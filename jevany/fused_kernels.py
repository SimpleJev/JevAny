"""Fused Qwen3.5 inference kernels: ``LoadOptions(fused_kernels=True)``, ``JEVANY_FUSED_KERNELS=1`` or
``--fused-kernels``.

At batch size one inside CUDA graphs (jevany.cudagraphs) a Qwen3.5 forward is a long chain of small kernels around
its GEMMs. ``fuse_qwen3_5`` rewrites a merged, eval-mode backbone in place:

* every zero-centred RMSNorm runs as one FLA kernel, with ``1 + weight`` precomputed in fp32;
* each MLP computes its gate and up projections as one GEMM followed by FLA's fused SwiGLU;
* each Gated DeltaNet layer computes its four input projections as one GEMM. FLA's chunked delta-rule kernel takes
  the raw gate and beta inputs and applies ``-exp(A_log) * softplus(a + dt_bias)``, the beta sigmoid, the q/k L2
  norm and the grouped value heads itself; the output norm is FLA's fused gated RMSNorm;
* each full-attention layer computes its query (with output gate), key and value projections as one GEMM;
* decoder weights are stored transposed, so cuBLAS reads both GEMM operands in their natural layout.

The fused matrices replace the originals and the original ``nn.Linear`` modules keep views of them, so memory does
not grow. A fused forward runs only for inference without a cache, attention mask or packed-sequence metadata, which
is how row-mode inference and CUDA-graph capture call the backbone; any other call takes the original forward. The
FLA kernels keep intermediates in fp32 where transformers rounds to BF16 between operations, so probabilities are
close to, not identical with, the unfused path.
"""
from __future__ import annotations

import inspect
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import torch
from torch import nn

MIN_FLA = "0.5.0"   # grouped value heads, and the gate computed inside the chunked delta-rule kernel


def _fla_version():
    for name in ("fla-core", "flash-linear-attention"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def load_kernels():
    """The flash-linear-attention functions the fused forwards call; ValueError when they are unavailable."""
    from packaging.version import InvalidVersion, Version
    found = _fla_version()
    try:
        recent = found is not None and Version(found) >= Version(MIN_FLA)
    except InvalidVersion:
        recent = False
    if not recent:
        raise ValueError(f"fused_kernels need flash-linear-attention>={MIN_FLA} (pip install 'jevany[fast]'); "
                         f"found {found}")
    from fla.modules.activations import swiglu
    from fla.modules.fused_norm_gate import rms_norm_gated
    from fla.modules.layernorm import rms_norm
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return SimpleNamespace(
        version=found, swiglu=swiglu, rms_norm=rms_norm, rms_norm_gated=rms_norm_gated,
        chunk_gated_delta_rule=chunk_gated_delta_rule,
        # flash-linear-attention 0.5.0 has no in-kernel beta sigmoid.
        beta_sigmoid_in_kernel="use_beta_sigmoid_in_kernel" in inspect.signature(chunk_gated_delta_rule).parameters,
    )


def _fuse(linears):
    """One contiguous (in x total out) matrix for bias-free linears; each linear keeps a view of its columns."""
    fused = torch.cat([linear.weight.detach().t() for linear in linears], dim=1).contiguous()
    start = 0
    for linear in linears:
        linear.weight = nn.Parameter(fused[:, start:start + linear.out_features].t(), requires_grad=False)
        start += linear.out_features
    return fused


def _transpose_storage(linear):
    """Store the (out x in) weight as the transpose of a contiguous (in x out) matrix: F.linear then computes
    x @ W^T with no transposed GEMM operand, which cuBLAS runs faster for these small-row shapes."""
    linear.weight = nn.Parameter(linear.weight.detach().t().contiguous().t(), requires_grad=False)


def _replace_forward(module, fused_forward, kernels):
    original = module.forward

    def forward(*args, **kwargs):
        return fused_forward(module, original, kernels, *args, **kwargs)

    module.forward = forward


def _rms_norm(self, original, kernels, hidden_states):
    if self.training:
        return original(hidden_states)
    return kernels.rms_norm(hidden_states, self.fused_weight, None, eps=self.eps)


def _mlp(self, original, kernels, x):
    if self.training:
        return original(x)
    gate_up = x @ self.fused_weight
    size = self.intermediate_size
    return self.down_proj(kernels.swiglu(gate_up[..., :size], gate_up[..., size:]))


def _gated_deltanet(self, original, kernels, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    if (self.training or cache_params is not None or attention_mask is not None
            or kwargs.get("cu_seq_lens_q") is not None):
        return original(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
    batch, length, _ = hidden_states.shape
    projected = hidden_states @ self.fused_weight                      # [qkv | z | b | a]
    qkv_end = 2 * self.key_dim + self.value_dim
    z_end = qkv_end + self.value_dim
    b_end = z_end + self.num_v_heads
    mixed = projected[..., :qkv_end].transpose(1, 2)
    if mixed.stride(0) % 8 or mixed.stride(2) % 8:
        mixed = mixed.contiguous()   # causal-conv1d reads channel-last input only with 16-byte aligned rows
    mixed = kernels.causal_conv1d(mixed, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation)
    query, key, value = torch.split(mixed.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    beta = projected[..., z_end:b_end]
    in_kernel = {"use_beta_sigmoid_in_kernel": True} if kernels.beta_sigmoid_in_kernel else {}
    core, _ = kernels.chunk_gated_delta_rule(
        query.reshape(batch, length, -1, self.head_k_dim), key.reshape(batch, length, -1, self.head_k_dim),
        value.reshape(batch, length, -1, self.head_v_dim), g=projected[..., b_end:],
        beta=beta if in_kernel else beta.sigmoid(), use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        A_log=self.A_log, dt_bias=self.dt_bias, output_final_state=False, **in_kernel)
    core = kernels.rms_norm_gated(core.reshape(-1, self.head_v_dim),
                                  projected[..., qkv_end:z_end].reshape(-1, self.head_v_dim),
                                  self.norm.weight, None, activation="swish", eps=self.norm.variance_epsilon)
    return self.out_proj(core.reshape(batch, length, -1))


def _attention(self, original, kernels, hidden_states, position_embeddings, attention_mask, past_key_values=None,
               **kwargs):
    if self.training or past_key_values is not None:
        return original(hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values,
                        **kwargs)
    qwen = kernels.qwen
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    projected = hidden_states @ self.fused_weight                      # [query and gate | key | value]
    query_end, key_end = self.fused_sizes
    query, gate = torch.chunk(projected[..., :query_end].view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    query = self.q_norm(query.reshape(hidden_shape)).transpose(1, 2)
    key = self.k_norm(projected[..., query_end:key_end].reshape(hidden_shape)).transpose(1, 2)
    value = projected[..., key_end:].reshape(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = qwen.apply_rotary_pos_emb(query, key, cos, sin)
    attention = qwen.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,
                                                           qwen.eager_attention_forward)
    output, weights = attention(self, query, key, value, attention_mask, dropout=0.0, scaling=self.scaling, **kwargs)
    output = output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate.reshape(*input_shape, -1))
    return self.o_proj(output), weights


def fuse_qwen3_5(lm):
    """Rewrite the Qwen3.5 modules of ``lm`` in place for fused inference and report what changed.

    ``lm`` must be a backbone with its LoRA merged, on one CUDA device, with its final weights loaded."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    modules = list(lm.modules())
    layers = [module for module in modules if isinstance(module, qwen.Qwen3_5DecoderLayer)]
    if not layers:
        raise ValueError("fused_kernels support Qwen3.5 backbones only")
    devices = {parameter.device for parameter in lm.parameters()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("fused_kernels need the whole backbone on one CUDA device")
    if any(hasattr(module, "lora_A") for module in modules):
        raise ValueError("fused_kernels need the LoRA merged into the backbone")
    kernels = load_kernels()
    kernels.qwen, kernels.causal_conv1d = qwen, qwen.causal_conv1d_fn
    counts = dict.fromkeys(("rms_norm", "mlp", "gated_deltanet", "attention", "transposed_linear"), 0)

    def bias_free(*linears):
        return all(linear.bias is None for linear in linears)

    with torch.no_grad():
        for module in modules:
            if isinstance(module, qwen.Qwen3_5RMSNorm):
                module.fused_weight = (1.0 + module.weight.detach().float()).contiguous()
                _replace_forward(module, _rms_norm, kernels)
                counts["rms_norm"] += 1
            elif (isinstance(module, qwen.Qwen3_5MLP) and module.config.hidden_act in ("silu", "swish")
                  and bias_free(module.gate_proj, module.up_proj)):
                module.fused_weight = _fuse([module.gate_proj, module.up_proj])
                _replace_forward(module, _mlp, kernels)
                counts["mlp"] += 1
            elif isinstance(module, qwen.Qwen3_5GatedDeltaNet) and module.activation in ("silu", "swish"):
                projections = [module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a]
                if bias_free(*projections):
                    module.fused_weight = _fuse(projections)
                    _replace_forward(module, _gated_deltanet, kernels)
                    counts["gated_deltanet"] += 1
            elif (isinstance(module, qwen.Qwen3_5Attention)
                  and bias_free(module.q_proj, module.k_proj, module.v_proj)):
                query, key = module.q_proj.out_features, module.k_proj.out_features
                module.fused_weight = _fuse([module.q_proj, module.k_proj, module.v_proj])
                module.fused_sizes = (query, query + key)
                _replace_forward(module, _attention, kernels)
                counts["attention"] += 1
        for layer in layers:
            for linear in layer.modules():
                if isinstance(linear, nn.Linear) and linear.weight.is_contiguous():
                    _transpose_storage(linear)
                    counts["transposed_linear"] += 1
    torch.cuda.empty_cache()
    return {"backbone": "qwen3_5", "flash_linear_attention": kernels.version, **counts}
