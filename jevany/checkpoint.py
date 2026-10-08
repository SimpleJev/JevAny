# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Trained checkpoints: a run directory or Hub repo holding a LoRA adapter, metadata, and a tokenizer.

This is the one place that knows the layout of `head.pt` and how a checkpoint becomes a `DecisionModel`:
`jevany.serve`, `jevany.benchmark` and `jevany.train --init_from` all go through it.

    ck = Checkpoint("SimpleJev/JevAny-Qwen3.8-27B-LoRA")  # or a local run directory; `@tag` pins a Hub revision
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
"""
import json
import math
import os
import re
from importlib import metadata as importlib_metadata
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from .model import DecisionModel, load_preprocessor
from .backbones import get_backbone_adapter
# Re-exported: the lightweight tools import these without the modelling stack.
from .placement import DEVICE_MAPS, add_placement_arguments

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")
COMPILE_MODES = frozenset({"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"})


def _compile_mode(value):
    value = str(value or "").strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return None
    if value in ("1", "true", "on", "yes"):
        return "reduce-overhead"
    if value not in COMPILE_MODES:
        raise ValueError(f"JEVANY_COMPILE must be 0, 1, or one of {sorted(COMPILE_MODES)}")
    return value


def _check_compile_runtime():
    """Reject a mismatched Torch/Triton pair before an expensive model load."""
    from packaging.requirements import Requirement
    try:
        installed = importlib_metadata.version("triton")
    except importlib_metadata.PackageNotFoundError as error:
        raise ValueError("torch.compile on CUDA requires the Triton version declared by torch") from error
    requirements = importlib_metadata.requires("torch") or ()
    for value in requirements:
        requirement = Requirement(value)
        if requirement.name.casefold() != "triton":
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if installed not in requirement.specifier:
            raise ValueError(
                f"torch.compile is disabled: torch requires {requirement}, but Triton {installed} is installed"
            )
        return


def is_hub_id(run):
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_run(run):
    """Local run directory or a Hub ID such as SimpleJev/JevAny-Qwen3.8-27B-LoRA, optionally pinned."""
    if os.path.isdir(run):
        if not (Path(run) / "head.pt").is_file():
            raise ValueError(f"{run}: missing head.pt; pass a trained JevAny checkpoint, not base weights")
        return str(run)
    if not is_hub_id(run):
        raise ValueError(f"checkpoint does not exist: {run}; use a local directory or owner/repo[@revision]")
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None,
                             allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja", "*.model", "*.tiktoken"])


@dataclass
class Meta:
    """Contents of `head.pt`. Every reader gets the same defaults for fields older checkpoints did not write.
    `extra` keeps the rest of the file (training args, suite hash, init provenance, temperature fit) so a
    read-modify-write round trip loses nothing."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    head_residual_dim: int = 0
    head_type: str = "linear"
    option_isolation: bool = False
    special_embeddings: bool = False
    multimodal: bool = False
    backbone_adapter: str = "auto"
    branch_mode: str = "auto"
    tokenizer_saved: bool = False
    weights_dtype: str = "fp32"
    decision_mode: str = "pointer"
    verbalizers: list = field(default_factory=list)
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "head_residual_dim", "head_type",
             "option_isolation", "special_embeddings", "multimodal", "backbone_adapter", "branch_mode",
             "tokenizer_saved", "weights_dtype", "decision_mode", "verbalizers", "temperature", "holdout")

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.KNOWN if k in d}, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}   # known fields win over a stray key in extra


def read_meta(run):
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu"))


def write_meta(run, meta):
    torch.save(meta.to_dict(), f"{run}/head.pt")


@dataclass(frozen=True)
class LoadOptions:
    """How a checkpoint is turned into a model. Defaults are the exact path every reported number uses; the fields
    are the same knobs the JEVANY_* environment variables expose to the command-line tools (see from_env).

    dtype        None = fp32 (bf16 when the checkpoint was trained with a bf16 backbone). bf16 halves memory for
                 serving large backbones; probabilities then differ from fp32 in the third decimal.
    merge        fold the LoRA into the base weights in fp32 before any cast. Exact in fp32; in bf16 it is faster (~15%)
                 and closer to fp32 than merging directly into bf16. Ignored for adapters that carry trained token embeddings.
    attn         attention backend; None = the model default (SDPA on CUDA, eager elsewhere). "sdpa" on MPS measured
                 parity with eager and is a few percent faster.
    lora_scale   WiSE-FT-style interpolation between base (0) and fine-tuned weights (1), at inference.
    temperature  None = the temperature the checkpoint carries (fitted by scripts/calibrate_checkpoint.py); 1.0 = raw logits.
    base_load_path
                 optional node-local mirror for base-model I/O. Checkpoint metadata still names the canonical base.
    merge_bf16   allow the faster but slightly approximate merge of a LoRA adapter into BF16 base weights.
    compile_mode opt into torch.compile. ``reduce-overhead`` also enables CUDA Graphs for compatible graph segments.
    device_map   None = the whole model on one device. An Accelerate placement strategy (DEVICE_MAPS) splits the
                 backbone's layers over every visible GPU, for bases larger than one card; layers still run one after
                 another, so this adds memory, not speed. The readout head stays on the requested device.
    max_memory_gib
                 per-GPU weight budget for device_map; leave room for activations on long requests.
    cuda_graphs  capture one CUDA graph per padded row length after loading (jevany.cudagraphs) and replay them at
                 inference. Row-mode backbones on one CUDA device only; capture adds tens of seconds to loading.
                 Probabilities are not bit-identical to the eager path.
    cuda_graph_max_tokens
                 largest captured row. Longer rows use eager inference. The conservative default avoids padding
                 regressions on long requests; raise it only after measuring the target model and workload.
    fused_kernels
                 fused Qwen3.5 inference kernels (jevany.fused_kernels): one-kernel RMSNorms, merged projections, the
                 Gated DeltaNet gate inside FLA's kernel and transposed weight storage. Needs the LoRA merged, one
                 CUDA device and flash-linear-attention; the gain comes inside CUDA graphs. Probabilities are close
                 to, not identical with, the unfused path.
    """
    dtype: torch.dtype | None = None
    merge: bool = True
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None
    base_load_path: str | None = None
    merge_bf16: bool = False
    compile_mode: str | None = None
    device_map: str | None = None
    max_memory_gib: float | None = None
    cuda_graphs: bool = False
    cuda_graph_max_tokens: int = 2048
    fused_kernels: bool = False

    def __post_init__(self):
        if self.compile_mode not in (None, *COMPILE_MODES):
            raise ValueError(f"compile_mode must be one of {sorted(COMPILE_MODES)}")
        if self.device_map is not None and self.device_map not in DEVICE_MAPS:
            raise ValueError(f"device_map must be one of {', '.join(DEVICE_MAPS)}")
        if self.max_memory_gib is not None:
            if self.device_map is None:
                raise ValueError("max_memory_gib requires device_map")
            if not (math.isfinite(self.max_memory_gib) and self.max_memory_gib > 0):
                raise ValueError("max_memory_gib must be finite and positive")
        if self.cuda_graphs and self.compile_mode:
            raise ValueError("cuda_graphs and compile_mode are alternative graph captures; enable one")
        if self.cuda_graphs and self.device_map is not None:
            raise ValueError("cuda_graphs require the whole model on one GPU; disable device_map")
        if type(self.cuda_graph_max_tokens) is not int or self.cuda_graph_max_tokens < 1:
            raise ValueError("cuda_graph_max_tokens must be a positive integer")
        if self.fused_kernels and self.compile_mode:
            raise ValueError("fused_kernels and compile_mode cannot be combined")
        if self.fused_kernels and self.device_map is not None:
            raise ValueError("fused_kernels require the whole model on one GPU; disable device_map")

    @classmethod
    def from_env(cls, env=os.environ):
        """Read checkpoint loading, placement and optional inference acceleration settings."""
        return cls(dtype={"bf16": torch.bfloat16, "fp16": torch.float16}.get(env.get("JEVANY_DTYPE", "")),
                   merge=env.get("JEVANY_MERGE", "1") != "0", attn=env.get("JEVANY_ATTN") or None,
                   lora_scale=float(env.get("JEVANY_LORA_SCALE", "1")),
                   temperature=float(env["JEVANY_TEMPERATURE"]) if env.get("JEVANY_TEMPERATURE") else None,
                   base_load_path=env.get("JEVANY_BASE_LOAD_PATH") or None,
                   merge_bf16=env.get("JEVANY_MERGE_BF16", "0") == "1",
                   compile_mode=_compile_mode(env.get("JEVANY_COMPILE")),
                   device_map=env.get("JEVANY_DEVICE_MAP") or None,
                   max_memory_gib=float(env["JEVANY_MAX_MEMORY_GIB"]) if env.get("JEVANY_MAX_MEMORY_GIB") else None,
                   cuda_graphs=env.get("JEVANY_CUDA_GRAPHS", "0") == "1",
                   cuda_graph_max_tokens=int(env.get("JEVANY_CUDA_GRAPH_MAX_TOKENS", "2048")),
                   fused_kernels=env.get("JEVANY_FUSED_KERNELS", "0") == "1")


def load_options_from_args(args, env=os.environ):
    """LoadOptions from the environment with explicit placement flags applied; None when neither is set."""
    overrides = {name: getattr(args, name) for name in ("device_map", "max_memory_gib")
                 if getattr(args, name, None) is not None}
    return replace(LoadOptions.from_env(env), **overrides) if overrides else None


class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)                    # what the caller asked for (a Hub id stays a Hub id in labels)
        self.path = resolve_run(run)
        self.meta = read_meta(self.path)

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    def load(self, device, opts=LoadOptions()):
        """Return an eval model with its LoRA and configured decision readout loaded."""
        if opts.compile_mode:
            if not str(device).startswith("cuda"):
                raise ValueError("torch.compile acceleration is supported only on CUDA")
            _check_compile_runtime()
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # The exact path keeps the FP32 adapter separate from BF16 base weights.
            # An explicit fast-mode opt-in permits the lossy BF16 merge.
            dtype, merge = torch.bfloat16, merge and opts.merge_bf16
        source, revision = meta.base, meta.base_revision
        if opts.base_load_path:
            if not Path(opts.base_load_path).is_dir():
                raise ValueError(f"base load path does not exist: {opts.base_load_path}")
            source, revision = opts.base_load_path, None
        tokenizer_source = self.path if meta.tokenizer_saved else source
        adapter_name = meta.backbone_adapter
        if adapter_name == "auto" and meta.multimodal:
            adapter_name = get_backbone_adapter("auto", multimodal=True, source=source, revision=revision).name
        if meta.tokenizer_saved and not self.file("tokenizer_config.json").is_file():
            raise ValueError(f"{self.path}: missing saved tokenizer_config.json")
        tok = load_preprocessor(tokenizer_source, revision=None if meta.tokenizer_saved else revision,
                                multimodal=meta.multimodal, backbone_adapter=adapter_name)
        adapter_config = self.adapter_config()
        # Direct-token scoring reuses the causal model's frozen vocabulary
        # projection, which is outside the feature-extraction PEFT wrapper.
        # Keep that association intact instead of merging the wrapper away.
        merge = (merge and meta.decision_mode != "lm_token"
                 and not adapter_config.get("trainable_token_indices"))
        if opts.fused_kernels:
            if not str(device).startswith("cuda"):
                raise ValueError("fused_kernels run on CUDA only")
            if not merge:
                raise ValueError("fused_kernels need the LoRA merged into the backbone (JEVANY_MERGE_BF16=1 for BF16 "
                                 "checkpoints); direct-token checkpoints and trainable token embeddings stay unmerged")
            from .fused_kernels import load_kernels
            load_kernels()   # fail before loading weights when flash-linear-attention is missing
        saved_args = meta.extra.get("args", {})
        lora_targets = saved_args.get("lora_targets", "all")
        explicit_targets = saved_args.get("lora_target_modules", "")
        if not explicit_targets and adapter_config.get("target_modules"):
            # Exact saved modules are the portable contract. Presets can expand
            # as new backbone implementations expose additional linear layers.
            explicit_targets = ",".join(sorted(adapter_config["target_modules"]))
        m = DecisionModel(source, tok, device, lora=meta.lora, revision=revision, head_dim=meta.head_dim,
                          head_residual_dim=meta.head_residual_dim, lora_targets=lora_targets,
                          lora_dropout=float(adapter_config.get("lora_dropout", 0.05)),
                          special_embeddings=meta.special_embeddings,
                          option_isolation=meta.option_isolation, dtype=torch.float32 if merge else dtype,
                          attn=opts.attn, multimodal=meta.multimodal,
                          backbone_adapter=adapter_name, branch_mode=meta.branch_mode,
                          lora_target_modules=explicit_targets,
                          decision_mode=meta.decision_mode, verbalizers=meta.verbalizers or None,
                          device_map=opts.device_map, max_memory_gib=opts.max_memory_gib)
        self.warm_start(m, meta)
        if opts.lora_scale != 1:
            for module in m.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for k in module.scaling: module.scaling[k] *= opts.lora_scale
            m.lora_scale = opts.lora_scale
        if merge:
            m.set_language_model(m.lm.merge_and_unload())
            if dtype != torch.float32:
                m.set_language_model(m.lm.to(dtype))
        if m.head is not None:
            m.head.load_state_dict(meta.head)
        m.eval()
        m.temperature = meta.temperature if opts.temperature is None else opts.temperature
        if m.head is not None:
            m.head.temperature = m.temperature
        m.inference_acceleration = {
            "compile_mode": opts.compile_mode,
            "lora_merged": bool(merge),
            "approximate_bf16_merge": bool(merge and meta.weights_dtype == "bf16"),
            "fused_kernels": None,
        }
        if opts.fused_kernels:
            from .fused_kernels import fuse_qwen3_5
            m.inference_acceleration["fused_kernels"] = fuse_qwen3_5(m.lm)
        if opts.compile_mode:
            m.lm.compile(mode=opts.compile_mode, fullgraph=False, dynamic=True)
        graphs = None
        if opts.cuda_graphs:
            from .cudagraphs import RowGraphs
            graphs = m.cuda_graphs = RowGraphs(m, max_tokens=opts.cuda_graph_max_tokens).capture()
        m.inference_acceleration["cuda_graphs"] = graphs.stats if graphs is not None else None
        return tok, m

    COMPAT_FIELDS = ("base", "base_revision", "lora", "head_dim", "head_residual_dim", "head_type",
                     "option_isolation", "special_embeddings", "multimodal", "decision_mode", "verbalizers")

    def warm_start(self, model, ours):
        """Delta training: load this checkpoint's adapter and pointer head into `model` (a fresh DecisionModel built with
        LoRA). `ours` is the Meta the new run will save; every architecture field is compared BEFORE loading, because peft
        loads matching keys silently and a half-loaded adapter still trains and still reports a loss. Returns provenance."""
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        from .suite import digest
        for name in self.COMPAT_FIELDS:
            theirs, mine = getattr(self.meta, name), getattr(ours, name)
            if theirs != mine and not (name == "base_revision" and None in (theirs, mine)):
                raise ValueError(f"--init_from {self.path}: {name} is {theirs!r} there and {mine!r} here")
        if self.meta.tokenizer_saved:
            for name in ("backbone_adapter", "branch_mode"):
                expected = getattr(self.meta, name)
                if name == "backbone_adapter" and expected == "auto":
                    expected = get_backbone_adapter("auto", multimodal=self.meta.multimodal,
                                                    source=self.meta.base, revision=self.meta.base_revision).name
                actual = getattr(ours, name)
                if name == "backbone_adapter" and actual == "auto":
                    actual = model.backbone_adapter
                if expected != actual:
                    raise ValueError(f"--init_from {self.path}: incompatible {name}")
        saved_tokens = self.adapter_config().get("trainable_token_indices")
        current_tokens = model.lm.peft_config["default"].trainable_token_indices
        if saved_tokens != current_tokens:
            raise ValueError(f"--init_from {self.path}: trainable token IDs differ from this tokenizer")
        weights = load_peft_weights(self.path, device="cpu")
        have = set(get_peft_model_state_dict(model.lm, save_embedding_layers=False))
        unexpected, missing = sorted(set(weights) - have), sorted(have - set(weights))
        if unexpected:
            raise ValueError(f"--init_from {self.path} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
        if missing:
            raise ValueError(f"--init_from {self.path} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(model.lm, weights)
        if model.head is not None:
            model.head.load_state_dict(self.meta.head)
        return {"init_from": self.requested, "resolved": self.path, "adapter_sha256": digest(self.file("adapter_model.safetensors")),
                "head_sha256": digest(self.file("head.pt")), "adapter_tensors": len(weights)}


def load(run, device, opts=LoadOptions()):
    """Convenience: Checkpoint(run).load(device, opts)."""
    return Checkpoint(run).load(device, opts)
