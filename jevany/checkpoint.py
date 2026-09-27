# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Trained checkpoints: a run directory or a Hub repo holding a LoRA adapter, `head.pt` and the tokenizer.

This is the one place that knows the layout of `head.pt` and how a checkpoint becomes a `DecisionModel`:
`jevany.serve`, `jevany.benchmark` and `jevany.train --init_from` all go through it.

    ck = Checkpoint("tianxinwei/JevAny-27B-RLCR")    # or a local run directory; `@tag` pins a Hub revision
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import DecisionModel, load_preprocessor

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")


def is_hub_id(run):
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_run(run):
    """Local run directory as given, or a Hub repo id such as tianxinwei/JevAny-27B-RLCR, optionally pinned to a revision."""
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


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
    query_readout: str = "decide"
    option_readout: str = "close"
    option_isolation: bool = False
    special_embeddings: bool = False
    multimodal: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "head_residual_dim", "head_type", "query_readout", "option_readout", "option_isolation", "special_embeddings", "multimodal", "weights_dtype", "temperature", "holdout")

    @classmethod
    def from_dict(cls, d):
        known = {k: d[k] for k in cls.KNOWN if k in d}
        if "head_type" not in known:
            known["head_type"] = "residual" if d.get("head_residual_dim", 0) else "linear"
        return cls(**known, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

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
    """
    dtype: torch.dtype | None = None
    merge: bool = True
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None
    base_load_path: str | None = None

    @classmethod
    def from_env(cls, env=os.environ):
        """Read JEVANY_DTYPE, JEVANY_MERGE, JEVANY_ATTN, JEVANY_LORA_SCALE,
        JEVANY_TEMPERATURE, and JEVANY_BASE_LOAD_PATH at command-line entry points."""
        return cls(dtype={"bf16": torch.bfloat16, "fp16": torch.float16}.get(env.get("JEVANY_DTYPE", "")),
                   merge=env.get("JEVANY_MERGE", "1") != "0", attn=env.get("JEVANY_ATTN") or None,
                   lora_scale=float(env.get("JEVANY_LORA_SCALE", "1")),
                   temperature=float(env["JEVANY_TEMPERATURE"]) if env.get("JEVANY_TEMPERATURE") else None,
                   base_load_path=env.get("JEVANY_BASE_LOAD_PATH") or None)


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
        """-> (tokenizer, DecisionModel) in eval mode with the LoRA applied and the pointer head loaded."""
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # Keep a fp32 adapter unmerged when the checkpoint was trained over a bf16 backbone.
            dtype, merge = torch.bfloat16, False
        source, revision = meta.base, meta.base_revision
        if opts.base_load_path:
            if not Path(opts.base_load_path).is_dir():
                raise ValueError(f"base load path does not exist: {opts.base_load_path}")
            source, revision = opts.base_load_path, None
        tok = load_preprocessor(source, revision=revision, multimodal=meta.multimodal)
        merge = merge and not self.adapter_config().get("trainable_token_indices")   # token-trained adapters stay unmerged
        lora_targets = meta.extra.get("args", {}).get("lora_targets", "all")
        m = DecisionModel(source, tok, device, lora=meta.lora, revision=revision, head_dim=meta.head_dim,
                          head_residual_dim=meta.head_residual_dim, head_type=meta.head_type,
                          query_readout=meta.query_readout, option_readout=meta.option_readout,
                          lora_targets=lora_targets,
                          lora_dropout=float(self.adapter_config().get("lora_dropout", 0.05)),
                          special_embeddings=meta.special_embeddings,
                          option_isolation=meta.option_isolation, dtype=torch.float32 if merge else dtype,
                          attn=opts.attn, multimodal=meta.multimodal)
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
        m.head.load_state_dict(meta.head); m.eval()
        m.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        return tok, m

    COMPAT_FIELDS = ("base", "base_revision", "head_dim", "query_readout", "option_readout", "option_isolation", "special_embeddings", "multimodal")

    @staticmethod
    def _verify_constructed_architecture(model, meta):
        actual = {
            "head_dim": model.head.q.out_features,
            "head_residual_dim": (model.head.residual[0].out_features
                                  if model.head.residual is not None else 0),
            "head_type": model.head.head_type,
            "query_readout": model.query_readout,
            "option_readout": model.option_readout,
        }
        expected = {name: getattr(meta, name) for name in actual}
        if actual != expected:
            raise ValueError(f"constructed model architecture does not match checkpoint metadata: {actual} != {expected}")

    @staticmethod
    def _expand_lora_weights(weights, destination):
        """Copy a lower-rank adapter into the leading subspace of a larger one."""
        expanded = {}
        for name, target in destination.items():
            if name not in weights:
                raise ValueError(f"parent adapter is missing {name}")
            source = weights[name]
            if source.shape == target.shape:
                expanded[name] = source
            elif ".lora_A." in name and source.shape[1:] == target.shape[1:] and source.shape[0] < target.shape[0]:
                # Keep the destination initialization in the new A rows. New B
                # columns start at zero, so these rows do not affect step-0
                # logits but receive gradients after B first moves.
                value = target.detach().cpu().clone()
                value[:source.shape[0]].copy_(source)
                expanded[name] = value
            elif ".lora_B." in name and source.shape[:-1] == target.shape[:-1] and source.shape[-1] < target.shape[-1]:
                value = torch.zeros_like(target, device="cpu")
                value[..., :source.shape[-1]].copy_(source)
                expanded[name] = value
            else:
                raise ValueError(f"cannot expand {name} from {tuple(source.shape)} to {tuple(target.shape)}")
        unexpected = sorted(set(weights) - set(destination))
        if unexpected:
            raise ValueError(f"parent adapter has unexpected tensors: {unexpected[:2]}")
        return expanded

    @staticmethod
    def _load_parent_head(head, weights, source_residual_dim):
        result = head.load_state_dict(weights, strict=False)
        allowed = {name for name in head.state_dict() if name.startswith("residual.")} if source_residual_dim == 0 else set()
        if set(result.missing_keys) != allowed or result.unexpected_keys:
            raise ValueError(f"parent pointer head is incompatible: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
        if allowed:
            for name, value in head.state_dict().items():
                if name.startswith("residual.2.") and torch.count_nonzero(value):
                    raise ValueError("new pointer residual output must remain zero at warm start")

    def warm_start(self, model, ours):
        """Delta training: load this checkpoint's adapter and pointer head into `model` (a fresh DecisionModel built with
        LoRA). `ours` is the Meta the new run will save; every architecture field is compared BEFORE loading, because peft
        loads matching keys silently and a half-loaded adapter still trains and still reports a loss. Returns provenance."""
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        from .suite import digest
        self._verify_constructed_architecture(model, ours)
        for name in self.COMPAT_FIELDS:
            theirs, mine = getattr(self.meta, name), getattr(ours, name)
            if theirs != mine and not (name == "base_revision" and None in (theirs, mine)):
                raise ValueError(f"--init_from {self.path}: {name} is {theirs!r} there and {mine!r} here")
        if ours.lora < self.meta.lora:
            raise ValueError(f"--init_from cannot shrink LoRA rank {self.meta.lora} to {ours.lora}")
        if (self.meta.head_type != ours.head_type
                and not (self.meta.head_type == "linear" and ours.head_type == "residual")):
            raise ValueError(f"pointer head may only preserve its type or grow from linear to residual; found {self.meta.head_type} to {ours.head_type}")
        source_config = self.adapter_config()
        destination_config = next(iter(model.lm.peft_config.values()))
        weights = load_peft_weights(self.path, device="cpu")
        destination = get_peft_model_state_dict(model.lm)
        if ours.lora != self.meta.lora:
            if (source_config.get("r") != self.meta.lora
                    or source_config.get("rank_pattern")
                    or source_config.get("alpha_pattern")
                    or source_config.get("lora_alpha") != 2 * self.meta.lora
                    or destination_config.r != ours.lora
                    or destination_config.rank_pattern
                    or destination_config.alpha_pattern
                    or destination_config.lora_alpha != 2 * ours.lora):
                raise ValueError("LoRA rank expansion requires uniform alpha/r=2 adapters")
            weights = self._expand_lora_weights(weights, destination)
        else:
            unexpected, missing = sorted(set(weights) - set(destination)), sorted(set(destination) - set(weights))
            if unexpected:
                raise ValueError(f"--init_from {self.path} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
            if missing:
                raise ValueError(f"--init_from {self.path} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(model.lm, weights)
        if not (self.meta.head_residual_dim == ours.head_residual_dim
                or self.meta.head_residual_dim == 0 < ours.head_residual_dim):
            raise ValueError("pointer residual head may only grow from zero at warm start")
        self._load_parent_head(model.head, self.meta.head, self.meta.head_residual_dim)
        return {"init_from": self.requested, "resolved": self.path, "adapter_sha256": digest(self.file("adapter_model.safetensors")),
                "head_sha256": digest(self.file("head.pt")), "adapter_tensors": len(weights),
                "source_lora_rank": self.meta.lora, "destination_lora_rank": ours.lora,
                "lora_expansion": "leading_subspace_zero_pad" if ours.lora != self.meta.lora else None,
                "source_lora_dropout": source_config.get("lora_dropout"),
                "destination_lora_dropout": destination_config.lora_dropout,
                "head_residual_expansion": self.meta.head_residual_dim == 0 < ours.head_residual_dim}


def load(run, device, opts=LoadOptions()):
    """Convenience: Checkpoint(run).load(device, opts)."""
    return Checkpoint(run).load(device, opts)
