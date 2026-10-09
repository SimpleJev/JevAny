# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""One inference runtime shared by Python applications and the HTTP server."""
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .api import SystemOneRequest, output_tokens, to_answers, to_record, validate_response
from .backbones import linear_attention_kernels
from .checkpoint import Checkpoint, LoadOptions
from .client import DecisionClient
from .device import default_device, sync
from .inference import InferenceOptions
from .model import ContextLengthError, DecisionModel
from .readout import resolve_readout_options

DEFAULT_CHECKPOINT = "SimpleJev/JevAny-Qwen3.8-27B-LoRA"


@dataclass
class DecisionRuntime:
    checkpoint: Checkpoint
    tok: object
    model: DecisionModel
    device: str
    model_id: str = "jevany"
    inference_options: InferenceOptions = field(default_factory=InferenceOptions)
    lock: Any = field(default_factory=threading.RLock, repr=False)
    prefix_cache: dict = field(default_factory=dict)
    prefix_hits: int = 0
    prefix_misses: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_name must be a nonempty string")

    @property
    def limits(self) -> dict[str, int | None]:
        options = self.inference_options
        window = self.model.inference_capabilities.context_window
        return {
            "state_tokens": min(options.max_state_tokens, window) if window else options.max_state_tokens,
            "branch_tokens": min(options.max_branch_tokens, window) if window else options.max_branch_tokens,
            "packed_tokens": options.max_packed_tokens,
            "choices": len(self.model.verbalizers) if self.model.decision_mode == "lm_token" else None,
        }

    @property
    def aliases(self) -> list[str]:
        return [] if self.model_id == "jevany-latest" else ["jevany-latest"]

    def describe(self) -> dict[str, Any]:
        """Describe the loaded model and effective settings without loading weights again."""
        with self.lock:
            capabilities = self.model.inference_capabilities
            return {
                "id": self.model_id, "aliases": self.aliases,
                "run": self.checkpoint.requested, "base": self.checkpoint.meta.base,
                "lora": self.checkpoint.meta.lora, "device": self.device,
                "device_map": getattr(self.model, "device_map", None),
                "devices": getattr(self.model, "devices", [self.device]),
                "temperature": self.model.temperature,
                "decision_mode": self.checkpoint.meta.decision_mode,
                "readout": "native",
                "backbone_adapter": self.model.backbone_adapter,
                "branch_mode": self.model.branch_mode,
                "acceleration": {
                    **getattr(self.model, "inference_acceleration", {
                        "compile_mode": None, "lora_merged": False, "approximate_bf16_merge": False,
                    }),
                    # Fused gated-DeltaNet kernels; None when the backbone has no such layers.
                    "linear_attention_kernels": (linear_attention_kernels()
                                                 if getattr(self.model, "hybrid", False) else None),
                },
                "capabilities": {**asdict(capabilities), "media_types": list(capabilities.media_types)},
                "limits": self.limits,
                "prefix_cache": {
                    "enabled": self.inference_options.prefix_cache_size > 0 and capabilities.prefix_cache,
                    "size": self.inference_options.prefix_cache_size,
                    "min_state_tokens": self.inference_options.prefix_min_tokens,
                    "hits": self.prefix_hits, "misses": self.prefix_misses,
                    "cached_states": len(self.prefix_cache),
                },
            }

    def clear_cache(self) -> None:
        """Release cached prefixes under the same lock as inference."""
        with self.lock:
            self.prefix_cache.clear()
            self.prefix_hits = self.prefix_misses = 0

    def probs(self, record: dict) -> tuple[list[list[float]], dict]:
        """Encode and score under one lock, including the shared media processor."""
        with self.lock:
            limits, options = self.limits, self.inference_options
            capabilities = self.model.inference_capabilities
            if record.get("media"):
                if any(item["type"] not in capabilities.media_types for item in record["media"]):
                    raise ValueError("checkpoint does not support the requested media type")
                maximum = capabilities.max_media_questions
                if maximum is not None and len(record["questions"]) > maximum:
                    raise ValueError(f"media requests support at most {maximum} question(s)")
            encoding = self.model.encode(
                self.tok, record, max_state=limits["state_tokens"], max_branch=limits["branch_tokens"], strict=True,
            )
            if len(encoding["ids"]) > limits["packed_tokens"]:
                raise ContextLengthError(
                    f"request exceeds {limits['packed_tokens']} packed tokens: {len(encoding['ids'])}")
            if capabilities.context_window is not None and max(encoding["pos"]) >= capabilities.context_window:
                raise ContextLengthError(
                    f"request exceeds backbone context window of {capabilities.context_window} tokens")
            state_tokens = encoding["seg"].count(0)
            key = (tuple(encoding["ids"][:state_tokens]), bool(encoding.get("option_isolation")))
            cache, hit = self.prefix_cache, False
            sync(self.device)
            start = time.perf_counter()
            eligible = (options.prefix_cache_size > 0 and capabilities.prefix_cache
                        and state_tokens >= options.prefix_min_tokens
                        and not record.get("media") and not encoding.get("multimodal"))
            if eligible and key in cache:
                prefix = cache.pop(key)
                probabilities = self.model.probs_with_prefix(encoding, prefix)
                cache[key] = prefix
                self.prefix_hits += 1
                hit = True
            elif eligible:
                probabilities, prefix = self.model.probs_and_prefix(encoding)
                cache[key] = prefix
                while len(cache) > options.prefix_cache_size:
                    cache.pop(next(iter(cache)))
                self.prefix_misses += 1
            else:
                probabilities = self.model.probs(encoding)
            sync(self.device)
            elapsed = time.perf_counter() - start
        return [p.tolist() for p in probabilities], {
            "tokens": len(encoding["ids"]), "state_tokens": state_tokens,
            "latency_ms": round(elapsed * 1000, 1), "prefix_cache_hit": hit,
        }

    def answer(self, request: SystemOneRequest) -> dict[str, Any]:
        if request.model not in (self.model_id, *self.aliases):
            raise ValueError(f"unknown model {request.model!r}; this deployment serves {self.model_id!r}")
        record, metadata = to_record(request)
        # Output accounting uses the same tokenizer as encoding; media processors
        # can mutate its settings, so serialize both operations together.
        with self.lock:
            probabilities, measurements = self.probs(record)
            answers = to_answers(probabilities, metadata)
            return validate_response(request, {
                "model": self.model_id, "answers": answers,
                "usage": {"input_tokens": measurements["tokens"], "output_tokens": output_tokens(self.tok, answers)},
                "latency_ms": measurements["latency_ms"],
            })


class JevModel(DecisionClient):
    """Load a checkpoint once and make decisions without starting an HTTP server."""

    def __init__(self, runtime: DecisionRuntime) -> None:
        self.runtime = runtime
        self.model_id = runtime.model_id

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path = DEFAULT_CHECKPOINT, *,
        device: str | None = None, dtype: str | None = None,
        model_name: str | None = None, options: LoadOptions | None = None,
        inference_options: InferenceOptions | None = None,
        readout: str = "native", choice_temperature: float | None = None,
        choice_native_weight: float | None = None, choice_max_tokens: int | None = None,
        letter_temperature: float | None = None,
        letter_pointer_weight: float | None = None, letter_max_tokens: int | None = None,
    ) -> "JevModel":
        """Load a local run or Hugging Face adapter ID (optionally ``owner/repo@revision``).

        The full backbone must fit on the selected device unless ``options.device_map``
        (or JEVANY_DEVICE_MAP) splits it over the visible GPUs. ``dtype`` accepts
        fp32, fp16 or bf16; omission uses the checkpoint/environment settings.
        ``readout='choice'`` replaces the checkpoint head with a training-free
        choice-token projection; its temperature, native blend, and prompt
        limit are deployment settings, not checkpoint metadata. ``letter`` and
        ``letter_*`` remain accepted legacy aliases. Files used by native media
        requests are trusted local paths.
        """
        import torch

        readout, choice_options = resolve_readout_options(
            readout,
            choice_temperature=choice_temperature,
            choice_native_weight=choice_native_weight,
            choice_max_tokens=choice_max_tokens,
            letter_temperature=letter_temperature,
            letter_pointer_weight=letter_pointer_weight,
            letter_max_tokens=letter_max_tokens,
        )
        if readout == "choice" and inference_options is not None:
            raise ValueError("inference_options apply only to native readout; use choice_max_tokens")

        device = default_device() if device is None else device
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps or cuda")
        options = options or LoadOptions.from_env()
        if readout == "native":
            inference_options = inference_options or InferenceOptions.from_env()
        if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
            raise ValueError("model_name must be a nonempty string")
        if dtype is not None:
            dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
            if dtype not in dtypes:
                raise ValueError("dtype must be fp32, fp16 or bf16")
            options = replace(options, dtype=dtypes[dtype])
        if device == "mps" and options.attn is None:
            options = replace(options, attn="sdpa")
        if readout == "choice" and options.cuda_graphs:
            raise ValueError("CUDA graph capture is available only for native readout")
        if readout == "choice" and options.temperature is not None:
            raise ValueError("JEVANY_TEMPERATURE applies to the native head; use choice_temperature")
        predictor = None
        if readout == "choice":
            from .letter_predictor import LetterReadoutPredictor

            predictor = LetterReadoutPredictor(
                checkpoint=checkpoint,
                device=device,
                options=options,
                temperature=choice_options.temperature,
                native_weight=choice_options.native_weight,
                max_tokens=choice_options.max_tokens,
            )
            loaded = predictor.checkpoint
        else:
            loaded = Checkpoint(checkpoint)
        if model_name is None:
            source = loaded.requested.partition("@")[0]
            if source == DEFAULT_CHECKPOINT:
                model_name = "jevany-qwen3.8-27b-lora"
            elif source in ("tianxinwei/JevAny-27B-SFT", "tianxinwei/JevAny-27B-RLCR"):
                model_name = "jevany-27b"
            else:
                model_name = Path(source).name or Path(loaded.path).resolve().name
        if predictor is not None:
            from .letter_runtime import LetterDecisionRuntime
            return cls(LetterDecisionRuntime(predictor, model_name))
        tokenizer, model = loaded.load(device, options)
        return cls(DecisionRuntime(loaded, tokenizer, model, device, model_name, inference_options))

    def describe(self) -> dict[str, Any]:
        """Return identity, backbone capabilities, effective limits and cache statistics."""
        return self.runtime.describe()

    def models(self) -> list[dict[str, Any]]:
        """List this local model using the same discovery method as JevClient."""
        return [self.describe()]

    def clear_cache(self) -> None:
        """Release this model's cached state prefixes."""
        self.runtime.clear_cache()

    def __call__(self, request: SystemOneRequest | dict) -> dict[str, Any]:
        validated = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        return self.runtime.answer(validated)
