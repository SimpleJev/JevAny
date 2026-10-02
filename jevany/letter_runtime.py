"""System One runtime adapter for the training-free option-letter predictor."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .api import SystemOneRequest, output_tokens, to_answers, to_record, validate_response


def _unlabelled_record(request: SystemOneRequest) -> dict[str, Any]:
    """Build the predictor record shape without exposing labels to the model.

    The native pointer branch uses :func:`jevany.data.materialize`, whose
    benchmark input contract includes a label.  Harmless first-option labels
    let the serving path reuse that encoder; ``api_request`` removes them
    before either readout sees the request.
    """

    record = request.model_dump(exclude={"model"})
    for question in record["questions"].values():
        if question["type"] == "choice":
            question["label"] = next(iter(question["criteria"]))
        elif question["type"] == "score":
            question["label"] = 0
        else:
            question["label"] = False
    return record


@dataclass
class LetterDecisionRuntime:
    """Expose ``LetterReadoutPredictor`` through the regular runtime contract."""

    predictor: Any
    model_id: str
    lock: Any = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_name must be a nonempty string")
        if self.predictor.checkpoint is None:
            raise ValueError("serving letter readout requires a JevAny checkpoint")

    @property
    def checkpoint(self):
        return self.predictor.checkpoint

    @property
    def aliases(self) -> list[str]:
        return [] if self.model_id == "jevany-latest" else ["jevany-latest"]

    def clear_cache(self) -> None:
        """Letter readout currently keeps no mutable prefix cache."""

    def describe(self) -> dict[str, Any]:
        """Describe the effective letter readout and its checkpoint overlay."""

        checkpoint = self.checkpoint
        pointer_model = self.predictor.pointer_model
        capabilities = pointer_model.inference_capabilities
        context_window = capabilities.context_window
        effective_window = self.predictor.effective_max_tokens
        acceleration = getattr(pointer_model, "inference_acceleration", {
            "compile_mode": None,
            "lora_merged": False,
            "approximate_bf16_merge": False,
            "cuda_graphs": None,
        })
        letter_readout = dict(self.predictor.provenance)
        if self.predictor.pointer_weight:
            letter_readout["pointer_temperature"] = pointer_model.temperature
        with self.lock:
            return {
                "id": self.model_id,
                "aliases": self.aliases,
                "run": checkpoint.requested,
                "base": checkpoint.meta.base,
                "lora": checkpoint.meta.lora,
                "device": self.predictor.device,
                "device_map": getattr(pointer_model, "device_map", None),
                "devices": getattr(pointer_model, "devices", [self.predictor.device]),
                "temperature": self.predictor.temperature,
                "decision_mode": checkpoint.meta.decision_mode,
                "readout": "letter",
                "letter_readout": letter_readout,
                "backbone_adapter": pointer_model.backbone_adapter,
                "branch_mode": "chat",
                "acceleration": acceleration,
                "capabilities": {
                    "context_window": context_window,
                    "prefix_cache": False,
                    "media_types": [],
                    "max_media_questions": None,
                },
                "limits": {
                    "state_tokens": effective_window,
                    "branch_tokens": effective_window,
                    "packed_tokens": effective_window,
                    "choices": 26,
                },
                "prefix_cache": {
                    "enabled": False,
                    "size": 0,
                    "min_state_tokens": 0,
                    "hits": 0,
                    "misses": 0,
                    "cached_states": 0,
                },
            }

    def answer(self, request: SystemOneRequest) -> dict[str, Any]:
        """Run letter inference and return a validated System One response."""

        if request.model not in (self.model_id, *self.aliases):
            raise ValueError(f"unknown model {request.model!r}; this deployment serves {self.model_id!r}")
        if request.media:
            raise ValueError("letter readout does not support media requests")
        _, metadata = to_record(request)
        with self.lock:
            prediction = self.predictor(_unlabelled_record(request))
            rows = [
                [prediction["probabilities"][item["id"]][key] for key in item["keys"]]
                for item in metadata
            ]
            answers = to_answers(rows, metadata)
            response = {
                "model": self.model_id,
                "answers": answers,
                "usage": {
                    "input_tokens": prediction["input_tokens"],
                    "output_tokens": output_tokens(self.predictor.tokenizer, answers),
                },
                "latency_ms": prediction["latency_ms"],
            }
        return validate_response(request, response)


__all__ = ["LetterDecisionRuntime"]
