"""Training-free choice-token inference over a frozen base or JevAny adapter.

This is the in-process backend for :mod:`jevany.letter_readout`.  It keeps the
existing pointer path unchanged: a JevAny checkpoint can instead be applied to
the chat prompt before the frozen vocabulary readout, and its pointer
distribution can optionally be combined with the letter distribution.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .api import SystemOneRequest, question_keys
from .checkpoint import Checkpoint, LoadOptions
from .data import api_request, materialize
from .device import sync
from .letter_readout import (
    CHOICE_SYMBOLS,
    SYSTEM_PROMPT,
    build_prompt,
    letter_token_ids,
    read_letter_distribution,
)
from .model import ContextLengthError, DecisionModel, load_preprocessor, safe_text, tokenizer_of
from .suite import digest


def _one_row_input_ids(value) -> torch.Tensor:
    """Normalize tokenizer/template output to one two-dimensional ID row."""

    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value, dtype=torch.long)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("chat template did not return input_ids") from error
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] != 1 or value.shape[1] == 0:
        raise ValueError("chat template did not return one input_ids row")
    return value


def _safe_chat_text(tokenizer, text: str) -> str:
    """Escape caller text that an arbitrary chat tokenizer treats as control."""

    escaped = safe_text(tokenizer, text)
    for token in getattr(tokenizer, "all_special_tokens", ()):
        if isinstance(token, str) and token.startswith(("<", "[")):
            replacement = token.replace("<", "‹").replace("[", "［")
            escaped = escaped.replace(token, replacement)
    return escaped


def _release_unused_output_head(decision_model, *, retain_native: bool = False) -> bool:
    """Drop the full-vocabulary head unless a native LM-token pass needs it."""

    adapter = getattr(decision_model, "adapter", None)
    adapter_head = getattr(adapter, "_output_embeddings", None)
    native_head = getattr(decision_model, "lm_head", None)
    if retain_native:
        return False
    if adapter_head is None and native_head is None:
        return False
    # Exact choice rows are copied from the tied embedding table or loaded
    # independently from safetensors.  With no LM-token native pass, both
    # aliases can therefore be detached to release the full vocabulary head.
    if native_head is not None:
        decision_model.lm_head = None
    if adapter_head is not None:
        adapter._output_embeddings = None
    return True


def question_options(question: dict) -> tuple[list[str], list[object]]:
    """Return response keys and Cygnet-compatible descriptions in one order."""

    qtype = question.get("type")
    criteria = question.get("criteria")
    keys = question_keys(qtype, criteria)
    if qtype == "choice":
        if not isinstance(criteria, dict):
            raise ValueError("choice criteria must be a mapping")
        descriptions = []
        for key, description in criteria.items():
            if description is None or (isinstance(description, str) and not description.strip()):
                descriptions.append(str(key))
            elif isinstance(description, str):
                descriptions.append(description)
            else:
                descriptions.append(f"{key}: {json.dumps(description, ensure_ascii=False)}")
        return keys, descriptions
    if qtype == "score":
        if not isinstance(criteria, list):
            raise ValueError("score criteria must be a list")
        descriptions = []
        for index, description in enumerate(criteria):
            if description is None or (isinstance(description, str) and not description.strip()):
                descriptions.append(f"Level {index}")
            elif isinstance(description, str):
                descriptions.append(description)
            else:
                descriptions.append(json.dumps(description, ensure_ascii=False))
        return keys, descriptions
    if qtype != "noul":
        raise ValueError(f"unsupported question type: {qtype!r}")
    criteria = criteria or {}
    if not isinstance(criteria, dict):
        raise ValueError("noul criteria must be a mapping when provided")
    # The application form of Cygnet fixes this order.  It also matches
    # JevAny's question_keys contract and makes P(true) unambiguous.
    descriptions = []
    for key, fallback in (("false", "No"), ("true", "Yes")):
        description = criteria.get(key)
        if description is None or (isinstance(description, str) and not description.strip()):
            descriptions.append(fallback)
        elif isinstance(description, str):
            descriptions.append(description)
        else:
            descriptions.append(json.dumps(description, ensure_ascii=False))
    return keys, descriptions


def geometric_blend(left: list[float], right: list[float], right_weight: float) -> tuple[list[float], list[float]]:
    """Log-linear pool two complete distributions and return probabilities/logits."""

    if isinstance(right_weight, bool) or not isinstance(right_weight, (int, float)):
        raise TypeError("pointer weight must be numeric")
    right_weight = float(right_weight)
    if not math.isfinite(right_weight) or not 0 <= right_weight <= 1:
        raise ValueError("pointer weight must be finite and in [0, 1]")
    if len(left) != len(right) or not left:
        raise ValueError("blended distributions must have the same non-zero length")
    for values in (left, right):
        if any(isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0 for value in values):
            raise ValueError("blended probabilities must be finite and non-negative")
        if not math.isclose(sum(float(value) for value in values), 1.0, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("blended probabilities must sum to one")
    floor = 1e-12
    logits = [
        (1 - right_weight) * math.log(max(float(a), floor))
        + right_weight * math.log(max(float(b), floor))
        for a, b in zip(left, right, strict=True)
    ]
    pivot = max(logits)
    weights = [math.exp(value - pivot) for value in logits]
    total = sum(weights)
    return [value / total for value in weights], logits


def _artifact_file(source: str | Path, revision: str | None, filename: str) -> Path:
    root = Path(source)
    if root.is_dir():
        path = root / filename
        if not path.is_file():
            raise ValueError(f"base model is missing {filename}: {root}")
        return path
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(str(source), filename, revision=revision))


def _untied_output_rows(
    source: str | Path,
    revision: str | None,
    token_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Load only the needed rows from an untied safetensors LM head.

    Safetensors indexed slices read only the selected rows; only the small
    exact-choice projection is retained on the accelerator.
    """

    root = Path(source)
    index_path = root / "model.safetensors.index.json" if root.is_dir() else None
    if index_path is None:
        from huggingface_hub.errors import EntryNotFoundError
        try:
            index_path = _artifact_file(source, revision, "model.safetensors.index.json")
        except EntryNotFoundError:
            index_path = None
    elif not index_path.is_file():
        index_path = None

    from safetensors import safe_open

    if index_path is not None:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("model weight index has no weight_map")
    else:
        single = _artifact_file(source, revision, "model.safetensors")
        with safe_open(single, framework="pt", device="cpu") as tensors:
            weight_map = {name: "model.safetensors" for name in tensors.keys()}

    weights = [name for name in weight_map if name == "lm_head.weight" or name.endswith(".lm_head.weight")]
    if len(weights) != 1:
        raise ValueError(f"expected one untied lm_head.weight in model weights, found {weights}")

    def selected(name):
        shard = _artifact_file(source, revision, weight_map[name])
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            # ``get_tensor`` materializes Qwen3.8-27B's complete 2.4 GiB head.
            # Safetensors slices perform indexed row reads and retain only the
            # exact choice-token projection we need.
            return tensors.get_slice(name)[token_ids].clone()

    rows = selected(weights[0])
    biases = [name for name in weight_map if name == "lm_head.bias" or name.endswith(".lm_head.bias")]
    if len(biases) > 1:
        raise ValueError(f"expected at most one lm_head.bias in the model index, found {biases}")
    return rows, selected(biases[0]) if biases else None


class LetterReadoutPredictor:
    """Benchmark predictor for an exact constrained first-token readout.

    Give either ``base`` for a frozen training-free model, or ``checkpoint``
    to apply a JevAny LoRA before the same readout.  ``native_weight > 0``
    additionally pools the checkpoint's native probabilities in log space;
    this costs one extra native-formatted prefill per request.  The legacy
    ``pointer_weight`` spelling remains an alias for ``native_weight``.
    """

    def __init__(
        self,
        *,
        base: str | Path | None = None,
        checkpoint: str | Path | None = None,
        device: str = "cuda",
        options: LoadOptions | None = None,
        revision: str | None = None,
        temperature: float = 1.0,
        pointer_weight: float | None = None,
        native_weight: float | None = None,
        return_components: bool = False,
        exact_kernels: bool = True,
        efficient_long_context_tokens: int | None = None,
        max_tokens: int = 16_384,
    ) -> None:
        if (base is None) == (checkpoint is None):
            raise ValueError("give exactly one of base or checkpoint")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be numeric")
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if pointer_weight is not None and native_weight is not None:
            raise ValueError("give at most one of native_weight or pointer_weight")
        weight_source = (
            "native_weight" if native_weight is not None
            else "pointer_weight" if pointer_weight is not None
            else "default"
        )
        selected_weight = (
            native_weight if native_weight is not None
            else pointer_weight if pointer_weight is not None
            else 0.0
        )
        # Reuse the same validation as the actual combination path.
        geometric_blend([1.0], [1.0], selected_weight)
        self.native_weight = float(selected_weight)
        # Public compatibility alias for existing runtime/CLI integrations.
        self.pointer_weight = self.native_weight
        if checkpoint is None and self.native_weight:
            raise ValueError("native_weight requires a JevAny checkpoint")
        if type(return_components) is not bool:
            raise TypeError("return_components must be a boolean")
        if type(exact_kernels) is not bool:
            raise TypeError("exact_kernels must be a boolean")
        self.return_components = return_components
        self.exact_kernels = exact_kernels
        self.exact_cuda_kernels_applied = str(device).startswith("cuda") and exact_kernels
        if (
            efficient_long_context_tokens is not None
            and (
                type(efficient_long_context_tokens) is not int
                or efficient_long_context_tokens < 2
            )
        ):
            raise ValueError("efficient_long_context_tokens must be an integer >= 2 or None")
        self.efficient_long_context_tokens = efficient_long_context_tokens
        if type(max_tokens) is not int or max_tokens < 2:
            raise ValueError("max_tokens must be an integer >= 2")
        self.max_tokens = max_tokens
        self.device = device
        self.options = options or LoadOptions()
        if self.options.temperature is not None:
            raise ValueError("native-head temperature is not used by choice readout; use temperature")
        if self.options.cuda_graphs:
            raise ValueError("CUDA graph capture is available only for native readout")
        if self.exact_cuda_kernels_applied:
            # Match LocalPredictor: benchmark evaluation disables approximate
            # TF32 and fused SDPA kernels, while exact_kernels=False retains
            # the serving-style PyTorch defaults.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        self.checkpoint: Checkpoint | None = None
        self.native_model = None
        self.native_decision_mode = None
        self._needs_native = checkpoint is not None and bool(
            self.native_weight or self.return_components
        )
        # Compatibility alias retained for callers predating generalized
        # native blending.
        self.pointer_model = None

        if checkpoint is not None:
            if revision is not None:
                raise ValueError("revision accompanies a base; pin checkpoint revisions in owner/repo@revision")
            loaded = self.checkpoint = Checkpoint(checkpoint)
            if loaded.meta.special_embeddings:
                raise ValueError("choice readout is not validated for checkpoints with trained special embeddings")
            self.preprocessor, decision_model = loaded.load(device, self.options)
            self.native_model = decision_model
            self.pointer_model = decision_model
            self.native_decision_mode = loaded.meta.decision_mode
            self.language_model = decision_model.lm
            canonical_base = loaded.meta.base
            canonical_revision = loaded.meta.base_revision
            projection_source = self.options.base_load_path or canonical_base
            projection_revision = None if self.options.base_load_path else canonical_revision
            checkpoint_id = loaded.requested
            adapter_scale = self.options.lora_scale
            adapter_applied = adapter_scale != 0
        else:
            source = str(self.options.base_load_path or base)
            source_revision = None if self.options.base_load_path else revision
            self.preprocessor = load_preprocessor(source, source_revision)
            dtype = self.options.dtype or (torch.bfloat16 if str(device).startswith("cuda") else torch.float32)
            decision_model = DecisionModel(
                source,
                self.preprocessor,
                device,
                revision=source_revision,
                dtype=dtype,
                attn=self.options.attn,
                branch_mode="rows",
                decision_mode="pointer",
                device_map=self.options.device_map,
                max_memory_gib=self.options.max_memory_gib,
            )
            decision_model.eval()
            self.language_model = decision_model.lm
            canonical_base, canonical_revision = str(base), revision
            projection_source, projection_revision = source, source_revision
            checkpoint_id, adapter_scale, adapter_applied = None, 0.0, False

        released_output_head = _release_unused_output_head(
            decision_model,
            retain_native=(self._needs_native and self.native_decision_mode == "lm_token"),
        )
        self.language_model.eval()
        override_config = (
            Path(self.options.base_load_path) / "config.json"
            if self.options.base_load_path else None
        )
        canonical_base_text = str(canonical_base)
        canonical_base_is_local = Path(canonical_base_text).is_absolute()
        self.base_loading = {
            "canonical_base": (
                Path(canonical_base_text).name if canonical_base_is_local else canonical_base_text
            ),
            "canonical_base_is_local": canonical_base_is_local,
            "canonical_revision": canonical_revision,
            "override_used": bool(self.options.base_load_path),
            "override_config_sha256": (
                digest(override_config) if override_config and override_config.is_file() else None
            ),
        }
        context_window = getattr(decision_model.inference_capabilities, "context_window", None)
        if context_window is not None and (type(context_window) is not int or context_window < 2):
            raise ValueError("backbone context window must be an integer >= 2")
        self.backbone_context_window = context_window
        self.effective_max_tokens = (
            min(self.max_tokens, context_window) if context_window is not None else self.max_tokens
        )
        self.tokenizer = tokenizer_of(self.preprocessor)
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if isinstance(chat_template, dict):
            chat_template = json.dumps(chat_template, ensure_ascii=False, sort_keys=True)
        self.chat_template_sha256 = (
            hashlib.sha256(chat_template.encode()).hexdigest()
            if isinstance(chat_template, str) else None
        )
        self.alias_token_ids = letter_token_ids(self.tokenizer, CHOICE_SYMBOLS)
        flat_token_ids, alias_rows = [], {}
        for letter, token_ids in self.alias_token_ids.items():
            start = len(flat_token_ids)
            flat_token_ids.extend(token_ids)
            alias_rows[letter] = tuple(range(start, len(flat_token_ids)))
        tied = bool(getattr(self.language_model.config, "tie_word_embeddings", False))
        if tied:
            embeddings = self.language_model.get_input_embeddings().weight
            if max(flat_token_ids) >= embeddings.shape[0]:
                raise ValueError("choice token ID exceeds the tied vocabulary projection")
            index = torch.tensor(flat_token_ids, dtype=torch.long, device=embeddings.device)
            projection = embeddings.index_select(0, index).detach().clone()
            projection_bias = None
        else:
            projection, projection_bias = _untied_output_rows(
                projection_source, projection_revision, flat_token_ids
            )
        model_dtype = next(self.language_model.parameters()).dtype
        self.letter_projection = projection.to(device=self.device, dtype=model_dtype)
        self.letter_bias = (projection_bias.to(device=self.device, dtype=model_dtype)
                            if projection_bias is not None else None)
        self.alias_rows = alias_rows
        self.output_softcap = getattr(self.language_model.config, "final_logit_softcapping", None)
        if self.output_softcap is not None:
            self.output_softcap = float(self.output_softcap)
            if not math.isfinite(self.output_softcap) or self.output_softcap <= 0:
                raise ValueError("final_logit_softcapping must be finite and positive")
        self.provenance = {
            "method": "training-free exact choice-token projection",
            "constraint_emulation": "exact decoded one-character option IDs",
            "prompt": "Cygnet-compatible A-Z; extended with a-z above 26 options",
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "chat_template_sha256": self.chat_template_sha256,
            "prompt_format_version": 2,
            "canonical_base": canonical_base,
            "canonical_revision": canonical_revision,
            "checkpoint": checkpoint_id,
            "adapter_applied": adapter_applied,
            "adapter_scale": adapter_scale,
            "tied_output_embeddings": tied,
            "unused_full_output_head_released": released_output_head,
            "choice_token_rows": len(flat_token_ids),
            "letter_choice_token_rows": len(flat_token_ids),
            "output_softcap": self.output_softcap,
            "temperature": self.temperature,
            "native_weight": self.native_weight,
            "native_weight_argument": weight_source,
            "native_decision_mode": self.native_decision_mode,
            "pointer_weight": self.pointer_weight,
            "native_temperature": (
                float(self.native_model.temperature) if self._needs_native else None
            ),
            "pointer_temperature": (
                float(self.native_model.temperature) if self._needs_native else None
            ),
            "return_components": self.return_components,
            "exact_kernels": self.exact_kernels,
            "exact_cuda_kernels_applied": self.exact_cuda_kernels_applied,
            "efficient_long_context_tokens": self.efficient_long_context_tokens,
            "long_context_attention": (
                "fused SDPA (Flash/Efficient, no math fallback) at or above the "
                "recorded threshold; exact math SDPA below it"
                if self.efficient_long_context_tokens is not None
                else None
            ),
            "max_tokens": self.max_tokens,
            "backbone_context_window": self.backbone_context_window,
            "effective_max_tokens": self.effective_max_tokens,
            "dtype": str(next(self.language_model.parameters()).dtype),
            "device": device,
        }

    def _uses_efficient_attention(self, token_count: int) -> bool:
        return (
            str(self.device).startswith("cuda")
            and getattr(self, "efficient_long_context_tokens", None) is not None
            and token_count >= self.efficient_long_context_tokens
        )

    def _attention_context(self, token_count: int):
        if not self._uses_efficient_attention(token_count):
            return nullcontext()
        # Exact math SDPA materializes an O(sequence^2) attention matrix.  A
        # few public JevJudge text records exceed 30K tokens, which can require
        # roughly 60 GiB for that matrix alone.  Release cached short-record
        # workspaces and force a memory-linear fused backend for those records.
        torch.cuda.empty_cache()
        return sdpa_kernel(
            [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION],
            set_priority=True,
        )

    def _chat_ids(self, prompt: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _safe_chat_text(self.tokenizer, prompt)},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        ids = _one_row_input_ids(ids)
        if ids.shape[1] + 1 > self.effective_max_tokens:
            raise ContextLengthError(
                f"choice prompt needs {ids.shape[1] + 1} tokens including the answer; "
                f"limit {self.effective_max_tokens}"
            )
        return ids.to(self.device)

    def _letter_question(self, state, question: dict) -> tuple[list[str], list[float], list[float], int]:
        keys, descriptions = question_options(question)
        if len(keys) > len(CHOICE_SYMBOLS):
            raise ValueError(
                f"exact choice-token readout supports at most {len(CHOICE_SYMBOLS)} options"
            )
        prompt = build_prompt(state, question.get("instructions") or "", descriptions)
        ids = self._chat_ids(prompt)
        with self._attention_context(int(ids.shape[1])):
            output = self.language_model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
            )
        hidden = output.last_hidden_state[0, -1]
        projection = self.letter_projection
        selected_logits = F.linear(
            hidden.to(projection.device, projection.dtype), projection, self.letter_bias
        ).float()
        if self.output_softcap is not None:
            selected_logits = self.output_softcap * torch.tanh(selected_logits / self.output_softcap)
        selected_logits = selected_logits.cpu()
        aliases = {
            symbol: self.alias_rows[symbol] for symbol in CHOICE_SYMBOLS[:len(keys)]
        }
        readout = read_letter_distribution(selected_logits, aliases, self.temperature)
        probabilities = [readout.calibrated_probabilities[letter] for letter in aliases]
        calibrated_logits = [readout.raw_log_masses[letter] / self.temperature for letter in aliases]
        return keys, probabilities, calibrated_logits, int(ids.shape[1])

    def _native(self, record: dict) -> tuple[list[list[float]], int]:
        internal = materialize(record)
        encoded = self.native_model.encode(
            self.preprocessor,
            internal,
            max_state=self.effective_max_tokens,
            max_branch=self.effective_max_tokens,
            strict=True,
        )
        if len(encoded["ids"]) > self.effective_max_tokens:
            raise ContextLengthError(
                f"native request needs {len(encoded['ids'])} packed tokens; "
                f"limit {self.effective_max_tokens}"
            )
        with self._attention_context(len(encoded["ids"])):
            rows = [F.softmax(logits, -1).float().cpu().tolist()
                    for logits in self.native_model.forward(encoded)]
        return rows, len(encoded["ids"])

    def _pointer(self, record: dict) -> tuple[list[list[float]], int]:
        """Compatibility alias for the generalized native checkpoint pass."""

        return self._native(record)

    @torch.inference_mode()
    def __call__(self, record: dict) -> dict:
        request = api_request(record)
        SystemOneRequest.model_validate(request)
        if request.get("media"):
            raise ValueError("choice readout is text-only and does not accept media")
        sync(self.device)
        started = time.perf_counter()
        letter_rows = [
            self._letter_question(request["state"], question)
            for question in request["questions"].values()
        ]
        native_rows, native_tokens = (None, 0)
        if self._needs_native:
            native_rows, native_tokens = self._native(record)
            if len(native_rows) != len(letter_rows):
                raise ValueError("native and choice question counts differ")

        probabilities, logits = {}, {}
        choice_probabilities, native_probabilities = {}, {}
        for index, (question_id, (keys, letter_p, letter_z, _tokens)) in enumerate(
            zip(request["questions"], letter_rows, strict=True)
        ):
            choice_probabilities[question_id] = dict(zip(keys, letter_p, strict=True))
            if native_rows is not None:
                if len(native_rows[index]) != len(keys):
                    raise ValueError(
                        f"native and choice option counts differ for question {question_id!r}"
                    )
                native_probabilities[question_id] = dict(
                    zip(keys, native_rows[index], strict=True)
                )
            if native_rows is None or not self.native_weight:
                selected_p, selected_z = letter_p, letter_z
            else:
                selected_p, selected_z = geometric_blend(
                    letter_p, native_rows[index], self.native_weight
                )
            probabilities[question_id] = dict(zip(keys, selected_p, strict=True))
            logits[question_id] = dict(zip(keys, selected_z, strict=True))
        sync(self.device)
        result = {
            "probabilities": probabilities,
            "logits": logits,
            "inference_temperature": self.temperature,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "input_tokens": sum(row[3] for row in letter_rows) + native_tokens,
            "efficient_long_context_attention_used": (
                any(self._uses_efficient_attention(row[3]) for row in letter_rows)
                or self._uses_efficient_attention(native_tokens)
            ),
            "readout": {
                "method": self.provenance["method"],
                "adapter_applied": self.provenance["adapter_applied"],
                "native_weight": self.native_weight,
                "native_decision_mode": self.native_decision_mode,
                "pointer_weight": self.pointer_weight,
            },
        }
        if self.return_components:
            result["component_probabilities"] = {"choice": choice_probabilities}
            if native_rows is not None:
                result["component_probabilities"]["native"] = native_probabilities
        return result


__all__ = ["LetterReadoutPredictor", "geometric_blend", "question_options"]
