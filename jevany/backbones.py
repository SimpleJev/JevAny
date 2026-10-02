"""Backbone and tokenizer adapters shared by training and inference.

The default adapter uses Transformers' base-model interface. Custom adapters can
subclass ``BackboneAdapter`` and be selected by an import path (``module:Class``)
in both the training recipe and the saved checkpoint.
"""
import functools
import importlib
from dataclasses import dataclass

import torch
from torch import nn
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
    DynamicCache, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase,
)
from transformers.pytorch_utils import Conv1D


LEGACY_TOKENS = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
DECISION_TOKENS = ["<|jev_state|>", "<|jev_question|>", "<|jev_option|>", "<|jev_end_option|>", "<|jev_decide|>"]
EXISTING_TOKEN_ALPHABETS = (
    LEGACY_TOKENS,
    DECISION_TOKENS,
    [f"<unused{index}>" for index in range(5)],
    [f"<|reserved_special_token_{index}|>" for index in range(2, 7)],
)


@dataclass(frozen=True)
class InferenceCapabilities:
    """Backbone constraints, independent of deployment limits.

    Prefix reuse is optional: opt in only when cache construction, copying and
    reordering work, plus cropping for packed branches. Unknown context windows
    use the operator's token limits. Media adapters declare accepted types and
    their question limit (None means no extra limit).
    """

    context_window: int | None = None
    prefix_cache: bool = False
    media_types: tuple[str, ...] = ()
    max_media_questions: int | None = None

    def __post_init__(self) -> None:
        for name in ("context_window", "max_media_questions"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if type(self.prefix_cache) is not bool:
            raise ValueError("prefix_cache must be a bool")
        if not isinstance(self.media_types, tuple) or any(t not in ("image", "video") for t in self.media_types):
            raise ValueError("media_types must be a tuple containing image and/or video")


def decision_tokens(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Return this tokenizer's persisted decision alphabet."""
    explicit = getattr(tokenizer, "init_kwargs", {}).get("jevany_decision_tokens")
    if explicit is not None:
        if not isinstance(explicit, list) or len(explicit) != 5 or len(set(explicit)) != 5:
            raise ValueError("jevany_decision_tokens must contain five distinct strings")
        return explicit
    schema = getattr(tokenizer, "init_kwargs", {}).get("jevany_token_schema", "legacy")
    if schema not in ("legacy", "v1"):
        raise ValueError(f"unsupported JevAny tokenizer schema: {schema!r}")
    return LEGACY_TOKENS if schema == "legacy" else DECISION_TOKENS


def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Reuse legacy delimiters or add five model-independent special tokens."""
    vocabulary = tokenizer.get_vocab()
    explicit = tokenizer.init_kwargs.get("jevany_decision_tokens")
    if explicit is not None:
        tokens = decision_tokens(tokenizer)
        if any(token not in vocabulary for token in tokens):
            raise ValueError("saved tokenizer is missing its explicit JevAny decision tokens")
        if len({tokenizer.convert_tokens_to_ids(token) for token in tokens}) != len(tokens):
            raise ValueError("JevAny decision tokens must have distinct token IDs")
        return tokenizer
    schema = tokenizer.init_kwargs.get("jevany_token_schema")
    if schema is None:
        unknown = getattr(tokenizer, "unk_token_id", None)
        for candidates in EXISTING_TOKEN_ALPHABETS:
            ids = [tokenizer.convert_tokens_to_ids(token) for token in candidates]
            if all(value is not None and value != unknown for value in ids) and len(set(ids)) == 5:
                tokenizer.init_kwargs["jevany_decision_tokens"] = list(candidates)
                break
        else:
            tokenizer.init_kwargs["jevany_token_schema"] = "v1"
            tokenizer.add_special_tokens({"extra_special_tokens": DECISION_TOKENS},
                                         replace_extra_special_tokens=False)
            tokenizer.init_kwargs["jevany_added_tokens"] = list(DECISION_TOKENS)
    tokens = decision_tokens(tokenizer)
    vocabulary = tokenizer.get_vocab()
    if any(token not in vocabulary for token in tokens):
        raise ValueError("saved tokenizer is missing its JevAny decision tokens")
    if len({tokenizer.convert_tokens_to_ids(token) for token in tokens}) != len(tokens):
        raise ValueError("JevAny decision tokens must have distinct token IDs")
    return tokenizer


def prepare_embeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Resize the base and initialize newly added delimiters deterministically.

    Only these rows need training; PEFT saves their learned values in the adapter.
    Existing vocabulary rows, including a padded vocabulary tail, are retained.
    """
    embedding = model.get_input_embeddings()
    if not isinstance(embedding, nn.Embedding):
        raise ValueError("backbone must expose an nn.Embedding via get_input_embeddings()")
    added = tokenizer.init_kwargs.get("jevany_added_tokens", [])
    ids = [tokenizer.convert_tokens_to_ids(token) for token in added]
    size = max(tokenizer.get_vocab().values()) + 1
    if size > embedding.num_embeddings and not ids:
        raise ValueError("tokenizer vocabulary exceeds base embeddings without added-token metadata")
    if ids:
        with torch.no_grad():
            initial = embedding.weight.mean(dim=0)
            # Gemma E4B also looks up token IDs in a frozen per-layer table.
            # Its new rows are not in PEFT's main-embedding adapter, so rebuild
            # them deterministically whenever the checkpoint loads its base.
            per_layer = bool(getattr(model.config, "hidden_size_per_layer_input", 0))
            per_layer_initial = (model.get_per_layer_input_embeddings().weight.mean(dim=0)
                                 if per_layer else None)
            if size > embedding.num_embeddings:
                model.resize_token_embeddings(size, mean_resizing=False)
            model.get_input_embeddings().weight[ids] = initial
            if per_layer:
                model.get_per_layer_input_embeddings().weight[ids] = per_layer_initial
    return ids


def frozen_weight_options(config: dict) -> dict:
    """Load released FP8 bases as ordinary weights for portable LoRA training."""
    if (config.get("quantization_config") or {}).get("quant_method") == "fp8":
        from transformers import FineGrainedFP8Config
        return {"quantization_config": FineGrainedFP8Config(dequantize=True)}
    return {}


LINEAR_ATTENTION_KERNELS = {"flash_linear_attention": "fla", "causal_conv1d": "causal_conv1d"}


@functools.cache
def linear_attention_kernels() -> dict[str, bool]:
    """Whether the optional packages behind Transformers' gated-DeltaNet fast path import.

    Hybrid backbones (Qwen3.5/3.8) otherwise run the reference PyTorch
    recurrence, which is exact but several times slower on long states.
    Transformers resolves the same imports once, when its modeling module loads.
    """
    status = {}
    for name, module in LINEAR_ATTENTION_KERNELS.items():
        try:
            importlib.import_module(module)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def peft_expert_remapped_leaves(model_type: str) -> set[str]:
    """Linear leaf names PEFT rewrites to fused-expert parameters for this model type.

    PEFT's Transformers v5 MoE conversion redirects every target ending in these
    names, including dense and shared-expert ``nn.Linear`` layers, to 3-D expert
    parameters that reject LoRA dropout. Without these private names (a PEFT release
    that dropped the conversion) nothing is excluded, matching earlier behaviour.
    """
    try:
        from peft.utils.transformers_weight_conversion import (
            _MODEL_TO_CONVERSION_PATTERN, _MOE_TARGET_MODULE_MAPPING,
        )
    except ImportError:
        return set()
    return set(_MOE_TARGET_MODULE_MAPPING.get(_MODEL_TO_CONVERSION_PATTERN.get(model_type), {}))


class BackboneAdapter:
    """Transformers text backbone contract; override methods for other layouts.

    Models expose ``get_input_embeddings`` and return ``last_hidden_state`` from
    token IDs, position IDs and an attention mask. The default execution uses
    independent causal rows unless packed-mask support is known.
    """

    name = "text"
    media_types: frozenset[str] = frozenset()
    conditional_parameters = False

    def load_preprocessor(self, name: str, revision: str | None = None):
        return prepare_tokenizer(AutoTokenizer.from_pretrained(name, revision=revision, fix_mistral_regex=True))

    def load_model(self, name: str, *, revision: str | None, dtype: torch.dtype,
                   attn: str, placement: dict | None = None) -> tuple[PreTrainedModel, PreTrainedModel | None]:
        """Load the backbone. ``placement`` holds ``device_map``/``max_memory`` for ``from_pretrained`` when the
        caller shards the model over several GPUs; adapters that cannot shard may omit the parameter."""
        config = AutoConfig.from_pretrained(name, revision=revision)
        if config.is_encoder_decoder:
            raise ValueError("the text adapter requires a decoder-only text base; select a suitable backbone_adapter")
        kwargs = {**frozen_weight_options(config.to_dict()), **(placement or {})}
        if getattr(config, "text_config", None) is not None:
            native = AutoModel.from_pretrained(name, revision=revision, dtype=dtype,
                                                attn_implementation=attn, **kwargs)
            decoder = getattr(native, "language_model", None)
            if decoder is None:
                raise ValueError("composite base has no language_model; provide a custom backbone_adapter")
            return decoder, None
        causal = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype,
                                                     attn_implementation=attn, **kwargs)
        decoder = causal.base_model
        if decoder is causal:
            # Llama 4 declares a base_model_prefix that does not name its decoder attribute.
            decoder = getattr(causal, "model", None)
            if not isinstance(decoder, PreTrainedModel) or decoder is causal:
                raise ValueError("causal model does not expose its base_model; provide a custom backbone_adapter")
        self._output_embeddings = causal.get_output_embeddings()
        return decoder, None

    def output_embeddings(self, language_model, multimodal_model):
        """Return the base model's original frozen vocabulary projection."""
        owner = multimodal_model if multimodal_model is not None else language_model
        head = owner.get_output_embeddings()
        return head if head is not None else getattr(self, "_output_embeddings", None)

    def supports_packed(self, config) -> bool:
        layer_types = set(getattr(config, "layer_types", None) or [])
        return (
            config.model_type in {"llama", "mistral"}
            and not layer_types.difference({"full_attention"})
            and not getattr(config, "sliding_window", None)
        )

    def lora_modules(self, model: nn.Module, preset: str, explicit: str = "",
                     lora_dropout: float = 0.05) -> str | list[str]:
        """Resolve a preset against actual modules, including fused projections."""
        if preset not in ("all", "dense", "attn", "qv"):
            raise ValueError("lora_targets must be all, dense, attn, or qv")
        linear = {name for name, module in model.named_modules() if isinstance(module, (nn.Linear, Conv1D))}
        # Mamba's fused kernel bypasses out_proj.forward. PEFT 0.21 remaps even
        # GLM's fully qualified dense MLP targets to incompatible fused experts.
        # Llama 4's router subclasses nn.Linear but returns a tuple. Layers PEFT would
        # remap to fused experts only fail with dropout; zero-dropout runs keep PEFT's remap.
        excluded_leaves = {
            "nemotron_h": {"out_proj"},
            "glm4_moe_lite": {"gate_proj", "up_proj", "down_proj"},
            "llama4_text": {"router"},
        }.get(model.config.model_type, set())
        if lora_dropout:
            excluded_leaves = excluded_leaves | peft_expert_remapped_leaves(model.config.model_type)
        excluded = {name for name in linear if name.rsplit(".", 1)[-1] in excluded_leaves}
        if explicit:
            targets = [name.strip() for name in explicit.split(",")]
            missing = [target for target in targets
                       if not target or not any(name == target or name.endswith("." + target) for name in linear)]
            if missing:
                raise ValueError(f"lora_target_modules do not name linear layers in this backbone: {missing}")
            incompatible = [target for target in targets
                            if any(name == target or name.endswith("." + target) for name in excluded)]
            if incompatible:
                raise ValueError(f"{model.config.model_type} modules incompatible with LoRA: {incompatible}; "
                                 "use 'all', 'attn', or explicit supported projection names")
            return targets
        linear -= excluded
        if preset in ("all", "dense"):
            # Retain the released Qwen adapter layout, including its dense ablation.
            if model.config.model_type.startswith("qwen"):
                names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
                if preset == "all":
                    names.update({"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"})
                targets = sorted(name for name in linear if name.rsplit(".", 1)[-1] in names)
                if targets:
                    return targets
            # PEFT's all-linear shorthand also selects fused MoE parameters.
            # Keep those frozen; this contract targets linear modules only.
            if not linear:
                raise ValueError("backbone has no supported linear LoRA targets; provide a custom backbone_adapter")
            return sorted(linear)
        names = {"q_proj", "v_proj"} if preset == "qv" else {
            "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "out_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "in_proj",
            "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",
        }
        targets = sorted(name for name in linear if name.rsplit(".", 1)[-1] in names)
        if not targets:
            raise ValueError(f"lora_targets={preset!r} has no matching layers; use 'all' or explicit lora_target_modules")
        return targets

    def inference_capabilities(self, config) -> InferenceCapabilities:
        """Declare serving constraints; unknown architectures run without caching.

        Override for a different context layout or a model-specific cache. The
        cache allowlist is separate from packed-mask support.
        """
        return InferenceCapabilities(
            context_window=getattr(config, "max_position_embeddings", None),
            prefix_cache=config.model_type in {
                # Qwen's recurrent 27B path failed BF16 prefix/full-forward parity.
                "llama", "mistral", "gpt2",
            },
            media_types=tuple(sorted(self.media_types)),
            max_media_questions=1 if self.media_types else None,
        )

    def attach_language_model(self, multimodal_model, language_model) -> None:
        """Attach a PEFT-wrapped language model to a multimodal parent."""
        if multimodal_model is not None:
            raise ValueError("multimodal adapter must implement attach_language_model()")

    def frozen_modules(self, multimodal_model) -> tuple[nn.Module, ...]:
        return ()

    def new_cache(self, model: PreTrainedModel) -> DynamicCache:
        """Create a cache supporting deepcopy/reorder and, for packed mode, crop."""
        return DynamicCache(config=model.config)

    def encode_media(self, processor, record, **kwargs) -> dict:
        raise ValueError("this backbone adapter does not support media")

    def forward_media(self, language_model, multimodal_model, inputs: dict) -> torch.Tensor:
        """Return token hidden states, retaining each processor's model inputs."""
        model = multimodal_model if multimodal_model is not None else language_model
        return model(**inputs).last_hidden_state


def _video_inputs(processor, paths: list[str], *, num_frames: int) -> dict:
    """Decode with the declared PyAV dependency and retain native frame sampling."""
    if not paths:
        return {"videos": None}
    import numpy as np
    from transformers.video_utils import load_video

    video_processor = processor.video_processor
    videos, metadata = [], []
    for path in paths:
        indices = inverse = None

        def sample(metadata, **kwargs):
            nonlocal indices, inverse
            indices = np.asarray(video_processor.sample_frames(
                metadata=metadata, num_frames=num_frames, fps=None,
                temporal_patch_size=getattr(video_processor, "temporal_patch_size", None),
            ))
            # PyAV decodes each selected frame once; restore repetitions for short clips.
            unique, inverse = np.unique(indices, return_inverse=True)
            return unique

        frames, info = load_video(path, backend="pyav", sample_indices_fn=sample)
        videos.append(frames[inverse])
        info.frames_indices = indices
        metadata.append(info)
    return {"videos": videos, "video_metadata": metadata, "do_sample_frames": False}


class VisionAdapter(BackboneAdapter):
    """Native Transformers vision model with a separately addressable decoder.

    Subclasses declare the decoder path and supported media. Override
    ``process_media`` for another processor protocol or ``forward_media`` for
    models whose output layout differs from the Transformers base-model API.
    """

    media_types = frozenset({"image"})
    language_model_path = "language_model"

    def load_preprocessor(self, name: str, revision: str | None = None):
        processor = AutoProcessor.from_pretrained(name, revision=revision, fix_mistral_regex=True)
        # Some multimodal repositories expose a tokenizer-only AutoProcessor
        # fallback. Text inference must still work; native media calls will
        # fail later with the adapter's normal processor capability error.
        prepare_tokenizer(getattr(processor, "tokenizer", processor))
        return processor

    def load_model(self, name: str, *, revision: str | None, dtype: torch.dtype,
                   attn: str, placement: dict | None = None) -> tuple[PreTrainedModel, PreTrainedModel | None]:
        config, _ = PretrainedConfig.get_config_dict(name, revision=revision)
        model = AutoModel.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn,
                                          **frozen_weight_options(config), **(placement or {}))
        model.requires_grad_(False)
        self._output_embeddings = model.get_output_embeddings()
        if not self.language_model_path:
            return model, None
        try:
            decoder = model.get_submodule(self.language_model_path)
        except AttributeError as error:
            raise ValueError(f"{self.name} requires a {self.language_model_path} decoder") from error
        return decoder, model

    def attach_language_model(self, multimodal_model, language_model) -> None:
        if self.language_model_path:
            multimodal_model.set_submodule(self.language_model_path, language_model)

    def encode_media(self, processor, record, **kwargs) -> dict:
        unsupported = {item["type"] for item in record.get("media", [])} - self.media_types
        if unsupported:
            raise ValueError(f"{self.name} does not support {', '.join(sorted(unsupported))}; "
                             f"supported media: {', '.join(sorted(self.media_types))}")
        from .model import encode_multimodal
        return encode_multimodal(processor, record, adapter=self, **kwargs)

    def process_media(self, processor, media: list[dict], text: str):
        """Encode images with the native placeholder and image processor."""
        from PIL import Image
        images = []
        for item in media:
            with Image.open(item["uri"]) as image:
                images.append(image.convert("RGB"))
        return processor(text=[processor.image_token * len(images) + text],
                         images=images, return_tensors="pt")


class QwenVisionAdapter(VisionAdapter):
    """Preserve the released Qwen processor's image/video encoding."""

    name = "qwen_vl"
    media_types = frozenset({"image", "video"})

    def process_media(self, processor, media: list[dict], text: str):
        prefix, images, videos = [], [], []
        for item in media:
            token = processor.image_token if item["type"] == "image" else processor.video_token
            prefix.append(processor.vision_start_token + token + processor.vision_end_token)
            (images if item["type"] == "image" else videos).append(item["uri"])
        kwargs = {"return_tensors": "pt"}
        if images:
            kwargs["size"] = {"shortest_edge": 32 * 32,
                              "longest_edge": max(128 * 128, (512 * 512) // len(images))}
        if videos:
            kwargs.update(size={"shortest_edge": 32 * 32, "longest_edge": 512 * 512},
                          num_frames=8, fps=None, max_video_tokens=512, cap_pixels_per_frame=True)
        return processor(text=["".join(prefix) + text], images=images or None,
                         **_video_inputs(processor, videos, num_frames=8), **kwargs)


class LlamaVisionAdapter(VisionAdapter):
    name = "llama_vision"
    # Cross-attention LoRA is unused on text-only records.
    conditional_parameters = True

    def supports_packed(self, config) -> bool:
        return False


class Gemma4VisionAdapter(VisionAdapter):
    name = "gemma4_vision"
    media_types = frozenset({"image", "video"})

    def process_media(self, processor, media: list[dict], text: str):
        prefix, images, videos = [], [], []
        for item in media:
            prefix.append(processor.image_token if item["type"] == "image" else processor.video_token)
            (images if item["type"] == "image" else videos).append(item["uri"])
        return processor(text=["".join(prefix) + text], images=images or None,
                         **_video_inputs(processor, videos, num_frames=4),
                         return_tensors="pt", videos_kwargs={"num_frames": 4, "fps": None})


class MuseVisionAdapter(VisionAdapter):
    name = "muse_vision"
    media_types = frozenset({"image", "video"})

    def process_media(self, processor, media: list[dict], text: str):
        prefix, images, videos = [], [], []
        for item in media:
            prefix.append(processor.image_token if item["type"] == "image" else processor.video_token)
            (images if item["type"] == "image" else videos).append(item["uri"])
        return processor(
            text=["".join(prefix) + text], images=images or None,
            **_video_inputs(processor, videos, num_frames=8),
            return_tensors="pt", images_kwargs={"max_image_tokens": max(1, 512 // len(media))},
            videos_kwargs={"num_frames": 8, "fps": None, "return_metadata": True,
                           "max_video_frame_tokens": max(1, 128 // len(media))},
        )


class MistralVisionAdapter(VisionAdapter):
    name = "mistral_vision"


class GLMVisionAdapter(VisionAdapter):
    name = "glm_vision"
    media_types = frozenset({"image", "video"})

    def process_media(self, processor, media: list[dict], text: str):
        prefix, images, videos = [], [], []
        for item in media:
            kind = item["type"]
            token = processor.image_token if kind == "image" else processor.video_token
            prefix.append(f"<|begin_of_{kind}|>{token}<|end_of_{kind}|>")
            (images if kind == "image" else videos).append(item["uri"])
        return processor(
            text=["".join(prefix) + text], images=images or None,
            **_video_inputs(processor, videos, num_frames=8),
            return_tensors="pt", videos_kwargs={"num_frames": 8, "fps": None, "return_metadata": True},
        )


def get_backbone_adapter(name: str = "auto", *, multimodal: bool = False,
                         source: str | None = None, revision: str | None = None) -> BackboneAdapter:
    """Load a built-in adapter or an explicitly requested ``module:Class``."""
    if name == "auto":
        if multimodal and source is None:
            raise ValueError("source is required to select a native media adapter automatically")
        if multimodal and source is not None:
            config, _ = PretrainedConfig.get_config_dict(source, revision=revision)
            model_type = config.get("model_type")
            types = {"qwen3_5": "qwen_vl", "qwen3_5_moe": "qwen_vl",
                     "mllama": "llama_vision", "gemma4": "gemma4_vision",
                     "gemma4_unified": "gemma4_vision", "mistral3": "mistral_vision",
                     "muse_glimmer": "muse_vision", "glm4v": "glm_vision"}
            name = types.get(model_type)
            if name is None:
                raise ValueError(f"no built-in media adapter for {model_type!r}; "
                                 "provide backbone_adapter='module:Class'")
        else:
            name = "text"
    builtins = {"text": BackboneAdapter, "qwen_vl": QwenVisionAdapter,
                "llama_vision": LlamaVisionAdapter, "glm_vision": GLMVisionAdapter,
                "gemma4_vision": Gemma4VisionAdapter, "mistral_vision": MistralVisionAdapter,
                "muse_vision": MuseVisionAdapter}
    if name in builtins:
        if multimodal and name == "text":
            raise ValueError("backbone_adapter='text' cannot be used with multimodal=true")
        return builtins[name]()
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute:
        raise ValueError(f"backbone_adapter must be auto, {', '.join(builtins)}, or module:Class")
    adapter_class = getattr(importlib.import_module(module), attribute)
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, BackboneAdapter):
        raise ValueError(f"{name} must subclass jevany.backbones.BackboneAdapter")
    return adapter_class()
