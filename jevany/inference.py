"""Inference limits shared by Python, the CLI and HTTP deployments."""
import argparse
import os
from dataclasses import dataclass, fields
from typing import Mapping


def _env_name(field_name: str) -> str:
    return "JEVANY_PREFIX_CACHE" if field_name == "prefix_cache_size" else "JEVANY_" + field_name.upper()


@dataclass(frozen=True)
class InferenceOptions:
    """Admission and cache limits for one loaded model.

    Token limits reject oversized inputs rather than truncating them. State and
    branch limits are also bounded by the backbone's context window. A branch
    includes the state; the packed limit covers the entire encoded request.
    Set prefix_cache_size=0 to disable prefix reuse. max_batch_size > 1 lets
    concurrent text requests share one forward pass; 1 scores each alone.
    """

    max_state_tokens: int = 8192
    max_branch_tokens: int = 8192
    max_packed_tokens: int = 8192
    prefix_cache_size: int = 4
    prefix_min_tokens: int = 384
    max_batch_size: int = 1

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            minimum = 0 if item.name.startswith("prefix_") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{item.name} must be an integer >= {minimum}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "InferenceOptions":
        """Read JEVANY_MAX_* and the existing JEVANY_PREFIX_* settings."""
        env = os.environ if env is None else env
        names = {item.name: _env_name(item.name) for item in fields(cls)}
        settings = {}
        for field_name, variable in names.items():
            if variable in env:
                try:
                    settings[field_name] = int(env[variable])
                except ValueError as error:
                    raise ValueError(f"{variable} must be an integer") from error
        return cls(**settings)


def add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same optional limit overrides to serve and local decide."""
    for item in fields(InferenceOptions):
        parser.add_argument("--" + item.name.replace("_", "-"), type=int, default=None,
                            help=f"override {item.name}; otherwise use JEVANY_* or runtime defaults")


def inference_options_from_args(args: argparse.Namespace) -> InferenceOptions:
    """CLI overrides take precedence over environment settings."""
    overrides = {_env_name(item.name): str(getattr(args, item.name)) for item in fields(InferenceOptions)
                 if getattr(args, item.name) is not None}
    return InferenceOptions.from_env({**os.environ, **overrides})


MEDIA_BUDGETS = ("capped", "full")


def media_budget() -> str:
    """Media encoding for Qwen vision backbones, from JEVANY_MEDIA (``jevany serve/decide --media``).

    ``capped`` (default) keeps the released encoding: the images of a request share a 512x512 pixel budget and a
    video is 8 frames within 512x512 pixels. ``full`` gives images the processor's native resolution and keeps every
    frame of a video at its native resolution, the protocol used for the JevAny-Robotics evaluations.
    """
    value = os.environ.get("JEVANY_MEDIA", "capped").strip().lower()
    if value not in MEDIA_BUDGETS:
        raise ValueError(f"JEVANY_MEDIA must be one of {', '.join(MEDIA_BUDGETS)}, got {value!r}")
    return value


def add_media_argument(parser: argparse.ArgumentParser) -> None:
    """--media for serve and local decide; same as JEVANY_MEDIA (see jevany.backbones.media_budget)."""
    parser.add_argument("--media", choices=["capped", "full"], default=None,
                        help="Qwen vision media encoding: capped (default, 512x512 budget, 8 video frames) or full "
                             "(native resolution, every video frame); same as JEVANY_MEDIA")


def apply_media_argument(args: argparse.Namespace) -> str:
    """Export --media before weights load and validate the effective setting."""
    if getattr(args, "media", None) is not None:
        os.environ["JEVANY_MEDIA"] = args.media
    return media_budget()
