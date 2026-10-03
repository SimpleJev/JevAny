"""Lightweight command-line configuration for decision readouts.

This module deliberately has no modelling imports so ``jevany decide --help``
and the remote client path continue to work without PyTorch installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math


# ``letter`` is accepted indefinitely as the legacy spelling, but is omitted
# from generated usage/help so new integrations consistently say ``choice``.
READOUTS = ("native", "choice")
_ACCEPTED_READOUTS = (*READOUTS, "letter")


def normalize_readout(readout: str) -> str:
    """Return the public readout name while accepting the legacy alias."""

    if readout == "letter":
        return "choice"
    if readout not in READOUTS:
        raise ValueError("readout must be native or choice")
    return readout


@dataclass(frozen=True)
class ChoiceReadoutOptions:
    """Runtime settings for the training-free choice-token readout."""

    temperature: float = 1.0
    native_weight: float = 0.0
    max_tokens: int = 16_384

    def __post_init__(self) -> None:
        for name in ("temperature", "native_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"choice {name.replace('_', ' ')} must be finite and numeric")
        if self.temperature <= 0:
            raise ValueError("choice temperature must be positive")
        if not 0 <= self.native_weight <= 1:
            raise ValueError("choice native weight must be in [0, 1]")
        if type(self.max_tokens) is not int or self.max_tokens < 2:
            raise ValueError("choice max tokens must be an integer >= 2")

    @property
    def pointer_weight(self) -> float:
        """Legacy name for :attr:`native_weight`."""

        return self.native_weight


@dataclass(frozen=True)
class LetterReadoutOptions:
    """Legacy spelling of :class:`ChoiceReadoutOptions`."""

    temperature: float = 1.0
    pointer_weight: float = 0.0
    max_tokens: int = 16_384

    def __post_init__(self) -> None:
        # Validate through the public representation so both APIs have exactly
        # the same accepted values.
        ChoiceReadoutOptions(self.temperature, self.pointer_weight, self.max_tokens)

    @property
    def native_weight(self) -> float:
        return self.pointer_weight

    def as_choice(self) -> ChoiceReadoutOptions:
        return ChoiceReadoutOptions(self.temperature, self.pointer_weight, self.max_tokens)


def add_readout_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared readout arguments, with legacy flags accepted but hidden."""

    parser.add_argument(
        "--readout", choices=_ACCEPTED_READOUTS, metavar="{native,choice}", default="native",
        help="native checkpoint head (default) or training-free choice-token readout",
    )
    parser.add_argument(
        "--choice-temperature", type=float,
        help="probability temperature for choice readout (default 1.0)",
    )
    parser.add_argument(
        "--choice-native-weight", type=float,
        help="log-linear weight on the checkpoint's native distribution (default 0)",
    )
    parser.add_argument(
        "--choice-max-tokens", type=int,
        help="maximum chat-prompt tokens including the one-token answer (default 16384)",
    )
    # Backward-compatible spellings. argparse.SUPPRESS keeps the preferred
    # surface compact without removing existing scripts.
    parser.add_argument("--letter-temperature", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--letter-pointer-weight", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--letter-max-tokens", type=int, help=argparse.SUPPRESS)


def resolve_readout_options(
    readout: str = "native", *,
    choice_temperature: float | None = None,
    choice_native_weight: float | None = None,
    choice_max_tokens: int | None = None,
    letter_temperature: float | None = None,
    letter_pointer_weight: float | None = None,
    letter_max_tokens: int | None = None,
) -> tuple[str, ChoiceReadoutOptions | None]:
    """Normalize aliases and resolve preferred/legacy setting spellings."""

    normalized = normalize_readout(readout)
    pairs = (
        ("temperature", "--choice-temperature", choice_temperature,
         "--letter-temperature", letter_temperature),
        ("native_weight", "--choice-native-weight", choice_native_weight,
         "--letter-pointer-weight", letter_pointer_weight),
        ("max_tokens", "--choice-max-tokens", choice_max_tokens,
         "--letter-max-tokens", letter_max_tokens),
    )
    values, used = {}, []
    for name, preferred_flag, preferred, legacy_flag, legacy in pairs:
        if preferred is not None and legacy is not None:
            raise ValueError(f"give at most one of {preferred_flag} and legacy {legacy_flag}")
        if preferred is not None:
            values[name] = preferred
            used.append(preferred_flag)
        elif legacy is not None:
            values[name] = legacy
            used.append(legacy_flag)
    if normalized == "native":
        if used:
            raise ValueError(f"{', '.join(used)} require --readout choice")
        return normalized, None
    return normalized, ChoiceReadoutOptions(**values)


def choice_options_from_args(args: argparse.Namespace) -> ChoiceReadoutOptions | None:
    """Return preferred choice settings from a shared CLI namespace."""

    _, options = resolve_readout_options(
        getattr(args, "readout", "native"),
        choice_temperature=getattr(args, "choice_temperature", None),
        choice_native_weight=getattr(args, "choice_native_weight", None),
        choice_max_tokens=getattr(args, "choice_max_tokens", None),
        letter_temperature=getattr(args, "letter_temperature", None),
        letter_pointer_weight=getattr(args, "letter_pointer_weight", None),
        letter_max_tokens=getattr(args, "letter_max_tokens", None),
    )
    return options


def letter_options_from_args(args: argparse.Namespace) -> LetterReadoutOptions | None:
    """Legacy wrapper returning the historical options type."""

    options = choice_options_from_args(args)
    return (None if options is None else LetterReadoutOptions(
        options.temperature, options.native_weight, options.max_tokens,
    ))


__all__ = [
    "READOUTS", "ChoiceReadoutOptions", "LetterReadoutOptions",
    "add_readout_arguments", "choice_options_from_args", "letter_options_from_args",
    "normalize_readout", "resolve_readout_options",
]
