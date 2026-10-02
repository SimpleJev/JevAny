"""Lightweight command-line configuration for decision readouts.

This module deliberately has no modelling imports so ``jevany decide --help``
and the remote client path continue to work without PyTorch installed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math


READOUTS = ("native", "letter")


@dataclass(frozen=True)
class LetterReadoutOptions:
    """Runtime settings for the training-free option-letter readout."""

    temperature: float = 1.0
    pointer_weight: float = 0.0
    max_tokens: int = 16_384

    def __post_init__(self) -> None:
        for name in ("temperature", "pointer_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"letter {name.replace('_', ' ')} must be finite and numeric")
        if self.temperature <= 0:
            raise ValueError("letter temperature must be positive")
        if not 0 <= self.pointer_weight <= 1:
            raise ValueError("letter pointer weight must be in [0, 1]")
        if type(self.max_tokens) is not int or self.max_tokens < 2:
            raise ValueError("letter max tokens must be an integer >= 2")


def add_readout_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same readout selector and letter-only knobs to a CLI parser."""

    parser.add_argument(
        "--readout", choices=READOUTS, default="native",
        help="native checkpoint head (default) or training-free option-letter readout",
    )
    parser.add_argument(
        "--letter-temperature", type=float,
        help="temperature for letter probability calibration (letter readout only; default 1.0)",
    )
    parser.add_argument(
        "--letter-pointer-weight", type=float,
        help="log-linear weight on the native pointer distribution (letter readout only; default 0)",
    )
    parser.add_argument(
        "--letter-max-tokens", type=int,
        help="maximum chat-prompt tokens including the one-letter answer (letter readout only; default 16384)",
    )


def letter_options_from_args(args: argparse.Namespace) -> LetterReadoutOptions | None:
    """Return letter settings, rejecting letter-only flags with native readout."""

    values = {
        "temperature": getattr(args, "letter_temperature", None),
        "pointer_weight": getattr(args, "letter_pointer_weight", None),
        "max_tokens": getattr(args, "letter_max_tokens", None),
    }
    if getattr(args, "readout", "native") == "native":
        used = ["--letter-" + name.replace("_", "-") for name, value in values.items() if value is not None]
        if used:
            raise ValueError(f"{', '.join(used)} require --readout letter")
        return None
    return LetterReadoutOptions(**{
        name: value for name, value in values.items() if value is not None
    })


__all__ = [
    "READOUTS", "LetterReadoutOptions", "add_readout_arguments", "letter_options_from_args",
]
