"""Configuration and Python entry point for the existing SFT/RLCR trainer."""
import argparse
import math
import tomllib
from pathlib import Path


class TrainingArgumentParser(argparse.ArgumentParser):
    """Report invalid Python API configuration without exiting the interpreter."""

    def error(self, message: str) -> None:
        raise ValueError(message)


def configure_parser(parser: argparse.ArgumentParser, argv: list[str] | None) -> argparse.Namespace:
    """Apply a flat TOML recipe, then CLI overrides; reject unknown or mistyped keys."""
    # Accept readable CLI spelling while preserving the original research scripts.
    for action in parser._actions:
        for flag in list(action.option_strings):
            if flag.startswith("--") and "_" in flag:
                alias = flag.replace("_", "-")
                if alias not in action.option_strings:
                    action.option_strings.append(alias)
                parser._option_string_actions[alias] = action
    probe = type(parser)(add_help=False)
    probe.add_argument("--config")
    config = probe.parse_known_args(argv)[0].config
    if config:
        try:
            with Path(config).open("rb") as source:
                settings = tomllib.load(source)
            actions = {action.dest: action for action in parser._actions}
            for key, value in settings.items():
                if key not in actions or key in ("help", "config", "dry_run"):
                    raise ValueError(f"unknown training setting: {key}")
                action = actions[key]
                expected = bool if isinstance(action, (argparse._StoreTrueAction, argparse.BooleanOptionalAction)) else action.type or str
                valid_type = type(value) is expected or (expected is float and type(value) is int)
                if not valid_type:
                    raise ValueError(f"{key} must be {expected.__name__}")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{key} must be finite")
                if action.choices is not None and value not in action.choices:
                    raise ValueError(f"{key} must be one of {list(action.choices)}")
            parser.set_defaults(**settings)
        except (OSError, ValueError) as error:
            parser.error(f"{config}: {error}")
    return parser.parse_args(argv)


def train(config: str | Path, *, output_dir: str | Path | None = None) -> Path:
    """Train from a TOML recipe and return the final checkpoint directory.

    Paths in recipes resolve from the current working directory, just like CLI
    arguments. Use torchrun for distributed training. Existing runs are never
    overwritten. init_from starts a new optimizer; resume restores the original
    schedule and requires a new output directory.

    Invalid settings raise ValueError; missing input files raise OSError.
    """
    from .train import main

    argv = ["--config", str(config)]
    if output_dir is not None:
        argv += ["--out", str(output_dir)]
    return main(argv, parser_class=TrainingArgumentParser)
