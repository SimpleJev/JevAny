"""Placement flags shared by the command-line tools, without importing PyTorch.

`jevany decide --help` and `jevany decide ... --base-url` must work in the base
installation, so the argument definitions live here instead of in
`jevany.checkpoint`, which loads the modelling stack. `jevany.checkpoint`
re-exports both names, and turning the parsed flags into `LoadOptions` stays
there because that dataclass needs torch.
"""
import argparse

# Accelerate placement strategies accepted for LoadOptions.device_map.
DEVICE_MAPS = ("auto", "balanced", "balanced_low_0", "sequential")


def add_placement_arguments(parser: argparse.ArgumentParser) -> None:
    """--device-map / --max-memory-gib for the command-line tools; explicit flags override the environment."""
    parser.add_argument("--device-map", choices=DEVICE_MAPS, default=None,
                        help="split the backbone over all visible GPUs (Accelerate); overrides JEVANY_DEVICE_MAP")
    parser.add_argument("--max-memory-gib", type=float, default=None,
                        help="per-GPU weight budget with --device-map; overrides JEVANY_MAX_MEMORY_GIB")
