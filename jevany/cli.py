"""The `jevany` command: train, serve, decide, explore demos, and prepare data."""
import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="jevany", description="Train and deploy Jev-style decision models.")
    from . import __version__
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=["train", "serve", "decide", "eval", "data", "demo"])
    if not argv or argv[0] in ("-h", "--help", "--version"):
        parser.parse_args(argv or ["--help"])
        return
    command = parser.parse_args(argv[:1]).command
    rest = argv[1:]
    try:
        if command == "train":
            from .train import main as train
            train(rest)
        elif command == "serve":
            from .serve import main as serve
            serve(rest)
        elif command == "eval":
            from .benchmark import main as evaluate
            evaluate(rest, prog="jevany eval")
        elif command == "data":
            data_main(rest)
        elif command == "demo":
            from .demos.server import main as demo
            demo(rest)
        else:
            decide_main(rest)
    except ImportError as error:
        parser.exit(2, f"{error}\nInstall the needed extra: 'jevany[train]', 'jevany[serve]' or 'jevany[local]'.\n")
    except (OSError, ValueError) as error:
        parser.exit(2, f"jevany: {error}\n")


def decide_main(argv: list[str]) -> None:
    # Only the --checkpoint branch loads the modelling stack: --help and
    # --base-url work in the base installation, which has no PyTorch.
    from dataclasses import fields
    from .inference import InferenceOptions, add_inference_arguments, inference_options_from_args
    from .placement import add_placement_arguments
    from .readout import add_readout_arguments, choice_options_from_args

    parser = argparse.ArgumentParser(prog="jevany decide")
    parser.add_argument("request", help="JSON request file; - reads stdin")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", help="load this checkpoint in-process")
    mode.add_argument("--base-url", default="http://127.0.0.1:8008", help="call a running server")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--model-name", help="identity for a locally loaded checkpoint")
    add_readout_arguments(parser)
    add_placement_arguments(parser)
    parser.add_argument("--cuda-graphs", action="store_true", help="capture CUDA graphs for the local checkpoint")
    parser.add_argument("--cuda-graph-max-tokens", type=int,
                        help="largest captured row; longer rows run eagerly (default 2048)")
    add_inference_arguments(parser)
    args = parser.parse_args(argv)
    try:
        choice_options = choice_options_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    content = sys.stdin.read() if args.request == "-" else Path(args.request).read_text(encoding="utf-8")
    from .api import SystemOneRequest
    request = SystemOneRequest.model_validate_json(content)
    if args.checkpoint:
        from dataclasses import replace
        from .checkpoint import LoadOptions, load_options_from_args
        from .runtime import JevModel
        if choice_options is not None and (
            args.cuda_graphs or args.cuda_graph_max_tokens is not None
            or any(getattr(args, item.name) is not None for item in fields(InferenceOptions))
        ):
            parser.error("CUDA graph and native inference-limit flags cannot be used with --readout choice; "
                         "use --choice-max-tokens")
        options = load_options_from_args(args)
        if args.cuda_graphs or args.cuda_graph_max_tokens is not None:
            options = options or LoadOptions.from_env()
            options = replace(options, cuda_graphs=True,
                              cuda_graph_max_tokens=(args.cuda_graph_max_tokens
                                                     if args.cuda_graph_max_tokens is not None
                                                     else options.cuda_graph_max_tokens))
        settings = ({
            "readout": "choice",
            "choice_temperature": choice_options.temperature,
            "choice_native_weight": choice_options.native_weight,
            "choice_max_tokens": choice_options.max_tokens,
        } if choice_options is not None else {
            "inference_options": inference_options_from_args(args),
        })
        client = JevModel.from_pretrained(
            args.checkpoint, device=args.device, dtype=args.dtype, model_name=args.model_name,
            options=options, **settings,
        )
    else:
        if (args.device or args.dtype or args.model_name is not None or args.readout != "native"
                or args.device_map is not None or args.max_memory_gib is not None or args.cuda_graphs
                or args.cuda_graph_max_tokens is not None
                or any(getattr(args, item.name) is not None for item in fields(InferenceOptions))):
            parser.error("device, dtype, placement, model-name, readout, cuda-graphs and inference limit options "
                         "require --checkpoint")
        from .client import JevClient
        client = JevClient(args.base_url)
    print(json.dumps(client(request), indent=2, ensure_ascii=False))


def data_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="jevany data", description="Prepare and validate labelled System One JSONL.")
    parser.add_argument("action", choices=["init", "validate", "convert", "check-suite", "build-sft", "build-rlcr"])
    if not argv or argv[0] in ("-h", "--help"):
        parser.parse_args(argv or ["--help"])
        return
    action = parser.parse_args(argv[:1]).action
    rest = argv[1:]
    if action == "convert":
        from .datasets.convert import main as convert
        convert(rest)
    elif action == "build-sft":
        from .datasets.build_sft import main as build
        build(rest)
    elif action == "build-rlcr":
        from .datasets.build_rlcr import main as build
        build(rest)
    else:
        sub = argparse.ArgumentParser(prog=f"jevany data {action}")
        if action == "init":
            sub.add_argument("--out", default="data/starter")
            from .datasets import init_starter
            print(init_starter(sub.parse_args(rest).out))
        elif action == "check-suite":
            sub.add_argument("path")
            sub.add_argument("--allow-test", action="store_true", help="also validate the locked test partition")
            from .suite import validate_suite
            args = sub.parse_args(rest)
            print(json.dumps(validate_suite(args.path, allow_test=args.allow_test), indent=2))
        else:
            sub.add_argument("path")
            from .data import validate_dataset
            print(json.dumps(validate_dataset(sub.parse_args(rest).path), indent=2))
