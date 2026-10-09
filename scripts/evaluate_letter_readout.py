#!/usr/bin/env python3
"""Evaluate Cygnet-style full-vocabulary letter readout on public JevBench.

The public set is a development diagnostic, not JevBench's sealed score.  Use
``--sample-per-tier`` for a quick smoke run before the complete 231 records.
"""

import argparse
from pathlib import Path

import torch

from jevany.benchmark import evaluate_records
from jevany.checkpoint import LoadOptions
from jevany.letter_predictor import LetterReadoutPredictor
from jevany.suite import digest, write_json
from scripts.evaluate_jevbench import load_records


CYGNET_RECIPE = "https://github.com/blockbrain-ai/cygnet-recipe"
CYGNET_COMMIT = "3cf591c692dec649f7c134449814610307c7bb3a"


def sample_tiers(records, count):
    if count == 0:
        return records
    selected = []
    for tier in ("easy", "original", "hard"):
        rows = [record for record in records if record["_meta"]["id"].startswith(f"{tier}-")]
        if len(rows) < count:
            raise ValueError(f"requested {count} {tier} records, only {len(rows)} exist")
        selected.extend(rows[:count])
    return selected


def tier_report(rows):
    result = {}
    for tier in ("easy", "original", "hard"):
        selected = [row for row in rows if row["id"].startswith(f"{tier}-")]
        correct = sum(
            max(range(len(row["p"])), key=row["p"].__getitem__) == row["label"]
            for row in selected
        )
        result[tier] = {
            "questions": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base", help="canonical frozen base model ID")
    source.add_argument("--checkpoint", help="JevAny checkpoint whose LoRA is applied before letter readout")
    parser.add_argument("--base-load-path", help="verified local copy of the canonical base weights")
    parser.add_argument("--revision", help="base revision; checkpoint revisions belong in owner/repo@revision")
    parser.add_argument("--suite", required=True, help="checksum-verified jevbench-public-v1.4.2.2 suite")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="post-readout calibration temperature; fit per model, do not copy Cygnet's 3.4 blindly")
    parser.add_argument("--pointer-weight", type=float, default=0.0,
                        help="log-linear JevAny pointer weight; requires --checkpoint")
    parser.add_argument("--lora-scale", type=float, default=1.0,
                        help="checkpoint adapter scale; 0 gives the exact pinned base with the adapter disabled")
    parser.add_argument("--max-tokens", type=int, default=16_384)
    parser.add_argument("--sample-per-tier", type=int, default=0,
                        help="evaluate the first N records from each tier; 0 runs all 231")
    args = parser.parse_args()
    if args.sample_per_tier < 0:
        parser.error("--sample-per-tier must be non-negative")
    if args.pointer_weight and not args.checkpoint:
        parser.error("--pointer-weight requires --checkpoint")

    suite = Path(args.suite)
    records, manifest = load_records(suite)
    records = sample_tiers(records, args.sample_per_tier)
    options = LoadOptions(
        dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype],
        merge=False,
        attn=args.attn,
        temperature=None,
        base_load_path=args.base_load_path,
        lora_scale=args.lora_scale,
    )
    predictor = LetterReadoutPredictor(
        base=args.base,
        checkpoint=args.checkpoint,
        device=args.device,
        options=options,
        revision=args.revision,
        temperature=args.temperature,
        pointer_weight=args.pointer_weight,
        max_tokens=args.max_tokens,
    )
    report, rows = evaluate_records(records, predictor, args.out)
    report.update(
        protocol="JevBench/public development diagnostic",
        suite=manifest["name"],
        suite_sha256=digest(suite / "development.jsonl"),
        source_sha256=manifest["source_files"],
        tiers=tier_report(rows),
        sample_per_tier=args.sample_per_tier,
        readout=predictor.provenance,
        base_loading=predictor.base_loading,
        method_reference={"repository": CYGNET_RECIPE, "commit": CYGNET_COMMIT},
    )
    write_json(Path(args.out) / "report.json", report)


if __name__ == "__main__":
    main()
