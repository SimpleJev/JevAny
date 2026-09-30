#!/usr/bin/env python3
"""Evaluate one JevAny checkpoint on the frozen public JevBench suite."""
import argparse
from pathlib import Path

import torch

from jevany.benchmark import evaluate_records
from jevany.checkpoint import LoadOptions, add_placement_arguments
from jevany.predictors import LocalPredictor
from jevany.suite import digest, load_split, read_manifest, write_json


EXPECTED_SUITE = "jevbench-public-v1.4.2.2"
EXPECTED_RECORDS = 231
EXPECTED_DEVELOPMENT_SHA256 = "b6d6eb34fdbbf46ec11be8657da5e513e3538990d7a00ca5582c55d8bd7941c6"


def load_records(root):
    """Load the checksum-verified canonical conversion used during training."""
    manifest = read_manifest(root)
    if manifest.get("name") != EXPECTED_SUITE:
        raise ValueError(f"expected frozen suite {EXPECTED_SUITE!r}")
    if manifest.get("files", {}).get("development.jsonl", {}).get("sha256") != EXPECTED_DEVELOPMENT_SHA256:
        raise ValueError("unexpected JevBench development manifest hash")
    records = load_split(root, "development")
    questions = sum(len(record["questions"]) for record in records)
    if len(records) != EXPECTED_RECORDS or questions != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} JevBench records/questions")
    return records, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--suite", required=True,
                        help="checksum-verified jevbench-public-v1.4.2.2 suite")
    parser.add_argument("--out", required=True)
    parser.add_argument("--attn", choices=("eager", "sdpa"), default=None)
    add_placement_arguments(parser)
    args = parser.parse_args()

    suite = Path(args.suite)
    records, manifest = load_records(suite)
    predictor = LocalPredictor(args.run, "cuda", LoadOptions(
        dtype=torch.bfloat16, merge=False, temperature=None, base_load_path=args.base,
        attn=args.attn, device_map=args.device_map, max_memory_gib=args.max_memory_gib,
    ), max_packed=16_384)
    report, rows = evaluate_records(records, predictor, args.out)
    tiers = {}
    for tier, expected in (("easy", 48), ("original", 72), ("hard", 111)):
        selected = [row for row in rows if row["id"].startswith(f"{tier}-")]
        if len(selected) != expected:
            raise ValueError(f"expected {expected} {tier} rows, found {len(selected)}")
        correct = sum(max(range(len(row["p"])), key=row["p"].__getitem__) == row["label"]
                      for row in selected)
        tiers[tier] = {"questions": expected, "correct": correct, "accuracy": correct / expected}
    report.update(protocol="JevBench/public", suite=manifest["name"],
                  suite_sha256=digest(suite / "development.jsonl"),
                  source_sha256=manifest["source_files"], tiers=tiers)
    write_json(Path(args.out) / "report.json", report)


if __name__ == "__main__":
    main()
