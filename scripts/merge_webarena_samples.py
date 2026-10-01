#!/usr/bin/env python3
"""Merge independently run WebArena task pairs into one validated sample cell."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from jevany.suite import write_json
from scripts.eval_bedrock_jev_webarena import compare, render_demo, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--demo-task-id", type=int, required=True)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--cache-read-price-per-million", type=float)
    parser.add_argument("--cache-write-price-per-million", type=float)
    parser.add_argument("--out", required=True)
    parser.add_argument("--demo-out")
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    if any(item.get("status") != "complete" for item in records):
        parser.error("all inputs must be complete")
    models = {item["bedrock_model"] for item in records}
    resolved = {item["resolved_bedrock_model"] for item in records}
    jev_models = {item["jev_model"] for item in records}
    if len(models) != 1 or len(resolved) != 1 or len(jev_models) != 1:
        parser.error("model identities differ across inputs")
    episodes = [episode for record in records for episode in record["episodes"]]
    indexed = {(item["task_id"], item["mode"]): item for item in episodes}
    task_ids = sorted({item["task_id"] for item in episodes})
    if len(indexed) != len(episodes) or any(
        (task_id, mode) not in indexed
        for task_id in task_ids for mode in ("baseline", "optional")
    ):
        parser.error("inputs contain duplicate or unpaired tasks")
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["mode"]].append(episode)
    prices = {
        "input": args.input_price_per_million,
        "output": args.output_price_per_million,
        "cache_read": args.cache_read_price_per_million,
        "cache_write": args.cache_write_price_per_million,
    }
    summaries = {mode: summarize(items, prices) for mode, items in grouped.items()}
    if args.demo_task_id not in task_ids:
        parser.error("demo task is absent")
    demo = {
        "task_id": args.demo_task_id,
        "baseline": indexed[(args.demo_task_id, "baseline")],
        "optional": indexed[(args.demo_task_id, "optional")],
    }
    output = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "WebArena sampled paired evaluation",
        "bedrock_model": next(iter(models)),
        "resolved_bedrock_model": next(iter(resolved)),
        "jev_model": next(iter(jev_models)),
        "config": {
            "episodes": len(task_ids),
            "task_ids": task_ids,
            "demo_task_id": args.demo_task_id,
            "source_files": [str(Path(path).resolve()) for path in args.inputs],
        },
        "summaries": summaries,
        "comparison": compare(summaries, episodes),
        "demo_case": demo,
        "episodes": episodes,
    }
    write_json(Path(args.out), output)
    if args.demo_out:
        path = Path(args.demo_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_demo(demo, next(iter(models)), next(iter(jev_models))),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
