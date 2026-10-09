#!/usr/bin/env python3
"""Evaluate a pinned Laya snapshot with JevAny's frozen scoring protocol."""
import argparse
import json
import sys
import time
from pathlib import Path

from jevany.api import question_keys, validate_response
from jevany.benchmark import evaluate_records
from jevany.data import api_request
from jevany.suite import digest, load_split, read_manifest, write_json


JEVBENCH_FILES = ("easy.jsonl", "original.jsonl", "hard.jsonl")


def load_jevbench(root):
    records = []
    for filename in JEVBENCH_FILES:
        dataset = Path(filename).stem.title()
        for raw in (json.loads(line) for line in (root / "jevbench" / filename).read_text().splitlines() if line):
            questions = {"decision": dict(raw["question"])}
            label = raw["expected"]
            if questions["decision"]["type"] == "noul":
                label = {"no": False, "yes": True, "false": False, "true": True,
                         0: False, 1: True}[label]
            questions["decision"].update(label=label, src=dataset)
            identity = f"{dataset}/{raw['id']}"
            records.append({
                "state": raw["state"], "questions": questions,
                "_meta": {"id": identity, "group_id": identity, "source": dataset,
                          "variant": "clean"},
            })
    return records


class LayaPredictor:
    temperature = 1.0

    def __init__(self, snapshot, device):
        sys.path.insert(0, str(snapshot))
        from rl_agent_api import RLAgent
        self.agent = RLAgent(str(snapshot), device=device)

    def __call__(self, record):
        request = api_request(record)
        started = time.perf_counter()
        response = self.agent.system_one(request["state"], request["questions"])
        elapsed = 1000 * (time.perf_counter() - started)
        validated = validate_response(request, response)
        probabilities = {}
        for qid, question in request["questions"].items():
            answer = validated["answers"][qid]
            keys = question_keys(question["type"], question.get("criteria"))
            if question["type"] == "noul":
                value = float(answer["noul"])
                probabilities[qid] = {"false": 1 - value, "true": value}
            else:
                probabilities[qid] = {key: float(answer["probabilities"][key]) for key in keys}
        return {"probabilities": probabilities, "latency_ms": elapsed,
                "input_tokens": (response.get("usage") or {}).get("input_tokens")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite")
    source.add_argument("--jevbench-root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.suite:
        records = load_split(args.suite, "development")
        heldout = tuple(read_manifest(args.suite).get("holdout_sources", []))
        source_hash = digest(Path(args.suite) / "manifest.json")
        protocol = "transfer-v9/development"
    else:
        records = load_jevbench(Path(args.jevbench_root))
        heldout = ()
        source_hash = {name: digest(Path(args.jevbench_root) / "jevbench" / name)
                       for name in JEVBENCH_FILES}
        protocol = "JevBench/public"
    predictor = LayaPredictor(Path(args.snapshot), args.device)
    report, _ = evaluate_records(records, predictor, args.out, heldout_sources=heldout,
                                 skip_overlong=True)
    report.update(model="convaiinnovations/laya", revision=Path(args.snapshot).name,
                  protocol=protocol, source_sha256=source_hash)
    write_json(Path(args.out) / "report.json", report)


if __name__ == "__main__":
    main()
