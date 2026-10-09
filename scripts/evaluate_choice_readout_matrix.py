#!/usr/bin/env python3
"""Evaluate training-free choice readout and a checkpoint's native readout.

The script loads one model once, then scores the same frozen model on JevBench,
Transfer-v9, Typed Decisions and JevJudge's text-only subset.  Checkpoint runs
save both readout components so ensemble weights can be selected offline on a
separate calibration split without repeating inference or looking at test
labels.  Frozen-base runs save the training-free choice component only.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import peft
import safetensors
import torch
import transformers

from jevany.api import question_keys
from jevany.benchmark import prediction_rows, summarize
from jevany.checkpoint import LoadOptions
from jevany.data import api_request
from jevany.letter_predictor import LetterReadoutPredictor
from jevany.suite import (
    ENCODING,
    digest,
    load_split,
    read_json,
    record_digest,
    write_json,
)


def _typed_rows(dataset: Path, split: str) -> list[dict]:
    path = dataset / "all" / f"{split}-00000-of-00001.parquet"
    return pq.read_table(path).to_pylist()


def _typed_record(row: dict) -> dict:
    state = json.loads(row["state"])
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    converted = {}
    for qid, question in questions.items():
        answer = gold[qid]
        value = dict(question)
        keys = question_keys(value["type"], value.get("criteria"))
        # Typed Decisions accuracy is ordered argmax agreement with the soft
        # teacher distribution.  The dataset's convenience `label` differs on
        # tied rows, so derive the target exactly as the benchmark scorer does.
        label_key = max(keys, key=lambda key: float(answer["probabilities"][key]))
        if value["type"] == "choice":
            value["label"] = label_key
        elif value["type"] == "noul":
            value["label"] = label_key == "true"
        else:
            value["label"] = int(label_key)
        value["target"] = answer["probabilities"]
        value["src"] = f"typed_decisions/{row['workflow']}/{qid}"
        converted[qid] = value
    return {
        "state": state,
        "questions": converted,
        "_meta": {
            "id": row["id"],
            "group_id": row["id"],
            "source": "LocalLLaMA/typed-decisions",
            "variant": "clean",
            "split": row["split"],
            "workflow": row["workflow"],
        },
    }


def _attach_soft_gold(rows: list[dict], record: dict) -> None:
    questions = record["questions"]
    for row in rows:
        target = questions[row["question"]].get("target")
        if target is not None:
            row["gold"] = [float(target[key]) for key in row["keys"]]


def _component_prediction(probabilities: dict) -> dict:
    return {"probabilities": probabilities}


def _typed_soft_metrics(rows: list[dict], bins: int = 15) -> dict:
    totals = [0] * bins
    correct = [0] * bins
    confidence = [0.0] * bins
    kl, brier = 0.0, 0.0
    for row in rows:
        p = [float(value) for value in row["p"]]
        gold = [float(value) for value in row["gold"]]
        p_total, gold_total = sum(p), sum(gold)
        p = [value / p_total for value in p]
        gold = [value / gold_total for value in gold]
        kl += sum(g * math.log(max(g, 1e-12) / max(value, 1e-12))
                  for g, value in zip(gold, p, strict=True))
        brier += sum((value - g) ** 2 for value, g in zip(p, gold, strict=True))
        predicted = max(range(len(p)), key=p.__getitem__)
        conf = max(p)
        bucket = min(bins - 1, int(conf * bins))
        totals[bucket] += 1
        correct[bucket] += int(predicted == row["label"])
        confidence[bucket] += conf
    n = len(rows)
    return {
        "n": n,
        "correct": sum(correct),
        "accuracy": sum(correct) / n,
        "kl_from_gold": kl / n,
        "brier": brier / n,
        "ece": sum(
            totals[index] / n * abs(
                correct[index] / totals[index] - confidence[index] / totals[index]
            )
            for index in range(bins) if totals[index]
        ),
        "mean_confidence": sum(confidence) / n,
    }


def evaluate(name: str, records: list[dict], predictor, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    component_rows: dict[str, list[dict]] = defaultdict(list)
    latencies = []
    efficient_long_context_records = 0
    started = time.time()
    with (destination / "predictions.jsonl").open("w", encoding=ENCODING) as stream:
        for index, record in enumerate(records, 1):
            result = predictor(record)
            efficient_long_context_records += int(
                result.get("efficient_long_context_attention_used", False)
            )
            components = result.get("component_probabilities") or {
                "choice": result["probabilities"]
            }
            saved = {}
            for component, probabilities in components.items():
                if probabilities is None:
                    continue
                rows = prediction_rows(record, _component_prediction(probabilities))
                _attach_soft_gold(rows, record)
                component_rows[component].extend(rows)
                saved[component] = probabilities
            stream.write(json.dumps({
                "id": record["_meta"]["id"],
                "request_sha256": record_digest(api_request(record)),
                "component_probabilities": saved,
                "latency_ms": result["latency_ms"],
                "input_tokens": result["input_tokens"],
            }, allow_nan=False) + "\n")
            stream.flush()
            latencies.append(float(result["latency_ms"]))
            if index % 25 == 0:
                print(f"{name}: {index}/{len(records)}", flush=True)
    reports = {}
    for component, rows in component_rows.items():
        write_json(destination / f"{component}-rows.json", rows)
        reports[component] = summarize(rows)
        if name == "typed_test":
            reports[component]["typed_soft_metrics"] = _typed_soft_metrics(rows)
    report = {
        "dataset": name,
        "records": len(records),
        "questions": sum(len(record["questions"]) for record in records),
        "components": reports,
        "components_executed": sorted(component_rows),
        "efficient_long_context_records": efficient_long_context_records,
        "latency_scope": (
            "end-to-end predictor call; checkpoint calls execute both choice and native "
            "components, so these values are not per-component efficiency measurements"
        ),
        "latency_ms": {
            "median": statistics.median(latencies),
            "p95": sorted(latencies)[min(len(latencies) - 1, int(.95 * len(latencies)))],
        },
        "wall_seconds": time.time() - started,
    }
    write_json(destination / "report.json", report)
    return report


def reuse_completed(
    name: str,
    records: list[dict],
    predictor,
    destination: Path,
) -> dict:
    """Load and validate one completed dataset from an interrupted matrix run."""

    report_path = destination / "report.json"
    if not report_path.is_file():
        raise RuntimeError(
            f"cannot resume incomplete dataset directory: {destination}; preserve or "
            "move that directory aside before retrying"
        )
    report = read_json(report_path)
    expected_questions = sum(len(record["questions"]) for record in records)
    expected_components = {"choice", "native"} if predictor.return_components else {"choice"}
    actual_components = set(report.get("components", {}))
    if (
        report.get("dataset") != name
        or report.get("records") != len(records)
        or report.get("questions") != expected_questions
        or actual_components != expected_components
    ):
        raise RuntimeError(f"completed report does not match requested resume dataset: {destination}")
    for component in expected_components:
        rows_path = destination / f"{component}-rows.json"
        if not rows_path.is_file() or len(read_json(rows_path)) != expected_questions:
            raise RuntimeError(f"completed component rows are missing or incomplete: {rows_path}")
    return report


def _correct(report: dict, component: str) -> int:
    clean = report["components"][component]["clean"]
    return round(clean["n"] * clean["acc"])


def _code_revision() -> str | None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _checkpoint_artifacts(predictor) -> dict | None:
    checkpoint = predictor.checkpoint
    if checkpoint is None:
        return None
    root = Path(checkpoint.path)
    files = {}
    for name in ("adapter_model.safetensors", "head.pt", "adapter_config.json"):
        path = root / name
        if path.is_file():
            files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return {"requested": checkpoint.requested, "resolved": str(root), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base")
    source.add_argument("--checkpoint")
    parser.add_argument("--base-load-path", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jevbench", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--typed", type=Path, required=True)
    parser.add_argument("--jevjudge", type=Path, required=True)
    parser.add_argument("--expected-jevbench-native-correct", type=int)
    parser.add_argument("--expected-transfer-native-correct", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--attn", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--max-tokens", type=int, default=65_536)
    parser.add_argument(
        "--efficient-long-context-tokens", type=int, default=8_192,
        help=(
            "use memory-linear fused SDPA at or above this many tokens; short "
            "parity panels retain exact math SDPA"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse validated completed dataset directories in an interrupted run",
    )
    parser.add_argument(
        "--preexisting-code-revision",
        help="code revision that produced reused reports (required when any are reused)",
    )
    args = parser.parse_args()
    if args.revision and args.checkpoint:
        parser.error("--revision accompanies --base, not --checkpoint")
    if args.base and (args.expected_jevbench_native_correct is not None
                      or args.expected_transfer_native_correct is not None):
        parser.error("native parity assertions require --checkpoint")

    if args.resume:
        if not args.out.is_dir():
            parser.error("--resume requires an existing --out directory")
    else:
        if args.preexisting_code_revision:
            parser.error("--preexisting-code-revision requires --resume")
        args.out.mkdir(parents=True, exist_ok=False)
    options = LoadOptions(
        dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype],
        merge=False,
        temperature=None,
        base_load_path=args.base_load_path,
        attn=args.attn,
    )
    predictor = LetterReadoutPredictor(
        base=args.base,
        checkpoint=args.checkpoint,
        revision=args.revision,
        device=args.device,
        options=options,
        temperature=1.0,
        native_weight=0.5 if args.checkpoint else 0.0,
        return_components=bool(args.checkpoint),
        exact_kernels=True,
        efficient_long_context_tokens=args.efficient_long_context_tokens,
        max_tokens=args.max_tokens,
    )

    current_code_revision = _code_revision()
    dataset_code_revisions = {}
    reused_datasets = []

    def run_dataset(name: str, records: list[dict]) -> dict:
        destination = args.out / name
        if args.resume and destination.exists():
            report = reuse_completed(name, records, predictor, destination)
            reused_datasets.append(name)
            dataset_code_revisions[name] = args.preexisting_code_revision
            return report
        report = evaluate(name, records, predictor, destination)
        dataset_code_revisions[name] = current_code_revision
        return report

    parity_datasets = {
        "jevbench_public": load_split(args.jevbench, "development"),
        "transfer_calibration": load_split(args.transfer, "development"),
    }
    reports = {
        name: run_dataset(name, records)
        for name, records in parity_datasets.items()
    }
    if args.checkpoint:
        checks = (
            ("jevbench_public", args.expected_jevbench_native_correct),
            ("transfer_calibration", args.expected_transfer_native_correct),
        )
        for dataset, expected in checks:
            if expected is not None:
                actual = _correct(reports[dataset], "native")
                if actual != expected:
                    raise RuntimeError(
                        f"{dataset} native parity failed: expected {expected}, got {actual}"
                    )
    heldout_datasets = {
        "transfer_test": load_split(args.transfer, "test", allow_test=True),
        "typed_test": [_typed_record(row) for row in _typed_rows(args.typed, "test")],
        "jevjudge_text": [
            record for record in load_split(args.jevjudge, "test", allow_test=True)
            if record["_meta"]["modality"] == "text"
        ],
    }
    reports.update({
        name: run_dataset(name, records)
        for name, records in heldout_datasets.items()
    })
    if reused_datasets and not args.preexisting_code_revision:
        parser.error("--preexisting-code-revision is required when reports are reused")

    typed_manifest = json.loads((args.typed / ".hf-fetch.json").read_text())
    manifest = {
        "model": args.model_name,
        "source": {"base": args.base, "checkpoint": args.checkpoint},
        "checkpoint_artifacts": _checkpoint_artifacts(predictor),
        "base_loading": predictor.base_loading,
        "predictor": predictor.provenance,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "safetensors": safetensors.__version__,
            "code_revision": current_code_revision,
        },
        "protocol": {
            "kernel_policy": "LocalPredictor-compatible exact CUDA policy",
            "dtype": args.dtype,
            "attn": args.attn,
            "temperature": 1.0,
            "blend_selection": (
                "not performed here; components are saved for fitting on Transfer-v9 "
                "development only"
            ),
            "efficiency_scope": (
                "base calls execute choice only; checkpoint calls execute choice and native. "
                "Do not compare component efficiency from these end-to-end timings"
            ),
            "long_context_attention": (
                "exact math SDPA below the configured threshold; memory-linear fused SDPA "
                "without a math fallback at or above it"
            ),
            "efficient_long_context_tokens": args.efficient_long_context_tokens,
            "resume": {
                "enabled": args.resume,
                "reused_datasets": reused_datasets,
                "preexisting_code_revision": args.preexisting_code_revision,
                "dataset_code_revisions": dataset_code_revisions,
            },
        },
        "datasets": {
            "jevbench_development_sha256": digest(args.jevbench / "development.jsonl"),
            "jevbench_manifest_sha256": digest(args.jevbench / "manifest.json"),
            "transfer_manifest_sha256": digest(args.transfer / "manifest.json"),
            "transfer_development_sha256": digest(args.transfer / "development.jsonl"),
            "transfer_test_sha256": digest(args.transfer / "test.jsonl"),
            "typed_revision": typed_manifest["resolved_revision"],
            "typed_test_sha256": digest(
                args.typed / "all" / "test-00000-of-00001.parquet"
            ),
            "jevjudge_test_sha256": digest(args.jevjudge / "test.jsonl"),
            "jevjudge_manifest_sha256": digest(args.jevjudge / "manifest.json"),
        },
        "reports": reports,
    }
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({
        "model": args.model_name,
        "results": {
            name: {
                component: report["clean"]["acc"]
                for component, report in value["components"].items()
            }
            for name, value in reports.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
