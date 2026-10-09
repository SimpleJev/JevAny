# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Score a predictor on a frozen suite partition (or your own labelled JSONL).

    jevany eval --run models/JevAny-Qwen3.8-27B-LoRA --suite data/eval-suite --out runs/eval
    jevany eval --run runs/my-jev --data data/starter/development.jsonl --out runs/my-jev/eval
    jevany eval --remote http://127.0.0.1:8008 --suite ... --out ...        # any System One endpoint

`python -m jevany.benchmark` takes the same arguments.

Every prediction becomes one row per question (prediction_rows); jevany.metrics scores rows; evaluate_records writes
predictions.jsonl, rows.json and report.json. Predictors live in jevany.predictors.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from jevany.api import question_keys, validate_distribution, with_date_facts
from jevany.checkpoint import LoadOptions
from jevany.data import api_request, load_records
from jevany.device import default_device
from jevany.metrics import EPSILON, grouped_metrics, metrics, unknowable_report
from jevany.model import ContextLengthError
from jevany.predictors import LocalPredictor, RemotePredictor
from jevany.readout import add_readout_arguments, choice_options_from_args, normalize_readout
from jevany.suite import ENCODING, digest, load_split, read_manifest, record_digest, write_json


def paired_flip(rows):
    """Measure whether paired cases flip when their correct answers differ."""
    pairs = {}
    for row in rows:
        if row.get("pair_id"):
            pair = pairs.setdefault((row["pair_id"], row.get("question", "decision")), {})
            pair[row["sibling"]] = row
    if not pairs:
        return None
    if any(set(pair) != {"a", "b"} for pair in pairs.values()):
        raise ValueError("incomplete contrastive pair")
    prediction = lambda row: row["keys"][max(range(len(row["p"])), key=row["p"].__getitem__)]
    truth = lambda row: row["keys"][row["label"]]
    changed = [pair for pair in pairs.values() if truth(pair["a"]) != truth(pair["b"])]
    invariant = [pair for pair in pairs.values() if truth(pair["a"]) == truth(pair["b"])]
    both_correct = lambda group: sum(all(prediction(row) == truth(row) for row in pair.values()) for pair in group) / len(group) if group else None
    result = {
        "pairs": len(changed),
        "flip_rate": sum(prediction(pair["a"]) != prediction(pair["b"]) for pair in changed) / len(changed) if changed else None,
        "both_correct_rate": both_correct(changed),
    }
    if invariant:
        result.update(
            invariant_pairs=len(invariant),
            invariance_rate=sum(prediction(pair["a"]) == prediction(pair["b"]) for pair in invariant) / len(invariant),
            invariant_both_correct_rate=both_correct(invariant),
        )
    return result


def labels(q):
    """(option keys, label index) of a labelled request question."""
    keys = question_keys(q["type"], q.get("criteria"))
    return keys, keys.index(q["label"]) if q["type"] == "choice" else int(q["label"])


def prediction_rows(record, prediction):
    if set(prediction["probabilities"]) != set(record["questions"]):
        raise ValueError("answer IDs differ from request IDs")
    meta = record["_meta"]
    rows = []
    for qid, q in record["questions"].items():
        keys, y = labels(q)
        values, total = validate_distribution(prediction["probabilities"][qid], keys)
        p = np.array(values)
        row = {"id": meta["id"], "group": meta["group_id"], "question": qid,
               "source": meta["source"], "task": q["src"], "type": q["type"],
               "variant": meta["variant"], "keys": keys, "label": y, "control_id": meta.get("control_id"),
               "pair_id": meta.get("pair_id"), "sibling": meta.get("sibling"), # suites frozen before parent_id existed stored the parent's id in group_id for variants
               "parent": meta.get("parent_id") or (meta["id"] if meta["variant"] == "clean" else meta["group_id"]),
               "p": p.tolist(), "raw_probability_sum": total, "zero_count": int((p == 0).sum())}
        if "logits" in prediction:
            raw_logits = prediction["logits"][qid]
            if set(raw_logits) != set(keys) or not all(math.isfinite(raw_logits[k]) for k in keys):
                raise ValueError("logit keys or values do not match the requested options")
            row["logits"] = [float(raw_logits[k]) for k in keys]
            row["inference_temperature"] = prediction["inference_temperature"]
        rows.append(row)
    return rows


def summarize(rows, temperature=1.0, heldout_sources=()):
    """Report over benchmark rows. heldout_sources: sources the scored model never trained on; their tasks are also
    reported as a separate block."""
    clean = [r for r in rows if r["variant"] == "clean"]
    tasks = grouped_metrics(clean, "task")
    variants = grouped_metrics(rows, "variant")
    lookup = {(r["id"], r["question"]): r for r in clean}
    diffs, flips = [], []
    for row in rows:
        if row["variant"] == "permuted" and row["type"] == "choice":
            original = lookup[(row["parent"], row["question"])]
            aligned = [row["p"][row["keys"].index(k)] for k in original["keys"]]
            diffs.append(float(np.max(np.abs(np.array(aligned) - original["p"]))))
            flips.append(int(np.argmax(aligned) != np.argmax(original["p"])))
    knowable = [r for r in clean if r["source"] != "unknowable"]     # unknowable records are scored on confidence, never on accuracy
    losses = [v["nll"] for k, v in tasks.items()
              if not k.startswith("unknowable_") or k.startswith("unknowable_control")]
    return {"objective": -float(np.mean(losses)) if losses else None,
            "paired_flip": paired_flip(clean), "unknowable": unknowable_report(clean),
            "clean": metrics(knowable) if knowable else {"n": 0}, "tasks": tasks, "variants": variants,
            "heldout_tasks": grouped_metrics([r for r in clean if r["source"] in heldout_sources], "task") if any(r["source"] in heldout_sources for r in clean) else {},
            "permutation": {"n": len(diffs), "mean_max_delta": float(np.mean(diffs)) if diffs else None,
                            "flip_rate": float(np.mean(flips)) if flips else None},
            "temperature": temperature, "calibrated_clean": metrics(knowable, temperature) if knowable else {"n": 0},
            "metric_policy": {"version": 3, "selective_ties": "whole_confidence_groups",
                              "coverage_at_error": "in-sample maximum over confidence thresholds; not a deployed error guarantee",
                              "aurc": "right-step integral over whole confidence groups",
                              "confident_error_rate": "high-confidence errors divided by all questions",
                              "error_rate_at_0_9": "errors divided by questions accepted at p_max >= 0.9",
                              "nll": "returned probabilities at T=1; logits when applying an additional temperature",
                              "nll_floor": EPSILON, "renormalize_returned_probabilities": True,
                              "raw_sums_outside_1e_5": sum(abs(r["raw_probability_sum"] - 1) > 1e-5 for r in rows),
                              "returned_zeros": sum(r["zero_count"] for r in rows)}}


def evaluate_records(records, predictor, directory, temperature=1.0, heldout_sources=(), skip_overlong=False):
    """skip_overlong: for external data that was not frozen to JevAny's context, records the model cannot encode (jevany.model
    MAX_STATE / MAX_PACKED) are counted in coverage["rejected_records"] and listed in rejected.json instead of aborting.
    Frozen suites never need this; reports must state that rejected records count as wrong in any headline number."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    coverage = {"requested_records": len(records), "requested_questions": sum(len(r["questions"]) for r in records),
                "evaluated_records": 0, "evaluated_questions": 0, "rejected_records": 0, "truncated_records": 0}
    rows, latencies, rejected = [], [], []
    with (directory / "predictions.jsonl").open("w", encoding=ENCODING) as output:
        for record in records:
            try:
                pred = predictor(record)
                new_rows = prediction_rows(record, pred)
            except ValueError as error:
                if skip_overlong and isinstance(error, ContextLengthError):
                    coverage["rejected_records"] += 1; rejected.append({"id": record["_meta"]["id"], "error": str(error)}); continue
                coverage["rejected_records"] += 1
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            except Exception as error:
                coverage["rejected_records"] += 1
                write_json(directory / "failure.json", {"coverage": coverage, "record_id": record["_meta"]["id"], "error_type": type(error).__name__})
                raise
            output.write(json.dumps({"request_sha256": record_digest(api_request(record)), "id": record["_meta"]["id"],
                                     "prediction": pred, "rows": new_rows}, allow_nan=False) + "\n")
            output.flush()
            rows.extend(new_rows)
            latencies.append(pred["latency_ms"])
            coverage["evaluated_records"] += 1
            coverage["evaluated_questions"] += len(new_rows)
            if coverage["evaluated_records"] % 50 == 0:
                print(f"evaluated {coverage['evaluated_records']}/{len(records)}", flush=True)
    write_json(directory / "rows.json", rows)
    if rejected: write_json(directory / "rejected.json", rejected)
    report = summarize(rows, temperature, heldout_sources)
    report.update(coverage=coverage, latency_ms={
                      "median": float(np.median(latencies)) if latencies else None,
                      "p95": float(np.quantile(latencies, .95)) if latencies else None},
                  calibration={"inference_temperature": getattr(predictor, "temperature", None),
                               "additional_temperature": temperature,
                               "logits_recorded": bool(rows) and all("logits" in r for r in rows)})
    write_json(directory / "report.json", report)
    return report, rows


EXAMPLES = """examples:
  jevany eval --run runs/my-jev --data data/starter/development.jsonl --out runs/my-jev/eval
  jevany eval --run SimpleJev/JevAny-Qwen3.5-4B-LoRA --suite data/eval-suite --out runs/eval
  jevany eval --run SimpleJev/JevAny-Qwen3.5-4B-LoRA --readout choice --suite data/eval-suite --out runs/choice
  jevany eval --remote http://127.0.0.1:8008 --data my-labelled.jsonl --out runs/remote-eval

--data scores your own labelled JSONL (jevany.data.load_records); --suite scores a
frozen suite's development partition. Reports land in <out>/report.json, with
per-question rows in rows.json and raw answers in predictions.jsonl. --out must not
exist yet. Local scoring needs the 'local' extra; --remote needs no model locally."""


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog, formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES,
        description="Score a checkpoint or a running endpoint on a frozen suite or your own labelled JSONL.")
    ap.add_argument("--run", help="checkpoint dir or Hub id (local scoring)")
    ap.add_argument("--remote", help="System One endpoint using the JevClient contract; non-loopback URLs require HTTPS")
    ap.add_argument("--remote-model", default="jevany-latest")
    add_readout_arguments(ap)
    ap.add_argument("--suite", help="frozen suite directory (scores its development partition)")
    ap.add_argument("--data", help="your own labelled requests, one JSON object per line (jevany.data.load_records); an alternative to --suite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=default_device())
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--date_facts", action="store_true", help="apply jevany.api.with_date_facts to every state before scoring (the opt-in serving preprocessor); reported in report.json")
    a = ap.parse_args(argv)
    try:
        choice_options = choice_options_from_args(a)
    except ValueError as error:
        ap.error(str(error))
    if bool(a.run) == bool(a.remote): ap.error("give exactly one of --run or --remote")
    if a.remote and choice_options is not None:
        ap.error("--readout choice is a local model setting; configure it on the remote server")
    if bool(a.suite) == bool(a.data): ap.error("give exactly one of --suite or --data")
    if a.data:
        records, heldout, split, source_hash = load_records(a.data), [], "custom", digest(Path(a.data))
    else:
        split = "test" if a.allow_test else "development"
        records = load_split(a.suite, split, allow_test=a.allow_test)
        heldout = read_manifest(a.suite).get("holdout_sources", []); source_hash = digest(Path(a.suite) / "manifest.json")
    if a.date_facts:
        records = [{**r, "state": with_date_facts(r["state"])} for r in records]
    if a.remote:
        predictor = RemotePredictor(a.remote, a.remote_model, os.environ.get("JEVANY_REMOTE_API_KEY", "local"))
    elif choice_options is not None:
        from jevany.letter_predictor import LetterReadoutPredictor
        predictor = LetterReadoutPredictor(
            checkpoint=a.run,
            device=a.device,
            options=LoadOptions.from_env(),
            temperature=choice_options.temperature,
            native_weight=choice_options.native_weight,
            max_tokens=choice_options.max_tokens,
        )
    else:
        predictor = LocalPredictor(a.run, a.device, LoadOptions.from_env())
    report, _ = evaluate_records(records, predictor, a.out, heldout_sources=tuple(heldout), skip_overlong=bool(a.data))
    native_temperature = (getattr(predictor.native_model, "temperature", None)
                          if choice_options is not None and predictor.native_weight else None)
    calibration_applied = (None if a.remote else predictor.temperature != 1.0
                           or native_temperature not in (None, 1.0))
    report.update(suite_sha256=source_hash, data=a.data, date_facts=a.date_facts, run=a.run or a.remote, split=split,
                  calibration_applied=calibration_applied,
                  base_loading=getattr(predictor, "base_loading", None),
                  remote={"base_url": a.remote, "requested_model": a.remote_model, "served_model": predictor.served_model} if a.remote else None)
    if not a.remote:
        report["readout"] = normalize_readout(a.readout)
    if choice_options is not None:
        choice_readout = dict(predictor.provenance)
        if native_temperature is not None:
            choice_readout["native_temperature"] = native_temperature
            choice_readout["pointer_temperature"] = native_temperature
            report["calibration"]["native_temperature"] = native_temperature
            report["calibration"]["pointer_temperature"] = native_temperature
        report["choice_readout"] = choice_readout
        # Kept as an output alias for consumers of reports written before the
        # public readout name changed to ``choice``.
        report["letter_readout"] = dict(choice_readout)
    write_json(Path(a.out) / "report.json", report)
    print(json.dumps({"objective": report["objective"], "clean": report["clean"], "coverage": report["coverage"]}, indent=2))


if __name__ == "__main__":
    main()
