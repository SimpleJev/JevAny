#!/usr/bin/env python3
"""Fit choice/native pooling on Transfer-v9 development and score frozen tests.

The blend weight and temperature are selected only from the deterministic
Transfer-v9 development partition emitted by ``evaluate_choice_readout_matrix``.
Those fixed values are then applied unchanged to every held-out dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from jevany.suite import read_json, write_json


EPS = 1e-12


def load_components(directory: Path) -> tuple[list[dict], list[dict] | None]:
    choice = read_json(directory / "choice-rows.json")
    native_path = directory / "native-rows.json"
    native = read_json(native_path) if native_path.exists() else None
    if native is not None:
        left = [(row["id"], row["question"], row["keys"], row["label"]) for row in choice]
        right = [(row["id"], row["question"], row["keys"], row["label"]) for row in native]
        if left != right:
            raise ValueError(f"component rows are not aligned: {directory}")
    return choice, native


def headline_indices(rows: list[dict]) -> list[int]:
    """Indices in the benchmark's accuracy cohort.

    Transfer-v9 also contains non-clean robustness variants and unknowable
    questions.  The benchmark never counts either in headline accuracy, so
    neither may influence weight or temperature selection.
    """

    return [
        index for index, row in enumerate(rows)
        if row.get("variant", "clean") == "clean"
        and row.get("source") != "unknowable"
    ]


def select_indices(rows: list[dict], indices: list[int]) -> list[dict]:
    return [rows[index] for index in indices]


def targets(rows: list[dict]) -> list[np.ndarray]:
    values = []
    for row in rows:
        if "gold" in row:
            target = np.asarray(row["gold"], dtype=np.float64)
            target /= target.sum()
        else:
            target = np.zeros(len(row["p"]), dtype=np.float64)
            target[row["label"]] = 1.0
        values.append(target)
    return values


def pooled_logits(choice: list[dict], native: list[dict] | None, weight: float) -> list[np.ndarray]:
    if native is None and weight:
        raise ValueError("native weight requires native component rows")
    result = []
    for index, row in enumerate(choice):
        left = np.log(np.maximum(np.asarray(row["p"], dtype=np.float64), EPS))
        if native is None:
            result.append(left)
        else:
            right = np.log(np.maximum(np.asarray(native[index]["p"], dtype=np.float64), EPS))
            result.append((1 - weight) * left + weight * right)
    return result


def probabilities(logits: list[np.ndarray], temperature: float) -> list[np.ndarray]:
    result = []
    for values in logits:
        scaled = values / temperature
        scaled -= scaled.max()
        p = np.exp(scaled)
        result.append(p / p.sum())
    return result


def soft_nll(rows: list[dict], ps: list[np.ndarray]) -> float:
    gold = targets(rows)
    return float(np.mean([
        -float(np.sum(target * np.log(np.maximum(p, EPS))))
        for target, p in zip(gold, ps, strict=True)
    ]))


def fit_temperature(rows: list[dict], logits: list[np.ndarray]) -> tuple[float, float]:
    """Golden-section search in log-temperature space with fixed bounds."""

    lo, hi = math.log(.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)

    def objective(log_temperature: float) -> float:
        return soft_nll(rows, probabilities(logits, math.exp(log_temperature)))

    f1, f2 = objective(x1), objective(x2)
    for _ in range(80):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = objective(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = objective(x2)
    point = (lo + hi) / 2
    return math.exp(point), objective(point)


def fit(rows: list[dict], choice: list[dict], native: list[dict] | None) -> dict:
    candidates = [0.0] if native is None else [index / 100 for index in range(101)]
    best = None
    trace = []
    for weight in candidates:
        logits = pooled_logits(choice, native, weight)
        temperature, nll = fit_temperature(rows, logits)
        t1 = probabilities(logits, 1.0)
        correct = sum(
            int(int(p.argmax()) == row["label"])
            for row, p in zip(rows, t1, strict=True)
        )
        item = {
            "native_weight": weight,
            "temperature": temperature,
            "correct": correct,
            "accuracy": correct / len(rows),
            "nll": nll,
        }
        trace.append(item)
        if best is None or (
            -correct, nll, abs(weight - .5)
        ) < (
            -best["correct"], best["nll"], abs(best["native_weight"] - .5)
        ):
            best = item
    return {"selected": best, "grid": trace}


def ece(rows: list[dict], ps: list[np.ndarray], bins: int = 15) -> float:
    totals = np.zeros(bins)
    correct = np.zeros(bins)
    confidence = np.zeros(bins)
    for row, p in zip(rows, ps, strict=True):
        conf = float(p.max())
        bucket = min(bins - 1, int(conf * bins))
        totals[bucket] += 1
        correct[bucket] += int(int(p.argmax()) == row["label"])
        confidence[bucket] += conf
    n = len(rows)
    return float(sum(
        totals[index] / n * abs(correct[index] / totals[index] - confidence[index] / totals[index])
        for index in range(bins) if totals[index]
    ))


def metrics(rows: list[dict], ps: list[np.ndarray], *, ece_bins: int = 10) -> dict:
    gold = targets(rows)
    cross_entropy = soft_nll(rows, ps)
    gold_entropy = float(np.mean([
        -float(np.sum(target * np.log(np.maximum(target, EPS)))) for target in gold
    ]))
    return {
        "n": len(rows),
        "correct": sum(int(int(p.argmax()) == row["label"]) for row, p in zip(rows, ps, strict=True)),
        "accuracy": float(np.mean([
            int(int(p.argmax()) == row["label"])
            for row, p in zip(rows, ps, strict=True)
        ])),
        "cross_entropy_from_gold": cross_entropy,
        "gold_entropy": gold_entropy,
        "kl_from_gold": cross_entropy - gold_entropy,
        "brier": float(np.mean([
            float(np.sum((p - target) ** 2))
            for target, p in zip(gold, ps, strict=True)
        ])),
        "ece": ece(rows, ps, bins=ece_bins),
        "mean_confidence": float(np.mean([p.max() for p in ps])),
    }


def evaluate(directory: Path, selected: dict) -> dict:
    choice, native = load_components(directory)
    configurations = {
        "choice_t1": (0.0, 1.0),
        "transfer_dev_calibrated_choice": (0.0, selected["choice_temperature"]),
    }
    if native is not None:
        configurations.update({
            "native_shipped": (1.0, 1.0),
            "fixed_blend_0_5": (.5, 1.0),
            "transfer_dev_tuned_blend": (
                selected["native_weight"], selected["blend_temperature"]
            ),
        })
    indices = headline_indices(choice)
    ece_bins = 15 if directory.name == "typed_test" else 10
    results = {}
    for name, (weight, temperature) in configurations.items():
        ps = probabilities(pooled_logits(choice, native, weight), temperature)
        headline = metrics(
            select_indices(choice, indices),
            [ps[index] for index in indices],
            ece_bins=ece_bins,
        )
        headline["cohort"] = "clean and knowable"
        headline["excluded_non_headline_rows"] = len(choice) - len(indices)
        headline["ece_bins"] = ece_bins
        results[name] = headline
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.run / "manifest.json")
    calibration_dir = args.run / "transfer_calibration"
    choice, native = load_components(calibration_dir)
    indices = headline_indices(choice)
    calibration_choice = select_indices(choice, indices)
    calibration_native = select_indices(native, indices) if native is not None else None
    choice_fit = fit(calibration_choice, calibration_choice, None)
    blend_fit = fit(calibration_choice, calibration_choice, calibration_native)
    selected = {
        "choice_temperature": choice_fit["selected"]["temperature"],
        "native_weight": blend_fit["selected"]["native_weight"],
        "blend_temperature": blend_fit["selected"]["temperature"],
        "selection_objective": (
            "hard-label accuracy on the Transfer-v9 development clean/knowable "
            "cohort; calibrated NLL and distance from 0.5 break ties"
        ),
    }
    datasets = {}
    for name in (
        "transfer_calibration", "transfer_test", "typed_test", "jevbench_public",
        "jevjudge_text"
    ):
        datasets[name] = evaluate(args.run / name, selected)
    result = {
        "model": manifest["model"],
        "source_manifest": str(args.run / "manifest.json"),
        "protocol": {
            "selection_split": (
                "Transfer-v9 development clean/knowable accuracy cohort; never any "
                "reported test panel"
            ),
            "selection_rows": len(calibration_choice),
            "weight_grid": "0.00 to 1.00 inclusive, step 0.01",
            "pool": "log-linear/geometric",
            "temperature_fit": "bounded scalar NLL minimization, T in [0.05, 20]",
            "ece": "15 equal-width bins for Typed Decisions; 10 for every other dataset",
            "selection_objective": "accuracy; calibrated NLL then distance from 0.5 break ties",
            "transfer": "the selected weight and temperature are frozen for every held-out dataset",
            "native_shipped": (
                "the checkpoint's native probabilities, including its shipped inference "
                "temperature; no additional temperature is applied"
            ),
            "fixed_blend_0_5": (
                "equal log-linear pooling of T=1 choice probabilities and shipped native "
                "probabilities; no additional temperature is applied"
            ),
            "transfer_dev_tuned_blend": (
                "the Transfer-dev-selected native weight followed by the reported scalar "
                "additional temperature"
            ),
        },
        "selected": selected,
        "selection": {"choice": choice_fit, "blend": blend_fit},
        "datasets": datasets,
    }
    write_json(args.out, result)
    print(json.dumps({
        "model": result["model"],
        "selected": selected,
        "test_accuracy": {
            dataset: {name: values["accuracy"] for name, values in scores.items()}
            for dataset, scores in datasets.items() if dataset != "transfer_calibration"
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
