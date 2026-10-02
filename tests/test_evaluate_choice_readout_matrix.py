import json
import math

import pytest

from scripts.evaluate_choice_readout_matrix import (
    _attach_soft_gold,
    _typed_record,
    _typed_soft_metrics,
    evaluate,
)


def test_typed_record_uses_ordered_argmax_when_soft_gold_is_tied():
    row = {
        "id": "tie-case",
        "split": "test",
        "workflow": "unit",
        "state": json.dumps({"context": "A tied teacher distribution"}),
        "questions": json.dumps({
            "decision": {
                "type": "choice",
                "criteria": {"first": "First option", "second": "Second option"},
            }
        }),
        # The convenience label intentionally disagrees with the benchmark's
        # ordered argmax rule and must not determine the converted target.
        "gold": json.dumps({
            "decision": {
                "label": "second",
                "probabilities": {"first": 0.5, "second": 0.5},
            }
        }),
    }

    record = _typed_record(row)

    assert record["questions"]["decision"]["label"] == "first"
    assert record["questions"]["decision"]["target"] == {
        "first": 0.5,
        "second": 0.5,
    }


def test_typed_soft_metrics_normalize_distributions_and_use_hard_labels():
    rows = [
        {"p": [8.0, 2.0], "gold": [3.0, 1.0], "label": 0},
        {"p": [4.0, 6.0], "gold": [1.0, 1.0], "label": 0},
    ]

    result = _typed_soft_metrics(rows, bins=2)

    expected_kl = (
        0.75 * math.log(0.75 / 0.8)
        + 0.25 * math.log(0.25 / 0.2)
        + 0.5 * math.log(0.5 / 0.4)
        + 0.5 * math.log(0.5 / 0.6)
    ) / 2
    assert result["n"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["kl_from_gold"] == pytest.approx(expected_kl)
    assert result["brier"] == pytest.approx(0.0125)
    assert result["ece"] == pytest.approx(0.2)
    assert result["mean_confidence"] == pytest.approx(0.7)


def test_attach_soft_gold_aligns_values_to_each_prediction_rows_key_order():
    record = {
        "questions": {
            "choice": {"target": {"alpha": 0.1, "beta": 0.9}},
            "score": {"target": {"0": 0.2, "1": 0.3, "2": 0.5}},
        }
    }
    rows = [
        {"question": "choice", "keys": ["beta", "alpha"]},
        {"question": "score", "keys": ["2", "0", "1"]},
    ]

    _attach_soft_gold(rows, record)

    assert rows[0]["gold"] == [0.9, 0.1]
    assert rows[1]["gold"] == [0.5, 0.2, 0.3]


def test_evaluate_base_only_result_saves_choice_component(tmp_path):
    record = {
        "state": "Choose the right option.",
        "questions": {
            "decision": {
                "type": "choice",
                "criteria": {"wrong": "Wrong", "right": "Right"},
                "label": "right",
                "target": {"right": 0.8, "wrong": 0.2},
                "src": "unit/base-only",
            }
        },
        "_meta": {
            "id": "base-only",
            "group_id": "base-only",
            "source": "unit",
            "variant": "clean",
        },
    }

    def predictor(_record):
        return {
            # Reverse insertion order relative to criteria to exercise keyed
            # probability alignment in the full evaluation path.
            "probabilities": {"decision": {"right": 0.9, "wrong": 0.1}},
            "latency_ms": 2.0,
            "input_tokens": 12,
        }

    destination = tmp_path / "base"
    report = evaluate("typed_test", [record], predictor, destination)

    assert set(report["components"]) == {"choice"}
    assert report["components"]["choice"]["clean"]["acc"] == pytest.approx(1.0)
    assert not (destination / "native-rows.json").exists()
    rows = json.loads((destination / "choice-rows.json").read_text())
    assert rows[0]["keys"] == ["wrong", "right"]
    assert rows[0]["p"] == pytest.approx([0.1, 0.9])
    assert rows[0]["gold"] == pytest.approx([0.2, 0.8])
    prediction = json.loads(
        (destination / "predictions.jsonl").read_text().strip()
    )
    assert set(prediction["component_probabilities"]) == {"choice"}

