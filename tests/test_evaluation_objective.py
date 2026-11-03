"""The evaluation objective scores knowable clean rows with equal task weight."""
import json
import math

import pytest

from jevany.benchmark import evaluate_records, summarize
from jevany.data import load_records


def row(identity, source, task, probability):
    return {
        "id": identity, "question": "q", "group": identity,
        "source": source, "task": task, "type": "choice", "variant": "clean",
        "keys": ["a", "b"], "label": 1, "p": [1 - probability, probability],
        "raw_probability_sum": 1.0, "zero_count": 0,
    }


@pytest.mark.parametrize("source,task,expected", [
    ("unknowable", "custom_choice", None),
    ("unknowable", "unknowable_control_choice", None),
    ("unknowable", "unknowable_choice", None),
    ("custom", "unknowable_fixture", math.log(0.8)),
    ("custom", "custom_choice", math.log(0.8)),
    ("unknowable_control", "unknowable_control_choice", math.log(0.8)),
    ("unknowable_control", "unknowable_fixture", math.log(0.8)),
])
def test_objective_selects_by_source_regardless_of_task_name(source, task, expected):
    report = summarize([row("record", source, task, 0.8)])

    if expected is None:
        assert report["objective"] is None
        assert report["clean"] == {"n": 0}
        assert report["unknowable"]["n"] == 1
    else:
        assert report["objective"] == pytest.approx(expected)
        assert report["clean"]["n"] == 1
    assert report["tasks"][task]["n"] == 1


def test_objective_excludes_unknowable_rows_within_a_shared_task():
    rows = [
        row("first", "custom", "shared", 0.8),
        row("second", "custom", "shared", 0.6),
        row("unknown", "unknowable", "shared", 0.01),
    ]

    report = summarize(rows)

    assert report["objective"] == pytest.approx((math.log(0.8) + math.log(0.6)) / 2)
    assert report["clean"]["n"] == 2
    assert report["unknowable"]["n"] == 1
    assert report["tasks"]["shared"]["n"] == 3
    assert report["variants"]["clean"]["n"] == 3


def test_objective_keeps_equal_task_weights_and_ignores_variants_and_calibration():
    rows = [
        row("first", "custom", "large", 0.8),
        row("second", "custom", "large", 0.6),
        row("small", "custom", "small", 0.25),
        {**row("permuted", "custom", "small", 0.99), "variant": "permuted", "parent": "small"},
    ]

    report = summarize(rows, temperature=2.0)

    expected = ((math.log(0.8) + math.log(0.6)) / 2 + math.log(0.25)) / 2
    assert report["objective"] == pytest.approx(expected)
    assert report["clean"]["n"] == 3
    assert report["variants"]["permuted"]["n"] == 1
    assert report["calibrated_clean"]["nll"] != pytest.approx(report["clean"]["nll"])


def test_evaluation_persists_objective_for_custom_rows_with_default_task_names(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({
        "state": source, "questions": {"q": {"type": "noul", "label": True}},
        "_meta": {"source": source},
    }) + "\n" for source in ("custom", "unknowable")))

    def predict(record):
        probability = 0.8 if record["_meta"]["source"] == "custom" else 0.01
        return {"probabilities": {"q": {"false": 1 - probability, "true": probability}},
                "latency_ms": 1}

    output = tmp_path / "evaluation"
    report, rows = evaluate_records(load_records(path), predict, output)

    assert len(rows) == 2
    assert report["objective"] == pytest.approx(math.log(0.8))
    assert report["clean"]["n"] == 1
    assert report["unknowable"]["n"] == 1
    assert report["tasks"]["custom_noul"]["n"] == 2
    assert report["coverage"]["evaluated_questions"] == 2
    assert json.loads((output / "report.json").read_text()) == report
    assert json.loads((output / "rows.json").read_text()) == rows
