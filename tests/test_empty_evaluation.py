"""Evaluation reports retain coverage when there are no scoreable rows."""
import json

from jevany.benchmark import evaluate_records
from jevany.data import load_records
from jevany.model import ContextLengthError


def records(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps({
        "state": state, "questions": {"q": {"type": "noul", "label": True}},
    }) + "\n" for state in ("first input", "second input")))
    return load_records(path)


def test_all_context_rejections_leave_a_complete_evaluation_report(tmp_path):
    panel = records(tmp_path)

    def reject(_record):
        raise ContextLengthError("request exceeds the model context")

    output = tmp_path / "evaluation"
    report, rows = evaluate_records(panel, reject, output, skip_overlong=True)
    assert rows == []
    assert report["coverage"] == {
        "requested_records": 2, "requested_questions": 2,
        "evaluated_records": 0, "evaluated_questions": 0,
        "rejected_records": 2, "truncated_records": 0,
    }
    assert report["objective"] is None
    assert report["clean"] == report["calibrated_clean"] == {"n": 0}
    assert report["latency_ms"] == {"median": None, "p95": None}
    assert report["calibration"]["logits_recorded"] is False
    assert json.loads((output / "report.json").read_text()) == report
    assert json.loads((output / "rows.json").read_text()) == []
    assert json.loads((output / "rejected.json").read_text()) == [
        {"id": record["_meta"]["id"], "error": "request exceeds the model context"} for record in panel
    ]
    assert (output / "predictions.jsonl").read_text() == ""
    assert not (output / "failure.json").exists()
    json.dumps(report, allow_nan=False)


def test_unknowable_only_data_has_confidence_results_without_accuracy(tmp_path):
    panel = records(tmp_path)
    for record in panel:
        record["_meta"]["source"] = "unknowable"
        record["questions"]["q"]["src"] = "unknowable_fixture"

    def predict(_record):
        return {"probabilities": {"q": {"false": 0.5, "true": 0.5}}, "latency_ms": 1}

    report, rows = evaluate_records(panel, predict, tmp_path / "evaluation")
    assert len(rows) == 2
    assert report["objective"] is None
    assert report["clean"] == report["calibrated_clean"] == {"n": 0}
    assert report["unknowable"]["n"] == 2
    assert report["unknowable"]["mean_max_p"] == 0.5
    assert report["latency_ms"] == {"median": 1, "p95": 1}
    assert report["coverage"]["evaluated_records"] == 2
    json.dumps(report, allow_nan=False)
