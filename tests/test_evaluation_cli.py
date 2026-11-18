"""Evaluation rejects occupied output paths before initializing a predictor."""
import json

import pytest

from jevany import benchmark, letter_predictor


@pytest.fixture(params=["native", "choice", "remote"])
def predictor_args(request):
    if request.param == "remote":
        return ["--remote", "http://localhost:8008"]
    return ["--run", "unused-checkpoint", "--readout", request.param]


@pytest.fixture
def labelled_data(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({
        "state": "evidence", "questions": {"q": {"type": "noul", "label": True}},
    }) + "\n", encoding="utf-8")
    return path


def patch_predictors(monkeypatch, factory):
    monkeypatch.setattr(benchmark, "LocalPredictor", factory)
    monkeypatch.setattr(benchmark, "RemotePredictor", factory)
    monkeypatch.setattr(letter_predictor, "LetterReadoutPredictor", factory)


@pytest.mark.parametrize("kind", ["directory", "file", "broken-symlink"])
@pytest.mark.parametrize("source", ["--data", "--suite"])
def test_cli_rejects_existing_output_before_reading_data_or_loading_predictor(
    tmp_path, labelled_data, predictor_args, kind, source, monkeypatch, capsys,
):
    output = tmp_path / "evaluation"
    if kind == "directory":
        output.mkdir()
        retained = output / "report.json"
    elif kind == "file":
        retained = output
    else:
        retained = tmp_path / "missing-target"
        output.symlink_to(retained, target_is_directory=True)
    if kind != "broken-symlink":
        retained.write_text("existing result", encoding="utf-8")

    def forbidden(*_args, **_kwargs):
        pytest.fail("data or predictor was loaded for an occupied output path")

    monkeypatch.setattr(benchmark, "load_records", forbidden)
    monkeypatch.setattr(benchmark, "load_split", forbidden)
    patch_predictors(monkeypatch, forbidden)
    with pytest.raises(SystemExit) as stopped:
        benchmark.main([
            *predictor_args, source, str(labelled_data),
            "--out", str(output), "--device", "cpu",
        ])
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "--out already exists" in error and str(output) in error
    if kind == "broken-symlink":
        assert output.is_symlink() and output.readlink() == retained
        assert not retained.exists()
    else:
        assert retained.read_text(encoding="utf-8") == "existing result"


def test_cli_evaluates_into_a_fresh_output_directory(
    tmp_path, labelled_data, predictor_args, monkeypatch, capsys,
):
    calls = []

    class Predictor:
        temperature = 1.0
        native_weight = 0.0
        provenance = {"method": "fixture"}
        served_model = "fixture"

        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, record):
            calls.append(record)
            return {"probabilities": {"q": {"false": 0.2, "true": 0.8}}, "latency_ms": 1}

    patch_predictors(monkeypatch, Predictor)
    output = tmp_path / "new-parent" / "evaluation"
    benchmark.main([
        *predictor_args, "--data", str(labelled_data),
        "--out", str(output), "--device", "cpu",
    ])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert report["clean"]["acc"] == 1.0
    assert report["coverage"]["evaluated_records"] == 1
    assert len(json.loads((output / "rows.json").read_text(encoding="utf-8"))) == 1
    assert len((output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(capsys.readouterr().out)["coverage"] == report["coverage"]


def test_output_created_during_model_loading_is_not_overwritten(tmp_path, labelled_data, monkeypatch):
    output = tmp_path / "evaluation"
    report = output / "report.json"

    def load(*_args, **_kwargs):
        output.mkdir()
        report.write_text("another evaluation", encoding="utf-8")
        return lambda _record: pytest.fail("prediction ran after the output path was occupied")

    monkeypatch.setattr(benchmark, "LocalPredictor", load)
    with pytest.raises(FileExistsError):
        benchmark.main([
            "--run", "unused-checkpoint", "--data", str(labelled_data),
            "--out", str(output), "--device", "cpu",
        ])
    assert report.read_text(encoding="utf-8") == "another evaluation"
    assert list(output.iterdir()) == [report]
