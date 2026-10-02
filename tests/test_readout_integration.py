import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jevany.api import Choice, Noul, Score, SystemOneRequest
from jevany.inference import InferenceOptions
from jevany.letter_runtime import LetterDecisionRuntime, _unlabelled_record
from jevany.readout import (
    LetterReadoutOptions,
    add_readout_arguments,
    letter_options_from_args,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(range(len(text.split()))))


class FakeLetterPredictor:
    def __init__(self):
        meta = SimpleNamespace(
            base="owner/base", lora=8, decision_mode="pointer",
        )
        self.checkpoint = SimpleNamespace(requested="owner/checkpoint", meta=meta)
        self.pointer_model = SimpleNamespace(
            inference_capabilities=SimpleNamespace(context_window=4096),
            inference_acceleration={
                "compile_mode": None, "lora_merged": False,
                "approximate_bf16_merge": False, "cuda_graphs": None,
            },
            device_map=None, devices=["cpu"], backbone_adapter="test", temperature=1.2,
        )
        self.device = "cpu"
        self.temperature = 1.0
        self.pointer_weight = 0.25
        self.max_tokens = 2048
        self.effective_max_tokens = 2048
        self.tokenizer = FakeTokenizer()
        self.provenance = {
            "method": "exact option-letter alias projection",
            "adapter_applied": True,
            "pointer_weight": 0.25,
        }
        self.last_record = None

    def __call__(self, record):
        self.last_record = record
        return {
            "probabilities": {
                "choice": {"left": 0.25, "right": 0.75},
                "noul": {"false": 0.1, "true": 0.9},
                "score": {"0": 0.2, "1": 0.8},
            },
            "latency_ms": 12.5,
            "input_tokens": 37,
        }


@pytest.fixture
def decision_request():
    return SystemOneRequest(
        state={"signal": "green"}, model="letter-model",
        questions={
            "choice": Choice(criteria={"left": "Left", "right": "Right"}),
            "noul": Noul(criteria={"false": "No", "true": "Yes"}),
            "score": Score(criteria=["low", "high"]),
        },
    )


def test_letter_cli_options_are_shared_and_native_rejects_letter_flags():
    parser = argparse.ArgumentParser()
    add_readout_arguments(parser)
    assert letter_options_from_args(parser.parse_args([])) is None
    actual = letter_options_from_args(parser.parse_args([
        "--readout", "letter", "--letter-temperature", "1.5",
        "--letter-pointer-weight", "0.25", "--letter-max-tokens", "2048",
    ]))
    assert actual == LetterReadoutOptions(temperature=1.5, pointer_weight=0.25, max_tokens=2048)
    with pytest.raises(ValueError, match="require --readout letter"):
        letter_options_from_args(parser.parse_args(["--letter-temperature", "2"]))
    for arguments, message in [
        (["--readout", "letter", "--letter-temperature", "nan"], "finite"),
        (["--readout", "letter", "--letter-temperature", "0"], "positive"),
        (["--readout", "letter", "--letter-pointer-weight", "1.1"], r"\[0, 1\]"),
        (["--readout", "letter", "--letter-max-tokens", "1"], ">= 2"),
    ]:
        with pytest.raises(ValueError, match=message):
            letter_options_from_args(parser.parse_args(arguments))


def test_letter_runtime_maps_requests_and_reports_effective_readout(decision_request):
    predictor = FakeLetterPredictor()
    runtime = LetterDecisionRuntime(predictor, "letter-model")

    response = runtime.answer(decision_request)

    assert response["model"] == "letter-model"
    assert response["answers"]["choice"]["choice"] == "right"
    assert response["answers"]["noul"]["noul"] == 0.9
    assert response["answers"]["score"]["score"] == 0.8
    assert response["usage"]["input_tokens"] == 37
    assert predictor.last_record["state"] == {"signal": "green"}
    assert [q["label"] for q in predictor.last_record["questions"].values()] == ["left", False, 0]
    description = runtime.describe()
    assert description["readout"] == "letter"
    assert description["letter_readout"]["pointer_weight"] == 0.25
    assert description["letter_readout"]["pointer_temperature"] == 1.2
    assert description["limits"] == {
        "state_tokens": 2048, "branch_tokens": 2048,
        "packed_tokens": 2048, "choices": 26,
    }
    assert description["capabilities"]["media_types"] == []
    assert not description["prefix_cache"]["enabled"]


def test_letter_runtime_rejects_wrong_model_and_media(decision_request):
    runtime = LetterDecisionRuntime(FakeLetterPredictor(), "letter-model")
    with pytest.raises(ValueError, match="unknown model"):
        runtime.answer(decision_request.model_copy(update={"model": "other"}))
    with pytest.raises(ValueError, match="does not support media"):
        runtime.answer(decision_request.model_copy(update={
            "media": [{"type": "image", "uri": "frame.png"}],
        }))


def test_unlabelled_record_adds_only_encoder_placeholders(decision_request):
    record = _unlabelled_record(decision_request)
    assert "model" not in record
    assert json.dumps(record)
    assert record["questions"]["choice"]["label"] == "left"
    assert record["questions"]["noul"]["label"] is False
    assert record["questions"]["score"]["label"] == 0


def test_public_loader_dispatches_to_letter_predictor(monkeypatch):
    from jevany import letter_predictor
    from jevany.runtime import JevModel

    predictor = FakeLetterPredictor()
    predictor.checkpoint.path = "/tmp/checkpoint"
    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return predictor

    monkeypatch.setattr(letter_predictor, "LetterReadoutPredictor", load)
    local = JevModel.from_pretrained(
        "owner/checkpoint", device="cpu", model_name="letter-model",
        readout="letter", letter_temperature=1.5,
        letter_pointer_weight=0.25, letter_max_tokens=2048,
    )
    assert isinstance(local.runtime, LetterDecisionRuntime)
    assert seen["checkpoint"] == "owner/checkpoint"
    assert seen["temperature"] == 1.5
    assert seen["pointer_weight"] == 0.25
    assert seen["max_tokens"] == 2048
    with pytest.raises(ValueError, match="require readout='letter'"):
        JevModel.from_pretrained("unused", device="cpu", letter_temperature=2)
    with pytest.raises(ValueError, match="only to native readout"):
        JevModel.from_pretrained(
            "unused", device="cpu", readout="letter",
            inference_options=InferenceOptions(),
        )


def test_create_app_rejects_cross_readout_settings():
    pytest.importorskip("fastapi")
    from jevany.serve import create_app

    with pytest.raises(ValueError, match="require readout='letter'"):
        create_app(readout="native", letter_options=LetterReadoutOptions())
    with pytest.raises(ValueError, match="only to native readout"):
        create_app(readout="letter", inference_options=InferenceOptions())


def test_decide_cli_forwards_letter_settings(tmp_path, monkeypatch, capsys):
    from jevany.cli import decide_main
    from jevany.runtime import JevModel

    source = tmp_path / "request.json"
    source.write_text(SystemOneRequest(
        state="state", model="letter-model",
        questions={"choice": Choice(criteria={"left": None, "right": None})},
    ).model_dump_json())
    seen = {}

    class Client:
        def __call__(self, request):
            return {
                "model": "letter-model",
                "answers": {"choice": {
                    "type": "choice", "choice": "right", "confidence": 0.5,
                    "probabilities": {"left": 0.25, "right": 0.75},
                }},
            }

    def load(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return Client()

    monkeypatch.setattr(JevModel, "from_pretrained", load)
    decide_main([
        str(source), "--checkpoint", "owner/checkpoint", "--device", "cpu",
        "--readout", "letter", "--letter-temperature", "1.5",
        "--letter-pointer-weight", "0.25", "--letter-max-tokens", "2048",
    ])
    assert json.loads(capsys.readouterr().out)["answers"]["choice"]["choice"] == "right"
    assert seen["args"] == ("owner/checkpoint",)
    assert seen["kwargs"]["readout"] == "letter"
    assert seen["kwargs"]["letter_temperature"] == 1.5
    assert seen["kwargs"]["letter_pointer_weight"] == 0.25
    assert seen["kwargs"]["letter_max_tokens"] == 2048
    assert "inference_options" not in seen["kwargs"]
    with pytest.raises(SystemExit):
        decide_main([str(source), "--letter-temperature", "2"])


def test_eval_cli_selects_letter_predictor_and_records_provenance(tmp_path, monkeypatch, capsys):
    from jevany import benchmark, letter_predictor

    data = tmp_path / "data.jsonl"
    data.write_text("placeholder\n")
    output = tmp_path / "evaluation"
    record = {
        "state": "state",
        "questions": {"choice": {
            "type": "choice", "criteria": {"left": None, "right": None}, "label": "right",
        }},
    }
    seen = {}

    class Predictor:
        temperature = 1.5
        pointer_weight = 0.25
        pointer_model = SimpleNamespace(temperature=1.2)
        provenance = {"method": "exact option-letter alias projection"}

        def __init__(self, **kwargs):
            seen.update(kwargs)

    def evaluate(records, predictor, directory, **kwargs):
        assert records == [record]
        assert isinstance(predictor, Predictor)
        Path(directory).mkdir()
        return ({"objective": 0.0, "clean": {}, "coverage": {}, "calibration": {}}, [])

    monkeypatch.setattr(benchmark, "load_records", lambda path: [record])
    monkeypatch.setattr(benchmark, "evaluate_records", evaluate)
    monkeypatch.setattr(letter_predictor, "LetterReadoutPredictor", Predictor)
    benchmark.main([
        "--run", "owner/checkpoint", "--data", str(data), "--out", str(output),
        "--device", "cpu", "--readout", "letter", "--letter-temperature", "1.5",
        "--letter-pointer-weight", "0.25", "--letter-max-tokens", "2048",
    ])
    assert json.loads(capsys.readouterr().out)["objective"] == 0.0
    assert seen["checkpoint"] == "owner/checkpoint"
    assert seen["temperature"] == 1.5
    assert seen["pointer_weight"] == 0.25
    assert seen["max_tokens"] == 2048
    report = json.loads((output / "report.json").read_text())
    assert report["readout"] == "letter"
    assert report["letter_readout"] == {
        **Predictor.provenance, "pointer_temperature": 1.2,
    }
    assert report["calibration_applied"] is True
    assert report["calibration"]["pointer_temperature"] == 1.2
