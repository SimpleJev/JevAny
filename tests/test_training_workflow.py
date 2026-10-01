"""Research workflows: rejected data, preflight, saving and resuming training."""
import json
from types import SimpleNamespace

import pytest

from jevany.datasets import init_starter
from jevany.model import ContextLengthError
from jevany.train import main, training_requests


def test_admission_preserves_rejection_reason_and_does_not_hide_encoding_errors(tmp_path):
    data = init_starter(tmp_path / "data") / "train.jsonl"
    args = SimpleNamespace(data=str(data), replay=0, max_state=16, max_branch=128, max_packed=256)

    class Encoder:
        calls = 0

        def encode(self, *unused, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ContextLengthError("state exceeds 16 tokens: 40")
            return {"ids": [1]}

    records, admission = training_requests(args, None, None, Encoder())
    assert len(records) == 23
    assert admission["requested_records"] == 24
    assert admission["admitted_records"] == 23
    assert admission["rejected_records"] == 1
    assert admission["rejected"][0]["reason"] == "state exceeds 16 tokens: 40"
    assert admission["rejected"][0]["id"] not in {r["_meta"]["id"] for r in records}

    class BrokenEncoder:
        def encode(self, *unused, **kwargs):
            raise ValueError("unsupported media type")

    with pytest.raises(ValueError, match="unsupported media type"):
        training_requests(args, None, None, BrokenEncoder())


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("bad_input", ["base", "evaluation"])
def test_preflight_rejects_bad_local_inputs_without_creating_run(tmp_path, monkeypatch, dry_run, bad_input):
    root = init_starter(tmp_path / "data")
    out = tmp_path / "run"
    args = ["--data", str(root / "train.jsonl"), "--out", str(out)]
    args += (["--base-load-path", str(tmp_path / "missing")] if bad_input == "base"
             else ["--eval-every-steps", "1", "--eval-suite", str(root / "development.jsonl")])
    if dry_run:
        args.append("--dry-run")
    monkeypatch.setattr("jevany.train.load_preprocessor",
                        lambda *args, **kwargs: pytest.fail("preflight must run before model loading"))
    with pytest.raises(ValueError, match="base load path|evaluation suite must be a directory"):
        main(args)
    assert not out.exists()


def test_python_training_configuration_errors_are_catchable(tmp_path):
    from jevany.training import train

    config = tmp_path / "bad.toml"
    config.write_text('batch = true\n')
    with pytest.raises(ValueError, match="batch must be int"):
        train(config)
    config.write_text('data = "irrelevant.jsonl"\naccum = 0\n')
    with pytest.raises(ValueError, match="must be positive"):
        train(config)


@pytest.mark.parametrize("epochs", [1, 3])
def test_saved_training_counts_include_rejected_input(tiny_run, tmp_path, epochs):
    root, _, _ = tiny_run
    data = tmp_path / "train.jsonl"
    question = {"q": {"type": "choice", "criteria": {"a": None, "b": None}, "label": "a"}}
    data.write_text("".join(json.dumps({"state": state, "questions": question}) + "\n"
                            for state in ("state", "state " * 40)))
    out = main(["--base", str(root / "base"), "--data", str(data), "--out", str(tmp_path / "run"),
                "--device", "cpu", "--lora", "2", "--head-dim", "8", "--lora-targets", "qv",
                "--max-steps", "1", "--accum", "1", "--max-state", "16", "--epochs", str(epochs),
                "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0"])
    metrics = json.loads((out / "training_metrics.json").read_text())
    assert metrics["records_seen"] == 1
    assert metrics["requested_records"] == 2 * epochs
    assert metrics["input_records"] == 2
    assert metrics["admitted_records"] == 1
    assert metrics["rejected_records"] == 1
    admission = json.loads((out / "data_admission.json").read_text())
    assert admission["rejected"][0]["reason"] == "state exceeds 16 tokens: 41"


@pytest.mark.parametrize("evaluate", [False, True])
def test_checkpoint_saving_is_independent_of_evaluation(tiny_run, tmp_path, evaluate):
    root, _, _ = tiny_run
    args = ["--base", str(root / "base"), "--data", str(root / "data" / "train.jsonl"),
            "--out", str(tmp_path / "run"), "--device", "cpu", "--lora", "2", "--head-dim", "8",
            "--lora-targets", "qv", "--max-steps", "3", "--accum", "1",
            "--checkpoint-every-steps", "2", "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0"]
    if evaluate:
        args += ["--eval-data", str(root / "data" / "development.jsonl"), "--eval-every-steps", "3"]
    out = main(args)
    assert (out / "checkpoints" / "step-000002" / "head.pt").is_file()
    assert (out / "checkpoints" / "step-000003" / "head.pt").is_file()
    if evaluate:
        evaluation = out / "training_eval" / "step-000003"
        summary = json.loads((evaluation / "summary.json").read_text())
        assert summary["coverage"]["evaluated_records"] == 8
        assert not summary["calibration_fitted"]
        assert summary["temperature"] == 1.0
        assert summary["clean"] == summary["calibrated_clean"]
        assert not (evaluation / "calibration").exists()
    else:
        assert not (out / "training_eval").exists()


def test_validation_data_must_be_held_out(tmp_path):
    data = init_starter(tmp_path / "data") / "train.jsonl"
    with pytest.raises(ValueError, match="overlaps training data"):
        main(["--data", str(data), "--eval-data", str(data), "--eval-every-steps", "1",
              "--out", str(tmp_path / "run"), "--dry-run"])


def test_overlong_validation_records_are_counted_without_aborting_training(tiny_run, tmp_path):
    root, _, _ = tiny_run
    data = tmp_path / "validation.jsonl"
    question = {"q": {"type": "choice", "criteria": {"a": None, "b": None}, "label": "a"}}
    data.write_text("".join(json.dumps({"state": state, "questions": question}) + "\n"
                            for state in ("choose a", "state " * 2200)))
    out = main(["--base", str(root / "base"), "--data", str(root / "data" / "train.jsonl"),
                "--out", str(tmp_path / "run"), "--device", "cpu", "--lora", "2", "--head-dim", "8",
                "--lora-targets", "qv", "--max-steps", "1", "--accum", "1",
                "--eval-data", str(data), "--eval-every-steps", "1"])
    evaluation = out / "training_eval" / "step-000001"
    summary = json.loads((evaluation / "summary.json").read_text())
    assert summary["coverage"]["requested_records"] == 2
    assert summary["coverage"]["evaluated_records"] == 1
    assert summary["coverage"]["rejected_records"] == 1
    assert len(json.loads((evaluation / "development" / "rejected.json").read_text())) == 1


def test_preflight_training_records_are_reused(tiny_run, tmp_path, monkeypatch):
    from jevany import train as trainer
    root, _, _ = tiny_run
    original_load = trainer.load_records
    calls = []

    def load(filename):
        calls.append(str(filename))
        return original_load(filename)

    monkeypatch.setattr(trainer, "load_records", load)
    data = root / "data" / "train.jsonl"
    main(["--base", str(root / "base"), "--data", str(data), "--out", str(tmp_path / "run"),
          "--device", "cpu", "--lora", "2", "--head-dim", "8", "--lora-targets", "qv",
          "--max-steps", "1", "--accum", "1"])
    assert calls == [str(data)]


def test_multi_question_evaluation_uses_typed_packed_limit(tiny_run):
    from jevany.checkpoint import Checkpoint
    from jevany.predictors import ModelPredictor

    _, sft, _ = tiny_run
    tok, model = Checkpoint(sft).load("cpu")
    predictor = ModelPredictor(model, tok, "cpu", max_packed=64)
    record = {"state": "state", "questions": {
        str(index): {"type": "choice", "instructions": "choose",
                     "criteria": {"a": None, "b": None}, "label": "a"}
        for index in range(8)
    }}
    with pytest.raises(ContextLengthError, match="packed request exceeds"):
        predictor(record)


def test_evaluation_does_not_misclassify_other_tokenizer_errors(tmp_path):
    from jevany.benchmark import evaluate_records
    from jevany.data import load_records

    data = init_starter(tmp_path / "data") / "development.jsonl"

    def broken_predictor(record):
        raise ValueError("tokenizer returned invalid tokens")

    with pytest.raises(ValueError, match="tokenizer returned invalid tokens"):
        evaluate_records(load_records(data), broken_predictor, tmp_path / "evaluation", skip_overlong=True)


@pytest.mark.parametrize("rlcr", [False, True])
@pytest.mark.parametrize("resume_step", [1, 3])
def test_resume_matches_continuous_training_across_partial_and_complete_epochs(tiny_run, tmp_path, rlcr, resume_step):
    import torch
    from safetensors.torch import load_file
    from jevany.checkpoint import Checkpoint

    root, sft, _ = tiny_run
    args = ["--base", str(root / "base"), "--data", str(root / "data" / "train.jsonl"),
            "--out", str(tmp_path / "continuous"), "--device", "cpu", "--lora", "2", "--head-dim", "8",
            "--lora-targets", "qv", "--lora-dropout", "0.2", "--epochs", "2", "--batch", "5", "--accum", "2",
            "--max-steps", "6", "--checkpoint-every-steps", "1"]
    if rlcr:
        args += ["--rlcr", "--init-from", str(sft)]
    continuous = main(args)
    checkpoint = continuous / "checkpoints" / f"step-{resume_step:06d}"
    resumed = main(["--resume", str(checkpoint), "--out", str(tmp_path / "resumed")])
    expected, actual = (load_file(str(directory / "adapter_model.safetensors"))
                        for directory in (continuous, resumed))
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    expected_head, actual_head = (Checkpoint(directory).meta.head for directory in (continuous, resumed))
    assert all(torch.equal(expected_head[key], actual_head[key]) for key in expected_head)
    original_state, resumed_state = (torch.load(directory / "trainer_state.pt", weights_only=True)
                                     for directory in (continuous, resumed))
    assert original_state["scheduler"] == resumed_state["scheduler"]
    for key in ("records_seen", "tokens_seen", "optimizer_step", "epoch", "next_microbatch"):
        assert original_state[key] == resumed_state[key]


def test_resume_rejects_changed_schedule_data_and_world_size(tiny_run, tmp_path):
    import torch
    from jevany.resume import load_trainer_state, read_resume_config

    root, _, _ = tiny_run
    data = tmp_path / "train.jsonl"
    data.write_bytes((root / "data" / "train.jsonl").read_bytes())
    out = main(["--base", str(root / "base"), "--data", str(data), "--out", str(tmp_path / "original"),
                "--device", "cpu", "--lora", "2", "--head-dim", "8", "--lora-targets", "qv",
                "--max-steps", "2", "--accum", "1", "--checkpoint-every-steps", "1"])
    checkpoint = out / "checkpoints" / "step-000001"
    retry_out = tmp_path / "retry"
    with pytest.raises(ValueError, match="original training settings: max_steps"):
        main(["--resume", str(checkpoint), "--out", str(retry_out), "--max-steps", "3"])
    saved = torch.load(checkpoint / "trainer_state.pt", weights_only=True)
    with pytest.raises(ValueError, match="original world size"):
        load_trainer_state(checkpoint, read_resume_config(checkpoint)["args"], saved["dataset_sha256"], 2, "cpu")
    data.write_text(data.read_text() + data.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="training data changed"):
        main(["--resume", str(checkpoint), "--out", str(retry_out)])
    assert not retry_out.exists()


def test_rng_state_round_trip():
    import random
    import numpy as np
    import torch
    from jevany.resume import capture_rng, restore_rng

    shuffle = random.Random(7)
    state = capture_rng(shuffle, "cpu")
    expected = (shuffle.random(), random.random(), np.random.random(), torch.rand(3))
    restore_rng(state, shuffle, "cpu")
    actual = (shuffle.random(), random.random(), np.random.random(), torch.rand(3))
    assert expected[:3] == actual[:3]
    assert torch.equal(expected[3], actual[3])


def test_resume_fingerprint_preserves_option_order():
    from jevany.resume import dataset_fingerprint

    question = {"type": "choice", "criteria": {"a": None, "b": None}, "label": "a"}
    first = {"state": "state", "questions": {"q": question}}
    reordered = {"state": "state", "questions": {
        "q": {**question, "criteria": {"b": None, "a": None}},
    }}
    assert dataset_fingerprint([first]) != dataset_fingerprint([reordered])


def test_resume_retains_prior_best_checkpoint_and_early_stop_state(tiny_run, tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file

    root, _, _ = tiny_run

    def evaluate(model, tok, dev, suite, out, step, *args):
        scores = {"acc": 0.9 if step == 1 else 0.5, "nll": 1.0}
        return {"clean": scores, "calibrated_clean": scores, "temperature": 1.0}

    monkeypatch.setattr("jevany.train.evaluate_during_training", evaluate)
    original = main(["--base", str(root / "base"), "--data", str(root / "data" / "train.jsonl"),
                     "--out", str(tmp_path / "original"), "--device", "cpu", "--lora", "2",
                     "--head-dim", "8", "--lora-targets", "qv", "--max-steps", "6", "--accum", "1",
                     "--eval-data", str(root / "data" / "development.jsonl"),
                     "--eval-every-steps", "1", "--early-stop-patience", "2", "--checkpoint-every-steps", "1"])
    resumed = main(["--resume", str(original / "checkpoints" / "step-000002"),
                    "--out", str(tmp_path / "resumed")])
    expected, actual = (load_file(str(directory / "adapter_model.safetensors"))
                        for directory in (original, resumed))
    assert all(torch.equal(expected[key], actual[key]) for key in expected)
    metrics = json.loads((resumed / "training_metrics.json").read_text())
    assert metrics["early_stopped"] and metrics["optimizer_steps"] == 3
    selected = json.loads((resumed / "selection.json").read_text())["selected_checkpoint"]
    assert selected == str((original / "checkpoints" / "step-000001").resolve())
    assert metrics["selected_checkpoint"] == selected
