"""RLCR continuation keeps the parent suite's evaluation boundaries."""
import json

import pytest

from jevany.datasets import init_starter
from jevany.datasets import build_rlcr
from jevany.suite import digest, load_split, read_jsonl, read_manifest, write_json


def parent_suite(tmp_path, **restrictions):
    root = init_starter(tmp_path / "parent")
    rows = read_jsonl(root / "train.jsonl")
    write_json(root / "manifest.json", {
        "files": {"train.jsonl": {
            "records": len(rows), "questions": sum(len(row["questions"]) for row in rows),
            "sha256": digest(root / "train.jsonl"),
        }},
        "trainable_sources": ["jevany_starter"],
        "eval_only_sources": ["evaluation_only"],
        "holdout_sources": ["held_out"],
        **restrictions,
    })
    return root


def build_args(root, output, **quotas):
    return ["--suite", str(root), "--out", str(output),
            "--quotas", json.dumps({**dict.fromkeys(build_rlcr.QUOTAS, 0), "core": 1, **quotas})]


def test_rlcr_preserves_parent_source_restrictions(tmp_path, monkeypatch):
    root = parent_suite(tmp_path)
    monkeypatch.setattr(build_rlcr, "reasoning_rows", lambda *_: [])
    output = tmp_path / "rlcr"
    build_rlcr.main(build_args(root, output))
    manifest = read_manifest(output)
    assert manifest["eval_only_sources"] == ["evaluation_only"]
    assert manifest["holdout_sources"] == ["held_out"]
    assert manifest["trainable_sources"] == ["jevany_starter"]
    assert manifest["parent_suite"]["manifest_sha256"] == digest(root / "manifest.json")
    assert len(load_split(output, "train")) == 1


@pytest.mark.parametrize("restriction", ["eval_only_sources", "holdout_sources", "trainable_sources"])
def test_rlcr_rejects_forbidden_parent_records_before_creating_output(tmp_path, monkeypatch, restriction):
    forbidden = ["other"] if restriction == "trainable_sources" else ["jevany_starter"]
    root = parent_suite(tmp_path, **{restriction: forbidden})
    monkeypatch.setattr(build_rlcr, "reasoning_rows", lambda *_: [])
    output = tmp_path / "rlcr"
    with pytest.raises(ValueError, match="training source.*jevany_starter"):
        build_rlcr.main(build_args(root, output))
    assert not output.exists()


def test_added_reasoning_data_cannot_override_parent_holdouts(tmp_path, monkeypatch):
    root = parent_suite(tmp_path, holdout_sources=["aqua_rat"])
    row = build_rlcr.reasoning_record("aqua_rat", 1, "sum", ["one", "two"], 1, "Add", "revision")
    monkeypatch.setattr(build_rlcr, "reasoning_rows", lambda *_: [row])
    output = tmp_path / "rlcr"
    with pytest.raises(ValueError, match="training source.*aqua_rat"):
        build_rlcr.main(build_args(root, output, math_reasoning=1))
    assert not output.exists()


def test_insufficient_records_leave_output_available_for_a_corrected_run(tmp_path, monkeypatch):
    root = parent_suite(tmp_path)
    monkeypatch.setattr(build_rlcr, "reasoning_rows", lambda *_: [])
    output = tmp_path / "rlcr"
    with pytest.raises(ValueError, match="requested"):
        build_rlcr.main(build_args(root, output, core=100))
    assert not output.exists()
