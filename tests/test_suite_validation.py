"""Frozen-suite checks run without models and preserve the locked-test boundary."""
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from jevany.cli import main
from jevany.suite import digest, read_manifest, validate_suite, write_json, write_jsonl


def record(split, *, state=None, group=None):
    return {"state": state or split, "questions": {
        "q": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "label": "a"},
    }, "_meta": {"source": "custom", "id": f"{split}/1", "group_id": group or f"{split}/1"}}


def make_suite(tmp_path, splits=("train", "development"), rows=None):
    root = tmp_path / "suite"
    root.mkdir()
    files = {}
    for split in splits:
        values = rows[split] if rows is not None else [record(split)]
        path = root / f"{split}.jsonl"
        write_jsonl(path, values)
        files[path.name] = {"sha256": digest(path), "records": len(values),
                            "questions": sum(len(row.get("questions", {})) for row in values)}
    write_json(root / "manifest.json", {"files": files, "trainable_sources": ["custom"]})
    return root


def test_suite_validates_records_and_optional_metadata(tmp_path):
    root = make_suite(tmp_path)
    report = validate_suite(root)
    assert report == {"partitions": {
        split: {"records": 1, "questions": 1, "sources": {"custom": 1}}
        for split in ("train", "development")
    }, "skipped": []}


@pytest.mark.parametrize("field,value,match", [
    ("sha256", "wrong", "checksum"), ("records", 2, "record count"),
    ("questions", 2, "question count"), ("records", True, "nonnegative integer"),
    ("questions", -1, "nonnegative integer"), ("sha256", None, "sha256"),
])
def test_suite_checks_file_manifest_fields(tmp_path, field, value, match):
    root = make_suite(tmp_path)
    manifest = read_manifest(root)
    manifest["files"]["train.jsonl"][field] = value
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match=match):
        validate_suite(root)


@pytest.mark.parametrize("manifest", [[], {}, {"files": []}, {"files": {}},
                                     {"files": {}, "holdout_sources": "custom"}])
def test_suite_reports_invalid_manifest_shapes(tmp_path, manifest):
    root = make_suite(tmp_path)
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="manifest.json"):
        validate_suite(root)


@pytest.mark.parametrize("restriction", ["eval_only_sources", "holdout_sources"])
def test_suite_rejects_heldout_sources_in_training(tmp_path, restriction):
    root = make_suite(tmp_path)
    manifest = read_manifest(root)
    manifest[restriction] = ["custom"]
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="training source"):
        validate_suite(root)


@pytest.mark.parametrize("kind", ["label", "metadata"])
def test_suite_rejects_bad_records_with_record_location(tmp_path, kind):
    row = record("train")
    if kind == "label":
        row["questions"]["q"]["label"] = "unknown"
    else:
        del row["_meta"]["group_id"]
    root = make_suite(tmp_path, splits=("train",), rows={"train": [row]})
    with pytest.raises(ValueError, match="train.jsonl record 1:"):
        validate_suite(root)


@pytest.mark.parametrize("kind", ["decision text", "group"])
def test_suite_rejects_cross_split_text_and_group_overlap(tmp_path, kind):
    train = record("train")
    development = record("development", state="TRAIN" if kind == "decision text" else None,
                         group="train/1" if kind == "group" else None)
    root = make_suite(tmp_path, rows={"train": [train], "development": [development]})
    with pytest.raises(ValueError, match=f"{kind} overlap between train and development"):
        validate_suite(root)


def test_repeated_records_in_one_partition_remain_supported(tmp_path):
    row = record("train")
    root = make_suite(tmp_path, splits=("train",), rows={"train": [row, copy.deepcopy(row)]})
    assert validate_suite(root)["partitions"]["train"]["records"] == 2


def test_locked_test_is_not_read_without_explicit_opt_in(tmp_path, monkeypatch, capsys):
    from jevany import suite

    root = make_suite(tmp_path, splits=("train", "test"))
    original = suite.load_split
    loaded = []

    def load(directory, split, allow_test=False):
        loaded.append(split)
        return original(directory, split, allow_test=allow_test)

    monkeypatch.setattr(suite, "load_split", load)
    main(["data", "check-suite", str(root)])
    assert json.loads(capsys.readouterr().out)["skipped"] == ["test"]
    assert loaded == ["train"]
    loaded.clear()
    main(["data", "check-suite", str(root), "--allow-test"])
    report = json.loads(capsys.readouterr().out)
    assert report["skipped"] == [] and report["partitions"]["test"]["records"] == 1
    assert loaded == ["train", "test"]


def test_test_only_suite_requires_opt_in_instead_of_checking_nothing(tmp_path):
    root = make_suite(tmp_path, splits=("test",))
    with pytest.raises(ValueError, match="allow-test"):
        validate_suite(root)
    assert validate_suite(root, allow_test=True)["partitions"]["test"]["records"] == 1


def test_suite_check_needs_no_model_dependencies(tmp_path):
    root = make_suite(tmp_path)
    script = """
import sys
class NoOptionalImports:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "transformers", "datasets", "boto3"}:
            raise ImportError("optional dependency unavailable: " + name)
        return None
sys.meta_path.insert(0, NoOptionalImports())
from jevany.cli import main
main(["data", "check-suite", sys.argv[1]])
"""
    result = subprocess.run([sys.executable, "-c", script, str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["partitions"]["train"]["records"] == 1


def test_suite_without_base_revision_map_accepts_explicit_pin():
    from jevany.train import pinned_revision

    args = SimpleNamespace(base="model", base_revision="revision")
    assert pinned_revision(args, {"files": {}}) == "revision"
    args.base_revision = ""
    with pytest.raises(ValueError, match="base not pinned"):
        pinned_revision(args, {"files": {}})


def test_remote_evaluation_accepts_suite_without_holdout_metadata(tmp_path, monkeypatch):
    from jevany import benchmark

    root = make_suite(tmp_path)
    seen = []
    monkeypatch.setattr(benchmark, "RemotePredictor", lambda *_: SimpleNamespace(served_model="fixture"))

    def evaluate(records, predictor, output, *, heldout_sources, **kwargs):
        seen.append(heldout_sources)
        Path(output).mkdir()
        return {"objective": 1, "clean": {}, "coverage": {}}, []

    monkeypatch.setattr(benchmark, "evaluate_records", evaluate)
    benchmark.main(["--remote", "http://localhost:8008", "--suite", str(root), "--out", str(tmp_path / "report")])
    assert seen == [()]
