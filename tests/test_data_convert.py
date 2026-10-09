"""Local multiple-choice imports preserve provenance and never mix splits."""
import json
import subprocess
import sys

import pytest

from jevany.cli import main
from jevany.data import api_request, materialize
from jevany.datasets.convert import convert_jsonl
from jevany.suite import digest, load_split, read_manifest


def write_export(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("split", ["train", "calibration", "development", "test"])
def test_conversion_preserves_keys_evidence_groups_and_revision(tmp_path, split):
    export = write_export(tmp_path / "export.jsonl", [
        {"question": {"stem": "Which way?"}, "choices": {"text": ["北", "南"], "label": ["N", "S"]},
         "answerKey": "S", "fact1": "The destination is south.", "group_id": "shared"},
        {"question": "Pick one", "options": "['zero', 'one']", "answer": "B"},
    ])
    output = convert_jsonl(export, tmp_path / "suite", source="custom", revision="release-1", split=split)
    rows = load_split(output, split, allow_test=split == "test")
    assert [row["questions"]["answer"]["label"] for row in rows] == ["S", "1"]
    assert rows[0]["questions"]["answer"]["criteria"] == {"N": "北", "S": "南"}
    assert rows[0]["state"] == {"fact1": "The destination is south."}
    assert rows[0]["_meta"]["group_id"] == "custom/shared"
    assert {row["_meta"]["split"] for row in rows} == {split}
    assert {row["_meta"]["revision"] for row in rows} == {"release-1"}
    assert rows[1]["_meta"]["id"] == f"custom/{split}/2"
    assert all("_meta" not in api_request(row) and "label" not in api_request(row)["questions"]["answer"]
               for row in rows)
    assert materialize(rows[0])["questions"][0]["label"] == 1
    manifest = read_manifest(output)
    assert manifest["sources"]["custom"]["input_sha256"] == digest(export)
    assert manifest["files"][f"{split}.jsonl"]["records"] == 2
    assert manifest["trainable_sources"] == (["custom"] if split == "train" else [])
    assert sorted(path.name for path in output.iterdir()) == sorted([f"{split}.jsonl", "manifest.json"])


@pytest.mark.parametrize("row", [
    [], {},
    {"question": "", "options": ["a", "b"], "answer": "a"},
    {"question": "Pick", "options": ["a", "b"], "answer": -1},
    {"question": "Pick", "options": ["a", "b"], "answer": 2},
    {"question": "Pick", "choices": {"text": ["a", "b"], "label": ["A"]}, "answer": "A"},
    {"question": "Pick", "options": ["None", "other"]},
    {"question": "Pick", "options": ["None", "other"], "answer": None},
    {"question": "Pick", "options": ["a", "b"], "answer": "a", "group_id": []},
])
def test_invalid_rows_report_physical_line_without_creating_output(tmp_path, row):
    export = tmp_path / "export.jsonl"
    export.write_text('\n' + json.dumps(row) + "\n")
    output = tmp_path / "suite"
    with pytest.raises(ValueError, match=r"export.jsonl:2:"):
        convert_jsonl(export, output, source="custom", revision="r1", split="train")
    assert not output.exists()


@pytest.mark.parametrize("content", ["", "\n\n", '{"question":'])
def test_empty_or_invalid_json_does_not_create_a_suite(tmp_path, content):
    export = tmp_path / "export.jsonl"
    export.write_text(content)
    output = tmp_path / "suite"
    with pytest.raises(ValueError, match="export.jsonl"):
        convert_jsonl(export, output, source="custom", revision="r1", split="train")
    assert not output.exists()


@pytest.mark.parametrize("overrides", [
    {"source": ""}, {"revision": " "}, {"split": "random"}, {"source": "mmlu"},
])
def test_conversion_rejects_invalid_provenance_and_eval_only_training(tmp_path, overrides):
    export = write_export(tmp_path / "export.jsonl", [{"question": "Pick", "options": ["a", "b"], "answer": "a"}])
    output = tmp_path / "suite"
    with pytest.raises(ValueError):
        convert_jsonl(export, output, **{"source": "custom", "revision": "r1", "split": "train", **overrides})
    assert not output.exists()


def test_converter_cli_creates_a_suite_and_refuses_to_overwrite_it(tmp_path, capsys):
    export = write_export(tmp_path / "export.jsonl", [{"question": "Pick", "options": ["a", "b"], "answer": "a"}])
    output = tmp_path / "suite"
    args = ["data", "convert", str(export), "--out", str(output), "--source", "custom",
            "--revision", "r1", "--split", "train"]
    main(args)
    before = (output / "train.jsonl").read_bytes()
    assert capsys.readouterr().out.strip() == str(output)
    with pytest.raises(SystemExit) as caught:
        main(args)
    assert caught.value.code == 2 and "refusing to overwrite" in capsys.readouterr().err
    assert (output / "train.jsonl").read_bytes() == before


def test_converter_runs_without_model_dataset_or_cloud_packages(tmp_path):
    export = write_export(tmp_path / "export.jsonl", [{"question": "Pick", "options": ["a", "b"], "answer": 1}])
    script = """
import sys

class NoOptionalImports:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "transformers", "datasets", "boto3"}:
            raise ImportError("optional dependency unavailable: " + name)
        return None

sys.meta_path.insert(0, NoOptionalImports())
from jevany.cli import main
main(["data", "convert", sys.argv[1], "--out", sys.argv[2], "--source", "fixture",
      "--revision", "r1", "--split", "development"])
"""
    result = subprocess.run([sys.executable, "-c", script, str(export), str(tmp_path / "suite")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(load_split(tmp_path / "suite", "development")) == 1
