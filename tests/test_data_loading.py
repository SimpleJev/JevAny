"""Labelled JSONL validates request fields and reports physical input lines."""
import json

import pytest

from jevany.data import load_records, materialize, validate_dataset


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
def test_unicode_separators_inside_json_strings_are_preserved(tmp_path, separator):
    row = {
        "state": {"note": f"north{separator}south"},
        "questions": {
            "q": {
                "type": "choice",
                "instructions": f"Choose{separator}one",
                "criteria": {"a": f"left{separator}right", "b": "other"},
                "label": "a",
            }
        },
    }
    path = tmp_path / "records.jsonl"
    path.write_text("\n" + json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    records = load_records(path, source="fixture")

    assert len(records) == 1
    assert records[0]["state"] == row["state"]
    question = records[0]["questions"]["q"]
    assert question["instructions"] == row["questions"]["q"]["instructions"]
    assert question["criteria"] == row["questions"]["q"]["criteria"]
    assert records[0]["_meta"]["row"] == 1
    assert records[0]["_meta"]["id"] == "fixture/1"
    assert validate_dataset(path)["questions"] == 1


def test_invalid_json_after_unicode_evidence_reports_physical_line(tmp_path):
    row = {"state": "first\u2028second", "questions": {"q": {"type": "noul", "label": True}}}
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n" + json.dumps(row, ensure_ascii=False) + '\n\n{"state":',
        encoding="utf-8", newline="\r\n",
    )

    with pytest.raises(ValueError, match=r"records.jsonl:4:"):
        load_records(path)


@pytest.mark.parametrize("media", [None, False, 0, "", {}])
def test_materialize_rejects_invalid_media(media):
    row = {"state": "evidence", "media": media, "questions": {"q": {"type": "noul", "label": True}}}

    with pytest.raises(ValueError, match="media"):
        materialize(row)


@pytest.mark.parametrize("media", [None, False, 0, "", {}])
def test_invalid_media_reports_physical_line(tmp_path, media):
    row = {"state": "evidence", "media": media, "questions": {"q": {"type": "noul", "label": True}}}
    path = tmp_path / "records.jsonl"
    path.write_text("\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"records.jsonl:2:") as error:
        load_records(path)

    assert "media" in str(error.value)


@pytest.mark.parametrize("include_media", [False, True], ids=["omitted", "empty"])
def test_records_without_media_remain_valid(tmp_path, include_media):
    row = {"state": "evidence", "questions": {"q": {"type": "noul", "label": True}}}
    if include_media:
        row["media"] = []
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    record = load_records(path)[0]

    assert record.get("media", []) == []
    assert "media" not in materialize(record)
    assert validate_dataset(path)["records"] == 1


def test_valid_media_paths_survive_loading_and_materialization(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image fixture")
    media = [
        {"type": "image", "uri": image.name},
        {"type": "video", "uri": "https://example.com/clip.mp4"},
    ]
    row = {"state": "evidence", "media": media, "questions": {"q": {"type": "noul", "label": True}}}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    record = load_records(path)[0]
    expected = [{"type": "image", "uri": str(image.resolve())}, media[1]]

    assert record["media"] == expected
    assert materialize(record)["media"] == expected
    assert validate_dataset(path)["records"] == 1
