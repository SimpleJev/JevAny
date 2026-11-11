"""Media identity is shared by dataset deduplication and overlap checks."""
import copy
import json
from pathlib import Path

import pytest

from jevany.data import load_records, resolve_media
from jevany.suite import digest, load_split, read_jsonl, read_manifest, semantic_hash, validate_suite, write_json, write_jsonl


TEXT_HASH = "15504894a79f9615b2dfc9f0a2b0fb3db86d11f24ea0c9cae874bf781bb11abb"
INVALID_MEDIA = [
    pytest.param(None, id="null"),
    pytest.param(False, id="false"),
    pytest.param(0, id="zero"),
    pytest.param("", id="empty-string"),
    pytest.param({}, id="empty-object"),
    pytest.param([None], id="null-item"),
    pytest.param(["asset.png"], id="string-item"),
    pytest.param([{}], id="empty-item"),
    pytest.param([{"type": "audio", "uri": "asset"}], id="unsupported-type"),
    pytest.param([{"type": "image"}], id="missing-uri"),
    pytest.param([{"type": "image", "uri": None}], id="null-uri"),
    pytest.param([{"type": "image", "uri": ""}], id="empty-uri"),
    pytest.param([{"type": "image", "uri": 1}], id="numeric-uri"),
    pytest.param([{"type": "image", "uri": "asset"}] * 33, id="too-many"),
]


def record(identifier, media=()):
    return {
        "state": "Inspect the evidence.",
        "questions": {"q": {"type": "choice", "instructions": "Which object?",
                            "criteria": {"a": "A", "b": "B"}, "label": "a"}},
        "media": list(media),
        "_meta": {"source": "custom", "id": identifier, "group_id": identifier,
                  "text_sha256": TEXT_HASH},
    }


def make_suite(tmp_path, partitions):
    root = tmp_path / "suite"
    root.mkdir()
    files = {}
    for split, rows in partitions.items():
        path = root / f"{split}.jsonl"
        write_jsonl(path, rows)
        files[path.name] = {"sha256": digest(path), "records": len(rows)}
    write_json(root / "manifest.json", {"files": files, "trainable_sources": ["custom"]})
    return root


def write_invalid_media(path, media, *, labelled=True):
    valid = record("valid")
    valid["state"] = "first\u2028second"
    invalid = record("invalid")
    invalid["media"] = media
    if not labelled:
        for row in (valid, invalid):
            del row["questions"]["q"]["label"]
            del row["_meta"]
    path.write_text(
        "\n" + json.dumps(valid, ensure_ascii=False) + "\n\n" + json.dumps(invalid) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("media", INVALID_MEDIA)
@pytest.mark.parametrize("reader", ["load_split", "validate_suite"])
def test_suite_media_reports_physical_line(tmp_path, media, reader):
    root = make_suite(tmp_path, {"train": [record("valid"), record("invalid")]})
    path = root / "train.jsonl"
    write_invalid_media(path, media)
    manifest = read_manifest(root)
    manifest["files"][path.name]["sha256"] = digest(path)
    write_json(root / "manifest.json", manifest)

    with pytest.raises(ValueError, match=r"train\.jsonl:4:") as error:
        if reader == "load_split":
            load_split(root, "train")
        else:
            validate_suite(root)

    assert "media" in str(error.value)


@pytest.mark.parametrize("resolve_media_paths", [False, True])
def test_empty_jsonl_remains_supported(tmp_path, resolve_media_paths):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n \n", encoding="utf-8")

    assert read_jsonl(path, resolve_media_paths=resolve_media_paths) == []


def test_plain_jsonl_reader_keeps_arbitrary_json_values(tmp_path):
    rows = [None, False, 0, "", [], {}, {"media": None}, record("relative", [
        {"type": "image", "uri": "asset.png"},
    ])]
    path = tmp_path / "raw.jsonl"
    write_jsonl(path, rows)

    assert read_jsonl(path) == rows


@pytest.mark.parametrize("include_media", [False, True], ids=["omitted", "empty"])
def test_suite_empty_media_preserves_request_identity(tmp_path, include_media):
    from jevany.data import api_request
    from jevany.suite import record_digest

    row = record("train")
    if not include_media:
        del row["media"]
    expected = copy.deepcopy(row)
    root = make_suite(tmp_path, {"train": [row]})

    loaded = load_split(root, "train")[0]

    assert loaded == expected
    assert api_request(loaded) == {"state": row["state"], "questions": {
        "q": {"type": "choice", "instructions": "Which object?", "criteria": {"a": "A", "b": "B"}},
    }}
    assert record_digest(api_request(loaded)) == record_digest(api_request(expected))
    assert semantic_hash(loaded) == TEXT_HASH


def test_resolved_jsonl_preserves_unlabelled_media_and_metadata(tmp_path):
    row = record("request", [
        {"type": "image", "uri": "asset.png", "provenance": "keep"},
        {"type": "video", "uri": "https://example.com/clip.mp4"},
        {"type": "image", "uri": str(tmp_path / "absolute.png")},
    ])
    del row["questions"]["q"]["label"]
    row["extra"] = {"preserve": True}
    path = tmp_path / "requests.jsonl"
    write_jsonl(path, [row])
    before = path.read_bytes()
    expected = copy.deepcopy(row)
    expected["media"][0]["uri"] = str(tmp_path / "asset.png")

    assert read_jsonl(path, resolve_media_paths=True) == [expected]
    assert path.read_bytes() == before


def test_media_resolution_accepts_the_api_limit_without_requiring_labels(tmp_path):
    row = {"media": [{"type": "image", "uri": f"{index}.png"} for index in range(32)]}

    assert resolve_media(row, tmp_path) is row
    assert row["media"] == [{"type": "image", "uri": str(tmp_path / f"{index}.png")} for index in range(32)]


def test_media_validation_happens_before_resolving_any_paths(tmp_path):
    row = {"media": [{"type": "image", "uri": "asset.png"}, {"type": "video", "uri": ""}]}
    before = copy.deepcopy(row)

    with pytest.raises(ValueError, match="media"):
        resolve_media(row, tmp_path)

    assert row == before


@pytest.mark.parametrize("row", [None, [], "record"])
def test_resolved_jsonl_rejects_non_objects_with_location(tmp_path, row):
    path = tmp_path / "records.jsonl"
    path.write_text("\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"records\.jsonl:2:.*object"):
        read_jsonl(path, resolve_media_paths=True)


@pytest.mark.parametrize("resolve_media_paths", [False, True])
def test_invalid_json_reports_physical_line(tmp_path, resolve_media_paths):
    row = record("valid")
    row["state"] = "first\u2028second"
    path = tmp_path / "records.jsonl"
    path.write_text("\n" + json.dumps(row, ensure_ascii=False) + '\n\n{"state":', encoding="utf-8")

    with pytest.raises(ValueError, match=r"records\.jsonl:4:"):
        read_jsonl(path, resolve_media_paths=resolve_media_paths)


def test_text_hash_preserves_legacy_digest_and_normalization():
    row = record("first")
    assert semantic_hash(row) == TEXT_HASH
    del row["media"]
    row["state"] = "inspect  THE evidence."
    row["questions"]["q"].update(label="b", target={"b": 1.0}, src="other")
    row["_meta"] = {"text_sha256": "old provenance"}
    assert semantic_hash(row) == TEXT_HASH


@pytest.mark.parametrize("first_media,second_media", [
    ([{"type": "image", "uri": "a.png"}], [{"type": "image", "uri": "b.png"}]),
    ([{"type": "video", "uri": "a.mp4"}], [{"type": "video", "uri": "b.mp4"}]),
    ([{"type": "image", "uri": "A.png"}], [{"type": "image", "uri": "a.png"}]),
    ([{"type": "image", "uri": "a  b.png"}], [{"type": "image", "uri": "a b.png"}]),
    ([{"type": "image", "uri": "a.png"}], [{"type": "video", "uri": "a.png"}]),
    ([{"type": "image", "uri": "a.png"}, {"type": "video", "uri": "b.mp4"}],
     [{"type": "video", "uri": "b.mp4"}, {"type": "image", "uri": "a.png"}]),
    ([{"type": "image", "uri": "a.png"}],
     [{"type": "image", "uri": "a.png"}, {"type": "image", "uri": "a.png"}]),
    ([], [{"type": "image", "uri": "a.png"}]),
], ids=["images", "videos", "uri-case", "uri-whitespace", "type", "order", "multiplicity", "text-vs-media"])
def test_builder_deduplication_and_exclusions_use_media(first_media, second_media):
    from jevany.datasets.build_sft import unique

    first = record("first", first_media)
    second = record("second", second_media)
    duplicate = copy.deepcopy(first)
    duplicate["state"] = "inspect  THE evidence."
    duplicate["questions"]["q"].update(label="b", target={"b": 1.0}, src="other")
    duplicate["_meta"] = {"id": "duplicate", "text_sha256": "different provenance",
                          "media_sha256": ["different provenance"]}
    duplicate["media"] = [dict(reversed(list(item.items()))) for item in duplicate["media"]]

    kept = unique([first, second, duplicate], set())
    assert [row["_meta"]["id"] for row in kept] == ["first", "second"]
    assert first["_meta"]["text_sha256"] != second["_meta"]["text_sha256"]
    assert unique([duplicate, second], {semantic_hash(first)}) == [second]


@pytest.mark.parametrize("overlap", ["none", "media", "group"])
def test_suite_overlap_uses_media_and_preserves_group_checks(tmp_path, overlap):
    train = record("train", [{"type": "image", "uri": "a.png"}])
    development = record("development", [{"type": "image", "uri": "a.png" if overlap == "media" else "b.png"}])
    development["questions"]["q"]["label"] = "b"
    if overlap == "group":
        development["_meta"]["group_id"] = "train"
    root = make_suite(tmp_path, {"train": [train], "development": [development]})
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    if overlap == "none":
        assert set(validate_suite(root)["partitions"]) == {"train", "development"}
    else:
        kind = "decision text" if overlap == "media" else "group"
        with pytest.raises(ValueError, match=f"{kind} overlap between train and development"):
            validate_suite(root)
    assert load_split(root, "train")[0]["_meta"]["text_sha256"] == TEXT_HASH
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


@pytest.mark.parametrize("training_source", ["data", "suite", "replay"])
@pytest.mark.parametrize("same_media", [False, True])
def test_training_preflight_compares_resolved_media(tmp_path, monkeypatch, training_source, same_media):
    from jevany.train import main

    train = record("train", [{"type": "image", "uri": "a.png"}])
    root = make_suite(tmp_path, {"train": [train]})
    # Preflight checks file existence; media decoding happens when training starts.
    (root / "a.png").write_bytes(b"fixture")
    (root / "b.png").write_bytes(b"fixture")
    evaluation = record("evaluation", [{"type": "image", "uri": str(root / ("a.png" if same_media else "b.png"))}])
    evaluation["questions"]["q"]["label"] = "b"
    eval_data = tmp_path / "evaluation.jsonl"
    write_jsonl(eval_data, [evaluation])
    out = tmp_path / "run"
    args = ["--eval-data", str(eval_data), "--eval-every-steps", "1",
            "--out", str(out), "--multimodal", "--dry-run"]
    if training_source == "data":
        args += ["--data", str(root / "train.jsonl")]
    else:
        args += ["--suite", str(root), "--base-revision", "fixture"]
        if training_source == "replay":
            extra = record("extra")
            extra["state"] = "Unrelated text."
            data = tmp_path / "extra.jsonl"
            write_jsonl(data, [extra])
            args += ["--data", str(data), "--replay", "1"]
    monkeypatch.setattr("jevany.train.load_preprocessor",
                        lambda *args, **kwargs: pytest.fail("dry-run must not load a model"))

    if same_media:
        with pytest.raises(ValueError, match="overlaps training data"):
            main(args)
    else:
        assert main(args) == out
    assert not out.exists()


@pytest.mark.parametrize("media", [
    [{"type": "audio", "uri": "a.png"}],
    [{"type": "image", "uri": ""}],
    [{"type": "image"}],
])
def test_invalid_media_still_fails_record_validation(tmp_path, media):
    path = tmp_path / "invalid.jsonl"
    write_jsonl(path, [record("invalid", media)])
    with pytest.raises(ValueError, match=r"invalid\.jsonl:1:") as error:
        load_records(path)
    assert "media" in str(error.value)


@pytest.mark.parametrize("media_type", ["image", "video", None])
def test_rlcr_can_exclude_its_own_parent_partition(tmp_path, media_type):
    from jevany.datasets import build_rlcr

    media = [{"type": media_type, "uri": "asset"}] if media_type else []
    root = make_suite(tmp_path, {"train": [record("train", media)]})
    (root / "asset").write_bytes(b"fixture")
    output = tmp_path / "rlcr"
    quotas = {**dict.fromkeys(build_rlcr.QUOTAS, 0), "core": 1}
    with pytest.raises(ValueError, match="core: requested 1 records but only 0 are available"):
        build_rlcr.main(["--suite", str(root), "--out", str(output),
                         "--exclude", str(root / "train.jsonl"), "--quotas", json.dumps(quotas)])
    assert not output.exists()


@pytest.mark.parametrize("media", INVALID_MEDIA)
def test_rlcr_exclusion_media_reports_physical_line(tmp_path, media):
    from jevany.datasets import build_rlcr

    root = make_suite(tmp_path, {"train": [record("train")]})
    excluded = tmp_path / "exclude.jsonl"
    write_invalid_media(excluded, media, labelled=False)
    output = tmp_path / "rlcr"
    quotas = dict.fromkeys(build_rlcr.QUOTAS, 0)

    with pytest.raises(ValueError, match=r"exclude\.jsonl:4:") as error:
        build_rlcr.main(["--suite", str(root), "--out", str(output),
                         "--exclude", str(excluded), "--quotas", json.dumps(quotas)])

    assert "media" in str(error.value)
    assert not output.exists()


def test_rlcr_accepts_unlabelled_exclusions(tmp_path):
    from jevany.datasets import build_rlcr

    train = record("train", [{"type": "image", "uri": "asset.png"}])
    root = make_suite(tmp_path, {"train": [train]})
    exclusion = copy.deepcopy(train)
    del exclusion["questions"]["q"]["label"]
    del exclusion["_meta"]
    exclusion["media"][0]["uri"] = "suite/asset.png"
    excluded = tmp_path / "exclude.jsonl"
    write_jsonl(excluded, [exclusion])
    output = tmp_path / "rlcr"
    quotas = {**dict.fromkeys(build_rlcr.QUOTAS, 0), "core": 1}

    with pytest.raises(ValueError, match="core: requested 1 records but only 0 are available"):
        build_rlcr.main(["--suite", str(root), "--out", str(output),
                         "--exclude", str(excluded), "--quotas", json.dumps(quotas)])

    assert not output.exists()


@pytest.fixture
def offline_sft(monkeypatch):
    from jevany.datasets import build_sft

    monkeypatch.setattr(build_sft, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(build_sft, "admitted", lambda *args: True)
    monkeypatch.setattr(build_sft, "load_source",
                        lambda *args: pytest.fail("unexpected dataset load"))
    return build_sft


def sft_args(output):
    return ["--out", str(output)] + [
        arg for flag in ("helpsteer", "agent", "hard-text", "scienceqa", "aokvqa", "video", "eval-per-source")
        for arg in (f"--{flag}", "0")
    ]


@pytest.mark.parametrize("media", INVALID_MEDIA)
@pytest.mark.parametrize("flag", ["--include-train", "--exclude"])
def test_sft_input_media_reports_physical_line(tmp_path, offline_sft, media, flag):
    path = tmp_path / "input.jsonl"
    write_invalid_media(path, media, labelled=flag == "--include-train")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match=r"input\.jsonl:4:") as error:
        offline_sft.main(sft_args(output) + [flag, str(path)])

    assert "media" in str(error.value)
    assert not (output / "manifest.json").exists()


def test_sft_included_files_keep_their_own_relative_media(tmp_path, offline_sft):
    output = tmp_path / "output"
    args = sft_args(output)
    expected = set()
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        asset = directory / "asset.png"
        asset.write_bytes(b"fixture")
        path = directory / "train.jsonl"
        write_jsonl(path, [record(name, [{"type": "image", "uri": "asset.png"}])])
        args += ["--include-train", str(path)]
        expected.add(str(asset))

    offline_sft.main(args)

    rows = load_split(output, "train")
    assert {row["media"][0]["uri"] for row in rows} == expected
    assert all(not Path(row["media"][0]["uri"]).is_absolute()
               for row in read_jsonl(output / "train.jsonl"))


@pytest.mark.parametrize("same_file", [False, True])
def test_sft_exclusions_resolve_media_against_the_exclusion_file(tmp_path, offline_sft, same_file):
    included = tmp_path / "included"
    excluded = tmp_path / "excluded"
    included.mkdir()
    excluded.mkdir()
    for directory in (included, excluded):
        (directory / "asset.png").write_bytes(b"fixture")
    train = record("train", [{"type": "image", "uri": "asset.png"}])
    write_jsonl(included / "train.jsonl", [train])
    exclusion = copy.deepcopy(train)
    exclusion["media"][0]["uri"] = "../included/asset.png" if same_file else "asset.png"
    del exclusion["questions"]["q"]["label"]
    del exclusion["_meta"]
    write_jsonl(excluded / "exclude.jsonl", [exclusion])
    output = tmp_path / "output"

    offline_sft.main(sft_args(output) + ["--include-train", str(included / "train.jsonl"),
                                       "--exclude", str(excluded / "exclude.jsonl")])

    assert len(read_jsonl(output / "train.jsonl")) == (0 if same_file else 1)


@pytest.mark.parametrize("extra_input", ["none", "exclude", "include"])
def test_sft_generated_media_uses_the_output_directory(tmp_path, monkeypatch, offline_sft, extra_input):
    from PIL import Image

    source_row = {"image": Image.new("RGB", (1, 1)), "question": "Which object?",
                  "choices": ["A", "B"], "answer": 0}
    output = tmp_path / "output"
    generated = offline_sft.visual_record(source_row, "scienceqa", 0, "train", output)
    monkeypatch.setattr(offline_sft, "load_source",
                        lambda source, split: ([source_row] * (1 if split == "train" else 2)
                                                if source == "scienceqa" else []))
    args = sft_args(output) + ["--scienceqa", "1", "--eval-per-source", "1", "--resume"]
    if extra_input == "exclude":
        excluded = copy.deepcopy(generated)
        excluded["media"][0]["uri"] = str(Path("output") / excluded["media"][0]["uri"])
        path = tmp_path / "exclude.jsonl"
        write_jsonl(path, [excluded])
        args += ["--exclude", str(path)]
    elif extra_input == "include":
        included = tmp_path / "included"
        imported = offline_sft.visual_record(source_row, "scienceqa", 0, "train", included)
        path = included / "train.jsonl"
        write_jsonl(path, [imported])
        args += ["--include-train", str(path)]

    offline_sft.main(args)

    counts = {"train": {"none": 1, "exclude": 0, "include": 2}[extra_input],
              "calibration": 1, "development": 1}
    for split, count in counts.items():
        raw = read_jsonl(output / f"{split}.jsonl")
        assert len(raw) == count
        assert all(not Path(item["uri"]).is_absolute() for row in raw for item in row["media"])
        assert all(Path(item["uri"]).is_file() for row in load_split(output, split) for item in row["media"])


def test_sft_generated_partitions_are_reproducible_across_output_roots(tmp_path, monkeypatch, offline_sft):
    from PIL import Image

    source_row = {"image": Image.new("RGB", (1, 1)), "question": "Which object?",
                  "choices": ["A", "B"], "answer": 0}
    monkeypatch.setattr(offline_sft, "load_source",
                        lambda source, split: ([source_row] * (1 if split == "train" else 2)
                                                if source == "scienceqa" else []))
    outputs = [tmp_path / name for name in ("first", "second")]
    for output in outputs:
        offline_sft.main(sft_args(output) + ["--scienceqa", "1", "--eval-per-source", "1"])

    for split in ("train", "calibration", "development"):
        filename = f"{split}.jsonl"
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
        for output in outputs:
            rows = read_jsonl(output / filename)
            assert len(rows) == 1
            assert rows[0]["_meta"]["text_sha256"] == semantic_hash(rows[0])
        assert read_manifest(outputs[0])["files"][filename] == read_manifest(outputs[1])["files"][filename]
