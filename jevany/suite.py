# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Read immutable JSONL suites and verify their manifests."""
import hashlib
import json
from collections import Counter
from pathlib import Path

from .data import EVAL_ONLY, materialize, resolve_media

ENCODING = "utf-8"
SPLITS = ("train", "calibration", "development", "test")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def record_digest(record):
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def semantic_hash(record):
    """Hash decision content without labels or provenance metadata."""
    questions = {
        question_id: {key: value for key, value in question.items()
                      if key in ("type", "instructions", "criteria")}
        for question_id, question in record["questions"].items()
    }
    payload = json.dumps({"state": record["state"], "questions": questions}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(" ".join(payload.casefold().split()).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding=ENCODING))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding=ENCODING)


def read_jsonl(path):
    with Path(path).open(encoding=ENCODING) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    body = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    Path(path).write_text(body, encoding=ENCODING)


def read_manifest(directory):
    return read_json(Path(directory) / "manifest.json")


def load_split(directory, split, allow_test=False):
    if split not in SPLITS:
        raise ValueError(f"unknown split: {split}")
    if split == "test" and not allow_test:
        raise ValueError("locked test requires explicit --allow-test")
    directory = Path(directory)
    manifest = read_manifest(directory)
    path = directory / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing suite partition: {path}")
    expected = manifest["files"][path.name]
    if digest(path) != expected["sha256"]:
        raise ValueError(f"suite checksum mismatch: {path}")
    records = [resolve_media(record, path.parent) for record in read_jsonl(path)]
    if len(records) != expected["records"]:
        raise ValueError(f"suite record count mismatch: {path}")
    return records


def validate_training(records, manifest):
    allowed = set(manifest.get("trainable_sources", []))
    forbidden = set(EVAL_ONLY) | set(manifest.get("eval_only_sources", [])) | set(manifest.get("holdout_sources", []))
    for record in records:
        source = record["_meta"]["source"]
        if source in forbidden or (allowed and source not in allowed):
            raise ValueError(f"eval-only or undeclared training source: {source}")
    if not records:
        raise ValueError("empty training partition")


def validate_suite(directory: str | Path, *, allow_test: bool = False) -> dict:
    """Check declared partitions, labelled records, source restrictions and split overlap.

    The locked test partition is skipped unless explicitly requested. Text
    overlap uses the same label-free semantic hash as the public data builders;
    related records also stay together by source and group_id.
    """
    directory = Path(directory)
    manifest = read_manifest(directory)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError(f"{directory}/manifest.json: files must be an object")
    for field in ("trainable_sources", "eval_only_sources", "holdout_sources"):
        sources = manifest.get(field, [])
        if not isinstance(sources, list) or any(not isinstance(source, str) or not source for source in sources):
            raise ValueError(f"{directory}/manifest.json: {field} must be a list of source names")
    partitions, skipped = {}, []
    seen_text, seen_groups = {}, {}
    for split in SPLITS:
        filename = f"{split}.jsonl"
        if filename not in manifest["files"]:
            continue
        if split == "test" and not allow_test:
            skipped.append(split)
            continue
        expected = manifest["files"][filename]
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise ValueError(f"{filename}: manifest entry must contain sha256 and records")
        for field in ("records", "questions"):
            if field == "questions" and field not in expected:
                continue
            if type(expected.get(field)) is not int or expected[field] < 0:
                raise ValueError(f"{filename}: {field} must be a nonnegative integer")
        records = load_split(directory, split, allow_test=allow_test)
        sources = Counter()
        for index, row in enumerate(records, start=1):
            try:
                materialize(row)
                meta = row.get("_meta")
                if not isinstance(meta, dict) or any(
                    not isinstance(meta.get(key), str) or not meta[key]
                    for key in ("source", "id", "group_id")
                ):
                    raise ValueError("_meta must contain non-empty source, id and group_id strings")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{filename} record {index}: {error}") from error
            sources[meta["source"]] += 1
            for kind, key, seen in (
                ("decision text", semantic_hash(row), seen_text),
                ("group", (meta["source"], meta["group_id"]), seen_groups),
            ):
                previous = seen.setdefault(key, split)
                if previous != split:
                    raise ValueError(f"{kind} overlap between {previous} and {split}: {meta['id']}")
        if split == "train":
            validate_training(records, manifest)
        questions = sum(len(row["questions"]) for row in records)
        if "questions" in expected and questions != expected["questions"]:
            raise ValueError(f"suite question count mismatch: {filename}")
        partitions[split] = {"records": len(records), "questions": questions, "sources": dict(sources)}
    if not partitions:
        if skipped:
            raise ValueError("suite contains only a locked test partition; pass --allow-test to check it")
        raise ValueError(f"{directory}/manifest.json: no supported JSONL partitions")
    return {"partitions": partitions, "skipped": skipped}
