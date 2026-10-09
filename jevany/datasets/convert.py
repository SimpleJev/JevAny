"""Convert local multiple-choice exports to labelled, revisioned System One JSONL."""
import argparse
import ast
import json
from pathlib import Path

from jevany.data import materialize
from jevany.suite import SPLITS, digest, semantic_hash, validate_training, write_json, write_jsonl


def metadata(source, identifier, split, state, questions, **extra):
    identifier = str(identifier)
    return {"source": source, "variant": "clean", "id": f"{source}/{identifier}",
            "group_id": f"{source}/{identifier}", "row": identifier, "split": split,
            "text_sha256": semantic_hash({"state": state, "questions": questions}), **extra}


def record(source, identifier, split, state, questions, media=None, **extra):
    result = {"state": state, "questions": questions}
    if media:
        result["media"] = media
    result["_meta"] = metadata(source, identifier, split, state, questions, **extra)
    return result


def choice_parts(row: dict) -> tuple[list[str], list[str]]:
    choices = row.get("choices", row.get("options"))
    if isinstance(choices, str):
        try:
            choices = ast.literal_eval(choices)
        except (ValueError, SyntaxError) as error:
            raise ValueError("choices must be a list of options or a text/label object") from error
    if isinstance(choices, dict):
        texts, keys = choices.get("text"), choices.get("label")
    else:
        texts, keys = choices, None
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("choices must contain a non-empty list of option texts")
    if keys is None:
        keys = [str(index) for index in range(len(texts))]
    if not isinstance(keys, (list, tuple)) or len(keys) != len(texts):
        raise ValueError("option labels and texts must be lists of equal length")
    keys = [str(value) for value in keys]
    if len(set(keys)) != len(keys):
        raise ValueError("option labels must be unique")
    return keys, [str(value) for value in texts]


def answer_key(answer, keys: list[str], choices: list[str]) -> str:
    if answer is None:
        raise ValueError("answer is required")
    answer = str(answer)
    if answer in keys:
        return answer
    if answer in choices:
        return keys[choices.index(answer)]
    if len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(answer.upper()) - ord("A")
        if index < len(keys):
            return keys[index]
    try:
        index = int(answer)
    except ValueError as error:
        raise ValueError(f"answer {answer!r} is not an option key, text, letter or index") from error
    if not 0 <= index < len(keys):
        raise ValueError(f"answer index must be in 0..{len(keys) - 1}, got {answer!r}")
    return keys[index]


def text_choice_record(row: dict, source: str, identifier: str | int, split: str) -> dict:
    try:
        keys, choices = choice_parts(row)
        label = answer_key(row.get("answerKey", row.get("answer")), keys, choices)
    except ValueError as error:
        raise ValueError(f"{source}/{identifier}: {error}") from error
    state = {key: row[key] for key in ("hint", "fact1", "fact2", "combinedfact") if row.get(key)}
    if not state:
        state = "Choose the best supported answer."
    question = row.get("question")
    if isinstance(question, dict):
        question = question.get("stem", str(question))
    questions = {"answer": {"type": "choice", "instructions": question, "criteria": dict(zip(keys, choices)),
                            "label": label,
                            "src": f"{source}_choice"}}
    return record(source, identifier, split, state, questions)


def convert_jsonl(
    path: str | Path, output_dir: str | Path, *, source: str, revision: str, split: str,
) -> Path:
    """Import one explicit split without downloads, model imports, or overwriting files.

    Each row needs a question, options/choices and answer/answerKey. Optional
    group_id values are preserved across splits. Validation completes before
    creating the output directory; invalid rows report the input line number.
    """
    for name, value in (("source", source), ("revision", revision)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    path, output_dir = Path(path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("each row must be an object")
                question = row.get("question")
                if isinstance(question, dict):
                    question = question.get("stem")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("question must be a non-empty string or a stem object")
                converted = text_choice_record(row, source, f"{split}/{line_number}", split)
                converted["_meta"].update(revision=revision, input_line=line_number)
                if "group_id" in row:
                    group = row["group_id"]
                    if type(group) not in (str, int) or not str(group).strip():
                        raise ValueError("group_id must be a non-empty string or integer")
                    converted["_meta"]["group_id"] = f"{source}/{group}"
                materialize(converted)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            rows.append(converted)
    if not rows:
        raise ValueError(f"{path}: no records")
    manifest = {
        "name": source, "base_revisions": {},
        "trainable_sources": [source] if split == "train" else [],
        "eval_only_sources": [], "holdout_sources": [],
        "sources": {source: {"revision": revision, "input_file": path.name, "input_sha256": digest(path)}},
    }
    if split == "train":
        validate_training(rows, manifest)
    output_dir.mkdir(parents=True)
    output = output_dir / f"{split}.jsonl"
    write_jsonl(output, rows)
    manifest["files"] = {output.name: {"records": len(rows), "questions": len(rows), "sha256": digest(output)}}
    write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="UTF-8 multiple-choice JSONL export")
    parser.add_argument("--out", required=True, help="new suite directory")
    parser.add_argument("--source", required=True, help="stable source name")
    parser.add_argument("--revision", required=True, help="revision of the input export")
    parser.add_argument("--split", required=True, choices=SPLITS)
    args = parser.parse_args(argv)
    print(convert_jsonl(args.path, args.out, source=args.source, revision=args.revision, split=args.split))


if __name__ == "__main__":
    main()
