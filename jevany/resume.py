"""State for continuing a training schedule from an optimizer-step boundary."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .suite import digest, write_json

VERSION = 1
MUTABLE_SETTINGS = {
    "config", "dry_run", "out", "resume", "base_load_path", "checkpoint_every_steps",
    "wandb_project", "wandb_name", "wandb_group", "wandb_entity", "wandb_mode",
}


def read_resume_config(directory: str | Path) -> dict:
    """Read a local training checkpoint's configuration without loading weights."""
    directory = Path(directory)
    filename = directory / "trainer_state.json"
    if not filename.is_file():
        raise ValueError(f"{directory}: no complete trainer state; use --init-from for a weights-only warm start")
    try:
        header = json.loads(filename.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{filename}: invalid trainer state metadata") from error
    if header.get("version") != VERSION or not isinstance(header.get("args"), dict):
        raise ValueError(f"{filename}: unsupported trainer state format")
    return header


def validate_resume_settings(args: dict, saved: dict) -> None:
    """A resume continues the original schedule; recipe changes need a warm start."""
    changed = sorted(key for key in saved if key not in MUTABLE_SETTINGS and args.get(key) != saved[key])
    if changed:
        raise ValueError(f"--resume must keep the original training settings: {', '.join(changed)}; "
                         "use --init-from to start a different experiment")


def dataset_fingerprint(records: list[dict]) -> str:
    """Identify admitted records, labels, ordering and the local media they use."""
    fingerprint = hashlib.sha256()
    media = set()
    for record in records:
        # Question and option insertion order affects encoding and seeded
        # augmentation, so it must be part of the resume identity.
        fingerprint.update(json.dumps(record, ensure_ascii=False, allow_nan=False).encode())
        fingerprint.update(b"\n")
        for item in record.get("media", []):
            media.add(item["uri"])
    for uri in sorted(media):
        fingerprint.update(uri.encode())
        if "://" not in uri:
            fingerprint.update(digest(uri).encode())
    return fingerprint.hexdigest()


def capture_rng(shuffle: random.Random, device: str) -> dict:
    """Capture the random generators used by one rank after its optimizer step."""
    numpy = np.random.get_state()
    result = {
        "shuffle": shuffle.getstate(), "python": random.getstate(),
        "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
        "torch": torch.get_rng_state(),
    }
    if device == "cuda":
        result["device"] = torch.cuda.get_rng_state()
    elif device == "mps":
        result["device"] = torch.mps.get_rng_state()
    return result


def restore_rng(state: dict, shuffle: random.Random, device: str) -> None:
    """Restore generators after model/optimizer construction has consumed randomness."""
    shuffle.setstate(state["shuffle"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"])
    if device == "cuda":
        torch.cuda.set_rng_state(state["device"])
    elif device == "mps":
        torch.mps.set_rng_state(state["device"])


def save_trainer_state(directory: str | Path, state: dict) -> None:
    """Write state last, so an interrupted model save cannot appear resumable."""
    directory = Path(directory)
    state["version"] = VERSION
    state["checkpoint_sha256"] = {
        name: digest(directory / name) for name in ("adapter_model.safetensors", "head.pt")
    }
    temporary = directory / ".trainer_state.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, directory / "trainer_state.pt")
    header = {key: state[key] for key in ("version", "args", "optimizer_step", "world_size", "dataset_sha256")}
    temporary = directory / ".trainer_state.json.tmp"
    write_json(temporary, header)
    os.replace(temporary, directory / "trainer_state.json")


def load_trainer_state(directory: str | Path, args: dict, dataset_sha256: str,
                       world_size: int, device: str) -> dict:
    """Load tensor/basic-type state and validate its data, schedule and weight files."""
    directory = Path(directory)
    header = read_resume_config(directory)
    validate_resume_settings(args, header["args"])
    state_file = directory / "trainer_state.pt"
    if not state_file.is_file():
        raise ValueError(f"{directory}: missing trainer_state.pt")
    state = torch.load(state_file, map_location="cpu", weights_only=True)
    if (state.get("version") != VERSION or state.get("args") != header["args"]
            or state.get("optimizer_step") != header["optimizer_step"]):
        raise ValueError(f"{directory}: trainer state and metadata disagree")
    if state["dataset_sha256"] != dataset_sha256:
        raise ValueError("--resume training data changed since this checkpoint "
                         "(record content/order, media paths or local media contents)")
    if state["world_size"] != world_size or state["device"] != device:
        raise ValueError("--resume requires the original world size and device type")
    for name, expected in state["checkpoint_sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"{directory}: {name} differs from the saved trainer state")
    return state
