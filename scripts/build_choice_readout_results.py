#!/usr/bin/env python3
"""Validate, summarize, and plot the five-run choice-readout matrix.

The input directory must contain ``base4``, ``pointer4``, ``direct4``,
``base27``, and ``pointer27`` subdirectories.  Each subdirectory contains the
``manifest.json`` emitted by ``evaluate_choice_readout_matrix.py`` and the
``ensemble.json`` emitted by ``fit_choice_ensemble.py``.

The builder fails closed on incomplete panels, component/count drift, or
cross-run dataset-hash drift.  It never substitutes partial coverage or a
media-stripped score for JevJudge full.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL = ROOT / "results/external-zero-shot-v1.json"
DEFAULT_MODEL_CATALOG = ROOT / "results/model-family-v2.json"
DEFAULT_JSON_OUT = ROOT / "results/choice-readout-v2.json"
DEFAULT_SVG_OUT = ROOT / "docs/choice-readout-results.svg"

INK = "#213248"
MUTED = "#64748B"
RULE = "#E4E9EF"
CHOICE = "#278577"
NATIVE = "#8493A6"
TUNED = "#165F55"
EXTERNAL = "#A17BB7"
TYPESAFE = "#C08A42"

RUN_SPECS = (
    {
        "key": "base4",
        "family": "Qwen3.5-4B",
        "public_repository": "Qwen/Qwen3.5-4B",
        "weights": "frozen_base",
        "native_readout": None,
        "native_decision_mode": None,
    },
    {
        "key": "pointer4",
        "family": "Qwen3.5-4B",
        "public_repository": "SimpleJev/JevAny-Qwen3.5-4B-LoRA",
        "weights": "jevany_sft",
        "native_readout": "pointer",
        "native_decision_mode": "pointer",
    },
    {
        "key": "direct4",
        "family": "Qwen3.5-4B",
        "public_repository": "SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA",
        "weights": "jevany_sft",
        "native_readout": "direct-token",
        "native_decision_mode": "lm_token",
    },
    {
        "key": "base27",
        "family": "Qwen3.8-27B",
        "public_repository": "Qwen/Qwen3.8-27B",
        "weights": "frozen_base",
        "native_readout": None,
        "native_decision_mode": None,
    },
    {
        "key": "pointer27",
        "family": "Qwen3.8-27B",
        "public_repository": "SimpleJev/JevAny-Qwen3.8-27B-LoRA",
        "weights": "jevany_sft",
        "native_readout": "pointer",
        "native_decision_mode": "pointer",
    },
)

DATASETS = {
    "transfer_calibration": {
        "label": "Transfer-v9 development",
        "role": "selection_only",
        "records": 1_264,
        "questions": 1_264,
        "headline_n": 1_046,
    },
    "transfer_test": {
        "label": "Transfer-v9 test",
        "role": "held_out_evaluation",
        "records": 1_264,
        "questions": 1_264,
        "headline_n": 1_046,
    },
    "typed_test": {
        "label": "Typed Decisions test",
        "role": "held_out_external_evaluation",
        "records": 400,
        "questions": 2_000,
        "headline_n": 2_000,
    },
    "jevbench_public": {
        "label": "JevBench public development",
        "role": "public_diagnostic_not_used_for_selection",
        "records": 231,
        "questions": 231,
        "headline_n": 231,
    },
    "jevjudge_text": {
        "label": "JevJudge text",
        "role": "held_out_external_evaluation",
        "records": 724,
        "questions": 724,
        "headline_n": 724,
    },
}

ZERO_SHOT_CONFIGS = (
    "choice_t1",
    "native_shipped",
    "fixed_blend_0_5",
)
TUNED_CONFIGS = (
    "transfer_dev_calibrated_choice",
    "transfer_dev_tuned_blend",
)
BASE_CONFIGS = {"choice_t1", "transfer_dev_calibrated_choice"}
CHECKPOINT_CONFIGS = BASE_CONFIGS | {
    "native_shipped",
    "fixed_blend_0_5",
    "transfer_dev_tuned_blend",
}


def _read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing input artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _probability(value: Any, context: str) -> float:
    if not _finite(value) or not 0 <= value <= 1:
        raise ValueError(f"{context} must be a finite number in [0, 1]")
    return float(value)


def _positive(value: Any, context: str) -> float:
    if not _finite(value) or value <= 0:
        raise ValueError(f"{context} must be a finite positive number")
    return float(value)


def _metric(metric: Any, expected_n: int, context: str) -> dict:
    if not isinstance(metric, dict):
        raise ValueError(f"{context} must be an object")
    if metric.get("n") != expected_n:
        raise ValueError(
            f"{context} expected n={expected_n}, got {metric.get('n')!r}"
        )
    correct = metric.get("correct")
    if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= expected_n:
        raise ValueError(f"{context}.correct must be an integer in [0, {expected_n}]")
    accuracy = _probability(metric.get("accuracy"), f"{context}.accuracy")
    if not math.isclose(accuracy, correct / expected_n, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"{context}.accuracy does not equal correct / n")
    for key in (
        "cross_entropy_from_gold",
        "gold_entropy",
        "kl_from_gold",
        "brier",
        "ece",
        "mean_confidence",
    ):
        if key in metric and not _finite(metric[key]):
            raise ValueError(f"{context}.{key} must be finite")
    return copy.deepcopy(metric)


def _manifest_component_accuracy(report: dict, component: str, expected_n: int, context: str) -> float:
    try:
        clean = report["components"][component]["clean"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{context}: missing {component} clean report") from error
    if clean.get("n") != expected_n:
        raise ValueError(
            f"{context}: {component} expected clean n={expected_n}, "
            f"got {clean.get('n')!r}"
        )
    return _probability(clean.get("acc"), f"{context}.{component}.clean.acc")


def _validate_source(spec: dict, manifest: dict, context: str) -> None:
    source = manifest.get("source")
    predictor = manifest.get("predictor")
    if not isinstance(source, dict) or not isinstance(predictor, dict):
        raise ValueError(f"{context}: missing source or predictor provenance")
    is_base = spec["weights"] == "frozen_base"
    if is_base:
        if not source.get("base") or source.get("checkpoint") is not None:
            raise ValueError(f"{context}: frozen-base run must use source.base only")
        if predictor.get("adapter_applied") is not False:
            raise ValueError(f"{context}: frozen-base run unexpectedly applied an adapter")
        if manifest.get("checkpoint_artifacts") is not None:
            raise ValueError(f"{context}: frozen-base run has checkpoint artifacts")
    else:
        if not source.get("checkpoint") or source.get("base") is not None:
            raise ValueError(f"{context}: checkpoint run must use source.checkpoint only")
        if predictor.get("adapter_applied") is not True:
            raise ValueError(f"{context}: checkpoint run did not apply its adapter")
        artifacts = manifest.get("checkpoint_artifacts")
        if not isinstance(artifacts, dict) or not isinstance(artifacts.get("files"), dict):
            raise ValueError(f"{context}: checkpoint artifact hashes are missing")
    if predictor.get("native_decision_mode") != spec["native_decision_mode"]:
        raise ValueError(
            f"{context}: expected native_decision_mode={spec['native_decision_mode']!r}, "
            f"got {predictor.get('native_decision_mode')!r}"
        )


def _immutable_revision(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    return None


def _snapshot_revision(path: Any, repository: str) -> str | None:
    """Return a Hub snapshot revision only when the path names this repository."""

    if not isinstance(path, str):
        return None
    encoded_repository = repository.replace("/", "--")
    match = re.search(
        rf"(?:^|/)models--{re.escape(encoded_repository)}/snapshots/([0-9a-f]{{40}})(?:/|$)",
        path,
    )
    return match.group(1) if match else None


def _public_source(spec: dict, manifest: dict) -> dict:
    """Expose reproducible public identity separately from local load paths.

    A local release directory is not silently treated as a public Hub revision.
    In that case the evaluated files remain identified by their SHA-256 values
    and the missing public revision is explicit.
    """

    repository = spec["public_repository"]
    base_loading = manifest.get("base_loading")
    base_loading = base_loading if isinstance(base_loading, dict) else {}
    base_repository = base_loading.get("canonical_base")
    base_revision = _immutable_revision(base_loading.get("canonical_revision"))
    if not isinstance(base_repository, str) or not base_repository:
        raise ValueError(f"{spec['key']}: canonical base repository is missing")
    if base_revision is None:
        raise ValueError(f"{spec['key']}: immutable canonical base revision is missing")

    artifacts = manifest.get("checkpoint_artifacts")
    artifact_hashes = {}
    if isinstance(artifacts, dict) and isinstance(artifacts.get("files"), dict):
        for filename, metadata in artifacts["files"].items():
            if not isinstance(metadata, dict) or not re.fullmatch(
                r"[0-9a-f]{64}", str(metadata.get("sha256", ""))
            ):
                raise ValueError(
                    f"{spec['key']}: checkpoint artifact {filename!r} lacks SHA-256"
                )
            artifact_hashes[filename] = {
                "sha256": metadata["sha256"],
                "bytes": metadata.get("bytes"),
            }

    if spec["weights"] == "frozen_base":
        if repository != base_repository:
            raise ValueError(
                f"{spec['key']}: public base repository does not match canonical base"
            )
        revision = base_revision
        revision_evidence = "manifest.base_loading.canonical_revision"
        repository_evidence = "manifest.base_loading.canonical_base"
        limitation = None
    else:
        source = manifest["source"]["checkpoint"]
        revision = _snapshot_revision(source, repository)
        revision_evidence = "manifest.source.checkpoint" if revision else None
        repository_evidence = (
            "manifest.source.checkpoint"
            if revision
            else "results/model-family-v2.json#released_models"
        )
        limitation = None if revision else (
            "The evaluated checkpoint came from a local release. Its exact files "
            "are pinned below by SHA-256, but the run manifest and release metadata "
            "do not prove an immutable public-repository revision for those bytes."
        )

    return {
        "repository": repository,
        "url": f"https://huggingface.co/{repository}",
        "revision": revision,
        "revision_status": "verified" if revision else "not_verified",
        "repository_evidence": repository_evidence,
        "revision_evidence": revision_evidence,
        "evaluated_artifact_sha256": artifact_hashes,
        "base_model": {
            "repository": base_repository,
            "revision": base_revision,
        },
        "limitation": limitation,
    }


def _load_run(run_root: Path, spec: dict) -> dict:
    directory = run_root / spec["key"]
    manifest_path = directory / "manifest.json"
    ensemble_path = directory / "ensemble.json"
    manifest = _read_object(manifest_path)
    ensemble = _read_object(ensemble_path)
    context = spec["key"]

    model = manifest.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{context}: manifest model must be a non-empty string")
    if ensemble.get("model") != model:
        raise ValueError(f"{context}: manifest and ensemble model names differ")
    _validate_source(spec, manifest, context)

    expected_components = {"choice"}
    if spec["native_readout"] is not None:
        expected_components.add("native")
    reports = manifest.get("reports")
    datasets = ensemble.get("datasets")
    if not isinstance(reports, dict) or not isinstance(datasets, dict):
        raise ValueError(f"{context}: missing report or ensemble datasets")
    if set(DATASETS) - set(reports) or set(DATASETS) - set(datasets):
        raise ValueError(f"{context}: one or more required datasets are missing")

    expected_configs = BASE_CONFIGS if spec["native_readout"] is None else CHECKPOINT_CONFIGS
    summarized = {}
    for dataset, expected in DATASETS.items():
        report = reports[dataset]
        if report.get("records") != expected["records"]:
            raise ValueError(
                f"{context}/{dataset}: expected {expected['records']} records, "
                f"got {report.get('records')!r}"
            )
        if report.get("questions") != expected["questions"]:
            raise ValueError(
                f"{context}/{dataset}: expected {expected['questions']} questions, "
                f"got {report.get('questions')!r}"
            )
        components = set(report.get("components", {}))
        executed = set(report.get("components_executed", components))
        if components != expected_components or executed != expected_components:
            raise ValueError(
                f"{context}/{dataset}: expected components {sorted(expected_components)}, "
                f"got reports={sorted(components)}, executed={sorted(executed)}"
            )
        component_accuracy = {
            component: _manifest_component_accuracy(
                report, component, expected["headline_n"], f"{context}/{dataset}"
            )
            for component in sorted(expected_components)
        }

        configurations = datasets[dataset]
        if not isinstance(configurations, dict) or set(configurations) != expected_configs:
            raise ValueError(
                f"{context}/{dataset}: expected ensemble configs "
                f"{sorted(expected_configs)}, got "
                f"{sorted(configurations) if isinstance(configurations, dict) else configurations!r}"
            )
        checked = {
            name: _metric(
                values,
                expected["headline_n"],
                f"{context}/{dataset}/{name}",
            )
            for name, values in configurations.items()
        }
        expected_excluded = expected["questions"] - expected["headline_n"]
        for name, values in checked.items():
            if values.get("excluded_non_headline_rows") != expected_excluded:
                raise ValueError(
                    f"{context}/{dataset}/{name}: expected "
                    f"excluded_non_headline_rows={expected_excluded}"
                )
        if not math.isclose(
            checked["choice_t1"]["accuracy"], component_accuracy["choice"],
            rel_tol=0, abs_tol=1e-12,
        ):
            raise ValueError(f"{context}/{dataset}: choice component and ensemble are not aligned")
        if checked["choice_t1"]["correct"] != checked["transfer_dev_calibrated_choice"]["correct"]:
            raise ValueError(
                f"{context}/{dataset}: temperature-only calibration changed hard decisions"
            )
        if "native" in expected_components and not math.isclose(
            checked["native_shipped"]["accuracy"], component_accuracy["native"],
            rel_tol=0, abs_tol=1e-12,
        ):
            raise ValueError(f"{context}/{dataset}: native component and ensemble are not aligned")

        summarized[dataset] = {
            "zero_shot": {
                key: checked[key] for key in ZERO_SHOT_CONFIGS if key in checked
            },
            "transfer_dev_tuned": {
                key: checked[key] for key in TUNED_CONFIGS if key in checked
            },
        }

    protocol = ensemble.get("protocol")
    selected = ensemble.get("selected")
    if not isinstance(protocol, dict) or protocol.get("selection_rows") != 1_046:
        raise ValueError(f"{context}: ensemble must select on 1,046 Transfer-dev rows")
    if not isinstance(selected, dict):
        raise ValueError(f"{context}: missing selected ensemble settings")
    choice_temperature = _positive(
        selected.get("choice_temperature"), f"{context}.choice_temperature"
    )
    blend_temperature = _positive(
        selected.get("blend_temperature"), f"{context}.blend_temperature"
    )
    native_weight = _probability(
        selected.get("native_weight"), f"{context}.native_weight"
    )
    if spec["native_readout"] is None and native_weight != 0:
        raise ValueError(f"{context}: frozen base selected a native weight")

    dataset_hashes = manifest.get("datasets")
    runtime = manifest.get("runtime")
    if not isinstance(dataset_hashes, dict) or not dataset_hashes:
        raise ValueError(f"{context}: dataset hashes are missing")
    if not isinstance(runtime, dict) or not runtime.get("code_revision"):
        raise ValueError(f"{context}: runtime/code revision is missing")

    return {
        "key": spec["key"],
        "model": model,
        "family": spec["family"],
        "weights": spec["weights"],
        "native_readout": spec["native_readout"],
        "artifacts": {
            "manifest": {
                "path": _display_path(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "ensemble": {
                "path": _display_path(ensemble_path),
                "sha256": _sha256(ensemble_path),
            },
        },
        "source": copy.deepcopy(manifest.get("source")),
        "public_source": _public_source(spec, manifest),
        "checkpoint_artifacts": copy.deepcopy(manifest.get("checkpoint_artifacts")),
        "base_loading": copy.deepcopy(manifest.get("base_loading")),
        "predictor": copy.deepcopy(manifest.get("predictor")),
        "runtime": copy.deepcopy(runtime),
        "dataset_hashes": copy.deepcopy(dataset_hashes),
        "evaluation_protocol": copy.deepcopy(manifest.get("protocol")),
        "ensemble_protocol": copy.deepcopy(protocol),
        "selected": {
            **copy.deepcopy(selected),
            "choice_temperature": choice_temperature,
            "blend_temperature": blend_temperature,
            "native_weight": native_weight,
        },
        "datasets": summarized,
    }


def _find_model(rows: Any, name: str, context: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError(f"{context}: models must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("model") == name]
    if len(matches) != 1:
        raise ValueError(f"{context}: expected exactly one {name!r} row")
    _probability(matches[0].get("accuracy"), f"{context}/{name}.accuracy")
    return copy.deepcopy(matches[0])


def _load_external(path: Path) -> dict:
    external = _read_object(path)
    typed = external.get("typed_decisions")
    text = external.get("jevjudge_text")
    if not isinstance(typed, dict) or not isinstance(text, dict):
        raise ValueError("external artifact is missing Typed Decisions or JevJudge text")

    typed_rows = typed.get("models")
    if not isinstance(typed_rows, list):
        raise ValueError("external Typed Decisions models must be a list")
    published = [
        row for row in typed_rows
        if isinstance(row, dict)
        and row.get("kind") == "published_only"
        and _finite(row.get("accuracy"))
    ]
    if not published:
        raise ValueError("external artifact has no published Typed Decisions baseline")
    typed_top = copy.deepcopy(max(published, key=lambda row: (row["accuracy"], row["model"])))
    _probability(typed_top.get("accuracy"), "Typed published top accuracy")
    typed_typesafe = _find_model(
        typed_rows, "Jev 1.13 (OpenRouter)", "external/typed_decisions"
    )

    text_rows = text.get("models")
    text_typesafe = _find_model(
        text_rows, "Jev 1.13 (OpenRouter)", "external/jevjudge_text"
    )
    kev_rows = [
        row for row in text_rows
        if isinstance(row, dict)
        and row.get("kind") == "open_kev"
        and _finite(row.get("accuracy"))
    ] if isinstance(text_rows, list) else []
    if not kev_rows:
        raise ValueError("external artifact has no scored JevJudge-text Kev baseline")
    text_kev = copy.deepcopy(max(kev_rows, key=lambda row: (row["accuracy"], row["model"])))
    for row in (text_typesafe, text_kev):
        if (row.get("answered"), row.get("requested")) != (724, 724):
            raise ValueError(
                f"external/jevjudge_text/{row['model']}: expected 724/724 coverage"
            )

    return {
        "artifact": {
            "path": _display_path(path),
            "sha256": _sha256(path),
            "artifact_version": external.get("artifact_version"),
            "provenance": copy.deepcopy(external.get("provenance")),
        },
        "typed_decisions": [typed_top, typed_typesafe],
        "jevjudge_text": [text_typesafe, text_kev],
    }


def _load_repository_catalog(path: Path) -> dict:
    catalog = _read_object(path)
    released = catalog.get("released_models")
    repositories = {
        row.get("repository")
        for row in released
        if isinstance(row, dict) and isinstance(row.get("repository"), str)
    } if isinstance(released, list) else set()
    required = {
        spec["public_repository"]
        for spec in RUN_SPECS
        if spec["weights"] != "frozen_base"
    }
    missing = required - repositories
    if missing:
        raise ValueError(
            "model release catalog is missing canonical repositories: "
            + ", ".join(sorted(missing))
        )
    return {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "release": catalog.get("release"),
    }


def build_results(
    run_root: Path,
    external_path: Path = DEFAULT_EXTERNAL,
    model_catalog_path: Path = DEFAULT_MODEL_CATALOG,
) -> dict:
    """Build a validated, JSON-serializable result artifact without writing it."""

    run_root = Path(run_root)
    repository_catalog = _load_repository_catalog(Path(model_catalog_path))
    runs = [_load_run(run_root, spec) for spec in RUN_SPECS]
    if len({run["model"] for run in runs}) != len(runs):
        raise ValueError("the five run manifests must have unique model labels")

    reference_hashes = runs[0]["dataset_hashes"]
    for run in runs[1:]:
        if run["dataset_hashes"] != reference_hashes:
            raise ValueError(
                f"{run['key']}: dataset hashes differ from {runs[0]['key']}"
            )
    revisions = {run["runtime"].get("code_revision") for run in runs}
    if len(revisions) != 1:
        raise ValueError("all five runs must use one code revision")

    external = _load_external(Path(external_path))
    result_datasets = {}
    for dataset, expected in DATASETS.items():
        result_datasets[dataset] = {
            **copy.deepcopy(expected),
            "runs": [
                {
                    "run": run["key"],
                    "model": run["model"],
                    "family": run["family"],
                    "weights": run["weights"],
                    "native_readout": run["native_readout"],
                    **copy.deepcopy(run["datasets"][dataset]),
                }
                for run in runs
            ],
        }
    result_datasets["typed_test"]["external_baselines"] = copy.deepcopy(
        external["typed_decisions"]
    )
    result_datasets["jevjudge_text"]["external_baselines"] = copy.deepcopy(
        external["jevjudge_text"]
    )
    result_datasets["jevjudge_full"] = {
        "label": "JevJudge full",
        "records": 3_220,
        "status": "unsupported",
        "accuracy": None,
        "reason": (
            "Training-free choice-token readout is text-only; image/video records "
            "are not stripped or relabeled as a full-suite result."
        ),
        "native_checkpoint_results": "results/external-zero-shot-v1.json#jevjudge_full",
    }

    public_runs = []
    for run in runs:
        public_runs.append({key: copy.deepcopy(value) for key, value in run.items() if key != "datasets"})

    return {
        "schema_version": 2,
        "artifact": "choice-readout-v2",
        "title": "Training-free choice-token readout evaluation",
        "generated_by": "scripts/build_choice_readout_results.py",
        "method": {
            "readout": (
                "Constrain the next token to exact one-character option IDs and "
                "renormalize their probability mass; this can change argmax decisions."
            ),
            "training": "No additional training for the choice-token readout.",
            "option_ids": "A-Z followed by a-z; at most 52 options.",
            "calibration": (
                "Scalar temperature changes probabilities but not argmax accuracy."
            ),
            "ensemble": (
                "Native + choice configurations use log-linear/geometric pooling."
            ),
        },
        "protocol_groups": {
            "zero_shot": {
                "meaning": (
                    "No Typed, JevJudge, JevBench, or Transfer-test labels tune the "
                    "readout configuration."
                ),
                "configurations": list(ZERO_SHOT_CONFIGS),
            },
            "transfer_dev_tuned": {
                "meaning": (
                    "Weight and additional temperature are selected only on Transfer-v9 "
                    "development, then frozen for every displayed evaluation panel."
                ),
                "configurations": list(TUNED_CONFIGS),
                "accuracy_note": (
                    "Temperature-only calibrated choice has the same hard accuracy as "
                    "choice_t1; only a changed blend weight can change its argmax."
                ),
            },
        },
        "expected_counts": copy.deepcopy(DATASETS),
        "dataset_hashes": copy.deepcopy(reference_hashes),
        "code_revision": next(iter(revisions)),
        "source_runs": public_runs,
        "external_source": external["artifact"],
        "repository_catalog": repository_catalog,
        "datasets": result_datasets,
        "legacy_results": {
            "status": "retained unchanged",
            "artifact": "results/letter-readout-v1.json",
        },
    }


def write_results(path: Path, artifact: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        artifact, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    path.write_text(payload, encoding="utf-8")


def _run_result(artifact: dict, dataset: str, run_key: str) -> dict:
    matches = [
        row for row in artifact["datasets"][dataset]["runs"]
        if row["run"] == run_key
    ]
    if len(matches) != 1:
        raise ValueError(f"{dataset}: expected exactly one {run_key} result")
    return matches[0]


def _plot_rows(artifact: dict, dataset: str) -> list[dict]:
    def accuracy(run: str, group: str, config: str) -> float:
        return _run_result(artifact, dataset, run)[group][config]["accuracy"]

    rows = [
        {
            "label": "Frozen Qwen3.5 4B · Choice",
            "group": "zero_shot",
            "kind": "choice",
            "accuracy": accuracy("base4", "zero_shot", "choice_t1"),
        },
        {
            "label": "JevAny 4B Pointer · Native",
            "group": "zero_shot",
            "kind": "native",
            "accuracy": accuracy("pointer4", "zero_shot", "native_shipped"),
        },
        {
            "label": "JevAny 4B Pointer · Choice",
            "group": "zero_shot",
            "kind": "choice",
            "accuracy": accuracy("pointer4", "zero_shot", "choice_t1"),
        },
        {
            "label": "JevAny 4B Direct-Token · Native",
            "group": "zero_shot",
            "kind": "native",
            "accuracy": accuracy("direct4", "zero_shot", "native_shipped"),
        },
        {
            "label": "JevAny 4B Direct-Token · Choice",
            "group": "zero_shot",
            "kind": "choice",
            "accuracy": accuracy("direct4", "zero_shot", "choice_t1"),
        },
        {
            "label": "Frozen Qwen3.8 27B · Choice",
            "group": "zero_shot",
            "kind": "choice",
            "accuracy": accuracy("base27", "zero_shot", "choice_t1"),
        },
        {
            "label": "JevAny 27B Pointer · Native",
            "group": "zero_shot",
            "kind": "native",
            "accuracy": accuracy("pointer27", "zero_shot", "native_shipped"),
        },
        {
            "label": "JevAny 27B Pointer · Choice",
            "group": "zero_shot",
            "kind": "choice",
            "accuracy": accuracy("pointer27", "zero_shot", "choice_t1"),
        },
        {
            "label": "JevAny 4B Pointer · Tuned blend",
            "group": "transfer_dev_tuned",
            "kind": "tuned",
            "accuracy": accuracy(
                "pointer4", "transfer_dev_tuned", "transfer_dev_tuned_blend"
            ),
        },
        {
            "label": "JevAny 4B Direct-Token · Tuned blend",
            "group": "transfer_dev_tuned",
            "kind": "tuned",
            "accuracy": accuracy(
                "direct4", "transfer_dev_tuned", "transfer_dev_tuned_blend"
            ),
        },
        {
            "label": "JevAny 27B Pointer · Tuned blend",
            "group": "transfer_dev_tuned",
            "kind": "tuned",
            "accuracy": accuracy(
                "pointer27", "transfer_dev_tuned", "transfer_dev_tuned_blend"
            ),
        },
    ]
    baselines = artifact["datasets"][dataset]["external_baselines"]
    for baseline in baselines:
        is_typesafe = baseline["model"] == "Jev 1.13 (OpenRouter)"
        label = (
            "TypeSafe Jev 1.13"
            if is_typesafe
            else f"{baseline['model']} · "
            + ("published" if dataset == "typed_test" else "open")
        )
        rows.append({
            "label": label,
            "group": "external",
            "kind": "typesafe" if is_typesafe else "external",
            "accuracy": baseline["accuracy"],
            "published": dataset == "typed_test" and not is_typesafe,
        })
    return rows


def render_svg(artifact: dict, output: Path) -> None:
    """Render the compact two-panel README figure from a validated artifact."""

    try:
        import matplotlib
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render the SVG") from error

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "jevany-choice-readout-v2",
        "text.color": INK,
        "hatch.linewidth": 0.7,
    })
    panels = (
        ("typed_test", "Typed Decisions", "2,000 decisions · accuracy (%)"),
        ("jevjudge_text", "JevJudge text", "724 records · accuracy (%)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 9.3), dpi=110, facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.988, bottom=0.12, top=0.78, wspace=0.30)
    fig.text(0.035, 0.955, "Training-free choice readout", fontsize=24, weight="bold")
    fig.text(
        0.035,
        0.914,
        "Zero-shot readouts and Transfer-dev-tuned blends · no Typed or JevJudge labels used for tuning",
        fontsize=12.2,
        color=MUTED,
    )
    fig.legend(
        handles=[
            Patch(facecolor=CHOICE, label="Choice T=1 · zero-shot"),
            Patch(facecolor=NATIVE, label="Native shipped · zero-shot"),
            Patch(facecolor=TUNED, hatch="///", label="Transfer-dev-tuned blend"),
            Patch(facecolor=TYPESAFE, label="TypeSafe Jev"),
            Patch(facecolor=EXTERNAL, hatch="////", label="Published / open baseline"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.988, 0.982),
        ncol=3,
        frameon=False,
        fontsize=10.1,
        handlelength=1.2,
        columnspacing=1.25,
    )

    description_parts = []
    colors = {
        "choice": CHOICE,
        "native": NATIVE,
        "tuned": TUNED,
        "external": EXTERNAL,
        "typesafe": TYPESAFE,
    }
    positions = list(range(8)) + list(range(9, 12)) + list(range(13, 15))
    for ax, (dataset, title, scope) in zip(axes, panels, strict=True):
        rows = _plot_rows(artifact, dataset)
        if len(rows) != len(positions):
            raise ValueError(f"{dataset}: plot requires exactly 15 rows")
        values = [row["accuracy"] * 100 for row in rows]
        xmax = min(100, max(70, int(math.ceil((max(values) + 4) / 10) * 10)))
        ax.axhspan(-0.7, 7.65, color="#F7FAFC", zorder=0)
        ax.axhspan(8.35, 11.65, color="#EDF7F4", zorder=0)
        ax.axhspan(12.35, 14.65, color="#FAF7FC", zorder=0)
        for y, row, value in zip(positions, rows, values, strict=True):
            bars = ax.barh(
                y,
                value,
                height=0.64,
                color=colors[row["kind"]],
                edgecolor="#805A2B" if row.get("published") else "none",
                linewidth=0.7 if row.get("published") else 0,
                zorder=3,
            )
            if row["kind"] == "tuned":
                bars[0].set_hatch("///")
            elif row.get("published"):
                bars[0].set_hatch("////")
            ax.text(
                min(value + xmax * 0.015, xmax * 0.985),
                y,
                f"{value:.1f}",
                ha="right" if value > xmax * 0.91 else "left",
                va="center",
                fontsize=9.2,
                color=INK,
                weight="bold" if row["group"] != "external" else "normal",
            )
        ax.text(0, -0.78, "ZERO-SHOT", fontsize=9.0, color=MUTED, weight="bold")
        ax.text(0, 8.22, "TRANSFER-DEV-TUNED", fontsize=9.0, color=TUNED, weight="bold")
        ax.text(0, 12.22, "EXTERNAL REFERENCE", fontsize=9.0, color=MUTED, weight="bold")
        ax.set_title(title, loc="left", fontsize=17, color=INK, weight="bold", pad=28)
        ax.text(0, 1.015, scope, transform=ax.transAxes, fontsize=10.2, color=MUTED)
        ax.set_xlim(0, xmax)
        ticks = list(range(0, xmax + 1, 20))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(value) for value in ticks], fontsize=9.2, color=MUTED)
        ax.set_yticks(positions)
        ax.set_yticklabels([row["label"] for row in rows], fontsize=9.0, color=INK)
        for tick, row in zip(ax.get_yticklabels(), rows, strict=True):
            if row["group"] != "external":
                tick.set_weight("bold" if row["kind"] == "tuned" else "normal")
        ax.set_ylim(15.0, -1.1)
        ax.tick_params(axis="x", length=0, pad=7)
        ax.tick_params(axis="y", length=0, pad=7)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=RULE, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axvline(0, color=RULE, linewidth=1)
        description_parts.append(
            f"{title}: " + ", ".join(
                f"{row['label']} {row['accuracy'] * 100:.2f}%" for row in rows
            )
        )

    fig.text(
        0.035,
        0.044,
        "JevJudge full is unsupported for this text-only readout. Choice temperature calibration is omitted here because it cannot change accuracy.",
        fontsize=10.1,
        color=MUTED,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Date": None,
        "Title": "Training-free choice readout accuracy",
        "Description": " ".join(description_parts),
    }
    fig.savefig(output, metadata=metadata)
    plt.close(fig)
    output.write_text(
        "\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--model-catalog", type=Path, default=DEFAULT_MODEL_CATALOG)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--svg-out", type=Path, default=DEFAULT_SVG_OUT)
    args = parser.parse_args(argv)

    artifact = build_results(args.run_root, args.external, args.model_catalog)
    write_results(args.json_out, artifact)
    render_svg(artifact, args.svg_out)
    print(f"Wrote {_display_path(args.json_out)} and {_display_path(args.svg_out)}")


if __name__ == "__main__":
    main()
