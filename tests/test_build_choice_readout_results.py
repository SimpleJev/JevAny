import copy
import json
from pathlib import Path

import pytest

from scripts.build_choice_readout_results import (
    DATASETS,
    RUN_SPECS,
    build_results,
    render_svg,
    write_results,
)


def _metric(n, correct):
    accuracy = correct / n
    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "cross_entropy_from_gold": 0.5,
        "gold_entropy": 0.1,
        "kl_from_gold": 0.4,
        "brier": 0.2,
        "ece": 0.03,
        "mean_confidence": 0.7,
        "cohort": "clean and knowable",
        "excluded_non_headline_rows": 0,
        "ece_bins": 15,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _external_fixture(path):
    value = {
        "artifact_version": 5,
        "provenance": {"fixture": {"sha256": "e" * 64}},
        "typed_decisions": {
            "models": [
                {
                    "model": "Published runner-up",
                    "kind": "published_only",
                    "accuracy": 0.74,
                },
                {
                    "model": "Published top",
                    "kind": "published_only",
                    "accuracy": 0.77,
                },
                {
                    "model": "Jev 1.13 (OpenRouter)",
                    "kind": "external_api",
                    "accuracy": 0.73,
                },
            ]
        },
        "jevjudge_text": {
            "models": [
                {
                    "model": "Jev 1.13 (OpenRouter)",
                    "kind": "external_api",
                    "accuracy": 0.65,
                    "answered": 724,
                    "requested": 724,
                },
                {
                    "model": "Kev-4B",
                    "kind": "open_kev",
                    "accuracy": 0.54,
                    "answered": 724,
                    "requested": 724,
                },
                {
                    "model": "Kev-27B",
                    "kind": "open_kev",
                    "accuracy": 0.64,
                    "answered": 724,
                    "requested": 724,
                },
            ]
        },
    }
    _write_json(path, value)


def _run_fixture(root, spec, ordinal):
    is_base = spec["weights"] == "frozen_base"
    canonical_base = f"Qwen/{spec['family']}"
    snapshot_revision = f"{ordinal + 1}" * 40
    encoded_repository = spec["public_repository"].replace("/", "--")
    checkpoint = (
        f"/fixture/models--{encoded_repository}/snapshots/{snapshot_revision}"
        if not is_base and spec["key"] != "direct4"
        else f"/fixture/local-release/{spec['key']}"
    )
    components = ["choice"] if is_base else ["choice", "native"]
    model = f"fixture-{spec['key']}"
    reports = {}
    ensemble_datasets = {}
    for dataset_index, (dataset, expected) in enumerate(DATASETS.items()):
        n = expected["headline_n"]
        choice_correct = n // 2 + ordinal + dataset_index
        native_correct = choice_correct + 1
        choice_accuracy = choice_correct / n
        native_accuracy = native_correct / n
        reports[dataset] = {
            "dataset": dataset,
            "records": expected["records"],
            "questions": expected["questions"],
            "components_executed": components,
            "components": {
                "choice": {"clean": {"n": n, "acc": choice_accuracy}},
            },
        }
        if not is_base:
            reports[dataset]["components"]["native"] = {
                "clean": {"n": n, "acc": native_accuracy}
            }
        choice = _metric(n, choice_correct)
        choice["excluded_non_headline_rows"] = expected["questions"] - n
        calibrated = copy.deepcopy(choice)
        calibrated["cross_entropy_from_gold"] = 0.45
        configurations = {
            "choice_t1": choice,
            "transfer_dev_calibrated_choice": calibrated,
        }
        if not is_base:
            native = _metric(n, native_correct)
            fixed = _metric(n, min(n, native_correct + 1))
            tuned = _metric(n, min(n, native_correct + 2))
            for value in (native, fixed, tuned):
                value["excluded_non_headline_rows"] = expected["questions"] - n
            configurations.update({
                "native_shipped": native,
                "fixed_blend_0_5": fixed,
                "transfer_dev_tuned_blend": tuned,
            })
        ensemble_datasets[dataset] = configurations

    hashes = {
        "jevbench_development_sha256": "1" * 64,
        "jevbench_manifest_sha256": "2" * 64,
        "transfer_manifest_sha256": "3" * 64,
        "transfer_development_sha256": "4" * 64,
        "transfer_test_sha256": "5" * 64,
        "typed_revision": "6" * 40,
        "typed_test_sha256": "7" * 64,
        "jevjudge_test_sha256": "8" * 64,
        "jevjudge_manifest_sha256": "9" * 64,
    }
    manifest = {
        "model": model,
        "source": {
            "base": spec["public_repository"] if is_base else None,
            "checkpoint": None if is_base else checkpoint,
        },
        "checkpoint_artifacts": None if is_base else {
            "requested": f"fixture/{spec['key']}",
            "resolved": f"/fixture/{spec['key']}",
            "files": {
                "adapter_model.safetensors": {
                    "sha256": f"{ordinal + 1}" * 64,
                    "bytes": 123,
                }
            },
        },
        "base_loading": {
            "requested": spec["family"],
            "resolved": "/fixture/base",
            "canonical_base": canonical_base,
            "canonical_revision": "a" * 40,
        },
        "predictor": {
            "method": "training-free exact choice-token projection",
            "adapter_applied": not is_base,
            "native_decision_mode": spec["native_decision_mode"],
            "canonical_base": canonical_base,
            "canonical_revision": "a" * 40,
        },
        "runtime": {
            "python": "3.12.0",
            "torch": "2.fixture",
            "code_revision": "b" * 40,
        },
        "protocol": {"temperature": 1.0, "kernel_policy": "fixture exact"},
        "datasets": hashes,
        "reports": reports,
    }
    ensemble = {
        "model": model,
        "protocol": {
            "selection_rows": 1046,
            "selection_split": "Transfer-v9 development",
        },
        "selected": {
            "choice_temperature": 1.2,
            "native_weight": 0.0 if is_base else 0.55,
            "blend_temperature": 1.1,
            "selection_objective": "fixture",
        },
        "datasets": ensemble_datasets,
    }
    directory = root / spec["key"]
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "ensemble.json", ensemble)


@pytest.fixture
def matrix(tmp_path):
    root = tmp_path / "runs"
    for ordinal, spec in enumerate(RUN_SPECS):
        _run_fixture(root, spec, ordinal)
    external = tmp_path / "external.json"
    _external_fixture(external)
    return root, external


def test_build_results_preserves_all_five_run_types_and_protocol_groups(matrix):
    root, external = matrix

    artifact = build_results(root, external)

    assert artifact["artifact"] == "choice-readout-v2"
    assert [run["key"] for run in artifact["source_runs"]] == [
        "base4", "pointer4", "direct4", "base27", "pointer27"
    ]
    assert {run["weights"] for run in artifact["source_runs"]} == {
        "frozen_base", "jevany_sft"
    }
    assert {run["native_readout"] for run in artifact["source_runs"]} == {
        None, "pointer", "direct-token"
    }
    typed = artifact["datasets"]["typed_test"]
    assert typed["questions"] == typed["headline_n"] == 2000
    assert typed["external_baselines"][0]["model"] == "Published top"
    assert artifact["datasets"]["jevjudge_text"]["external_baselines"][1]["model"] == "Kev-27B"
    assert artifact["datasets"]["jevjudge_full"] == {
        "label": "JevJudge full",
        "records": 3220,
        "status": "unsupported",
        "accuracy": None,
        "reason": (
            "Training-free choice-token readout is text-only; image/video records "
            "are not stripped or relabeled as a full-suite result."
        ),
        "native_checkpoint_results": "results/external-zero-shot-v1.json#jevjudge_full",
    }
    base = next(row for row in typed["runs"] if row["run"] == "base4")
    direct = next(row for row in typed["runs"] if row["run"] == "direct4")
    assert set(base["zero_shot"]) == {"choice_t1"}
    assert set(base["transfer_dev_tuned"]) == {"transfer_dev_calibrated_choice"}
    assert set(direct["zero_shot"]) == {
        "choice_t1", "native_shipped", "fixed_blend_0_5"
    }
    assert set(direct["transfer_dev_tuned"]) == {
        "transfer_dev_calibrated_choice", "transfer_dev_tuned_blend"
    }
    assert len(artifact["source_runs"][0]["artifacts"]["manifest"]["sha256"]) == 64
    assert artifact["source_runs"][0]["runtime"]["torch"] == "2.fixture"

    sources = {run["key"]: run["public_source"] for run in artifact["source_runs"]}
    assert sources["base4"] == {
        "repository": "Qwen/Qwen3.5-4B",
        "url": "https://huggingface.co/Qwen/Qwen3.5-4B",
        "revision": "a" * 40,
        "revision_status": "verified",
        "repository_evidence": "manifest.base_loading.canonical_base",
        "revision_evidence": "manifest.base_loading.canonical_revision",
        "evaluated_artifact_sha256": {},
        "base_model": {
            "repository": "Qwen/Qwen3.5-4B",
            "revision": "a" * 40,
        },
        "limitation": None,
    }
    assert sources["pointer4"]["revision"] == "2" * 40
    assert sources["pointer4"]["revision_status"] == "verified"
    assert sources["pointer27"]["revision"] == "5" * 40
    assert sources["direct4"]["repository"] == (
        "SimpleJev/JevAny-Qwen3.5-4B-Direct-Token-LoRA"
    )
    assert sources["direct4"]["revision"] is None
    assert sources["direct4"]["revision_status"] == "not_verified"
    assert sources["direct4"]["revision_evidence"] is None
    assert "do not prove an immutable public-repository revision" in (
        sources["direct4"]["limitation"]
    )
    assert sources["direct4"]["evaluated_artifact_sha256"] == {
        "adapter_model.safetensors": {
            "sha256": "3" * 64,
            "bytes": 123,
        }
    }


def test_build_results_rejects_component_count_drift(matrix):
    root, external = matrix
    path = root / "pointer4" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["reports"]["typed_test"]["components"]["native"]["clean"]["n"] = 1999
    _write_json(path, manifest)

    with pytest.raises(ValueError, match="native expected clean n=2000"):
        build_results(root, external)


def test_build_results_rejects_native_component_metric_misalignment(matrix):
    root, external = matrix
    path = root / "direct4" / "ensemble.json"
    ensemble = json.loads(path.read_text())
    native = ensemble["datasets"]["jevjudge_text"]["native_shipped"]
    native["correct"] += 1
    native["accuracy"] = native["correct"] / native["n"]
    _write_json(path, ensemble)

    with pytest.raises(ValueError, match="native component and ensemble are not aligned"):
        build_results(root, external)


def test_build_results_rejects_cross_run_dataset_hash_drift(matrix):
    root, external = matrix
    path = root / "direct4" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["datasets"]["typed_test_sha256"] = "0" * 64
    _write_json(path, manifest)

    with pytest.raises(ValueError, match="dataset hashes differ"):
        build_results(root, external)


def test_build_results_rejects_release_catalog_without_canonical_repositories(
    matrix, tmp_path
):
    root, external = matrix
    catalog = tmp_path / "model-catalog.json"
    _write_json(catalog, {"release": "fixture", "released_models": []})

    with pytest.raises(ValueError, match="missing canonical repositories"):
        build_results(root, external, catalog)


def test_synthetic_artifact_and_svg_are_written_only_to_requested_paths(matrix, tmp_path):
    pytest.importorskip("matplotlib")
    root, external = matrix
    artifact = build_results(root, external)
    json_out = tmp_path / "out" / "choice.json"
    svg_out = tmp_path / "out" / "choice.svg"

    write_results(json_out, artifact)
    render_svg(artifact, svg_out)

    assert json.loads(json_out.read_text())["schema_version"] == 2
    svg = svg_out.read_text()
    assert "Training-free choice readout" in svg
    assert "Typed Decisions" in svg
    assert "JevJudge text" in svg
    assert "TRANSFER-DEV-TUNED" in svg
