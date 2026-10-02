import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

try:
    import matplotlib
    import pypdf
except ImportError:
    if os.environ.get("JEVANY_REPORT_TESTS_REQUIRED") == "1":
        raise
    pytest.skip("report dependencies are installed in the report-test job", allow_module_level=True)

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter

from scripts.build_external_report_appendix import (
    CHOICE_DATASET_SPECS,
    CHOICE_RUN_SPECS,
    _choice_matrix_rows,
    build_appendix,
    load_choice_artifact,
    merge_report,
    page_invariants,
)
from scripts.build_choice_readout_results import (
    DATASETS as PRODUCER_DATASETS,
    RUN_SPECS as PRODUCER_RUN_SPECS,
    build_results,
    write_results,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _external_artifact(path):
    value = {
        "artifact_version": 5,
        "provenance": {"fixture": {"sha256": "e" * 64}},
        "typed_decisions": {
            "models": [
                {"model": "JevAny-Qwen3.8-27B", "accuracy": 0.728},
                {
                    "model": "Jev 1.13 (OpenRouter)",
                    "kind": "external_api",
                    "accuracy": 0.727,
                },
                {
                    "model": "Published top",
                    "kind": "published_only",
                    "accuracy": 0.770,
                },
            ],
        },
        "jevjudge_text": {
            "models": [
                {"model": "JevAny-Qwen3.8-27B", "accuracy": 0.6644},
                {
                    "model": "Jev 1.13 (OpenRouter)",
                    "kind": "external_api",
                    "accuracy": 0.6506,
                    "answered": 724,
                    "requested": 724,
                },
                {
                    "model": "Kev-27B",
                    "kind": "open_kev",
                    "accuracy": 0.6423,
                    "answered": 724,
                    "requested": 724,
                },
            ],
        },
        "jevjudge_full": {
            "models": [
                {
                    "model": "JevAny-Qwen3.8-27B",
                    "kind": "ours",
                    "accuracy": 0.6227,
                    "skill_role": 0.3555,
                    "skill_role_ci_95": [0.3284, 0.3818],
                    "nll": 0.878,
                    "ece": 0.094,
                },
                {
                    "model": "Jeff-Qwen3.5-2B",
                    "kind": "open_external",
                    "accuracy": 0.4823,
                    "skill_role": 0.1357,
                    "skill_role_ci_95": [0.1080, 0.1599],
                    "nll": 1.105,
                    "ece": 0.137,
                },
            ],
        },
    }
    _write_json(path, value)


def _metric(n, correct):
    return {"n": n, "correct": correct, "accuracy": correct / n}


def _choice_artifact(path):
    source_runs = []
    for index, (key, (family, weights, native)) in enumerate(CHOICE_RUN_SPECS.items()):
        source_runs.append({
            "key": key,
            "model": f"fixture-{key}",
            "family": family,
            "weights": weights,
            "native_readout": native,
            "selected": {
                "native_weight": 0.0 if native is None else 0.40 + index / 100,
                "choice_temperature": 1.20 + index / 100,
                "blend_temperature": 1.10 + index / 100,
            },
            "ensemble_protocol": {"selection_rows": 1_046},
        })

    datasets = {}
    for dataset_index, (dataset, _label, records, questions, n) in enumerate(
        CHOICE_DATASET_SPECS
    ):
        runs = []
        for run_index, (key, (family, weights, native)) in enumerate(
            CHOICE_RUN_SPECS.items()
        ):
            choice_correct = n // 2 + dataset_index * 2 + run_index
            row = {
                "run": key,
                "model": f"fixture-{key}",
                "family": family,
                "weights": weights,
                "native_readout": native,
                "zero_shot": {"choice_t1": _metric(n, choice_correct)},
                "transfer_dev_tuned": {
                    "transfer_dev_calibrated_choice": _metric(n, choice_correct),
                },
            }
            if native is not None:
                row["zero_shot"].update({
                    "native_shipped": _metric(n, choice_correct + 1),
                    "fixed_blend_0_5": _metric(n, choice_correct + 2),
                })
                row["transfer_dev_tuned"]["transfer_dev_tuned_blend"] = _metric(
                    n, choice_correct + 3
                )
            runs.append(row)
        datasets[dataset] = {
            "records": records,
            "questions": questions,
            "headline_n": n,
            "runs": runs,
        }
    datasets["jevjudge_full"] = {
        "label": "JevJudge full",
        "records": 3_220,
        "status": "unsupported",
        "accuracy": None,
        "reason": "text-only fixture",
    }
    value = {
        "schema_version": 2,
        "artifact": "choice-readout-v2",
        "method": {
            "option_ids": "A-Z followed by a-z; at most 52 options.",
            "calibration": "Scalar temperature changes probabilities but not argmax accuracy.",
        },
        "source_runs": source_runs,
        "datasets": datasets,
    }
    _write_json(path, value)
    return value


def _producer_metric(n, correct, excluded):
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "cross_entropy_from_gold": 0.5,
        "gold_entropy": 0.1,
        "kl_from_gold": 0.4,
        "brier": 0.2,
        "ece": 0.03,
        "mean_confidence": 0.7,
        "cohort": "clean and knowable",
        "excluded_non_headline_rows": excluded,
        "ece_bins": 15,
    }


def _producer_run(root, spec, ordinal):
    is_base = spec["weights"] == "frozen_base"
    canonical_base = f"Qwen/{spec['family']}"
    encoded_repository = spec["public_repository"].replace("/", "--")
    checkpoint = (
        f"/fixture/models--{encoded_repository}/snapshots/{str(ordinal + 1) * 40}"
        if spec["key"] != "direct4"
        else f"/fixture/local-release/{spec['key']}"
    )
    components = ["choice"] if is_base else ["choice", "native"]
    model = f"producer-{spec['key']}"
    reports = {}
    ensemble_datasets = {}
    for dataset_index, (dataset, expected) in enumerate(PRODUCER_DATASETS.items()):
        n = expected["headline_n"]
        excluded = expected["questions"] - n
        choice_correct = n // 2 + ordinal + dataset_index
        native_correct = choice_correct + 1
        reports[dataset] = {
            "dataset": dataset,
            "records": expected["records"],
            "questions": expected["questions"],
            "components_executed": components,
            "components": {
                "choice": {"clean": {"n": n, "acc": choice_correct / n}},
            },
        }
        if not is_base:
            reports[dataset]["components"]["native"] = {
                "clean": {"n": n, "acc": native_correct / n}
            }
        choice = _producer_metric(n, choice_correct, excluded)
        calibrated = dict(choice)
        calibrated["cross_entropy_from_gold"] = 0.45
        configurations = {
            "choice_t1": choice,
            "transfer_dev_calibrated_choice": calibrated,
        }
        if not is_base:
            configurations.update({
                "native_shipped": _producer_metric(n, native_correct, excluded),
                "fixed_blend_0_5": _producer_metric(n, native_correct + 1, excluded),
                "transfer_dev_tuned_blend": _producer_metric(
                    n, native_correct + 2, excluded
                ),
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
                },
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
            "python": "3.13.5",
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
            "selection_rows": 1_046,
            "selection_split": "Transfer-v9 development",
        },
        "selected": {
            "choice_temperature": 1.2,
            "native_weight": 0.0 if is_base else 0.50 + ordinal / 100,
            "blend_temperature": 1.0 + ordinal / 10,
            "selection_objective": "fixture",
        },
        "datasets": ensemble_datasets,
    }
    directory = root / spec["key"]
    directory.mkdir(parents=True)
    _write_json(directory / "manifest.json", manifest)
    _write_json(directory / "ensemble.json", ensemble)


def _base_report(path, pages=21):
    metadata = {
        "Title": "Synthetic uniquely identified base report",
        "CreationDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
    }
    with PdfPages(path, metadata=metadata) as pdf:
        for page_number in range(1, pages + 1):
            fig = plt.figure(figsize=(8.5, 11), facecolor="white")
            fig.text(
                0.1,
                0.8,
                f"UNIQUE BASE PAGE {page_number:02d}",
                fontsize=16,
                url=f"https://example.test/base/{page_number}",
            )
            fig.text(0.1, 0.7, f"resource marker {page_number * 17}", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)


def _annotation_uris(page):
    uris = []
    for reference in page.get("/Annots", []):
        annotation = reference.get_object()
        action = annotation.get("/A")
        if action is not None:
            action = action.get_object()
            if action.get("/URI") is not None:
                uris.append(str(action["/URI"]))
    return uris


@pytest.fixture
def appendix_inputs(tmp_path):
    external = tmp_path / "external.json"
    choice = tmp_path / "choice.json"
    chart = tmp_path / "chart.png"
    _external_artifact(external)
    _choice_artifact(choice)
    plt.imsave(chart, [[0.0, 0.5], [0.75, 1.0]], cmap="viridis")
    return external, chart, choice


@pytest.fixture
def producer_artifact(tmp_path):
    root = tmp_path / "producer-runs"
    for ordinal, spec in enumerate(PRODUCER_RUN_SPECS):
        _producer_run(root, spec, ordinal)
    external = tmp_path / "producer-external.json"
    _external_artifact(external)
    artifact = build_results(root, external)
    choice = tmp_path / "producer-choice.json"
    write_results(choice, artifact)
    return external, choice


def test_build_appendix_renders_four_pages_directly_from_choice_json(
    appendix_inputs, tmp_path
):
    external, chart, choice = appendix_inputs
    output = tmp_path / "appendices.pdf"

    build_appendix(external, chart, choice, output)

    pdf = PdfReader(output)
    assert len(pdf.pages) == 4
    method_text = pdf.pages[2].extract_text()
    results_text = pdf.pages[3].extract_text()
    assert "APPENDIX M" in method_text
    assert "52 exact IDs" in method_text
    assert "not temperature calibration" in method_text
    assert "Choice-token accuracy matrix" in results_text
    assert "Matched runs" in results_text
    assert "JevJudge text 724" in results_text
    # base4 JevBench fixture: floor(231 / 2) = 115 => 49.78%.
    assert "49.78" in results_text


def test_producer_artifact_flows_into_pointer_direct_and_tuned_pdf_rows(
    producer_artifact, tmp_path
):
    external, choice = producer_artifact
    chart = tmp_path / "producer-chart.png"
    output = tmp_path / "producer-appendices.pdf"
    plt.imsave(chart, [[0.0, 0.5], [0.75, 1.0]], cmap="viridis")

    loaded = load_choice_artifact(choice)
    source_by_key = {row["key"]: row for row in loaded["source_runs"]}
    assert source_by_key["pointer4"]["native_readout"] == "pointer"
    assert source_by_key["direct4"]["native_readout"] == "direct-token"
    assert source_by_key["pointer4"]["selected"]["native_weight"] == 0.51
    assert source_by_key["direct4"]["selected"]["native_weight"] == 0.52
    rows = _choice_matrix_rows(loaded)
    assert [(row["model"], row["readout"], row["run"]) for row in rows] == [
        ("Frozen Qwen3.5-4B", "Choice T=1", "base4"),
        ("JevAny 4B Pointer", "Native", "pointer4"),
        ("", "Choice T=1", "pointer4"),
        ("", "Tuned stack", "pointer4"),
        ("JevAny 4B Direct-Token", "Native", "direct4"),
        ("", "Choice T=1", "direct4"),
        ("", "Tuned stack", "direct4"),
        ("Frozen Qwen3.8-27B", "Choice T=1", "base27"),
        ("JevAny 27B Pointer", "Native", "pointer27"),
        ("", "Choice T=1", "pointer27"),
        ("", "Tuned stack", "pointer27"),
    ]
    matrix = {(row["run"], row["readout"]): row for row in rows}
    assert matrix[("pointer4", "Native")]["metrics"][0]["correct"] == 120
    assert matrix[("pointer4", "Tuned stack")]["metrics"][0]["correct"] == 122
    assert matrix[("direct4", "Native")]["metrics"][0]["correct"] == 121
    assert matrix[("direct4", "Tuned stack")]["metrics"][0]["correct"] == 123
    build_appendix(external, chart, choice, output)

    text = PdfReader(output).pages[3].extract_text()
    assert "JevAny 4B Pointer" in text
    assert "JevAny 4B Direct-Token" in text
    assert text.count("Tuned stack") == 3
    assert "4B Pointer: w=0.51, T=1.10" in text
    assert "4B Direct-Token: w=0.52, T=1.20" in text
    assert "27B Pointer: w=0.54, T=1.40" in text
    # JevBench producer values: Pointer native/tuned and Direct native/tuned.
    for accuracy in ("51.95", "52.81", "52.38", "53.25"):
        assert accuracy in text


def test_appendix_and_in_place_merge_are_deterministic_and_replace_old_pages(
    appendix_inputs, tmp_path
):
    external, chart, choice = appendix_inputs
    appendix_a = tmp_path / "appendices-a.pdf"
    appendix_b = tmp_path / "appendices-b.pdf"
    base = tmp_path / "base.pdf"
    merged_a = tmp_path / "merged-a.pdf"
    merged_b = tmp_path / "merged-b.pdf"
    inplace = tmp_path / "inplace.pdf"
    build_appendix(external, chart, choice, appendix_a)
    build_appendix(external, chart, choice, appendix_b)
    assert appendix_a.read_bytes() == appendix_b.read_bytes()
    appendix_metadata = PdfReader(appendix_a).metadata
    assert appendix_metadata["/CreationDate"] == "D:20261002000000Z"
    assert appendix_metadata["/ModDate"] == "D:20261002000000Z"
    assert appendix_metadata["/Title"] == "JevAny Technical Report — Evaluation Appendices"

    _base_report(base)
    original = PdfReader(base)
    original_invariants = [page_invariants(original.pages[index]) for index in range(19)]
    assert "UNIQUE BASE PAGE 20" in original.pages[19].extract_text()
    assert "UNIQUE BASE PAGE 21" in original.pages[20].extract_text()
    merge_report(base, appendix_a, merged_a, base_pages=19)
    merge_report(base, appendix_b, merged_b, base_pages=19)
    assert merged_a.read_bytes() == merged_b.read_bytes()

    merged = PdfReader(merged_a)
    assert len(merged.pages) == 23
    for index in range(19):
        assert f"UNIQUE BASE PAGE {index + 1:02d}" in merged.pages[index].extract_text()
        assert page_invariants(merged.pages[index]) == original_invariants[index]
        assert _annotation_uris(merged.pages[index]) == [
            f"https://example.test/base/{index + 1}"
        ]
    generated_text = [merged.pages[index].extract_text() for index in range(19, 23)]
    assert all("UNIQUE BASE PAGE 20" not in text for text in generated_text)
    assert all("UNIQUE BASE PAGE 21" not in text for text in generated_text)
    assert "APPENDIX L" in generated_text[0]
    assert "APPENDIX L" in generated_text[1]
    assert "APPENDIX M" in generated_text[2]
    assert "APPENDIX M" in generated_text[3]
    assert "Training-free choice-token readout" in generated_text[2]
    assert "Choice-token accuracy matrix" in generated_text[3]
    merged_metadata = merged.metadata
    assert merged_metadata["/CreationDate"] == "D:20261002000000Z"
    assert merged_metadata["/ModDate"] == "D:20261002000000Z"
    assert merged_metadata["/Title"] == (
        "JevAny: Toward General Decision Intelligence — merged technical report"
    )

    shutil.copyfile(base, inplace)
    merge_report(inplace, appendix_a, inplace, base_pages=19)
    first_inplace = inplace.read_bytes()
    assert len(PdfReader(inplace).pages) == 23
    merge_report(inplace, appendix_a, inplace, base_pages=19)

    assert inplace.read_bytes() == first_inplace == merged_a.read_bytes()


def test_merge_report_rejects_any_base_page_count_other_than_19(
    appendix_inputs, tmp_path
):
    external, chart, choice = appendix_inputs
    appendix = tmp_path / "appendices.pdf"
    base = tmp_path / "base.pdf"
    build_appendix(external, chart, choice, appendix)
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with base.open("wb") as stream:
        writer.write(stream)

    with pytest.raises(ValueError, match="exactly 19"):
        merge_report(base, appendix, tmp_path / "merged.pdf", base_pages=18)


def test_choice_artifact_requires_full_3220_to_be_unsupported(tmp_path):
    choice = tmp_path / "choice.json"
    value = _choice_artifact(choice)
    value["datasets"]["jevjudge_full"]["accuracy"] = 0.5
    _write_json(choice, value)

    with pytest.raises(ValueError, match="must be explicitly unsupported"):
        load_choice_artifact(choice)


def test_published_report_uses_one_current_27b_release():
    report = Path(__file__).resolve().parents[1] / "reports" / "JevAny_Tech_Report.pdf"
    pdf = PdfReader(report)
    assert len(pdf.pages) == 23

    expected = {
        1: ("86.04", "90.04"),
        4: ("44,319", "86.04", "90.04", "0.388", "0.195", "0.026"),
        7: ("86.04",),
        8: ("44,319", "39.43", "1,261.7", "2,082"),
    }
    stale = (
        "85.76", "90.48", "22,160", "18.83", "602.7", "1,423",
        "0.392", "0.200", "0.030",
    )
    for page_number, values in expected.items():
        text = pdf.pages[page_number - 1].extract_text() or ""
        assert all(value in text for value in values)
        assert all(value not in text for value in stale)
    compute_text = pdf.pages[7].extract_text() or ""
    assert "estimated cumulative seconds-per-record timing" in compute_text
    boundary_text = (pdf.pages[9].extract_text() or "").replace("-\n", "-")
    assert "The merged PDF is release-synchronized." in boundary_text
    assert "The original PDF is pre-" not in boundary_text

    method_text = pdf.pages[21].extract_text() or ""
    assert "current 27B release checkpoint: step 44,319" in method_text
    assert "matched prompt-v2/runtime reruns" in method_text
