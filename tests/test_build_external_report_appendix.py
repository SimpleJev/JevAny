import json

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("pypdf")

import matplotlib.pyplot as plt
from pypdf import PdfReader, PdfWriter

from scripts.build_external_report_appendix import (
    CHOICE_DATASET_SPECS,
    CHOICE_RUN_SPECS,
    build_appendix,
    load_choice_artifact,
    merge_report,
)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _external_artifact(path):
    value = {
        "typed_decisions": {
            "models": [
                {"model": "JevAny-Qwen3.8-27B", "accuracy": 0.728},
                {"model": "Jev 1.13 (OpenRouter)", "accuracy": 0.727},
            ],
        },
        "jevjudge_text": {
            "models": [
                {"model": "JevAny-Qwen3.8-27B", "accuracy": 0.6644},
                {"model": "Jev 1.13 (OpenRouter)", "accuracy": 0.6506},
                {"model": "Kev-27B", "accuracy": 0.6423},
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


@pytest.fixture
def appendix_inputs(tmp_path):
    external = tmp_path / "external.json"
    choice = tmp_path / "choice.json"
    chart = tmp_path / "chart.png"
    _external_artifact(external)
    _choice_artifact(choice)
    plt.imsave(chart, [[0.0, 0.5], [0.75, 1.0]], cmap="viridis")
    return external, chart, choice


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


def test_merge_report_preserves_first_19_pages_and_writes_exactly_23(
    appendix_inputs, tmp_path
):
    external, chart, choice = appendix_inputs
    appendix = tmp_path / "appendices.pdf"
    base = tmp_path / "base.pdf"
    merged = tmp_path / "merged.pdf"
    build_appendix(external, chart, choice, appendix)
    writer = PdfWriter()
    for _ in range(21):
        writer.add_blank_page(width=612, height=792)
    with base.open("wb") as stream:
        writer.write(stream)

    merge_report(base, appendix, merged, base_pages=19)

    assert len(PdfReader(merged).pages) == 23


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
