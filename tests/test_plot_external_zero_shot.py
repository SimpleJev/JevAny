"""Validation checks for the tracked external-comparison chart artifact."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.plot_external_zero_shot import PANELS, load_results


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/external-zero-shot-v1.json"


@pytest.fixture
def artifact():
    return json.loads(SOURCE.read_text())


def load_modified(tmp_path, artifact, panel, model, **changes):
    data = deepcopy(artifact)
    row = next(row for row in data[panel]["models"] if row["model"] == model)
    row.update(changes)
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    return load_results(path)


def test_tracked_chart_artifact_has_one_valid_thirteen_model_cohort():
    data = load_results(SOURCE)
    order = data["main_comparison_order"]
    assert len(order) == 13
    assert {panel["metric"] for panel in PANELS} == {"accuracy"}
    for panel in ("typed_decisions", "jevjudge_full", "jevjudge_text"):
        assert [row["model"] for row in data[panel]["ordered_models"]] == order
        assert data[panel]["chart_metric"] == "accuracy"


def test_published_jev_has_typed_accuracy_and_explicit_jevjudge_dashes():
    data = load_results(SOURCE)
    typed = {row["model"]: row for row in data["typed_decisions"]["ordered_models"]}
    full = {row["model"]: row for row in data["jevjudge_full"]["ordered_models"]}
    text = {row["model"]: row for row in data["jevjudge_text"]["ordered_models"]}

    assert typed["TypeSafe Jev 1.13.0"]["accuracy"] == pytest.approx(0.727)
    assert typed["TypeSafe Jev 1.13.0"]["status"] == "dataset-card result; not locally rerun"
    for panel in (full, text):
        assert panel["TypeSafe Jev 1.13.0"]["accuracy"] is None
        assert panel["TypeSafe Jev 1.13.0"]["status"] == "no matching published or local JevJudge result"


@pytest.mark.parametrize("value", [-0.01, 1.01, True, float("inf")])
def test_accuracy_must_be_a_finite_probability(tmp_path, artifact, value):
    with pytest.raises(ValueError, match="invalid accuracy"):
        load_modified(
            tmp_path,
            artifact,
            "typed_decisions",
            "JevAny-Qwen3.8-27B",
            accuracy=value,
        )


@pytest.mark.parametrize(
    "ci",
    [None, [0.2], [0.4, 0.3], [0.36, 0.40], [0.30, float("nan")]],
)
def test_skill_role_ci_must_be_finite_ordered_and_contain_score(tmp_path, artifact, ci):
    with pytest.raises(ValueError, match="skill_role CI"):
        load_modified(
            tmp_path,
            artifact,
            "jevjudge_full",
            "JevAny-Qwen3.8-27B",
            skill_role_ci_95=ci,
        )


@pytest.mark.parametrize("value", [1.01, True, float("inf")])
def test_skill_role_must_be_finite_and_bounded(tmp_path, artifact, value):
    with pytest.raises(ValueError, match="invalid skill_role"):
        load_modified(
            tmp_path,
            artifact,
            "jevjudge_full",
            "JevAny-Qwen3.8-27B",
            skill_role=value,
        )


def test_skill_role_can_be_less_than_minus_one_after_chance_correction(tmp_path, artifact):
    data = load_modified(
        tmp_path,
        artifact,
        "jevjudge_full",
        "JevAny-Qwen3.8-27B",
        skill_role=-1.2,
        skill_role_ci_95=[-1.4, -1.1],
    )
    assert data["jevjudge_full"]["ordered_models"][0]["skill_role"] == -1.2


@pytest.mark.parametrize(
    ("panel", "changes", "coverage"),
    [
        ("jevjudge_full", {"answered": 3219}, 3220),
        ("jevjudge_full", {"requested": 3219}, 3220),
        ("jevjudge_text", {"answered": 723}, 724),
        ("jevjudge_text", {"requested": 723}, 724),
    ],
)
def test_scored_jevjudge_rows_require_exact_coverage(tmp_path, artifact, panel, changes, coverage):
    with pytest.raises(ValueError, match=rf"exact {coverage}/{coverage} coverage"):
        load_modified(tmp_path, artifact, panel, "JevAny-Qwen3.8-27B", **changes)


def test_dash_rows_require_null_metrics_and_an_unavailable_status(tmp_path, artifact):
    with pytest.raises(ValueError, match="non-null fields"):
        load_modified(
            tmp_path,
            artifact,
            "jevjudge_full",
            "Kev-27B",
            skill_role=0.5,
        )
    with pytest.raises(ValueError, match="needs a no-matching status"):
        load_modified(
            tmp_path,
            artifact,
            "typed_decisions",
            "Kev-27B",
            status="local run",
        )
    with pytest.raises(ValueError, match="has an unavailable status"):
        load_modified(
            tmp_path,
            artifact,
            "jevjudge_full",
            "JevAny-Qwen3.8-27B",
            status="incompatible: fixture",
        )


def test_model_kind_must_match_across_all_panels(tmp_path, artifact):
    with pytest.raises(ValueError, match="inconsistent kind"):
        load_modified(
            tmp_path,
            artifact,
            "jevjudge_text",
            "Kev-27B",
            kind="open_rerun",
        )
