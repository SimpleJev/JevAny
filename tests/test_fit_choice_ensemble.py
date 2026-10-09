import pytest

from scripts.fit_choice_ensemble import (
    fit,
    headline_indices,
    metrics,
    pooled_logits,
    probabilities,
)


def row(identity, p, label, gold=None):
    value = {
        "id": identity,
        "question": "decision",
        "keys": ["left", "right"],
        "label": label,
        "p": p,
    }
    if gold is not None:
        value["gold"] = gold
    return value


def test_fit_selects_native_weight_on_calibration_accuracy_before_nll():
    choice = [row("a", [.9, .1], 1), row("b", [.8, .2], 1)]
    native = [row("a", [.1, .9], 1), row("b", [.2, .8], 1)]

    result = fit(choice, choice, native)

    assert result["selected"]["correct"] == 2
    assert result["selected"]["native_weight"] > .5
    assert len(result["grid"]) == 101


def test_soft_gold_metrics_use_ordered_labels_and_report_kl():
    rows = [row("a", [.75, .25], 0, gold=[.5, .5])]
    ps = probabilities(pooled_logits(rows, None, 0), 1.0)

    result = metrics(rows, ps)

    assert result["accuracy"] == 1.0
    assert result["brier"] == pytest.approx(.125)
    assert result["kl_from_gold"] == pytest.approx(
        result["cross_entropy_from_gold"] - result["gold_entropy"]
    )


def test_choice_only_fit_has_no_native_weight():
    rows = [row("a", [.7, .3], 0), row("b", [.4, .6], 1)]

    result = fit(rows, rows, None)

    assert result["selected"]["native_weight"] == 0.0
    assert len(result["grid"]) == 1


def test_transfer_selection_uses_only_clean_knowable_headline_rows():
    rows = [
        {**row("headline", [.7, .3], 0), "variant": "clean", "source": "task"},
        {**row("variant", [.7, .3], 0), "variant": "permuted", "source": "task"},
        {**row("unknown", [.7, .3], 0), "variant": "clean", "source": "unknowable"},
    ]

    assert headline_indices(rows) == [0]
