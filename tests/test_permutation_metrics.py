"""Permutation metrics compare decisions in each request's option order."""
import json

import pytest

from jevany.benchmark import evaluate_records, summarize


def row(identity, keys, probabilities, *, question="q", variant="clean", parent=None):
    return {
        "id": identity, "question": question, "group": parent or identity,
        "source": "custom", "task": "custom_choice", "type": "choice",
        "variant": variant, "parent": parent or identity,
        "keys": keys, "label": keys.index("a"), "p": probabilities,
        "raw_probability_sum": sum(probabilities), "zero_count": probabilities.count(0),
    }


@pytest.mark.parametrize("original_p,keys,permuted_p,flip,delta", [
    pytest.param([0.5, 0.5], ["b", "a"], [0.5, 0.5], 1, 0, id="uniform-flip"),
    pytest.param([0.5, 0.5], ["a", "b"], [0.5, 0.5], 0, 0, id="uniform-stable"),
    pytest.param([0.4, 0.4, 0.2], ["c", "b", "a"], [0.2, 0.4, 0.4], 1, 0, id="partial-tie-flip"),
    pytest.param([0.4, 0.4, 0.2], ["c", "a", "b"], [0.2, 0.4, 0.4], 0, 0, id="partial-tie-stable"),
    pytest.param([0.2, 0.8], ["b", "a"], [0.5, 0.5], 0, 0.3, id="new-tie-same-choice"),
    pytest.param([0.5, 0.5], ["b", "a"], [0.8, 0.2], 1, 0.3, id="resolved-tie-flip"),
    pytest.param([0.5, 0.5], ["b", "a"], [0.2, 0.8], 0, 0.3, id="resolved-tie-stable"),
    pytest.param([0.7, 0.3], ["b", "a"], [0.3, 0.7], 0, 0, id="unique-winner-stable"),
    pytest.param([0.7, 0.3], ["b", "a"], [0.7, 0.3], 1, 0.4, id="unique-winner-flip"),
    pytest.param([0.2, 0.6, 0.2], ["c", "b", "a"], [0.2, 0.6, 0.2], 0, 0, id="nonmaximal-tie"),
])
def test_permutation_compares_selected_keys_in_each_option_order(original_p, keys, permuted_p, flip, delta):
    original = row("original", list("abc"[:len(original_p)]), original_p)
    permuted = row("permuted", keys, permuted_p, variant="permuted", parent="original")

    result = summarize([original, permuted])["permutation"]

    assert result == {"n": 1, "mean_max_delta": pytest.approx(delta), "flip_rate": flip}


@pytest.mark.parametrize("reverse", [False, True])
def test_permutation_pairs_each_parent_and_question_and_averages(reverse):
    rows = [
        row("first", ["a", "b"], [0.5, 0.5]),
        row("first", ["a", "b"], [0.8, 0.2], question="other"),
        row("second", ["a", "b"], [0.2, 0.8]),
        row("first-permuted", ["b", "a"], [0.5, 0.5], variant="permuted", parent="first"),
        row("first-permuted", ["b", "a"], [0.2, 0.8], question="other", variant="permuted", parent="first"),
        row("second-permuted", ["b", "a"], [0.5, 0.5], variant="permuted", parent="second"),
    ]
    if reverse:
        rows.reverse()

    result = summarize(rows)["permutation"]

    assert result == {"n": 3, "mean_max_delta": pytest.approx(0.1), "flip_rate": pytest.approx(1 / 3)}


@pytest.mark.parametrize("population", ["empty", "clean-only", "non-choice"])
def test_permutation_without_choice_variants_has_no_measurements(population):
    original = row("original", ["a", "b"], [0.5, 0.5])
    rows = [] if population == "empty" else [original]
    if population == "non-choice":
        original.update(type="noul", keys=["false", "true"])
        rows.append({**original, "id": "permuted", "variant": "permuted"})

    assert summarize(rows)["permutation"] == {"n": 0, "mean_max_delta": None, "flip_rate": None}


def test_evaluation_persists_tie_flip_and_aligned_probability_delta(tmp_path):
    records = [
        {
            "state": "Choose an option.",
            "questions": {"q": {
                "type": "choice", "criteria": {key: key for key in keys},
                "label": "a", "src": "custom_choice",
            }},
            "_meta": {
                "id": identity, "group_id": "original", "source": "custom",
                "variant": variant, "parent_id": "original",
            },
        }
        for identity, keys, variant in [
            ("original", ["a", "b"], "clean"),
            ("permuted", ["b", "a"], "permuted"),
        ]
    ]

    def predict(_record):
        return {"probabilities": {"q": {"a": 0.5, "b": 0.5}}, "latency_ms": 1}

    output = tmp_path / "evaluation"
    report, rows = evaluate_records(records, predict, output)

    assert [result["keys"] for result in rows] == [["a", "b"], ["b", "a"]]
    assert report["permutation"] == {"n": 1, "mean_max_delta": 0.0, "flip_rate": 1.0}
    assert json.loads((output / "report.json").read_text()) == report
    assert json.loads((output / "rows.json").read_text()) == rows
