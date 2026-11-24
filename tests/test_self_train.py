import random

import pytest

from jevany.self_train import sample_answer


@pytest.mark.parametrize(("temperature", "draw", "expected"), [
    (1.0, 0.79, "a"),
    (1.0, 0.81, "b"),
    (0.5, 0.94, "a"),
    (0.5, 0.95, "b"),
    (2.0, 0.66, "a"),
    (2.0, 0.67, "b"),
])
def test_sampling_applies_temperature(temperature, draw, expected, monkeypatch):
    rng = random.Random(0)
    monkeypatch.setattr(rng, "random", lambda: draw)
    assert sample_answer({"a": 0.8, "b": 0.2}, rng, temperature) == expected


@pytest.mark.parametrize("temperature", [1e-4, 5e-324])
def test_sampling_preserves_ties_at_low_temperature(temperature):
    rng = random.Random(31)
    answers = [sample_answer({"a": 0.5, "b": 0.5}, rng, temperature) for _ in range(20)]
    assert answers == random.Random(31).choices(["a", "b"], k=20)


def test_sampling_selects_the_maximum_at_low_temperature():
    assert sample_answer({"a": 0.75, "b": 0.25}, random.Random(0), 1e-4) == "a"


@pytest.mark.parametrize("temperature", [1.0, 10.0, 1000.0])
def test_sampling_never_selects_zero_probability(temperature):
    probabilities = {"impossible": 0.0, "certain": 1.0, "also_impossible": 0.0}
    rng = random.Random(31)
    assert [sample_answer(probabilities, rng, temperature) for _ in range(20)] == ["certain"] * 20


@pytest.mark.parametrize(("draw", "expected"), [(0.005, "rare"), (0.02, "certain")])
def test_sampling_preserves_small_positive_probabilities(draw, expected, monkeypatch):
    rng = random.Random(0)
    monkeypatch.setattr(rng, "random", lambda: draw)
    assert sample_answer({"rare": 1e-20, "certain": 1.0}, rng, 10.0) == expected


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_sampling_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        sample_answer({"a": 0.5, "b": 0.5}, random.Random(0), temperature)


@pytest.mark.parametrize("probabilities", [
    {},
    {"a": 0.0, "b": 0.0},
    {"a": -0.1, "b": 1.1},
    {"a": float("nan"), "b": 1.0},
    {"a": float("inf"), "b": 1.0},
])
def test_sampling_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match="probabilities"):
        sample_answer(probabilities, random.Random(0))
