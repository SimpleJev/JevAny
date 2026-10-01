import math

import pytest

from jevany.metrics import EPSILON, fit_temperature, metrics, nll_at_temperature, paired_bootstrap


def test_default_nll_scores_the_returned_probability_distribution():
    row = {
        "p": [0.8, 0.2],
        "logits": [0.0, 0.0],
        "label": 0,
        "type": "noul",
    }

    assert nll_at_temperature(row) == pytest.approx(-math.log(0.8))
    assert metrics([row])["nll"] == pytest.approx(-math.log(0.8))


def test_additional_temperature_uses_recorded_logits():
    row = {
        "p": [0.8, 0.2],
        "logits": [0.0, 0.0],
        "label": 0,
        "type": "noul",
    }

    assert nll_at_temperature(row, temperature=2.0) == pytest.approx(math.log(2.0))


def test_default_nll_floors_a_returned_zero_even_when_logits_are_recorded():
    row = {
        "p": [0.0, 1.0],
        "logits": [-100.0, 0.0],
        "label": 0,
        "type": "noul",
    }

    assert nll_at_temperature(row) == pytest.approx(-math.log(EPSILON))


def test_temperature_fitting_still_fits_recorded_logits():
    # Three positive labels and one negative label have an optimum p=0.75.
    # Logits producing p=0.9 therefore have an exactly representable T=2 optimum
    # on fit_temperature's candidate grid.
    logit = 2 * math.log(3)
    rows = [
        {
            "p": [0.9, 0.1],
            "logits": [logit, 0.0],
            "label": label,
            "type": "noul",
            "variant": "clean",
            "source": "source",
            "task": "task",
            "inference_temperature": 1.0,
        }
        for label in (0, 0, 0, 1)
    ]

    assert fit_temperature(rows, aggregation="micro") == pytest.approx(2.0)


def test_temperature_fitting_uses_logits_at_the_t_one_grid_point():
    # Raw logits are calibrated at T=1 for a 3:1 label ratio. Deliberately
    # inconsistent returned p values must not create a discontinuity exactly
    # at T=1 in the fitting grid.
    rows = [
        {
            "p": [0.99, 0.01],
            "logits": [math.log(3), 0.0],
            "label": label,
            "type": "noul",
            "variant": "clean",
            "source": "source",
            "task": "task",
            "inference_temperature": 1.0,
        }
        for label in (0, 0, 0, 1)
    ]

    assert fit_temperature(rows, aggregation="micro") == pytest.approx(1.0)


def test_paired_bootstrap_nll_uses_returned_probabilities():
    common = {
        "variant": "clean",
        "source": "source",
        "group": "group",
        "task": "task",
        "type": "noul",
        "question": "decision",
        "keys": ["false", "true"],
        "label": 0,
    }
    candidate = [{**common, "id": "record", "p": [0.8, 0.2], "logits": [0.0, 0.0]}]
    reference = [{**common, "id": "record", "p": [0.5, 0.5], "logits": [0.0, 0.0]}]

    result = paired_bootstrap(candidate, reference, samples=10, seed=0, metric="nll", aggregation="micro")

    assert result["micro_nll_delta"] == pytest.approx(-math.log(0.8) + math.log(0.5))
