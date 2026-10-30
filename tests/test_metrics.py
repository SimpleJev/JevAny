import json
import math
import subprocess
import sys

import pytest

from jevany.metrics import EPSILON, fit_temperature, metrics, nll_at_temperature, paired_bootstrap


@pytest.mark.parametrize("probabilities,label,kind,brier,rps", [
    ([1.0], 0, "choice", 0.0, None),
    ([0.8, 0.2], 0, "choice", 0.08, None),
    ([0.8, 0.2], 1, "choice", 1.28, None),
    ([0.0, 0.0, 1.0], 1, "choice", 2.0, None),
    ([0.25, 0.75], 1, "noul", 0.125, None),
    ([0.2, 0.5, 0.3], 0, "score", 0.98, 0.365),
    ([0.2, 0.5, 0.3], 1, "score", 0.38, 0.065),
    ([0.2, 0.5, 0.3], 2, "score", 0.78, 0.265),
])
def test_brier_and_ranked_scores(probabilities, label, kind, brier, rps):
    result = metrics([{"p": probabilities, "label": label, "type": kind}])
    assert result["brier"] == pytest.approx(brier)
    if rps is not None:
        assert result["ranked_probability_score"] == pytest.approx(rps)
    else:
        assert "ranked_probability_score" not in result


def test_brier_and_ranked_scores_after_temperature_scaling():
    result = metrics([{"p": [0.8, 0.2], "label": 0, "type": "score"}], temperature=2.0)
    assert result["brier"] == pytest.approx(2 / 9)
    assert result["ranked_probability_score"] == pytest.approx(1 / 9)


@pytest.mark.parametrize("operation", ["summary", "bootstrap"])
def test_many_options_use_bounded_metric_memory(operation):
    # Isolate allocation tracing from pytest and other tests. A 4096-by-4096
    # float64 target matrix consumes 128 MiB; a single target vector is 32 KiB.
    script = """
import json
import sys
import tracemalloc
from jevany.metrics import metrics, paired_bootstrap

count = 4096
row = {
    "p": [1 / count] * count, "label": 0, "type": "choice",
    "variant": "clean", "source": "example", "task": "routing",
    "group": "example", "question": "q", "id": "row",
    "keys": [str(index) for index in range(count)],
}
reference = {**row, "p": [1.0] + [0.0] * (count - 1)}
tracemalloc.stop()
tracemalloc.start()
if sys.argv[1] == "summary":
    brier = metrics([row])["brier"]
else:
    brier = paired_bootstrap(
        [row], [reference], samples=5, metric="brier",
    )["macro_brier_delta"]
_, peak = tracemalloc.get_traced_memory()
print(json.dumps({"brier": brier, "peak_bytes": peak}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, operation], capture_output=True, text=True, check=True,
    )
    result = json.loads(completed.stdout)
    assert result["brier"] == pytest.approx(1 - 1 / 4096)
    assert result["peak_bytes"] < 8 * 1024 * 1024


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
