import pytest

from scripts.evaluate_letter_readout import sample_tiers, tier_report


def records(per_tier=3):
    return [
        {"_meta": {"id": f"{tier}-{index}"}}
        for tier in ("easy", "original", "hard")
        for index in range(per_tier)
    ]


def test_sample_tiers_is_deterministic_and_balanced():
    selected = sample_tiers(records(), 2)
    assert [row["_meta"]["id"] for row in selected] == [
        "easy-0", "easy-1", "original-0", "original-1", "hard-0", "hard-1"
    ]
    assert sample_tiers(records(), 0) == records()
    with pytest.raises(ValueError, match="only 3 exist"):
        sample_tiers(records(), 4)


def test_tier_report_uses_argmax_accuracy():
    rows = [
        {"id": "easy-0", "p": [0.8, 0.2], "label": 0},
        {"id": "original-0", "p": [0.2, 0.8], "label": 0},
        {"id": "hard-0", "p": [0.1, 0.9], "label": 1},
    ]
    result = tier_report(rows)
    assert result["easy"] == {"questions": 1, "correct": 1, "accuracy": 1.0}
    assert result["original"] == {"questions": 1, "correct": 0, "accuracy": 0.0}
    assert result["hard"] == {"questions": 1, "correct": 1, "accuracy": 1.0}
