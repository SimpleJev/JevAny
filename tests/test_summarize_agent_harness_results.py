from pathlib import Path

import pytest

from scripts.summarize_agent_harness_results import row_from_result, validate_result


def result(optional_tokens=50):
    episodes = []
    for seed in (10, 11):
        for mode in ("baseline", "optional"):
            episodes.append({"environment": "toy", "mode": mode, "seed": seed})
    summaries = {
        "toy/baseline": {
            "success_rate": 1.0,
            "mean_delegation_calls": 0.0,
            "mean_jev_decisions": 0.0,
            "usage": {"inputTokens": 100, "outputTokens": 10},
        },
        "toy/optional": {
            "success_rate": 1.0,
            "mean_delegation_calls": 1.0,
            "mean_jev_decisions": 2.0,
            "usage": {"inputTokens": optional_tokens, "outputTokens": 5},
        },
    }
    paired = {
        key: {"bootstrap_95_ci": interval}
        for key, interval in {
            "bedrock_calls": [-2, -1], "tokens": [-60, -30], "latency_ms": [-50, -10]
        }.items()
    }
    return {
        "status": "complete",
        "bedrock_model": "us.anthropic.claude-opus-4-7",
        "resolved_bedrock_model": "arn:model",
        "jev_model": "jev-27b",
        "config": {"episodes": 2},
        "summaries": summaries,
        "comparisons": {"toy": {
            "success_rate_delta": 0.0,
            "bedrock_call_reduction": 0.5,
            "bedrock_token_reduction": 0.5,
            "wall_time_reduction": 0.2,
            "paired": paired,
        }},
        "episodes": episodes,
    }


def test_row_validates_pairs_and_identifies_clear_win(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("{}")
    row = row_from_result(result(), path)
    assert row["pairs"] == 2
    assert row["estimated_bedrock_cost_reduction"] > 0
    assert row["clear_quality_cost_latency_win"]


def test_validation_rejects_unpaired_seed(tmp_path):
    data = result()
    data["episodes"].pop()
    with pytest.raises(ValueError, match="unpaired seeds"):
        validate_result(data, tmp_path / "bad.json")
