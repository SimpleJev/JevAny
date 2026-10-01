import json
import sys
from types import SimpleNamespace

import pytest

import scripts.eval_bedrock_jev_agent as evaluator
from scripts.eval_bedrock_jev_agent import compare, render_demo, select_demo, summarize


def episode(mode, success, tokens, calls, latency, *, seed=1, jev=0):
    return {
        "environment": "fixture", "mode": mode, "seed": seed, "success": success,
        "reward": float(success), "actions": [{"index": 0, "controller": "jev" if jev else "bedrock",
        "action_name": "right", "confidence": 0.9 if jev else None, "reward": float(success),
        "done": success}], "bedrock_calls": calls, "direct_actions": 0 if jev else 1,
        "delegation_calls": int(bool(jev)), "jev_decisions": jev, "low_confidence_returns": 0,
        "usage": {"inputTokens": tokens, "outputTokens": 10, "totalTokens": tokens + 10},
        "total_latency_ms": latency, "bedrock_latency_ms": latency - jev,
        "jev_latency_ms": float(jev),
        "transcript": [{"turn": 0, "text": "Routine move.",
                        "tool": "delegate_to_jev" if jev else "take_action",
                        "tool_input": {"subgoal": "Move right."} if jev else {"action_key": "1"}}],
    }


def test_summary_comparison_and_demo_selection():
    baseline = [episode("baseline", False, 200, 2, 1000)]
    optional = [episode("optional", True, 80, 1, 600, jev=1)]
    summaries = {
        "fixture/baseline": summarize(baseline, 2, 10),
        "fixture/optional": summarize(optional, 2, 10),
    }
    delta = compare(summaries, baseline + optional)["fixture"]
    assert delta["success_rate_delta"] == 1
    assert delta["bedrock_call_reduction"] == 0.5
    assert delta["bedrock_token_reduction"] > 0.5
    assert delta["paired"]["success"]["bootstrap_95_ci"] == [1.0, 1.0]
    case = select_demo(baseline + optional)
    assert case["seed"] == 1
    markdown = render_demo(case, "opus", "jev")
    assert "Frontier + optional Jev" in markdown
    assert "Routine move." in markdown


def test_compare_requires_exactly_matching_paired_seed_sets():
    baseline = [episode("baseline", True, 100, 1, 100, seed=1)]
    optional = [episode("optional", True, 80, 1, 90, seed=2, jev=1)]
    summaries = {
        "fixture/baseline": summarize(baseline),
        "fixture/optional": summarize(optional),
    }

    with pytest.raises(ValueError, match="unpaired seed sets"):
        compare(summaries, baseline + optional)


def test_compare_rejects_duplicate_results_for_a_pair():
    baseline = episode("baseline", True, 100, 1, 100, seed=1)
    optional = episode("optional", True, 80, 1, 90, seed=1, jev=1)
    summaries = {
        "fixture/baseline": summarize([baseline, baseline]),
        "fixture/optional": summarize([optional]),
    }

    with pytest.raises(ValueError, match="duplicate paired result"):
        compare(summaries, [baseline, baseline, optional])


def test_demo_selection_is_predeclared_not_selected_from_outcomes():
    first_pair = [
        episode("baseline", True, 80, 1, 100, seed=1),
        episode("optional", False, 200, 2, 200, seed=1, jev=1),
    ]
    later_cherry_pick = [
        episode("baseline", False, 500, 4, 500, seed=2),
        episode("optional", True, 10, 1, 50, seed=2, jev=1),
    ]

    case = select_demo(first_pair + later_cherry_pick)

    assert case["seed"] == 1
    assert case["selection"] == "predeclared first paired seed"


def test_main_records_bedrock_error_continues_pair_and_always_closes_env(
    monkeypatch, tmp_path,
):
    repository = tmp_path / "RAGEN"
    (repository / "ragen" / "env").mkdir(parents=True)
    (repository / "ragen" / "env" / "base.py").write_text("", encoding="utf-8")
    output_path = tmp_path / "result.json"
    environments = []

    class CloseTrackingEnvironment:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def make_environment(name, path):
        value = CloseTrackingEnvironment()
        environments.append(value)
        return value

    class Result:
        def as_dict(self):
            return {
                "mode": "optional", "seed": 7, "success": True, "reward": 1.0,
                "actions": [], "bedrock_calls": 1, "direct_actions": 0,
                "delegation_calls": 1, "jev_decisions": 1, "jev_failures": 0,
                "low_confidence_returns": 0, "protocol_repairs": 0,
                "termination_reason": "terminal_success",
                "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
                "bedrock_latency_ms": 10.0, "jev_latency_ms": 2.0,
                "total_latency_ms": 12.0, "transcript": [],
            }

    class Agent:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, env, goal, *, seed, mode, max_actions):
            if mode == "baseline":
                raise TimeoutError("bedrock timed out")
            return Result()

    monkeypatch.setattr(evaluator, "environment", make_environment)
    monkeypatch.setattr(evaluator, "runtime_client", lambda *args, **kwargs: (object(), "resolved"))
    monkeypatch.setattr(evaluator, "BedrockJevAgent", Agent)
    monkeypatch.setattr(
        evaluator.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="ragen-sha\n"),
    )
    monkeypatch.setattr(sys, "argv", [
        "eval_bedrock_jev_agent.py",
        "--ragen-repo", str(repository),
        "--environment", "frozen_lake",
        "--mode", "baseline",
        "--mode", "optional",
        "--episodes", "1",
        "--seed", "7",
        "--out", str(output_path),
    ])

    evaluator.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert [(item["mode"], item["status"]) for item in payload["episodes"]] == [
        ("baseline", "error"), ("optional", "complete"),
    ]
    assert payload["episodes"][0]["error_type"] == "TimeoutError"
    assert all(environment.closed for environment in environments)
