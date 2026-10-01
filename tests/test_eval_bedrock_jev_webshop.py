from scripts.eval_bedrock_jev_webshop import compare, render_demo, select_demo, summarize


def episode(mode, success, tokens, calls, latency, *, seed=1, jev=0):
    return {
        "mode": mode,
        "split": "test",
        "seed": seed,
        "goal": "Buy the blue shirt.",
        "success": success,
        "reward": float(success),
        "actions": [{
            "index": 0,
            "controller": "jev" if jev else "bedrock",
            "action": "click[buy now]",
            "confidence": 0.9 if jev else None,
            "reward": float(success),
            "done": success,
        }],
        "bedrock_calls": calls,
        "direct_actions": 0 if jev else 1,
        "search_actions": 0,
        "delegation_calls": int(bool(jev)),
        "jev_decisions": jev,
        "low_confidence_returns": 0,
        "usage": {"inputTokens": tokens, "outputTokens": 10, "totalTokens": tokens + 10},
        "total_latency_ms": latency,
        "bedrock_latency_ms": latency - jev,
        "jev_latency_ms": float(jev),
        "transcript": [{
            "turn": 0,
            "text": "Routine click.",
            "tool": "delegate_clicks" if jev else "click",
            "tool_input": {"decisions": [{
                "subgoal": "Choose it.", "candidate_keys": ["0", "1"],
            }]} if jev else {"action_key": "0"},
        }],
    }


def test_summary_comparison_and_demo_selection():
    baseline = [episode("baseline", False, 200, 2, 1000)]
    optional = [episode("optional", True, 80, 1, 600, jev=1)]
    summaries = {
        "baseline": summarize(baseline, 2, 10),
        "optional": summarize(optional, 2, 10),
    }
    delta = compare(summaries, baseline + optional)
    assert delta["success_rate_delta"] == 1
    assert delta["bedrock_call_reduction"] == 0.5
    assert delta["bedrock_token_reduction"] > 0.5
    case = select_demo(baseline + optional)
    assert case["seed"] == 1
    markdown = render_demo(case, "opus", "jev")
    assert "Frontier + optional Jev" in markdown
    assert "Routine click." in markdown


def test_demo_selection_is_preregistered_not_best_outcome():
    episodes = [
        episode("baseline", True, 100, 2, 1000, seed=11),
        episode("optional", False, 200, 3, 1500, seed=11),
        episode("baseline", False, 200, 2, 1000, seed=12),
        episode("optional", True, 50, 1, 500, seed=12, jev=1),
    ]
    case = select_demo(episodes, seed=11)
    assert case["seed"] == 11
    assert not case["optional"]["success"]
    assert case["selection"] == "predeclared first paired seed"
