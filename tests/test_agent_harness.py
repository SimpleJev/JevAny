from copy import deepcopy

import pytest

from jevany.agent_harness import BedrockJevAgent, estimated_bedrock_cost


def decision(choice="1", confidence=0.9):
    return {"answers": {"action": {
        "type": "choice", "choice": choice, "confidence": confidence,
        "probabilities": {"0": 1 - confidence, "1": confidence},
    }}}


class Environment:
    ACTION_LOOKUP = {1: "left", 2: "right"}

    def reset(self, seed=None):
        self.position = 0
        return "position 0"

    def get_all_actions(self):
        return [1, 2]

    def step(self, action):
        self.position += 1 if action == 2 else -1
        done = self.position >= 2
        return f"position {self.position}", float(done), done, {
            "success": done, "action_is_effective": True,
        }


class Client:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.requests = []

    def converse(self, **request):
        self.requests.append(deepcopy(request))
        name, tool_input, text = next(self.calls)
        return {
            "output": {"message": {"role": "assistant", "content": [
                {"text": text},
                {"toolUse": {"toolUseId": f"tool-{len(self.requests)}", "name": name,
                             "input": tool_input}},
            ]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 100, "outputTokens": 20},
        }


class RawClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def converse(self, **request):
        self.requests.append(deepcopy(request))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def tool_response(*calls):
    return {
        "output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": tool_id, "name": name, "input": tool_input}}
            for tool_id, name, tool_input in calls
        ]}},
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 2},
    }


def test_baseline_calls_bedrock_for_each_action():
    client = Client([
        ("take_action", {"action_key": "1"}, "Move right."),
        ("take_action", {"action_key": "1"}, "Move right again."),
    ])
    episode = BedrockJevAgent(client, "opus", lambda request: decision()).run(
        Environment(), "reach position 2", seed=7, mode="baseline",
    )
    assert episode.success
    assert episode.bedrock_calls == episode.direct_actions == 2
    assert episode.jev_decisions == episode.delegation_calls == 0
    assert episode.usage == {"inputTokens": 200, "outputTokens": 40}
    assert all(len(request["toolConfig"]["tools"]) == 1 for request in client.requests)


def test_optional_delegation_reduces_frontier_calls_and_records_rationale():
    client = Client([("delegate_to_jev", {"steps": 2, "subgoal": "Move right to position 2."},
                      "This local path is routine.")])
    episode = BedrockJevAgent(client, "opus", lambda request: decision()).run(
        Environment(), "reach position 2", seed=7, mode="optional",
    )
    assert episode.success
    assert episode.bedrock_calls == 1
    assert episode.delegation_calls == 1 and episode.jev_decisions == 2
    assert episode.direct_actions == 0
    assert [action.controller for action in episode.actions] == ["jev", "jev"]
    assert episode.transcript[0]["text"] == "This local path is routine."
    assert episode.transcript[0]["tool_input"]["subgoal"] == "Move right to position 2."
    assert len(client.requests[0]["toolConfig"]["tools"]) == 2


@pytest.mark.parametrize("history_limit", [0, 1])
def test_delegation_respects_the_requested_history_limit(history_limit):
    requests = []

    def decide(request):
        requests.append(deepcopy(request))
        return decision()

    agent = BedrockJevAgent(
        None, "unused", decide, history_limit=history_limit,
    )
    episode = agent.run(Environment(), "reach position 2", mode="jev_only")
    assert episode.success and len(requests) == 2
    assert len(requests[1]["state"]["recent_actions"]) == history_limit


def test_delegation_rejects_a_negative_history_limit():
    with pytest.raises(ValueError, match="history_limit"):
        BedrockJevAgent(None, "unused", lambda request: decision(), history_limit=-1)


def test_low_confidence_returns_control_to_frontier_agent():
    client = Client([
        ("delegate_to_jev", {"steps": 2, "subgoal": "Move right."}, "Try the cheap policy."),
        ("take_action", {"action_key": "1"}, "I will handle this."),
        ("take_action", {"action_key": "1"}, "Finish."),
    ])
    confidences = iter((0.6,))
    agent = BedrockJevAgent(
        client, "opus", lambda request: decision(confidence=next(confidences)),
        confidence_threshold=0.7,
    )
    episode = agent.run(Environment(), "reach position 2", mode="optional")
    assert episode.success
    assert episode.low_confidence_returns == 1
    assert episode.jev_decisions == 1
    assert episode.direct_actions == 2
    assert all(action.controller == "bedrock" for action in episode.actions)


def test_jev_only_uses_no_bedrock_calls():
    episode = BedrockJevAgent(Client([]), "opus", lambda request: decision()).run(
        Environment(), "reach position 2", mode="jev_only",
    )
    assert episode.success and episode.bedrock_calls == 0
    assert episode.jev_decisions == 2


def test_jev_only_stops_instead_of_spinning_on_low_confidence():
    episode = BedrockJevAgent(
        Client([]), "opus", lambda request: decision(confidence=0.6),
        confidence_threshold=0.7,
    ).run(Environment(), "reach position 2", mode="jev_only")
    assert not episode.success and not episode.actions
    assert episode.jev_decisions == episode.low_confidence_returns == 1


def test_cost_requires_explicit_prices():
    usage = {"inputTokens": 2_000_000, "outputTokens": 1_000_000}
    assert estimated_bedrock_cost(usage, None, 10) is None
    assert estimated_bedrock_cost(usage, 3, 15) == 21


def test_multiple_tool_uses_are_rejected_with_one_result_per_tool_id_then_repaired():
    client = RawClient([
        tool_response(
            ("first", "take_action", {"action_key": "1"}),
            ("second", "take_action", {"action_key": "1"}),
        ),
        tool_response(("third", "take_action", {"action_key": "1"})),
        tool_response(("fourth", "take_action", {"action_key": "1"})),
    ])

    episode = BedrockJevAgent(client, "opus", lambda request: decision()).run(
        Environment(), "reach position 2", mode="baseline",
    )

    assert episode.success
    assert episode.protocol_repairs == 1
    assert episode.direct_actions == 2
    repair_blocks = client.requests[1]["messages"][-1]["content"]
    assert [block["toolResult"]["toolUseId"] for block in repair_blocks] == ["first", "second"]
    assert all(block["toolResult"]["status"] == "error" for block in repair_blocks)


class DelegationStopEnvironment(Environment):
    def __init__(self, *, reward=0.0, effective=True):
        self.first_reward = reward
        self.first_effective = effective

    def step(self, action):
        self.position += 1 if action == 2 else -1
        first = self.position == 1
        done = self.position >= 2
        return f"position {self.position}", self.first_reward if first else float(done), done, {
            "success": done,
            "action_is_effective": self.first_effective if first else True,
        }


@pytest.mark.parametrize(
    ("environment", "expected_reason"),
    [
        (DelegationStopEnvironment(reward=-1.0), "negative_reward"),
        (DelegationStopEnvironment(effective=False), "ineffective_action"),
    ],
)
def test_delegation_returns_control_after_anomalous_transition(environment, expected_reason):
    client = Client([
        ("delegate_to_jev", {"steps": 3, "subgoal": "Move right."}, "Delegate routine moves."),
        ("take_action", {"action_key": "1"}, "Recover directly."),
    ])

    episode = BedrockJevAgent(client, "opus", lambda request: decision()).run(
        environment, "reach position 2", mode="optional",
    )

    assert episode.success
    assert episode.jev_decisions == 1
    assert [action.controller for action in episode.actions] == ["jev", "bedrock"]
    delegation_result = client.requests[1]["messages"][-1]["content"][0]["toolResult"]
    assert delegation_result["status"] == "success"
    assert delegation_result["content"][0]["json"]["stop_reason"] == expected_reason


def test_jev_backend_failure_is_counted_and_control_returns_to_bedrock():
    client = Client([
        ("delegate_to_jev", {"steps": 2, "subgoal": "Move right."}, "Delegate."),
        ("take_action", {"action_key": "1"}, "Fallback one."),
        ("take_action", {"action_key": "1"}, "Fallback two."),
    ])

    episode = BedrockJevAgent(
        client, "opus", lambda request: (_ for _ in ()).throw(TimeoutError("jev timeout")),
    ).run(Environment(), "reach position 2", mode="optional")

    assert episode.success
    assert episode.jev_decisions == episode.jev_failures == 1
    assert episode.direct_actions == 2
    result = client.requests[1]["messages"][-1]["content"][0]["toolResult"]["content"][0]["json"]
    assert result["stop_reason"] == "backend_error"
    assert result["error_type"] == "TimeoutError"


def test_termination_reason_distinguishes_action_budget_and_protocol_failure():
    action_limited = BedrockJevAgent(
        Client([("take_action", {"action_key": "1"}, "One move.")]),
        "opus", lambda request: decision(),
    ).run(Environment(), "reach position 2", mode="baseline", max_actions=1)
    assert action_limited.termination_reason == "max_actions"

    empty_response = {
        "output": {"message": {"role": "assistant", "content": [{"text": "No tool."}]}},
        "usage": {"inputTokens": 1, "outputTokens": 1},
    }
    protocol_limited = BedrockJevAgent(
        RawClient([empty_response, empty_response, empty_response]),
        "opus", lambda request: decision(),
    ).run(Environment(), "reach position 2", mode="baseline")
    assert protocol_limited.termination_reason == "model_protocol_error"
    assert protocol_limited.protocol_repairs == 3
    assert not protocol_limited.success


def test_cache_cost_requires_and_uses_separate_cache_prices():
    usage = {
        "inputTokens": 1_000_000,
        "outputTokens": 1_000_000,
        "cacheReadInputTokens": 1_000_000,
        "cacheWriteInputTokens": 1_000_000,
    }
    assert estimated_bedrock_cost(usage, 3, 15) is None
    assert estimated_bedrock_cost(
        usage, 3, 15, cache_read_per_million=0.3, cache_write_per_million=3.75,
    ) == 22.05
class ScalarAction:
    def item(self):
        return 1


class ScalarActionEnv(Environment):
    def get_all_actions(self):
        return [ScalarAction()]

    ACTION_LOOKUP = {1: "advance"}


def test_environment_scalar_actions_are_normalized_for_json():
    client = Client([("take_action", {"action_key": "0"}, "Advance.")])
    episode = BedrockJevAgent(client, "frontier", lambda request: decision()).run(
        ScalarActionEnv(), "finish", seed=1, mode="baseline", max_actions=1,
    )
    assert episode.actions[0].action == 1
