from jevany.webshop_harness import BedrockJevWebShopAgent


class FakeShop:
    def __init__(self):
        self.page = "search"

    def reset(self, *, seed, mode):
        self.page = "search"
        return "Search page", {}

    def get_instruction_text(self):
        return "Buy the blue shirt."

    def get_available_actions(self):
        if self.page == "search":
            return ["search[<content>]" ]
        if self.page == "results":
            return ["click[back to search]", "click[blue-shirt]"]
        if self.page == "product":
            return ["click[large]", "click[buy now]"]
        return []

    def step(self, action):
        if action.startswith("search["):
            self.page = "results"
        elif action == "click[blue-shirt]":
            self.page = "product"
        elif action == "click[buy now]":
            self.page = "done"
            return "Purchased", 1.0, True, {"success": 1, "action_is_effective": True}
        return self.page, 0.0, False, {"success": 0, "action_is_effective": True}

    def close(self):
        pass


class FakeBedrock:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        name, tool_input = next(self.calls)
        return {
            "output": {"message": {"role": "assistant", "content": [{
                "text": f"using {name}",
            }, {"toolUse": {
                "toolUseId": f"call-{len(self.requests)}",
                "name": name,
                "input": tool_input,
            }}]}},
            "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12},
        }


def jev_first(request):
    keys = list(request["questions"]["action"]["criteria"])
    probabilities = {key: 0.0 for key in keys}
    probabilities["1"] = 1.0
    return {
        "model": request["model"],
        "answers": {"action": {
            "type": "choice", "choice": "1", "confidence": 0.9,
            "probabilities": probabilities,
        }},
    }


def test_baseline_keeps_search_and_clicks_under_bedrock_control():
    client = FakeBedrock([
        ("search", {"query": "blue shirt"}),
        ("click", {"action_key": "1"}),
        ("click", {"action_key": "1"}),
    ])
    episode = BedrockJevWebShopAgent(client, "frontier", jev_first).run(
        FakeShop(), seed=7, mode="baseline",
    )
    assert episode.success
    assert episode.search_actions == 1
    assert episode.direct_actions == 3
    assert episode.jev_decisions == 0
    assert all(action.controller == "bedrock" for action in episode.actions)
    assert all(
        tool["toolSpec"]["name"] != "delegate_clicks"
        for request in client.requests
        for tool in request["toolConfig"]["tools"]
    )


def test_optional_mode_delegates_only_finite_clicks():
    client = FakeBedrock([
        ("search", {"query": "blue shirt"}),
        ("delegate_clicks", {"steps": 2, "subgoal": "Open the shirt and buy it."}),
    ])
    requests = []

    def decide(request):
        requests.append(request)
        choice = "1"
        keys = list(request["questions"]["action"]["criteria"])
        probabilities = {key: 0.0 for key in keys}
        probabilities[choice] = 1.0
        return {
            "model": request["model"],
            "answers": {"action": {
                "type": "choice", "choice": choice, "confidence": 0.9,
                "probabilities": probabilities,
            }},
        }

    episode = BedrockJevWebShopAgent(client, "frontier", decide).run(
        FakeShop(), seed=7, mode="optional",
    )
    assert episode.success
    assert [action.controller for action in episode.actions] == ["bedrock", "jev", "jev"]
    assert episode.search_actions == 1
    assert episode.delegation_calls == 1
    assert episode.jev_decisions == 2
    assert len(requests[0]["questions"]["action"]["criteria"]) == 2
    assert all(
        value.startswith("click[")
        for request in requests
        for value in request["questions"]["action"]["criteria"].values()
    )


def test_low_confidence_returns_control_without_clicking():
    client = FakeBedrock([
        ("search", {"query": "blue shirt"}),
        ("delegate_clicks", {"steps": 2, "subgoal": "Open a matching product."}),
        ("click", {"action_key": "1"}),
        ("click", {"action_key": "1"}),
    ])

    def uncertain(request):
        keys = list(request["questions"]["action"]["criteria"])
        probabilities = {key: 0.0 for key in keys}
        probabilities["1"] = 1.0
        return {
            "model": request["model"],
            "answers": {"action": {
                "type": "choice", "choice": "1", "confidence": 0.4,
                "probabilities": probabilities,
            }},
        }

    episode = BedrockJevWebShopAgent(
        client, "frontier", uncertain, confidence_threshold=0.6,
    ).run(FakeShop(), seed=7, mode="optional")
    assert episode.success
    assert episode.low_confidence_returns == 1
    assert episode.jev_decisions == 1
    assert all(action.controller == "bedrock" for action in episode.actions)


def test_jev_failure_returns_control_and_is_counted():
    client = FakeBedrock([
        ("search", {"query": "blue shirt"}),
        ("delegate_clicks", {"steps": 2, "subgoal": "Open a matching product."}),
        ("click", {"action_key": "1"}),
        ("click", {"action_key": "1"}),
    ])

    def unavailable(_request):
        raise TimeoutError("decision backend timeout")

    episode = BedrockJevWebShopAgent(client, "frontier", unavailable).run(
        FakeShop(), seed=7, mode="optional",
    )
    assert episode.success
    assert episode.jev_failures == 1
    assert episode.jev_decisions == 1
    assert all(action.controller == "bedrock" for action in episode.actions)
