"""A tool-routing application using only the public decision client."""
import math

import pytest

from examples import tool_routing
from jevany.api import to_answers, to_record
from jevany.client import DecisionClient


class Client(DecisionClient):
    model_id = "fixture"

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        _, metadata = to_record(request)
        return {"answers": to_answers([self.probabilities], metadata)}


def test_tool_routing_uses_option_probability_and_keeps_execution_with_the_caller():
    client = Client([0.8, 0.1, 0.05, 0.05])
    result = tool_routing.run(client, min_probability=0.8)
    assert result["tool"] == "order_status"
    assert result["probability"] == 0.8
    assert not result["deferred"]
    assert result["decision"]["answers"]["tool"]["confidence"] < 0.8
    request = client.requests[0]
    assert request.state == {"message": tool_routing.MESSAGE}
    assert request.questions["tool"].criteria == tool_routing.TOOLS
    assert "label" not in request.questions["tool"].model_dump()


def test_tool_routing_defers_when_the_distribution_is_uncertain():
    result = tool_routing.run(Client([0.4, 0.3, 0.2, 0.1]))
    assert result["tool"] is None
    assert result["deferred"]
    assert result["probability"] == pytest.approx(0.4)
    assert result["decision"]["answers"]["tool"]["choice"] == "order_status"


def test_tool_routing_can_choose_a_different_tool_and_accept_a_custom_message():
    client = Client([0.02, 0.93, 0.03, 0.02])
    result = tool_routing.run(client, "How many days do I have to return my purchase?")
    assert result["tool"] == "return_policy"
    assert client.requests[0].state["message"].startswith("How many days")


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf])
def test_tool_routing_rejects_invalid_thresholds_before_calling_the_model(threshold):
    client = Client([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="min_probability"):
        tool_routing.run(client, min_probability=threshold)
    assert not client.requests


def test_tool_routing_cli_validates_before_loading_a_checkpoint(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["tool_routing", "--checkpoint", "unused", "--min-probability", "nan"])
    monkeypatch.setattr(tool_routing, "client", lambda args: pytest.fail("must validate before loading"))
    with pytest.raises(SystemExit) as caught:
        tool_routing.main()
    assert caught.value.code == 2
    assert "--min-probability" in capsys.readouterr().err


def test_tool_routing_runs_with_a_local_checkpoint(tiny_run):
    from jevany import JevModel

    _, sft, _ = tiny_run
    result = tool_routing.run(JevModel.from_pretrained(sft, device="cpu"))
    assert result["tool"] is None or result["tool"] in tool_routing.TOOLS
    assert result["deferred"] == (result["tool"] is None)
    assert 0 <= result["probability"] <= 1
    assert set(result["decision"]["answers"]["tool"]["probabilities"]) == set(tool_routing.TOOLS)
