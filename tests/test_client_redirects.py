"""Exercise urllib's redirect chain with an in-memory HTTP transport."""
import io
import json
from email.message import Message
from types import SimpleNamespace
import urllib.request
import urllib.response

import pytest

from jevany.client import DecisionHTTPError, JevClient
from jevany.openai_compat import ChatCompletionsGenerator


@pytest.fixture
def transport(monkeypatch):
    state = SimpleNamespace(target="https://first.invalid/redirected", code=302, calls=[], payload={})

    class FixtureTransport(urllib.request.BaseHandler):
        handler_order = 100

        def http_open(self, request):
            state.calls.append(request)
            headers = Message()
            if len(state.calls) == 1:
                code, body = state.code, b""
                headers["Location"] = state.target
            else:
                code, body = 200, json.dumps(state.payload).encode()
                headers["Content-Type"] = "application/json"
            response = urllib.response.addinfourl(io.BytesIO(body), headers, request.full_url, code)
            response.msg = "OK" if code == 200 else "Redirect"
            return response

        https_open = http_open

    build_opener = urllib.request.build_opener
    monkeypatch.setattr(
        urllib.request, "build_opener",
        lambda *handlers: build_opener(urllib.request.ProxyHandler({}), FixtureTransport(), *handlers),
    )
    monkeypatch.setattr(urllib.request, "_opener", None)
    return state


def call_client(kind, transport):
    if kind == "planner":
        transport.payload = {"choices": [{"message": {"content": "planned"}, "finish_reason": "stop"}]}
        return ChatCompletionsGenerator(
            "fixture", base_url="https://first.invalid", api_key="fixture-only",
        ).generate("plan a task")
    client = JevClient("https://first.invalid", api_key="fixture-only")
    if kind == "models":
        transport.payload = {"models": [{"id": "fixture"}]}
        return client.models()
    transport.payload = {"answers": {"q": {"type": "noul", "noul": 0.75}}}
    return client.system_one("evidence", {"q": {"type": "noul"}})


@pytest.mark.parametrize("kind", ["models", "decision", "planner"])
@pytest.mark.parametrize("target", [
    "https://other.invalid/redirected",
    "https://first.invalid:444/redirected",
    "http://first.invalid/redirected",
    "https://user:password@first.invalid/redirected",
])
def test_redirects_cannot_leave_the_configured_origin(kind, target, transport):
    transport.target = target
    with pytest.raises(DecisionHTTPError, match="redirect") as caught:
        call_client(kind, transport)
    assert caught.value.code == 302
    assert len(transport.calls) == 1
    assert "fixture-only" not in str(caught.value)


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_same_origin_model_redirects_keep_authorization(code, transport):
    transport.code = code
    transport.target = "https://FIRST.invalid:443/redirected"
    assert call_client("models", transport) == [{"id": "fixture"}]
    assert len(transport.calls) == 2
    assert transport.calls[1].get_method() == "GET"
    assert transport.calls[1].get_header("Authorization") == "Bearer fixture-only"


@pytest.mark.parametrize("kind", ["decision", "planner"])
@pytest.mark.parametrize("code", [307, 308])
def test_same_origin_post_redirects_preserve_the_request(kind, code, transport):
    transport.code = code
    assert call_client(kind, transport)
    first, redirected = transport.calls
    assert redirected.get_method() == "POST"
    assert redirected.data == first.data
    assert redirected.get_header("Content-type") == "application/json"
    assert redirected.get_header("Authorization") == "Bearer fixture-only"


@pytest.mark.parametrize("kind", ["decision", "planner"])
@pytest.mark.parametrize("code", [301, 302, 303])
def test_post_redirects_cannot_silently_discard_the_body(kind, code, transport):
    transport.code = code
    with pytest.raises(DecisionHTTPError, match="redirect"):
        call_client(kind, transport)
    assert len(transport.calls) == 1
