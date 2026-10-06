"""The base installation's client path: no PyTorch, and HTTP failures that explain themselves."""
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

from jevany.client import ERROR_BODY_LIMIT, DecisionHTTPError, JevClient, error_detail

# Make `import torch` fail the way a base installation does, so a test environment
# that happens to have the modelling extras still proves the lightweight path.
NO_TORCH = """
import sys


class NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return None


sys.meta_path.insert(0, NoTorch())
"""

HELP_SCRIPT = NO_TORCH + """
from jevany.cli import main

try:
    main(["decide", "--help"])
except SystemExit as stop:
    assert stop.code in (0, None), stop.code
assert "torch" not in sys.modules, "decide --help imported torch"
print("help-ok")
"""

REMOTE_SCRIPT = NO_TORCH + """
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

ANSWER = {"model": "sft", "answers": {"team": {"type": "choice", "choice": "billing",
          "confidence": 0.6, "probabilities": {"billing": 0.8, "shipping": 0.2}}}}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps(ANSWER).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
from jevany.cli import main

with TemporaryDirectory() as directory:
    request = Path(directory) / "request.json"
    request.write_text(json.dumps({"state": "I was charged twice.", "model": "sft", "questions": {
        "team": {"type": "choice", "criteria": {"billing": None, "shipping": None}}}}))
    main(["decide", str(request), "--base-url", f"http://127.0.0.1:{server.server_port}"])
server.shutdown()
assert "torch" not in sys.modules, "remote decide imported torch"
"""


def run(script):
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_decide_help_does_not_need_torch():
    assert "help-ok" in run(HELP_SCRIPT)


def test_remote_decide_answers_without_torch():
    assert json.loads(run(REMOTE_SCRIPT))["answers"]["team"]["choice"] == "billing"


def failing_client(status, body, content_type="application/json", api_key="secret-key"):
    client = JevClient("http://127.0.0.1:8008", api_key=api_key, model="sft")

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, status, "Unprocessable Entity",
            {"content-type": content_type}, io.BytesIO(body),
        )

    return client, urlopen


@pytest.mark.parametrize("body,expected", [
    (b'{"detail":"unknown model \'wrong\'; this deployment serves \'sft\'"}',
     "unknown model 'wrong'; this deployment serves 'sft'"),
    (b'{"detail":[{"loc":["body","questions"],"msg":"field required"}]}', "field required"),
    (b'<html><body>\n502 Bad Gateway\n</body></html>', "502 Bad Gateway"),
    (b'', None),
])
def test_http_errors_show_the_server_reason_and_stay_inspectable(monkeypatch, body, expected):
    client, urlopen = failing_client(422, body)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.HTTPError) as caught:
        client.system_one("state", {"team": {"type": "choice", "criteria": {"a": None, "b": None}}})
    error = caught.value
    assert isinstance(error, DecisionHTTPError)
    assert error.code == 422 and error.status == 422 and error.reason == "Unprocessable Entity"
    assert error.read() == body and error.body == body
    assert str(error).startswith("HTTP Error 422: Unprocessable Entity")
    if expected is None:
        assert str(error) == "HTTP Error 422: Unprocessable Entity"
    else:
        assert expected in str(error)
    # The request's Authorization header never reaches the message.
    assert "secret-key" not in str(error) and "Bearer" not in str(error)


def test_error_bodies_are_read_with_a_bound_and_release_the_response(monkeypatch):
    oversized = b'{"detail":"' + b'z' * (ERROR_BODY_LIMIT * 2) + b'"}'

    class Body(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.amounts, self.released = [], False

        def read(self, amount=-1):
            self.amounts.append(amount)
            return super().read(amount)

        def close(self):
            self.released = True
            super().close()

    body = Body(oversized)

    def urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 502, "Bad Gateway", {}, body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = JevClient("http://127.0.0.1:8008")
    with pytest.raises(urllib.error.HTTPError) as caught:
        client.system_one("state", {"done": {"type": "noul"}})
    error = caught.value
    assert body.amounts == [ERROR_BODY_LIMIT], "the error body must be read with a bound"
    assert body.released, "the original response must be closed"
    assert len(error.body) == ERROR_BODY_LIMIT
    assert error.read() == error.body
    assert error.code == 502 and "zzz" in error.detail and len(error.detail) <= 403


def test_successful_responses_are_not_truncated_to_the_error_limit(monkeypatch):
    names = [f"option-{index:05d}-with-a-long-descriptive-name" for index in range(3000)]
    payload = {"model": "sft", "answers": {"team": {
        "type": "choice", "choice": names[0], "confidence": 0.0,
        "probabilities": {name: 1 / len(names) for name in names}}}}
    body = json.dumps(payload).encode()
    assert len(body) > ERROR_BODY_LIMIT
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: io.BytesIO(body))
    client = JevClient("http://127.0.0.1:8008", model="sft")
    result = client.system_one(
        "state", {"team": {"type": "choice", "criteria": {name: None for name in names}}})
    assert len(result["answers"]["team"]["probabilities"]) == len(names)


def test_error_detail_collapses_whitespace_and_bounds_length():
    assert error_detail(b'{"message":"a\\n  b"}') == "a b"
    assert error_detail(b'{"detail":"' + b'x' * 900 + b'"}').endswith("...")
    assert len(error_detail(b'{"detail":"' + b'x' * 900 + b'"}')) <= 403
    assert error_detail(b"   ") == ""


def test_models_round_trip_rejects_payloads_that_are_not_a_model_list(monkeypatch):
    client = JevClient("http://127.0.0.1:8008")
    served = [{"id": "sft", "aliases": ["jevany-latest"], "capabilities": {"media_types": ["image"]},
               "limits": {"media_enabled": True}}]
    seen = {}

    def urlopen(request, timeout):
        seen["url"], seen["method"] = request.full_url, request.get_method()
        return io.BytesIO(json.dumps(seen["payload"]).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    seen["payload"] = {"models": served}
    assert client.models() == served
    assert seen["url"] == "http://127.0.0.1:8008/v1/models" and seen["method"] == "GET"
    for payload in ({"models": []}, {"models": "sft"}, {"data": served}, ["sft"]):
        seen["payload"] = payload
        with pytest.raises(ValueError, match="models"):
            client.models()


@pytest.mark.parametrize("choice", [["a"], {"name": "a"}, None, 0, "missing"])
def test_malformed_choice_responses_raise_a_catchable_value_error(monkeypatch, choice):
    payload = {"answers": {"route": {
        "type": "choice", "choice": choice, "confidence": 0.5,
        "probabilities": {"a": 0.75, "b": 0.25},
    }}}
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda request, timeout: io.BytesIO(json.dumps(payload).encode()),
    )
    client = JevClient()
    with pytest.raises(ValueError, match="unknown choice.*route"):
        client.system_one("state", {"route": {"type": "choice", "criteria": {"a": None, "b": None}}})
