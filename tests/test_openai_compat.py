"""The public text-planning path runs against local HTTP fixtures without accounts."""
import io
import json
import subprocess
import sys
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevany.client import DecisionHTTPError
from jevany.openai_compat import ChatCompletionsGenerator


def completion(text="compiled"):
    return {"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}


def test_generator_maps_planning_parameters_without_mutating_them(monkeypatch):
    params = {"max_output_tokens": 42, "temperature": 0.2, "top_p": 0.9, "stop_sequences": ["END"]}

    def urlopen(request, timeout):
        assert request.full_url == "https://planner.example.com/prefix/v1/chat/completions"
        assert request.get_header("Authorization") == "Bearer test-only"
        assert request.get_method() == "POST" and timeout == 17
        assert json.loads(request.data) == {
            "model": "planner", "messages": [{"role": "user", "content": "编译问题"}], "stream": False,
            "max_completion_tokens": 42, "temperature": 0.2, "top_p": 0.9, "stop": ["END"],
        }
        return io.BytesIO(json.dumps(completion()).encode())

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
    planner = ChatCompletionsGenerator(
        "planner", base_url="https://planner.example.com/prefix/v1/", api_key="test-only", timeout=17,
    )
    assert planner.generate("编译问题", params) == {
        "text": "compiled", "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        "stop_reason": "stop",
    }
    assert params["max_output_tokens"] == 42 and params["stop_sequences"] == ["END"]


@pytest.mark.parametrize("kwargs", [
    {"model": ""}, {"model": " "}, {"base_url": "http://remote.example.com/v1"},
    {"base_url": "https://user:password@example.com/v1"},
    {"base_url": "file:///tmp/model"}, {"base_url": "https://example.com/v1?key=value"},
    {"timeout": 0}, {"timeout": float("nan")},
])
def test_generator_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ChatCompletionsGenerator(**{"model": "planner", **kwargs})


@pytest.mark.parametrize("prompt,params", [
    ("", None), (None, None), ("compile", {"stream": True}),
    ("compile", {"temperature": float("nan")}),
])
def test_invalid_generation_does_not_make_a_request(monkeypatch, prompt, params):
    monkeypatch.setattr("jevany.client._urlopen", lambda *_a, **_k: pytest.fail("invalid request sent"))
    with pytest.raises(ValueError):
        ChatCompletionsGenerator("planner").generate(prompt, params)


@pytest.mark.parametrize("payload", [
    [], {}, {"choices": []}, {"choices": [None]}, {"choices": [{"message": None}]},
    {"choices": [{"message": {"content": None}, "finish_reason": "tool_calls"}]},
    {"choices": [{"message": {"content": ["text"]}}]},
    completion(""), completion("   "),
])
def test_generator_rejects_missing_or_non_text_completions(monkeypatch, payload):
    monkeypatch.setattr("jevany.client._urlopen", lambda *_a, **_k: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="chat completion"):
        ChatCompletionsGenerator("planner").generate("compile")


def test_generator_preserves_server_errors_without_retrying(monkeypatch):
    attempts = []

    def urlopen(request, timeout):
        attempts.append(request)
        raise urllib.error.HTTPError(
            request.full_url, 422, "Rejected", {}, io.BytesIO(b'{"error":{"message":"model unavailable"}}'),
        )

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
    with pytest.raises(DecisionHTTPError, match="model unavailable"):
        ChatCompletionsGenerator("planner").generate("compile")
    assert len(attempts) == 1


def test_harness_explains_truncated_planner_json(monkeypatch):
    from jevany.harness import JevHarness

    payload = completion('{"questions":{"q":{"type":"n')
    payload["choices"][0]["finish_reason"] = "length"
    monkeypatch.setattr("jevany.client._urlopen", lambda *_a, **_k: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(ValueError, match="planner.*length"):
        JevHarness(ChatCompletionsGenerator("planner"), None).compile("compile a task", {})


def test_harness_example_compiles_and_decides_with_local_servers(monkeypatch, capsys):
    from examples.harness import main

    questions = {"route": {"type": "choice", "criteria": {"review": "Human", "approve": "Automatic"}}}
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, request, self.headers.get("Authorization")))
            if self.path == "/v1/chat/completions":
                payload = completion(json.dumps({"questions": questions}))
            elif self.path == "/v1/systemone":
                payload = {"model": "decision", "answers": {"route": {
                    "type": "choice", "choice": "review", "confidence": 0.8,
                    "probabilities": {"review": 0.9, "approve": 0.1},
                }}}
            else:
                self.send_error(404)
                return
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.delenv("JEVANY_PLANNER_API_KEY", raising=False)
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        main(["--planner-url", url + "/v1", "--planner-model", "fixture",
              "--base-url", url, "--task", "Route this case",
              "--evidence", '{"case":"ambiguous evidence"}'])
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    result = json.loads(capsys.readouterr().out)
    assert result["decision"]["answers"]["route"]["choice"] == "review"
    assert result["request"]["state"]["evidence"] == {"case": "ambiguous evidence"}
    assert result["planner"]["usage"]["prompt_tokens"] == 7
    assert [entry[0] for entry in received] == ["/v1/chat/completions", "/v1/systemone"]
    assert received[0][2] is None
    assert "ambiguous evidence" not in received[0][1]["messages"][0]["content"]
    assert received[1][1]["questions"] == result["request"]["questions"]


def test_public_planner_does_not_import_cloud_or_model_dependencies():
    script = """
import io
import sys

class NoOptionalImports:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "transformers", "boto3", "datasets", "openai"}:
            raise ImportError("optional dependency unavailable: " + name)
        return None

sys.meta_path.insert(0, NoOptionalImports())
from jevany.openai_compat import ChatCompletionsGenerator
from jevany.harness import JevHarness
import jevany.client
jevany.client._urlopen = lambda *a, **k: io.BytesIO(
    b'{"choices":[{"message":{"content":"compiled"},"finish_reason":"stop"}]}')
assert ChatCompletionsGenerator("fixture").generate("compile")["text"] == "compiled"
print("base-install-ok")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "base-install-ok"
