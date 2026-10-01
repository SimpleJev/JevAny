"""Connecting a model from the Playground and asking one custom question."""
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jevany.client import JevClient
from jevany.demos.server import DemoApplication

ANSWER = {"model": "sft", "latency_ms": 12.5, "answers": {"decision": {
    "type": "choice", "choice": "billing", "confidence": 0.6,
    "probabilities": {"billing": 0.7, "shipping": 0.2, "accounts": 0.1}}}}


class Endpoint:
    """A loopback System One endpoint whose /v1/models payload the test controls."""

    def __init__(self):
        self.models = [{
            "id": "sft", "aliases": ["jevany-latest"], "base": "Qwen/Qwen3.5-4B",
            "device": "cpu", "decision_mode": "pointer",
            "capabilities": {"media_types": [], "context_window": 4096},
            "limits": {"media_enabled": False, "state_tokens": 8192},
        }]
        self.status = 200
        self.requests = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def reply(self, value, status=200):
                body = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path != "/v1/models":
                    return self.reply({"detail": "not found"}, 404)
                if endpoint.status != 200:
                    return self.reply({"detail": "model is not ready"}, endpoint.status)
                self.reply({"models": endpoint.models})

            def do_POST(self):
                endpoint.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.reply(ANSWER)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=2)


@pytest.fixture
def endpoint():
    served = Endpoint()
    yield served
    served.close()


@pytest.fixture
def app():
    result = DemoApplication(timeout=10)
    yield result
    result.close()


def test_unconfigured_playground_reports_no_model_but_still_serves_replays(app):
    connection = app.config()["connection"]
    assert connection["configured"] is False and connection["reachable"] is False
    assert connection["model"] is None and connection["editable"] is True
    assert connection["default_base_url"] == "http://127.0.0.1:8008"
    # Nothing can carry an image without a model, whatever the launch default was.
    assert app.images is True and connection["image_requests"] is False
    assert connection["media"]["usable"] is False
    assert "connect a model first" in connection["media"]["reason"]
    with pytest.raises(ValueError, match="connect a model before enabling image input"):
        app.set_images(True)
    with pytest.raises(ValueError, match="connect a model"):
        app.decide("state", "question?", [{"name": "a"}, {"name": "b"}])


def test_a_configured_endpoint_stays_text_only_until_it_is_tested(endpoint, tmp_path):
    app = DemoApplication(JevClient(endpoint.url, timeout=10, model="sft"),
                          media_root=tmp_path, timeout=10)
    try:
        connection = app.connection()
        assert connection["configured"] is True and connection["reachable"] is False
        assert connection["served"] is None
        # --media-root alone does not make an unprobed endpoint image-capable.
        assert connection["images"] is True and connection["image_requests"] is False
        assert connection["media"]["usable"] is False
        assert "Test and connect" in connection["media"]["reason"]
        with pytest.raises(ValueError, match="Test and connect"):
            app.set_images(True)
        endpoint.models[0]["capabilities"]["media_types"] = ["image"]
        endpoint.models[0]["limits"]["media_enabled"] = True
        assert app.connect(endpoint.url)["image_requests"] is False
        assert app.set_images(True)["image_requests"] is True
    finally:
        app.close()


def test_config_keeps_the_keys_the_page_already_relied_on(app, endpoint):
    config = app.config()
    assert set(config) >= {"cases", "model", "images", "media_transport", "installed", "connection"}
    assert set(config["cases"]) == {"arm", "crafter", "doom"}
    assert config["model"] is None and config["media_transport"] == "inline"
    assert set(config["installed"]) == set(config["cases"])
    app.connect(endpoint.url)
    assert app.config()["model"] == "sft"


def test_connecting_reports_identity_and_capabilities_from_v1_models(app, endpoint):
    connection = app.connect(endpoint.url)
    assert connection["configured"] and connection["reachable"]
    assert connection["model"] == "sft" and connection["base_url"] == endpoint.url
    assert connection["served"]["base"] == "Qwen/Qwen3.5-4B"
    assert connection["served"]["aliases"] == ["jevany-latest"]
    assert connection["served"]["decision_mode"] == "pointer"
    assert connection["checked"] and isinstance(app.client, JevClient)
    # A text-only checkpoint keeps image input off and says why.
    assert connection["images"] is False and connection["image_requests"] is False
    assert connection["media"]["model_media_types"] == []
    assert "text-only" in connection["media"]["reason"]
    with pytest.raises(ValueError, match="text-only"):
        app.set_images(True)


def test_a_failed_connection_keeps_the_working_one(app, endpoint):
    app.connect(endpoint.url)
    working = app.client
    for bad, message in [
        ("http://127.0.0.1:1", "cannot reach"),
        ("https://127.0.0.1:1", "cannot reach"),
        ("ftp://127.0.0.1:8008", "HTTP or HTTPS"),
        ("http://example.com:8008", "must use HTTPS"),
    ]:
        with pytest.raises(ValueError, match=message):
            app.connect(bad)
        assert app.client is working and app.connection()["reachable"]
    endpoint.status = 503
    with pytest.raises(ValueError, match="model is not ready"):
        app.connect(endpoint.url)
    assert app.client is working and app.connection()["reachable"]


@pytest.mark.parametrize("broken,message", [
    ({"capabilities": ["image"]}, "capabilities for 'sft' that is not an object"),
    ({"limits": "8192"}, "limits for 'sft' that is not an object"),
    ({"limits": {"media_enabled": "false"}}, "media_enabled for 'sft' .* not a boolean"),
    ({"capabilities": {"media_types": "image"}}, "media_types for 'sft' .* not a list"),
    ({"capabilities": {"media_types": [{"type": "image"}]}}, "media_types for 'sft' .* not a list"),
    ({"aliases": "jevany-latest"}, "aliases for 'sft' that are not a list"),
    ({"device": {"name": "cuda"}}, "device for 'sft' that is not a name"),
    ({"id": ""}, "did not report a model id"),
    ({"id": None}, "did not report a model id"),
])
def test_a_malformed_selected_model_entry_is_never_adopted(app, endpoint, tmp_path, broken, message):
    endpoint.models[0]["capabilities"]["media_types"] = ["image"]
    endpoint.models[0]["limits"]["media_enabled"] = True
    app.media_root = tmp_path
    app.connect(endpoint.url)
    app.set_images(True)
    working, probe, served = app.client, app.probe, app.connection()["served"]
    endpoint.models = [{**copy.deepcopy(endpoint.models[0]), **broken}]
    with pytest.raises(ValueError, match=message):
        app.connect(endpoint.url)
    # The working connection, its probe and its image setting all survive.
    assert app.client is working and app.probe is probe
    assert app.connection()["served"] == served
    assert app.images is True and app.image_requests is True


def test_a_requested_model_the_endpoint_does_not_serve_names_what_it_serves(app, endpoint):
    with pytest.raises(ValueError, match="does not serve 'wrong-model'") as caught:
        app.connect(endpoint.url, "wrong-model")
    assert "sft" in str(caught.value) and "jevany-latest" in str(caught.value)
    assert app.client is None and app.connection()["configured"] is False
    assert app.connect(endpoint.url, "jevany-latest")["model"] == "jevany-latest"


def test_images_need_the_model_the_server_and_a_shared_directory(app, endpoint, tmp_path):
    endpoint.models[0]["capabilities"]["media_types"] = ["image"]
    endpoint.models[0]["limits"]["media_enabled"] = True
    connection = app.connect(endpoint.url)
    assert connection["images"] is False
    assert connection["media"]["transport"] == "inline"
    with pytest.raises(ValueError, match="--media-root"):
        app.set_images(True)
    app.media_root = tmp_path
    connection = app.set_images(True)
    assert connection["images"] and connection["image_requests"]
    assert connection["media"]["transport"] == "shared-files" and connection["media"]["usable"]
    endpoint.models[0]["limits"]["media_enabled"] = False
    app.connect(endpoint.url)
    assert app.image_requests is False
    with pytest.raises(ValueError, match="JEVANY_MEDIA_ROOT"):
        app.set_images(True)


def test_custom_decision_sends_one_choice_question_and_returns_probabilities(app, endpoint):
    app.connect(endpoint.url)
    result = app.decide(
        "  A customer was charged twice.  ", "Which team should handle this?",
        [{"name": "billing", "description": "Payments"}, {"name": "shipping", "description": ""},
         {"name": "accounts"}],
    )
    sent = endpoint.requests[-1]
    assert sent["model"] == "sft" and sent["state"] == "A customer was charged twice."
    question = sent["questions"]["decision"]
    assert question["type"] == "choice" and question["instructions"] == "Which team should handle this?"
    assert question["criteria"] == {"billing": "Payments", "shipping": None, "accounts": None}
    assert result["choice"] == "billing" and result["confidence"] == 0.6
    assert result["options"] == ["billing", "shipping", "accounts"]
    assert result["probabilities"]["billing"] == 0.7
    assert result["model"] == "sft" and result["latency_ms"] == 12.5


@pytest.mark.parametrize("state,question,options,message", [
    ("", "q?", [{"name": "a"}, {"name": "b"}], "state is required"),
    ("s", "  ", [{"name": "a"}, {"name": "b"}], "question is required"),
    ("s", "q?", [{"name": "a"}], "between 2 and 12"),
    ("s", "q?", [{"name": "a"}, {"name": "a"}], "duplicate option"),
    ("s", "q?", [{"name": "a"}, {"name": ""}], "option name is required"),
    ("s", "q?", [{"name": "a"}, {"name": "b", "description": "x" * 601}], "at most 600"),
    ("s", "q?", "a,b", "between 2 and 12"),
])
def test_custom_decision_rejects_incomplete_forms_before_calling_the_model(
    app, endpoint, state, question, options, message,
):
    app.connect(endpoint.url)
    with pytest.raises(ValueError, match=message):
        app.decide(state, question, options)
    assert endpoint.requests == []


def test_an_in_process_model_cannot_be_replaced_from_the_browser(app, endpoint):
    class Model:
        model_id = "in-process"

        def __call__(self, request):
            return ANSWER

    app.client = Model()
    connection = app.connection()
    assert connection["configured"] and connection["editable"] is False
    assert connection["base_url"] is None and connection["model"] == "in-process"
    with pytest.raises(ValueError, match="in-process"):
        app.connect(endpoint.url)
    assert isinstance(app.client, Model)


def test_new_endpoints_over_http_keep_the_same_origin_and_size_checks(app, endpoint):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from jevany.demos.server import make_server

    server = make_server(app, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def post(path, payload, origin=base):
        request = Request(base + path, data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json", "Origin": origin})
        with urlopen(request) as response:
            return json.load(response)

    try:
        connection = post("/api/connect", {"base_url": endpoint.url})
        assert connection["reachable"] and connection["model"] == "sft"
        assert post("/api/images", {"enabled": False})["images"] is False
        result = post("/api/decide", {"state": "charged twice", "question": "Which team?",
                                      "options": [{"name": "billing"}, {"name": "shipping"},
                                                  {"name": "accounts"}]})
        assert result["choice"] == "billing"
        # A state longer than the decide body limit is refused, not truncated.
        with pytest.raises(HTTPError) as caught:
            post("/api/decide", {"state": "x" * 40000, "question": "Which team?",
                                 "options": [{"name": "billing"}, {"name": "shipping"}]})
        assert caught.value.code == 400 and b"32768 bytes" in caught.value.read()
        for path in ("/api/connect", "/api/images", "/api/decide"):
            with pytest.raises(HTTPError) as caught:
                post(path, {}, origin="https://unrelated.example")
            assert caught.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_connect_validates_the_timeout_and_the_url_before_any_request(app):
    with pytest.raises(ValueError, match="model URL is required"):
        app.connect("")
    with pytest.raises(ValueError, match="timeout"):
        app.connect("http://127.0.0.1:8008", None, 0)
    with pytest.raises(ValueError, match="timeout"):
        app.connect("http://127.0.0.1:8008", None, "fast")
    assert app.client is None
