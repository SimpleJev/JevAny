"""Remote evaluation uses the public client contract and retries transient failures."""
import io
import json
import urllib.error
import urllib.request

import pytest

from jevany.client import DecisionHTTPError
from jevany.predictors import RemotePredictor


RECORD = {"state": "state", "questions": {
    "route": {"type": "choice", "criteria": {"a": "A", "b": "B"}, "label": "a"},
    "done": {"type": "noul", "label": True},
}}
RESPONSE = {"model": "served", "answers": {
    "route": {"type": "choice", "choice": "a", "confidence": 0.6,
              "probabilities": {"a": 0.8, "b": 0.2}},
    "done": {"type": "noul", "noul": 0.9},
}, "usage": {"input_tokens": 11}}


@pytest.mark.parametrize("url", [
    "http://decision.example.com", "ftp://example.com", "file:///tmp/model",
    "decision.example.com", "https://user:password@example.com",
    "https://example.com?key=value", "https://example.com#fragment",
])
def test_remote_evaluation_rejects_unsafe_or_ambiguous_endpoints(url):
    with pytest.raises(ValueError):
        RemotePredictor(url)


@pytest.mark.parametrize("kwargs", [
    {"timeout": 0}, {"timeout": float("inf")}, {"timeout": float("nan")},
    {"retries": 0}, {"retries": -1}, {"retries": 1.5}, {"retries": True},
])
def test_remote_evaluation_rejects_invalid_retry_configuration(kwargs):
    with pytest.raises(ValueError):
        RemotePredictor("http://localhost:8008", **kwargs)


def test_remote_evaluation_preserves_predictions_and_omits_labels_and_empty_credentials(monkeypatch):
    def urlopen(request, timeout):
        assert request.full_url == "http://127.0.0.1:8008/v1/systemone"
        assert request.get_header("Authorization") is None
        payload = json.loads(request.data)
        assert payload["model"] == "requested"
        assert "label" not in payload["questions"]["route"]
        assert timeout == 12
        return io.BytesIO(json.dumps(RESPONSE).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    predictor = RemotePredictor("http://127.0.0.1:8008/", model="requested", api_key=None, timeout=12)
    result = predictor(RECORD)
    assert result["probabilities"]["route"] == {"a": 0.8, "b": 0.2}
    assert result["probabilities"]["done"] == pytest.approx({"false": 0.1, "true": 0.9})
    assert result["input_tokens"] == 11 and result["latency_ms"] >= 0
    assert predictor.served_model == "served"


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_permanent_http_errors_surface_once_with_the_server_explanation(monkeypatch, status):
    attempts = []

    def urlopen(request, timeout):
        attempts.append(request)
        raise urllib.error.HTTPError(
            request.full_url, status, "Rejected", {},
            io.BytesIO(b'{"detail":"unknown model requested"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time.sleep", lambda _: pytest.fail("permanent errors must not retry"))
    with pytest.raises(DecisionHTTPError, match="unknown model requested") as caught:
        RemotePredictor("https://decision.example.com", api_key="private")(RECORD)
    assert caught.value.code == status and len(attempts) == 1
    assert "private" not in str(caught.value)


@pytest.mark.parametrize("failure", [429, 503, "connection", "timeout"])
@pytest.mark.parametrize("recover", [False, True])
def test_transient_failures_retry_without_sleeping_after_the_last_attempt(monkeypatch, failure, recover):
    attempts, sleeps = [], []

    def urlopen(request, timeout):
        attempts.append(request)
        if recover and len(attempts) == 3:
            return io.BytesIO(json.dumps(RESPONSE).encode())
        if failure == "connection":
            raise urllib.error.URLError("connection refused")
        if failure == "timeout":
            raise TimeoutError("timed out")
        raise urllib.error.HTTPError(
            request.full_url, failure, "Unavailable", {},
            io.BytesIO(b'{"detail":"try later"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time.sleep", sleeps.append)
    predictor = RemotePredictor("http://localhost:8008")
    if recover:
        assert predictor(RECORD)["input_tokens"] == 11
    else:
        with pytest.raises((urllib.error.URLError, TimeoutError)):
            predictor(RECORD)
    assert len(attempts) == 3 and sleeps == [1, 2]


@pytest.mark.parametrize("body", [b"not json", b'{"answers":{}}'])
def test_bad_responses_are_not_retried(monkeypatch, body):
    attempts = []

    def urlopen(request, timeout):
        attempts.append(request)
        return io.BytesIO(body)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time.sleep", lambda _: pytest.fail("bad responses must not retry"))
    with pytest.raises(ValueError):
        RemotePredictor("http://localhost:8008")(RECORD)
    assert len(attempts) == 1
