"""Remote evaluation uses the public client contract and retries transient failures."""
import http.client
import io
import json
from types import SimpleNamespace
import urllib.error

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

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
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

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
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

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time.sleep", sleeps.append)
    predictor = RemotePredictor("http://localhost:8008")
    if recover:
        assert predictor(RECORD)["input_tokens"] == 11
    else:
        with pytest.raises((urllib.error.URLError, TimeoutError)):
            predictor(RECORD)
    assert len(attempts) == 3 and sleeps == [1, 2]


def interrupted_response(status, transfer):
    body = b'{"detail":"response interrupted"}'
    if transfer == "chunked":
        headers = b"Transfer-Encoding: chunked\r\n"
        wire_body = f"{len(body):x}\r\n".encode() + body + b"\r\n10\r\npartial"
    else:
        headers = f"Content-Length: {len(body) + 10}\r\n".encode()
        wire_body = body

    class Socket:
        def makefile(self, *_args):
            return io.BytesIO(f"HTTP/1.1 {status} Test\r\n".encode() + headers + b"\r\n" + wire_body)

    response = http.client.HTTPResponse(Socket())
    response.begin()
    return response


@pytest.mark.parametrize("transfer", ["length", "chunked"])
@pytest.mark.parametrize("retries", [1, 3])
@pytest.mark.parametrize("recover", [False, True])
def test_interrupted_success_responses_obey_retry_budget(monkeypatch, transfer, retries, recover):
    responses, sleeps = [], []
    attempts = []

    def urlopen(request, timeout):
        attempts.append(request)
        if recover and len(attempts) == retries:
            return io.BytesIO(json.dumps(RESPONSE).encode())
        response = interrupted_response(200, transfer)
        responses.append(response)
        return response

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time.sleep", sleeps.append)
    predictor = RemotePredictor("http://localhost:8008", retries=retries)
    if recover:
        assert predictor(RECORD)["input_tokens"] == 11
        assert predictor.served_model == "served"
    else:
        with pytest.raises(http.client.IncompleteRead):
            predictor(RECORD)
        assert predictor.served_model is None
    assert len(attempts) == retries
    assert sleeps == [2 ** attempt for attempt in range(retries - 1)]
    assert all(response.isclosed() for response in responses)


@pytest.mark.parametrize("status,expected_attempts", [(401, 1), (403, 1), (422, 1), (429, 3), (503, 3)])
def test_interrupted_error_bodies_preserve_http_status_and_retry_policy(monkeypatch, status, expected_attempts):
    responses, sleeps = [], []

    def urlopen(request, timeout):
        response = interrupted_response(status, "chunked")
        responses.append(response)
        raise urllib.error.HTTPError(request.full_url, status, "Rejected", response.headers, response)

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time.sleep", sleeps.append)
    with pytest.raises(DecisionHTTPError, match="response interrupted") as caught:
        RemotePredictor("http://localhost:8008")(RECORD)
    assert caught.value.code == status
    assert caught.value.body == b'{"detail":"response interrupted"}'
    assert caught.value.read() == caught.value.body
    assert len(responses) == expected_attempts
    assert sleeps == [2 ** attempt for attempt in range(expected_attempts - 1)]
    assert all(response.isclosed() for response in responses)


@pytest.mark.parametrize("failure", [None, 429, 503, "connection", "timeout", "interrupted"])
def test_remote_evaluation_latency_includes_attempts_and_backoff(monkeypatch, tmp_path, failure):
    from jevany.benchmark import evaluate_records

    elapsed, attempts, sleeps = 0.0, [], []

    def sleep(seconds):
        nonlocal elapsed
        elapsed += seconds
        sleeps.append(seconds)

    def urlopen(request, timeout):
        nonlocal elapsed
        elapsed += 0.25
        attempts.append(request)
        if failure is not None and len(attempts) < 3:
            if failure == "connection":
                raise urllib.error.URLError("connection refused")
            if failure == "timeout":
                raise TimeoutError("timed out")
            if failure == "interrupted":
                return interrupted_response(200, "chunked")
            raise urllib.error.HTTPError(
                request.full_url, failure, "Unavailable", {},
                io.BytesIO(b'{"detail":"try later"}'),
            )
        return io.BytesIO(json.dumps(RESPONSE).encode())

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time", SimpleNamespace(
        perf_counter=lambda: elapsed, sleep=sleep,
    ))
    predictor = RemotePredictor("http://localhost:8008")
    record = {
        **RECORD,
        "questions": {qid: {**question, "src": f"fixture_{question['type']}"}
                      for qid, question in RECORD["questions"].items()},
        "_meta": {"id": "fixture/0", "group_id": "0", "source": "fixture", "variant": "clean"},
    }
    output = tmp_path / "evaluation"
    report, _ = evaluate_records([record], predictor, output)
    expected_ms = 250.0 if failure is None else 3750.0
    prediction = json.loads((output / "predictions.jsonl").read_text())["prediction"]
    saved_report = json.loads((output / "report.json").read_text())
    assert prediction["latency_ms"] == pytest.approx(expected_ms)
    assert report["latency_ms"] == saved_report["latency_ms"] == {
        "median": expected_ms, "p95": expected_ms,
    }
    assert len(attempts) == (1 if failure is None else 3)
    assert sleeps == ([] if failure is None else [1, 2])

    # The next prediction starts a fresh interval, even after earlier retries.
    assert predictor(RECORD)["latency_ms"] == pytest.approx(250.0)


@pytest.mark.parametrize("body", [b"not json", b'{"answers":{}}'])
def test_bad_responses_are_not_retried(monkeypatch, body):
    attempts = []

    def urlopen(request, timeout):
        attempts.append(request)
        return io.BytesIO(body)

    monkeypatch.setattr("jevany.client._urlopen", urlopen)
    monkeypatch.setattr("jevany.predictors.time.sleep", lambda _: pytest.fail("bad responses must not retry"))
    with pytest.raises(ValueError):
        RemotePredictor("http://localhost:8008")(RECORD)
    assert len(attempts) == 1
