"""Remote evaluation preserves typed context rejections without hiding other errors."""
import io
import json
from types import SimpleNamespace
import urllib.error
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from jevany.backbones import InferenceCapabilities
from jevany.benchmark import evaluate_records
from jevany.client import DecisionHTTPError
from jevany.data import load_records
from jevany.inference import InferenceOptions
from jevany.model import ContextLengthError
from jevany.predictors import RemotePredictor
from jevany.runtime import DecisionRuntime
from jevany.serve import create_app


@pytest.fixture
def remote(monkeypatch):
    def answer(request):
        if request.model == "missing":
            raise ValueError("unknown model 'missing'")
        if request.state == "overlong":
            raise ContextLengthError("request exceeds 8192 packed tokens: 9000")
        if request.state == "invalid":
            raise ValueError("unsupported request")
        return {"model": "fixture", "answers": {"q": {"type": "noul", "noul": 0.75}},
                "usage": {"input_tokens": 3}}

    model = SimpleNamespace(runtime=SimpleNamespace(answer=answer))
    with TestClient(create_app(model=model)) as server:
        def predictor(model_id="fixture"):
            result = RemotePredictor("https://fixture.invalid", model=model_id, retries=1)

            def open_request(request):
                response = server.request(request.get_method(), urlsplit(request.full_url).path,
                                          content=request.data, headers=dict(request.header_items()))
                if response.status_code >= 400:
                    error = urllib.error.HTTPError(request.full_url, response.status_code,
                                                   response.reason_phrase, response.headers,
                                                   io.BytesIO(response.content))
                    raise DecisionHTTPError(error, response.content)
                return response.json()

            monkeypatch.setattr(result.client, "_open", open_request)
            return result

        yield predictor


def panel(tmp_path, states):
    path = tmp_path / "panel.jsonl"
    path.write_text("".join(json.dumps({
        "state": state, "questions": {"q": {"type": "noul", "label": True}},
    }) + "\n" for state in states))
    return load_records(path)


def test_remote_custom_data_counts_context_rejections_and_continues(tmp_path, remote):
    records = panel(tmp_path, ["short", "overlong", "another short input"])
    output = tmp_path / "evaluation"
    report, rows = evaluate_records(records, remote(), output, skip_overlong=True)
    assert report["coverage"] == {
        "requested_records": 3, "requested_questions": 3,
        "evaluated_records": 2, "evaluated_questions": 2,
        "rejected_records": 1, "truncated_records": 0,
    }
    assert len(rows) == 2
    rejected = json.loads((output / "rejected.json").read_text())
    assert rejected == [{"id": records[1]["_meta"]["id"],
                         "error": "request exceeds 8192 packed tokens: 9000"}]
    assert not (output / "failure.json").exists()


def test_remote_frozen_suites_still_fail_on_a_context_rejection(tmp_path, remote):
    records = panel(tmp_path, ["overlong"])
    output = tmp_path / "evaluation"
    with pytest.raises(ContextLengthError, match="packed tokens"):
        evaluate_records(records, remote(), output)
    failure = json.loads((output / "failure.json").read_text())
    assert failure["error_type"] == "ContextLengthError"
    assert failure["record_id"] == records[0]["_meta"]["id"]


@pytest.mark.parametrize("state,model_id", [("short", "missing"), ("invalid", "fixture")])
def test_remote_custom_data_does_not_skip_unrelated_422_errors(tmp_path, remote, state, model_id):
    records = panel(tmp_path, [state])
    output = tmp_path / "evaluation"
    with pytest.raises(DecisionHTTPError) as caught:
        evaluate_records(records, remote(model_id), output, skip_overlong=True)
    assert caught.value.code == 422
    assert not caught.value.headers.get("X-JevAny-Error-Code")
    assert (output / "failure.json").exists()
    assert not (output / "rejected.json").exists()


@pytest.mark.parametrize("ids,positions,window,packed", [
    ([1, 2, 3], [0, 1, 2], 8, 2),
    ([1, 2, 3], [0, 1, 8], 8, 4),
])
def test_runtime_capacity_checks_raise_context_length_errors(ids, positions, window, packed):
    model = SimpleNamespace(
        decision_mode="pointer",
        inference_capabilities=InferenceCapabilities(context_window=window),
        encode=lambda *_args, **_kwargs: {"ids": ids, "pos": positions},
    )
    runtime = DecisionRuntime(None, None, model, "cpu",
                              inference_options=InferenceOptions(max_packed_tokens=packed))
    with pytest.raises(ContextLengthError, match="request exceeds"):
        runtime.probs({"state": "text", "questions": []})
