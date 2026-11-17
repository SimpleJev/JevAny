"""JSON evidence has the same meaning in local and HTTP requests."""
import io
import json
from types import SimpleNamespace

import pytest

from jevany.api import Media, SystemOneRequest, to_record
from jevany.client import JevClient


@pytest.fixture(params=["http", "local"])
def media_client(request, monkeypatch):
    received = []
    response = {"answers": {"q": {"type": "noul", "noul": 0.5}}}
    if request.param == "http":
        def urlopen(request, timeout):
            received.append(json.loads(request.data))
            return io.BytesIO(json.dumps(response).encode())

        monkeypatch.setattr("jevany.client._urlopen", urlopen)
        client = JevClient()
    else:
        from jevany import JevModel

        def answer(request):
            received.append(request.model_dump(mode="json"))
            return response

        client = JevModel(SimpleNamespace(model_id="jevany-latest", answer=answer))
    return client, received


@pytest.mark.parametrize("media", [
    False, 0, 0.0, "", {},
    [{"type": "image", "uri": ""}],
    [{"type": "audio", "uri": "audio.wav"}],
    [{"type": "image", "uri": f"{index}.png"} for index in range(33)],
], ids=["false", "zero", "zero-float", "empty-string", "empty-object",
        "empty-uri", "unsupported-type", "too-many"])
def test_clients_reject_invalid_media_before_dispatch(media_client, media):
    client, received = media_client
    with pytest.raises(ValueError, match="media"):
        client.system_one("evidence", {"q": {"type": "noul"}}, media=media)
    assert received == []


@pytest.mark.parametrize("kwargs,expected", [
    ({}, []),
    ({"media": None}, []),
    ({"media": []}, []),
    ({"media": [{"type": "image", "uri": "frame.png"}, Media(type="video", uri="clip.mp4")]},
     [{"type": "image", "uri": "frame.png"}, {"type": "video", "uri": "clip.mp4"}]),
    ({"media": [{"type": "image", "uri": f"{index}.png"} for index in range(32)]},
     [{"type": "image", "uri": f"{index}.png"} for index in range(32)]),
], ids=["omitted", "none", "empty-list", "image-and-video", "max-count"])
def test_clients_preserve_valid_media(media_client, kwargs, expected):
    client, received = media_client
    result = client.system_one("evidence", {"q": {"type": "noul"}}, **kwargs)
    assert result["answers"]["q"]["noul"] == 0.5
    assert len(received) == 1
    assert received[0]["media"] == expected


@pytest.mark.parametrize("count", [32, 33])
@pytest.mark.parametrize("as_json", [False, True], ids=["python", "json"])
def test_request_media_count_limit(count, as_json):
    request = {"state": "evidence", "questions": {"q": {"type": "noul"}},
               "media": [{"type": "image", "uri": f"{index}.png"} for index in range(count)]}
    validate = SystemOneRequest.model_validate_json if as_json else SystemOneRequest.model_validate
    payload = json.dumps(request) if as_json else request

    if count == 32:
        assert len(validate(payload).media) == 32
    else:
        with pytest.raises(ValueError, match="media"):
            validate(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["state", "instructions", "choice", "noul", "score"])
def test_nonfinite_evidence_is_rejected_by_python_and_json_validation(field, value):
    nested = {"measurements": [1, {"value": value}]}
    question = {"type": "noul"}
    request = {"state": "evidence", "questions": {"q": question}}
    if field == "state":
        request["state"] = nested
    elif field == "instructions":
        question["instructions"] = nested
    elif field == "choice":
        question.update(type="choice", criteria={"a": nested, "b": None})
    elif field == "noul":
        question["criteria"] = {"true": nested}
    else:
        question.update(type="score", criteria=["low", nested])

    with pytest.raises(ValueError, match="finite"):
        SystemOneRequest.model_validate(request)
    with pytest.raises(ValueError, match="finite"):
        SystemOneRequest.model_validate_json(json.dumps(request))


def test_finite_json_evidence_survives_the_wire_without_changing_model_input():
    values = [None, True, False, 0, -7, 1.25, 1e100, "NaN", "∞", {"nested": ["证据"]}]
    request = SystemOneRequest(state={"values": values}, questions={
        "q": {"type": "choice", "instructions": values, "criteria": {"a": values, "b": None}},
    })
    decoded = SystemOneRequest.model_validate_json(request.model_dump_json())
    assert decoded == request
    assert to_record(decoded) == to_record(request)


def test_client_rejects_nonfinite_evidence_before_sending(monkeypatch):
    monkeypatch.setattr(
        "jevany.client._urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid evidence was sent"),
    )
    with pytest.raises(ValueError, match="finite"):
        JevClient().system_one({"measurement": float("nan")}, {"q": {"type": "noul"}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_http_rejects_nonfinite_evidence_with_a_json_error(value):
    from fastapi.testclient import TestClient
    from jevany.serve import create_app

    received = []

    def answer(request):
        received.append(request)
        return {"answers": {"q": {"type": "noul", "noul": 0.5}}}

    model = SimpleNamespace(runtime=SimpleNamespace(answer=answer))
    with TestClient(create_app(model=model), raise_server_exceptions=False) as client:
        payload = {"state": {"measurement": [value]}, "questions": {"q": {"type": "noul"}}}
        response = client.post("/v1/systemone", content=json.dumps(payload),
                               headers={"content-type": "application/json"})
        assert response.status_code == 422, response.text
        assert response.json()["detail"][0]["loc"] == ["body", "state"]
        assert "finite" in response.json()["detail"][0]["msg"]
        assert received == []
        payload["state"]["measurement"] = [0.25]
        assert client.post("/v1/systemone", json=payload).status_code == 200
        assert len(received) == 1
