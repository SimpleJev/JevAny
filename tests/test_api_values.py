"""JSON evidence has the same meaning in local and HTTP requests."""
import json
from types import SimpleNamespace

import pytest

from jevany.api import SystemOneRequest, to_record
from jevany.client import JevClient


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
