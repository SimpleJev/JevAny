"""Serving contracts with real, offline Transformers checkpoints."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from jevany import Choice, InferenceOptions, JevModel, Noul, Score, SystemOneRequest
from jevany.api import to_record
from jevany.backbones import BackboneAdapter, InferenceCapabilities
from jevany.checkpoint import Meta, write_meta
from jevany.inference import add_inference_arguments, inference_options_from_args
from jevany.model import DecisionModel, load_tokenizer
from jevany.serve import create_app
from test_backbones import make_base


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_checkpoint(root, family="llama", adapter="auto", legacy=False):
    torch.manual_seed(43)
    base, checkpoint = root / "base", root / "checkpoint"
    make_base(base, family, legacy=legacy)
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8, backbone_adapter=adapter)
    model.lm.save_pretrained(checkpoint, save_embedding_layers=False)
    tokenizer.save_pretrained(checkpoint)
    write_meta(checkpoint, Meta(
        base=str(base), head=model.head.state_dict(), lora=2, head_dim=8,
        special_embeddings=model.special_embeddings, tokenizer_saved=True,
        backbone_adapter=adapter, branch_mode=model.branch_mode,
    ))
    return checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    return make_checkpoint(tmp_path)


@pytest.fixture
def request_body():
    return SystemOneRequest(state="state " * 20, questions={
        "choice": Choice(instructions="choose", criteria={"a": None, "b": None}),
        "noul": Noul(instructions="choose yes"),
        "score": Score(instructions="choose other", criteria=["a", "b"]),
    })


def assert_answers_close(actual, expected):
    assert actual.keys() == expected.keys()
    for name, answer in actual.items():
        reference = expected[name]
        assert answer["type"] == reference["type"]
        if answer["type"] == "noul":
            assert answer["noul"] == reference["noul"]
        else:
            assert answer["probabilities"] == pytest.approx(reference["probabilities"], abs=2e-6)
            assert answer["confidence"] == reference["confidence"]
            if answer["type"] == "choice":
                assert answer["choice"] == reference["choice"]
            else:
                assert answer["score"] == reference["score"]
                assert answer["legend"] == reference["legend"]


@pytest.mark.parametrize("family", ["qwen35", "qwen35_moe", "llama", "gemma", "mistral", "glm", "nemotron", "gpt2"])
def test_checkpoint_python_http_and_prefix_parity(tmp_path, family, request_body):
    checkpoint = make_checkpoint(tmp_path, family)
    local = JevModel.from_pretrained(
        checkpoint, device="cpu", model_name=family,
        inference_options=InferenceOptions(prefix_min_tokens=1),
    )
    record, _ = to_record(request_body)
    encoded = local.runtime.model.encode(local.runtime.tok, record)
    expected_probabilities = local.runtime.model.probs(encoded)
    first = local(request_body)
    second = local(request_body)
    assert_answers_close(first["answers"], second["answers"])
    assert first["usage"] == second["usage"]
    for key, index in (("choice", 0), ("score", 2)):
        assert list(first["answers"][key]["probabilities"].values()) == pytest.approx(
            expected_probabilities[index].tolist(), abs=2e-6)
    assert first["answers"]["noul"]["noul"] == round(expected_probabilities[1][1].item(), 2)
    cache_enabled = int(local.runtime.model.inference_capabilities.prefix_cache)
    assert local.describe()["prefix_cache"]["hits"] == cache_enabled
    assert local.describe()["prefix_cache"]["misses"] == cache_enabled
    application = create_app(model=local)
    with TestClient(application) as http:
        assert application.state.server is local.runtime
        response = http.post("/v1/systemone", json=request_body.model_dump())
        assert response.status_code == 200
        assert_answers_close(response.json()["answers"], first["answers"])
        assert response.json()["model"] == family
        assert response.json()["usage"] == first["usage"]
        description = http.get("/v1/models").json()["models"][0]
        assert description["id"] == family
        assert description["capabilities"]["context_window"] == 512
        assert description["capabilities"]["media_types"] == []
        assert description["limits"]["branch_tokens"] == 512
        assert description["prefix_cache"]["hits"] == 2 * cache_enabled
        assert description["branch_mode"] == local.runtime.model.branch_mode
        kernels = description["acceleration"]["linear_attention_kernels"]
        assert (kernels is None) == (not local.runtime.model.hybrid)
        assert kernels is None or set(kernels) == {"flash_linear_attention", "causal_conv1d"}
        assert http.get("/health").status_code == 200
    # An injected model remains usable after the HTTP app shuts down.
    assert local(request_body)["model"] == family
    local.clear_cache()
    assert local.describe()["prefix_cache"]["cached_states"] == 0


@pytest.mark.parametrize("options", [
    {"max_state_tokens": 0}, {"max_branch_tokens": -1}, {"max_packed_tokens": 2.5},
    {"prefix_cache_size": -1}, {"prefix_min_tokens": True},
])
def test_invalid_inference_options(options):
    with pytest.raises(ValueError, match="must be an integer"):
        InferenceOptions(**options)


def test_environment_settings_and_cli_overrides(monkeypatch):
    assert InferenceOptions.from_env({}) == InferenceOptions()
    monkeypatch.setenv("JEVANY_MAX_STATE_TOKENS", "128")
    monkeypatch.setenv("JEVANY_PREFIX_CACHE", "2")
    monkeypatch.setenv("JEVANY_PREFIX_MIN_TOKENS", "32")
    parser = argparse.ArgumentParser()
    add_inference_arguments(parser)
    actual = inference_options_from_args(parser.parse_args(["--prefix-cache-size", "0"]))
    assert actual == InferenceOptions(max_state_tokens=128, prefix_cache_size=0, prefix_min_tokens=32)
    with pytest.raises(ValueError, match="JEVANY_PREFIX_CACHE must be an integer"):
        InferenceOptions.from_env({"JEVANY_PREFIX_CACHE": "bad"})
    with pytest.raises(ValueError, match="prefix_cache_size"):
        InferenceOptions.from_env({"JEVANY_PREFIX_CACHE": "-1"})


def test_load_options_and_empty_model_fail_before_loading(tmp_path, monkeypatch):
    import jevany.runtime as runtime

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid options must fail before loading a checkpoint")

    monkeypatch.setattr(runtime, "Checkpoint", unexpected)
    for settings, error in [
        ({"device": ""}, "device"), ({"device": "bad"}, "device"),
        ({"dtype": "int8"}, "dtype"), ({"model_name": ""}, "model_name"),
        ({"model_name": " "}, "model_name"),
    ]:
        with pytest.raises(ValueError, match=error):
            JevModel.from_pretrained(tmp_path, **settings)
    monkeypatch.setenv("JEVANY_PREFIX_CACHE", "-1")
    with pytest.raises(ValueError, match="prefix_cache_size"):
        JevModel.from_pretrained(tmp_path)


def test_current_directory_checkpoint_and_explicit_empty_selector(checkpoint, monkeypatch):
    monkeypatch.chdir(checkpoint)
    local = JevModel.from_pretrained(".", device="cpu")
    assert local.model_id == "checkpoint"
    assert local.system_one("state", {"q": Noul()})["model"] == "checkpoint"
    with pytest.raises(ValueError):
        local.system_one("state", {"q": Noul()}, model="")


def test_unknown_architecture_defaults_to_no_cache():
    capabilities = BackboneAdapter().inference_capabilities(
        SimpleNamespace(model_type="custom_decoder", max_position_embeddings=1024))
    assert capabilities == InferenceCapabilities(context_window=1024)


def test_custom_cacheless_adapter_checkpoint_serves(tmp_path, monkeypatch, request_body):
    module = tmp_path / "cacheless_adapter.py"
    module.write_text(
        "from dataclasses import replace\n"
        "from jevany.backbones import BackboneAdapter\n"
        "class Cacheless(BackboneAdapter):\n"
        "    def inference_capabilities(self, config):\n"
        "        return replace(super().inference_capabilities(config), prefix_cache=False)\n"
        "    def new_cache(self, model):\n"
        "        raise AssertionError('cache must never be constructed')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    checkpoint = make_checkpoint(tmp_path, adapter="cacheless_adapter:Cacheless")
    local = JevModel.from_pretrained(
        checkpoint, device="cpu", inference_options=InferenceOptions(prefix_min_tokens=0))
    expected = local(request_body)
    with TestClient(create_app(model=local)) as http:
        actual = http.post("/v1/systemone", json=request_body.model_dump())
        assert actual.status_code == 200
        assert actual.json()["answers"] == expected["answers"]
        description = http.get("/v1/models").json()["models"][0]
        assert description["backbone_adapter"] == "cacheless_adapter:Cacheless"
        assert not description["prefix_cache"]["enabled"]
        assert description["prefix_cache"]["hits"] == description["prefix_cache"]["misses"] == 0


def test_model_context_and_invalid_requests_fail_before_forward(checkpoint, request_body, monkeypatch):
    local = JevModel.from_pretrained(checkpoint, device="cpu")

    def unexpected(*args, **kwargs):
        raise AssertionError("invalid requests must not reach model forward")

    monkeypatch.setattr(local.runtime.model, "probs", unexpected)
    too_long = request_body.model_copy(update={"state": "state " * 520})
    with pytest.raises(ValueError, match="state exceeds 512"):
        local(too_long)
    with TestClient(create_app(model=local)) as http:
        for body, error in [
            (too_long.model_dump(), "state exceeds 512"),
            ({**request_body.model_dump(), "model": "other-model"}, "unknown model"),
            ({**request_body.model_dump(), "questions": {}}, None),
            ({**request_body.model_dump(), "model": ""}, None),
        ]:
            response = http.post("/v1/systemone", json=body)
            assert response.status_code == 422
            if error:
                assert error in response.json()["detail"]
    with pytest.raises(ValueError, match="does not support.*media"):
        local(request_body.model_dump() | {"media": [{"type": "image", "uri": "unused.png"}]})


def test_state_branch_and_packed_limits(checkpoint, request_body):
    baseline = JevModel.from_pretrained(checkpoint, device="cpu")
    record, _ = to_record(request_body)
    encoded = baseline.runtime.model.encode(baseline.runtime.tok, record)
    state_tokens = encoded["seg"].count(0)
    branch_tokens = state_tokens + max(encoded["seg"].count(i) for i in (1, 2, 3))
    packed_tokens = len(encoded["ids"])
    exact = InferenceOptions(max_state_tokens=state_tokens, max_branch_tokens=branch_tokens,
                             max_packed_tokens=packed_tokens, prefix_cache_size=0)
    baseline.runtime.inference_options = exact
    assert baseline(request_body)["answers"]
    for name, limit, error in [
        ("max_state_tokens", state_tokens, "state exceeds"),
        ("max_branch_tokens", branch_tokens, "branch too long"),
        ("max_packed_tokens", packed_tokens, "packed tokens"),
    ]:
        baseline.runtime.inference_options = replace(exact, **{name: limit - 1})
        with pytest.raises(ValueError, match=error):
            baseline(request_body)


def test_cache_eviction_failure_recovery_and_concurrent_requests(checkpoint, request_body, monkeypatch):
    local = JevModel.from_pretrained(
        checkpoint, device="cpu",
        inference_options=InferenceOptions(prefix_cache_size=1, prefix_min_tokens=0))
    first = local(request_body)
    local(request_body.model_copy(update={"state": "other"}))
    assert len(local.runtime.prefix_cache) == 1
    assert_answers_close(local(request_body)["answers"], first["answers"])
    forward = local.runtime.model.lm.forward

    def fail_after_forward(*args, **kwargs):
        forward(*args, **kwargs)
        raise RuntimeError("injected backend failure")

    with monkeypatch.context() as patch:
        patch.setattr(local.runtime.model.lm, "forward", fail_after_forward)
        with pytest.raises(RuntimeError, match="injected"):
            local(request_body)
    assert not local.runtime.prefix_cache
    assert_answers_close(local(request_body)["answers"], first["answers"])
    with TestClient(create_app(model=local)) as http:
        def call(index):
            return (local(request_body) if index % 2 else
                    http.post("/v1/systemone", json=request_body.model_dump()).json())

        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(call, range(8)))
    for output in outputs:
        assert_answers_close(output["answers"], first["answers"])
    assert len(local.runtime.prefix_cache) == 1
    assert local.describe()["prefix_cache"]["hits"] == 8


def test_factory_loads_once_at_startup_and_releases_runtime(checkpoint, request_body, monkeypatch):
    loader = JevModel.from_pretrained
    loaded = []

    def load(*args, **kwargs):
        local = loader(*args, **kwargs)
        loaded.append(local)
        return local

    monkeypatch.setattr(JevModel, "from_pretrained", load)
    application = create_app(checkpoint, device="cpu", model_name="factory",
                             inference_options=InferenceOptions(prefix_min_tokens=0))
    assert not loaded
    with TestClient(application) as http:
        assert len(loaded) == 1
        for _ in range(2):
            assert http.post("/v1/systemone", json=request_body.model_dump()).json()["model"] == "factory"
        assert len(loaded[0].runtime.prefix_cache) == 1
    assert application.state.server is None
    assert not loaded[0].runtime.prefix_cache
    with pytest.raises(ValueError, match="either"):
        create_app(checkpoint, model=loaded[0])


def test_apps_keep_identity_limits_and_cache_separate(checkpoint, request_body):
    first = create_app(checkpoint, device="cpu", model_name="first",
                       inference_options=InferenceOptions(prefix_min_tokens=0))
    second = create_app(checkpoint, device="cpu", model_name="second",
                        inference_options=InferenceOptions(max_state_tokens=10, prefix_cache_size=0))
    with TestClient(first) as a, TestClient(second) as b:
        assert a.post("/v1/systemone", json=request_body.model_dump()).json()["model"] == "first"
        assert b.post("/v1/systemone", json=request_body.model_dump()).status_code == 422
        assert a.get("/v1/models").json()["models"][0]["prefix_cache"]["cached_states"] == 1
        other = b.get("/v1/models").json()["models"][0]
        assert other["id"] == "second"
        assert other["limits"]["state_tokens"] == 10
        assert not other["prefix_cache"]["enabled"]


def test_unready_and_failed_startup(tmp_path, request_body):
    application = create_app(tmp_path / "missing", device="cpu")
    # A client without lifespan sees a proper unavailable status.
    http = TestClient(application)
    try:
        assert http.get("/health").status_code == 503
        assert http.get("/v1/models").status_code == 503
        assert http.post("/v1/systemone", json=request_body.model_dump()).status_code == 503
    finally:
        http.close()
    with pytest.raises(ValueError, match="checkpoint does not exist"):
        with TestClient(application):
            pass


def test_local_decide_cli_and_remote_option_rejection(checkpoint, request_body, tmp_path, capsys):
    from jevany.cli import main
    source = tmp_path / "request.json"
    source.write_text(request_body.model_dump_json())
    main(["decide", str(source), "--checkpoint", str(checkpoint), "--device", "cpu",
          "--model-name", "cli", "--prefix-cache-size", "0", "--max-state-tokens", "128"])
    assert json.loads(capsys.readouterr().out)["model"] == "cli"
    with pytest.raises(SystemExit):
        main(["decide", str(source), "--prefix-cache-size", "0"])


@pytest.mark.parametrize("family", ["qwen35", "qwen35_moe", "llama", "gemma4", "gemma4_per_layer",
                                  "gemma4_unified", "mistral3", "muse", "glm"])
def test_native_media_checkpoint_python_http_parity(tmp_path, family, monkeypatch):
    from PIL import Image
    from jevany import serve
    from jevany.model import load_preprocessor
    from test_multimodal_backbones import make_vision_base

    torch.manual_seed(43)
    base, checkpoint = tmp_path / "base", tmp_path / "checkpoint"
    make_vision_base(base, family)
    processor = load_preprocessor(base, multimodal=True)
    model = DecisionModel(base, processor, "cpu", lora=2, head_dim=8, multimodal=True)
    model.lm.save_pretrained(checkpoint, save_embedding_layers=False)
    processor.save_pretrained(checkpoint)
    write_meta(checkpoint, Meta(
        base=str(base), head=model.head.state_dict(), lora=2, head_dim=8,
        multimodal=True, special_embeddings=model.special_embeddings, tokenizer_saved=True,
        backbone_adapter=model.backbone_adapter, branch_mode=model.branch_mode,
    ))
    local = JevModel.from_pretrained(
        checkpoint, device="cpu", inference_options=InferenceOptions(prefix_min_tokens=0))
    image = tmp_path / "image.png"
    Image.new("RGB", (28, 28), "red").save(image)
    request = SystemOneRequest(
        state="state", media=[{"type": "image", "uri": str(image)}],
        questions={"color": Choice(instructions="choose", criteria={"red": None, "blue": None})},
    )
    expected = local(request)
    assert local.describe()["prefix_cache"]["cached_states"] == 0
    monkeypatch.setattr(serve, "MEDIA_ROOT", str(tmp_path))
    with TestClient(create_app(model=local)) as http:
        body = request.model_dump()
        body["media"][0]["uri"] = image.name
        actual = http.post("/v1/systemone", json=body)
        assert actual.status_code == 200, actual.text
        assert actual.json()["answers"] == expected["answers"]
        assert actual.json()["usage"] == expected["usage"]
        description = http.get("/v1/models").json()["models"][0]
        assert description["limits"]["media_enabled"]
        assert "image" in description["capabilities"]["media_types"]
        assert description["capabilities"]["max_media_questions"] == 1
        assert description["prefix_cache"]["cached_states"] == 0
        # A native media checkpoint also keeps the standard text endpoint.
        text = request.model_copy(update={"media": []})
        text_expected = local(text)
        text_actual = http.post("/v1/systemone", json=text.model_dump())
        assert text_actual.status_code == 200, text_actual.text
        assert_answers_close(text_actual.json()["answers"], text_expected["answers"])
        body["questions"]["second"] = {"type": "noul"}
        invalid = http.post("/v1/systemone", json=body)
        assert invalid.status_code == 422
        assert "at most 1 question" in invalid.json()["detail"]


def _varied_requests(request_body):
    return [request_body.model_copy(update={"state": "state " * (4 + 3 * index)}) for index in range(8)]


@pytest.mark.parametrize("family", ["llama", "qwen35"])
def test_request_batching_matches_serial_scoring(tmp_path, request_body, family):
    import time
    checkpoint = make_checkpoint(tmp_path, family)
    serial = JevModel.from_pretrained(checkpoint, device="cpu", inference_options=InferenceOptions())
    requests = _varied_requests(request_body)
    expected = [serial(request) for request in requests]
    assert serial.describe()["batching"] == {"enabled": False, "max_batch_size": 1, "batches": 0, "batched_requests": 0}

    batched = JevModel.from_pretrained(checkpoint, device="cpu",
                                       inference_options=InferenceOptions(max_batch_size=8, prefix_cache_size=0))
    runtime = batched.runtime
    runtime.model_lock.acquire()   # hold the model so every request queues before the first forward
    with ThreadPoolExecutor(len(requests)) as pool:
        try:
            futures = [pool.submit(batched, request) for request in requests]
            deadline = time.monotonic() + 30
            while len(runtime.pending) < len(requests) and time.monotonic() < deadline:
                time.sleep(0.01)
            queued = len(runtime.pending)
        finally:
            runtime.model_lock.release()
        actual = [future.result(timeout=60) for future in futures]
    assert queued == len(requests)
    for left, right in zip(actual, expected):
        assert_answers_close(left["answers"], right["answers"])
        assert left["usage"] == right["usage"]
    stats = batched.describe()["batching"]
    assert stats["batched_requests"] == len(requests) and stats["batches"] < len(requests)
    alone = batched(requests[0])   # an idle runtime scores one request alone
    assert_answers_close(alone["answers"], expected[0]["answers"])
    with TestClient(create_app(model=batched)) as http:
        response = http.post("/v1/systemone", json=requests[1].model_dump())
        assert response.status_code == 200
        assert_answers_close(response.json()["answers"], expected[1]["answers"])
        assert http.post("/v1/systemone", json=requests[1].model_dump() | {"questions": {}}).status_code == 422


def test_request_batching_isolates_failures_and_respects_token_budget(checkpoint, request_body, monkeypatch):
    local = JevModel.from_pretrained(checkpoint, device="cpu",
                                     inference_options=InferenceOptions(max_batch_size=8, prefix_cache_size=0))
    runtime = local.runtime
    record = to_record(request_body)[0]
    admitted = [runtime._admit(record) for _ in range(3)]
    from jevany.runtime import _Pending, _padded_tokens
    tokens = _padded_tokens([admitted[0][1]], runtime.model.branch_mode)
    runtime.inference_options = replace(runtime.inference_options, max_packed_tokens=2 * tokens)
    runtime.pending.extend(_Pending(item) for item in admitted)
    assert len(runtime._take_batch()) == 2 and len(runtime.pending) == 1
    runtime.pending.clear()

    slots = [_Pending(item) for item in admitted]
    original = runtime.model.probs
    broken = id(slots[1].admitted[1])
    single_forward = runtime.model.forward_batch
    def forward_batch(encodings):
        if len(encodings) > 1:
            raise RuntimeError("batch")
        return single_forward(encodings)
    monkeypatch.setattr(runtime.model, "forward_batch", forward_batch)
    def probs(encoding):
        if id(encoding) == broken:
            raise ValueError("bad request")
        return original(encoding)
    monkeypatch.setattr(runtime.model, "probs", probs)
    with runtime.model_lock:
        runtime._run_batch(slots)
    assert all(slot.done for slot in slots)
    assert slots[0].error is None and slots[2].error is None and slots[0].result[0] == slots[2].result[0]
    assert isinstance(slots[1].error, ValueError)
    assert runtime.batching and local.describe()["batching"]["enabled"]
    monkeypatch.setattr(runtime.model, "multimodal", True)
    assert not runtime.batching and not local.describe()["batching"]["enabled"]
