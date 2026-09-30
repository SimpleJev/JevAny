"""Multi-GPU placement: option plumbing on any machine, sharded parity on two or more CUDA devices."""
import argparse
import json

import pytest
import torch

from jevany import checkpoint
from jevany.checkpoint import LoadOptions, add_placement_arguments, load_options_from_args
from jevany.model import DecisionModel, load_tokenizer
from test_backbones import RECORD, make_base


def test_placement_options_from_env_and_validation():
    assert LoadOptions.from_env({}).device_map is None
    options = LoadOptions.from_env({"JEVANY_DEVICE_MAP": "auto", "JEVANY_MAX_MEMORY_GIB": "22"})
    assert (options.device_map, options.max_memory_gib) == ("auto", 22.0)
    with pytest.raises(ValueError, match="device_map must be one of"):
        LoadOptions(device_map="cuda:1")
    with pytest.raises(ValueError, match="requires device_map"):
        LoadOptions(max_memory_gib=20)
    for bad in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            LoadOptions(device_map="auto", max_memory_gib=bad)


def test_placement_flags_override_environment():
    parser = argparse.ArgumentParser()
    add_placement_arguments(parser)
    env = {"JEVANY_DEVICE_MAP": "sequential", "JEVANY_MAX_MEMORY_GIB": "10", "JEVANY_DTYPE": "bf16"}
    assert load_options_from_args(parser.parse_args([]), env) is None   # callers fall back to from_env
    options = load_options_from_args(parser.parse_args(["--device-map", "balanced", "--max-memory-gib", "31"]), env)
    assert (options.device_map, options.max_memory_gib, options.dtype) == ("balanced", 31.0, torch.bfloat16)
    with pytest.raises(SystemExit):
        parser.parse_args(["--device-map", "cuda:0"])


def test_remote_decide_rejects_placement(tmp_path, capsys):
    from jevany import Choice, SystemOneRequest
    from jevany.cli import main
    request = tmp_path / "request.json"
    request.write_text(SystemOneRequest(state="state", questions={
        "q": Choice(instructions="choose", criteria={"a": None, "b": None})}).model_dump_json())
    with pytest.raises(SystemExit):
        main(["decide", str(request), "--device-map", "auto"])
    assert "require --checkpoint" in capsys.readouterr().err


def test_checkpoint_load_passes_placement(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "adapter_config.json").write_text(json.dumps({}), encoding="utf-8")
    ck = checkpoint.Checkpoint.__new__(checkpoint.Checkpoint)
    ck.requested, ck.path = "owner/checkpoint", str(run)
    ck.meta = checkpoint.Meta(base="owner/base", head={}, weights_dtype="bf16")
    seen = {}

    class Head:
        temperature = 1.0
        def load_state_dict(self, state):
            pass

    class Model:
        def __init__(self, source, tok, device, **kwargs):
            seen.update(device=device, **kwargs)
            self.head = Head()
        def eval(self):
            return self

    monkeypatch.setattr(checkpoint, "DecisionModel", Model)
    monkeypatch.setattr(checkpoint, "load_preprocessor", lambda *args, **kwargs: object())
    monkeypatch.setattr(ck, "warm_start", lambda model, meta: None)
    ck.load("cuda", LoadOptions(device_map="auto", max_memory_gib=31))
    assert (seen["device"], seen["device_map"], seen["max_memory_gib"]) == ("cuda", "auto", 31)


def test_device_map_requires_cuda(tmp_path):
    base = tmp_path / "base"
    make_base(base, "llama")
    with pytest.raises(ValueError, match="use device='cuda'"):
        DecisionModel(base, load_tokenizer(base), "cpu", lora=2, head_dim=8, device_map="auto")
    with pytest.raises(ValueError, match="requires device_map"):
        DecisionModel(base, load_tokenizer(base), "cpu", lora=2, head_dim=8, max_memory_gib=1)


def test_adapter_without_placement_is_rejected(tmp_path, monkeypatch):
    from jevany.backbones import BackboneAdapter

    class Legacy(BackboneAdapter):
        def load_model(self, name, *, revision, dtype, attn):
            pytest.fail("a legacy adapter must be rejected before loading")

    monkeypatch.setattr("jevany.model.get_backbone_adapter", lambda *args, **kwargs: Legacy())
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    base = tmp_path / "base"
    make_base(base, "llama")
    with pytest.raises(ValueError, match="does not support device_map"):
        DecisionModel(base, load_tokenizer(base), "cuda", lora=2, head_dim=8, device_map="auto")


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.parametrize("family,decision_mode", [("qwen35", "pointer"), ("llama", "pointer"),
                                                  ("gpt2", "lm_token")])
def test_sharded_forward_matches_single_device(tmp_path, family, decision_mode):
    base = tmp_path / "base"
    make_base(base, family, legacy=decision_mode == "lm_token")
    tokenizer = load_tokenizer(base)
    verbalizers = ["yes", "no"] if decision_mode == "lm_token" else None
    common = dict(lora=2, head_dim=8, decision_mode=decision_mode, verbalizers=verbalizers)
    single = DecisionModel(base, tokenizer, "cuda", **common).eval()
    # A budget of ~60% of the weights per GPU forces Accelerate to use both cards.
    size = sum(p.numel() * p.element_size() for p in single.lm.parameters())
    sharded = DecisionModel(base, tokenizer, "cuda", device_map="sequential",
                            max_memory_gib=0.6 * size / 2**30, **common).eval()
    if single.head is not None:
        sharded.head.load_state_dict(single.head.state_dict())
    sharded.lm.load_state_dict(single.lm.state_dict())
    assert len(sharded.devices) >= 2
    assert not sharded.inference_capabilities.prefix_cache
    encoded = single.encode(tokenizer, RECORD)
    for left, right in zip(single.probs(encoded), sharded.probs(encoded)):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
