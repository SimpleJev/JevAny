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


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="needs two usable CUDA devices")
def test_disk_offloaded_adapter_is_rejected(tmp_path):
    base = tmp_path / "base"
    make_base(base, "qwen35", legacy=True)
    tokenizer = load_tokenizer(base)
    reference = DecisionModel(base, tokenizer, "cuda", lora=2, head_dim=8)
    size = sum(parameter.numel() * parameter.element_size() for parameter in reference.lm.parameters())
    with pytest.raises(ValueError, match="offloaded adapter or readout weights"):
        DecisionModel(base, tokenizer, "cuda", lora=2, head_dim=8, device_map="sequential",
                      max_memory_gib=0.6 * size / 2**30)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="needs two usable CUDA devices")
@pytest.mark.parametrize("family,decision_mode", [("qwen35", "pointer"), ("llama", "pointer"),
                                                  ("gpt2", "lm_token")])
def test_sharded_forward_matches_single_device(tmp_path, family, decision_mode):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    base = tmp_path / "base"
    make_base(base, family, legacy=decision_mode == "lm_token")
    tokenizer = load_tokenizer(base)
    verbalizers = ["yes", "no"] if decision_mode == "lm_token" else None
    common = dict(lora=2, head_dim=8, decision_mode=decision_mode, verbalizers=verbalizers)
    torch.manual_seed(0)
    single = DecisionModel(base, tokenizer, "cuda", **common).eval()
    adapter_state = {name: value.detach().cpu() for name, value in get_peft_model_state_dict(single.lm).items()}
    # A budget of ~90% of the weights per GPU forces two-card placement without
    # introducing disk-offloaded meta tensors in these two-layer fixtures.
    size = sum(p.numel() * p.element_size() for p in single.lm.parameters())
    torch.manual_seed(0)
    sharded = DecisionModel(base, tokenizer, "cuda", device_map="sequential",
                            max_memory_gib=0.9 * size / 2**30, **common).eval()
    set_peft_model_state_dict(sharded.lm, adapter_state)
    if single.head is not None:
        sharded.head.load_state_dict(single.head.state_dict())
    assert len(sharded.devices) >= 2
    assert not sharded.inference_capabilities.prefix_cache
    encoded = single.encode(tokenizer, RECORD)
    for left, right in zip(single.probs(encoded), sharded.probs(encoded)):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="needs two usable CUDA devices")
def test_sharded_readout_transfers_only_selected_vectors(tmp_path, monkeypatch):
    from test_readout_selection import ConvertedShapes

    base = tmp_path / "base"
    make_base(base, "llama", legacy=True)
    tokenizer = load_tokenizer(base)
    reference = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8)
    size = sum(p.numel() * p.element_size() for p in reference.lm.parameters())
    model = DecisionModel(base, tokenizer, "cuda:0", lora=2, head_dim=8,
                          device_map="sequential", max_memory_gib=0.9 * size / 2**30).eval()
    readout = model._question_readout
    calls = []

    def observe(hidden, decide, options):
        assert hidden.device == torch.device("cuda:1")
        assert model.head.q.weight.device == torch.device("cuda:0")
        with ConvertedShapes() as converted:
            logits = readout(hidden, decide, options)
        assert tuple(hidden.shape) not in converted.shapes
        assert (len(options) + 1, hidden.shape[1]) in converted.shapes
        calls.append(True)
        return logits

    monkeypatch.setattr(model, "_question_readout", observe)
    model.probs(model.encode(tokenizer, RECORD))
    assert len(calls) == len(RECORD["questions"])
