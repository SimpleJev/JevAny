"""Loading option validation and CLI precedence without model loads or listeners."""
import argparse
import io
import os
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from jevany import Choice, JevModel, SystemOneRequest
from jevany.checkpoint import LoadOptions, load_options_from_args


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("fp32", torch.float32),
    ("fp16", torch.float16), ("bf16", torch.bfloat16),
])
def test_load_options_accept_supported_dtype_settings(value, expected):
    env = {} if value is None else {"JEVANY_DTYPE": value}
    assert LoadOptions.from_env(env).dtype == expected


@pytest.mark.parametrize("value", ["bfloat16", "float16", "float32", "b16", "BF16", " bf16 "])
def test_load_options_reject_invalid_dtype_settings(value):
    with pytest.raises(ValueError, match="JEVANY_DTYPE must be fp32, fp16 or bf16"):
        LoadOptions.from_env({"JEVANY_DTYPE": value})


@pytest.mark.parametrize("readout", ["native", "choice"])
def test_invalid_dtype_fails_before_checkpoint_resolution(monkeypatch, readout):
    def unexpected_resolve(*args, **kwargs):
        pytest.fail("invalid dtype reached checkpoint resolution")

    monkeypatch.setattr("jevany.checkpoint.resolve_run", unexpected_resolve)
    monkeypatch.setenv("JEVANY_DTYPE", "bfloat16")
    with pytest.raises(ValueError, match="JEVANY_DTYPE must be fp32, fp16 or bf16"):
        JevModel.from_pretrained("unused/checkpoint", device="cpu", readout=readout)


@pytest.mark.parametrize("readout", ["native", "choice"])
@pytest.mark.parametrize("options", [None, LoadOptions(dtype=torch.float16, merge=False)])
@pytest.mark.parametrize("dtype,expected", [
    ("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16),
])
def test_explicit_dtype_overrides_invalid_environment_and_options(monkeypatch, readout, options,
                                                                 dtype, expected):
    def load(*args, **kwargs):
        actual = kwargs["options"] if readout == "choice" else args[1]
        assert actual.dtype == expected
        assert not actual.merge
        raise LoaderReached

    monkeypatch.setenv("JEVANY_DTYPE", "bfloat16")
    monkeypatch.setenv("JEVANY_MERGE", "0")
    monkeypatch.setattr("jevany.runtime.Checkpoint", lambda _: SimpleNamespace(
        requested="unused/checkpoint", load=load,
    ))
    monkeypatch.setattr("jevany.letter_predictor.LetterReadoutPredictor", load)
    with pytest.raises(LoaderReached):
        JevModel.from_pretrained("unused/checkpoint", device="cpu", readout=readout,
                                 dtype=dtype, options=options)
    assert os.environ["JEVANY_DTYPE"] == "bfloat16"


def test_dtype_flag_overrides_environment_without_mutating_it():
    env = {"JEVANY_DTYPE": "bfloat16", "JEVANY_MERGE": "0"}
    before = dict(env)
    options = load_options_from_args(argparse.Namespace(dtype="bf16"), MappingProxyType(env))
    assert options == LoadOptions(dtype=torch.bfloat16, merge=False)
    assert env == before


@pytest.mark.parametrize("lora_scale", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("from_environment", [False, True])
def test_load_options_reject_nonfinite_lora_scale(lora_scale, from_environment):
    with pytest.raises(ValueError, match="lora_scale must be finite"):
        if from_environment:
            LoadOptions.from_env({"JEVANY_LORA_SCALE": str(lora_scale)})
        else:
            LoadOptions(lora_scale=lora_scale)


@pytest.mark.parametrize("lora_scale", [-1.0, 0.0, 0.5, 1.0, 2.0])
def test_load_options_accept_finite_lora_scale(lora_scale):
    assert LoadOptions(lora_scale=lora_scale).lora_scale == lora_scale
    assert LoadOptions.from_env({"JEVANY_LORA_SCALE": str(lora_scale)}).lora_scale == lora_scale


@pytest.mark.parametrize("lora_scale", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("readout", ["native", "choice"])
def test_nonfinite_lora_scale_fails_before_checkpoint_resolution(monkeypatch, lora_scale, readout):
    def unexpected_resolve(*args, **kwargs):
        pytest.fail("invalid lora_scale reached checkpoint resolution")

    monkeypatch.setattr("jevany.checkpoint.resolve_run", unexpected_resolve)
    monkeypatch.setenv("JEVANY_LORA_SCALE", lora_scale)
    with pytest.raises(ValueError, match="lora_scale must be finite"):
        JevModel.from_pretrained("unused/checkpoint", device="cpu", readout=readout)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("from_environment", [False, True])
def test_load_options_reject_invalid_temperature(temperature, from_environment):
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        if from_environment:
            LoadOptions.from_env({"JEVANY_TEMPERATURE": str(temperature)})
        else:
            LoadOptions(temperature=temperature)


@pytest.mark.parametrize("temperature", [None, 0.5, 1.0, 2.0])
def test_load_options_accept_valid_temperature(temperature):
    assert LoadOptions(temperature=temperature).temperature == temperature
    value = "" if temperature is None else str(temperature)
    assert LoadOptions.from_env({"JEVANY_TEMPERATURE": value}).temperature == temperature


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("decision_mode", ["pointer", "lm_token"])
def test_checkpoint_rejects_invalid_temperature_before_loading_assets(tmp_path, monkeypatch,
                                                                    temperature, decision_mode):
    from jevany import checkpoint

    checkpoint.write_meta(tmp_path, checkpoint.Meta(
        base="unused", temperature=temperature, decision_mode=decision_mode,
    ))
    saved = checkpoint.Checkpoint(tmp_path)

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid temperature reached model asset loading")

    monkeypatch.setattr(checkpoint, "load_preprocessor", unexpected_load)
    monkeypatch.setattr(checkpoint, "get_backbone_adapter", unexpected_load)
    monkeypatch.setattr(checkpoint, "DecisionModel", unexpected_load)
    monkeypatch.setattr(saved, "warm_start", unexpected_load)
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        saved.load("cpu")


@pytest.mark.parametrize("value", ["bad", "0", "-1", "4096"])
def test_graph_token_flag_overrides_environment_before_validation(value):
    env = {
        "JEVANY_CUDA_GRAPH_MAX_TOKENS": value, "JEVANY_CUDA_GRAPHS": "0",
        "JEVANY_FUSED_KERNELS": "1", "JEVANY_DTYPE": "bf16", "JEVANY_MERGE": "0",
        "JEVANY_LORA_SCALE": "0.5",
    }
    before = dict(env)
    options = load_options_from_args(argparse.Namespace(cuda_graph_max_tokens=1024), MappingProxyType(env))
    assert options == LoadOptions(
        cuda_graphs=True, cuda_graph_max_tokens=1024, fused_kernels=True,
        dtype=torch.bfloat16, merge=False, lora_scale=0.5,
    )
    assert env == before


def test_graph_token_flag_accepts_minimum_limit():
    assert load_options_from_args(argparse.Namespace(cuda_graph_max_tokens=1), {}) == LoadOptions(
        cuda_graphs=True, cuda_graph_max_tokens=1,
    )


@pytest.mark.parametrize("flag", ["cuda_graphs", "fused_kernels"])
@pytest.mark.parametrize("value", ["1", "0", "true", ""])
def test_acceleration_boolean_flags_preserve_other_environment_settings(flag, value):
    args = argparse.Namespace(cuda_graphs=False, fused_kernels=False)
    setattr(args, flag, True)
    env = {"JEVANY_CUDA_GRAPHS": value, "JEVANY_FUSED_KERNELS": value,
           "JEVANY_CUDA_GRAPH_MAX_TOKENS": "512"}
    expected = {"cuda_graphs": value == "1", "fused_kernels": value == "1", flag: True}
    assert load_options_from_args(args, env) == LoadOptions(cuda_graph_max_tokens=512, **expected)


@pytest.mark.parametrize("env", [
    {}, {"JEVANY_CUDA_GRAPH_MAX_TOKENS": "bad"},
    {"JEVANY_CUDA_GRAPHS": "1", "JEVANY_FUSED_KERNELS": "1", "JEVANY_CUDA_GRAPH_MAX_TOKENS": "512"},
])
def test_loading_flags_preserve_no_override_sentinel(env):
    assert load_options_from_args(argparse.Namespace(), env) is None
    args = argparse.Namespace(device_map=None, max_memory_gib=None, cuda_graphs=False,
                              cuda_graph_max_tokens=None, fused_kernels=False)
    assert load_options_from_args(args, env) is None


@pytest.mark.parametrize("env,args,message", [
    ({"JEVANY_CUDA_GRAPH_MAX_TOKENS": "bad"}, {"cuda_graphs": True}, "invalid literal"),
    ({"JEVANY_CUDA_GRAPH_MAX_TOKENS": "0"}, {"fused_kernels": True}, "positive integer"),
    ({"JEVANY_CUDA_GRAPH_MAX_TOKENS": "-1"}, {"cuda_graphs": True}, "positive integer"),
    ({}, {"cuda_graph_max_tokens": 0}, "positive integer"),
    ({"JEVANY_CUDA_GRAPH_MAX_TOKENS": "512"}, {"cuda_graph_max_tokens": -1}, "positive integer"),
    ({"JEVANY_LORA_SCALE": "bad"}, {"cuda_graph_max_tokens": 512}, "could not convert string to float"),
    ({"JEVANY_MAX_MEMORY_GIB": "10"}, {"cuda_graphs": True}, "max_memory_gib requires device_map"),
    ({"JEVANY_COMPILE": "fastest"}, {"cuda_graphs": True}, "JEVANY_COMPILE"),
    ({"JEVANY_COMPILE": "1"}, {"cuda_graph_max_tokens": 512}, "enable one"),
    ({"JEVANY_COMPILE": "1"}, {"fused_kernels": True}, "cannot be combined"),
    ({"JEVANY_DEVICE_MAP": "auto"}, {"cuda_graphs": True}, "one GPU"),
    ({}, {"device_map": "auto", "fused_kernels": True}, "one GPU"),
    ({"JEVANY_CUDA_GRAPH_MAX_TOKENS": "bad"},
     {"device_map": "auto", "cuda_graph_max_tokens": 512}, "one GPU"),
])
def test_loading_flags_validate_final_combined_settings(env, args, message):
    with pytest.raises(ValueError, match=message):
        load_options_from_args(argparse.Namespace(**args), env)


class LoaderReached(Exception):
    pass


@pytest.fixture(params=["decide", "serve", "benchmark"])
def loading_entrypoint(request, monkeypatch, tmp_path):
    from jevany.cli import main

    for variable in tuple(os.environ):
        if variable.startswith("JEVANY_"):
            monkeypatch.delenv(variable)
    command = request.param
    seen = {}

    def load(*args, **kwargs):
        seen["options"] = args[2] if command == "benchmark" else kwargs["options"]
        raise LoaderReached

    if command == "benchmark":
        from scripts import benchmark_latency

        monkeypatch.setattr(benchmark_latency, "LocalPredictor", load)
        monkeypatch.setattr(benchmark_latency, "load_split", lambda *args: [{"_meta": {"id": "row"}}])
        flags = ["--run", "owner/checkpoint", "--suite", "unused", "--out", str(tmp_path / "latency.json"),
                 "--device", "cpu"]
        run = benchmark_latency.main
    else:
        flags = [command, "--checkpoint", "owner/checkpoint", "--device", "cpu"]
        run = main
        if command == "decide":
            monkeypatch.setattr(JevModel, "from_pretrained", load)
            body = SystemOneRequest(state="state", questions={"q": Choice(criteria={"a": None, "b": None})})
            monkeypatch.setattr("sys.stdin", io.StringIO(body.model_dump_json()))
            flags.insert(1, "-")
        else:
            monkeypatch.setattr("jevany.serve.create_app", load)
    return SimpleNamespace(command=command, seen=seen, run=lambda extra: run([*flags, *extra]))


@pytest.mark.parametrize("loading_entrypoint", ["decide", "serve"], indirect=True)
@pytest.mark.parametrize("acceleration", [False, True])
def test_entrypoint_dtype_overrides_invalid_environment(loading_entrypoint, monkeypatch, acceleration):
    monkeypatch.setenv("JEVANY_DTYPE", "bfloat16")
    flags = ["--dtype", "bf16"]
    if acceleration:
        flags += ["--cuda-graph-max-tokens", "512"]
    with pytest.raises(LoaderReached):
        loading_entrypoint.run(flags)
    assert loading_entrypoint.seen["options"] == LoadOptions(
        dtype=torch.bfloat16, cuda_graphs=acceleration,
        cuda_graph_max_tokens=512 if acceleration else 2048,
    )
    assert os.environ["JEVANY_DTYPE"] == "bfloat16"


@pytest.mark.parametrize("value", ["bad", "0", "-1", "4096"])
def test_entrypoint_graph_token_override_reaches_loader(loading_entrypoint, monkeypatch, value):
    monkeypatch.setenv("JEVANY_CUDA_GRAPH_MAX_TOKENS", value)
    monkeypatch.setenv("JEVANY_CUDA_GRAPHS", "0")
    monkeypatch.setenv("JEVANY_MERGE", "0")
    with pytest.raises(LoaderReached):
        loading_entrypoint.run(["--cuda-graph-max-tokens", "1024"])
    assert loading_entrypoint.seen["options"] == LoadOptions(
        cuda_graphs=True, cuda_graph_max_tokens=1024, merge=False,
    )
    assert os.environ["JEVANY_CUDA_GRAPH_MAX_TOKENS"] == value
    assert os.environ["JEVANY_CUDA_GRAPHS"] == "0"


@pytest.mark.parametrize("flags,expected", [
    (["--cuda-graphs"], LoadOptions(cuda_graphs=True, cuda_graph_max_tokens=512)),
    (["--fused-kernels"], LoadOptions(fused_kernels=True, cuda_graph_max_tokens=512)),
    (["--cuda-graphs", "--fused-kernels"],
     LoadOptions(cuda_graphs=True, fused_kernels=True, cuda_graph_max_tokens=512)),
])
def test_entrypoint_boolean_flags_reach_loader(loading_entrypoint, monkeypatch, flags, expected):
    monkeypatch.setenv("JEVANY_CUDA_GRAPH_MAX_TOKENS", "512")
    monkeypatch.setenv("JEVANY_CUDA_GRAPHS", "0")
    monkeypatch.setenv("JEVANY_FUSED_KERNELS", "0")
    with pytest.raises(LoaderReached):
        loading_entrypoint.run(flags)
    assert loading_entrypoint.seen["options"] == expected


@pytest.mark.parametrize("env", [
    {},
    {"JEVANY_CUDA_GRAPHS": "1", "JEVANY_FUSED_KERNELS": "1", "JEVANY_CUDA_GRAPH_MAX_TOKENS": "512"},
])
def test_entrypoint_without_loading_flags_preserves_fallback(loading_entrypoint, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(LoaderReached):
        loading_entrypoint.run([])
    expected = LoadOptions.from_env(env) if loading_entrypoint.command == "benchmark" else None
    assert loading_entrypoint.seen["options"] == expected


@pytest.mark.parametrize("env,flags,message", [
    ({}, ["--cuda-graph-max-tokens", "0"], "positive integer"),
    ({}, ["--cuda-graph-max-tokens", "-1"], "positive integer"),
    ({"JEVANY_CUDA_GRAPH_MAX_TOKENS": "bad"}, ["--cuda-graphs"], "invalid literal"),
    ({"JEVANY_COMPILE": "1"}, ["--cuda-graph-max-tokens", "512"], "enable one"),
    ({"JEVANY_DEVICE_MAP": "auto"}, ["--cuda-graph-max-tokens", "512"], "one GPU"),
    ({"JEVANY_COMPILE": "1"}, ["--fused-kernels"], "cannot be combined"),
])
def test_entrypoint_rejects_invalid_final_settings(loading_entrypoint, monkeypatch, capsys, env, flags, message):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    error_type = ValueError if loading_entrypoint.command == "benchmark" else SystemExit
    with pytest.raises(error_type) as error:
        loading_entrypoint.run(flags)
    if error_type is SystemExit:
        assert error.value.code == 2
        assert message in capsys.readouterr().err
    else:
        assert message in str(error.value)
    assert not loading_entrypoint.seen


def test_entrypoint_rejects_malformed_graph_token_flag(loading_entrypoint, capsys):
    with pytest.raises(SystemExit) as error:
        loading_entrypoint.run(["--cuda-graph-max-tokens", "bad"])
    assert error.value.code == 2
    assert "invalid int value" in capsys.readouterr().err
    assert not loading_entrypoint.seen


@pytest.mark.parametrize("flags", [
    ["--cuda-graphs"], ["--cuda-graph-max-tokens", "512"], ["--fused-kernels"],
])
def test_entrypoint_choice_readout_rejects_acceleration(loading_entrypoint, capsys, flags):
    with pytest.raises(SystemExit) as error:
        loading_entrypoint.run(["--readout", "choice", *flags])
    assert error.value.code == 2
    assert "native" in capsys.readouterr().err
    assert not loading_entrypoint.seen
