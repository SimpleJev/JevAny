"""Fused Qwen3.5 inference kernels: option plumbing and weight sharing on CPU; parity with the unfused path on CUDA
with flash-linear-attention installed."""
import importlib.util

import pytest
import torch
from torch import nn

from jevany.checkpoint import Checkpoint, LoadOptions
from jevany.fused_kernels import _fuse, _transpose_storage, fuse_qwen3_5
from jevany.model import DecisionModel, load_tokenizer
from test_backbones import make_base

fla = pytest.mark.skipif(not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
                         reason="needs CUDA and flash-linear-attention")


def test_fused_kernel_option_from_env():
    assert not LoadOptions.from_env({}).fused_kernels
    assert LoadOptions.from_env({"JEVANY_FUSED_KERNELS": "1"}).fused_kernels
    assert not LoadOptions.from_env({"JEVANY_FUSED_KERNELS": "0"}).fused_kernels
    with pytest.raises(ValueError, match="cannot be combined"):
        LoadOptions(fused_kernels=True, compile_mode="default")
    with pytest.raises(ValueError, match="one GPU"):
        LoadOptions(fused_kernels=True, device_map="auto")


def test_fused_projections_share_one_transposed_matrix():
    torch.manual_seed(0)
    linears = [nn.Linear(6, size, bias=False) for size in (4, 3, 5)]
    x = torch.randn(2, 6)
    expected = [linear(x) for linear in linears]
    fused = _fuse(linears)
    assert fused.shape == (6, 12) and fused.is_contiguous()
    torch.testing.assert_close(x @ fused, torch.cat(expected, dim=-1))
    for linear, reference in zip(linears, expected):
        assert linear.weight.shape == (linear.out_features, 6)
        assert linear.weight.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
        torch.testing.assert_close(linear(x), reference)


def test_transposed_storage_keeps_linear_outputs():
    torch.manual_seed(0)
    linear, x = nn.Linear(6, 4, bias=False), torch.randn(3, 6)
    expected = linear(x)
    _transpose_storage(linear)
    assert linear.weight.shape == (4, 6) and linear.weight.t().is_contiguous()
    torch.testing.assert_close(linear(x), expected)


@pytest.mark.parametrize("family,message", [("qwen35", "one CUDA device"), ("llama", "Qwen3.5")])
def test_fusion_checks_backbone_and_device(tmp_path, family, message):
    base = tmp_path / "base"
    make_base(base, family, legacy=True)
    model = DecisionModel(base, load_tokenizer(base), "cpu", lora=2, head_dim=8)
    with pytest.raises(ValueError, match=message):
        fuse_qwen3_5(model.lm)


def test_checkpoint_and_choice_readout_reject_unsupported_fused_kernels(tmp_path):
    from jevany import JevModel
    from test_serving import make_checkpoint
    checkpoint = make_checkpoint(tmp_path, "qwen35", legacy=True)
    with pytest.raises(ValueError, match="CUDA only"):
        Checkpoint(checkpoint).load("cpu", LoadOptions(fused_kernels=True))
    with pytest.raises(ValueError, match="native readout"):
        JevModel.from_pretrained(checkpoint, device="cpu", options=LoadOptions(fused_kernels=True), readout="choice")


def test_remote_decide_rejects_fused_kernels(tmp_path, capsys):
    from jevany import Choice, SystemOneRequest
    from jevany.cli import main
    request = tmp_path / "request.json"
    request.write_text(SystemOneRequest(state="state", questions={
        "q": Choice(instructions="choose", criteria={"a": None, "b": None})}).model_dump_json())
    with pytest.raises(SystemExit):
        main(["decide", str(request), "--fused-kernels"])
    assert "require --checkpoint" in capsys.readouterr().err


def _qwen35_backbone(dtype):
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=64, layer_types=["linear_attention"] * 3 + ["full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
        max_position_embeddings=1024)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3_5TextModel(config)
    with torch.no_grad():   # norm weights start at their identity values; exercise the scales too
        for module in model.modules():
            if type(module).__name__ in ("Qwen3_5RMSNorm", "Qwen3_5RMSNormGated"):
                module.weight.add_(0.1 * torch.randn_like(module.weight))
    return model.to("cuda", dtype).eval()


@fla
@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 1e-3), (torch.bfloat16, 3e-2)])
def test_fused_backbone_matches_transformers(dtype, tolerance):
    model = _qwen35_backbone(dtype)
    ids = torch.randint(0, 128, (1, 200), device="cuda")
    positions = torch.arange(200, device="cuda")[None]
    with torch.no_grad():
        expected = model(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state.float()
        counts = fuse_qwen3_5(model)
        actual = model(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state.float()
    assert {key: counts[key] for key in ("rms_norm", "mlp", "gated_deltanet", "attention")} == {
        "rms_norm": 11, "mlp": 4, "gated_deltanet": 3, "attention": 1}
    error = ((actual - expected).norm() / expected.norm()).item()
    print(f"relative error {dtype}: {error:.2e}")
    assert error < tolerance


@fla
def test_checkpoint_fused_kernels_replay_and_report(tmp_path):
    from jevany import Choice, JevModel, SystemOneRequest
    from test_serving import make_checkpoint
    checkpoint = make_checkpoint(tmp_path, "qwen35", legacy=True)
    request = SystemOneRequest(state="state " * 20, questions={
        "choice": Choice(instructions="choose", criteria={"a": None, "b": None})})
    plain = JevModel.from_pretrained(checkpoint, device="cuda", options=LoadOptions(cuda_graphs=True))
    fused = JevModel.from_pretrained(checkpoint, device="cuda",
                                     options=LoadOptions(cuda_graphs=True, fused_kernels=True))
    expected, actual = plain(request), fused(request)
    assert actual["answers"]["choice"]["probabilities"] == pytest.approx(
        expected["answers"]["choice"]["probabilities"], abs=1e-3)
    acceleration = fused.describe()["acceleration"]
    assert acceleration["fused_kernels"]["gated_deltanet"] == 1 and acceleration["fused_kernels"]["attention"] == 1
    assert acceleration["cuda_graphs"]["graph_calls"] == 1
    assert plain.describe()["acceleration"]["fused_kernels"] is None
