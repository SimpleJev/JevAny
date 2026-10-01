"""Readout preserves logits/gradients while converting only selected token vectors."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from jevany.model import DecisionModel, PointerHead


class ConvertedShapes(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._to_copy.default:
            self.shapes.append(tuple(args[0].shape))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("source,target", [
    ("cpu", "cpu"),
    pytest.param("cuda:0", "cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA")),
    pytest.param("cuda:1", "cuda:0", marks=pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two CUDA devices")),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["pointer", "lm_token"])
@pytest.mark.parametrize("training", [False, True])
def test_selected_readout_matches_full_conversion(source, target, dtype, mode, training):
    torch.manual_seed(17)
    model = DecisionModel.__new__(DecisionModel)
    torch.nn.Module.__init__(model)
    model.decision_mode, model.device, model.temperature = mode, target, 1.7
    model.training = training
    if mode == "pointer":
        model.head = PointerHead(8, dp=4, residual_dim=4).to(target).train(training)
        model.head.temperature = model.temperature
        parameters = list(model.head.parameters())
    else:
        model.lm_head = torch.nn.Linear(8, 32, bias=False).to(device=target, dtype=dtype)
        model.register_buffer("_verbalizer_index", torch.tensor([2, 9, 17], device=target))
        parameters = list(model.lm_head.parameters())
    hidden = torch.randn(257, 8, device=source, dtype=dtype, requires_grad=True)
    decide, options = 128, [10, 13, 129]

    # Previous implementation transferred and converted the entire row first.
    full = hidden.to(target).float()
    if mode == "pointer":
        expected = model.head(full[decide], full[torch.tensor(options, device=target)])
    else:
        weight = model.lm_head.weight
        if not training:
            weight = weight.index_select(0, model._verbalizer_index)
        expected = torch.nn.functional.linear(full[decide].to(dtype), weight).float()
        if not training:
            expected = expected / model.temperature
    with ConvertedShapes() as converted:
        actual = model._question_readout(hidden, decide, options)
    assert tuple(hidden.shape) not in converted.shapes
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual_grad = torch.autograd.grad(actual.square().sum(), [hidden, *parameters])
    expected_grad = torch.autograd.grad(expected.square().sum(), [hidden, *parameters])
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize("branch_mode", ["packed", "rows"])
def test_backbone_hidden_stays_native_until_readout(tmp_path, monkeypatch, branch_mode):
    from test_backbones import make_base, RECORD
    from jevany.model import load_tokenizer

    base = tmp_path / "base"
    make_base(base, "llama", legacy=True)
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8,
                          dtype=torch.bfloat16, branch_mode=branch_mode).eval()
    encoded = model.encode(tokenizer, RECORD)
    native_dtypes = []
    readout = model._question_readout

    def observe(hidden, decide, options):
        native_dtypes.append(hidden.dtype)
        return readout(hidden, decide, options)

    monkeypatch.setattr(model, "_question_readout", observe)
    with torch.no_grad():
        model(encoded)
        # Callers explicitly requesting all hidden states retain the old API.
        full = model.hidden_batch([encoded])
        single = model.hidden(encoded)
    assert native_dtypes == [torch.bfloat16] * len(RECORD["questions"])
    assert full.dtype == single.dtype == torch.float32
    assert full.device == single.device == torch.device("cpu")
    torch.testing.assert_close(single, full[0, :len(encoded["ids"])], atol=0, rtol=0)
