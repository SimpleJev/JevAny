"""Native precision, gradient parity and the adapter's forward/cache contract."""
import pytest
import torch

from jevany.backbones import BackboneAdapter
from jevany.model import DecisionModel, load_tokenizer
from test_backbones import RECORD, make_base


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class RecordingAdapter(BackboneAdapter):
    def __init__(self):
        self.calls = []

    def forward(self, model, **inputs):
        output = super().forward(model, **inputs)
        self.calls.append((bool(inputs.get("use_cache")), output.past_key_values is not None,
                           output.last_hidden_state.dtype))
        return output


@pytest.mark.parametrize("branch_mode", ["packed", "rows"])
def test_scoring_and_prefix_reuse_go_through_adapter(tmp_path, branch_mode):
    base = tmp_path / "base"
    make_base(base, "llama")
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8, branch_mode=branch_mode)
    model.adapter = RecordingAdapter()
    model.eval()
    encoded = model.encode(tokenizer, RECORD)
    # The base enables caching by default; ordinary decision scoring opts out.
    assert model.lm.config.use_cache
    expected = model.probs(encoded)
    assert model.adapter.calls == [(False, False, torch.float32)]
    first, prefix = model.probs_and_prefix(encoded)
    for _ in range(2):
        reused = model.probs_with_prefix(encoded, prefix)
        for direct, miss, hit in zip(expected, first, reused):
            torch.testing.assert_close(direct, miss, atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(direct, hit, atol=2e-6, rtol=1e-5)
    assert all(requested and returned for requested, returned, _ in model.adapter.calls[1:])


@pytest.mark.parametrize("branch_mode", ["packed", "rows"])
def test_bf16_readout_and_gradients_match_full_hidden_upcast(tmp_path, branch_mode):
    torch.manual_seed(17)
    base = tmp_path / "base"
    make_base(base, "llama")
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8, lora_dropout=0,
                          dtype=torch.bfloat16, branch_mode=branch_mode)
    model.adapter = RecordingAdapter()
    model.train()
    encoded = model.encode(tokenizer, RECORD)
    logits = model(encoded)
    sum(value.square().sum() for value in logits).backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    assert model.adapter.calls == [(False, False, torch.bfloat16)]
    if branch_mode == "packed":
        with torch.no_grad():
            assert model.hidden_batch([encoded]).dtype == torch.bfloat16
    model.zero_grad(set_to_none=True)

    # Emulate the old full-sequence FP32 conversion using the same frozen base,
    # adapter weights and head, then compare both outputs and backpropagation.
    def full_upcast(module, args, output):
        output.last_hidden_state = output.last_hidden_state.float()
        return output

    with model.lm.register_forward_hook(full_upcast):
        reference = model(encoded)
        sum(value.square().sum() for value in reference).backward()
    for actual, expected in zip(logits, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=0, atol=0)
