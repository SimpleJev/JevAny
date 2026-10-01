"""Fast coverage for pointer and direct-token readouts without model weights."""
from types import SimpleNamespace

import pytest
import torch

from jevany.api import SystemOneRequest
from jevany.checkpoint import Meta
from jevany.model import DecisionModel, PointerHead
from jevany.train import lm_token_loss


def direct_model(vocabulary_size=8):
    model = DecisionModel.__new__(DecisionModel)
    torch.nn.Module.__init__(model)
    model.decision_mode = "lm_token"
    model.device = "cpu"
    model.verbalizers = ["A", "B", "C"]
    model.verbalizer_ids = [1, 4, 6]
    model.register_buffer("_verbalizer_index", torch.tensor(model.verbalizer_ids), persistent=False)
    model.temperature = 1.0
    model.lm_head = torch.nn.Linear(3, vocabulary_size, bias=False)
    model.option_isolation = False
    model.multimodal = False
    model.mm = None
    model.adapter = SimpleNamespace(frozen_modules=lambda _: [])
    return model


def test_lm_token_training_uses_full_vocabulary_logits():
    model = direct_model()
    model.temperature = 1.7
    model.train()
    hidden = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    logits = model._question_readout(hidden, 0, [0, 1])
    assert logits.shape == (8,)
    expected = torch.nn.functional.linear(hidden[0], model.lm_head.weight).float()
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    actual_gradient, = torch.autograd.grad(logits.square().sum(), hidden, retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected.square().sum(), hidden)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_lm_token_eval_selects_and_calibrates_candidates():
    model = direct_model()
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.arange(24).reshape(8, 3))
    model.temperature = 2.0
    model.eval()
    logits = model._question_readout(torch.tensor([[1.0, 0.0, 0.0]]), 0, [0, 1])
    assert torch.equal(logits, torch.tensor([1.5, 6.0]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("options", [1, 2, 3])
def test_candidate_projection_matches_full_vocabulary_readout(dtype, options):
    torch.manual_seed(17)
    model = direct_model(vocabulary_size=1024)
    model.to(dtype)
    model.temperature = 1.7
    model.eval()
    hidden = torch.randn(4, 3, dtype=dtype)
    full = torch.nn.functional.linear(hidden[2], model.lm_head.weight).float()
    expected = full[model.verbalizer_ids[:options]] / model.temperature
    actual = model._question_readout(hidden, 2, list(range(options)))
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_lm_token_ce_has_full_vocabulary_denominator():
    logits = torch.tensor([0.0, 2.0, -1.0, 0.5, 1.0, -2.0, 3.0])
    question = {"options": ["a", "b", "c"], "label": 2,
                "target": [0.25, 0.25, 0.5]}
    loss = lm_token_loss(logits, question, [1, 4, 6], "cpu")
    expected = -(torch.tensor(question["target"]) *
                 torch.log_softmax(logits, -1)[[1, 4, 6]]).sum()
    assert torch.allclose(loss, expected)


def test_lm_token_rejects_more_than_frozen_verbalizer_table(monkeypatch):
    model = direct_model()
    tokenizer = SimpleNamespace()
    record = {"state": "s", "questions": [{"instr": "choose",
                                              "options": [str(i) for i in range(4)],
                                              "label": 0}]}
    with pytest.raises(ValueError, match="at most 3 options"):
        model.encode(tokenizer, record)


def test_pointer_api_accepts_more_than_255_choices():
    request = SystemOneRequest.model_validate({
        "state": "s",
        "questions": {"q": {"type": "choice", "instructions": "choose",
                               "criteria": {str(i): None for i in range(256)}}},
    })
    assert len(request.questions["q"].criteria) == 256


def test_residual_pointer_starts_as_linear_pointer():
    torch.manual_seed(1)
    head = PointerHead(5, dp=3, residual_dim=4)
    decide, options = torch.randn(5), torch.randn(7, 5)
    expected = (head.k(options) @ head.q(decide)) * head.scale
    assert torch.equal(head(decide, options), expected)


def test_legacy_metadata_defaults_to_pointer():
    meta = Meta.from_dict({"base": "base"})
    assert meta.decision_mode == "pointer"
    assert meta.verbalizers == []
