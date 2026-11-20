"""Fast coverage for pointer and direct-token readouts without model weights."""
from types import SimpleNamespace

import pytest
import torch

from jevany.api import MAX_OPTIONS, SystemOneRequest
from jevany.checkpoint import Meta
from jevany.data import materialize
from jevany.model import SPECIAL, DecisionModel, PointerHead
from jevany.train import encode_batch, lm_token_loss


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


@pytest.fixture
def training_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    tokens = ["[UNK]", "state", ":"] + SPECIAL + [
        token for i in range(256) for token in (f"o{i}", f"V{i}")
    ]
    tokenizer = Tokenizer(WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", additional_special_tokens=SPECIAL,
    )


@pytest.mark.parametrize(("mode", "limit", "options"), [
    ("lm_token", 255, 255),
    ("lm_token", 255, 254),
    ("lm_token", 3, 3),
    ("lm_token", 4, 3),
    ("pointer", MAX_OPTIONS, 255),
])
@pytest.mark.parametrize("augmentation", ["p_none_pair", "p_none_distract", "p_distract"])
def test_training_augmentations_respect_model_option_limit(training_tokenizer, mode, limit, options, augmentation):
    model = direct_model()
    model.decision_mode = mode
    model.verbalizers = [f"V{i}" for i in range(limit)] if mode == "lm_token" else []
    request = {
        "state": "state",
        "questions": {"q": {
            "type": "choice", "criteria": {f"o{i}": None for i in range(options)}, "label": "o0",
        }},
        "_meta": {"id": "training/0"},
    }
    args = SimpleNamespace(
        seed=0, p_none=0, p_none_pair=0, p_none_distract=0, p_distract=0,
        max_state=16, max_branch=2048, max_packed=2048,
    )
    setattr(args, augmentation, 1)
    model.encode(training_tokenizer, materialize(request), max_state=16, max_branch=2048, strict=True)

    variants = encode_batch(model, training_tokenizer, args, [request], epoch=0)

    expected = (
        [options, options + 1, options] if options < limit else [options]
    ) if augmentation == "p_none_pair" else [min(options + 1, limit)]
    assert [len(variant.rec["questions"][0]["options"]) for variant in variants] == expected
    assert [len(variant.enc["opt_idx"][0]) for variant in variants] == expected


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
