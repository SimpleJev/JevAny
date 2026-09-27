from types import SimpleNamespace

import torch

from jevany.api import to_answers
from jevany.checkpoint import Meta
from jevany.model import PointerHead


def test_noul_response_keeps_probability_precision():
    probability = 0.731234567
    metadata = [{"id": "q", "type": "noul", "keys": ["false", "true"]}]

    assert to_answers([[1 - probability, probability]], metadata)["q"]["noul"] == probability


def test_checkpoint_metadata_describes_final_head_and_readouts():
    metadata = Meta.from_dict({
        "base": "Qwen/Qwen3.5-4B",
        "head": {},
        "lora": 32,
        "head_dim": 256,
        "head_residual_dim": 256,
        "head_type": "residual",
        "query_readout": "question_mean",
        "option_readout": "close",
    })

    assert metadata.lora == 32
    assert metadata.head_type == "residual"
    assert metadata.head_residual_dim == 256
    assert metadata.query_readout == "question_mean"
    assert metadata.option_readout == "close"


def test_residual_pointer_head_has_zero_initialized_residual_output():
    head = PointerHead(32, dp=16, residual_dim=8, head_type="residual")

    assert head.head_type == "residual"
    assert head.residual is not None
    assert torch.count_nonzero(head.residual[2].weight) == 0
    assert torch.count_nonzero(head.residual[2].bias) == 0


def test_strict_models_endpoint_redacts_local_paths(monkeypatch):
    from jevany import serve

    local_checkpoint = "/private/checkpoints/step-000358"
    fake = SimpleNamespace(
        checkpoint=SimpleNamespace(
            requested=local_checkpoint,
            meta=SimpleNamespace(base="/private/models/Qwen3.5-4B", lora=32),
        ),
        model=SimpleNamespace(head=SimpleNamespace(temperature=0.7)),
        model_id="jevany-4b-final",
        model_aliases=(),
        strict_model_id=True,
        provenance={"checkpoint_source": "local"},
        device="cuda",
        prefix_hits=0,
        prefix_misses=0,
        prefix_cache={},
    )
    monkeypatch.setattr(serve.app.state, "server", fake, raising=False)

    row = serve.models()["models"][0]
    assert row["run"] == "jevany-4b-final"
    assert row["base"] == "Qwen3.5-4B"
    assert local_checkpoint not in str(row)
