"""CUDA-graph replay for row-mode inference: option plumbing everywhere, parity with the eager path on CUDA."""
import pytest
import torch

from jevany.checkpoint import LoadOptions
from jevany.cudagraphs import RowGraphs
from jevany.model import DecisionModel, load_tokenizer
from test_backbones import RECORD, make_base

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_cuda_graph_option_from_env():
    assert not LoadOptions.from_env({}).cuda_graphs
    configured = LoadOptions.from_env({"JEVANY_CUDA_GRAPHS": "1", "JEVANY_CUDA_GRAPH_MAX_TOKENS": "1024"})
    assert configured.cuda_graphs and configured.cuda_graph_max_tokens == 1024
    assert not LoadOptions.from_env({"JEVANY_CUDA_GRAPHS": "0"}).cuda_graphs
    with pytest.raises(ValueError, match="enable one"):
        LoadOptions(cuda_graphs=True, compile_mode="reduce-overhead")
    with pytest.raises(ValueError, match="positive integer"):
        LoadOptions(cuda_graph_max_tokens=0)


def test_cuda_graphs_require_one_cuda_device(tmp_path):
    base = tmp_path / "base"
    make_base(base, "qwen35", legacy=True)
    model = DecisionModel(base, load_tokenizer(base), "cpu", lora=2, head_dim=8)
    assert model.branch_mode == "rows"
    with pytest.raises(ValueError, match="one CUDA device"):
        RowGraphs(model)


def test_cuda_graphs_reject_packed_backbones(tmp_path):
    base = tmp_path / "base"
    make_base(base, "llama")
    model = DecisionModel(base, load_tokenizer(base), "cpu", lora=2, head_dim=8)
    assert model.branch_mode == "packed"
    with pytest.raises(ValueError, match="row-mode"):
        RowGraphs(model)


def test_cuda_graphs_reject_trainable_token_embeddings(tmp_path):
    base = tmp_path / "base"
    make_base(base, "qwen35")
    model = DecisionModel(base, load_tokenizer(base), "cpu", lora=2, head_dim=8)
    assert model.special_embeddings
    with pytest.raises(ValueError, match="trainable token embeddings"):
        RowGraphs(model)


def test_remote_decide_rejects_cuda_graphs(tmp_path, capsys):
    from jevany import Choice, SystemOneRequest
    from jevany.cli import main
    request = tmp_path / "request.json"
    request.write_text(SystemOneRequest(state="state", questions={
        "q": Choice(instructions="choose", criteria={"a": None, "b": None})}).model_dump_json())
    with pytest.raises(SystemExit):
        main(["decide", str(request), "--cuda-graphs"])
    assert "require --checkpoint" in capsys.readouterr().err


ONE_QUESTION = {**RECORD, "questions": RECORD["questions"][:1]}


@cuda
@pytest.mark.parametrize("family,dtype,decision_mode", [
    ("qwen35", torch.float32, "pointer"), ("qwen35", torch.bfloat16, "pointer"),
    ("gemma", torch.float32, "pointer"), ("gpt2", torch.float32, "lm_token"),
])
def test_graph_replay_matches_eager(tmp_path, family, dtype, decision_mode):
    base = tmp_path / "base"
    make_base(base, family, legacy=True)
    tokenizer = load_tokenizer(base)
    torch.manual_seed(0)
    model = DecisionModel(base, tokenizer, "cuda", lora=2, head_dim=8, dtype=dtype, decision_mode=decision_mode,
                          verbalizers=["yes", "no"] if decision_mode == "lm_token" else None).eval()
    assert model.branch_mode == "rows"
    encoded = model.encode(tokenizer, ONE_QUESTION)
    expected = model.probs(encoded)
    model.cuda_graphs = RowGraphs(model, lengths=(16, 32, 64)).capture()
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    for _ in range(2):   # the second replay refills the same static inputs
        for left, right in zip(expected, model.probs(encoded)):
            torch.testing.assert_close(left, right, atol=tolerance, rtol=0)
    assert (model.cuda_graphs.stats["graph_calls"], model.cuda_graphs.stats["eager_calls"]) == (2, 0)


@cuda
def test_multi_question_and_long_rows_run_eagerly(tmp_path):
    base = tmp_path / "base"
    make_base(base, "qwen35", legacy=True)
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cuda", lora=2, head_dim=8).eval()
    several, one = model.encode(tokenizer, RECORD), model.encode(tokenizer, ONE_QUESTION)
    expected = model.probs(several), model.probs(one)
    model.cuda_graphs = RowGraphs(model, lengths=(8,)).capture()   # shorter than any row
    for encoded, reference in zip((several, one), expected):
        for left, right in zip(reference, model.probs(encoded)):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=0)
    assert (model.cuda_graphs.stats["graph_calls"], model.cuda_graphs.stats["eager_calls"]) == (0, 2)


@cuda
def test_checkpoint_option_captures_and_runtime_reports(tmp_path):
    from jevany import Choice, JevModel, SystemOneRequest
    from test_serving import make_checkpoint
    checkpoint = make_checkpoint(tmp_path, "qwen35", legacy=True)
    eager = JevModel.from_pretrained(checkpoint, device="cuda")
    graphs = JevModel.from_pretrained(
        checkpoint, device="cuda", options=LoadOptions(cuda_graphs=True, cuda_graph_max_tokens=192))
    request = SystemOneRequest(state="state " * 20, questions={
        "choice": Choice(instructions="choose", criteria={"a": None, "b": None})})
    expected, actual = eager(request), graphs(request)
    assert actual["answers"]["choice"]["probabilities"] == pytest.approx(
        expected["answers"]["choice"]["probabilities"], abs=1e-5)
    stats = graphs.describe()["acceleration"]["cuda_graphs"]
    assert stats["buckets"] == [128, 192]
    assert (stats["graph_calls"], stats["eager_calls"]) == (1, 0)
    assert eager.describe()["acceleration"]["cuda_graphs"] is None
