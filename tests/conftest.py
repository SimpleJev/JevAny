"""Shared, offline model artifacts for training and serving integration tests."""
import pytest


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from jevany.datasets import init_starter
    from jevany.model import SPECIAL
    from jevany.train import main

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("framework")
    base = root / "base"
    vocabulary = {token: index for index, token in enumerate(
        ["[UNK]", "[PAD]", "[EOS]"] + SPECIAL + ["yes", "no", "choose", "a", "b", "state"])}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                  pad_token="[PAD]", eos_token="[EOS]", additional_special_tokens=SPECIAL)
    fast.save_pretrained(base)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=1, eos_token_id=2,
    ))
    model.save_pretrained(base)
    data = init_starter(root / "data") / "train.jsonl"
    args = ["--base", str(base), "--data", str(data), "--device", "cpu", "--lora", "2",
            "--head-dim", "8", "--lora-targets", "qv", "--max-steps", "2", "--accum", "1",
            "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0"]
    sft = main(args + ["--out", str(root / "sft")])
    rlcr = main(args + ["--out", str(root / "rlcr"), "--init-from", str(sft), "--rlcr"])
    yield root, sft, rlcr
    torch.set_num_threads(old_threads)
