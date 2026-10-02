import json
import math

import pytest

from jevany.letter_readout import (
    LETTERS,
    SYSTEM_PROMPT,
    build_prompt,
    letter_log_masses,
    letter_token_ids,
    read_letter_distribution,
    temper_probabilities,
)


class FakeTokenizer:
    def __init__(self, decoded_tokens):
        self.decoded_tokens = list(decoded_tokens)
        self.decoded_ids = []

    def __len__(self):
        return len(self.decoded_tokens)

    def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        (token_id,) = token_ids
        self.decoded_ids.append(token_id)
        return self.decoded_tokens[token_id]


def test_build_prompt_matches_cygnet_structured_rendering_and_option_order():
    state = {"order": {"id": "A-17", "paid_with": "gift card"}, "city": "Zürich"}
    options = {"refund": "Refund the card", "credit": "Issue store credit"}

    prompt = build_prompt(state, {"task": "Choose the remedy"}, options)

    expected = "\n".join([
        json.dumps(state, ensure_ascii=False, indent=1),
        "",
        json.dumps({"task": "Choose the remedy"}, ensure_ascii=False),
        "",
        "Options:",
        "A. Refund the card",
        "B. Issue store credit",
        "",
        "Answer with the letter of exactly one option, and nothing else:",
    ])
    assert prompt == expected
    assert prompt.index("A. Refund") < prompt.index("B. Issue")
    assert "thinking" not in prompt.lower()
    assert "LETTER and nothing else" in SYSTEM_PROMPT


def test_prompt_accepts_all_letters_and_rejects_invalid_option_counts():
    prompt = build_prompt("state", "choose", [f"option {index}" for index in range(26)])
    assert f"{LETTERS[-1]}. option 25" in prompt
    with pytest.raises(ValueError, match="at least one"):
        build_prompt("state", "choose", [])
    with pytest.raises(ValueError, match="at most 26"):
        build_prompt("state", "choose", list(range(27)))
    with pytest.raises(TypeError, match="ordered mapping"):
        build_prompt("state", "choose", "not an option sequence")


def test_mapping_options_keep_labels_for_blank_and_structured_descriptions():
    prompt = build_prompt(
        "state",
        "choose",
        {"calm": None, "angry": "  ", "review": {"owner": "ops", "priority": 2}},
    )

    assert "A. calm" in prompt
    assert "B. angry" in prompt
    assert 'C. review: {"owner": "ops", "priority": 2}' in prompt


def test_letter_token_ids_scans_full_vocab_and_preserves_exact_duplicate_ids():
    tokenizer = FakeTokenizer([
        "<bos>", "A", "A", " A", "A.", "a", "B", " B", "[b]", "AA", "The", "Ａ",
    ])

    aliases = letter_token_ids(tokenizer, ("A", "B"))

    assert aliases == {"A": (1, 2), "B": (6,)}
    assert tokenizer.decoded_ids == list(range(len(tokenizer)))


def test_letter_token_ids_uses_batch_decode_and_emulates_exact_choice_mask():
    class BatchTokenizer(FakeTokenizer):
        def __init__(self, decoded_tokens):
            super().__init__(decoded_tokens)
            self.batches = []

        def batch_decode(self, rows, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            self.batches.append(rows)
            return [self.decoded_tokens[row[0]] for row in rows]

    tokenizer = BatchTokenizer(["A", " A", "A.", "B", " b", "not a letter"])
    assert letter_token_ids(tokenizer, ("A", "B")) == {"A": (0,), "B": (3,)}
    assert tokenizer.batches == [[[0], [1], [2], [3], [4], [5]]]
    assert tokenizer.decoded_ids == []


def test_duplicate_token_aliases_all_contribute_via_stable_logsumexp():
    token_ids = {"A": (1, 2), "B": (3,)}
    # The absolute logits are deliberately large. Two distinct A token IDs,
    # even though they may decode to identical text, carry twice B's mass.
    logits = [-5000.0, 1000.0, 1000.0, 1000.0]

    masses = letter_log_masses(logits, token_ids)
    readout = read_letter_distribution(logits, token_ids, temperature=2.0)

    assert masses["A"] == pytest.approx(1000.0 + math.log(2.0))
    assert masses["B"] == pytest.approx(1000.0)
    assert readout.raw_log_masses == masses
    assert readout.raw_probabilities == pytest.approx({"A": 2 / 3, "B": 1 / 3})
    expected_a = math.sqrt(2 / 3) / (math.sqrt(2 / 3) + math.sqrt(1 / 3))
    assert readout.calibrated_probabilities == pytest.approx({"A": expected_a, "B": 1 - expected_a})
    assert readout.temperature == 2.0


def test_calibration_uses_log_masses_before_raw_softmax_underflow():
    readout = read_letter_distribution([0.0, -1000.0], {"A": (0,), "B": (1,)}, temperature=2.0)

    assert readout.raw_probabilities["B"] == 0.0
    assert readout.calibrated_probabilities["B"] > 0.0
    assert math.log(readout.calibrated_probabilities["B"]) == pytest.approx(-500.0)


def test_temperature_one_is_an_exact_identity_and_order_is_preserved():
    probabilities = {"C": 0.7, "A": 0.2, "B": 0.1}

    calibrated = temper_probabilities(probabilities, 1.0)

    assert calibrated == probabilities
    assert list(calibrated) == ["C", "A", "B"]
    assert calibrated is not probabilities


@pytest.mark.parametrize("temperature", [0, -1, math.nan, math.inf, -math.inf])
def test_temperature_must_be_strictly_positive_and_finite(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        temper_probabilities({"A": 0.5, "B": 0.5}, temperature)


@pytest.mark.parametrize("probabilities", [
    {},
    {"A": -0.1, "B": 1.1},
    {"A": math.nan, "B": 1.0},
    {"A": 0.0, "B": 0.0},
    {"A": 0.2, "B": 0.2},
])
def test_invalid_probability_distributions_are_rejected(probabilities):
    with pytest.raises(ValueError):
        temper_probabilities(probabilities, 3.4)


def test_invalid_letters_token_maps_and_logits_are_rejected():
    tokenizer = FakeTokenizer(["A", "not B"])
    with pytest.raises(ValueError, match="only single uppercase"):
        letter_token_ids(tokenizer, ("A", "b"))
    with pytest.raises(ValueError, match="duplicates"):
        letter_token_ids(tokenizer, ("A", "A"))
    with pytest.raises(ValueError, match="no one-token aliases for: B"):
        letter_token_ids(tokenizer, ("A", "B"))

    with pytest.raises(ValueError, match="outside vocab_logits"):
        letter_log_masses([0.0], {"A": (1,)})
    with pytest.raises(ValueError, match="more than one letter"):
        letter_log_masses([0.0], {"A": (0,), "B": (0,)})
    with pytest.raises(ValueError, match=r"NaN or \+inf"):
        letter_log_masses([math.nan], {"A": (0,)})
    with pytest.raises(ValueError, match="zero finite logit mass"):
        letter_log_masses([-math.inf], {"A": (0,)})
