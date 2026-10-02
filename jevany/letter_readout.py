"""One-token option-letter readout primitives for frozen causal LMs.

The Cygnet-compatible prompt and letter aggregation semantics are adapted from
the MIT-licensed ``blockbrain-ai/cygnet-recipe`` shim at commit ``3cf591c``
(Copyright 2026 Nood Co and contributors; https://github.com/blockbrain-ai/cygnet-recipe).
Cygnet credits the one-token option-letter readout idea to NInfer, Apache-2.0
(https://github.com/igorls/ninfer).  This module contains no serving/backend
integration: callers remain responsible for constrained decoding and for
enabling or disabling a model's thinking mode.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_PROMPT = (
    "You are a calibration engine. You never answer in prose. You are given a state, a question and "
    "a numbered set of options, and you choose exactly one option. You reply with that option's "
    "LETTER and nothing else — a single character, no words, no punctuation, no explanation."
)

_NEGATIVE_INFINITY = float("-inf")


@dataclass(frozen=True)
class LetterReadout:
    """Raw and post-hoc calibrated distributions in option-letter order."""

    raw_log_masses: dict[str, float]
    raw_probabilities: dict[str, float]
    calibrated_probabilities: dict[str, float]
    temperature: float


def _ordered_descriptions(options: Mapping[Any, Any] | Sequence[Any]) -> list[Any]:
    if isinstance(options, Mapping):
        # Match Cygnet's application path: a blank description falls back to
        # its option label; structured descriptions include the label and
        # deterministic JSON; ordinary text stays verbatim.
        descriptions = []
        for label, description in options.items():
            if description is None or (isinstance(description, str) and not description.strip()):
                descriptions.append(str(label))
            elif isinstance(description, str):
                descriptions.append(description)
            else:
                descriptions.append(f"{label}: {json.dumps(description, ensure_ascii=False)}")
    elif isinstance(options, Sequence) and not isinstance(options, (str, bytes, bytearray)):
        descriptions = list(options)
    else:
        raise TypeError("options must be an ordered mapping or a non-string sequence")
    if not descriptions:
        raise ValueError("options must contain at least one option")
    if len(descriptions) > len(LETTERS):
        raise ValueError(f"options may contain at most {len(LETTERS)} entries")
    return descriptions


def build_prompt(state: Any, instructions: Any, options: Mapping[Any, Any] | Sequence[Any]) -> str:
    """Build Cygnet's measured user prompt, assigning options A through Z.

    Mapping insertion order or sequence order is the option order.  Structured
    state is rendered with ``indent=1`` as in Cygnet.  This function only
    builds text; thinking/template settings are deliberately caller-controlled.
    """

    descriptions = _ordered_descriptions(options)
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)
    instruction_text = (
        instructions if isinstance(instructions, str)
        else json.dumps(instructions, ensure_ascii=False)
    )
    lines = [state_text.rstrip(), "", instruction_text.rstrip(), "", "Options:"]
    lines.extend(f"{LETTERS[index]}. {description}" for index, description in enumerate(descriptions))
    lines.extend(("", "Answer with the letter of exactly one option, and nothing else:"))
    return "\n".join(lines)


def _validate_letters(letters: Sequence[str]) -> tuple[str, ...]:
    if isinstance(letters, (bytes, bytearray)):
        raise TypeError("letters must be a sequence of A-Z strings")
    selected = tuple(letters)
    if not selected:
        raise ValueError("letters must not be empty")
    if len(selected) > len(LETTERS):
        raise ValueError(f"letters may contain at most {len(LETTERS)} entries")
    if any(type(letter) is not str or len(letter) != 1 or letter not in LETTERS for letter in selected):
        raise ValueError("letters must contain only single uppercase A-Z strings")
    if len(set(selected)) != len(selected):
        raise ValueError("letters must not contain duplicates")
    return selected


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        # Some tokenizer-compatible test/dedicated runtimes do not expose the
        # cleanup keyword.  Never fall back to convert_ids_to_tokens: raw BPE
        # pieces are not the decoded surface text scored by the model.
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    if not isinstance(decoded, str):
        raise TypeError(f"tokenizer.decode returned {type(decoded).__name__} for token id {token_id}")
    return decoded


def letter_token_ids(tokenizer: Any, letters: Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Scan the vocabulary for every token ID decoding to each exact letter.

    Token text is intentionally never used as a dictionary key: distinct token
    IDs may decode to identical text, and every such ID contributes probability
    mass.  Tokens decoding to ``" A"``, ``"A."``, or ``"a"`` are excluded:
    Cygnet's structured-output choice admits the exact uppercase string only.
    ``len(tokenizer)`` must describe the full vocabulary, including added tokens.
    """

    selected = _validate_letters(letters)
    try:
        vocab_size = len(tokenizer)
    except (TypeError, AttributeError) as error:
        raise TypeError("tokenizer must define the complete vocabulary via len(tokenizer)") from error
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("tokenizer vocabulary must have a positive integer size")

    aliases: dict[str, list[int]] = {letter: [] for letter in selected}

    def admit(token_id, decoded):
        if not isinstance(decoded, str):
            raise TypeError(
                f"tokenizer decode returned {type(decoded).__name__} for token id {token_id}"
            )
        if decoded in aliases:
            aliases[decoded].append(token_id)

    batch_decode = getattr(tokenizer, "batch_decode", None)
    if callable(batch_decode):
        # Fast tokenizers amortize Python/Rust crossings over a chunk.  A
        # 248k-token vocabulary otherwise takes minutes when decoded one ID at
        # a time.  Chunks keep the temporary list bounded.
        chunk_size = 4096
        for start in range(0, vocab_size, chunk_size):
            stop = min(start + chunk_size, vocab_size)
            token_ids = list(range(start, stop))
            try:
                decoded = batch_decode(
                    [[token_id] for token_id in token_ids],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                decoded = batch_decode([[token_id] for token_id in token_ids], skip_special_tokens=False)
            if len(decoded) != len(token_ids):
                raise ValueError("tokenizer.batch_decode returned the wrong number of tokens")
            for token_id, surface in zip(token_ids, decoded, strict=True):
                admit(token_id, surface)
    else:
        for token_id in range(vocab_size):
            admit(token_id, _decode_token(tokenizer, token_id))

    missing = [letter for letter, token_ids in aliases.items() if not token_ids]
    if missing:
        raise ValueError(f"tokenizer has no one-token aliases for: {', '.join(missing)}")
    return {letter: tuple(token_ids) for letter, token_ids in aliases.items()}


def _logaddexp(left: float, right: float) -> float:
    if left == _NEGATIVE_INFINITY:
        return right
    if right == _NEGATIVE_INFINITY:
        return left
    high = max(left, right)
    return high + math.log1p(math.exp(-abs(left - right)))


def _logit(value: Any, token_id: int) -> float:
    if isinstance(value, bool):
        raise TypeError(f"logit for token id {token_id} must be numeric, not bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"logit for token id {token_id} must be numeric") from error
    if math.isnan(number) or number == math.inf:
        raise ValueError(f"logit for token id {token_id} must not be NaN or +inf")
    return number


def letter_log_masses(
    vocab_logits: Sequence[Any], token_ids: Mapping[str, Sequence[int]]
) -> dict[str, float]:
    """Log-sum-exp all token logits belonging to each option letter.

    ``vocab_logits`` contains all admitted token rows at the answer position:
    either a complete vocabulary vector or a compact vector whose indices were
    remapped in ``token_ids``. Normalizing exact-letter rows emulates Cygnet's
    structured-output choice mask. Callers must not pre-deduplicate logits by
    decoded text because several token IDs can decode to one exact letter.
    """

    if not isinstance(token_ids, Mapping) or not token_ids:
        raise ValueError("token_ids must be a non-empty ordered mapping")
    try:
        vocab_size = len(vocab_logits)
    except (TypeError, AttributeError) as error:
        raise TypeError("vocab_logits must be a sized complete vocabulary vector") from error
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("vocab_logits must not be empty")

    letters = _validate_letters(tuple(token_ids))
    seen_token_ids: set[int] = set()
    masses: dict[str, float] = {}
    for letter in letters:
        aliases = token_ids[letter]
        if isinstance(aliases, (str, bytes, bytearray)):
            raise TypeError(f"token IDs for {letter} must be a sequence of integers")
        aliases = tuple(aliases)
        if not aliases:
            raise ValueError(f"letter {letter} has no token IDs")
        mass = _NEGATIVE_INFINITY
        for token_id in aliases:
            if type(token_id) is not int or not 0 <= token_id < vocab_size:
                raise ValueError(f"token id {token_id!r} for {letter} is outside vocab_logits")
            if token_id in seen_token_ids:
                raise ValueError(f"token id {token_id} is assigned to more than one letter")
            seen_token_ids.add(token_id)
            mass = _logaddexp(mass, _logit(vocab_logits[token_id], token_id))
        if mass == _NEGATIVE_INFINITY:
            raise ValueError(f"letter {letter} has zero finite logit mass")
        masses[letter] = mass
    return masses


def probabilities_from_log_masses(log_masses: Mapping[str, Any]) -> dict[str, float]:
    """Normalize ordered per-letter log masses with a stable softmax."""

    if not isinstance(log_masses, Mapping) or not log_masses:
        raise ValueError("log_masses must be a non-empty ordered mapping")
    values: list[float] = []
    for letter, value in log_masses.items():
        if isinstance(value, bool):
            raise TypeError(f"log mass for {letter} must be numeric, not bool")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"log mass for {letter} must be numeric") from error
        if not math.isfinite(number):
            raise ValueError(f"log mass for {letter} must be finite")
        values.append(number)
    pivot = max(values)
    weights = [math.exp(value - pivot) for value in values]
    total = sum(weights)
    return {letter: weight / total for letter, weight in zip(log_masses, weights, strict=True)}


def _validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool):
        raise TypeError("temperature must be a finite positive number")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as error:
        raise TypeError("temperature must be a finite positive number") from error
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return temperature


def temper_probabilities(
    probabilities: Mapping[str, Any], temperature: float = 1.0
) -> dict[str, float]:
    """Apply ``p ** (1 / temperature)`` and renormalize, preserving order."""

    temperature = _validate_temperature(temperature)
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("probabilities must be a non-empty ordered mapping")

    values: list[float] = []
    for key, value in probabilities.items():
        if isinstance(value, bool):
            raise TypeError(f"probability for {key} must be numeric, not bool")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"probability for {key} must be numeric") from error
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"probability for {key} must be finite and non-negative")
        values.append(number)
    total = sum(values)
    if total <= 0 or not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"probabilities must sum to 1, got {total}")
    if temperature == 1.0:
        return {key: value for key, value in zip(probabilities, values, strict=True)}

    # This is algebraically p ** (1 / T), evaluated in log space so a valid
    # but very small temperature cannot underflow every option to zero.
    scaled_logs = [
        math.log(probability) / temperature if probability > 0 else _NEGATIVE_INFINITY
        for probability in values
    ]
    pivot = max(scaled_logs)
    weights = [math.exp(value - pivot) if value != _NEGATIVE_INFINITY else 0.0
               for value in scaled_logs]
    normalizer = sum(weights)
    return {key: weight / normalizer for key, weight in zip(probabilities, weights, strict=True)}


def read_letter_distribution(
    vocab_logits: Sequence[Any],
    token_ids: Mapping[str, Sequence[int]],
    temperature: float = 1.0,
) -> LetterReadout:
    """Aggregate admitted token logits and return raw plus calibrated probabilities."""

    temperature = _validate_temperature(temperature)
    masses = letter_log_masses(vocab_logits, token_ids)
    raw = probabilities_from_log_masses(masses)
    if temperature == 1.0:
        calibrated = dict(raw)
    else:
        # softmax(log_mass / T) is exactly p ** (1/T), with the common raw
        # normalizer cancelled.  Calibrating before materializing tiny raw
        # probabilities avoids losing recoverable mass to float underflow.
        calibrated = probabilities_from_log_masses(
            {letter: mass / temperature for letter, mass in masses.items()}
        )
    return LetterReadout(
        raw_log_masses=masses,
        raw_probabilities=raw,
        calibrated_probabilities=calibrated,
        temperature=temperature,
    )


__all__ = [
    "LETTERS",
    "SYSTEM_PROMPT",
    "LetterReadout",
    "build_prompt",
    "letter_token_ids",
    "letter_log_masses",
    "probabilities_from_log_masses",
    "temper_probabilities",
    "read_letter_distribution",
]
