"""None-pair augmentation preserves user option keys and descriptions."""
from copy import deepcopy
import random

import pytest

from jevany.data import NONE_OPTIONS, materialize, none_pair


@pytest.mark.parametrize("label", ["none_of_these", "other"])
def test_none_pair_skips_when_all_none_option_keys_are_occupied(label):
    criteria = {key: f"Existing description for {key}" for key, _ in NONE_OPTIONS}
    criteria["none_of_these"] = "An existing option, not a generated none option"
    request = {
        "state": "evidence",
        "questions": {"q": {"type": "choice", "criteria": criteria, "label": label}},
    }
    original = deepcopy(request)
    materialize(request)

    assert none_pair(request, random.Random(0)) == []
    assert request == original


@pytest.mark.parametrize("use_fallback", [False, True], ids=["regular", "fallback"])
def test_none_pair_adds_a_fresh_key_and_preserves_existing_options(use_fallback):
    criteria = (
        {key: f"Existing description for {key}" for key, _ in NONE_OPTIONS}
        if use_fallback else {"a": "First option", "b": "Second option"}
    )
    criteria["answer"] = "The correct option"
    request = {
        "state": {"evidence": "answer"},
        "media": [{"type": "image", "uri": "frame.png"}],
        "questions": {"q": {
            "type": "choice", "instructions": "Choose one", "criteria": criteria, "label": "answer",
        }},
    }
    original = deepcopy(request)

    present, absent = none_pair(request, random.Random(0))

    present_question, absent_question = present["questions"]["q"], absent["questions"]["q"]
    added = set(present_question["criteria"]) - set(criteria)
    assert len(added) == 1
    added_key = added.pop()
    if use_fallback:
        assert added_key == "none_of_these"
    else:
        assert added_key in dict(NONE_OPTIONS)
    assert {key: present_question["criteria"][key] for key in criteria} == criteria
    assert present_question["label"] == "answer"
    assert absent_question["label"] == added_key
    assert absent_question["criteria"] == {
        key: value for key, value in present_question["criteria"].items() if key != "answer"
    }
    for variant in (present, absent):
        assert variant["state"] == request["state"]
        assert variant["media"] == request["media"]
        assert variant["questions"]["q"]["instructions"] == "Choose one"
        materialize(variant)
    assert request == original
