"""None-pair augmentation preserves options and produces valid requests."""
from copy import deepcopy
import random

import pytest

from jevany.api import MAX_OPTIONS
from jevany.data import NONE_OPTIONS, materialize, none_pair


@pytest.mark.parametrize("options", [MAX_OPTIONS - 1, MAX_OPTIONS])
def test_none_pair_respects_request_option_limit(options):
    request = {
        "state": "evidence",
        "questions": {"q": {
            "type": "choice", "criteria": {f"o{i}": None for i in range(options)}, "label": "o0",
        }},
    }
    original = deepcopy(request)
    materialize(request)

    variants = none_pair(request, random.Random(0))

    if options == MAX_OPTIONS:
        assert variants == []
    else:
        assert [len(variant["questions"]["q"]["criteria"]) for variant in variants] == [
            MAX_OPTIONS, MAX_OPTIONS - 1,
        ]
        for variant in variants:
            materialize(variant)
    assert request == original


def test_none_pair_selects_a_question_with_room_for_another_option():
    request = {
        "state": "evidence",
        "questions": {
            "full": {
                "type": "choice", "criteria": {f"o{i}": None for i in range(MAX_OPTIONS)}, "label": "o0",
            },
            "small": {"type": "choice", "criteria": {"a": None, "b": None, "c": None}, "label": "a"},
        },
    }
    original = deepcopy(request)
    materialize(request)

    variants = none_pair(request, random.Random(1))

    assert len(variants) == 2
    for variant in variants:
        assert set(variant["questions"]) == {"small"}
        materialize(variant)
    assert request == original


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
