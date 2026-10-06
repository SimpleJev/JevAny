"""Planner JSON is checked before constructing or running a decision tree."""
import copy

import pytest

from jevany.harness import json_object
from jevany.symbolic import JevTree


TREE = {
    "root": "start",
    "nodes": {"start": {"question": {"criteria": {"go": "Continue"}}, "branches": {"go": "done"}}},
    "outcomes": {"done": {"status": "finished"}},
}


@pytest.mark.parametrize("text", [
    '[{"questions":{}}]', '```json\n[{"questions":{}}]\n```',
    'Here is the plan:\n```json\n[{"questions":{}}]\n```',
    '"{}"', "null", "true", "42",
])
def test_planner_rejects_json_values_that_are_not_objects(text):
    with pytest.raises(ValueError):
        json_object(text)


@pytest.mark.parametrize("text", [
    '{"questions":{}}',
    '```json\n{"questions":{}}\n```',
    'Here is the plan:\n{"questions":{}}',
])
def test_planner_keeps_plain_fenced_and_prose_wrapped_objects(text):
    assert json_object(text) == {"questions": {}}


@pytest.mark.parametrize("value", [
    None, [], {}, {"root": "start"}, {**TREE, "nodes": None}, {**TREE, "nodes": []},
    {**TREE, "root": []}, {**TREE, "outcomes": []},
    {**TREE, "nodes": {"start": {}}},
    {**TREE, "nodes": {"start": {"question": [], "branches": {}}}},
    {**TREE, "nodes": {"start": {"question": {"criteria": {"go": "Continue"}}, "branches": {"go": []}}}},
])
def test_malformed_tree_shapes_raise_actionable_value_errors(value):
    with pytest.raises(ValueError, match="root|nodes|outcomes|dictionary"):
        JevTree.from_dict(value)


def test_structural_validation_keeps_input_unchanged_and_preserves_outcomes():
    value = copy.deepcopy(TREE)
    tree = JevTree.from_dict(value)
    assert value == TREE
    assert tree.nodes["start"].question.type == "choice"
    assert tree.outcomes == {"done": {"status": "finished"}}
    result = tree.run("state", lambda request: {"answers": {"branch": {
        "type": "choice", "choice": "go", "confidence": 1, "probabilities": {"go": 1},
    }}})
    assert result["outcome"] == {"status": "finished"}
