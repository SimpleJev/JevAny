import pytest

from jevany.terminal_harness import (
    TerminalCompletion,
    TerminalProposal,
    delegate_at_rate,
    delegation_limit,
    jev_terminal_request,
    parse_frontier_response,
    validate_command,
)


def response(name, values):
    return {"output": {"message": {"content": [{"toolUse": {
        "toolUseId": "call-1", "name": name, "input": values,
    }}]}}}


def proposal_values():
    return {
        "reasoning_summary": "Inspect before editing.",
        "subgoal": "Locate the relevant files.",
        "routine": True,
        "composable_sequence": False,
        "delegate_steps": 2,
        "frontier_choice": 0,
        "candidates": [
            {"label": "list", "command": "ls -la", "expected": "files", "wait_seconds": 1},
            {"label": "find", "command": "find . -maxdepth 2 -type f", "expected": "paths", "wait_seconds": 2},
        ],
    }


def test_parse_proposal_and_build_jev_request():
    tool_id, item = parse_frontier_response(response("propose_terminal_options", proposal_values()))
    assert tool_id == "call-1"
    assert isinstance(item, TerminalProposal)
    request = jev_terminal_request("fix it", item, "$", list(item.candidates), [], "jev")
    assert request["questions"]["action"]["criteria"]["0"].startswith("list:")


def test_parse_completion():
    _, item = parse_frontier_response(response(
        "complete_task", {"summary": "done", "verification": "tests pass"}
    ))
    assert isinstance(item, TerminalCompletion)


@pytest.mark.parametrize("command", [
    "torsocks curl x", "sudo apt install tor", "/usr/bin/tor",
    "t'o'r --version", r"t\or --version", "to${EMPTY}r --version",
])
def test_tor_commands_are_rejected(command):
    with pytest.raises(ValueError, match="prohibited"):
        validate_command(command)


@pytest.mark.parametrize("command", [
    "find / -type f", "pwd\nfind / -type f", "/usr/bin/find / -name '*.py'",
])
def test_recursive_root_scan_is_rejected(command):
    with pytest.raises(ValueError, match="filesystem root"):
        validate_command(command)


def test_invalid_frontier_choice_is_rejected():
    values = proposal_values()
    values["frontier_choice"] = 3
    with pytest.raises(ValueError, match="frontier_choice"):
        parse_frontier_response(response("propose_terminal_options", values))


def test_nonroutine_precise_edit_may_have_one_candidate():
    values = proposal_values()
    values["routine"] = False
    values["candidates"] = values["candidates"][:1]
    _, item = parse_frontier_response(response("propose_terminal_options", values))
    assert len(item.candidates) == 1


def test_routine_decision_still_requires_options():
    values = proposal_values()
    values["candidates"] = values["candidates"][:1]
    with pytest.raises(ValueError, match="at least two candidates"):
        parse_frontier_response(response("propose_terminal_options", values))


def test_empty_wait_command_is_normalized():
    values = proposal_values()
    values["routine"] = False
    values["candidates"] = [{
        "label": "wait for build", "command": "", "expected": "build output", "wait_seconds": 2,
    }]
    values["frontier_choice"] = 0
    _, item = parse_frontier_response(response("propose_terminal_options", values))
    assert item.candidates[0].command == "<WAIT>"


def test_harness_levels_form_monotonic_delegation_ladder():
    _, single = parse_frontier_response(response("propose_terminal_options", proposal_values()))
    assert [delegation_limit(single, level) for level in range(4)] == [0, 1, 1, 1]

    values = proposal_values()
    values["composable_sequence"] = True
    _, sequence = parse_frontier_response(response("propose_terminal_options", values))
    assert [delegation_limit(sequence, level) for level in range(4)] == [0, 0, 1, 2]


def test_delegation_rate_is_deterministic():
    assert [delegate_at_rate(index, 0.5) for index in range(1, 7)] == [
        False, True, False, True, False, True,
    ]
    assert all(delegate_at_rate(index, 1.0) for index in range(1, 4))
    assert not any(delegate_at_rate(index, 0.0) for index in range(1, 4))


@pytest.mark.parametrize("level", [-1, 4, True])
def test_invalid_harness_level_is_rejected(level):
    _, item = parse_frontier_response(response("propose_terminal_options", proposal_values()))
    with pytest.raises(ValueError, match="harness_level"):
        delegation_limit(item, level)


def test_agent_modes_keep_legacy_defaults_and_force_baseline_rate(tmp_path):
    pytest.importorskip("harbor")
    from jevany.terminal_bench_agent import AgentHarness

    baseline = AgentHarness(logs_dir=tmp_path / "base", mode="baseline", delegation_rate=0.75)
    optional = AgentHarness(logs_dir=tmp_path / "optional", mode="optional", delegation_rate=0.25)
    assert (baseline.harness_level, baseline.delegation_rate) == (0, 0.0)
    assert (optional.harness_level, optional.delegation_rate) == (3, 0.25)


def test_agent_rejects_out_of_range_tradeoff_controls(tmp_path):
    pytest.importorskip("harbor")
    from jevany.terminal_bench_agent import AgentHarness

    with pytest.raises(ValueError, match="delegation_rate"):
        AgentHarness(logs_dir=tmp_path, mode="optional", delegation_rate=1.1)
    with pytest.raises(ValueError, match="harness_level"):
        AgentHarness(logs_dir=tmp_path, mode="optional", harness_level=4)
