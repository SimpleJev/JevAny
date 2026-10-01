#!/usr/bin/env python3
"""Render the Terminal-Bench v4 delegation-rate frontier.

This is intentionally separate from the sample-v3 summarizer.  Source trials are
read-only, pending cells remain explicit, and every completed cell retains its
source hash and complete frontier candidate-menu trace in the JSON output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TASKS = ("regex-log", "sqlite-db-truncate")
RATE_SPECS = (
    ("d0", 0.0),
    ("d50", 0.5),
    ("d100", 1.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_model_id(value: str | None) -> str | None:
    if value and ":inference-profile/" in value:
        return value.split(":inference-profile/", 1)[1]
    return value


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _duration_seconds(data: dict[str, Any]) -> float | None:
    started = _parse_time(data.get("started_at"))
    finished = _parse_time(data.get("finished_at"))
    if started is None or finished is None:
        return None
    return (finished - started).total_seconds()


def _reward(data: dict[str, Any]) -> float | None:
    value = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return float(value) if isinstance(value, (int, float)) else None


def _trial_result(run_root: Path, task: str, rate_id: str) -> Path | None:
    matches = sorted((run_root / task / rate_id / "jobs").glob("*/*/result.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"{task}/{rate_id}: expected one trial result, found {len(matches)}")
    return matches[0]


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _candidate_trace(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain every v4 candidate menu and its selected execution(s)."""
    trace = []
    for turn in turns:
        if turn.get("kind") != "commands":
            continue
        candidates = []
        for candidate in turn.get("candidates") or []:
            candidates.append({
                "index": candidate.get("index"),
                "label": candidate.get("label"),
                "command": candidate.get("command"),
                "expected": candidate.get("expected"),
            })
        executed = []
        for action in turn.get("executed") or []:
            output = action.get("output")
            executed.append({
                "controller": action.get("controller"),
                "candidate_index": action.get("candidate"),
                "label": action.get("label"),
                "command": action.get("command"),
                "confidence": action.get("confidence"),
                "jev_latency_ms": action.get("jev_latency_ms"),
                "command_completed": action.get("command_completed"),
                "error": action.get("error"),
                "output_sha256": (
                    hashlib.sha256(output.encode("utf-8")).hexdigest()
                    if isinstance(output, str) else None
                ),
            })
        trace.append({
            "turn": turn.get("turn"),
            "reasoning_summary": turn.get("reasoning_summary"),
            "subgoal": turn.get("subgoal"),
            "routine": turn.get("routine"),
            "composable_sequence": turn.get("composable_sequence"),
            "eligible_for_delegation": turn.get("eligible_for_delegation"),
            "policy_targeted": turn.get("policy_targeted"),
            "actual_delegated": turn.get("actual_delegated"),
            "frontier_choice": turn.get("frontier_choice"),
            "candidate_count": turn.get("candidate_count"),
            "candidates": candidates,
            "executed": executed,
        })
    return trace


def _load_trial(path: Path, rate_id: str, configured_rate: float) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    agent = data.get("agent_result") or {}
    metadata = agent.get("metadata") or {}
    usage = metadata.get("usage") or {}
    input_tokens = agent.get("n_input_tokens", usage.get("inputTokens", 0)) or 0
    output_tokens = agent.get("n_output_tokens", usage.get("outputTokens", 0)) or 0
    eligible = int(metadata.get("eligible_decisions") or 0)
    targeted = int(metadata.get("target_delegated_decisions") or 0)
    actual = int(metadata.get("actual_delegated_decisions") or 0)
    delegated_commands = int(metadata.get("delegated_commands") or 0)
    direct_commands = int(metadata.get("direct_commands") or 0)
    turns = metadata.get("turns") or []
    fallback_turns = [
        turn for turn in turns
        if turn.get("kind") == "commands"
        and turn.get("policy_targeted")
        and not turn.get("actual_delegated")
    ]
    fallback_commands = sum(
        action.get("controller") == "frontier" and bool(action.get("command"))
        for turn in fallback_turns
        for action in turn.get("executed") or []
    )
    reward = _reward(data)
    finished = bool(data.get("finished_at"))
    return {
        "status": "complete" if finished and reward is not None else "incomplete",
        "rate_id": rate_id,
        "configured_delegation_rate": configured_rate,
        "recorded_delegation_rate": metadata.get("delegation_rate"),
        "harness_level": metadata.get("harness_level"),
        "mode": metadata.get("mode"),
        "reward": reward,
        "frontier_calls": int(metadata.get("frontier_calls") or 0),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "total_tokens": int(input_tokens) + int(output_tokens),
        "estimated_bedrock_cost_usd": agent.get(
            "cost_usd", metadata.get("estimated_bedrock_cost_usd")
        ),
        "wall_time_seconds": _duration_seconds(data),
        "eligible_decisions": eligible,
        "target_delegated_decisions": targeted,
        "actual_delegated_decisions": actual,
        "target_eligible_replacement_rate": metadata.get(
            "target_replacement_rate", _ratio(targeted, eligible)
        ),
        "actual_eligible_replacement_rate": metadata.get(
            "actual_replacement_rate", _ratio(actual, eligible)
        ),
        "delegated_commands": delegated_commands,
        "direct_commands": direct_commands,
        "command_replacement_rate": metadata.get(
            "actual_command_replacement_rate",
            _ratio(delegated_commands, delegated_commands + direct_commands),
        ),
        "frontier_fallback_decisions": max(0, targeted - actual),
        "frontier_fallback_commands": int(fallback_commands),
        "jev_calls": int(metadata.get("jev_calls") or 0),
        "jev_failures": int(metadata.get("jev_failures") or 0),
        "jev_low_confidence_returns": int(metadata.get("jev_low_confidence_returns") or 0),
        "protocol_repairs": int(metadata.get("protocol_repairs") or 0),
        "completed_by_frontier": bool(metadata.get("completed_by_frontier")),
        "termination_reason": metadata.get("termination_reason"),
        "frontier_model": metadata.get("frontier_model"),
        "resolved_frontier_model": _public_model_id(metadata.get("resolved_frontier_model")),
        "jev_model": metadata.get("jev_model"),
        "candidate_trace": _candidate_trace(turns),
        "source": str(path),
        "source_sha256": _sha256(path),
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
    }


def _reduction(reference: float | int | None, value: float | int | None) -> float | None:
    if reference in (None, 0) or value is None:
        return None
    return (float(reference) - float(value)) / float(reference)


def _comparison(reference: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_rate_id": reference["rate_id"],
        "reward_delta_vs_d0": cell["reward"] - reference["reward"],
        "quality_preserved_vs_d0": cell["reward"] >= reference["reward"],
        "frontier_call_reduction_vs_d0": _reduction(reference["frontier_calls"], cell["frontier_calls"]),
        "token_reduction_vs_d0": _reduction(reference["total_tokens"], cell["total_tokens"]),
        "cost_reduction_vs_d0": _reduction(
            reference["estimated_bedrock_cost_usd"], cell["estimated_bedrock_cost_usd"]
        ),
        "wall_time_reduction_vs_d0": _reduction(
            reference["wall_time_seconds"], cell["wall_time_seconds"]
        ),
    }


def _pareto_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Maximize quality, resource savings, and actual eligible replacement."""
    metrics = (
        "reward_delta_vs_d0",
        "frontier_call_reduction_vs_d0",
        "token_reduction_vs_d0",
        "cost_reduction_vs_d0",
        "wall_time_reduction_vs_d0",
    )
    left_values = [left["comparison_vs_d0"][key] for key in metrics]
    right_values = [right["comparison_vs_d0"][key] for key in metrics]
    left_values.append(left["actual_eligible_replacement_rate"])
    right_values.append(right["actual_eligible_replacement_rate"])
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _task_frontier(cells: list[dict[str, Any]]) -> dict[str, Any]:
    reference = next(
        (cell for cell in cells if cell["rate_id"] == "d0" and cell["status"] == "complete"),
        None,
    )
    if reference is None:
        return {
            "status": "pending_reference",
            "reference_rate_id": "d0",
            "quality_preserving_rates": [],
            "pareto_rate_ids": [],
        }
    complete = [cell for cell in cells if cell["status"] == "complete"]
    for cell in complete:
        cell["comparison_vs_d0"] = _comparison(reference, cell)
    eligible = [
        cell for cell in complete
        if cell["comparison_vs_d0"]["quality_preserved_vs_d0"]
    ]
    frontier = [
        cell for cell in eligible
        if not any(_pareto_dominates(other, cell) for other in eligible if other is not cell)
    ]
    return {
        "status": "complete" if len(complete) == len(RATE_SPECS) else "partial",
        "reference_rate_id": "d0",
        "quality_constraint": "reward >= reward(d0)",
        "objectives": [
            "maximize reward delta",
            "maximize frontier-call/token/cost/time reduction",
            "maximize actual eligible-decision replacement",
        ],
        "quality_preserving_rates": [cell["rate_id"] for cell in eligible],
        "pareto_rate_ids": [cell["rate_id"] for cell in frontier],
    }


def summarize(run_root: Path, tasks: tuple[str, ...]) -> dict[str, Any]:
    task_rows = []
    for task in tasks:
        cells = []
        for rate_id, configured_rate in RATE_SPECS:
            path = _trial_result(run_root, task, rate_id)
            if path is None:
                cells.append({
                    "status": "pending",
                    "rate_id": rate_id,
                    "configured_delegation_rate": configured_rate,
                })
            else:
                cells.append(_load_trial(path, rate_id, configured_rate))
        frontier = _task_frontier(cells)
        task_rows.append({
            "task": task,
            "status": "complete" if all(cell["status"] == "complete" for cell in cells) else "partial",
            "cells": cells,
            "frontier": frontier,
        })
    return {
        "schema_version": 1,
        "protocol": "candidate-harness-v4-delegation-rate-frontier",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "rate_definitions": [
            {"rate_id": rate_id, "configured_delegation_rate": rate}
            for rate_id, rate in RATE_SPECS
        ],
        "tasks": task_rows,
        "complete_cells": sum(
            cell["status"] == "complete" for task in task_rows for cell in task["cells"]
        ),
        "predeclared_cells": len(task_rows) * len(RATE_SPECS),
        "candidate_trace_contract": (
            "Every completed cell retains every v4 command turn, full generated candidate "
            "menu, frontier choice, policy target, actual controller selection, and source SHA-256. "
            "Terminal outputs remain in the immutable hashed raw result."
        ),
    }


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", "<br>")


def _cell_table(task: dict[str, Any]) -> list[str]:
    lines = [
        "| Rate | Status | Reward | Calls | Tokens | Cost | Time | Target/actual eligible | Command replacement | Repairs | Fallbacks | Jev fail/low-conf |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in task["cells"]:
        if cell["status"] != "complete":
            lines.append(f"| `{cell['rate_id']}` | {cell['status']} | — | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{cell['rate_id']}` | complete | {cell['reward']:.0f} | "
            f"{cell['frontier_calls']} | {cell['total_tokens']:,} | "
            f"${cell['estimated_bedrock_cost_usd']:.3f} | {cell['wall_time_seconds']:.1f}s | "
            f"{cell['target_delegated_decisions']}/{cell['eligible_decisions']} "
            f"({_percent(cell['target_eligible_replacement_rate'])}) / "
            f"{cell['actual_delegated_decisions']}/{cell['eligible_decisions']} "
            f"({_percent(cell['actual_eligible_replacement_rate'])}) | "
            f"{cell['delegated_commands']}/{cell['delegated_commands'] + cell['direct_commands']} "
            f"({_percent(cell['command_replacement_rate'])}) | {cell['protocol_repairs']} | "
            f"{cell['frontier_fallback_decisions']} decisions / {cell['frontier_fallback_commands']} commands | "
            f"{cell['jev_failures']}/{cell['jev_low_confidence_returns']} |"
        )
    return lines


def _delta_table(task: dict[str, Any]) -> list[str]:
    lines = [
        "| Rate | Reward Δ | Calls saved | Tokens saved | Cost saved | Time saved | Quality preserved |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cell in task["cells"]:
        comparison = cell.get("comparison_vs_d0")
        if comparison is None:
            lines.append(f"| `{cell['rate_id']}` | n/a | n/a | n/a | n/a | n/a | pending |")
            continue
        lines.append(
            f"| `{cell['rate_id']}` | {comparison['reward_delta_vs_d0']:+.0f} | "
            f"{_percent(comparison['frontier_call_reduction_vs_d0'], signed=True)} | "
            f"{_percent(comparison['token_reduction_vs_d0'], signed=True)} | "
            f"{_percent(comparison['cost_reduction_vs_d0'], signed=True)} | "
            f"{_percent(comparison['wall_time_reduction_vs_d0'], signed=True)} | "
            f"{comparison['quality_preserved_vs_d0']} |"
        )
    return lines


def _trace_index(task: dict[str, Any]) -> list[str]:
    lines = [
        "| Rate | Turn | Eligible/targeted/actual | Frontier choice | Candidate menu | Executed |",
        "|---|---:|---|---:|---|---|",
    ]
    any_trace = False
    for cell in task["cells"]:
        for turn in cell.get("candidate_trace") or []:
            any_trace = True
            menu = "; ".join(
                f"{candidate['index']}: {candidate['label']} — {candidate['command']}"
                for candidate in turn["candidates"]
            )
            selected = "; ".join(
                f"{action['controller']}:{action.get('candidate_index')} {action.get('label') or action.get('error') or ''}"
                for action in turn["executed"]
            )
            flags = "/".join(str(bool(turn.get(key)))[0] for key in (
                "eligible_for_delegation", "policy_targeted", "actual_delegated"
            ))
            lines.append(
                f"| `{cell['rate_id']}` | {turn['turn']} | {flags} | {turn.get('frontier_choice')} | "
                f"{_escape(menu)} | {_escape(selected)} |"
            )
    if not any_trace:
        lines.append("| — | — | — | — | pending | — |")
    return lines


def render_markdown(output: dict[str, Any]) -> str:
    lines = [
        "# Terminal-Bench 2 delegation-rate frontier — v4",
        "",
        f"Completed {output['complete_cells']}/{output['predeclared_cells']} predeclared task×rate cells.",
        "`d0`, `d50`, and `d100` target 0%, 50%, and 100% of eligible routine decisions. Actual eligible replacement can be lower after Jev failure or low confidence.",
        "",
    ]
    if output["complete_cells"] < output["predeclared_cells"]:
        lines.extend([
            "> Partial snapshot: pending cells remain visible and are excluded from task frontiers until their result exists.",
            "",
        ])
    for task in output["tasks"]:
        lines.extend([f"## `{task['task']}`", ""])
        lines.extend(_cell_table(task))
        lines.extend(["", "Relative to the same-task `d0` reference:", ""])
        lines.extend(_delta_table(task))
        frontier = task["frontier"]
        lines.extend([
            "",
            "Quality-preserving Pareto rates: "
            + (", ".join(f"`{rate}`" for rate in frontier["pareto_rate_ids"]) or "pending")
            + ". The frontier jointly rewards quality, resource savings, and actual eligible replacement.",
            "",
            "### Candidate trace index",
            "",
            "`Eligible/targeted/actual` is shown as T/F flags. The JSON companion retains reasoning summaries, expected outcomes, all full candidate fields, execution metadata, and source hashes.",
            "",
        ])
        lines.extend(_trace_index(task))
        lines.append("")
    lines.extend([
        "## Method and provenance",
        "",
        "- Pareto dominance is computed within each task only, after requiring reward ≥ that task's `d0` reward.",
        "- Objectives maximize reward delta, calls/tokens/cost/time saved versus `d0`, and actual eligible-decision replacement.",
        "- Fallback decisions are policy-targeted eligible decisions with no actual Jev execution. Fallback commands count frontier executions on those turns.",
        "- This is one exploratory attempt per cell; it is a demo frontier, not a confidence interval or a population estimate.",
        "- Raw trial files are not modified. Each completed JSON cell includes its repository-relative source path and SHA-256.",
        "",
        f"Generated: `{output['generated_at_utc']}`",
        "",
        f"Run root: `{output['run_root']}`",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/terminal-bench/frontier-v4")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--out-json", default="runs/terminal-bench/frontier-v4-summary.json")
    parser.add_argument("--out-md", default="runs/terminal-bench/frontier-v4-summary.md")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    output = summarize(Path(args.run_root), tuple(args.tasks))
    if args.require_complete and output["complete_cells"] != output["predeclared_cells"]:
        raise SystemExit(
            f"completed {output['complete_cells']}/{output['predeclared_cells']} cells"
        )
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(output), encoding="utf-8")


if __name__ == "__main__":
    main()
