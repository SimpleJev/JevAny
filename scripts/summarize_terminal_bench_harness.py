#!/usr/bin/env python3
"""Summarize the predeclared Terminal-Bench Jev harness sample.

The script reads Harbor trial results without modifying them.  Incomplete task/mode
pairs stay visible as pending so a partial snapshot cannot silently become a failed
or cherry-picked benchmark row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODES = ("baseline", "optional")
TIER_LABELS = {
    "easy": "easy (historical 4/4)",
    "medium": "medium (historical 2/4)",
    "hard": "hard (historical 0/4)",
}


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
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    value = rewards.get("reward")
    return float(value) if isinstance(value, (int, float)) else None


def _candidate_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [turn for turn in turns if turn.get("kind") == "commands"]
    counts = [int(turn.get("candidate_count") or 0) for turn in decisions]
    multi = sum(count >= 2 for count in counts)
    trace = []
    for turn in decisions:
        selected = []
        for command in turn.get("executed") or []:
            selected.append({
                "controller": command.get("controller"),
                "candidate_index": command.get("candidate"),
                "label": command.get("label"),
                "command": command.get("command"),
                "confidence": command.get("confidence"),
                "error": command.get("error"),
            })
        trace.append({
            "turn": turn.get("turn"),
            "subgoal": turn.get("subgoal"),
            "routine": turn.get("routine"),
            "composable_sequence": turn.get("composable_sequence"),
            "candidate_count": turn.get("candidate_count"),
            "frontier_choice": turn.get("frontier_choice"),
            "selected": selected,
        })
    return {
        "decision_turns": len(decisions),
        "candidates_generated": sum(counts),
        "mean_candidates_per_decision": sum(counts) / len(counts) if counts else None,
        "multi_candidate_turns": multi,
        "multi_candidate_rate": multi / len(counts) if counts else None,
        "trace": trace,
        "retention_note": (
            "v3 retained candidate_count, frontier_choice, and executed choices; "
            "unselected candidate texts were not persisted"
        ),
    }


def _trial_result(run_root: Path, task: str, mode: str) -> Path | None:
    matches = sorted((run_root / task / mode / "jobs").glob("*/*/result.json"))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"{task}/{mode}: expected one trial result, found {len(matches)}")
    return matches[0]


def _load_trial(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    agent = data.get("agent_result") or {}
    metadata = agent.get("metadata") or {}
    usage = metadata.get("usage") or {}
    input_tokens = agent.get("n_input_tokens")
    output_tokens = agent.get("n_output_tokens")
    if input_tokens is None:
        input_tokens = usage.get("inputTokens", 0)
    if output_tokens is None:
        output_tokens = usage.get("outputTokens", 0)
    turns = metadata.get("turns") or []
    direct = int(metadata.get("direct_commands") or 0)
    delegated = int(metadata.get("delegated_commands") or 0)
    command_total = direct + delegated
    reward = _reward(data)
    return {
        "status": "complete" if data.get("finished_at") and reward is not None else "incomplete",
        "source": str(path),
        "sha256": _sha256(path),
        "trial_id": data.get("id"),
        "reward": reward,
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "wall_time_seconds": _duration_seconds(data),
        "frontier_model": metadata.get("frontier_model"),
        "resolved_frontier_model": _public_model_id(metadata.get("resolved_frontier_model")),
        "jev_model": metadata.get("jev_model"),
        "frontier_calls": int(metadata.get("frontier_calls") or 0),
        "jev_calls": int(metadata.get("jev_calls") or 0),
        "jev_failures": int(metadata.get("jev_failures") or 0),
        "jev_low_confidence_returns": int(metadata.get("jev_low_confidence_returns") or 0),
        "delegated_commands": delegated,
        "direct_commands": direct,
        "command_decision_replacement_rate": delegated / command_total if command_total else None,
        "protocol_repairs": int(metadata.get("protocol_repairs") or 0),
        "completed_by_frontier": bool(metadata.get("completed_by_frontier")),
        "termination_reason": metadata.get("termination_reason"),
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(input_tokens or 0) + int(output_tokens or 0),
        "estimated_bedrock_cost_usd": agent.get("cost_usd", metadata.get("estimated_bedrock_cost_usd")),
        "frontier_candidates": _candidate_summary(turns),
    }


def _reduction(baseline: float | None, optional: float | None) -> float | None:
    if baseline in (None, 0) or optional is None:
        return None
    return (baseline - optional) / baseline


def _paired_metrics(baseline: dict[str, Any], optional: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward_delta": optional["reward"] - baseline["reward"],
        "frontier_call_reduction": _reduction(baseline["frontier_calls"], optional["frontier_calls"]),
        "token_reduction": _reduction(baseline["total_tokens"], optional["total_tokens"]),
        "cost_reduction": _reduction(
            baseline["estimated_bedrock_cost_usd"], optional["estimated_bedrock_cost_usd"]
        ),
        "wall_time_reduction": _reduction(baseline["wall_time_seconds"], optional["wall_time_seconds"]),
        "quality_preserved": optional["reward"] >= baseline["reward"],
    }


def _task_rows(run_root: Path, tiers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for tier in tiers:
        for task in tier["tasks"]:
            modes: dict[str, Any] = {}
            for mode in MODES:
                path = _trial_result(run_root, task, mode)
                modes[mode] = {"status": "pending"} if path is None else _load_trial(path)
            complete = all(modes[mode].get("status") == "complete" for mode in MODES)
            row = {
                "tier": tier["tier"],
                "tier_label": tier["label"],
                "historical_frontier_passes": tier["historical_frontier_passes"],
                "task": task,
                "status": "complete" if complete else "pending",
                "modes": modes,
                "comparison": _paired_metrics(modes["baseline"], modes["optional"]) if complete else None,
            }
            rows.append(row)
    return rows


def _tier_spec(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = next(
        benchmark
        for tier in manifest["tiers"]
        for benchmark in tier["benchmarks"]
        if benchmark["name"] == "terminal-bench-2"
    )
    tiers = []
    for key, tasks in benchmark["samples"].items():
        difficulty, _, historical = key.partition("_")
        passes = historical.replace("_of_", "/")
        tiers.append({
            "tier": difficulty,
            "label": TIER_LABELS.get(difficulty, f"{difficulty} (historical {passes})"),
            "historical_frontier_passes": passes,
            "tasks": tasks,
        })
    return tiers


def _sum(rows: list[dict[str, Any]], mode: str, field: str) -> float:
    return sum(float(row["modes"][mode][field]) for row in rows)


def _aggregate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    complete = [row for row in rows if row["status"] == "complete"]
    result: dict[str, Any] = {
        "label": label,
        "predeclared_tasks": len(rows),
        "complete_pairs": len(complete),
        "pending_tasks": [row["task"] for row in rows if row["status"] != "complete"],
    }
    if not complete:
        return result
    base_rewards = _sum(complete, "baseline", "reward")
    optional_rewards = _sum(complete, "optional", "reward")
    base_calls = _sum(complete, "baseline", "frontier_calls")
    optional_calls = _sum(complete, "optional", "frontier_calls")
    base_tokens = _sum(complete, "baseline", "total_tokens")
    optional_tokens = _sum(complete, "optional", "total_tokens")
    base_cost = _sum(complete, "baseline", "estimated_bedrock_cost_usd")
    optional_cost = _sum(complete, "optional", "estimated_bedrock_cost_usd")
    base_time = _sum(complete, "baseline", "wall_time_seconds")
    optional_time = _sum(complete, "optional", "wall_time_seconds")
    delegated = _sum(complete, "optional", "delegated_commands")
    direct = _sum(complete, "optional", "direct_commands")
    result.update({
        "baseline_success_rate": base_rewards / len(complete),
        "optional_success_rate": optional_rewards / len(complete),
        "success_rate_delta": (optional_rewards - base_rewards) / len(complete),
        "frontier_call_reduction": _reduction(base_calls, optional_calls),
        "token_reduction": _reduction(base_tokens, optional_tokens),
        "cost_reduction": _reduction(base_cost, optional_cost),
        "wall_time_reduction": _reduction(base_time, optional_time),
        "optional_jev_calls": int(_sum(complete, "optional", "jev_calls")),
        "optional_delegated_commands": int(delegated),
        "optional_direct_commands": int(direct),
        "command_decision_replacement_rate": delegated / (delegated + direct) if delegated + direct else None,
    })
    return result


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = ("reward_delta", "frontier_call_reduction", "token_reduction", "cost_reduction", "wall_time_reduction")
    left_values = [left["comparison"][field] for field in fields]
    right_values = [right["comparison"][field] for field in fields]
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _frontier(rows: list[dict[str, Any]]) -> list[str]:
    eligible = [
        row for row in rows
        if row["status"] == "complete"
        and row["comparison"]["quality_preserved"]
        and row["modes"]["optional"]["reward"] > 0
        and row["modes"]["optional"]["delegated_commands"] > 0
    ]
    return [
        row["task"] for row in eligible
        if not any(_dominates(other, row) for other in eligible if other is not row)
    ]


def _showcase(rows: list[dict[str, Any]]) -> str | None:
    eligible = [
        row for row in rows
        if row["status"] == "complete"
        and row["comparison"]["quality_preserved"]
        and row["modes"]["optional"]["reward"] > 0
        and row["modes"]["optional"]["delegated_commands"] > 0
    ]
    if not eligible:
        return None
    def rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
        comparison = row["comparison"]
        return (
            comparison["reward_delta"],
            comparison["frontier_call_reduction"] or 0,
            comparison["cost_reduction"] or 0,
            comparison["wall_time_reduction"] or 0,
        )
    return max(eligible, key=rank)["task"]


def _percent(value: float | None, *, signed: bool = True) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _num(value: Any, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}"


def _task_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Level | Task | Status | Success B→J | Frontier calls B→J | Tokens B→J | Cost B→J | Time B→J | Jev replaced decisions |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["status"] != "complete":
            present = ", ".join(mode for mode in MODES if row["modes"][mode].get("status") == "complete") or "none"
            lines.append(f"| {row['tier_label']} | `{row['task']}` | pending ({present} present) | — | — | — | — | — | — |")
            continue
        base, optional = row["modes"]["baseline"], row["modes"]["optional"]
        replaced = optional["command_decision_replacement_rate"]
        lines.append(
            f"| {row['tier_label']} | `{row['task']}` | complete | "
            f"{base['reward']:.0f}→{optional['reward']:.0f} ({row['comparison']['reward_delta']:+.0f}) | "
            f"{base['frontier_calls']}→{optional['frontier_calls']} ({_percent(row['comparison']['frontier_call_reduction'])}) | "
            f"{base['total_tokens']:,}→{optional['total_tokens']:,} ({_percent(row['comparison']['token_reduction'])}) | "
            f"${base['estimated_bedrock_cost_usd']:.3f}→${optional['estimated_bedrock_cost_usd']:.3f} ({_percent(row['comparison']['cost_reduction'])}) | "
            f"{base['wall_time_seconds']:.1f}s→{optional['wall_time_seconds']:.1f}s ({_percent(row['comparison']['wall_time_reduction'])}) | "
            f"{optional['delegated_commands']}/{optional['delegated_commands'] + optional['direct_commands']} ({_percent(replaced, signed=False)}) |"
        )
    return lines


def _aggregate_table(aggregates: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Level | Complete/predeclared | Success B→J | Calls saved | Tokens saved | Cost saved | Time saved | Jev decision share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        if not item["complete_pairs"]:
            lines.append(f"| {item['label']} | 0/{item['predeclared_tasks']} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {item['label']} | {item['complete_pairs']}/{item['predeclared_tasks']} | "
            f"{item['baseline_success_rate']:.0%}→{item['optional_success_rate']:.0%} ({_percent(item['success_rate_delta'])}) | "
            f"{_percent(item['frontier_call_reduction'])} | {_percent(item['token_reduction'])} | "
            f"{_percent(item['cost_reduction'])} | {_percent(item['wall_time_reduction'])} | "
            f"{item['optional_delegated_commands']}/{item['optional_delegated_commands'] + item['optional_direct_commands']} "
            f"({_percent(item['command_decision_replacement_rate'], signed=False)}) |"
        )
    return lines


def render_markdown(output: dict[str, Any]) -> str:
    rows = output["rows"]
    complete = [row for row in rows if row["status"] == "complete"]
    pending = [row["task"] for row in rows if row["status"] != "complete"]
    lines = [
        "# Terminal-Bench 2 Jev harness — sample-v3 snapshot",
        "",
        f"Predeclared sample: {len(rows)} tasks / {len(rows) * 2} runs; complete paired rows: {len(complete)}/{len(rows)}.",
        "Baseline executes the frontier LLM's selected candidate. Optional mode lets Jev choose bounded routine candidates while the frontier LLM retains diagnosis, edits, verification, and completion.",
        "Positive `saved` percentages favor Jev optional mode; negative values are regressions.",
        "",
    ]
    if pending:
        lines.extend([
            f"> Partial snapshot: pending paired rows are {', '.join(f'`{task}`' for task in pending)}. They remain pending and are excluded from aggregates.",
            "",
        ])
    lines.extend(["## Per-task trade-off", ""])
    lines.extend(_task_table(rows))
    lines.extend(["", "## Level aggregates", ""])
    lines.extend(_aggregate_table(output["tier_aggregates"] + [output["overall_aggregate"]]))
    lines.extend([
        "",
        "## Quality-preserving Pareto frontier",
        "",
    ])
    frontier = output["quality_preserving_pareto_frontier"]
    if frontier:
        lines.append(
            "Non-dominated among completed, quality-preserving rows with actual Jev delegation "
            "over reward delta plus calls/tokens/cost/time saved: "
            + ", ".join(f"`{task}`" for task in frontier)
            + "."
        )
    else:
        lines.append("No completed row currently qualifies.")
    showcase = output["showcase_task"]
    lines.extend(["", "## Demo candidate", ""])
    if showcase is None:
        lines.append("No quality-preserving delegated row is currently available.")
    else:
        row = next(row for row in rows if row["task"] == showcase)
        base, optional = row["modes"]["baseline"], row["modes"]["optional"]
        candidates = optional["frontier_candidates"]
        lines.extend([
            f"`{showcase}` is selected by the predeclared ranking: reward delta first, then frontier-call, cost, and time reduction.",
            "",
            f"- Outcome: {base['reward']:.0f}→{optional['reward']:.0f}; frontier calls {base['frontier_calls']}→{optional['frontier_calls']}; Jev executed {optional['delegated_commands']} of {optional['delegated_commands'] + optional['direct_commands']} command decisions.",
            f"- Candidate surface: {candidates['candidates_generated']} candidate slots across {candidates['decision_turns']} command turns; {candidates['multi_candidate_turns']} turns exposed multiple options.",
            "- The complete selected-command trace and raw result hashes are retained in the JSON companion.",
            "",
            "| Turn | Subgoal | Routine | Options | Controller / selected label |",
            "|---:|---|---|---:|---|",
        ])
        for turn in candidates["trace"]:
            selected = "; ".join(
                f"{choice.get('controller')}: {choice.get('label') or choice.get('error') or 'unlabeled'}"
                for choice in turn["selected"]
            ) or "none"
            subgoal = str(turn.get("subgoal") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {turn.get('turn')} | {subgoal} | {turn.get('routine')} | "
                f"{turn.get('candidate_count')} | {selected.replace('|', '\\|')} |"
            )
        lines.extend([
            "",
            "Note: sample-v3 persisted each option count, the frontier index, and executed choice, but not the text of unselected options. A v4 trace is required for a fully replayable option-menu demo.",
        ])
    lines.extend([
        "",
        "## Interpretation and provenance",
        "",
        "- This is one exploratory paired attempt per task, not a statistically powered benchmark. Do not infer population-level significance.",
        "- Difficulty is fixed from four retained historical frontier-agent runs (4/4, 2/4, 0/4), not assigned after observing these Jev results.",
        "- `Jev decision share` is delegated commands / (delegated + direct commands), not Jev calls / frontier calls. Failed or low-confidence Jev calls therefore do not inflate replacement.",
        "- Wall time uses Harbor trial start/finish timestamps. Bedrock dollars are the run-recorded estimates; Jev serving cost is not monetized.",
        "- Every source result is preserved. The JSON companion records repository-relative source paths, SHA-256 hashes, exact model IDs, and selected-command traces.",
        "",
        f"Generated: `{output['generated_at_utc']}`",
        "",
        f"Run root: `{output['run_root']}`",
        "",
        f"Manifest: `{output['manifest']}`",
        "",
    ])
    return "\n".join(lines)


def summarize(run_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tiers = _tier_spec(manifest)
    rows = _task_rows(run_root, tiers)
    tier_aggregates = [
        _aggregate([row for row in rows if row["tier"] == tier["tier"]], tier["label"])
        for tier in tiers
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "manifest": str(manifest_path),
        "protocol": "candidate-harness-v3",
        "selection_policy": manifest.get("selection_policy"),
        "tier_definition": tiers,
        "rows": rows,
        "tier_aggregates": tier_aggregates,
        "overall_aggregate": _aggregate(rows, "all completed tiers"),
        "quality_preserving_pareto_frontier": _frontier(rows),
        "showcase_task": _showcase(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/terminal-bench/sample-v3")
    parser.add_argument("--manifest", default="configs/agent_task_ladder.json")
    parser.add_argument("--out-json", default="runs/terminal-bench/sample-v3-summary.json")
    parser.add_argument("--out-md", default="runs/terminal-bench/sample-v3-summary.md")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    output = summarize(Path(args.run_root), Path(args.manifest))
    pending = [row["task"] for row in output["rows"] if row["status"] != "complete"]
    if args.require_complete and pending:
        raise SystemExit(f"pending paired rows: {', '.join(pending)}")

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown(output), encoding="utf-8")


if __name__ == "__main__":
    main()
