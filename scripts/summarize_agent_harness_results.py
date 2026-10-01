#!/usr/bin/env python3
"""Validate paired agent-harness runs and render one auditable result matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from jevany.agent_harness import estimated_bedrock_cost
from jevany.suite import write_json
from scripts.eval_bedrock_jev_agent import render_demo


LIST_PRICES = {
    "us.anthropic.claude-opus-4-7": {
        "input": 5.50,
        "output": 27.50,
        "source": "AWS public Price List, checked 2026-09-30",
    },
    "us.anthropic.claude-sonnet-4-6": {
        "input": 3.30,
        "output": 16.50,
        "source": "AWS public Price List, checked 2026-09-30",
    },
    "us.openai.gpt-5.6-sol": {
        "input": 4.40,
        "output": 22.00,
        "source": "AWS model card short-context Geo CRIS, checked 2026-09-30",
        "catalog_caveat": "No matching public Price List SKU; cached-token price unavailable.",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_model_id(value: str | None) -> str | None:
    """Remove account-specific ARN prefixes from published result summaries."""

    if value and ":inference-profile/" in value:
        return value.split(":inference-profile/", 1)[1]
    return value


def _mode_summary(data: dict, environment: str, mode: str) -> dict:
    summaries = data["summaries"]
    return summaries[f"{environment}/{mode}"] if f"{environment}/{mode}" in summaries else summaries[mode]


def _comparison(data: dict, environment: str) -> dict:
    if "comparisons" in data:
        return data["comparisons"][environment]
    return data["comparison"]


def validate_result(data: dict, path: Path) -> tuple[str, list[int]]:
    if data.get("status") != "complete":
        raise ValueError(f"{path}: result is not complete")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{path}: episodes must be a non-empty list")
    environments = {item.get("environment", "webshop") for item in episodes}
    if len(environments) != 1:
        raise ValueError(f"{path}: expected one environment, got {sorted(environments)}")
    environment = next(iter(environments))
    indexed = {}
    for item in episodes:
        key = (item.get("mode"), item.get("seed"))
        if key in indexed:
            raise ValueError(f"{path}: duplicate episode {key}")
        indexed[key] = item
    baseline = {seed for mode, seed in indexed if mode == "baseline"}
    optional = {seed for mode, seed in indexed if mode == "optional"}
    if baseline != optional:
        raise ValueError(f"{path}: unpaired seeds {baseline} != {optional}")
    expected = data.get("config", {}).get("episodes")
    if expected is not None and len(baseline) != expected:
        raise ValueError(f"{path}: expected {expected} pairs, found {len(baseline)}")
    _mode_summary(data, environment, "baseline")
    _mode_summary(data, environment, "optional")
    _comparison(data, environment)
    return environment, sorted(baseline)


def _cost(summary: dict, model: str) -> float | None:
    if "estimated_bedrock_cost_usd" in summary:
        return summary["estimated_bedrock_cost_usd"]
    price = LIST_PRICES.get(model)
    if price is None:
        return None
    usage = summary["usage"]
    return estimated_bedrock_cost(usage, price["input"], price["output"])


def _reduction(baseline: float | None, optional: float | None) -> float | None:
    if baseline in (None, 0) or optional is None:
        return None
    return (baseline - optional) / baseline


def row_from_result(data: dict, path: Path) -> dict:
    environment, seeds = validate_result(data, path)
    baseline = _mode_summary(data, environment, "baseline")
    optional = _mode_summary(data, environment, "optional")
    comparison = _comparison(data, environment)
    model = data["bedrock_model"]
    baseline_cost, optional_cost = _cost(baseline, model), _cost(optional, model)
    paired = comparison.get("paired", {})
    clear_win = (
        comparison["success_rate_delta"] >= 0
        and paired.get("bedrock_calls", {}).get("bootstrap_95_ci", [0, 0])[1] < 0
        and paired.get("tokens", {}).get("bootstrap_95_ci", [0, 0])[1] < 0
        and paired.get("latency_ms", {}).get("bootstrap_95_ci", [0, 0])[1] < 0
    )
    return {
        "source": str(path),
        "sha256": _sha256(path),
        "benchmark": environment,
        "pairs": len(seeds),
        "seeds": seeds,
        "bedrock_model": model,
        "resolved_bedrock_model": _public_model_id(data.get("resolved_bedrock_model")),
        "jev_model": data["jev_model"],
        "baseline_success_rate": baseline["success_rate"],
        "optional_success_rate": optional["success_rate"],
        "success_rate_delta": comparison["success_rate_delta"],
        "bedrock_call_reduction": comparison["bedrock_call_reduction"],
        "bedrock_token_reduction": comparison["bedrock_token_reduction"],
        "wall_time_reduction": comparison["wall_time_reduction"],
        "estimated_bedrock_cost_baseline_usd": baseline_cost,
        "estimated_bedrock_cost_optional_usd": optional_cost,
        "estimated_bedrock_cost_reduction": _reduction(baseline_cost, optional_cost),
        "mean_optional_delegation_calls": optional["mean_delegation_calls"],
        "mean_optional_jev_decisions": optional["mean_jev_decisions"],
        "baseline_error_rate": baseline.get("error_rate", 0),
        "optional_error_rate": optional.get("error_rate", 0),
        "paired": paired,
        "clear_quality_cost_latency_win": clear_win,
    }


def _percent(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def render_markdown(output: dict) -> str:
    rows = output["rows"]
    lines = [
        "# Jev optional-delegation agent benchmark",
        "",
        f"Validated {sum(row['pairs'] for row in rows)} paired seeds across {len(rows)} model/benchmark/checkpoint cells.",
        "The frontier model autonomously chose between direct action and bounded Jev delegation.",
        "",
        "| Frontier model | Benchmark | Jev | n | Success base → optional | Calls saved | Tokens saved | Bedrock $ saved | Time saved | Jev calls/episode |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        jev = "4B" if "3.5-4B" in row["jev_model"] else "27B"
        lines.append(
            f"| `{row['bedrock_model']}` | `{row['benchmark']}` | {jev} | {row['pairs']} | "
            f"{row['baseline_success_rate']:.0%} → {row['optional_success_rate']:.0%} | "
            f"{_percent(row['bedrock_call_reduction'], signed=True)} | "
            f"{_percent(row['bedrock_token_reduction'], signed=True)} | "
            f"{_percent(row['estimated_bedrock_cost_reduction'], signed=True)} | "
            f"{_percent(row['wall_time_reduction'], signed=True)} | "
            f"{row['mean_optional_delegation_calls']:.2f} |"
        )
    winners = [row for row in rows if row["clear_quality_cost_latency_win"]]
    lines.extend([
        "",
        "## Evidence-backed reading",
        "",
    ])
    if winners:
        for row in winners:
            lines.append(
                f"- Clear multi-metric win: `{row['bedrock_model']}` on `{row['benchmark']}` with "
                f"{row['jev_model']}: success {_percent(row['baseline_success_rate'])} → "
                f"{_percent(row['optional_success_rate'])}, calls saved "
                f"{_percent(row['bedrock_call_reduction'])}, tokens saved "
                f"{_percent(row['bedrock_token_reduction'])}, and time saved "
                f"{_percent(row['wall_time_reduction'])}. Paired bootstrap intervals for calls, "
                "tokens, and latency exclude zero."
            )
    else:
        lines.append("- No cell meets the predeclared quality + calls + tokens + latency win rule.")
    lines.extend([
        "- Other cells are mixed or negative and remain in the table; they are not presented as wins.",
        "- Dollar figures are estimated Bedrock list-price costs, not Cost Explorer billing. `n/a` means the run used cached tokens whose applicable price could not be verified.",
        "- Jev GPU rental/energy cost is not monetized. One already-allocated H200 served the 27B matrix; cold start was excluded after readiness, while per-request Jev latency remains in wall time.",
        "- FrozenLake and Sokoban are controlled mechanism tests. WebShop and WebArena are external interactive-agent checks; the WebArena row is a predeclared six-task sample, not the full benchmark.",
        "",
        "## Provenance",
        "",
        f"Generated: `{output['generated_at_utc']}`",
        "",
        f"Pricing basis: `{output['pricing_checked_at']}`",
        "",
        "Every row retains the raw JSON path, SHA-256, exact model IDs, seeds, paired deltas, and bootstrap intervals in the machine-readable matrix.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--demo-dir")
    args = parser.parse_args()
    paths = [Path(value) for value in args.inputs]
    records = [
        (row_from_result(data, path), data)
        for path in paths
        for data in [json.loads(path.read_text(encoding="utf-8"))]
    ]
    records.sort(key=lambda item: (
        item[0]["bedrock_model"], item[0]["benchmark"], item[0]["jev_model"],
    ))
    rows = [row for row, _ in records]
    demo_files = []
    if args.demo_dir:
        demo_dir = Path(args.demo_dir)
        demo_dir.mkdir(parents=True, exist_ok=True)
        for row, data in records:
            case = data.get("demo_case")
            if not row["clear_quality_cost_latency_win"] or not case:
                continue
            model_name = row["bedrock_model"].replace(".", "-").replace("/", "-")
            path = demo_dir / f"{model_name}-{row['benchmark']}-winner.md"
            path.write_text(
                render_demo(case, row["bedrock_model"], row["jev_model"]),
                encoding="utf-8",
            )
            demo_files.append({"path": str(path), "source": row["source"]})
    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pricing_checked_at": "2026-09-30",
        "pricing": LIST_PRICES,
        "rows": rows,
        "demo_files": demo_files,
    }
    write_json(Path(args.out_json), output)
    markdown = render_markdown(output)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
