#!/usr/bin/env python3
"""Paired Bedrock-agent evaluation with optional multi-step Jev delegation."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from jevany.agent_harness import BedrockJevAgent, estimated_bedrock_cost
from jevany.harness import HTTPDecisionClient
from jevany.suite import write_json
from scripts.eval_ragen import GOALS, environment


REGION_PREFIXES = ("us.", "eu.", "ap.", "global.")
TOKEN_FIELDS = (
    "inputTokens", "outputTokens", "totalTokens", "cacheReadInputTokens",
    "cacheWriteInputTokens",
)


def runtime_client(profile: str, region: str, model: str, timeout: int = 900):
    import boto3
    from botocore.config import Config

    user_config = Path.home() / ".aws" / "config"
    if user_config.is_file():
        os.environ["AWS_CONFIG_FILE"] = str(user_config)
    session = boto3.Session(profile_name=profile, region_name=region)
    model_id = model
    if model.startswith(REGION_PREFIXES):
        account = session.client("sts", region_name=region).get_caller_identity()["Account"]
        model_id = f"arn:aws:bedrock:{region}:{account}:inference-profile/{model}"
    client = session.client(
        "bedrock-runtime", region_name=region,
        config=Config(
            connect_timeout=10, read_timeout=timeout, max_pool_connections=4,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )
    return client, model_id


def summarize(episodes: list[dict], input_price=None, output_price=None,
              cache_read_price=None, cache_write_price=None) -> dict:
    if not episodes:
        raise ValueError("cannot summarize an empty episode list")
    usage = {key: sum(item["usage"].get(key, 0) for item in episodes) for key in TOKEN_FIELDS}
    summary = {
        "episodes": len(episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "mean_reward": statistics.fmean(item["reward"] for item in episodes),
        "mean_actions": statistics.fmean(len(item["actions"]) for item in episodes),
        "mean_bedrock_calls": statistics.fmean(item["bedrock_calls"] for item in episodes),
        "mean_bedrock_failures": statistics.fmean(item.get("bedrock_failures", 0) for item in episodes),
        "mean_direct_actions": statistics.fmean(item["direct_actions"] for item in episodes),
        "mean_delegation_calls": statistics.fmean(item["delegation_calls"] for item in episodes),
        "mean_jev_decisions": statistics.fmean(item["jev_decisions"] for item in episodes),
        "mean_jev_failures": statistics.fmean(item.get("jev_failures", 0) for item in episodes),
        "error_rate": sum(item.get("status") == "error" for item in episodes) / len(episodes),
        "mean_total_latency_ms": statistics.fmean(item["total_latency_ms"] for item in episodes),
        "mean_bedrock_latency_ms": statistics.fmean(item["bedrock_latency_ms"] for item in episodes),
        "mean_jev_latency_ms": statistics.fmean(item["jev_latency_ms"] for item in episodes),
        "usage": usage,
    }
    cost = estimated_bedrock_cost(
        usage, input_price, output_price,
        cache_read_per_million=cache_read_price,
        cache_write_per_million=cache_write_price,
    )
    if cost is not None:
        summary["estimated_bedrock_cost_usd"] = cost
    return summary


def reduction(baseline: float, optional: float) -> float | None:
    return None if baseline == 0 else (baseline - optional) / baseline


def bootstrap_interval(values: list[float], *, samples: int = 10_000,
                       seed: int = 20260930) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap empty values")
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples) - 1]]


def compare(summaries: dict, episodes: list[dict]) -> dict:
    output = {}
    environments = sorted({key.split("/", 1)[0] for key in summaries})
    for name in environments:
        baseline = summaries.get(f"{name}/baseline")
        optional = summaries.get(f"{name}/optional")
        if not baseline or not optional:
            continue
        indexed = {}
        for record in (item for item in episodes if item["environment"] == name):
            key = (record["mode"], record["seed"])
            if key in indexed:
                raise ValueError(f"duplicate paired result: {name}/{key[0]}/{key[1]}")
            indexed[key] = record
        baseline_seeds = {seed for mode, seed in indexed if mode == "baseline"}
        optional_seeds = {seed for mode, seed in indexed if mode == "optional"}
        if baseline_seeds != optional_seeds:
            raise ValueError(f"unpaired seed sets for {name}: {baseline_seeds} != {optional_seeds}")

        def token_count(record):
            return record["usage"].get("totalTokens", (
                record["usage"].get("inputTokens", 0) + record["usage"].get("outputTokens", 0)
            ))

        baseline_tokens = baseline["usage"]["totalTokens"]
        optional_tokens = optional["usage"]["totalTokens"]
        deltas = {key: [] for key in ("success", "reward", "bedrock_calls", "tokens", "latency_ms")}
        for paired_seed in sorted(baseline_seeds):
            left = indexed[("baseline", paired_seed)]
            right = indexed[("optional", paired_seed)]
            deltas["success"].append(float(right["success"]) - float(left["success"]))
            deltas["reward"].append(right["reward"] - left["reward"])
            deltas["bedrock_calls"].append(right["bedrock_calls"] - left["bedrock_calls"])
            deltas["tokens"].append(token_count(right) - token_count(left))
            deltas["latency_ms"].append(right["total_latency_ms"] - left["total_latency_ms"])
        item = {
            "success_rate_delta": optional["success_rate"] - baseline["success_rate"],
            "reward_delta": optional["mean_reward"] - baseline["mean_reward"],
            "bedrock_call_reduction": reduction(
                baseline["mean_bedrock_calls"], optional["mean_bedrock_calls"],
            ),
            "bedrock_token_reduction": reduction(baseline_tokens, optional_tokens),
            "wall_time_reduction": reduction(
                baseline["mean_total_latency_ms"], optional["mean_total_latency_ms"],
            ),
            "paired": {
                key: {
                    "mean_optional_minus_baseline": statistics.fmean(values),
                    "bootstrap_95_ci": bootstrap_interval(values),
                }
                for key, values in deltas.items()
            },
        }
        if "estimated_bedrock_cost_usd" in baseline:
            item["estimated_cost_reduction"] = reduction(
                baseline["estimated_bedrock_cost_usd"],
                optional["estimated_bedrock_cost_usd"],
            )
        output[name] = item
    return output


def select_demo(episodes: list[dict], *, seed: int | None = None) -> dict | None:
    """Return the pre-registered seed without inspecting outcomes or usage."""
    paired = defaultdict(dict)
    for item in episodes:
        paired[(item["environment"], item["seed"])][item["mode"]] = item
    if seed is None and paired:
        seed = min(episode_seed for _, episode_seed in paired)
    for (environment_name, episode_seed), pair in sorted(paired.items()):
        if episode_seed != seed:
            continue
        if "baseline" not in pair or "optional" not in pair:
            continue
        return {
            "environment": environment_name, "seed": episode_seed,
            "selection": "predeclared first paired seed",
            "baseline": pair["baseline"], "optional": pair["optional"],
        }
    return None


def render_demo(case: dict, model: str, jev_model: str) -> str:
    baseline, optional = case["baseline"], case["optional"]

    def tokens(item):
        return item["usage"].get("inputTokens", 0) + item["usage"].get("outputTokens", 0)

    lines = [
        "# Frontier model + Jev delegation case",
        "",
        f"Benchmark: `{case['environment']}` · seed `{case['seed']}`  ",
        f"Frontier model: `{model}` · decision model: `{jev_model}`",
        "",
        "| Mode | Success | Actions | Bedrock calls | Bedrock tokens | Jev decisions | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("Frontier only", baseline), ("Frontier + optional Jev", optional)):
        lines.append(
            f"| {label} | {'yes' if item['success'] else 'no'} | {len(item['actions'])} | "
            f"{item['bedrock_calls']} | {tokens(item):,} | {item['jev_decisions']} | "
            f"{item['total_latency_ms'] / 1000:.2f}s |"
        )
    lines.extend(["", "## Why and when the frontier model delegated", ""])
    delegated = False
    for turn in optional["transcript"]:
        if turn.get("tool") == "delegate_to_jev":
            delegated = True
            subgoal = (turn.get("tool_input") or {}).get("subgoal", "(missing subgoal)")
            lines.append(
                f"- Turn {turn['turn'] + 1}: {turn.get('text') or '(no public rationale)'} "
                f"**Delegated subgoal:** {subgoal}"
            )
    if not delegated:
        lines.append("- The frontier model did not delegate in this episode.")
    lines.extend(["", "## Optional-Jev action trace", "", "| # | Controller | Action | Confidence | Reward | Terminal |", "|---:|---|---|---:|---:|---:|"])
    for action in optional["actions"]:
        confidence = "—" if action["confidence"] is None else f"{action['confidence']:.3f}"
        lines.append(
            f"| {action['index'] + 1} | {action['controller']} | {action['action_name']} | "
            f"{confidence} | {action['reward']:.3f} | {'yes' if action['done'] else 'no'} |"
        )
    lines.extend([
        "", "This is a paired benchmark replay, not a hand-authored product demonstration. "
        "The environment seed and complete machine-readable trajectory are retained with the evaluation result.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ragen-repo", default="../RAGEN")
    parser.add_argument("--environment", action="append", choices=sorted(GOALS), dest="environments")
    parser.add_argument("--mode", action="append", choices=("baseline", "optional", "jev_only"), dest="modes")
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--profile", default="bedrock")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--jev-url", default="http://127.0.0.1:8008")
    parser.add_argument("--jev-model", default="SimpleJev/JevAny-Qwen3.8-27B-LoRA")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--max-turns", type=int, default=64)
    parser.add_argument("--max-delegate-steps", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--cache-read-price-per-million", type=float)
    parser.add_argument("--cache-write-price-per-million", type=float)
    parser.add_argument("--mode-order", choices=("counterbalanced", "fixed"), default="counterbalanced")
    parser.add_argument(
        "--demo-seed", type=int,
        help="Pre-register a demo seed; defaults to --seed and never selects by outcome.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--demo-out")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_actions < 1:
        parser.error("episode and action counts must be positive")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    environments = args.environments or ["frozen_lake", "sokoban"]
    modes = args.modes or ["baseline", "optional", "jev_only"]
    if len(set(modes)) != len(modes):
        parser.error("each --mode may be specified at most once")
    repository = Path(args.ragen_repo).resolve()
    if not (repository / "ragen" / "env" / "base.py").exists():
        parser.error(f"RAGEN checkout not found under {repository}")
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()

    client, resolved_model = runtime_client(args.profile, args.region, args.bedrock_model)
    agent = BedrockJevAgent(
        client, resolved_model, HTTPDecisionClient(args.jev_url), jev_model=args.jev_model,
        max_output_tokens=args.max_output_tokens, max_turns=args.max_turns,
        max_delegate_steps=args.max_delegate_steps,
        confidence_threshold=args.confidence_threshold,
        temperature=args.temperature,
    )
    episodes = []
    for environment_name in environments:
        for index in range(args.episodes):
            seed = args.seed + index
            ordered_modes = modes if args.mode_order == "fixed" or index % 2 == 0 else list(reversed(modes))
            for mode in ordered_modes:
                env = environment(environment_name, repository)
                try:
                    result = agent.run(
                        env, GOALS[environment_name], seed=seed, mode=mode,
                        max_actions=args.max_actions,
                    ).as_dict()
                    result["status"] = (
                        "error" if result["termination_reason"] == "bedrock_backend_error"
                        else "complete"
                    )
                except Exception as error:
                    result = {
                        "mode": mode, "seed": seed, "status": "error", "success": False,
                        "reward": 0.0, "actions": [], "bedrock_calls": 0,
                        "bedrock_failures": 0,
                        "direct_actions": 0, "delegation_calls": 0, "jev_decisions": 0,
                        "jev_failures": 0, "low_confidence_returns": 0,
                        "protocol_repairs": 0, "termination_reason": "backend_error",
                        "error_type": type(error).__name__, "usage": {},
                        "bedrock_latency_ms": 0.0, "jev_latency_ms": 0.0,
                        "total_latency_ms": 0.0, "transcript": [],
                    }
                finally:
                    close = getattr(env, "close", None)
                    if close:
                        close()
                result.update(environment=environment_name)
                episodes.append(result)
                write_json(output_path, {
                    "schema_version": 1, "status": "running",
                    "bedrock_model": args.bedrock_model, "jev_model": args.jev_model,
                    "episodes": episodes,
                })
                print(json.dumps({
                    "environment": environment_name, "seed": seed, "mode": mode,
                    "success": result["success"], "actions": len(result["actions"]),
                    "bedrock_calls": result["bedrock_calls"],
                    "jev_decisions": result["jev_decisions"],
                    "tokens": result["usage"].get("totalTokens", 0),
                }, sort_keys=True), flush=True)

    grouped = defaultdict(list)
    for item in episodes:
        grouped[f"{item['environment']}/{item['mode']}"] .append(item)
    summaries = {
        key: summarize(
            value, args.input_price_per_million, args.output_price_per_million,
            args.cache_read_price_per_million, args.cache_write_price_per_million,
        )
        for key, value in sorted(grouped.items())
    }
    demo_seed = args.seed if args.demo_seed is None else args.demo_seed
    if not args.seed <= demo_seed < args.seed + args.episodes:
        parser.error("--demo-seed must be inside the evaluated seed range")
    demo = select_demo(episodes, seed=demo_seed)
    output = {
        "schema_version": 1,
        "benchmark": "RAGEN paired discrete-agent evaluation",
        "ragen_revision": revision,
        "bedrock_model": args.bedrock_model,
        "resolved_bedrock_model": resolved_model,
        "jev_model": args.jev_model,
        "config": {
            "environments": environments, "modes": modes, "episodes": args.episodes,
            "seed_start": args.seed, "max_actions": args.max_actions,
            "max_turns": args.max_turns, "max_delegate_steps": args.max_delegate_steps,
            "confidence_threshold": args.confidence_threshold,
            "max_output_tokens": args.max_output_tokens, "temperature": args.temperature,
            "mode_order": args.mode_order,
            "demo_seed": demo_seed,
            "input_price_per_million": args.input_price_per_million,
            "output_price_per_million": args.output_price_per_million,
            "cache_read_price_per_million": args.cache_read_price_per_million,
            "cache_write_price_per_million": args.cache_write_price_per_million,
        },
        "summaries": summaries,
        "comparisons": compare(summaries, episodes),
        "demo_case": demo,
        "episodes": episodes,
    }
    output["status"] = "complete"
    write_json(output_path, output)
    if args.demo_out and demo:
        demo_path = Path(args.demo_out)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_path.write_text(render_demo(demo, args.bedrock_model, args.jev_model), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "comparisons": output["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
