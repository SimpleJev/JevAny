#!/usr/bin/env python3
"""Paired Bedrock baseline/optional-Jev evaluation on sampled WebArena tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path

import requests

from jevany.agent_harness import estimated_bedrock_cost
from jevany.harness import HTTPDecisionClient
from jevany.suite import write_json
from jevany.webarena_harness import BedrockJevWebArenaAgent


TOKEN_FIELDS = (
    "inputTokens", "outputTokens", "totalTokens", "cacheReadInputTokens",
    "cacheWriteInputTokens", "cacheWriteInputTokens5m", "cacheWriteInputTokens1h",
    "cacheWriteInputTokens30m",
)
REGION_PREFIXES = ("us.", "eu.", "ap.", "global.")


class WebArenaEnvClient:
    def __init__(self, base_url: str, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.env_id = None

    def create(self) -> None:
        response = requests.post(f"{self.base_url}/create", timeout=self.timeout)
        response.raise_for_status()
        self.env_id = response.json()["env_idx"]

    def reset(self, task_id: int) -> dict:
        if self.env_id is None:
            self.create()
        response = requests.post(
            f"{self.base_url}/reset",
            json={"env_idx": self.env_id, "seed": 0, "idx": task_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def observe(self) -> str:
        response = requests.get(
            f"{self.base_url}/observation?env_idx={self.env_id}", timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json())

    def step(self, action: str) -> dict:
        response = requests.post(
            f"{self.base_url}/step",
            json={"env_idx": self.env_id, "action": action}, timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        if self.env_id is None:
            return
        try:
            requests.post(
                f"{self.base_url}/close", json={"env_idx": self.env_id}, timeout=60,
            )
        except requests.RequestException:
            pass


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
    return session.client(
        "bedrock-runtime", region_name=region,
        config=Config(connect_timeout=10, read_timeout=timeout, max_pool_connections=4,
                      retries={"max_attempts": 5, "mode": "adaptive"}),
    ), model_id


def summarize(episodes: list[dict], prices: dict[str, float | None]) -> dict:
    usage = {key: sum(item.get("usage", {}).get(key, 0) for item in episodes)
             for key in TOKEN_FIELDS}
    result = {
        "episodes": len(episodes),
        "success_rate": sum(item.get("success", False) for item in episodes) / len(episodes),
        "mean_reward": statistics.fmean(item.get("reward", 0.0) for item in episodes),
        "mean_actions": statistics.fmean(len(item.get("actions", [])) for item in episodes),
        "mean_bedrock_calls": statistics.fmean(item.get("bedrock_calls", 0) for item in episodes),
        "mean_delegation_calls": statistics.fmean(item.get("delegation_calls", 0) for item in episodes),
        "mean_jev_decisions": statistics.fmean(item.get("jev_decisions", 0) for item in episodes),
        "mean_total_latency_ms": statistics.fmean(item.get("total_latency_ms", 0) for item in episodes),
        "error_rate": sum(item.get("status") == "error" for item in episodes) / len(episodes),
        "usage": usage,
    }
    cost = estimated_bedrock_cost(
        usage, prices["input"], prices["output"],
        cache_read_per_million=prices["cache_read"],
        cache_write_per_million=prices["cache_write"],
    )
    if cost is not None:
        result["estimated_bedrock_cost_usd"] = cost
    return result


def bootstrap_interval(values: list[float], samples: int = 10000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples))
    return [means[int(samples * 0.025)], means[int(samples * 0.975)]]


def _reduction(baseline: float, optional: float) -> float | None:
    return None if baseline == 0 else (baseline - optional) / baseline


def compare(summaries: dict, episodes: list[dict]) -> dict:
    baseline, optional = summaries["baseline"], summaries["optional"]
    paired = defaultdict(dict)
    for item in episodes:
        paired[item["task_id"]][item["mode"]] = item
    deltas = {"success": [], "bedrock_calls": [], "tokens": [], "latency_ms": []}
    for task_id, pair in sorted(paired.items()):
        if set(pair) != {"baseline", "optional"}:
            raise ValueError(f"unpaired task {task_id}: {sorted(pair)}")
        left, right = pair["baseline"], pair["optional"]
        deltas["success"].append(float(right["success"]) - float(left["success"]))
        deltas["bedrock_calls"].append(right["bedrock_calls"] - left["bedrock_calls"])
        deltas["tokens"].append(right["usage"].get("totalTokens", 0) - left["usage"].get("totalTokens", 0))
        deltas["latency_ms"].append(right["total_latency_ms"] - left["total_latency_ms"])
    base_tokens = baseline["usage"].get("totalTokens", 0)
    opt_tokens = optional["usage"].get("totalTokens", 0)
    output = {
        "success_rate_delta": optional["success_rate"] - baseline["success_rate"],
        "reward_delta": optional["mean_reward"] - baseline["mean_reward"],
        "bedrock_call_reduction": _reduction(baseline["mean_bedrock_calls"], optional["mean_bedrock_calls"]),
        "bedrock_token_reduction": _reduction(base_tokens, opt_tokens),
        "wall_time_reduction": _reduction(baseline["mean_total_latency_ms"], optional["mean_total_latency_ms"]),
        "paired": {key: {"mean_optional_minus_baseline": statistics.fmean(values),
                         "bootstrap_95_ci": bootstrap_interval(values)}
                   for key, values in deltas.items()},
    }
    if "estimated_bedrock_cost_usd" in baseline and "estimated_bedrock_cost_usd" in optional:
        output["estimated_bedrock_cost_reduction"] = _reduction(
            baseline["estimated_bedrock_cost_usd"], optional["estimated_bedrock_cost_usd"]
        )
    return output


def render_demo(case: dict, model: str, jev_model: str) -> str:
    lines = [
        "# Frontier model + optional Jev on WebArena", "",
        f"Task: `webarena_{case['task_id']}`  ",
        f"Frontier: `{model}` · Jev: `{jev_model}`", "",
        f"Objective: {case['optional']['goal']}", "",
        "| Mode | Success | Reward | Browser actions | Frontier calls | Tokens | Jev decisions | Wall time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("Frontier only", case["baseline"]),
                        ("Frontier + optional Jev", case["optional"])):
        lines.append(
            f"| {label} | {'yes' if item['success'] else 'no'} | {item['reward']:.3f} | "
            f"{len(item['actions'])} | {item['bedrock_calls']} | "
            f"{item['usage'].get('totalTokens', 0):,} | {item['jev_decisions']} | "
            f"{item['total_latency_ms']/1000:.2f}s |"
        )
    lines += ["", "## Optional-Jev trace", "",
              "| # | Controller | Action | Confidence |", "|---:|---|---|---:|"]
    for action in case["optional"]["actions"]:
        confidence = "—" if action["confidence"] is None else f"{action['confidence']:.3f}"
        lines.append(f"| {action['index']+1} | {action['controller']} | `{action['action']}` | {confidence} |")
    lines += ["", "The demo task was fixed in the sample manifest before this run; it was not selected after seeing results.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://127.0.0.1:36005")
    parser.add_argument("--task-ids", default="264,278,300,158,4,763")
    parser.add_argument("--demo-task-id", type=int, default=264)
    parser.add_argument("--bedrock-model", default="us.anthropic.claude-opus-4-7")
    parser.add_argument("--profile", default="bedrock")
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--jev-url", default="http://127.0.0.1:18201")
    parser.add_argument("--jev-model", default="SimpleJev/JevAny-Qwen3.8-27B-LoRA")
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=18)
    parser.add_argument("--max-delegate-steps", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--cache-read-price-per-million", type=float)
    parser.add_argument("--cache-write-price-per-million", type=float)
    parser.add_argument("--out", required=True)
    parser.add_argument("--demo-out")
    args = parser.parse_args()
    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    if len(task_ids) != len(set(task_ids)) or args.demo_task_id not in task_ids:
        parser.error("task ids must be unique and include --demo-task-id")
    client, resolved_model = runtime_client(args.profile, args.region, args.bedrock_model)
    agent = BedrockJevWebArenaAgent(
        client, resolved_model, HTTPDecisionClient(args.jev_url), jev_model=args.jev_model,
        max_output_tokens=args.max_output_tokens, max_turns=args.max_turns,
        max_delegate_steps=args.max_delegate_steps,
        confidence_threshold=args.confidence_threshold,
    )
    output_path = Path(args.out)
    episodes = []
    for index, task_id in enumerate(task_ids):
        modes = ["baseline", "optional"] if index % 2 == 0 else ["optional", "baseline"]
        for mode in modes:
            env = WebArenaEnvClient(args.env_url)
            try:
                result = agent.run(env, task_id=task_id, mode=mode,
                                   max_actions=args.max_actions).as_dict()
                result["status"] = "error" if result["termination_reason"] == "bedrock_backend_error" else "complete"
            except Exception as error:
                result = {
                    "mode": mode, "task_id": task_id, "goal": "", "success": False,
                    "reward": 0.0, "actions": [], "bedrock_calls": 0,
                    "bedrock_failures": 0, "direct_actions": 0,
                    "delegation_calls": 0, "jev_decisions": 0, "jev_failures": 0,
                    "low_confidence_returns": 0, "protocol_repairs": 0,
                    "termination_reason": "backend_error", "usage": {},
                    "bedrock_latency_ms": 0.0, "jev_latency_ms": 0.0,
                    "total_latency_ms": 0.0, "transcript": [], "status": "error",
                    "error_type": type(error).__name__, "error": str(error),
                }
            finally:
                env.close()
            result["environment"] = "webarena"
            result["seed"] = task_id
            episodes.append(result)
            write_json(output_path, {"schema_version": 1, "status": "running",
                                     "benchmark": "WebArena sampled paired evaluation",
                                     "bedrock_model": args.bedrock_model,
                                     "jev_model": args.jev_model, "episodes": episodes})
            print(json.dumps({"task_id": task_id, "mode": mode,
                              "success": result["success"],
                              "frontier_calls": result["bedrock_calls"],
                              "jev_decisions": result["jev_decisions"]}), flush=True)
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[episode["mode"]].append(episode)
    prices = {"input": args.input_price_per_million,
              "output": args.output_price_per_million,
              "cache_read": args.cache_read_price_per_million,
              "cache_write": args.cache_write_price_per_million}
    summaries = {mode: summarize(items, prices) for mode, items in grouped.items()}
    by_task = defaultdict(dict)
    for episode in episodes:
        by_task[episode["task_id"]][episode["mode"]] = episode
    demo = {"task_id": args.demo_task_id, **by_task[args.demo_task_id]}
    output = {
        "schema_version": 1, "status": "complete",
        "benchmark": "WebArena sampled paired evaluation",
        "bedrock_model": args.bedrock_model, "resolved_bedrock_model": resolved_model,
        "jev_model": args.jev_model,
        "config": {"episodes": len(task_ids), "task_ids": task_ids,
                   "demo_task_id": args.demo_task_id, "max_actions": args.max_actions,
                   "max_turns": args.max_turns, "max_delegate_steps": args.max_delegate_steps,
                   "confidence_threshold": args.confidence_threshold},
        "summaries": summaries, "comparison": compare(summaries, episodes),
        "demo_case": demo, "episodes": episodes,
    }
    write_json(output_path, output)
    if args.demo_out:
        demo_path = Path(args.demo_out)
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo_path.write_text(render_demo(demo, args.bedrock_model, args.jev_model), encoding="utf-8")
    print(json.dumps({"summaries": summaries, "comparison": output["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
