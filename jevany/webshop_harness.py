"""Bedrock controller with optional bounded Jev delegation for WebShop."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol

from .agent import action_request
from .api import validate_response


class ConverseClient(Protocol):
    def converse(self, **request) -> dict: ...


DecisionFunction = Callable[[dict], dict]


@dataclass(frozen=True)
class WebShopAction:
    index: int
    controller: str
    action: str
    confidence: float | None
    reward: float
    done: bool
    info: dict


@dataclass(frozen=True)
class WebShopEpisode:
    mode: str
    split: str
    seed: int
    goal: str
    success: bool
    reward: float
    actions: tuple[WebShopAction, ...]
    bedrock_calls: int
    bedrock_failures: int
    direct_actions: int
    search_actions: int
    delegation_calls: int
    jev_decisions: int
    jev_failures: int
    low_confidence_returns: int
    protocol_repairs: int
    termination_reason: str
    usage: dict
    bedrock_latency_ms: float
    jev_latency_ms: float
    total_latency_ms: float
    transcript: tuple[dict, ...]

    def as_dict(self) -> dict:
        value = asdict(self)
        value["actions"] = [asdict(action) for action in self.actions]
        value["transcript"] = list(self.transcript)
        return value


def _observation(value: Any) -> Any:
    return value[0] if isinstance(value, tuple) else value


def _usage_sum(total: dict, current: Mapping | None) -> None:
    current = current or {}
    for key, value in current.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = total.get(key, 0) + value
    detail_tokens = 0
    for detail in current.get("cacheDetails", []) or []:
        if not isinstance(detail, Mapping):
            raise ValueError("Bedrock usage cacheDetails entries must be objects")
        ttl = str(detail.get("ttl", "")).strip().lower()
        ttl = {"pt5m": "5m", "pt1h": "1h", "pt30m": "30m"}.get(ttl, ttl)
        if ttl not in {"5m", "1h", "30m"}:
            raise ValueError(f"unsupported Bedrock cache write TTL: {ttl!r}")
        tokens = detail.get("inputTokens")
        if isinstance(tokens, bool) or not isinstance(tokens, (int, float)) or tokens < 0:
            raise ValueError("Bedrock cacheDetails inputTokens must be non-negative")
        total[f"cacheWriteInputTokens{ttl}"] = total.get(f"cacheWriteInputTokens{ttl}", 0) + tokens
        detail_tokens += tokens
    reported = current.get("cacheWriteInputTokens", 0)
    if detail_tokens and reported != detail_tokens:
        raise ValueError(
            f"Bedrock cacheDetails total {detail_tokens} != cacheWriteInputTokens {reported}"
        )


def _available_actions(env: Any) -> tuple[bool, list[str]]:
    actions = [str(action) for action in env.get_available_actions()]
    search_available = any(action.startswith("search[") for action in actions)
    clicks = [action for action in actions if action.startswith("click[")]
    return search_available, clicks


class BedrockJevWebShopAgent:
    """Solve WebShop while letting the frontier model decide when Jev acts.

    Search queries remain a frontier-model responsibility because their action
    space is open-ended. Jev receives only the finite click choices exposed by
    the current page and may execute a bounded sequence before returning control.
    """

    MODES = {"baseline", "optional"}

    def __init__(
        self,
        client: ConverseClient,
        model: str,
        decide: DecisionFunction,
        *,
        jev_model: str = "jevany-latest",
        max_output_tokens: int = 512,
        max_turns: int = 16,
        max_delegate_steps: int = 4,
        confidence_threshold: float = 0.0,
        history_limit: int = 8,
    ):
        if min(max_output_tokens, max_turns, max_delegate_steps, history_limit) < 1:
            raise ValueError("token, turn, delegation, and history limits must be positive")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self.client = client
        self.model = model
        self.decide = decide
        self.jev_model = jev_model
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns
        self.max_delegate_steps = max_delegate_steps
        self.confidence_threshold = confidence_threshold
        self.history_limit = history_limit

    def _tools(self, search_available: bool, clicks: list[str], mode: str) -> list[dict]:
        tools: list[dict] = []
        if search_available:
            tools.append({"toolSpec": {
                "name": "search",
                "description": (
                    "Search the shop. Use concise product keywords; omit size, color, and other "
                    "options that can be selected on the product page. Only you can choose queries."
                ),
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
                    "required": ["query"],
                    "additionalProperties": False,
                }},
            }})
        if clicks:
            keys = [str(index) for index in range(len(clicks))]
            descriptions = ", ".join(f"{key}={clicks[int(key)]}" for key in keys)
            tools.append({"toolSpec": {
                "name": "click",
                "description": f"Execute one click. Current keys: {descriptions}.",
                "inputSchema": {"json": {
                    "type": "object",
                    "properties": {"action_key": {"type": "string", "enum": keys}},
                    "required": ["action_key"],
                    "additionalProperties": False,
                }},
            }})
            if mode == "optional":
                tools.append({"toolSpec": {
                    "name": "delegate_clicks",
                    "description": (
                        "Delegate several routine finite-choice page clicks to Jev. Jev sees the "
                        "fresh page after every click and returns control on completion, low confidence, "
                        "a search page, or the requested horizon. Retain control for ambiguous choices."
                    ),
                    "inputSchema": {"json": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "integer", "minimum": 1,
                                "maximum": self.max_delegate_steps,
                            },
                            "subgoal": {
                                "type": "string", "minLength": 1, "maxLength": 500,
                            },
                        },
                        "required": ["steps", "subgoal"],
                        "additionalProperties": False,
                    }},
                }})
        return tools

    @staticmethod
    def _system(mode: str) -> str:
        common = (
            "You are shopping in a text web environment. Satisfy the user's product constraints and "
            "complete the purchase within the action budget. Search uses free text; all click actions "
            "must use a key from the current page. Inspect every tool result as the trusted new state. "
            "Call exactly one state-changing tool per response and give one short public rationale."
        )
        if mode == "optional":
            return common + (
                " You may autonomously delegate routine finite-choice clicking to Jev. Keep control of "
                "search queries and novel, ambiguous, or high-risk choices. Give Jev a concrete subgoal "
                "that preserves the user's product, option, price, and completion constraints."
            )
        return common

    def run(
        self,
        env: Any,
        *,
        seed: int,
        split: str = "test",
        mode: str = "optional",
        max_actions: int = 10,
    ) -> WebShopEpisode:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if max_actions < 1:
            raise ValueError("max_actions must be positive")

        started = time.perf_counter()
        observation = _observation(env.reset(seed=seed, mode=split))
        goal = str(env.get_instruction_text())
        actions_taken: list[WebShopAction] = []
        history: list[dict] = []
        transcript: list[dict] = []
        usage: dict = {}
        total_reward = 0.0
        success = done = False
        bedrock_calls = bedrock_failures = direct_actions = search_actions = 0
        delegation_calls = jev_decisions = jev_failures = low_confidence_returns = 0
        protocol_repairs = 0
        termination_reason = "running"
        bedrock_latency = jev_latency = 0.0

        def execute(action: str, controller: str, confidence: float | None) -> dict:
            nonlocal observation, total_reward, success, done, direct_actions, search_actions
            next_observation, reward, terminal, raw_info = env.step(action)
            info = dict(raw_info or {})
            observation = _observation(next_observation)
            done = bool(terminal)
            success = success or bool(info.get("success")) or float(reward) >= 1.0
            total_reward += float(reward)
            if controller == "bedrock":
                direct_actions += 1
                if action.startswith("search["):
                    search_actions += 1
            actions_taken.append(WebShopAction(
                len(actions_taken), controller, action, confidence,
                float(reward), done, info,
            ))
            history.append({
                "controller": controller,
                "action": action,
                "reward": float(reward),
                "effective": info.get("action_is_effective"),
            })
            return {
                "action": action,
                "reward": float(reward),
                "done": done,
                "success": success,
                "effective": info.get("action_is_effective"),
                "observation": observation,
            }

        def delegate(steps: int, subgoal: str) -> dict:
            nonlocal jev_decisions, jev_failures, jev_latency, low_confidence_returns
            delegated = []
            reason = "horizon"
            for _ in range(min(steps, self.max_delegate_steps, max_actions - len(actions_taken))):
                search_available, clicks = _available_actions(env)
                if done:
                    reason = "terminal"
                    break
                if search_available or not clicks:
                    reason = "frontier_control_required"
                    break
                choices = {str(index): action for index, action in enumerate(clicks)}
                request = action_request(
                    f"Shopping goal: {goal}\nDelegated subgoal: {subgoal}",
                    observation,
                    choices,
                    history[-self.history_limit:],
                    self.jev_model,
                )
                before = time.perf_counter()
                jev_decisions += 1
                try:
                    response = validate_response(request, self.decide(request))
                except Exception as error:
                    jev_failures += 1
                    reason = "backend_error"
                    return {
                        "subgoal": subgoal,
                        "executed": len(delegated),
                        "stop_reason": reason,
                        "error_type": type(error).__name__,
                        "steps": delegated,
                        "current_observation": observation,
                        "success": success,
                        "done": done,
                    }
                finally:
                    jev_latency += (time.perf_counter() - before) * 1000
                answer = response["answers"]["action"]
                key = str(answer["choice"])
                confidence = float(answer["confidence"])
                if confidence < self.confidence_threshold:
                    low_confidence_returns += 1
                    reason = "low_confidence"
                    break
                if key not in choices:
                    raise ValueError(f"decision backend returned unknown click {key!r}")
                delegated.append(execute(choices[key], "jev", confidence))
                if done:
                    reason = "terminal"
                    break
                if delegated[-1]["reward"] < 0:
                    reason = "negative_reward"
                    break
                if delegated[-1].get("effective") is False:
                    reason = "ineffective_action"
                    break
            return {
                "subgoal": subgoal,
                "executed": len(delegated),
                "stop_reason": reason,
                "steps": delegated,
                "current_observation": observation,
                "success": success,
                "done": done,
            }

        messages = [{"role": "user", "content": [{"text": (
            f"Shopping goal:\n{goal}\n\nInitial page:\n{observation}\n\nStart acting."
        )}]}]
        repair_turns = 0
        while not done and len(actions_taken) < max_actions and bedrock_calls < self.max_turns:
            search_available, clicks = _available_actions(env)
            tools = self._tools(search_available, clicks, mode)
            if not tools:
                break
            request = {
                "modelId": self.model,
                "system": [{"text": self._system(mode)}],
                "messages": messages,
                "toolConfig": {"tools": tools},
                "inferenceConfig": {"maxTokens": self.max_output_tokens},
            }
            before = time.perf_counter()
            bedrock_calls += 1
            try:
                response = self.client.converse(**request)
            except Exception as error:
                bedrock_failures += 1
                transcript.append({
                    "turn": bedrock_calls - 1,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                termination_reason = "bedrock_backend_error"
                break
            finally:
                bedrock_latency += (time.perf_counter() - before) * 1000
            _usage_sum(usage, response.get("usage"))
            message = response.get("output", {}).get("message", {})
            content = message.get("content", [])
            messages.append({"role": "assistant", "content": content})
            public_text = "\n".join(block["text"] for block in content if "text" in block)
            tool_uses = [block["toolUse"] for block in content if "toolUse" in block]
            transcript.append({
                "turn": bedrock_calls - 1,
                "text": public_text,
                "tool": tool_uses[0]["name"] if len(tool_uses) == 1 else None,
                "tool_input": tool_uses[0].get("input") if len(tool_uses) == 1 else None,
            })
            if len(tool_uses) != 1:
                repair_turns += 1
                protocol_repairs += 1
                if repair_turns > 2:
                    termination_reason = "model_protocol_error"
                    break
                if tool_uses and all(call.get("toolUseId") for call in tool_uses):
                    messages.append({"role": "user", "content": [{"toolResult": {
                        "toolUseId": call["toolUseId"],
                        "content": [{"json": {"error": "Call exactly one state-changing tool."}}],
                        "status": "error",
                    }} for call in tool_uses]})
                else:
                    messages.append({"role": "user", "content": [{"text": (
                        "The shop is not terminal. Call exactly one available tool."
                    )}]})
                continue
            repair_turns = 0
            call = tool_uses[0]
            tool_input = call.get("input") or {}
            try:
                if call["name"] == "search" and search_available:
                    query = tool_input.get("query")
                    if not isinstance(query, str) or not query.strip() or len(query) > 200:
                        raise ValueError("query must be a non-empty string of at most 200 characters")
                    result = execute(f"search[{query.strip()}]", "bedrock", None)
                elif call["name"] == "click" and clicks:
                    key = str(tool_input.get("action_key"))
                    if not key.isdigit() or int(key) >= len(clicks):
                        raise ValueError(f"unknown click key {key!r}")
                    result = execute(clicks[int(key)], "bedrock", None)
                elif call["name"] == "delegate_clicks" and mode == "optional":
                    steps = tool_input.get("steps")
                    if (
                        isinstance(steps, bool)
                        or not isinstance(steps, int)
                        or not 1 <= steps <= self.max_delegate_steps
                    ):
                        raise ValueError(
                            f"steps must be an integer in [1, {self.max_delegate_steps}]"
                        )
                    subgoal = tool_input.get("subgoal")
                    if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal) > 500:
                        raise ValueError("subgoal must be a non-empty string of at most 500 characters")
                    delegation_calls += 1
                    result = delegate(steps, subgoal.strip())
                else:
                    raise ValueError(f"unavailable tool {call.get('name')!r}")
                status = "success"
            except (KeyError, TypeError, ValueError) as error:
                result, status = {"error": str(error), "observation": observation}, "error"
                protocol_repairs += 1
            if not call.get("toolUseId"):
                termination_reason = "model_protocol_error"
                break
            messages.append({"role": "user", "content": [{"toolResult": {
                "toolUseId": call["toolUseId"],
                "content": [{"json": result}],
                "status": status,
            }}]})

        if termination_reason == "running":
            termination_reason = (
                "terminal_success" if success else "terminal_failure" if done
                else "max_actions" if len(actions_taken) >= max_actions else "max_turns"
            )
        return WebShopEpisode(
            mode, split, seed, goal, success, total_reward, tuple(actions_taken),
            bedrock_calls, bedrock_failures, direct_actions, search_actions, delegation_calls,
            jev_decisions, jev_failures, low_confidence_returns, protocol_repairs,
            termination_reason, usage, bedrock_latency,
            jev_latency, (time.perf_counter() - started) * 1000,
            tuple(transcript),
        )
