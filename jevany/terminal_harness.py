"""Validated candidate-policy protocol for Terminal-Bench delegation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .agent import action_request


PROPOSE_TOOL = "propose_terminal_options"
COMPLETE_TOOL = "complete_task"
_TOR = re.compile(r"(?i)(?<![A-Za-z0-9_])(?:tor|torsocks)(?![A-Za-z0-9_])")
_ROOT_SCAN = re.compile(
    r"(?i)(?:^|[;&|\n]\s*)(?:sudo\s+)?(?:/usr/bin/|/bin/)?find\s+/(?:\s|$)"
)
_SHELL_FRAGMENT = re.compile(r"['\"\\]|\$\{[^}\n]*\}")


FRONTIER_TOOLS = [
    {"toolSpec": {
        "name": PROPOSE_TOOL,
        "description": (
            "Reason about the terminal state and propose 1-4 bounded next-command options "
            "(2-4 for a routine decision; a precise non-routine edit may be a singleton). "
            "Routine, composable options may be delegated to Jev for a short sequence; "
            "ambiguous edits and irreversible decisions remain under frontier control."
        ),
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "reasoning_summary": {"type": "string", "minLength": 1, "maxLength": 2000},
                "subgoal": {"type": "string", "minLength": 1, "maxLength": 1000},
                "routine": {"type": "boolean"},
                "composable_sequence": {"type": "boolean"},
                "delegate_steps": {"type": "integer", "minimum": 1, "maximum": 4},
                "frontier_choice": {"type": "integer", "minimum": 0, "maximum": 3},
                "candidates": {
                    "type": "array", "minItems": 1, "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "minLength": 1, "maxLength": 200},
                            "command": {"type": "string", "minLength": 1, "maxLength": 8000},
                            "expected": {"type": "string", "minLength": 1, "maxLength": 500},
                            "wait_seconds": {"type": "number", "minimum": 0.1, "maximum": 60},
                        },
                        "required": ["label", "command", "expected", "wait_seconds"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["reasoning_summary", "subgoal", "routine", "composable_sequence",
                         "delegate_steps", "frontier_choice", "candidates"],
            "additionalProperties": False,
        }},
    }},
    {"toolSpec": {
        "name": COMPLETE_TOOL,
        "description": "Declare the task complete only after the solution has been verified.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
                "verification": {"type": "string", "minLength": 1, "maxLength": 2000},
            },
            "required": ["summary", "verification"],
            "additionalProperties": False,
        }},
    }},
]


SYSTEM_PROMPT = """You are the frontier reasoning model responsible for completing a
Terminal-Bench task. You own diagnosis, planning, edits, verification, and the final
completion decision. Every response must call exactly one provided tool.

For propose_terminal_options, provide 2-4 concrete non-interactive terminal options.
The harness executes commands verbatim in a persistent shell. Use literal <WAIT> to
observe a still-running command and <CTRL_C> to interrupt it. Never use Tor or torsocks.

Set routine=true for basic inspection, navigation, compilation, and test decisions that
a cheaper local model can safely select. Most candidate lists are alternatives: set
composable_sequence=false and Jev will choose exactly one. Set composable_sequence=true
only when every candidate is useful at most once, the candidates are safe in any order,
and executing two of them before you reason again cannot derail the task. Never use a
composable sequence for edits or competing approaches. Set routine=false for precise edits,
ambiguous strategy choices, credential/network decisions, or irreversible actions;
frontier_choice is then executed directly. Even for routine decisions, frontier_choice is
the safe fallback if Jev is unavailable or below its confidence threshold.
Routine decisions need at least two real options. A precise non-routine edit may contain
one candidate; do not invent a second unsafe edit merely to make the list plural.
Do not relabel a basic inspection, navigation, wait/poll, compilation, or verification
decision as non-routine to avoid generating alternatives: those basic decisions must be
routine=true with 2-4 real options so Jev gets an actual choice surface.

Work only from the user instruction and visible workspace. Do not search for hidden tests,
grader files, task metadata, or solution files. An empty /app directory may be intentional.
Never recursively scan the filesystem root; inspect focused paths such as /app instead.

Do not mark complete merely because a command succeeded. Inspect the task, implement the
solution, run relevant verification, and use complete_task only when the requested artifact
or behavior is genuinely finished."""


@dataclass(frozen=True)
class TerminalCandidate:
    index: int
    label: str
    command: str
    expected: str
    wait_seconds: float

    @property
    def choice_text(self) -> str:
        rendered = self.command.replace("\n", " ⏎ ")
        return f"{self.label}: {rendered} (expect: {self.expected})"


@dataclass(frozen=True)
class TerminalProposal:
    reasoning_summary: str
    subgoal: str
    routine: bool
    composable_sequence: bool
    delegate_steps: int
    frontier_choice: int
    candidates: tuple[TerminalCandidate, ...]


@dataclass(frozen=True)
class TerminalCompletion:
    summary: str
    verification: str


def _text(value: Any, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty string of at most {maximum} characters")
    return value.strip()


def validate_command(command: Any) -> str:
    command = _text(command, "command", 8000)
    # Also inspect a conservative de-obfuscated form so trivial shell token
    # splitting (for example ``t'o'r`` or ``t\\or``) cannot bypass the local
    # policy gate. This complements, rather than replaces, the system prompt.
    deobfuscated = _SHELL_FRAGMENT.sub("", command)
    if _TOR.search(command) or _TOR.search(deobfuscated):
        raise ValueError("Tor and torsocks are prohibited")
    if _ROOT_SCAN.search(command):
        raise ValueError("recursive scans of the filesystem root are prohibited")
    if "\x00" in command:
        raise ValueError("command contains a NUL byte")
    return command


def delegation_limit(proposal: TerminalProposal, harness_level: int) -> int:
    """Return the maximum Jev-selected commands for one frontier decision.

    The levels make the harness frontier explicit and reproducible:
    0 = frontier only; 1 = non-composable routine choices; 2 = all routine
    choices, one command; 3 = safe composable routine batches, capped at two.
    """
    if isinstance(harness_level, bool) or not isinstance(harness_level, int) \
            or not 0 <= harness_level <= 3:
        raise ValueError("harness_level must be an integer in [0, 3]")
    if harness_level == 0 or not proposal.routine:
        return 0
    if harness_level == 1 and proposal.composable_sequence:
        return 0
    if harness_level <= 2 or not proposal.composable_sequence:
        return 1
    return min(proposal.delegate_steps, 2, len(proposal.candidates))


def delegate_at_rate(eligible_index: int, delegation_rate: float) -> bool:
    """Deterministically sample a fraction of eligible decisions.

    ``eligible_index`` is one-based. At 0.5 this selects decisions 2, 4, ...;
    at 1.0 it selects every eligible decision. This avoids RNG confounds in
    paired benchmark runs.
    """
    if isinstance(eligible_index, bool) or not isinstance(eligible_index, int) or eligible_index < 1:
        raise ValueError("eligible_index must be a positive integer")
    if isinstance(delegation_rate, bool) or not isinstance(delegation_rate, (int, float)) \
            or not 0 <= delegation_rate <= 1:
        raise ValueError("delegation_rate must be in [0, 1]")
    return int(eligible_index * delegation_rate) > int((eligible_index - 1) * delegation_rate)


def parse_frontier_response(response: Mapping[str, Any]) -> tuple[str, TerminalProposal | TerminalCompletion]:
    content = response.get("output", {}).get("message", {}).get("content", [])
    calls = [block.get("toolUse") for block in content if isinstance(block, Mapping) and block.get("toolUse")]
    if len(calls) != 1:
        raise ValueError(f"expected exactly one tool call, received {len(calls)}")
    call = calls[0]
    name = call.get("name")
    values = call.get("input")
    if not isinstance(values, Mapping):
        raise ValueError("tool input must be an object")
    if name == COMPLETE_TOOL:
        return str(call.get("toolUseId", "")), TerminalCompletion(
            _text(values.get("summary"), "summary", 2000),
            _text(values.get("verification"), "verification", 2000),
        )
    if name != PROPOSE_TOOL:
        raise ValueError(f"unexpected tool {name!r}")
    routine = values.get("routine")
    if not isinstance(routine, bool):
        raise ValueError("routine must be boolean")
    raw_candidates = values.get("candidates")
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 4:
        raise ValueError("candidates must contain 1-4 entries")
    candidates = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError("each candidate must be an object")
        wait = raw.get("wait_seconds")
        if isinstance(wait, bool) or not isinstance(wait, (int, float)) or not 0.1 <= wait <= 60:
            raise ValueError("wait_seconds must be in [0.1, 60]")
        label = _text(raw.get("label"), "label", 200)
        raw_command = raw.get("command")
        if (not isinstance(raw_command, str) or not raw_command.strip()) and "wait" in label.lower():
            raw_command = "<WAIT>"
        candidates.append(TerminalCandidate(
            index=index,
            label=label,
            command=validate_command(raw_command),
            expected=_text(raw.get("expected"), "expected", 500),
            wait_seconds=float(wait),
        ))
    choice = values.get("frontier_choice")
    if isinstance(choice, bool) or not isinstance(choice, int) or not 0 <= choice < len(candidates):
        raise ValueError("frontier_choice does not identify a candidate")
    steps = values.get("delegate_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 4:
        raise ValueError("delegate_steps must be in [1, 4]")
    composable = values.get("composable_sequence")
    if not isinstance(composable, bool):
        raise ValueError("composable_sequence must be boolean")
    if not routine and composable:
        raise ValueError("non-routine decisions cannot be composable sequences")
    # A routine singleton is not a genuine decision surface for Jev. Keep the
    # contract strict so basic decisions are regenerated with real options;
    # precise non-routine edits may still be singletons.
    if routine and len(candidates) == 1:
        raise ValueError("routine decisions must contain at least two candidates")
    return str(call.get("toolUseId", "")), TerminalProposal(
        reasoning_summary=_text(values.get("reasoning_summary"), "reasoning_summary", 2000),
        subgoal=_text(values.get("subgoal"), "subgoal", 1000),
        routine=routine,
        composable_sequence=composable,
        delegate_steps=steps,
        frontier_choice=choice,
        candidates=tuple(candidates),
    )


def jev_terminal_request(
    task: str,
    proposal: TerminalProposal,
    observation: str,
    remaining: list[TerminalCandidate],
    history: list[dict],
    model: str,
) -> dict:
    choices = {str(candidate.index): candidate.choice_text for candidate in remaining}
    return action_request(
        f"Terminal task: {task}\nCurrent subgoal: {proposal.subgoal}\n"
        f"Frontier reasoning summary: {proposal.reasoning_summary}",
        observation,
        choices,
        history,
        model,
    )
