#!/usr/bin/env python3
"""Render cache-busted, animated decision-process demos for the README."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

import render_agent_harness_demo as ui


def header(draw, badge: str, title: str, subtitle: str, active: int) -> None:
    ui.label(draw, (44, 38), badge)
    draw.text((44, 78), title, fill=ui.TEXT, font=ui.font(27, True))
    draw.text((44, 113), subtitle, fill=ui.MUTED, font=ui.font(15))
    steps = ("GOAL", "COMPARE", "SELECT", "EXECUTE", "VERIFY", "IMPACT")
    for index, name in enumerate(steps):
        x = 44 + index * 146
        color = ui.ACCENT if index == active else ui.BORDER
        fill = "#0b2426" if index == active else ui.PANEL_2
        draw.rounded_rectangle((x, 151, x + 130, 184), 10, fill=fill, outline=color, width=2 if index == active else 1)
        draw.text((x + 14, 160), name, fill=ui.TEXT if index == active else ui.MUTED, font=ui.font(11, index == active))


def candidate_card(draw, x: int, title: str, consequence: str, status: str) -> None:
    palette = {
        "candidate": (ui.PANEL_2, ui.BORDER, ui.MUTED, "OPTION"),
        "focus": ("#14223a", ui.BLUE, ui.BLUE, "EVALUATE"),
        "selected": ("#0b2426", ui.ACCENT, ui.ACCENT, "SELECT"),
        "rejected": ("#29161f", ui.RED, ui.RED, "REJECT"),
    }
    fill, outline, accent, badge = palette[status]
    draw.rounded_rectangle((x, 207, x + 276, 330), 14, fill=fill, outline=outline, width=3 if status in {"focus", "selected"} else 1)
    draw.text((x + 17, 224), title, fill=ui.TEXT, font=ui.font(16, status == "selected"))
    draw.text((x + 17, 263), consequence, fill=accent, font=ui.font(13, True))
    draw.text((x + 17, 296), badge, fill=accent, font=ui.font(11, True))


def terminal_frame(phase: str, focus: int | None = None, typed: float = 1.0) -> Image.Image:
    image, draw = ui.canvas()
    active = {"goal": 0, "compare": 1, "select": 2, "execute": 3, "output": 3, "verify": 4}[phase]
    titles = {
        "goal": ("Recover rows from a damaged SQLite page", "Terminal-Bench 2 · sqlite-db-truncate"),
        "compare": ("Compare three real commands", "Criterion: obtain bytes even when SQLite metadata is damaged"),
        "select": ("Jev selects raw-page inspection", "The other commands cannot expose the same row bytes at this step"),
        "execute": ("Execute the selected command", "The terminal runs exactly one candidate"),
        "output": ("The environment returns useful bytes", "Page type, cell count, and offsets become visible"),
        "verify": ("LLM parses; verifier checks", "Jev selects the inspection · LLM owns recovery and completion"),
    }
    header(draw, "TERMINAL-BENCH", *titles[phase], active)

    statuses = ["candidate", "candidate", "candidate"]
    if phase == "compare" and focus is not None:
        statuses[focus] = "focus"
    elif phase in {"select", "execute", "output", "verify"}:
        statuses = ["selected", "rejected", "rejected"]
    candidate_card(draw, 44, "A · od raw page", "works on raw bytes", statuses[0])
    candidate_card(draw, 342, "B · sqlite schema", "needs readable metadata", statuses[1])
    candidate_card(draw, 640, "C · check tools", "returns paths, not DB bytes", statuses[2])

    draw.rounded_rectangle((44, 351, 916, 480), 14, fill="#080e1b", outline=ui.BORDER)
    if phase == "goal":
        draw.text((68, 371), "DECISION GOAL", fill=ui.BLUE, font=ui.font(12, True))
        draw.text((68, 402), "Find recoverable records without trusting a valid SQLite header.", fill=ui.TEXT, font=ui.font(18, True))
        draw.text((68, 440), "Three commands are valid; only one executes.", fill=ui.MUTED, font=ui.font(14))
    elif phase == "compare":
        labels = ("raw bytes -> parser input", "schema -> may depend on metadata", "tool paths -> no record data")
        draw.text((68, 370), "FOCUS MOVES ACROSS ALL OPTIONS", fill=ui.BLUE, font=ui.font(12, True))
        for index, value in enumerate(labels):
            color = ui.BLUE if index == focus else ui.MUTED
            prefix = ">" if index == focus else "·"
            draw.text((72, 399 + index * 25), f"{prefix} {value}", fill=color, font=ui.font(14, index == focus))
    elif phase == "select":
        draw.text((68, 370), "JEV CHOICE", fill=ui.ACCENT, font=ui.font(12, True))
        draw.text((68, 401), "A · od raw page", fill=ui.TEXT, font=ui.font(23, True))
        draw.text((350, 406), "confidence 0.81", fill=ui.ACCENT, font=ui.font(17, True))
        draw.text((68, 444), "Reason shown: raw bytes remain inspectable even when schema access is unreliable.", fill=ui.MUTED, font=ui.font(13))
    elif phase == "execute":
        command = "$ od -A x -t x1z -v /app/trunc.db | head -80"
        chars = max(1, round(len(command) * typed))
        draw.text((68, 373), "TERMINAL", fill=ui.ACCENT, font=ui.font(12, True))
        draw.text((68, 410), command[:chars] + ("▋" if typed < 1 else ""), fill=ui.TEXT, font=ui.font(16, True, mono=True))
        draw.text((68, 450), "selected command is running...", fill=ui.MUTED, font=ui.font(13))
    elif phase == "output":
        draw.text((68, 368), "REAL TERMINAL OUTPUT", fill=ui.ACCENT, font=ui.font(12, True))
        draw.text((68, 396), "000000  0d 00 00 00 0a 0f 49 00 0f f0 0f df ...", fill=ui.TEXT, font=ui.font(15, True, mono=True))
        draw.text((68, 427), "0x0d leaf page", fill=ui.ACCENT, font=ui.font(16, True))
        draw.text((275, 427), "10 cells", fill=ui.ACCENT, font=ui.font(16, True))
        draw.text((420, 427), "content @ 0x0f49", fill=ui.ACCENT, font=ui.font(16, True))
        draw.text((68, 457), "Environment changed from unknown bytes -> structured recovery evidence", fill=ui.MUTED, font=ui.font(13))
    else:
        draw.text((68, 368), "LLM RECOVERY", fill=ui.BLUE, font=ui.font(12, True))
        draw.text((68, 398), "parse cell pointers  ->  recover 10 rows  ->  write recover.json", fill=ui.TEXT, font=ui.font(16, True))
        draw.rounded_rectangle((68, 435, 430, 469), 9, fill="#0b2426", outline=ui.ACCENT, width=2)
        draw.text((88, 443), "INDEPENDENT VERIFIER · PASS", fill=ui.ACCENT, font=ui.font(13, True))
    return image


def terminal_impact_frame() -> Image.Image:
    image, draw = ui.canvas()
    header(draw, "TERMINAL-BENCH", "Same task and reward; less LLM work", "The selected inspections replace routine frontier calls", 5)
    columns = (
        (44, "LLM ONLY", ui.MUTED, "diagnose -> inspect -> repeat -> parse", "15 LLM calls", "187.9 s", "reward 1"),
        (490, "LLM + JEV", ui.ACCENT, "Jev raw page -> bytes -> LLM parse", "8 LLM calls", "144.7 s", "reward 1"),
    )
    for x, title, color, path, calls, elapsed, reward in columns:
        draw.rounded_rectangle((x, 213, x + 426, 431), 17, fill=ui.PANEL_2 if x == 44 else "#0b2426", outline=color, width=2)
        draw.text((x + 24, 235), title, fill=color, font=ui.font(15, True))
        draw.text((x + 24, 273), path, fill=ui.MUTED, font=ui.font(12, True))
        draw.text((x + 24, 309), calls, fill=ui.TEXT, font=ui.font(24, True))
        draw.text((x + 24, 350), elapsed, fill=ui.TEXT, font=ui.font(24, True))
        draw.text((x + 24, 393), reward, fill=ui.ACCENT, font=ui.font(17, True))
    draw.text((44, 459), "Impact: 7 fewer LLM calls · 43.2 s faster · success preserved", fill=ui.ACCENT, font=ui.font(16, True))
    return image


def draw_grid(draw, player: tuple[float, float], reached: bool = False) -> None:
    grid_x, grid_y, cell = 72, 220, 68
    for row in range(4):
        for col in range(4):
            x, y = grid_x + col * cell, grid_y + row * cell
            fill = "#0b2426" if (row, col) == (0, 3) else ui.PANEL_2
            if (row, col) == (3, 3):
                fill = "#29161f"
            draw.rounded_rectangle((x, y, x + 56, y + 56), 10, fill=fill, outline=ui.BORDER)
            mark = "G" if (row, col) == (0, 3) else "O" if (row, col) == (3, 3) else ""
            if mark:
                color = ui.ACCENT if mark == "G" else ui.RED
                draw.text((x + 18, y + 13), mark, fill=color, font=ui.font(24, True, mono=True))
    row, col = player
    cx = grid_x + col * cell + 28
    cy = grid_y + row * cell + 28
    color = ui.ACCENT if reached else ui.BLUE
    draw.ellipse((cx - 21, cy - 21, cx + 21, cy + 21), fill=color, outline="#dbeafe", width=2)
    draw.text((cx - 9, cy - 14), "✓" if reached else "P", fill="#07111d", font=ui.font(20, True))


def lake_card(draw, x: int, y: int, action: str, note: str, status: str) -> None:
    palette = {
        "candidate": (ui.PANEL_2, ui.BORDER, ui.MUTED, "OPTION"),
        "focus": ("#14223a", ui.BLUE, ui.BLUE, "CHECK"),
        "selected": ("#0b2426", ui.ACCENT, ui.ACCENT, "SELECT"),
        "rejected": ("#29161f", ui.RED, ui.RED, "REJECT"),
        "alternate": ("#292313", ui.AMBER, ui.AMBER, "ALT"),
    }
    fill, outline, ink, badge = palette[status]
    draw.rounded_rectangle((x, y, x + 216, y + 73), 12, fill=fill, outline=outline, width=3 if status in {"focus", "selected"} else 1)
    draw.text((x + 15, y + 11), action, fill=ui.TEXT, font=ui.font(16, status == "selected"))
    draw.text((x + 15, y + 43), note, fill=ink, font=ui.font(12, True))
    draw.text((x + 151, y + 13), badge, fill=ink, font=ui.font(9, True))


LAKE_POSITIONS = ((1, 0), (1, 1), (1, 2), (1, 3))
LAKE_ACTIONS = ("Right", "Right", "Right", "Up")
LAKE_CONF = (0.99, 0.99, 0.82, 1.0)
LAKE_REASON = (
    {"Left": "wall", "Down": "away", "Right": "route", "Up": "alternate"},
    {"Left": "backtrack", "Down": "away", "Right": "route", "Up": "alternate"},
    {"Left": "backtrack", "Down": "away", "Right": "route", "Up": "alternate"},
    {"Left": "backtrack", "Down": "away", "Right": "wall", "Up": "goal"},
)


def lake_frame(stage: int, mode: str, player: tuple[float, float] | None = None) -> Image.Image:
    image, draw = ui.canvas()
    pos = player or LAKE_POSITIONS[stage]
    active = 1 if mode in {"candidate", "focus"} else 2 if mode == "selected" else 3
    title = "Compare four directions" if mode in {"candidate", "focus"} else f"Jev selects {LAKE_ACTIONS[stage]}"
    subtitle = f"Player {LAKE_POSITIONS[stage]}  ·  Goal (0, 3)  ·  Step {stage + 1}/4"
    header(draw, "FROZENLAKE", title, subtitle, active)
    draw_grid(draw, pos)

    draw.rounded_rectangle((390, 207, 916, 468), 15, fill="#080e1b", outline=ui.BORDER)
    for index, action in enumerate(("Left", "Down", "Right", "Up")):
        note = LAKE_REASON[stage][action]
        if mode == "candidate":
            status, note = "candidate", "evaluate"
        elif mode == "focus":
            status = "focus" if action == LAKE_ACTIONS[stage] else "candidate"
            note = "toward delegated route" if status == "focus" else "compare"
        elif action == LAKE_ACTIONS[stage]:
            status = "selected"
        elif note == "alternate":
            status = "alternate"
        else:
            status = "rejected"
        lake_card(draw, 414 + (index % 2) * 238, 228 + (index // 2) * 88, action, note, status)
    if mode == "selected":
        draw.text((414, 415), f"confidence {LAKE_CONF[stage]:.0%}  -> execute", fill=ui.ACCENT, font=ui.font(15, True))
    elif mode == "moving":
        draw.text((414, 415), "ENVIRONMENT STATE UPDATES", fill=ui.BLUE, font=ui.font(14, True))
    else:
        draw.text((414, 415), "Goal + current state + 4 consequences", fill=ui.MUTED, font=ui.font(14))
    return image


def lake_goal_frame() -> Image.Image:
    image, draw = ui.canvas()
    header(draw, "FROZENLAKE", "The selected route reaches the goal", "Every Jev action changed the environment state", 4)
    draw_grid(draw, (0, 3), reached=True)
    draw.rounded_rectangle((390, 213, 916, 452), 16, fill="#0b2426", outline=ui.ACCENT, width=2)
    draw.text((426, 246), "ENVIRONMENT FEEDBACK", fill=ui.ACCENT, font=ui.font(13, True))
    draw.text((426, 294), "Right -> Right -> Right -> Up", fill=ui.TEXT, font=ui.font(23, True))
    draw.text((426, 350), "GOAL REACHED", fill=ui.ACCENT, font=ui.font(31, True))
    draw.text((426, 405), "reward 1  ·  4/4 actions effective", fill=ui.TEXT, font=ui.font(16, True))
    return image


def lake_impact_frame() -> Image.Image:
    image, draw = ui.canvas()
    header(draw, "FROZENLAKE", "Same route; three fewer LLM calls", "Paired seed 3000 · identical start, goal, and reward", 5)
    rows = (
        ("LLM ONLY", "Right¹ -> Right² -> Right³ -> Up⁴", "4 LLM calls", "19.7 s", ui.MUTED),
        ("LLM + JEV", "1 LLM plan -> 4 Jev state decisions", "1 LLM call", "16.7 s", ui.ACCENT),
    )
    for index, (name, path, calls, elapsed, color) in enumerate(rows):
        y = 216 + index * 112
        draw.rounded_rectangle((44, y, 916, y + 92), 14, fill="#0b2426" if index else ui.PANEL_2, outline=color, width=2)
        draw.text((68, y + 15), name, fill=color, font=ui.font(14, True))
        draw.text((225, y + 15), path, fill=ui.TEXT, font=ui.font(16, True))
        draw.text((708, y + 15), calls, fill=ui.TEXT, font=ui.font(15, True))
        draw.text((708, y + 50), elapsed, fill=ui.MUTED, font=ui.font(14))
    draw.text((44, 462), "Impact: reward 1 -> 1  ·  calls 4 -> 1  ·  tokens 2,338 -> 663", fill=ui.ACCENT, font=ui.font(16, True))
    return image


def webshop_frame(phase: str, focus: int | None = None) -> Image.Image:
    image, draw = ui.canvas()
    group = "color" if phase.startswith("color") else "size"
    active = 0 if phase == "goal" else 1 if "compare" in phase else 2 if "select" in phase else 3 if "execute" in phase else 4
    titles = {
        "goal": ("Buy the matching blazer", "Goal: z-dark green · small · under $80"),
        "color_compare": ("LLM generates three color candidates", "Criterion: match the exact requested color"),
        "color_select": ("Jev selects z-dark green", "The alternatives are different valid colors"),
        "color_execute": ("Execute the color choice", "Hidden product state records the selection"),
        "size_compare": ("LLM generates three size candidates", "Criterion: match the exact requested size"),
        "size_select": ("Jev selects small", "The alternatives are different valid sizes"),
        "size_execute": ("Execute the size choice", "Both requested options are now selected"),
        "buy": ("LLM keeps completion control", "Buy Now is never included in the Jev menu"),
        "success": ("Purchase succeeds", "Environment verifier returns reward 1"),
    }
    header(draw, "WEBSHOP", *titles[phase], active)

    draw.rounded_rectangle((44, 210, 330, 462), 15, fill=ui.PANEL_2, outline=ui.BORDER)
    draw.text((66, 232), "TARGET PRODUCT", fill=ui.BLUE, font=ui.font(12, True))
    draw.text((66, 270), "Women's blazer", fill=ui.TEXT, font=ui.font(20, True))
    draw.text((66, 312), "long sleeve", fill=ui.MUTED, font=ui.font(14))
    draw.text((66, 345), "z-dark green", fill=ui.ACCENT, font=ui.font(17, True))
    draw.text((66, 379), "size small", fill=ui.ACCENT, font=ui.font(17, True))
    draw.text((66, 418), "price < $80", fill=ui.TEXT, font=ui.font(15, True))

    draw.rounded_rectangle((354, 207, 916, 468), 15, fill="#080e1b", outline=ui.BORDER)
    if phase == "goal":
        draw.text((382, 238), "LLM PLAN", fill=ui.BLUE, font=ui.font(12, True))
        draw.text((382, 279), "1. Search and open a matching product", fill=ui.TEXT, font=ui.font(17, True))
        draw.text((382, 323), "2. Generate bounded option menus", fill=ui.TEXT, font=ui.font(17, True))
        draw.text((382, 367), "3. Jev selects routine options", fill=ui.TEXT, font=ui.font(17, True))
        draw.text((382, 411), "4. LLM verifies and buys", fill=ui.TEXT, font=ui.font(17, True))
        return image

    if phase in {"buy", "success"}:
        draw.text((382, 235), "SELECTED PRODUCT STATE", fill=ui.ACCENT, font=ui.font(12, True))
        draw.rounded_rectangle((382, 272, 888, 328), 12, fill="#0b2426", outline=ui.ACCENT, width=2)
        draw.text((406, 288), "color: z-dark green  ·  size: small", fill=ui.TEXT, font=ui.font(18, True))
        if phase == "buy":
            draw.rounded_rectangle((382, 358, 610, 419), 13, fill="#14223a", outline=ui.BLUE, width=2)
            draw.text((414, 376), "LLM · BUY NOW", fill=ui.BLUE, font=ui.font(17, True))
            draw.text((641, 376), "completion stays with LLM", fill=ui.MUTED, font=ui.font(14))
        else:
            draw.rounded_rectangle((382, 355, 888, 427), 14, fill="#0b2426", outline=ui.ACCENT, width=3)
            draw.text((412, 372), "PURCHASE COMPLETE", fill=ui.ACCENT, font=ui.font(24, True))
            draw.text((772, 377), "reward 1", fill=ui.TEXT, font=ui.font(18, True))
        return image

    actions = (
        ("z-dark green", "z-army green", "z-khaki")
        if group == "color" else ("x-small", "small", "medium")
    )
    selected = 0 if group == "color" else 1
    for index, action in enumerate(actions):
        x = 380 + index * 174
        if "compare" in phase:
            status = "focus" if focus == index else "candidate"
            note = "compare exact value" if focus == index else "valid option"
        elif index == selected:
            status, note = "selected", "exact match"
        else:
            status, note = "rejected", "wrong color" if group == "color" else "wrong size"
        fill, outline, ink, badge = {
            "candidate": (ui.PANEL_2, ui.BORDER, ui.MUTED, "OPTION"),
            "focus": ("#14223a", ui.BLUE, ui.BLUE, "COMPARE"),
            "selected": ("#0b2426", ui.ACCENT, ui.ACCENT, "SELECT"),
            "rejected": ("#29161f", ui.RED, ui.RED, "REJECT"),
        }[status]
        draw.rounded_rectangle((x, 242, x + 158, 350), 13, fill=fill, outline=outline, width=3 if status in {"focus", "selected"} else 1)
        draw.text((x + 14, 261), action, fill=ui.TEXT, font=ui.font(14, status == "selected"))
        draw.text((x + 14, 299), note, fill=ink, font=ui.font(11, True))
        draw.text((x + 14, 325), badge, fill=ink, font=ui.font(9, True))
    if "select" in phase:
        draw.text((382, 390), f"Jev confidence 100%  ->  {actions[selected]}", fill=ui.ACCENT, font=ui.font(16, True))
    elif "execute" in phase:
        state = "color: z-dark green" if group == "color" else "color: z-dark green  ·  size: small"
        draw.text((382, 382), "ENVIRONMENT STATE UPDATED", fill=ui.BLUE, font=ui.font(12, True))
        draw.text((382, 416), state, fill=ui.ACCENT, font=ui.font(17, True, mono=True))
    else:
        draw.text((382, 390), "Goal + exact criterion + 3 mutually exclusive actions", fill=ui.MUTED, font=ui.font(13))
    return image


def webshop_impact_frame() -> Image.Image:
    image, draw = ui.canvas()
    header(draw, "WEBSHOP", "Same task and reward; less LLM work", "Paired seed 3107 · identical goal and product catalog", 5)
    rows = (
        ("LLM ONLY", "select + inspect + repeat + buy", "9 calls", "38,852 tokens", "18.54 s", ui.MUTED),
        ("LLM + JEV", "2 Jev choices -> LLM buys", "4 calls", "14,256 tokens", "7.83 s", ui.ACCENT),
    )
    for index, (name, path, calls, tokens, elapsed, color) in enumerate(rows):
        y = 214 + index * 112
        draw.rounded_rectangle((44, y, 916, y + 92), 14, fill="#0b2426" if index else ui.PANEL_2, outline=color, width=2)
        draw.text((68, y + 15), name, fill=color, font=ui.font(14, True))
        draw.text((216, y + 15), path, fill=ui.TEXT, font=ui.font(15, True))
        draw.text((690, y + 12), calls, fill=ui.TEXT, font=ui.font(15, True))
        draw.text((690, y + 38), tokens, fill=ui.MUTED, font=ui.font(12))
        draw.text((690, y + 63), elapsed, fill=ui.MUTED, font=ui.font(12))
    draw.text((44, 462), "Impact: reward 1 -> 1  ·  calls 9 -> 4  ·  time 18.54s -> 7.83s", fill=ui.ACCENT, font=ui.font(16, True))
    return image


def tween(frames: list[Image.Image], holds: list[int], steps: int = 2, transition_ms: int = 55):
    output, durations = [], []
    for index, frame in enumerate(frames):
        output.append(frame)
        durations.append(holds[index])
        if index + 1 < len(frames):
            next_frame = frames[index + 1]
            for step in range(1, steps + 1):
                output.append(Image.blend(frame, next_frame, step / (steps + 1)))
                durations.append(transition_ms)
    return output, durations


def save(path: str, keyframes: list[Image.Image], holds: list[int]) -> None:
    frames, durations = tween(keyframes, holds)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"{output} ({sum(durations) / 1000:.2f}s, {len(frames)} frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="docs/demos/jev-agent-harness-traces.json")
    parser.add_argument("--terminal-out", default="docs/demos/jev-decision-terminal-v2.gif")
    parser.add_argument("--frozen-lake-out", default="docs/demos/jev-decision-frozen-lake-v2.gif")
    parser.add_argument("--webshop-out", default="docs/demos/jev-decision-webshop-v2.gif")
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))

    sqlite = trace["sqlite"]
    frozen = trace["frozen_lake"]
    webshop = trace["webshop"]
    assert sqlite["jev_choice"] == "Inspect raw page" and sqlite["confidence"] == 0.81
    assert frozen["action_menu"] == ["Left", "Down", "Right", "Up"]
    for run in (sqlite["baseline"], sqlite["jev_harness"]):
        source = Path(run["source"])
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == run["sha256"]
    source = Path(frozen["source"])
    if source.exists():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == frozen["source_sha256"]
    source = Path(webshop["source"])
    if source.exists():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == webshop["source_sha256"]
    assert webshop["candidate_groups"][0]["actions"] == [
        "click[z-dark green]", "click[z-army green]", "click[z-khaki]",
    ]

    terminal_frames = [
        terminal_frame("goal"),
        terminal_frame("compare", 0),
        terminal_frame("compare", 1),
        terminal_frame("compare", 2),
        terminal_frame("select"),
        terminal_frame("execute", typed=0.35),
        terminal_frame("execute", typed=0.7),
        terminal_frame("execute", typed=1.0),
        terminal_frame("output"),
        terminal_frame("verify"),
        terminal_impact_frame(),
    ]
    terminal_holds = [520, 130, 130, 130, 650, 90, 90, 330, 620, 700, 1550]

    lake_frames = [lake_frame(0, "candidate"), lake_frame(0, "focus"), lake_frame(0, "selected")]
    lake_holds = [430, 180, 520]
    for stage in range(4):
        start = LAKE_POSITIONS[stage]
        target = LAKE_POSITIONS[stage + 1] if stage < 3 else (0, 3)
        if stage > 0:
            lake_frames.extend([lake_frame(stage, "focus"), lake_frame(stage, "selected")])
            lake_holds.extend([140, 430 if stage < 3 else 650])
        for step in (0.25, 0.5, 0.75, 1.0):
            moving = (start[0] + (target[0] - start[0]) * step, start[1] + (target[1] - start[1]) * step)
            lake_frames.append(lake_frame(stage, "moving", moving))
            lake_holds.append(70)
    lake_frames.extend([lake_goal_frame(), lake_impact_frame()])
    lake_holds.extend([650, 1600])

    webshop_frames = [webshop_frame("goal")]
    webshop_holds = [420]
    for phase in ("color", "size"):
        for focus in range(3):
            webshop_frames.append(webshop_frame(f"{phase}_compare", focus))
            webshop_holds.append(110)
        webshop_frames.extend([
            webshop_frame(f"{phase}_select"), webshop_frame(f"{phase}_execute"),
        ])
        webshop_holds.extend([540, 420])
    webshop_frames.extend([webshop_frame("buy"), webshop_frame("success"), webshop_impact_frame()])
    webshop_holds.extend([620, 650, 1500])

    save(args.terminal_out, terminal_frames, terminal_holds)
    save(args.frozen_lake_out, lake_frames, lake_holds, )
    save(args.webshop_out, webshop_frames, webshop_holds)


if __name__ == "__main__":
    main()
