#!/usr/bin/env python3
"""Render paired README animations from recorded totals and illustrated steps.

Both lanes share a clock and playback rate. Only total wall times are measured;
step durations are schematic. A completed lane stays frozen until the loop ends.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from render_agent_harness_demo import ACCENT, BLUE, BORDER, MUTED, PANEL, PANEL_2, TEXT


WIDTH, HEIGHT = 1080, 700
PLAY_MS = 1600
FRAME_MS = 40
END_HOLD_MS = 400
BACKGROUND = "#070b14"
GREEN_PANEL = "#0b2426"


@lru_cache
def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    family = "DejaVuSansMono" if mono else "DejaVuSans"
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{family}{suffix}.ttf", size)


@dataclass(frozen=True)
class Step:
    actor: str
    title: str
    detail: str
    state: int


def steps_for(key: str, trace: dict, assisted: bool) -> list[Step]:
    if key == "webshop":
        opening = [
            Step("LLM", "Search for a blazer", "Long sleeve, under $80", 0),
            Step("LLM", "Open product", "Generate color + size menus" if assisted else "Inspect product options", 1),
        ]
        choices = [
            Step("Jev" if assisted else "LLM", "Select z-dark green", "3 colors → exact match" if assisted else "Choose the requested color", 2),
            Step("Jev" if assisted else "LLM", "Select small", "3 sizes → exact match" if assisted else "Choose the requested size", 3),
        ]
        recheck = [] if assisted else [
            Step("LLM", "Inspect features", "Check product details", 3),
            Step("LLM", "Repeat color + size", "Return to the product", 3),
        ]
        return opening + choices + recheck + [Step("LLM", "Buy now", "Complete the purchase", 3)]
    if key == "frozen_lake":
        steps = [Step("LLM", "Plan a route", "Delegate; check every new state", 0)] if assisted else []
        return steps + [
            Step("Jev" if assisted else "LLM", f"Move {step['action'].lower()}",
                 "Compare 4 directions" if assisted else "Read state; choose next move", i + 1)
            for i, step in enumerate(trace["steps"])
        ]
    if assisted:
        return [
            Step("LLM", "Propose 3 commands", "Raw page, schema, tools", 0),
            Step("Jev", "Choose raw page", "Run od; read page bytes", 1),
            Step("LLM", "Parse cell pointers", "Recover records from raw bytes", 2),
            Step("LLM", "Write recover.json", "Check the recovered output", 3),
        ]
    return [
        Step("LLM", "Inspect database", "Choose diagnostic commands", 0),
        Step("LLM", "Read page structure", "Read and interpret observations", 1),
        Step("LLM", "Parse cell pointers", "Recover records from raw bytes", 2),
        Step("LLM", "Write recover.json", "Check the recovered output", 3),
    ]


def text(draw: ImageDraw.ImageDraw, x: float, y: float, value: str,
         size: int = 24, color: str = TEXT, bold: bool = False, mono: bool = False) -> None:
    draw.text((x, y), value, font=font(size, bold, mono), fill=color)


def webshop_scene(draw: ImageDraw.ImageDraw, x: int, state: int, color: str) -> None:
    if state == 0:
        text(draw, x + 22, 220, "SEARCH", 22, MUTED, True)
        draw.rounded_rectangle((x + 20, 264, x + 444, 324), 12, fill=PANEL_2, outline=BORDER)
        text(draw, x + 38, 279, "Women's green blazer", 26)
        text(draw, x + 22, 350, "small, long sleeve, < $80", 24, MUTED)
        return
    # A simple product sketch keeps the environment legible at README width.
    jacket = [
        (x + 55, 224), (x + 93, 216), (x + 126, 224),
        (x + 152, 265), (x + 131, 284), (x + 121, 267),
        (x + 125, 368), (x + 57, 368), (x + 61, 267),
        (x + 50, 284), (x + 28, 265),
    ]
    draw.polygon(jacket, fill="#277464" if state >= 2 else "#3d4e67")
    draw.line((x + 93, 222, x + 91, 357), fill="#9fbab5", width=3)
    text(draw, x + 179, 216, "Women's blazer", 26, bold=True)
    for y, label, selected in ((265, "z-dark green", state >= 2), (327, "small", state >= 3)):
        draw.rounded_rectangle((x + 176, y, x + 441, y + 48), 10,
                               fill=GREEN_PANEL if selected else PANEL_2,
                               outline=color if selected else BORDER, width=2)
        text(draw, x + 192, y + 9, label, 24, color if selected else MUTED)
        if selected:
            text(draw, x + 407, y + 7, "✓", 26, color, True)


def lake_scene(draw: ImageDraw.ImageDraw, x: int, trace: dict, step: Step,
               fraction: float, finished: bool, color: str) -> None:
    positions = [tuple(s["player"]) for s in trace["steps"]] + [(0, 3)]
    state = step.state
    player = positions[state]
    if state and not finished:
        start, target = positions[state - 1], positions[state]
        move = min(1.0, max(0.0, (fraction - 0.45) / 0.4))
        player = tuple(a + (b - a) * move for a, b in zip(start, target))
    gx, gy, cell = x + 26, 208, 46
    for row in range(4):
        for col in range(4):
            left, top = gx + col * cell, gy + row * cell
            goal, hole = (row, col) == (0, 3), (row, col) == (3, 3)
            fill = GREEN_PANEL if goal else "#29161f" if hole else PANEL_2
            draw.rounded_rectangle((left, top, left + 39, top + 39), 7, fill=fill, outline=BORDER)
            if goal or hole:
                text(draw, left + 10, top + 4, "G" if goal else "×", 23, ACCENT if goal else "#fb7185", True)
    cy, cx = gy + player[0] * cell + 19, gx + player[1] * cell + 19
    draw.ellipse((cx - 15, cy - 15, cx + 15, cy + 15), fill=color)
    text(draw, x + 241, 216, "4 directions", 24, MUTED)
    for i, action in enumerate(trace["action_menu"]):
        left, top = x + 238 + (i % 2) * 105, 262 + (i // 2) * 57
        selected = state > 0 and action == trace["steps"][state - 1]["action"]
        draw.rounded_rectangle((left, top, left + 96, top + 44), 9,
                               fill=GREEN_PANEL if selected else PANEL_2,
                               outline=color if selected else BORDER, width=2)
        text(draw, left + 12, top + 8, action, 23, color if selected else MUTED, selected)


def terminal_scene(draw: ImageDraw.ImageDraw, x: int, state: int,
                   assisted: bool, finished: bool) -> None:
    draw.rounded_rectangle((x + 20, 207, x + 444, 393), 12, fill="#080e1b", outline=BORDER)
    text(draw, x + 38, 220, "trunc.db", 23, MUTED, mono=True)
    if state == 0:
        lines = ("raw page | schema | tools", "Jev chooses the inspection") if assisted else (
            "damaged SQLite page", "LLM chooses inspections",
        )
    elif state == 1:
        lines = ("$ od ... trunc.db", "0d 00 00 00 0a 0f 49 ...") if assisted else (
            "inspect page bytes", "interpret page structure",
        )
    elif state == 2:
        lines = ("cell pointers → records", "parse recoverable rows")
    else:
        lines = ("recover.json", "10 rows, verifier pass" if finished else "write rows; check output")
    for i, line in enumerate(lines):
        text(draw, x + 38, 271 + i * 55, line, 23, ACCENT if finished else TEXT, mono=True)


def pair_for(key: str, trace: dict) -> tuple[dict, dict]:
    pair = trace["pair"] if key == "frozen_lake" else trace
    baseline, assisted = pair["baseline"], pair["jev_harness"]
    for run in (baseline, assisted):
        if not 0 < run["wall_time_seconds"] < float("inf"):
            raise ValueError(f"{key}: wall_time_seconds must be finite and positive")
    return baseline, assisted


def finish_ms(run: dict, longest: float) -> int:
    # GIF timestamps have 10 ms precision; include the actual finish boundary.
    return round(PLAY_MS * run["wall_time_seconds"] / longest / 10) * 10


def render_frame(key: str, trace: dict, tick: int) -> Image.Image:
    baseline, assisted = pair_for(key, trace)
    longest = max(run["wall_time_seconds"] for run in (baseline, assisted))
    titles = {
        "webshop": "WebShop: buy the matching blazer",
        "frozen_lake": "FrozenLake: reach the goal",
        "sqlite": "Terminal-Bench: recover SQLite rows",
    }
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    text(draw, 28, 21, titles[key], 28, bold=True)
    for lane, run in enumerate((baseline, assisted)):
        x = 24 + lane * 528
        color = ACCENT if lane else BLUE
        end = finish_ms(run, longest)
        finished = tick >= end
        elapsed = run["wall_time_seconds"] if finished else tick / PLAY_MS * longest
        steps = steps_for(key, trace, bool(lane))
        progress = min(tick / end, 1.0)
        position = min(progress * len(steps), len(steps) - 0.000001)
        index = int(position)
        step, fraction = steps[index], position - index

        draw.rounded_rectangle((x, 73, x + 504, 608), 18, fill=PANEL, outline=color, width=2)
        text(draw, x + 22, 90, "LLM + Jev" if lane else "LLM only", 30, color, True)
        text(draw, x + 22, 131, f"{elapsed:.2f} s" if key == "webshop" else f"{elapsed:.1f} s", 40, bold=True)
        text(draw, x + 280, 148, "Completed" if finished else "Running…", 28, ACCENT if finished else MUTED, True)
        inner = x + 20
        if key == "webshop":
            webshop_scene(draw, inner, step.state, color)
        elif key == "frozen_lake":
            lake_scene(draw, inner, trace, step, fraction, finished, color)
        else:
            terminal_scene(draw, inner, step.state, bool(lane), finished)

        draw.rounded_rectangle((x + 20, 414, x + 484, 533), 12,
                               fill=GREEN_PANEL if finished else PANEL_2)
        if finished:
            text(draw, x + 36, 427, "✓ Completed", 32, ACCENT, True)
            outcome = {"webshop": "Purchased, reward 1", "frozen_lake": "Goal reached, reward 1",
                       "sqlite": "10 rows recovered, reward 1"}[key]
            text(draw, x + 36, 480, outcome, 25)
        else:
            actor_color = ACCENT if step.actor == "Jev" else BLUE
            text(draw, x + 36, 425, f"{step.actor}: {step.title}", 26, actor_color, True)
            text(draw, x + 36, 477, step.detail, 24, MUTED)
        draw.rounded_rectangle((x + 22, 551, x + 482, 560), 4, fill=BORDER)
        length = round(460 * progress)
        if length:
            draw.rounded_rectangle((x + 22, 551, x + 22 + length, 560), 4, fill=color)
        text(draw, x + 22, 577, f"Total: {run['llm_calls']} LLM calls", 23, MUTED)
    saved = baseline["wall_time_seconds"] - assisted["wall_time_seconds"]
    text(draw, 28, 626, f"Recorded pair: {saved:.2f} s saved ({saved / baseline['wall_time_seconds']:.0%} less time)",
         27, ACCENT, True)
    text(draw, 28, 668, f"Illustrated steps; measured totals at {longest * 1000 / PLAY_MS:.2f}× playback", 23, MUTED)
    return image


def save(path: str, key: str, trace: dict) -> None:
    pair = pair_for(key, trace)
    longest = max(run["wall_time_seconds"] for run in pair)
    finishes = {finish_ms(run, longest) for run in pair}
    # Keep finish boundaries without introducing sub-20 ms frame delays.
    ticks = sorted(finishes | {
        tick for tick in range(0, PLAY_MS + 1, FRAME_MS)
        if all(abs(tick - end) >= 20 for end in finishes)
    })
    durations = [b - a for a, b in zip(ticks, ticks[1:])] + [END_HOLD_MS]
    # A shared palette prevents flicker and lets GIF store just changed pixels.
    samples = [render_frame(key, trace, tick).resize((270, 175)) for tick in ticks[::10]]
    sheet = Image.new("RGB", (270, 175 * len(samples)))
    for i, sample in enumerate(samples):
        sheet.paste(sample, (0, 175 * i))
    palette = sheet.quantize(colors=128)
    frames = [
        render_frame(key, trace, tick).quantize(palette=palette, dither=Image.Dither.NONE)
        for tick in ticks
    ]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    print(f"{output}: {len(frames)} frames, {sum(durations) / 1000:.1f}s, "
          f"Jev completes at {finish_ms(pair[1], longest) / 1000:.2f}s, "
          f"{output.stat().st_size / 1024:.0f} KiB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="docs/demos/jev-agent-harness-traces.json")
    parser.add_argument("--terminal-out", default="docs/demos/jev-decision-terminal-v2.gif")
    parser.add_argument("--frozen-lake-out", default="docs/demos/jev-decision-frozen-lake-v2.gif")
    parser.add_argument("--webshop-out", default="docs/demos/jev-decision-webshop-v2.gif")
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    for key, output in (("webshop", args.webshop_out), ("frozen_lake", args.frozen_lake_out),
                        ("sqlite", args.terminal_out)):
        save(output, key, trace[key])


if __name__ == "__main__":
    main()
