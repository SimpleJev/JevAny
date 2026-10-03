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
import shutil
import subprocess

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFont

from render_agent_harness_demo import ACCENT, AMBER, BLUE, BORDER, MUTED, PANEL, PANEL_2, TEXT
from render_agent_robot_icons import ROBOT_COLORS, actor_icon


WIDTH, HEIGHT = 1080, 700
PLAY_MS = 2400
FRAME_MS = 40
END_HOLD_MS = 600
BACKGROUND = "#070b14"
GREEN_PANEL = "#0b2426"
BLUE_PANEL = "#14223a"
AMBER_PANEL = "#292313"
TASK_CONTEXT = {
    "webshop": "z-dark green · small · under $80",
    "frozen_lake": "Start [1, 0] → goal [0, 3]",
    "sqlite": "Target: 10 verified rows",
}


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


def right_text(draw: ImageDraw.ImageDraw, right: float, y: float, value: str,
               size: int = 18, color: str = MUTED, bold: bool = False) -> None:
    text(draw, right - draw.textlength(value, font=font(size, bold)), y,
         value, size, color, bold)


def actor_style(actor: str) -> tuple[str, str]:
    return (ACCENT, GREEN_PANEL) if actor == "Jev" else (
        (BLUE, BLUE_PANEL) if actor == "LLM" else (AMBER, AMBER_PANEL)
    )


def wrapped_text(draw: ImageDraw.ImageDraw, x: int, y: int, value: str,
                 width: int, size: int = 17, color: str = MUTED) -> None:
    lines = [""]
    for word in value.split():
        candidate_line = f"{lines[-1]} {word}".strip()
        if draw.textlength(candidate_line, font=font(size)) > width and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = candidate_line
    for index, line in enumerate(lines):
        text(draw, x, y + index * 20, line, size, color)


def phase_panel(image: Image.Image, draw: ImageDraw.ImageDraw, x: int, key: str,
                step: Step, index: int, count: int, finished: bool, tick: int) -> None:
    actor = ("Verifier" if key == "sqlite" else "Env") if finished else step.actor
    color, fill = actor_style(actor)
    draw.rounded_rectangle((x + 20, 414, x + 484, 533), 12, fill=PANEL_2)
    draw.rounded_rectangle((x + 29, 423, x + 123, 524), 10,
                           fill=fill, outline=color, width=2)
    portrait = actor_icon(actor, 78, phase=0 if finished else tick // 100 % 6)
    image.paste(portrait, (x + 37, 424), portrait)
    label = "CHECK" if actor == "Verifier" else "RESULT" if actor == "Env" else actor.upper()
    center = x + 76
    text(draw, center - draw.textlength(label, font=font(13, True)) / 2,
         503, label, 13, color, True)
    text(draw, x + 141, 429,
         "VERIFIED RESULT" if finished else f"PHASE {index + 1} / {count}", 13, MUTED)
    title = "✓ Task complete" if finished else step.title
    text(draw, x + 141, 452, title, 24, ACCENT if finished else TEXT, True)
    outcome = {"webshop": "Purchased, reward 1", "frozen_lake": "Goal reached, reward 1",
               "sqlite": "10 rows recovered, reward 1"}[key]
    wrapped_text(draw, x + 141, 491, outcome if finished else step.detail, 322)


def actor_rail(image: Image.Image, draw: ImageDraw.ImageDraw, x: int, steps: list[Step],
               current: int, finished: bool, tick: int) -> None:
    # Each controller keeps one position, including when it regains control.
    actors = list(dict.fromkeys(step.actor for step in steps))
    gap = 8
    width = (460 - gap * (len(actors) - 1)) / len(actors)
    for index, actor in enumerate(actors):
        left = x + 22 + index * (width + gap)
        color, fill = actor_style(actor)
        active = not finished and steps[current].actor == actor
        draw.rounded_rectangle((left, 546, left + width, 589), 9,
                               fill=fill if active else PANEL,
                               outline=color if active else BORDER, width=2 if active else 1)
        portrait = actor_icon(actor, 36, active, tick // 100 % 6 if active else 0)
        image.paste(portrait, (round(left + 10), 548), portrait)
        text(draw, left + 55, 557, actor.upper(), 15, color if active else MUTED, True)
        if active:
            draw.ellipse((left + width - 18, 564, left + width - 12, 570), fill=color)
            text(draw, left + width - 69, 561, "ACTIVE", 9, color, True)
    text(draw, x + 22, 594, "TASK COMPLETE" if finished else "ACTIVE CONTROLLER", 10, MUTED)


def savings_strip(draw: ImageDraw.ImageDraw, baseline: dict, assisted: dict) -> None:
    # Static totals from this recorded pair; percentages use the baseline.
    tokens_saved = baseline["tokens"] - assisted["tokens"]
    time_saved = baseline["wall_time_seconds"] - assisted["wall_time_seconds"]
    rewards = baseline["reward"], assisted["reward"]
    both_succeeded = all(reward == 1 for reward in rewards)
    outcome = "Both succeeded" if both_succeeded else (
        "Same reward" if rewards[0] == rewards[1] else "Reward changed"
    )
    cards = [
        ("LLM TOKENS SAVED", f"{tokens_saved:,}",
         f"{tokens_saved / baseline['tokens']:.1%}",
         f"{baseline['tokens']:,} → {assisted['tokens']:,}", ACCENT),
        ("TIME SAVED", f"{time_saved:g} s",
         f"{time_saved / baseline['wall_time_seconds']:.1%}",
         f"{baseline['wall_time_seconds']:g} s → {assisted['wall_time_seconds']:g} s", ACCENT),
        ("TASK RESULT", outcome, "✓" if both_succeeded else "",
         f"Reward {rewards[0]:g} → {rewards[1]:g}", AMBER),
    ]
    for index, (label, value, badge, detail, color) in enumerate(cards):
        x = 24 + index * 348
        metric = index < 2
        draw.rounded_rectangle((x, 616, x + 336, 676), 10,
                               fill=GREEN_PANEL if metric else PANEL,
                               outline=ACCENT if metric else BORDER)
        text(draw, x + 14, 619, label, 14, MUTED, True)
        if metric:
            text(draw, x + 14, 631, badge, 42, color, True)
            right_text(draw, x + 322, 620, value, 28, TEXT, True)
            right_text(draw, x + 322, 651, detail, 15)
        else:
            text(draw, x + 14, 632, value, 26, TEXT, True)
            right_text(draw, x + 322, 637, badge, 24, color, True)
            text(draw, x + 14, 657, detail, 15, MUTED)


def candidate(draw: ImageDraw.ImageDraw, x: int, y: int, width: int,
              label: str, selected: bool, color: str = ACCENT) -> None:
    fill = BLUE_PANEL if color == BLUE else GREEN_PANEL
    draw.rounded_rectangle((x, y, x + width, y + 41), 8,
                           fill=fill if selected else PANEL_2,
                           outline=color if selected else BORDER, width=2)
    text(draw, x + 13, y + 7, label, 21, color if selected else MUTED, selected)
    if selected:
        text(draw, x + width - 32, y + 5, "✓", 23, color, True)


def webshop_scene(draw: ImageDraw.ImageDraw, x: int, trace: dict, step: Step,
                   assisted: bool, finished: bool, color: str) -> None:
    state = step.state
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
    if assisted and step.actor == "Jev" and not finished:
        decision = trace["candidate_groups"][state - 2]
        kind = "color" if state == 2 else "size"
        text(draw, x + 179, 216, f"LLM {kind} menu", 24, BLUE, True)
        for index, action in enumerate(decision["actions"]):
            candidate(draw, x + 176, 256 + index * 47, 265,
                      action.removeprefix("click[").removesuffix("]"),
                      action == decision["selected"])
        text(draw, x + 29, 377, "3 alternatives", 17, MUTED)
        return
    text(draw, x + 179, 216, "Women's blazer", 26, bold=True)
    for y, label, selected in ((265, "z-dark green", state >= 2), (327, "small", state >= 3)):
        draw.rounded_rectangle((x + 176, y, x + 441, y + 48), 10,
                               fill=(GREEN_PANEL if assisted else BLUE_PANEL) if selected else PANEL_2,
                               outline=color if selected else BORDER, width=2)
        text(draw, x + 192, y + 9, label, 24, color if selected else MUTED)
        if selected:
            text(draw, x + 407, y + 7, "✓", 26, color, True)


def lake_scene(draw: ImageDraw.ImageDraw, x: int, trace: dict, step: Step,
               fraction: float, finished: bool, color: str, assisted: bool) -> None:
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
    text(draw, x + 241, 216, "1 LLM plan" if assisted else "4 directions",
         24, BLUE if assisted else MUTED)
    for i, action in enumerate(trace["action_menu"]):
        left, top = x + 238 + (i % 2) * 105, 262 + (i // 2) * 57
        selected = state > 0 and action == trace["steps"][state - 1]["action"]
        draw.rounded_rectangle((left, top, left + 96, top + 44), 9,
                               fill=(GREEN_PANEL if assisted else BLUE_PANEL) if selected else PANEL_2,
                               outline=color if selected else BORDER, width=2)
        text(draw, left + 12, top + 8, action, 23, color if selected else MUTED, selected)


def terminal_scene(draw: ImageDraw.ImageDraw, x: int, trace: dict, state: int,
                   assisted: bool, finished: bool) -> None:
    draw.rounded_rectangle((x + 20, 207, x + 444, 393), 12, fill="#080e1b", outline=BORDER)
    if assisted and state < 2:
        text(draw, x + 38, 220, "trunc.db · 3 inspections", 19, MUTED, mono=True)
        for index, option in enumerate(trace["options"]):
            candidate(draw, x + 32, 250 + index * 46, 398, option["name"],
                      state == 1 and option["name"] == trace["jev_choice"])
        return
    text(draw, x + 38, 220, "trunc.db", 23, MUTED, mono=True)
    if state == 0:
        lines = ("damaged SQLite page", "LLM chooses inspections")
    elif state == 1:
        lines = ("inspect page bytes", "interpret page structure")
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
    right_text(draw, WIDTH - 28, 30, TASK_CONTEXT[key], 17)
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
        calls = str(run["llm_calls"])
        text(draw, x + 22, 124, calls, 44, color, True)
        label_x = x + 34 + draw.textlength(calls, font=font(44, True))
        text(draw, label_x, 131, "RECORDED TOTAL", 10, MUTED)
        text(draw, label_x, 149, "LLM call" if run["llm_calls"] == 1 else "LLM calls", 21, TEXT)
        text(draw, x + 300, 105, "COMPLETED" if finished else "ELAPSED TIME",
             11, ACCENT if finished else MUTED, True)
        draw.rounded_rectangle((x + 300, 125, x + 484, 188), 10,
                               fill=BACKGROUND, outline=BORDER)
        text(draw, x + 314, 133,
             f"{elapsed:.2f} s" if key == "webshop" else f"{elapsed:.1f} s", 29, TEXT, True)
        draw.rounded_rectangle((x + 314, 178, x + 470, 182), 2, fill=BORDER)
        length = round(156 * progress)
        if length:
            draw.rounded_rectangle((x + 314, 178, x + 314 + length, 182), 2, fill=color)
        inner = x + 20
        if key == "webshop":
            webshop_scene(draw, inner, trace, step, bool(lane), finished, color)
        elif key == "frozen_lake":
            lake_scene(draw, inner, trace, step, fraction, finished, color, bool(lane))
        else:
            terminal_scene(draw, inner, trace, step.state, bool(lane), finished)

        phase_panel(image, draw, x, key, step, index, len(steps), finished, tick)
        actor_rail(image, draw, x, steps, index, finished, tick)
    savings_strip(draw, baseline, assisted)
    text(draw, 28, 681,
         f"One recorded pair · illustrated steps · measured totals at {longest * 1000 / PLAY_MS:.2f}× playback",
         13, MUTED)
    right_text(draw, WIDTH - 28, 681, "LLM only → LLM + Jev", 13)
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
    # Keep controller and portrait colors exact even when small art is undersampled.
    # Remaining entries retain the sampled colors for antialiased text and art.
    theme = (BACKGROUND, PANEL, PANEL_2, BORDER, TEXT, MUTED, ACCENT, BLUE,
             AMBER, GREEN_PANEL, BLUE_PANEL, AMBER_PANEL) + ROBOT_COLORS
    colors = palette.getpalette()[:128 * 3]
    colors += [channel for color in theme for channel in ImageColor.getrgb(color)]
    colors += list(ImageColor.getrgb(BACKGROUND)) * ((768 - len(colors)) // 3)
    palette.putpalette(colors)
    frames = []
    for tick in ticks:
        original = render_frame(key, trace, tick)
        frame = original.quantize(palette=palette, dither=Image.Dither.NONE)
        # Quantization can reuse a nearby swatch even for an exact palette color.
        # Pin flat theme pixels; retain sampled colors for antialiased edges.
        channels = original.split()
        for index, color in enumerate(theme, start=128):
            masks = [
                channel.point([255 if value == target else 0 for value in range(256)])
                for channel, target in zip(channels, ImageColor.getrgb(color))
            ]
            mask = ImageChops.multiply(ImageChops.multiply(masks[0], masks[1]), masks[2])
            frame.paste(index, mask=mask)
        frames.append(frame)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    print(f"{output}: {len(frames)} frames, {sum(durations) / 1000:.1f}s, "
          f"Jev completes at {finish_ms(pair[1], longest) / 1000:.2f}s, "
          f"{output.stat().st_size / 1024:.0f} KiB")


def save_overview(path: str, sources: list[str]) -> None:
    """Join the three approved loops without changing their pixels or timing."""
    frames = []
    durations = []
    for source in sources:
        with Image.open(source) as clip:
            if clip.size != (WIDTH, HEIGHT):
                raise ValueError(f"{source}: expected a {WIDTH}×{HEIGHT} replay")
            for index in range(clip.n_frames):
                clip.seek(index)
                original = clip.convert("RGB")
                frame = original.quantize(colors=256, dither=Image.Dither.NONE)
                if ImageChops.difference(original, frame.convert("RGB")).getbbox():
                    raise ValueError(f"{source}: frame {index} cannot be preserved exactly")
                frames.append(frame)
                durations.append(clip.info["duration"])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    print(f"{output}: {len(frames)} frames, {sum(durations) / 1000:.1f}s, "
          f"{output.stat().st_size / 1024:.0f} KiB")


def save_site_media(sources: list[str]) -> None:
    """Export only this presentation's videos and posters at native resolution."""
    executable = shutil.which("ffmpeg")
    if not executable:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as error:
            raise SystemExit("Install ffmpeg or imageio-ffmpeg to export site media.") from error
        executable = get_ffmpeg_exe()
    directory = Path(__file__).resolve().parents[1] / "site/assets/media/docs"
    directory.mkdir(parents=True, exist_ok=True)
    for source in sources:
        name = Path(source).stem
        common = [
            executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ignore_loop", "1", "-i", source, "-an", "-vf", "setsar=1,fps=25",
            "-pix_fmt", "yuv420p", "-threads", "2",
        ]
        subprocess.run(common + [
            "-c:v", "libx264", "-crf", "26", "-preset", "fast",
            "-movflags", "+faststart", str(directory / f"{name}.mp4"),
        ], check=True)
        subprocess.run(common + [
            "-c:v", "libvpx-vp9", "-crf", "36", "-b:v", "0", "-cpu-used", "3",
            "-row-mt", "1", str(directory / f"{name}.webm"),
        ], check=True)
        with Image.open(source) as image:
            image.seek(min(12, image.n_frames - 1))
            image.convert("RGB").save(directory / f"{name}.webp", quality=88, method=6)
        print(f"Prepared site media: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="docs/demos/jev-agent-harness-traces.json")
    parser.add_argument("--terminal-out", "--sqlite-out", default="docs/demos/jev-decision-terminal-v2.gif")
    parser.add_argument("--frozen-lake-out", default="docs/demos/jev-decision-frozen-lake-v2.gif")
    parser.add_argument("--webshop-out", default="docs/demos/jev-decision-webshop-v2.gif")
    parser.add_argument("--overview-out", "--out", default="docs/demos/jev-agent-harness.gif")
    parser.add_argument("--site-media", action="store_true",
                        help="Also export MP4, WebM, and WebP for these four GIFs")
    args = parser.parse_args()
    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    for key, output in (("webshop", args.webshop_out), ("frozen_lake", args.frozen_lake_out),
                        ("sqlite", args.terminal_out)):
        save(output, key, trace[key])
    clips = [args.webshop_out, args.frozen_lake_out, args.terminal_out]
    save_overview(args.overview_out, clips)
    if args.site_media:
        save_site_media([args.overview_out, *clips])


if __name__ == "__main__":
    main()
