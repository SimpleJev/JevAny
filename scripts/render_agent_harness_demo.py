#!/usr/bin/env python3
"""Render the recorded SQLite and WebShop harness traces as a README GIF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 960, 540
BACKGROUND = "#070911"
PANEL = "#11182a"
PANEL_2 = "#172136"
BORDER = "#35415d"
TEXT = "#f7f8ff"
MUTED = "#9ca8c2"
ACCENT = "#2dd4bf"
BLUE = "#60a5fa"
AMBER = "#f59e0b"
RED = "#fb7185"


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        filename = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    else:
        filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{filename}", size)


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, WIDTH - 18, HEIGHT - 18), 25, fill=PANEL, outline=BORDER, width=2)
    return image, draw


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, color: str = ACCENT) -> None:
    x, y = xy
    box = draw.textbbox((0, 0), value, font=font(13, True))
    width = box[2] - box[0] + 24
    draw.rounded_rectangle((x, y, x + width, y + 28), 14, fill="#0b2426")
    draw.text((x + 12, y + 6), value, fill=color, font=font(13, True))


def heading(draw: ImageDraw.ImageDraw, title: str, subtitle: str, badge: str) -> None:
    label(draw, (44, 40), badge)
    draw.text((44, 82), title, fill=TEXT, font=font(28, True))
    draw.text((44, 119), subtitle, fill=MUTED, font=font(16))


def pipeline(draw: ImageDraw.ImageDraw, active: int) -> None:
    names = ("1  LLM plans", "2  Jev selects", "3  LLM verifies")
    xs = (44, 346, 648)
    for index, (name, x) in enumerate(zip(names, xs)):
        selected = index == active
        fill = "#0b2426" if selected else PANEL_2
        outline = ACCENT if selected else BORDER
        draw.rounded_rectangle((x, 162, x + 268, 208), 13, fill=fill, outline=outline, width=2)
        draw.text((x + 18, 176), name, fill=TEXT if selected else MUTED, font=font(15, selected))
        if index < 2:
            draw.text((x + 279, 176), ">", fill=MUTED, font=font(17, True))


def option(draw: ImageDraw.ImageDraw, y: int, title: str, detail: str, selected: bool = False) -> None:
    fill = "#0b2426" if selected else PANEL_2
    outline = ACCENT if selected else BORDER
    draw.rounded_rectangle((44, y, 480, y + 65), 12, fill=fill, outline=outline, width=2 if selected else 1)
    draw.text((62, y + 11), title, fill=TEXT, font=font(16, selected))
    draw.text((62, y + 36), detail, fill=ACCENT if selected else MUTED, font=font(12, mono=True))


def sqlite_frame(stage: int) -> Image.Image:
    image, draw = canvas()
    titles = (
        ("LLM creates a real choice", "Three valid ways to inspect a truncated SQLite database", "RECORDED TRACE"),
        ("Jev handles the bounded choice", "It selects one option; it does not solve the recovery task", "SQLITE"),
        ("LLM owns interpretation and completion", "The delegated observation returns to the frontier model", "SQLITE"),
    )
    heading(draw, *titles[stage])
    pipeline(draw, stage)
    option(draw, 234, "Inspect raw page", "od -A x -t x1z -v trunc.db | head", stage >= 1)
    option(draw, 310, "Read SQLite schema", "sqlite3 trunc.db '.schema'", False)
    option(draw, 386, "Check available tools", "which sqlite3 python3", False)

    draw.rounded_rectangle((510, 234, 916, 451), 15, fill="#0c1322", outline=BORDER, width=1)
    if stage == 0:
        draw.text((534, 258), "Frontier fallback", fill=MUTED, font=font(13, True))
        draw.text((534, 286), "Inspect raw page", fill=TEXT, font=font(22, True))
        draw.text((534, 329), "Jev receives the same", fill=MUTED, font=font(15))
        draw.text((534, 354), "bounded 3-option menu.", fill=MUTED, font=font(15))
    elif stage == 1:
        draw.text((534, 258), "JEV CHOICE", fill=ACCENT, font=font(13, True))
        draw.text((534, 287), "Inspect raw page", fill=TEXT, font=font(24, True))
        draw.text((534, 330), "confidence", fill=MUTED, font=font(14))
        draw.text((795, 325), "0.81", fill=ACCENT, font=font(25, True))
        draw.rounded_rectangle((534, 372, 888, 390), 9, fill="#233149")
        draw.rounded_rectangle((534, 372, 821, 390), 9, fill=ACCENT)
        draw.text((534, 410), "Above 0.55 gate -> execute", fill=MUTED, font=font(14))
    else:
        draw.text((534, 255), "OBSERVATION", fill=BLUE, font=font(13, True))
        draw.text((534, 283), "SQLite leaf page", fill=TEXT, font=font(21, True))
        draw.text((534, 315), "10 rows recovered", fill=TEXT, font=font(18))
        draw.text((534, 350), "LLM writes + verifies recover.json", fill=MUTED, font=font(14))
        draw.rounded_rectangle((534, 386, 890, 432), 12, fill="#0b2426", outline=ACCENT, width=2)
        draw.text((556, 398), "VERIFIER PASS", fill=ACCENT, font=font(18, True))
    draw.text((44, 479), "Recorded run: calls 15 -> 8  |  tokens -40%  |  time 187.9s -> 144.7s", fill=MUTED, font=font(14))
    return image


def sqlite_comparison_frame() -> Image.Image:
    image, draw = canvas()
    heading(draw, "The useful observation arrives sooner", "Same SQLite task and reward; fewer frontier-model calls", "SQLITE RESULT")

    cards = (
        (44, "LLM ONLY", "15", "202,050", "187.9s", MUTED),
        (490, "LLM + JEV", "8", "121,293", "144.7s", ACCENT),
    )
    for x, title, calls, tokens, elapsed, color in cards:
        draw.rounded_rectangle((x, 177, x + 426, 414), 17, fill=PANEL_2, outline=color, width=2)
        draw.text((x + 24, 199), title, fill=color, font=font(15, True))
        draw.text((x + 24, 248), calls, fill=TEXT, font=font(34, True))
        draw.text((x + 96, 263), "LLM calls", fill=MUTED, font=font(15))
        draw.text((x + 24, 307), tokens, fill=TEXT, font=font(28, True))
        draw.text((x + 183, 317), "tokens", fill=MUTED, font=font(15))
        draw.text((x + 24, 359), elapsed, fill=TEXT, font=font(28, True))
        draw.text((x + 151, 369), "wall time", fill=MUTED, font=font(15))
    draw.text((44, 457), "Jev chooses the raw-page inspection -> LLM parses the rows -> verifier passes", fill=TEXT, font=font(16, True))
    draw.text((44, 487), "Reward: 1 -> 1", fill=ACCENT, font=font(14, True))
    return image


def swatch(draw: ImageDraw.ImageDraw, x: int, color: str, name: str, selected: bool = False) -> None:
    outline = ACCENT if selected else BORDER
    draw.rounded_rectangle((x, 338, x + 120, 399), 11, fill=PANEL_2, outline=outline, width=2 if selected else 1)
    draw.ellipse((x + 13, 351, x + 39, 377), fill=color, outline="#dbeafe")
    draw.text((x + 49, 354), name, fill=TEXT if selected else MUTED, font=font(12, selected))


def webshop_frame(stage: int) -> Image.Image:
    image, draw = canvas()
    titles = (
        ("LLM finds the matching product", "Target: lace-up boots, black, size 11.5, under $160", "WEBSHOP"),
        ("Jev executes the local choice", "The LLM delegates the first option selection", "WEBSHOP"),
        ("LLM completes the purchase", "Observe -> select size -> buy -> reward 1", "WEBSHOP"),
    )
    heading(draw, *titles[stage])
    pipeline(draw, stage)

    draw.rounded_rectangle((44, 234, 590, 452), 15, fill="#f8fafc", outline="#cbd5e1", width=1)
    draw.rounded_rectangle((68, 258, 212, 407), 12, fill="#dbeafe")
    draw.rounded_rectangle((101, 276, 179, 386), 12, fill="#111827")
    draw.rectangle((108, 364, 128, 397), fill="#111827")
    draw.rectangle((151, 364, 171, 397), fill="#111827")
    draw.text((238, 258), "Men's lace-up boots", fill="#172136", font=font(21, True))
    draw.text((238, 293), "rubber outsole  |  under $160", fill="#475569", font=font(14))
    actions = ("click[black]", "click[11.5]", "click[buy now]")
    for index, value in enumerate(actions):
        x = 238 + index * 105
        selected = (stage == 1 and index == 0) or (stage == 2 and index > 0)
        draw.rounded_rectangle(
            (x, 342, x + 96, 382), 9,
            fill="#d1fae5" if selected else "#e2e8f0",
            outline="#0f766e" if selected else "#94a3b8",
            width=2 if selected else 1,
        )
        draw.text((x + 8, 355), value, fill="#115e59" if selected else "#475569", font=font(10, selected, mono=True))
    draw.rounded_rectangle((238, 408, 558, 440), 8, fill="#ecfeff", outline="#67e8f9")
    draw.text((250, 416), "goal: black  +  11.5  +  price < $160", fill="#155e75", font=font(11, True))

    draw.rounded_rectangle((618, 234, 916, 452), 15, fill="#0c1322", outline=BORDER, width=1)
    if stage == 0:
        draw.text((642, 258), "LLM PLAN", fill=BLUE, font=font(13, True))
        draw.text((642, 289), "1. select black", fill=TEXT, font=font(18, True))
        draw.text((642, 323), "2. select 11.5", fill=TEXT, font=font(18, True))
        draw.text((642, 357), "3. buy now", fill=TEXT, font=font(18, True))
        draw.text((642, 405), "3-step local subgoal", fill=MUTED, font=font(14))
    elif stage == 1:
        draw.text((642, 258), "JEV CHOICE", fill=ACCENT, font=font(13, True))
        draw.text((642, 290), "click[black]", fill=TEXT, font=font(23, True))
        draw.text((642, 333), "confidence  0.89", fill=ACCENT, font=font(18, True))
        draw.rounded_rectangle((642, 375, 888, 393), 9, fill="#233149")
        draw.rounded_rectangle((642, 375, 861, 393), 9, fill=ACCENT)
        draw.text((642, 414), "action -> observation", fill=MUTED, font=font(14))
    else:
        draw.text((642, 258), "LLM FINISH", fill=BLUE, font=font(13, True))
        draw.text((642, 289), "click[11.5]", fill=TEXT, font=font(19, True))
        draw.text((642, 321), "click[buy now]", fill=TEXT, font=font(19, True))
        draw.rounded_rectangle((642, 366, 890, 421), 12, fill="#0b2426", outline=ACCENT, width=2)
        draw.text((672, 381), "BUY -> REWARD 1", fill=ACCENT, font=font(17, True))
    draw.text((44, 479), "Recorded seed 3106: reward 0 -> 1  |  LLM calls 11 -> 5  |  tokens 47,360 -> 15,705", fill=MUTED, font=font(14))
    return image


def webshop_comparison_frame() -> Image.Image:
    image, draw = canvas()
    heading(draw, "Recorded pair: reward 0 -> 1", "Same WebShop task, seed 3106", "WEBSHOP RESULT")

    draw.rounded_rectangle((44, 177, 456, 432), 17, fill=PANEL_2, outline=BORDER, width=2)
    draw.text((68, 200), "LLM ONLY", fill=MUTED, font=font(15, True))
    baseline = ("search", "inspect features", "back", "search again", "step budget ends")
    for index, value in enumerate(baseline):
        y = 242 + index * 34
        draw.ellipse((68, y + 3, 78, y + 13), fill=MUTED)
        draw.text((94, y), value, fill=TEXT if index < 4 else RED, font=font(15, index == 4))
    draw.text((68, 399), "reward 0  |  11 calls  |  20.9s", fill=RED, font=font(16, True))

    draw.rounded_rectangle((480, 177, 916, 432), 17, fill="#0b2426", outline=ACCENT, width=2)
    draw.text((504, 200), "LLM + JEV", fill=ACCENT, font=font(15, True))
    harness = ("LLM searches", "LLM opens match", "Jev selects black", "LLM selects 11.5", "LLM buys")
    for index, value in enumerate(harness):
        y = 242 + index * 34
        draw.ellipse((504, y + 3, 514, y + 13), fill=ACCENT)
        draw.text((530, y), value, fill=TEXT, font=font(15, index == 2))
    draw.text((504, 399), "reward 1  |  5 calls  |  9.9s", fill=ACCENT, font=font(16, True))

    draw.text((44, 469), "Observed result: 11 -> 5 LLM calls  |  20.9s -> 9.9s", fill=TEXT, font=font(16, True))
    return image


def frozen_lake_frame(stage: int) -> Image.Image:
    image, draw = canvas()
    actions = ("Right", "Right", "Right", "Up")
    confidences = (0.99, 0.99, 0.82, 1.0)
    positions = ((1, 0), (1, 1), (1, 2), (1, 3))
    heading(
        draw,
        "One LLM plan, four Jev decisions",
        "Route: Right -> Right -> Right -> Up",
        f"FROZENLAKE  ·  STEP {stage + 1}/4",
    )

    grid_x, grid_y, cell = 65, 175, 70
    for row in range(4):
        for col in range(4):
            x0, y0 = grid_x + col * cell, grid_y + row * cell
            fill = "#0b2426" if (row, col) == (0, 3) else PANEL_2
            if (row, col) == (3, 3):
                fill = "#29161f"
            if (row, col) == positions[stage]:
                fill = "#17375e"
            draw.rounded_rectangle((x0, y0, x0 + 58, y0 + 58), 10, fill=fill, outline=BORDER)
            mark = "G" if (row, col) == (0, 3) else "O" if (row, col) == (3, 3) else ""
            if (row, col) == positions[stage]:
                mark = "P"
            if mark:
                color = ACCENT if mark == "G" else RED if mark == "O" else BLUE
                draw.text((x0 + 19, y0 + 13), mark, fill=color, font=font(25, True, mono=True))

    draw.rounded_rectangle((390, 175, 894, 455), 16, fill="#0c1322", outline=BORDER)
    draw.text((418, 201), "CURRENT STATE", fill=MUTED, font=font(13, True))
    draw.text((418, 233), f"player {positions[stage]}  ->  goal (0, 3)", fill=TEXT, font=font(19, True))
    draw.text((418, 285), "JEV CHOICE", fill=ACCENT, font=font(13, True))
    draw.text((418, 319), actions[stage], fill=TEXT, font=font(29, True))
    draw.text((697, 325), f"{confidences[stage]:.0%}", fill=ACCENT, font=font(23, True))
    draw.rounded_rectangle((418, 370, 850, 390), 10, fill="#233149")
    draw.rounded_rectangle((418, 370, 418 + int(432 * confidences[stage]), 390), 10, fill=ACCENT)
    result = "GOAL -> REWARD 1" if stage == 3 else "safe state -> continue"
    draw.text((418, 415), result, fill=ACCENT if stage == 3 else MUTED, font=font(17, True))
    draw.text((44, 485), "Recorded seed 3000: LLM calls 4 -> 1  |  tokens 2,338 -> 663  |  reward 1", fill=MUTED, font=font(14))
    return image


def frozen_lake_comparison_frame() -> Image.Image:
    image, draw = canvas()
    heading(draw, "Success stays at 100%; LLM work falls", "Ten paired FrozenLake tasks with GPT-5.6-sol", "FROZENLAKE RESULT")
    metrics = (
        ("Success", "100%", "100%", "same"),
        ("Mean LLM calls", "4.5", "1.6", "-64.4%"),
        ("Mean tokens", "3,058", "1,128", "-63.1%"),
        ("Mean time", "14.8s", "9.2s", "-37.6%"),
    )
    draw.text((52, 181), "METRIC", fill=MUTED, font=font(13, True))
    draw.text((365, 181), "LLM ONLY", fill=MUTED, font=font(13, True))
    draw.text((552, 181), "LLM + JEV", fill=ACCENT, font=font(13, True))
    draw.text((760, 181), "CHANGE", fill=ACCENT, font=font(13, True))
    for index, (name, baseline, harness, change) in enumerate(metrics):
        y = 218 + index * 58
        draw.rounded_rectangle((44, y, 916, y + 46), 10, fill=PANEL_2)
        draw.text((62, y + 13), name, fill=TEXT, font=font(15, True))
        draw.text((365, y + 12), baseline, fill=MUTED, font=font(17, True))
        draw.text((552, y + 12), harness, fill=TEXT, font=font(17, True))
        draw.text((760, y + 12), change, fill=ACCENT, font=font(17, True))
    draw.text((44, 478), "The LLM delegates the route once; Jev chooses each next move from the new state.", fill=TEXT, font=font(16, True))
    return image


def summary_frame() -> Image.Image:
    image, draw = canvas()
    heading(draw, "Delegate selection, not problem solving", "Replace useful local calls; keep reasoning and recovery with the LLM", "KEY TAKEAWAY")

    cards = (
        (44, "Controlled", "100% success", "LLM calls -64.4%", ACCENT),
        (342, "WebShop", "50% -> 60%", "success +10 points", BLUE),
        (640, "Terminal-Bench", "1/6 -> 3/6", "LLM calls -9.0%", AMBER),
    )
    for x, title, metric, note, color in cards:
        draw.rounded_rectangle((x, 177, x + 276, 306), 16, fill=PANEL_2, outline=color, width=2)
        draw.text((x + 20, 197), title, fill=MUTED, font=font(14, True))
        draw.text((x + 20, 231), metric, fill=TEXT, font=font(23, True))
        draw.text((x + 20, 270), note, fill=color, font=font(14, True))

    draw.rounded_rectangle((44, 333, 470, 450), 15, fill="#0b2426", outline=ACCENT, width=1)
    draw.text((66, 355), "GOOD FIT", fill=ACCENT, font=font(13, True))
    draw.text((66, 385), "Bounded + reversible + locally judgeable", fill=TEXT, font=font(15, True))
    draw.text((66, 416), "2-4 materially different valid options", fill=MUTED, font=font(13))

    draw.rounded_rectangle((490, 333, 916, 450), 15, fill="#29161f", outline=RED, width=1)
    draw.text((512, 355), "NOT A FIX", fill=RED, font=font(13, True))
    draw.text((512, 385), "Missing task capability or global reasoning", fill=TEXT, font=font(15, True))
    draw.text((512, 416), "Jev cannot rescue an incapable agent", fill=MUTED, font=font(13))
    draw.text((44, 482), "96 controlled/web pairs  +  6 terminal pairs", fill=MUTED, font=font(14))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/demos/jev-agent-harness.gif")
    parser.add_argument("--sqlite-out", default="docs/demos/jev-agent-harness-sqlite.gif")
    parser.add_argument("--webshop-out", default="docs/demos/jev-agent-harness-webshop.gif")
    parser.add_argument("--frozen-lake-out", default="docs/demos/jev-agent-harness-frozen-lake.gif")
    parser.add_argument("--trace", default="docs/demos/jev-agent-harness-traces.json")
    args = parser.parse_args()

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    assert trace["summary"] == {
        "controlled_and_web_pairs": 96,
        "terminal_pairs": 6,
        "frozen_lake": {"success": "100% -> 100%", "llm_calls_change": "-64.4%"},
        "webshop": {"success": "50% -> 60%"},
        "terminal_bench": {"success": "1/6 -> 3/6", "llm_calls_change": "-9.0%"},
    }
    sqlite = trace["sqlite"]
    assert [option["name"] for option in sqlite["options"]] == [
        "Inspect raw page", "Read SQLite schema", "Check available tools"
    ]
    assert sqlite["jev_choice"] == "Inspect raw page" and sqlite["confidence"] == 0.81
    assert sqlite["baseline"] == {
        "reward": 1, "llm_calls": 15, "tokens": 202050, "wall_time_seconds": 187.9,
        "source": "runs/terminal-bench/frontier-v4/sqlite-db-truncate/d0/jobs/jev-tb2-sqlite-db-truncate-baseline/sqlite-db-truncate__JEG29Mk/result.json",
        "sha256": "009be2b0f3b3302240186e30981e869408a5d8434eb3b065ef565b91e5b83546",
    }
    assert sqlite["jev_harness"] == {
        "reward": 1, "llm_calls": 8, "tokens": 121293, "wall_time_seconds": 144.7,
        "source": "runs/terminal-bench/frontier-v4/sqlite-db-truncate/d100/jobs/jev-tb2-sqlite-db-truncate-optional/sqlite-db-truncate__yq4Z9qx/result.json",
        "sha256": "d64fc17bbc5fcc6119f6f6d90ba600452c79bcc781ca1a172f46431057d10ee4",
    }

    webshop = trace["webshop"]
    assert webshop["seed"] == 3106
    assert webshop["goal"] == "Men's lace-up boots, black, size 11.5, under $160"
    assert webshop["baseline"]["reward"] == 0 and webshop["baseline"]["llm_calls"] == 11
    assert webshop["baseline"]["tokens"] == 47360 and webshop["baseline"]["wall_time_seconds"] == 20.9
    assert webshop["jev_harness"]["llm_calls"] == 5
    assert webshop["jev_harness"]["tokens"] == 15705
    assert webshop["jev_harness"]["wall_time_seconds"] == 9.9
    assert trace["webshop"]["jev_harness"]["path"][2] == {
        "controller": "jev", "action": "click[black]", "confidence": 0.89,
        "action_is_effective": False,
    }
    assert webshop["jev_harness"]["reward"] == 1

    frozen = trace["frozen_lake"]
    assert [step["action"] for step in frozen["steps"]] == ["Right", "Right", "Right", "Up"]
    assert [step["confidence"] for step in frozen["steps"]] == [0.99, 0.99, 0.82, 1.0]
    assert frozen["pair"] == {
        "baseline": {"reward": 1, "llm_calls": 4, "tokens": 2338, "wall_time_seconds": 19.7},
        "jev_harness": {"reward": 1, "llm_calls": 1, "tokens": 663, "wall_time_seconds": 16.7},
    }
    assert frozen["ten_pair_result"] == {
        "baseline_success": 1.0, "jev_harness_success": 1.0,
        "baseline_mean_llm_calls": 4.5, "jev_harness_mean_llm_calls": 1.6,
        "baseline_mean_tokens": 3057.6, "jev_harness_mean_tokens": 1128.0,
        "baseline_mean_wall_time_seconds": 14.8, "jev_harness_mean_wall_time_seconds": 9.2,
    }

    for demo in (frozen, webshop):
        source = Path(demo["source"])
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == demo["source_sha256"]
    for run in (sqlite["baseline"], sqlite["jev_harness"]):
        source = Path(run["source"])
        if source.exists():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == run["sha256"]

    sqlite_frames = [sqlite_frame(i) for i in range(3)] + [sqlite_comparison_frame()]
    webshop_frames = [webshop_frame(i) for i in range(3)] + [webshop_comparison_frame()]
    frozen_lake_frames = [frozen_lake_frame(i) for i in range(4)] + [frozen_lake_comparison_frame()]

    def save(output_name: str, frames: list[Image.Image], durations: list[int]) -> None:
        output = Path(output_name)
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
        print(output)

    save(args.sqlite_out, sqlite_frames, [1500, 1600, 1900, 2300])
    save(args.webshop_out, webshop_frames, [1500, 1600, 1900, 2300])
    save(args.frozen_lake_out, frozen_lake_frames, [1100, 1100, 1100, 1800, 2500])
    save(
        args.out,
        sqlite_frames + webshop_frames + frozen_lake_frames + [summary_frame()],
        [1500, 1600, 1900, 2300, 1500, 1600, 1900, 2300, 1100, 1100, 1100, 1800, 2500, 2500],
    )


if __name__ == "__main__":
    main()
