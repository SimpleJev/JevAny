"""Animate within-row latency reductions for the 4B and 27–30B releases.

Writes docs/efficiency-ablation.gif. Each row is normalized to its own measured
baseline because the 4B and large-model audits use different hardware or fixed
evaluation panels. Exact median latency remains visible on every completed bar.
"""

from pathlib import Path
import json

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
A100_SOURCE = ROOT / "results" / "efficiency-a100-v1.json"
H200_SOURCE = ROOT / "results" / "efficiency-h200-best-v1.json"
GIF_OUT = ROOT / "docs" / "efficiency-ablation.gif"

INK, MUTED, RULE, PAGE = "#213248", "#64748B", "#E4E9EF", "#FCFCFB"
GRAPH, DEFAULT = "#E8884A", "#8493A6"
LANE_COLOR = {
    "Default": DEFAULT,
    "Graphs off": DEFAULT,
    "+ kernels, fused SDPA": "#6DBBA9",
    "+ fused SDPA": "#6DBBA9",
    "+ CUDA graphs": "#278577",
}

SMALL_RELEASES = [
    ("JevAny-Qwen3.5-4B", "Qwen3.5-4B"),
    ("JevAny-Qwen3.5-4B-Direct-Token", "Qwen3.5-4B-DT"),
    ("JevAny-Gemma-4B", "Gemma-4B"),
]
SMALL_STAGES = [
    ("Default", "Default"),
    ("+ kernels, fused SDPA", "+ kernels + fused SDPA"),
    ("+ CUDA graphs", "+ kernels + fused SDPA + CUDA graphs"),
]


def load_groups() -> list[dict]:
    a100_rows = json.loads(A100_SOURCE.read_text())["rows"]
    a100_index = {(row["model"], row["configuration"], row["gpus"]): row for row in a100_rows}
    small_rows = []
    for model, short in SMALL_RELEASES:
        stops = [
            (stage, a100_index[(model, configuration, 1)]["transfer"]["median_ms"])
            for stage, configuration in SMALL_STAGES
        ]
        small_rows.append({"name": short, "panel": "", "digits": 1, "stops": stops})

    h200_rows = {row["model"]: row for row in json.loads(H200_SOURCE.read_text())["rows"]}
    qwen = h200_rows["JevAny-Qwen3.8-27B"]
    muse = h200_rows["JevAny-Muse-Glimmer-30B"]
    large_rows = [
        {
            "name": "Qwen3.8-27B",
            "panel": "JevBench · 231",
            "digits": 2,
            "stops": [
                ("Graphs off", qwen["before"]["median_ms"]),
                ("+ CUDA graphs", qwen["after"]["median_ms"]),
            ],
        },
        {
            "name": "Muse-Glimmer-30B",
            "panel": "Transfer · 44",
            "digits": 2,
            "stops": [
                ("Default", muse["before"]["median_ms"]),
                ("+ fused SDPA", muse["intermediate"]["median_ms"]),
                ("+ CUDA graphs", muse["after"]["median_ms"]),
            ],
        },
    ]
    return [
        {
            "title": "4B releases",
            "detail": "1× A100-40GB · Transfer-v9 · 1,046 scored",
            "rows": small_rows,
        },
        {
            "title": "27–30B releases",
            "detail": "1× H200 · fixed panel shown per row",
            "rows": large_rows,
        },
    ]


def speedup_text(total: float, final: float, digits: int) -> str:
    ratio = total / final
    return f"{ratio:.2f}×" if digits == 2 or ratio < 2 else f"{ratio:.1f}×"


def latency_text(latency: float, digits: int) -> str:
    return f"{latency:.{digits}f} ms"


# Render at 1.5× density so labels stay crisp at GitHub README width.
SCALE = 1.5
WIDTH, HEIGHT = round(1080 * SCALE), round(810 * SCALE)
FRAME_MS, PLAY_MS, START_HOLD_MS, END_HOLD_MS = 40, 2200, 240, 900
BAR_X0, BAR_X1, LABEL_X = (round(value * SCALE) for value in (390, 850, 40))
LANE_H, LANE_GAP, MODEL_GAP, GROUP_GAP, TOP = (
    round(value * SCALE) for value in (24, 7, 14, 22, 128)
)


def px(value: float) -> int:
    return round(value * SCALE)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(
        str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), px(size)
    )


def render_frame(groups: list[dict], progress: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAGE)
    draw = ImageDraw.Draw(image)
    draw.text((LABEL_X, px(24)), "How JevAny gets faster — 4B to 30B", font=font(26, True), fill=INK)
    draw.text(
        (LABEL_X, px(64)),
        "Median latency per request · batch size 1 · each row normalized to its own baseline",
        font=font(14),
        fill=MUTED,
    )
    draw.text(
        (WIDTH - LABEL_X, px(27)),
        f"{min(progress, 1.0) * 100:3.0f}%",
        font=font(23, True),
        fill=INK,
        anchor="ra",
    )

    axis_y = HEIGHT - px(48)
    for tick in range(0, 101, 20):
        x = BAR_X0 + (BAR_X1 - BAR_X0) * tick / 100
        draw.line([(x, TOP + px(25)), (x, axis_y - px(12))], fill=RULE, width=px(1))
        draw.text((x, axis_y), f"{tick}%", font=font(11), fill=MUTED, anchor="ma")

    y = TOP
    for group_index, group in enumerate(groups):
        draw.text((LABEL_X, y), group["title"], font=font(15, True), fill=INK, anchor="la")
        title_width = draw.textlength(group["title"], font=font(15, True))
        draw.text(
            (LABEL_X + title_width + px(12), y + px(1)),
            group["detail"],
            font=font(12),
            fill=MUTED,
            anchor="la",
        )
        draw.line([(LABEL_X, y + px(25)), (WIDTH - LABEL_X, y + px(25))], fill=RULE, width=px(1))
        y += px(38)

        for row in group["rows"]:
            total = row["stops"][0][1]
            for stop_index, (stage, latency) in enumerate(row["stops"]):
                center_y = y + LANE_H / 2
                if stop_index == 0:
                    draw.text((LABEL_X, center_y), row["name"], font=font(13, True), fill=INK, anchor="lm")
                    if row["panel"]:
                        name_width = draw.textlength(row["name"], font=font(13, True))
                        draw.text(
                            (LABEL_X + name_width + px(8), center_y),
                            row["panel"],
                            font=font(10),
                            fill=MUTED,
                            anchor="lm",
                        )
                draw.text(
                    (BAR_X0 - px(12), center_y), stage, font=font(11), fill=MUTED, anchor="rm"
                )
                draw.rounded_rectangle(
                    [BAR_X0, y, BAR_X1, y + LANE_H], radius=px(4), fill="#EEF2F5"
                )
                relative_latency = latency / total
                length = min(progress, relative_latency) * (BAR_X1 - BAR_X0)
                if length >= 1:
                    draw.rounded_rectangle(
                        [BAR_X0, y, BAR_X0 + length, y + LANE_H],
                        radius=px(4),
                        fill=LANE_COLOR[stage],
                    )
                if progress >= relative_latency:
                    label_x = BAR_X0 + length + px(8)
                    latency_label = latency_text(latency, row["digits"])
                    draw.text(
                        (label_x, center_y), latency_label, font=font(12, True), fill=INK, anchor="lm"
                    )
                    if stop_index:
                        latency_width = draw.textlength(latency_label, font=font(12, True))
                        draw.text(
                            (label_x + latency_width + px(8), center_y),
                            speedup_text(total, latency, row["digits"]),
                            font=font(12, True),
                            fill=GRAPH,
                            anchor="lm",
                        )
                y += LANE_H + LANE_GAP
            y += MODEL_GAP - LANE_GAP
        if group_index + 1 < len(groups):
            y += GROUP_GAP

    draw.text(
        (LABEL_X, HEIGHT - px(18)),
        "Hardware and panels differ; compare optimization stages only within each row.",
        font=font(11),
        fill=MUTED,
        anchor="ld",
    )
    return image


def draw_gif() -> None:
    groups = load_groups()
    progresses = [0.0] * (START_HOLD_MS // FRAME_MS)
    progresses += [t / PLAY_MS for t in range(0, PLAY_MS + FRAME_MS, FRAME_MS)]
    progresses += [1.0] * (END_HOLD_MS // FRAME_MS)
    frames = [render_frame(groups, progress) for progress in progresses]
    palette = frames[-1].quantize(colors=128)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    frames[0].save(
        GIF_OUT,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False,
        disposal=1,
    )


def main() -> None:
    draw_gif()
    print(f"wrote {GIF_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
