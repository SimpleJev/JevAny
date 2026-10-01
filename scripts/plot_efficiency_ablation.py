"""Animate the JevAny 4B latency ablation from results/efficiency-a100-v1.json.

Writes docs/efficiency-ablation.gif: every Transfer-v9 configuration races on the
same slowed-down clock.
"""

from pathlib import Path
import json

import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "efficiency-a100-v1.json"
GIF_OUT = ROOT / "docs" / "efficiency-ablation.gif"

INK, MUTED, RULE, PAGE = "#213248", "#64748B", "#E4E9EF", "#FCFCFB"
REMAINING, GRAPH, DEFAULT = "#278577", "#E8884A", "#8493A6"

# Only releases that share one GPU and one fixed panel; 27B and 30B are reported
# from H200 runs on different panels, so their latencies must not share a clock.
RELEASES = [
    ("JevAny-Qwen3.5-4B", "JevAny-4B", 1),
    ("JevAny-Qwen3.5-4B-Direct-Token", "JevAny-4B-DT", 1),
    ("JevAny-Gemma-4B", "JevAny-Gemma-4B", 1),
]
# Stage name -> configuration row. Kernels and fused SDPA are one stage: next to
# CUDA graphs they move latency too little to read as separate steps.
STAGES = [
    ("Default", "Default"),
    ("+ kernels, fused SDPA", "+ kernels + fused SDPA"),
    ("+ CUDA graphs", "+ kernels + fused SDPA + CUDA graphs"),
]


def load_ladders(suite: str) -> list[tuple[str, int, list[tuple[str, float]]]]:
    rows = json.loads(SOURCE.read_text())["rows"]
    index = {(r["model"], r["configuration"], r["gpus"]): r for r in rows}
    ladders = []
    for model, short, gpus in RELEASES:
        stops = [
            (stage, index[(model, config, gpus)][suite]["median_ms"])
            for stage, config in STAGES
            if (model, config, gpus) in index
        ]
        ladders.append((short, gpus, stops))
    return ladders


def speedup_text(total: float, final: float) -> str:
    ratio = total / final
    return f"{ratio:.2f}×" if ratio < 2 else f"{ratio:.1f}×"


# GIF: one ms of model latency plays as SLOWDOWN ms of animation.
WIDTH, HEIGHT = 1080, 600
FRAME_MS, SLOWDOWN, START_HOLD_MS, END_HOLD_MS = 40, 20, 240, 900
LANE_COLOR = {"Default": DEFAULT, "+ kernels, fused SDPA": "#6DBBA9", "+ CUDA graphs": REMAINING}
BAR_X0, BAR_X1, LABEL_X = 372, 930, 40
LANE_H, LANE_GAP, GROUP_GAP, TOP = 28, 10, 20, 132


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


def render_frame(ladders, clock_ms: float, axis_ms: float) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), PAGE)
    draw = ImageDraw.Draw(image)
    draw.text((LABEL_X, 26), "How the 4B releases get faster", font=font(26, True), fill=INK)
    draw.text((LABEL_X, 66), "Transfer-v9 median latency per request · one A100-40GB · batch size 1 · "
              f"played {SLOWDOWN}× slower than real time", font=font(14), fill=MUTED)
    draw.text((WIDTH - LABEL_X, 26), f"{min(clock_ms, axis_ms):5.0f} ms", font=font(26, True),
              fill=INK, anchor="ra")

    scale = (BAR_X1 - BAR_X0) / axis_ms
    for tick in range(0, int(axis_ms) + 1, 20):
        x = BAR_X0 + tick * scale
        draw.line([(x, TOP - 14), (x, HEIGHT - 52)], fill=RULE, width=1)
        draw.text((x, HEIGHT - 44), f"{tick} ms", font=font(12), fill=MUTED, anchor="ma")

    y = TOP
    for short, gpus, stops in ladders:
        name = short + ("" if gpus == 1 else f" · {gpus} GPUs")
        draw.text((LABEL_X, y + LANE_H / 2), name, font=font(15, True), fill=INK, anchor="lm")
        total = stops[0][1]
        for stage, latency in stops:
            done = clock_ms >= latency
            draw.text((BAR_X0 - 12, y + LANE_H / 2), stage, font=font(12), fill=MUTED, anchor="rm")
            draw.rounded_rectangle([BAR_X0, y, BAR_X1, y + LANE_H], radius=4, fill="#EEF2F5")
            length = min(clock_ms, latency) * scale
            if length >= 1:
                draw.rounded_rectangle([BAR_X0, y, BAR_X0 + length, y + LANE_H], radius=4,
                                       fill=LANE_COLOR[stage])
            if done:
                label_x = BAR_X0 + length + 8
                draw.text((label_x, y + LANE_H / 2), f"{latency:.0f} ms", font=font(13, True),
                          fill=INK, anchor="lm")
                if stage != "Default":
                    width = draw.textlength(f"{latency:.0f} ms", font=font(13, True))
                    draw.text((label_x + width + 10, y + LANE_H / 2), speedup_text(total, latency),
                              font=font(13, True), fill=GRAPH, anchor="lm")
            y += LANE_H + LANE_GAP
        y += GROUP_GAP
    return image


def draw_gif() -> None:
    ladders = load_ladders("transfer")
    slowest = max(stops[0][1] for _, _, stops in ladders)
    axis_ms = (int(slowest) // 20 + 1) * 20
    play_ms = slowest * SLOWDOWN
    clocks = [0.0] * (START_HOLD_MS // FRAME_MS)
    clocks += [t / SLOWDOWN for t in range(0, int(play_ms) + FRAME_MS, FRAME_MS)]
    clocks += [slowest] * (END_HOLD_MS // FRAME_MS)
    frames = [render_frame(ladders, clock, axis_ms) for clock in clocks]
    palette = frames[-1].quantize(colors=128)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(GIF_OUT, save_all=True, append_images=frames[1:], duration=FRAME_MS,
                   loop=0, optimize=False, disposal=1)


def main() -> None:
    draw_gif()
    print(f"wrote {GIF_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
