"""Draw the JevAny 4B latency ablation from results/efficiency-a100-v1.json.

Writes docs/efficiency-ablation.svg (both suites) and docs/efficiency-ablation.gif
(Transfer-v9, every configuration racing on the same slowed-down clock).
"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "efficiency-a100-v1.json"
SVG_OUT = ROOT / "docs" / "efficiency-ablation.svg"
GIF_OUT = ROOT / "docs" / "efficiency-ablation.gif"

INK, MUTED, RULE, PAGE = "#213248", "#64748B", "#E4E9EF", "#FCFCFB"
REMAINING, KERNEL, GRAPH, DEFAULT = "#278577", "#A8D5CB", "#E8884A", "#8493A6"

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
SUITES = [("transfer", "Transfer-v9"), ("jevbench", "JevBench public")]


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


def draw_svg() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2), facecolor=PAGE,
                             gridspec_kw={"wspace": 0.22})
    fig.text(0.04, 0.95, "Where the latency goes: each acceleration option on the 4B releases",
             ha="left", va="top", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.04, 0.86, "Bar length is the Default median latency, split into what each option "
             "removes and what is left. One A100-40GB, batch size 1.",
             ha="left", va="top", fontsize=11.5, color=MUTED)
    fig.subplots_adjust(left=0.04, right=0.975, top=0.70, bottom=0.2)
    renderer = fig.canvas.get_renderer()

    for ax, (suite, title) in zip(axes, SUITES):
        ladders = load_ladders(suite)
        ax.set_facecolor(PAGE)
        ax.set_xlim(0, max(stops[0][1] for _, _, stops in ladders) * 1.2)
        ax.set_ylim(-0.75, len(ladders) - 0.2)
        to_px = ax.transData.transform

        for y, (short, gpus, stops) in zip(range(len(ladders) - 1, -1, -1), ladders):
            total, final = stops[0][1], stops[-1][1]
            segments = [
                (KERNEL if stage == "+ kernels, fused SDPA" else GRAPH, prev - ms, "#20483F" if stage != "+ CUDA graphs" else "white")
                for (_, prev), (stage, ms) in zip(stops, stops[1:]) if prev - ms > 0
            ]
            segments.append((REMAINING, final, "white"))
            left = 0.0
            for color, width, text_color in segments:
                ax.barh(y, width, left=left, height=0.46, color=color, lw=0)
                label = f"{width:.0f} ms" if color == REMAINING else f"−{width:.0f}"
                text = ax.text(left + width / 2, y, label, ha="center", va="center",
                               fontsize=10, fontweight="bold", color=text_color)
                text_px = text.get_window_extent(renderer).width
                bar_px = to_px((left + width, y))[0] - to_px((left, y))[0]
                if text_px + 8 > bar_px:
                    text.set_position((left + width / 2, y - 0.27))
                    text.set_va("top")
                    text.set_color(MUTED)
                    text.set_fontsize(9)
                left += width

            ax.text(total * 1.02, y, speedup_text(total, final), ha="left", va="center",
                    fontsize=12, fontweight="bold", color=INK)
            ax.text(0, y + 0.31, short + ("" if gpus == 1 else f" · {gpus} GPUs"), ha="left",
                    va="bottom", fontsize=11.5, fontweight="bold", color=INK)
            ax.text(total * 1.02, y + 0.31, f"from {total:.0f} ms", ha="left", va="bottom",
                    fontsize=9, color=MUTED)

        ax.set_title(title, loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=14)
        ax.set_xlabel("Median latency per request (ms)", fontsize=11, color=MUTED)
        ax.set_yticks([])
        ax.tick_params(axis="x", colors=MUTED, labelsize=10)
        ax.xaxis.grid(True, color=RULE, lw=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)

    handles = [Patch(facecolor=KERNEL, label="saved by kernels / fused SDPA"),
               Patch(facecolor=GRAPH, label="saved by CUDA graphs"),
               Patch(facecolor=REMAINING, label="remaining latency")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.035, 0.0), ncol=3,
               frameon=False, fontsize=10.5, labelcolor=INK)
    fig.savefig(SVG_OUT, format="svg", facecolor=PAGE, metadata={"Date": None})
    plt.close(fig)


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
    draw_svg()
    draw_gif()
    print(f"wrote {SVG_OUT.relative_to(ROOT)} and {GIF_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
