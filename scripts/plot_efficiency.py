#!/usr/bin/env python3
"""Plot accuracy against measured inference latency from the efficiency report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter


JEVANY = "#2f7ed8"
JEVANY_ARROW = "#7db7f4"
OTHER = "#8c8d89"
OTHER_ARROW = "#c2c2bf"
FRONTIER = "#f26634"

LABELS = {
    "JevAny-Qwen3.5-4B": "JevAny-4B",
    "JevAny-Qwen3.5-4B-Direct-Token": "JevAny-4B-DT",
    "JevAny-Gemma-4B": "JevAny-Gemma-E4B",
    "JevAny-Qwen3.8-27B": "JevAny-27B",
    "JevAny-Muse-Glimmer-30B": "JevAny-Muse-30B",
    "Open-Jev-27B-v1.1": "Open-Jev-27B",
    "Open-Jev-9B": "Open-Jev-9B",
    "Bespoke-Nimble-9B": "Nimble-9B",
    "Intern-Decision-4B": "Intern-4B",
    "Intern-Decision-2B": "Intern-2B",
    "Intern-Decision-0.8B": "Intern-0.8B",
    "decider-4b": "decider-4b",
    "decider-2b": "decider-2b",
    "Jev-Omni": "Jev-Omni-12B",
    "Laya": "Laya-0.4B",
}

OFFSETS = {
    "transfer": {
        "JevAny-Qwen3.5-4B": (-12, 18, "right"),
        "JevAny-Qwen3.5-4B-Direct-Token": (-12, -22, "right"),
        "JevAny-Gemma-4B": (-12, 0, "right"),
        "JevAny-Qwen3.8-27B": (0, 17, "center"),
        "JevAny-Muse-Glimmer-30B": (8, -20, "left"),
        "Bespoke-Nimble-9B": (8, 3, "left"),
        "Intern-Decision-4B": (8, -14, "left"),
        "Intern-Decision-2B": (8, -14, "left"),
        "Intern-Decision-0.8B": (8, -14, "left"),
        "decider-4b": (8, -14, "left"),
        "decider-2b": (8, -14, "left"),
        "Jev-Omni": (-8, 12, "right"),
        "Laya": (8, -12, "left"),
        "Open-Jev-27B-v1.1": (8, -14, "left"),
        "Open-Jev-9B": (8, -14, "left"),
    },
    "jevbench": {
        "JevAny-Qwen3.5-4B": (-12, -14, "right"),
        "JevAny-Qwen3.5-4B-Direct-Token": (-12, 16, "right"),
        "JevAny-Gemma-4B": (-12, -18, "right"),
        "JevAny-Qwen3.8-27B": (0, 17, "center"),
        "JevAny-Muse-Glimmer-30B": (8, -20, "left"),
        "Bespoke-Nimble-9B": (8, -14, "left"),
        "Intern-Decision-4B": (0, 15, "center"),
        "Intern-Decision-2B": (8, -14, "left"),
        "Intern-Decision-0.8B": (8, -14, "left"),
        "decider-4b": (8, -13, "left"),
        "decider-2b": (8, 9, "left"),
        "Jev-Omni": (8, -14, "left"),
        "Laya": (8, -12, "left"),
        "Open-Jev-27B-v1.1": (8, -14, "left"),
        "Open-Jev-9B": (8, -14, "left"),
    },
}


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for row in rows:
        result.setdefault(row["model"], []).append(row)
    return result


def before_and_after(rows: list[dict], suite: str) -> tuple[dict | None, list[dict], dict]:
    before = next((row for row in rows if row["configuration"] == "Default"), None)
    accelerated = [row for row in rows if row["configuration"].startswith("+")]
    if not accelerated:
        accelerated = rows
    best = min(accelerated, key=lambda row: row[suite]["median_ms"])
    return before, accelerated, best


def frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    result = []
    best_accuracy = float("-inf")
    for point in sorted(points):
        if point[1] > best_accuracy:
            result.append(point)
            best_accuracy = point[1]
    return result


def plot(data: dict, output: Path) -> None:
    models = grouped(data["rows"])
    fig, axes = plt.subplots(1, 2, figsize=(18, 10.133), dpi=150)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.19, wspace=0.17)
    best_points: dict[str, list[tuple[float, float, str]]] = {"transfer": [], "jevbench": []}

    for ax, suite, title in zip(axes, ("transfer", "jevbench"), ("Transfer", "JevBench public")):
        for model, rows in models.items():
            ours = model.startswith("JevAny-")
            color = JEVANY if ours else OTHER
            arrow = JEVANY_ARROW if ours else OTHER_ARROW
            before, accelerated, best = before_and_after(rows, suite)
            bx = before[suite]["median_ms"] if before else None
            by = before[suite]["accuracy"] if before else None
            ax.scatter(
                [row[suite]["median_ms"] for row in accelerated],
                [row[suite]["accuracy"] for row in accelerated],
                s=85 if ours else 58, c=color, edgecolors="white", linewidths=1.3, zorder=4,
            )
            if before:
                ax.scatter([bx], [by], s=90 if ours else 62, facecolors="white",
                           edgecolors=color, linewidths=2, zorder=5)
                ax.annotate("", xy=(best[suite]["median_ms"], best[suite]["accuracy"]),
                            xytext=(bx, by),
                            arrowprops=dict(arrowstyle="-|>", color=arrow, lw=1.8), zorder=2)
            x, y = best[suite]["median_ms"], best[suite]["accuracy"]
            best_points[suite].append((x, y, model))
            dx, dy, align = OFFSETS.get(suite, {}).get(model, (7, -10, "left"))
            ax.annotate(LABELS.get(model, model), (x, y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=10.5,
                        color="#151515" if ours else "#666666",
                        fontweight="bold" if ours else "normal", ha=align)

        edge = frontier(best_points[suite])
        ax.plot([p[0] for p in edge], [p[1] for p in edge], color=FRONTIER, lw=2.2, zorder=1)
        for x, y, model in edge:
            if model.startswith("JevAny-"):
                ax.scatter([x], [y], s=520, color="#cfe3fb", alpha=0.9,
                           edgecolors="none", zorder=0)

        ax.set_xscale("log")
        ax.set_xlim(6.7, 3200)
        ax.set_xticks([10, 20, 50, 100, 200, 500, 1000, 2000])
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f"{value / 1000:g} s" if value >= 1000 else f"{value:g} ms"
        ))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, color="#e5e5e2", linewidth=1)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=12)
        ax.set_xlabel("Median latency per request (log scale)", fontsize=13, color="#5d5d5a")
        ax.set_ylabel("Accuracy", fontsize=13, color="#5d5d5a")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
        ax.tick_params(labelsize=11, colors="#5d5d5a")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#ddddda")

    axes[0].set_ylim(49.2, 89.2)
    axes[1].set_ylim(57.7, 91.8)
    fig.suptitle("Accuracy vs measured latency, before and after acceleration",
                 x=0.055, y=0.965, ha="left", fontsize=23, fontweight="bold")
    fig.text(0.055, 0.915,
             "Same A100-40GB nodes, batch size 1, one request = one question; accuracy re-measured after each change.",
             fontsize=13.5, color="#5d5d5a")
    legend = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=JEVANY,
               markeredgecolor="white", markersize=9, label="JevAny, accelerated"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=JEVANY, markeredgewidth=2, markersize=9, label="JevAny, before"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=OTHER,
               markeredgecolor="white", markersize=8, label="Third-party, after"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=OTHER, markeredgewidth=2, markersize=8, label="Third-party, before"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#cfe3fb",
               markeredgecolor="none", markersize=16, label="JevAny on the frontier"),
        Line2D([], [], color=FRONTIER, lw=2.5, label="Pareto frontier (after)"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.052, 0.88),
               ncol=6, frameon=False, fontsize=11.5, handlelength=1.4, columnspacing=1.4)
    fig.text(0.055, 0.055,
             "After: flash-linear-attention + causal-conv1d for models with Gated DeltaNet layers, plus fused SDPA.\n"
             "Single-GPU JevAny models also use CUDA graphs; 27B/30B models stay layer-sharded over 40 GB GPUs.\n"
             "JevAny times the model call; third-party timings include each runtime's tokenization.",
             fontsize=10.5, color="#5d5d5a", wrap=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/efficiency-a100-v1.json")
    parser.add_argument("--output", default="docs/efficiency-latency.png")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    plot(data, Path(args.output))


if __name__ == "__main__":
    main()
