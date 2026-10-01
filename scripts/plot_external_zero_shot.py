#!/usr/bin/env python3
"""Render the external zero-shot README comparison from its tracked artifact."""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/external-zero-shot-v1.json"
SVG_OUT = ROOT / "docs/external-zero-shot.svg"
PNG_OUT = ROOT / "docs/external-zero-shot.png"

INK = "#213248"
MUTED = "#64748B"
RULE = "#E4E9EF"
COLORS = {
    "ours": "#278577",
    "open_rerun": "#8493A6",
    "published_only": "#A17BB7",
}
HATCHES = {"ours": None, "open_rerun": None, "published_only": "////"}


def load_results() -> dict:
    data = json.loads(SOURCE.read_text())
    for suite in ("typed_decisions", "jevjudge_text"):
        seen = set()
        for row in data[suite]["models"]:
            if row["model"] in seen:
                raise ValueError(f"{suite}: duplicate model {row['model']}")
            seen.add(row["model"])
            score = row["accuracy"]
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise ValueError(f"{suite}: invalid accuracy for {row['model']}")
            if row["kind"] not in COLORS:
                raise ValueError(f"{suite}: invalid kind for {row['model']}")
        data[suite]["models"].sort(key=lambda row: (-row["accuracy"], row["model"]))
    return data


def short_label(name: str) -> str:
    replacements = {
        "JevAny-Qwen3.8-27B": "JevAny Qwen3.8 27B",
        "JevAny-Qwen3.5-4B-Direct-Token": "JevAny Qwen3.5 4B · DT",
        "JevAny-Qwen3.5-4B": "JevAny Qwen3.5 4B",
        "JevAny-Muse-Glimmer-30B": "JevAny Muse 30B",
        "JevAny-Gemma-4B": "JevAny Gemma 4B",
        "prima-ratio + Gemma4 12B": "prima-ratio + Gemma 12B",
        "convaiinnovations/laya": "Laya",
    }
    return replacements.get(name, name)


def draw_panel(ax, rows: list[dict], title: str, scope: str) -> None:
    labels = [short_label(row["model"]) for row in rows]
    values = [row["accuracy"] * 100 for row in rows]
    bars = ax.barh(
        range(len(rows)),
        values,
        height=0.67,
        color=[COLORS[row["kind"]] for row in rows],
        edgecolor=["#79558D" if row["kind"] == "published_only" else "none" for row in rows],
        linewidth=0.6,
        zorder=3,
    )
    for bar, row, value in zip(bars, rows, values):
        bar.set_hatch(HATCHES[row["kind"]])
        ax.text(
            min(value + 1.0, 80.7),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left" if value < 77 else "right",
            va="center",
            fontsize=9.4,
            color=INK,
            weight="bold" if row["kind"] == "ours" else "normal",
        )
    ax.set_title(title, loc="left", fontsize=17, color=INK, weight="bold", pad=24)
    ax.text(0, 1.015, scope, transform=ax.transAxes, fontsize=10.5, color=MUTED)
    ax.set_xlim(0, 82)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0", "20", "40", "60", "80"], fontsize=9.5, color=MUTED)
    ax.set_yticks(range(len(rows)), labels, fontsize=9.5, color=INK)
    for tick, row in zip(ax.get_yticklabels(), rows):
        if row["kind"] == "ours":
            tick.set_weight("bold")
    ax.invert_yaxis()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0, color=RULE, linewidth=1)


def main() -> None:
    data = load_results()
    order = data["main_comparison_order"]
    panels = {}
    for suite in ("typed_decisions", "jevjudge_text"):
        by_model = {row["model"]: row for row in data[suite]["models"]}
        missing = [model for model in order if model not in by_model]
        if missing:
            raise ValueError(f"{suite}: missing common-cohort models: {missing}")
        panels[suite] = [by_model[model] for model in order]
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "jevany-external-zero-shot-v1",
        "text.color": INK,
        "hatch.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), dpi=100, facecolor="white")
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.17, top=0.72, wspace=0.48)
    fig.text(0.035, 0.955, "External zero-shot decision accuracy", fontsize=24, weight="bold")
    fig.text(
        0.035,
        0.910,
        "The same ten models in the same order · full public splits",
        fontsize=12.5,
        color=MUTED,
    )
    legend = [
        Patch(facecolor=COLORS["ours"], label="JevAny"),
        Patch(facecolor=COLORS["open_rerun"], label="Open · locally rerun"),
    ]
    fig.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.975),
        ncol=2,
        frameon=False,
        fontsize=11.5,
        handlelength=1.2,
        columnspacing=1.7,
    )
    draw_panel(
        axes[0],
        panels["typed_decisions"],
        "Typed Decisions",
        "2,000 decisions · teacher agreement (%)",
    )
    draw_panel(
        axes[1],
        panels["jevjudge_text"],
        "JevJudge common text set",
        "724/724 full-context requests · accuracy (%)",
    )
    fig.text(
        0.035,
        0.045,
        "JevJudge text is a four-role diagnostic, not the official full-multimodal headline; native inputs use the "
        "same 65,536-token ceiling with no truncation.",
        fontsize=10.2,
        color=MUTED,
    )
    description = " ".join(
        f"{suite}: "
        + ", ".join(f"{row['model']} {row['accuracy'] * 100:.2f}%" for row in panels[key])
        for suite, key in (
            ("Typed Decisions", "typed_decisions"),
            ("JevJudge common text set", "jevjudge_text"),
        )
    )
    metadata = {
        "Date": None,
        "Title": "External zero-shot decision accuracy",
        "Description": description,
    }
    fig.savefig(SVG_OUT, metadata=metadata)
    SVG_OUT.write_text("\n".join(line.rstrip() for line in SVG_OUT.read_text().splitlines()) + "\n")
    fig.savefig(PNG_OUT, dpi=160, metadata={"Software": "JevAny plot_external_zero_shot.py"})
    plt.close(fig)
    print(f"Wrote {SVG_OUT.relative_to(ROOT)} and {PNG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
