#!/usr/bin/env python3
"""Render the external zero-shot README comparison from its tracked artifact."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/external-zero-shot-v1.json"
SVG_OUT = ROOT / "docs/external-zero-shot.svg"
PNG_OUT = ROOT / "docs/external-zero-shot.png"

INK = "#213248"
MUTED = "#64748B"
RULE = "#E4E9EF"
COLORS = {
    "ours": "#278577",
    "open_kev": "#9A6FB0",
    "open_rerun": "#8493A6",
    "published_only": "#A17BB7",
}
HATCHES = {"ours": None, "open_kev": None, "open_rerun": None, "published_only": "////"}
PANELS = (
    {
        "key": "typed_decisions",
        "metric": "accuracy",
        "title": "Typed Decisions",
        "scope": "2,000 decisions · accuracy (%)",
        "xmax": 82,
        "ticks": [0, 20, 40, 60, 80],
        "score_range": (0, 1),
        "coverage": None,
    },
    {
        "key": "jevjudge_full",
        "metric": "accuracy",
        "title": "JevJudge full",
        "scope": "3,220 text / image / video · accuracy (%)",
        "xmax": 72,
        "ticks": [0, 20, 40, 60],
        "score_range": (0, 1),
        "coverage": 3220,
    },
    {
        "key": "jevjudge_text",
        "metric": "accuracy",
        "title": "JevJudge text-only",
        "scope": "724-record text subset · accuracy (%)",
        "xmax": 72,
        "ticks": [0, 20, 40, 60],
        "score_range": (0, 1),
        "coverage": 724,
    },
)


def _finite_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _unavailable_status(status) -> bool:
    if not isinstance(status, str) or not status.strip():
        return False
    normalized = status.strip().lower()
    return normalized.startswith("incompatible:") or "no matching" in normalized


def _validate_row(row: dict, panel: dict) -> None:
    suite = panel["key"]
    model = row["model"]
    metric = panel["metric"]
    score = row.get(metric)
    unavailable = score is None

    if unavailable:
        status = row.get("status")
        normalized = status.strip().lower() if isinstance(status, str) else ""
        if suite == "typed_decisions" and "no matching" not in normalized:
            raise ValueError(f"{suite}: unavailable {model} needs a no-matching status")
        if suite == "jevjudge_full" and not (
            normalized.startswith("incompatible:") or "no matching" in normalized
        ):
            raise ValueError(f"{suite}: unavailable {model} needs an incompatibility or no-matching status")
        if suite == "jevjudge_text" and not (
            row.get("kind") == "published_only" and "no matching" in normalized
        ):
            raise ValueError(f"{suite}: unavailable {model} needs a published no-matching status")
        fields = ("accuracy", "skill_role", "skill_role_ci_95", "answered", "requested")
        present = [field for field in fields if row.get(field) is not None]
        if present:
            raise ValueError(f"{suite}: unavailable {model} has non-null fields: {present}")
        return

    if _unavailable_status(row.get("status")):
        raise ValueError(f"{suite}: scored {model} has an unavailable status")
    low, high = panel["score_range"]
    if not _finite_number(score) or (low is not None and score < low) or score > high:
        raise ValueError(f"{suite}: invalid {metric} for {model}")

    accuracy = row.get("accuracy")
    if not _finite_number(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError(f"{suite}: invalid accuracy for {model}")

    coverage = panel["coverage"]
    if coverage is not None and (row.get("answered"), row.get("requested")) != (coverage, coverage):
        raise ValueError(f"{suite}: {model} must have exact {coverage}/{coverage} coverage")

    skill_role = row.get("skill_role")
    ci = row.get("skill_role_ci_95")
    if suite == "jevjudge_full":
        if not _finite_number(skill_role) or skill_role > 1:
            raise ValueError(f"{suite}: invalid skill_role for {model}")
        if not isinstance(ci, list) or len(ci) != 2 or not all(_finite_number(value) for value in ci):
            raise ValueError(f"{suite}: invalid skill_role CI for {model}")
        ci_low, ci_high = ci
        if not ci_low <= skill_role <= ci_high <= 1:
            raise ValueError(f"{suite}: skill_role CI must be ordered and contain the score for {model}")
    elif skill_role is not None or ci is not None:
        raise ValueError(f"{suite}: unexpected skill_role data for {model}")


def load_results(source: Path = SOURCE) -> dict:
    data = json.loads(source.read_text())
    order = data["main_comparison_order"]
    if len(order) != 13 or len(order) != len(set(order)) or not all(isinstance(model, str) and model for model in order):
        raise ValueError("main comparison order must contain 13 unique, non-empty model names")
    cohort_kinds = {}
    for panel in PANELS:
        suite = data[panel["key"]]
        if suite.get("chart_metric") != panel["metric"]:
            raise ValueError(f"{panel['key']}: chart_metric must be {panel['metric']}")
        by_model = {}
        for row in suite["models"]:
            model = row.get("model")
            if not isinstance(model, str) or not model:
                raise ValueError(f"{panel['key']}: invalid model name")
            if model in by_model:
                raise ValueError(f"{panel['key']}: duplicate model {model}")
            if row.get("kind") not in COLORS:
                raise ValueError(f"{panel['key']}: invalid kind for {model}")
            by_model[model] = row
        missing = [model for model in order if model not in by_model]
        if missing:
            raise ValueError(f"{panel['key']}: missing fixed-cohort models: {missing}")
        ordered = [by_model[model] for model in order]
        for row in ordered:
            model = row["model"]
            previous = cohort_kinds.setdefault(model, row["kind"])
            if row["kind"] != previous:
                raise ValueError(f"{panel['key']}: inconsistent kind for {model}")
            _validate_row(row, panel)
        suite["ordered_models"] = ordered
    return data


def short_label(name: str) -> str:
    replacements = {
        "JevAny-Qwen3.8-27B": "JevAny Qwen3.8 27B",
        "JevAny-Qwen3.5-4B-Direct-Token": "JevAny Qwen3.5 4B · DT",
        "JevAny-Qwen3.5-4B": "JevAny Qwen3.5 4B · Pointer",
        "JevAny-Muse-Glimmer-30B": "JevAny Muse 30B",
        "JevAny-Gemma-4B": "JevAny Gemma 4B",
        "TypeSafe Jev 1.13.0": "TypeSafe Jev 1.13.0",
        "OpenDecider-small": "OpenDecider small",
        "Bongard-mini": "Bongard mini",
        "Jeff-Gemma4-E2B": "Jeff Gemma4 E2B",
        "Jeff-Qwen3.5-2B": "Jeff Qwen3.5 2B",
        "Jeff-Qwen3.5-0.8B": "Jeff Qwen3.5 0.8B",
        "Kev-27B": "Kev 27B",
        "Kev-4B": "Kev 4B",
    }
    return replacements.get(name, name)


def draw_panel(ax, rows: list[dict], panel: dict, show_labels: bool) -> None:
    metric = panel["metric"]
    for index, row in enumerate(rows):
        score = row.get(metric)
        color = COLORS[row["kind"]]
        if score is None:
            ax.text(
                panel["xmax"] * 0.018,
                index,
                "—",
                ha="left",
                va="center",
                fontsize=14,
                color="#A8B1BD",
                weight="bold",
            )
            continue
        value = score * 100
        published = row["kind"] == "published_only"
        bar = ax.barh(
            index,
            value,
            height=0.66,
            color=color,
            edgecolor="#79558D" if published else "none",
            linewidth=0.7 if published else 0,
            zorder=3,
        )
        bar[0].set_hatch(HATCHES[row["kind"]])
        ax.text(
            min(value + panel["xmax"] * 0.018, panel["xmax"] * 0.985),
            index,
            f"{value:.1f}",
            ha="right" if value > panel["xmax"] * 0.91 else "left",
            va="center",
            fontsize=9.1,
            color=INK,
            weight="bold" if row["kind"] == "ours" else "normal",
        )

    ax.set_title(panel["title"], loc="left", fontsize=16, color=INK, weight="bold", pad=25)
    ax.text(0, 1.014, panel["scope"], transform=ax.transAxes, fontsize=10.1, color=MUTED)
    ax.set_xlim(0, panel["xmax"])
    ax.set_xticks(panel["ticks"])
    ax.set_xticklabels([str(value) for value in panel["ticks"]], fontsize=9.2, color=MUTED)
    ax.set_yticks(range(len(rows)))
    if show_labels:
        ax.set_yticklabels([short_label(row["model"]) for row in rows], fontsize=9.4, color=INK)
        for tick, row in zip(ax.get_yticklabels(), rows):
            if row["kind"] == "ours":
                tick.set_weight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.invert_yaxis()
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(0, color=RULE, linewidth=1)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = load_results()
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "svg.hashsalt": "jevany-external-zero-shot-v2",
        "text.color": INK,
        "hatch.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(21, 8.2), dpi=110, facecolor="white", sharey=True)
    fig.subplots_adjust(left=0.17, right=0.988, bottom=0.16, top=0.72, wspace=0.20)
    fig.text(0.035, 0.955, "External zero-shot decision accuracy", fontsize=24, weight="bold")
    fig.text(
        0.035,
        0.910,
        "Same 13-model order · complete stated sets · — = unsupported native input or no matching result",
        fontsize=12.2,
        color=MUTED,
    )
    legend = [
        Patch(facecolor=COLORS["ours"], label="JevAny"),
        Patch(facecolor=COLORS["open_kev"], label="Kev · open"),
        Patch(facecolor=COLORS["open_rerun"], label="Other open · locally rerun"),
        Patch(facecolor=COLORS["published_only"], edgecolor="#79558D", hatch="////", label="Published only"),
    ]
    fig.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.988, 0.972),
        ncol=4,
        frameon=False,
        fontsize=10.8,
        handlelength=1.2,
        columnspacing=1.5,
    )
    for index, (ax, panel) in enumerate(zip(axes, PANELS)):
        draw_panel(ax, data[panel["key"]]["ordered_models"], panel, show_labels=index == 0)
    fig.text(
        0.035,
        0.045,
        "All three panels use accuracy. Typed accuracy is agreement with teacher-derived labels; JevJudge full "
        "covers 3,220 multimodal records and text-only is its 724-record subset.",
        fontsize=10.2,
        color=MUTED,
    )
    description = " ".join(
        f"{panel['title']}: "
        + ", ".join(
            f"{row['model']} "
            + ("unavailable" if row.get(panel["metric"]) is None else f"{row[panel['metric']] * 100:.2f}%")
            for row in data[panel["key"]]["ordered_models"]
        )
        for panel in PANELS
    )
    metadata = {"Date": None, "Title": "External zero-shot decision accuracy", "Description": description}
    fig.savefig(SVG_OUT, metadata=metadata)
    SVG_OUT.write_text("\n".join(line.rstrip() for line in SVG_OUT.read_text().splitlines()) + "\n")
    fig.savefig(PNG_OUT, dpi=170, metadata={"Software": "JevAny plot_external_zero_shot.py"})
    plt.close(fig)
    print(f"Wrote {SVG_OUT.relative_to(ROOT)} and {PNG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
