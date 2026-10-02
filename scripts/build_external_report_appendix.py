#!/usr/bin/env python3
"""Build and reproducibly merge the external and choice-readout appendices."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from pypdf import PdfReader, PdfWriter
from pypdf._cmap import get_encoding
from pypdf.generic import (
    ArrayObject,
    ContentStream,
    IndirectObject,
    NumberObject,
    TextStringObject,
)


ROOT = Path(__file__).resolve().parents[1]
INK = "#213248"
MUTED = "#64748B"
RULE = "#DCE4EC"
TEAL = "#278577"
PALE = "#F3F7FA"
PDF_TIMESTAMP = datetime(2026, 10, 2, tzinfo=timezone.utc)
PDF_DATE_LITERAL = "D:20261002000000Z"
EXTERNAL_APPENDIX_PAGES = 2
CHOICE_APPENDIX_PAGES = 2
APPENDIX_PAGES = EXTERNAL_APPENDIX_PAGES + CHOICE_APPENDIX_PAGES
REPORT_BASE_PAGES = 19
MERGED_REPORT_PAGES = REPORT_BASE_PAGES + APPENDIX_PAGES
APPENDIX_METADATA = {
    "Title": "JevAny Technical Report — Evaluation Appendices",
    "Author": "SimpleJev",
    "Subject": "External evaluation and training-free choice-token readout",
    "Creator": "scripts/build_external_report_appendix.py",
    "Producer": "Matplotlib PDF backend",
    "CreationDate": PDF_TIMESTAMP,
    "ModDate": PDF_TIMESTAMP,
}
MERGED_METADATA = {
    "/Title": "JevAny: Toward General Decision Intelligence — merged technical report",
    "/Author": "SimpleJev",
    "/Subject": (
        "JevAny model, bounded decision harness, external evaluation, and "
        "training-free choice-token readout"
    ),
    "/Creator": "scripts/build_external_report_appendix.py",
    "/Producer": "pypdf",
    "/CreationDate": PDF_DATE_LITERAL,
    "/ModDate": PDF_DATE_LITERAL,
}

# The source report predates the final 27B release checkpoint.  Synchronize the
# handful of release-headline fields while merging so the one published PDF
# does not call two different checkpoints "current".  The original layout,
# resources, links, and every unrelated result remain untouched.
CURRENT_RELEASE_TEXT = {
    1: (("85.76", "86.04", 2), ("90.48", "90.04", 2)),
    4: (
        ("22,160", "44,319", 1),
        ("85.76", "86.04", 1),
        ("90.48", "90.04", 1),
        ("0.392", "0.388", 1),
        ("0.200", "0.195", 1),
        ("0.030", "0.026", 1),
    ),
    7: (("85.76", "86.04", 1),),
    8: (
        ("22,160", "44,319", 1),
        ("18.83", "39.43", 1),
        ("602.7", "1,261.7", 1),
        ("1,423", "2,082", 1),
    ),
}

CURRENT_COMPUTE_PROVENANCE = (
    (
        "onds;itsvalueusestherecordedrunstartandcheckpointtimestamp."
        "Direct-token,Qwen27B,and",
        "onds; its value uses the recorded run start and checkpoint timestamp. "
        "Direct-token and Muse use",
    ),
    (
        "Museusecheckpoint-nativeelapsedtime;Qwen4Bpointerusesterminaltrainertelemetrybecause",
        "checkpoint-native elapsed time; Qwen 4B pointer uses terminal trainer telemetry. "
        "Qwen 27B uses",
    ),
    (
        "thereleasedcheckpointisthecompletedstep13,850run."
        "Peakwithin-runparallelismwas40GPUs.",
        "estimated cumulative seconds-per-record timing. "
        "Peak within-run parallelism was 40 GPUs.",
    ),
)


@contextmanager
def atomic_destination(destination: Path):
    """Yield a same-directory temporary path, then atomically replace destination."""
    destination = destination.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        temporary.chmod(0o644)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def header(
    fig,
    title: str,
    subtitle: str,
    page: int,
    *,
    appendix: str = "L",
    section: str = "External evaluation",
    pages: int = EXTERNAL_APPENDIX_PAGES,
) -> None:
    fig.text(0.065, 0.955, f"SimpleJev  ·  TECHNICAL REPORT  ·  APPENDIX {appendix}", color=TEAL,
             fontsize=8.5, weight="bold")
    fig.text(0.065, 0.905, title, color=INK, fontsize=22, weight="bold")
    fig.text(0.065, 0.872, subtitle, color=MUTED, fontsize=9.5)
    fig.lines.append(plt.Line2D([0.065, 0.935], [0.845, 0.845], transform=fig.transFigure,
                                color=RULE, linewidth=1))
    fig.text(0.065, 0.035, "JevAny · Toward General Decision Intelligence", color=MUTED, fontsize=7.5)
    fig.text(0.935, 0.035, f"{section} · {page}/{pages}", color=MUTED,
             fontsize=7.5, ha="right")


def card(fig, x: float, title: str, body: str) -> None:
    patch = FancyBboxPatch((x, 0.17), 0.27, 0.16, transform=fig.transFigure,
                           boxstyle="round,pad=0.012,rounding_size=0.012",
                           facecolor=PALE, edgecolor=RULE, linewidth=0.8)
    fig.patches.append(patch)
    fig.text(x + 0.018, 0.294, title, color=INK, fontsize=10.5, weight="bold")
    fig.text(x + 0.018, 0.265, body, color=MUTED, fontsize=8.4, va="top", linespacing=1.35)


def overview_page(pdf: PdfPages, data: dict, chart: Path) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    header(fig, "External decision-suite evaluation",
           "Accuracy on Typed Decisions · JevJudge-Public v0.3 full multimodal suite · text-only subset", 1)
    chart_image = mpimg.imread(chart)
    # The source asset is a wide, three-panel README graphic. Rendering it as
    # one image makes its labels roughly four points on a portrait report page.
    # Crop the panels (including their titles and axes) and use a 1 + 2 layout
    # so every label is materially larger while the source pixels stay intact.
    height, width = chart_image.shape[:2]
    if width >= 3 and height >= 3:
        y_start, y_stop = int(height * 0.15), int(height * 0.91)
        panels = (
            (chart_image[y_start:y_stop, int(width * 0.02):int(width * 0.43)],
             [0.065, 0.500, 0.870, 0.300]),
            (chart_image[y_start:y_stop, int(width * 0.42):int(width * 0.71)],
             [0.065, 0.115, 0.420, 0.335]),
            (chart_image[y_start:y_stop, int(width * 0.70):int(width * 0.995)],
             [0.515, 0.115, 0.420, 0.335]),
        )
        for panel, bounds in panels:
            ax = fig.add_axes(bounds)
            ax.imshow(panel)
            ax.axis("off")
    else:
        # Keep tiny fixture images usable in report-generation tests.
        ax = fig.add_axes([0.065, 0.115, 0.870, 0.685])
        ax.imshow(chart_image)
        ax.axis("off")
    full = {row["model"]: row for row in data["jevjudge_full"]["models"]}
    text = {row["model"]: row for row in data["jevjudge_text"]["models"]}
    typed = {row["model"]: row for row in data["typed_decisions"]["models"]}
    qwen_full = full["JevAny-Qwen3.8-27B"]
    qwen_typed = typed["JevAny-Qwen3.8-27B"]
    qwen_text = text["JevAny-Qwen3.8-27B"]
    fig.text(
        0.065, 0.818,
        "Same 13-model order · — unsupported/no result · teal JevAny · gold Jev API · purple Kev · gray other open · hatch published",
        color=MUTED, fontsize=7.8,
    )
    fig.text(
        0.065, 0.795,
        f"Qwen3.8-27B topline  ·  Typed {qwen_typed['accuracy'] * 100:.2f}%  ·  "
        f"JevJudge full {qwen_full['accuracy'] * 100:.2f}%  ·  text-only {qwen_text['accuracy'] * 100:.2f}%",
        color=TEAL, fontsize=8.2, weight="bold",
    )
    pdf.savefig(fig)
    plt.close(fig)


def results_page(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    header(fig, "JevJudge full-multimodal results",
           "Accuracy headline · official chance-corrected skill_role · returned-probability calibration", 2)
    ax = fig.add_axes([0.065, 0.50, 0.87, 0.31])
    ax.axis("off")
    scored = [row for row in data["jevjudge_full"]["models"] if row.get("skill_role") is not None]
    columns = (0.00, 0.42, 0.72, 0.86, 0.97)
    headers = ("Model", "Accuracy", "Official skill_role (95% CI)", "NLL", "ECE")
    for x, label in zip(columns, headers):
        ax.text(x, 1.02, label, transform=ax.transAxes, fontsize=8.5, color=MUTED,
                weight="bold", ha="right" if x else "left")
    ax.plot([0, 1], [0.975, 0.975], transform=ax.transAxes, color=RULE, linewidth=1)
    for index, row in enumerate(scored):
        y = 0.89 - index * 0.125
        low, high = row["skill_role_ci_95"]
        values = (
            row["model"].replace("JevAny-", ""),
            f"{row['accuracy'] * 100:.2f}%",
            f"{row['skill_role'] * 100:.2f}% [{low * 100:.2f}, {high * 100:.2f}]",
            f"{row['nll']:.3f}",
            f"{row['ece']:.3f}",
        )
        color = TEAL if row["kind"] == "ours" else INK
        weight = "bold" if row["model"] == "JevAny-Qwen3.8-27B" else "normal"
        for x, value in zip(columns, values):
            ax.text(x, y, value, transform=ax.transAxes, fontsize=8.5, color=color,
                    weight=weight, ha="right" if x else "left", va="center")
        ax.plot([0, 1], [y - 0.06, y - 0.06], transform=ax.transAxes,
                color="#EEF2F6", linewidth=0.7)

    fig.text(0.065, 0.455, "Why skill_role differs from accuracy", color=INK, fontsize=12, weight="bold")
    fig.text(0.065, 0.417,
             "Plain accuracy counts correct items / 3,220. skill_role removes each family's chance or majority floor, "
             "averages families inside each role, then equally averages five roles. Zero is approximately chance; "
             "scores can be negative. Uniform predictions obtain 40.0% plain accuracy but −0.77% skill_role.",
             color=MUTED, fontsize=8.5, wrap=True, linespacing=1.45, va="top")

    fig.text(0.065, 0.335, "Compatibility and text-only results", color=INK, fontsize=12, weight="bold")
    fig.text(0.065, 0.300,
             "Jev 1.13's OpenRouter endpoint, OpenDecider-small, and Jeff-Gemma4 are text-only; Bongard-mini has no video path; Kev does not "
             "consume image or video media. Their full result is —, never a media-stripped score. On the 724-record "
             "text-only subset, Qwen3.8-27B scores 66.44%, Jev 1.13 65.06%, and Kev-27B 64.23%.",
             color=MUTED, fontsize=8.5, wrap=True, linespacing=1.45, va="top")

    fig.text(0.065, 0.215, "Protocol and provenance", color=INK, fontsize=12, weight="bold")
    protocol = (
        "Dataset HuanxinSheng/JevJudge-Public@4d576ded · 65,536-token ceiling · native checkpoint media processors · "
        "1,000 source-stratified group_id bootstrap replicates, seed 20261001 · NLL = −mean log returned p(gold). "
        "Aggregate artifacts and exact hashes: results/external-zero-shot-v1.json and docs/EXTERNAL_EVALUATION.md. "
        "The superseded Kev-27B 38.54% text row used malformed 7–63-token inputs and is excluded."
    )
    fig.text(0.065, 0.177, protocol, color=MUTED, fontsize=8.5, wrap=True, linespacing=1.45, va="top")
    pdf.savefig(fig)
    plt.close(fig)


CHOICE_RUN_SPECS = {
    "base4": ("Qwen3.5-4B", "frozen_base", None),
    "pointer4": ("Qwen3.5-4B", "jevany_sft", "pointer"),
    "direct4": ("Qwen3.5-4B", "jevany_sft", "direct-token"),
    "base27": ("Qwen3.8-27B", "frozen_base", None),
    "pointer27": ("Qwen3.8-27B", "jevany_sft", "pointer"),
}
CHOICE_DATASET_SPECS = (
    ("jevbench_public", "JevBench\npublic", 231, 231, 231),
    ("transfer_test", "Transfer-v9\ntest", 1_264, 1_264, 1_046),
    ("typed_test", "Typed\ntest", 400, 2_000, 2_000),
    ("jevjudge_text", "JevJudge\ntext", 724, 724, 724),
)


def _read_json_object(path: Path, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing {name}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {name}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}")
    return value


def _choice_metric(row: dict, group: str, config: str, n: int, context: str) -> dict:
    try:
        metric = row[group][config]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{context}: missing {group}/{config}") from error
    if not isinstance(metric, dict) or metric.get("n") != n:
        raise ValueError(f"{context}/{group}/{config}: expected n={n}")
    correct = metric.get("correct")
    accuracy = metric.get("accuracy")
    if isinstance(correct, bool) or not isinstance(correct, int) or not 0 <= correct <= n:
        raise ValueError(f"{context}/{group}/{config}: invalid correct count")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(accuracy)
        or not math.isclose(float(accuracy), correct / n, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError(f"{context}/{group}/{config}: accuracy does not equal correct / n")
    return metric


def load_choice_artifact(path: Path) -> dict:
    """Load the validated v2 matrix consumed by Appendix M."""

    artifact = _read_json_object(path, "choice-readout artifact")
    if artifact.get("schema_version") != 2 or artifact.get("artifact") != "choice-readout-v2":
        raise ValueError("choice-readout artifact must use schema_version=2")
    method = artifact.get("method")
    if not isinstance(method, dict) or method.get("option_ids") != (
        "A-Z followed by a-z; at most 52 options."
    ):
        raise ValueError("choice-readout artifact does not declare the v2 52-ID method")

    source_rows = artifact.get("source_runs")
    if not isinstance(source_rows, list):
        raise ValueError("choice-readout artifact has no source_runs list")
    source_by_key = {
        row.get("key"): row for row in source_rows if isinstance(row, dict)
    }
    if set(source_by_key) != set(CHOICE_RUN_SPECS) or len(source_rows) != len(source_by_key):
        raise ValueError("choice-readout artifact must contain the five required source runs")
    for key, (family, weights, native_readout) in CHOICE_RUN_SPECS.items():
        row = source_by_key[key]
        if (row.get("family"), row.get("weights"), row.get("native_readout")) != (
            family, weights, native_readout
        ):
            raise ValueError(f"choice-readout source run {key!r} has inconsistent identity")
        selected = row.get("selected")
        if not isinstance(selected, dict):
            raise ValueError(f"choice-readout source run {key!r} has no selected settings")
        for field in ("native_weight", "choice_temperature", "blend_temperature"):
            value = selected.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"choice-readout source run {key!r} has invalid {field}")
        if not 0 <= selected["native_weight"] <= 1:
            raise ValueError(f"choice-readout source run {key!r} has invalid native_weight")
        if selected["choice_temperature"] <= 0 or selected["blend_temperature"] <= 0:
            raise ValueError(f"choice-readout source run {key!r} has invalid temperature")
        if weights == "frozen_base" and selected["native_weight"] != 0:
            raise ValueError(f"frozen-base run {key!r} selected a native weight")
        ensemble_protocol = row.get("ensemble_protocol")
        if not isinstance(ensemble_protocol, dict) or (
            ensemble_protocol.get("selection_rows") != 1_046
        ):
            raise ValueError(f"choice-readout source run {key!r} has invalid selection rows")

    datasets = artifact.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError("choice-readout artifact has no datasets object")
    for dataset, _label, records, questions, n in CHOICE_DATASET_SPECS:
        panel = datasets.get(dataset)
        if not isinstance(panel, dict) or (
            panel.get("records"), panel.get("questions"), panel.get("headline_n")
        ) != (records, questions, n):
            raise ValueError(
                f"choice-readout dataset {dataset!r} must declare "
                f"records/questions/headline_n={records}/{questions}/{n}"
            )
        rows = panel.get("runs")
        if not isinstance(rows, list):
            raise ValueError(f"choice-readout dataset {dataset!r} has no runs list")
        by_key = {row.get("run"): row for row in rows if isinstance(row, dict)}
        if set(by_key) != set(CHOICE_RUN_SPECS) or len(rows) != len(by_key):
            raise ValueError(f"choice-readout dataset {dataset!r} must contain five runs")
        for key, (family, weights, native_readout) in CHOICE_RUN_SPECS.items():
            row = by_key[key]
            if (row.get("family"), row.get("weights"), row.get("native_readout")) != (
                family, weights, native_readout
            ):
                raise ValueError(f"{dataset}/{key}: inconsistent run identity")
            _choice_metric(row, "zero_shot", "choice_t1", n, f"{dataset}/{key}")
            if native_readout is not None:
                _choice_metric(row, "zero_shot", "native_shipped", n, f"{dataset}/{key}")
                _choice_metric(
                    row,
                    "transfer_dev_tuned",
                    "transfer_dev_tuned_blend",
                    n,
                    f"{dataset}/{key}",
                )

    full = datasets.get("jevjudge_full")
    if not isinstance(full, dict) or (
        full.get("records"), full.get("status"), full.get("accuracy")
    ) != (3_220, "unsupported", None):
        raise ValueError("choice-token JevJudge full must be explicitly unsupported")
    return artifact


def _rounded_box(fig, x, y, width, height, *, facecolor=PALE, edgecolor=RULE):
    patch = FancyBboxPatch(
        (x, y), width, height, transform=fig.transFigure,
        boxstyle="round,pad=0.010,rounding_size=0.010",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8,
    )
    fig.patches.append(patch)


def choice_method_page(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    header(
        fig,
        "Training-free choice-token readout",
        "A constrained readout, probability calibration, and stacking are different operations",
        1,
        appendix="M",
        section="Choice-token readout",
        pages=CHOICE_APPENDIX_PAGES,
    )
    fig.text(
        0.065, 0.815,
        "The frozen language model scores one exact option ID at the answer position. "
        "No explanation is generated and no additional training is required for this readout.",
        color=INK, fontsize=10.0, wrap=True, linespacing=1.45,
    )

    pipeline = (
        ("1", "Ordered options", "Preserve candidate order"),
        ("2", "52 exact IDs", "A–Z followed by a–z"),
        ("3", "One-token projection", "Read next-token mass"),
        ("4", "Decision distribution", "Renormalize over IDs"),
    )
    for index, (number, title, body) in enumerate(pipeline):
        x = 0.065 + index * 0.222
        _rounded_box(fig, x, 0.685, 0.185, 0.095, facecolor="#F5F9F8")
        fig.text(x + 0.014, 0.750, number, color=TEAL, fontsize=9, weight="bold")
        fig.text(x + 0.042, 0.750, title, color=INK, fontsize=9.2, weight="bold")
        fig.text(x + 0.014, 0.712, body, color=MUTED, fontsize=7.8)
        if index < len(pipeline) - 1:
            fig.text(x + 0.198, 0.730, "→", color=TEAL, fontsize=14, weight="bold")

    semantic_cards = (
        (
            "Choice-token · T=1",
            "Training-free constrained\nprojection. This is a readout,\n"
            "not temperature calibration;\nit can disagree with native.",
        ),
        (
            "Temperature calibration",
            "Fit scalar T on separate\ndevelopment rows. T changes\n"
            "probabilities; it cannot change\nargmax or accuracy.",
        ),
        (
            "Native + choice stack",
            "Log-linear pooling can change\nrankings. Select native weight\n"
            "and additional temperature on\nTransfer-v9 development only.",
        ),
    )
    for index, (title, body) in enumerate(semantic_cards):
        x = 0.065 + index * 0.299
        _rounded_box(fig, x, 0.475, 0.270, 0.150)
        fig.text(x + 0.016, 0.590, title, color=INK, fontsize=9.6, weight="bold")
        fig.text(
            x + 0.016, 0.558, body, color=MUTED, fontsize=7.8,
            va="top", linespacing=1.35,
        )

    sources = {row["key"]: row for row in data["source_runs"]}
    table = (
        ("Frozen base", "—", "Choice-token", f"{sources['base4']['family']}, {sources['base27']['family']}"),
        ("JevAny SFT", "Pointer", "Choice-token", f"{sources['pointer4']['family']}, {sources['pointer27']['family']}"),
        ("JevAny SFT", "Direct-Token", "Choice-token", sources["direct4"]["family"]),
    )
    fig.text(0.065, 0.425, "Five-run comparison", color=INK, fontsize=12, weight="bold")
    columns = (0.065, 0.245, 0.420, 0.610)
    for x, label in zip(columns, ("Weights", "Native reference", "Training-free path", "Backbone family")):
        fig.text(x, 0.390, label, color=MUTED, fontsize=8.0, weight="bold")
    fig.lines.append(plt.Line2D([0.065, 0.935], [0.377, 0.377], transform=fig.transFigure,
                                color=RULE, linewidth=1))
    for index, values in enumerate(table):
        y = 0.346 - index * 0.052
        for x, value in zip(columns, values):
            fig.text(x, y, value, color=INK, fontsize=8.4)
        fig.lines.append(plt.Line2D([0.065, 0.935], [y - 0.019, y - 0.019],
                                    transform=fig.transFigure, color="#EEF2F6", linewidth=0.7))

    _rounded_box(
        fig, 0.065, 0.123, 0.870, 0.075,
        facecolor="#FFF8E8", edgecolor="#E7C46A",
    )
    fig.text(0.083, 0.176, "Release sync and protocol boundary", color="#8A5A00", fontsize=9.5, weight="bold")
    fig.text(
        0.083, 0.151,
        "The merged report and README identify the current 27B release checkpoint: step 44,319.",
        color=INK, fontsize=8.0,
    )
    fig.text(
        0.083, 0.132,
        "Appendix M native rows are matched prompt-v2/runtime reruns; Appendix L's earlier external-study row can differ by one decision.",
        color=INK, fontsize=8.0,
    )
    _rounded_box(fig, 0.065, 0.061, 0.870, 0.047, facecolor="#F7FAFC")
    fig.text(0.083, 0.091, "Scope", color=INK, fontsize=8.7, weight="bold")
    fig.text(
        0.137, 0.091,
        "Text only · up to 52 IDs · JevJudge text 724 supported · full multimodal unsupported · agent tasks unchanged",
        color=MUTED, fontsize=7.6,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _dataset_run(data: dict, dataset: str, run: str) -> dict:
    matches = [row for row in data["datasets"][dataset]["runs"] if row["run"] == run]
    if len(matches) != 1:
        raise ValueError(f"{dataset}: expected one {run} row")
    return matches[0]


def _choice_matrix_rows(data: dict) -> list[dict]:
    sources = {row["key"]: row for row in data["source_runs"]}
    definitions = (
        ("Frozen Qwen3.5-4B", "Choice T=1", "base4", "zero_shot", "choice_t1", "—"),
        ("JevAny 4B Pointer", "Native", "pointer4", "zero_shot", "native_shipped", "shipped"),
        ("", "Choice T=1", "pointer4", "zero_shot", "choice_t1", "—"),
        ("", "Tuned stack", "pointer4", "transfer_dev_tuned", "transfer_dev_tuned_blend", None),
        ("JevAny 4B Direct-Token", "Native", "direct4", "zero_shot", "native_shipped", "shipped"),
        ("", "Choice T=1", "direct4", "zero_shot", "choice_t1", "—"),
        ("", "Tuned stack", "direct4", "transfer_dev_tuned", "transfer_dev_tuned_blend", None),
        ("Frozen Qwen3.8-27B", "Choice T=1", "base27", "zero_shot", "choice_t1", "—"),
        ("JevAny 27B Pointer", "Native", "pointer27", "zero_shot", "native_shipped", "shipped"),
        ("", "Choice T=1", "pointer27", "zero_shot", "choice_t1", "—"),
        ("", "Tuned stack", "pointer27", "transfer_dev_tuned", "transfer_dev_tuned_blend", None),
    )
    rows = []
    for model, readout, run, group, config, weight in definitions:
        if weight is None:
            weight = f"{sources[run]['selected']['native_weight']:.2f}"
        metrics = []
        for dataset, _label, _records, _questions, n in CHOICE_DATASET_SPECS:
            metric = _choice_metric(
                _dataset_run(data, dataset, run), group, config, n, f"{dataset}/{run}"
            )
            metrics.append(metric)
        rows.append({
            "model": model,
            "readout": readout,
            "run": run,
            "weight": weight,
            "metrics": metrics,
            "tuned": readout == "Tuned stack",
            "native": readout == "Native",
        })
    return rows


def choice_results_page(pdf: PdfPages, data: dict) -> None:
    fig = plt.figure(figsize=(8.5, 11), facecolor="white")
    header(
        fig,
        "Choice-token accuracy matrix",
        "Matched runs · current 27B release step 44,319 · Transfer-dev settings frozen before evaluation",
        2,
        appendix="M",
        section="Choice-token readout",
        pages=CHOICE_APPENDIX_PAGES,
    )
    rows = _choice_matrix_rows(data)
    ax = fig.add_axes([0.055, 0.285, 0.89, 0.530])
    ax.axis("off")
    x_model, x_readout, x_weight = 0.00, 0.315, 0.495
    x_metrics = (0.635, 0.755, 0.875, 0.995)
    ax.text(x_model, 1.035, "Model", transform=ax.transAxes, fontsize=8.0,
            color=MUTED, weight="bold")
    ax.text(x_readout, 1.035, "Readout", transform=ax.transAxes, fontsize=8.0,
            color=MUTED, weight="bold")
    ax.text(x_weight, 1.035, "w_native", transform=ax.transAxes, fontsize=8.0,
            color=MUTED, weight="bold", ha="right")
    for x, (_dataset, label, _records, _questions, n) in zip(
        x_metrics, CHOICE_DATASET_SPECS, strict=True
    ):
        ax.text(x, 1.035, f"{label}\n(n={n:,})", transform=ax.transAxes,
                fontsize=7.7, color=MUTED, weight="bold", ha="right", va="bottom")
    ax.plot([0, 1], [0.995, 0.995], transform=ax.transAxes, color=RULE, linewidth=1)

    group_spans = ((0, 0), (1, 3), (4, 6), (7, 7), (8, 10))
    step, first_y = 0.082, 0.930
    for group_index, (start, end) in enumerate(group_spans):
        top = first_y - start * step + 0.034
        bottom = first_y - end * step - 0.034
        if group_index % 2:
            patch = FancyBboxPatch(
                (-0.012, bottom), 1.022, top - bottom,
                transform=ax.transAxes, boxstyle="round,pad=0.004,rounding_size=0.005",
                facecolor="#F7FAFC", edgecolor="none", zorder=0,
            )
            ax.add_patch(patch)

    maxima = [max(row["metrics"][index]["accuracy"] for row in rows) for index in range(4)]
    for row_index, row in enumerate(rows):
        y = first_y - row_index * step
        color = TEAL if row["tuned"] else INK
        ax.text(x_model, y, row["model"], transform=ax.transAxes, fontsize=8.0,
                color=INK, va="center", weight="bold" if row["model"] else "normal")
        ax.text(x_readout, y, row["readout"], transform=ax.transAxes, fontsize=8.0,
                color=color, va="center", weight="bold" if row["tuned"] else "normal")
        ax.text(x_weight, y, row["weight"], transform=ax.transAxes, fontsize=8.0,
                color=MUTED, va="center", ha="right")
        for metric_index, (x, metric) in enumerate(zip(x_metrics, row["metrics"], strict=True)):
            best = math.isclose(metric["accuracy"], maxima[metric_index], rel_tol=0, abs_tol=1e-12)
            ax.text(
                x, y, f"{metric['accuracy'] * 100:.2f}", transform=ax.transAxes,
                fontsize=8.1, color=TEAL if best else color, va="center", ha="right",
                weight="bold" if best or row["tuned"] else "normal",
            )
        ax.plot([0, 1], [y - 0.040, y - 0.040], transform=ax.transAxes,
                color="#EEF2F6", linewidth=0.6, zorder=1)

    sources = {row["key"]: row for row in data["source_runs"]}
    tuned = []
    for key, label in (("pointer4", "4B Pointer"), ("direct4", "4B Direct-Token"),
                       ("pointer27", "27B Pointer")):
        selected = sources[key]["selected"]
        tuned.append(
            f"{label}: w={selected['native_weight']:.2f}, T={selected['blend_temperature']:.2f}"
        )
    fig.text(0.065, 0.235, "Transfer-dev selection", color=INK, fontsize=10.5, weight="bold")
    fig.text(0.065, 0.205, " · ".join(tuned), color=TEAL, fontsize=8.4, weight="bold")
    fig.text(
        0.065, 0.169,
        "Weights and additional temperatures use only 1,046 clean/knowable Transfer-v9 development rows.\n"
        "Scalar T changes probabilities, not the accuracy values above.",
        color=MUTED, fontsize=8.2, linespacing=1.35,
    )
    fig.text(
        0.065, 0.125,
        "Native = matched native rerun at the checkpoint's shipped inference temperature.\n"
        "JevBench is a public development diagnostic; Transfer, Typed, and JevJudge text are held-out panels.",
        color=MUTED, fontsize=8.2, linespacing=1.35,
    )
    fig.text(
        0.065, 0.082,
        "Choice-token is text-only: JevJudge text 724 is reported; full JevJudge 3,220 is unsupported.\n"
        "Source and complete calibration metrics: results/choice-readout-v2.json.",
        color=MUTED, fontsize=8.2, linespacing=1.35,
    )
    pdf.savefig(fig)
    plt.close(fig)


def build_appendix(
    data_path: Path,
    chart_path: Path,
    choice_data_path: Path,
    output: Path,
) -> None:
    data = _read_json_object(data_path, "external-evaluation artifact")
    choice_data = load_choice_artifact(choice_data_path)
    with atomic_destination(output) as temporary:
        with PdfPages(temporary, metadata=APPENDIX_METADATA) as pdf:
            overview_page(pdf, data, chart_path)
            results_page(pdf, data)
            choice_method_page(pdf, choice_data)
            choice_results_page(pdf, choice_data)
        appendix = PdfReader(temporary)
        if len(appendix.pages) != APPENDIX_PAGES:
            raise RuntimeError(
                f"expected a {APPENDIX_PAGES}-page appendix, got {len(appendix.pages)} pages"
            )


def _canonical_pdf_object(value, active: set[int] | None = None):
    """Resolve PDF references into a stable, object-number-independent value."""

    if isinstance(value, IndirectObject):
        return _canonical_pdf_object(value.get_object(), active)
    if active is None:
        active = set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in active:
            return ("cycle",)
        active.add(marker)
        try:
            items = tuple(sorted(
                (
                    str(key),
                    _canonical_pdf_object(item, active),
                )
                for key, item in value.items()
                if str(key) != "/Length"
            ))
            stream = None
            if hasattr(value, "get_data"):
                stream = hashlib.sha256(value.get_data()).hexdigest()
            return ("dict", items, stream)
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_pdf_object(item, active) for item in value)
    if isinstance(value, bytes):
        return ("bytes", hashlib.sha256(value).hexdigest())
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def page_invariants(page) -> tuple:
    """Return page properties that must survive the merge unchanged."""
    contents = page.get_contents()
    content_bytes = contents.get_data() if contents is not None else b""
    annotations = []
    for reference in page.get("/Annots", []):
        annotation = reference.get_object()
        # /P is a back-reference to the owning page and would make the
        # otherwise stable annotation target depend on PDF object numbers.
        annotations.append(_canonical_pdf_object({
            key: value for key, value in annotation.items() if str(key) != "/P"
        }))
    return (
        tuple(float(value) for value in page.mediabox),
        tuple(float(value) for value in page.cropbox),
        page.rotation,
        content_bytes,
        page.extract_text() or "",
        _canonical_pdf_object(page.get("/Resources")),
        tuple(annotations),
    )


def _font_text_maps(page) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    maps = {}
    fonts = page["/Resources"]["/Font"]
    for name, reference in fonts.items():
        _encoding, character_map = get_encoding(reference.get_object())
        reverse = {
            unicode_text: glyph
            for glyph, unicode_text in character_map.items()
            if isinstance(glyph, str)
            and isinstance(unicode_text, str)
            and len(unicode_text) == 1
        }
        maps[str(name)] = (character_map, reverse)
    return maps


def _decode_text_object(value, character_map: dict[str, str]) -> str:
    return "".join(character_map.get(glyph, glyph) for glyph in str(value))


def _encode_text_object(
    text: str,
    reverse_map: dict[str, str],
    prototype,
) -> TextStringObject:
    try:
        glyphs = "".join(reverse_map[character] for character in text)
    except KeyError as error:
        raise RuntimeError(f"release-sync font cannot encode {error.args[0]!r}") from error
    original_length = len(prototype.original_bytes)
    glyph_count = len(str(prototype))
    if glyph_count == 0 or original_length % glyph_count:
        raise RuntimeError("release-sync text object has an unsupported encoding width")
    byte_width = original_length // glyph_count
    if byte_width not in {1, 2}:
        raise RuntimeError(f"release-sync text uses unsupported {byte_width}-byte glyphs")
    raw = b"".join(ord(glyph).to_bytes(byte_width, "big") for glyph in glyphs)
    result = TextStringObject(glyphs)
    result._original_bytes = raw
    return result


def _word_array(text: str, reverse_map: dict[str, str], prototype) -> ArrayObject:
    words = text.split()
    result = ArrayObject()
    for index, word in enumerate(words):
        if index:
            result.append(NumberObject(-300))
        result.append(_encode_text_object(word, reverse_map, prototype))
    return result


def synchronize_current_release(page, page_number: int) -> bool:
    """Replace the superseded 27B headline fields in the retained report core."""

    replacements = CURRENT_RELEASE_TEXT.get(page_number)
    if replacements is None:
        return False
    extracted = page.extract_text() or ""
    legacy_counts = {old: extracted.count(old) for old, _new, _count in replacements}
    if not any(legacy_counts.values()):
        return False
    for old, _new, expected_count in replacements:
        if legacy_counts[old] != expected_count:
            raise RuntimeError(
                f"base report page {page_number}: expected {expected_count} occurrences "
                f"of legacy value {old!r}, found {legacy_counts[old]}"
            )

    text_maps = _font_text_maps(page)
    content = ContentStream(page.get_contents(), page.indirect_reference.pdf)
    active_font = None
    replaced_counts = {old: 0 for old, _new, _count in replacements}
    provenance_replaced = 0
    for operands, operator in content.operations:
        if operator == b"Tf":
            active_font = str(operands[0])
            continue
        if operator not in {b"Tj", b"TJ"} or active_font is None:
            continue
        character_map, reverse_map = text_maps[active_font]
        values = operands[0] if operator == b"TJ" else ArrayObject([operands[0]])
        decoded_line = "".join(
            _decode_text_object(value, character_map)
            for value in values
            if hasattr(value, "original_bytes")
        )

        if page_number == 8:
            provenance = next(
                (replacement for legacy, replacement in CURRENT_COMPUTE_PROVENANCE
                 if decoded_line == legacy),
                None,
            )
            if provenance is not None:
                prototype = next(
                    value for value in values if hasattr(value, "original_bytes")
                )
                operands[0] = _word_array(provenance, reverse_map, prototype)
                provenance_replaced += 1
                continue

        for index, value in enumerate(values):
            if not hasattr(value, "original_bytes"):
                continue
            decoded = _decode_text_object(value, character_map)
            updated = decoded
            for old, new, _expected_count in replacements:
                count = updated.count(old)
                if count:
                    updated = updated.replace(old, new)
                    replaced_counts[old] += count
            if updated != decoded:
                values[index] = _encode_text_object(updated, reverse_map, value)
        if operator == b"Tj":
            operands[0] = values[0]

    for old, _new, expected_count in replacements:
        if replaced_counts[old] != expected_count:
            raise RuntimeError(
                f"base report page {page_number}: replaced {replaced_counts[old]} of "
                f"{expected_count} expected {old!r} values"
            )
    if page_number == 8 and provenance_replaced != len(CURRENT_COMPUTE_PROVENANCE):
        raise RuntimeError(
            "base report page 8: could not synchronize the 27B compute provenance"
        )
    # In-place list edits do not invalidate ContentStream's raw-byte cache.
    content.operations = content.operations
    synchronized_parts = []
    active_font = None
    for operands, operator in content.operations:
        if operator == b"Tf":
            active_font = str(operands[0])
            continue
        if operator not in {b"Tj", b"TJ"} or active_font is None:
            continue
        character_map, _reverse_map = text_maps[active_font]
        values = operands[0] if operator == b"TJ" else [operands[0]]
        synchronized_parts.extend(
            _decode_text_object(value, character_map)
            for value in values
            if hasattr(value, "original_bytes")
        )
    synchronized = "".join(synchronized_parts)
    for old, new, expected_count in replacements:
        if old in synchronized or synchronized.count(new) < expected_count:
            raise RuntimeError(
                f"base report page {page_number}: failed to synchronize {old!r} to {new!r}; "
                f"old={synchronized.count(old)}, new={synchronized.count(new)}"
            )
    page.replace_contents(content)
    return True


def synchronize_report_boundary(page, page_number: int) -> bool:
    """Update the old appendix claim that the merged core is byte-unchanged."""

    if page_number != 10:
        return False
    extracted = page.extract_text() or ""
    if "The original PDF is pre-" not in extracted:
        return False
    replacements = {
        "original ": "merged ",
        "pre-": "release-",
        "serv": "synch",
        "ed unchanged.": "ronized.",
    }
    counts = {old: 0 for old in replacements}
    content = ContentStream(page.get_contents(), page.indirect_reference.pdf)
    for operands, operator in content.operations:
        if operator not in {b"Tj", b"TJ"}:
            continue
        values = operands[0] if operator == b"TJ" else ArrayObject([operands[0]])
        for index, value in enumerate(values):
            if not isinstance(value, TextStringObject):
                continue
            updated = str(value)
            for old, new in replacements.items():
                count = updated.count(old)
                if count:
                    updated = updated.replace(old, new)
                    counts[old] += count
            if updated != str(value):
                values[index] = TextStringObject(updated)
        if operator == b"Tj":
            operands[0] = values[0]
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(
            f"base report page 10: could not synchronize appendix boundary: {counts}"
        )
    content.operations = content.operations
    page.replace_contents(content)
    return True


def merge_report(base_report: Path, appendix_report: Path, merged_output: Path, base_pages: int) -> None:
    if base_pages != REPORT_BASE_PAGES:
        raise ValueError(f"--base-pages must be exactly {REPORT_BASE_PAGES}")
    base = PdfReader(base_report)
    appendix = PdfReader(appendix_report)
    if len(base.pages) < base_pages:
        raise ValueError(
            f"base report has {len(base.pages)} pages, fewer than --base-pages={base_pages}"
        )
    if len(appendix.pages) != APPENDIX_PAGES:
        raise ValueError(
            f"appendix must have exactly {APPENDIX_PAGES} pages, got {len(appendix.pages)}"
        )

    writer = PdfWriter()
    writer.pdf_header = "%PDF-1.7"
    synchronized_pages = set()
    expected_base_invariants = {}
    for index in range(base_pages):
        writer.add_page(base.pages[index])
        changed = synchronize_current_release(writer.pages[-1], index + 1)
        changed = synchronize_report_boundary(writer.pages[-1], index + 1) or changed
        if changed:
            synchronized_pages.add(index)
            expected_base_invariants[index] = page_invariants(writer.pages[-1])
    for page in appendix.pages:
        writer.add_page(page)
    writer.add_metadata(MERGED_METADATA)

    with atomic_destination(merged_output) as temporary:
        with temporary.open("wb") as stream:
            writer.write(stream)
        merged = PdfReader(temporary)
        expected_pages = MERGED_REPORT_PAGES
        if len(merged.pages) != expected_pages:
            raise RuntimeError(f"expected {expected_pages} merged pages, got {len(merged.pages)}")
        for index in range(base_pages):
            expected = (
                expected_base_invariants[index]
                if index in synchronized_pages
                else page_invariants(base.pages[index])
            )
            actual = page_invariants(merged.pages[index])
            if index in synchronized_pages:
                # A writer-attached page temporarily extracts its glyph IDs,
                # while the serialized page resolves them through ToUnicode.
                # Compare every invariant except that transient text view.
                expected = expected[:4] + expected[5:]
                actual = actual[:4] + actual[5:]
            if expected != actual:
                raise RuntimeError(f"base page {index + 1} changed during merge")
        for page_number, replacements in CURRENT_RELEASE_TEXT.items():
            page_text = merged.pages[page_number - 1].extract_text() or ""
            is_release_page = (page_number - 1) in synchronized_pages or any(
                page_text.count(new) >= expected_count
                for _old, new, expected_count in replacements
            )
            if not is_release_page:
                continue
            for old, new, expected_count in replacements:
                if old in page_text or page_text.count(new) < expected_count:
                    raise RuntimeError(
                        f"serialized base page {page_number} does not contain the current "
                        f"release value {new!r}"
                    )
        boundary_text = merged.pages[9].extract_text() or ""
        boundary_text = boundary_text.replace("-\n", "-")
        if "The original PDF is pre-" in boundary_text or (
            "The merged PDF is release-synchronized." not in boundary_text
            and 9 in synchronized_pages
        ):
            raise RuntimeError("serialized base page 10 has a stale appendix boundary")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/external-zero-shot-v1.json")
    parser.add_argument("--chart", type=Path, default=ROOT / "docs/external-zero-shot.png")
    parser.add_argument(
        "--choice-data", type=Path, default=ROOT / "results/choice-readout-v2.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-report",
        type=Path,
        help="report whose first --base-pages pages are retained before appending",
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        help="merged report; may be the same path as --base-report",
    )
    parser.add_argument("--base-pages", type=int, default=REPORT_BASE_PAGES)
    args = parser.parse_args(argv)
    if (args.base_report is None) != (args.merged_output is None):
        parser.error("--base-report and --merged-output must be provided together")
    if args.base_pages != REPORT_BASE_PAGES:
        parser.error(f"--base-pages must be exactly {REPORT_BASE_PAGES}")
    if args.base_report is not None:
        appendix_path = args.output.expanduser().absolute()
        base_path = args.base_report.expanduser().absolute()
        merged_path = args.merged_output.expanduser().absolute()
        if appendix_path in {base_path, merged_path}:
            parser.error("--output must differ from --base-report and --merged-output")

    build_appendix(args.data, args.chart, args.choice_data, args.output)
    print(f"Wrote reproducible appendix: {args.output}")
    if args.base_report is not None:
        merge_report(args.base_report, args.output, args.merged_output, args.base_pages)
        print(
            f"Wrote reproducible {MERGED_REPORT_PAGES}-page merged report: "
            f"{args.merged_output}"
        )


if __name__ == "__main__":
    main()
