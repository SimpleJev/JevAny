#!/usr/bin/env python3
"""Build and reproducibly merge the two-page external-evaluation appendix."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
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


ROOT = Path(__file__).resolve().parents[1]
INK = "#213248"
MUTED = "#64748B"
RULE = "#DCE4EC"
TEAL = "#278577"
PALE = "#F3F7FA"
PDF_TIMESTAMP = datetime(2026, 10, 1, tzinfo=timezone.utc)
PDF_DATE_LITERAL = "D:20261001000000Z"
APPENDIX_METADATA = {
    "Title": "JevAny Technical Report — External Evaluation Appendix",
    "Author": "SimpleJev",
    "Subject": "Typed Decisions and JevJudge-Public v0.3 evaluation",
    "Creator": "scripts/build_external_report_appendix.py",
    "Producer": "Matplotlib PDF backend",
    "CreationDate": PDF_TIMESTAMP,
    "ModDate": PDF_TIMESTAMP,
}
MERGED_METADATA = {
    "/Title": "JevAny: Toward General Decision Intelligence — with Agent-Harness and External-Evaluation Appendices",
    "/Author": "SimpleJev",
    "/Subject": "JevAny model, bounded decision harness, and external decision-suite evaluation",
    "/Creator": "scripts/build_external_report_appendix.py",
    "/Producer": "pypdf",
    "/CreationDate": PDF_DATE_LITERAL,
    "/ModDate": PDF_DATE_LITERAL,
}


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


def header(fig, title: str, subtitle: str, page: int) -> None:
    fig.text(0.065, 0.955, "SimpleJev  ·  TECHNICAL REPORT  ·  APPENDIX L", color=TEAL,
             fontsize=8.5, weight="bold")
    fig.text(0.065, 0.905, title, color=INK, fontsize=22, weight="bold")
    fig.text(0.065, 0.872, subtitle, color=MUTED, fontsize=9.5)
    fig.lines.append(plt.Line2D([0.065, 0.935], [0.845, 0.845], transform=fig.transFigure,
                                color=RULE, linewidth=1))
    fig.text(0.065, 0.035, "JevAny · Toward General Decision Intelligence", color=MUTED, fontsize=7.5)
    fig.text(0.935, 0.035, f"External evaluation · {page}/2", color=MUTED,
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
    ax = fig.add_axes([0.055, 0.42, 0.89, 0.36])
    ax.imshow(mpimg.imread(chart))
    ax.axis("off")
    full = {row["model"]: row for row in data["jevjudge_full"]["models"]}
    text = {row["model"]: row for row in data["jevjudge_text"]["models"]}
    typed = {row["model"]: row for row in data["typed_decisions"]["models"]}
    qwen = full["JevAny-Qwen3.8-27B"]
    jeff = full["Jeff-Qwen3.5-2B"]
    card(fig, 0.065, "Full-suite accuracy",
         f"Qwen3.8-27B: {qwen['accuracy'] * 100:.2f}%\n"
         f"Jeff-2B: {jeff['accuracy'] * 100:.2f}%\n"
         f"Gap: {(qwen['accuracy'] - jeff['accuracy']) * 100:.2f} points")
    card(fig, 0.365, "Typed accuracy",
         f"JevAny Qwen27: {typed['JevAny-Qwen3.8-27B']['accuracy'] * 100:.2f}%\n"
         f"Jev 1.13: {typed['Jev 1.13 (OpenRouter)']['accuracy'] * 100:.2f}%\n"
         "Typed result published · gap 0.10 pt")
    card(fig, 0.665, "Text-only accuracy",
         f"JevAny Qwen27: {text['JevAny-Qwen3.8-27B']['accuracy'] * 100:.2f}%\n"
         f"Jev 1.13: {text['Jev 1.13 (OpenRouter)']['accuracy'] * 100:.2f}%\n"
         f"Kev-27B: {text['Kev-27B']['accuracy'] * 100:.2f}%")
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
             "Jev 1.13, OpenDecider-small, and Jeff-Gemma4 are text-only; Bongard-mini has no video path; Kev does not "
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


def build_appendix(data_path: Path, chart_path: Path, output: Path) -> None:
    data = json.loads(data_path.read_text())
    with atomic_destination(output) as temporary:
        with PdfPages(temporary, metadata=APPENDIX_METADATA) as pdf:
            overview_page(pdf, data, chart_path)
            results_page(pdf, data)
        appendix = PdfReader(temporary)
        if len(appendix.pages) != 2:
            raise RuntimeError(f"expected a two-page appendix, got {len(appendix.pages)} pages")


def page_invariants(page) -> tuple:
    """Return page properties that must survive the merge unchanged."""
    contents = page.get_contents()
    content_bytes = contents.get_data() if contents is not None else b""
    return (
        tuple(float(value) for value in page.mediabox),
        tuple(float(value) for value in page.cropbox),
        page.rotation,
        content_bytes,
        page.extract_text() or "",
        len(page.get("/Annots", [])),
    )


def merge_report(base_report: Path, appendix_report: Path, merged_output: Path, base_pages: int) -> None:
    if base_pages < 1:
        raise ValueError("--base-pages must be positive")
    base = PdfReader(base_report)
    appendix = PdfReader(appendix_report)
    if len(base.pages) < base_pages:
        raise ValueError(
            f"base report has {len(base.pages)} pages, fewer than --base-pages={base_pages}"
        )
    if len(appendix.pages) != 2:
        raise ValueError(f"appendix must have exactly two pages, got {len(appendix.pages)}")

    writer = PdfWriter()
    writer.pdf_header = "%PDF-1.7"
    for index in range(base_pages):
        writer.add_page(base.pages[index])
    for page in appendix.pages:
        writer.add_page(page)
    writer.add_metadata(MERGED_METADATA)

    with atomic_destination(merged_output) as temporary:
        with temporary.open("wb") as stream:
            writer.write(stream)
        merged = PdfReader(temporary)
        expected_pages = base_pages + len(appendix.pages)
        if len(merged.pages) != expected_pages:
            raise RuntimeError(f"expected {expected_pages} merged pages, got {len(merged.pages)}")
        for index in range(base_pages):
            if page_invariants(base.pages[index]) != page_invariants(merged.pages[index]):
                raise RuntimeError(f"base page {index + 1} changed during merge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/external-zero-shot-v1.json")
    parser.add_argument("--chart", type=Path, default=ROOT / "docs/external-zero-shot.png")
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
    parser.add_argument("--base-pages", type=int, default=19)
    args = parser.parse_args()
    if (args.base_report is None) != (args.merged_output is None):
        parser.error("--base-report and --merged-output must be provided together")
    if args.base_pages < 1:
        parser.error("--base-pages must be positive")
    if args.base_report is not None:
        appendix_path = args.output.expanduser().absolute()
        base_path = args.base_report.expanduser().absolute()
        merged_path = args.merged_output.expanduser().absolute()
        if appendix_path in {base_path, merged_path}:
            parser.error("--output must differ from --base-report and --merged-output")

    build_appendix(args.data, args.chart, args.output)
    print(f"Wrote reproducible appendix: {args.output}")
    if args.base_report is not None:
        merge_report(args.base_report, args.output, args.merged_output, args.base_pages)
        print(
            f"Wrote reproducible {args.base_pages + 2}-page merged report: "
            f"{args.merged_output}"
        )


if __name__ == "__main__":
    main()
