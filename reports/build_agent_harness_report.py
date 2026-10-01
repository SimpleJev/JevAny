#!/usr/bin/env python3
"""Build the agent-harness appendix and append it to the released report.

The Markdown file is authoritative.  This small renderer intentionally supports
only the constructs used by the appendix and has no Python dependencies beyond
pypdf.  It requires ``groff`` with the PDF device.

The original ``JevAny_Tech_Report.pdf`` is read but never modified.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

from pypdf import PdfReader, PdfWriter


HERE = Path(__file__).resolve().parent
MARKDOWN = HERE / "JevAny_Tech_Report_Agent_Harness_Appendix.md"
ORIGINAL = HERE / "JevAny_Tech_Report.pdf"
APPENDIX_PDF = HERE / "JevAny_Tech_Report_Agent_Harness_Appendix.pdf"
COMBINED_PDF = HERE / "JevAny_Tech_Report_with_Agent_Harness.pdf"


UNICODE_ASCII = str.maketrans(
    {
        "→": "->",
        "−": "-",
        "–": "-",
        "—": "--",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "…": "...",
        "•": "*",
        "≥": ">=",
        "≤": "<=",
    }
)


def plain(text: str) -> str:
    """Convert the appendix's inline Markdown to safe roff text."""

    text = text.translate(UNICODE_ASCII)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    text = text.replace("\\", r"\e")
    text = text.replace("-", r"\-")
    if text.startswith((".", "'")):
        text = r"\&" + text
    return text


def emit_paragraph(out: list[str], lines: list[str]) -> None:
    if not lines:
        return
    out.extend([".PP", plain(" ".join(s.strip() for s in lines))])
    lines.clear()


def emit_table(out: list[str], rows: list[list[str]]) -> None:
    """Render wide Markdown tables as compact, wrapping record lists."""

    if len(rows) < 2:
        return
    header = rows[0]
    out.extend([".RS 0.18i", ".ps -1", ".vs -1"])
    for row in rows[2:]:
        values = row + [""] * (len(header) - len(row))
        fields = [f"{h}: {v}" for h, v in zip(header, values) if h and v]
        out.extend([r".IP \(bu 2", plain("; ".join(fields))])
    out.extend([".vs +1", ".ps +1", ".RE"])


def markdown_to_ms(source: str) -> str:
    out = [
        ".ds LH JevAny Agent-Harness Appendix",
        ".ds RH SimpleJev Technical Report",
        ".ds CF \\\\n%",
        ".po 0.75i",
        ".ll 7.0i",
        ".ps 10",
        ".vs 12",
        ".hy 14",
    ]
    paragraph: list[str] = []
    table: list[list[str]] = []
    in_code = False
    code: list[str] = []
    seen_h2 = False

    def flush_table() -> None:
        nonlocal table
        if table:
            emit_table(out, table)
            table = []

    for raw in source.splitlines():
        line = raw.rstrip()

        if line.startswith("```"):
            emit_paragraph(out, paragraph)
            flush_table()
            if in_code:
                out.extend([".DS L", ".ft CR", ".ps 8", ".vs 9"])
                for code_line in code:
                    chunks = textwrap.wrap(
                        code_line,
                        width=94,
                        subsequent_indent="    ",
                        replace_whitespace=False,
                        drop_whitespace=False,
                    ) or [""]
                    out.extend(plain(chunk) for chunk in chunks)
                out.extend([".vs 12", ".ps 10", ".ft R", ".DE"])
                code = []
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue

        if line.startswith("|") and line.endswith("|"):
            emit_paragraph(out, paragraph)
            table.append([cell.strip() for cell in line.strip("|").split("|")])
            continue
        flush_table()

        if not line:
            emit_paragraph(out, paragraph)
            continue
        if line.startswith("# "):
            emit_paragraph(out, paragraph)
            out.extend([".TL", plain(line[2:]), ".AU", "SimpleJev", ".AI", "Technical report appendix"])
        elif line.startswith("## "):
            emit_paragraph(out, paragraph)
            if not seen_h2:
                out.append(".bp")
                seen_h2 = True
            else:
                out.append(".ne 4")
            out.extend([".SH", plain(line[3:])])
        elif line.startswith("### "):
            emit_paragraph(out, paragraph)
            out.extend([".SH", plain(line[4:])])
        elif re.match(r"^\d+\.\s", line):
            emit_paragraph(out, paragraph)
            marker, body = line.split(" ", 1)
            out.extend([f'.IP "{plain(marker)}" 4', plain(body)])
        elif line.startswith("- "):
            emit_paragraph(out, paragraph)
            out.extend([r".IP \(bu 2", plain(line[2:])])
        else:
            paragraph.append(line.rstrip("  "))

    emit_paragraph(out, paragraph)
    flush_table()
    return "\n".join(out) + "\n"


def merge_pdfs() -> None:
    writer = PdfWriter()
    original = PdfReader(ORIGINAL)
    appendix = PdfReader(APPENDIX_PDF)
    for page in original.pages:
        writer.add_page(page)
    for page in appendix.pages:
        writer.add_page(page)
    writer.add_metadata(
        {
            "/Title": "JevAny: Toward General Decision Intelligence — with Agent-Harness Appendix",
            "/Author": "SimpleJev",
            "/Subject": "JevAny model and bounded decision harness evaluation",
        }
    )
    with COMBINED_PDF.open("wb") as handle:
        writer.write(handle)


def main() -> None:
    if not ORIGINAL.exists():
        raise SystemExit(f"missing released report: {ORIGINAL}")
    source = MARKDOWN.read_text(encoding="utf-8")
    roff = markdown_to_ms(source)
    with tempfile.TemporaryDirectory(prefix="jevany-report-") as temp_dir:
        ms_path = Path(temp_dir) / "appendix.ms"
        ms_path.write_text(roff, encoding="utf-8")
        with APPENDIX_PDF.open("wb") as output:
            subprocess.run(
                ["groff", "-Tpdf", "-ms", str(ms_path)],
                check=True,
                stdout=output,
            )
    merge_pdfs()
    print(f"wrote {APPENDIX_PDF}")
    print(f"wrote {COMBINED_PDF}")


if __name__ == "__main__":
    main()
