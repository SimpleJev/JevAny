"""Build the static homepage data and local documentation from repository sources.

Install site/tools/requirements.txt, then run python site/tools/build_content.py.
Generated HTML is committed with the site; visitors need no Markdown renderer.
"""

from collections import defaultdict
from html import escape, unescape
import json
import os
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

import markdown

from localization import build_chinese_homepage, translate


SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
PUBLIC_URL = "https://simplejev.github.io/JevAny/"
RESULTS = json.loads((ROOT / "results/model-family-v2.json").read_text())
CATALOG = json.loads((ROOT / "docs/supported-models.json").read_text())
LOGOS = json.loads((SITE / "assets/model-logos/sources.json").read_text())["families"]
BASELINE_LOGOS = {
    "Kev-4B": "kev.svg",
    "Kev-27B": "kev.svg",
    "Jev 1.13.0": "typesafe.png",
    "Laya": "laya.svg",
}
METRICS = {
    "jevbench_public_accuracy": ("JevBench", True),
    "transfer_v9_accuracy": ("Transfer", True),
    "transfer_v9_nll": ("NLL", False),
    "transfer_v9_brier": ("Brier", False),
    "transfer_v9_ece": ("ECE", False),
}
SPECIAL_DOCS = {
    "README.md": "quickstart.html",
    "README.zh-CN.md": "zh.html",
    "CONTRIBUTING.md": "contributing.html",
    "examples/README.md": "examples.html",
}
DOCUMENTS = {}
ATTACHMENTS = set()


def model_name(model):
    return model.get("model") or model["repository"].split("/")[1].removesuffix("-LoRA")


def display(value, accuracy):
    return f"{value * 100:.2f}%" if accuracy else f"{value:.3f}"


def benchmarks():
    models = RESULTS["released_models"] + RESULTS["references"]
    models = sorted(models, key=lambda model: model["jevbench_public_accuracy"], reverse=True)
    tabs = "".join(
        f'<button type="button" data-metric="{key}" aria-pressed="{str(i == 0).lower()}">'
        f'{name} {"↑" if accuracy else "↓"}</button>'
        for i, (key, (name, accuracy)) in enumerate(METRICS.items())
    )
    rows = []
    table = []
    for model in models:
        name = model_name(model)
        ours = "repository" in model
        attrs = " ".join(f'data-{key.replace("_", "-")}="{model[key]}"' for key in METRICS)
        if ours:
            family = next(f for f in ("Qwen", "Gemma", "Muse") if f in name)
            logo = f"jevany-{family.lower()}.svg"
        else:
            logo = BASELINE_LOGOS[name]
        label = (
            f'<span class="benchmark-model" translate="no">'
            f'<img src="assets/model-logos/{logo}" width="24" height="24" alt="">'
            f'<span>{escape(name)}</span></span>'
        )
        value = display(model["jevbench_public_accuracy"], True)
        rows.append(
            f'<li class="benchmark-row {"ours" if ours else "baseline"}" {attrs}>'
            f'{label}'
            f'<span class="benchmark-track" aria-hidden="true"><i style="width:{value}"></i></span>'
            f'<span class="benchmark-value">{value}</span></li>'
        )
        values = "".join(f"<td>{display(model[key], accuracy)}</td>" for key, (_, accuracy) in METRICS.items())
        tiers = "".join(f'<td>{display(model["jevbench_tiers"][tier], True)}</td>' for tier in ("easy", "original", "hard"))
        table.append(f'<tr><th scope="row">{label}</th>{values}{tiers}</tr>')
    return f"""
      <div class="benchmark-panel">
        <div class="benchmark-toolbar"><div class="metric-buttons" role="group" aria-label="Benchmark metric" hidden>{tabs}</div>
          <span class="chart-legend"><i></i> JevAny <i></i> Baselines</span></div>
        <div class="chart-heading"><h3 id="chart-title">JevBench accuracy</h3><p id="chart-description">231 public development items · Higher is better</p></div>
        <div class="chart-axis" aria-hidden="true"><span>0%</span><span id="chart-axis-end">100%</span></div>
        <ol class="benchmark-chart" aria-labelledby="chart-title" aria-describedby="chart-description">{"".join(rows)}</ol>
        <p id="metric-status" class="sr-only" role="status"></p>
      </div>
      <details class="disclosure benchmark-table"><summary>Full metrics &amp; JevBench difficulty tiers <span aria-hidden="true">+</span></summary>
        <div class="table-scroll" tabindex="0" role="region" aria-label="All benchmark results">
          <table><caption>Current releases and baselines. ↑ Higher is better. ↓ Lower is better. Easy, Original and Hard are JevBench tiers.</caption>
          <thead><tr><th scope="col">Model</th>{"".join(f'<th scope="col">{name} {"↑" if acc else "↓"}</th>' for name, acc in METRICS.values())}
          <th scope="col">Easy ↑</th><th scope="col">Original ↑</th><th scope="col">Hard ↑</th></tr></thead>
          <tbody>{"".join(table)}</tbody></table>
        </div>
      </details>"""


def cases():
    records = []
    category = ""
    for line in (ROOT / "docs/CASES.md").read_text().splitlines():
        if line.startswith("## "):
            category = line[3:]
        match = re.match(r"\| (.+?) \| (.+?) \| .*?demos/cases/([\w]+)\.gif", line)
        if match:
            title, description, name = match.groups()
            records.append({"id": name, "title": title, "description": description, "category": category})
    assert len(records) == 30
    cards = "".join(
        f'<a class="case-card" id="case-{case["id"]}" href="assets/media/{case["id"]}.mp4" data-case="{case["id"]}">'
        f'<img src="assets/media/{case["id"]}.webp" loading="lazy" width="480" height="270" alt="">'
        f'<span class="case-card-title">{escape(case["title"])} <span aria-hidden="true">▷</span></span>'
        f'<span class="case-card-description">{escape(case["description"])}</span></a>'
        for case in records
    )
    return records, f'<details class="disclosure case-library" id="case-library"><summary>Explore all 30 cases <span aria-hidden="true">+</span></summary><div class="case-grid">{cards}</div></details>'


def models():
    blurbs = {
        "JevAny-Gemma-4B": "Compact Gemma release.",
        "JevAny-Qwen3.5-4B": "Compact, with flexible choice counts.",
        "JevAny-Qwen3.5-4B-Direct-Token": "Highest JevBench accuracy among released 4B models.",
        "JevAny-Qwen3.8-27B": "Default release. Highest released accuracy on both suites.",
        "JevAny-Muse-Glimmer-30B": "A Muse Glimmer alternative.",
    }
    cards = []
    for model in sorted(RESULTS["released_models"], key=lambda m: m["jevbench_public_accuracy"], reverse=True):
        name = model_name(model)
        family = next(f for f in ("Qwen", "Gemma", "Muse") if f in name)
        cards.append(
            f'<article class="checkpoint-card"><img src="assets/model-logos/jevany-{family.lower()}.svg" width="48" height="48" alt="">'
            f'<h3 translate="no">{escape(name.removeprefix("JevAny-"))}</h3><p>{blurbs[name]}</p>'
            f'<a href="https://huggingface.co/{model["repository"]}"><span class="hf-mark" aria-hidden="true">🤗</span>Model card</a>'
            f'<a href="#get-started" data-local-model="{escape(model["repository"], quote=True)}">Run locally →</a>'
            f'<details class="checkpoint-details"><summary>Checkpoint details</summary>'
            f'<p>{escape(model["readout"].capitalize())} readout</p><code>{escape(model["repository"])}</code></details></article>'
        )
    families = defaultdict(list)
    for model in CATALOG["models"]:
        families[model["family"]].append(model)
    support = []
    for family, entries in families.items():
        rows = "".join(
            f'<tr><th scope="row"><code>{escape(m["id"])}</code></th><td translate="no">{escape(m["size"])}</td>'
            f'<td>{escape(", ".join(["text"] + m["media"]))}</td></tr>' for m in entries
        )
        support.append(
            f'<details class="disclosure family-details" id="family-{family.lower()}">'
            f'<summary><img src="assets/model-logos/{LOGOS[family]}" width="24" height="24" alt="">'
            f'{family}<small>{len(entries)} {"model" if len(entries) == 1 else "models"}</small><span aria-hidden="true">+</span></summary>'
            f'<div class="table-scroll" tabindex="0" role="region" aria-label="{family} supported models"><table>'
            f'<thead><tr><th scope="col">Base model ID</th><th scope="col">Size</th><th scope="col">Inputs</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></details>'
        )
    return (
        f'<div class="checkpoint-grid">{"".join(cards)}</div>'
        '<p class="readout-note">Pointer supports up to 4,096 options within the context limit; direct-token supports up to 255. Both use the same API. '
        '<a href="docs/TRAINING.html#pointer-and-direct-token-readouts">Readout guide →</a></p>'
        f'<div id="supported-models" class="support-list"><div class="support-heading"><h3>26 supported base models</h3>'
        f'<a class="text-link" href="docs/TRAINING.html#backbone-support">Setup requirements →</a></div>{"".join(support)}</div>'
    )


def document_path(source):
    relative = source.relative_to(ROOT).as_posix()
    name = SPECIAL_DOCS.get(relative, relative.removeprefix("docs/").replace("/", "-").removesuffix(".md") + ".html")
    return SITE / "docs" / name


def relative_url(target, page):
    return os.path.relpath(target, page.parent).replace(os.sep, "/")


def local_url(url, source, page):
    parts = urlsplit(unescape(url))
    suffix = (f"?{parts.query}" if parts.query else "") + (f"#{parts.fragment}" if parts.fragment else "")
    # README links use public URLs. Resolve them through the same source pipeline
    # so previews stay local and referenced guides/downloads are rebuilt.
    if url.startswith(PUBLIC_URL):
        path = unquote(parts.path.removeprefix(urlsplit(PUBLIC_URL).path)) or "index.html"
        destination = (SITE / path).resolve()
        destination.relative_to(SITE.resolve())
        if path.startswith("files/"):
            original = ROOT / path.removeprefix("files/")
            return local_url(os.path.relpath(original, source.parent) + suffix, source, page)
        if path.startswith("docs/"):
            candidates = [ROOT / name for name in SPECIAL_DOCS]
            candidates.extend(ROOT.glob("*.md"))
            for directory in ("docs", "reports", "results"):
                candidates.extend((ROOT / directory).rglob("*.md"))
            original = next((item for item in candidates if document_path(item) == destination), None)
            if original is not None:
                return local_url(os.path.relpath(original, source.parent) + suffix, source, page)
        return relative_url(destination, page) + suffix
    # Convert links back to this repository to the same local documentation.
    prefix = "https://github.com/SimpleJev/JevAny/blob/main/"
    if url.startswith(prefix):
        return local_url(os.path.relpath(ROOT / parts.path.split("/blob/main/")[1], source.parent)
                         + suffix, source, page)
    if parts.scheme or url.startswith("//") or not parts.path:
        return url
    target = (source.parent / unquote(parts.path)).resolve()
    target.relative_to(ROOT)  # Documentation links must remain in this repository.
    if target.name in ("evaluation-overview.svg", "evaluation-summary.svg", "evaluation-checkpoints.svg"):
        return relative_url(SITE / "index.html", page) + "#benchmarks"
    if target.suffix == ".md":
        destination = document_path(target)
        DOCUMENTS.setdefault(target, destination)
    elif target == ROOT / "LICENSE":
        destination = SITE / "docs/license.html"
    elif target.suffix == ".gif" and target.parent == ROOT / "docs/demos/cases":
        destination = SITE / f"assets/media/{target.stem}.mp4"
    elif target == ROOT / "docs/demos/jevany-cases.gif":
        return relative_url(SITE / "index.html", page) + "#case-library"
    elif target.suffix == ".gif" and target.parent == ROOT / "docs/demos":
        destination = SITE / f"assets/media/docs/{target.stem}.mp4"
    elif target.is_dir():
        destination = SITE / "files" / target.relative_to(ROOT) / "index.html"
        if destination not in ATTACHMENTS:
            ATTACHMENTS.add(destination)
            links = []
            for child in sorted(target.iterdir()):
                if child.is_file() and child.suffix in (".py", ".md", ".json", ".toml", ".txt", ".sh", ".svg"):
                    link = local_url(child.name, target / "_index.md", destination)
                    links.append(f'<li><a href="{escape(link, quote=True)}">{escape(child.name)}</a></li>')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(doc_shell(target.name, "<ul>" + "".join(links) + "</ul>", "", destination))
    else:
        assert target.is_file(), f"Missing documentation attachment: {target}"
        destination = SITE / "files" / target.relative_to(ROOT)
        if destination not in ATTACHMENTS:
            ATTACHMENTS.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, destination)
    return relative_url(destination, page) + suffix


def doc_shell(title, body, toc, page, chinese=False):
    home = relative_url(SITE / ("zh.html" if chinese else "index.html"), page)
    text = translate if chinese else lambda value: value
    css = relative_url(SITE / "styles.css", page)
    icon = relative_url(SITE / "assets/favicon.svg", page)
    links = [
        ("Quickstart", "quickstart.html"), ("Training", "TRAINING.html"),
        ("Data", "DATA.html"), ("Deployment", "DEPLOYMENT.html"),
        ("API reference", "API.html"), ("Examples", "examples.html"),
        ("Cases", "CASES.html"), ("Evaluation", "EVALUATION.html"),
        ("Method", "ALGORITHM.html"), ("Integrations", "INTEGRATIONS.html"),
        ("中文文档", "zh.html"),
    ]
    nav = ""
    for label, filename in links:
        current = ' aria-current="page"' if page.name == filename else ""
        label = text(label)
        if chinese and filename != "zh.html":
            label += "（英文）"
        nav += f'<a href="{relative_url(SITE / "docs" / filename, page)}"{current}>{label}</a>'
    language_links = ""
    if page.name in {"quickstart.html", "zh.html"}:
        en_current = ' aria-current="page"' if not chinese else ""
        zh_current = ' aria-current="page"' if chinese else ""
        language_links = (
            '<div class="language-switch" role="group" aria-label="' + text("Language") + '">'
            f'<a href="quickstart.html" lang="en" hreflang="en" title="English"{en_current}>EN</a>'
            f'<a href="zh.html" lang="zh-CN" hreflang="zh-CN" title="中文"{zh_current}>中文</a></div>'
        )
    return f"""<!doctype html>
<html lang="{"zh-CN" if chinese else "en"}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#0a1114">
<title>{escape(title)} — JevAny</title><link rel="stylesheet" href="{css}"><link rel="icon" href="{icon}" type="image/svg+xml"></head>
<body class="docs-page"><a class="skip-link" href="#content">{text("Skip to content")}</a>
<header class="site-header wrap"><a class="brand" href="{home}"><img src="{icon}" width="30" height="30" alt=""><span>JevAny<span class="brand-period">.</span></span></a>
<nav aria-label="{text("Main navigation")}"><a href="{home}#benchmarks">{text("Benchmarks")}</a><a href="{home}#models">{text("Models")}</a><a href="{home}#get-started">{text("Get started")}</a></nav>{language_links}</header>
<div class="docs-layout wrap"><aside class="docs-sidebar"><a class="text-link" href="{home}">{text("← Project homepage")}</a><nav aria-label="{text("Documentation")}">{nav}</nav>
<details class="doc-toc"><summary>{text("On this page")}</summary>{toc}</details></aside>
<main id="content" class="doc-content"><p class="eyebrow">{text("JEVANY / DOCUMENTATION")}</p><h1>{escape(title)}</h1>{body}</main></div>
<footer class="site-footer wrap"><a href="{home}">{text("← Back to JevAny")}</a><a href="{relative_url(SITE / 'docs/license.html', page)}">{text("Apache-2.0 code")}</a></footer></body></html>
"""


def render_doc(source, page):
    text = source.read_text()
    text = re.sub(r"\A---\n.*?\n---\n+", "", text, flags=re.S)
    chinese = source.name == "README.zh-CN.md"
    match = re.match(r"# (.+)\n", text)
    title = match[1] if match else ("开始使用 JevAny" if chinese else "Get started with JevAny")
    if match:
        text = text[match.end():]
    # The website renders benchmark results as HTML, and owns its navigation.
    text = re.sub(r"\[!\[[^\]]*\]\(docs/evaluation-[^)]+\)\]\([^)]+\)", "[Explore interactive benchmark results](site-benchmarks)", text)
    if source.name.startswith("README") and source.parent == ROOT:
        start = text.find("**JevAny")
        if start >= 0:
            text = text[start:]
    if source.name == "CASES.md":
        text = re.sub(r"\[!\[[^\]]*\]\(demos/jevany-cases.gif\)\]\([^)]+\)", "[Explore all 30 replays on the homepage](site-cases)", text)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = md.convert(text)
    body = body.replace('href="site-benchmarks"', f'href="{relative_url(SITE / "index.html", page)}#benchmarks"')
    body = body.replace('href="site-cases"', f'href="{relative_url(SITE / "index.html", page)}#case-library"')
    body = re.sub(
        r'<a href="[^"]+"><img[^>]*src="[^"]*jevany-cases\.gif"[^>]*></a>',
        '<a href="../index.html#case-library">Explore all 30 application replays →</a>', body)
    # Case recordings use the same lightweight videos as the homepage.
    body = re.sub(
        r'<a href="demos/cases/(\w+)\.gif"><img[^>]*></a>',
        lambda m: f'<a href="../assets/media/{m[1]}.mp4" class="doc-replay">Watch replay ▷</a>',
        body,
    )
    def media_tag(match):
        attributes = match[0]
        src = re.search(r'\bsrc="([^"]+)"', attributes)
        if not src:
            return attributes
        target = (source.parent / unquote(src[1])).resolve()
        if target.suffix != ".gif" or target.parent != ROOT / "docs/demos":
            return attributes
        alt = re.search(r'\balt="([^"]*)"', attributes)
        label = alt[1] if alt else "Recorded JevAny replay"
        return (f'<video class="doc-video" controls playsinline preload="none" aria-label="{label}" '
                f'poster="../assets/media/docs/{target.stem}.webp">'
                f'<source src="../assets/media/docs/{target.stem}.webm" type="video/webm">'
                f'<source src="../assets/media/docs/{target.stem}.mp4" type="video/mp4">'
                'Your browser does not support video.</video>')
    body = re.sub(r"<img\b[^>]*>", media_tag, body)
    body = re.sub(r'<a href="[^"]+">(<video\b.*?</video>)</a>', r"\1", body)
    # Already-local URLs added above do not need source-relative resolution.
    body = re.sub(
        r'\b(href|src)="([^"]+)"',
        lambda m: m[0] if m[2].startswith(("../index.html", "../assets/")) else
        f'{m[1]}="{escape(local_url(m[2], source, page), quote=True)}"',
        body,
    )
    body = re.sub(r"<table>(.*?)</table>", r'<div class="table-scroll" tabindex="0" role="region" aria-label="Documentation table"><table>\1</table></div>', body, flags=re.S)
    body = re.sub(r"<pre>", '<pre tabindex="0" role="region" aria-label="Code example">', body)
    body = body.replace("<img ", '<img loading="lazy" ')
    if chinese:
        body = body.replace("../index.html", "../zh.html")
        for message in ("Explore interactive benchmark results", "Explore all 30 application replays →",
                        "Documentation table", "Code example", "Your browser does not support video."):
            body = body.replace(message, translate(message))
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(doc_shell(title, body, md.toc, page, chinese))


def main():
    global ROOT
    ROOT = ROOT.resolve()
    for family in ("qwen", "gemma", "muse"):
        filename = f"jevany-{family}.svg"
        shutil.copyfile(ROOT / "docs/model-logos" / filename, SITE / "assets/model-logos" / filename)
    for filename in (*sorted(set(BASELINE_LOGOS.values())), "LICENSE-laya"):
        shutil.copyfile(ROOT / "docs/model-logos" / filename, SITE / "assets/model-logos" / filename)
    for source in sorted((ROOT / "docs").glob("*.md")):
        DOCUMENTS[source] = document_path(source)
    for name in SPECIAL_DOCS:
        source = ROOT / name
        DOCUMENTS[source] = document_path(source)
    records, case_html = cases()
    replacements = {"cases": case_html, "benchmarks": benchmarks(), "models": models()}
    index = SITE / "index.html"
    html = index.read_text()
    for name, content in replacements.items():
        html = re.sub(f"(<!-- generated:{name}:start -->).*?(<!-- generated:{name}:end -->)",
                      lambda m: m[1] + "\n" + content.strip() + "\n      " + m[2], html, flags=re.S)
    index.write_text(html)
    build_chinese_homepage(html)
    data = SITE / "assets/data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "cases.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    for source in ("results/model-family-v2.json", "docs/supported-models.json"):
        shutil.copyfile(ROOT / source, data / Path(source).name)
    rendered = set()
    while pending := [source for source in DOCUMENTS if source not in rendered]:
        for source in pending:
            render_doc(source, DOCUMENTS[source])
            rendered.add(source)
    license_page = SITE / "docs/license.html"
    license_page.write_text(doc_shell("Apache License 2.0", f"<pre>{escape((ROOT / 'LICENSE').read_text())}</pre>", "", license_page))
    print(f"Built homepage content, {len(rendered) + 1} documentation pages, {len(ATTACHMENTS)} local attachments.")


if __name__ == "__main__":
    main()
