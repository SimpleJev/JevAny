"""Check public assets and links without installing the Python application."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse, urlsplit
import xml.etree.ElementTree as ET

import markdown


SITE = Path(__file__).resolve().parents[1]
PUBLIC_URL = "https://simplejev.github.io/JevAny/"


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.assets: list[str] = []
        self.h1_count = 0
        self.benchmark_rows: list[dict] = []
        self.background_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if "id" in attributes:
            identifier = attributes["id"]
            assert identifier not in self.ids, f"Duplicate id: {identifier}"
            self.ids.add(identifier)
        if tag == "a" and attributes.get("name"):
            self.ids.add(attributes["name"])
        if tag == "h1":
            self.h1_count += 1
        if tag == "a" and "href" in attributes:
            self.links.append(attributes.get("href", ""))
        for name in ("src", "poster"):
            if name in attributes:
                self.assets.append(attributes[name])
        if "srcset" in attributes:
            self.assets.extend(source.strip().split()[0] for source in attributes["srcset"].split(","))
        if tag == "link" and attributes.get("rel") in ("stylesheet", "icon", "preload"):
            self.assets.append(attributes["href"])
        if tag == "img":
            assert "alt" in attributes, "Image without alt text"
        if tag == "video":
            assert "autoplay" not in attributes, "JS must respect motion preferences"
            if "ambient-video" in attributes.get("class", "").split():
                self.background_count += 1
        if "benchmark-row" in attributes.get("class", "").split():
            self.benchmark_rows.append(attributes)


def check_readme_links(pages: dict[Path, Page]) -> int:
    readmes = {"README.md", "README.zh-CN.md"}
    count = 0
    for name in sorted(readmes):
        parser = Page()
        parser.feed(markdown.markdown(
            (SITE.parent / name).read_text(), extensions=["tables", "fenced_code", "toc"],
        ))
        for link in parser.links:
            parts = urlsplit(link)
            if link.startswith(PUBLIC_URL):
                target = (SITE / unquote(parts.path.removeprefix("/JevAny/"))).resolve()
                if target.is_dir():
                    target /= "index.html"
                assert target.is_file(), f"Missing website target in {name}: {link}"
                if parts.fragment:
                    assert unquote(parts.fragment) in pages[target].ids, f"Broken website anchor in {name}: {link}"
                count += 1
            elif not parts.scheme:
                if not parts.path:
                    assert unquote(parts.fragment) in parser.ids, f"Broken README anchor in {name}: {link}"
                else:
                    target = SITE.parent / unquote(parts.path)
                    assert target.exists(), f"Broken repository link in {name}: {link}"
                    assert target.suffix != ".md" or target.name in readmes, f"Link to the website guide in {name}: {link}"
        for asset in parser.assets:
            if not urlsplit(asset).scheme:
                assert (SITE.parent / unquote(asset)).is_file(), f"Missing README image in {name}: {asset}"
    return count


def main() -> None:
    pages = {}
    for path in SITE.rglob("*.html"):
        parser = Page()
        parser.feed(path.read_text())
        assert parser.h1_count == 1, f"Expected one h1: {path}"
        pages[path.resolve()] = parser
    links_count = 0
    assets_count = 0
    for path, parser in pages.items():
        for link in parser.links:
            assert link and link != "#", f"Placeholder link in {path}"
            parts = urlsplit(link)
            if parts.scheme:
                assert not link.startswith("https://github.com/SimpleJev/JevAny/blob/"), link
                assert not link.startswith(PUBLIC_URL), f"Preview link points to production in {path.name}: {link}"
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            assert target.exists(), f"Broken link in {path.name}: {link}"
            if target.is_dir():
                target = target / "index.html"
            if parts.fragment and target.suffix == ".html":
                assert unquote(parts.fragment) in pages[target].ids, f"Broken anchor in {path.name}: {link}"
        for asset in parser.assets:
            assert not urlparse(asset).scheme, f"External runtime dependency: {asset}"
            assert (path.parent / unquote(asset)).is_file(), f"Missing asset in {path.name}: {asset}"
        links_count += len(parser.links)
        assets_count += len(parser.assets)
    css = (SITE / "styles.css").read_text()
    assets = re.findall(r'url\("([^"]+)"\)', css)
    for asset in assets:
        assert not urlparse(asset).scheme, f"External runtime dependency: {asset}"
        assert (SITE / unquote(asset)).is_file(), f"Missing asset: {asset}"
    cases = json.loads((SITE / "assets/data/cases.json").read_text())
    messages = json.loads((SITE / "locales/zh-CN.json").read_text())
    generated_messages = (SITE / "assets/i18n/zh-CN.js").read_text().split("window.jevanyMessages = ", 1)[1]
    assert json.loads(generated_messages.removesuffix(";\n")) == messages, "Rebuild the Chinese homepage"
    for case in cases:
        assert case["title"] in messages and case["description"] in messages, f"Untranslated case: {case['id']}"
    for source, translation in messages.items():
        assert set(re.findall(r"\{\w+\}", source)) == set(re.findall(r"\{\w+\}", translation)), source
    sources = {p.stem for p in (SITE.parent / "docs/demos/cases").glob("*.gif")}
    assert len(cases) == len({case["id"] for case in cases}) == 30
    assert {case["id"] for case in cases} == sources
    backgrounds = ("grid", "grid-mobile")
    assert pages[(SITE / "index.html").resolve()].background_count == 1
    assert pages[(SITE / "zh.html").resolve()].background_count == 1
    for name in backgrounds:
        for extension in ("mp4", "webm", "webp"):
            assert (SITE / f"assets/media/ambient/{name}.{extension}").stat().st_size > 1000
    for name in sources:
        for extension in ("mp4", "webm", "webp"):
            assert (SITE / f"assets/media/{name}.{extension}").stat().st_size > 1000
    results = json.loads((SITE / "assets/data/model-family-v2.json").read_text())
    assert results == json.loads((SITE.parent / "results/model-family-v2.json").read_text())
    catalog = json.loads((SITE / "assets/data/supported-models.json").read_text())
    assert catalog == json.loads((SITE.parent / "docs/supported-models.json").read_text())
    expected = sorted(results["released_models"] + results["references"], key=lambda m: m["jevbench_public_accuracy"], reverse=True)
    rows = pages[(SITE / "index.html").resolve()].benchmark_rows
    assert rows == pages[(SITE / "zh.html").resolve()].benchmark_rows
    assert len(rows) == len(expected) == 9
    for row, model in zip(rows, expected):
        for metric in ("jevbench_public_accuracy", "transfer_v9_accuracy", "transfer_v9_nll", "transfer_v9_brier", "transfer_v9_ece"):
            assert float(row[f'data-{metric.replace("_", "-")}']) == model[metric]
    assert (SITE / "assets/social.jpg").is_file()
    assert sum(p.stat().st_size for p in (SITE / "assets").rglob("*") if p.is_file()) < 25_000_000
    ET.parse(SITE / "assets/favicon.svg")
    ET.parse(SITE / "sitemap.xml")
    readme_links = check_readme_links(pages)
    print(f"PASS: {len(pages)} pages, {links_count} links, {assets_count + len(assets)} assets, {readme_links} README website links, one background stream, 30 replays, exact benchmark data.")


if __name__ == "__main__":
    main()
