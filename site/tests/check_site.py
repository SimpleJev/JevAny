"""Check public assets and links without installing the Python application."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


SITE = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.assets: list[str] = []
        self.h1_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if "id" in attributes:
            identifier = attributes["id"]
            assert identifier not in self.ids, f"Duplicate id: {identifier}"
            self.ids.add(identifier)
        if tag == "h1":
            self.h1_count += 1
        if tag == "a":
            self.links.append(attributes.get("href", ""))
        for name in ("src", "poster"):
            if name in attributes:
                self.assets.append(attributes[name])
        if tag == "link" and attributes.get("rel") in ("stylesheet", "icon", "preload"):
            self.assets.append(attributes["href"])
        if tag == "img":
            assert "alt" in attributes, "Image without alt text"
        if tag == "video":
            assert "autoplay" not in attributes, "JS must respect motion preferences"


def main() -> None:
    parser = Page()
    parser.feed((SITE / "index.html").read_text())
    assert parser.h1_count == 1
    for link in parser.links:
        assert link and link != "#", "Placeholder link"
        if link.startswith("#"):
            assert link[1:] in parser.ids, f"Broken anchor: {link}"
        elif not urlparse(link).scheme:
            assert (SITE / unquote(link.split("#")[0])).exists(), link
    css = (SITE / "styles.css").read_text()
    assets = parser.assets + re.findall(r'url\("([^"]+)"\)', css)
    for asset in assets:
        assert not urlparse(asset).scheme, f"External runtime dependency: {asset}"
        assert (SITE / unquote(asset)).is_file(), f"Missing asset: {asset}"
    backgrounds = json.loads((SITE / "assets/media/backgrounds.json").read_text())
    sources = {p.stem for p in (SITE.parent / "docs/demos/cases").glob("*.gif")}
    assert len(backgrounds) == len(set(backgrounds)) == 30
    assert set(backgrounds) == sources, "Background catalog differs from archived cases"
    for name in backgrounds:
        for extension in ("mp4", "webm"):
            assert (SITE / f"assets/media/ambient/{name}.{extension}").stat().st_size > 1000
    for name in ("peg_insertion", "drone", "lab", "frontend", "sql", "chess"):
        for extension in ("mp4", "webm", "webp"):
            assert (SITE / f"assets/media/{name}.{extension}").stat().st_size > 1000
    assert (SITE / "assets/social.jpg").is_file()
    assert sum(p.stat().st_size for p in (SITE / "assets").rglob("*") if p.is_file()) < 20_000_000
    ET.parse(SITE / "assets/favicon.svg")
    ET.parse(SITE / "sitemap.xml")
    print(f"PASS: {len(parser.links)} links, {len(assets)} static assets, 30 backgrounds, 6 replays.")


if __name__ == "__main__":
    main()
