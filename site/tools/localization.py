"""Render the Chinese homepage from the English HTML and shared message catalog."""

from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re


SITE = Path(__file__).resolve().parents[1]
MESSAGES = json.loads((SITE / "locales/zh-CN.json").read_text())
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


def translate(text):
    key = re.sub(r"\s+", " ", text.strip())
    if not key:
        return text
    if key not in MESSAGES:
        if re.search(r"[A-Za-z]", key):
            raise ValueError(f"Missing Chinese translation: {key!r}")
        return text
    return text[:len(text) - len(text.lstrip())] + MESSAGES[key] + text[len(text.rstrip()):]


class ChineseHomepage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.output = []
        self.stack = []

    def handle_decl(self, decl):
        self.output.append(f"<!{decl}>")

    def handle_comment(self, data):
        self.output.append(f"<!--{data}-->")

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        skipped = (self.stack and self.stack[-1][1]) or tag in {"code", "script", "style"} or attrs.get("translate") == "no" or attrs.get("aria-hidden") == "true"
        if not skipped:
            for name in ("aria-label", "title", "alt", "label"):
                if attrs.get(name):
                    attrs[name] = translate(attrs[name])
            if tag == "meta" and (attrs.get("name") == "description" or attrs.get("property") in {"og:title", "og:description"}):
                attrs["content"] = translate(attrs["content"])
        if tag == "html":
            attrs["lang"] = "zh-CN"
        if tag == "link" and attrs.get("rel") == "canonical":
            attrs["href"] = "https://simplejev.org/JevAny/zh.html"
        if tag == "meta" and attrs.get("property") == "og:url":
            attrs["content"] = "https://simplejev.org/JevAny/zh.html"
        if attrs.get("data-language"):
            attrs.pop("aria-current", None)
            if attrs["data-language"] == "zh-CN":
                attrs["aria-current"] = "page"
        if tag == "a" and not attrs.get("data-language"):
            href = attrs.get("href", "")
            if href == "./":
                attrs["href"] = "zh.html"
            elif href.startswith("docs/quickstart.html"):
                attrs["href"] = href.replace("docs/quickstart.html", "docs/zh.html").replace("#run-locally", "#本地体验")
            elif href.startswith("docs/") and not href.startswith("docs/zh.html"):
                attrs["hreflang"] = "en"
                attrs["title"] = "英文文档"
        if tag == "script" and attrs.get("src") == "app.js":
            self.output.append('<script src="assets/i18n/zh-CN.js" defer></script>\n  ')
        serialized = "".join(f' {key}="{escape(value, quote=True)}"' if value is not None else f" {key}" for key, value in attrs.items())
        self.output.append(f"<{tag}{serialized}>")
        if tag not in VOID_TAGS:
            self.stack.append((tag, skipped))

    def handle_endtag(self, tag):
        self.output.append(f"</{tag}>")
        self.stack.pop()

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data):
        tag, skipped = self.stack[-1] if self.stack else ("", False)
        if tag in {"script", "style"}:
            self.output.append(data)
        else:
            self.output.append(escape(data if skipped else translate(data), quote=False))


def build_chinese_homepage(html):
    page = ChineseHomepage()
    page.feed(html)
    (SITE / "zh.html").write_text("".join(page.output))
    destination = SITE / "assets/i18n/zh-CN.js"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("// Generated from site/locales/zh-CN.json.\nwindow.jevanyMessages = "
                           + json.dumps(MESSAGES, indent=2, ensure_ascii=False) + ";\n")
