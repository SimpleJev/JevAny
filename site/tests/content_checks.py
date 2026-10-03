"""Check published-link routing without network access or the Python application."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_content as builder


class PublishedLinkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.site = self.root / "site"
        self.source = self.root / "README.md"
        self.page = self.site / "docs/quickstart.html"
        self.guide = self.root / "docs/experiments/guide.md"
        self.guide.parent.mkdir(parents=True)
        self.guide.write_text("# Guide\n")
        self.patches = patch.multiple(
            builder, ROOT=self.root, SITE=self.site, DOCUMENTS={}, ATTACHMENTS=set(),
        )
        self.patches.start()
        self.addCleanup(self.patches.stop)

    def test_homepage_keeps_language_query_and_fragment(self):
        for path, expected in (
            ("?from=readme#models", "../index.html?from=readme#models"),
            ("zh.html#case-library", "../zh.html#case-library"),
        ):
            self.assertEqual(
                builder.local_url("https://simplejev.org/JevAny/" + path, self.source, self.page),
                expected,
            )

    def test_published_guide_registers_its_source(self):
        url = "https://simplejev.org/JevAny/docs/experiments-guide.html#guide"
        self.assertEqual(builder.local_url(url, self.source, self.page), "experiments-guide.html#guide")
        self.assertEqual(builder.DOCUMENTS[self.guide], self.site / "docs/experiments-guide.html")

    def test_published_download_is_copied_from_source(self):
        data = self.root / "results/run.json"
        data.parent.mkdir()
        data.write_text('{"accuracy": 0.9}\n')
        url = "https://simplejev.org/JevAny/files/results/run.json?download=1"
        self.assertEqual(builder.local_url(url, self.source, self.page), "../files/results/run.json?download=1")
        self.assertEqual((self.site / "files/results/run.json").read_bytes(), data.read_bytes())

    def test_published_media_stays_on_preview_origin(self):
        url = "https://simplejev.org/JevAny/assets/media/docs/playground-arm.mp4"
        self.assertEqual(builder.local_url(url, self.source, self.page), "../assets/media/docs/playground-arm.mp4")

    def test_external_links_remain_external(self):
        for url in (
            "https://huggingface.co/SimpleJev/JevAny",
            "https://simplejev.org/AnotherProject/docs/API.html",
            "https://simplejev.org.example.com/JevAny/docs/API.html",
        ):
            self.assertEqual(builder.local_url(url, self.source, self.page), url)

    def test_published_path_cannot_escape_site(self):
        with self.assertRaises(ValueError):
            builder.local_url("https://simplejev.org/JevAny/%2e%2e/private.html", self.source, self.page)

    def test_relative_guides_still_register_the_source(self):
        self.assertEqual(
            builder.local_url("docs/experiments/guide.md#guide", self.source, self.page),
            "experiments-guide.html#guide",
        )
        self.assertIn(self.guide, builder.DOCUMENTS)


if __name__ == "__main__":
    unittest.main()
