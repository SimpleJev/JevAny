"""Exercise the published /JevAny/ path in Chromium and save optional previews."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
from threading import Thread
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


SITE = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if not path.startswith("/JevAny/"):
            self.send_error(404)
            return
        self.path = self.path[len("/JevAny"):]
        super().do_GET()

    def log_message(self, *_args: object) -> None:
        pass


def check(url: str, output: Path) -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--disable-dev-shm-usage"])
        context = browser.new_context(
            viewport={"width": 1440, "height": 1000},
            permissions=["clipboard-read", "clipboard-write"],
        )
        errors: list[str] = []
        context.on("page", lambda new_page: new_page.on("pageerror", lambda error: errors.append(str(error))))
        page = context.new_page()
        page.goto(url, wait_until="networkidle")
        page.evaluate("document.fonts.ready")
        assert page.title() == "JevAny — Any model. Your next move."
        assert page.locator(".ambient-video.is-visible").count() == 1
        page.wait_for_function("document.querySelector('.ambient-video.is-visible').currentTime > 0")

        # A real 16-second rotation must crossfade into the second layer.
        first_source = page.locator(".ambient-video.is-visible").get_attribute("src")
        page.wait_for_function(
            "first => document.querySelector('.ambient-video.is-visible').getAttribute('src') !== first",
            arg=first_source, timeout=25000,
        )
        page.wait_for_timeout(4500)
        assert page.locator(".ambient-video.is-visible").count() == 1
        assert page.locator(".ambient-video:not(.is-visible)").evaluate("v => v.paused")

        # All six selectors must load real, decodable videos with the right labels.
        page.locator("#in-action").scroll_into_view_if_needed()
        for name in ("peg_insertion", "drone", "lab", "frontend", "sql", "chess"):
            page.locator(f'[data-case="{name}"]').click()
            page.wait_for_function("document.querySelector('#replay-video').readyState >= 2")
            assert name in page.locator("#replay-video").get_attribute("src")
            assert page.locator(f'[data-case="{name}"]').get_attribute("aria-pressed") == "true"
            assert page.locator(".replay-choice[aria-pressed=true]").count() == 1

        page.locator("#motion-toggle").click()
        assert page.locator("#motion-toggle").get_attribute("aria-pressed") == "true"
        assert page.locator("video").evaluate_all("vs => vs.every(v => v.paused)")
        times = page.locator("video").evaluate_all("vs => vs.map(v => v.currentTime)")
        page.wait_for_timeout(500)
        assert times == page.locator("video").evaluate_all("vs => vs.map(v => v.currentTime)")
        page.locator("#motion-toggle").click()
        page.locator("#in-action").scroll_into_view_if_needed()
        page.wait_for_function("!document.querySelector('#replay-video').paused")
        page.locator("#replay-video").evaluate("v => v.pause()")
        page.wait_for_timeout(100)
        page.locator("#get-started").scroll_into_view_if_needed()
        page.wait_for_timeout(150)
        page.locator("#in-action").scroll_into_view_if_needed()
        page.wait_for_timeout(150)
        assert page.locator("#replay-video").evaluate("v => v.paused"), "Native pause lost"

        # Keyboard tabs, actual recipe content, and clipboard contents.
        page.locator("#tab-preview").click()
        page.locator("#tab-preview").press("ArrowRight")
        assert page.locator("#tab-train").get_attribute("aria-selected") == "true"
        assert "data validate" in page.locator("#quickstart-code").inner_text()
        page.locator("#tab-train").press("End")
        assert page.locator("#tab-serve").get_attribute("aria-selected") == "true"
        assert "--device cuda" in page.locator("#quickstart-code").inner_text()
        page.locator("#tab-serve").press("Home")
        page.locator("#copy-code").click()
        assert page.evaluate("navigator.clipboard.readText()") == page.locator("#quickstart-code").inner_text()
        page.evaluate("Object.defineProperty(navigator, 'clipboard', {value: undefined, configurable: true})")
        page.locator("#copy-code").click()
        assert "Commands selected" in page.locator("#copy-status").inner_text()

        # Responsive layout and preview artifacts; keep videos still in screenshots.
        page.evaluate("window.getSelection().removeAllRanges()")
        page.locator('[data-case="peg_insertion"]').click()
        page.locator("#motion-toggle").click()
        for width, height in ((1440, 1000), (1280, 800), (768, 1024), (640, 900), (540, 900), (390, 844), (320, 700)):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.wait_for_timeout(150)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Overflow at {width}"
            assert page.locator("img").evaluate_all("imgs => imgs.every(i => i.complete && i.naturalWidth > 0)")
            if width == 320:
                assert "accuracy Qwen3.8-27B" in page.locator(".facts > a:last-child").inner_text()
                for selector in (".api-code", ".terminal pre"):
                    code = page.locator(selector)
                    assert code.get_attribute("tabindex") == "0"
                    code.focus()
                    code.press("ArrowRight")
                    page.wait_for_timeout(150)
                    assert code.evaluate("el => el.scrollLeft > 0"), f"Keyboard scroll failed: {selector}"
            if width in (1440, 390):
                page.screenshot(path=str(output / f"homepage-{width}.png"), full_page=True)
                page.screenshot(path=str(output / f"hero-{width}.png"))

        reduced = browser.new_context(viewport={"width": 390, "height": 844}, reduced_motion="reduce")
        reduced.on("page", lambda new_page: new_page.on("pageerror", lambda error: errors.append(str(error))))
        reduced_page = reduced.new_page()
        background_requests: list[str] = []
        reduced_page.on("request", lambda request: background_requests.append(request.url) if "/ambient/" in request.url else None)
        reduced_page.goto(url, wait_until="networkidle")
        assert reduced_page.locator("#motion-toggle").get_attribute("aria-pressed") == "true"
        assert reduced_page.locator("video").evaluate_all("vs => vs.every(v => v.paused)")
        assert not background_requests, background_requests
        reduced_page.locator("#in-action").scroll_into_view_if_needed()
        reduced_page.locator("#replay-video").evaluate("v => v.play()")
        reduced_page.wait_for_function("document.querySelector('#replay-video').currentTime > 0")
        assert not background_requests
        reduced.close()

        # Save-Data also keeps backgrounds off. Decorative failure must not break the page.
        saved = browser.new_context()
        saved.on("page", lambda new_page: new_page.on("pageerror", lambda error: errors.append(str(error))))
        saved.add_init_script("Object.defineProperty(navigator, 'connection', {value: {saveData: true}})")
        saved_page = saved.new_page()
        saved_page.goto(url, wait_until="networkidle")
        assert saved_page.locator("#motion-toggle").get_attribute("aria-pressed") == "true"
        assert saved_page.locator(".ambient-video").evaluate_all("vs => vs.every(v => !v.getAttribute('src'))")
        saved.close()
        failed = browser.new_context()
        failed_page = failed.new_page()
        failed_page.on("pageerror", lambda error: errors.append(str(error)))
        failed_page.route("**/backgrounds.json", lambda route: route.abort())
        failed_page.goto(url, wait_until="networkidle")
        assert failed_page.locator(".hero h1").is_visible()
        assert failed_page.locator(".ambient-poster").is_visible()
        failed.close()

        # Hold the next background request open; the load guard must skip it.
        stalled = browser.new_context()
        stalled.on("page", lambda new_page: new_page.on("pageerror", lambda error: errors.append(str(error))))
        stalled_page = stalled.new_page()
        pending_routes = []
        stalled_page.route("**/ambient/chess.*", lambda route: pending_routes.append(route))
        stalled_page.goto(url, wait_until="networkidle")
        stalled_page.wait_for_function(
            "document.querySelector('.ambient-video.is-visible')?.currentSrc.includes('satellite')",
            timeout=65000,
        )
        assert pending_routes, "The intended background request was not held"
        for route in pending_routes:
            route.abort()
        stalled.close()

        plain = browser.new_context(java_script_enabled=False)
        plain_page = plain.new_page()
        plain_page.goto(url, wait_until="networkidle")
        assert plain_page.locator(".hero h1").is_visible()
        assert plain_page.locator(".noscript-note").is_visible()
        assert "jevany demo" in plain_page.locator("#quickstart-code").inner_text()
        plain.close()
        assert not errors, errors
        result = {
            "status": "passed",
            "viewports": [1440, 1280, 768, 640, 540, 390, 320],
            "checks": ["background crossfade", "six decoded replays", "global pause",
                       "native pause", "keyboard tabs", "clipboard and fallback",
                       "no overflow", "reduced motion", "save data", "network fallback",
                       "stalled background recovery", "keyboard code scrolling", "no JavaScript fallback"],
            "console_errors": errors,
        }
        (output / "browser-results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Check an already running or published site")
    parser.add_argument("--output", type=Path, help="Directory for screenshots and results")
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="jevany-site-check-"))
    output.mkdir(parents=True, exist_ok=True)
    server = None
    try:
        if args.url:
            url = args.url
        else:
            server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(SITE)))
            Thread(target=server.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{server.server_port}/JevAny/"
        check(url, output)
    finally:
        if server:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    main()
