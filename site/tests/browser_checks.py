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
        page.add_style_tag(content="html { scroll-behavior: auto !important; }")
        assert page.title() == "JevAny: Decision Models"
        assert page.locator(".ambient-video").count() == 1
        page.wait_for_function("[...document.querySelectorAll('.ambient-video')].every(v => v.currentTime > 0)")
        assert page.locator(".ambient-video").get_attribute("src").endswith("/grid.webm")
        assert page.locator(".ambient-video").evaluate("v => v.videoWidth") == 720
        assert page.locator(".ambient").evaluate("e => getComputedStyle(e).filter") == "none"
        first_transform = page.locator(".model-track").evaluate("e => getComputedStyle(e).transform")
        page.wait_for_timeout(300)
        assert first_transform != page.locator(".model-track").evaluate("e => getComputedStyle(e).transform")
        assert page.locator(".model-group[aria-hidden=true] a").evaluate_all("links => links.every(a => a.tabIndex === -1)")
        assert page.locator('.model-group').first.locator('a[href="#family-muse"] img').get_attribute("src").endswith("/meta-color.svg")
        assert page.locator(".checkpoint-card .hf-mark").count() == 5
        assert page.locator(".hero-accent").evaluate("e => getComputedStyle(e).fontFamily").startswith('"IBM Plex Sans Condensed"')
        assert page.locator(".hero-accent").evaluate("e => getComputedStyle(e).fontStyle") == "normal"
        star = page.locator(".hero .star-button")
        assert star.is_visible() and star.get_attribute("href") == "https://github.com/SimpleJev/JevAny"
        # Check the navigation without authenticating or changing a real GitHub account.
        context.route("https://github.com/SimpleJev/JevAny", lambda route: route.fulfill(body="<h1>GitHub repository</h1>", content_type="text/html"))
        with page.expect_popup() as popup:
            star.click()
        popup.value.wait_for_load_state()
        assert popup.value.url == "https://github.com/SimpleJev/JevAny"
        popup.value.close()

        # All six selectors must load real, decodable videos with the right labels.
        page.locator("#in-action").scroll_into_view_if_needed()
        page.wait_for_function("document.documentElement.classList.contains('models-offscreen')")
        assert page.locator(".model-track").evaluate("e => getComputedStyle(e).animationPlayState") == "paused"
        for name in ("peg_insertion", "drone", "lab", "frontend", "sql", "chess"):
            page.locator(f'.replay-choice[data-case="{name}"]').click()
            page.wait_for_function("document.querySelector('#replay-video').readyState >= 2")
            assert name in page.locator("#replay-video").get_attribute("src")
            assert page.locator(f'.replay-choice[data-case="{name}"]').get_attribute("aria-pressed") == "true"
            assert page.locator(".replay-choice[aria-pressed=true]").count() == 1

        # Every archived case plays here, including those outside the featured six.
        page.locator('a[href="#case-library"]').first.click()
        assert page.locator("#case-library").get_attribute("open") is not None
        assert page.locator(".case-card").count() == 30
        for card in page.locator(".case-card").all():
            name = card.get_attribute("data-case")
            card.click()
            page.wait_for_function("document.querySelector('#replay-video').readyState >= 2")
            assert name in page.locator("#replay-video").get_attribute("src")
            assert page.url == url + "#case-library"
        page.locator("#case-library > summary").click()

        # Metrics must sort in the right direction and display the source values.
        results = json.loads((SITE / "assets/data/model-family-v2.json").read_text())
        models = results["released_models"] + results["references"]
        for metric in ("jevbench_public_accuracy", "transfer_v9_accuracy", "transfer_v9_nll", "transfer_v9_brier", "transfer_v9_ece"):
            page.locator(f'[data-metric="{metric}"]').click()
            accuracy = metric.endswith("accuracy")
            expected = sorted((model[metric] for model in models), reverse=accuracy)
            labels = page.locator(".benchmark-value").all_text_contents()
            assert labels == [f"{value * 100:.2f}%" if accuracy else f"{value:.3f}" for value in expected]
            assert page.locator('[data-metric][aria-pressed="true"]').count() == 1
        page.locator('[data-metric="jevbench_public_accuracy"]').click()
        page.locator(".benchmark-table summary").click()
        assert page.locator(".benchmark-table tbody tr").count() == 9
        assert page.locator(".benchmark-table tbody tr").first.inner_text().endswith("81.08%")
        page.locator(".benchmark-table summary").click()
        model_link = page.locator(".model-group").first.locator('a[href="#family-qwen"]')
        model_link.focus()
        model_link.press("Enter")
        assert page.locator("#family-qwen").get_attribute("open") is not None
        assert page.locator("#family-qwen tbody tr").count() == 9
        page.locator("#family-qwen summary").click()

        page.locator("#motion-toggle").click()
        assert page.locator("#motion-toggle").get_attribute("aria-pressed") == "true"
        assert page.locator("video").evaluate_all("vs => vs.every(v => v.paused)")
        times = page.locator("video").evaluate_all("vs => vs.map(v => v.currentTime)")
        page.wait_for_timeout(500)
        assert times == page.locator("video").evaluate_all("vs => vs.map(v => v.currentTime)")
        transform = page.locator(".model-track").evaluate("e => getComputedStyle(e).transform")
        page.wait_for_timeout(250)
        assert transform == page.locator(".model-track").evaluate("e => getComputedStyle(e).transform")
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

        # Model cards select the corresponding local launch commands.
        for model in results["released_models"]:
            checkpoint = model["repository"]
            page.locator(f'[data-local-model="{checkpoint}"]').click()
            assert page.url.endswith("#get-started")
            assert page.locator("#local-model").input_value() == checkpoint
            commands = page.locator("#quickstart-code").inner_text()
            assert f"--checkpoint {checkpoint}" in commands
            assert "--device cuda --dtype bf16" in commands
            assert "jevany train" not in commands
            assert page.locator("#demo-launch").is_visible()
        page.locator("#local-model").select_option("cpu-starter")
        commands = page.locator("#quickstart-code").inner_text()
        assert "jevany train" in commands and "--weights-dtype fp32" in commands
        assert "--out runs/cpu-jev" in commands and "--checkpoint runs/cpu-jev" in commands
        assert "--device cpu --dtype fp32" in commands
        assert "16 GB" in page.locator("#recipe-note").inner_text()
        assert "--base-url http://127.0.0.1:8008 --text-only" in page.locator("#demo-code").inner_text()
        for value in ("SimpleJev/JevAny-Qwen3.5-4B-LoRA", "SimpleJev/JevAny-Qwen3.8-27B-LoRA"):
            page.locator("#local-model").select_option(value)
            assert f"--checkpoint {value}" in page.locator("#quickstart-code").inner_text()
        page.locator("#local-model").select_option("cpu-starter")

        # Keyboard tabs, actual recipe content, and clipboard contents.
        page.locator("#tab-local").click()
        page.locator("#tab-local").press("ArrowRight")
        assert page.locator("#tab-train").get_attribute("aria-selected") == "true"
        assert "data validate" in page.locator("#quickstart-code").inner_text()
        assert page.locator("#local-model-control").is_hidden()
        assert page.locator("#demo-launch").is_hidden()
        page.locator("#tab-train").press("End")
        assert page.locator("#tab-serve").get_attribute("aria-selected") == "true"
        assert "--device cuda" in page.locator("#quickstart-code").inner_text()
        page.locator("#tab-serve").press("Home")
        page.locator("#copy-code").click()
        assert page.evaluate("navigator.clipboard.readText()") == page.locator("#quickstart-code").inner_text()
        page.locator("#copy-demo").click()
        assert page.evaluate("navigator.clipboard.readText()") == page.locator("#demo-code").inner_text()
        page.evaluate("Object.defineProperty(navigator, 'clipboard', {value: undefined, configurable: true})")
        page.locator("#copy-code").click()
        assert "Commands selected" in page.locator("#copy-status").inner_text()

        # Responsive layout and preview artifacts; keep videos still in screenshots.
        page.evaluate("window.getSelection().removeAllRanges()")
        page.wait_for_function("[...document.querySelectorAll('.copy-label')].every(e => e.textContent === 'Copy commands')")
        page.locator('.replay-choice[data-case="peg_insertion"]').click()
        page.locator("#motion-toggle").click()
        for width, height in ((1440, 1000), (1280, 800), (768, 1024), (640, 900), (540, 900), (390, 844), (320, 700)):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            page.wait_for_timeout(150)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Overflow at {width}"
            assert page.locator("img:not([loading=lazy])").evaluate_all("imgs => imgs.every(i => i.complete && i.naturalWidth > 0)")
            if width == 320:
                for selector in (".api-code", ".terminal pre"):
                    code = page.locator(selector).first
                    assert code.get_attribute("tabindex") == "0"
                    code.focus()
                    code.press("ArrowRight")
                    page.wait_for_timeout(150)
                    assert code.evaluate("el => el.scrollLeft > 0"), f"Keyboard scroll failed: {selector}"
            if width in (1440, 390):
                page.screenshot(path=str(output / f"homepage-{width}.png"), full_page=True)
                page.screenshot(path=str(output / f"hero-{width}.png"))
                page.locator("#benchmarks").screenshot(path=str(output / f"benchmarks-{width}.png"))
                page.locator("#get-started").screenshot(path=str(output / f"local-cpu-{width}.png"))
                page.locator("#local-model").select_option("SimpleJev/JevAny-Qwen3.5-4B-LoRA")
                page.locator("#get-started").screenshot(path=str(output / f"local-gpu-{width}.png"))
                page.locator("#local-model").select_option("cpu-starter")
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#motion-toggle").click()
        page.wait_for_function("document.querySelector('.ambient-video').videoWidth === 360")
        assert page.locator(".ambient-video").get_attribute("src").endswith("/grid-mobile.webm")
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.wait_for_function("document.querySelector('.ambient-video').videoWidth === 720")
        page.locator("#motion-toggle").click()
        page.set_viewport_size({"width": 320, "height": 700})

        # Real site pages, their deep links and static attachments remain on this origin.
        for document in ("quickstart.html#quickstart", "API.html", "TRAINING.html#backbone-support", "EVALUATION.html#transfer-breakdown", "zh.html"):
            page.goto(url + "docs/" + document, wait_until="networkidle")
            assert page.locator("h1").count() == 1
            assert page.locator(".doc-content").is_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), document
        page.goto(url + "docs/API.html", wait_until="networkidle")
        page.screenshot(path=str(output / "docs-320.png"), full_page=True)
        for document in SITE.glob("docs/*.html"):
            response = page.request.get(url + "docs/" + document.name)
            assert response.ok, document
        assert page.request.get(url + "files/reports/JevAny_Tech_Report.pdf").body().startswith(b"%PDF")
        page.goto(url + "docs/examples.html", wait_until="networkidle")
        page.locator(".doc-video").first.evaluate("v => v.play()")
        page.wait_for_function("document.querySelector('.doc-video').currentTime > 0")

        reduced = browser.new_context(viewport={"width": 390, "height": 844}, reduced_motion="reduce")
        reduced.on("page", lambda new_page: new_page.on("pageerror", lambda error: errors.append(str(error))))
        reduced_page = reduced.new_page()
        background_requests: list[str] = []
        reduced_page.on("request", lambda request: background_requests.append(request.url) if "/ambient/" in request.url and request.url.endswith((".mp4", ".webm")) else None)
        reduced_page.goto(url, wait_until="networkidle")
        assert reduced_page.locator("#motion-toggle").get_attribute("aria-pressed") == "true"
        assert reduced_page.locator("video").evaluate_all("vs => vs.every(v => v.paused)")
        assert not background_requests, background_requests
        assert reduced_page.locator(".model-track").evaluate("e => getComputedStyle(e).animationName") == "none"
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
        failed_page.route("**/ambient/*.webm", lambda route: route.abort())
        failed_page.route("**/ambient/*.mp4", lambda route: route.abort())
        failed_page.route("**/assets/data/cases.json", lambda route: route.abort())
        failed_page.goto(url, wait_until="networkidle")
        assert failed_page.locator(".hero h1").is_visible()
        assert failed_page.locator(".ambient-poster").is_visible()
        failed_page.locator("#case-library summary").click()
        assert failed_page.locator('[data-case="warehouse_rover"]').get_attribute("href").endswith(".mp4")
        failed_page.locator('[data-metric="transfer_v9_accuracy"]').click()
        assert failed_page.locator(".benchmark-value").first.inner_text() == "86.04%"
        failed.close()

        # MP4 fallback decodes when VP9 support is not available.
        mp4 = browser.new_context()
        mp4.add_init_script("""const original = HTMLMediaElement.prototype.canPlayType;
            HTMLMediaElement.prototype.canPlayType = function(type) {
                return type.includes('webm') ? '' : original.call(this, type);
            };""")
        mp4_page = mp4.new_page()
        mp4_page.goto(url, wait_until="networkidle")
        mp4_decode_supported = bool(mp4_page.evaluate("""document.createElement('video').canPlayType('video/mp4; codecs="avc1.42E01E"')"""))
        if mp4_decode_supported:
            mp4_page.wait_for_function("[...document.querySelectorAll('.ambient-video')].every(v => v.currentTime > 0)")
        assert mp4_page.locator(".ambient-video").evaluate_all("vs => vs.every(v => v.currentSrc.endsWith('.mp4'))")
        mp4.close()

        plain = browser.new_context(java_script_enabled=False)
        plain_page = plain.new_page()
        plain_page.goto(url, wait_until="networkidle")
        assert plain_page.locator(".hero h1").is_visible()
        assert plain_page.locator(".noscript-note").is_visible()
        assert "jevany demo" in plain_page.locator("#demo-code").inner_text()
        assert "jevany serve --checkpoint runs/cpu-jev" in plain_page.locator("#quickstart-code").inner_text()
        assert plain_page.locator("#local-model-control").is_hidden()
        assert plain_page.locator(".benchmark-value").first.inner_text() == "90.04%"
        plain_page.locator("#case-library summary").click()
        assert plain_page.locator(".case-card").first.is_visible()
        plain_page.goto(url + "docs/API.html", wait_until="networkidle")
        assert "One decision API" in plain_page.locator("h1").inner_text()
        plain.close()
        assert not errors, errors
        result = {
            "status": "passed",
            "viewports": [1440, 1280, 768, 640, 540, 390, 320],
            "checks": ["one preblurred background stream", "responsive background resolution",
                       "scrolling model logos and offscreen pause", "Muse identity", "Hugging Face links",
                       "upright heading font", "Star link without authentication",
                       "30 decoded replays", "five benchmark metrics and correct sorting",
                       "full results table", "model support deep links", "local documentation and PDF", "global pause",
                       "native pause", "CPU/GPU model selection and model card links",
                       "keyboard tabs", "clipboard and fallback",
                       "no overflow", "reduced motion", "save data", "network fallback",
                       "MP4 fallback", "keyboard code scrolling", "no JavaScript fallback"],
            "console_errors": errors,
            "mp4_browser_decode": "passed" if mp4_decode_supported else "not available in this Chromium build; fallback URLs checked",
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
