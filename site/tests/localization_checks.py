"""Check language selection, translated interactions and navigation in Chromium."""

import json
from pathlib import Path

from playwright.sync_api import Browser, expect


SITE = Path(__file__).resolve().parents[1]


def check_localization(browser: Browser, url: str, output: Path) -> None:
    messages = json.loads((SITE / "locales/zh-CN.json").read_text())
    cases = json.loads((SITE / "assets/data/cases.json").read_text())
    errors = []
    context = browser.new_context(
        locale="zh-CN", reduced_motion="reduce",
        permissions=["clipboard-read", "clipboard-write"],
    )
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))

    # Browser preference applies at the root; explicit URLs override saved choices.
    page.goto(url + "?source=test#models", wait_until="networkidle")
    assert page.url == url + "zh.html?source=test#models"
    assert page.locator("html").get_attribute("lang") == "zh-CN"
    assert page.title() == messages["JevAny: Train and deploy System 1 decision models"]
    assert page.locator('meta[name="description"]').get_attribute("content").startswith("使用 JevAny")
    assert page.locator('link[rel="canonical"]').get_attribute("href").endswith("/zh.html")
    assert page.locator('.language-switch a[aria-current="page"]').inner_text() == "中文"
    assert page.locator('.language-switch a[lang="en"]').get_attribute("href") == "index.html?source=test#models"
    page.locator('.language-switch a[lang="en"]').focus()
    page.keyboard.press("Enter")
    page.wait_for_url(url + "index.html?source=test#models")
    assert page.locator("html").get_attribute("lang") == "en"
    page.goto(url, wait_until="networkidle")
    assert page.locator("html").get_attribute("lang") == "en"
    page.locator('.language-switch a[lang="zh-CN"]').click()
    page.wait_for_url(url + "zh.html")
    page.reload(wait_until="networkidle")
    assert page.locator("html").get_attribute("lang") == "zh-CN"
    page.goto(url, wait_until="networkidle")
    assert page.url == url + "zh.html"

    # The highlighted link opens the matching existing docs, with a way home.
    assert page.locator(".docs-link").get_attribute("href") == "docs/zh.html"
    page.locator(".docs-link").click()
    page.wait_for_url(url + "docs/zh.html")
    assert page.locator(".docs-sidebar > .text-link").inner_text() == "← 项目主页"
    page.locator(".site-header .brand").click()
    page.wait_for_url(url + "zh.html")
    page.locator('.doc-cards a[href="docs/zh.html#本地体验"]').click()
    assert page.locator('a[name="本地体验"]').count() == 1
    page.goto(url + "zh.html", wait_until="networkidle")
    assert page.locator('.doc-cards a[href="docs/API.html"]').get_attribute("hreflang") == "en"
    assert "英文" in page.locator(".documentation-note").inner_text()

    page.locator("#case-library summary").click()
    for case in cases:
        card = page.locator(f'.case-card[data-case="{case["id"]}"]')
        assert messages[case["title"]] in card.locator(".case-card-title").inner_text()
        assert card.locator(".case-card-description").inner_text() == messages[case["description"]]
    page.locator('.case-card[data-case="warehouse_rover"]').click()
    assert page.locator("#replay-description").inner_text() == messages[next(c["description"] for c in cases if c["id"] == "warehouse_rover")]
    assert page.locator("#replay-video").get_attribute("aria-label") == "JevAny 历史回放：仓库导航"
    page.locator("#case-library summary").click()

    for metric, title in (
        ("jevbench_public_accuracy", "JevBench 准确率"),
        ("transfer_v9_accuracy", "Transfer 准确率"),
        ("transfer_v9_nll", "负对数似然"),
        ("transfer_v9_brier", "Brier 分数"),
        ("transfer_v9_ece", "期望校准误差"),
    ):
        page.locator(f'[data-metric="{metric}"]').click()
        assert page.locator("#chart-title").inner_text() == title
        direction = "越高越好" if metric.endswith("accuracy") else "越低越好"
        assert direction in page.locator("#chart-description").inner_text()
        assert "已按" in page.locator("#metric-status").inner_text()
    page.locator('[data-metric="jevbench_public_accuracy"]').click()
    assert page.locator(".benchmark-value").first.inner_text() == "90.04%"
    assert page.locator("#motion-toggle").inner_text().endswith("继续动画")

    # Machine-readable commands, checkpoint IDs and values stay identical.
    english = context.new_page()
    english.goto(url + "index.html", wait_until="networkidle")
    for recipe in ("local", "train", "serve"):
        page.locator(f"#tab-{recipe}").click()
        english.locator(f"#tab-{recipe}").click()
        assert page.locator("#quickstart-code").inner_text() == english.locator("#quickstart-code").inner_text()
        assert " · " in page.locator("#recipe-note").inner_text()
        expected_note = "终端" if recipe != "local" else "内存"
        assert expected_note in page.locator("#recipe-note").inner_text()
    english.close()
    page.bring_to_front()
    page.locator("#tab-local").click()
    page.locator("#local-model").select_option("SimpleJev/JevAny-Qwen3.8-27B-LoRA")
    assert "54 GB 显存" in page.locator("#recipe-note").inner_text()
    page.locator("#copy-code").click()
    expect(page.locator("#copy-code .copy-label")).to_have_text("已复制")
    assert page.evaluate("navigator.clipboard.readText()") == page.locator("#quickstart-code").inner_text()
    page.evaluate("Object.defineProperty(navigator, 'clipboard', {value: undefined, configurable: true})")
    page.locator("#copy-demo").click()
    expect(page.locator("#copy-status")).to_contain_text("已选中命令")
    page.evaluate("window.getSelection().removeAllRanges()")
    page.locator("#local-model").select_option("cpu-starter")

    for language, filename in (("en", "index.html"), ("zh-CN", "zh.html")):
        page.goto(url + filename, wait_until="networkidle")
        for width, height in ((1440, 1000), (1280, 800), (1024, 900), (800, 900), (768, 1024), (640, 900), (540, 900), (390, 844), (320, 700)):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (language, width)
            assert page.locator(".docs-link").is_visible()
            assert page.locator(".language-switch").is_visible()
            assert page.locator(".docs-link").evaluate("e => e.getBoundingClientRect().right <= innerWidth")
            assert page.locator(".language-switch").evaluate(
                "e => e.getBoundingClientRect().right <= document.querySelector('.docs-link').getBoundingClientRect().left"
            )
            if width in (1440, 390, 320):
                page.screenshot(path=str(output / f"locale-{language}-{width}.png"))
                if language == "zh-CN" and width != 320:
                    page.locator("#get-started").screenshot(path=str(output / f"localized-quickstart-{width}.png"))
                    page.locator("#documentation").screenshot(path=str(output / f"localized-docs-{width}.png"))
        if language == "en":
            page.locator(".docs-link").click()
            page.wait_for_url(url + "docs/quickstart.html")
    context.close()

    failed_catalog = browser.new_context(locale="zh-CN", reduced_motion="reduce")
    failed_page = failed_catalog.new_page()
    failed_page.on("pageerror", lambda error: errors.append(str(error)))
    failed_page.route("**/assets/data/cases.json", lambda route: route.abort())
    failed_page.goto(url + "zh.html", wait_until="networkidle")
    failed_page.locator('.replay-choice[data-case="drone"]').click()
    assert failed_page.locator("#replay-video").get_attribute("aria-label") == "JevAny 历史回放：无人机配送"
    assert failed_page.locator("#replay-description").inner_text() == messages["Plan the route, land, and deliver with energy left in reserve."]
    failed_catalog.close()

    # Invalid preferences and denied storage must not trap visitors in redirects.
    for locale, setup, expected in (
        ("en-US", "localStorage.setItem('jevany-language', 'invalid')", "en"),
        ("zh-TW", "localStorage.setItem('jevany-language', 'invalid')", "zh-CN"),
        ("fr-FR", "", "en"),
        ("zh-CN", "Object.defineProperty(window, 'localStorage', {get() {throw new Error('Storage denied')}})", "zh-CN"),
    ):
        fallback = browser.new_context(locale=locale, reduced_motion="reduce")
        if setup:
            fallback.add_init_script(setup)
        fallback_page = fallback.new_page()
        fallback_page.on("pageerror", lambda error: errors.append(str(error)))
        fallback_page.goto(url, wait_until="networkidle")
        assert fallback_page.locator("html").get_attribute("lang") == expected
        fallback_page.locator('.language-switch a[lang="en"]').click()
        fallback_page.wait_for_url(url + "index.html")
        assert fallback_page.locator("html").get_attribute("lang") == "en"
        fallback.close()

    # Static translations, navigation and disclosures work without JavaScript.
    plain = browser.new_context(java_script_enabled=False, viewport={"width": 390, "height": 844})
    plain_page = plain.new_page()
    plain_page.goto(url)
    plain_page.locator('.language-switch a[lang="zh-CN"]').click()
    assert plain_page.locator("html").get_attribute("lang") == "zh-CN"
    assert plain_page.locator("#hero-title").inner_text() == "任意模型。\n你的 下一步。"
    assert plain_page.locator(".noscript-note").is_visible()
    assert plain_page.locator("#recipe-note").inner_text().startswith("CPU · 建议")
    plain_page.locator("#case-library summary").click()
    assert "机器人装配" in plain_page.locator(".case-card").first.inner_text()
    assert plain_page.locator(".docs-link").get_attribute("href") == "docs/zh.html"
    plain_page.locator('.language-switch a[lang="en"]').click()
    assert plain_page.locator("html").get_attribute("lang") == "en"
    plain.close()
    assert not errors, errors
    print("PASS: Chinese/English navigation, persistence, localized interactions, 9 viewport widths, storage and no-JS fallbacks.")
