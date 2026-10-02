# JevAny homepage

The homepage is a static site, published at https://simplejev.github.io/JevAny/.
It has no runtime dependencies or third-party requests.

Preview from the repository root:

```bash
python3 -m http.server 4173 --bind 127.0.0.1 --directory site
```

Open http://127.0.0.1:4173. For browser checks:

```bash
python3 -m pip install -r site/tests/requirements.txt
python3 -m playwright install chromium
python3 site/tests/content_checks.py
python3 site/tools/build_content.py
python3 site/tests/check_site.py
python3 site/tests/browser_checks.py
```

The homepage includes interactive benchmark results, all 30 archived replays,
released checkpoints and the supported-model catalog. Guides, the technical
report and linked reference files are hosted on the site. GitHub remains an
explicit source-code link; checkpoint download links point to Hugging Face.

The README and `agent-harness.html` use the same paired comparisons from
`scripts/render_agent_decision_demos_v2.py`. Three individual GIFs show WebShop,
FrozenLake, and SQLite recovery. `docs/demos/jev-agent-harness.gif` joins those
replays in that order into a nine-second overview, preserving every frame and
its timing. The gallery uses matching MP4/WebM videos and WebP posters.

Both lanes retain the measured completion-time ratio within each three-second loop.
Recorded LLM call totals are prominent alongside each lane's clock. Task
constraints stay visible; large actor panels identify the current phase,
and recorded menus highlight the selected option. The footer shows LLM tokens
saved, wall time saved, and the task result for this recorded pair. Large
highlighted percentages lead the two savings cards, with absolute savings and
original totals alongside them. Percentages use the LLM-only baseline. These
are full-run measurements for one recorded pair.
A fixed role bar shows each controller once and highlights only
the current controller. When control returns to the LLM, the same LLM position
lights up again. Both roles dim when the task completes, and completed lanes
remain frozen.

Shared robot portraits in `scripts/render_agent_robot_icons.py` distinguish the
blue LLM from the smaller teal Jev. Planning lights and short motion trails
animate only on the active controller; inactive portraits stay gray and still.
These decorative animations do not represent measured inference time. Amber
result symbols mark completion. Icons are drawn and antialiased locally with
Pillow, so all three cases use the same artwork without external image assets.

`agent-harness.html` leads with the overview, followed by the three individual
replays and a short explanation of each delegation: choosing within LLM-provided
menus, retaining one plan across local moves, and choosing useful evidence.
It uses the same navy, blue, teal, and amber palette as the GIFs.
Playback controls provide pause, seek, and fullscreen beneath the frame,
keeping the savings visible even when paused. Native controls remain available
when JavaScript is disabled. Only the overview
starts automatically; videos pause offscreen and when the tab is hidden, and
only one plays at a time. Reduced-motion preferences disable automatic playback.
A manual pause or selection takes precedence over automatic playback.

The gallery links to the recorded traces, delegation protocol, and full results.
WebShop uses seed 3107 and Jev 4B from the D4 supplement; SQLite and FrozenLake
keep their recorded cases. Savings describe one recorded pair, with the broader
D2 matrix, separate D4 supplement, fallback cases, and uncertainty documented
in the report. Rendering uses existing recordings and runs no model or environment.

Regenerate all four GIFs and their site media from the repository root:

```bash
python3 -m pip install pillow imageio-ffmpeg
python3 scripts/render_agent_decision_demos_v2.py --site-media
```

Omit `--site-media` to export only GIFs. Video conversion uses ffmpeg;
imageio-ffmpeg is optional when ffmpeg is already on PATH. The legacy command
`python3 scripts/render_agent_harness_demo.py` also exports this presentation.

Regenerate the homepage's data sections and documentation from the repository:

```bash
python3 -m pip install -r site/tools/requirements.txt
python3 site/tools/build_content.py
```

This reads `results/model-family-v2.json`, `docs/supported-models.json`, and the
existing Markdown guides. It updates the marked sections of `index.html`,
`assets/data`, `docs`, and `files`. Edit the source guides rather than generated
pages. GitHub Actions rebuilds and checks the site before publishing changes to
`main`, uploading only the public site files.

README navigation links point to the published website. Keep guide content in
the source Markdown; the builder converts public URLs back to relative site
links and includes their source documents and downloads in each build.
The link checker verifies README destinations and section anchors against the
generated pages, so local previews do not need the live website.

The header switches between English (`index.html`) and Simplified Chinese
(`zh.html`), preserving the current section. The root URL uses the saved choice,
then the browser language; explicit page URLs always take precedence. Both pages
remain readable and linked without JavaScript. Docs opens the corresponding
quickstart, with English-only reference guides identified on the Chinese page.

Edit English homepage copy in `index.html` and Chinese translations in
`locales/zh-CN.json`, then run the content builder. It generates `zh.html` and
`assets/i18n/zh-CN.js` from the same catalog, including interactive messages.
Missing homepage translations stop the build. Keep code, model identifiers and
benchmark values unchanged; mark display-only identifiers with `translate="no"`.

Thirty replay videos and the six scenes in the background come from
`docs/demos/cases`. The scenes are joined without gutters and blurred together,
with blended intermediate frames encoded into one 24 fps video. Landscape screens
load a 720×480 video; tall portrait screens load a 360×720 version. Only one background stream plays, and
the browser does not apply a live blur filter. Model-family logos scroll across
the top and pause outside the viewport. The motion toggle pauses both the logo strip and videos;
reduced-motion and data-saving preferences disable automatic playback.
Individual replays also have native video controls.
Each featured case plays twice before advancing to the next. The selected
case's highlight fills over both plays, following the video's actual progress.
Pausing or leaving the player preserves that progress. Selecting any case
disables automatic switching for the rest of the page visit and loops that
recording instead; its highlight stays full to mark the selection.

Regenerate media with `python3 site/tools/prepare_media.py` after installing
ffmpeg and Pillow. Use `--background-only` to regenerate just the two background
layouts. Foreground videos preserve the source timing; background scenes play
at half speed. The generator creates VP9 WebM and H.264
MP4 files with static WebP posters, including videos for the documentation.

Brand marks reuse the project's existing assets. Font licenses live in
`assets/fonts`; model-logo licenses and provenance live in `assets/model-logos`.
DM Sans and IBM Plex Sans Condensed are self-hosted. The latter uses IBM's
original outlines with a Latin character subset.

The prominent Star link opens the GitHub repository. Visitors confirm the star
on GitHub; the website does not request GitHub credentials or tokens.

Design references: [Supabase](https://supabase.com/) for the path from product
purpose to examples; [Ollama](https://ollama.com/) for a concise introduction and
quickstart; [vLLM](https://vllm.ai/) for visible model support and documentation.
The page uses JevAny's own brand and recordings.
