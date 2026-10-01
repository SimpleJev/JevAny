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
python3 site/tests/check_site.py
python3 site/tests/browser_checks.py
```

GitHub Actions checks the site before publishing changes to `main`. The workflow
uploads only the public HTML, CSS, JavaScript, metadata and assets.

The six featured videos and 30 blurred backgrounds come from the archived
`docs/demos/cases` GIFs. Regenerate them with `tools/prepare_media.py` after
installing ffmpeg and Pillow. Foreground videos preserve the source timing;
backgrounds use a blurred scene crop at half speed. The generator creates
VP9 WebM and H.264 MP4 files; the page chooses a supported format.
Native video controls and
the page's motion toggle pause playback. Reduced-motion and data-saving
preferences disable automatic playback.

Brand marks reuse the project's existing assets. Font licenses live in
`assets/fonts`; model-logo licenses and provenance live in `assets/model-logos`.
DM Sans and Instrument Serif are self-hosted from Google Fonts.

Design references: [Supabase](https://supabase.com/) for the path from product
purpose to examples; [Ollama](https://ollama.com/) for a concise introduction and
quickstart; [vLLM](https://vllm.ai/) for visible model support and documentation.
The page uses JevAny's own brand and recordings.
