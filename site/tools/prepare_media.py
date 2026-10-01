"""Prepare homepage media from the repository's archived GIFs.

Requires ffmpeg on PATH and Pillow: python -m pip install Pillow
Run from any directory: python site/tools/prepare_media.py
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageFilter, ImageOps


SITE = Path(__file__).resolve().parents[1]
SOURCE = SITE.parent / "docs" / "demos" / "cases"
OUTPUT = SITE / "assets" / "media"
FEATURED = ("peg_insertion", "drone", "lab", "frontend", "sql", "chess")


def convert(source: Path, filters: str, destination: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-threads", "2", "-i", str(source), "-an", "-vf", filters,
            "-c:v", "libx264", "-preset", "slow", "-crf", "26",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-threads", "2", str(destination),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-threads", "2", "-i", str(destination), "-an",
            "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36",
            "-row-mt", "1", "-cpu-used", "2", "-threads", "2",
            str(destination.with_suffix(".webm")),
        ],
        check=True,
    )


def prepare(source: Path) -> None:
    convert(
        source,
        "crop=iw*0.73:ih*0.66:iw*0.27:ih*0.20,"
        "scale=384:216,setsar=1,fps=10,gblur=sigma=12,setpts=2*PTS",
        OUTPUT / "ambient" / f"{source.stem}.mp4",
    )
    if source.stem in FEATURED:
        convert(
            source, "scale=1280:720,setsar=1,fps=20",
            OUTPUT / f"{source.stem}.mp4",
        )
        with Image.open(source) as image:
            image.seek(min(10, image.n_frames - 1))
            ImageOps.contain(image.convert("RGB"), (1280, 720)).save(
                OUTPUT / f"{source.stem}.webp", quality=82
            )
    print(f"Prepared {source.stem}", flush=True)


def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("Install ffmpeg and add it to PATH before preparing media.")
    (OUTPUT / "ambient").mkdir(parents=True, exist_ok=True)
    sources = sorted(SOURCE.glob("*.gif"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(prepare, sources))
    # Start with different scene palettes before visiting the remaining cases.
    first = ["lab", "chess", "satellite", "peg_insertion", "frontend", "drone"]
    names = first + [p.stem for p in sources if p.stem not in first]
    (OUTPUT / "backgrounds.json").write_text(json.dumps(names, indent=2) + "\n")
    with Image.open(SOURCE / "lab.gif") as image:
        image.seek(min(20, image.n_frames - 1))
        crop = image.convert("RGB").crop((432, 180, 1600, 774))
        ImageOps.fit(crop, (768, 432)).filter(ImageFilter.GaussianBlur(24)).save(
            OUTPUT / "ambient.webp", quality=75
        )


if __name__ == "__main__":
    main()
