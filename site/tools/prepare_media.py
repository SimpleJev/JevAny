"""Prepare homepage media from the repository's archived GIFs.

Requires ffmpeg on PATH and Pillow: python -m pip install Pillow
Run from any directory: python site/tools/prepare_media.py
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageOps


SITE = Path(__file__).resolve().parents[1]
SOURCE = SITE.parent / "docs" / "demos" / "cases"
OUTPUT = SITE / "assets" / "media"
BACKGROUNDS = ("peg_insertion", "drone", "lab", "chess", "conveyor_sorting", "warehouse_rover")


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
    encode_webm(destination)


def encode_webm(destination: Path) -> None:
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
    convert(source, "scale=1280:720,setsar=1,fps=20", OUTPUT / f"{source.stem}.mp4")
    with Image.open(source) as image:
        image.seek(min(10, image.n_frames - 1))
        ImageOps.contain(image.convert("RGB"), (1280, 720)).save(
            OUTPUT / f"{source.stem}.webp", quality=82
        )
    print(f"Prepared {source.stem}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-only", action="store_true", help="Regenerate just the two background layouts")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("Install ffmpeg and add it to PATH before preparing media.")
    (OUTPUT / "ambient").mkdir(parents=True, exist_ok=True)
    prepare_background()
    if args.background_only:
        return
    sources = sorted(SOURCE.glob("*.gif"))
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(prepare, sources))
    prepare_documentation()


def prepare_background() -> None:
    """Encode the six scenes as one softly blended 24 fps background."""
    for name, width, height, columns, rows in (
        ("grid", 720, 480, 3, 2),
        ("grid-mobile", 360, 720, 2, 3),
    ):
        tile_width = width // columns
        tile_height = height // rows
        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
        filters = []
        positions = []
        for index, case in enumerate(BACKGROUNDS):
            command += ["-threads", "1", "-ignore_loop", "1", "-stream_loop", "-1", "-i", str(SOURCE / f"{case}.gif")]
            filters.append(
                f"[{index}:v]setpts=2*(PTS-STARTPTS),fps=10,"
                "crop=iw*0.70:ih*0.60:iw*0.28:ih*0.18,"
                f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=increase,"
                f"crop={tile_width}:{tile_height},setsar=1[v{index}]"
            )
            positions.append(f"{index % columns * tile_width}_{index // columns * tile_height}")
        # Blur across scene boundaries, then blend frames offline for smoother motion.
        filters.append(
            "".join(f"[v{i}]" for i in range(6))
            + f"xstack=inputs=6:layout={'|'.join(positions)},gblur=sigma=12,eq=saturation=0.75,"
            "minterpolate=fps=24:mi_mode=blend,trim=duration=24,format=yuv420p[out]"
        )
        destination = OUTPUT / "ambient" / f"{name}.mp4"
        command += [
            "-filter_complex_threads", "2", "-filter_complex", ";".join(filters), "-map", "[out]",
            "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "28",
            "-movflags", "+faststart", "-threads", "2", str(destination),
        ]
        subprocess.run(command, check=True)
        encode_webm(destination)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(destination), "-frames:v", "1", "-q:v", "78", str(destination.with_suffix(".webp"))],
            check=True,
        )
        print(f"Prepared background {name}: {width}×{height}, 24 fps", flush=True)


def prepare_documentation() -> None:
    """Use controllable videos in documentation instead of large animated GIFs."""
    output = OUTPUT / "docs"
    output.mkdir(exist_ok=True)
    for source in sorted(SOURCE.parent.glob("*.gif")):
        if source.stem == "jevany-cases":
            continue
        convert(source, "scale=w='min(1280,iw)':h=-2,setsar=1,fps=20", output / f"{source.stem}.mp4")
        with Image.open(source) as image:
            image.seek(min(10, image.n_frames - 1))
            ImageOps.contain(image.convert("RGB"), (1280, 720)).save(
                output / f"{source.stem}.webp", quality=82)
        print(f"Prepared documentation replay {source.stem}", flush=True)


if __name__ == "__main__":
    main()
