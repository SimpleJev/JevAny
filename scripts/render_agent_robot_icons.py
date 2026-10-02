"""Small, deterministic robot portraits for the recorded agent demos."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageColor, ImageDraw

from render_agent_harness_demo import ACCENT, AMBER, BLUE


PORTRAIT_STYLES = {
    "LLM": ("#c3dcff", "#4674a9", "#10233f"),
    "Jev": ("#a9f0df", "#237768", "#092d2b"),
    "idle": ("#43516a", "#334158", "#121b2c"),
}
IDLE_EDGE, EYE_LIGHT, IDLE_LIGHT = "#52617a", "#f1fbff", "#8290a6"
ROBOT_COLORS = tuple(color for style in PORTRAIT_STYLES.values() for color in style) + (
    IDLE_EDGE, EYE_LIGHT, IDLE_LIGHT,
)


@lru_cache(maxsize=128)
def actor_icon(actor: str, size: int, active: bool = True, phase: int = 0) -> Image.Image:
    """Return an antialiased portrait; motion is decorative, not measured time."""
    image = Image.new("RGBA", (size * 3, size * 3))
    draw = ImageDraw.Draw(image)
    scale = size * 3 / 100
    accent = BLUE if actor == "LLM" else ACCENT if actor == "Jev" else AMBER
    edge = accent if active else IDLE_EDGE
    shell, joint, face = PORTRAIT_STYLES[
        "idle" if not active else "LLM" if actor == "LLM" else "Jev"
    ]
    light = EYE_LIGHT if active else IDLE_LIGHT
    phase = phase % 6 if active else 0

    def coords(values: tuple | list) -> tuple:
        return tuple(round(value * scale) for value in values)

    def rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
        return (*ImageColor.getrgb(color), alpha)

    def line(points: tuple | list, fill: str | tuple, width: float = 2) -> None:
        draw.line(coords(points), fill=fill, width=max(1, round(width * scale)),
                  joint="curve")

    def rect(box: tuple, radius: float, fill: str | tuple,
             outline: str | tuple | None = None, width: float = 1.5) -> None:
        draw.rounded_rectangle(coords(box), round(radius * scale), fill=fill,
                               outline=outline, width=max(1, round(width * scale)))

    def ellipse(box: tuple, fill: str | tuple, outline: str | tuple | None = None,
                width: float = 1) -> None:
        draw.ellipse(coords(box), fill=fill, outline=outline,
                     width=max(1, round(width * scale)))

    def polygon(points: list[tuple], fill: str | tuple) -> None:
        draw.polygon([coords(point) for point in points], fill=fill)

    if active:
        ellipse((7, 13, 95, 98), rgba(accent, 13))
        ellipse((15, 22, 89, 94), rgba(accent, 14))
    ellipse((24, 88, 80, 95), rgba(accent if active else edge, 35))

    if actor == "LLM":
        # Broad shell, stable stance, and three small planning lights.
        rect((30, 68, 73, 85), 8, joint, edge)
        rect((32, 82, 45, 90), 4, shell, edge)
        rect((59, 82, 72, 90), 4, shell, edge)
        rect((20, 69, 30, 80), 4, joint, edge)
        rect((74, 69, 84, 80), 4, joint, edge)
        rect((11, 42, 21, 59), 4, joint, edge)
        rect((82, 42, 92, 59), 4, joint, edge)
        line((51, 18, 51, 28), edge, 3)
        ellipse((47, 11, 55, 19), shell, edge)
        rect((19, 26, 84, 73), 15, shell, edge, 2)
        rect((26, 36, 77, 65), 10, face)
        line((36, 31, 65, 31), rgba(light, 190), 2)
        rect((36, 44, 42, 55), 3, edge)
        rect((61, 44, 67, 55), 3, edge)
        ellipse((37, 45, 40, 48), light)
        ellipse((62, 45, 65, 48), light)
        rect((46, 59, 57, 61), 1, rgba(edge, 150))
        for index in range(3):
            left = 42 + index * 8
            ellipse((left, 76, left + 4, 80), edge if active else joint)
        if active:
            line((66, 20, 75, 11, 87, 17), rgba(accent, 100), 1)
            for index, (cx, cy) in enumerate(((66, 20), (75, 11), (87, 17))):
                focused = index == phase // 2
                radius = 3 if focused else 2
                ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                        light if focused else accent)
    elif actor == "Jev":
        # Compact shell, forward stance, and short moving trails.
        if active:
            for index, y in enumerate((43, 55, 67)):
                shift = (phase * 3 + index * 4) % 10
                line((3 + shift, y, 23 + shift // 2, y),
                     rgba(accent, 115 + index * 45), 2)
        rect((42, 66, 77, 80), 7, joint, edge)
        polygon([(43, 77), (53, 79), (44, 88), (31, 88)], shell)
        polygon([(66, 78), (75, 75), (87, 86), (77, 89)], shell)
        line((34, 89, 44, 89), edge, 2)
        line((77, 89, 87, 86), edge, 2)
        rect((27, 59, 39, 68), 4, joint, edge)
        rect((82, 56, 92, 64), 4, joint, edge)
        rect((32, 31, 87, 69), 12, shell, edge, 2)
        line((45, 35, 74, 32), rgba(light, 190), 2)
        rect((40, 42, 83, 62), 8, face)
        rect((48, 47, 54, 55), 3, edge)
        rect((68, 46, 74, 54), 3, edge)
        ellipse((49, 47, 52, 49), light)
        ellipse((69, 46, 72, 48), light)
        polygon([(65, 12), (54, 29), (63, 28), (59, 39),
                 (77, 20), (67, 21), (73, 12)], edge)
        polygon([(60, 69), (54, 76), (59, 76), (56, 81),
                 (66, 73), (61, 73)], light)
    else:
        # Environment feedback and verification share the amber result symbol.
        polygon([(50, 17), (82, 29), (78, 60), (67, 78), (50, 90),
                 (31, 78), (21, 60), (18, 29)], rgba(accent, 40))
        line((50, 17, 82, 29, 78, 60, 67, 78, 50, 90,
              31, 78, 21, 60, 18, 29, 50, 17), edge, 2)
        line((33, 51, 46, 64, 69, 38), light, 6)

    return image.resize((size, size), Image.Resampling.LANCZOS)
