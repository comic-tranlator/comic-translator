import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from src.util.path import ASSETS_DIR

FONTS_DIR = ASSETS_DIR / "fonts"
FONT_REGISTRY_FILE = FONTS_DIR / "__registry__.json"


class TypeFace(StrEnum):
    SERIF = "serif"
    SANS_SERIF = "sans-serif"
    SCRIPT = "script"
    MONOSPACED = "monospaced"


class FontWeight(StrEnum):
    THIN = "thin"
    REGULAR = "regular"
    BOLD = "bold"
    BLACK = "black"


class Posture(StrEnum):
    ROMAN = "roman"
    ITALIC = "italic"


@dataclass
class FontEntry:
    path: Path
    typeface: TypeFace
    weight: FontWeight
    posture: Posture
    langs: list[str]


def load_font_registry() -> dict[str, FontEntry]:
    raw: dict[str, dict] = json.loads(FONT_REGISTRY_FILE.read_text())
    registry: dict[str, FontEntry] = {}
    for name, entry in raw.items():
        registry[name] = FontEntry(
            path=FONTS_DIR / entry["file"],
            typeface=TypeFace(entry["typeface"]),
            weight=FontWeight(entry["weight"]),
            posture=Posture(entry["posture"]),
            langs=entry["langs"],
        )
    return registry
