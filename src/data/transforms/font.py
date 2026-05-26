from typing import Any

from src.util.font import FontWeight, Posture, TypeFace

TYPEFACE2ID = {
    TypeFace.SERIF: 0,
    TypeFace.SANS_SERIF: 1,
    TypeFace.MONOSPACED: 2,
    TypeFace.SCRIPT: 3,
}

POSTURE2ID = {
    Posture.ROMAN: 0,
    Posture.ITALIC: 1,
}

FONT_WEIGHT2NUM = {
    FontWeight.THIN: 100,
    FontWeight.REGULAR: 300,
    FontWeight.BOLD: 600,
    FontWeight.BLACK: 900,
}


class PrepareFontAttributes:
    def __call__(self, x: dict[str, Any]) -> dict[str, Any]:
        x["typeface"] = TYPEFACE2ID[x["typeface"]]
        x["posture"] = POSTURE2ID[x["posture"]]
        x["weight"] = FONT_WEIGHT2NUM[x["weight"]]

        return x
