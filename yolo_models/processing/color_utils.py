from typing import Tuple
import re

HEX_REGEX = re.compile("^#[0-9a-f]{6}$", re.IGNORECASE)


def check_regex_color(hex_color: str):
    if HEX_REGEX.match(hex_color) is None:
        raise ValueError(f"Incorrect hex color: '{hex_color}'")


def rgb_to_hex(rgb: Tuple[int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex_color: str) -> Tuple[int]:
    check_regex_color(hex_color)
    return tuple(int(hex_color[i: i + 2], 16) for i in range(1, len(hex_color), 2))
