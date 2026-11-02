"""Sentence splitting and simulated LLM text streaming.

An LLM sends text as tokens. Here a "text unit" stands in for a token, so that runs do
not depend on any one tokenizer:

    English  - one word together with the space and punctuation that follow it
    Mandarin - one character (CJK ideograph), with any following punctuation attached

Units are released at a fixed rate (units per second) starting at t = 0.
"""

from __future__ import annotations

import re

_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_ZH_END = "。！？"
_EN_END = ".!?"


def is_mandarin(text: str) -> bool:
    return re.search(f"[{_CJK}]", text) is not None


def split_sentences(text: str) -> list[str]:
    """Split after sentence-final punctuation (. ! ? and the full-width 。！？).

    English splits only where the mark is followed by whitespace, so "3.5" and "e.g."
    inside a word stay whole. Empty pieces are dropped; whitespace is stripped.
    """
    text = text.strip()
    if not text:
        return []
    if is_mandarin(text):
        parts = re.split(f"(?<=[{_ZH_END}])", text)
    else:
        parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def text_units(text: str) -> list[str]:
    """Cut text into units; "".join(units) == text."""
    if is_mandarin(text):
        units: list[str] = []
        for ch in text:
            if units and not re.match(f"[{_CJK}A-Za-z0-9]", ch):
                units[-1] += ch  # punctuation / space rides on the previous unit
            else:
                units.append(ch)
        return units
    units = re.findall(r"\S+\s*", text)
    lead = text[: len(text) - len(text.lstrip())]
    if lead and units:
        units[0] = lead + units[0]
    return units


def schedule(units: list[str], rate_per_s: float | None) -> list[tuple[float, str]]:
    """Release times for each unit. rate None or <= 0 means all at t = 0 (one piece)."""
    if not units:
        return []
    if not rate_per_s or rate_per_s <= 0:
        return [(0.0, "".join(units))]
    return [(i / rate_per_s, u) for i, u in enumerate(units)]


def first_sentence_complete_s(text: str, rate_per_s: float) -> float:
    """When the last unit of the first sentence is released (the earliest moment a
    sentence-level TTS can start)."""
    first = split_sentences(text)[0]
    units = text_units(text)
    acc = ""
    for i, u in enumerate(units):
        acc += u
        if len(acc.strip()) >= len(first):
            return i / rate_per_s
    return (len(units) - 1) / rate_per_s
