"""Stated text normalizers for WER (English) and CER (Mandarin) (chapter 9).

A WER or CER number is only meaningful with the normalizer that produced it. Each normalizer here is
a named, ordered list of steps; `describe()` prints the steps so a result file can record them.
The same normalizer is always applied to the reference and to the hypothesis.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

# ---------------------------------------------------------------- steps

_WS = re.compile(r"\s+")
# punctuation: Unicode categories P* and S* (symbols), except the apostrophe inside English words
_APOS_INSIDE = re.compile(r"(?<=[A-Za-z])'(?=[A-Za-z])")


def collapse_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()


_LANG_TAG = re.compile(r"<[a-z]{2,3}(?:-[A-Za-z]{2,4})?>")


def drop_lang_tags(s: str) -> str:
    """Remove language tags such as '<zh-CN>' that Nemotron 3.5 ASR appends after the terminal
    punctuation (its NeMo streaming script calls this option strip_lang_tags)."""
    return _LANG_TAG.sub(" ", s)


def lower(s: str) -> str:
    return s.lower()


def width_nfkc(s: str) -> str:
    """Full-width -> half-width (e.g. 'ＡＢＣ１２３，' -> 'ABC123,'), by Unicode NFKC."""
    return unicodedata.normalize("NFKC", s)


def strip_punct(s: str) -> str:
    """Remove every punctuation/symbol character, keeping apostrophes inside words (don't, it's)."""
    s = _APOS_INSIDE.sub("\x00", s)
    out = []
    for ch in s:
        if ch == "\x00":
            out.append("'")
        elif unicodedata.category(ch)[0] in "PS":
            out.append(" ")
        else:
            out.append(ch)
    return collapse_ws("".join(out))


def remove_spaces(s: str) -> str:
    return _WS.sub("", s)


_t2s = None


def trad_to_simp(s: str) -> str:
    """Traditional -> Simplified Chinese with OpenCC's t2s table (opencc-python-reimplemented)."""
    global _t2s
    if _t2s is None:
        from opencc import OpenCC

        _t2s = OpenCC("t2s")
    return _t2s.convert(s)


def arabic_to_chinese_numerals(s: str) -> str:
    """'2019年' -> '二零一九年', '35岁' -> '三十五岁' (cn2an's an2cn transform).

    AISHELL-1 transcripts write numbers as Chinese characters, so a hypothesis with Arabic digits is
    scored as wrong on every digit unless this step runs.
    """
    if not re.search(r"\d", s):
        return s
    import cn2an

    try:
        return cn2an.transform(s, "an2cn")
    except Exception:  # cn2an raises on some mixed strings; fall back to digit-by-digit reading
        digits = "零一二三四五六七八九"
        return re.sub(r"\d", lambda m: digits[int(m.group())], s)


# ---------------------------------------------------------------- normalizers

class Normalizer:
    def __init__(self, name: str, steps: list[Callable[[str], str]]):
        self.name = name
        self.steps = steps

    def __call__(self, s: str) -> str:
        for f in self.steps:
            s = f(s)
        return collapse_ws(s)

    def describe(self) -> dict:
        return {"name": self.name, "steps": [f.__name__ for f in self.steps]}


EN = {
    "raw": Normalizer("en-raw", [collapse_ws]),
    "lower": Normalizer("en-lower", [lower]),
    "lower+punct": Normalizer("en-lower+punct", [drop_lang_tags, width_nfkc, lower, strip_punct]),
}
EN_DEFAULT = EN["lower+punct"]

ZH = {
    "raw": Normalizer("zh-raw", [collapse_ws]),
    "nospace": Normalizer("zh-nospace", [remove_spaces]),
    "nospace+punct": Normalizer("zh-nospace+punct", [strip_punct, remove_spaces]),
    "nospace+punct+tags": Normalizer("zh-nospace+punct+tags", [drop_lang_tags, strip_punct, remove_spaces]),
    "full": Normalizer("zh-full", [drop_lang_tags, width_nfkc, trad_to_simp, lower, arabic_to_chinese_numerals,
                                   strip_punct, remove_spaces]),
}
ZH_DEFAULT = ZH["full"]


# ---------------------------------------------------------------- tokenization for scoring

_CJK = r"㐀-䶿一-鿿豈-﫿"
_TOKEN_ZH = re.compile(rf"[{_CJK}]|[A-Za-z0-9']+|\S")


def words(s: str) -> list[str]:
    """English scoring units: whitespace-separated words."""
    return s.split()


def zh_units(s: str) -> list[str]:
    """Mandarin scoring units: one token per CJK character, one per embedded Latin word or number,
    one per any other visible character. Whitespace is never a unit."""
    return _TOKEN_ZH.findall(s)


def zh_units_with_spaces(s: str) -> list[str]:
    """What a naive character scorer does: every character, including spaces, is a unit."""
    return list(s)
