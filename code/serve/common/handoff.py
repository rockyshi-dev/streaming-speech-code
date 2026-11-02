"""The LLM -> TTS text handoff: sentence aggregation of streamed tokens (chapter 18).

LLM tokens arrive a few characters at a time. A TTS that takes whole requests needs complete sentences, so the
aggregator buffers text and releases a segment when it sees a sentence end: one of . ! ? ; (followed by a space
or end of stream) or one of 。！？； (no space needed in Mandarin). Optionally, the first segment may be cut
earlier at a comma (， , 、) once it has `first_min_chars` characters, which trades prosody for an earlier
first audio. Markdown symbols an LLM might emit (* # `) are dropped because a TTS would read them.
"""
from __future__ import annotations

import re

_END_ZH = "。！？；"
_END_EN = ".!?;"
_COMMA = "，,、"
_STRIP = re.compile(r"[*#`]+")


class SentenceAggregator:
    def __init__(self, first_min_chars: int | None = None, min_chars: int = 2):
        self.buf = ""
        self.first_min = first_min_chars
        self.min_chars = min_chars
        self.n_out = 0

    def _cut(self, final: bool) -> int | None:
        b = self.buf
        for i, ch in enumerate(b):
            if ch in _END_ZH:
                return i + 1
            if ch in _END_EN:
                nxt = b[i + 1] if i + 1 < len(b) else None
                if nxt is None and final:
                    return i + 1
                if nxt is not None and nxt.isspace():
                    return i + 1
        if self.n_out == 0 and self.first_min is not None and len(b) >= self.first_min:
            for i, ch in enumerate(b):
                if ch in _COMMA and i + 1 >= self.first_min:
                    return i + 1
        return None

    def feed(self, text: str) -> list[str]:
        """Add streamed text; return the segments that are now complete."""
        self.buf += _STRIP.sub("", text)
        out = []
        while (k := self._cut(final=False)) is not None:
            seg, self.buf = self.buf[:k].strip(), self.buf[k:]
            if len(seg) >= self.min_chars:
                out.append(seg)
                self.n_out += 1
        return out

    def flush(self) -> list[str]:
        """End of the LLM stream: return what is left (possibly a sentence without a final mark)."""
        out = []
        while (k := self._cut(final=True)) is not None:
            seg, self.buf = self.buf[:k].strip(), self.buf[k:]
            if len(seg) >= self.min_chars:
                out.append(seg)
        rest = self.buf.strip()
        self.buf = ""
        if len(rest) >= self.min_chars:
            out.append(rest)
        self.n_out += len(out)
        return out
