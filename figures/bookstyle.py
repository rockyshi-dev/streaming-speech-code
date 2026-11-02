"""The handbook's colour roles. Plot scripts do:

    from figures.bookstyle import setup, figure, save, patch, synthetic
    C, R = setup()          # C: theme colours; R: the book's colour roles (defined in setup() below)
    ax.barh(y, w, color=R.lat.endpoint)

The plotting helpers and colours come from figures/_fallback/figkit.py (plain matplotlib).
"""
from types import SimpleNamespace

import os

if os.environ.get("NO_FIGKIT"):
    from figures._fallback import figkit as fk
else:
    try:
        import figkit as fk
    except ImportError:  # the bundled helpers; an installed module with the same API is used when present
        from figures._fallback import figkit as fk

figure, save, patch, synthetic = fk.figure, fk.save, fk.patch, fk.synthetic


def setup():
    C = fk.setup()
    s = C.series
    roles = SimpleNamespace(
        focus=C.focus,
        past=C.past, current=C.current, lookahead=C.lookahead, future=C.future,
        lat=SimpleNamespace(lookahead=C.lookahead, compute=s[1], emission=s[2], endpoint=C.focus,
                            transport=C.muted),
        sys=SimpleNamespace(asr=s[0], llm=s[4], tts=s[2]),
        base=C.base, muted=C.muted, ink=C.ink)
    return C, roles
