"""The bundled plotting helpers: figure widths, theme colours (palette.json, the mono-accent theme) and SVG
output, on plain matplotlib."""
import json
import os
import pathlib
from types import SimpleNamespace

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

_T = json.loads((pathlib.Path(__file__).parent / "palette.json").read_text())
WIDTH_IN = {"column": 7.0, "wide": 9.0}


def setup(theme=None):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#E6E6E6", "axes.axisbelow": True})
    return SimpleNamespace(name="fallback", focus=_T["focus"], base=_T["base"], series=list(_T["series"]),
                           ink=_T["ink"], muted=_T["muted"], paper="#FFFFFF", line=_T["cellLine"],
                           past=_T["past"], current=_T["current"], lookahead=_T["lookahead"],
                           future=_T["future"], seq=LinearSegmentedColormap.from_list("seq", _T["seq"]),
                           div=LinearSegmentedColormap.from_list("div", _T["div"]))


def figure(width="column", aspect=0.55, **kw):
    w = WIDTH_IN[width]
    return plt.subplots(figsize=(w, w * aspect), **kw)


def synthetic(ax):
    ax.text(1.0, 1.02, "synthetic data", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic")


def patch(color, label):
    from matplotlib.patches import Patch
    return Patch(facecolor=color, edgecolor="none", label=label)


def save(fig, path=None):
    path = str(path or os.environ["FIGKIT_OUT"])
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None} if path.endswith(".svg") else None)
