"""How a p95 behaves with few requests: 2,000 draws each of n = 20, 100 and 1,000 latencies from one
log-normal distribution (median 500 ms), with numpy's default linear percentile. At n = 20 the p95
sits between the two largest values, so it swings with them. Synthetic."""
import numpy as np

from figures.bookstyle import setup, figure, save, synthetic

C, R = setup()
rng = np.random.default_rng(0)
true_p95 = 500 * np.exp(0.4 * 1.6449)
fig, ax = figure(aspect=0.4)
for k, n in enumerate([20, 100, 1000]):
    p = np.percentile(rng.lognormal(np.log(500), 0.4, size=(2000, n)), 95, axis=1)
    lo, hi = np.percentile(p, [5, 95])
    ax.plot([lo, hi], [k, k], color=R.base, lw=6, solid_capstyle="butt")
    ax.plot(np.median(p), k, "o", color=R.focus, ms=6)
    ax.text(hi + 15, k, f"{lo:,.0f} to {hi:,.0f} ms", va="center", fontsize=9)
ax.axvline(true_p95, color=C.ink, lw=0.8, ls=":")
ax.text(true_p95 + 5, 2.45, f"true p95 = {true_p95:,.0f} ms", fontsize=9)
ax.set_yticks(range(3), ["20 requests", "100 requests", "1,000 requests"])
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, 2.8)
ax.set_xlabel("measured p95 (ms): middle 90% of 2,000 repeated measurements")
ax.grid(axis="y", visible=False)
synthetic(ax)
save(fig)
