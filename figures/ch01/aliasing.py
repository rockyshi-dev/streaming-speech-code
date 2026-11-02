"""Aliasing: cosines at 6 kHz and 10 kHz sampled at 16 kHz give identical samples, because
cos(2*pi*10k*n/16k) = cos(2*pi*(16k-10k)*n/16k). Synthetic."""
import numpy as np
from figures.bookstyle import setup, figure, save, synthetic

C, R = setup()
sr = 16000
t = np.linspace(0, 1.0e-3, 2000)
n = np.arange(0, 17)
fig, ax = figure(aspect=0.38)
ax.plot(t * 1e3, np.cos(2 * np.pi * 10000 * t), color=R.muted, lw=1.0, label="10 kHz (above the 8 kHz limit)")
ax.plot(t * 1e3, np.cos(2 * np.pi * 6000 * t), color=R.focus, lw=1.6, label="6 kHz")
ax.plot(n / sr * 1e3, np.cos(2 * np.pi * 10000 * n / sr), "o", color=C.ink, ms=5,
        label="samples at 16 kHz (the same for both)")
ax.set_xlabel("time (ms)")
ax.set_ylabel("amplitude")
ax.set_xlim(0, 1.0)
ax.set_ylim(-1.25, 1.6)
ax.legend(loc="upper right", ncols=3, fontsize=8, frameon=False)
synthetic(ax)
save(fig)
