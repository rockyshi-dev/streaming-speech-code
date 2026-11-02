"""Gain across a crossfade of 3,840 samples (160 ms at 24 kHz, CosyVoice 2's overlap): CosyVoice 2's
Hamming halves (np.hamming(2 * 3840), as in cosyvoice/cli/model.py and utils/common.py fade_in_out)
and a linear crossfade. Amplitude sum: what two identical signals add up to. Power sum: what two
unrelated signals add up to in energy. Computed from the window definitions."""
import numpy as np

from figures.bookstyle import setup, figure, save, synthetic

C, R = setup()
L = 3840
w = np.hamming(2 * L)
up, down = w[:L], w[L:]
lin_up = np.linspace(0, 1, L)
t = np.arange(L) / 24000 * 1000
fig, ax = figure(aspect=0.45)
ax.plot(t, 20 * np.log10(up + down), color=R.focus, lw=1.6, label="Hamming, identical signals")
ax.plot(t, 10 * np.log10(up ** 2 + down ** 2), color=R.focus, lw=1.6, ls="--", label="Hamming, unrelated signals")
ax.plot(t, 20 * np.log10(lin_up + lin_up[::-1]), color=R.base, lw=2.4, label="linear, identical signals (0 dB)")
ax.plot(t, 10 * np.log10(lin_up ** 2 + lin_up[::-1] ** 2), color=R.base, lw=1.6, ls="--", label="linear, unrelated signals")
ax.axhline(0, color=C.ink, lw=0.8, ls=":")
ax.set_xlabel("position in the 160 ms overlap (ms)")
ax.set_ylabel("gain (dB)")
ax.set_xlim(0, t[-1])
ax.set_ylim(-3.6, 1.2)
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2), ncols=2)
synthetic(ax)
save(fig)
