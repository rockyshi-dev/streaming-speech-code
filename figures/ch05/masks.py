"""Attention masks over 12 encoder steps: full; regular band (left 2, right 2), whose reach compounds
across layers; chunk-aware (chunk 4, one chunk of left context), built with the same rule as NeMo
v3.0.0 conformer_encoder.py lines 823-842. Rows attend to columns."""
import numpy as np

from figures.bookstyle import setup, figure, save

C, R = setup()
n = 12
i = np.arange(n)
full = np.ones((n, n))
band = ((i[None, :] - i[:, None] <= 2) & (i[:, None] - i[None, :] <= 2)).astype(float)
right = 3
chunk = right + 1
left_chunks = 4 // chunk
cid = i // chunk
diff = cid[:, None] - cid[None, :]
chunked = ((diff <= left_chunks) & (diff >= 0)).astype(float)
fig, axes = figure("wide", aspect=0.36, ncols=3)
for ax, m, title in zip(axes, [full, band, chunked],
                        ["full (offline)", "regular: right context 2", "chunk-aware: chunk 4, left 1 chunk"]):
    ax.imshow(m, cmap=C.seq, vmin=0, vmax=1.6)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("attends to step"); ax.grid(False)
axes[0].set_ylabel("step")
save(fig)
