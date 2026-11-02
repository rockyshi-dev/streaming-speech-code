"""One NeMo cache-aware setting, att_context_size [70,13] with 8x subsampling (80 ms per encoder
frame): a chunk is 14 frames = 1,120 ms; the first frame of a chunk sees 13 future frames
(1,040 ms), the last sees none; the average is 520 ms. From NeMo conformer_encoder.py and
arXiv:2312.17279 Sec. 3.1."""
from figures.bookstyle import setup, figure, save

C, R = setup()
r, step = 13, 80
fig, ax = figure(aspect=0.36)
for i in range(r + 1):
    la = (r - i) * step
    ax.bar(i * step + step / 2, la, step * 0.8, color=R.lookahead)
ax.axhline((r * step) / 2, color=C.ink, lw=0.8, ls=":")
ax.text((r + 1) * step - 10, r * step / 2 + 25, "average 520 ms", ha="right", fontsize=9)
ax.text(step / 2 + 50, r * step - 10, "worst 1,040 ms", fontsize=9, va="top")
ax.annotate("", xy=(0, -170), xytext=((r + 1) * step, -170),
            arrowprops=dict(arrowstyle="<->", color=C.ink, lw=0.8), annotation_clip=False)
ax.text((r + 1) * step / 2, -215, "chunk: 14 frames = 1,120 ms", ha="center", va="top", fontsize=9)
ax.set_xlim(0, (r + 1) * step)
ax.set_ylim(0, 1150)
ax.set_xlabel("position of the encoder frame inside the chunk (ms)", labelpad=28)
ax.set_ylabel("future audio it waits for (ms)")
save(fig)
