"""160 ms of audio (the running example's chunk) at the frame rate of each codec or token stream in
chapter 11, with the number of codes per frame. Rates from code and configs: EnCodec 24 kHz and
DAC 24 kHz 75 Hz (hop 320); CosyVoice 2/3 tokens 25 Hz; XTTS 21.53 Hz; Mimi and the Qwen3-TTS 12 Hz
tokenizer 12.5 Hz (hop 1,920)."""
from figures.bookstyle import setup, figure, save, synthetic

C, R = setup()
rows = [("EnCodec / DAC 24 kHz", 75.0, "8 codes per frame at 6 kbit/s"),
        ("CosyVoice 2/3 tokens", 25.0, "1 token per frame"),
        ("XTTS VQ-VAE", 22050 / 1024, "1 code per frame"),
        ("Mimi, Qwen3-TTS 12 Hz", 12.5, "8 to 32 codes per frame")]
fig, ax = figure(aspect=0.32)
for y, (name, hz, note) in enumerate(reversed(rows)):
    step = 1000 / hz
    t = 0.0
    k = 0
    while t < 160 - 1e-6:
        w = min(step, 160 - t)
        ax.barh(y, w * 0.94, 0.55, left=t, color=R.focus if k == 0 else R.base)
        t += step
        k += 1
    ax.text(166, y, f"{step:.1f} ms per frame; {note}", va="center", fontsize=9)
ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 160)
ax.set_xlabel("160 ms of audio (ms); each bar is one frame, the first highlighted")
ax.grid(axis="y", visible=False)
synthetic(ax)
save(fig)
