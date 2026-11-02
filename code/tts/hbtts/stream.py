"""Client-side record of one streamed TTS request, and what is computed from it.

All times are on the client's monotonic clock (`time.perf_counter()`), in seconds,
relative to `t0`.

t0
    The moment the client sends the request. When text is streamed in (simulated LLM
    output), t0 is the moment the first piece of text is sent.
time to first audio (TTFA)
    Arrival time of the received piece that contains the first audible sample
    (see `hbtts.audio.first_audible_index`), minus t0. Header bytes and leading
    inaudible samples do not count.
first PCM byte
    Arrival time of the first piece that carries at least one PCM sample, minus t0.
    Reported as a diagnostic; it is not TTFA.
real-time factor (RTF)
    (arrival time of the last audio piece - t0) / duration of all received audio.
    Below 1.0 means audio was produced faster than it plays.
underrun
    The simulated player (one sample per 1/sr seconds) needs a sample that has not
    arrived. Each such event counts once; the player waits for the piece and resumes.
    `underruns` starts the player at first audio (the arrival of the piece holding the
    first audible sample, i.e. at TTFA) plus an optional pre-buffer, and plays every
    sample received, the inaudible lead-in included. `underruns_from_first_pcm` starts
    it at the first PCM piece instead, as a player that starts on any byte would.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import first_audible_index


@dataclass
class PlaybackResult:
    underruns: int
    stall_s: float
    start_s: float  # player start time, relative to t0
    end_s: float  # time the last sample finishes playing, relative to t0


def simulate_playback(
    arrivals: list[float],
    n_samples: list[int],
    sr: int,
    prebuffer_s: float = 0.0,
    start_at: float | None = None,
    eps: float = 1e-6,
) -> PlaybackResult:
    """Real-time player fed by pieces arriving at `arrivals` (s, relative to t0).

    Piece i holds n_samples[i] samples. Pieces with zero samples are ignored. Playback
    starts at `start_at` (default: the first non-empty arrival) plus `prebuffer_s`, and
    never before the first piece has arrived. Each time the player reaches the start of
    piece i before piece i has arrived, one underrun is counted and playback resumes
    when it arrives.
    """
    if len(arrivals) != len(n_samples):
        raise ValueError("arrivals and n_samples differ in length")
    pieces = [(a, n) for a, n in zip(arrivals, n_samples) if n > 0]
    if not pieces:
        return PlaybackResult(0, 0.0, float("nan"), float("nan"))
    if any(pieces[i + 1][0] < pieces[i][0] for i in range(len(pieces) - 1)):
        raise ValueError("arrival times must be non-decreasing")
    start = max(pieces[0][0], pieces[0][0] if start_at is None else start_at) + prebuffer_s
    clock = start  # time at which the player will need the next piece
    underruns = 0
    stall = 0.0
    for a, n in pieces:
        if a > clock + eps:
            underruns += 1
            stall += a - clock
            clock = a
        clock += n / sr
    return PlaybackResult(underruns, stall, start, clock)


@dataclass
class StreamRecord:
    sr: int
    t0: float = 0.0
    t_text_done: float | None = None  # last text piece sent (relative to t0), if streamed in
    arrivals: list[float] = field(default_factory=list)  # relative to t0
    pieces: list[np.ndarray] = field(default_factory=list)  # float32 samples per piece
    error: str | None = None

    def add(self, t_abs: float, samples: np.ndarray) -> None:
        self.arrivals.append(t_abs - self.t0)
        self.pieces.append(np.asarray(samples, dtype=np.float32).reshape(-1))

    def audio(self) -> np.ndarray:
        if not self.pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.pieces)

    def summary(self, prebuffer_s: float = 0.0) -> dict:
        counts = [len(p) for p in self.pieces]
        total = int(sum(counts))
        out: dict = {"sr": self.sr, "n_pieces": len(counts), "n_samples": total,
                     "audio_s": total / self.sr if self.sr else 0.0, "error": self.error,
                     "t_text_done_s": self.t_text_done}
        if total == 0:
            out.update(ttfa_s=None, first_pcm_s=None, rtf=None, underruns=None,
                       stall_s=None, last_audio_s=None)
            return out
        nz = [i for i, c in enumerate(counts) if c > 0]
        out["first_pcm_s"] = self.arrivals[nz[0]]
        x = self.audio()
        k = first_audible_index(x, self.sr)
        if k is None:
            out["ttfa_s"] = None
            out["leading_inaudible_s"] = total / self.sr
        else:
            cum = np.cumsum(counts)
            piece = int(np.searchsorted(cum, k, side="right"))
            out["ttfa_s"] = self.arrivals[piece]
            out["leading_inaudible_s"] = k / self.sr
        out["last_audio_s"] = self.arrivals[nz[-1]]
        out["rtf"] = out["last_audio_s"] / out["audio_s"]
        start = out["ttfa_s"] if out["ttfa_s"] is not None else None
        pb = simulate_playback(self.arrivals, counts, self.sr, prebuffer_s, start_at=start)
        out["underruns"] = pb.underruns
        out["stall_s"] = pb.stall_s
        out["prebuffer_s"] = prebuffer_s
        pb0 = simulate_playback(self.arrivals, counts, self.sr, prebuffer_s)
        out["underruns_from_first_pcm"] = pb0.underruns
        out["stall_from_first_pcm_s"] = pb0.stall_s
        # inter-arrival time vs the audio duration each piece carries (non-empty pieces)
        arr = np.array([self.arrivals[i] for i in nz])
        dur = np.array([counts[i] / self.sr for i in nz])
        out["piece_dur_s"] = summarize(dur)
        if len(arr) > 1:
            gaps = np.diff(arr)
            out["inter_arrival_s"] = summarize(gaps)
            # a gap longer than the audio already queued would stall; this is the per-piece view
            out["gap_over_prev_dur"] = summarize(gaps / dur[:-1])
        return out


def summarize(v) -> dict:
    """n, mean, p50, p95, p99, min, max of a 1-D sample (linear-interpolated percentiles)."""
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }
