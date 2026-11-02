"""A latency harness for streaming recognizers (chapter 3; reused by chapters 6, 7, 9, 15 and 18).

The harness plays one utterance into a recognizer chunk by chunk, as a microphone would, and records
for every chunk when its audio existed, when the recognizer got to it and finished it, and what the
partial result was afterwards. From that log it computes, per utterance:

  first-token delay   time the first non-empty partial result is available, minus speech onset
  final latency       time the final result is available, minus the end of speech (and, separately,
                      minus the end of the audio the client sent)
  endpoint delay      time the recognizer's endpointer fired, minus the end of speech
  RTF                 total compute time / audio duration

Clock. t = 0 is the moment the first sample of the utterance is captured. A chunk of audio covering
[a, b) seconds "arrives" at t = b: it cannot be processed before its last sample exists.

Pacing. `pace="realtime"` sleeps until each chunk's arrival time and reads the wall clock.
`pace="simulated"` does not sleep: it measures each chunk's compute time and advances a virtual
clock, start_i = max(arrival_i, done_{i-1}), done_i = start_i + compute_i. That is the same schedule a
real-time run would have on an otherwise idle machine, without waiting 7 s per 7 s utterance; a
backlog (compute slower than real time) shows up as start_i > arrival_i, exactly as it would live.
`compare_pacing()` checks the two agree on a given machine.

What it does not include: transport (network, client buffering) and anything outside the recognizer
call. A number from this harness is recognizer-side latency on the stated hardware.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Protocol

import numpy as np

from .metrics import percentiles


class StreamingRecognizer(Protocol):
    """What the harness needs from a recognizer. Adapters live in hbasr.backends_*."""

    sample_rate: int

    def start(self) -> None: ...                 # open a new stream
    def accept(self, x: np.ndarray) -> None: ... # feed float32 samples; decode everything decodable
    def partial(self) -> str: ...                # current partial result
    def endpoint(self) -> bool: ...              # has the endpointer fired since start()?
    def finish(self) -> str: ...                 # end of input: flush, return the final result


@dataclass
class ChunkEvent:
    index: int
    audio_start_s: float
    audio_end_s: float
    arrival_s: float
    start_s: float
    done_s: float
    compute_s: float
    partial: str
    endpoint: bool


@dataclass
class StreamLog:
    utt_id: str
    chunk_ms: float
    pace: str
    audio_s: float
    events: list = field(default_factory=list)
    final_text: str = ""
    final_start_s: float = 0.0
    final_done_s: float = 0.0
    final_compute_s: float = 0.0
    endpoint_s: Optional[float] = None
    endpoint_text: Optional[str] = None
    speech_onset_s: Optional[float] = None
    speech_offset_s: Optional[float] = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict) -> "StreamLog":
        d = dict(d)
        d["events"] = [ChunkEvent(**e) for e in d["events"]]
        return StreamLog(**d)

    # ------------------------------------------------------------ derived quantities

    def emissions(self) -> list[tuple[float, str]]:
        """Distinct partial results in order, with the time each first became available; final last."""
        out, prev = [], ""
        for e in self.events:
            if e.partial != prev:
                out.append((e.done_s, e.partial))
                prev = e.partial
        if self.final_text != prev:
            out.append((self.final_done_s, self.final_text))
        return out

    def first_partial_s(self) -> Optional[float]:
        for e in self.events:
            if e.partial.strip():
                return e.done_s
        return self.final_done_s if self.final_text.strip() else None

    def compute_total_s(self) -> float:
        return sum(e.compute_s for e in self.events) + self.final_compute_s

    def summary(self) -> dict:
        fp = self.first_partial_s()
        on, off = self.speech_onset_s, self.speech_offset_s
        return {
            "utt_id": self.utt_id,
            "audio_s": self.audio_s,
            "rtf": self.compute_total_s() / self.audio_s if self.audio_s else None,
            "first_partial_s": fp,
            "first_token_delay_s": (fp - on) if (fp is not None and on is not None) else None,
            "final_latency_after_input_end_s": self.final_done_s - self.audio_s,
            "final_latency_after_speech_end_s": (self.final_done_s - off) if off is not None else None,
            "endpoint_delay_s": (self.endpoint_s - off) if (self.endpoint_s is not None and off is not None) else None,
            "max_backlog_s": max([e.start_s - e.arrival_s for e in self.events] or [0.0]),
            "n_chunks": len(self.events),
        }


def run_stream(
    rec: StreamingRecognizer,
    x: np.ndarray,
    utt_id: str = "utt",
    chunk_ms: float = 160.0,
    pace: str = "simulated",
    speech: Optional[tuple[float, float]] = None,
    stop_at_endpoint: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> StreamLog:
    """Stream `x` (float32 at rec.sample_rate) into `rec` in `chunk_ms` chunks and log every chunk."""
    if pace not in ("simulated", "realtime"):
        raise ValueError(pace)
    sr = rec.sample_rate
    n = int(round(sr * chunk_ms / 1000.0))
    log = StreamLog(utt_id, chunk_ms, pace, x.size / sr)
    if speech is not None:
        log.speech_onset_s, log.speech_offset_s = speech
    rec.start()
    t0 = clock()
    done_prev = 0.0
    for i, a in enumerate(range(0, x.size, n)):
        c = x[a:a + n]
        a_s, b_s = a / sr, (a + c.size) / sr
        if pace == "realtime":
            wait = b_s - (clock() - t0)
            if wait > 0:
                time.sleep(wait)
            start = clock() - t0
            rec.accept(c)
            ep = rec.endpoint()
            p = rec.partial()
            done = clock() - t0
            compute = done - start
        else:
            c0 = clock()
            rec.accept(c)
            ep = rec.endpoint()
            p = rec.partial()
            compute = clock() - c0
            start = max(b_s, done_prev)
            done = start + compute
        done_prev = done
        log.events.append(ChunkEvent(i, a_s, b_s, b_s, start, done, compute, p, bool(ep)))
        if ep and log.endpoint_s is None:
            log.endpoint_s = done
            log.endpoint_text = p
            if stop_at_endpoint:
                break
    # end of input
    if pace == "realtime":
        start = clock() - t0
        final = rec.finish()
        done = clock() - t0
        compute = done - start
    else:
        c0 = clock()
        final = rec.finish()
        compute = clock() - c0
        start = max(log.events[-1].arrival_s if log.events else 0.0, done_prev)
        done = start + compute
    log.final_text, log.final_start_s, log.final_done_s, log.final_compute_s = final, start, done, compute
    return log


def aggregate(summaries: list[dict]) -> dict:
    """Corpus-level statistics of per-utterance summaries: p50/p90/p95/mean per quantity, plus the
    corpus RTF (sum of compute over sum of audio, not a mean of ratios)."""
    keys = ["first_token_delay_s", "final_latency_after_input_end_s", "final_latency_after_speech_end_s",
            "endpoint_delay_s", "max_backlog_s", "rtf"]
    out = {k: percentiles([s[k] for s in summaries]) for k in keys}
    audio = sum(s["audio_s"] for s in summaries)
    comp = sum((s["rtf"] or 0.0) * s["audio_s"] for s in summaries)
    out["corpus_rtf"] = comp / audio if audio else None
    out["audio_total_s"] = audio
    out["utterances"] = len(summaries)
    return out


def compare_pacing(make_rec: Callable[[], StreamingRecognizer], x: np.ndarray, chunk_ms: float = 160.0) -> dict:
    """Run the same utterance with real-time and simulated pacing; return both summaries and texts."""
    rt = run_stream(make_rec(), x, "rt", chunk_ms, "realtime")
    sim = run_stream(make_rec(), x, "sim", chunk_ms, "simulated")
    return {
        "realtime": rt.summary() | {"final_text": rt.final_text},
        "simulated": sim.summary() | {"final_text": sim.final_text},
        "same_final_text": rt.final_text == sim.final_text,
        "same_partials": [e.partial for e in rt.events] == [e.partial for e in sim.events],
    }


def emission_lags(log: StreamLog, unit_times: list[float], tokenize: Callable[[str], list[str]]) -> list[float]:
    """Per-unit emission lag: for final unit i, the time from `unit_times[i]` (where the model itself
    places the unit in the audio, e.g. a transducer's emission frame) to the first partial result from
    which units 0..i never change again. This measures how long the recognizer holds a unit back after
    the model has seen it. With `unit_times` from a forced alignment of the reference (None for units
    that are not correctly recognized), the same function gives the per-word emission latency."""
    final = tokenize(log.final_text)
    ems = [(t, tokenize(s)) for t, s in log.emissions()]
    lags = []
    for i, ut in enumerate(unit_times[: len(final)]):
        if ut is None:
            lags.append(None)
            continue
        target = final[: i + 1]
        t_stable = None
        for j in range(len(ems) - 1, -1, -1):  # walk back while the prefix is held
            if ems[j][1][: i + 1] == target:
                t_stable = ems[j][0]
            else:
                break
        lags.append(None if t_stable is None else t_stable - ut)
    return lags
