"""CPU-only tests for the measurement logic in hbtts (no models, no GPU)."""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hbtts.audio import WavStreamParser, first_audible_index, int16_bytes_to_float  # noqa: E402
from hbtts.errorrate import cer, edit_ops, normalize_en, normalize_zh, pooled, wer  # noqa: E402
from hbtts.stream import StreamRecord, simulate_playback, summarize  # noqa: E402
from hbtts.textstream import (first_sentence_complete_s, schedule, split_sentences,  # noqa: E402
                              text_units)

SR = 24000


# ---------- first audible sample ----------

def test_silence_is_not_audible():
    assert first_audible_index(np.zeros(SR), SR) is None


def test_quiet_noise_floor_is_not_audible():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 10 ** (-50 / 20), SR)  # -50 dBFS RMS
    assert first_audible_index(x, SR) is None


def test_onset_after_leading_silence():
    x = np.zeros(SR)
    x[4800:] = 0.1 * np.sin(2 * np.pi * 200 * np.arange(SR - 4800) / SR)  # -23 dBFS from 200 ms
    k = first_audible_index(x, SR)
    assert k == 4800  # 4800 is a multiple of the 240-sample window


def test_onset_inside_a_window_reports_window_start():
    x = np.zeros(SR)
    x[4900:] = 0.5
    # window 4800..5039 holds 140 samples of 0.5 -> RMS 0.38 >= 0.01
    assert first_audible_index(x, SR) == 4800


def test_threshold_is_minus_40_dbfs():
    win = 240
    just_below = np.full(win * 4, 10 ** (-40.5 / 20))
    just_above = np.full(win * 4, 10 ** (-39.5 / 20))
    assert first_audible_index(just_below, SR) is None
    assert first_audible_index(just_above, SR) == 0


# ---------- WAV header stripping ----------

def _wav_header(sr=SR, data_size=0xFFFFFFFF, extra_chunk=False):
    fmt = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16)
    lst = struct.pack("<4sI", b"LIST", 4) + b"INFO" if extra_chunk else b""
    data = struct.pack("<4sI", b"data", data_size)
    body = b"WAVE" + fmt + lst + data
    return struct.pack("<4sI", b"RIFF", 0xFFFFFFFF) + body


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("split", [1, 3, 7, 44, 1000])
def test_wav_header_stripped_across_pieces(extra, split):
    pcm = (np.arange(100, dtype="<i2") * 7).tobytes()
    stream = _wav_header(extra_chunk=extra) + pcm
    p = WavStreamParser()
    out = b"".join(p.feed(stream[i : i + split]) for i in range(0, len(stream), split))
    assert out == pcm
    assert p.sample_rate == SR and p.channels == 1 and p.bits == 16
    assert p.header_bytes == len(_wav_header(extra_chunk=extra))


def test_header_bytes_are_never_audio():
    # a loud-looking header must not trigger first audio
    p = WavStreamParser()
    pcm = p.feed(_wav_header())
    assert pcm == b"" and p.header_done


def test_int16_conversion():
    b = np.array([0, 16384, -32768, 32767], dtype="<i2").tobytes() + b"\x01"
    x = int16_bytes_to_float(b)
    assert x.tolist() == pytest.approx([0.0, 0.5, -1.0, 32767 / 32768])


# ---------- playback simulation ----------

def test_no_underrun_when_pieces_arrive_ahead():
    # 4 pieces of 0.5 s each, arriving every 0.25 s
    r = simulate_playback([0.1, 0.35, 0.6, 0.85], [12000] * 4, SR)
    assert r.underruns == 0 and r.stall_s == 0.0
    assert r.start_s == pytest.approx(0.1) and r.end_s == pytest.approx(2.1)


def test_underrun_counted_and_playback_resumes():
    # piece 0: 0.2 s at t=0; piece 1 arrives at 0.5 (player needed it at 0.2) -> 1 underrun 0.3 s
    # piece 2 needed at 0.5 + 0.2 = 0.7, arrives at 0.6 -> fine
    r = simulate_playback([0.0, 0.5, 0.6], [4800, 4800, 4800], SR)
    assert r.underruns == 1
    assert r.stall_s == pytest.approx(0.3)
    assert r.end_s == pytest.approx(0.5 + 0.4)


def test_prebuffer_absorbs_a_late_piece():
    r0 = simulate_playback([0.0, 0.5], [4800, 4800], SR, prebuffer_s=0.0)
    r1 = simulate_playback([0.0, 0.5], [4800, 4800], SR, prebuffer_s=0.3)
    assert r0.underruns == 1 and r1.underruns == 0


def test_empty_pieces_ignored():
    r = simulate_playback([0.0, 0.05, 0.1], [0, 2400, 0], SR)
    assert r.underruns == 0 and r.start_s == pytest.approx(0.05)


def test_exactly_on_time_is_not_an_underrun():
    r = simulate_playback([0.0, 0.1, 0.2], [2400, 2400, 2400], SR)
    assert r.underruns == 0


def test_unsorted_arrivals_rejected():
    with pytest.raises(ValueError):
        simulate_playback([0.2, 0.1], [10, 10], SR)


# ---------- stream record ----------

def test_ttfa_skips_header_and_leading_silence():
    rec = StreamRecord(sr=SR, t0=10.0)
    rec.add(10.05, np.zeros(2400))  # first PCM piece, silent
    rec.add(10.10, np.zeros(2400))  # still silent
    tone = 0.2 * np.sin(2 * np.pi * 300 * np.arange(2400) / SR)
    rec.add(10.15, tone)
    rec.add(10.20, tone)
    s = rec.summary()
    assert s["first_pcm_s"] == pytest.approx(0.05)
    assert s["ttfa_s"] == pytest.approx(0.15)
    assert s["leading_inaudible_s"] == pytest.approx(0.2)
    assert s["audio_s"] == pytest.approx(0.4)
    assert s["rtf"] == pytest.approx(0.20 / 0.4)
    assert s["underruns"] == 0


def test_player_starts_at_first_audio_not_first_byte():
    # a silent 80 ms first piece at 0.03 s, then 2 s of speech at 0.36 s, then 2 s at 0.70 s
    rec = StreamRecord(sr=SR, t0=0.0)
    rec.add(0.03, np.zeros(1920))
    rec.add(0.36, np.full(48000, 0.1))
    rec.add(0.70, np.full(48000, 0.1))
    s = rec.summary()
    assert s["ttfa_s"] == pytest.approx(0.36) and s["first_pcm_s"] == pytest.approx(0.03)
    assert s["underruns"] == 0
    assert s["underruns_from_first_pcm"] == 1
    assert s["stall_from_first_pcm_s"] == pytest.approx(0.36 - 0.03 - 0.08)


def test_start_at_is_never_before_first_piece():
    r = simulate_playback([0.5, 0.6], [2400, 2400], SR, start_at=0.0)
    assert r.start_s == pytest.approx(0.5) and r.underruns == 0


def test_record_counts_underruns():
    rec = StreamRecord(sr=SR, t0=0.0)
    for t in (0.0, 0.3, 0.6):  # 0.1 s pieces every 0.3 s -> 2 underruns of 0.2 s
        rec.add(t, np.full(2400, 0.1))
    s = rec.summary()
    assert s["underruns"] == 2 and s["stall_s"] == pytest.approx(0.4)
    assert s["gap_over_prev_dur"]["p50"] == pytest.approx(3.0)


def test_empty_record():
    s = StreamRecord(sr=SR).summary()
    assert s["ttfa_s"] is None and s["n_samples"] == 0


def test_summarize_percentiles():
    s = summarize(list(range(1, 101)))
    assert s["n"] == 100 and s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05)
    assert summarize([None, float("nan")]) == {"n": 0}


# ---------- text streaming ----------

def test_split_sentences_en():
    t = "It costs 3.5 dollars. Really? Yes! Fine; done."
    assert split_sentences(t) == ["It costs 3.5 dollars.", "Really?", "Yes!", "Fine; done."]


def test_split_sentences_zh():
    assert split_sentences("今天很好。你呢？我也好！") == ["今天很好。", "你呢？", "我也好！"]


def test_text_units_roundtrip():
    for t in ["Hello there, world. Bye.", "  lead space", "今天很好。你呢？", "我有3个GPU。"]:
        assert "".join(text_units(t)) == t
    assert text_units("今天很好。") == ["今", "天", "很", "好。"]
    assert text_units("Hello there, world.") == ["Hello ", "there, ", "world."]


def test_schedule_rate_and_all_at_once():
    u = ["a ", "b ", "c"]
    assert schedule(u, None) == [(0.0, "a b c")]
    assert schedule(u, 4.0) == [(0.0, "a "), (0.25, "b "), (0.5, "c")]


def test_first_sentence_complete():
    # "One two three." is units 0..2 -> released at 2/10 s
    assert first_sentence_complete_s("One two three. Four five.", 10.0) == pytest.approx(0.2)
    assert first_sentence_complete_s("你好。再见。", 10.0) == pytest.approx(0.1)


# ---------- error rates ----------

def test_edit_ops_counts():
    assert edit_ops(list("abc"), list("abc")) == (0, 0, 0)
    assert edit_ops(list("abc"), list("axc")) == (1, 0, 0)
    assert edit_ops(list("abc"), list("ac")) == (0, 1, 0)
    assert edit_ops(list("abc"), list("abxc")) == (0, 0, 1)
    assert sum(edit_ops(list("kitten"), list("sitting"))) == 3


def test_wer_and_normalization():
    r = wer("Hello, World! It's me.", "hello world its me")
    assert r["n_ref"] == 4 and r["sub"] == 1 and r["rate"] == pytest.approx(0.25)
    assert normalize_en("Don't  STOP—now.") == "don't stop now"


def test_cer_ignores_punctuation_and_spaces():
    r = cer("今天，天气 很好。", "今天天气很好")
    assert r["rate"] == 0.0 and r["n_ref"] == 6
    assert normalize_zh("我有3个GPU。") == "我有3个gpu"


def test_pooled_rate_is_not_mean_of_rates():
    rows = [{"n_ref": 1, "sub": 1, "del": 0, "ins": 0}, {"n_ref": 9, "sub": 0, "del": 0, "ins": 0}]
    assert pooled(rows) == pytest.approx(0.1)
