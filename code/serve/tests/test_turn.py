import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.turn import WIN_MS, Endpointer  # noqa: E402


def run(ep, probs):
    out = []
    for i, p in enumerate(probs):
        out += ep.step(p, (i + 1) * WIN_MS / 1000, None)
    return out


def test_silence_endpoint_after_threshold():
    ep = Endpointer(mode="silence", silence_ms=320, start_ms=96)
    ev = run(ep, [0.9] * 10 + [0.1] * 20)
    kinds = [e["type"] for e in ev]
    assert kinds == ["speech_start", "endpoint"]
    # speech starts on the 3rd speech window (96 ms); endpoint on the 10th silent window (320 ms of silence)
    assert abs(ev[0]["t"] - 3 * 0.032) < 1e-9
    assert abs(ev[1]["t"] - 20 * 0.032) < 1e-9


def test_short_pause_does_not_end_turn():
    ep = Endpointer(mode="silence", silence_ms=320)
    ev = run(ep, [0.9] * 10 + [0.1] * 5 + [0.9] * 5 + [0.1] * 12)
    ends = [e for e in ev if e["type"] == "endpoint"]
    assert len(ends) == 1 and abs(ends[0]["t"] - 30 * 0.032) < 1e-9


def test_hysteresis_band_neither_speech_nor_silence():
    ep = Endpointer(mode="silence", silence_ms=64)
    ev = run(ep, [0.9] * 5 + [0.4] * 10)  # 0.4 is in [0.35, 0.5): does not count as silence
    assert [e["type"] for e in ev] == ["speech_start"]


def test_smart_complete_ends_at_trigger():
    ep = Endpointer(mode="smart", trigger_ms=192, fallback_ms=3000, turn_model=lambda a: 0.9)
    ev = run(ep, [0.9] * 10 + [0.1] * 10)
    e = [x for x in ev if x["type"] == "endpoint"][0]
    assert e["reason"] == "smart_turn" and abs(e["t"] - 16 * 0.032) < 1e-9


def test_smart_incomplete_waits_for_fallback():
    calls = []
    ep = Endpointer(mode="smart", trigger_ms=192, fallback_ms=640, turn_model=lambda a: calls.append(1) or 0.1)
    ev = run(ep, [0.9] * 10 + [0.1] * 30)
    e = [x for x in ev if x["type"] == "endpoint"][0]
    assert e["reason"] == "fallback" and abs(e["t"] - 30 * 0.032) < 1e-9
    assert len(calls) == 1  # asked once per pause


def test_smart_asks_again_after_speech_resumes():
    calls = []
    ep = Endpointer(mode="smart", trigger_ms=96, fallback_ms=3000, turn_model=lambda a: calls.append(1) or 0.1)
    run(ep, [0.9] * 5 + [0.1] * 4 + [0.9] * 5 + [0.1] * 4)
    assert len(calls) == 2


def test_band_windows_extend_a_started_silence():
    ep = Endpointer(mode="silence", silence_ms=320)
    ev = run(ep, [0.9] * 10 + [0.1] * 2 + [0.4] * 8 + [0.1] * 5)
    e = [x for x in ev if x["type"] == "endpoint"][0]
    assert abs(e["t"] - 20 * 0.032) < 1e-9  # 2 silent + 8 band windows = 320 ms
