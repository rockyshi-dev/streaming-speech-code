import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.player import Player  # noqa: E402


def test_continuous_stream_no_underrun():
    p = Player(sr=1000)
    for j in range(5):
        p.add(0.1 * j, 100)  # each piece 0.1 s, arriving just in time
    assert p.underruns() == 0
    assert p.play_time(0) == 0.0
    assert abs(p.play_time(250) - 0.25) < 1e-9


def test_late_piece_is_an_underrun_and_shifts_later_audio():
    p = Player(sr=1000)
    p.add(0.0, 100)
    p.add(0.3, 100)  # player ran dry at 0.1
    p.add(0.3, 100)
    assert p.underruns() == 1
    assert abs(p.play_time(100) - 0.3) < 1e-9
    assert abs(p.play_time(200) - 0.4) < 1e-9


def test_prebuffer_delays_start():
    p = Player(sr=1000, prebuffer_s=0.05)
    p.add(1.0, 100)
    assert abs(p.play_time(10) - 1.06) < 1e-9


def test_stop():
    p = Player(sr=1000)
    p.add(0.0, 1000)
    p.stop(0.25)
    assert p.samples_played() == 250
    assert p.play_time(300) is None and p.play_time(100) is not None


def test_gap_inside_inaudible_lead_in_not_counted_after_first_audible():
    p = Player(sr=1000)
    p.add(0.0, 80)    # quiet lead-in
    p.add(0.4, 2000)  # first audible sample is in here (index 90)
    p.add(2.3, 100)   # arrives in time
    assert p.underruns() == 1
    assert p.underruns(after_sample=90) == 0
