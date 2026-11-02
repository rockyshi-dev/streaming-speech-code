import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ch17.jitter import Net, arrivals_tcp, arrivals_udp, concealed_fraction, depth_for, late_fraction, trace  # noqa: E402


def test_no_jitter_no_concealment():
    send = np.arange(100) * 20.0
    delay = np.full(100, 30.0)
    lost = np.zeros(100, bool)
    assert concealed_fraction(arrivals_udp(send, delay, lost), 20, 0) == 0.0


def test_one_late_packet():
    send = np.arange(10) * 20.0
    delay = np.full(10, 30.0)
    delay[5] += 25  # 25 ms late
    lost = np.zeros(10, bool)
    udp = arrivals_udp(send, delay, lost)
    assert concealed_fraction(udp, 20, 24) == 0.1
    assert concealed_fraction(udp, 20, 25) == 0.0
    # TCP: in-order delivery pushes packet 6 behind packet 5 (head-of-line); 6 is only 5 ms late
    tcp = arrivals_tcp(send, delay, lost, 200)
    assert tcp[6] == tcp[5] and concealed_fraction(tcp, 20, 4) == 0.2


def test_loss_udp_vs_tcp():
    send = np.arange(10) * 20.0
    delay = np.full(10, 30.0)
    lost = np.zeros(10, bool)
    lost[3] = True
    assert concealed_fraction(arrivals_udp(send, delay, lost), 20, 1000) == 0.1  # gone at any depth
    assert late_fraction(arrivals_udp(send, delay, lost), 20, 0) == 0.0
    tcp = arrivals_tcp(send, delay, lost, 200)
    assert concealed_fraction(tcp, 20, 200) == 0.0 and concealed_fraction(tcp, 20, 199) > 0


def test_depth_monotone():
    s, d, l = trace(Net(), 5000, 1)
    udp = arrivals_udp(s, d, l)
    fr = [concealed_fraction(udp, 20, x) for x in range(0, 200, 10)]
    assert all(a >= b for a, b in zip(fr, fr[1:]))
    assert depth_for(udp, 20, 0.5) is not None


def test_reference_is_fastest_packet_not_first():
    send = np.arange(10) * 20.0
    delay = np.full(10, 30.0)
    delay[0] = 80.0  # first packet slow: anchoring on it would hide all later jitter
    lost = np.zeros(10, bool)
    assert late_fraction(arrivals_udp(send, delay, lost), 20, 0) == 0.1
    assert late_fraction(arrivals_udp(send, delay, lost), 20, 50) == 0.0
