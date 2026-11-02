"""A toy jitter buffer under SYNTHETIC network jitter: the trade between buffer depth and late packets.

Model (all synthetic, seeded; nothing here is a measurement of a real network):
  * a sender emits one packet every `frame_ms` (20 ms, one Opus frame);
  * each packet gets an independent one-way delay = base + jitter, jitter drawn from a gamma distribution
    (mean `jitter_mean_ms`, shape `shape`), plus rare spikes (probability `spike_p`, extra `spike_ms`), and is
    lost with probability `loss`;
  * UDP-like delivery (WebRTC): each packet arrives at send + delay, possibly out of order; a lost packet is gone;
  * TCP-like delivery (WebSocket): bytes are delivered in order, so packet i cannot arrive before packet i-1
    (head-of-line blocking), and a lost segment arrives `rto_ms` later than it would have, delaying everything
    behind it. That repair time is an ASSUMPTION of the model and the TCP numbers depend on it, so two values
    are reported: 200 ms (Linux's minimum retransmission timeout) and about one round trip
    (2 * base + 10 ms), roughly what fast retransmit achieves on a steady 20 ms packet stream.

A fixed jitter buffer of depth D plays packet i at  ref + D + i * frame_ms, where ref is the arrival time the
fastest packet implies for packet 0 (min over i of arrival_i - i * frame_ms): D is the delay the buffer adds
on top of the network's best case, which a receiver learns after a few packets. A packet that arrives after
its playout time is late and is concealed; a lost UDP packet is concealed at any depth and is counted
separately. Larger D means fewer late packets and D more milliseconds of user-perceived latency.

    python ch17/jitter.py --out $BOOK_DATA/runs/ch17_jitter
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import paths  # noqa: E402
from common.util import RESULTS, run_meta, write_json  # noqa: E402


@dataclass
class Net:
    frame_ms: float = 20.0
    base_ms: float = 30.0
    jitter_mean_ms: float = 8.0
    shape: float = 2.0
    spike_p: float = 0.01
    spike_ms: float = 120.0
    loss: float = 0.01
    rto_ms: float = 200.0


def trace(net: Net, n: int, seed: int = 0):
    """send times (ms), per-packet delay (ms), lost flags."""
    rng = np.random.default_rng(seed)
    send = np.arange(n) * net.frame_ms
    jit = rng.gamma(net.shape, net.jitter_mean_ms / net.shape, n)
    spikes = (rng.random(n) < net.spike_p) * net.spike_ms
    delay = net.base_ms + jit + spikes
    lost = rng.random(n) < net.loss
    return send, delay, lost


def arrivals_udp(send, delay, lost):
    a = send + delay
    return np.where(lost, np.inf, a)


def arrivals_tcp(send, delay, lost, rto_ms):
    """In-order delivery; a lost packet is resent after rto_ms (its own delay applies again)."""
    a = send + delay + np.where(lost, rto_ms, 0.0)
    return np.maximum.accumulate(a)


def late_fraction(arrival, frame_ms, depth_ms):
    """Fraction of packets that arrived but after their playout time (lost packets excluded)."""
    i = np.arange(len(arrival))
    ok = np.isfinite(arrival)
    ref = np.min(arrival[ok] - i[ok] * frame_ms)
    play = ref + depth_ms + i * frame_ms
    return float(np.sum(ok & (arrival > play)) / len(arrival))


def concealed_fraction(arrival, frame_ms, depth_ms):
    """Late plus lost, as a fraction of all packets."""
    return late_fraction(arrival, frame_ms, depth_ms) + float(np.mean(~np.isfinite(arrival)))


def depth_for(arrival, frame_ms, target, max_ms=1000, step=1):
    """Smallest depth (ms) whose late fraction is <= target; None if none up to max_ms."""
    for d in np.arange(0, max_ms + step, step):
        if late_fraction(arrival, frame_ms, d) <= target:
            return float(d)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{paths.BOOK_DATA}/runs/ch17_jitter")
    ap.add_argument("--packets", type=int, default=30000, help="10 minutes of 20 ms frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    scenarios = {
        "calm": Net(jitter_mean_ms=4, spike_p=0.0, loss=0.0),
        "wifi": Net(jitter_mean_ms=10, spike_p=0.01, spike_ms=120, loss=0.01),
        "mobile": Net(base_ms=60, jitter_mean_ms=25, shape=1.5, spike_p=0.02, spike_ms=250, loss=0.03),
    }
    depths = list(range(0, 301, 10))
    seeds = list(range(a.seed, a.seed + a.seeds))
    res = {"meta": run_meta(["numpy"]), "synthetic": True, "depths_ms": depths, "seeds": seeds, "scenarios": {}}
    for name, net in scenarios.items():
        fast = 2 * net.base_ms + 10
        per = {"udp": [], "tcp_rto200": [], "tcp_fast": []}
        d1 = {k: [] for k in per}
        d01 = {k: [] for k in per}
        lost_rate = []
        for sd in seeds:
            s_, d_, l_ = trace(net, a.packets, sd)
            arr = {"udp": arrivals_udp(s_, d_, l_), "tcp_rto200": arrivals_tcp(s_, d_, l_, 200.0),
                   "tcp_fast": arrivals_tcp(s_, d_, l_, fast)}
            lost_rate.append(float(np.mean(l_)))
            for k, x in arr.items():
                per[k].append([late_fraction(x, net.frame_ms, dd) for dd in depths])
                d1[k].append(depth_for(x, net.frame_ms, 0.01))
                d01[k].append(depth_for(x, net.frame_ms, 0.001))
        med = lambda v: None if any(x is None for x in v) else float(np.median(v))  # noqa: E731
        res["scenarios"][name] = {
            "net": asdict(net), "tcp_fast_repair_ms": fast,
            "udp_lost_mean": round(float(np.mean(lost_rate)), 5),
            "late_fraction_mean": {k: [round(float(x), 5) for x in np.mean(v, axis=0)] for k, v in per.items()},
            "depth_for_1pct_ms": {k: {"median": med(v), "per_seed": v} for k, v in d1.items()},
            "depth_for_0.1pct_ms": {k: {"median": med(v), "per_seed": v} for k, v in d01.items()},
        }
        print(name, {k: med(v) for k, v in d1.items()}, {k: med(v) for k, v in d01.items()})
    res["notes"] = ("Late fractions exclude lost packets (a lost UDP packet is concealed at any depth; "
                    "udp_lost_mean gives that rate). TCP never loses a packet but delivers it late; its repair "
                    "time is assumed (200 ms or about one round trip). Depth is the delay added above the "
                    "fastest packet's network delay. Medians over the listed seeds.")
    write_json(Path(a.out) / "jitter.json", res)
    write_json(RESULTS / "ch17_jitter_synthetic.json", res)


if __name__ == "__main__":
    main()
