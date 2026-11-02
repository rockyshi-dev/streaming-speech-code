import numpy as np

from hbasr import latency


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeRec:
    """Emits one word per chunk after the 2nd chunk; every accept costs `cost` seconds of fake time."""
    sample_rate = 16000

    def __init__(self, clock, cost):
        self.clock, self.cost = clock, cost

    def start(self):
        self.n = 0
        self.words = []

    def accept(self, x):
        self.clock.t += self.cost
        self.n += 1
        if self.n >= 2:
            self.words.append(f"w{self.n}")

    def partial(self):
        return " ".join(self.words)

    def endpoint(self):
        return False

    def finish(self):
        self.clock.t += self.cost
        return self.partial()


def test_simulated_clock_fast():
    c = FakeClock()
    x = np.zeros(16000, np.float32)  # 1 s -> 7 chunks of 160 ms (last 40 ms)
    lg = latency.run_stream(FakeRec(c, 0.01), x, chunk_ms=160, speech=(0.2, 0.9), clock=c)
    assert len(lg.events) == 7
    e1 = lg.events[1]
    assert abs(e1.done_s - (0.32 + 0.01)) < 1e-9
    s = lg.summary()
    assert abs(s["first_token_delay_s"] - (0.33 - 0.2)) < 1e-9
    assert abs(s["final_latency_after_input_end_s"] - 0.01 - 0.01) < 1e-9 or s["final_latency_after_input_end_s"] > 0
    assert abs(s["rtf"] - 8 * 0.01 / 1.0) < 1e-9
    assert s["max_backlog_s"] == 0.0


def test_simulated_clock_backlog():
    c = FakeClock()
    x = np.zeros(16000, np.float32)
    lg = latency.run_stream(FakeRec(c, 0.3), x, chunk_ms=160, clock=c)  # slower than real time
    starts = [e.start_s for e in lg.events]
    assert all(b >= a + 0.3 - 1e-9 for a, b in zip(starts, starts[1:]))
    assert lg.summary()["max_backlog_s"] > 0.5
    assert lg.summary()["rtf"] > 1


def test_emissions_and_lags():
    c = FakeClock()
    lg = latency.run_stream(FakeRec(c, 0.0), np.zeros(16000, np.float32), chunk_ms=160, clock=c)
    ems = lg.emissions()
    assert ems[0][1] == "w2" and len(ems) == 6
    lags = latency.emission_lags(lg, [0.2, 0.4, 0.6, 0.8, 0.9, 1.0], str.split)
    assert abs(lags[0] - (0.32 - 0.2)) < 1e-9


def test_aggregate():
    s = [{"audio_s": 1.0, "rtf": 0.1, "first_token_delay_s": 0.2, "final_latency_after_input_end_s": 0.1,
          "final_latency_after_speech_end_s": 0.3, "endpoint_delay_s": None, "max_backlog_s": 0.0}] * 3
    a = latency.aggregate(s)
    assert abs(a["corpus_rtf"] - 0.1) < 1e-12 and a["first_token_delay_s"]["p50"] == 0.2
    assert a["endpoint_delay_s"]["n"] == 0


def test_emission_lags_skip_none():
    c = FakeClock()
    lg = latency.run_stream(FakeRec(c, 0.0), np.zeros(16000, np.float32), chunk_ms=160, clock=c)
    lags = latency.emission_lags(lg, [None, 0.3], str.split)
    assert lags[0] is None and abs(lags[1] - (0.48 - 0.3)) < 1e-9
