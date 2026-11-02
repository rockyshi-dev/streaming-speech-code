"""A virtual audio player for timing (chapter 18): it plays received pieces in order at the sample rate.

The player starts when the first piece arrives plus `prebuffer_s`. Piece j starts playing at
max(arrival_j, end of piece j-1); if arrival_j is later than the end of piece j-1 the player ran dry
(an underrun) and waits. The time sample k is heard is then the start of its piece plus its offset / sr.
Device output latency (the sound card's own buffer) is not modelled; it adds a constant on a real device.

stop(t) records a barge-in: everything scheduled after t is not played.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Player:
    sr: int
    prebuffer_s: float = 0.0
    arrivals: list = field(default_factory=list)
    lengths: list = field(default_factory=list)
    stopped_at: float | None = None

    def add(self, t: float, n: int) -> None:
        if n > 0:
            self.arrivals.append(t)
            self.lengths.append(n)

    def schedule(self) -> list[tuple[float, float]]:
        """(start, end) play time of each piece."""
        out = []
        clock = None
        for j, (a, n) in enumerate(zip(self.arrivals, self.lengths)):
            start = a + self.prebuffer_s if j == 0 else max(a, clock)
            clock = start + n / self.sr
            out.append((start, clock))
        return out

    def underruns(self, after_sample: int = 0) -> int:
        """Times the player ran dry. With after_sample=k, only gaps after sample k has been reached count
        (k = the first audible sample: a pause inside an inaudible lead-in is not heard as a gap)."""
        s = self.schedule()
        stop = self.stopped_at if self.stopped_at is not None else float("inf")
        n, pos = 0, 0
        for j in range(len(s)):
            if j and pos > after_sample and self.arrivals[j] > s[j - 1][1] + 1e-6 and s[j - 1][1] < stop:
                n += 1
            pos += self.lengths[j]
        return n

    def play_time(self, k: int) -> float | None:
        """Wall time at which sample k (0-based, over all pieces) is heard; None if never (or after stop)."""
        pos = 0
        for (start, _), n in zip(self.schedule(), self.lengths):
            if k < pos + n:
                t = start + (k - pos) / self.sr
                if self.stopped_at is not None and t >= self.stopped_at:
                    return None
                return t
            pos += n
        return None

    def stop(self, t: float) -> None:
        self.stopped_at = t

    def samples_played(self) -> int:
        """Samples heard before stop (all of them if never stopped)."""
        total = 0
        for (start, end), n in zip(self.schedule(), self.lengths):
            if self.stopped_at is None or end <= self.stopped_at:
                total += n
            elif start < self.stopped_at:
                total += int((self.stopped_at - start) * self.sr)
        return total

    def playing_at(self, t: float) -> bool:
        return any(s <= t < e for s, e in self.schedule())
