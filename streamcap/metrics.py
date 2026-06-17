"""Session metrics: end-to-end caption latency and throughput."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SessionStats:
    started_at: float = field(default_factory=time.monotonic)
    latencies: list[float] = field(default_factory=list)
    display_latencies: list[float] = field(default_factory=list)
    words_committed: int = 0
    audio_seconds: float = 0.0
    transcribe_calls: int = 0
    transcribe_seconds: float = 0.0

    def add_commit(self, latency: float, n_words: int) -> None:
        self.latencies.append(latency)
        self.words_committed += n_words

    def add_display(self, latency: float) -> None:
        self.display_latencies.append(latency)

    @staticmethod
    def _pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        return float(np.percentile(values, p))

    def percentile(self, p: float) -> float | None:
        return self._pct(self.latencies, p)

    @property
    def rtf(self) -> float | None:
        """Real-time factor of ASR compute: < 1.0 means faster than the audio."""
        if self.audio_seconds == 0:
            return None
        return self.transcribe_seconds / self.audio_seconds

    def summary(self) -> str:
        dur = time.monotonic() - self.started_at
        p50, p95 = self.percentile(50), self.percentile(95)
        parts = [
            f"session {dur:.0f}s",
            f"audio {self.audio_seconds:.0f}s",
            f"words {self.words_committed}",
            f"commits {len(self.latencies)}",
        ]
        if p50 is not None:
            parts.append(f"commit latency p50 {p50:.2f}s p95 {p95:.2f}s")
        d50 = self._pct(self.display_latencies, 50)
        d95 = self._pct(self.display_latencies, 95)
        if d50 is not None:
            parts.append(f"display latency p50 {d50:.2f}s p95 {d95:.2f}s")
        if self.rtf is not None:
            parts.append(f"rtf {self.rtf:.2f}")
        return " | ".join(parts)
