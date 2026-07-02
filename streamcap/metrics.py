"""Session metrics: end-to-end caption latency, throughput and inference cost.

The cost baseline is what this session would have cost if every captured
second had been streamed to a cloud STT API, the architecture of a thin
desktop client with no local model. Rate: Deepgram Nova-3 streaming,
pay-as-you-go, $0.0077/min (verified July 2026).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

CLOUD_RATE_PER_MIN = 0.0077  # USD, Deepgram Nova-3 streaming PAYG


@dataclass
class SessionStats:
    started_at: float = field(default_factory=time.monotonic)
    latencies: list[float] = field(default_factory=list)
    display_latencies: list[float] = field(default_factory=list)
    words_committed: int = 0
    audio_seconds: float = 0.0
    cloud_seconds: float = 0.0  # audio routed to a paid cloud API
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

    @property
    def local_pct(self) -> float:
        if self.audio_seconds == 0:
            return 100.0
        return 100.0 * (1.0 - self.cloud_seconds / self.audio_seconds)

    @property
    def cloud_cost(self) -> float:
        return self.cloud_seconds / 60.0 * CLOUD_RATE_PER_MIN

    @property
    def baseline_cost(self) -> float:
        """Cost of this session on a stream-everything thin client."""
        return self.audio_seconds / 60.0 * CLOUD_RATE_PER_MIN

    @property
    def saved(self) -> float:
        return self.baseline_cost - self.cloud_cost

    def snapshot(self) -> dict:
        return {
            "session_s": round(time.monotonic() - self.started_at, 1),
            "audio_s": round(self.audio_seconds, 1),
            "words": self.words_committed,
            "commits": len(self.latencies),
            "display_p50": self._pct(self.display_latencies, 50),
            "display_p95": self._pct(self.display_latencies, 95),
            "commit_p50": self._pct(self.latencies, 50),
            "commit_p95": self._pct(self.latencies, 95),
            "rtf": self.rtf,
            "local_pct": round(self.local_pct, 1),
            "cloud_cost": round(self.cloud_cost, 4),
            "baseline_cost": round(self.baseline_cost, 4),
            "saved": round(self.saved, 4),
            "lat_series": [round(v, 3) for v in self.display_latencies[-150:]],
        }

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
