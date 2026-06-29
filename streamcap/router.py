"""Cost-aware hybrid routing between local Whisper and a cloud streaming API.

Policy (the Hybrid LLM / cost-aware routing idea applied to ASR): serve from
the local model while its rolling word confidence stays healthy; when it
degrades — heavy accents, crosstalk, noise — fail over to the paid cloud
stream so quality never drops. Periodically probe local again so the session
returns to $0 inference as soon as it can. Silence is never billed: the same
RMS gate that protects the local model gates what is sent to the cloud.

Drop-in replacement for StreamingTranscriber from the consumer's point of
view: same feed()/stats/pending_text/detected_language surface.
"""

from __future__ import annotations

import time

import numpy as np

from .asr import Commit, StreamingTranscriber
from .audio import TARGET_RATE, AudioBlock
from .cloud import CloudStream


class HybridRouter:
    def __init__(
        self,
        local: StreamingTranscriber,
        api_key: str,
        mode: str = "auto",  # auto | local | cloud
        low_confidence: float = 0.55,
        high_confidence: float = 0.70,
        min_local_seconds: float = 5.0,
        probe_after_seconds: float = 20.0,
    ):
        self.local = local
        self.mode = mode
        self.low_confidence = low_confidence
        self.high_confidence = high_confidence
        self.min_local_seconds = min_local_seconds
        self.probe_after_seconds = probe_after_seconds

        self._api_key = api_key
        self._cloud: CloudStream | None = None
        self._cloud_since: float | None = None
        self._cloud_retry_at = 0.0  # backoff after a failed cloud session
        self.active = "cloud" if mode == "cloud" else "local"
        if self.active == "cloud":
            self._open_cloud()

    # -- transcriber-compatible surface ------------------------------------

    @property
    def stats(self):
        return self.local.stats

    @property
    def language(self) -> str | None:
        return self.local.language

    @property
    def detected_language(self):
        return self.local.detected_language

    @property
    def pending_text(self) -> str:
        if self.active == "cloud" and self._cloud is not None:
            return self._cloud.interim_text
        return self.local.pending_text

    @property
    def device(self) -> str:
        if self.active == "cloud":
            return "deepgram · cloud"
        return f"{self.local.device} · local"

    @property
    def compute_type(self) -> str:
        return self.local.compute_type

    def set_language(self, language: str | None) -> None:
        if self.active == "cloud":
            self._fall_back_to_local(reason="language change")
        self.local.set_language(language)

    # -- routing ------------------------------------------------------------

    def feed(self, block: AudioBlock) -> list[Commit]:
        if self.active == "local":
            commits = self.local.feed(block)
            self._maybe_fail_over()
            return commits
        return self._feed_cloud(block)

    def _maybe_fail_over(self) -> None:
        if self.mode != "auto" or time.monotonic() < self._cloud_retry_at:
            return
        confidence = self.local.rolling_confidence
        if (
            confidence is not None
            and confidence < self.low_confidence
            and self.local.stats.audio_seconds >= self.min_local_seconds
        ):
            self._open_cloud()

    def _feed_cloud(self, block: AudioBlock) -> list[Commit]:
        stats = self.local.stats
        if stats.audio_seconds == 0.0:
            stats.started_at = time.monotonic()
        seconds = len(block.samples) / TARGET_RATE
        stats.audio_seconds += seconds

        if self._cloud is None or self._cloud.error:
            self._fall_back_to_local(reason="cloud unavailable")
            return []

        # never pay for silence
        if float(np.sqrt(np.mean(block.samples**2))) >= 3e-4:
            self._cloud.send(block)
            stats.cloud_seconds += seconds

        commits: list[Commit] = []
        for result in self._cloud.poll():
            latency = self._cloud.latency_of(result) or 0.0
            n_words = len(result.text.split())
            stats.add_commit(latency, n_words=n_words)
            stats.add_display(latency)
            commits.append(self._to_commit(result, latency))

        if (
            self.mode == "auto"
            and self._cloud_since is not None
            and time.monotonic() - self._cloud_since > self.probe_after_seconds
        ):
            self._fall_back_to_local(reason="probing local again")
        return commits

    def _open_cloud(self) -> None:
        if self._cloud is None:
            self._cloud = CloudStream(self._api_key, language=self.local.language)
            self._cloud_since = time.monotonic()
            self.active = "cloud"
            # the local model stops being fed; drop its buffer so a later
            # probe starts fresh instead of committing stale audio
            self.local._drop_buffer()

    def _fall_back_to_local(self, reason: str) -> None:
        if self._cloud is not None:
            if self._cloud.error:
                # bad key, network down, quota — don't hammer the endpoint on
                # every confidence dip; local keeps captioning meanwhile
                self._cloud_retry_at = time.monotonic() + 60.0
            self._cloud.close()
            self._cloud = None
        self._cloud_since = None
        self.active = "local"
        self.local.reset_confidence()

    def _to_commit(self, result, latency: float) -> Commit:
        """Map a cloud result's stream offsets onto the session clock."""
        t0 = self.local.stats.started_at
        wall_start = self._cloud._wall_time_at(result.start) if self._cloud else None
        wall_end = self._cloud._wall_time_at(result.end) if self._cloud else None
        return Commit(
            text=result.text,
            latency=latency,
            language=self.detected_language,
            start=max(0.0, (wall_start or t0) - t0),
            end=max(0.0, (wall_end or t0) - t0),
        )

    def drain(self, seconds: float = 3.0) -> list[Commit]:
        """Flush the cloud stream and collect its remaining final results."""
        commits: list[Commit] = []
        if self._cloud is not None:
            self._cloud.finish()  # CloseStream makes the server emit pending finals
        deadline = time.monotonic() + seconds
        while self._cloud is not None and time.monotonic() < deadline:
            for result in self._cloud.poll():
                latency = self._cloud.latency_of(result) or 0.0
                self.local.stats.add_commit(latency, n_words=len(result.text.split()))
                commits.append(self._to_commit(result, latency))
            time.sleep(0.2)
        self.close()
        return commits

    def close(self) -> None:
        if self._cloud is not None:
            self._cloud.close()
            self._cloud = None
