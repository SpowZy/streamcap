"""Deterministic tests for the streaming core: agreement, endpointing,
gap resync, language locking, trimming, routing, metrics and export.

A fake Whisper model is injected so the streaming logic is tested exactly,
without GPU or network. The fake emits one word per 0.5 s of voiced audio at
stable timestamps, which is what LocalAgreement needs to converge.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamcap.asr import Commit, StreamingTranscriber
from streamcap.audio import TARGET_RATE, AudioBlock
from streamcap.export import write_srt, write_vtt
from streamcap.metrics import CLOUD_RATE_PER_MIN, SessionStats

BLOCK = 0.25  # seconds per fed block
WORD_EVERY = 0.5  # fake model: one word per half second of voiced audio


class FakeModel:
    """Emits words w0, w1, ... at fixed offsets for the voiced prefix of the
    buffer, mimicking faster_whisper's (segments, info) surface. The word
    prefix is derived from the audio amplitude (0.1 -> w, 0.2 -> x) so tests
    can tell utterances apart across resets without external state."""

    PREFIXES = {1: "w", 2: "x"}

    def __init__(self, language="en", language_probability=0.95, probability=0.9):
        self.language = language
        self.language_probability = language_probability
        self.probability = probability
        self.word_offset = 0  # lets tests relabel words across utterances
        self.calls = 0

    def detect_language(self, buffer):
        return self.language, self.language_probability, []

    def transcribe(self, buffer, **kwargs):
        self.calls += 1
        words = []
        # position words inside each contiguous voiced run at its true buffer
        # offset, exactly like real word timestamps behave
        mask = np.abs(buffer) > 1e-3
        idx = np.flatnonzero(mask)
        runs = []
        if idx.size:
            splits = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate(([idx[0]], idx[splits + 1]))
            ends = np.concatenate((idx[splits], [idx[-1]])) + 1
            runs = list(zip(starts / TARGET_RATE, ends / TARGET_RATE))
        for run_start, run_end in runs:
            amp = float(np.max(np.abs(buffer[int(run_start * TARGET_RATE): int(run_end * TARGET_RATE)])))
            prefix = self.PREFIXES.get(int(round(amp / 0.1)), "y")
            i = 0
            while (i + 1) * WORD_EVERY <= (run_end - run_start) + 0.1:
                words.append(
                    SimpleNamespace(
                        word=f" {prefix}{i + self.word_offset}",
                        start=run_start + i * WORD_EVERY,
                        end=run_start + i * WORD_EVERY + 0.4,
                        probability=self.probability,
                    )
                )
                i += 1
        info = SimpleNamespace(
            language=self.language, language_probability=self.language_probability
        )
        segment = SimpleNamespace(words=words)

        def gen():
            yield segment

        return gen(), info


def make_transcriber(fake=None, language=None, **kwargs) -> StreamingTranscriber:
    fake = fake or FakeModel()
    original = StreamingTranscriber._load_model
    StreamingTranscriber._load_model = staticmethod(lambda size, device: ("cpu", "int8", fake))
    try:
        t = StreamingTranscriber(model_size="fake", device="cpu", language=language, **kwargs)
    finally:
        StreamingTranscriber._load_model = original
    t._fake = fake
    return t


def blocks(seconds: float, t0: float, voiced: bool = True):
    """Yield AudioBlocks of BLOCK seconds starting at wall time t0."""
    n = int(round(seconds / BLOCK))
    samples = int(BLOCK * TARGET_RATE)
    for i in range(n):
        data = (np.ones(samples, dtype=np.float32) * 0.1) if voiced else np.zeros(samples, dtype=np.float32)
        yield AudioBlock(samples=data, captured_at=t0 + i * BLOCK)


def feed_all(t, block_iter) -> list[Commit]:
    out = []
    for b in block_iter:
        out += t.feed(b)
    return out


def committed_words(commits: list[Commit]) -> list[str]:
    return " ".join(c.text for c in commits).split()


# -- streaming core ---------------------------------------------------------


def test_agreement_commits_and_no_duplicates():
    t = make_transcriber()
    t0 = time.monotonic() - 10.0
    commits = feed_all(t, blocks(4.0, t0))
    words = committed_words(commits)
    assert words == sorted(set(words), key=lambda w: int(w[1:])), f"duplicated or reordered: {words}"
    assert len(words) >= 4  # the stable prefix must have committed


def test_endpoint_finalizes_pending_no_words_lost():
    """Speech then silence: every produced word must be committed, and the
    pending (violet) tail must be promoted, not eaten."""
    t = make_transcriber()
    t0 = time.monotonic() - 20.0
    commits = feed_all(t, blocks(3.0, t0))  # 6 words: w0..w5
    commits += feed_all(t, blocks(2.0, t0 + 3.0, voiced=False))
    words = committed_words(commits)
    assert words == [f"w{i}" for i in range(6)], f"words lost or duplicated: {words}"
    assert t.pending_text == ""


def test_endpoint_stops_decoding_during_silence():
    t = make_transcriber()
    t0 = time.monotonic() - 60.0
    feed_all(t, blocks(3.0, t0))
    feed_all(t, blocks(1.0, t0 + 3.0, voiced=False))
    calls = t._fake.calls
    feed_all(t, blocks(30.0, t0 + 4.0, voiced=False))
    assert t._fake.calls == calls, "silence must not be re-decoded"


def test_short_utterance_below_min_first_is_not_lost():
    """'Yes.' (0.5 s) then silence: shorter than min_first_seconds but must
    still be transcribed via the endpoint flush."""
    t = make_transcriber()
    t0 = time.monotonic() - 10.0
    commits = feed_all(t, blocks(0.5, t0))
    commits += feed_all(t, blocks(1.5, t0 + 0.5, voiced=False))
    assert committed_words(commits) == ["w0"]


class Clock:
    """Controllable monotonic clock: audio pacing without real sleeps."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def paced_feed(t, clock: Clock, seconds: float, voiced: bool = True, amp: float = 0.1):
    """Feed blocks in lockstep with the fake clock, like a live capture."""
    out = []
    samples = int(BLOCK * TARGET_RATE)
    for _ in range(int(round(seconds / BLOCK))):
        clock.t += BLOCK
        data = (np.ones(samples, dtype=np.float32) * amp) if voiced else np.zeros(samples, dtype=np.float32)
        out += t.feed(AudioBlock(samples=data, captured_at=clock.t - BLOCK))
    return out


def test_gap_resync_kills_phantom_latency(monkeypatch):
    """The 107 s bug: a paused source must not inflate later latencies."""
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    t = make_transcriber()
    commits = paced_feed(t, clock, 3.0)  # w0..w5
    clock.t += 120.0  # user pauses the video; no blocks arrive at all
    commits += paced_feed(t, clock, 3.0, amp=0.2)  # x0..x5
    commits += paced_feed(t, clock, 1.5, voiced=False)
    post_gap = [c for c in commits if "x" in c.text]
    assert post_gap, "no commits after the gap"
    worst = max(c.latency for c in post_gap)
    assert worst < 5.0, f"phantom latency survived the gap: {worst:.1f}s"
    # and the stats percentiles must not be poisoned either
    assert t.stats.percentile(95) < 5.0, f"p95 poisoned: {t.stats.percentile(95):.1f}s"


def test_gap_promotes_pending_tail(monkeypatch):
    """Words still pending when the source pauses must be promoted, not lost."""
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    t = make_transcriber()
    commits = paced_feed(t, clock, 3.0)  # w0..w5
    clock.t += 120.0
    commits += paced_feed(t, clock, 2.0, amp=0.2)
    words = committed_words(commits)
    assert [w for w in words if w.startswith("w")] == [f"w{i}" for i in range(6)], (
        f"pre-gap words lost: {words}"
    )


def test_language_two_window_lock_resists_jingle():
    """A music sting misdetected as Spanish must not lock the session."""
    fake = FakeModel(language="es")
    t = make_transcriber(fake)
    t0 = time.monotonic() - 30.0
    gen = blocks(20.0, t0)
    t.feed(next(gen))
    t.feed(next(gen))
    t.feed(next(gen))
    t.feed(next(gen))
    t.feed(next(gen))  # first decode: candidate es, not locked
    assert t._locked_language is None
    fake.language = "en"
    for b in gen:
        t.feed(b)
    assert t._locked_language == "en"


def test_long_silence_unlocks_language():
    t = make_transcriber()
    t0 = time.monotonic() - 30.0
    # locking needs two agreeing detection windows of >= 3 s each
    feed_all(t, blocks(6.0, t0))
    assert t._locked_language == "en"
    feed_all(t, blocks(1.0, t0 + 6.0, voiced=False))
    assert t._locked_language == "en", "short pause must keep the lock"
    feed_all(t, blocks(3.0, t0 + 7.0, voiced=False))
    assert t._locked_language is None, "long silence must unlock"


def test_language_audit_breaks_covert_translation_lock():
    """A language switch with no silence gap does not collapse token
    confidence (Whisper just translates, fluently); the encoder posterior
    audit must break the stale lock anyway."""
    t = make_transcriber()
    t0 = time.monotonic() - 60.0
    feed_all(t, blocks(6.0, t0))
    assert t._locked_language == "en"
    t._fake.language = "fr"  # the source switched; words still decode confidently
    feed_all(t, blocks(14.0, t0 + 6.0))
    assert t._locked_language == "fr", "stale lock survived the language audit"


def test_pinned_language_never_unlocks():
    t = make_transcriber(language="fr")
    t0 = time.monotonic() - 30.0
    feed_all(t, blocks(3.0, t0))
    feed_all(t, blocks(5.0, t0 + 3.0, voiced=False))
    assert t._locked_language == "fr"


def test_forced_trim_keeps_shifted_pending():
    """Un-agreeable audio (music): trimming must shift the hypothesis, not
    wipe it, so on-screen words still get a chance to commit."""
    t = make_transcriber()
    t._buffer = np.ones(int(9.0 * TARGET_RATE), dtype=np.float32) * 0.1
    t._buffer_start = time.monotonic() - 9.0
    t._committed_end = 0.0
    from streamcap.asr import Hyp

    t._pending = [Hyp(raw=" tail", norm="tail", end=8.5, start=8.1)]
    t._trim()
    assert len(t._buffer) / TARGET_RATE == pytest.approx(8.0, abs=0.01)
    assert t._pending and t._pending[0].end == pytest.approx(7.5, abs=0.01)


def test_repeated_word_across_utterances_not_deduped():
    """'Yes.' pause 'Yes.': the second one must survive the boundary dedup."""
    t = make_transcriber()
    t0 = time.monotonic() - 20.0
    commits = feed_all(t, blocks(0.5, t0))
    commits += feed_all(t, blocks(1.5, t0 + 0.5, voiced=False))
    t._fake.word_offset = 0  # same word again
    commits += feed_all(t, blocks(0.5, t0 + 2.0))
    commits += feed_all(t, blocks(1.5, t0 + 2.5, voiced=False))
    assert committed_words(commits) == ["w0", "w0"]


def test_set_language_resets_stream():
    t = make_transcriber()
    t0 = time.monotonic() - 20.0
    feed_all(t, blocks(3.0, t0))
    t.set_language("fr")
    assert t.pending_text == ""
    assert t._locked_language == "fr"
    assert len(t._buffer) == 0


# -- metrics ----------------------------------------------------------------


def test_metrics_costs_and_percentiles():
    s = SessionStats()
    s.audio_seconds = 600.0
    s.cloud_seconds = 60.0
    assert s.local_pct == pytest.approx(90.0)
    assert s.cloud_cost == pytest.approx(1 * CLOUD_RATE_PER_MIN)
    assert s.baseline_cost == pytest.approx(10 * CLOUD_RATE_PER_MIN)
    assert s.saved == pytest.approx(9 * CLOUD_RATE_PER_MIN)
    for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
        s.add_commit(v, n_words=2)
    assert s.percentile(50) == pytest.approx(0.6)
    assert s.words_committed == 10
    snap = s.snapshot()
    assert snap["commit_p50"] == pytest.approx(0.6)
    assert snap["local_pct"] == pytest.approx(90.0)


# -- export -----------------------------------------------------------------


def test_export_formats_and_sorting(tmp_path):
    from streamcap.export import save_session

    transcript = [
        {"start": 5.0, "end": 6.2, "text": "second"},
        {"start": 1.0, "end": 2.5, "text": "first"},
    ]
    out = save_session(transcript, directory=str(tmp_path))
    srt = (out / "captions.srt").read_text(encoding="utf-8")
    assert srt.index("first") < srt.index("second"), "SRT must be time-ordered"
    assert "00:00:01,000 --> 00:00:02,500" in srt
    vtt = (out / "captions.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")
    assert "00:00:05.000 --> 00:00:06.200" in vtt
    txt = (out / "transcript.txt").read_text(encoding="utf-8")
    assert txt.splitlines() == ["first", "second"]


def test_export_empty_session_writes_nothing(tmp_path):
    from streamcap.export import save_session

    assert save_session([], directory=str(tmp_path)) is None
    assert not list(tmp_path.iterdir())


# -- hybrid router ----------------------------------------------------------


class FakeCloud:
    instances: list["FakeCloud"] = []

    def __init__(self, api_key, language=None):
        FakeCloud.instances.append(self)
        self.language = language
        self.interim_text = ""
        self.error = None
        self.sent_blocks = 0
        self.closed = False
        self._results = []

    def send(self, block):
        self.sent_blocks += 1

    def poll(self):
        out, self._results = self._results, []
        return out

    def latency_of(self, result):
        return 0.3

    def _wall_time_at(self, offset):
        return time.monotonic()

    def finish(self):
        pass

    def close(self):
        self.closed = True


@pytest.fixture()
def router_env(monkeypatch):
    import streamcap.router as router_mod

    FakeCloud.instances = []
    monkeypatch.setattr(router_mod, "CloudStream", FakeCloud)
    local = make_transcriber()
    r = router_mod.HybridRouter(local, api_key="test", mode="auto", min_local_seconds=0.0)
    return r, local


def voiced_block(t0=None):
    samples = int(BLOCK * TARGET_RATE)
    return AudioBlock(
        samples=np.ones(samples, dtype=np.float32) * 0.1,
        captured_at=(t0 or time.monotonic()) - BLOCK,
    )


def silent_block():
    samples = int(BLOCK * TARGET_RATE)
    return AudioBlock(samples=np.zeros(samples, dtype=np.float32), captured_at=time.monotonic() - BLOCK)


def test_router_fails_over_on_low_confidence(router_env):
    r, local = router_env
    local._word_probs.extend([0.30] * 40)
    r.feed(voiced_block())
    assert r.active == "cloud"
    assert len(FakeCloud.instances) == 1


def test_router_falls_back_on_cloud_error_with_cooldown(router_env):
    r, local = router_env
    local._word_probs.extend([0.30] * 40)
    r.feed(voiced_block())
    assert r.active == "cloud"
    FakeCloud.instances[0].error = "auth failed"
    r.feed(voiced_block())
    assert r.active == "local"
    assert FakeCloud.instances[0].closed
    # low confidence again, inside the cooldown: must NOT reopen the cloud
    local._word_probs.extend([0.30] * 40)
    r.feed(voiced_block())
    assert r.active == "local"
    assert len(FakeCloud.instances) == 1, "reconnect storm: cloud reopened during cooldown"


def test_router_never_bills_silence(router_env):
    r, local = router_env
    local._word_probs.extend([0.30] * 40)
    r.feed(voiced_block())
    assert r.active == "cloud"
    cloud = FakeCloud.instances[-1]
    sent_before = cloud.sent_blocks
    billed_before = r.stats.cloud_seconds
    r.feed(silent_block())
    assert cloud.sent_blocks == sent_before
    assert r.stats.cloud_seconds == billed_before


def test_router_exposes_full_transcriber_surface(router_env):
    """The overlay's F8/F7 handlers read these off whichever engine they get;
    a missing passthrough kills the caption worker mid-session."""
    r, local = router_env
    assert r.language is None
    r.set_language("fr")  # what F8 (clear) does with the current language
    assert r.language == "fr"
    assert local.language == "fr"
    assert r.detected_language == "fr"
    assert isinstance(r.pending_text, str)
    assert r.compute_type == local.compute_type


def test_router_probes_back_to_local(router_env):
    r, local = router_env
    local._word_probs.extend([0.30] * 40)
    r.feed(voiced_block())
    assert r.active == "cloud"
    # rewind the clock instead of a probe_after of 0.0: Windows monotonic
    # granularity can make two feeds land on the same tick
    r._cloud_since = time.monotonic() - (r.probe_after_seconds + 1.0)
    r.feed(voiced_block())  # probe window elapsed -> back to local
    assert r.active == "local"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
