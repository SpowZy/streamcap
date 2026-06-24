"""Streaming transcription on top of faster-whisper.

Whisper is an offline model; this module makes it behave like a streaming one
using the LocalAgreement policy from the simultaneous-ASR literature: the
rolling audio window is re-transcribed as audio arrives, and only the longest
common prefix between two consecutive hypotheses is committed as final text.
Unstable tail words are held back, which is what keeps captions from
flickering or rewriting themselves.

References worth reading on this family of approaches: WhisperRT
(arXiv:2508.12301), Simul-Whisper, and the UFAL whisper_streaming work.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from .audio import TARGET_RATE, AudioBlock
from .metrics import SessionStats

_WORD_STRIP = re.compile(r"[^\w]+", re.UNICODE)

SILENCE_RMS = 3e-4  # below this, a block is treated as silence
GAP_RESET_SECONDS = 1.0  # missing wall time that marks a stream discontinuity


def _norm(word: str) -> str:
    cleaned = _WORD_STRIP.sub("", word).lower()
    return cleaned or word.strip().lower()


@dataclass
class Hyp:
    raw: str
    norm: str
    end: float  # seconds from buffer start
    start: float = 0.0


@dataclass
class Commit:
    text: str
    latency: float  # seconds from audio playback to commit
    language: str | None
    start: float = 0.0  # session-relative start time of this caption (s)
    end: float = 0.0  # session-relative end time (s)


class StreamingTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        language: str | None = None,
        window_seconds: float = 8.0,
        step_seconds: float = 0.6,
        min_first_seconds: float = 1.0,
        endpoint_seconds: float = 0.7,
        unlock_seconds: float = 2.5,
        vad_filter: bool = False,
    ):
        # vad_filter costs ~110 ms per call (measured, 8 s window on GPU) and the
        # RMS silence gate + hypothesis agreement already suppress most
        # silence hallucinations, so it stays off by default.
        self.vad_filter = vad_filter
        self.language = language
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.min_first_seconds = min_first_seconds
        self.endpoint_seconds = endpoint_seconds
        self.unlock_seconds = unlock_seconds
        self.stats = SessionStats()
        # Whisper's per-window language detection flaps on short buffers, which
        # destabilizes hypotheses and stalls agreement. Lock the language once
        # detected with confidence; a long silence unlocks it so the session
        # can follow a switch to different-language content.
        self._locked_language: str | None = language
        self.detected_language: str | None = language
        self.pending_text: str = ""

        self.device, self.compute_type, self._model = self._load_model(model_size, device)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start: float | None = None  # monotonic time of buffer[0]
        self._committed_end = 0.0  # seconds into buffer already final
        self._pending: list[Hyp] = []  # last hypothesis tail, not yet agreed
        self._new_audio = 0.0  # seconds accumulated since last step
        self._last_commit: Hyp | None = None  # guards against boundary re-emission
        self._word_probs: deque[float] = deque(maxlen=80)  # router's quality signal
        self._session_t0: float | None = None  # wall time of the session's first sample
        self._lang_candidate: str | None = None  # detection must agree twice to lock
        self._silence_run = 0.0  # consecutive seconds of silent input blocks
        self._voiced_new = 0.0  # voiced seconds since the last decode
        self._decode_ema: float | None = None  # smoothed decode wall time
        self._lang_audit_due = 0.0  # audio_seconds mark of the next audit
        self._lang_disagreements = 0  # consecutive audits contradicting the lock

    @staticmethod
    def _load_model(model_size: str, device: str):
        from faster_whisper import WhisperModel

        from .gpu import prepare_cuda_dlls

        candidates = []
        if device in ("auto", "cuda"):
            candidates.append(("cuda", "float16"))
        if device in ("auto", "cpu"):
            candidates.append(("cpu", "int8"))

        last_error: Exception | None = None
        for dev, compute in candidates:
            try:
                if dev == "cuda":
                    prepare_cuda_dlls()
                model = WhisperModel(model_size, device=dev, compute_type=compute)
                # transcribe() is lazy — consuming the generator forces CUDA
                # init and the first kernel launch to fail here, not mid-session
                segments, _ = model.transcribe(np.zeros(TARGET_RATE, dtype=np.float32), beam_size=1)
                for _ in segments:
                    pass
                return dev, compute, model
            except Exception as e:  # noqa: BLE001 - fall through to next backend
                last_error = e
        raise RuntimeError(f"could not load whisper model on any backend: {last_error}")

    def feed(self, block: AudioBlock) -> list[Commit]:
        if self.stats.audio_seconds == 0.0:
            self.stats.started_at = time.monotonic()  # exclude model load time
            self._session_t0 = block.captured_at
        seconds = len(block.samples) / TARGET_RATE
        self.stats.audio_seconds += seconds

        commits: list[Commit] = []

        # Stream discontinuity: paused sources, loopback stalls and queue
        # overflows leave a hole in wall time that the buffer knows nothing
        # about. Without a resync, every commit after the hole inherits it as
        # phantom latency (a 2-minute pause reads as a 120 s p50).
        if self._buffer_start is not None:
            expected_end = self._buffer_start + len(self._buffer) / TARGET_RATE
            if block.captured_at - expected_end > GAP_RESET_SECONDS:
                # record=False: this text was captured before a pause we only
                # learn about now — its promotion delay is not pipeline latency
                commits += self._flush_tail(record=False)
                self._drop_buffer()
                self._unlock_language()  # a source switch often means a new language
                self._buffer_start = None
                self._silence_run = 0.0

        voiced = bool(len(block.samples)) and float(np.sqrt(np.mean(block.samples**2))) >= SILENCE_RMS
        self._silence_run = 0.0 if voiced else self._silence_run + seconds

        # Endpoint: input has gone quiet, so no further audio can improve the
        # pending hypothesis. Promote it to final (instead of letting it be
        # trimmed away) and decode nothing until speech returns — silence is
        # never re-decoded, and never hallucinated over. Short pauses keep
        # the audio window (language detection and the wall-clock map need
        # the context); only a long silence resets it and unlocks the
        # language for a possible source switch.
        if self._silence_run >= self.endpoint_seconds:
            commits += self._flush_tail(record=True)
            if self._silence_run >= self.unlock_seconds:
                self._drop_buffer()
                self._buffer_start = None
                self._unlock_language()
                return commits
            if self._locked_language is not None:
                # utterance done and the language is settled: reset the window
                # so stale silence never pollutes word alignment and the next
                # decodes stay small
                self._drop_buffer()
                self._buffer_start = None
                return commits
            # still detecting: keep buffering through the short pause so
            # detection gets a window of usable length (>= 3 s)
            if self._buffer_start is None:
                self._buffer_start = block.captured_at
            self._buffer = np.concatenate([self._buffer, block.samples])
            self._new_audio += seconds
            self._trim()
            return commits

        if self._buffer_start is None:
            self._buffer_start = block.captured_at
        self._buffer = np.concatenate([self._buffer, block.samples])
        self._new_audio += seconds
        if voiced:
            self._voiced_new += seconds

        buffered = len(self._buffer) / TARGET_RATE
        if buffered < self.min_first_seconds or self._new_audio < self._effective_step():
            return commits
        self._new_audio = 0.0
        self._voiced_new = 0.0

        return commits + self._step()

    def _effective_step(self) -> float:
        """Decode cadence, self-regulating under load.

        If a decode costs more wall time than the configured step (thermal
        throttling, CPU-only machines, a busy GPU), decoding at the configured
        cadence would fall behind real time, overflow the capture queue and
        drop audio. Stretch the step to stay ahead; latency degrades
        gracefully instead of the pipeline collapsing.
        """
        if self._decode_ema is None:
            return self.step_seconds
        return min(max(self.step_seconds, 1.3 * self._decode_ema), 2.5)

    @property
    def rolling_confidence(self) -> float | None:
        """Mean word probability over the recent window; None until warmed up.

        Averaged over the newest 40 words rather than the whole history so a
        mid-session quality drop (source switched language, heavy accent) is
        detected in seconds instead of being diluted by minutes of good audio.
        """
        if len(self._word_probs) < 20:
            return None
        return float(np.mean(list(self._word_probs)[-40:]))

    def reset_confidence(self) -> None:
        self._word_probs.clear()

    def set_language(self, language: str | None) -> None:
        """Pin (or unpin, with None) the language and restart the stream state.

        Switching audio sources mid-session rarely leaves a silence long
        enough to trigger the automatic unlock, so this is the deterministic
        path: drop the buffer, clear the hypothesis and the confidence
        window, and decode fresh in the requested language.
        """
        self.language = language
        self._drop_buffer()
        self._locked_language = language
        self.detected_language = language
        self._lang_candidate = None
        self._lang_disagreements = 0
        self._lang_audit_due = self.stats.audio_seconds + 3.0
        self._last_commit = None
        self.reset_confidence()

    def _drop_buffer(self) -> None:
        seconds = len(self._buffer) / TARGET_RATE
        if self._buffer_start is not None:
            self._buffer_start += seconds
        self._buffer = np.zeros(0, dtype=np.float32)
        self._committed_end = 0.0
        self._pending = []
        self.pending_text = ""
        self._new_audio = 0.0
        self._voiced_new = 0.0
        # boundary dedup only makes sense within a contiguous buffer; keeping
        # a stale marker here would eat a legitimately repeated word
        self._last_commit = None

    def _unlock_language(self) -> None:
        if self.language is None:  # only when the session is in auto-detect
            self._locked_language = None
            self._lang_candidate = None

    def _flush_tail(self, record: bool) -> list[Commit]:
        """Decode any not-yet-decoded audio, then promote the whole pending
        hypothesis to final text.

        Called at utterance boundaries (endpoint, stream gap) where no more
        audio will arrive for this buffer: the freshest hypothesis IS the
        final answer, and waiting for a second agreement pass would either
        stall or silently eat the end of the sentence. `record=False` keeps
        after-the-fact promotions (a resumed pause) out of the latency
        percentiles — that delay measures the pause, not the pipeline.
        """
        if self._buffer_start is None:
            self._pending = []
            self.pending_text = ""
            return []
        # short utterances never reach min_first_seconds; without this final
        # decode their words would be dropped without ever being transcribed
        if self._voiced_new > 0.0 and len(self._buffer) >= int(0.3 * TARGET_RATE):
            self._new_audio = 0.0
            self._voiced_new = 0.0
            current = [w for w in self._transcribe() if w.end > self._committed_end + 0.05]
            while (
                current
                and self._last_commit is not None
                and current[0].norm == self._last_commit.norm
                and current[0].end <= self._committed_end + 0.4
            ):
                current.pop(0)
            self._pending = current

        words, self._pending = self._pending, []
        self.pending_text = ""
        if not words:
            return []
        text = "".join(w.raw for w in words).strip()
        if not text:
            return []
        # word timings can overshoot the audio we actually hold (Whisper pads
        # its window); measure against real samples, not hallucinated ones
        buffered = len(self._buffer) / TARGET_RATE
        latency = max(0.0, time.monotonic() - (self._buffer_start + min(words[-1].end, buffered)))
        if record:
            self.stats.add_commit(latency, n_words=len(words))
        self._committed_end = words[-1].end
        self._last_commit = words[-1]
        t0 = self._session_t0 if self._session_t0 is not None else self._buffer_start
        return [
            Commit(
                text=text,
                latency=latency if record else 0.0,
                language=self.detected_language,
                start=max(0.0, self._buffer_start + words[0].start - t0),
                end=max(0.0, self._buffer_start + words[-1].end - t0),
            )
        ]

    def _transcribe(self) -> list[Hyp]:
        t0 = time.monotonic()
        segments, info = self._model.transcribe(
            self._buffer,
            language=self._locked_language,
            beam_size=1,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=self.vad_filter,
        )
        words: list[Hyp] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Hyp(raw=w.word, norm=_norm(w.word), end=w.end, start=w.start))
                if w.probability is not None:
                    self._word_probs.append(w.probability)
        decode_seconds = time.monotonic() - t0
        self.stats.transcribe_calls += 1
        self.stats.transcribe_seconds += decode_seconds
        ema = self._decode_ema
        self._decode_ema = decode_seconds if ema is None else 0.3 * decode_seconds + 0.7 * ema
        buffered = len(self._buffer) / TARGET_RATE
        if self._locked_language is None and info.language_probability and buffered >= 3.0:
            # music beds and jingles fool short-window detection (Spanish on
            # an intro sting is a classic) — require a window of at least 3 s
            # AND two consecutive windows agreeing before locking
            if info.language_probability > 0.8:
                if info.language == self._lang_candidate:
                    self._locked_language = info.language
                self._lang_candidate = info.language
            else:
                self._lang_candidate = None
        self.detected_language = self._locked_language or info.language
        return words

    def _step(self) -> list[Commit]:
        current = [w for w in self._transcribe() if w.end > self._committed_end + 0.05]

        # Word timings shift slightly between decodes; if the last committed
        # word reappears just past the committed boundary, drop the duplicate.
        while (
            current
            and self._last_commit is not None
            and current[0].norm == self._last_commit.norm
            and current[0].end <= self._committed_end + 0.4
        ):
            current.pop(0)

        # LocalAgreement: commit the prefix both consecutive hypotheses agree on.
        agreed = 0
        for prev, cur in zip(self._pending, current):
            if prev.norm != cur.norm:
                break
            agreed += 1

        commits: list[Commit] = []
        if agreed:
            committed = current[:agreed]
            text = "".join(w.raw for w in committed).strip()
            self._committed_end = committed[-1].end
            self._last_commit = committed[-1]
            # word timings can overshoot the audio we actually hold (Whisper
            # pads its window); measure against real samples
            buffered = len(self._buffer) / TARGET_RATE
            latency = max(
                0.0, time.monotonic() - (self._buffer_start + min(self._committed_end, buffered))
            )
            self.stats.add_commit(latency, n_words=agreed)
            t0 = self._session_t0 if self._session_t0 is not None else self._buffer_start
            commits.append(
                Commit(
                    text=text,
                    latency=latency,
                    language=self.detected_language,
                    start=max(0.0, self._buffer_start + committed[0].start - t0),
                    end=max(0.0, self._buffer_start + committed[-1].end - t0),
                )
            )

        # A wrong-language lock sometimes shows up as a confidence collapse
        # (noise, crosstalk). Re-detect before anything more expensive gets a
        # say — this must run ahead of the router.
        confidence = self.rolling_confidence
        if (
            self.language is None
            and self._locked_language is not None
            and confidence is not None
            and confidence < 0.55
        ):
            self.set_language(None)
            return commits

        # But a language switch usually does NOT collapse token confidence:
        # Whisper forced to the old language covertly *translates* the new
        # one, fluently and confidently. The encoder's language posterior is
        # the signal that can't be fooled that way — audit it periodically.
        if self._audit_language():
            return commits

        self._pending = current[agreed:]
        self.pending_text = "".join(w.raw for w in self._pending).strip()
        if current:
            # freshness of the newest word on screen (committed or pending)
            buffered = len(self._buffer) / TARGET_RATE
            display_latency = max(
                0.0, time.monotonic() - (self._buffer_start + min(current[-1].end, buffered))
            )
            self.stats.add_display(display_latency)
        self._trim()
        return commits

    def _audit_language(self) -> bool:
        """Every few seconds, check the encoder's language posterior against
        the locked language; two consecutive contradictions break the lock.

        Returns True when the lock was just broken (callers should stop the
        current step: the buffer has been reset).
        """
        if (
            self.language is not None  # pinned by the user: never audit
            or self._locked_language is None
            or self.stats.audio_seconds < self._lang_audit_due
            or len(self._buffer) < 3 * TARGET_RATE
        ):
            return False
        self._lang_audit_due = self.stats.audio_seconds + 3.0
        t0 = time.monotonic()
        language, probability, _ = self._model.detect_language(self._buffer)
        self.stats.transcribe_seconds += time.monotonic() - t0  # honest RTF
        if language != self._locked_language and probability > 0.8:
            self._lang_disagreements += 1
            # an unambiguous posterior breaks the lock at once; a marginal
            # one needs a second opinion 3 s later
            if probability >= 0.95 or self._lang_disagreements >= 2:
                # unlock IN PLACE: the buffered foreign-language audio is both
                # the detection context for the re-lock and speech the user
                # hasn't seen committed yet — dropping it would eat a phrase
                self._locked_language = None
                self._lang_candidate = None
                self._lang_disagreements = 0
                self._pending = []  # hypotheses from the stale language
                self.pending_text = ""
                self.reset_confidence()
                return True
        else:
            self._lang_disagreements = 0
        return False

    def _trim(self) -> None:
        buffered = len(self._buffer) / TARGET_RATE
        if buffered <= self.window_seconds:
            return
        cut = self._committed_end
        if cut < 1.0:
            # nothing stable to cut at (e.g. music): bound compute anyway, but
            # keep the shifted hypothesis so hard-to-agree words still get a
            # chance to commit instead of silently vanishing off screen
            cut = buffered - self.window_seconds
        samples = int(cut * TARGET_RATE)
        self._buffer = self._buffer[samples:]
        self._buffer_start += cut
        self._committed_end = max(0.0, self._committed_end - cut)
        self._pending = [
            Hyp(raw=w.raw, norm=w.norm, end=w.end - cut, start=w.start - cut)
            for w in self._pending
            if w.end - cut > 0
        ]
        if self._last_commit is not None:
            self._last_commit = Hyp(
                raw=self._last_commit.raw,
                norm=self._last_commit.norm,
                end=self._last_commit.end - cut,
                start=self._last_commit.start - cut,
            )
