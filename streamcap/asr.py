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


def _norm(word: str) -> str:
    cleaned = _WORD_STRIP.sub("", word).lower()
    return cleaned or word.strip().lower()


@dataclass
class Hyp:
    raw: str
    norm: str
    end: float  # seconds from buffer start


@dataclass
class Commit:
    text: str
    latency: float  # seconds from audio playback to commit
    language: str | None


class StreamingTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        language: str | None = None,
        window_seconds: float = 8.0,
        step_seconds: float = 0.6,
        min_first_seconds: float = 1.0,
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
                # force lazy CUDA init to fail here, not mid-session
                model.transcribe(np.zeros(TARGET_RATE, dtype=np.float32), beam_size=1)
                return dev, compute, model
            except Exception as e:  # noqa: BLE001 - fall through to next backend
                last_error = e
        raise RuntimeError(f"could not load whisper model on any backend: {last_error}")

    def feed(self, block: AudioBlock) -> list[Commit]:
        if self.stats.audio_seconds == 0.0:
            self.stats.started_at = time.monotonic()  # exclude model load time
        if self._buffer_start is None:
            self._buffer_start = block.captured_at
        self._buffer = np.concatenate([self._buffer, block.samples])
        seconds = len(block.samples) / TARGET_RATE
        self._new_audio += seconds
        self.stats.audio_seconds += seconds

        buffered = len(self._buffer) / TARGET_RATE
        if buffered < self.min_first_seconds or self._new_audio < self.step_seconds:
            return []
        self._new_audio = 0.0

        if self._is_silence():
            self._drop_buffer()
            return []

        return self._step()

    @property
    def rolling_confidence(self) -> float | None:
        """Mean word probability over the recent window; None until warmed up."""
        if len(self._word_probs) < 20:
            return None
        return float(np.mean(self._word_probs))

    def reset_confidence(self) -> None:
        self._word_probs.clear()

    def _is_silence(self) -> bool:
        return float(np.sqrt(np.mean(self._buffer**2))) < 3e-4

    def _drop_buffer(self) -> None:
        seconds = len(self._buffer) / TARGET_RATE
        if self._buffer_start is not None:
            self._buffer_start += seconds
        self._buffer = np.zeros(0, dtype=np.float32)
        self._committed_end = 0.0
        self._pending = []
        self.pending_text = ""
        if self.language is None:
            self._locked_language = None  # allow a language switch after silence

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
                words.append(Hyp(raw=w.word, norm=_norm(w.word), end=w.end))
                if w.probability is not None:
                    self._word_probs.append(w.probability)
        self.stats.transcribe_calls += 1
        self.stats.transcribe_seconds += time.monotonic() - t0
        if self._locked_language is None and info.language_probability and info.language_probability > 0.8:
            self._locked_language = info.language
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
            latency = time.monotonic() - (self._buffer_start + self._committed_end)
            self.stats.add_commit(latency, n_words=agreed)
            commits.append(Commit(text=text, latency=latency, language=self.detected_language))

        self._pending = current[agreed:]
        self.pending_text = "".join(w.raw for w in self._pending).strip()
        if current:
            # freshness of the newest word on screen (committed or pending)
            display_latency = time.monotonic() - (self._buffer_start + current[-1].end)
            self.stats.add_display(display_latency)
        self._trim()
        return commits

    def _trim(self) -> None:
        buffered = len(self._buffer) / TARGET_RATE
        if buffered <= self.window_seconds:
            return
        cut = self._committed_end
        if cut < 1.0:
            # nothing stable to cut at (e.g. music): bound compute anyway
            cut = buffered - self.window_seconds
            self._pending = []
        samples = int(cut * TARGET_RATE)
        self._buffer = self._buffer[samples:]
        self._buffer_start += cut
        self._committed_end = max(0.0, self._committed_end - cut)
        self._pending = [
            Hyp(raw=w.raw, norm=w.norm, end=w.end - cut) for w in self._pending if w.end - cut > 0
        ]
        if self._last_commit is not None:
            self._last_commit = Hyp(
                raw=self._last_commit.raw,
                norm=self._last_commit.norm,
                end=self._last_commit.end - cut,
            )
