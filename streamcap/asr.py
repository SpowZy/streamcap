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
        window_seconds: float = 12.0,
        step_seconds: float = 0.7,
        min_first_seconds: float = 1.0,
    ):
        self.language = language
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.min_first_seconds = min_first_seconds
        self.stats = SessionStats()
        self.detected_language: str | None = language

        self.device, self.compute_type, self._model = self._load_model(model_size, device)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start: float | None = None  # monotonic time of buffer[0]
        self._committed_end = 0.0  # seconds into buffer already final
        self._pending: list[Hyp] = []  # last hypothesis tail, not yet agreed
        self._new_audio = 0.0  # seconds accumulated since last step

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

    def _is_silence(self) -> bool:
        return float(np.sqrt(np.mean(self._buffer**2))) < 3e-4

    def _drop_buffer(self) -> None:
        seconds = len(self._buffer) / TARGET_RATE
        if self._buffer_start is not None:
            self._buffer_start += seconds
        self._buffer = np.zeros(0, dtype=np.float32)
        self._committed_end = 0.0
        self._pending = []

    def _transcribe(self) -> list[Hyp]:
        t0 = time.monotonic()
        segments, info = self._model.transcribe(
            self._buffer,
            language=self.language,
            beam_size=1,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        words: list[Hyp] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Hyp(raw=w.word, norm=_norm(w.word), end=w.end))
        self.stats.transcribe_calls += 1
        self.stats.transcribe_seconds += time.monotonic() - t0
        if self.language is None and info.language_probability and info.language_probability > 0.5:
            self.detected_language = info.language
        return words

    def _step(self) -> list[Commit]:
        current = [w for w in self._transcribe() if w.end > self._committed_end + 0.05]

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
            latency = time.monotonic() - (self._buffer_start + self._committed_end)
            self.stats.add_commit(latency, n_words=agreed)
            commits.append(Commit(text=text, latency=latency, language=self.detected_language))

        self._pending = current[agreed:]
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
