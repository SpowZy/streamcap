"""System-audio capture via WASAPI loopback.

Captures whatever the machine is currently playing (meetings, videos, calls)
without a virtual audio cable, then downmixes to mono and resamples to
16 kHz float32 — the input format Whisper models expect.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass

import numpy as np
import pyaudiowpatch as pyaudio

TARGET_RATE = 16_000


@dataclass
class AudioBlock:
    """A chunk of captured audio plus the wall-clock time of its first sample."""

    samples: np.ndarray  # float32 mono at TARGET_RATE
    captured_at: float  # time.monotonic() when the first sample was played


def _downmix(interleaved: np.ndarray, channels: int) -> np.ndarray:
    if channels == 1:
        return interleaved
    return interleaved.reshape(-1, channels).mean(axis=1)


def _resample(x: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    if src_rate == dst_rate:
        return x.astype(np.float32, copy=False)
    factor = src_rate / dst_rate
    if factor.is_integer():
        # Integer decimation with a box filter: cheap anti-aliasing that is
        # plenty for speech (48 kHz -> 16 kHz is the common WASAPI case).
        k = int(factor)
        usable = len(x) - (len(x) % k)
        return x[:usable].reshape(-1, k).mean(axis=1).astype(np.float32)
    n_dst = int(round(len(x) * dst_rate / src_rate))
    src_t = np.arange(len(x)) / src_rate
    dst_t = np.arange(n_dst) / dst_rate
    return np.interp(dst_t, src_t, x).astype(np.float32)


class LoopbackCapture:
    """Streams the default output device's loopback as 16 kHz mono float32 blocks."""

    def __init__(self, block_seconds: float = 0.25):
        self._pa = pyaudio.PyAudio()
        self.device = self._find_loopback_device()
        self._rate = int(self.device["defaultSampleRate"])
        self._channels = max(1, int(self.device["maxInputChannels"]))
        self._frames_per_buffer = int(self._rate * block_seconds)
        self.blocks: "queue.Queue[AudioBlock]" = queue.Queue(maxsize=256)
        self._stream = None

    def _find_loopback_device(self) -> dict:
        wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for candidate in self._pa.get_loopback_device_info_generator():
            if default_out["name"] in candidate["name"]:
                return candidate
        raise RuntimeError(
            f"no WASAPI loopback device found for default output {default_out['name']!r}"
        )

    def _on_block(self, in_data, frame_count, time_info, status):
        now = time.monotonic()
        pcm = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        mono = _downmix(pcm, self._channels)
        samples = _resample(mono, self._rate)
        block = AudioBlock(samples=samples, captured_at=now - frame_count / self._rate)
        try:
            self.blocks.put_nowait(block)
        except queue.Full:
            pass  # consumer stalled; dropping audio beats unbounded memory growth
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._rate,
            input=True,
            input_device_index=self.device["index"],
            frames_per_buffer=self._frames_per_buffer,
            stream_callback=self._on_block,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._pa.terminate()


def list_loopback_devices() -> list[dict]:
    pa = pyaudio.PyAudio()
    try:
        return list(pa.get_loopback_device_info_generator())
    finally:
        pa.terminate()
