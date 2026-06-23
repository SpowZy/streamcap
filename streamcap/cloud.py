"""Deepgram streaming client — the paid fallback leg of the hybrid router.

Runs a websocket session on its own asyncio thread. Only voiced audio is
sent (the router applies the same RMS gate as the local path), so cloud
spend tracks speech, not wall-clock. Latency is measured against the wall
time each chunk was actually captured, not against stream offsets.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import TARGET_RATE, AudioBlock


def load_api_key() -> str | None:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parent.parent / "local" / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.startswith("DEEPGRAM_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


@dataclass
class CloudResult:
    text: str
    start: float  # seconds of *sent* audio at which this result starts
    end: float  # seconds of *sent* audio at which this result ends
    is_final: bool


class CloudStream:
    """One streaming session against Deepgram's realtime endpoint."""

    def __init__(self, api_key: str, language: str | None = None):
        self._api_key = api_key
        # nova-3 multilingual covers the common European languages; Arabic
        # needs nova-2 with an explicit language code.
        if language == "ar":
            self._params = "model=nova-2&language=ar"
        elif language:
            self._params = f"model=nova-3&language={language}"
        else:
            self._params = "model=nova-3&language=multi"
        self._send_q: "queue.Queue[bytes | None]" = queue.Queue(maxsize=512)
        self._results: "queue.Queue[CloudResult]" = queue.Queue()
        self._timeline: list[tuple[float, float]] = []  # (cumulative sent s, wall time chunk ended)
        self._sent_seconds = 0.0
        self.interim_text = ""
        self.error: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- feeding ----------------------------------------------------------

    def send(self, block: AudioBlock) -> None:
        pcm = (np.clip(block.samples, -1, 1) * 32767).astype(np.int16).tobytes()
        seconds = len(block.samples) / TARGET_RATE
        self._sent_seconds += seconds
        self._timeline.append((self._sent_seconds, block.captured_at + seconds))
        if len(self._timeline) > 2048:
            del self._timeline[:1024]
        try:
            self._send_q.put_nowait(pcm)
        except queue.Full:
            pass

    @property
    def sent_seconds(self) -> float:
        return self._sent_seconds

    def _wall_time_at(self, sent_offset: float) -> float | None:
        """Wall time when the audio at `sent_offset` seconds was captured."""
        for cum, wall in self._timeline:
            if cum >= sent_offset:
                return wall - (cum - sent_offset)
        return self._timeline[-1][1] if self._timeline else None

    def latency_of(self, result: CloudResult) -> float | None:
        wall = self._wall_time_at(result.end)
        return None if wall is None else time.monotonic() - wall

    def poll(self) -> list[CloudResult]:
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def finish(self) -> None:
        """Stop sending and ask the server to flush its final results."""
        self._send_q.put(None)

    def close(self) -> None:
        self.finish()
        self._thread.join(timeout=5)

    # -- websocket thread --------------------------------------------------

    def _run(self) -> None:
        try:
            asyncio.run(self._session())
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    async def _session(self) -> None:
        import websockets

        url = (
            "wss://api.deepgram.com/v1/listen?"
            f"{self._params}&encoding=linear16&sample_rate={TARGET_RATE}"
            "&channels=1&interim_results=true&smart_format=true"
        )
        async with websockets.connect(
            url, additional_headers={"Authorization": f"Token {self._api_key}"}
        ) as ws:
            async def sender() -> None:
                loop = asyncio.get_running_loop()
                while True:
                    chunk = await loop.run_in_executor(None, self._send_q.get)
                    if chunk is None:
                        await ws.send(json.dumps({"type": "CloseStream"}))
                        return
                    await ws.send(chunk)

            async def receiver() -> None:
                async for message in ws:
                    data = json.loads(message)
                    if data.get("type") != "Results":
                        continue
                    alt = data["channel"]["alternatives"][0]
                    text = alt.get("transcript", "")
                    start = data.get("start", 0.0)
                    end = start + data.get("duration", 0.0)
                    if data.get("is_final"):
                        self.interim_text = ""
                        if text:
                            self._results.put(
                                CloudResult(text=text, start=start, end=end, is_final=True)
                            )
                    elif text:
                        self.interim_text = text

            send_task = asyncio.create_task(sender())
            await receiver()
            send_task.cancel()
