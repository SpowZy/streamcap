"""Record N seconds of system audio via loopback to a wav file (sanity check)."""

from __future__ import annotations

import sys
import time
import wave

sys.path.insert(0, ".")

import numpy as np


def main(seconds: float = 5.0, out_path: str = "local/capture_test.wav") -> None:
    from streamcap.audio import TARGET_RATE, LoopbackCapture

    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = LoopbackCapture()
    print(f"capturing {seconds}s from: {cap.device['name']}")
    cap.start()
    chunks = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            chunks.append(cap.blocks.get(timeout=1.0).samples)
        except Exception:
            pass
    cap.stop()

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    rms = float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0
    print(f"captured {len(audio) / TARGET_RATE:.2f}s, rms={rms:.5f} "
          f"({'signal detected' if rms > 3e-4 else 'silence'})")

    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 5.0)
