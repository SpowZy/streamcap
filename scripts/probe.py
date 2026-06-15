"""Environment probe: audio devices + ASR backends available on this machine."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np


def main() -> None:
    from streamcap.audio import TARGET_RATE, list_loopback_devices

    print("== loopback devices ==")
    for d in list_loopback_devices():
        print(f"[{d['index']:>3}] {d['name']}  ({int(d['defaultSampleRate'])} Hz, {d['maxInputChannels']} ch)")

    print("\n== asr backends (tiny model smoke test) ==")
    from streamcap.asr import StreamingTranscriber

    for device in ("cuda", "cpu"):
        try:
            t = StreamingTranscriber(model_size="tiny", device=device)
            import time

            t0 = time.monotonic()
            t._model.transcribe(np.zeros(TARGET_RATE * 5, dtype=np.float32), beam_size=1)
            dt = time.monotonic() - t0
            print(f"{device}: ok ({t.compute_type}), 5s of silence transcribed in {dt:.2f}s")
        except Exception as e:
            print(f"{device}: unavailable ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
