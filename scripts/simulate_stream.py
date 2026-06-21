"""Deterministic pipeline test: stream a wav file through the transcriber
as if it were live audio, printing commits and their latency.

    python scripts/simulate_stream.py local/test_en.wav --model small --speed 1.0
"""

from __future__ import annotations

import argparse
import sys
import time
import wave

sys.path.insert(0, ".")

import numpy as np


def load_wav_16k_mono(path: str) -> np.ndarray:
    from streamcap.audio import TARGET_RATE, _downmix, _resample

    with wave.open(path, "rb") as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width}")
    return _resample(_downmix(pcm, channels), rate, TARGET_RATE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("wav")
    p.add_argument("--model", default="small")
    p.add_argument("--device", default="auto")
    p.add_argument("--language", default=None)
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real-time pacing")
    p.add_argument("--route", default="local", choices=["local", "cloud", "auto"])
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from streamcap.asr import StreamingTranscriber
    from streamcap.audio import TARGET_RATE, AudioBlock

    audio = load_wav_16k_mono(args.wav)
    print(f"{args.wav}: {len(audio) / TARGET_RATE:.1f}s of audio")

    t = StreamingTranscriber(model_size=args.model, device=args.device, language=args.language)
    from streamcap.__main__ import build_engine

    t = build_engine(t, args.route)
    print(f"model ready: {t.device}\n")

    block_len = int(TARGET_RATE * 0.25)
    start = time.monotonic()
    for i in range(0, len(audio), block_len):
        chunk = audio[i : i + block_len]
        if args.speed > 0:
            # absolute-clock pacing: processing time must not delay the stream,
            # exactly like a live capture thread
            target = start + (i + len(chunk)) / TARGET_RATE / args.speed
            time.sleep(max(0.0, target - time.monotonic()))
        for commit in t.feed(AudioBlock(samples=chunk, captured_at=time.monotonic() - len(chunk) / TARGET_RATE)):
            lang = f" [{commit.language}]" if commit.language else ""
            print(f"{commit.text}   ({commit.latency:.2f}s){lang}", flush=True)

    if hasattr(t, "drain"):
        for commit in t.drain():
            print(f"{commit.text}   ({commit.latency:.2f}s) [drain]", flush=True)

    print("\n" + t.stats.summary())
    s = t.stats
    print(f"cloud audio {s.cloud_seconds:.1f}s | cloud spend ${s.cloud_cost:.4f} | saved ${s.saved:.4f}")


if __name__ == "__main__":
    main()
