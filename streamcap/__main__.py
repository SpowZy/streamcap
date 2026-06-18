"""Live captions in the console: python -m streamcap"""

from __future__ import annotations

import argparse
import queue
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="streamcap", description="Real-time captions for system audio.")
    p.add_argument("--model", default="small", help="faster-whisper model size (tiny/base/small/medium/large-v3)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--language", default=None, help="force a language code (default: auto-detect)")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: run until Ctrl+C)")
    p.add_argument("--overlay", action="store_true", help="show the on-screen caption overlay instead of console output")
    p.add_argument("--list-devices", action="store_true", help="list WASAPI loopback devices and exit")
    return p.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()

    from .audio import LoopbackCapture
    from .asr import StreamingTranscriber

    if args.list_devices:
        from .audio import list_loopback_devices

        for d in list_loopback_devices():
            print(f"[{d['index']:>3}] {d['name']}  ({int(d['defaultSampleRate'])} Hz, {d['maxInputChannels']} ch)")
        return 0

    if args.overlay:
        from .overlay import run_overlay

        return run_overlay(args.model, args.device, args.language, args.duration)

    print(f"loading {args.model} ({args.device}) ...", flush=True)
    transcriber = StreamingTranscriber(model_size=args.model, device=args.device, language=args.language)
    print(f"model ready on {transcriber.device} ({transcriber.compute_type})")

    capture = LoopbackCapture()
    print(f"capturing: {capture.device['name']}")
    print("play anything — captions below. Ctrl+C to stop.\n")
    capture.start()

    import time

    deadline = time.monotonic() + args.duration if args.duration else None
    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                block = capture.blocks.get(timeout=1.0)
            except queue.Empty:
                continue
            for commit in transcriber.feed(block):
                lang = f" [{commit.language}]" if commit.language else ""
                print(f"{commit.text}   ({commit.latency:.2f}s){lang}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        print("\n" + transcriber.stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
