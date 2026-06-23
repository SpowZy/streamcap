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
    p.add_argument("--dashboard", action="store_true", help="serve the live metrics dashboard on http://127.0.0.1:8765")
    p.add_argument("--route", default=None, choices=["local", "cloud", "auto"],
                   help="inference routing (default: auto when a Deepgram key is configured, else local)")
    p.add_argument("--list-devices", action="store_true", help="list WASAPI loopback devices and exit")
    return p.parse_args()


def build_engine(transcriber, route: str | None):
    """Wrap the local transcriber in the hybrid router when a key is available."""
    from .cloud import load_api_key

    key = load_api_key()
    mode = route or ("auto" if key else "local")
    if mode != "local":
        if not key:
            print("no DEEPGRAM_API_KEY configured, falling back to local-only")
            return transcriber
        from .router import HybridRouter

        return HybridRouter(transcriber, key, mode=mode)
    return transcriber


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

        return run_overlay(
            args.model, args.device, args.language, args.duration,
            dashboard=args.dashboard, route=args.route,
        )

    print(f"loading {args.model} ({args.device}) ...", flush=True)
    transcriber = StreamingTranscriber(model_size=args.model, device=args.device, language=args.language)
    engine = build_engine(transcriber, args.route)
    print(f"model ready: {engine.device} ({transcriber.compute_type})")

    if args.dashboard:
        from .dashboard import start_dashboard

        def provider() -> dict:
            snap = engine.stats.snapshot()
            snap.update(
                model=args.model,
                device=engine.device,
                language=engine.detected_language or "auto",
            )
            return snap

        print(f"dashboard: {start_dashboard(provider)}")

    capture = LoopbackCapture()
    print(f"capturing: {capture.device['name']}")
    print("play anything — captions below. Ctrl+C to stop.\n")
    capture.start()

    import time

    transcript: list[dict] = []
    deadline = time.monotonic() + args.duration if args.duration else None
    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                block = capture.blocks.get(timeout=1.0)
            except queue.Empty:
                continue
            for commit in engine.feed(block):
                lang = f" [{commit.language}]" if commit.language else ""
                print(f"{commit.text}   ({commit.latency:.2f}s){lang}", flush=True)
                transcript.append({"start": commit.start, "end": commit.end, "text": commit.text})
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        if hasattr(engine, "drain"):
            for commit in engine.drain():
                transcript.append({"start": commit.start, "end": commit.end, "text": commit.text})
        elif hasattr(engine, "close"):
            engine.close()
        from .export import save_session

        saved = save_session(transcript)
        print("\n" + engine.stats.summary())
        if saved:
            print(f"session exported: {saved} (captions.srt, captions.vtt, transcript.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
