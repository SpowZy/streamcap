"""Accuracy / speed / cost evaluation across local model sizes and the cloud API.

Scans a directory for (clip.wav, clip.txt) reference pairs, transcribes each
clip with every backend, and prints a markdown table of WER, real-time factor
and cost per audio hour. Language is inferred from a _xx suffix in the file
name (test_en.wav -> en).

    python scripts/eval_wer.py --dir local
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

LOCAL_CONFIGS = [
    ("tiny", "cpu", "int8"),
    ("base", "cpu", "int8"),
    ("small", "cpu", "int8"),
    ("small", "cuda", "float16"),
]

# cloud reference points, streaming rates (July 2026)
DEEPGRAM_STREAMING_PER_HOUR = 0.462  # nova-3 PAYG
ASSEMBLYAI_STREAMING_PER_HOUR = 0.45  # universal-3 pro


def normalize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    return text.split()


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(ref) + 1)
    d[0, :] = np.arange(len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(d[len(ref), len(hyp)]) / max(1, len(ref))


def transcribe_local(audio: np.ndarray, model_size: str, device: str, compute: str, language: str):
    from streamcap.gpu import prepare_cuda_dlls

    if device == "cuda":
        prepare_cuda_dlls()
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device=device, compute_type=compute)
    t0 = time.monotonic()
    segments, _ = model.transcribe(audio, language=language, beam_size=1)
    text = " ".join(s.text for s in segments)
    return text, time.monotonic() - t0


def transcribe_deepgram(wav_path: Path, language: str, api_key: str):
    url = f"https://api.deepgram.com/v1/listen?model=nova-3&language={language}&smart_format=true"
    if language == "ar":
        url = "https://api.deepgram.com/v1/listen?model=nova-2&language=ar&smart_format=true"
    req = urllib.request.Request(
        url,
        data=wav_path.read_bytes(),
        headers={"Authorization": f"Token {api_key}", "Content-Type": "audio/wav"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    text = payload["results"]["channels"][0]["alternatives"][0]["transcript"]
    return text, time.monotonic() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="local")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from scripts.simulate_stream import load_wav_16k_mono
    from streamcap.audio import TARGET_RATE
    from streamcap.cloud import load_api_key

    pairs = []
    for wav in sorted(Path(args.dir).glob("*.wav")):
        ref = wav.with_suffix(".txt")
        if ref.is_file():
            lang = wav.stem.rsplit("_", 1)[-1]
            pairs.append((wav, ref.read_text(encoding="utf-8").strip(), lang))
    if not pairs:
        print(f"no (wav, txt) pairs found in {args.dir}/")
        return

    rows = []
    for wav, reference, lang in pairs:
        audio = load_wav_16k_mono(str(wav))
        duration = len(audio) / TARGET_RATE
        for model_size, device, compute in LOCAL_CONFIGS:
            try:
                text, seconds = transcribe_local(audio, model_size, device, compute, lang)
                rows.append((wav.stem, f"{model_size} · {device}/{compute}", wer(reference, text),
                             seconds / duration, 0.0))
            except Exception as e:  # noqa: BLE001
                print(f"skip {model_size}/{device}: {e}")
        key = load_api_key()
        if key:
            try:
                text, seconds = transcribe_deepgram(wav, lang, key)
                rows.append((wav.stem, "deepgram nova-3 · cloud", wer(reference, text),
                             seconds / duration, DEEPGRAM_STREAMING_PER_HOUR))
            except Exception as e:  # noqa: BLE001
                print(f"skip deepgram: {e}")

    print("\n| clip | backend | WER | RTF | $/audio-hr |")
    print("|---|---|---|---|---|")
    for clip, backend, w, rtf, cost in rows:
        print(f"| {clip} | {backend} | {w:.1%} | {rtf:.2f} | ${cost:.3f} |")
    print(
        f"\ncloud reference: AssemblyAI streaming $0.15/hr (Universal-Streaming) to "
        f"${ASSEMBLYAI_STREAMING_PER_HOUR:.2f}/hr (Universal-3 Pro)"
    )


if __name__ == "__main__":
    main()
