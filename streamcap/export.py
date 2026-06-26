"""Session transcript export: SRT, VTT and plain text."""

from __future__ import annotations

from pathlib import Path


def _stamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(transcript: list[dict], path: Path) -> None:
    lines = []
    for i, entry in enumerate(transcript, 1):
        lines += [
            str(i),
            f"{_stamp(entry['start'], ',')} --> {_stamp(entry['end'], ',')}",
            entry["text"],
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(transcript: list[dict], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for entry in transcript:
        lines += [f"{_stamp(entry['start'], '.')} --> {_stamp(entry['end'], '.')}", entry["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_txt(transcript: list[dict], path: Path) -> None:
    path.write_text("\n".join(e["text"] for e in transcript), encoding="utf-8")


def save_session(transcript: list[dict], directory: str = "sessions") -> Path | None:
    if not transcript:
        return None
    import time

    # cloud drain can append out of order relative to local commits
    transcript = sorted(transcript, key=lambda e: (e["start"], e["end"]))
    out = Path(directory) / time.strftime("%Y-%m-%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    write_srt(transcript, out / "captions.srt")
    write_vtt(transcript, out / "captions.vtt")
    write_txt(transcript, out / "transcript.txt")
    return out
