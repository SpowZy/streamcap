# Architecture

A walkthrough of the pipeline, the decisions behind it, and the measurements that drove them. Written so every choice can be defended line by line.

## Pipeline

```
LoopbackCapture (audio.py)
  WASAPI loopback of the default output device · native rate (48 kHz stereo typical)
  → downmix to mono → resample to 16 kHz float32 (box-filter decimation when integer ratio)
  → 250 ms AudioBlocks, each stamped with the wall time of its first sample
      ↓ bounded queue (drops under backpressure — dropped audio beats unbounded memory)
StreamingTranscriber (asr.py)                     HybridRouter (router.py)
  rolling window ≤ 8 s                              rolling word-probability signal
  re-decode every ≥ 0.6 s of new audio              < 0.55 → open Deepgram stream (cloud.py)
  LocalAgreement commit policy                      probe local again after 20 s (hysteresis)
  language lock after confident detect              RMS gate on the cloud leg: silence unbilled
      ↓ commits (final) + pending tail (unstable)
Overlay (overlay.py) · Dashboard (dashboard.py) · Export (export.py)
```

## The streaming problem

Whisper is a batch model: it expects a complete utterance. Naively re-transcribing a growing buffer produces captions that rewrite themselves — the flicker every caption product fights. The fix here is the **LocalAgreement** policy from the simultaneous-ASR literature (UFAL whisper_streaming; Simul-Whisper; WhisperRT is the fine-tuned-causal-model evolution of the same idea): keep the tail hypothesis private, and only commit the longest common prefix of two consecutive decodes. Agreement across decodes is also an effective hallucination filter — a phantom phrase rarely survives two independent passes at different window positions.

Two consequences worth knowing:

- **Commit latency is bounded below by one step interval** (a word must be seen twice), which is why commit p50 sits near 2× the step. Display latency — what the user perceives — is one decode behind real time: p50 0.45 s measured.
- **A single early disagreement blocks everything behind it.** Anything that destabilizes hypotheses destroys latency. This is why language flapping was the worst bug in development: auto-detect on an 8 s window alternated fr/en between decodes, agreement never happened, and latency climbed from 1 s to 9 s. The language lock (detect once at ≥ 0.8 confidence, unlock only after a silence long enough to drop the buffer) fixed it.

## Measured tuning

All numbers from an i7-10875H + RTX 2080 Super, `small` model, fp16, 8 s window:

| decision | measurement | choice |
|---|---|---|
| `vad_filter` | +110 ms per decode call | off by default; RMS gate + agreement cover silence |
| `word_timestamps` | +100 ms per call | kept — needed for commit boundaries, trimming, latency accounting |
| window 12 s → 8 s | ~700 ms → ~505 ms per call | 8 s; RTF 0.47 → keeps up with real time with headroom |
| decode cadence | cost < step ⇒ stable latency | 0.6 s step; the loop self-balances under backlog |

The trim policy cuts the buffer at the committed boundary (safe: that text is final), and force-cuts if nothing commits for a full window (music, noise) so decode cost stays bounded.

## The router

The quality signal is a rolling mean of Whisper word probabilities (deque of 80, warm after 20). Policy:

- **local → cloud** when rolling confidence < 0.55 with ≥ 5 s of audio observed — heavy accent, crosstalk, noise
- **cloud → local** probe after 20 s; the local buffer is dropped on failover so the probe starts fresh rather than committing stale audio
- the same RMS gate runs in front of the cloud socket: **silence is never billed** (measured: 21 s clip → 19 s billed)
- cloud errors fail back to local silently — captions degrade in accuracy, never in availability

Cloud results carry Deepgram's stream offsets; a timeline of (cumulative-sent-seconds → wall-clock) maps each final back to when that audio actually played, so cloud and local latencies are measured on the same clock. Cost accounting is symmetric: `cloud_seconds` at the streaming list rate vs a stream-everything baseline over the whole session — the dashboard's "saved" figure is that difference, not a marketing estimate.

## Latency accounting

Every AudioBlock is stamped at capture. Display latency = now − (wall time of the newest decoded word). Commit latency = now − (wall time of the last committed word). Both are computed against capture stamps, not against processing milestones — there is nowhere for queueing delay to hide. p50/p95 stream to the dashboard live; the session summary prints them on exit.

## What I would build next

1. **WhisperRT-style causal encoder** (arXiv:2508.12301, weights released) — removes the re-decode entirely; the LocalAgreement machinery stays as a safety layer.
2. **Diarization** — streaming speaker turns (pyannote or cloud-side labels) rendered as speaker chips in the overlay.
3. **Real benchmark fixtures** — a Common Voice en/fr/ar slice through `eval_wer.py`, tracked in CI, so quality regressions are a failing table instead of a user report.
4. **Summaries & RAG over session history** — transcripts are already exported per session; embedding them (pgvector/Chroma) makes "ask everything you've heard" a small feature, not a project.
