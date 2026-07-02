# Architecture

A walkthrough of the pipeline, the decisions behind it, and the measurements that drove them. Everything here is defensible line by line, and most of it was learned the hard way during tuning.

## Pipeline

```
LoopbackCapture (audio.py)
  WASAPI loopback of the default output device, native rate (48 kHz stereo typical)
  -> downmix to mono -> resample to 16 kHz float32 (box-filter decimation when the ratio is integer)
  -> 250 ms AudioBlocks, each stamped with the wall time of its first sample
      | bounded queue (drops under backpressure; dropped audio beats unbounded memory)
      v
StreamingTranscriber (asr.py)                     HybridRouter (router.py)
  rolling window <= 8 s, adaptive decode cadence    rolling word-probability signal
  LocalAgreement commit policy                      < 0.55 -> open Deepgram stream (cloud.py)
  endpointing: silence finalizes the tail           probe local again after 20 s (hysteresis)
  wall-clock gap detection and resync               60 s cooldown after a cloud error
  language lock + periodic posterior audit          RMS gate on the cloud leg: silence unbilled
      |
      v commits (final) + pending tail (unstable)
Overlay (overlay.py) . Dashboard (dashboard.py) . Export (export.py)
```

## The streaming problem

Whisper is a batch model: it expects a complete utterance. Naively re-transcribing a growing buffer produces captions that rewrite themselves, which is the flicker every caption product fights. The fix here is the LocalAgreement policy from the simultaneous-ASR literature (UFAL whisper_streaming, Simul-Whisper; WhisperRT is the fine-tuned causal evolution of the same idea): keep the tail hypothesis private, and only commit the longest common prefix of two consecutive decodes. Agreement across decodes doubles as a hallucination filter, since a phantom phrase rarely survives two independent passes at different window positions.

Two consequences worth knowing:

- Commit latency is bounded below by one step interval, because a word must be seen twice. That is why commit p50 sits near 2x the step. Display latency, which is what the user actually perceives, runs one decode behind real time: p50 0.45 to 0.50 s measured.
- A single early disagreement blocks everything behind it. Anything that destabilizes hypotheses destroys latency. Language flapping was the worst offender during development: auto-detect on short windows alternated fr/en between decodes, agreement never happened, and latency climbed from 1 s to 9 s.

## Endpointing and stream gaps

Agreement alone has two failure modes at utterance boundaries, and both of them eat words.

**End of speech.** The final words of a sentence may never be seen twice, since no more audio arrives to re-decode against. They sit in the unstable tail until the trim policy throws them away. streamcap tracks per-block silence; after 0.7 s of it, the freshest hypothesis is by definition the final answer, so the tail is flushed through one last decode and promoted to committed text. Decoding then stops until speech returns. Silence is never re-decoded, never hallucinated over, and never billed.

**Holes in wall time.** A paused video, a loopback stall or a capture-queue overflow leaves a gap the buffer knows nothing about. Unhandled, every commit after the hole inherits it as phantom latency, so a 2 minute pause reads as a 120 s p50 forever. Every block carries its capture timestamp; when a block arrives later than the buffer's expected end, the tail is finalized, the window resets, and the clock resyncs. The promotion delay is excluded from the latency percentiles because it measures the pause, not the pipeline.

## Language handling

Three mechanisms, layered by how language actually fails in the wild:

1. **Lock.** Detection runs only on windows of 3 s or more and must agree across two consecutive decodes before locking. Music stings and jingles fool short-window detection (Spanish on an intro sting is a classic). Once locked, decodes pin the language, which is what keeps hypotheses stable.
2. **Posterior audit.** A language switch mid-stream usually does not collapse token confidence. Whisper forced to the old language covertly translates the new one, fluently and with high word probabilities, so no token-level signal can catch it. The encoder's language posterior can. It gets audited every ~3 s of audio; an unambiguous contradiction (p >= 0.95) breaks the lock at once, a marginal one needs a second opinion. The unlock happens in place: the buffered foreign-language audio is kept, because it is both the context the re-detection needs and speech the user has not seen committed yet.
3. **Unlock on boundaries.** A long silence (>= 2.5 s) or a wall-clock gap also unlocks, since a source switch usually hides there. `F7` remains the deterministic manual override.

A confidence collapse (rolling mean under 0.55) still triggers re-detection as a catch-all for noise and crosstalk.

## Measured tuning

All numbers from an i7-10875H + RTX 2080 Super, `small` model, fp16, 8 s window:

| decision | measurement | choice |
|---|---|---|
| `vad_filter` | +110 ms per decode call | off by default; the RMS gate plus agreement already cover silence |
| `word_timestamps` | +100 ms per call | kept, needed for commit boundaries, trimming and latency accounting |
| window 12 s vs 8 s | ~700 ms vs ~505 ms per call | 8 s keeps up with real time with headroom |
| decode cadence | decode cost varies 3-4x with GPU thermals and contention | adaptive: step = max(0.6 s, 1.3 x EMA of decode time), capped at 2.5 s |

The adaptive cadence is what keeps the pipeline honest on machines it does not control. If decodes get slow (thermal throttling, a busy GPU, CPU-only laptops), a fixed cadence falls behind real time, overflows the capture queue and silently drops audio. Stretching the step degrades latency gracefully instead. Measured on this machine: the same session runs at RTF 0.19 cold and RTF 0.45 with the GPU throttled, captions intact in both cases.

The trim policy cuts the buffer at the committed boundary, which is safe because that text is final, and force-cuts if nothing commits for a full window (music, noise) so decode cost stays bounded. The pending hypothesis is shifted rather than discarded on a force-cut, so hard-to-agree words still get their chance to commit.

## The router

The quality signal is a rolling mean of Whisper word probabilities over the newest 40 words, warm after 20. Policy:

- local to cloud when rolling confidence drops under 0.55 with at least 5 s of audio observed: heavy accents, crosstalk, noise
- cloud back to local via a probe after 20 s; the local buffer is dropped on failover so the probe starts fresh instead of committing stale audio
- a failed cloud session (bad key, network, quota) backs off for 60 s instead of reconnecting on every confidence dip; captions degrade in accuracy, never in availability
- the same RMS gate runs in front of the cloud socket, so silence is never billed (measured: a 21 s clip produced 19 s of billed stream, and chunks dropped under backpressure never enter the billing timeline)

Cloud results carry Deepgram's stream offsets. A timeline of cumulative-sent-seconds to wall-clock maps each final result back to when that audio actually played, so cloud and local latencies are measured on the same clock. Cost accounting is symmetric: `cloud_seconds` at the streaming list rate against a stream-everything baseline over the whole session. The dashboard's "saved" figure is that difference, not a marketing estimate.

## Latency accounting

Every AudioBlock is stamped at capture. Display latency is now minus the wall time of the newest decoded word; commit latency is now minus the wall time of the last committed word. Both are computed against capture stamps, not processing milestones, so queueing delay has nowhere to hide. Whisper's word timings can overshoot the audio actually held (it pads its decode window), so measurements clamp to real buffered samples. p50/p95 stream to the dashboard live and the session summary prints them on exit.

## What I would build next

1. WhisperRT-style causal encoder (arXiv:2508.12301, weights released): removes the re-decode entirely, the LocalAgreement machinery stays as a safety layer.
2. Diarization: streaming speaker turns (pyannote, or cloud-side labels) rendered as speaker chips in the overlay. A desktop app also has a cheaper trick available: the microphone and the loopback are separate physical streams, so "me vs them" labeling costs nothing.
3. Real benchmark fixtures: a Common Voice en/fr/ar slice through `eval_wer.py`, tracked in CI, so quality regressions become a failing table instead of a user report.
4. Summaries and RAG over session history: transcripts already export per session; embedding them (pgvector, Chroma) turns "search everything you have heard" into a small feature rather than a project.
