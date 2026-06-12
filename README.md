# streamcap

Local-first real-time captions for anything your computer plays — meetings, lectures, videos, calls — with cost-aware cloud fallback and a built-in latency/accuracy/cost benchmark harness.

Work in progress. Full architecture notes, benchmarks and demo video coming.

## Quickstart

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m streamcap
```

Play any audio on your machine and captions stream to the console with end-to-end latency per commit. `Ctrl+C` prints session stats (p50/p95 latency, words committed).
