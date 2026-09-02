# STT Provider Benchmark — Whisper vs Sarvam Saaras vs NVIDIA Parakeet vs Moonshine

## What was actually run here (read this first)

I built and ran a real benchmark harness against the existing
`STTProvider` architecture (`stt_provider.py`). It works, and it produced
clean, honest output — but **no provider could actually transcribe
anything in this sandbox**, for one specific, verifiable reason:

**This container's network egress allowlist blocks every host that hosts
STT model weights or the Sarvam API** — confirmed directly, not assumed:

| Host needed | For | Result |
|---|---|---|
| `openaipublic.azureedge.net` | Whisper model weights | `403 host_not_allowed` |
| `huggingface.co` | Moonshine / Parakeet weights | `403 host_not_allowed` |
| `api.sarvam.ai` | Saaras STT | `403 host_not_allowed` |

This is a sandbox restriction, not a real finding about any provider — even
your **existing, already-working Whisper setup** can't load its model here.
So rather than burn time/context fighting a network wall that isn't a
property of any provider, I stopped there for execution and instead
delivered something you can actually run: a real harness, ready to point at
your real dev machine (which already has `SARVAM_API_KEY` configured and
presumably a working Whisper install, per the existing repo).

**What I did verify by actually running it:**
- The harness imports the real `WhisperSTTProvider` / `SaarasSTTProvider`
  from your existing `stt_provider.py` unmodified, adds two new adapters
  (`ParakeetSTTProvider`, `MoonshineSTTProvider`) in the *same* contract,
  and produces a clean comparison table.
- Each provider fails with its own precise, correct reason (see table
  below) — e.g. Saaras correctly picked up the real `SARVAM_API_KEY` from
  `.env` and got exactly as far as making the HTTPS request before being
  blocked, proving the harness's config-loading is correct, not just its
  error-handling.
- 4 synthetic test clips (`benchmark/samples/*.wav`), generated offline
  with `espeak-ng` (no network needed), each tagged with known ground-truth
  fillers/repetitions/false-starts, ready to feed the harness.

**Run it for real, in 3 lines, on a machine with normal internet access:**
```bash
cd assessment/benchmark
pip install -r ../requirements.txt python-dotenv
python3 run_benchmark.py
```
It'll pick up your existing `.env` and produce real transcripts, real
latency, and real retention numbers for whichever providers you have
credentials/packages for.

---

## 1. Benchmark results (as actually executed)

| Provider | Ran? | Failure reason (exact) |
|---|---|---|
| **Whisper** | No | `403 Forbidden` fetching model weights from `openaipublic.azureedge.net` (sandbox network block only — this is your existing, working production provider) |
| **Saaras (Sarvam)** | No | `403 Forbidden` calling `api.sarvam.ai` (sandbox network block; the API key itself loaded and was used correctly) |
| **Parakeet** | No | `nemo_toolkit` not installed — and even if installed, weight download needs `huggingface.co`/NGC, also blocked here |
| **Moonshine** | No | `moonshine_onnx` not installed — same weight-hosting issue |

No transcripts, WER, or retention numbers exist yet for any provider —
**anything I told you about accuracy or filler-retention right now would
be invented, so I'm not going to do that.** The honest next step is running
`run_benchmark.py` in an environment with real network access; it'll take
minutes, not hours, once weights can download.

---

## 2. Which provider is best for assessment, and why — what I can say without live numbers

I can't crown a winner without real transcripts, but the *architecture*
already tells you something concrete, and it's worth saying now:

- **Saaras is architecturally disqualified from ever powering your
  Pronunciation score, regardless of accuracy.** This isn't new — it's
  already documented in your own `stt_provider.py`: Saaras's timestamps are
  sentence-level only, no per-word confidence at all. Your Pronunciation
  scorer (`WhisperConfidenceProvider`) needs per-word confidence to work;
  with Saaras as STT it silently falls back to a flat default score
  (72.0) with no real signal. So even if Saaras transcribes fillers
  beautifully, it breaks a downstream parameter your assessment already
  relies on.
- **Parakeet is the heaviest possible fit for your stated goals.** It's a
  0.6B-parameter model, effectively needs a GPU to run at reasonable
  latency, and needs `nemo_toolkit` (a large, GPU-oriented dependency
  chain) — the opposite of "CPU-only, offline-friendly," which the rest of
  this project (Whisper, LanguageTool, MATTR vocabulary scoring) is built
  around.
- **Moonshine is the one genuinely worth testing seriously**, on paper: CPU-
  first, small (tens-to-low-hundreds of MB), designed for edge/offline use
  — same operating profile as your current Whisper setup, just lighter.
  Whether it *preserves* fillers/repetitions as well as Whisper is exactly
  what's unverified and exactly what the harness will tell you once it can
  run.
- **Whisper stays the only provider proven, end-to-end, in your actual
  pipeline** — it already has word-level timestamps, per-word confidence,
  and everything downstream (Pronunciation, Clarity, Fluency, Filler
  detection) is built and tuned against its output shape.

## 3. What information Whisper currently loses (from reading the code, not a benchmark)

This is worth flagging independent of any benchmark, because it's visible
directly in `stt_provider.py` and `filler_detector.py`:

- **`condition_on_previous_text=False`** is deliberately set (see the
  comment in `WhisperSTTProvider.transcribe`) specifically to reduce
  hallucination/truncation on short clips. That's a real, reasonable
  tradeoff, but it does mean Whisper is already tuned toward "don't
  invent words" over "capture every disfluency" — the two aren't always
  the same optimization target.
- Whisper (like most general-purpose ASR) has a known, well-documented
  tendency to **smooth over disfluencies** — dropping repeated words and
  cleaning up false starts — because its training data (mostly clean
  captions/transcripts) rewards fluent-sounding output. Your own filler
  detector (`filler_detector.py`) exists partly to work around Whisper's
  transcript, not assume it's verbatim.
- This is precisely why the "retained the word" vs "preserved how it was
  spoken" distinction in the harness matters, and precisely what the
  Whisper row in a real benchmark run will quantify.

## 4. Would switching STT improve filler/fluency assessment?

Unverified — genuinely can't say yet without a real run. Directionally:
switching STT could only help your *filler/repetition/false-start*
signal if the alternative (a) actually transcribes disfluencies more
verbatim than Whisper does, **and** (b) still gives you the word-level
timestamps/confidence your Pronunciation/Clarity scores depend on. Saaras
already fails (b) by design. Parakeet and Moonshine are unverified on
both — that's exactly what the harness will tell you once weights can
download.

## 5. Tradeoffs — architectural, not measured

| Provider | CPU-only? | Offline? | RAM ballpark | Word timestamps | Word confidence |
|---|---|---|---|---|---|
| Whisper (current) | Yes | Yes | ~1-2 GB (base/small) | Yes | Yes |
| Saaras | Yes (hosted) | **No** | N/A (API) | Sentence-level only | **No** |
| Parakeet 0.6B | **No (GPU strongly preferred)** | Yes | ~4-6 GB | Yes (NeMo API) | Not exposed in NeMo's public output |
| Moonshine Tiny/Base | Yes | Yes | ~200-400 MB | **No** (public API is text-only) | **No** |

These are documented/vendor-stated characteristics, not measurements from
this session — flag that distinction to your mentor explicitly.

---

## Files delivered

- `run_benchmark.py` — orchestrator, produces the comparison table + JSON
- `candidate_providers.py` — Parakeet & Moonshine adapters (same `STTResult` contract as your existing `stt_provider.py`; nothing production is touched)
- `analyze.py` — the assessment-oriented retention analyzer (filler/repetition/false-start, not WER)
- `samples_manifest.py` — 4 ground-truth-tagged synthetic test scripts
- `samples/*.wav` — the 4 synthesized audio clips
- `benchmark_results.json` — raw output from the run in this sandbox (all `available: false`, with exact reasons)
