# VoiceCoach by Auravo

Browser-based English speech assessment tool. Record up to two minutes of speech, get it transcribed locally via OpenAI Whisper, and receive scored feedback across seven dimensions: **Pace, Filler Words, Pronunciation, Grammar, Clarity, Vocabulary Coverage, CEFR Level** — plus a rule-based **Voice Archetype** read on your speaking style.

Live at **https://voassess.auravo.ai**. Full architecture, API reference, and scoring internals are documented in [DESIGN.md](DESIGN.md). Server setup, deployment steps, and production troubleshooting are in [PRODUCTION.md](PRODUCTION.md).

## Features

- 🎙️ In-browser recording with live waveform (MediaRecorder + Web Audio API)
- 🗣️ Local speech-to-text via OpenAI Whisper (`base` model) — no audio leaves the server
- ✅ Grammar checking via `language_tool_python`, with "Hear correction" TTS
- 🔊 Per-word pronunciation confidence with "Hear it" TTS playback
- 📊 Seven-metric scoring (0–100) with radar chart + historical trend lines (Chart.js)
- 📚 Vocabulary Coverage — lexical diversity + advanced-word usage proxy
- 🏅 CEFR Level (A1–C2) — heuristic estimate from vocabulary, grammar, pronunciation, pace, and sentence complexity
- 🎭 Voice Archetype — rule-based categorical read on speaking style (e.g. "The Orator", "The Storyteller")
- 🔐 Auth via PocketBase — sign in with an emailed one-time code, sign up with a password for new accounts
- 📁 Full session history stored in SQLite
- 🎓 Guided **English Assessment**: state your name → repeat sentences shown on pictures →
  repeat sentences you hear → describe/compare pictures → a final free-speech assessment,
  all rolled into one comprehensive report

## Prerequisites

- Python 3.11+
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` (audio conversion)
- Java JRE (optional — enables real grammar checking via `language_tool_python`; without it the app falls back to a crude repeated-word check)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`language_tool_python` downloads a ~200 MB grammar rule set on first use.

## Run

```bash
python app.py
```

Starts Uvicorn on `http://0.0.0.0:5050`. First startup loads the Whisper model (~5–10s cold start).

| Route | Description |
|---|---|
| `/` | Redirects to `/login` |
| `/login` | Sign in / sign up (PocketBase-backed) |
| `/app` | Main SPA — Record / English Assessment / History / Progress tabs |

Auth is delegated to PocketBase at `pb.auravo.ai`. `POST /assess`, `GET /history`, `POST /assess/stage`,
`POST /assessment/finalize`, and `GET /assessment/history` all require a valid PocketBase session
token (`x-pb-token` header, verified server-side) and are scoped to the requesting user — see
[DESIGN.md §16](DESIGN.md#16-security-considerations).

## Project layout

```
app.py              FastAPI backend — transcription, scoring, persistence
index.html / script.js / style.css     Main app SPA
login.html / login.js / login.css      Login page
auth-check.js        Auth guard (ES module, runs before script.js)
requirements.txt
data/                SQLite DB + raw/converted recordings (git-ignored, auto-created)
```

## Notes

- Recording files (`data/recordings/`) are never cleaned up automatically.
- See [DESIGN.md](DESIGN.md) for scoring formulas, API request/response shapes, database schema, and known limitations.
