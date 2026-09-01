# FAQ — VoiceCoach by Auravo

Two sections: [For Users](#for-users) (anyone recording an assessment) and [For Developers](#for-developers--maintainers) (running, extending, or debugging the app). See [README.md](README.md) for setup and [DESIGN.md](DESIGN.md) for full architecture.

---

## For Users

### What is VoiceCoach?
A browser-based tool that records up to two minutes of spoken English and scores it across seven dimensions — **Pace, Filler Words, Pronunciation, Grammar, Clarity, Vocabulary Coverage, CEFR Level** — with a 0–100 overall score, a fun **Voice Archetype** read on your speaking style, and specific feedback on what to improve.

### How do I use it?
Sign in at [voassess.auravo.ai](https://voassess.auravo.ai), hit record, speak for up to 2 minutes, and stop. Your transcript, scores, and feedback appear within seconds.

### Does my audio get sent to a third party?
No. Speech-to-text runs locally on the server using OpenAI's Whisper model — audio never leaves the server for transcription. There's no external STT API call.

### What do the scores mean?
- **Pace** — words per minute; 120–150 WPM scores highest.
- **Filler Words** — frequency of "um," "like," "you know," etc.
- **Pronunciation** — derived from Whisper's transcription confidence per word.
- **Grammar** — errors detected per word, via automated grammar checking.
- **Clarity** — an average of the other four scores.
- **Vocabulary Coverage** — lexical diversity + how much of your vocabulary goes beyond a common-word baseline.
- **CEFR Level** — a heuristic A1–C2 estimate blending vocabulary, grammar, pronunciation, pace, and sentence complexity.

The **Overall** score weights the original five: Pronunciation 25%, Pace/Filler/Grammar 20% each, Clarity 15%.
Vocabulary Coverage and CEFR Level are shown alongside as additional insight but aren't folded into Overall.

### What's "Voice Archetype"?
A fun, rule-based label (e.g. "The Orator", "The Storyteller", "The Diplomat") describing your speaking style
based on the combination of scores in that session — not a scored metric itself, just a personality-style summary.

### Why did I get a low pronunciation score even though I spoke clearly?
Pronunciation scoring is a proxy based on transcription confidence, not true phoneme analysis. It can be pulled down by background noise, a strong accent, or a very short recording — see [Known Limitations](DESIGN.md#17-known-limitations) for detail.

### It flagged "gonna" / "wanna" as a grammar error — is that a bug?
It's a known false positive. The grammar checker is tuned for written English and sometimes flags informal spoken contractions.

### Can I see my past sessions?
Yes — the History and Progress tabs show your own past assessments and trend charts, scoped to your account.

### What browsers/devices are supported?
Any modern browser with `MediaRecorder` and `getUserMedia` support (Chrome, Edge, Firefox, Safari). On iOS Safari, the "Hear it" / "Hear correction" playback buttons require a tap to start audio — this is a platform requirement, not a bug.

### How do I sign in?
With a one-time code emailed to you — no password needed. Enter your email on the login page,
we'll send a code, enter it and you're in. New here? Use the Sign Up tab to create an account
with a password first, then sign back in anytime with an emailed code.

---

## For Developers / Maintainers

### What's the stack?
FastAPI + Uvicorn backend, vanilla JS/HTML/CSS frontend (no build step), OpenAI Whisper (`base` model) for STT, `language_tool_python` for grammar, PocketBase for auth, SQLite for storage, Chart.js for visualizations. Full breakdown in [DESIGN.md §3](DESIGN.md#3-tech-stack).

### How do I run it locally?
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Serves on `http://0.0.0.0:5050`. First run downloads the Whisper model and (if Java is present) a ~200 MB grammar rule set — see [README.md](README.md).

### `language_tool_python` isn't working — what happens?
The app degrades gracefully: `GRAMMAR_OK` is set to `False` and grammar scoring falls back to a crude regex that only catches stuttered repeated words. Install a Java JRE to enable real grammar checking.

### Why isn't `/assess` or `/history` callable without logging in?
Both API routes (and `/assess/stage`, `/assessment/finalize`, `/assessment/history`) require an `x-pb-token` header carrying the caller's PocketBase session token; the backend verifies it server-side via PocketBase's `auth-refresh` endpoint before touching the database. See [DESIGN.md §16](DESIGN.md#16-security-considerations).

### Is there per-user data isolation?
Yes. The `assessments` and `english_assessments` tables have a `user_id` column populated from the verified PocketBase token, and `/history`/`/assessment/history` filter by it — each user only ever sees their own sessions.

### Why does the app feel slow / block under concurrent load?
Whisper inference is synchronous and blocks the event loop; Uvicorn runs single-worker by default, so simultaneous `/assess` calls queue. Run with `--workers N` for concurrency — note each worker loads its own copy of the Whisper model in RAM.

### Where are recordings stored, and do they get cleaned up?
Raw (`.webm`) and converted (`.wav`) files land in `data/recordings/`, auto-created, and are **never deleted automatically**. At ~7 MB/session, this grows unbounded — plan for periodic cleanup or add retention logic.

### I'm getting a 413 or 502/504 from `/assess` — why?
- **413** — nginx's `client_max_body_size` (default 50M in this deployment) was exceeded.
- **502/504** — Whisper or the grammar tool took longer than nginx's `proxy_read_timeout` (300s in this deployment). Increase these values in the nginx config if longer recordings or slower hardware are expected.

### How is the overall score computed?
```
overall = pace×0.20 + filler×0.20 + pronunciation×0.25 + grammar×0.20 + clarity×0.15
```
Full formulas for each metric are in [DESIGN.md §8](DESIGN.md#8-scoring-algorithms).

### Can I swap in a bigger/smaller Whisper model?
Yes — change the model name where Whisper loads in `app.py`. Tradeoffs (RAM, latency, WER) are tabled in [DESIGN.md §15](DESIGN.md#15-deployment--infrastructure). `base` is the current default, chosen for a balance of accuracy and CPU inference time.

### Where's the database, and what's in it?
Single SQLite file at `data/assessments.db`, one `assessments` table (auto-created on startup). Note pronunciation/grammar *issue detail* (the wrong→correct pairs) is not persisted — only returned live in the `/assess` response. Full schema in [DESIGN.md §10](DESIGN.md#10-database-schema).

### How does TTS ("Hear it" / "Read Aloud") work — does it call an API?
No server calls — it uses the browser-native Web Speech API (`speechSynthesis`) entirely client-side. Quality depends on the OS/browser's installed voices.
