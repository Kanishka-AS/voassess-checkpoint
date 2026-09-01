# VoiceCoach by Auravo — Design Document

**Version:** 2.0  
**Stack:** FastAPI · OpenAI Whisper · language_tool_python · PocketBase · Vanilla JS  
**Hosted at:** https://voassess.auravo.ai  
**Auth backend:** https://pb.auravo.ai

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [File Structure](#4-file-structure)
5. [Authentication Flow](#5-authentication-flow)
6. [Recording Pipeline](#6-recording-pipeline)
7. [Assessment Pipeline](#7-assessment-pipeline)
8. [Scoring Algorithms](#8-scoring-algorithms)
9. [API Reference](#9-api-reference)
10. [Database Schema](#10-database-schema)
11. [Frontend Architecture](#11-frontend-architecture)
12. [Chart System](#12-chart-system)
13. [TTS Architecture](#13-tts-architecture)
14. [Error Handling](#14-error-handling)
15. [Deployment & Infrastructure](#15-deployment--infrastructure)
16. [Security Considerations](#16-security-considerations)
17. [Known Limitations](#17-known-limitations)
18. [Potential Improvements](#18-potential-improvements)
19. [English Assessment (Guided Flow)](#19-english-assessment-guided-flow)

---

## 1. Project Overview

VoiceCoach is a browser-based English speech assessment tool. Users record up to two minutes of spoken English; the system transcribes the audio locally using OpenAI Whisper, analyses it across five quality dimensions, and presents scored feedback with word-level detail for pronunciation and grammar issues.

### Core Capabilities

| Feature | Description |
|---|---|
| Speech recording | Up to 2 min via browser MediaRecorder API with live waveform visualisation |
| Local STT | OpenAI Whisper `base` model — no external API calls |
| Pronunciation analysis | Per-word Whisper confidence as clarity proxy; interactive TTS demo for each flagged word |
| Grammar checking | language_tool_python (Java-backed); wrong→correct pairs with "Hear correction" TTS |
| Five-metric scoring | Pace, Filler Words, Pronunciation, Grammar, Clarity (0–100 each) |
| Vocabulary Coverage | Lexical-diversity + advanced-vocabulary-usage proxy (0–100), informational — not folded into Overall |
| CEFR Level | Heuristic A1–C2 estimate blending vocabulary, grammar, pronunciation, pace, sentence complexity |
| Voice Archetype | Rule-based categorical read on speaking style (e.g. "The Orator") — not a scored metric |
| Personalised feedback | Rule-based text feedback tied to each metric's score range |
| Session history | All sessions stored in SQLite; expandable cards show transcript + feedback |
| Progress charts | Radar + trend lines per metric across all sessions (Chart.js 4) |
| Authentication | PocketBase email one-time code (OTP) for sign-in; password Sign Up for new accounts. Password Sign In and Google OAuth2 remain enabled on the shared backend but aren't exposed on this app's login page. |

---

## 2. Architecture

```
Browser
├── login.html  ──── PocketBase JS SDK (pb.auravo.ai) ──── auth token in localStorage
│
└── index.html (app)
    ├── auth-check.js  (ES module — validates token, injects user pill, guards route)
    ├── script.js      (recording, UI, charts)
    └── style.css

        │  POST /assess  (multipart: audio + duration)
        │  GET  /history
        ▼
   FastAPI  (app.py · port 5050)
   ├── ffmpeg  — WebM → 16 kHz mono WAV
   ├── Whisper base  — STT + word timestamps + avg_logprob
   ├── language_tool_python  — grammar match objects
   └── SQLite  (data/assessments.db)

Nginx reverse proxy (voassess.auravo.ai)
└── client_max_body_size 50M
    proxy_read_timeout   300s
```

### Request lifecycle for one assessment

```
1.  User clicks Stop
2.  Browser creates Blob (WebM/Opus, ~1–3 MB)
3.  POST /assess  multipart: audio + duration
4.  Nginx forwards to FastAPI (port 5050)
5.  ffmpeg converts → 16 kHz mono WAV
6.  Whisper transcribes with word_timestamps=True
7.  Scoring functions run in sequence
8.  language_tool_python checks grammar
9.  Results inserted into SQLite
10. JSON response → displayResults() → UI update
```

---

## 3. Tech Stack

| Layer | Technology | Version / Notes |
|---|---|---|
| HTTP server | FastAPI | ≥ 0.104 |
| ASGI runtime | Uvicorn | ≥ 0.24 with standard extras |
| Speech-to-text | OpenAI Whisper | `base` model, local CPU inference |
| Grammar checking | language_tool_python | ≥ 2.7.1; requires Java JRE |
| Audio conversion | ffmpeg | system binary; WebM → 16 kHz WAV |
| Database | SQLite | Python stdlib `sqlite3` |
| Auth backend | PocketBase | hosted at pb.auravo.ai |
| Auth client | PocketBase JS SDK | `pocketbase@0.27.3` CDN ES module |
| Charts | Chart.js | 4.4.0 CDN |
| TTS (feedback) | Web Speech API | `SpeechSynthesis` — browser-native, no server |
| Frontend | Vanilla JS + HTML + CSS | No build step; ES modules for auth files |
| Reverse proxy | Nginx | body limit 50 MB, read timeout 300 s |

---

## 4. File Structure

```
voice-assessment-app/
├── app.py              — FastAPI application (all backend logic)
├── index.html          — Main SPA (Record / History / Progress tabs)
├── script.js           — Frontend logic (recording, results, charts)
├── style.css           — Main app stylesheet (Auravo dark warm theme)
├── auth-check.js       — ES module: PocketBase auth guard + user pill
├── login.html          — Standalone login/register page
├── login.js            — ES module: OTP sign-in, sign-up
├── login.css           — Login page stylesheet (two-panel layout)
├── requirements.txt    — Python dependencies
├── DESIGN.md           — This document
└── data/
    ├── assessments.db  — SQLite database (auto-created on first run)
    └── recordings/     — Raw WebM + converted WAV files (auto-created)
        ├── rec_YYYYMMDD_HHMMSS.webm
        └── rec_YYYYMMDD_HHMMSS.wav
```

---

## 5. Authentication Flow

Authentication is fully delegated to PocketBase at `pb.auravo.ai`. The app itself has no user table and no session management.

### Login page flow

```
/login  →  login.html
           login.js (ES module)
           │
           ├── Email Code (OTP)  →  step 1: pb.collection('users').requestOTP(email) → {otpId}
           │                        step 2: pb.collection('users').authWithOTP(otpId, code)
           │                        On success → window.location.replace('/app')
           │                        Default/first tab — the only sign-*in* method.
           │
           └── Sign Up  →  pb.collection('users').create({name, email, password, passwordConfirm})
                          then authWithPassword (once, immediately) → redirect to /app
                          Kept only so brand-new users can create an account — OTP
                          requires an existing record and can't do that itself.
```

**Password Sign In and Google OAuth2 are intentionally not exposed on this page** — OTP is the
only sign-*in* method by design (email code only, no password to manage). Both methods remain
fully enabled on the shared PocketBase `users` collection itself (other apps sharing this auth
backend still use them), this is a voassess-side UI choice, not a server-side policy change.

The Email Code tab is a single two-step form (`#formOtp` in `login.html`) — the same submit
handler (`handleOtpSubmit`) branches on whether the code field is currently hidden to decide
whether it's requesting a code or verifying one. `otpId` is held in a module-level variable
between the two steps. The tab is disabled server-side-aware: if
`listAuthMethods().otp.enabled` is false, the tab is disabled with an explanatory tooltip and
an error banner, since there'd otherwise be no way to sign in at all.

### App guard (auth-check.js)

Every page load of `/app` runs `auth-check.js` (ES module, loaded first):

```
auth-check.js
├── new PocketBase('https://pb.auravo.ai')
├── if (!pb.authStore.isValid) → window.location.replace('/login')  [STOP]
└── else
    ├── read pb.authStore.record  (name / email)
    ├── inject .user-pill into .header-inner
    └── wire up Sign Out → pb.authStore.clear() → redirect to /login
```

PocketBase SDK stores the auth token in `localStorage` automatically. Token refresh is handled by the SDK on each `authStore` access.

### Token persistence

- Key: `pocketbase_auth` in `localStorage`
- Contains: `token` (JWT) + `record` (user record snapshot — `authStore.model` also still works, it's a deprecated alias for the same value in this SDK version, not yet removed)
- Validity checked via `pb.authStore.isValid` (checks expiry, not server round-trip)

---

## 6. Recording Pipeline

### Browser-side

```
navigator.mediaDevices.getUserMedia({ audio: true })
    │
    ├── Web Audio API  →  AnalyserNode  →  canvas waveform  (60 fps rAF loop)
    │                      fftSize=256, getByteTimeDomainData()
    │
    └── MediaRecorder (preferred MIME order):
        audio/webm;codecs=opus  →  audio/webm  →  audio/ogg  →  audio/mp4

        ondataavailable (every 200 ms)  →  audioChunks[]
        onstop  →  new Blob(audioChunks)  →  submitAssessment(blob, seconds)
```

**Timer:** max 120 seconds (`MAX_SECS`). `setInterval` increments `recordSeconds` every second; auto-stops at limit.

**Waveform:** `AnalyserNode` reads `Uint8Array(128)` of time-domain amplitude values; drawn as a continuous polyline on a `<canvas>`. Idle state shows a flat centre line.

### Server-side conversion

```python
subprocess.run([
    "ffmpeg", "-y", "-i", raw_path,   # input: browser WebM/Opus
    "-ar", "16000",                   # output: 16 kHz (Whisper requirement)
    "-ac", "1",                       # mono
    wav_path
])
```

Whisper processes the WAV directly. Both files are retained in `data/recordings/` for potential replay or debugging.

---

## 7. Assessment Pipeline

All scoring runs synchronously in a single `/assess` request after transcription completes.

```
Whisper.transcribe(wav, language="en", word_timestamps=True)
        │
        ├── transcript  (full text string)
        ├── segments[]  (each: {text, avg_logprob, words:[{word, probability}]})
        │
        ├── PACE
        │   wpm = (word_count / duration_secs) * 60
        │   pace_score = score_pace(wpm)
        │
        ├── FILLERS
        │   regex scan of 20 filler phrases in lowercased transcript
        │   filler_score = score_fillers(count, word_count)
        │
        ├── GRAMMAR  (if Java/language_tool available)
        │   errors = LanguageTool("en-US").check(transcript)
        │   grammar_score = score_grammar(len(errors), word_count)
        │   grammar_issues = extract_grammar_issues(transcript, errors)  [up to 8]
        │   fallback: count repeated consecutive words via regex \b(\w+)\s+\1\b
        │
        ├── PRONUNCIATION
        │   avg_logprob across all segments  →  score_pronunciation()
        │   per-word probability < 0.78  →  extract_mispronounced()  [up to 8]
        │
        ├── CLARITY
        │   clarity = (pace + filler + grammar + pronunciation) / 4  (equal weights)
        │
        ├── OVERALL
        │   overall = pace×0.20 + filler×0.20 + pronunciation×0.25
        │           + grammar×0.20 + clarity×0.15
        │
        ├── VOCABULARY COVERAGE  (informational — not part of Overall)
        │   type-token ratio + advanced-word ratio + unique-word breadth
        │   → score_vocabulary()
        │
        ├── CEFR LEVEL  (informational — not part of Overall)
        │   composite of vocabulary/grammar/pronunciation/pace/sentence-complexity
        │   → score_cefr()  →  {score, level: A1–C2}
        │
        └── VOICE ARCHETYPE  (informational — not a numeric score)
            rule-based classification from the metrics above
            → determine_voice_archetype()
```

### Weight rationale

Pronunciation is weighted highest (0.25) because it is the primary differentiator of spoken English quality. Clarity is weighted lowest (0.15) because it is already a composite of the other four, making it partially redundant in the overall calculation.

---

## 8. Scoring Algorithms

### Pace (WPM → 0–100)

| WPM range | Score | Interpretation |
|---|---|---|
| 120–150 | 100 | Ideal |
| 110–119 or 151–160 | 88 | Slightly off |
| 100–109 or 161–175 | 72 | Noticeable |
| 80–99 or 176–195 | 52 | Poor |
| 60–79 or 196–220 | 32 | Very poor |
| < 60 or > 220 | 15 | Unacceptable |

### Filler Words (count/words ratio → 0–100)

| Ratio | Score |
|---|---|
| 0 | 100 |
| < 2% | 90 |
| < 5% | 70 |
| < 8% | 50 |
| < 12% | 28 |
| ≥ 12% | 10 |

**Filler word list (20 phrases):** um, uh, er, ah, like, you know, basically, actually, literally, right, so yeah, i mean, kind of, sort of, you see, okay so, anyway, so basically, well.

### Grammar (error/word ratio → 0–100)

| Ratio | Score |
|---|---|
| 0 | 100 |
| < 2% | 90 |
| < 5% | 74 |
| < 8% | 56 |
| < 12% | 36 |
| ≥ 12% | 16 |

### Pronunciation (Whisper avg_logprob → 0–100)

Whisper returns `avg_logprob` per segment in range [−∞, 0], where 0 is maximum confidence. Score is mapped linearly from [−1.0, −0.1] to [0, 100]:

```
score = (avg_logprob + 1.0) / 0.9 × 100
clamped to [10, 100]
```

**Word-level flagging:** Per-word `probability < 0.78` triggers inclusion in the mispronounced list. Words shorter than 3 characters and 100+ common function words (articles, pronouns, prepositions, auxiliaries) are excluded to avoid false positives. Results are sorted worst-first and capped at 8 words.

### Clarity (composite)

```
clarity = (pace_score + filler_score + grammar_score + pronunciation_score) / 4
```

### Overall

```
overall = pace×0.20 + filler×0.20 + pronunciation×0.25 + grammar×0.20 + clarity×0.15
```

### Vocabulary Coverage (lightweight lexical proxy)

Not a certified lexical analysis — a practical heuristic blending three signals over the
transcript's words. Deliberately strict: raw type-token ratio inflates trivially on short
samples (fewer words means fewer chances to repeat one), so a length penalty is applied on
top of the usual blend — a single short sentence scores low, not spuriously high:

```
diversity_score = min(unique_words / total_words / 0.75, 1.0) × 100     # type-token ratio
length_factor    = min(total_words / 80, 1.0)                            # full credit needs ~80+ words
diversity_score *= (0.4 + 0.6 × length_factor)                           # short samples cap at 40% of raw

sophistication_score = min(advanced_word_ratio / 0.5, 1.0) × 100        # unique words outside
                                                                            the ~250-word _CORE_VOCAB
                                                                            baseline, len > 3
breadth_score         = min(unique_words / 70, 1.0) × 100

vocabulary_score = diversity_score×0.35 + sophistication_score×0.40 + breadth_score×0.25
```

`_CORE_VOCAB` is a hardcoded ~250-word list approximating everyday A1/A2 vocabulary — words
outside it (and longer than 3 characters) count as "advanced" usage. This is an approximation
for self-practice feedback, not a linguistically validated wordlist.

### CEFR Level (heuristic A1–C2 estimate)

Also deliberately strict — thresholds are set high (B2+ requires genuinely strong, sustained
speech) and the whole composite is dampened for short samples, same rationale as Vocabulary
Coverage above:

```
avg_sentence_length = word_count / sentence_count
complexity_score    = min(avg_sentence_length / 16, 1.0) × 100

composite = vocabulary_score×0.30 + grammar_score×0.25 + pronunciation_score×0.20
          + pace_score×0.10 + complexity_score×0.15

evidence_factor = min(word_count / 60, 1.0)                              # full credit needs ~60+ words
cefr_score = composite × (0.5 + 0.5 × evidence_factor)
```

| `cefr_score` | Level |
|---|---|
| < 35 | A1 |
| < 50 | A2 |
| < 65 | B1 |
| < 80 | B2 |
| < 92 | C1 |
| ≥ 92 | C2 |

Like Vocabulary Coverage, this is a self-practice approximation, not a certified CEFR
placement result — no calibration against real CEFR-rated speech samples has been done. The
same formula (with the same thresholds) is used for both the single-recording version above
and `aggregate_cefr()`'s assessment-wide version (§19) — the latter's `evidence_factor` is
based on the combined word count across Describe & Compare + Full Assessment, which is
naturally larger, so it's rarely the limiting factor there.

### Voice Archetype (rule-based, not a numeric score)

A fun, categorical read on speaking style, evaluated as an ordered if/elif chain over the
metrics above (first matching rule wins) in `determine_voice_archetype()`: **The Orator**
(high pronunciation + pace + clarity), **The Storyteller** (high vocabulary + clarity),
**The Analyst** (high grammar + CEFR + low filler), **The Sprinter** (fast pace, weaker
pronunciation), **The Explorer** (high filler count), **The Diplomat** (balanced, no metric
more than 20 points from another), **The Steady Narrator** (solid all-round), falling back to
**The Rising Voice** (encouraging default for developing speakers). Each returns an emoji,
description, and 2–3 trait tags — not persisted beyond the archetype name itself.

### Score colour thresholds (frontend)

| Range | Class | Colour |
|---|---|---|
| ≥ 80 | `good` | `#10b981` green |
| 60–79 | `ok` | `#f59e0b` amber |
| < 60 | `poor` | `#ef4444` red |

---

## 9. API Reference

### `POST /assess`

Accepts a multipart form with the recorded audio file and duration.

**Request**

```
Content-Type: multipart/form-data

audio     : File   — WebM/Opus recording from browser MediaRecorder
duration  : float  — recording length in seconds (from JS timer)
```

**Response 200**

```json
{
  "id": 42,
  "transcript": "Today I want to talk about...",
  "pace":         { "score": 88.0, "wpm": 147.3 },
  "filler":       { "score": 70.0, "count": 4, "words": ["um×2","like×2"] },
  "pronunciation":{ "score": 76.4, "issues": [{"word":"particularly","confidence":61}] },
  "grammar":      { "score": 90.0, "errors": 2, "issues": [
                    {"wrong":"there is","correct":"there are","message":"...","context":"..."}
                  ]},
  "clarity":      { "score": 81.1 },
  "vocabulary":   { "score": 73.0, "unique_words": 41, "total_words": 63, "advanced_ratio": 27.3 },
  "cefr":         { "score": 77.6, "level": "C1", "avg_sentence_length": 12.0 },
  "archetype":    { "archetype": "The Analyst", "emoji": "🧠",
                    "description": "Precise and structured...",
                    "traits": ["Grammatically precise", "Minimal filler", "Structured"] },
  "overall":      81.2,
  "feedback":     "Great pace at 147 WPM...",
  "duration":     63
}
```

`vocabulary` and `cefr` are informational — they do not factor into `overall` (see §7/§8).
`archetype` is categorical, not a score; only `archetype.archetype` (the label) is persisted.

**Error responses**

| Status | Cause |
|---|---|
| 400 | Empty transcript (inaudible or silent recording) |
| 413 | nginx body size limit exceeded (fix: `client_max_body_size 50M`) |
| 500 | ffmpeg conversion failure or unexpected exception |
| 502/504 | Whisper or grammar tool took too long (fix: increase `proxy_read_timeout`) |

---

### `GET /history`

Returns up to 60 most recent assessments, newest first.

**Response 200**

```json
[
  {
    "id": 42, "timestamp": "20250508_143022", "duration": 63,
    "overall": 81.2, "pace": 88.0, "filler": 70.0,
    "pronunciation": 76.4, "grammar": 90.0, "clarity": 81.1,
    "wpm": 147.3, "filler_count": 4, "grammar_errors": 2,
    "transcript": "Today I want to...", "feedback": "Great pace...",
    "vocabulary": 73.0, "cefr_score": 77.6, "cefr_level": "C1", "archetype": "The Analyst"
  }
]
```

---

### Static file routes

| Route | File served |
|---|---|
| `GET /` | Redirect → `/login` |
| `GET /login` | `login.html` |
| `GET /login.css` | `login.css` |
| `GET /login.js` | `login.js` |
| `GET /auth-check.js` | `auth-check.js` |
| `GET /app` | `index.html` |
| `GET /style.css` | `style.css` |
| `GET /script.js` | `script.js` |

---

## 10. Database Schema

The `assessments` table in `data/assessments.db` (a second table, `english_assessments`,
covers the guided flow — see §19):

```sql
CREATE TABLE IF NOT EXISTS assessments (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT NOT NULL,       -- "YYYYMMDD_HHMMSS"
    duration             REAL,                -- seconds
    transcript           TEXT,
    pace_score           REAL,
    filler_score         REAL,
    pronunciation_score  REAL,
    grammar_score        REAL,
    clarity_score        REAL,
    overall_score        REAL,
    pace_wpm             REAL,
    filler_count         INTEGER,
    grammar_errors       INTEGER,
    filler_words         TEXT,               -- "um×2, like×1" (comma-separated)
    feedback             TEXT,
    vocabulary_score     REAL,
    cefr_score           REAL,
    cefr_level           TEXT,                -- "A1"–"C2"
    archetype            TEXT                 -- just the label, e.g. "The Storyteller"
);
```

The four `vocabulary_score`/`cefr_score`/`cefr_level`/`archetype` columns were added after the
table already existed in production, so `init_db()` adds them via `ALTER TABLE ... ADD COLUMN`
(wrapped in a try/except for "column already exists") rather than relying on
`CREATE TABLE IF NOT EXISTS`, which is a no-op once the table is present. Rows written before
this change have `NULL` in these four columns.

**Notes:**
- Pronunciation issues and grammar issue detail (wrong/correct pairs) are **not** persisted — they are derived live from each transcription and returned in the `/assess` response only. The same applies to the Voice Archetype's description/emoji/traits — only the label is stored.
- The `filler_words` column stores a human-readable summary string, not JSON.
- `user_id` (TEXT) stores the PocketBase user ID that created the row, added the same migration-safe way as the other late columns. Rows written before this change have `NULL` here and are invisible to `/history` (which filters by `user_id`) — no data was deleted, but pre-fix rows aren't attributable to a specific account and stay hidden rather than being guessed at.

---

## 11. Frontend Architecture

The frontend is a zero-build Vanilla JS SPA. There is no bundler, no framework, and no npm dependency beyond CDN links.

### File responsibilities

| File | Scope | Module type |
|---|---|---|
| `auth-check.js` | Auth guard + user pill injection | ES module (`type="module"`) |
| `login.js` | OTP sign-in / sign-up | ES module |
| `script.js` | All app logic (recording, results, tabs, charts) | Classic script |

`auth-check.js` is loaded **before** `script.js` in `index.html` to ensure the auth guard fires first:

```html
<script type="module" src="auth-check.js"></script>
<script src="script.js"></script>
```

`auth-check.js` also exposes `window.authFetch(url, options)` — a `fetch()` wrapper that attaches
the PocketBase session token as an `x-pb-token` header, retries once via `authRefresh()` on a 401,
and redirects to `/login` if that also fails. Every data-touching call in `script.js`/`assessment.js`
(`/assess`, `/history`, `/assess/stage`, `/assessment/finalize`, `/assessment/history`) goes through
`authFetch` rather than bare `fetch` so the backend can resolve and scope requests to a real user.

### State variables (`script.js`)

| Variable | Type | Purpose |
|---|---|---|
| `mediaRecorder` | MediaRecorder \| null | Active recorder instance |
| `audioChunks` | Blob[] | Collected audio data chunks |
| `recordingTimer` | number | setInterval ID for the countdown |
| `recordSeconds` | number | Elapsed recording seconds |
| `audioCtx` | AudioContext \| null | Web Audio context for waveform |
| `analyser` | AnalyserNode \| null | FFT node for amplitude data |
| `waveAnim` | number | rAF handle for waveform loop |
| `charts` | `{[canvasId]: Chart}` | Chart.js instance registry (prevent memory leaks on re-render) |
| `_activeTtsBtn` | Element \| null | Currently speaking TTS button (for `playing` CSS class) |

### Tab switching

Clicking a `.tab-btn` sets it `active`, hides all `.tab-content`, shows `#tab-{id}`, then calls `loadHistory()` or `loadProgress()` lazily (data is always re-fetched from the server, never cached).

### Result display sequence (`displayResults`)

```
1.  Animate SVG ring (stroke-dashoffset interpolation via CSS transition)
2.  Colour overall score element by threshold
3.  setMetric() × 5  (score text + progress bar + detail text)
4.  Word count badge
5.  Transcript text
6.  Feedback text
7.  displayPronunciationIssues() → build pronun-panel HTML
8.  displayGrammarIssues()       → build grammar-panel HTML
9.  renderRadar('result-radar')  → Chart.js radar
10. Unhide results-wrap
11. Smooth scroll to results
```

---

## 12. Chart System

All charts use **Chart.js 4** loaded from CDN. Each chart is stored in the `charts` object keyed by canvas ID. Before rendering, the existing chart instance is `.destroy()`ed to prevent memory leaks and canvas conflicts.

### Radar chart (Speech Profile)

- Used in: Results tab (per-session) and Progress tab (latest session)
- Canvas IDs: `result-radar`, `progress-radar`
- Data: 5 axes — Pace, Filler Words, Pronunciation, Grammar, Clarity

### Line charts (Trend)

- Canvas IDs: `overall-trend`, `chart-pace`, `chart-filler`, `chart-pronunciation`, `chart-grammar`, `chart-clarity`
- X-axis: session index labels (`#1`, `#2`, …) in chronological order
- Y-axis: 0–100, gridlines at 20-unit intervals
- Data sourced from `/history`, reversed to chronological order

### Per-metric colours

| Metric | Colour |
|---|---|
| Pace | `#06b6d4` cyan |
| Filler Words | `#f59e0b` amber |
| Pronunciation | `#10b981` green |
| Grammar | `#3b82f6` blue |
| Clarity | `#8b5cf6` purple |

---

## 13. TTS Architecture

All text-to-speech uses the browser-native **Web Speech API** (`window.speechSynthesis`). No audio is fetched from the server.

```
_speak(text, rate, btn)
├── Cancel any currently playing utterance
├── Remove 'playing' class from previously active button
├── Create SpeechSynthesisUtterance({ text, lang:'en-US', rate })
├── Add 'playing' class to btn
├── speechSynthesis.speak(utt)
└── utt.onend / utt.onerror → remove 'playing' class
```

### TTS entry points

| Function | Rate | Trigger |
|---|---|---|
| `speakWord(btn, word)` | 0.72 | ▶ Hear it (pronunciation panel) |
| `speakCorrection(btn, correction)` | 0.85 | ▶ Hear correction (grammar panel) |
| feedback speak button | 0.92 | 🔊 Read Aloud (feedback card) |

Slower rate for individual words (0.72) makes phonemes more distinguishable for learning purposes.

### Limitations

- Voice quality and language support depends on the OS/browser.
- `speechSynthesis` on iOS Safari requires a user gesture to initiate (the ▶ buttons satisfy this).
- Only one utterance plays at a time — starting a new one cancels the current.

---

## 14. Error Handling

### Frontend error states

| HTTP status | Handling |
|---|---|
| 400 | JSON `{detail}` parsed and shown in `#record-status` |
| 413 | Detected before JSON parse; shows nginx body limit message |
| 500 | JSON `{detail}` parsed and shown |
| 502 / 504 | Detected before JSON parse; shows timeout message |
| Network failure | `catch(err)` → `setStatus(err.message)` |

### Backend error states

| Condition | Response |
|---|---|
| ffmpeg not found or fails | HTTP 500 with stderr output |
| Empty transcript | HTTP 400: "Could not transcribe — please speak clearly" |
| Grammar tool unavailable | Graceful degradation: repeated-word regex fallback, `GRAMMAR_OK=False` |
| Whisper segments missing `avg_logprob` | Returns default score 72.0 |

### Grammar tool degradation

If `language_tool_python` is unavailable (Java not installed or import fails), the backend sets `GRAMMAR_OK = False` and uses a fallback:

```python
ge = len(re.findall(r"\b(\w+)\s+\1\b", tl))  # count repeated consecutive words
```

This detects only the most obvious error type (stuttered repeated words) but avoids crashing. `grammar_issues` list will be empty, so the grammar detail panel is hidden.

---

## 15. Deployment & Infrastructure

### Process model

```
systemd / manual:  python app.py
                   → uvicorn on 0.0.0.0:5050

Nginx (voassess.auravo.ai):
    proxy_pass http://127.0.0.1:5050;
    client_max_body_size 50M;
    proxy_read_timeout   300s;
    proxy_send_timeout   300s;
```

### Startup sequence

On server start, `app.py` does:

1. Creates `data/` and `data/recordings/` directories
2. Initialises SQLite database + creates `assessments` table if absent
3. Loads Whisper `base` model into memory (~145 MB RAM, ~5–10 s cold start)
4. Attempts to load `language_tool_python` (downloads grammar rules on first run, ~200 MB)
5. Binds FastAPI on port 5050

### Whisper model size comparison

| Model | Parameters | WER | RAM | Inference (CPU) |
|---|---|---|---|---|
| tiny | 39M | ~12% | ~390 MB | ~3 s/min audio |
| **base** | **74M** | **~9%** | **~580 MB** | **~5 s/min audio** |
| small | 244M | ~7% | ~960 MB | ~12 s/min audio |
| medium | 769M | ~5.5% | ~3 GB | ~30 s/min audio |

`base` is the current choice — good accuracy, acceptable cold inference (~5–10 s for a 1-minute recording on a mid-range CPU VPS).

### Python dependencies

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
openai-whisper>=20231117
python-multipart>=0.0.6
httpx>=0.27.0
language-tool-python>=2.7.1
```

---

## 16. Security Considerations

| Area | Current state | Notes |
|---|---|---|
| Auth enforcement | Frontend guard (auth-check.js) **and** backend | All data-touching routes (`/assess`, `/history`, `/assess/stage`, `/assessment/finalize`, `/assessment/history`) require an `x-pb-token` header, verified server-side against PocketBase's `auth-refresh` endpoint (`require_user()` in app.py) before any DB read/write. Results are scoped to `user_id`. `/assessment/manifest` stays public — static reference content, no user data. |
| File storage | Raw WebM + WAV stored on disk | No automatic cleanup; disk can fill up over time. |
| Input sanitisation | HTML escaping via `esc()` in JS | All user-visible strings from transcript/feedback go through `esc()` before `innerHTML`. |
| SQL injection | Parameterised queries (`?` placeholders) | No string interpolation in SQL. |
| CORS | `allow_origins=["*"]` | Appropriate for a public tool; tighten to specific origins if access control matters. |
| Command injection | `subprocess.run` with list args (not shell=True) | ffmpeg args are derived from server-generated paths, not user input. |
| Whisper model | Local inference | No audio leaves the server; no third-party STT API. |

---

## 17. Known Limitations

### Pronunciation scoring

Whisper `avg_logprob` is a **transcription confidence** metric, not a true phoneme-level pronunciation score. It correlates with clarity and enunciation but:
- Accented speech may score lower even when pronunciation is consistent
- Background noise lowers scores regardless of pronunciation quality
- Short recordings (< 10 words) produce unreliable segment-level statistics

### Grammar tool

- Requires Java JRE on the server. If unavailable, falls back to a very crude regex.
- `language_tool_python` downloads a ~200 MB rule set on first use.
- Some false positives: informal spoken contractions (e.g. "gonna", "wanna") are flagged.

### Single-instance concurrency

Whisper inference blocks the Python event loop. FastAPI/Uvicorn is single-worker by default; two simultaneous `/assess` requests will queue. For concurrent use, run multiple workers (`--workers N`) and note that Whisper model loading per-worker multiplies RAM usage.

### Audio file accumulation

`data/recordings/*.webm` and `*.wav` are never deleted. A 2-minute recording is ~3 MB (WebM) + ~3.8 MB (WAV) = ~7 MB per session. 1000 sessions ≈ 7 GB disk.

---

## 18. Potential Improvements

### Short-term

| Improvement | Effort | Impact |
|---|---|---|
| Update chart colours to Auravo palette (`#e05530` radar) | Low | Visual consistency |
| Update waveform stroke colour to Auravo coral | Low | Visual consistency |
| Automatic cleanup of old recording files (keep last N) | Low | Disk management |
| Persist pronunciation + grammar issues in DB | Medium | Full history detail |

### Medium-term

| Improvement | Effort | Impact |
|---|---|---|
| Whisper `small` model option for higher accuracy | Low (config change) | Better transcription |
| Background task queue (e.g. `asyncio.to_thread`) | Medium | Concurrent request support |
| Vocabulary drill mode (speak a displayed sentence, score it) | High | Active practice |
| Export report as PDF | Medium | Shareable results |

### Long-term

| Improvement | Effort | Impact |
|---|---|---|
| True phoneme-level pronunciation (Montreal Forced Aligner or Kaldi) | Very High | Accurate pronunciation feedback |
| GPT/Claude-powered feedback narrative (richer, adaptive) | Medium | Better personalisation |
| Spaced repetition for flagged words | High | Retention-oriented learning |
| Mobile app (React Native / Expo) | High | Accessibility |

---

## 19. English Assessment (Guided Flow)

A second, guided assessment lives alongside the free-form Record tab: a five-stage wizard
(`index.html` → `#tab-assessment`, driven entirely by `assessment.js`, a self-contained
classic script that shares only CSS with `script.js`, no state).

### Stages

| # | Stage | Source assets | Recording timing | What's captured |
|---|---|---|---|---|
| 1 | State your name | — | Manual start, 15s max | Short recording, transcribed; name extracted via regex, used to personalise the report. Not scored. |
| 2 | Picture Talk | `PictureTalk/Picture{1,2,3}.png` | **Auto-starts immediately**, 15s max, manual early-stop | Each image has its target sentence baked into the picture itself. User reads it aloud. |
| 3 | Listen & Repeat | `MediaRepeat/clip-*.wav` | **Auto-starts immediately**, 15s max, manual early-stop | User listens to a reference clip, then repeats it from memory (no text shown). |
| 4 | Describe & Compare | `PictureDescribe/{PicD-1..4,Pic4-5}.png` | 30s skippable prep, then auto-record for `speak_secs` (120s or 180s per item) | 2 single-picture "describe" prompts, 2 "compare two pictures" prompts, 1 text-only follow-up — prompt text is read from each slide image and hardcoded into the manifest. |
| 5 | Full Assessment | — | 30s skippable prep, then auto-record, 180s (3 min) max | The same free-speech pipeline as the Record tab — this is the stage the report's headline **Overall** score comes from. |

Reference sentences for stages 2–3 aren't derivable from the assets automatically — Picture
Talk's captions were read directly off the images, and Media Repeat's were transcribed once
with Whisper and hardcoded into `ASSESSMENT_MANIFEST` in `app.py`. The manifest served to the
browser via `GET /assessment/manifest` strips the `sentence` field from `media_repeat` items
(the point of that stage is listening, not reading the answer off the screen).

### Timing & recording UX

Two distinct patterns, both in `assessment.js`:

- **Stages 2–3 (15s)** — recording starts the instant the screen renders (`renderRecorder({autoStart:true, maxSecs:15})`), no waiting period. The visible timer counts *down* from 15s and auto-stops the recording at 0, but the user can stop manually at any point to move on immediately — there's no minimum recording length enforced.
- **Stages 4–5 (30s prep)** — `prepCountdown()` shows a 30-second countdown with an "I'm Ready — Record Now" button. Clicking it, or letting the countdown reach 0, both transition into `renderRecorder({autoStart:true, ...})` the same way; the countdown is purely about *when recording begins*, never a hard requirement to wait it out.

Stage 1 (name) keeps the original manual-start recorder (click to begin), unchanged.

No per-stage or per-item score is displayed anywhere in the wizard — each screen goes
straight from recording to the next stage with no intermediate results screen. The only
scores shown are in the single comprehensive report at the very end (`renderReport()`,
built from `POST /assessment/finalize`).

### PDF export

The report screen's primary action is **Export PDF Report**, not a "start over" button —
`downloadPdfReport()` in `assessment.js` builds a multi-page PDF entirely client-side with
[jsPDF](https://github.com/parallax/jsPDF) (loaded from CDN in `index.html`, same pattern as
Chart.js — no server-side PDF dependency). It covers the overall score, Voice Archetype, the
five core Full Assessment metrics, assessment-wide Vocabulary/CEFR, the three guided-stage
section scores, the full transcript, and personalised feedback, paginating automatically via
a running `y` cursor and `ensureSpace()` page-break check. "Start New Assessment" is still
available as a secondary (`.assess-btn-secondary`) button next to it.

### Scoring

- **Picture Talk / Listen & Repeat** — not free speech; scored as a repeat-accuracy task:
  `score = word_accuracy × 0.6 + pronunciation_score × 0.4`, where `word_accuracy` is a
  `difflib.SequenceMatcher` ratio over normalised (lowercased, punctuation-stripped) word
  sequences between the reference sentence and what Whisper transcribed.
- **Describe & Compare / Full Assessment** — both reuse the exact same five-metric pipeline
  as `POST /assess` (`score_free_speech()` in `app.py`, factored out of the original `/assess`
  route so both entry points share one implementation).
- **Overall report score** — is the Full Assessment stage's `overall`, not a blend of all five
  stages. Picture Talk / Listen & Repeat / Describe & Compare are surfaced as supporting
  section averages, not folded into the headline number.
- **Vocabulary Coverage (report-level)** — unlike the Record tab, the wizard's Vocabulary
  Coverage is **not** taken from any single recording. `aggregate_vocabulary()` concatenates
  the transcripts of all Describe & Compare items *and* the Full Assessment recording, then
  runs the same `score_vocabulary()` used elsewhere over that combined text — a larger,
  more representative sample than either alone.
- **CEFR Level (report-level)** — likewise computed assessment-wide by `aggregate_cefr()`,
  not from a single recording: repeat-task accuracy and pronunciation are averaged across
  Picture Talk + Listen & Repeat; grammar, pronunciation, pace, and sentence-complexity are
  averaged across Describe & Compare + Full Assessment (using the same combined transcript
  as Vocabulary Coverage); the "state your name" stage contributes no signal (no assessable
  linguistic content). Weights: `vocab×0.25 + grammar×0.20 + pronunciation×0.20 + pace×0.10
  + complexity×0.10 + repeat_accuracy×0.15`, bucketed into A1–C2 the same way as the
  single-recording version (§8).
- Voice Archetype is **not** re-aggregated — the report shows the Full Assessment
  recording's own archetype, unchanged.

### API

| Route | Purpose |
|---|---|
| `GET /assets/picture-talk/*`, `/assets/media-repeat/*`, `/assets/picture-describe/*` | Static-mounted asset directories |
| `GET /assessment/manifest` | Public manifest (stages, prompts, timings) |
| `POST /assess/stage` | Requires `x-pb-token`. Score one recording. Form fields: `stage_type` (`name`\|`picture_talk`\|`media_repeat`\|`picture_describe`\|`final`), `stage_id`, `audio`, `duration`. Does not persist — recordings land in `data/recordings/` with a `stage_*` filename prefix, same no-cleanup caveat as the Record tab. |
| `POST /assessment/finalize` | Requires `x-pb-token`. JSON body `{name, stages:[...]}` (the full list of `/assess/stage` responses collected client-side). Computes section averages, persists one row tagged with the caller's `user_id`, returns the full report. |
| `GET /assessment/history` | Requires `x-pb-token`. Last 30 of the caller's own finalized comprehensive reports, newest first. |

### Database

New table `english_assessments` — one row per completed wizard run: per-section average
scores, the seven Full-Assessment metrics (the original five plus `vocabulary_score` and
`cefr_score`/`cefr_level`) and `archetype`, and a `stage_results` column holding the complete
per-stage JSON (transcripts, per-item scores, issues) for later inspection, plus a `user_id`
column scoping each row to its creator (same as the `assessments` table, §10).
