"""
VoiceCoach – Voice Assessment Backend
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import subprocess, re, os, json, difflib
from pathlib import Path
from datetime import datetime
import httpx
import whisper

# Load .env (if present) into the process environment BEFORE any project
# module reads os.environ — pronunciation_provider.py's and stt_provider.py's
# Saaras adapters read SARVAM_API_KEY etc. at *import* time (constructing the
# module-level pronunciation_registry / this file's stt_registry), so this
# must run before those imports below, not just before app.py's own code
# uses os.environ. Safe to skip if python-dotenv isn't installed — real
# deployments should set actual process env vars instead of relying on a
# .env file anyway (see .env.example).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from vocabulary import score_vocabulary
from languagetool_provider import LanguageToolProvider, LanguageToolUnavailable
from grammar_heuristics import augment_grammar_issues
from filler_detector import detect_fillers, summarize_words
from audio_utils import wav_duration_seconds, analyze_pauses
from pronunciation_provider import (
    pronunciation_registry, resolve_pronunciation, PronunciationProviderError,
    PRONUNCIATION_PROVIDER_NAMES,
)
from stt_provider import (
    STTProviderRegistry, STTProviderError, STT_PROVIDER_NAMES,
    WhisperSTTProvider, SaarasSTTProvider,
)
from groq_provider import generate_teacher_report
from grammar_context_validator import apply_contextual_validation
import repository as db

# HTTP LanguageTool server (preferred grammar/analyze source — see score_free_speech()).
# Construction is cheap (just stores config, no network call, no JVM startup),
# unlike the local `language_tool_python.LanguageTool(...)` below, so this is
# safe to create at import time even if the server isn't running yet.
_lt_provider = LanguageToolProvider()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RECORDINGS_DIR = DATA_DIR / "recordings"
DATA_DIR.mkdir(exist_ok=True)
RECORDINGS_DIR.mkdir(exist_ok=True)
# Database persistence (schema, connections, queries) lives entirely in
# repository.py — see that module's docstring. DB_PATH is kept here as an
# alias only because a couple of call sites below historically referenced
# app.DB_PATH; it now just points at the repository module's configured path.
DB_PATH = db.DB_PATH

# Debug user ID for storing debug assessments
DEBUG_USER_ID = "debug_user"

# Tutor view password - set in .env or use default
TUTOR_VIEW_PASSWORD = os.environ.get("TUTOR_VIEW_PASSWORD", "tutor123")

PICTURE_TALK_DIR     = BASE_DIR / "PictureTalk"
MEDIA_REPEAT_DIR     = BASE_DIR / "MediaRepeat"
PICTURE_DESCRIBE_DIR = BASE_DIR / "PictureDescribe"

PB_URL = "https://pb.auravo.ai"

async def verify_user_token(pb_token: str) -> dict | None:
    """Return the PocketBase user record if the session token is valid, else None."""
    if not pb_token:
        return None
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.post(f"{PB_URL}/api/collections/users/auth-refresh",
                              headers={"Authorization": pb_token})
            if r.status_code != 200:
                return None
            return r.json().get("record")
    except httpx.RequestError:
        return None

async def require_user(request: Request) -> dict:
    """Auth gate for data-touching routes. Every recording/history endpoint must
    resolve to a real PocketBase user so results are scoped per-user, not shared
    globally — see DESIGN.md §16 (this was previously an open gap)."""
    user = await verify_user_token(request.headers.get("x-pb-token", ""))
    if not user:
        raise HTTPException(401, "Login required — please sign in again.")
    return user

app = FastAPI(title="VoiceCoach")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/assets/picture-talk",     StaticFiles(directory=str(PICTURE_TALK_DIR)),     name="picture-talk")
app.mount("/assets/media-repeat",     StaticFiles(directory=str(MEDIA_REPEAT_DIR)),     name="media-repeat")
app.mount("/assets/picture-describe", StaticFiles(directory=str(PICTURE_DESCRIBE_DIR)), name="picture-describe")

# ── Whisper (local STT) ───────────────────────────────────────────────────────
print("Loading Whisper model (base)…")
_whisper = whisper.load_model("base")

# STT provider registry — constructed here (not as a module-level singleton
# inside stt_provider.py) because WhisperSTTProvider wraps the already-loaded
# `_whisper` model above rather than loading its own.
#
# Saaras (Sarvam) is the primary/default STT provider — it preserves raw
# fillers/disfluencies ("um", "uh", repetitions, false starts) the way this
# pipeline's transcript needs to, which Whisper tends to normalize away.
# Whisper stays registered as the always-available fallback for an explicit
# whisper request that fails (see resolve_stt() below) and as the "whisper"
# option in the stt_provider selector.
DEFAULT_STT_PROVIDER = "saaras"
stt_registry = STTProviderRegistry({
    "whisper": WhisperSTTProvider(_whisper),
    "saaras": SaarasSTTProvider(),
})


def validate_stt_provider(name: str) -> str:
    """Shared validation for the `stt_provider` request field. Same pattern
    as validate_pronunciation_provider() — an unknown name is a 422, not a
    silent default."""
    if name not in STT_PROVIDER_NAMES:
        raise HTTPException(
            422,
            f"Invalid stt_provider '{name}'. Must be one of: {', '.join(STT_PROVIDER_NAMES)}.",
        )
    return name


def resolve_stt(provider_name: str, wav_path: Path):
    """Resolve + transcribe through the registry, then apply the same
    fallback-with-honesty rule pronunciation providers use: if a
    *non-default* requested STT provider can't produce a transcript (not
    configured, request failed, etc.), fall back to the default provider
    (Saaras) for the actual transcript/segments, but return BOTH results so
    the caller can report what really happened (never silently relabel one
    provider's output as another's).

    If the DEFAULT provider itself (Saaras) fails, there is deliberately no
    fallback to Whisper here — raise HTTPException(400) with Saaras's own
    detail instead. Silently substituting Whisper's transcript would hide a
    real failure behind output from a model that's known to normalize away
    the fillers/disfluencies ("um", "uh", repetitions, false starts) this
    pipeline needs to preserve — see stt_provider.SaarasSTTProvider. A clear
    error is more useful here than a quietly-degraded transcript."""
    requested = stt_registry.get(provider_name).transcribe(wav_path)
    if requested.available:
        return requested, requested
    if provider_name == DEFAULT_STT_PROVIDER:
        raise HTTPException(400, requested.detail or "Could not transcribe — please speak clearly and try again.")
    fallback = stt_registry.get(DEFAULT_STT_PROVIDER).transcribe(wav_path)
    if not fallback.available:
        raise HTTPException(400, fallback.detail or "Could not transcribe — please speak clearly and try again.")
    return requested, fallback
print("Whisper ready.")

# ── Grammar tool (optional – requires Java) ───────────────────────────────────
try:
    import language_tool_python
    _grammar = language_tool_python.LanguageTool("en-US")
    GRAMMAR_OK = True
    print("Grammar tool ready.")
except Exception as e:
    GRAMMAR_OK = False
    print(f"Grammar tool unavailable ({e}); grammar score will be estimated.")

# ── Database ──────────────────────────────────────────────────────────────────
# Schema creation, connections, and all queries live in repository.py (the
# persistence/repository layer) — see that module's docstring. Scoring code
# in this file never touches sqlite directly; it only ever hands plain
# result dicts to the db.save_*()/db.get_*()/db.list_*() functions below.
db.init_db()

# ── English Assessment manifest ────────────────────────────────────────────
# Reference sentences for PictureTalk are the captions baked into the images.
# Reference sentences for MediaRepeat were transcribed once from the source
# clips with Whisper. PictureDescribe prompts/timings mirror the text baked
# into each slide image.
ASSESSMENT_MANIFEST = {
    "picture_talk": [
        {"id": "pt1", "image": "Picture1.png",
         "sentence": "I study for three hours every day. I use different colored pens for making notes."},
        {"id": "pt2", "image": "Picture2.png",
         "sentence": "I enjoy visiting the supermarket. I buy chocolates and chewing gum."},
        {"id": "pt3", "image": "Picture3.png",
         "sentence": "Industrialization was one of the most important developments in human history "
                      "because it transformed the way people lived, worked, and produced goods."},
    ],
    "media_repeat": [
        {"id": "mr1", "audio": "clip-1786358471410-1.wav",
         "sentence": "I like apples."},
        {"id": "mr2", "audio": "clip-1786358735764-2.wav",
         "sentence": "I sleep for seven hours every day."},
        {"id": "mr3", "audio": "clip-1786358849827-3.wav",
         "sentence": "Climate change is one of the biggest problems in the world today."},
    ],
    "picture_describe": [
        {"id": "pd1", "image": "PicD-1.png", "mode": "describe",
         "title": "Describe the Picture", "prep_label": "Look at the picture",
         "prep_secs": 45, "speak_secs": 120,
         "prompts": ["What have you observed?", "What place is this?",
                     "Describe all the details — colours, objects, emotions you can observe."]},
        {"id": "pd2", "image": "PicD-2.png", "mode": "describe",
         "title": "Describe the Picture", "prep_label": "Look at the picture",
         "prep_secs": 45, "speak_secs": 120,
         "prompts": ["What have you observed?", "What place is this?",
                     "Describe all the details — colours, objects, emotions you can observe."]},
        {"id": "pd3", "image": "PicD-3.png", "mode": "compare",
         "title": "Compare the Two Pictures", "prep_label": "Prepare your answer",
         "prep_secs": 60, "speak_secs": 180,
         "prompts": ["What do both scenarios indicate?", "Describe the images in detail.",
                     "Describe why you would prefer one over the other."]},
        {"id": "pd4", "image": "PicD-4.png", "mode": "compare",
         "title": "Compare the Two Pictures", "prep_label": "Prepare your answer",
         "prep_secs": 60, "speak_secs": 180,
         "prompts": ["What do both scenarios indicate?", "Describe the images in detail.",
                     "Describe why you would prefer one over the other."]},
        {"id": "pd5", "image": "Pic4-5.png", "mode": "followup",
         "title": "Follow-up Questions", "prep_label": "Prepare your answer",
         "prep_secs": 60, "speak_secs": 180,
         "prompts": ["How is your own daily life similar to, or different from, the pictures you saw today?",
                     "What's one change you'd like to see in the place you live?"]},
    ],
}

def public_manifest() -> dict:
    """Manifest sent to the browser — media_repeat's reference sentence is withheld
    since that stage is 'listen and repeat', not 'read and repeat'."""
    m = json.loads(json.dumps(ASSESSMENT_MANIFEST))  # deep copy
    for item in m["media_repeat"]:
        item.pop("sentence", None)
    return m

def word_accuracy(reference: str, hypothesis: str) -> float:
    """Word-level similarity between a reference sentence and what was actually
    said, using a sequence-matching ratio. 0-100, 100 = exact match."""
    norm = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()
    ref_words, hyp_words = norm(reference), norm(hypothesis)
    if not ref_words:
        return 0.0
    ratio = difflib.SequenceMatcher(None, ref_words, hyp_words).ratio()
    return round(ratio * 100, 1)

def extract_name(transcript: str) -> str:
    """Pull a name out of a free-form 'state your name' recording."""
    t = transcript.strip().rstrip(".")
    m = re.search(r"(?:my name is|i am|i'm|this is)\s+(.+)", t, re.IGNORECASE)
    return (m.group(1) if m else t).strip().title()

# ── Filler word list ──────────────────────────────────────────────────────────
FILLERS = [
    "um", "uh", "er", "ah", "like", "you know", "basically", "actually",
    "literally", "right", "so yeah", "i mean", "kind of", "sort of",
    "you see", "okay so", "anyway", "so basically", "well",
]

# ── Filler marker insertion ──────────────────────────────────────────────────
def insert_filler_markers(transcript: str, filler_occurrences: list) -> str:
    """
    Insert [word] markers into the transcript at the correct positions.

    `filler_occurrences` must be filler_detector.detect_fillers()'s
    `occurrences` list — SPOKEN filler evidence only, with `start`/`end`
    as transcript character offsets. There is deliberately no branch here
    for acoustic-only events: an acoustic low-energy/silence segment has no
    transcript character offset (its `start`/`end` are audio seconds), so
    there is no valid position to splice it into this text at. Marking one
    used to be done by reusing the audio-seconds number as if it were a
    character index, which is exactly how markers ended up landing inside
    words (e.g. "thinkin[filler]g") instead of between them. Acoustic
    hesitations are surfaced separately via `acoustic_hesitations` /
    `acoustic_hesitation_count` and are never spliced into the transcript.
    """
    if not filler_occurrences:
        return transcript

    # Sort by start offset (descending so we don't mess up positions)
    sorted_fillers = sorted(filler_occurrences, key=lambda x: x.get('start', 0), reverse=True)

    result = list(transcript)
    inserted_count = 0

    for f in sorted_fillers:
        start = f.get('start', 0)
        end = f.get('end', 0)
        word = f.get('word', '')

        # Convert to integer indices (truncate float to int)
        start_idx = int(start)
        end_idx = int(end)

        if start_idx < len(transcript) and end_idx <= len(transcript):
            # Check if the word exists at this position
            adj_start = start_idx + inserted_count
            adj_end = end_idx + inserted_count
            if adj_end <= len(result):
                actual_word = ''.join(result[adj_start:adj_end]).strip().lower()
                if actual_word == word.lower():
                    # Replace with marked version
                    result[adj_start:adj_end] = f'[{word}]'
                    # Word length stays the same, but we added 2 chars for brackets
                    inserted_count += 2

    return ''.join(result)


# ── Scoring helpers ───────────────────────────────────────────────────────────
def score_pace(wpm: float) -> float:
    """Ideal range 120–150 WPM."""
    if   120 <= wpm <= 150: return 100
    elif 110 <= wpm <  120 or 150 < wpm <= 160: return 88
    elif 100 <= wpm <  110 or 160 < wpm <= 175: return 72
    elif  80 <= wpm <  100 or 175 < wpm <= 195: return 52
    elif  60 <= wpm <   80 or 195 < wpm <= 220: return 32
    else: return 15

def score_fillers(count: int, words: int) -> float:
    ratio = count / max(words, 1)
    if   ratio == 0:      return 100
    elif ratio < 0.02:    return 90
    elif ratio < 0.05:    return 70
    elif ratio < 0.08:    return 50
    elif ratio < 0.12:    return 28
    else:                 return 10

def score_grammar(errors: int, words: int) -> float:
    rate = errors / max(words, 1)
    if   rate == 0:    return 100
    elif rate < 0.02:  return 90
    elif rate < 0.05:  return 74
    elif rate < 0.08:  return 56
    elif rate < 0.12:  return 36
    else:              return 16

# Vocabulary scoring (MATTR + wordfreq lexical sophistication + content-word
# repetition/variety, with short-response evidence dampening) lives in
# vocabulary.py — see that module's docstring for the full methodology.
# `score_vocabulary` is imported below and used exactly as before: it's
# called with a transcript and returns a dict with `score`, `unique_words`,
# `total_words`, `advanced_ratio` (plus new informational fields), so
# nothing downstream (score_cefr, determine_voice_archetype, the frontend)
# needed to change.

def score_cefr(vocab_score: float, grammar_score: float, pronun_score: float,
                pace_score: float, transcript: str) -> dict:
    """
    Heuristic CEFR (A1-C2) estimate — blends vocabulary, grammar, pronunciation,
    pace, and sentence complexity into a composite, then buckets it into a band.
    This is an approximation for self-practice feedback, not a certified placement test.

    Deliberately strict: thresholds are set high (reaching B2+ should require
    genuinely strong, sustained speech) and the whole composite is dampened for
    short samples — a single short sentence can't reliably demonstrate a high level.
    """
    sentences = [s for s in re.split(r"[.!?]+", transcript) if s.strip()]
    words = transcript.split()
    word_count = len(words)
    avg_sentence_len = word_count / max(len(sentences), 1)
    complexity_score = min(avg_sentence_len / 16, 1.0) * 100

    composite = (vocab_score * 0.30 + grammar_score * 0.25 + pronun_score * 0.20
                 + pace_score * 0.10 + complexity_score * 0.15)

    evidence_factor = min(word_count / 60, 1.0)      # full credit only from ~60+ words spoken
    composite = round(composite * (0.5 + 0.5 * evidence_factor), 1)

    if   composite < 35: level = "A1"
    elif composite < 50: level = "A2"
    elif composite < 65: level = "B1"
    elif composite < 80: level = "B2"
    elif composite < 92: level = "C1"
    else:                level = "C2"

    return {"score": composite, "level": level, "avg_sentence_length": round(avg_sentence_len, 1)}

def determine_voice_archetype(pace_s: float, filler_s: float, pronun_s: float,
                               grammar_s: float, clarity_s: float, vocab_s: float,
                               cefr_level: str) -> dict:
    """Rule-based, fun categorical read on speaking style — evaluated in priority
    order, first match wins, always ending in an encouraging fallback."""
    cefr_rank = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}.get(cefr_level, 3)
    band = [pace_s, filler_s, pronun_s, grammar_s]

    if pronun_s >= 82 and pace_s >= 80 and clarity_s >= 78:
        return {"archetype": "The Orator", "emoji": "🎙️",
                "description": "Commanding and confident — your pace and clarity make you easy "
                                "to follow and engaging to listen to.",
                "traits": ["Confident delivery", "Strong pronunciation", "Natural rhythm"]}
    if vocab_s >= 75 and clarity_s >= 65:
        return {"archetype": "The Storyteller", "emoji": "📖",
                "description": "Rich, descriptive language — you paint a picture with words and "
                                "hold attention through vocabulary, not just volume.",
                "traits": ["Wide vocabulary", "Descriptive", "Engaging"]}
    if grammar_s >= 85 and cefr_rank >= 4 and filler_s >= 75:
        return {"archetype": "The Analyst", "emoji": "🧠",
                "description": "Precise and structured — your grammar accuracy and controlled "
                                "delivery suggest a methodical, technical communication style.",
                "traits": ["Grammatically precise", "Minimal filler", "Structured"]}
    if pace_s >= 85 and pronun_s < 65:
        return {"archetype": "The Sprinter", "emoji": "⚡",
                "description": "Energetic and fast-paced — you communicate with urgency and "
                                "enthusiasm; slowing down slightly could sharpen clarity further.",
                "traits": ["High energy", "Fast pace", "Enthusiastic"]}
    if filler_s < 50:
        return {"archetype": "The Explorer", "emoji": "🧭",
                "description": "Free-flowing and exploratory — you think out loud as you speak, "
                                "which brings authenticity even as fillers creep in.",
                "traits": ["Spontaneous", "Authentic", "Room to tighten delivery"]}
    if all(s >= 60 for s in band) and max(band) - min(band) <= 20:
        return {"archetype": "The Diplomat", "emoji": "🤝",
                "description": "Measured and balanced — no single trait dominates; you "
                                "communicate steadily across every dimension.",
                "traits": ["Balanced", "Consistent", "Composed"]}
    if pace_s >= 70 and pronun_s >= 60 and clarity_s >= 55:
        return {"archetype": "The Steady Narrator", "emoji": "🌊",
                "description": "Calm and reliable — a natural, easy-to-follow speaking style "
                                "that listeners find comfortable.",
                "traits": ["Natural rhythm", "Reliable", "Easy to follow"]}
    return {"archetype": "The Rising Voice", "emoji": "🌱",
            "description": "You're building your speaking foundation — every recording "
                            "sharpens pace, clarity, and confidence. Keep practicing.",
            "traits": ["Developing fluency", "Growth mindset", "Practice-driven"]}


# score_pronunciation() / extract_mispronounced() moved to
# pronunciation_provider.py (as the internals of WhisperConfidenceProvider)
# as part of the pronunciation-provider migration — logic unchanged, just
# relocated so it's one provider among several behind a common interface.
# See pronunciation_provider.py and DEFAULT_PRONUNCIATION_PROVIDER below.
#
# DEFAULT changed from "whisper_confidence" to "allosaurus_g2p": pronunciation
# assessment must not depend on Whisper (Sarvam Saaras v3 is the primary STT
# and the single source of truth for transcription — see
# pronunciation_provider.py's module docstring for the full rationale).
# "allosaurus_g2p" assesses audio against the expected pronunciation of the
# transcript's words (G2P + Allosaurus acoustic evidence, fully offline/CPU)
# instead of treating any ASR's own confidence as a pronunciation signal.
# WhisperConfidenceProvider still exists and can still be selected explicitly
# (pronunciation_provider="whisper_confidence") but is no longer the default
# or the fallback used when another explicitly-requested provider is
# unavailable — see resolve_pronunciation()'s fallback in score_free_speech()
# below, which now falls back to this same Whisper-free default.

DEFAULT_PRONUNCIATION_PROVIDER = "allosaurus_g2p"


def validate_pronunciation_provider(name: str) -> str:
    """Shared validation for the `pronunciation_provider` request field,
    used by every route that accepts it. Rejects unknown values with a 422
    (FastAPI's standard validation-error status) rather than silently
    defaulting — an invalid provider name is a client error, not something
    to paper over."""
    if name not in PRONUNCIATION_PROVIDER_NAMES:
        raise HTTPException(
            422,
            f"Invalid pronunciation_provider '{name}'. Must be one of: "
            f"{', '.join(PRONUNCIATION_PROVIDER_NAMES)}.",
        )
    return name


def extract_grammar_issues(text: str, errors: list) -> list:
    """
    Turn language_tool matches into {wrong, correct, message, context} dicts.
    Uses snake_case attribute names as per language_tool_python API.
    """
    issues = []
    for m in errors:
        if not m.replacements:
            continue
        wrong   = m.matched_text          # the exact incorrect substring
        correct = m.replacements[0]
        if not wrong.strip() or wrong.lower() == correct.lower():
            continue
        issues.append({
            "wrong":   wrong,
            "correct": correct,
            "message": m.message,
            "context": m.context,         # sentence context string
        })
    return issues[:8]

def resolve_grammar(transcript: str, lt_grammar: dict, linguistic_analysis: dict = None) -> tuple:
    """
    Shared grammar-resolution path used by score_free_speech() and
    debug_analyze_text() (kept as one function so both stay in sync instead
    of re-implementing the same branching twice).

    Picks the base grammar signal exactly as before — prefer the HTTP
    LanguageTool server, fall back to local language_tool_python, fall back
    to the naive regex scan — then layers grammar_heuristics.py's learner-
    focused pattern detectors on top via augment_grammar_issues().

    That layer exists because LanguageTool (in any of the three paths above)
    can genuinely report 0 errors on a sentence that's obviously malformed
    for an English learner (wrong/missing "be" auxiliary, subject-verb
    agreement, missing past tense, "don't"/"doesn't" mismatch — see
    grammar_heuristics.py's module docstring). It never replaces
    LanguageTool's own findings, only adds real *additional* mistakes it
    would otherwise miss, and never double-counts one LanguageTool already
    caught (see augment_grammar_issues()'s de-duplication).

    `linguistic_analysis` (LanguageTool's /v2/analyze token/POS/lemma data,
    or None if that call failed/was unavailable — see
    LanguageToolProvider.check_and_analyze) is passed straight through to
    augment_grammar_issues(), which uses it to additionally run
    grammar_pos_rules.py's noun/indefinite-pronoun-subject and lexicon-based
    rules. Purely additive: None here means exactly the same regex-only
    behavior as before this parameter existed.

    ── Contextual validation (Groq) ────────────────────────────────────
    Everything above only produces CANDIDATE issues — tools built for
    proofreading written text, which regularly flag things that are not
    actually spoken-grammar mistakes (e.g. "forty one" vs "forty-one" is a
    hyphenation convention; "gonna" is a normal informal spoken form, not
    a grammar error). Before those candidates are allowed to affect the
    grammar score or the learner-facing issue list, they're passed through
    grammar_context_validator.apply_contextual_validation(), which uses Groq
    to read each candidate in its full sentence context and classify it as
    true_grammar_error / spoken_usage_issue / written_only_issue /
    style_or_register / not_an_error (falling back to a small offline
    heuristic classifier if Groq is unavailable — see that module's
    docstring). ONLY candidates judged `true_grammar_error` remain in
    `grammar_issues` / count toward `ge` after this point; everything else
    is preserved separately as non-scoring "context notes" (see
    `grammar_context` in the returned tuple) rather than discarded, so
    nothing here silently invents or hides evidence — it only decides what
    is allowed to count as a spoken grammar mistake.

    Returns (ge, grammar_issues, grammar_source, heuristic_added_count,
    grammar_context). `ge`/`grammar_issues`/`grammar_source` are exactly
    the fields the frontend already consumes (unchanged shape, now
    validated); `heuristic_added_count` is additive, for callers that want
    to report it; `grammar_context` is a new, additive debug/notes dict
    (see build below) — callers that ignore it get unchanged behavior
    other than the (intentional) score/issue-list correction itself.
    """
    if lt_grammar is not None:
        ge = lt_grammar["errors"]
        grammar_issues = lt_grammar["issues"]
        grammar_source = "languagetool_http"
    elif GRAMMAR_OK:
        errors = _grammar.check(transcript)
        ge = len(errors)
        grammar_issues = extract_grammar_issues(transcript, errors)
        grammar_source = "language_tool_python_local"
    else:
        ge = len(re.findall(r"\b(\w+)\s+\1\b", transcript.lower()))
        grammar_issues = []
        grammar_source = "regex_fallback"

    ge, grammar_issues, heuristic_added = augment_grammar_issues(
        transcript, ge, grammar_issues, linguistic_analysis=linguistic_analysis)

    # See grammar_context_validator.apply_contextual_validation() for the
    # full rationale/mechanics. In short: `ge`/`grammar_issues` above are
    # still just CANDIDATES from tools built for written-text proofreading;
    # this call is what actually decides which candidates are allowed to
    # count as a spoken grammar mistake.
    ge, grammar_issues, grammar_context = apply_contextual_validation(
        transcript, ge, grammar_issues)

    return ge, grammar_issues, grammar_source, heuristic_added, grammar_context


def score_clarity(pace: float, filler: float, grammar: float, pronun: float) -> float:
    return round(pace * 0.25 + filler * 0.25 + grammar * 0.25 + pronun * 0.25, 1)


def score_fluency(pause_info: dict, filler_count: int, hesitation_count: int,
                   word_count: int) -> dict:
    """Real fluency score, distinct from Pace and Clarity.

    Pace only measures average words/minute — a speaker can hit a perfect
    120-150 WPM average while still stopping dead for three seconds every
    sentence (the average just doesn't show it). Clarity is a pure
    re-blend of pace/filler/grammar/pronunciation and adds no new
    evidence. Fluency instead measures *continuity of delivery*: how often
    and how long the speaker actually pauses (from real word-timestamp
    gaps — see audio_utils.analyze_pauses), plus the rate of filler words
    and word-repetition hesitations per word spoken. This is informational,
    like Vocabulary/CEFR — it is not folded into `overall` or `clarity`, so
    none of the existing weighting/architecture changes.

    Informational only when pause data isn't available (e.g. an STT
    provider without word-level timestamps) — falls back to the
    filler/hesitation signal alone and says so via `pause_data_available`.
    """
    disfluency_rate = (filler_count + hesitation_count) / max(word_count, 1)
    disfluency_penalty = min(disfluency_rate * 250, 45)

    if pause_info.get("available"):
        # Long (>=1.2s) hesitation-style pauses cost far more than the
        # ordinary short gaps that occur naturally between phrases.
        pause_penalty = min(
            pause_info["pause_rate_per_min"] * 1.5 + pause_info["long_pause_count"] * 7,
            45,
        )
    else:
        pause_penalty = 0.0

    score = round(max(5.0, min(100.0, 100 - disfluency_penalty - pause_penalty)), 1)
    return {
        "score": score,
        "pause_data_available": bool(pause_info.get("available")),
        "pause_count": pause_info.get("pause_count", 0),
        "long_pause_count": pause_info.get("long_pause_count", 0),
        "avg_pause_ms": pause_info.get("avg_pause_ms", 0.0),
        "pause_rate_per_min": pause_info.get("pause_rate_per_min", 0.0),
        "hesitation_count": hesitation_count,
    }

def build_feedback(wpm, pace_s, fc, filler_words, ge, pronun_s, clarity_s, overall,
                    grammar_s: float = None) -> str:
    parts = []

    if wpm < 100:
        parts.append(f"Your pace ({wpm:.0f} WPM) is too slow — aim for 120–150 WPM for natural delivery.")
    elif wpm > 175:
        parts.append(f"You're speaking fast ({wpm:.0f} WPM) — slow down to 120–150 WPM so listeners can follow.")
    else:
        parts.append(f"Great pace at {wpm:.0f} WPM — right in the natural zone.")

    if fc >= 10:
        sample = ", ".join(filler_words[:3]) if filler_words else ""
        parts.append(f"You used {fc} filler words ({sample}). Practice pausing silently instead of filling gaps.")
    elif fc >= 4:
        parts.append(f"Reduce fillers — you used {fc}. A short pause sounds more confident than 'um' or 'uh'.")
    else:
        parts.append("Excellent control of filler words!")

    # Grammar feedback is thresholded on the *score* (score_grammar()'s
    # rate-based bands), not the raw error count `ge`. The two used to be
    # judged independently: `ge` is a raw count with no notion of transcript
    # length, while `grammar_s` already folds in `errors / words`. A single
    # error in a 13-word transcript scores 56 (the `rate < 0.08` band) — bad
    # enough that score_grammar() itself doesn't call it "clean" — but the
    # old `ge >= 2` check let that same single error fall into the `else`
    # branch and print "Grammar looks clean!", contradicting the score shown
    # elsewhere in the same response. Reusing score_grammar()'s own bands
    # here keeps this message and the score permanently in agreement.
    #
    # `grammar_s` is optional (default None) only so any external caller
    # still invoking build_feedback() with the old 8-argument signature
    # doesn't immediately break — it falls back to the previous count-based
    # thresholds in that case. Every in-repo call site now passes grammar_s.
    if grammar_s is not None:
        if grammar_s < 56:
            parts.append(f"{ge} grammar issue{'s' if ge != 1 else ''} detected — review subject-verb agreement and sentence structure.")
        elif grammar_s < 90:
            parts.append(f"Minor grammar issues ({ge}) — keep practicing for cleaner delivery.")
        else:
            parts.append("Grammar looks clean!")
    else:
        if ge >= 6:
            parts.append(f"{ge} grammar issues detected — review subject-verb agreement and sentence structure.")
        elif ge >= 2:
            parts.append(f"Minor grammar issues ({ge}) — keep practicing for cleaner delivery.")
        else:
            parts.append("Grammar looks clean!")

    if pronun_s >= 85:
        parts.append("Pronunciation is very clear throughout.")
    elif pronun_s >= 65:
        parts.append("Good pronunciation — focus on enunciating consonants at the ends of words.")
    else:
        parts.append("Work on clearer enunciation — speak each word fully before moving to the next.")

    return "  ".join(parts)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse("/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    return (BASE_DIR / "login.html").read_text()

@app.get("/login.css")
async def login_css():
    return FileResponse(str(BASE_DIR / "login.css"), media_type="text/css")

@app.get("/login.js")
async def login_js():
    return FileResponse(str(BASE_DIR / "login.js"), media_type="application/javascript")

@app.get("/auth-check.js")
async def auth_check_js():
    return FileResponse(str(BASE_DIR / "auth-check.js"), media_type="application/javascript")

@app.get("/app", response_class=HTMLResponse)
async def app_page():
    return (BASE_DIR / "index.html").read_text()

@app.get("/style.css")
async def css():
    return FileResponse(str(BASE_DIR / "style.css"), media_type="text/css")

@app.get("/script.js")
async def js():
    return FileResponse(str(BASE_DIR / "script.js"), media_type="application/javascript")

@app.get("/assessment.js")
async def assessment_js():
    return FileResponse(str(BASE_DIR / "assessment.js"), media_type="application/javascript")


def save_and_convert(raw_bytes: bytes, ts: str, prefix: str = "rec") -> Path:
    """Write the uploaded browser recording to disk and convert it to a
    16-kHz mono WAV for Whisper. Returns the WAV path."""
    raw_path = RECORDINGS_DIR / f"{prefix}_{ts}.webm"
    wav_path = RECORDINGS_DIR / f"{prefix}_{ts}.wav"
    raw_path.write_bytes(raw_bytes)

    conv = subprocess.run(
        ["ffmpeg", "-y", "-i", str(raw_path),
         "-ar", "16000", "-ac", "1", str(wav_path)],
        capture_output=True
    )
    if conv.returncode != 0:
        raise HTTPException(500, f"Audio conversion failed: {conv.stderr.decode()}")
    return wav_path

def transcribe_wav(wav_path: Path):
    """STT – word_timestamps=True gives per-word probability for pronunciation analysis.

    Thin wrapper over WhisperSTTProvider (see stt_provider.py) — kept for any
    external caller that still imports transcribe_wav() directly. /assess,
    /assess/stage, and /debug/analyze-audio call resolve_stt()/stt_registry
    instead, which is what actually supports switching to Saaras."""
    result = stt_registry.get("whisper").transcribe(wav_path)
    if not result.available:
        raise HTTPException(400, result.detail or "Could not transcribe — please speak clearly and try again.")
    return result.transcript, result.segments

def score_free_speech(transcript: str, segments: list, duration: float, wav_path: Path = None,
                       pronunciation_provider: str = DEFAULT_PRONUNCIATION_PROVIDER) -> dict:
    """The full five-metric scoring pipeline (pace/filler/grammar/pronunciation/
    clarity/overall) shared by the free-recording tab and the picture-description
    and final stages of the guided English Assessment.

    `duration` must be the actual audio duration (seconds), not a
    client-reported value — see audio_utils.wav_duration_seconds() and its
    call sites below. It drives wpm/pace and (via score_cefr) the CEFR
    pace component.

    `wav_path` is passed through to the pronunciation provider (audio-based
    providers like Saaras need the actual recording, not just Whisper's
    segments). `pronunciation_provider` selects which PronunciationProvider
    (see pronunciation_provider.py) is used; defaults to the pre-existing
    Whisper-confidence behavior. This function never branches on the
    provider name itself — see resolve_pronunciation()."""
    word_count = len(transcript.split())

    # Single LanguageTool call up front — both grammar checking (`lt_grammar`)
    # and context-aware filler detection (`linguistic_analysis`, POS/lemma
    # data) need it, and it's a network round trip, so this function must not
    # call it twice. Behavior/timeout/fallback is unchanged from before —
    # just moved earlier so filler detection can reuse the same result.
    #
    # `lt_errors` (why /v2/check and/or /v2/analyze failed, if either did)
    # used to be discarded here (bound to `_lt_errors` and dropped) even
    # though debug_analyze_text() below captures and returns the identical
    # tuple's third element as `languagetool_errors`. That asymmetry is why
    # /assess, /assess/stage, and /debug/analyze-audio could never explain
    # *why* linguistic_analysis was null — see "languagetool_errors" in the
    # returned dict below, now populated the same way for every caller.
    lt_grammar, linguistic_analysis, lt_errors = _lt_provider.check_and_analyze(transcript)

    # Pace
    wpm      = (word_count / max(duration, 1)) * 60
    pace_s   = score_pace(wpm)

    # ── Fillers — context-aware detection, spoken vs. acoustic kept separate ──
    # Replaces the old flat regex/word-list scan over FILLERS. Reuses the
    # existing, separately-tested module as-is (tests/test_filler_detector.py)
    # — no filler-classification logic duplicated here. score_fillers()'s
    # formula/thresholds are unchanged; only what gets counted changed.
    # Repetitions/hesitations ("I I", "the the") are a distinct disfluency
    # signal and are kept out of the filler count — see `hesitations` below.
    #
    # The wav_path is passed through so detect_fillers() can also run its
    # RMS-energy acoustic hesitation detector (catches genuine low-energy/
    # silence gaps that Whisper's transcript wouldn't show). That signal is
    # int­entionally kept OUT of `fillers["count"]` — see filler_detector.py's
    # module docstring. `filler_count` below must stay equal to
    # `fillers["count"]` (spoken evidence only): it is what feeds
    # score_fillers() -> filler_s -> clarity_s -> overall, so an acoustic
    # silence must never inflate it the way "spoken filler usage" should.
    fillers = detect_fillers(
        transcript,
        linguistic_analysis,
        duration_seconds=duration,
        audio_path=wav_path
    )
    filler_count = fillers["count"]  # SPOKEN fillers only — see filler_detector.py
    acoustic_hesitation_count = fillers["acoustic_hesitation_count"]
    found = summarize_words(fillers["occurrences"])
    filler_s = score_fillers(filler_count, word_count)

    # Grammar — prefer the HTTP LanguageTool server (stock /v2/check gives
    # richer, rule-based matches with category/rule-id/offsets than the old
    # local-Java path could expose). /v2/analyze runs alongside it for its
    # token/POS/lemma data; either can fail independently (see
    # LanguageToolProvider.check_and_analyze). If the HTTP server is
    # unavailable entirely, fall straight back to the existing local
    # language_tool_python / naive-regex behavior — unchanged from before.
    ge, grammar_issues, grammar_source, heuristic_added, grammar_context = resolve_grammar(
        transcript, lt_grammar, linguistic_analysis)
    grammar_s = score_grammar(ge, word_count)

    # Pronunciation & Clarity — resolved through the provider registry (see
    # pronunciation_provider.py). If the requested provider couldn't produce
    # a real assessment (not configured / not implemented — gop, local_llm,
    # or an incomplete saaras integration), fall back to the default
    # pronunciation provider (allosaurus_g2p — Whisper-free, see
    # pronunciation_provider.py) for the *score itself* (so clarity/overall/
    # cefr/archetype downstream still get a real number to blend), but keep
    # the honest available/detail info in the response's
    # pronunciation.provider_status so the UI never claims a provider ran
    # when it didn't — see requirement "UI should not say 'Saaras' if the
    # backend actually used another provider."
    pronun_result = resolve_pronunciation(pronunciation_provider, transcript, segments, wav_path)
    if pronun_result.available:
        pronun_s = pronun_result.score
        pronun_issues = pronun_result.issues
        pronunciation_provider_used = pronun_result.provider
        pronunciation_provider_detail = None
        pronunciation_provider_methodology = pronun_result.methodology
    else:
        # If the caller already requested the default provider and it's the
        # one that just failed, re-resolving it again would just repeat the
        # exact same (already-known) failure for no benefit — reuse
        # pronun_result instead of calling it a second time.
        fallback = (pronun_result if pronunciation_provider == DEFAULT_PRONUNCIATION_PROVIDER
                    else resolve_pronunciation(DEFAULT_PRONUNCIATION_PROVIDER, transcript, segments, wav_path))
        pronun_s = fallback.score
        pronun_issues = fallback.issues
        pronunciation_provider_used = fallback.provider
        pronunciation_provider_detail = pronun_result.detail
        pronunciation_provider_methodology = fallback.methodology
    clarity_s = score_clarity(pace_s, filler_s, grammar_s, pronun_s)
    overall   = round(pace_s * 0.20 + filler_s * 0.20 + pronun_s * 0.25 +
                      grammar_s * 0.20 + clarity_s * 0.15, 1)

    # Vocabulary, CEFR, Fluency, and Voice Archetype are additional,
    # informational metrics — they are not folded into `overall`, which
    # keeps its original five-metric meaning.
    vocab = score_vocabulary(transcript)
    cefr  = score_cefr(vocab["score"], grammar_s, pronun_s, pace_s, transcript)

    # Fluency — see score_fluency()'s docstring for why this is a distinct
    # signal from Pace/Clarity. Built from real word-timestamp pause gaps
    # (audio_utils.analyze_pauses) plus the filler/hesitation counts already
    # computed above; no new model call, no new audio pass.
    pause_info = analyze_pauses(segments, duration)
    fluency = score_fluency(pause_info, filler_count, len(fillers["hesitations"]), word_count)
    # Acoustic hesitation count is informational only here — it is exposed
    # alongside the fluency signal (real word-timestamp pause gaps +
    # spoken filler/repetition rate) rather than folded into
    # score_fluency()'s formula. The brief is explicit: don't redesign the
    # scoring system, and don't double-penalize the same hesitation once as
    # a filler and again as a fluency deduction unless that's already the
    # existing, intended behavior. score_fluency()'s formula is unchanged.
    fluency["acoustic_hesitation_count"] = acoustic_hesitation_count

    archetype = determine_voice_archetype(pace_s, filler_s, pronun_s, grammar_s,
                                           clarity_s, vocab["score"], cefr["level"])

    feedback = build_feedback(wpm, pace_s, filler_count, found, ge,
                               pronun_s, clarity_s, overall, grammar_s)

    # Evidence/reliability flag — Vocabulary and CEFR already dampen their
    # own composites for short samples (evidence_factor), but the five core
    # metrics (pace, filler, grammar, pronunciation, clarity) and Overall
    # did not carry any equivalent signal: a 4-word, 3-second recording with
    # zero grammar errors and no fillers can mathematically produce a
    # near-perfect Overall score with almost no evidence behind it (too few
    # words for WPM to mean anything, too few words for LanguageTool to
    # have anything to flag, too short for Whisper's avg_logprob to be
    # stable). This doesn't change any score — it only tells the caller
    # (API/UI) when the numbers above should be shown with a caveat rather
    # than presented as equally trustworthy as a 60+ word response.
    low_evidence = word_count < 25 or duration < 12
    evidence = {
        "word_count": word_count,
        "duration_seconds": round(duration, 1),
        "low_evidence": low_evidence,
        "reason": (
            f"Only {word_count} word(s) over {duration:.1f}s — scores are "
            "unreliable at this length; speak for at least ~12s and 25+ words "
            "for a trustworthy assessment." if low_evidence else None
        ),
    }
    if low_evidence:
        feedback += "  Note: this response is quite short, so these scores are a rough " \
                    "read rather than a reliable assessment — try a longer response " \
                    "for a more accurate result."

    # NEW: Insert filler markers into transcript
    transcript_with_fillers = insert_filler_markers(transcript, fillers.get("occurrences", []))

    return {
        "transcript": transcript,
        "transcript_with_fillers": transcript_with_fillers,  # NEW: transcript with [filler] markers
        "pace":         {"score": round(pace_s, 1),   "wpm": round(wpm, 1)},
        "filler":       {
            "score": round(filler_s, 1), "count": filler_count, "words": found,
            "occurrences": fillers["occurrences"],
            "rate_per_min": fillers["rate_per_min"],
            # SPOKEN filler evidence — count/score/occurrences above are
            # derived from this and this alone.
            "spoken_count": filler_count,
            "spoken_fillers": found,
            # ACOUSTIC hesitation evidence — RMS-energy low-energy/silence
            # segments. Kept fully separate: informational disfluency
            # signal only, never folded into score/count/occurrences above.
            "acoustic_hesitations": fillers.get("acoustic_hesitations", []),
            "acoustic_hesitation_count": acoustic_hesitation_count,
        },
        "pronunciation":{
            "score": round(pronun_s, 1), "issues": pronun_issues,
            "provider": pronunciation_provider_used,
            "requested_provider": pronunciation_provider,
            "available": pronun_result.available,
            "detail": pronunciation_provider_detail,
            "methodology": pronunciation_provider_methodology,
        },
        "grammar":      {"score": round(grammar_s, 1), "errors": ge, "issues": grammar_issues},
        "grammar_source": grammar_source,
        "grammar_heuristic_issues_added": heuristic_added,
        # Contextual-validation (Groq) evidence — see grammar_context_validator.py.
        # `grammar.errors`/`grammar.issues` above already reflect this validation
        # (only true_grammar_error candidates remain); this field is additive and
        # explains *why*: which candidates were reclassified away, the non-scoring
        # written-English/style/usage notes, and the full candidate+judgment trail.
        "grammar_context": grammar_context,
        "clarity":      {"score": round(clarity_s, 1)},
        "vocabulary":   vocab,
        "cefr":         cefr,
        "fluency":      fluency,
        "archetype":    archetype,
        "overall": overall,
        "feedback": feedback,
        "hesitations": fillers["hesitations"],
        "linguistic_analysis": linguistic_analysis,
        "languagetool_errors": lt_errors,
        "evidence": evidence,
    }


@app.post("/assess")
async def assess(request: Request, audio: UploadFile = File(...), duration: float = Form(0),
                  pronunciation_provider: str = Form(DEFAULT_PRONUNCIATION_PROVIDER),
                  stt_provider: str = Form(DEFAULT_STT_PROVIDER)):
    user = await require_user(request)
    pronunciation_provider = validate_pronunciation_provider(pronunciation_provider)
    stt_provider = validate_stt_provider(stt_provider)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = save_and_convert(await audio.read(), ts)
    # Authoritative duration, computed from the actual converted audio —
    # NOT the client-sent `duration` form field, which is an unreliable
    # browser wall-clock value (see audio_utils.wav_duration_seconds()).
    # This is what feeds wpm/pace/CEFR below and is what gets stored/
    # returned as "duration" — the same field name/position as before,
    # now populated with the accurate value that field was always meant
    # to hold. The raw client value is still received (kept as `duration`
    # the parameter) but only used as a fallback if the WAV is somehow
    # unreadable.
    audio_duration = wav_duration_seconds(wav_path) or duration
    stt_requested, stt_used = resolve_stt(stt_provider, wav_path)
    transcript, segments = stt_used.transcript, stt_used.segments
    r = score_free_speech(transcript, segments, audio_duration, wav_path,
                           pronunciation_provider=pronunciation_provider)
    r["stt"] = {
        "provider": stt_used.provider,
        "requested_provider": stt_provider,
        "available": stt_requested.available,
        "detail": None if stt_requested.available else stt_requested.detail,
    }

    # Persist. Vocabulary Coverage / CEFR Level aren't shown on the immediate
    # Quick Assessment results screen, but History and Progress do show them,
    # so they're still computed and stored here. The full per-parameter
    # result dict (issues, occurrences, linguistic analysis, etc.) is stored
    # too — see repository.save_assessment() — so it can be retrieved later
    # via GET /assessments/{id} or GET /assessments/{id}/parameters.
    aid = db.save_assessment(timestamp=ts, duration=audio_duration, result=r, user_id=user["id"])

    return {"id": aid, "duration": audio_duration, "duration_client_sent": duration, **r}


@app.get("/history")
async def history(request: Request):
    user = await require_user(request)
    return db.list_assessments(user["id"])


@app.get("/assessments/{assessment_id}")
async def get_assessment(assessment_id: int, request: Request):
    """One previous Quick Assessment by id, with its full per-parameter
    result (issues, occurrences, linguistic analysis, etc.) included under
    `full_result`. Scoped to the logged-in user, same as /history."""
    user = await require_user(request)
    result = db.get_assessment(assessment_id, user["id"])
    if result is None:
        raise HTTPException(404, f"No assessment found with id {assessment_id}")
    return result


@app.get("/assessments/{assessment_id}/parameters")
async def get_assessment_parameters(assessment_id: int, request: Request):
    """Just the individual parameter results (pace, filler, pronunciation,
    grammar, clarity, vocabulary, cefr, archetype, plus disfluency/
    linguistic evidence) for one previous Quick Assessment."""
    user = await require_user(request)
    result = db.get_assessment_parameters(assessment_id, user["id"])
    if result is None:
        raise HTTPException(404, f"No assessment found with id {assessment_id}")
    return result


def find_manifest_item(stage_type: str, stage_id: str) -> dict:
    for item in ASSESSMENT_MANIFEST.get(stage_type, []):
        if item["id"] == stage_id:
            return item
    raise HTTPException(404, f"Unknown stage: {stage_type}/{stage_id}")


@app.get("/assessment/manifest")
async def assessment_manifest():
    return public_manifest()


@app.get("/pronunciation-providers")
async def pronunciation_providers():
    """Which pronunciation providers exist and whether each is currently
    usable — so the UI can grey out / mark "Coming Soon" options the
    backend actually confirms are unavailable, rather than trusting
    frontend-only state (per requirement: 'The UI can show it as available
    only if the backend confirms it is configured')."""
    return {
        "default": DEFAULT_PRONUNCIATION_PROVIDER,
        "providers": pronunciation_registry.status(),
    }


@app.get("/stt-providers")
async def stt_providers():
    """Same idea as /pronunciation-providers, one layer earlier in the
    pipeline: which speech-to-text engines exist and are currently usable."""
    return {
        "default": DEFAULT_STT_PROVIDER,
        "providers": stt_registry.status(),
    }


@app.post("/assess/stage")
async def assess_stage(
    request: Request,
    stage_type: str = Form(...),
    stage_id: str = Form(""),
    audio: UploadFile = File(...),
    duration: float = Form(0),
    pronunciation_provider: str = Form(DEFAULT_PRONUNCIATION_PROVIDER),
    stt_provider: str = Form(DEFAULT_STT_PROVIDER),
):
    await require_user(request)
    if stage_type not in ("name", "picture_talk", "media_repeat", "picture_describe", "final"):
        raise HTTPException(400, f"Unknown stage_type: {stage_type}")
    pronunciation_provider = validate_pronunciation_provider(pronunciation_provider)
    stt_provider = validate_stt_provider(stt_provider)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = save_and_convert(await audio.read(), ts, prefix=f"stage_{stage_type}_{stage_id or 'x'}")
    audio_duration = wav_duration_seconds(wav_path) or duration
    stt_requested, stt_used = resolve_stt(stt_provider, wav_path)
    transcript, segments = stt_used.transcript, stt_used.segments
    stt_info = {
        "provider": stt_used.provider,
        "requested_provider": stt_provider,
        "available": stt_requested.available,
        "detail": None if stt_requested.available else stt_requested.detail,
    }

    if stage_type == "name":
        return {"stage_type": stage_type, "stage_id": stage_id,
                "transcript": transcript, "name": extract_name(transcript), "stt": stt_info}

    if stage_type in ("picture_talk", "media_repeat"):
        item = find_manifest_item(stage_type, stage_id)
        reference = item["sentence"]
        accuracy_s = word_accuracy(reference, transcript)
        pronun_result = resolve_pronunciation(pronunciation_provider, transcript, segments, wav_path)
        if pronun_result.available:
            pronun_s = pronun_result.score
            pronun_provider_used = pronun_result.provider
            pronun_detail = None
        else:
            fallback = resolve_pronunciation(DEFAULT_PRONUNCIATION_PROVIDER, transcript, segments, wav_path)
            pronun_s = fallback.score
            pronun_provider_used = fallback.provider
            pronun_detail = pronun_result.detail
        stage_score = round(accuracy_s * 0.6 + pronun_s * 0.4, 1)
        return {
            "stage_type": stage_type, "stage_id": stage_id,
            "transcript": transcript, "reference": reference,
            "accuracy_score": accuracy_s, "pronunciation_score": pronun_s,
            "pronunciation_provider": pronun_provider_used,
            "pronunciation_requested_provider": pronunciation_provider,
            "pronunciation_provider_detail": pronun_detail,
            "score": stage_score,
            "stt": stt_info,
        }

    # picture_describe and final both use the full free-speech pipeline.
    # Uses the authoritative WAV-derived duration, same as /assess.
    r = score_free_speech(transcript, segments, audio_duration, wav_path,
                           pronunciation_provider=pronunciation_provider)
    r["stt"] = stt_info
    return {"stage_type": stage_type, "stage_id": stage_id, "score": r["overall"],
            "duration": audio_duration, "duration_client_sent": duration, **r}


def aggregate_vocabulary(stages: list) -> dict:
    """Vocabulary Coverage computed from the combined transcripts of every
    free-speech stage (Describe & Compare + Full Assessment) — a larger, more
    representative sample of spontaneous speech than any single recording."""
    texts = [s["transcript"] for s in stages
             if s.get("stage_type") in ("picture_describe", "final") and s.get("transcript")]
    return score_vocabulary(" ".join(texts))

def aggregate_cefr(stages: list, vocab_score: float) -> dict:
    """CEFR estimate blended from all four scoreable stage types: repeat-accuracy
    and pronunciation from Picture Talk + Listen & Repeat, and vocabulary/grammar/
    pace/sentence-complexity from Describe & Compare + Full Assessment. ('State your
    name' has no assessable linguistic content and contributes no signal.)"""
    def avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    repeat_stages = [s for s in stages if s.get("stage_type") in ("picture_talk", "media_repeat")]
    free_stages   = [s for s in stages if s.get("stage_type") in ("picture_describe", "final")]

    repeat_accuracy = avg([s.get("accuracy_score") for s in repeat_stages])
    repeat_pronun   = avg([s.get("pronunciation_score") for s in repeat_stages])
    free_grammar    = avg([s["grammar"]["score"] for s in free_stages])
    free_pronun     = avg([s["pronunciation"]["score"] for s in free_stages])
    free_pace       = avg([s["pace"]["score"] for s in free_stages])
    pronun_score    = avg([v for v in (repeat_pronun, free_pronun) if v])

    combined_text = " ".join(s["transcript"] for s in free_stages if s.get("transcript"))
    sentences = [s for s in re.split(r"[.!?]+", combined_text) if s.strip()]
    words = combined_text.split()
    word_count = len(words)
    avg_sentence_len = word_count / max(len(sentences), 1)
    complexity_score = min(avg_sentence_len / 16, 1.0) * 100

    composite = (
        vocab_score * 0.25 + free_grammar * 0.20 + pronun_score * 0.20 +
        free_pace * 0.10 + complexity_score * 0.10 + repeat_accuracy * 0.15)

    # Same short-sample dampening as score_cefr() — a handful of words across all
    # stages combined still isn't enough evidence for a high CEFR estimate.
    evidence_factor = min(word_count / 60, 1.0)
    composite = round(composite * (0.5 + 0.5 * evidence_factor), 1)

    if   composite < 35: level = "A1"
    elif composite < 50: level = "A2"
    elif composite < 65: level = "B1"
    elif composite < 80: level = "B2"
    elif composite < 92: level = "C1"
    else:                level = "C2"

    return {"score": composite, "level": level, "avg_sentence_length": round(avg_sentence_len, 1)}


@app.post("/assessment/finalize")
async def assessment_finalize(request: Request, payload: dict):
    user = await require_user(request)
    name   = payload.get("name", "")
    stages = payload.get("stages", [])

    def avg(stage_type):
        scores = [s["score"] for s in stages if s.get("stage_type") == stage_type and s.get("score") is not None]
        return round(sum(scores) / len(scores), 1) if scores else None

    picture_talk_score     = avg("picture_talk")
    media_repeat_score     = avg("media_repeat")
    picture_describe_score = avg("picture_describe")

    final_stage = next((s for s in stages if s.get("stage_type") == "final"), None)
    if not final_stage:
        raise HTTPException(400, "Missing final assessment stage")

    overall_score = final_stage["overall"]

    # Vocabulary Coverage and CEFR are computed assessment-wide (see §7/§8 of DESIGN.md),
    # not just from the Full Assessment recording alone.
    vocab_agg = aggregate_vocabulary(stages)
    cefr_agg  = aggregate_cefr(stages, vocab_agg["score"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Groq teacher report — same existing mechanism already used by
    # /debug/analyze-audio (see groq_provider.generate_teacher_report()),
    # now also wired into the production guided-assessment path per its own
    # docstring ("the `final` stage of a guided assessment"). Additive,
    # second step on top of the deterministic evidence above (final_stage);
    # never touches any score/evidence. If GROQ_API_KEY is unset or the
    # request/parse fails, teacher_report is just null and the rest of the
    # response/report is unaffected — no duplicate frontend Groq call exists
    # anywhere in this codebase.
    try:
        teacher_report_result = generate_teacher_report(
            final_stage.get("transcript", ""),
            final_stage.get("transcript_with_fillers"),
            final_stage,
            vocab_agg=vocab_agg,
            cefr_agg=cefr_agg,
            name=name,
        )
        teacher_report = teacher_report_result.report if teacher_report_result.available else None
        teacher_report_detail = teacher_report_result.detail
    except Exception as e:
        teacher_report = None
        teacher_report_detail = f"Teacher report generation raised an unexpected error: {e}"

    aid = db.save_english_assessment(
        timestamp=ts, name=name,
        picture_talk_score=picture_talk_score, media_repeat_score=media_repeat_score,
        picture_describe_score=picture_describe_score, overall_score=overall_score,
        final_stage=final_stage, vocab_score=vocab_agg["score"], cefr_score=cefr_agg["score"],
        cefr_level=cefr_agg["level"], archetype=final_stage["archetype"]["archetype"],
        stages=stages, user_id=user["id"],
        teacher_report=teacher_report, teacher_report_detail=teacher_report_detail)

    return {
        "id": aid, "timestamp": ts, "name": name,
        "overall_score": overall_score,
        "vocabulary": vocab_agg,
        "cefr": cefr_agg,
        "sections": {
            "picture_talk":     {"score": picture_talk_score,
                                  "items": [s for s in stages if s.get("stage_type") == "picture_talk"]},
            "media_repeat":     {"score": media_repeat_score,
                                  "items": [s for s in stages if s.get("stage_type") == "media_repeat"]},
            "picture_describe": {"score": picture_describe_score,
                                  "items": [s for s in stages if s.get("stage_type") == "picture_describe"]},
        },
        "final": final_stage,
        "teacher_report": teacher_report,
        "teacher_report_detail": teacher_report_detail,
    }


@app.get("/assessment/history")
async def assessment_history(request: Request):
    user = await require_user(request)
    return db.list_english_assessments(user["id"])


@app.get("/assessment/{assessment_id}")
async def get_english_assessment(assessment_id: int, request: Request):
    """One previous guided assessment by id, with the full per-stage detail
    (transcripts, scores, issues) included under `stages`. Scoped to the
    logged-in user, same as /assessment/history."""
    user = await require_user(request)
    result = db.get_english_assessment(assessment_id, user["id"])
    if result is None:
        raise HTTPException(404, f"No assessment found with id {assessment_id}")
    return result


@app.get("/assessment/{assessment_id}/parameters")
async def get_english_assessment_parameters(assessment_id: int, request: Request):
    """Just the individual parameter/section results for one previous guided
    assessment: section scores, overall parameter scores, and each stage's
    own breakdown."""
    user = await require_user(request)
    result = db.get_english_assessment_parameters(assessment_id, user["id"])
    if result is None:
        raise HTTPException(404, f"No assessment found with id {assessment_id}")
    return result


# ── TUTOR VIEW ENDPOINTS (standalone, no PocketBase) ──────────────────────────

@app.get("/tutor")
async def tutor_page():
    """Tutor view page - shows all debug assessments in a clean table."""
    tutor_html_path = BASE_DIR / "tutor.html"
    if not tutor_html_path.exists():
        raise HTTPException(404, "tutor.html not found. Please create it in the project root.")
    return FileResponse(str(tutor_html_path), media_type="text/html")


@app.get("/tutor/debug-data")
async def tutor_debug_data(password: str = ""):
    """
    API endpoint for the tutor view to fetch saved debug assessments.
    Protected by a simple password (default: tutor123).
    """
    if password != TUTOR_VIEW_PASSWORD:
        raise HTTPException(401, "Unauthorized - incorrect password")
    
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute("""
        SELECT id, timestamp, duration, transcript,
               overall_score, pace_score, filler_score,
               pronunciation_score, grammar_score, clarity_score,
               vocabulary_score, cefr_score, cefr_level, archetype,
               pace_wpm, filler_count, filler_words, grammar_errors,
               feedback, full_result
        FROM assessments 
        WHERE user_id = 'debug_user'
        ORDER BY id DESC
    """).fetchall()
    conn.close()
    
    result = []
    for r in rows:
        row_dict = dict(r)
        # If full_result exists, use it for complete data
        if r['full_result']:
            try:
                full = json.loads(r['full_result'])
                # Merge but keep top-level fields for display
                for key in ['transcript', 'overall', 'pace', 'filler', 'pronunciation', 
                           'grammar', 'clarity', 'vocabulary', 'cefr', 'archetype']:
                    if key in full:
                        row_dict[key] = full[key]
            except:
                pass
        result.append(row_dict)
    
    return result


# ── DEBUG-ONLY ROUTE (added for local backend testing, not used by production) ─
# Isolated addition — does not modify any existing route, function, or response
# contract above. Reuses score_vocabulary() (imported from vocabulary.py exactly
# as app.py already does), score_grammar(), extract_grammar_issues(), GRAMMAR_OK,
# and _grammar exactly as they already exist. Intentionally skips require_user()
# so it can be called from the standalone debug-ui/ page without a PocketBase
# login (per explicit instruction — do not expose this in production).
#
# Text-only input has no audio/duration, so pace, pronunciation, clarity, overall,
# and CEFR (which all depend on Whisper segments or recording duration) are not
# computed here — only the two metrics that are pure functions of transcript text.
@app.post("/debug/analyze-text")
async def debug_analyze_text(payload: dict):
    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        raise HTTPException(400, "transcript is required")

    word_count = len(transcript.split())

    vocab = score_vocabulary(transcript)

    # Same preference order as score_free_speech(): HTTP LanguageTool server
    # first (richer /v2/check matches + /v2/analyze token data), falling back
    # to the existing local language_tool_python / naive-regex path unchanged.
    lt_grammar, linguistic_analysis, lt_errors = _lt_provider.check_and_analyze(transcript)

    ge, grammar_issues, grammar_source, heuristic_added, grammar_context = resolve_grammar(
        transcript, lt_grammar, linguistic_analysis)
    grammar_s = score_grammar(ge, word_count)

    # Build a result dict for persistence (matches score_free_speech() shape)
    result_for_db = {
        "transcript": transcript,
        "overall": None,
        "pace": {"score": None, "wpm": None},
        "filler": {"score": None, "count": 0, "words": [], "occurrences": [], "rate_per_min": None,
                   "spoken_count": 0, "spoken_fillers": [],
                   "acoustic_hesitations": [], "acoustic_hesitation_count": 0},
        "pronunciation": {"score": None, "issues": [], "provider": "text_only", "requested_provider": "text_only", "available": True, "detail": None, "methodology": None},
        "grammar": {"score": round(grammar_s, 1), "errors": ge, "issues": grammar_issues},
        "grammar_context": grammar_context,
        "clarity": {"score": None},
        "vocabulary": vocab,
        "cefr": {"score": None, "level": None},
        "archetype": {"archetype": "Text Analysis Only"},
        "feedback": "Text-only analysis: only grammar and vocabulary are available.",
        "hesitations": [],
        "linguistic_analysis": linguistic_analysis,
        "languagetool_errors": lt_errors,
        "grammar_source": grammar_source,
    }

    # Persist debug text analysis to database
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        aid = db.save_assessment(
            timestamp=ts,
            duration=0,  # No audio duration
            result=result_for_db,
            user_id=DEBUG_USER_ID
        )
        debug_assessment_id = aid
        saved_to_db = True
        save_error = None
    except Exception as e:
        debug_assessment_id = None
        saved_to_db = False
        save_error = str(e)

    return {
        "transcript": transcript,
        "word_count": word_count,
        "vocabulary": vocab,
        "grammar": {"score": round(grammar_s, 1), "errors": ge, "issues": grammar_issues},
        "grammar_context": grammar_context,
        "grammar_source": grammar_source,
        "grammar_heuristic_issues_added": heuristic_added,
        "grammar_tool_available": GRAMMAR_OK,
        "linguistic_analysis": linguistic_analysis,
        "languagetool_errors": lt_errors,
        "debug_assessment_id": debug_assessment_id,
        "saved_to_db": saved_to_db,
        "save_error": save_error,
        "note": ("Text-only debug analysis: pace, pronunciation, clarity, overall, and CEFR "
                 "require audio (Whisper timing/segments) and are not computed here. "
                 "'grammar.errors'/'grammar.issues' are POST-VALIDATION: LanguageTool + "
                 "heuristic candidates are first run through grammar_context_validator "
                 "(Groq, or an offline heuristic fallback) and only candidates judged "
                 "'true_grammar_error' remain — see 'grammar_context' for the reclassified "
                 "written-English/style/usage notes and the full candidate+judgment trail. "
                 "'linguistic_analysis' comes from the LanguageTool /v2/analyze endpoint "
                 "(token/lemma/POS data only — it does not indicate grammatical correctness "
                 "or pronunciation quality; use 'grammar' for correctness). It is null if "
                 "the LanguageTool server or /v2/analyze specifically was unreachable — see "
                 "'languagetool_errors'. 'grammar_source' shows which grammar path was "
                 "actually used for this request."),
    }


# ── DEBUG-ONLY ROUTE: audio → existing pipeline ─────────────────────────────────
# Isolated addition — does not modify /assess, /assess/stage, save_and_convert(),
# transcribe_wav(), or score_free_speech(). This route calls those three exactly
# as /assess does, so the transcript/scores it returns are byte-for-byte what the
# production pipeline would produce for the same audio. It intentionally skips
# require_user() (same rationale as /debug/analyze-text — no PocketBase login
# available to a standalone local debug page; do not expose this in production).
#
# Two fields are added on top of score_free_speech()'s normal return value.
# Both are read-only derivations from data score_free_speech() already computed
# internally (Whisper's segments/words) — nothing here changes what any existing
# route returns, and no scoring formula, threshold, or the FILLERS list itself
# is touched:
#
#   - word_timings: provider-neutral word-level timing data. {"source": the
#     STT provider that actually ran (see resolve_stt() — never assume
#     Whisper), "available": bool, "words": [...]}. Whisper populates
#     `words` from its word_timestamps=True output (word, start, end,
#     duration, probability); Saaras's REST endpoint documents chunk/
#     sentence-level timestamps only (no word-level, no confidence — see
#     stt_provider.py module docstring), so SaarasSTTProvider deliberately
#     returns segments=[] and `words` here is correctly empty with
#     available=False and source="saaras" — the UI must show that as "Saaras
#     doesn't provide word-level timestamps," not as "Whisper Word Timings /
#     No data returned" (which reads as a Whisper failure that never
#     happened). This field replaces the old `whisper_words` name, which
#     hard-coded an assumption the data in it was always Whisper's.
#
#   - filler_occurrences: legacy, Whisper-only, informational cross-check —
#     individual timestamped hits from matching the *existing* FILLERS list
#     against Whisper's word sequence (word_timings). Requires word-level
#     timing data, so with Saaras selected this is always empty regardless of
#     whether fillers were actually found — that's expected, not a detection
#     failure. The authoritative filler score/count/occurrences (works with
#     any STT provider, since it runs on the transcript text + optional
#     LanguageTool analysis, not on audio word-timing data) is `filler.score`
#     / `filler.count` / `filler.occurrences` from score_free_speech() — see
#     debug-ui/app.js renderFillers(), which now renders `filler.occurrences`
#     as the primary table and this field only as a secondary, clearly
#     labeled cross-check.
def _build_word_timings(segments: list, source: str) -> dict:
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            words.append({
                "word": w.get("word", ""),
                "start": start,
                "end": end,
                "duration": round(end - start, 3) if start is not None and end is not None else None,
                "probability": w.get("probability"),
            })
    return {"source": source, "available": bool(words), "words": words}


def _find_filler_occurrences(flat_words: list) -> list:
    """Match the existing FILLERS list against Whisper's word sequence to
    recover per-occurrence timestamps. Longest phrases matched first so e.g.
    'so basically' isn't swallowed by a bare 'so'/'basically' match first."""
    norm = lambda s: s.strip().strip(".,!?;:()[]\"'—-").lower()
    phrases = sorted(FILLERS, key=lambda p: -len(p.split()))
    tokens = [norm(w["word"]) for w in flat_words]

    occurrences = []
    i = 0
    n = len(flat_words)
    while i < n:
        matched = None
        for phrase in phrases:
            parts = phrase.split()
            k = len(parts)
            if i + k <= n and tokens[i:i + k] == parts:
                matched = (phrase, k)
                break
        if matched:
            phrase, k = matched
            span = flat_words[i:i + k]
            probs = [w["probability"] for w in span if w["probability"] is not None]
            occurrences.append({
                "word": phrase,
                "start": span[0]["start"],
                "end": span[-1]["end"],
                "type": "multi-word" if k > 1 else "single-word",
                "confidence": round(min(probs) * 100, 1) if probs else None,
                "reason": f"matched filler list entry '{phrase}'",
            })
            i += k
        else:
            i += 1
    return occurrences


@app.post("/debug/analyze-audio")
async def debug_analyze_audio(audio: UploadFile = File(...), duration: float = Form(0),
                               pronunciation_provider: str = Form(DEFAULT_PRONUNCIATION_PROVIDER),
                               stt_provider: str = Form(DEFAULT_STT_PROVIDER)):
    pronunciation_provider = validate_pronunciation_provider(pronunciation_provider)
    stt_provider = validate_stt_provider(stt_provider)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = save_and_convert(await audio.read(), ts, prefix="debug")
    # Authoritative duration from the actual converted WAV — see
    # audio_utils.wav_duration_seconds(). Falls back to the client value
    # only if the WAV is somehow unreadable (0.0 frames/rate).
    wav_duration = wav_duration_seconds(wav_path)
    audio_duration = wav_duration or duration
    stt_requested, stt_used = resolve_stt(stt_provider, wav_path)
    transcript, segments = stt_used.transcript, stt_used.segments
    r = score_free_speech(transcript, segments, audio_duration, wav_path,
                           pronunciation_provider=pronunciation_provider)
    r["stt"] = {
        "provider": stt_used.provider,
        "requested_provider": stt_provider,
        "available": stt_requested.available,
        "detail": None if stt_requested.available else stt_requested.detail,
    }

    # Groq teacher report — additive, second step on top of the deterministic
    # evidence above (see groq_provider.py). Never touches scores/evidence;
    # if GROQ_API_KEY is unset or the request fails, teacher_report is just
    # null and the rest of this response is unaffected.
    try:
        teacher_report_result = generate_teacher_report(
            transcript, r.get("transcript_with_fillers"), r)
        r["teacher_report"] = teacher_report_result.report if teacher_report_result.available else None
        r["teacher_report_detail"] = teacher_report_result.detail
    except Exception as e:
        r["teacher_report"] = None
        r["teacher_report_detail"] = f"Teacher report generation raised an unexpected error: {e}"

    word_timings = _build_word_timings(segments, stt_used.provider)

    # Persist debug audio analysis to database
    try:
        aid = db.save_assessment(
            timestamp=ts,
            duration=audio_duration,
            result=r,
            user_id=DEBUG_USER_ID
        )
        debug_assessment_id = aid
        saved_to_db = True
        save_error = None
    except Exception as e:
        debug_assessment_id = None
        saved_to_db = False
        save_error = str(e)

    return {
        "duration": audio_duration,
        "duration_client_sent": duration,
        "duration_source": "wav_frames" if wav_duration else "client_fallback",
        **r,
        "word_timings": word_timings,
        "filler_occurrences": _find_filler_occurrences(word_timings["words"]),
        "debug_assessment_id": debug_assessment_id,
        "saved_to_db": saved_to_db,
        "save_error": save_error,
        "note": ("Audio debug analysis: transcript/scores come from the same "
                 "save_and_convert() → resolve_stt() → score_free_speech() "
                 "path used by production /assess. 'word_timings' is a "
                 "provider-neutral, additive, read-only debug field — see "
                 "'word_timings.source' for which STT provider it actually "
                 "came from (never assume Whisper) and 'word_timings.available' "
                 "for whether that provider returns word-level data at all "
                 "(Saaras does not). It does not feed the score. "
                 "'filler_occurrences' (naive FILLERS-list match against "
                 "word_timings) is a legacy, Whisper-only cross-check — it is "
                 "always empty when word_timings.available is False, which is "
                 "expected, not a detection failure. The authoritative, "
                 "context-aware, provider-independent filler evidence is "
                 "'filler.occurrences' / 'hesitations' in the main response "
                 "body (from filler_detector.detect_fillers(), which is what "
                 "filler.score/filler.count actually use). 'duration' is "
                 "computed from the actual converted audio, not the "
                 "client-sent value — see 'duration_client_sent' and "
                 "'duration_source' to compare the two."),
    }


# ── DEBUG UI static mount (isolated, additive) ─────────────────────────────────
# Serves the standalone debug page from debug-ui/. Does not touch or interfere
# with the existing production static routes (/, /app, /login, etc.) above.
app.mount("/debug-ui", StaticFiles(directory=str(BASE_DIR / "debug-ui"), html=True), name="debug-ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5050, reload=False)