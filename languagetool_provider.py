"""
Isolated LanguageTool HTTP integration.

Talks to a real LanguageTool server over HTTP — the locally patched LanguageTool
6.8 build that adds a custom /v2/analyze endpoint alongside the stock /v2/check.
Deliberately kept independent of the rest of app.py:

- check_grammar() and analyze() call two unrelated endpoints. Either can fail
  without affecting the other — see check_and_analyze().
- Nothing here touches Whisper, the DB, or auth. It only turns transcript text
  into LanguageTool's JSON and back into plain dicts that match the shape
  app.py's existing grammar code already produces (wrong/correct/message/
  context), plus the new fields /v2/check and /v2/analyze make available.
- app.py decides what to do when this provider is unavailable. This module
  never falls back to anything itself — it raises LanguageToolUnavailable and
  lets the caller keep its own existing fallback behavior.

Configuration:
    LANGUAGETOOL_URL   Base URL of the LanguageTool server, e.g.
                        "http://localhost:8010" (no trailing /v2/...).
                        Defaults to "http://localhost:8010" if unset.
"""
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import httpx

LANGUAGETOOL_URL = os.environ.get("LANGUAGETOOL_URL", "http://localhost:8010").rstrip("/")
DEFAULT_LANGUAGE = "en-US"
DEFAULT_TIMEOUT = httpx.Timeout(10.0, connect=3.0)

# /v2/analyze only supplies *optional* POS/lemma data (used for nicer filler
# detection and the debug UI's Token Analysis panel) — it must never be
# allowed to make the overall grammar assessment slow. Its own HTTP call is
# given a shorter timeout than /v2/check, AND check_and_analyze() below caps
# how long it will *wait* on that call independently of this value, so a
# slow-but-technically-alive server can't stall the response either.
ANALYZE_HTTP_TIMEOUT = httpx.Timeout(4.0, connect=1.5)
ANALYZE_WAIT_BUDGET_SECONDS = 3.0


class LanguageToolUnavailable(Exception):
    """Raised when the LanguageTool server can't be reached, errors, or
    returns a response this provider doesn't recognize. Callers decide how
    to fall back — this module never falls back on its own."""


class LanguageToolProvider:
    """Thin client for a LanguageTool server exposing /v2/check (stock) and
    /v2/analyze (locally patched, returns AnalyzeResponse/SentenceAnalysis/
    TokenAnalysis/AlternativeReading JSON)."""

    def __init__(self, base_url: str = None, language: str = DEFAULT_LANGUAGE,
                 timeout: httpx.Timeout = DEFAULT_TIMEOUT,
                 analyze_timeout: httpx.Timeout = ANALYZE_HTTP_TIMEOUT,
                 analyze_wait_budget: float = ANALYZE_WAIT_BUDGET_SECONDS):
        self.base_url = (base_url or LANGUAGETOOL_URL).rstrip("/")
        self.language = language
        self.timeout = timeout
        # See ANALYZE_HTTP_TIMEOUT / ANALYZE_WAIT_BUDGET_SECONDS above —
        # /v2/analyze is optional and must fail fast without blocking
        # grammar scoring (see check_and_analyze()).
        self.analyze_timeout = analyze_timeout
        self.analyze_wait_budget = analyze_wait_budget

    # ---- /v2/check (stock LanguageTool grammar checking) ----------------------
    def check_grammar(self, text: str) -> dict:
        """
        Calls the server's stock /v2/check endpoint.

        Returns: {"errors": int, "issues": [{wrong, correct, message, context,
                                               rule_id, category, offset, length}]}
        Preserves the wrong/correct/message/context shape app.py's existing
        extract_grammar_issues() already produces, and adds rule_id/category/
        offset/length — real fields /v2/check returns that weren't available
        through the old local language_tool_python.check() path.

        Raises LanguageToolUnavailable on any network/HTTP/parse failure.
        """
        if not text or not text.strip():
            return {"errors": 0, "issues": []}

        try:
            resp = httpx.post(
                f"{self.base_url}/v2/check",
                data={"text": text, "language": self.language},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise LanguageToolUnavailable(f"/v2/check failed: {e}") from e

        matches = data.get("matches")
        if matches is None:
            raise LanguageToolUnavailable("/v2/check returned unexpected shape (no 'matches')")

        issues = []
        for m in matches:
            offset = m.get("offset", 0)
            length = m.get("length", 0)
            wrong = text[offset:offset + length]
            replacements = m.get("replacements") or []
            correct = replacements[0].get("value", "") if replacements else ""
            if not wrong.strip() or wrong.lower() == correct.lower():
                continue
            rule = m.get("rule") or {}
            category = (rule.get("category") or {}).get("name", "")
            issues.append({
                "wrong": wrong,
                "correct": correct,
                "message": m.get("message", ""),
                "context": (m.get("context") or {}).get("text", ""),
                "rule_id": rule.get("id", ""),
                "category": category,
                "offset": offset,
                "length": length,
            })

        return {"errors": len(matches), "issues": issues[:8]}

    # ---- /v2/analyze (locally patched token/POS/lemma endpoint) ---------------
    def analyze(self, text: str) -> dict:
        """
        Calls the locally patched /v2/analyze endpoint.

        Returns the AnalyzeResponse JSON largely as-is — it's already a clean
        structure ({"language", "sentences": [{"text", "tokens": [{"text",
        "lemma", "partOfSpeech", "posTag", "startOffset", "endOffset",
        "alternatives"}]}]}) — no reshaping needed beyond validating its shape.

        Raises LanguageToolUnavailable on any failure. Callers should treat
        that as "no token/POS/lemma data available" without it affecting
        grammar checking.
        """
        if not text or not text.strip():
            return {"language": self.language, "sentences": []}

        try:
            resp = httpx.post(
                f"{self.base_url}/v2/analyze",
                data={"text": text, "language": self.language},
                timeout=self.analyze_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise LanguageToolUnavailable(f"/v2/analyze failed: {e}") from e

        if "sentences" not in data:
            raise LanguageToolUnavailable("/v2/analyze returned unexpected shape (no 'sentences')")

        return data

    # ---- both, independently -----------------------------------------------
    def check_and_analyze(self, text: str) -> tuple:
        """
        Runs check_grammar() and analyze() concurrently in a small thread pool
        (they're two independent, unrelated HTTP calls — no reason to make one
        wait on the other). Neither call's failure affects the other, AND
        /v2/analyze's optional nature is enforced here, not just via its own
        (already shorter) HTTP timeout: we cap how long we're willing to
        *wait* on its future to `self.analyze_wait_budget` seconds regardless
        of what its underlying httpx timeout is. If it's still running past
        that budget, we stop waiting and report it as unavailable — the
        background thread is left to finish on its own (pool.shutdown(wait=
        False) below, instead of the previous `with` block, which would have
        blocked exiting until every submitted task finished even after we'd
        already given up on its result). Grammar checking (`check_grammar`)
        is essential, so it's still awaited for its own full timeout.

        Returns (grammar_or_None, analysis_or_None, errors_dict) where
        errors_dict has a "check" and/or "analyze" key (with the failure
        message) for whichever call(s) failed.
        """
        errors = {}
        pool = ThreadPoolExecutor(max_workers=2)
        future_check = pool.submit(self.check_grammar, text)
        future_analyze = pool.submit(self.analyze, text)

        # Essential call — wait up to its own configured HTTP timeout (plus
        # a small margin for scheduling) so a slow-but-alive server still
        # gets a fair chance; grammar scoring depends on this.
        check_wait_budget = self.timeout.read + self.timeout.connect + 2
        try:
            grammar = future_check.result(timeout=check_wait_budget)
        except FutureTimeoutError:
            grammar = None
            errors["check"] = f"/v2/check exceeded its {check_wait_budget:.1f}s wait budget"
        except LanguageToolUnavailable as e:
            grammar = None
            errors["check"] = str(e)

        # Optional call — capped independently, see docstring above.
        try:
            analysis = future_analyze.result(timeout=self.analyze_wait_budget)
        except FutureTimeoutError:
            analysis = None
            errors["analyze"] = (
                f"/v2/analyze exceeded its {self.analyze_wait_budget:.1f}s optional-call "
                "wait budget; grammar scoring continued without it"
            )
        except LanguageToolUnavailable as e:
            analysis = None
            errors["analyze"] = str(e)

        # Don't block here waiting for a straggler thread — let it finish (or
        # not) in the background. Safe because `pool` isn't shared/reused.
        pool.shutdown(wait=False)

        return grammar, analysis, errors