"""
Focused tests for languagetool_provider.py.

All HTTP calls are mocked via unittest.mock.patch on languagetool_provider.httpx.post —
no real LanguageTool server is required to run these tests. Response fixtures below
mirror the actual /v2/check and /v2/analyze JSON shapes (the latter matches the
AnalyzeResponse/SentenceAnalysis/TokenAnalysis/AlternativeReading DTOs).
"""
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from languagetool_provider import LanguageToolProvider, LanguageToolUnavailable


def _mock_response(json_data, status_ok=True):
    resp = MagicMock()
    resp.json.return_value = json_data
    if status_ok:
        resp.raise_for_status.return_value = None
    else:
        resp.raise_for_status.side_effect = Exception("HTTP error")
    return resp


CHECK_RESPONSE_FIXTURE = {
    "matches": [
        {
            "message": "Possible agreement error.",
            "replacements": [{"value": "have"}],
            "offset": 2,
            "length": 3,
            "context": {"text": "I has a book.", "offset": 2, "length": 3},
            "rule": {"id": "AGREEMENT", "category": {"name": "Grammar"}},
        }
    ]
}

ANALYZE_RESPONSE_FIXTURE = {
    "language": "en-US",
    "sentences": [
        {
            "text": "I went to Chennai yesterday.",
            "tokens": [
                {"text": "I", "lemma": "I", "partOfSpeech": "Personal pronoun",
                 "posTag": "PRP", "startOffset": 0, "endOffset": 1},
                {"text": "went", "lemma": "go", "partOfSpeech": "Verb, past tense",
                 "posTag": "VBD", "startOffset": 2, "endOffset": 6},
                {"text": "to", "lemma": "to", "partOfSpeech": "Preposition or subordinating conjunction",
                 "posTag": "IN", "startOffset": 7, "endOffset": 9,
                 "alternatives": [{"partOfSpeech": "Infinitive marker 'to'", "posTag": "TO"}]},
                {"text": "Chennai", "lemma": "Chennai", "partOfSpeech": "Proper noun, singular",
                 "posTag": "NNP", "startOffset": 10, "endOffset": 17},
            ],
        }
    ],
}


# ---- 1. /v2/check successful response --------------------------------------
def test_check_grammar_success():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", return_value=_mock_response(CHECK_RESPONSE_FIXTURE)):
        result = provider.check_grammar("I has a book.")
    assert result["errors"] == 1
    assert len(result["issues"]) == 1
    issue = result["issues"][0]
    assert issue["wrong"] == "has"
    assert issue["correct"] == "have"
    assert issue["message"] == "Possible agreement error."
    assert issue["context"] == "I has a book."
    assert issue["rule_id"] == "AGREEMENT"
    assert issue["category"] == "Grammar"
    assert issue["offset"] == 2
    assert issue["length"] == 3


# ---- 2. /v2/analyze successful response ------------------------------------
def test_analyze_success():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", return_value=_mock_response(ANALYZE_RESPONSE_FIXTURE)):
        result = provider.analyze("I went to Chennai yesterday.")
    assert result["language"] == "en-US"
    assert len(result["sentences"]) == 1
    assert len(result["sentences"][0]["tokens"]) == 4


# ---- 3. malformed/empty transcript ------------------------------------------
def test_empty_transcript_short_circuits_without_http_call():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post") as mock_post:
        check_result = provider.check_grammar("")
        analyze_result = provider.analyze("   ")
    mock_post.assert_not_called()
    assert check_result == {"errors": 0, "issues": []}
    assert analyze_result == {"language": "en-US", "sentences": []}


# ---- 4. LanguageTool unavailable (connection error) -------------------------
def test_check_grammar_raises_when_server_unreachable():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", side_effect=ConnectionError("refused")):
        try:
            provider.check_grammar("Hello there.")
            assert False, "expected LanguageToolUnavailable"
        except LanguageToolUnavailable:
            pass


def test_analyze_raises_when_server_unreachable():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", side_effect=ConnectionError("refused")):
        try:
            provider.analyze("Hello there.")
            assert False, "expected LanguageToolUnavailable"
        except LanguageToolUnavailable:
            pass


# ---- 5. /v2/analyze unavailable while /v2/check still works -----------------
def test_check_and_analyze_degrades_independently():
    provider = LanguageToolProvider(base_url="http://localhost:8010")

    def side_effect(url, **kwargs):
        if url.endswith("/v2/check"):
            return _mock_response(CHECK_RESPONSE_FIXTURE)
        raise ConnectionError("analyze endpoint down")

    with patch("languagetool_provider.httpx.post", side_effect=side_effect):
        grammar, analysis, errors = provider.check_and_analyze("I has a book.")

    assert grammar is not None
    assert grammar["errors"] == 1
    assert analysis is None
    assert "analyze" in errors
    assert "check" not in errors


def test_check_and_analyze_both_fail_independently_reported():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", side_effect=ConnectionError("server down")):
        grammar, analysis, errors = provider.check_and_analyze("Hello there.")
    assert grammar is None
    assert analysis is None
    assert "check" in errors
    assert "analyze" in errors


# ---- 6. POS/lemma parsing ----------------------------------------------------
def test_pos_lemma_parsing():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", return_value=_mock_response(ANALYZE_RESPONSE_FIXTURE)):
        result = provider.analyze("I went to Chennai yesterday.")
    tokens = {t["text"]: t for t in result["sentences"][0]["tokens"]}
    assert tokens["went"]["lemma"] == "go"
    assert tokens["went"]["posTag"] == "VBD"


# ---- 7. proper noun detection -------------------------------------------------
def test_proper_noun_detection():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", return_value=_mock_response(ANALYZE_RESPONSE_FIXTURE)):
        result = provider.analyze("I went to Chennai yesterday.")
    tokens = {t["text"]: t for t in result["sentences"][0]["tokens"]}
    assert tokens["Chennai"]["posTag"] == "NNP"
    assert tokens["Chennai"]["partOfSpeech"] == "Proper noun, singular"


# ---- 8. alternative POS readings ----------------------------------------------
def test_alternative_pos_readings():
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", return_value=_mock_response(ANALYZE_RESPONSE_FIXTURE)):
        result = provider.analyze("I went to Chennai yesterday.")
    tokens = {t["text"]: t for t in result["sentences"][0]["tokens"]}
    to_token = tokens["to"]
    assert to_token["posTag"] == "IN"
    assert to_token["alternatives"][0]["posTag"] == "TO"
    assert to_token["alternatives"][0]["partOfSpeech"] == "Infinitive marker 'to'"
    # A token with no alternatives simply doesn't have the key present or has an empty list —
    # AnalyzeResponse.java's @JsonInclude(NON_NULL) omits it when null, so both are valid.
    went_token = tokens["went"]
    assert not went_token.get("alternatives")


# ---- 9. sentence boundaries / 10. multiple sentences ---------------------------
def test_multiple_sentences_boundaries():
    two_sentence_response = {
        "language": "en-US",
        "sentences": [
            {"text": "I went to Chennai.", "tokens": [
                {"text": "I", "lemma": "I", "partOfSpeech": "Personal pronoun", "posTag": "PRP",
                 "startOffset": 0, "endOffset": 1},
            ]},
            {"text": "It was great.", "tokens": [
                {"text": "It", "lemma": "it", "partOfSpeech": "Personal pronoun", "posTag": "PRP",
                 "startOffset": 20, "endOffset": 22},
            ]},
        ],
    }
    provider = LanguageToolProvider(base_url="http://localhost:8010")
    with patch("languagetool_provider.httpx.post", return_value=_mock_response(two_sentence_response)):
        result = provider.analyze("I went to Chennai. It was great.")
    assert len(result["sentences"]) == 2
    assert result["sentences"][0]["text"] == "I went to Chennai."
    assert result["sentences"][1]["text"] == "It was great."


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed.append((t.__name__, repr(e)))
    print(f"PASSED: {passed}/{len(tests)}")
    for name, err in failed:
        print(f"  FAILED {name}: {err}")
