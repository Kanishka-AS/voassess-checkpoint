"""
Regression tests for repository.get_english_assessment_parameters().

Bug this guards against: the per-metric detail (pace/filler/pronunciation/
grammar/clarity/fluency/archetype/hesitations) for a guided (wizard)
assessment is stored under full_result["final"] (the Full Assessment
stage's complete score_free_speech() output — see
save_english_assessment()/assessment_finalize() in app.py), not at the top
level of the saved JSON. The pre-fix implementation read
`full.get('pace')` etc. directly, which returned None every time, silently
breaking GET /assessment/{id}/parameters for every guided assessment ever
saved, even though the correct data existed one level down in the exact
same row.

Uses a real (temporary, on-disk) SQLite DB via repository.py's own
init_db()/save_english_assessment() — no mocking of the persistence layer
itself, since that's exactly what's under test.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import repository as db


def _fresh_db(tmp_path):
    db.DB_PATH = tmp_path / "test_assessments.db"
    db.init_db()


def _sample_final_stage():
    return {
        "pace": {"score": 88.0, "wpm": 140.2},
        "filler": {"score": 90.0, "count": 1, "words": ["um×1"]},
        "pronunciation": {"score": 76.4, "issues": []},
        "grammar": {"score": 82.0, "errors": 2, "issues": []},
        "clarity": {"score": 84.1},
        "fluency": {"score": 91.5, "pause_data_available": True,
                    "pause_count": 2, "long_pause_count": 0},
        "archetype": {"archetype": "The Analyst", "emoji": "🧠"},
        "overall": 85.0,
        "hesitations": [],
        "evidence": {"word_count": 80, "duration_seconds": 40.0,
                     "low_evidence": False, "reason": None},
    }


def test_guided_assessment_parameters_returns_real_metric_data(tmp_path):
    _fresh_db(tmp_path)
    final_stage = _sample_final_stage()
    aid = db.save_english_assessment(
        timestamp="20260901_000000", name="Test User",
        picture_talk_score=90.0, media_repeat_score=88.0, picture_describe_score=80.0,
        overall_score=85.0, final_stage=final_stage,
        vocab_score=70.0, cefr_score=65.0, cefr_level="B1", archetype="The Analyst",
        stages=[dict(stage_type="final", **final_stage)], user_id="u1",
    )

    params = db.get_english_assessment_parameters(aid, "u1")

    # Before the fix, every one of these was None.
    assert params["pace"] == final_stage["pace"]
    assert params["filler"] == final_stage["filler"]
    assert params["pronunciation"] == final_stage["pronunciation"]
    assert params["grammar"] == final_stage["grammar"]
    assert params["clarity"] == final_stage["clarity"]
    assert params["fluency"] == final_stage["fluency"]
    assert params["archetype"] == final_stage["archetype"]
    assert params["overall_score"] == 85.0
    assert params["picture_talk_score"] == 90.0


def test_guided_assessment_parameters_scoped_to_owner(tmp_path):
    """A user_id mismatch must return None, not another user's data."""
    _fresh_db(tmp_path)
    final_stage = _sample_final_stage()
    aid = db.save_english_assessment(
        timestamp="20260901_000000", name="Test User",
        picture_talk_score=90.0, media_repeat_score=88.0, picture_describe_score=80.0,
        overall_score=85.0, final_stage=final_stage,
        vocab_score=70.0, cefr_score=65.0, cefr_level="B1", archetype="The Analyst",
        stages=[dict(stage_type="final", **final_stage)], user_id="owner",
    )
    assert db.get_english_assessment_parameters(aid, "someone_else") is None


def test_missing_assessment_returns_none(tmp_path):
    _fresh_db(tmp_path)
    assert db.get_english_assessment_parameters(9999, "u1") is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
