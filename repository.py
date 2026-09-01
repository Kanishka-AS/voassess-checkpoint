"""
Repository layer for database persistence.

This module handles all database operations for the VoiceCoach application.
It provides functions to save and retrieve assessment results from SQLite.

The database stores:
- Quick assessments (from /assess and debug routes)
- Guided English assessments (from /assessment/finalize)

All functions enforce user isolation via user_id filtering.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Database path - same as used in app.py
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "assessments.db"


def _json_dumps(obj):
    """Safely convert to JSON, handling non-serializable types."""
    if obj is None:
        return None
    try:
        return json.dumps(obj, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(obj))


def _json_loads(data):
    """Safely load JSON, returning None for null/empty."""
    if data is None or data == "":
        return None
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError):
        return None


def _get_conn():
    """Get a database connection with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database schema if it doesn't exist."""
    conn = sqlite3.connect(str(DB_PATH))
    
    # assessments table - with full_result JSON column for complete persistence
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            duration REAL,
            transcript TEXT,
            overall_score REAL,
            pace_score REAL,
            filler_score REAL,
            pronunciation_score REAL,
            grammar_score REAL,
            clarity_score REAL,
            vocabulary_score REAL,
            cefr_score REAL,
            cefr_level TEXT,
            archetype TEXT,
            pace_wpm REAL,
            filler_count INTEGER,
            filler_words TEXT,
            grammar_errors INTEGER,
            feedback TEXT,
            full_result TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add columns if they don't exist (migration-safe)
    columns_to_add = [
        ("vocabulary_score", "REAL"),
        ("cefr_score", "REAL"),
        ("cefr_level", "TEXT"),
        ("archetype", "TEXT"),
        ("user_id", "TEXT"),
        ("full_result", "TEXT"),
    ]
    
    for col, coltype in columns_to_add:
        try:
            conn.execute(f"ALTER TABLE assessments ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    
    # english_assessments table - with full_result JSON column
    conn.execute("""
        CREATE TABLE IF NOT EXISTS english_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            name TEXT,
            picture_talk_score REAL,
            media_repeat_score REAL,
            picture_describe_score REAL,
            overall_score REAL,
            pace_score REAL,
            filler_score REAL,
            pronunciation_score REAL,
            grammar_score REAL,
            clarity_score REAL,
            vocabulary_score REAL,
            cefr_score REAL,
            cefr_level TEXT,
            archetype TEXT,
            full_result TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add columns to english_assessments
    columns_to_add_guided = [
        ("vocabulary_score", "REAL"),
        ("cefr_score", "REAL"),
        ("cefr_level", "TEXT"),
        ("archetype", "TEXT"),
        ("user_id", "TEXT"),
        ("full_result", "TEXT"),
        ("pace_score", "REAL"),
        ("filler_score", "REAL"),
        ("pronunciation_score", "REAL"),
        ("grammar_score", "REAL"),
        ("clarity_score", "REAL"),
    ]
    
    for col, coltype in columns_to_add_guided:
        try:
            conn.execute(f"ALTER TABLE english_assessments ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    
    conn.commit()
    conn.close()
    print("Database initialized.")


def save_assessment(timestamp: str, duration: float, result: Dict[str, Any], user_id: str) -> int:
    """
    Save a quick assessment result to the database.
    
    Args:
        timestamp: ISO format timestamp string
        duration: Duration in seconds
        result: The complete assessment result dict from score_free_speech()
        user_id: The user's ID (or 'debug_user' for debug assessments)
    
    Returns:
        The ID of the newly inserted row
    """
    conn = _get_conn()
    try:
        # Extract key fields from the result for easy querying
        overall = result.get('overall')
        pace = result.get('pace', {})
        filler = result.get('filler', {})
        pronunciation = result.get('pronunciation', {})
        grammar = result.get('grammar', {})
        clarity = result.get('clarity', {})
        vocabulary = result.get('vocabulary', {})
        cefr = result.get('cefr', {})
        archetype = result.get('archetype', {})
        
        full_result_json = _json_dumps(result)
        
        cur = conn.execute("""
            INSERT INTO assessments (
                user_id, timestamp, duration, transcript,
                overall_score, pace_score, filler_score,
                pronunciation_score, grammar_score, clarity_score,
                vocabulary_score, cefr_score, cefr_level, archetype,
                pace_wpm, filler_count, filler_words, grammar_errors,
                feedback, full_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            timestamp,
            duration,
            result.get('transcript'),
            overall,
            pace.get('score'),
            filler.get('score'),
            pronunciation.get('score'),
            grammar.get('score'),
            clarity.get('score'),
            vocabulary.get('score'),
            cefr.get('score'),
            cefr.get('level'),
            archetype.get('archetype'),
            pace.get('wpm'),
            filler.get('count'),
            _json_dumps(filler.get('words')),
            grammar.get('errors'),
            result.get('feedback'),
            full_result_json
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def save_english_assessment(
    timestamp: str,
    name: str,
    picture_talk_score: Optional[float],
    media_repeat_score: Optional[float],
    picture_describe_score: Optional[float],
    overall_score: float,
    final_stage: Dict[str, Any],
    vocab_score: float,
    cefr_score: float,
    cefr_level: str,
    archetype: str,
    stages: List[Dict[str, Any]],
    user_id: str
) -> int:
    """
    Save a guided English assessment result to the database.
    
    Args:
        timestamp: ISO format timestamp string
        name: User's name
        picture_talk_score: Average score for picture talk stage
        media_repeat_score: Average score for media repeat stage
        picture_describe_score: Average score for picture describe stage
        overall_score: Overall assessment score
        final_stage: The final stage result dict
        vocab_score: Vocabulary score
        cefr_score: CEFR score
        cefr_level: CEFR level (A1-C2)
        archetype: Voice archetype label
        stages: List of all stage results
        user_id: The user's ID
    
    Returns:
        The ID of the newly inserted row
    """
    conn = _get_conn()
    try:
        # Build the complete result as JSON
        full_result = {
            "timestamp": timestamp,
            "name": name,
            "overall_score": overall_score,
            "vocabulary": {"score": vocab_score},
            "cefr": {"score": cefr_score, "level": cefr_level},
            "sections": {
                "picture_talk": {"score": picture_talk_score},
                "media_repeat": {"score": media_repeat_score},
                "picture_describe": {"score": picture_describe_score},
            },
            "final": final_stage,
            "stages": stages,
        }
        full_result_json = _json_dumps(full_result)
        
        cur = conn.execute("""
            INSERT INTO english_assessments (
                user_id, timestamp, name,
                picture_talk_score, media_repeat_score, picture_describe_score,
                overall_score,
                pace_score, filler_score, pronunciation_score,
                grammar_score, clarity_score,
                vocabulary_score, cefr_score, cefr_level, archetype,
                full_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            timestamp,
            name,
            picture_talk_score,
            media_repeat_score,
            picture_describe_score,
            overall_score,
            final_stage.get('pace', {}).get('score'),
            final_stage.get('filler', {}).get('score'),
            final_stage.get('pronunciation', {}).get('score'),
            final_stage.get('grammar', {}).get('score'),
            final_stage.get('clarity', {}).get('score'),
            vocab_score,
            cefr_score,
            cefr_level,
            archetype,
            full_result_json
        ))
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def get_assessment(assessment_id: int, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a single assessment by ID with user ownership check.
    
    Args:
        assessment_id: The assessment ID
        user_id: The user's ID (must match)
    
    Returns:
        The assessment dict, or None if not found or not owned by user
    """
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT id, timestamp, duration, transcript,
                   overall_score, pace_score, filler_score,
                   pronunciation_score, grammar_score, clarity_score,
                   vocabulary_score, cefr_score, cefr_level, archetype,
                   pace_wpm, filler_count, filler_words, grammar_errors,
                   feedback, full_result
            FROM assessments
            WHERE id = ? AND user_id = ?
        """, (assessment_id, user_id)).fetchone()
        
        if not row:
            return None
        
        # If full_result exists, use it for complete data
        if row['full_result']:
            try:
                full = _json_loads(row['full_result'])
                if full:
                    full['id'] = row['id']
                    # Ensure all key fields are present
                    for key in ['transcript', 'overall', 'pace', 'filler', 
                               'pronunciation', 'grammar', 'clarity', 
                               'vocabulary', 'cefr', 'archetype', 'feedback']:
                        if key not in full:
                            # Fallback to extracted fields
                            if key == 'overall':
                                full['overall'] = row['overall_score']
                            elif key == 'pace':
                                full['pace'] = {'score': row['pace_score'], 'wpm': row['pace_wpm']}
                            elif key == 'filler':
                                full['filler'] = {
                                    'score': row['filler_score'],
                                    'count': row['filler_count'],
                                    'words': _json_loads(row['filler_words']) or []
                                }
                            # ... other fallbacks
                    return full
            except:
                pass
        
        # Fallback: build from extracted fields
        return {
            'id': row['id'],
            'timestamp': row['timestamp'],
            'duration': row['duration'],
            'transcript': row['transcript'],
            'overall': row['overall_score'],
            'pace': {'score': row['pace_score'], 'wpm': row['pace_wpm']},
            'filler': {
                'score': row['filler_score'],
                'count': row['filler_count'],
                'words': _json_loads(row['filler_words']) or [],
                'occurrences': [],
                'rate_per_min': None
            },
            'pronunciation': {
                'score': row['pronunciation_score'],
                'issues': [],
                'provider': None,
                'requested_provider': None,
                'available': False,
                'detail': None
            },
            'grammar': {
                'score': row['grammar_score'],
                'errors': row['grammar_errors'],
                'issues': []
            },
            'clarity': {'score': row['clarity_score']},
            'vocabulary': {'score': row['vocabulary_score']},
            'cefr': {'score': row['cefr_score'], 'level': row['cefr_level']},
            'archetype': {'archetype': row['archetype']},
            'feedback': row['feedback'],
            'hesitations': [],
            'linguistic_analysis': None,
            'languagetool_errors': {}
        }
    finally:
        conn.close()


def get_assessment_parameters(assessment_id: int, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get just the parameter results for an assessment.
    
    Args:
        assessment_id: The assessment ID
        user_id: The user's ID (must match)
    
    Returns:
        Dict with parameter results, or None if not found
    """
    full = get_assessment(assessment_id, user_id)
    if not full:
        return None
    
    # Extract just the parameter sections
    return {
        'id': full['id'],
        'timestamp': full.get('timestamp'),
        'pace': full.get('pace'),
        'filler': full.get('filler'),
        'pronunciation': full.get('pronunciation'),
        'grammar': full.get('grammar'),
        'clarity': full.get('clarity'),
        'vocabulary': full.get('vocabulary'),
        'cefr': full.get('cefr'),
        'archetype': full.get('archetype'),
        'hesitations': full.get('hesitations', []),
        'linguistic_analysis': full.get('linguistic_analysis'),
        'languagetool_errors': full.get('languagetool_errors'),
    }


def list_assessments(user_id: str, limit: int = 60) -> List[Dict[str, Any]]:
    """
    List assessments for a user, newest first.
    
    Args:
        user_id: The user's ID
        limit: Maximum number of records to return
    
    Returns:
        List of assessment dicts
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT id, timestamp, duration, transcript,
                   overall_score, pace_score, filler_score,
                   pronunciation_score, grammar_score, clarity_score,
                   vocabulary_score, cefr_score, cefr_level, archetype,
                   pace_wpm, filler_count, filler_words, grammar_errors,
                   feedback, full_result
            FROM assessments
            WHERE user_id = ?
            ORDER BY id DESC LIMIT ?
        """, (user_id, limit)).fetchall()
        
        result = []
        for row in rows:
            # Try to use full_result if available
            if row['full_result']:
                try:
                    full = _json_loads(row['full_result'])
                    if full:
                        full['id'] = row['id']
                        result.append(full)
                        continue
                except:
                    pass
            
            # Fallback
            result.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'duration': row['duration'],
                'transcript': row['transcript'],
                'overall': row['overall_score'],
                'pace': {'score': row['pace_score'], 'wpm': row['pace_wpm']},
                'filler': {
                    'score': row['filler_score'],
                    'count': row['filler_count'],
                    'words': _json_loads(row['filler_words']) or []
                },
                'pronunciation': {'score': row['pronunciation_score'], 'issues': []},
                'grammar': {
                    'score': row['grammar_score'],
                    'errors': row['grammar_errors'],
                    'issues': []
                },
                'clarity': {'score': row['clarity_score']},
                'vocabulary': {'score': row['vocabulary_score']},
                'cefr': {'score': row['cefr_score'], 'level': row['cefr_level']},
                'archetype': {'archetype': row['archetype']},
                'feedback': row['feedback']
            })
        
        return result
    finally:
        conn.close()


def get_english_assessment(assessment_id: int, user_id: str) -> Optional[Dict[str, Any]]:
    """Get a guided assessment by ID with user ownership check."""
    conn = _get_conn()
    try:
        row = conn.execute("""
            SELECT id, timestamp, name, picture_talk_score, media_repeat_score,
                   picture_describe_score, overall_score,
                   pace_score, filler_score, pronunciation_score,
                   grammar_score, clarity_score,
                   vocabulary_score, cefr_score, cefr_level, archetype,
                   full_result
            FROM english_assessments
            WHERE id = ? AND user_id = ?
        """, (assessment_id, user_id)).fetchone()
        
        if not row:
            return None
        
        if row['full_result']:
            try:
                full = _json_loads(row['full_result'])
                if full:
                    full['id'] = row['id']
                    return full
            except:
                pass
        
        return {
            'id': row['id'],
            'timestamp': row['timestamp'],
            'name': row['name'],
            'overall_score': row['overall_score'],
            'picture_talk_score': row['picture_talk_score'],
            'media_repeat_score': row['media_repeat_score'],
            'picture_describe_score': row['picture_describe_score'],
            'pace': row['pace_score'],
            'filler': row['filler_score'],
            'pronunciation': row['pronunciation_score'],
            'grammar': row['grammar_score'],
            'clarity': row['clarity_score'],
            'vocabulary': {'score': row['vocabulary_score']},
            'cefr': {'score': row['cefr_score'], 'level': row['cefr_level']},
            'archetype': {'archetype': row['archetype']},
        }
    finally:
        conn.close()


def get_english_assessment_parameters(assessment_id: int, user_id: str) -> Optional[Dict[str, Any]]:
    """Get just the parameters for a guided assessment."""
    full = get_english_assessment(assessment_id, user_id)
    if not full:
        return None
    
    return {
        'id': full['id'],
        'timestamp': full.get('timestamp'),
        'name': full.get('name'),
        'picture_talk_score': full.get('picture_talk_score'),
        'media_repeat_score': full.get('media_repeat_score'),
        'picture_describe_score': full.get('picture_describe_score'),
        'overall_score': full.get('overall_score'),
        'pace': full.get('pace'),
        'filler': full.get('filler'),
        'pronunciation': full.get('pronunciation'),
        'grammar': full.get('grammar'),
        'clarity': full.get('clarity'),
        'vocabulary': full.get('vocabulary'),
        'cefr': full.get('cefr'),
        'archetype': full.get('archetype'),
    }


def list_english_assessments(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    """List guided assessments for a user, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT id, timestamp, name, picture_talk_score, media_repeat_score,
                   picture_describe_score, overall_score,
                   pace_score, filler_score, pronunciation_score,
                   grammar_score, clarity_score,
                   vocabulary_score, cefr_score, cefr_level, archetype,
                   full_result
            FROM english_assessments
            WHERE user_id = ?
            ORDER BY id DESC LIMIT ?
        """, (user_id, limit)).fetchall()
        
        result = []
        for row in rows:
            if row['full_result']:
                try:
                    full = _json_loads(row['full_result'])
                    if full:
                        full['id'] = row['id']
                        result.append(full)
                        continue
                except:
                    pass
            
            result.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'name': row['name'],
                'picture_talk_score': row['picture_talk_score'],
                'media_repeat_score': row['media_repeat_score'],
                'picture_describe_score': row['picture_describe_score'],
                'overall_score': row['overall_score'],
                'pace': row['pace_score'],
                'filler': row['filler_score'],
                'pronunciation': row['pronunciation_score'],
                'grammar': row['grammar_score'],
                'clarity': row['clarity_score'],
                'vocabulary': {'score': row['vocabulary_score']},
                'cefr': {'score': row['cefr_score'], 'level': row['cefr_level']},
                'archetype': {'archetype': row['archetype']},
            })
        
        return result
    finally:
        conn.close()
