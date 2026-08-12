"""
session_history persistence tests (8 cases).

Covers the claims made about this module: schema creation, idempotent
re-initialisation, save/load round-trips, ordering, per-session sample
retrieval, deletion, clearing, and the v2 -> v3 task_mode migration.
"""

import sqlite3
from datetime import datetime, timedelta

import pytest

import session_history


def _history(n=80, start=None, level_cycle=("LOW", "MODERATE", "HIGH")):
    """Synthetic in-memory session history shaped like app.py produces."""
    start = start or datetime(2026, 8, 11, 10, 0, 0)
    return [
        {
            "ts": start + timedelta(seconds=i * 0.5),
            "stress_index": 0.30 + (i % 10) * 0.02,
            "level": level_cycle[i % len(level_cycle)],
            "daisee": 0.50,
            "behavior": 0.40,
        }
        for i in range(n)
    ]


def _tables(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def test_schema_creation(temp_db):
    """init_db() creates both tables."""
    session_history.init_db()
    assert {"sessions", "samples"} <= _tables(temp_db)


def test_idempotent_reinit(temp_db):
    """Calling init_db() repeatedly must not error or duplicate columns."""
    for _ in range(3):
        session_history.init_db()

    conn = sqlite3.connect(str(temp_db))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)")]
    finally:
        conn.close()
    assert cols.count("task_mode") == 1


def test_save_load_roundtrip(temp_db):
    """80 samples in, 80 samples out, values preserved."""
    session_history.init_db()
    history = _history(80)

    sid = session_history.save_session(history, task_mode="Studying")
    assert sid is not None

    loaded = session_history.load_session_samples(sid)
    assert len(loaded) == 80
    assert loaded[0]["level"] == history[0]["level"]
    assert loaded[0]["stress_index"] == pytest.approx(history[0]["stress_index"])
    assert loaded[-1]["ts"] == history[-1]["ts"]


def test_empty_history_returns_none(temp_db):
    """Saving an empty session is a no-op, not a crash or a phantom row."""
    session_history.init_db()
    assert session_history.save_session([]) is None
    assert session_history.session_count() == 0


def test_recent_sessions_newest_first(temp_db):
    """load_recent_sessions() orders newest first and honours its limit."""
    session_history.init_db()
    for day in range(3):
        session_history.save_session(
            _history(10, start=datetime(2026, 8, 1 + day, 9, 0, 0)),
            task_mode="Default",
        )

    recent = session_history.load_recent_sessions(n=2)
    assert len(recent) == 2
    assert recent[0]["id"] > recent[1]["id"]
    assert session_history.session_count() == 3


def test_summary_statistics(temp_db):
    """Percentages reflect the level distribution and sum to 100."""
    session_history.init_db()
    # 30 rows cycling LOW/MODERATE/HIGH -> 10 each
    sid = session_history.save_session(_history(30), task_mode="Hospital")

    row = next(s for s in session_history.load_recent_sessions(10)
               if s["id"] == sid)
    assert row["sample_count"] == 30
    assert row["task_mode"] == "Hospital"
    assert row["pct_low"] == pytest.approx(100 / 3, abs=0.01)
    assert (row["pct_low"] + row["pct_moderate"]
            + row["pct_high"]) == pytest.approx(100.0)


def test_delete_session_removes_its_samples(temp_db):
    """Deleting a session removes its samples and leaves others intact."""
    session_history.init_db()
    keep = session_history.save_session(_history(10))
    drop = session_history.save_session(_history(10))

    session_history.delete_session(drop)

    assert session_history.load_session_samples(drop) == []
    assert len(session_history.load_session_samples(keep)) == 10
    assert session_history.session_count() == 1


def test_clear_all_sessions(temp_db):
    """clear_all_sessions() empties both tables."""
    session_history.init_db()
    sid = session_history.save_session(_history(10))
    session_history.clear_all_sessions()

    assert session_history.session_count() == 0
    assert session_history.load_session_samples(sid) == []


def test_v2_to_v3_migration_adds_task_mode(temp_db):
    """
    A legacy v2 database (no task_mode column) is migrated in place by
    init_db(), and existing rows survive with the 'Default' backfill.
    """
    temp_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(temp_db))
    try:
        conn.execute("""
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
                duration_sec REAL NOT NULL, sample_count INTEGER NOT NULL,
                mean_stress REAL NOT NULL, max_stress REAL NOT NULL,
                pct_low REAL NOT NULL, pct_moderate REAL NOT NULL,
                pct_high REAL NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO sessions (started_at, ended_at, duration_sec,
                sample_count, mean_stress, max_stress,
                pct_low, pct_moderate, pct_high)
            VALUES ('2026-05-01T10:00:00', '2026-05-01T10:01:00',
                    60.0, 100, 0.3, 0.6, 80.0, 15.0, 5.0)
        """)
        conn.commit()
        pre_cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)")]
    finally:
        conn.close()
    assert "task_mode" not in pre_cols          # genuinely a v2 schema

    session_history.init_db()                    # performs the migration

    rows = session_history.load_recent_sessions(10)
    assert len(rows) == 1                        # legacy row preserved
    assert rows[0]["task_mode"] == "Default"     # backfilled
