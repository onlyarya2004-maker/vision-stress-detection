"""
Session History DB (v2 - with per-frame samples)
=================================================
Persists completed sessions to a local SQLite database. Each session has
a one-row summary in the `sessions` table plus N per-frame rows in the
`samples` table (one per recorded frame). Per-frame storage allows us
to re-render any past session's timeline graph on demand.

Database location: ../data/sessions.db  (auto-created on first call)

Schema:
    sessions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at      TEXT     ISO8601 timestamp,
        ended_at        TEXT     ISO8601 timestamp,
        duration_sec    REAL,
        sample_count    INTEGER,
        mean_stress     REAL,
        max_stress      REAL,
        pct_low         REAL,
        pct_moderate    REAL,
        pct_high        REAL
    )
    samples (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id      INTEGER FK -> sessions.id,
        ts              TEXT,
        stress_index    REAL,
        level           TEXT,
        daisee          REAL,
        behavior        REAL
    )

Storage cost: ~80-100 samples per session, 6 columns each ~ 8 KB per
session. 100 sessions ~ 800 KB total. Negligible.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path('../data/sessions.db')


def _get_conn():
    """Open a fresh connection. Caller is responsible for closing it."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they do not exist. Safe to call repeatedly."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at      TEXT NOT NULL,
                ended_at        TEXT NOT NULL,
                duration_sec    REAL NOT NULL,
                sample_count    INTEGER NOT NULL,
                mean_stress     REAL NOT NULL,
                max_stress      REAL NOT NULL,
                pct_low         REAL NOT NULL,
                pct_moderate    REAL NOT NULL,
                pct_high        REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL,
                ts              TEXT NOT NULL,
                stress_index    REAL NOT NULL,
                level           TEXT NOT NULL,
                daisee          REAL NOT NULL,
                behavior        REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_samples_session
            ON samples(session_id)
        """)
        conn.commit()
    finally:
        conn.close()


def save_session(history: List[Dict]) -> Optional[int]:
    """
    Save a completed session to the DB (summary + every frame).

    Args:
        history: list of records, each with keys:
            'ts' (datetime), 'stress_index' (float), 'level' (str),
            'daisee' (float), 'behavior' (float)

    Returns:
        The new session's id, or None if history was empty.
    """
    if not history:
        return None

    started_at = history[0]['ts']
    ended_at   = history[-1]['ts']
    duration   = (ended_at - started_at).total_seconds()
    n          = len(history)

    stress_vals = [r['stress_index'] for r in history]
    mean_stress = sum(stress_vals) / n
    max_stress  = max(stress_vals)

    counts = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0}
    for r in history:
        lvl = r.get('level', 'LOW')
        counts[lvl] = counts.get(lvl, 0) + 1
    pct_low      = 100.0 * counts['LOW']      / n
    pct_moderate = 100.0 * counts['MODERATE'] / n
    pct_high     = 100.0 * counts['HIGH']     / n

    conn = _get_conn()
    try:
        cursor = conn.execute("""
            INSERT INTO sessions (
                started_at, ended_at, duration_sec, sample_count,
                mean_stress, max_stress,
                pct_low, pct_moderate, pct_high
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            started_at.isoformat(),
            ended_at.isoformat(),
            duration, n,
            mean_stress, max_stress,
            pct_low, pct_moderate, pct_high,
        ))
        session_id = cursor.lastrowid

        # Insert all samples in one batch
        sample_rows = [
            (
                session_id,
                r['ts'].isoformat(),
                float(r['stress_index']),
                str(r.get('level', 'LOW')),
                float(r.get('daisee', 0.0)),
                float(r.get('behavior', 0.0)),
            )
            for r in history
        ]
        conn.executemany("""
            INSERT INTO samples
                (session_id, ts, stress_index, level, daisee, behavior)
            VALUES (?, ?, ?, ?, ?, ?)
        """, sample_rows)

        conn.commit()
        return session_id
    finally:
        conn.close()


def load_recent_sessions(n: int = 10) -> List[Dict]:
    """Return the most recent N session summaries, newest first."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT * FROM sessions
            ORDER BY id DESC
            LIMIT ?
        """, (n,)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def load_session_samples(session_id: int) -> List[Dict]:
    """
    Load all per-frame samples for a session in chronological order.
    Returns dicts shaped like the in-memory `st.session_state.history`,
    so they pass straight into the existing render functions.
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT ts, stress_index, level, daisee, behavior
            FROM samples
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,)).fetchall()
        return [
            {
                'ts':           datetime.fromisoformat(r['ts']),
                'stress_index': float(r['stress_index']),
                'level':        str(r['level']),
                'daisee':       float(r['daisee']),
                'behavior':     float(r['behavior']),
            }
            for r in rows
        ]
    finally:
        conn.close()


def session_count() -> int:
    """Total number of saved sessions in the DB."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def clear_all_sessions():
    """Delete every saved session and all its samples."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM samples")
        conn.execute("DELETE FROM sessions")
        conn.commit()
    finally:
        conn.close()


def delete_session(session_id: int):
    """Delete a specific session and its samples."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM samples  WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?",         (session_id,))
        conn.commit()
    finally:
        conn.close()