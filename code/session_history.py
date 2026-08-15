"""
Session History DB (v3 - task-aware sessions)
==============================================
Persists completed sessions to a local SQLite database. Each session has
a one-row summary in the `sessions` table plus N per-frame rows in the
`samples` table (one per recorded frame). Per-frame storage allows us
to re-render any past session's timeline graph on demand.

v3 adds a `task_mode` column to the sessions table so we know which
context ('Default' / 'Studying' / 'Hospital' / 'Driving') the session
was recorded under. Old v2 DBs are migrated in place via ALTER TABLE.

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
        pct_high        REAL,
        task_mode       TEXT     'Default' | 'Studying' | 'Hospital' | 'Driving'
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
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import config

DB_PATH = Path(config.DB_PATH)

# The seven behavioural signals, stored per frame so the self-retraining loop
# has real inputs to learn from (the fused index alone is not trainable data).
SIGNAL_COLUMNS = ['eye_strain', 'blink_stress', 'brow_tension', 'jaw_tension',
                  'head_instability', 'neck_strain', 'slouch']


def _get_conn():
    """Open a fresh connection. Caller is responsible for closing it."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables if they do not exist. Safe to call repeatedly.

    Also migrates legacy DBs in place by adding the task_mode column to
    the sessions table if missing (v2 -> v3).
    """
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
                pct_high        REAL NOT NULL,
                task_mode       TEXT NOT NULL DEFAULT 'Default'
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

        # Migration: add task_mode column to existing v2 sessions tables
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if 'task_mode' not in existing_cols:
            conn.execute(
                "ALTER TABLE sessions "
                "ADD COLUMN task_mode TEXT NOT NULL DEFAULT 'Default'"
            )

        # Migration v3 -> v4: self_report holds the user's end-of-session
        # answer to "how stressed did you actually feel?" (1-5), NULL when
        # skipped. This is the label source for Phase 3's continual
        # adaptation - the system's predictions are NOT ground truth, the
        # user's own report is.
        if 'self_report' not in existing_cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN self_report INTEGER"
            )

        # Migration v4 -> v5, part 1: per-signal columns on samples. Without
        # these, live sessions store only the FUSED index and there is nothing
        # for the self-retraining loop to learn from. NULL on legacy rows.
        sample_cols = {row[1] for row in
                       conn.execute("PRAGMA table_info(samples)").fetchall()}
        for col in SIGNAL_COLUMNS:
            if col not in sample_cols:
                conn.execute(f"ALTER TABLE samples ADD COLUMN {col} REAL")

        # Migration v4 -> v5, part 2: the intervention agent's action log.
        # One row per intervention shown; outcome is filled in once the
        # evaluation window has elapsed (did the index actually drop?).
        # This table is what lets the agent LEARN which interventions work.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interventions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER,
                ts              TEXT NOT NULL,
                kind            TEXT NOT NULL,
                trigger_signal  TEXT,
                index_before    REAL NOT NULL,
                index_after     REAL,
                outcome         TEXT NOT NULL DEFAULT 'pending',
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        conn.commit()
    finally:
        conn.close()


def save_session(history: List[Dict], task_mode: str = 'Default') -> Optional[int]:
    """
    Save a completed session to the DB (summary + every frame).

    Args:
        history: list of records, each with keys:
            'ts' (datetime), 'stress_index' (float), 'level' (str),
            'daisee' (float), 'behavior' (float)
        task_mode: which context the session was recorded under -
            'Default' / 'Studying' / 'Hospital' / 'Driving'.

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
                pct_low, pct_moderate, pct_high,
                task_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            started_at.isoformat(),
            ended_at.isoformat(),
            duration, n,
            mean_stress, max_stress,
            pct_low, pct_moderate, pct_high,
            str(task_mode),
        ))
        session_id = cursor.lastrowid

        # Insert all samples in one batch, per-signal values included when
        # the record carries them (live sessions do; legacy replays may not).
        sig_cols = ", ".join(SIGNAL_COLUMNS)
        sig_marks = ", ".join("?" for _ in SIGNAL_COLUMNS)
        sample_rows = [
            (
                session_id,
                r['ts'].isoformat(),
                float(r['stress_index']),
                str(r.get('level', 'LOW')),
                float(r.get('daisee', 0.0)),
                float(r.get('behavior', 0.0)),
                *[None if r.get(c) is None else float(r[c])
                  for c in SIGNAL_COLUMNS],
            )
            for r in history
        ]
        conn.executemany(f"""
            INSERT INTO samples
                (session_id, ts, stress_index, level, daisee, behavior,
                 {sig_cols})
            VALUES (?, ?, ?, ?, ?, ?, {sig_marks})
        """, sample_rows)

        conn.commit()
        return session_id
    finally:
        conn.close()


def set_self_report(session_id: int, rating: Optional[int]):
    """
    Record the user's own end-of-session stress rating (1-5), or clear it
    with None. This is Phase 3's label source: the system's prediction is
    not ground truth - the user's report is what the continual-adaptation
    loop trains against.
    """
    if rating is not None:
        rating = int(rating)
        if not 1 <= rating <= 5:
            raise ValueError(f"self_report must be 1-5 or None, got {rating}")
    conn = _get_conn()
    try:
        conn.execute("UPDATE sessions SET self_report = ? WHERE id = ?",
                     (rating, session_id))
        conn.commit()
    finally:
        conn.close()


def labeled_sessions(min_samples: int = 10) -> List[Dict]:
    """Sessions that have a self-report, for the retraining loop."""
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT * FROM sessions
            WHERE self_report IS NOT NULL AND sample_count >= ?
            ORDER BY id ASC
        """, (min_samples,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_interventions(session_id: Optional[int], records: List[Dict]):
    """
    Persist the intervention agent's action log for a session.
    Each record: {'ts': datetime, 'kind', 'trigger_signal',
                  'index_before', 'index_after' (or None), 'outcome'}.
    """
    if not records:
        return
    conn = _get_conn()
    try:
        conn.executemany("""
            INSERT INTO interventions
                (session_id, ts, kind, trigger_signal,
                 index_before, index_after, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (session_id, r['ts'].isoformat(), str(r['kind']),
             r.get('trigger_signal'),
             float(r['index_before']),
             None if r.get('index_after') is None else float(r['index_after']),
             str(r.get('outcome', 'pending')))
            for r in records
        ])
        conn.commit()
    finally:
        conn.close()


def intervention_stats() -> Dict[str, Dict[str, int]]:
    """
    Per-kind helped/failed counts across all history. This is the agent's
    memory: what has actually worked on this install, learned from measured
    outcomes rather than assumed.
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT kind, outcome, COUNT(*) FROM interventions
            WHERE outcome IN ('helped', 'no_effect')
            GROUP BY kind, outcome
        """).fetchall()
    finally:
        conn.close()
    stats: Dict[str, Dict[str, int]] = {}
    for kind, outcome, n in rows:
        stats.setdefault(kind, {'helped': 0, 'no_effect': 0})[outcome] = n
    return stats


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