"""
Session History DB (v6 - multi-user profiles)
================================================
Persists completed sessions to a local SQLite database. Each session has
a one-row summary in the `sessions` table plus N per-frame rows in the
`samples` table (one per recorded frame). Per-frame storage allows us
to re-render any past session's timeline graph on demand.

v3 adds a `task_mode` column to the sessions table so we know which
context ('Default' / 'Studying' / 'Hospital' / 'Driving') the session
was recorded under. Old v2 DBs are migrated in place via ALTER TABLE.

v6 (Phase 1 prerequisite for team recordings) adds a `users` table and
`sessions.user_id`, so multiple people can each get their own persisted
calibration baseline instead of sharing the single process-wide one
BehavioralFeatureEngine has always used. The engine itself is unchanged -
app.py swaps `engine.baseline` for the active user's saved one on
selection, the same mechanism "Reset baseline" already used to blow it
back to generic defaults. Legacy sessions (recorded before this existed)
keep `user_id = NULL`, meaning "the original single-user PP1 data" - never
silently reassigned to any profile.

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
        user_id         INTEGER  FK -> users.id, NULL for anonymous/legacy sessions
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
    users (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        name            TEXT     UNIQUE,
        created_at      TEXT     ISO8601 timestamp,
        baseline_json   TEXT     JSON personal calibration baseline, NULL
                                 until this user's first completed calibration
    )
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import config
from temporal_model import TEMPORAL_FEATURE_ORDER

DB_PATH = Path(config.DB_PATH)

# The seven behavioural signals, stored per frame so the self-retraining loop
# has real inputs to learn from (the fused index alone is not trainable data).
SIGNAL_COLUMNS = ['eye_strain', 'blink_stress', 'brow_tension', 'jaw_tension',
                  'head_instability', 'neck_strain', 'slouch']

# The 3 extra per-frame values temporal_model.TEMPORAL_FEATURE_ORDER needs
# beyond SIGNAL_COLUMNS - raw EAR/blink-rate diagnostics the derived signals
# above partially discard. Kept as a separate list (not merged into
# SIGNAL_COLUMNS) since these are GRU-training inputs, not fusion-weighted
# behavioral signals - a different purpose, even though both are per-frame
# floats living in the same `samples` table. Derived from
# TEMPORAL_FEATURE_ORDER (not hand-typed) so a future change to that
# vocabulary is picked up here automatically instead of silently drifting
# out of sync with what the GRU retrain script actually expects.
TEMPORAL_EXTRA_COLUMNS = [c for c in TEMPORAL_FEATURE_ORDER if c not in SIGNAL_COLUMNS]


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

        # Migration v5 -> v6: multi-user profiles (Phase 1 prerequisite for
        # team recordings). users.baseline_json holds a JSON-serialized
        # personal baseline (behavioral + DaiSEE), NULL until that user's
        # first completed calibration. sessions.user_id is nullable so
        # existing single-user history is never silently reassigned.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL UNIQUE,
                created_at      TEXT NOT NULL,
                baseline_json   TEXT
            )
        """)
        if 'user_id' not in existing_cols:
            conn.execute(
                "ALTER TABLE sessions ADD COLUMN user_id INTEGER "
                "REFERENCES users(id)"
            )

        # Migration v6 -> v7: the 3 extra per-frame values temporal_model.py's
        # GRU needs (TEMPORAL_FEATURE_ORDER's last 3 dims) beyond the 7
        # SIGNAL_COLUMNS. BehavioralFeatureEngine.extract() already computes
        # these every frame, but until this migration they were used
        # transiently and never persisted - meaning no live recording, no
        # matter how many get made, could ever be turned into a real GRU
        # training sequence. Same nullable-on-legacy-rows pattern as v5a.
        sample_cols = {row[1] for row in
                       conn.execute("PRAGMA table_info(samples)").fetchall()}
        for col in TEMPORAL_EXTRA_COLUMNS:
            if col not in sample_cols:
                conn.execute(f"ALTER TABLE samples ADD COLUMN {col} REAL")

        # Migration v7 -> v8: samples.daisee REAL NOT NULL -> REAL (nullable).
        # app.py stored `result.get('daisee_raw') or 0.0`, which collapsed
        # two different things into the same value: the CNN branch
        # genuinely scoring near-zero, and the CNN branch not running at
        # all this frame (it only runs every CNN_EVERY_N_FRAMES frames).
        # datasets/feature_traces/*.csv already preserves this distinction
        # (an empty field for a skipped frame, read back as None by
        # ablation_study.load_trace) - this migration brings the live DB up
        # to the same standard. SQLite can't drop a NOT NULL constraint via
        # ALTER TABLE, so the table is rebuilt; existing rows keep whatever
        # 0.0 they already have (the ambiguity in already-recorded data is
        # not recoverable), but every row saved after this migration
        # correctly distinguishes "CNN said 0" from "CNN didn't run."
        daisee_notnull = next(
            (row[3] for row in conn.execute("PRAGMA table_info(samples)").fetchall()
             if row[1] == 'daisee'), 0)
        if daisee_notnull:
            current_cols = [row[1] for row in
                           conn.execute("PRAGMA table_info(samples)").fetchall()]
            col_list = ", ".join(current_cols)
            conn.execute(f"""
                CREATE TABLE samples_v8 (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id         INTEGER NOT NULL,
                    ts                 TEXT NOT NULL,
                    stress_index       REAL NOT NULL,
                    level              TEXT NOT NULL,
                    daisee             REAL,
                    behavior           REAL NOT NULL,
                    eye_strain         REAL,
                    blink_stress       REAL,
                    brow_tension       REAL,
                    jaw_tension        REAL,
                    head_instability   REAL,
                    neck_strain        REAL,
                    slouch             REAL,
                    ear                REAL,
                    ear_asymmetry      REAL,
                    blinks_per_minute  REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            conn.execute(f"INSERT INTO samples_v8 ({col_list}) SELECT {col_list} FROM samples")
            conn.execute("DROP TABLE samples")
            conn.execute("ALTER TABLE samples_v8 RENAME TO samples")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id)")

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Users / per-user calibration baselines
# ---------------------------------------------------------------------------
def create_user(name: str) -> int:
    """
    Create a new user profile. Raises ValueError if the name is blank or
    already taken - callers are UI code that should surface this, not a
    silent no-op that leaves the user thinking a profile was created.
    """
    name = name.strip()
    if not name:
        raise ValueError("name must not be blank")
    conn = _get_conn()
    try:
        try:
            cursor = conn.execute(
                "INSERT INTO users (name, created_at) VALUES (?, ?)",
                (name, datetime.now().isoformat()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"a user named {name!r} already exists")
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def list_users() -> List[Dict]:
    """All user profiles, alphabetical (case-insensitive)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, name, created_at, "
            "(baseline_json IS NOT NULL) AS has_baseline "
            "FROM users ORDER BY name COLLATE NOCASE"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_user_baseline(user_id: int, baseline: Dict):
    """Persist a user's calibration baseline (a plain JSON-able dict - see
    app.py for how it's assembled from BehavioralBaseline + the DaiSEE
    baseline value)."""
    conn = _get_conn()
    try:
        conn.execute("UPDATE users SET baseline_json = ? WHERE id = ?",
                     (json.dumps(baseline), user_id))
        conn.commit()
    finally:
        conn.close()


def load_user_baseline(user_id: int) -> Optional[Dict]:
    """The user's saved baseline dict, or None if they have never
    completed a calibration."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT baseline_json FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row is None or row['baseline_json'] is None:
            return None
        return json.loads(row['baseline_json'])
    finally:
        conn.close()


def summarize_history(history: List[Dict]) -> Dict:
    """
    Reduce a per-frame history list to the same summary stats stored in the
    `sessions` table row (duration/mean/max/pct_low/pct_moderate/pct_high/
    sample_count). Pure function, no DB access - shared by save_session()
    and session_report.py so the two never drift on what "mean stress" or
    "% time high" means.
    """
    if not history:
        raise ValueError("history must not be empty")

    started_at = history[0]['ts']
    ended_at   = history[-1]['ts']
    n          = len(history)

    stress_vals = [r['stress_index'] for r in history]
    mean_stress = sum(stress_vals) / n
    max_stress  = max(stress_vals)

    counts = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0}
    for r in history:
        # `.get(..., 'LOW')` only substitutes the default for a MISSING
        # key, not a present None - a record with level=None would count
        # toward `n` but no bucket, so the percentages below wouldn't sum
        # to 100%. `or` degrades both cases the same way.
        lvl = r.get('level') or 'LOW'
        counts[lvl] = counts.get(lvl, 0) + 1

    return {
        'duration_sec': (ended_at - started_at).total_seconds(),
        'sample_count': n,
        'mean_stress': mean_stress,
        'max_stress': max_stress,
        'pct_low':      100.0 * counts['LOW']      / n,
        'pct_moderate': 100.0 * counts['MODERATE'] / n,
        'pct_high':     100.0 * counts['HIGH']     / n,
    }


def save_session(history: List[Dict], task_mode: str = 'Default',
                 user_id: Optional[int] = None) -> Optional[int]:
    """
    Save a completed session to the DB (summary + every frame).

    Args:
        history: list of records, each with keys:
            'ts' (datetime), 'stress_index' (float), 'level' (str),
            'daisee' (float), 'behavior' (float)
        task_mode: which context the session was recorded under -
            'Default' / 'Studying' / 'Hospital' / 'Driving'.
        user_id: the active user profile's id, or None for anonymous /
            single-user usage (unchanged pre-v6 behavior).

    Returns:
        The new session's id, or None if history was empty.
    """
    if not history:
        return None

    started_at = history[0]['ts']
    ended_at   = history[-1]['ts']
    stats = summarize_history(history)
    duration     = stats['duration_sec']
    n            = stats['sample_count']
    mean_stress  = stats['mean_stress']
    max_stress   = stats['max_stress']
    pct_low      = stats['pct_low']
    pct_moderate = stats['pct_moderate']
    pct_high     = stats['pct_high']

    conn = _get_conn()
    try:
        cursor = conn.execute("""
            INSERT INTO sessions (
                started_at, ended_at, duration_sec, sample_count,
                mean_stress, max_stress,
                pct_low, pct_moderate, pct_high,
                task_mode, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            started_at.isoformat(),
            ended_at.isoformat(),
            duration, n,
            mean_stress, max_stress,
            pct_low, pct_moderate, pct_high,
            str(task_mode),
            user_id,
        ))
        session_id = cursor.lastrowid

        # Insert all samples in one batch, per-signal values included when
        # the record carries them (live sessions do; legacy replays may not).
        extra_cols = SIGNAL_COLUMNS + TEMPORAL_EXTRA_COLUMNS
        sig_cols = ", ".join(extra_cols)
        sig_marks = ", ".join("?" for _ in extra_cols)
        sample_rows = [
            (
                session_id,
                r['ts'].isoformat(),
                float(r['stress_index']),
                str(r.get('level', 'LOW')),
                None if r.get('daisee') is None else float(r['daisee']),
                float(r.get('behavior', 0.0)),
                *[None if r.get(c) is None else float(r[c])
                  for c in extra_cols],
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


def intervention_stats_by_mode() -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Per-(task_mode, kind) helped/failed counts, via a join against
    sessions - no schema change needed, sessions.task_mode already exists.
    {mode: {kind: {'helped':, 'no_effect':}}}.

    An intervention whose session_id is NULL or points at a deleted
    session is excluded here (its context is unknown) but still counted
    by the global, mode-blind intervention_stats() above - this function
    only adds a context-aware VIEW on the same data, it doesn't change
    what counts as a measured outcome.
    """
    conn = _get_conn()
    try:
        rows = conn.execute("""
            SELECT s.task_mode, i.kind, i.outcome, COUNT(*)
            FROM interventions i
            JOIN sessions s ON s.id = i.session_id
            WHERE i.outcome IN ('helped', 'no_effect')
            GROUP BY s.task_mode, i.kind, i.outcome
        """).fetchall()
    finally:
        conn.close()
    stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    for mode, kind, outcome, n in rows:
        stats.setdefault(mode, {}).setdefault(
            kind, {'helped': 0, 'no_effect': 0})[outcome] = n
    return stats


def load_recent_sessions(n: int = 10) -> List[Dict]:
    """
    Return the most recent N session summaries, newest first.

    Deliberately NOT filtered by user_id: this is one shared history log
    across every profile on the install, by design - only per-user
    calibration baselines (users.baseline_json, see save_user_baseline /
    load_user_baseline above) are actually separated per person. The
    sidebar's "Past sessions" list is meant to show everyone's recordings
    on this install, not just the active profile's. If that ever needs to
    change (e.g. for multi-person team recordings where users should not
    see each other's history), filter explicitly by the caller passing the
    active user_id - don't assume this is an oversight to "fix" silently.
    """
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
                'daisee':       None if r['daisee'] is None else float(r['daisee']),
                'behavior':     float(r['behavior']),
            }
            for r in rows
        ]
    finally:
        conn.close()


def load_session_signal_rows(session_id: int) -> List[Dict]:
    """
    Load the per-frame BEHAVIOR SIGNAL values for a session (schema v5a's
    columns: eye_strain, blink_stress, brow_tension, jaw_tension,
    head_instability, neck_strain, slouch) - not returned by
    load_session_samples() above, which only carries the fused-score
    columns. This is what a meta-learner retrain needs as its feature
    vector; NULL on any legacy row recorded before v5a.
    """
    conn = _get_conn()
    try:
        rows = conn.execute(f"""
            SELECT {', '.join(SIGNAL_COLUMNS)}
            FROM samples
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_session_temporal_rows(session_id: int) -> List[Dict]:
    """
    Load all 10 of temporal_model.TEMPORAL_FEATURE_ORDER's per-frame values
    for a session (SIGNAL_COLUMNS + TEMPORAL_EXTRA_COLUMNS, schema v7) -
    what a GRU retrain/fine-tune needs to build a real sequence from a live
    recording. NULL on any row recorded before v7 (ear/ear_asymmetry/
    blinks_per_minute did not exist as columns yet).
    """
    conn = _get_conn()
    try:
        cols = SIGNAL_COLUMNS + TEMPORAL_EXTRA_COLUMNS
        rows = conn.execute(f"""
            SELECT {', '.join(cols)}
            FROM samples
            WHERE session_id = ?
            ORDER BY id ASC
        """, (session_id,)).fetchall()
        return [dict(r) for r in rows]
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
        # interventions.session_id has no ON DELETE CASCADE, so it must be
        # cleared before sessions or PRAGMA foreign_keys=ON (set in
        # _get_conn) raises IntegrityError the moment any session has a
        # logged intervention - which any live session with the agent
        # enabled can produce.
        conn.execute("DELETE FROM interventions")
        conn.execute("DELETE FROM samples")
        conn.execute("DELETE FROM sessions")
        conn.commit()
    finally:
        conn.close()


def delete_session(session_id: int):
    """Delete a specific session and its samples."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM interventions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM samples       WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions      WHERE id = ?",         (session_id,))
        conn.commit()
    finally:
        conn.close()