"""
Tests for app.py's session_analytics_dataframe() (Phase 5.1).

app.py has no other test file in this codebase - it needs a real Streamlit
script-run context for anything touching st.session_state/widgets, which
pytest doesn't provide. session_analytics_dataframe() is the exception:
it's a pure function of session_history + pandas, no st.* calls, so it
imports and runs fine under Streamlit's "bare mode" (a harmless warning,
not a failure - verified before relying on it here). Nothing else from
app.py is tested this way; this is not a general precedent for testing
Streamlit UI code, just this one genuinely pure function.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

import session_history
from app import session_analytics_dataframe
from session_history import SIGNAL_COLUMNS


def _session(n=20, base=0.3, task_mode='Default', start=None, seed=0):
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 8, 1, 9, 0, 0)
    rows = []
    for i in range(n):
        row = {
            'ts': start + timedelta(seconds=i * 0.5),
            'stress_index': float(np.clip(base + rng.normal(0, 0.05), 0, 1)),
            'level': 'LOW', 'daisee': 0.5, 'behavior': base,
        }
        for c in SIGNAL_COLUMNS:
            row[c] = float(np.clip(base + rng.normal(0, 0.05), 0, 1))
        rows.append(row)
    sid = session_history.save_session(rows, task_mode=task_mode)
    return sid


def test_returns_none_for_empty_history(temp_db):
    session_history.init_db()
    assert session_analytics_dataframe() is None


def test_returns_one_row_per_session(temp_db):
    session_history.init_db()
    _session(start=datetime(2026, 8, 1, 9, 0, 0), seed=1)
    _session(start=datetime(2026, 8, 2, 14, 0, 0), seed=2)
    df = session_analytics_dataframe()
    assert len(df) == 2


def test_derived_time_columns_are_correct(temp_db):
    session_history.init_db()
    _session(start=datetime(2026, 8, 3, 15, 30, 0), seed=1)   # a Monday, 15:00 hour
    df = session_analytics_dataframe()
    row = df.iloc[0]
    assert row['hour'] == 15
    assert row['weekday'] == 'Monday'
    assert str(row['date']) == '2026-08-03'


def test_sorted_chronologically_regardless_of_save_order(temp_db):
    session_history.init_db()
    later = _session(start=datetime(2026, 8, 5, 9, 0, 0), seed=1)
    earlier = _session(start=datetime(2026, 8, 1, 9, 0, 0), seed=2)
    df = session_analytics_dataframe()
    assert list(df['id']) == [earlier, later]


def test_task_mode_column_present_for_context_breakdown(temp_db):
    session_history.init_db()
    _session(task_mode='Studying', start=datetime(2026, 8, 1, 9, 0, 0), seed=1)
    _session(task_mode='Driving', start=datetime(2026, 8, 2, 9, 0, 0), seed=2)
    df = session_analytics_dataframe()
    assert set(df['task_mode']) == {'Studying', 'Driving'}


def test_self_report_is_nan_when_not_answered(temp_db):
    session_history.init_db()
    _session(seed=1)
    df = session_analytics_dataframe()
    assert df['self_report'].isna().all()


def test_self_report_present_when_answered(temp_db):
    session_history.init_db()
    sid = _session(seed=1)
    session_history.set_self_report(sid, 4)
    df = session_analytics_dataframe()
    assert df.loc[df['id'] == sid, 'self_report'].iloc[0] == 4
