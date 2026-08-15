"""
Self-retraining loop tests.

The guards ARE the feature: a self-improving system that trains on two
sessions, or on its own predictions, is worse than one that never adapts.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

import session_history
from retrain_from_sessions import build_dataset
from session_history import SIGNAL_COLUMNS


def _session(n=40, base=0.2, jitter_seed=0, start=None):
    """History rows carrying per-signal values (as the live app now records)."""
    rng = np.random.default_rng(jitter_seed)
    start = start or datetime(2026, 8, 16, 9, 0, 0)
    rows = []
    for i in range(n):
        row = {
            'ts': start + timedelta(seconds=i * 0.5),
            'stress_index': base, 'level': 'LOW',
            'daisee': 0.5, 'behavior': base,
        }
        for c in SIGNAL_COLUMNS:
            row[c] = float(np.clip(base + rng.normal(0, 0.05), 0, 1))
        rows.append(row)
    return rows


def _labelled(rating, base, seed, start=None):
    sid = session_history.save_session(_session(base=base, jitter_seed=seed,
                                                start=start))
    session_history.set_self_report(sid, rating)
    return sid


def test_signals_roundtrip_through_samples(temp_db):
    """Schema v5: per-signal values survive save -> DB -> dataset build."""
    session_history.init_db()
    _labelled(5, base=0.8, seed=1)
    X, y, groups, skipped = build_dataset()
    assert X.shape == (40, len(SIGNAL_COLUMNS))
    assert set(y) == {1}
    assert not skipped


def test_ambiguous_ratings_are_excluded(temp_db):
    """A '3 - neutral' is ambiguous by the user's own account: not training data."""
    session_history.init_db()
    _labelled(3, base=0.5, seed=2)
    X, y, groups, skipped = build_dataset()
    assert len(X) == 0
    assert any('ambiguous' in why for _, why in skipped)


def test_sessions_without_signal_data_are_skipped(temp_db):
    """Pre-v5 sessions (no per-signal columns) must be skipped, not zero-filled."""
    session_history.init_db()
    rows = [{'ts': datetime(2026, 8, 1, 9, 0, 0) + timedelta(seconds=i),
             'stress_index': 0.4, 'level': 'LOW', 'daisee': 0.5,
             'behavior': 0.4} for i in range(20)]           # no signals
    sid = session_history.save_session(rows)
    session_history.set_self_report(sid, 5)
    X, y, groups, skipped = build_dataset()
    assert len(X) == 0
    assert any('no per-signal' in why for _, why in skipped)


def test_labels_follow_the_user_not_the_system(temp_db):
    """
    A session the SYSTEM scored low but the USER rated 5 must be labelled
    stressed. The user's report is ground truth; anything else is the model
    grading its own homework.
    """
    session_history.init_db()
    sid = _labelled(5, base=0.1, seed=3)      # system saw calm signals
    X, y, groups, _ = build_dataset()
    assert set(y[groups == sid]) == {1}


def test_dataset_spans_multiple_sessions_with_groups(temp_db):
    session_history.init_db()
    a = _labelled(1, base=0.1, seed=4, start=datetime(2026, 8, 10, 9, 0, 0))
    b = _labelled(5, base=0.8, seed=5, start=datetime(2026, 8, 11, 9, 0, 0))
    X, y, groups, _ = build_dataset()
    assert set(groups) == {a, b}
    assert set(y) == {0, 1}
    # groups are what makes leave-one-SESSION-out possible
    assert (groups == a).sum() == 40 and (groups == b).sum() == 40
