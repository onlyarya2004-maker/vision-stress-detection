"""
correlate_daisee_selfreport.py tests (Phase 6.1).

Uses the shared `temp_db` fixture (conftest.py) so these hit a real,
throwaway SQLite database via session_history.save_session()/
set_self_report() - the same round trip the live app performs - rather
than mocking internals.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

import session_history
from correlate_daisee_selfreport import run_daisee_selfreport_study, render


def _history(n, daisee_value, start=None):
    """Synthetic session history with a constant per-frame DaiSEE score,
    shaped like app.py produces (see test_session_history.py's _history())."""
    start = start or datetime(2026, 9, 1, 10, 0, 0)
    return [
        {
            "ts": start + timedelta(seconds=i * 0.5),
            "stress_index": 0.30,
            "level": "LOW",
            "daisee": daisee_value,
            "behavior": 0.30,
        }
        for i in range(n)
    ]


def _save_labeled_session(daisee_value, self_report, n=20):
    session_history.init_db()
    sid = session_history.save_session(_history(n, daisee_value))
    session_history.set_self_report(sid, self_report)
    return sid


# ---------------------------------------------------------------------------
# run_daisee_selfreport_study()
# ---------------------------------------------------------------------------

def test_no_labeled_sessions_returns_empty_insufficient_result(temp_db):
    results = run_daisee_selfreport_study()
    assert results['rows'] == []
    assert results['insufficient'] is True


def test_single_labeled_session_is_insufficient(temp_db):
    _save_labeled_session(daisee_value=0.7, self_report=5)
    results = run_daisee_selfreport_study()
    assert len(results['rows']) == 1
    assert results['insufficient'] is True


def test_unlabeled_sessions_are_excluded(temp_db):
    """A session with no self-report answered must not count."""
    session_history.init_db()
    session_history.save_session(_history(20, 0.7))     # no set_self_report call
    _save_labeled_session(daisee_value=0.3, self_report=1)
    _save_labeled_session(daisee_value=0.8, self_report=5)
    results = run_daisee_selfreport_study()
    assert len(results['rows']) == 2


def test_perfect_positive_correlation_hand_checked(temp_db):
    """Three sessions with mean_daisee and self_report both strictly
    increasing together must give Pearson r = Spearman r = +1.0 exactly -
    an unambiguous, hand-known answer, not just 'runs without crashing'."""
    _save_labeled_session(daisee_value=0.1, self_report=1)
    _save_labeled_session(daisee_value=0.5, self_report=3)
    _save_labeled_session(daisee_value=0.9, self_report=5)
    results = run_daisee_selfreport_study()
    assert results['insufficient'] is False
    assert results['pearson'] == pytest.approx(1.0)
    assert results['spearman'] == pytest.approx(1.0)


def test_perfect_negative_correlation_hand_checked(temp_db):
    _save_labeled_session(daisee_value=0.9, self_report=1)
    _save_labeled_session(daisee_value=0.5, self_report=3)
    _save_labeled_session(daisee_value=0.1, self_report=5)
    results = run_daisee_selfreport_study()
    assert results['pearson'] == pytest.approx(-1.0)
    assert results['spearman'] == pytest.approx(-1.0)


def test_constant_daisee_gives_nan_correlation_not_a_crash(temp_db):
    """Zero variance in one series makes correlation mathematically
    undefined - this must come back as NaN, not raise or silently print
    a ConstantInputWarning-driven wrong number (see correlate_physiology.py's
    own precedent for this exact guard)."""
    _save_labeled_session(daisee_value=0.5, self_report=1)
    _save_labeled_session(daisee_value=0.5, self_report=5)
    results = run_daisee_selfreport_study()
    assert np.isnan(results['pearson'])
    assert np.isnan(results['spearman'])


def test_binary_agreement_hand_checked(temp_db):
    """4 sessions: 2 clearly agree (predicted matches reported), 1 clearly
    disagrees, 1 is neutral (rating 3) and must be excluded entirely."""
    _save_labeled_session(daisee_value=0.2, self_report=1)   # relaxed/relaxed -> agree
    _save_labeled_session(daisee_value=0.8, self_report=5)   # stressed/stressed -> agree
    _save_labeled_session(daisee_value=0.9, self_report=2)   # stressed/relaxed -> disagree
    _save_labeled_session(daisee_value=0.5, self_report=3)   # neutral -> excluded
    results = run_daisee_selfreport_study()
    assert results['total'] == 3          # the neutral session excluded
    assert results['agree'] == 2


def test_min_samples_filters_short_sessions(temp_db):
    """A session shorter than min_samples must not appear at all - matches
    labeled_sessions()'s own sample_count filter."""
    _save_labeled_session(daisee_value=0.5, self_report=3, n=3)
    _save_labeled_session(daisee_value=0.7, self_report=5, n=20)
    results = run_daisee_selfreport_study(min_samples=10)
    assert len(results['rows']) == 1
    assert results['rows'][0]['n_frames'] == 20


def test_safe_to_run_before_any_session_ever_saved(temp_db):
    """Must not raise sqlite3.OperationalError on a completely fresh
    database where init_db() has never been called - this is the
    standalone-script equivalent of a brand-new checkout."""
    results = run_daisee_selfreport_study()   # temp_db points at a file that doesn't exist yet
    assert results == {'rows': [], 'insufficient': True}


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------

def test_render_no_sessions_message(temp_db):
    out = []
    render(run_daisee_selfreport_study(), out)
    text = "\n".join(out)
    assert "no labeled sessions found" in text


def test_render_insufficient_lists_the_one_session(temp_db):
    _save_labeled_session(daisee_value=0.6, self_report=4)
    out = []
    render(run_daisee_selfreport_study(), out)
    text = "\n".join(out)
    assert "Only 1 labeled session" in text
    assert "self_report=4" in text


def test_render_sufficient_shows_correlations_and_agreement(temp_db):
    _save_labeled_session(daisee_value=0.1, self_report=1)
    _save_labeled_session(daisee_value=0.9, self_report=5)
    out = []
    render(run_daisee_selfreport_study(), out)
    text = "\n".join(out)
    assert "Pearson  r = +1.000" in text
    assert "Binary agreement" in text
    assert "2/2" in text
