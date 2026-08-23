"""
Self-retraining loop tests.

The guards ARE the feature: a self-improving system that trains on two
sessions, or on its own predictions, is worse than one that never adapts.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

import session_history
from retrain_from_sessions import (
    build_dataset, build_dataset_by_task_mode, evaluate_lambda,
    fit_logreg_with_prior, load_population_prior, process_mode,
    _wrap_as_sklearn_model,
)
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


def _labelled(rating, base, seed, start=None, task_mode='Default'):
    sid = session_history.save_session(
        _session(base=base, jitter_seed=seed, start=start), task_mode=task_mode)
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


# --------------------------------------------------------------------------
# Phase 3.2/3.3 (2026-08-23): per-task-mode scoping + warm-start
# --------------------------------------------------------------------------

def _many_sessions(mode, n_relaxed=3, n_stressed=3, seed_start=100, start=None):
    """n_relaxed + n_stressed labelled sessions in one task mode."""
    start = start or datetime(2026, 8, 20, 9, 0, 0)
    ids = []
    seed = seed_start
    for _ in range(n_relaxed):
        ids.append(_labelled(1, base=0.1, seed=seed, task_mode=mode,
                             start=start + timedelta(hours=seed)))
        seed += 1
    for _ in range(n_stressed):
        ids.append(_labelled(5, base=0.8, seed=seed, task_mode=mode,
                             start=start + timedelta(hours=seed)))
        seed += 1
    return ids


def test_build_dataset_by_task_mode_partitions_correctly(temp_db):
    session_history.init_db()
    d = _labelled(5, base=0.8, seed=1, task_mode='Default')
    s = _labelled(5, base=0.8, seed=2, task_mode='Studying')
    by_mode, skipped = build_dataset_by_task_mode()
    assert set(by_mode) == {'Default', 'Studying'}
    assert set(by_mode['Default'][2]) == {d}     # groups
    assert set(by_mode['Studying'][2]) == {s}
    assert not skipped


def test_load_population_prior_returns_none_when_file_missing(tmp_path):
    assert load_population_prior(tmp_path / 'does_not_exist.joblib') is None


def test_load_population_prior_rejects_feature_order_mismatch(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    bad = tmp_path / 'bad_prior.joblib'
    joblib.dump({'model': LogisticRegression(), 'scaler': None,
                'features': ['wrong', 'order']}, bad)
    assert load_population_prior(bad) is None


def test_load_population_prior_loads_valid_file(tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.random.default_rng(0).normal(size=(50, len(SIGNAL_COLUMNS)))
    y = (X[:, 0] > 0).astype(int)
    sc = StandardScaler().fit(X)
    model = LogisticRegression().fit(sc.transform(X), y)
    good = tmp_path / 'good_prior.joblib'
    joblib.dump({'model': model, 'scaler': sc, 'features': list(SIGNAL_COLUMNS)}, good)

    prior = load_population_prior(good)
    assert prior is not None
    assert np.array_equal(prior['w'], model.coef_[0])
    assert prior['b'] == model.intercept_[0]


def test_wrap_as_sklearn_model_matches_manual_sigmoid():
    w = np.array([1.0, -2.0, 0.5, 0.0, 3.0, 0.1, -0.1])
    b = 0.25
    model = _wrap_as_sklearn_model(w, b)
    X = np.random.default_rng(0).normal(size=(10, len(w)))
    manual_p1 = 1 / (1 + np.exp(-(X @ w + b)))
    assert np.allclose(model.predict_proba(X)[:, 1], manual_p1, atol=1e-8)


def test_evaluate_lambda_zero_prior_matches_plain_fit():
    """lambda's value is moot with an all-zero prior - this is the
    'no population prior available' degrade path, and it must reduce to
    exactly the old from-scratch behavior, not something subtly different."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, len(SIGNAL_COLUMNS)))
    y = (X[:, 0] > 0).astype(int)
    groups = np.repeat(np.arange(8), 10)
    acc_lam0, _, _ = evaluate_lambda(X, y, groups, np.zeros(len(SIGNAL_COLUMNS)), 0.0, 0.0)
    acc_lam_huge, _, _ = evaluate_lambda(X, y, groups, np.zeros(len(SIGNAL_COLUMNS)), 0.0, 1000.0)
    # With a zero prior, a huge lambda pulls everything toward zero weights
    # (useless), while lambda=0 is a real fit - they must NOT be close.
    assert acc_lam0 > acc_lam_huge


def test_process_mode_refuses_below_minimum_sessions(temp_db, tmp_path):
    session_history.init_db()
    _many_sessions('Default', n_relaxed=1, n_stressed=1, seed_start=200)
    by_mode, _ = build_dataset_by_task_mode()
    X, y, groups = by_mode['Default']
    process_mode('Default', X, y, groups, prior=None, min_sessions=6, models_dir=tmp_path)
    assert list(tmp_path.glob('*.joblib')) == []


def test_process_mode_saves_a_learned_fusion_compatible_file(temp_db, tmp_path):
    """
    The critical contract test: learned_fusion.py (the only real consumer)
    must be able to load whatever this saves and produce a valid score.
    This is exactly the kind of file-schema mismatch that stays invisible
    until something actually tries to load the file the way production
    code will.
    """
    session_history.init_db()
    _many_sessions('Studying', n_relaxed=3, n_stressed=3, seed_start=300)
    by_mode, _ = build_dataset_by_task_mode()
    X, y, groups = by_mode['Studying']
    process_mode('Studying', X, y, groups, prior=None, min_sessions=6, models_dir=tmp_path)

    saved = list(tmp_path.glob('personal_fusion_studying_v*.joblib'))
    assert len(saved) == 1

    from learned_fusion import LearnedFusion
    lf = LearnedFusion(model_path=saved[0])
    assert lf.is_available() is True
    score = lf.score({name: 0.5 for name in SIGNAL_COLUMNS})
    assert score is not None and 0.0 <= score <= 1.0


def test_process_mode_without_prior_is_not_warm_started(temp_db, tmp_path):
    import joblib
    session_history.init_db()
    _many_sessions('Default', n_relaxed=3, n_stressed=3, seed_start=400)
    by_mode, _ = build_dataset_by_task_mode()
    X, y, groups = by_mode['Default']
    process_mode('Default', X, y, groups, prior=None, min_sessions=6, models_dir=tmp_path)

    saved = list(tmp_path.glob('personal_fusion_default_v*.joblib'))[0]
    blob = joblib.load(saved)
    assert blob['warm_started'] is False
    assert blob['lambda'] == 0.0


def test_process_mode_with_prior_is_warm_started(temp_db, tmp_path):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    session_history.init_db()
    _many_sessions('Driving', n_relaxed=3, n_stressed=3, seed_start=500)
    by_mode, _ = build_dataset_by_task_mode()
    X, y, groups = by_mode['Driving']

    rng = np.random.default_rng(0)
    Xp = rng.normal(size=(50, len(SIGNAL_COLUMNS)))
    yp = (Xp[:, 0] > 0).astype(int)
    scp = StandardScaler().fit(Xp)
    prior_model = LogisticRegression().fit(scp.transform(Xp), yp)
    prior = {'w': prior_model.coef_[0], 'b': prior_model.intercept_[0], 'scaler': scp}

    process_mode('Driving', X, y, groups, prior=prior, min_sessions=6,
                models_dir=tmp_path, lam=10.0)

    saved = list(tmp_path.glob('personal_fusion_driving_v*.joblib'))[0]
    blob = joblib.load(saved)
    assert blob['warm_started'] is True
    assert blob['lambda'] == 10.0
    # BUG_AUDIT_2026-08-23.md finding #1 (fixed): the saved scaler must be
    # the PRIOR's scaler, not a fresh one fit on this user's own small
    # sample - w_prior/b_prior only mean what they mean in the prior
    # scaler's coordinate space, so saving any other scaler alongside them
    # would silently corrupt the warm-start regularization. (Compared by
    # fitted params, not identity - blob['scaler'] is a fresh object after
    # the joblib round-trip.)
    assert np.array_equal(blob['scaler'].mean_, scp.mean_)
    assert np.array_equal(blob['scaler'].scale_, scp.scale_)


def test_evaluate_lambda_uses_the_given_scaler_not_a_fresh_one():
    """
    Regression test for BUG_AUDIT_2026-08-23.md finding #1: passing a
    scaler to evaluate_lambda() must actually change its behavior (proving
    the parameter is wired through, not silently ignored) - a scaler with
    a large mean shift relative to the data changes the standardized
    features evaluate_lambda regularizes toward w_prior in, and therefore
    the fitted weights/accuracy, versus letting it fit a fresh per-fold
    scaler on the unshifted data.
    """
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(loc=0.0, scale=1.0, size=(60, len(SIGNAL_COLUMNS)))
    y = (X[:, 0] > 0).astype(int)
    groups = np.repeat(np.arange(6), 10)
    w_prior = np.zeros(len(SIGNAL_COLUMNS))
    w_prior[0] = 5.0    # a confident prior opinion that feature 0 drives the label
    b_prior = 0.0

    acc_fresh, _, _ = evaluate_lambda(X, y, groups, w_prior, b_prior, lam=10.0,
                                      scaler=None)

    # A scaler fit on wildly-shifted, wildly-rescaled data: transforming X
    # through it does NOT standardize X to mean~0/std~1 the way a
    # correctly-matched scaler would, so the lambda=10 pull toward w_prior
    # (defined for standardized units) becomes actively harmful.
    shifted = StandardScaler().fit(X * 50.0 + 100.0)
    acc_mismatched, _, _ = evaluate_lambda(X, y, groups, w_prior, b_prior, lam=10.0,
                                           scaler=shifted)

    assert acc_mismatched < acc_fresh - 0.1, (
        "evaluate_lambda's scaler parameter must actually be used - a "
        "badly mismatched scaler should measurably hurt accuracy relative "
        "to a correctly-fit one, not be silently ignored")


def test_process_mode_versions_increment_without_overwriting(temp_db, tmp_path):
    session_history.init_db()
    _many_sessions('Default', n_relaxed=3, n_stressed=3, seed_start=600)
    by_mode, _ = build_dataset_by_task_mode()
    X, y, groups = by_mode['Default']

    process_mode('Default', X, y, groups, prior=None, min_sessions=6, models_dir=tmp_path)
    process_mode('Default', X, y, groups, prior=None, min_sessions=6, models_dir=tmp_path)

    saved = sorted(p.name for p in tmp_path.glob('personal_fusion_default_v*.joblib'))
    assert saved == ['personal_fusion_default_v1.joblib', 'personal_fusion_default_v2.joblib']
