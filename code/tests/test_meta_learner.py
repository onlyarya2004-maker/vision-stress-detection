"""
Learned-fusion tests.

The point of the meta-learner is that it can express things the hand-picked
weights cannot - in particular a NEGATIVE weight for a signal that is
anti-correlated with stress. config.BEHAVIOR_WEIGHTS are all positive by
construction, so an inverted signal actively subtracts from accuracy there.

These tests use synthetic traces where the right answer is known, so a wrong
implementation is distinguishable from genuinely hard data.
"""

import csv

import numpy as np
import pytest

import config
from train_meta_learner import (
    SIGNALS, handtuned_predictions, handtuned_scores, load_traces,
)


def _write_trace(directory, subject, task, rows):
    """Write a feature CSV in the layout evaluate_video_file.py produces."""
    from evaluate_video_file import FEATURE_COLUMNS
    path = directory / f"features_vid_s{subject}_T{task}.csv"
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)
        w.writeheader()
        for r in rows:
            full = {c: 0.0 for c in FEATURE_COLUMNS}
            full.update(r)
            full['level'] = 'LOW'
            w.writerow(full)
    return path


def _rows(n, seed=0, **signal_means):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        row = {'t': i / 30.0}
        for s in SIGNALS:
            mu = signal_means.get(s, 0.1)
            row[s] = float(np.clip(mu + rng.normal(0, 0.05), 0, 1))
        out.append(row)
    return out


@pytest.fixture
def synthetic_traces(tmp_path):
    """
    Three subjects. brow_tension is INVERTED (higher at rest) and carries the
    largest hand-tuned weight, 0.30. slouch is the genuine signal but carries
    only 0.10. This is the shape the real UBFC-Phys data showed.
    """
    for subj in (1, 2, 3):
        _write_trace(tmp_path, subj, 1,          # T1 = rest
                     _rows(200, seed=subj, brow_tension=0.8, slouch=0.05))
        _write_trace(tmp_path, subj, 2,          # T2 = stressed
                     _rows(200, seed=subj + 10, brow_tension=0.1, slouch=0.9))
    return tmp_path


def test_load_traces_labels_tasks_correctly(synthetic_traces):
    X, y, groups, names = load_traces(synthetic_traces)
    assert set(names) == set(SIGNALS)
    assert set(np.unique(y)) == {0, 1}
    assert len(np.unique(groups)) == 3
    # T1 -> relaxed, T2 -> stressed, equal counts here
    assert (y == 0).sum() == (y == 1).sum()


def test_load_traces_groups_by_subject(synthetic_traces):
    _, _, groups, _ = load_traces(synthetic_traces)
    assert set(np.unique(groups)) == {'s1', 's2', 's3'}


def test_handtuned_scores_match_the_config_formula():
    """The baseline must reproduce config's weighting exactly."""
    names = list(SIGNALS)
    X = np.full((1, len(names)), 0.5)
    got = handtuned_scores(X, names)[0]
    w = np.array([config.BEHAVIOR_WEIGHTS[s] for s in SIGNALS])
    expected = np.clip(0.5 * w.sum() / w.sum()
                       * config.BEHAVIOR_AMPLIFICATION, 0, 1)
    assert got == pytest.approx(expected)


def test_handtuned_fusion_fails_on_inverted_data(synthetic_traces):
    """
    With the dominant signal inverted, the hand-picked weights should do
    poorly - that is the premise of the whole exercise. If this ever passes
    easily, the synthetic data no longer reflects the real failure.
    """
    X, y, _, names = load_traces(synthetic_traces)
    acc = (handtuned_predictions(X, names) == y).mean()
    assert acc < 0.75, f"hand-tuned unexpectedly strong ({acc:.2f})"


def test_learned_fusion_beats_handtuned_on_inverted_data(synthetic_traces):
    """
    A learned weighting must recover what the fixed one cannot, and it must
    do so out-of-fold, never having seen the test subject.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    X, y, groups, names = load_traces(synthetic_traces)
    hand_acc = (handtuned_predictions(X, names) == y).mean()

    oof = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, class_weight='balanced')
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    learned_acc = (oof == y).mean()

    assert learned_acc > hand_acc, (
        f"learned {learned_acc:.2f} did not beat hand-tuned {hand_acc:.2f}")


def test_learned_weights_can_be_negative(synthetic_traces):
    """
    The core capability: expressing that a signal points the WRONG way.
    config.BEHAVIOR_WEIGHTS cannot represent this - every entry is positive.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y, _, names = load_traces(synthetic_traces)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=2000, class_weight='balanced')
    m.fit(sc.transform(X), y)

    brow = m.coef_[0][names.index('brow_tension')]
    assert brow < 0, f"inverted signal got a positive coefficient ({brow:+.2f})"
    assert all(w > 0 for w in config.BEHAVIOR_WEIGHTS.values()), (
        "premise check: hand-tuned weights are all positive")


def test_single_subject_cannot_be_loso_evaluated(tmp_path):
    """LOSO needs at least two subjects; one must not silently 'work'."""
    _write_trace(tmp_path, 1, 1, _rows(50, seed=1))
    _write_trace(tmp_path, 1, 2, _rows(50, seed=2))
    _, _, groups, _ = load_traces(tmp_path)
    assert len(np.unique(groups)) == 1
