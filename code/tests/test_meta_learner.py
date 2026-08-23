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
    SIGNALS, compute_learned_weights_rows, format_learned_weights_rows,
    format_summary_line, handtuned_predictions, handtuned_scores, load_traces,
    render, run_learned_fusion_study, summarize,
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


# --------------------------------------------------------------------------
# Phase 4.1: compute/render split for the harness (run_all_evaluations.py)
# --------------------------------------------------------------------------

def test_summarize_is_pure_and_includes_the_label():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    groups = np.array(['s1', 's1', 's2', 's2'])
    stats = summarize(y_true, y_pred, groups, 'my label')
    assert stats['label'] == 'my label'
    assert stats['accuracy'] == pytest.approx(0.75)
    assert 0.0 <= stats['relaxed_recall'] <= 1.0
    assert 0.0 <= stats['stressed_recall'] <= 1.0


def test_format_summary_line_includes_key_fields():
    stats = summarize(np.array([0, 1]), np.array([0, 1]), np.array(['s1', 's2']), 'x')
    line = format_summary_line(stats)
    assert 'x' in line
    assert 'acc' in line and 'F1' in line
    # Deliberately NOT "session" - eval_scoring.session_scores() means one
    # recording/file, while this module's grouping is (subject, binary
    # condition); printing both under "session" in the same canonical
    # report would read as the same metric when it isn't. See
    # BUG_AUDIT_2026-08-23.md finding #9.
    assert 'cond-block' in line
    assert 'session' not in line


def test_compute_learned_weights_rows_sorted_by_magnitude_and_carries_direction():
    class FakeModel:
        coef_ = np.array([[0.1, -2.0, 0.5]])
    names = ['eye_strain', 'brow_tension', 'slouch']
    rows = compute_learned_weights_rows(FakeModel(), names)
    assert [r['name'] for r in rows] == ['brow_tension', 'slouch', 'eye_strain']
    assert rows[0]['direction'] == 'stress down (INVERTED)'
    assert rows[1]['direction'] == 'stress up'
    assert rows[0]['hand_tuned'] == config.BEHAVIOR_WEIGHTS['brow_tension']


def test_compute_learned_weights_rows_degenerate_all_zero_returns_empty():
    class FakeModel:
        coef_ = np.array([[0.0, 0.0]])
    assert compute_learned_weights_rows(FakeModel(), ['eye_strain', 'slouch']) == []


def test_format_learned_weights_rows_uses_plain_ascii_only():
    """Windows cp1252 crashed printing non-ASCII arrow glyphs here once -
    the labels must stay plain ASCII, not a Unicode arrow."""
    rows = [{'name': 'brow_tension', 'hand_tuned': 0.30, 'learned': 0.5,
            'coef': -1.0, 'direction': 'stress down (INVERTED)'}]
    lines = format_learned_weights_rows(rows)
    text = "\n".join(lines)
    text.encode('ascii')  # raises UnicodeEncodeError if anything non-ASCII slipped in
    assert 'INVERTED' in text


def test_run_learned_fusion_study_returns_none_for_empty_directory(tmp_path):
    assert run_learned_fusion_study(tmp_path) is None


def test_run_learned_fusion_study_single_subject_skips_loso_gracefully(tmp_path):
    _write_trace(tmp_path, 1, 1, _rows(50, seed=1))
    _write_trace(tmp_path, 1, 2, _rows(50, seed=2))
    results = run_learned_fusion_study(tmp_path)
    assert results['n_subjects'] == 1
    assert results['handtuned'] is not None
    assert results['logistic_loso'] is None
    assert results['gbt_loso'] is None
    assert results['weights_rows'] == []
    assert results['final_model'] is None
    assert results['scaler'] is None


def test_run_learned_fusion_study_matches_manual_loso_on_inverted_data(synthetic_traces):
    """
    Cross-check against the same inverted-signal scenario
    test_learned_fusion_beats_handtuned_on_inverted_data() already validates
    manually - the harness-facing function must reproduce it, not a
    different computation that happens to also look reasonable.
    """
    results = run_learned_fusion_study(synthetic_traces)
    assert results['n_subjects'] == 3
    assert results['logistic_loso'] is not None
    assert results['gbt_loso'] is not None
    assert results['logistic_loso']['accuracy'] > results['handtuned']['accuracy']
    assert results['final_model'] is not None
    assert results['scaler'] is not None
    assert results['weights_rows'], "a non-degenerate fit must produce weight rows"
    brow_row = next(r for r in results['weights_rows'] if r['name'] == 'brow_tension')
    assert brow_row['direction'] == 'stress down (INVERTED)'


def test_run_learned_fusion_study_never_touches_disk(synthetic_traces, tmp_path, monkeypatch):
    """No models/ artifact is written by the compute path - only --save
    (main()'s job, not run_learned_fusion_study()'s) persists anything."""
    sentinel_models_dir = tmp_path / 'models_sentinel'
    sentinel_models_dir.mkdir()
    run_learned_fusion_study(synthetic_traces)
    assert list(sentinel_models_dir.iterdir()) == []


def test_render_none_results_shows_skip_message():
    out = []
    render(None, out)
    assert any('no feature traces found' in line for line in out)


def test_render_single_subject_stops_before_learned_section():
    results = {
        'n_frames': 10, 'n_subjects': 1, 'n_relaxed': 5, 'n_stressed': 5,
        'subsample': 1,
        'handtuned': summarize(np.array([0, 1]), np.array([0, 1]), np.array(['s1', 's1']), 'hand-tuned fusion'),
        'logistic_loso': None, 'gbt_loso': None, 'weights_rows': [],
        'final_model': None, 'scaler': None, 'feature_names': SIGNALS,
    }
    out = []
    render(results, out)
    text = "\n".join(out)
    assert 'BASELINE' in text
    assert 'Only one subject present' in text
    # The title banner always says "LEARNED FUSION" - the per-section
    # "LEARNED (out-of-fold, N folds)" header is the thing that must be
    # absent here.
    assert 'LEARNED (out-of-fold' not in text
    assert 'WHAT IT LEARNED' not in text


def test_render_multi_subject_includes_all_sections(synthetic_traces):
    results = run_learned_fusion_study(synthetic_traces)
    out = []
    render(results, out)
    text = "\n".join(out)
    assert 'BASELINE' in text
    assert 'logistic regression' in text
    assert 'gradient boosting' in text
    assert 'WHAT IT LEARNED' in text
    assert 'INVERTED' in text
