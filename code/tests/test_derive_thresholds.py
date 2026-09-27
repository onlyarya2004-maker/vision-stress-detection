"""
derive_thresholds.py tests.

Real UBFC-Phys/team recordings aren't part of the repo, so
run_loso_derivation()/run_sensitivity_sweep() are tested against small
synthetic subject pools engineered so the correct answer is known by
construction, not just "doesn't crash".
"""

import csv

import numpy as np
import pytest

from derive_thresholds import (
    load_ubfc_subjects, load_team_subjects,
    run_loso_derivation, run_sensitivity_sweep, render,
)


# --------------------------------------------------------------------------
# load_ubfc_subjects
# --------------------------------------------------------------------------

def _write_ubfc_trace(path, scores):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'stress_index_smoothed'])
        for i, s in enumerate(scores):
            w.writerow([float(i), s])


def test_load_ubfc_subjects_labels_t1_relaxed_t2_t3_stressed(tmp_path):
    _write_ubfc_trace(tmp_path / 'features_vid_s7_T1.csv', [0.1, 0.2])
    _write_ubfc_trace(tmp_path / 'features_vid_s7_T2.csv', [0.7, 0.8])
    _write_ubfc_trace(tmp_path / 'features_vid_s7_T3.csv', [0.6])

    by_subject = load_ubfc_subjects(tmp_path)
    assert set(by_subject) == {'U07'}
    labels = sorted(l for _, l in by_subject['U07'])
    assert labels == [0, 0, 1, 1, 1]   # 2 relaxed (T1) + 3 stressed (T2/T3)


def test_load_ubfc_subjects_ignores_non_matching_files(tmp_path):
    (tmp_path / 'not_a_trace.csv').write_text("junk")
    by_subject = load_ubfc_subjects(tmp_path)
    assert by_subject == {}


# --------------------------------------------------------------------------
# load_team_subjects
# --------------------------------------------------------------------------

def _write_team_csv(path, scores):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stress_index', 'level'])
        for s in scores:
            w.writerow([s, 'LOW' if s < 0.36 else 'MODERATE'])


def test_load_team_subjects_prefixes_and_labels_by_condition(tmp_path):
    _write_team_csv(tmp_path / 'relaxed_S1_default_2026-09-01.csv', [0.1, 0.15])
    _write_team_csv(tmp_path / 'stressed_S1_default_2026-09-01.csv', [0.7])

    by_subject = load_team_subjects(tmp_path, mode='default')
    assert set(by_subject) == {'TS1'}
    labels = sorted(l for _, l in by_subject['TS1'])
    assert labels == [0, 0, 1]


# --------------------------------------------------------------------------
# run_loso_derivation
# --------------------------------------------------------------------------

def test_run_loso_derivation_returns_none_for_single_subject():
    assert run_loso_derivation({'A': [(0.1, 0), (0.9, 1)]}) is None


def test_run_loso_derivation_recovers_a_clean_separating_threshold():
    """
    Three subjects, each with a clean gap between relaxed (~0.1-0.2) and
    stressed (~0.7-0.9) scores. The LOSO-derived optimal threshold should
    land somewhere in the gap, and held-out accuracy at that threshold
    should be perfect - a hand-known answer, not just "doesn't crash".
    """
    # Every subject shares the exact same stressed-cluster floor (0.80).
    # roc_curve()'s tie-broken threshold always lands on the training
    # pool's lowest stressed score, which - since it's identical across
    # subjects here - equals the held-out subject's own floor too. A
    # varying floor per subject (tried first) let a fold's threshold end
    # up just above the one held-out subject's floor, misclassifying
    # exactly one frame - a real, correct generalisation-gap effect, but
    # not what this test is trying to isolate (that the machinery finds a
    # separating threshold and scores held-out accuracy correctly at all).
    subjects = {
        'A': [(0.05, 0), (0.15, 0), (0.80, 1), (0.90, 1)],
        'B': [(0.08, 0), (0.18, 0), (0.80, 1), (0.92, 1)],
        'C': [(0.10, 0), (0.20, 0), (0.80, 1), (0.95, 1)],
    }
    results = run_loso_derivation(subjects)
    assert results is not None
    assert results['n_folds'] == 3
    assert 0.20 <= results['mean_optimal_threshold'] <= 0.85
    assert results['mean_acc_optimal'] == pytest.approx(1.0)


def test_run_loso_derivation_reports_fixed_threshold_comparison():
    subjects = {
        'A': [(0.10, 0), (0.80, 1)],
        'B': [(0.12, 0), (0.78, 1)],
    }
    results = run_loso_derivation(subjects)
    assert results is not None
    for row in results['fold_rows']:
        assert 'acc_fixed' in row and 'acc_optimal' in row


def test_run_loso_derivation_skips_fold_with_single_class_training_pool():
    """
    If every OTHER subject shares one label, roc_curve() can't fit - that
    fold must be skipped, not crash the whole study.
    """
    subjects = {
        'A': [(0.1, 0)],          # only relaxed
        'B': [(0.1, 0)],          # only relaxed
        'C': [(0.9, 1), (0.1, 0)],
    }
    results = run_loso_derivation(subjects)
    # Holding out C: training pool (A+B) is single-class -> C's fold skipped.
    # Holding out A or B: training pool has both classes -> those folds run.
    assert results is not None
    held_out_subjects = {r['subject'] for r in results['fold_rows']}
    assert 'C' not in held_out_subjects


# --------------------------------------------------------------------------
# run_sensitivity_sweep
# --------------------------------------------------------------------------

def test_run_sensitivity_sweep_returns_none_for_single_class_data():
    assert run_sensitivity_sweep({'A': [(0.1, 0), (0.2, 0)]}) is None


def test_run_sensitivity_sweep_hand_checked_accuracy_at_each_threshold():
    subjects = {'A': [(0.1, 0), (0.4, 1), (0.6, 1)]}
    rows = run_sensitivity_sweep(subjects, thresholds=[0.3, 0.5])
    by_th = {r['threshold']: r for r in rows}
    # threshold 0.3: predict stressed for 0.4,0.6 (correct), relaxed for 0.1 (correct) -> 3/3
    assert by_th[0.3]['accuracy'] == pytest.approx(1.0)
    # threshold 0.5: predict stressed only for 0.6 -> 0.4 misclassified as relaxed -> 2/3
    assert by_th[0.5]['accuracy'] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def test_render_handles_none_results():
    out = []
    render(None, None, out)
    text = "\n".join(out)
    assert 'fewer than 2 subjects' in text
    assert 'no usable pooled data' in text


def test_render_states_moderate_high_limitation_explicitly():
    out = []
    render(None, None, out)
    text = "\n".join(out)
    assert 'MODERATE/HIGH split has no ground truth' in text


def test_render_marks_current_threshold_in_sweep_table():
    import config
    out = []
    sweep = [{'threshold': config.LOW_TO_MODERATE_CUTOFF, 'accuracy': 0.9, 'balanced_accuracy': 0.85}]
    render(None, sweep, out)
    text = "\n".join(out)
    assert '<- current' in text
