"""
correlate_physiology.py tests.

Real UBFC-Phys EDA/trace data isn't part of the repo (see README.md's
dataset section), so run_correlation_study() is tested against synthetic
tmp_path data, engineered so the correlation/direction-agreement math is
checked against a hand-known answer, not just "doesn't crash".
"""

import csv

import numpy as np
import pytest

from correlate_physiology import (
    align, align_hr, bvp_to_heart_rate, load_bvp, load_eda, load_index,
    run_correlation_study, run_hr_correlation_study, render, render_hr,
)


# --------------------------------------------------------------------------
# load_index / load_eda
# --------------------------------------------------------------------------

def test_load_index_reads_t_and_smoothed_index(tmp_path):
    path = tmp_path / 'features_vid_s1_T1.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'stress_index_smoothed', 'other_col'])
        w.writerow(['0.0', '0.30', 'x'])
        w.writerow(['0.5', '0.35', 'y'])
    t, x = load_index(path)
    assert list(t) == [0.0, 0.5]
    assert list(x) == pytest.approx([0.30, 0.35])


def test_load_index_skips_malformed_rows(tmp_path):
    path = tmp_path / 'features_vid_s1_T1.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'stress_index_smoothed'])
        w.writerow(['0.0', '0.30'])
        w.writerow(['not_a_number', '0.35'])
        w.writerow(['1.0', '0.40'])
    t, x = load_index(path)
    assert len(t) == 2
    assert list(t) == [0.0, 1.0]


def test_load_index_sorts_out_of_order_rows_by_time(tmp_path):
    """
    np.interp() in align() requires monotonic x - unlike
    ablation_study.load_trace() (same file schema), this loader used to
    trust file order rather than sorting explicitly (BUG_AUDIT_2026-08-23.md
    finding #19). Rows written out of time order must still come back
    sorted, with (t, index) pairs kept together.
    """
    path = tmp_path / 'features_vid_s1_T1.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'stress_index_smoothed'])
        w.writerow(['1.0', '0.40'])   # written out of order
        w.writerow(['0.0', '0.30'])
        w.writerow(['0.5', '0.35'])
    t, x = load_index(path)
    assert list(t) == [0.0, 0.5, 1.0]
    assert list(x) == pytest.approx([0.30, 0.35, 0.40])


def test_load_index_handles_empty_file_without_crashing(tmp_path):
    path = tmp_path / 'features_vid_s1_T1.csv'
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'stress_index_smoothed'])
    t, x = load_index(path)
    assert len(t) == 0 and len(x) == 0


def test_load_eda_skips_dropout_markers_and_blank_lines(tmp_path):
    path = tmp_path / 'eda_s1_T1.csv'
    path.write_text("0.15\n\ndropout\n0.22\n0.19\n")
    vals = load_eda(path)
    assert list(vals) == pytest.approx([0.15, 0.22, 0.19])


# --------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------

def test_align_returns_none_for_empty_eda():
    t_idx, x_idx = np.array([0.0, 1.0, 2.0]), np.array([0.1, 0.2, 0.3])
    x, e = align(t_idx, x_idx, np.array([]))
    assert x is None and e is None


def test_align_returns_none_when_overlap_too_short():
    # index trace only covers [0, 2]s, EDA runs [0, 100]s at 4Hz -> overlap
    # window (0-2s = 8 samples) is below the 20-sample floor.
    t_idx, x_idx = np.array([0.0, 2.0]), np.array([0.1, 0.3])
    eda = np.zeros(400)  # 100s @ 4Hz
    x, e = align(t_idx, x_idx, eda)
    assert x is None and e is None


def test_align_resamples_index_onto_eda_clock():
    t_idx = np.linspace(0, 10, 11)          # 0..10s, 1 sample/s
    x_idx = t_idx.copy()                     # index == time, for a known answer
    eda = np.zeros(40)                       # 10s @ 4Hz
    x, e = align(t_idx, x_idx, eda)
    assert x is not None
    assert len(x) == len(e) == 40
    # x is the index interpolated onto the EDA clock -> should still equal
    # time-in-seconds at each of those 4Hz sample points.
    expected_t = np.arange(40) / 4.0
    assert x == pytest.approx(expected_t, abs=1e-9)


# --------------------------------------------------------------------------
# run_correlation_study - end to end
# --------------------------------------------------------------------------

def _write_trace(path, t_vals, index_vals):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'stress_index_smoothed'])
        for t, x in zip(t_vals, index_vals):
            w.writerow([t, x])


def test_run_correlation_study_returns_none_for_empty_directory(tmp_path):
    assert run_correlation_study(tmp_path, tmp_path) is None


def test_run_correlation_study_marks_missing_eda_file(tmp_path):
    t = np.linspace(0, 60, 601)
    _write_trace(tmp_path / 'features_vid_s1_T1.csv', t, np.sin(t))
    # No eda_s1_T1.csv anywhere under an s1/s1/ subfolder.
    results = run_correlation_study(tmp_path, tmp_path)
    assert results is not None
    assert results['per_trace'] == [{'label': 's1_T1', 'status': 'no_eda'}]


def test_run_correlation_study_computes_perfect_positive_correlation(tmp_path):
    t = np.linspace(0, 60, 601)             # 10Hz index, 60s
    index_vals = np.sin(t / 5.0) * 0.5 + 0.5
    _write_trace(tmp_path / 'features_vid_s1_T1.csv', t, index_vals)

    eda_dir = tmp_path / 's1' / 's1'
    eda_dir.mkdir(parents=True)
    t_eda = np.arange(240) / 4.0            # 60s @ 4Hz
    eda_vals = np.sin(t_eda / 5.0) * 2.0 + 3.0   # same shape, different scale/offset
    (eda_dir / 'eda_s1_T1.csv').write_text("\n".join(f"{v:.6f}" for v in eda_vals))

    results = run_correlation_study(tmp_path, tmp_path)
    assert results is not None
    row = results['per_trace'][0]
    assert row['status'] == 'ok'
    assert row['pearson'] == pytest.approx(1.0, abs=1e-3)
    assert row['spearman'] == pytest.approx(1.0, abs=1e-3)


def test_run_correlation_study_direction_agreement_hand_checked(tmp_path):
    """
    Two subjects, engineered so the direction-agreement answer is known by
    construction: s1's index rises with EDA from rest to stress (agree),
    s2's index falls while EDA rises (disagree).
    """
    def write_subject(subj, rest_idx, rest_eda, stress_idx, stress_eda):
        t = np.linspace(0, 30, 301)
        for task, idx_val, eda_val in ((1, rest_idx, rest_eda),
                                       (2, stress_idx, stress_eda),
                                       (3, stress_idx, stress_eda)):
            _write_trace(tmp_path / f'features_vid_s{subj}_T{task}.csv',
                        t, np.full_like(t, idx_val))
            eda_dir = tmp_path / f's{subj}' / f's{subj}'
            eda_dir.mkdir(parents=True, exist_ok=True)
            eda_samples = np.full(120, eda_val)  # 30s @ 4Hz
            (eda_dir / f'eda_s{subj}_T{task}.csv').write_text(
                "\n".join(f"{v:.6f}" for v in eda_samples))

    write_subject(1, rest_idx=0.1, rest_eda=0.1, stress_idx=0.8, stress_eda=0.9)  # agree
    write_subject(2, rest_idx=0.8, rest_eda=0.1, stress_idx=0.1, stress_eda=0.9)  # disagree

    results = run_correlation_study(tmp_path, tmp_path)
    assert results['total'] == 2
    assert results['agree'] == 1
    by_subject = {row['subject']: row['agree'] for row in results['agreement_rows']}
    assert by_subject[1] is True
    assert by_subject[2] is False


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def test_render_handles_none_results():
    out = []
    render(None, out)
    assert any('no feature traces found' in line for line in out)


def test_render_generated_line_only_appears_when_passed():
    results = {'traces_dir': 'X', 'ubfc_dir': 'Y', 'per_trace': [],
              'agreement_rows': [], 'agree': 0, 'total': 0}
    out_without = []
    render(results, out_without)
    assert not any('generated' in line for line in out_without)

    out_with = []
    render(results, out_with, generated_line='generated : 2026-01-01')
    assert any('generated : 2026-01-01' in line for line in out_with)


def test_render_formats_each_trace_status_distinctly():
    results = {
        'traces_dir': 'X', 'ubfc_dir': 'Y',
        'per_trace': [
            {'label': 's1_T1', 'status': 'no_eda'},
            {'label': 's2_T1', 'status': 'nan', 'n': 50},
            {'label': 's3_T1', 'status': 'ok', 'n': 50, 'pearson': 0.5, 'spearman': 0.4},
        ],
        'agreement_rows': [], 'agree': 0, 'total': 0,
    }
    out = []
    render(results, out)
    text = "\n".join(out)
    assert 'no EDA file' in text
    assert 'NaN in EDA' in text
    assert '+0.50' in text and '+0.40' in text


# --------------------------------------------------------------------------
# _find_physio_file - tolerates both the doubled s{N}/s{N}/ layout (most
# subjects) and the flat s{N}/ layout (s11) - see module docstring.
# --------------------------------------------------------------------------

def test_run_correlation_study_finds_eda_under_flat_s11_style_layout(tmp_path):
    """
    The original lookup only tried s{N}/s{N}/eda_*.csv, so a subject whose
    files sit directly under s{N}/ (s11's actual layout) was silently
    reported as 'no EDA file' even though the file exists. _find_physio_file
    must find it either way.
    """
    t = np.linspace(0, 60, 601)
    _write_trace(tmp_path / 'features_vid_s11_T1.csv', t, np.sin(t))

    flat_dir = tmp_path / 's11'
    flat_dir.mkdir(parents=True)
    t_eda = np.arange(240) / 4.0
    (flat_dir / 'eda_s11_T1.csv').write_text(
        "\n".join(f"{v:.6f}" for v in np.sin(t_eda)))

    results = run_correlation_study(tmp_path, tmp_path)
    assert results['per_trace'][0]['status'] == 'ok'


# --------------------------------------------------------------------------
# bvp_to_heart_rate
# --------------------------------------------------------------------------

def _synthetic_bvp(bpm, seconds=30, fs=64.0):
    """A clean PPG-like waveform at a known, hand-chosen heart rate."""
    t = np.arange(0, seconds, 1.0 / fs)
    freq_hz = bpm / 60.0
    return t, np.sin(2 * np.pi * freq_hz * t)


def test_bvp_to_heart_rate_recovers_known_rate():
    _, bvp = _synthetic_bvp(bpm=72, seconds=30)
    t_hr, hr = bvp_to_heart_rate(bvp)
    assert len(hr) > 10
    assert np.median(hr) == pytest.approx(72, abs=2)


def test_bvp_to_heart_rate_handles_too_short_signal():
    t_hr, hr = bvp_to_heart_rate(np.zeros(10))   # far under fs*5 samples
    assert len(t_hr) == 0 and len(hr) == 0


def test_bvp_to_heart_rate_handles_flat_signal_without_crashing():
    # No peaks at all in a flat line - must return empty, not divide-by-zero.
    t_hr, hr = bvp_to_heart_rate(np.zeros(64 * 30))
    assert len(t_hr) == 0 and len(hr) == 0


def test_bvp_to_heart_rate_drops_physiologically_impossible_intervals():
    # One huge gap spliced into an otherwise-clean 72bpm signal would
    # produce a sub-30bpm "beat" if not filtered out.
    _, bvp = _synthetic_bvp(bpm=72, seconds=20)
    gap = np.zeros(64 * 5)
    bvp = np.concatenate([bvp[:64 * 10], gap, bvp[64 * 10:]])
    t_hr, hr = bvp_to_heart_rate(bvp)
    assert np.all((hr >= 30) & (hr <= 220))


# --------------------------------------------------------------------------
# align_hr
# --------------------------------------------------------------------------

def test_align_hr_returns_none_for_too_few_beats():
    t_idx, x_idx = np.array([0.0, 10.0]), np.array([0.1, 0.5])
    x, h = align_hr(t_idx, x_idx, np.array([1.0, 2.0]), np.array([70.0, 71.0]))
    assert x is None and h is None


def test_align_hr_resamples_index_onto_heart_rate_timestamps():
    t_idx = np.linspace(0, 10, 101)
    x_idx = t_idx.copy()
    t_hr = np.linspace(1, 9, 20)          # irregular-ish, within [0,10]
    hr = np.full(20, 70.0)
    x, h = align_hr(t_idx, x_idx, t_hr, hr)
    assert x is not None
    assert x == pytest.approx(t_hr, abs=1e-9)
    assert list(h) == pytest.approx([70.0] * 20)


# --------------------------------------------------------------------------
# run_hr_correlation_study / render_hr - end to end
# --------------------------------------------------------------------------

def test_run_hr_correlation_study_marks_missing_bvp_file(tmp_path):
    t = np.linspace(0, 60, 601)
    _write_trace(tmp_path / 'features_vid_s1_T1.csv', t, np.sin(t))
    results = run_hr_correlation_study(tmp_path, tmp_path)
    assert results is not None
    assert results['per_trace'] == [{'label': 's1_T1', 'status': 'no_bvp'}]


def test_run_hr_correlation_study_direction_agreement_hand_checked(tmp_path):
    """
    Two subjects engineered so the answer is known by construction: s1's
    index rises with heart rate from rest to stress (agree), s2's index
    falls while heart rate rises (disagree). Rest = 65bpm, stress = 95bpm -
    clearly distinct after peak detection.
    """
    def write_subject(subj, rest_idx, rest_bpm, stress_idx, stress_bpm):
        t = np.linspace(0, 30, 301)
        for task, idx_val, bpm in ((1, rest_idx, rest_bpm),
                                   (2, stress_idx, stress_bpm),
                                   (3, stress_idx, stress_bpm)):
            _write_trace(tmp_path / f'features_vid_s{subj}_T{task}.csv',
                        t, np.full_like(t, idx_val))
            bvp_dir = tmp_path / f's{subj}' / f's{subj}'
            bvp_dir.mkdir(parents=True, exist_ok=True)
            _, bvp = _synthetic_bvp(bpm=bpm, seconds=30)
            (bvp_dir / f'bvp_s{subj}_T{task}.csv').write_text(
                "\n".join(f"{v:.6f}" for v in bvp))

    write_subject(1, rest_idx=0.1, rest_bpm=65, stress_idx=0.8, stress_bpm=95)  # agree
    write_subject(2, rest_idx=0.8, rest_bpm=65, stress_idx=0.1, stress_bpm=95)  # disagree

    results = run_hr_correlation_study(tmp_path, tmp_path)
    assert results['total'] == 2
    assert results['agree'] == 1
    by_subject = {row['subject']: row['agree'] for row in results['agreement_rows']}
    assert by_subject[1] is True
    assert by_subject[2] is False


def test_render_hr_handles_none_results():
    out = []
    render_hr(None, out)
    assert any('no feature traces found' in line for line in out)


def test_render_hr_formats_each_trace_status_distinctly():
    results = {
        'traces_dir': 'X', 'ubfc_dir': 'Y',
        'per_trace': [
            {'label': 's1_T1', 'status': 'no_bvp'},
            {'label': 's2_T1', 'status': 'no_peaks'},
            {'label': 's3_T1', 'status': 'nan', 'n': 50},
            {'label': 's4_T1', 'status': 'ok', 'n': 50, 'pearson': 0.5, 'spearman': 0.4},
        ],
        'agreement_rows': [], 'agree': 0, 'total': 0,
    }
    out = []
    render_hr(results, out)
    text = "\n".join(out)
    assert 'no BVP file' in text
    assert 'too few detected heartbeats' in text
    assert 'NaN in heart rate' in text
    assert '+0.50' in text and '+0.40' in text
