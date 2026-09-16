"""
wesad_calibration.py tests.

The real WESAD .pkl dataset (~15 subjects, 700Hz chest signals) isn't part
of the repo and takes ~30 minutes to process - so, matching this project's
established pattern (ablation_study, correlate_physiology, temporal_model
all test their pure math against synthetic data rather than skip testing
entirely), every pure signal-processing function here is tested against
hand-constructed synthetic signals with a known right answer. This module
had ZERO test coverage before this file, despite computing the actual
numbers ("thresholds validated against the WESAD reference dataset") this
project reports in its methodology.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

from wesad_calibration import (
    _normalize_keys, find_signal, extract_hr_rmssd, extract_eda_features,
    extract_resp_rate, windows_for_label, _slide,
)

FS = 700  # WESAD chest sampling rate


# ---------------------------------------------------------------------------
# _normalize_keys
# ---------------------------------------------------------------------------

def test_normalize_keys_decodes_bytes_keys():
    raw = {b'signal': {b'chest': {b'ECG': [1, 2, 3]}}}
    out = _normalize_keys(raw)
    assert out == {'signal': {'chest': {'ECG': [1, 2, 3]}}}


def test_normalize_keys_leaves_str_keys_and_non_dicts_untouched():
    raw = {'already_str': [1, 2, 3], b'mixed': 'value'}
    out = _normalize_keys(raw)
    assert out == {'already_str': [1, 2, 3], 'mixed': 'value'}


def test_normalize_keys_recurses_through_nested_dicts():
    raw = {b'a': {b'b': {b'c': 42}}}
    assert _normalize_keys(raw) == {'a': {'b': {'c': 42}}}


# ---------------------------------------------------------------------------
# find_signal
# ---------------------------------------------------------------------------

def test_find_signal_matches_case_insensitively():
    chest = {'RESP': np.array([1.0, 2.0, 3.0])}
    result = find_signal(chest, 'Resp', 'RESP')
    assert list(result) == [1.0, 2.0, 3.0]


def test_find_signal_tries_every_candidate():
    """Multi-sample array, matching real usage - a single-sample array
    would squeeze() to a 0-d scalar, which is not a realistic WESAD
    window (60s at 700Hz = 42,000 samples) and isn't what this test is
    checking (candidate-matching, not squeeze()'s own edge behaviour)."""
    chest = {'Temp': np.array([36.5, 36.6, 36.4])}
    result = find_signal(chest, 'TEMP', 'Temp')
    assert list(result) == [36.5, 36.6, 36.4]


def test_find_signal_raises_keyerror_when_nothing_matches():
    chest = {'ECG': np.array([1.0])}
    with pytest.raises(KeyError):
        find_signal(chest, 'Resp', 'RESP')


def test_find_signal_squeezes_extra_dimensions():
    """WESAD chest arrays are stored as (N, 1) column vectors - squeeze()
    must flatten them, or every downstream slice/filter call breaks."""
    chest = {'ECG': np.array([[1.0], [2.0], [3.0]])}
    result = find_signal(chest, 'ECG')
    assert result.shape == (3,)


# ---------------------------------------------------------------------------
# windows_for_label / _slide
# ---------------------------------------------------------------------------

def test_slide_produces_non_overlapping_windows_at_zero_overlap():
    windows = list(_slide(0, 10, win=3, overlap=0.0))
    assert windows == [(0, 3), (3, 6), (6, 9)]


def test_slide_produces_half_overlapping_windows():
    """win=4, overlap=0.5 -> step=2: (0,4),(2,6),(4,8) - hand-computed."""
    windows = list(_slide(0, 8, win=4, overlap=0.5))
    assert windows == [(0, 4), (2, 6), (4, 8)]


def test_slide_yields_nothing_when_block_shorter_than_window():
    assert list(_slide(0, 2, win=5, overlap=0.5)) == []


def test_windows_for_label_finds_single_contiguous_block():
    labels = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    # block is indices 2..7 (length 6); win=3, overlap=0 -> step=3
    windows = list(windows_for_label(labels, target_label=1,
                                     win_samples=3, overlap=0.0))
    assert windows == [(2, 5), (5, 8)]


def test_windows_for_label_finds_multiple_separate_blocks():
    labels = [1, 1, 1, 0, 0, 1, 1, 1]
    windows = list(windows_for_label(labels, target_label=1,
                                     win_samples=3, overlap=0.0))
    assert windows == [(0, 3), (5, 8)]


def test_windows_for_label_handles_block_extending_to_end_of_array():
    """A target-label block that never transitions back before the array
    ends must still be yielded - this is the trailing in_block flush."""
    labels = [0, 1, 1, 1]
    windows = list(windows_for_label(labels, target_label=1,
                                     win_samples=3, overlap=0.0))
    assert windows == [(1, 4)]


def test_windows_for_label_returns_nothing_for_absent_label():
    labels = [0, 0, 0, 0]
    assert list(windows_for_label(labels, target_label=1,
                                  win_samples=2, overlap=0.0)) == []


# ---------------------------------------------------------------------------
# extract_hr_rmssd - synthetic ECG with a known, exact R-R interval
# ---------------------------------------------------------------------------

def _synthetic_ecg(bpm, duration_s, fs=FS, pulse_width_samples=15):
    """A train of narrow Gaussian pulses at exactly `bpm`, on a flat
    baseline - enough to drive the bandpass+find_peaks R-peak detector to
    a known, hand-computable heart rate."""
    n = int(duration_s * fs)
    ecg = np.zeros(n)
    rr_samples = int(round(60.0 / bpm * fs))
    t_pulse = np.arange(-pulse_width_samples, pulse_width_samples + 1)
    pulse = np.exp(-(t_pulse ** 2) / (2 * (pulse_width_samples / 3) ** 2))
    for center in range(rr_samples, n - pulse_width_samples, rr_samples):
        lo, hi = center - pulse_width_samples, center + pulse_width_samples + 1
        ecg[lo:hi] += pulse
    return ecg


def test_extract_hr_rmssd_recovers_known_heart_rate():
    ecg = _synthetic_ecg(bpm=72, duration_s=20)
    hr, rmssd = extract_hr_rmssd(ecg)
    assert hr == pytest.approx(72, abs=2.0)
    # Perfectly regular pulses -> near-zero beat-to-beat variability.
    assert rmssd == pytest.approx(0.0, abs=5.0)


def test_extract_hr_rmssd_recovers_a_different_known_heart_rate():
    """A second, distinct rate - guards against a test that would pass
    for any constant output regardless of the real signal."""
    ecg = _synthetic_ecg(bpm=100, duration_s=20)
    hr, _ = extract_hr_rmssd(ecg)
    assert hr == pytest.approx(100, abs=3.0)


def test_extract_hr_rmssd_returns_nan_on_flat_signal():
    """No peaks at all -> undefined heart rate, must be NaN, not a crash
    or a fabricated number."""
    flat = np.zeros(FS * 10)
    hr, rmssd = extract_hr_rmssd(flat)
    assert np.isnan(hr) and np.isnan(rmssd)


# ---------------------------------------------------------------------------
# extract_eda_features
# ---------------------------------------------------------------------------

def test_extract_eda_features_recovers_tonic_level():
    """A constant EDA signal has no phasic component at all - tonic_mean
    must equal the constant level, and scr_per_min must be ~0."""
    duration_s = 30
    eda = np.full(FS * duration_s, 5.0)
    tonic_mean, tonic_std, scr_per_min = extract_eda_features(eda)
    assert tonic_mean == pytest.approx(5.0, abs=0.1)
    assert scr_per_min == pytest.approx(0.0, abs=1.0)


def test_extract_eda_features_counts_injected_scr_peaks():
    """Inject exactly 6 sharp phasic bumps into a 60-second window ->
    expect close to 6 SCRs/min (peaks-per-minute over exactly 1 minute)."""
    duration_s = 60
    n = FS * duration_s
    eda = np.full(n, 2.0)
    rng = np.random.RandomState(0)
    for center in np.linspace(FS * 5, n - FS * 5, 6, dtype=int):
        width = 200
        t = np.arange(-width, width)
        eda[center - width:center + width] += 2.0 * np.exp(-(t ** 2) / (2 * (width / 3) ** 2))
    tonic_mean, tonic_std, scr_per_min = extract_eda_features(eda)
    assert scr_per_min == pytest.approx(6.0, abs=2.0)


# ---------------------------------------------------------------------------
# extract_resp_rate
# ---------------------------------------------------------------------------

def test_extract_resp_rate_recovers_known_breathing_rate():
    """A clean sinusoid at 15 breaths/min (0.25 Hz) over 60s must recover
    close to 15 breaths/min.

    This is a real-bug regression test, not a routine correctness check:
    the (b, a) transfer-function form of this filter (fixed to sos in
    this same change) confirmed fabricated an identical, wrong ~13
    breaths/min for EVERY true rate from 15 to 30 - a tight tolerance
    here is what would have caught it, which is why it's exact (abs=0.5),
    not the loose abs=2.0 that happened to let 13 slip past a 15 target."""
    duration_s = 60
    t = np.arange(FS * duration_s) / FS
    breaths_per_min = 15
    freq_hz = breaths_per_min / 60.0
    resp = np.sin(2 * np.pi * freq_hz * t)
    rate = extract_resp_rate(resp)
    assert rate == pytest.approx(breaths_per_min, abs=0.5)


def test_report_text_is_pure_ascii_regression():
    """Regression test for a real bug: run_calibration() crashed with
    UnicodeEncodeError printing its report to a plain Windows console
    (cp1252 codepage) because of a literal U+2265 ('>=' typeset as '≥') in
    one lines.append(...) call - the JSON and .txt file were written
    correctly first (explicit encoding='utf-8'), but the confirmatory
    print() then crashed before the run could finish cleanly. Since a
    full run needs the real WESAD dataset (~30 min), this instead
    statically parses the source and asserts every string literal passed
    to lines.append() is pure ASCII, which is exactly what would have
    caught this before it ever reached a live console."""
    source_path = Path(__file__).resolve().parent.parent / 'wesad_calibration.py'
    tree = ast.parse(source_path.read_text(encoding='utf-8'))

    non_ascii = []
    for node in ast.walk(tree):
        is_lines_append = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'append'
        )
        if not is_lines_append:
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    if not sub.value.isascii():
                        non_ascii.append((sub.lineno, sub.value))

    assert non_ascii == [], (
        f"non-ASCII string literal(s) in a lines.append(...) call would "
        f"crash printing on a plain Windows console: {non_ascii}"
    )


def test_extract_resp_rate_recovers_a_faster_known_rate():
    """A second, distinct rate - guards specifically against the fixed
    bug's failure mode (every rate in 15-30 bpm collapsing to the same
    ~13 bpm output), which a single test point could not distinguish
    from a coincidentally-correct constant."""
    duration_s = 60
    t = np.arange(FS * duration_s) / FS
    breaths_per_min = 22
    freq_hz = breaths_per_min / 60.0
    resp = np.sin(2 * np.pi * freq_hz * t)
    rate = extract_resp_rate(resp)
    assert rate == pytest.approx(breaths_per_min, abs=0.5)
