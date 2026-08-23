"""
eval_scoring.py tests.

This module computes every accuracy percentage and confidence interval this
project reports (run_all_evaluations.py, ablation_study.py both build on
it) and previously had no dedicated test file - it was only exercised
indirectly through ablation_study.py's tests, which never asserted on
wilson()/frame_scores()/session_scores() in isolation against a hand-known
answer. These tests do that directly.
"""

import math

import pytest

from eval_scoring import wilson, pct, ci_str, frame_scores, session_scores


def _rows(levels, stress_index=0.5):
    return [{'level': lvl, 'stress_index': stress_index} for lvl in levels]


# ---------------------------------------------------------------------------
# wilson()
# ---------------------------------------------------------------------------

def _wilson_reference(successes, n):
    """
    Independent closed-form Wilson score implementation - the same textbook
    formula eval_scoring.py's own no-statsmodels fallback uses, but written
    out separately here so this is a real external check rather than the
    module re-validating itself.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054                          # 95%
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@pytest.mark.parametrize("successes,n", [(8, 10), (0, 10), (10, 10), (1, 1), (37, 152)])
def test_wilson_matches_closed_form_reference(successes, n):
    lo, hi = wilson(successes, n)
    ref_lo, ref_hi = _wilson_reference(successes, n)
    assert lo == pytest.approx(ref_lo, abs=1e-9)
    assert hi == pytest.approx(ref_hi, abs=1e-9)


def test_wilson_matches_statsmodels_directly():
    """The audit's suggested cross-check, when statsmodels is present."""
    pytest.importorskip("statsmodels")
    from statsmodels.stats.proportion import proportion_confint

    lo, hi = wilson(8, 10)
    ref_lo, ref_hi = proportion_confint(8, 10, alpha=0.05, method="wilson")
    assert lo == pytest.approx(ref_lo, abs=1e-9)
    assert hi == pytest.approx(ref_hi, abs=1e-9)


def test_wilson_zero_n_returns_zero_interval():
    assert wilson(0, 0) == (0.0, 0.0)


def test_wilson_interval_bounds_stay_within_unit_range():
    for successes, n in [(0, 5), (5, 5), (3, 5)]:
        lo, hi = wilson(successes, n)
        assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# pct() / ci_str()
# ---------------------------------------------------------------------------

def test_pct_formats_two_decimal_percent():
    assert pct(0.5) == "50.00%"
    assert pct(1.0) == "100.00%"
    assert pct(0.1234) == "12.34%"
    assert pct(0.0) == "0.00%"


def test_ci_str_formats_one_decimal_percent_bounds():
    assert ci_str((0.1, 0.9)) == "[10.0%, 90.0%]"
    assert ci_str((0.12345, 0.6789)) == "[12.3%, 67.9%]"


# ---------------------------------------------------------------------------
# frame_scores()
# ---------------------------------------------------------------------------

def test_frame_scores_hand_computed_accuracy_and_confusion():
    # relaxed: 3 frames, 2 correctly LOW, 1 wrongly MODERATE
    relaxed_files = [('r1.csv', _rows(['LOW', 'LOW', 'MODERATE'], stress_index=0.2))]
    # stressed: 4 frames, 3 correctly M/H, 1 wrongly LOW
    stressed_files = [('s1.csv', _rows(['HIGH', 'LOW', 'MODERATE', 'HIGH'], stress_index=0.7))]

    result = frame_scores(relaxed_files, stressed_files)

    assert result['n'] == 7
    assert result['relaxed_n'] == 3
    assert result['relaxed_ok'] == 2
    assert result['stressed_n'] == 4
    assert result['stressed_ok'] == 3
    assert result['correct'] == 5
    assert result['accuracy'] == pytest.approx(5 / 7)
    assert result['relaxed_recall'] == pytest.approx(2 / 3)
    assert result['stressed_recall'] == pytest.approx(3 / 4)
    assert result['confusion']['relaxed'] == {'LOW': 2, 'MODERATE': 1}
    assert result['confusion']['stressed'] == {'HIGH': 2, 'LOW': 1, 'MODERATE': 1}
    assert result['ci'] == wilson(5, 7)
    assert result['relaxed_ci'] == wilson(2, 3)
    assert result['stressed_ci'] == wilson(3, 4)
    assert result['relaxed_mean'] == pytest.approx(0.2)
    assert result['stressed_mean'] == pytest.approx(0.7)
    assert result['gap'] == pytest.approx(0.7 - 0.2)


def test_frame_scores_all_correct_is_full_accuracy():
    relaxed_files = [('r.csv', _rows(['LOW'] * 5))]
    stressed_files = [('s.csv', _rows(['HIGH'] * 5))]
    result = frame_scores(relaxed_files, stressed_files)
    assert result['accuracy'] == pytest.approx(1.0)
    assert result['relaxed_recall'] == pytest.approx(1.0)
    assert result['stressed_recall'] == pytest.approx(1.0)


def test_frame_scores_all_wrong_is_zero_accuracy():
    relaxed_files = [('r.csv', _rows(['HIGH'] * 5))]
    stressed_files = [('s.csv', _rows(['LOW'] * 5))]
    result = frame_scores(relaxed_files, stressed_files)
    assert result['accuracy'] == pytest.approx(0.0)
    assert result['relaxed_recall'] == pytest.approx(0.0)
    assert result['stressed_recall'] == pytest.approx(0.0)


def test_frame_scores_empty_relaxed_side():
    result = frame_scores([], [('s.csv', _rows(['HIGH', 'HIGH']))])
    assert result['relaxed_n'] == 0
    assert result['relaxed_recall'] == 0.0
    assert result['relaxed_mean'] is None
    assert result['stressed_n'] == 2
    assert result['accuracy'] == pytest.approx(1.0)
    # gap needs BOTH sides present, one side being empty must not crash
    # statistics.mean() on an empty sequence
    assert result['gap'] is None


def test_frame_scores_empty_stressed_side():
    result = frame_scores([('r.csv', _rows(['LOW', 'LOW']))], [])
    assert result['stressed_n'] == 0
    assert result['stressed_recall'] == 0.0
    assert result['stressed_mean'] is None
    assert result['gap'] is None


def test_frame_scores_totally_empty_returns_zero_not_crash():
    result = frame_scores([], [])
    assert result['n'] == 0
    assert result['accuracy'] == 0.0
    assert result['ci'] == (0.0, 0.0)
    assert result['gap'] is None


# ---------------------------------------------------------------------------
# session_scores()
# ---------------------------------------------------------------------------

def test_session_scores_hand_computed():
    relaxed_files = [
        ('r1.csv', _rows(['LOW', 'LOW', 'MODERATE'])),    # majority LOW -> correct
        ('r2.csv', _rows(['HIGH', 'MODERATE'])),           # majority not-LOW -> wrong
    ]
    stressed_files = [
        ('s1.csv', _rows(['HIGH', 'HIGH', 'HIGH'])),       # majority M/H -> correct
    ]
    result = session_scores(relaxed_files, stressed_files)
    assert result['n'] == 3
    assert result['correct'] == 2
    assert result['accuracy'] == pytest.approx(2 / 3)
    assert result['ci'] == wilson(2, 3)
    ok_by_name = {name: ok for name, _, ok in result['detail']}
    assert ok_by_name == {'r1.csv': True, 'r2.csv': False, 's1.csv': True}


def test_session_scores_majority_vote_tie_counts_as_not_ok():
    """session_scores() uses a strict '>' for the majority vote - an exact
    50/50 split must NOT count as correct, since neither level actually won
    the frame count. This is the majority-vote tie case the audit calls
    out explicitly."""
    stressed_files = [('tie.csv', _rows(['HIGH', 'HIGH', 'LOW', 'LOW']))]
    result = session_scores([], stressed_files)
    assert result['n'] == 1
    assert result['correct'] == 0
    name, condition, ok = result['detail'][0]
    assert (name, condition, ok) == ('tie.csv', 'stressed', False)


def test_session_scores_relaxed_majority_vote_tie_counts_as_not_ok():
    relaxed_files = [('tie.csv', _rows(['LOW', 'LOW', 'MODERATE', 'HIGH']))]
    result = session_scores(relaxed_files, [])
    assert result['correct'] == 0
    assert result['detail'][0][2] is False


def test_session_scores_skips_files_with_no_rows():
    result = session_scores([('empty.csv', [])], [])
    assert result['n'] == 0
    assert result['detail'] == []


def test_session_scores_totally_empty_returns_zero_not_crash():
    result = session_scores([], [])
    assert result['n'] == 0
    assert result['accuracy'] == 0.0
    assert result['ci'] == (0.0, 0.0)
