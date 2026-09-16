"""
diagnose_features.py tests.

Pure-Python CSV/statistics tool (no models, no camera) with zero prior
test coverage despite real branching logic (verdict() has 6 distinct
outcomes) that directly informs which behavioral signals are trusted.
"""

import csv

import pytest

from diagnose_features import col, mean, pstdev, cohens_d, verdict, split_by_task


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# col() / mean() / pstdev()
# ---------------------------------------------------------------------------

def test_col_extracts_numeric_values():
    rows = [{'x': '1.0'}, {'x': '2.5'}, {'x': '3.0'}]
    assert col(rows, 'x') == [1.0, 2.5, 3.0]


def test_col_skips_blank_and_none_values():
    """The CNN doesn't run every frame - blank/None entries in a trace
    column must be skipped, not crash or count as zero."""
    rows = [{'x': '1.0'}, {'x': ''}, {'x': 'None'}, {'x': '2.0'}]
    assert col(rows, 'x') == [1.0, 2.0]


def test_col_skips_unparseable_strings():
    rows = [{'x': '1.0'}, {'x': 'not_a_number'}]
    assert col(rows, 'x') == [1.0]


def test_col_missing_key_returns_empty():
    assert col([{'y': '1.0'}], 'x') == []


def test_mean_of_empty_column_is_nan():
    assert mean([{'x': ''}], 'x') != mean([{'x': ''}], 'x')  # NaN != NaN


def test_mean_hand_computed():
    rows = [{'x': '2.0'}, {'x': '4.0'}, {'x': '6.0'}]
    assert mean(rows, 'x') == pytest.approx(4.0)


def test_pstdev_of_single_value_is_zero():
    """len(v) <= 1 must return 0.0, not raise (statistics.pstdev needs
    >= 1 point but is only meaningful with >= 2)."""
    assert pstdev([{'x': '5.0'}], 'x') == 0.0


def test_pstdev_hand_computed():
    rows = [{'x': '2.0'}, {'x': '4.0'}]
    # population stdev of [2, 4]: mean=3, variance=((1)^2+(1)^2)/2=1, sd=1
    assert pstdev(rows, 'x') == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cohens_d()
# ---------------------------------------------------------------------------

def test_cohens_d_hand_computed():
    """rest=[1,1,1,1], stressed=[3,3,3,3] but with real spread to make d
    computable: rest=[1,2], stressed=[3,4] -> pooled sd = sqrt((0.25+0.25)/2)
    = 0.5, mean diff = 3.5-1.5 = 2.0, d = 2.0/0.5 = 4.0."""
    rest = [{'x': '1.0'}, {'x': '2.0'}]
    stressed = [{'x': '3.0'}, {'x': '4.0'}]
    assert cohens_d(rest, stressed, 'x') == pytest.approx(4.0)


def test_cohens_d_negative_when_stressed_is_lower():
    rest = [{'x': '3.0'}, {'x': '4.0'}]
    stressed = [{'x': '1.0'}, {'x': '2.0'}]
    assert cohens_d(rest, stressed, 'x') < 0


def test_cohens_d_nan_with_insufficient_data():
    rest = [{'x': '1.0'}]     # only 1 point
    stressed = [{'x': '2.0'}, {'x': '3.0'}]
    d = cohens_d(rest, stressed, 'x')
    assert d != d  # NaN


def test_cohens_d_zero_when_pooled_variance_is_effectively_zero():
    """Both groups perfectly constant (no variance at all) -> pooled sd
    ~0 -> must return exactly 0.0, not divide-by-zero -> inf/nan."""
    rest = [{'x': '5.0'}, {'x': '5.0'}]
    stressed = [{'x': '5.0'}, {'x': '5.0'}]
    assert cohens_d(rest, stressed, 'x') == 0.0


# ---------------------------------------------------------------------------
# verdict()
# ---------------------------------------------------------------------------

def test_verdict_dead_when_no_variance():
    assert verdict(d=0.5, rest_mean=0.5, pooled_sd=1e-9) == "DEAD (no variance)"


def test_verdict_saturated_when_rest_mean_near_max():
    assert verdict(d=0.1, rest_mean=0.97, pooled_sd=0.1) == "SATURATED at rest"


def test_verdict_na_for_nan_d():
    assert verdict(d=float('nan'), rest_mean=0.5, pooled_sd=0.1) == "n/a"


def test_verdict_inverted_for_negative_d_beyond_threshold():
    assert verdict(d=-0.3, rest_mean=0.5, pooled_sd=0.1) == "INVERTED"


def test_verdict_useless_for_small_magnitude_d():
    assert verdict(d=0.1, rest_mean=0.5, pooled_sd=0.1) == "useless"


def test_verdict_weak_for_moderate_d():
    assert verdict(d=0.3, rest_mean=0.5, pooled_sd=0.1) == "weak"


def test_verdict_useful_for_larger_d():
    assert verdict(d=0.7, rest_mean=0.5, pooled_sd=0.1) == "useful"


def test_verdict_strong_for_large_d():
    assert verdict(d=1.5, rest_mean=0.5, pooled_sd=0.1) == "STRONG"


def test_verdict_boundary_at_exactly_0point2_is_not_inverted_or_useless():
    """d == -0.2 exactly must hit the <= -0.2 branch (INVERTED), not fall
    through to the abs(d) < 0.2 'useless' branch - boundary behaviour that
    a >= vs > typo could silently flip."""
    assert verdict(d=-0.2, rest_mean=0.5, pooled_sd=0.1) == "INVERTED"


def test_verdict_dead_takes_priority_over_saturated():
    """A signal that is both dead (no variance) AND has rest_mean > 0.95
    must report DEAD, not SATURATED - dead is checked first, and a
    reordering of these branches would silently change which diagnosis a
    genuinely broken signal gets."""
    assert verdict(d=0.0, rest_mean=0.99, pooled_sd=1e-9) == "DEAD (no variance)"


# ---------------------------------------------------------------------------
# split_by_task()
# ---------------------------------------------------------------------------

def test_split_by_task_groups_t1_as_rest_and_t2_t3_as_stressed(tmp_path):
    _write_csv(tmp_path / "features_vid_s1_T1.csv", [{'eye_strain': '0.1'}])
    _write_csv(tmp_path / "features_vid_s1_T2.csv", [{'eye_strain': '0.5'}])
    _write_csv(tmp_path / "features_vid_s1_T3.csv", [{'eye_strain': '0.6'}])
    rest, stressed, files = split_by_task(tmp_path)
    assert len(rest) == 1
    assert len(stressed) == 2
    assert len(files) == 3


def test_split_by_task_ignores_files_with_no_task_number(tmp_path):
    _write_csv(tmp_path / "features_untagged.csv", [{'eye_strain': '0.1'}])
    rest, stressed, files = split_by_task(tmp_path)
    assert rest == [] and stressed == []
    assert len(files) == 1   # still listed, just not grouped


def test_split_by_task_empty_directory_returns_empty(tmp_path):
    rest, stressed, files = split_by_task(tmp_path)
    assert rest == [] and stressed == [] and files == []
