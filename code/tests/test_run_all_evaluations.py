"""
run_all_evaluations.py tests.

This is the canonical evaluation harness - the module every reported
accuracy number ultimately runs through - and previously had no test file
at all. These tests focus on collect()/matches_mode() (the mode-filtering
logic BUG_AUDIT_2026-08-23.md finding #4 was about: a modern-but-corrupted
filename must NOT silently count as Default), find_unclaimed_files() (the
structural fix that catches a repeat of the relaaxed_2.csv founding bug),
and render_per_subject()'s subject-grouping.

Real evaluation_data/ CSVs aren't touched here - everything runs against
synthetic tmp_path files, matching the house style in test_ablation_study.py
and test_correlate_physiology.py.
"""

import csv

import pytest

from app_naming import parse_filename
from eval_scoring import pct
from run_all_evaluations import collect, find_unclaimed_files, render_per_subject


def _write_csv(path, levels, stress_index=0.5):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stress_index', 'level', 'daisee', 'behavior'])
        for lvl in levels:
            w.writerow([stress_index, lvl, 0.0, 0.0])


def _rows(levels, stress_index=0.5):
    """In-memory equivalent of _write_csv()'s output, for render_per_subject()
    tests that build (name, rows) tuples directly instead of round-tripping
    through disk."""
    return [{'level': lvl, 'stress_index': stress_index} for lvl in levels]


# ---------------------------------------------------------------------------
# collect() / matches_mode()
# ---------------------------------------------------------------------------

def test_collect_mode_none_returns_everything(tmp_path):
    _write_csv(tmp_path / 'relaxed_S01_default_2026-08-12.csv', ['LOW'])
    _write_csv(tmp_path / 'relaxed_S02_studying_2026-08-12.csv', ['LOW'])
    relaxed, _ = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode=None)
    names = {n for n, _ in relaxed}
    assert names == {'relaxed_S01_default_2026-08-12.csv',
                     'relaxed_S02_studying_2026-08-12.csv'}


def test_collect_mode_filters_canonical_names_by_encoded_mode(tmp_path):
    _write_csv(tmp_path / 'relaxed_S01_default_2026-08-12.csv', ['LOW'])
    _write_csv(tmp_path / 'relaxed_S02_studying_2026-08-12.csv', ['LOW'])

    default_relaxed, _ = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode='default')
    studying_relaxed, _ = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode='studying')

    assert [n for n, _ in default_relaxed] == ['relaxed_S01_default_2026-08-12.csv']
    assert [n for n, _ in studying_relaxed] == ['relaxed_S02_studying_2026-08-12.csv']


def test_collect_legacy_two_part_name_counts_only_as_default(tmp_path):
    """relaxed_1.csv predates per-mode recording - it belongs to Default
    only, never to any other mode section."""
    _write_csv(tmp_path / 'relaxed_1.csv', ['LOW'])

    default_relaxed, _ = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode='default')
    studying_relaxed, _ = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode='studying')

    assert [n for n, _ in default_relaxed] == ['relaxed_1.csv']
    assert studying_relaxed == []


def test_collect_modern_corrupted_name_excluded_from_every_mode(tmp_path):
    """
    BUG_AUDIT_2026-08-23.md finding #4: a hand-edited subject ID with a
    stray underscore produces a 5-part name that LOOKS canonical but fails
    parse_filename() (test_export_naming.py proves this exact name doesn't
    parse). Before the fix this silently counted as Default; it must now
    count for NO mode section, including Default.
    """
    bad_name = 'relaxed_S_01_default_2026-08-12.csv'
    _write_csv(tmp_path / bad_name, ['LOW'])
    assert parse_filename(bad_name) is None   # premise check

    for mode in ('default', 'studying', 'hospital', 'driving'):
        relaxed, _ = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode=mode)
        assert relaxed == [], f"mode={mode} unexpectedly claimed the corrupted name"


def test_collect_tolerates_ubfc_traceability_suffix(tmp_path):
    """A real 7-part canonical name (evaluate_video_file.py's UBFC-Phys
    traceability suffix) must still parse and filter correctly - it is NOT
    the 'modern but corrupted' case despite having >=4 underscore parts."""
    name = 'relaxed_U01_default_2026-08-15_vid_s1_T1.csv'
    _write_csv(tmp_path / name, ['LOW'])
    relaxed, _ = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode='default')
    assert [n for n, _ in relaxed] == [name]


def test_collect_stressed_glob_independently_filtered(tmp_path):
    _write_csv(tmp_path / 'stressed_S01_hospital_2026-08-12.csv', ['HIGH'])
    _, stressed_hospital = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode='hospital')
    _, stressed_driving = collect(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', mode='driving')
    assert len(stressed_hospital) == 1
    assert stressed_driving == []


# ---------------------------------------------------------------------------
# find_unclaimed_files()
# ---------------------------------------------------------------------------

ALL_MODES = ('default', 'studying', 'hospital', 'driving')


def test_find_unclaimed_files_empty_when_everything_claimed(tmp_path):
    _write_csv(tmp_path / 'relaxed_S01_default_2026-08-12.csv', ['LOW'])
    _write_csv(tmp_path / 'stressed_S01_studying_2026-08-12.csv', ['HIGH'])
    unclaimed = find_unclaimed_files(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', ALL_MODES)
    assert unclaimed == []


def test_find_unclaimed_files_reports_modern_corrupted_name(tmp_path):
    """The direct regression test for finding #4's fix: a corrupted-but-
    modern-looking name must show up in the unclaimed-files report instead
    of silently inflating Default's numbers."""
    bad_name = 'relaxed_S_01_default_2026-08-12.csv'
    _write_csv(tmp_path / bad_name, ['LOW'])
    unclaimed = find_unclaimed_files(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', ALL_MODES)
    assert unclaimed == [bad_name]


def test_find_unclaimed_files_legacy_name_needs_default_in_the_checked_modes(tmp_path):
    _write_csv(tmp_path / 'relaxed_1.csv', ['LOW'])

    # "default" not among the checked modes -> legacy file goes unclaimed
    unclaimed = find_unclaimed_files(tmp_path, 'relaxed_*.csv', 'stressed_*.csv',
                                     ('studying', 'hospital', 'driving'))
    assert unclaimed == ['relaxed_1.csv']

    # with "default" included, it's claimed and disappears from the report
    unclaimed_with_default = find_unclaimed_files(
        tmp_path, 'relaxed_*.csv', 'stressed_*.csv', ALL_MODES)
    assert unclaimed_with_default == []


def test_find_unclaimed_files_mix_of_claimed_and_unclaimed(tmp_path):
    _write_csv(tmp_path / 'relaxed_S01_default_2026-08-12.csv', ['LOW'])       # claimed
    _write_csv(tmp_path / 'relaxed_S_01_hospital_2026-08-12.csv', ['LOW'])     # corrupted
    _write_csv(tmp_path / 'stressed_S02_driving_2026-08-12.csv', ['HIGH'])     # claimed
    unclaimed = find_unclaimed_files(tmp_path, 'relaxed_*.csv', 'stressed_*.csv', ALL_MODES)
    assert unclaimed == ['relaxed_S_01_hospital_2026-08-12.csv']


# ---------------------------------------------------------------------------
# render_per_subject()
# ---------------------------------------------------------------------------

def test_render_per_subject_noops_below_two_subjects():
    relaxed_files = [('relaxed_S01_default_2026-08-12.csv', _rows(['LOW', 'LOW']))]
    stressed_files = [('stressed_S01_default_2026-08-12.csv', _rows(['HIGH', 'HIGH']))]
    out = []
    render_per_subject(relaxed_files, stressed_files, out)
    assert out == []


def test_render_per_subject_groups_by_subject_when_two_or_more():
    relaxed_files = [
        ('relaxed_S01_default_2026-08-12.csv', _rows(['LOW', 'LOW'])),        # 2/2 correct
        ('relaxed_S02_default_2026-08-12.csv', _rows(['LOW', 'HIGH'])),       # 1/2 correct
    ]
    stressed_files = [
        ('stressed_S01_default_2026-08-12.csv', _rows(['HIGH', 'HIGH'])),     # 2/2 correct
        ('stressed_S02_default_2026-08-12.csv', _rows(['HIGH', 'LOW'])),      # 1/2 correct
    ]
    out = []
    render_per_subject(relaxed_files, stressed_files, out)
    text = "\n".join(out)

    assert 'PER-SUBJECT breakdown' in text
    assert 'S01' in text and 'S02' in text
    # S01: 4/4 correct = 100%; S02: 2/4 correct = 50% - hand-computed
    assert pct(1.0) in text
    assert pct(0.5) in text


def test_render_per_subject_skips_names_that_do_not_parse():
    """A legacy 2-part name has no encoded subject - it must not be
    miscounted as its own 'subject' in the breakdown."""
    relaxed_files = [
        ('relaxed_S01_default_2026-08-12.csv', _rows(['LOW', 'LOW'])),
        ('relaxed_1.csv', _rows(['LOW'])),
        ('relaxed_S02_default_2026-08-12.csv', _rows(['LOW', 'LOW'])),
    ]
    out = []
    render_per_subject(relaxed_files, [], out)
    text = "\n".join(out)
    assert 'S01' in text and 'S02' in text
    assert 'relaxed_1' not in text


def test_render_per_subject_ignores_subjects_with_no_rows():
    """A parseable name whose file happened to produce zero usable rows
    (e.g. every row malformed) must not appear as a spurious n=0 subject."""
    relaxed_files = [
        ('relaxed_S01_default_2026-08-12.csv', _rows(['LOW', 'LOW'])),
        ('relaxed_S02_default_2026-08-12.csv', []),
    ]
    stressed_files = [
        ('stressed_S02_default_2026-08-12.csv', _rows(['HIGH', 'HIGH'])),
    ]
    out = []
    render_per_subject(relaxed_files, stressed_files, out)
    text = "\n".join(out)
    # S02 has stressed rows even though its relaxed file was empty, so it
    # still participates - this checks the "skip if fr['n']==0" branch
    # rather than a subject vanishing outright.
    assert 'S01' in text and 'S02' in text
