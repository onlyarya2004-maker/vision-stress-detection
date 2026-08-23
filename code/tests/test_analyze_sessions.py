"""
analyze_sessions.py tests.

This script is documented in README.md as "superseded by the harness"
(run_all_evaluations.py) but is still present and runnable, so
BUG_AUDIT_2026-08-23.md finding #18 flagged its missing task-mode filter -
unlike the harness, it would silently pool Studying/Hospital/Driving
recordings into one mode-blind number if pointed at mixed-mode data. Kept
deliberately small given the file's "superseded, lower-priority" status.
"""

import csv

from analyze_sessions import collect_files, matches_mode


def _write_csv(path, level='LOW'):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stress_index', 'level', 'daisee', 'emotion', 'behavior'])
        w.writerow([0.2, level, 0.1, 0.0, 0.2])


def test_collect_files_defaults_to_unfiltered(tmp_path):
    """No --mode passed -> every relaxed_*/stressed_* file counts, matching
    this script's behavior before the filter was added."""
    _write_csv(tmp_path / 'relaxed_S01_default_2026-08-12.csv')
    _write_csv(tmp_path / 'relaxed_S02_studying_2026-08-12.csv')
    relaxed, _ = collect_files(tmp_path, mode=None)
    assert len(relaxed) == 2


def test_collect_files_mode_filters_by_encoded_task_mode(tmp_path):
    _write_csv(tmp_path / 'relaxed_S01_default_2026-08-12.csv')
    _write_csv(tmp_path / 'relaxed_S02_studying_2026-08-12.csv')
    default_relaxed, _ = collect_files(tmp_path, mode='default')
    studying_relaxed, _ = collect_files(tmp_path, mode='studying')
    assert [p.name for p in default_relaxed] == ['relaxed_S01_default_2026-08-12.csv']
    assert [p.name for p in studying_relaxed] == ['relaxed_S02_studying_2026-08-12.csv']


def test_matches_mode_legacy_name_is_default_only(tmp_path):
    path = tmp_path / 'relaxed_1.csv'
    _write_csv(path)
    assert matches_mode(path, None) is True
    assert matches_mode(path, 'default') is True
    assert matches_mode(path, 'studying') is False
