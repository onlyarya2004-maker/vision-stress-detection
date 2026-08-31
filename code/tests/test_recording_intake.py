"""
recording_intake.py tests.

This is the pipeline-entry-point check for RECORDING_PROTOCOL.md's founding
bug: a bad filename (relaaxed_2.csv) or an unlabelled export must be caught
BEFORE the file ever reaches evaluation_data/, not discovered later by
run_all_evaluations.py's find_unclaimed_files(). Everything here runs
against synthetic tmp_path CSVs built with the real history_to_csv() header
(ts, stress_index, level, daisee, emotion, behavior) - matching the house
style in test_run_all_evaluations.py.
"""

import csv
from datetime import datetime, timedelta

from recording_intake import validate_file, run_intake, print_summary, MIN_DURATION_SECONDS

BASE = datetime(2026, 8, 12, 10, 0, 0)


def _write_session_csv(path, ts_list, level="LOW", stress_index=0.2):
    """Build a synthetic exported session CSV with the real app.py
    history_to_csv() header/column order."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["ts", "stress_index", "level", "daisee", "emotion", "behavior"])
        w.writeheader()
        for ts in ts_list:
            w.writerow({
                "ts": ts.isoformat(), "stress_index": f"{stress_index:.4f}",
                "level": level, "daisee": "0.1000", "emotion": "0.0000",
                "behavior": "0.1000",
            })


def _long_enough_ts(n=14, step_seconds=5):
    """>= MIN_DURATION_SECONDS worth of timestamps (65s with the defaults)."""
    return [BASE + timedelta(seconds=i * step_seconds) for i in range(n)]


def _write_header_only_csv(path):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(["ts", "stress_index", "level", "daisee", "emotion", "behavior"])


# ---------------------------------------------------------------------------
# validate_file()
# ---------------------------------------------------------------------------

def test_valid_canonical_file_passes(tmp_path):
    path = tmp_path / "relaxed_S01_default_2026-08-12.csv"
    _write_session_csv(path, _long_enough_ts())
    meta, reasons = validate_file(path)
    assert reasons == []
    assert meta == {"condition": "relaxed", "subject": "S01",
                    "mode": "default", "date": "2026-08-12"}


def test_unlabelled_export_flagged_with_specific_message(tmp_path):
    path = tmp_path / "stress_session_2026-08-12_143000.csv"
    _write_session_csv(path, _long_enough_ts())
    meta, reasons = validate_file(path)
    assert meta is None
    assert any("labels not set in the app" in r for r in reasons)
    # must be the distinct, actionable message - not the generic parse failure
    assert not any("does not match the canonical" in r for r in reasons)


def test_malformed_filename_flagged_with_name_shown(tmp_path):
    # the actual historical typo that started this project's data-integrity audit
    path = tmp_path / "relaaxed_2.csv"
    _write_session_csv(path, _long_enough_ts())
    meta, reasons = validate_file(path)
    assert meta is None
    assert any("relaaxed_2.csv" in r for r in reasons)
    assert any("does not match the canonical" in r for r in reasons)


def test_invalid_mode_flagged_even_though_name_parses(tmp_path):
    path = tmp_path / "relaxed_S01_yoga_2026-08-12.csv"
    _write_session_csv(path, _long_enough_ts())
    meta, reasons = validate_file(path)
    assert meta["mode"] == "yoga"
    assert any("mode 'yoga'" in r for r in reasons)


def test_too_short_recording_flagged_with_measured_duration(tmp_path):
    path = tmp_path / "relaxed_S01_default_2026-08-12.csv"
    _write_session_csv(path, [BASE, BASE + timedelta(seconds=30)])
    meta, reasons = validate_file(path)
    assert meta is not None                      # name itself is fine
    assert any("30.0s measured" in r and f"{MIN_DURATION_SECONDS:.0f}s minimum" in r
              for r in reasons)


def test_empty_header_only_file_flagged(tmp_path):
    path = tmp_path / "relaxed_S01_default_2026-08-12.csv"
    _write_header_only_csv(path)
    meta, reasons = validate_file(path)
    assert any("empty or header-only" in r for r in reasons)


def test_destination_collision_is_flagged_not_overwritten(tmp_path):
    incoming = tmp_path / "incoming"
    dest = tmp_path / "dest"
    incoming.mkdir()
    dest.mkdir()
    name = "relaxed_S01_default_2026-08-12.csv"
    _write_session_csv(incoming / name, _long_enough_ts())
    _write_session_csv(dest / name, _long_enough_ts())     # already "claimed"
    meta, reasons = validate_file(incoming / name, dest=dest)
    assert any("already exists" in r for r in reasons)


# ---------------------------------------------------------------------------
# run_intake()
# ---------------------------------------------------------------------------

def test_valid_file_is_moved_into_dest(tmp_path):
    incoming = tmp_path / "incoming"
    dest = tmp_path / "dest"
    incoming.mkdir()
    name = "relaxed_S01_default_2026-08-12.csv"
    _write_session_csv(incoming / name, _long_enough_ts())

    moved, flagged = run_intake(incoming, dest)

    assert flagged == []
    assert [n for n, _ in moved] == [name]
    assert not (incoming / name).exists()
    assert (dest / name).exists()


def test_flagged_files_stay_in_incoming_untouched(tmp_path):
    incoming = tmp_path / "incoming"
    dest = tmp_path / "dest"
    incoming.mkdir()
    name = "stress_session_2026-08-12_143000.csv"
    _write_session_csv(incoming / name, _long_enough_ts())

    moved, flagged = run_intake(incoming, dest)

    assert moved == []
    assert [n for n, _ in flagged] == [name]
    assert (incoming / name).exists()          # left in place
    assert not (dest / name).exists()           # never moved
    assert not dest.exists()                    # not even created for nothing


def test_dry_run_validates_without_moving_anything(tmp_path):
    incoming = tmp_path / "incoming"
    dest = tmp_path / "dest"
    incoming.mkdir()
    name = "relaxed_S01_default_2026-08-12.csv"
    _write_session_csv(incoming / name, _long_enough_ts())

    moved, flagged = run_intake(incoming, dest, dry_run=True)

    assert [n for n, _ in moved] == [name]      # correctly validated as passing
    assert flagged == []
    assert (incoming / name).exists()           # but nothing actually moved
    assert not dest.exists()


def test_empty_incoming_folder_is_a_clean_noop(tmp_path):
    incoming = tmp_path / "incoming"
    dest = tmp_path / "dest"
    incoming.mkdir()

    moved, flagged = run_intake(incoming, dest)

    assert moved == []
    assert flagged == []
    assert not dest.exists()


def test_multiple_files_produce_correct_combined_result(tmp_path, capsys):
    incoming = tmp_path / "incoming"
    dest = tmp_path / "dest"
    incoming.mkdir()

    good_name = "relaxed_S01_default_2026-08-12.csv"
    short_name = "stressed_S02_default_2026-08-12.csv"
    bad_name = "relaaxed_3.csv"

    _write_session_csv(incoming / good_name, _long_enough_ts())
    _write_session_csv(incoming / short_name, [BASE, BASE + timedelta(seconds=10)])
    _write_session_csv(incoming / bad_name, _long_enough_ts())

    moved, flagged = run_intake(incoming, dest)

    assert [n for n, _ in moved] == [good_name]
    assert {n for n, _ in flagged} == {short_name, bad_name}
    assert (dest / good_name).exists()
    assert (incoming / short_name).exists()
    assert (incoming / bad_name).exists()

    print_summary(incoming, dest, False, moved, flagged)
    out = capsys.readouterr().out
    assert "1 file(s) moved" in out
    assert "2 file(s) flagged" in out
    assert good_name in out
    assert short_name in out
    assert bad_name in out
    # the exact next-step command must be the last line, per
    # RECORDING_PROTOCOL.md's "After recording" step 2 - printed, never
    # auto-run, so the multi-minute harness stays a deliberate step.
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[-1] == "cd code && python run_all_evaluations.py"


def test_print_summary_on_empty_run_still_prints_next_step(tmp_path, capsys):
    incoming = tmp_path / "incoming"
    dest = tmp_path / "dest"
    incoming.mkdir()
    print_summary(incoming, dest, False, [], [])
    out = capsys.readouterr().out
    assert "no CSV files found" in out
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[-1] == "cd code && python run_all_evaluations.py"
