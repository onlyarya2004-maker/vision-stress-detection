"""
Recording intake
=================
Validates freshly-exported session CSVs BEFORE they enter evaluation_data/,
so a bad filename or a too-short recording is caught at the pipeline entry
point instead of relying on run_all_evaluations.py's after-the-fact
find_unclaimed_files() check (see that function's docstring, and
RECORDING_PROTOCOL.md's relaaxed_2.csv incident - the founding bug that
check exists to catch a repeat of).

Scans a staging folder (default evaluation_data/incoming/) for *.csv files
dropped there straight from the app's "Download session as CSV" button,
validates each one against the canonical naming scheme (app_naming.py) and
RECORDING_PROTOCOL.md's rules, and moves only the files that pass every
check into evaluation_data/live_recordings/ - where the harness already
looks for them (kept apart from evaluation_data/'s root since 2026-09-16,
matching the existing baseline_v1/ and ubfc_phys/ subfolder convention, so
the folder isn't a flat pile of dozens of CSVs). Anything that fails stays
in the staging folder untouched, with the specific reason(s) printed.

Pure file/CSV processing - no camera, no Streamlit, no TensorFlow/MediaPipe.

Usage
-----
    cd code && python recording_intake.py
    python recording_intake.py --dry-run
    python recording_intake.py --incoming ../evaluation_data/incoming --dest ../evaluation_data/live_recordings
"""

import argparse
import csv
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import config
from app_naming import parse_filename, VALID_CONDITIONS

# RECORDING_PROTOCOL.md, "Before you start": "Aim for 60 seconds minimum."
# Not the same constant as config.INTERVENTION_MIN_EVAL_SECONDS, which
# happens to share the value 60.0 but governs an unrelated thing (the live
# intervention agent's outcome-evaluation window) - conflating the two would
# make an unrelated future change to intervention timing silently move this
# dataset-intake rule too.
MIN_DURATION_SECONDS = 60.0

# Sourced from config.py (not stress_fusion.py) so this stays stdlib-only -
# stress_fusion.py pulls in numpy, which this lightweight, camera-free
# script has no other reason to need.
VALID_MODES = {mode.lower() for mode in config.TASK_MODES.keys()}

# export_filename()'s deliberate fallback for an unlabelled export (see
# app_naming.py) - "stress_session_<YYYY-MM-DD>_<HHMMSS>.csv". Matched
# separately from the generic "doesn't parse" case so a recorder who forgot
# to set Subject ID/Condition before exporting gets told exactly that,
# rather than a generic-looking parse failure that reads like corruption.
UNLABELLED_RE = re.compile(r'^stress_session_\d{4}-\d{2}-\d{2}_\d{6}\.csv$', re.IGNORECASE)


def _read_rows_and_timestamps(path):
    """
    Read one exported session CSV (see app.py's history_to_csv() for the
    exact header: ts, stress_index, level, daisee, emotion, behavior) and
    return (rows, ts_values) - the raw DictReader rows, and whichever 'ts'
    values parsed as ISO timestamps, in file order.
    """
    rows = []
    ts_values = []
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            raw_ts = r.get("ts")
            if raw_ts:
                try:
                    ts_values.append(datetime.fromisoformat(raw_ts))
                except ValueError:
                    pass
    return rows, ts_values


def _duration_seconds(ts_values):
    """
    Same measure session_history.summarize_history() uses for a live
    session: last timestamp minus first, in the file's own row order (the
    app always writes frames chronologically). A single timestamp still
    yields a defined duration (0.0s) rather than "unknown" - one frame is a
    real, measurably-too-short recording, not a missing-column failure.
    """
    if not ts_values:
        return None
    if len(ts_values) == 1:
        return 0.0
    return (ts_values[-1] - ts_values[0]).total_seconds()


def validate_file(path, dest=None):
    """
    Validate one staged CSV. Returns (meta, reasons):
      meta    - the parsed {condition, subject, mode, date} dict, or None
                if the filename didn't parse.
      reasons - list of human-readable failure strings; empty means the
                file passed every check.

    `dest`, when given, adds a same-name-already-there collision check so a
    re-run never silently overwrites a file already accepted into the
    evaluation dataset.
    """
    reasons = []

    meta = parse_filename(path.name)
    if meta is None:
        if UNLABELLED_RE.match(path.name):
            reasons.append("labels not set in the app - re-export with Subject ID "
                            "and Condition filled in")
        else:
            reasons.append("filename does not match the canonical "
                            "{condition}_{subject}_{mode}_{date}.csv scheme "
                            f"(check for a typo): '{path.name}'")
    else:
        # Implied by a successful parse (parse_filename only accepts
        # parts[0] in VALID_CONDITIONS) - checked explicitly anyway rather
        # than trusted silently, per this project's own "loud and specific"
        # validation ethos (see find_unclaimed_files() in
        # run_all_evaluations.py).
        if meta["condition"] not in VALID_CONDITIONS:
            reasons.append(f"condition '{meta['condition']}' is not 'relaxed' or 'stressed'")
        if meta["mode"] not in VALID_MODES:
            reasons.append(f"mode '{meta['mode']}' is not one of "
                            f"{sorted(VALID_MODES)}")
        if dest is not None and (Path(dest) / path.name).exists():
            reasons.append(f"a file named '{path.name}' already exists in {dest} "
                            "- resolve the collision manually")

    rows, ts_values = _read_rows_and_timestamps(path)
    if not rows:
        reasons.append("file is empty or header-only - no recorded frames")
    else:
        duration = _duration_seconds(ts_values)
        if duration is None:
            reasons.append("could not read the 'ts' column to measure duration")
        elif duration < MIN_DURATION_SECONDS:
            reasons.append(f"recording too short: {duration:.1f}s measured, "
                            f"{MIN_DURATION_SECONDS:.0f}s minimum "
                            "(RECORDING_PROTOCOL.md)")

    return meta, reasons


def run_intake(incoming, dest, dry_run=False):
    """
    Scan `incoming` for *.csv files, validate each, and (unless dry_run)
    move the ones that pass every check into `dest`. Returns (moved,
    flagged): moved is [(name, meta), ...], flagged is [(name, reasons), ...].
    """
    incoming = Path(incoming)
    dest = Path(dest)

    moved = []
    flagged = []
    for path in sorted(incoming.glob("*.csv")):
        meta, reasons = validate_file(path, dest=dest)
        if reasons:
            flagged.append((path.name, reasons))
            continue
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest / path.name))
        moved.append((path.name, meta))

    return moved, flagged


def print_summary(incoming, dest, dry_run, moved, flagged):
    print("=" * 78)
    print("RECORDING INTAKE")
    print("=" * 78)
    print(f"staging : {incoming}")
    print(f"dest    : {dest}")
    if dry_run:
        print("mode    : DRY RUN - nothing will be moved")
    print()

    total = len(moved) + len(flagged)
    if total == 0:
        print("no CSV files found in the staging folder - nothing to do")
    else:
        verb = "would be moved" if dry_run else "moved"
        print(f"{len(moved)} file(s) {verb} into {dest}:")
        for name, meta in moved:
            print(f"  {name:45s} subject={meta['subject']:6s} "
                  f"condition={meta['condition']:9s} mode={meta['mode']:9s} "
                  f"date={meta['date']}")
        print()
        print(f"{len(flagged)} file(s) flagged - left in {incoming}:")
        for name, reasons in flagged:
            print(f"  {name}")
            for reason in reasons:
                print(f"    - {reason}")

    print()
    print("Next step (not run automatically):")
    print("cd code && python run_all_evaluations.py")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incoming", default="../evaluation_data/incoming",
                     help="staging folder to scan (default: ../evaluation_data/incoming, "
                          "i.e. when run from code/)")
    ap.add_argument("--dest", default="../evaluation_data/live_recordings",
                     help="destination folder for validated CSVs "
                          "(default: ../evaluation_data/live_recordings)")
    ap.add_argument("--dry-run", action="store_true",
                     help="validate and report only - never move anything")
    args = ap.parse_args()

    incoming = Path(args.incoming)
    dest = Path(args.dest)

    if not incoming.exists():
        incoming.mkdir(parents=True, exist_ok=True)
        keep = incoming / ".gitkeep"
        if not keep.exists():
            keep.write_text(
                "Drop freshly-exported session CSVs here. python recording_intake.py "
                "validates them and moves the good ones into evaluation_data/.\n"
            )
        print(f"created staging folder: {incoming}")
        print()

    moved, flagged = run_intake(incoming, dest, dry_run=args.dry_run)
    print_summary(incoming, dest, args.dry_run, moved, flagged)

    return 0


if __name__ == "__main__":
    sys.exit(main())
