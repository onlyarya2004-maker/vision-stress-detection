"""
Evaluation-CSV naming
=====================
Kept out of app.py so it can be imported and tested without pulling in
Streamlit (importing app.py would execute st.set_page_config and the whole
page body).

The scheme is load-bearing: run_all_evaluations.py locates recordings by
globbing relaxed_*/stressed_*, so a filename that does not match means the
recording silently contributes nothing to any reported number. PP1 lost 48
valid relaxed samples exactly this way, to a file typo'd as 'relaaxed_2.csv'.
"""

from datetime import datetime

VALID_CONDITIONS = ('relaxed', 'stressed')


def export_filename(subject, condition, task_mode, when=None):
    """
    Build the canonical evaluation-CSV filename:

        {condition}_{subject}_{mode}_{YYYY-MM-DD}.csv
        e.g. relaxed_S01_default_2026-08-12.csv

    Condition comes first so the harness's existing relaxed_*/stressed_*
    globs keep matching without modification.

    If the labels are not properly set, this deliberately returns a
    timestamped 'stress_session_*' name that does NOT match those globs, so
    an unlabelled recording is obviously unlabelled rather than quietly
    mislabelled as ground truth.
    """
    when = when or datetime.now()
    # Underscores are the field separator, so one inside subject/mode would
    # produce a name parse_filename can't invert (see its own guard) -
    # strip them here too so the app's own export path can never generate
    # a filename it can't later parse back correctly.
    subject = (subject or '').strip().replace(' ', '').replace('_', '')
    condition = (condition or '').strip().lower()
    task_mode = str(task_mode).strip().lower().replace('_', '')

    if condition not in VALID_CONDITIONS or not subject:
        return f"stress_session_{when:%Y-%m-%d_%H%M%S}.csv"

    return f"{condition}_{subject}_{task_mode}_{when:%Y-%m-%d}.csv"


def parse_filename(name):
    """
    Inverse of export_filename: recover (condition, subject, mode) from a
    canonical name, or None if it is not one. Used by the evaluation harness
    to group results per subject for multi-subject reporting.
    """
    stem = name[:-4] if name.lower().endswith('.csv') else name
    parts = stem.split('_')
    # Exactly 4, not "at least 4": a subject or mode containing an
    # underscore produces >4 parts, which used to be silently accepted
    # with every field after condition shifted/truncated - a subject "S_01"
    # would parse as subject="S", mode="01", date="default", corrupting the
    # per-subject breakdown table with no error. Reject it instead, so it's
    # obviously unparsed rather than quietly wrong - same principle as the
    # unlabelled-recording fallback below.
    if len(parts) != 4 or parts[0].lower() not in VALID_CONDITIONS:
        return None
    return {'condition': parts[0].lower(), 'subject': parts[1],
            'mode': parts[2].lower(), 'date': parts[3]}
