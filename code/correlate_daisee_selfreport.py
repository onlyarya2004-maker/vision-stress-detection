"""
DaiSEE branch construct-validity check (Phase 6.1)
===================================================
Checks whether the CNN branch's own output tracks what the user actually
reported feeling, via the self-report loop (Phase 3.1) - the closest thing
this project has to independent stress ground truth. DaiSEE's own labels
are affect categories (engagement/frustration/confusion/boredom), not
stress itself, and the weighted formula that converts those into a binary
training label (see daisee_preprocessing.py / rebucket_to_binary.py) has
never been checked against real stress ground truth, because DaiSEE has
none to check it against.

This does NOT validate that training-label formula in isolation - that
would need a dataset with both video and independent stress ground truth,
which doesn't exist. What it DOES check is the construct-validity question
that actually matters for this system: does the CNN branch's live output
correlate with genuine self-reported stress.

Two views, same shape as correlate_physiology.py's precedent (Phase 4.4):
  CONTINUOUS - Pearson/Spearman between each session's mean DaiSEE stress
               probability (the raw, un-baseline-shifted 'daisee' column -
               see stress_fusion.daisee_raw_score(), a value in [0,1]) and
               the raw 1-5 self-report rating.
  BINARY     - does the CNN branch's own implied call (mean probability
               > 0.5 -> "stressed") agree with the self-reported label
               (1-2 -> relaxed, 4-5 -> stressed, 3 excluded as ambiguous -
               the same mapping self-report uses everywhere else it's
               binarized, e.g. retrain_from_sessions.py)?

Needs >= 2 labeled sessions to produce a real number (correlation is
undefined on 0 or 1 points). Grows automatically as normal app use
accumulates self-reports - no dedicated recording session required.

Phase 6.1: run_daisee_selfreport_study()/render() below are called
directly from run_all_evaluations.py, folded into the one canonical file
rather than living only in a separately-generated report - same pattern
as correlate_physiology.py. Still runnable on its own too.

Usage
-----
    cd code && python correlate_daisee_selfreport.py
    python correlate_daisee_selfreport.py --min-samples 10
    python correlate_daisee_selfreport.py --out ../evaluation_data/DAISEE_SELFREPORT.txt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

import session_history

STRESSED_MIN = 4   # self-report >= 4 -> stressed
RELAXED_MAX = 2    # self-report <= 2 -> relaxed
NEUTRAL = 3        # excluded from the binary agreement check as ambiguous


def _session_mean_daisee(session_id):
    """Mean per-frame DaiSEE stress probability for one session, or None
    if the session has no samples (shouldn't happen for a real saved
    session, but load_session_samples() can return [] if it does)."""
    samples = session_history.load_session_samples(session_id)
    if not samples:
        return None
    return float(np.mean([s['daisee'] for s in samples]))


# ---------------------------------------------------------------------------
# Computation - separated from rendering so run_all_evaluations.py can call
# this directly and fold the result into the one canonical file, instead of
# this living only in its own separately-generated report.
# ---------------------------------------------------------------------------
def run_daisee_selfreport_study(min_samples=10):
    """
    Returns a results dict. `insufficient` is True when fewer than 2
    labeled sessions have usable samples - correlation is undefined on 0
    or 1 points, same not-enough-data contract as
    ablation_study.run_ablation_study() and
    correlate_physiology.run_correlation_study().

    init_db() is called first so this is safe to run standalone against a
    fresh checkout where the app has never been launched (no sessions
    table yet) - without it, labeled_sessions() would raise
    sqlite3.OperationalError instead of just reporting zero sessions.
    """
    session_history.init_db()

    sessions = session_history.labeled_sessions(min_samples=min_samples)
    rows = []
    for sess in sessions:
        mean_daisee = _session_mean_daisee(sess['id'])
        if mean_daisee is None:
            continue
        rows.append({
            'session_id': sess['id'],
            'self_report': int(sess['self_report']),
            'mean_daisee': mean_daisee,
            'n_frames': sess['sample_count'],
        })

    if len(rows) < 2:
        return {'rows': rows, 'insufficient': True}

    daisee_vals = np.array([r['mean_daisee'] for r in rows])
    report_vals = np.array([r['self_report'] for r in rows], dtype=float)

    # Constant input (e.g. every session had the same self-report rating)
    # makes pearsonr/spearmanr's correlation undefined - guard explicitly
    # rather than let scipy emit a ConstantInputWarning and return NaN
    # implicitly, matching correlate_physiology.py's own precedent.
    sd, sr = np.std(daisee_vals), np.std(report_vals)
    pearson = float(stats.pearsonr(daisee_vals, report_vals)[0]) if sd > 0 and sr > 0 else np.nan
    spearman = float(stats.spearmanr(daisee_vals, report_vals)[0]) if sd > 0 and sr > 0 else np.nan

    agree = total = 0
    for r in rows:
        if r['self_report'] == NEUTRAL:
            continue
        reported_stressed = r['self_report'] >= STRESSED_MIN
        predicted_stressed = r['mean_daisee'] > 0.5
        agree += int(reported_stressed == predicted_stressed)
        total += 1

    return {
        'rows': rows,
        'insufficient': False,
        'pearson': pearson,
        'spearman': spearman,
        'agree': agree,
        'total': total,
    }


def render(results, out, generated_line=None):
    """
    generated_line: a "generated : <timestamp>" line to include, for
    standalone use (main() below). Omitted when called from
    run_all_evaluations.py, which already stamps one timestamp for the
    whole canonical file - no other section repeats its own.
    """
    out.append("\n" + "=" * 78)
    out.append("DAISEE BRANCH CONSTRUCT VALIDITY vs SELF-REPORT (Phase 6.1)")
    out.append("=" * 78)
    if generated_line:
        out.append(f"  {generated_line}")
    out.append("  This checks whether the CNN branch's own output tracks genuine")
    out.append("  self-reported stress. It does NOT validate DaiSEE's own training-label")
    out.append("  formula in isolation - no dataset exists with both video and independent")
    out.append("  stress ground truth to check that against.")
    out.append("")

    n = len(results['rows'])
    if n == 0:
        out.append("  (no labeled sessions found - answer the end-of-session self-report")
        out.append("   prompt to start accumulating data here)")
        return

    if results['insufficient']:
        out.append(f"  Only {n} labeled session(s) so far - correlation needs >= 2.")
        out.append("  Per-session data collected so far:")
        for r in results['rows']:
            out.append(f"    session {r['session_id']:>4}  self_report={r['self_report']}"
                       f"  mean_daisee={r['mean_daisee']:.3f}  n_frames={r['n_frames']}")
        return

    out.append(f"  {'session':>9}{'self_report':>13}{'mean_daisee':>13}{'n_frames':>10}")
    out.append("  " + "-" * 47)
    for r in results['rows']:
        out.append(f"  {r['session_id']:>9}{r['self_report']:>13}{r['mean_daisee']:>13.3f}"
                   f"{r['n_frames']:>10}")

    out.append("")
    out.append(f"  n = {n} labeled sessions")
    out.append(f"  Pearson  r = {results['pearson']:+.3f}")
    out.append(f"  Spearman r = {results['spearman']:+.3f}")
    out.append("  CAUTION: n this small means these correlations are indicative only,")
    out.append("  not inferential - they will firm up as more sessions accumulate.")

    out.append("")
    if results['total']:
        pct = 100.0 * results['agree'] / results['total']
        out.append("  Binary agreement (CNN branch mean-probability>0.5 'stressed' call vs")
        out.append(f"  self-reported label, rating {NEUTRAL} excluded as ambiguous): "
                   f"{results['agree']}/{results['total']} = {pct:.1f}%")
        out.append("  (chance is ~50%)")
    else:
        out.append(f"  (every labeled session so far is rating {NEUTRAL} - no binary")
        out.append("   agreement check possible yet)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-samples', type=int, default=10,
                    help="minimum frame count for a session to count (default: 10, "
                         "matches session_history.labeled_sessions()'s own default)")
    ap.add_argument('--root', default='..',
                    help="project root (default: .. when run from code/)")
    ap.add_argument('--out', default=None,
                    help="output file (default: <root>/evaluation_data/"
                         "DAISEE_SELFREPORT_RESULTS_<date>.txt)")
    args = ap.parse_args()

    results = run_daisee_selfreport_study(min_samples=args.min_samples)

    out = []
    render(results, out,
          generated_line=f"generated : {datetime.now().isoformat(timespec='seconds')}")
    text = "\n".join(out)
    print(text)

    root = Path(args.root).resolve()
    out_path = (Path(args.out) if args.out else
                root / "evaluation_data" / f"DAISEE_SELFREPORT_RESULTS_{datetime.now():%Y-%m-%d}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n\nwritten to: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
