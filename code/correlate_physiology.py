"""
Stress index vs physiology (Phase 1.4)
======================================
Correlates the system's continuous stress index against UBFC-Phys's recorded
EDA (electrodermal activity) - the classic sympathetic-arousal signal.

This asks a harder question than the accuracy tables: not "does the index
clear a threshold on stressed clips", but "does it MOVE WITH measured
physiology". Two levels, because they can disagree:

  WITHIN-task  - per (subject, task): Pearson/Spearman between the smoothed
                 index and EDA over the 3 minutes. Honest caveat printed with
                 the results: EDA drifts slowly and is autocorrelated, so
                 within-task r is indicative, not inferential.

  BETWEEN-task - per subject: do the task MEANS move together? If EDA rises
                 from rest to stressor, does the index rise too? This is the
                 claim the system actually makes, and with 3 tasks per
                 subject it reduces to direction agreement.

Inputs are all already on disk (video long deleted):
  datasets/feature_traces/features_vid_s{N}_T{K}.csv   stress index over time
  datasets/UBFC-Phys/s{N}/s{N}/eda_s{N}_T{K}.csv       EDA @ 4 Hz

Phase 4.4: run_correlation_study()/render() below are called directly from
run_all_evaluations.py, so this is folded into the one regenerated canonical
file rather than living only in its own standalone report. Still runnable on
its own too (same functions, same output) - see Usage.

Usage
-----
    cd code && python correlate_physiology.py
    python correlate_physiology.py --traces ../datasets/feature_traces \
        --ubfc ../datasets/UBFC-Phys
    python correlate_physiology.py --out ../evaluation_data/CORRELATION.txt
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

EDA_FS = 4.0                      # Empatica E4 EDA sample rate
NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)


def load_index(path):
    """(t, smoothed stress index) from a feature trace, sorted by t.

    np.interp() in align() silently misbehaves on unsorted input - the same
    schema's loader in ablation_study.py (load_trace()) already sorts for
    exactly this reason. Currently safe here too (the only writer produces
    monotonic rows), but sorting explicitly rather than relying on that
    holding forever costs nothing.
    """
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            rows.append((float(r['t']), float(r['stress_index_smoothed'])))
        except (KeyError, ValueError):
            continue
    rows.sort(key=lambda row: row[0])
    if not rows:
        return np.array([]), np.array([])
    t, x = zip(*rows)
    return np.array(t), np.array(x)


def load_eda(path):
    """
    One EDA reading per line. Skips any line that doesn't parse (a header
    row, a dropout marker, a truncated download) rather than crashing -
    matching load_index()'s pattern. A single bad EDA file used to abort
    the whole run and lose every already-computed subject's results with
    it, since output was only ever written once, at the very end.
    """
    vals = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            vals.append(float(l))
        except ValueError:
            continue
    return np.array(vals)


def align(t_idx, x_idx, eda):
    """
    Resample the stress index onto the EDA clock (4 Hz). The index trace
    starts after the calibration window, so EDA is trimmed to the overlap.
    """
    if len(eda) == 0:
        return None, None
    t_eda = np.arange(len(eda)) / EDA_FS
    lo, hi = max(t_idx[0], t_eda[0]), min(t_idx[-1], t_eda[-1])
    keep = (t_eda >= lo) & (t_eda <= hi)
    if keep.sum() < 20:
        return None, None
    x_on_eda = np.interp(t_eda[keep], t_idx, x_idx)
    return x_on_eda, eda[keep]


# ---------------------------------------------------------------------------
# Computation - separated from rendering so run_all_evaluations.py can call
# this directly and fold the result into the one canonical file, instead of
# this living only in its own separately-generated report.
# ---------------------------------------------------------------------------
def run_correlation_study(traces_dir, ubfc_dir):
    """
    Returns a results dict, or None if no feature traces are found (e.g. the
    dataset-derived traces aren't present in this checkout - same contract
    as ablation_study.run_ablation_study()).

    per_trace entries only exist for traces that produced SOME output line
    (no-EDA-file, NaN-dropout, or a real correlation) - a trace skipped for
    being too short or having no clock overlap with its EDA file produces no
    entry, matching the original script's silent `continue`.
    """
    traces = sorted(Path(traces_dir).glob('features_*.csv'))
    if not traces:
        return None

    per_trace = []
    per_subject = {}
    for tr in traces:
        m = NAME_RE.search(tr.stem)
        if not m:
            continue
        subj, task = int(m.group(1)), int(m.group(2))
        label = f"s{subj}_T{task}"
        eda_path = (Path(ubfc_dir) / f"s{subj}" / f"s{subj}"
                    / f"eda_s{subj}_T{task}.csv")
        if not eda_path.exists():
            per_trace.append({'label': label, 'status': 'no_eda'})
            continue

        t_idx, x_idx = load_index(tr)
        if len(t_idx) < 20:
            continue
        x, e = align(t_idx, x_idx, load_eda(eda_path))
        if x is None:
            continue

        # NaN std (an unparsed dropout marker survived as a literal "nan")
        # must not look identical to genuinely-zero variance, so it's
        # called out explicitly instead of falling through to the same
        # blank-looking "N/A" the zero-variance case would print.
        sx, se = np.std(x), np.std(e)
        if np.isnan(sx) or np.isnan(se):
            per_trace.append({'label': label, 'status': 'nan', 'n': len(x)})
            continue
        pr = stats.pearsonr(x, e)[0] if sx > 0 and se > 0 else np.nan
        sr = stats.spearmanr(x, e)[0] if sx > 0 and se > 0 else np.nan
        per_trace.append({'label': label, 'status': 'ok', 'n': len(x),
                          'pearson': float(pr), 'spearman': float(sr)})

        d = per_subject.setdefault(subj, {})
        d[task] = {'idx_mean': float(np.mean(x)), 'eda_mean': float(np.mean(e))}

    # ---- between-task: does the index move the way EDA moves? ----
    agreement_rows = []
    agree = total = 0
    for subj in sorted(per_subject):
        d = per_subject[subj]
        if 1 not in d or not ({2, 3} & set(d)):
            continue
        eda_rest = d[1]['eda_mean']
        idx_rest = d[1]['idx_mean']
        eda_stress = float(np.mean([d[t]['eda_mean'] for t in (2, 3) if t in d]))
        idx_stress = float(np.mean([d[t]['idx_mean'] for t in (2, 3) if t in d]))
        same = (eda_stress - eda_rest) * (idx_stress - idx_rest) > 0
        agree += int(same)
        total += 1
        agreement_rows.append({
            'subject': subj, 'eda_rest': eda_rest, 'eda_stress': eda_stress,
            'idx_rest': idx_rest, 'idx_stress': idx_stress, 'agree': same,
        })

    return {
        'traces_dir': Path(traces_dir).resolve(),
        'ubfc_dir': Path(ubfc_dir).resolve(),
        'per_trace': per_trace,
        'agreement_rows': agreement_rows,
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
    out.append("\n" + "=" * 74)
    out.append("STRESS INDEX vs EDA  (UBFC-Phys ground truth)")
    out.append("=" * 74)
    if results is None:
        out.append("  (no feature traces found under datasets/feature_traces/ - skipped)")
        return

    if generated_line:
        out.append(f"  {generated_line}")
    out.append(f"  traces : {results['traces_dir']}")
    out.append(f"  ubfc   : {results['ubfc_dir']}")
    out.append("")
    out.append("Within-task r is indicative only - EDA drifts and both series are")
    out.append("autocorrelated. The between-task direction test is the real claim.")
    out.append("")

    out.append(f"  {'trace':22s}{'n':>6}{'Pearson':>10}{'Spearman':>10}")
    out.append("  " + "-" * 50)
    for row in results['per_trace']:
        label = row['label']
        if row['status'] == 'no_eda':
            out.append(f"  {label:<22} (no EDA file)")
        elif row['status'] == 'nan':
            out.append(f"  {label:<22}{row['n']:>6}  (NaN in EDA - dropout marker?)")
        else:
            out.append(f"  {label:<22}{row['n']:>6}{row['pearson']:>+10.2f}"
                       f"{row['spearman']:>+10.2f}")

    out.append("")
    out.append("  BETWEEN-TASK (rest T1 -> stressors T2/T3): direction agreement")
    out.append(f"  {'subject':10s}{'EDA rest':>10}{'EDA stress':>12}"
               f"{'idx rest':>10}{'idx stress':>12}   agree?")
    out.append("  " + "-" * 62)
    for row in results['agreement_rows']:
        out.append(f"  s{row['subject']:<9}{row['eda_rest']:>10.3f}{row['eda_stress']:>12.3f}"
                   f"{row['idx_rest']:>10.3f}{row['idx_stress']:>12.3f}   "
                   f"{'YES' if row['agree'] else 'NO'}")

    if results['total']:
        out.append(f"\n  direction agreement: {results['agree']}/{results['total']} subjects")
        out.append("  (chance is ~50%; this is the number that grows meaningful as")
        out.append("   more subjects are processed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', default='../datasets/feature_traces')
    ap.add_argument('--ubfc', default='../datasets/UBFC-Phys')
    ap.add_argument('--root', default='..',
                    help="project root (default: .. when run from code/)")
    ap.add_argument('--out', default=None,
                    help="output file (default: <root>/evaluation_data/"
                         "CORRELATION_RESULTS_<date>.txt)")
    args = ap.parse_args()

    results = run_correlation_study(args.traces, args.ubfc)
    if results is None:
        print("no feature traces found")
        return 1

    out = []
    render(results, out,
          generated_line=f"generated : {datetime.now().isoformat(timespec='seconds')}")
    text = "\n".join(out)
    print(text)

    root = Path(args.root).resolve()
    out_path = (Path(args.out) if args.out else
                root / "evaluation_data" / f"CORRELATION_RESULTS_{datetime.now():%Y-%m-%d}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n\nwritten to: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
