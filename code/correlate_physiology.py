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

Usage
-----
    python correlate_physiology.py
    python correlate_physiology.py --traces ../datasets/feature_traces \
        --ubfc ../datasets/UBFC-Phys
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

EDA_FS = 4.0                      # Empatica E4 EDA sample rate
NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)


def load_index(path):
    """(t, smoothed stress index) from a feature trace."""
    t, x = [], []
    for r in csv.DictReader(open(path)):
        try:
            t.append(float(r['t']))
            x.append(float(r['stress_index_smoothed']))
        except (KeyError, ValueError):
            continue
    return np.array(t), np.array(x)


def load_eda(path):
    return np.array([float(l) for l in open(path) if l.strip()])


def align(t_idx, x_idx, eda):
    """
    Resample the stress index onto the EDA clock (4 Hz). The index trace
    starts after the calibration window, so EDA is trimmed to the overlap.
    """
    t_eda = np.arange(len(eda)) / EDA_FS
    lo, hi = max(t_idx[0], t_eda[0]), min(t_idx[-1], t_eda[-1])
    keep = (t_eda >= lo) & (t_eda <= hi)
    if keep.sum() < 20:
        return None, None
    x_on_eda = np.interp(t_eda[keep], t_idx, x_idx)
    return x_on_eda, eda[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', default='../datasets/feature_traces')
    ap.add_argument('--ubfc', default='../datasets/UBFC-Phys')
    args = ap.parse_args()

    traces = sorted(Path(args.traces).glob('features_*.csv'))
    if not traces:
        print("no feature traces found")
        return 1

    print("=" * 74)
    print("STRESS INDEX vs EDA  (UBFC-Phys ground truth)")
    print("=" * 74)
    print("Within-task r is indicative only - EDA drifts and both series are")
    print("autocorrelated. The between-task direction test is the real claim.\n")

    per_subject = {}
    print(f"  {'trace':22s}{'n':>6}{'Pearson':>10}{'Spearman':>10}")
    print("  " + "-" * 50)
    for tr in traces:
        m = NAME_RE.search(tr.stem)
        if not m:
            continue
        subj, task = int(m.group(1)), int(m.group(2))
        eda_path = (Path(args.ubfc) / f"s{subj}" / f"s{subj}"
                    / f"eda_s{subj}_T{task}.csv")
        if not eda_path.exists():
            print(f"  s{subj}_T{task:<18} (no EDA file)")
            continue

        t_idx, x_idx = load_index(tr)
        if len(t_idx) < 20:
            continue
        x, e = align(t_idx, x_idx, load_eda(eda_path))
        if x is None:
            continue

        pr = stats.pearsonr(x, e)[0] if np.std(x) > 0 and np.std(e) > 0 else np.nan
        sr = stats.spearmanr(x, e)[0] if np.std(x) > 0 and np.std(e) > 0 else np.nan
        print(f"  s{subj}_T{task:<18}{len(x):>6}{pr:>+10.2f}{sr:>+10.2f}")

        d = per_subject.setdefault(subj, {})
        d[task] = {'idx_mean': float(np.mean(x)), 'eda_mean': float(np.mean(e))}

    # ---- between-task: does the index move the way EDA moves? ----
    print("\n  BETWEEN-TASK (rest T1 -> stressors T2/T3): direction agreement")
    print(f"  {'subject':10s}{'EDA rest':>10}{'EDA stress':>12}"
          f"{'idx rest':>10}{'idx stress':>12}   agree?")
    print("  " + "-" * 62)
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
        print(f"  s{subj:<9}{eda_rest:>10.3f}{eda_stress:>12.3f}"
              f"{idx_rest:>10.3f}{idx_stress:>12.3f}   {'YES' if same else 'NO'}")

    if total:
        print(f"\n  direction agreement: {agree}/{total} subjects")
        print("  (chance is ~50%; this is the number that grows meaningful as")
        print("   more subjects are processed - target s15-s20)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
