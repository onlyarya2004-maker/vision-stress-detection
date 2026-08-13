"""
Per-signal diagnostics
======================
Answers the question the fused stress index cannot: *which behavioural signal
is actually doing anything?*

This exists because the first cross-subject run produced a flat output -
relaxed averaged 0.436, stressed 0.445 - with no way to see why. Breaking the
fusion open showed 0.65 of the weight budget sitting on signals that were
ANTI-correlated with stress on that subject, while blink_stress (d = +3.94,
weight 0.15) was carrying the discrimination almost alone.

Reads the per-frame feature traces written by evaluate_video_file.py.

Usage
-----
    # one subject's traces
    python diagnose_features.py --dir ../datasets/feature_traces

    # compare two runs (e.g. before vs after a calibration change)
    python diagnose_features.py --dir ../datasets/feature_traces \
        --compare ../datasets/feature_traces_BEFORE_calibfix
"""

import argparse
import csv
import re
import statistics as st
import sys
from pathlib import Path

import config

SIGNALS = ['eye_strain', 'blink_stress', 'brow_tension', 'jaw_tension',
           'head_instability', 'neck_strain', 'slouch']

# UBFC-Phys: T1 is the rest task, T2 (speech) and T3 (arithmetic) are stressors
TASK_RE = re.compile(r'_T(\d)', re.IGNORECASE)


def load(path):
    return list(csv.DictReader(open(path)))


def col(rows, key):
    """Numeric column, skipping blanks (the CNN does not run every frame)."""
    out = []
    for r in rows:
        v = r.get(key, '')
        if v not in ('', 'None', None):
            try:
                out.append(float(v))
            except ValueError:
                pass
    return out


def mean(rows, key):
    v = col(rows, key)
    return st.mean(v) if v else float('nan')


def pstdev(rows, key):
    v = col(rows, key)
    return st.pstdev(v) if len(v) > 1 else 0.0


def cohens_d(rest, stressed, key):
    a, b = col(rest, key), col(stressed, key)
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    pooled = ((st.pstdev(a) ** 2 + st.pstdev(b) ** 2) / 2) ** 0.5
    if pooled < 1e-9:
        return 0.0
    return (st.mean(b) - st.mean(a)) / pooled


def verdict(d, rest_mean, pooled_sd):
    if pooled_sd < 1e-6:
        return "DEAD (no variance)"
    if rest_mean > 0.95:
        return "SATURATED at rest"
    if d != d:                      # NaN
        return "n/a"
    if d <= -0.2:
        return "INVERTED"
    if abs(d) < 0.2:
        return "useless"
    if d < 0.5:
        return "weak"
    if d < 1.0:
        return "useful"
    return "STRONG"


def split_by_task(directory):
    """Group traces into (rest_rows, stressed_rows) using UBFC T1/T2/T3."""
    rest, stressed, files = [], [], []
    for p in sorted(Path(directory).glob('features_*.csv')):
        m = TASK_RE.search(p.stem)
        rows = load(p)
        files.append((p.name, len(rows)))
        if not m:
            continue
        (rest if m.group(1) == '1' else stressed).extend(rows)
    return rest, stressed, files


def report(directory, label):
    rest, stressed, files = split_by_task(directory)
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    for name, n in files:
        print(f"  {name:34s} {n:>6} frames")
    if not rest or not stressed:
        print("  need both a T1 trace and at least one T2/T3 trace")
        return None

    print(f"\n  rest frames {len(rest)}  |  stressed frames {len(stressed)}")
    print(f"\n  {'signal':18s}{'wt':>5}{'rest':>8}{'stress':>8}"
          f"{'d':>8}   verdict")
    print("  " + "-" * 62)

    results = {}
    for s in SIGNALS:
        rm, sm = mean(rest, s), mean(stressed, s)
        d = cohens_d(rest, stressed, s)
        pooled = ((pstdev(rest, s) ** 2 + pstdev(stressed, s) ** 2) / 2) ** 0.5
        v = verdict(d, rm, pooled)
        results[s] = {'rest': rm, 'stressed': sm, 'd': d, 'verdict': v}
        w = config.BEHAVIOR_WEIGHTS[s]
        print(f"  {s:18s}{w:>5.2f}{rm:>8.3f}{sm:>8.3f}{d:>+8.2f}   {v}")

    bad = sum(config.BEHAVIOR_WEIGHTS[s] for s, r in results.items()
              if r['verdict'] in ('INVERTED', 'DEAD (no variance)',
                                  'SATURATED at rest'))
    good = sum(config.BEHAVIOR_WEIGHTS[s] for s, r in results.items()
               if r['verdict'] in ('useful', 'STRONG'))
    print(f"\n  weight on inverted/dead/saturated : {bad:.2f}")
    print(f"  weight on useful/strong           : {good:.2f}")
    if bad > good:
        print("  -> the weighting is working AGAINST the signal on this data.")

    print(f"\n  fused: rest {mean(rest, 'stress_index_smoothed'):.3f}  "
          f"stressed {mean(stressed, 'stress_index_smoothed'):.3f}  "
          f"gap {mean(stressed, 'stress_index_smoothed') - mean(rest, 'stress_index_smoothed'):+.3f}")
    return results


def compare(after, before):
    print(f"\n{'=' * 74}\nBEFORE vs AFTER\n{'=' * 74}")
    print(f"  {'signal':18s}{'rest before':>13}{'rest after':>12}"
          f"{'d before':>10}{'d after':>9}")
    print("  " + "-" * 62)
    for s in SIGNALS:
        b, a = before.get(s), after.get(s)
        if not b or not a:
            continue
        print(f"  {s:18s}{b['rest']:>13.3f}{a['rest']:>12.3f}"
              f"{b['d']:>+10.2f}{a['d']:>+9.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True, help='feature trace directory')
    ap.add_argument('--compare', help='a second directory to compare against')
    args = ap.parse_args()

    after = report(args.dir, f"CURRENT  ({args.dir})")
    if args.compare:
        before = report(args.compare, f"BASELINE ({args.compare})")
        if after and before:
            compare(after, before)
    return 0


if __name__ == '__main__':
    sys.exit(main())
