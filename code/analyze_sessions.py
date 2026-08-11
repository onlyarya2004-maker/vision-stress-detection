"""
Self-evaluation analysis
========================
Reads relaxed_*.csv and stressed_*.csv files exported from the live app and
reports how well the system distinguishes the two states.

Place all CSVs in the same folder as this script (or pass --dir).

Usage:
    python analyze_sessions.py
    python analyze_sessions.py --dir /path/to/csvs
"""

import argparse
import csv
import statistics
from pathlib import Path
from collections import Counter


def load_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    'stress_index': float(r['stress_index']),
                    'level':        r['level'],
                    'daisee':       float(r['daisee']),
                    'emotion':      float(r['emotion']),
                    'behavior':     float(r['behavior']),
                })
            except (KeyError, ValueError):
                continue
    return rows


def summarize(rows, label):
    if not rows:
        print(f"  {label}: no data")
        return None
    si = [r['stress_index'] for r in rows]
    da = [r['daisee']       for r in rows]
    be = [r['behavior']     for r in rows]
    levels = Counter(r['level'] for r in rows)
    print(f"  {label}: n={len(rows):>4}  "
          f"stress_idx={statistics.mean(si):.3f}  "
          f"daisee={statistics.mean(da):.3f}  "
          f"behavior={statistics.mean(be):.3f}  "
          f"levels={dict(levels)}")
    return {
        'n':            len(rows),
        'stress_index': statistics.mean(si),
        'daisee':       statistics.mean(da),
        'behavior':     statistics.mean(be),
        'levels':       dict(levels),
    }


def evaluate(relaxed_rows, stressed_rows):
    """
    "Correct classification" definitions:
      - Relaxed sample is correct if level == LOW
      - Stressed sample is correct if level in {MODERATE, HIGH}
    """
    # Relaxed: should be LOW
    if relaxed_rows:
        relaxed_correct = sum(1 for r in relaxed_rows if r['level'] == 'LOW')
        relaxed_acc = relaxed_correct / len(relaxed_rows)
    else:
        relaxed_correct, relaxed_acc = 0, 0.0

    # Stressed: should be MODERATE or HIGH
    if stressed_rows:
        stressed_correct = sum(1 for r in stressed_rows if r['level'] in ('MODERATE', 'HIGH'))
        stressed_acc = stressed_correct / len(stressed_rows)
    else:
        stressed_correct, stressed_acc = 0, 0.0

    total      = len(relaxed_rows) + len(stressed_rows)
    correct    = relaxed_correct + stressed_correct
    overall    = correct / total if total else 0.0

    # Confusion matrix
    cm = {
        'relaxed_LOW': sum(1 for r in relaxed_rows  if r['level'] == 'LOW'),
        'relaxed_MOD': sum(1 for r in relaxed_rows  if r['level'] == 'MODERATE'),
        'relaxed_HIGH':sum(1 for r in relaxed_rows  if r['level'] == 'HIGH'),
        'stress_LOW':  sum(1 for r in stressed_rows if r['level'] == 'LOW'),
        'stress_MOD':  sum(1 for r in stressed_rows if r['level'] == 'MODERATE'),
        'stress_HIGH': sum(1 for r in stressed_rows if r['level'] == 'HIGH'),
    }

    return {
        'relaxed_correct': relaxed_correct, 'relaxed_total': len(relaxed_rows),
        'relaxed_acc':     relaxed_acc,
        'stressed_correct': stressed_correct, 'stressed_total': len(stressed_rows),
        'stressed_acc':    stressed_acc,
        'overall_acc':     overall,
        'confusion':       cm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='.', help='Folder containing CSVs')
    args = parser.parse_args()

    folder = Path(args.dir)
    relaxed_files  = sorted(folder.glob('relaxed_*.csv'))
    stressed_files = sorted(folder.glob('stressed_*.csv'))
    if not relaxed_files and not stressed_files:
        # Also accept the auto-named exports
        all_csvs = sorted(folder.glob('stress_session_*.csv'))
        if all_csvs:
            print(f"Found {len(all_csvs)} stress_session_*.csv files but no "
                  f"relaxed_*.csv / stressed_*.csv. Please rename them so the "
                  f"script knows which is which.")
        else:
            print(f"No CSVs found in {folder.resolve()}.")
        return

    print(f"Found {len(relaxed_files)} relaxed file(s), "
          f"{len(stressed_files)} stressed file(s)\n")

    print("=== Per-file summary ===")
    relaxed_all  = []
    stressed_all = []
    for f in relaxed_files:
        rows = load_csv(f)
        summarize(rows, f.name)
        relaxed_all.extend(rows)
    for f in stressed_files:
        rows = load_csv(f)
        summarize(rows, f.name)
        stressed_all.extend(rows)

    print("\n=== Pooled summary ===")
    summarize(relaxed_all,  "RELAXED ")
    summarize(stressed_all, "STRESSED")

    if not relaxed_all or not stressed_all:
        print("\nCannot evaluate without both relaxed and stressed samples.")
        return

    eval_result = evaluate(relaxed_all, stressed_all)

    print("\n=== Live system accuracy ===")
    print(f"  Relaxed correctly classified as LOW:        "
          f"{eval_result['relaxed_correct']}/{eval_result['relaxed_total']} "
          f"= {eval_result['relaxed_acc']*100:.1f}%")
    print(f"  Stressed correctly classified as MOD/HIGH:  "
          f"{eval_result['stressed_correct']}/{eval_result['stressed_total']} "
          f"= {eval_result['stressed_acc']*100:.1f}%")
    print(f"  Overall accuracy:                           "
          f"{eval_result['overall_acc']*100:.1f}%")

    print("\n=== Confusion matrix ===")
    print(f"  {'':<12}{'pred LOW':>10}{'pred MOD':>10}{'pred HIGH':>10}")
    print(f"  {'relaxed':<12}"
          f"{eval_result['confusion']['relaxed_LOW']:>10}"
          f"{eval_result['confusion']['relaxed_MOD']:>10}"
          f"{eval_result['confusion']['relaxed_HIGH']:>10}")
    print(f"  {'stressed':<12}"
          f"{eval_result['confusion']['stress_LOW']:>10}"
          f"{eval_result['confusion']['stress_MOD']:>10}"
          f"{eval_result['confusion']['stress_HIGH']:>10}")

    # Discriminative gap
    rs = statistics.mean(r['stress_index'] for r in relaxed_all)
    ss = statistics.mean(r['stress_index'] for r in stressed_all)
    gap = ss - rs
    print(f"\n=== Discrimination ===")
    print(f"  Mean stress_index (relaxed):   {rs:.3f}")
    print(f"  Mean stress_index (stressed):  {ss:.3f}")
    print(f"  Gap:                           {gap:+.3f}")
    if gap > 0.10:
        print("  ✅ System distinguishes the two states.")
    elif gap > 0.0:
        print("  ⚠️ Weak separation. Consider re-tuning thresholds.")
    else:
        print("  ❌ INVERTED or no separation. System is broken.")


if __name__ == '__main__':
    main()
