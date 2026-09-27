"""
ROC / Youden's J threshold derivation (Phase 1.4b)
===================================================
Derives the LOW-vs-stressed decision boundary from data instead of by hand,
using ROC analysis (Youden's J statistic) validated with leave-one-subject-
out (LOSO) cross-validation across every subject with labeled ground truth
in this project: UBFC-Phys (T1=relaxed, T2/T3=stressed) and the team's own
recordings.

What this does NOT attempt
---------------------------
The live system has TWO thresholds - config.LOW_TO_MODERATE_CUTOFF (0.36)
and config.MODERATE_TO_HIGH_CUTOFF (0.55) - but the only ground truth
anywhere in this project is BINARY (relaxed vs stressed). There is no
severity label separating a "moderately stressed" frame from a "highly
stressed" one, so Youden's J (which needs a binary target) can only derive
the LOW-vs-(MODERATE-or-HIGH) boundary. The second boundary is reported
here as a hand-set severity split within the "stressed" score range,
unvalidated by this or any other study in this project - stated plainly
rather than inventing a fake ordinal signal (e.g. treating UBFC-Phys's T2
speech task and T3 arithmetic task as "mild vs severe", which they are not
by design) just to justify a second cut.

Method
------
For each subject, held out in turn:
  1. Pool every OTHER subject's (score, label) frames.
  2. Compute the ROC curve (sklearn.metrics.roc_curve) and take the
     threshold maximising Youden's J = TPR - FPR.
  3. Apply that fold's threshold to the held-out subject's own frames and
     score accuracy/balanced accuracy against it - never validated on the
     same data used to pick the threshold - and compare against the
     CURRENT fixed threshold (0.36) applied to that same held-out subject.

Also reports a sensitivity sweep: accuracy across a range of fixed
thresholds evaluated on the full pooled dataset, so "how much would
results change if we'd picked 0.30 instead of 0.36" has an actual,
measured answer instead of a guess.

Usage
-----
    cd code && python derive_thresholds.py
    python derive_thresholds.py --out ../evaluation_data/THRESHOLD_RESULTS.txt
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve, balanced_accuracy_score

import config
from app_naming import parse_filename
from correlate_physiology import load_index
# run_all_evaluations imports THIS module (to fold its results into the one
# canonical report), so importing `collect` back from it at module level
# would be circular. Deferred into load_team_subjects() below instead -
# by call time both modules are fully initialised either way this file is
# entered (standalone script or imported by run_all_evaluations).

UBFC_NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)

# Sweep range for the sensitivity analysis - fine enough to see the shape of
# the accuracy-vs-threshold curve without pretending to more precision than
# a hand-tuned cutoff needs.
SWEEP_THRESHOLDS = np.round(np.arange(0.20, 0.71, 0.05), 2)


# ---------------------------------------------------------------------------
# Data loading - two ground-truth sources, combined into one subject pool.
# ---------------------------------------------------------------------------
def load_ubfc_subjects(trace_dir):
    """
    {subject_id: [(score, label), ...]} from UBFC-Phys feature traces.
    subject_id is prefixed "U" (e.g. "U07") so it can never collide with a
    team recording's subject id in the combined pool. Ground truth: T1 is
    the rest/relaxed task (label 0), T2/T3 are the stressor tasks (label 1)
    - the UBFC-Phys protocol's own task design, the same convention already
    used throughout correlate_physiology.py and ablation_study.py.
    """
    by_subject = {}
    for path in sorted(Path(trace_dir).glob('features_*.csv')):
        m = UBFC_NAME_RE.search(path.stem)
        if not m:
            continue
        subj, task = int(m.group(1)), int(m.group(2))
        label = 0 if task == 1 else 1
        t, x = load_index(path)
        if len(x) == 0:
            continue
        subject_id = f"U{subj:02d}"
        by_subject.setdefault(subject_id, []).extend(
            (float(score), label) for score in x)
    return by_subject


def load_team_subjects(folder, mode='default'):
    """
    {subject_id: [(score, label), ...]} from the team's own recordings.
    subject_id is prefixed "T" (e.g. "TS1") for the same collision-avoidance
    reason as load_ubfc_subjects(). Reuses run_all_evaluations.collect() so
    this is bound by the exact same mode-filtering/naming rules as every
    other reported number, not a separate ad-hoc file scan.
    """
    from run_all_evaluations import collect   # deferred - see module docstring note
    relaxed, stressed = collect(folder, 'relaxed_*.csv', 'stressed_*.csv', mode=mode)
    by_subject = {}
    for condition, files, label in (('relaxed', relaxed, 0), ('stressed', stressed, 1)):
        for name, rows in files:
            meta = parse_filename(name)
            if not meta:
                continue
            subject_id = f"T{meta['subject']}"
            by_subject.setdefault(subject_id, []).extend(
                (float(r['stress_index']), label) for r in rows)
    return by_subject


# ---------------------------------------------------------------------------
# LOSO threshold derivation
# ---------------------------------------------------------------------------
def run_loso_derivation(all_subjects):
    """
    Returns a results dict, or None if fewer than 2 subjects have usable
    (both-class) data - Youden's J needs both classes present in the
    training pool, and a held-out subject with only one class can still be
    scored (plain accuracy), just not balanced-accuracy.
    """
    subjects = sorted(all_subjects)
    if len(subjects) < 2:
        return None

    fold_rows = []
    for held_out in subjects:
        train_scores, train_labels = [], []
        for subj in subjects:
            if subj == held_out:
                continue
            for score, label in all_subjects[subj]:
                train_scores.append(score)
                train_labels.append(label)

        if len(set(train_labels)) < 2:
            continue   # can't fit a ROC curve without both classes in training

        fpr, tpr, roc_thresholds = roc_curve(train_labels, train_scores)
        youden_j = tpr - fpr
        optimal_threshold = float(roc_thresholds[int(np.argmax(youden_j))])

        test_pairs = all_subjects[held_out]
        test_scores = np.array([s for s, _ in test_pairs])
        test_labels = np.array([l for _, l in test_pairs])
        if len(test_labels) == 0:
            continue

        pred_optimal = (test_scores >= optimal_threshold).astype(int)
        pred_fixed = (test_scores >= config.LOW_TO_MODERATE_CUTOFF).astype(int)
        both_classes_in_test = len(set(test_labels.tolist())) > 1

        fold_rows.append({
            'subject': held_out,
            'n': len(test_labels),
            'optimal_threshold': optimal_threshold,
            'acc_optimal': float(np.mean(pred_optimal == test_labels)),
            'acc_fixed': float(np.mean(pred_fixed == test_labels)),
            'balanced_acc_optimal': (
                float(balanced_accuracy_score(test_labels, pred_optimal))
                if both_classes_in_test else None),
            'balanced_acc_fixed': (
                float(balanced_accuracy_score(test_labels, pred_fixed))
                if both_classes_in_test else None),
        })

    if not fold_rows:
        return None

    thresholds = [r['optimal_threshold'] for r in fold_rows]
    return {
        'fold_rows': fold_rows,
        'n_folds': len(fold_rows),
        'mean_optimal_threshold': float(np.mean(thresholds)),
        'std_optimal_threshold': float(np.std(thresholds)),
        'mean_acc_optimal': float(np.mean([r['acc_optimal'] for r in fold_rows])),
        'mean_acc_fixed': float(np.mean([r['acc_fixed'] for r in fold_rows])),
    }


def run_sensitivity_sweep(all_subjects, thresholds=SWEEP_THRESHOLDS):
    """
    Accuracy/balanced-accuracy on the FULL pooled dataset (every subject,
    no held-out split - this is a descriptive sweep of the score
    distribution, not a generalisation claim) at each candidate threshold.
    Answers "how much would the headline number move if a different cutoff
    had been picked", independent of the LOSO study above.
    """
    scores, labels = [], []
    for frames in all_subjects.values():
        for s, l in frames:
            scores.append(s)
            labels.append(l)
    scores, labels = np.array(scores), np.array(labels)
    if len(labels) == 0 or len(set(labels.tolist())) < 2:
        return None

    rows = []
    for th in thresholds:
        pred = (scores >= th).astype(int)
        rows.append({
            'threshold': float(th),
            'accuracy': float(np.mean(pred == labels)),
            'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
        })
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(loso_results, sweep_results, out, generated_line=None):
    out.append("\n" + "=" * 78)
    out.append("ROC / YOUDEN'S J THRESHOLD DERIVATION (LOSO)")
    out.append("=" * 78)
    if generated_line:
        out.append(f"  {generated_line}")
    out.append(f"  current fixed cutoffs : LOW/MODERATE={config.LOW_TO_MODERATE_CUTOFF:.2f}  "
               f"MODERATE/HIGH={config.MODERATE_TO_HIGH_CUTOFF:.2f}")
    out.append("")
    out.append("Ground truth is binary (relaxed/stressed) everywhere in this project, so")
    out.append("only the LOW-vs-stressed boundary can be derived this way. The")
    out.append("MODERATE/HIGH split has no ground truth to validate against and is NOT")
    out.append("addressed by this study - it remains a hand-set severity cut, reported")
    out.append("here as an explicit limitation, not silently assumed solved.")
    out.append("")

    if loso_results is None:
        out.append("  (fewer than 2 subjects with usable both-class data - skipped)")
    else:
        out.append(f"  LOSO folds run: {loso_results['n_folds']}")
        out.append(f"  mean optimal threshold : {loso_results['mean_optimal_threshold']:.3f}  "
                   f"(std {loso_results['std_optimal_threshold']:.3f})")
        out.append(f"  mean held-out accuracy, per-fold optimal threshold : "
                   f"{loso_results['mean_acc_optimal']*100:.2f}%")
        out.append(f"  mean held-out accuracy, CURRENT fixed threshold ({config.LOW_TO_MODERATE_CUTOFF:.2f}) : "
                   f"{loso_results['mean_acc_fixed']*100:.2f}%")
        out.append("")
        out.append(f"  {'subject':10s}{'n':>7}{'opt.thr':>9}{'acc@opt':>9}"
                   f"{'acc@fixed':>11}{'bal@opt':>9}{'bal@fixed':>11}")
        out.append("  " + "-" * 66)
        for r in loso_results['fold_rows']:
            bal_opt = f"{r['balanced_acc_optimal']*100:.1f}%" if r['balanced_acc_optimal'] is not None else "n/a"
            bal_fix = f"{r['balanced_acc_fixed']*100:.1f}%" if r['balanced_acc_fixed'] is not None else "n/a"
            out.append(f"  {r['subject']:<10}{r['n']:>7}{r['optimal_threshold']:>9.3f}"
                       f"{r['acc_optimal']*100:>8.1f}%{r['acc_fixed']*100:>10.1f}%"
                       f"{bal_opt:>9}{bal_fix:>11}")

    out.append("")
    out.append("-" * 78)
    out.append("SENSITIVITY SWEEP (full pooled dataset, not held-out)")
    out.append("-" * 78)
    if sweep_results is None:
        out.append("  (no usable pooled data - skipped)")
        return
    out.append(f"  {'threshold':>10s}{'accuracy':>11}{'balanced acc':>15}")
    out.append("  " + "-" * 36)
    for row in sweep_results:
        marker = "  <- current" if abs(row['threshold'] - config.LOW_TO_MODERATE_CUTOFF) < 1e-9 else ""
        out.append(f"  {row['threshold']:>10.2f}{row['accuracy']*100:>10.1f}%"
                   f"{row['balanced_accuracy']*100:>14.1f}%{marker}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', default='../datasets/feature_traces')
    ap.add_argument('--team', default='../evaluation_data/live_recordings')
    ap.add_argument('--root', default='..')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    all_subjects = {}
    all_subjects.update(load_ubfc_subjects(args.traces))
    all_subjects.update(load_team_subjects(args.team))

    if not all_subjects:
        print("no labeled data found under --traces or --team")
        return 1

    loso_results = run_loso_derivation(all_subjects)
    sweep_results = run_sensitivity_sweep(all_subjects)

    out = []
    render(loso_results, sweep_results, out,
          generated_line=f"generated : {datetime.now().isoformat(timespec='seconds')}")
    text = "\n".join(out)
    print(text)

    root = Path(args.root).resolve()
    out_path = (Path(args.out) if args.out else
                root / "evaluation_data" / f"THRESHOLD_RESULTS_{datetime.now():%Y-%m-%d}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n\nwritten to: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
