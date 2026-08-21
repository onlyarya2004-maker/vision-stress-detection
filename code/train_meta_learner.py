"""
Learned fusion (Phase 3)
========================
Replaces the hand-picked fusion weights with weights LEARNED from data.

Why
---
The seven behavioural weights in config.BEHAVIOR_WEIGHTS were chosen from
affective-computing intuition, never fitted. On your own recordings that was
invisible - any weighting works once thresholds are tuned to one person. The
first cross-subject run showed what it costs:

    brow_tension   weight 0.30   Cohen's d  -0.78   INVERTED
    jaw_tension    weight 0.25   Cohen's d  -0.70   INVERTED
    slouch         weight 0.10   Cohen's d  +2.17   STRONG  (after calib fix)
    neck_strain    weight 0.05   Cohen's d  +1.34   STRONG
    head_instab.   weight 0.05   Cohen's d  +1.23   STRONG

0.55 of the budget on signals pointing the wrong way; 0.20 on the ones that
work. No amount of per-signal fixing corrects a weighting problem.

Protocol
--------
Leave-one-subject-out cross-validation, the same protocol wesad_calibration.py
uses, so the numbers are directly comparable and no subject is ever in its own
training set. Against it we score the CURRENT hand-tuned fusion recomputed
from the identical feature rows - so the comparison isolates the weighting and
nothing else.

Interpretability is preserved deliberately. Logistic regression over seven
named signals yields coefficients that ARE the new weights, readable in the
same terms as the old ones. The system does not become a black box; it stops
being a guess.

Usage
-----
    python train_meta_learner.py --dir ../datasets/feature_traces
    python train_meta_learner.py --dir ../datasets/feature_traces --save ../models/meta_learner.joblib
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config

SIGNALS = ['eye_strain', 'blink_stress', 'brow_tension', 'jaw_tension',
           'head_instability', 'neck_strain', 'slouch']

# features_vid_s12_T3.csv -> subject 12, task 3
NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_traces(directory, include_daisee=False):
    """
    Returns X, y, groups, feature_names.

    y: 0 = relaxed (UBFC-Phys T1 rest), 1 = stressed (T2 speech, T3 arithmetic)
    groups: subject id, so LOSO never trains and tests on the same person.
    """
    names = list(SIGNALS) + (['daisee_raw'] if include_daisee else [])
    X, y, groups = [], [], []
    per_subject = defaultdict(int)

    for path in sorted(Path(directory).glob('features_*.csv')):
        m = NAME_RE.search(path.stem)
        if not m:
            print(f"  [skip] {path.name}: cannot parse subject/task")
            continue
        subject, task = f"s{int(m.group(1))}", int(m.group(2))
        label = 0 if task == 1 else 1

        for row in csv.DictReader(open(path)):
            try:
                vals = [float(row[n]) for n in names]
            except (KeyError, ValueError):
                continue          # blank daisee on frames the CNN skipped
            X.append(vals)
            y.append(label)
            groups.append(subject)
            per_subject[subject] += 1

    if not X:
        raise SystemExit(f"no usable rows in {directory}")
    print(f"  loaded {len(X)} frames from {len(per_subject)} subject(s): "
          + ", ".join(f"{k}={v}" for k, v in sorted(per_subject.items())))
    return np.array(X), np.array(y), np.array(groups), names


# ---------------------------------------------------------------------------
# The incumbent: current hand-tuned fusion, recomputed from the same rows
# ---------------------------------------------------------------------------
def handtuned_scores(X, names):
    """
    Reproduce config's weighted average + 1.2x amplification. The DaiSEE
    branch is omitted because we are comparing the BEHAVIOURAL weighting;
    including it would only add a constant offset to both sides.
    """
    idx = {n: i for i, n in enumerate(names)}
    w = np.array([config.BEHAVIOR_WEIGHTS[s] for s in SIGNALS])
    cols = np.array([idx[s] for s in SIGNALS])
    raw = X[:, cols] @ w / w.sum()
    return np.clip(raw * config.BEHAVIOR_AMPLIFICATION, 0.0, 1.0)


def handtuned_predictions(X, names):
    """Apply the live LOW/MODERATE/HIGH cutoff: >= 0.36 counts as stressed."""
    return (handtuned_scores(X, names)
            >= config.LOW_TO_MODERATE_CUTOFF).astype(int)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def summarize(y_true, y_pred, groups, label):
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rel = y_true == 0
    stz = y_true == 1
    rel_recall = (y_pred[rel] == 0).mean() if rel.any() else float('nan')
    stz_recall = (y_pred[stz] == 1).mean() if stz.any() else float('nan')

    # Session level: one verdict per subject-condition by majority vote.
    sess_ok = sess_n = 0
    for g in np.unique(groups):
        for lab in (0, 1):
            m = (groups == g) & (y_true == lab)
            if m.sum() == 0:
                continue
            sess_n += 1
            sess_ok += int((y_pred[m] == lab).mean() > 0.5)

    print(f"  {label:22s} acc {acc*100:6.2f}%   F1 {f1:.3f}   "
          f"relaxed-recall {rel_recall*100:5.1f}%   "
          f"stressed-recall {stz_recall*100:5.1f}%   "
          f"session {sess_ok}/{sess_n}")
    return {'accuracy': acc, 'f1': f1, 'relaxed_recall': rel_recall,
            'stressed_recall': stz_recall,
            'session_correct': sess_ok, 'session_total': sess_n}


def loso_evaluate(make_model, X, y, groups, label, scale=True):
    """Leave-one-subject-out CV. Returns pooled out-of-fold predictions."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    logo = LeaveOneGroupOut()
    oof = np.zeros_like(y)
    n_subjects = len(np.unique(groups))
    if n_subjects < 2:
        print(f"  {label}: needs >=2 subjects for LOSO (have {n_subjects})")
        return None, None

    per_fold = []
    for tr, te in logo.split(X, y, groups):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        model = make_model()
        model.fit(Xtr, y[tr])
        pred = model.predict(Xte)
        oof[te] = pred
        per_fold.append(((pred == y[te]).mean(), groups[te][0]))

    stats = summarize(y, oof, groups, label)
    stats['per_fold'] = per_fold
    return oof, stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_learned_weights(model, names):
    """
    Turn logistic-regression coefficients into weights comparable with the
    hand-picked ones. Sign matters: a negative coefficient means the signal
    genuinely points the other way.
    """
    coef = model.coef_[0]
    total = np.abs(coef).sum()
    if total < 1e-12:
        return
    print(f"\n  {'signal':18s}{'hand-tuned':>12}{'learned':>10}"
          f"{'coef':>9}   direction")
    print("  " + "-" * 60)
    order = np.argsort(-np.abs(coef))
    for i in order:
        n = names[i]
        hand = config.BEHAVIOR_WEIGHTS.get(n)
        hand_s = f"{hand:.2f}" if hand is not None else "  -"
        learned = abs(coef[i]) / total
        # Plain ASCII, not an arrow glyph - Windows' default console
        # codepage (cp1252) can't encode U+2191/U+2193 and print() crashes
        # partway through this table, which used to also skip the --save
        # step entirely since it runs later in main() than this report.
        direction = "stress up" if coef[i] > 0 else "stress down (INVERTED)"
        print(f"  {n:18s}{hand_s:>12}{learned:>10.2f}{coef[i]:>+9.2f}"
              f"   {direction}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', default='../datasets/feature_traces',
                    help='directory of per-frame feature traces')
    ap.add_argument('--include-daisee', action='store_true',
                    help='add the CNN branch output as an eighth feature')
    ap.add_argument('--save', help='path to persist the fitted model')
    ap.add_argument('--subsample', type=int, default=1,
                    help='use every Nth frame (frames are autocorrelated; '
                         'subsampling speeds fitting without changing much)')
    args = ap.parse_args()

    print("=" * 74)
    print("LEARNED FUSION - leave-one-subject-out")
    print("=" * 74)
    X, y, groups, names = load_traces(args.dir, args.include_daisee)
    if args.subsample > 1:
        X, y, groups = X[::args.subsample], y[::args.subsample], groups[::args.subsample]
        print(f"  subsampled to {len(X)} frames (every {args.subsample})")
    print(f"  class balance: {int((y == 0).sum())} relaxed / "
          f"{int((y == 1).sum())} stressed")

    print(f"\n{'-' * 74}\nBASELINE - current hand-tuned weights\n{'-' * 74}")
    summarize(y, handtuned_predictions(X, names), groups,
              "hand-tuned fusion")

    n_subjects = len(np.unique(groups))
    if n_subjects < 2:
        print("\n  Only one subject present - LOSO needs at least two.")
        print("  Process more UBFC-Phys subjects, then re-run.")
        return 0

    print(f"\n{'-' * 74}\nLEARNED (out-of-fold, {n_subjects} folds)\n{'-' * 74}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier

    loso_evaluate(
        lambda: LogisticRegression(max_iter=2000, class_weight='balanced'),
        X, y, groups, "logistic regression")
    loso_evaluate(
        lambda: GradientBoostingClassifier(random_state=0),
        X, y, groups, "gradient boosting", scale=False)

    # Fit once on everything, purely to read off interpretable weights.
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    final = LogisticRegression(max_iter=2000, class_weight='balanced')
    final.fit(sc.transform(X), y)

    # Save before the diagnostic report below, not after - the model is
    # the actually-important output; the printed table is just commentary
    # on it. A crash while printing (this table's direction labels used to
    # be non-ASCII arrows, which crash mid-print on Windows' default
    # console codepage) used to also silently skip --save, since it ran
    # later in the function and this exact thing already happened once.
    if args.save:
        try:
            import joblib
            out = Path(args.save)
            out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({'model': final, 'scaler': sc, 'features': names},
                        out)
            print(f"\n  saved: {out}")
        except ImportError:
            print("\n  joblib not available; model not saved")

    print(f"\n{'-' * 74}\nWHAT IT LEARNED (fitted on all subjects)\n{'-' * 74}")
    report_learned_weights(final, names)

    print("\n  Signals marked INVERTED carry a NEGATIVE learned coefficient:")
    print("  the model found them anti-correlated with stress on this data,")
    print("  which the hand-picked weights could not express - they are all")
    print("  positive by construction.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
