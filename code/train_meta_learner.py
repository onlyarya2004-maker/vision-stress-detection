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

Phase 4.1: run_learned_fusion_study()/render() below are called directly
from run_all_evaluations.py (same compute/render split as ablation_study.py
and correlate_physiology.py), so the LOSO comparison is a regenerated
canonical-file row instead of only ever living in this script's own output.
The harness path never writes models/meta_learner.joblib - saving a new
model is this script's job (--save), not a side effect of regenerating
reported numbers.

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
    """Pure - returns the metrics dict, does not print. format_summary_line()
    below renders it, so the CLI and the harness share one format."""
    from sklearn.metrics import accuracy_score, f1_score

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rel = y_true == 0
    stz = y_true == 1
    rel_recall = (y_pred[rel] == 0).mean() if rel.any() else float('nan')
    stz_recall = (y_pred[stz] == 1).mean() if stz.any() else float('nan')

    # "Condition-block" level: one verdict per (subject, binary condition)
    # by majority vote - e.g. a subject's T2 and T3 UBFC-Phys clips (two
    # different stressors, two different recordings) are POOLED into one
    # stressed verdict here. This is deliberately NOT the same unit as
    # eval_scoring.session_scores()'s "session", which is one recording/
    # file - that function scores T2 and T3 as two separate sessions. Both
    # numbers get printed in the same canonical report (see
    # run_all_evaluations.py), so the label here is "cond-block", never
    # bare "session", to keep the two from reading as the same metric.
    # See BUG_AUDIT_2026-08-23.md finding #9. Not changed to match
    # eval_scoring's definition - that would silently move every reported
    # number here with no clear win; only the label changed.
    cb_ok = cb_n = 0
    for g in np.unique(groups):
        for lab in (0, 1):
            m = (groups == g) & (y_true == lab)
            if m.sum() == 0:
                continue
            cb_n += 1
            cb_ok += int((y_pred[m] == lab).mean() > 0.5)

    return {'label': label, 'accuracy': acc, 'f1': f1,
            'relaxed_recall': rel_recall, 'stressed_recall': stz_recall,
            'cond_block_correct': cb_ok, 'cond_block_total': cb_n}


def format_summary_line(stats):
    return (f"  {stats['label']:22s} acc {stats['accuracy']*100:6.2f}%   "
           f"F1 {stats['f1']:.3f}   "
           f"relaxed-recall {stats['relaxed_recall']*100:5.1f}%   "
           f"stressed-recall {stats['stressed_recall']*100:5.1f}%   "
           f"cond-block {stats['cond_block_correct']}/{stats['cond_block_total']}")


def loso_evaluate(make_model, X, y, groups, label, scale=True):
    """
    Leave-one-subject-out CV. Returns (oof, stats), or (None, None) if fewer
    than 2 subjects are present - LOSO is undefined with only one group, and
    this must not silently "work" on the wrong protocol.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    logo = LeaveOneGroupOut()
    oof = np.zeros_like(y)
    n_subjects = len(np.unique(groups))
    if n_subjects < 2:
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
def compute_learned_weights_rows(model, names):
    """
    Turn logistic-regression coefficients into weights comparable with the
    hand-picked ones. Sign matters: a negative coefficient means the signal
    genuinely points the other way. Returns [] if the model is degenerate
    (all-zero coefficients).
    """
    coef = model.coef_[0]
    total = np.abs(coef).sum()
    if total < 1e-12:
        return []
    rows = []
    order = np.argsort(-np.abs(coef))
    for i in order:
        n = names[i]
        rows.append({
            'name': n,
            'hand_tuned': config.BEHAVIOR_WEIGHTS.get(n),
            'learned': abs(coef[i]) / total,
            'coef': float(coef[i]),
            # Plain ASCII, not an arrow glyph - Windows' default console
            # codepage (cp1252) can't encode U+2191/U+2193 and print()
            # crashes partway through this table, which used to also skip
            # the --save step entirely since it ran later in main() than
            # this report.
            'direction': "stress up" if coef[i] > 0 else "stress down (INVERTED)",
        })
    return rows


def format_learned_weights_rows(rows):
    lines = [f"  {'signal':18s}{'hand-tuned':>12}{'learned':>10}"
            f"{'coef':>9}   direction",
            "  " + "-" * 60]
    for r in rows:
        hand_s = f"{r['hand_tuned']:.2f}" if r['hand_tuned'] is not None else "  -"
        lines.append(f"  {r['name']:18s}{hand_s:>12}{r['learned']:>10.2f}"
                     f"{r['coef']:>+9.2f}   {r['direction']}")
    return lines


# ---------------------------------------------------------------------------
# Computation - separated from rendering (and from --save) so
# run_all_evaluations.py can call this directly and fold the LOSO comparison
# into the one canonical file. Never writes to disk itself.
# ---------------------------------------------------------------------------
def run_learned_fusion_study(trace_dir, include_daisee=False, subsample=1):
    """
    Returns a results dict, or None if no trace CSVs are found (same
    contract as ablation_study.run_ablation_study() /
    correlate_physiology.run_correlation_study()).

    'final_model'/'scaler' are the logistic regression fit on ALL subjects
    (for the interpretable-weights table) - included so main()'s --save can
    reuse them directly instead of fitting a second, possibly-divergent
    copy. Both are None when there are fewer than 2 subjects, matching
    loso_evaluate()'s "LOSO is undefined here" stance.
    """
    traces = sorted(Path(trace_dir).glob('features_*.csv'))
    if not traces:
        return None

    X, y, groups, names = load_traces(trace_dir, include_daisee)
    if subsample > 1:
        X, y, groups = X[::subsample], y[::subsample], groups[::subsample]

    handtuned_stats = summarize(y, handtuned_predictions(X, names), groups,
                                "hand-tuned fusion")

    results = {
        'n_frames': len(X), 'n_subjects': len(np.unique(groups)),
        'n_relaxed': int((y == 0).sum()), 'n_stressed': int((y == 1).sum()),
        'subsample': subsample,
        'handtuned': handtuned_stats,
        'logistic_loso': None, 'gbt_loso': None, 'weights_rows': [],
        'final_model': None, 'scaler': None, 'feature_names': names,
    }
    if results['n_subjects'] < 2:
        return results

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    _, log_stats = loso_evaluate(
        lambda: LogisticRegression(max_iter=2000, class_weight='balanced'),
        X, y, groups, "logistic regression")
    _, gbt_stats = loso_evaluate(
        lambda: GradientBoostingClassifier(random_state=0),
        X, y, groups, "gradient boosting", scale=False)
    results['logistic_loso'] = log_stats
    results['gbt_loso'] = gbt_stats

    # Fit once on everything, purely to read off interpretable weights (and
    # for --save to persist, if asked).
    sc = StandardScaler().fit(X)
    final = LogisticRegression(max_iter=2000, class_weight='balanced')
    final.fit(sc.transform(X), y)
    results['final_model'] = final
    results['scaler'] = sc
    results['weights_rows'] = compute_learned_weights_rows(final, names)
    return results


def render(results, out):
    out.append("\n" + "=" * 74)
    out.append("LEARNED FUSION - leave-one-subject-out")
    out.append("=" * 74)
    if results is None:
        out.append("  (no feature traces found under datasets/feature_traces/ - skipped)")
        return

    out.append(f"  {results['n_frames']} frames from {results['n_subjects']} subject(s)"
               + (f", subsampled every {results['subsample']}" if results['subsample'] > 1 else ""))
    out.append(f"  class balance: {results['n_relaxed']} relaxed / "
               f"{results['n_stressed']} stressed")

    out.append(f"\n{'-' * 74}\nBASELINE - current hand-tuned weights\n{'-' * 74}")
    out.append(format_summary_line(results['handtuned']))

    if results['n_subjects'] < 2:
        out.append("\n  Only one subject present - LOSO needs at least two.")
        out.append("  Process more UBFC-Phys subjects, then re-run.")
        return

    out.append(f"\n{'-' * 74}\nLEARNED (out-of-fold, {results['n_subjects']} folds)\n{'-' * 74}")
    out.append(format_summary_line(results['logistic_loso']))
    out.append(format_summary_line(results['gbt_loso']))

    if results['weights_rows']:
        out.append(f"\n{'-' * 74}\nWHAT IT LEARNED (fitted on all subjects)\n{'-' * 74}")
        out.extend(format_learned_weights_rows(results['weights_rows']))
        out.append("\n  Signals marked INVERTED carry a NEGATIVE learned coefficient:")
        out.append("  the model found them anti-correlated with stress on this data,")
        out.append("  which the hand-picked weights could not express - they are all")
        out.append("  positive by construction.")


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

    results = run_learned_fusion_study(args.dir, args.include_daisee, args.subsample)
    if results is None:
        raise SystemExit(f"no usable rows in {args.dir}")

    out = []
    render(results, out)

    # Save before printing - the model is the actually-important output;
    # the printed table is just commentary on it. A crash while printing
    # (this table's direction labels used to be non-ASCII arrows, which
    # crash mid-print on Windows' default console codepage) used to also
    # silently skip --save, since it ran later in the function and this
    # exact thing already happened once.
    if args.save and results['final_model'] is not None:
        try:
            import joblib
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({'model': results['final_model'], 'scaler': results['scaler'],
                        'features': results['feature_names']}, save_path)
            out.append(f"\n  saved: {save_path}")
        except ImportError:
            out.append("\n  joblib not available; model not saved")

    print("\n".join(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
