"""
PROTOTYPE - warm-start personalization (not wired into production)
====================================================================
Tests a specific hypothesis before touching retrain_from_sessions.py or
any live code: does initializing a user's personal fusion model from the
population-level meta-learner (train_meta_learner.py), instead of training
from scratch on just their own handful of sessions, produce a better model
under realistically LOW personal data?

Why this is a separate script, not a change to retrain_from_sessions.py
--------------------------------------------------------------------------
retrain_from_sessions.py's honesty guards (>=6 labeled sessions, session-
held-out evaluation, never-automatic activation) exist for a reason, and
there is currently only 1 real self-reported session in this install - not
enough to validate ANYTHING about personalization on real personal data.
This script validates the underlying idea on data that DOES have enough
volume to test it honestly: UBFC-Phys, which already has 10 subjects x 3
tasks. Only after this shows a real, reproducible improvement (or doesn't)
should the idea move into the production retraining script.

Protocol (LOSO + simulated few-shot personalization)
--------------------------------------------------------------------------
For each UBFC-Phys subject S in turn:
  1. POPULATION PRIOR: StandardScaler + LogisticRegression fit on the other
     9 subjects' frames - identical recipe to train_meta_learner.py's own
     LOSO fold, so this prior is exactly what production code would produce
     with subject S held out.
  2. SIMULATED PERSONAL DATA: subject S's T1 (relaxed) and T2 (stressed)
     frames, SUBSAMPLED to PERSONAL_FRAMES_PER_TASK each (default 100 -
     chosen from this install's own real session sample counts, which
     range 23-204 frames/session, average ~97 - see PLAN_FOR_PP2.md). This
     stands in for "a couple of real labeled sessions", not thousands of
     frames - the actual regime retrain_from_sessions.py will face.
  3. HELD-OUT EVAL: subject S's T3 (arithmetic stress) frames, ALL of them,
     never touched by fitting - a genuinely unseen task, not just unseen
     frames from a task the model has already partially seen.
  4. Three conditions scored on that same held-out T3 data:
     - population_only : the prior, unmodified, zero personal data used
     - personal_only    : fresh LogisticRegression on JUST the small
                           personal subsample - what retrain_from_sessions.py
                           does today
     - warm_started     : MAP logistic regression - personal data as the
                           likelihood, the prior's (w, b) as a Gaussian
                           prior mean, swept over a few regularization
                           strengths (lambda) to see where any benefit lives
  Repeated for all 10 subjects; every condition scored the same way, no
  cherry-picking of which subjects/lambdas get reported.

Usage
-----
    python prototype_warmstart_personalization.py
"""

import argparse
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from train_meta_learner import SIGNALS

PERSONAL_FRAMES_PER_TASK = 100   # ~1 real session's worth, see module docstring
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
SEED = 0


# ---------------------------------------------------------------------------
# MAP logistic regression with a Gaussian prior centered at (w_prior, b_prior)
# ---------------------------------------------------------------------------
def fit_logreg_with_prior(X, y, w_prior, b_prior, lam):
    """
    minimize  BCE(X, y; w, b)  +  (lam/2) * (||w - w_prior||^2 + (b-b_prior)^2)

    lam=0 reduces to plain MLE on (X, y) alone - identical in spirit to
    "personal_only" (the prior only sets the optimizer's starting point,
    which does not change the converged optimum of a convex problem).
    lam -> large pulls (w, b) back toward the prior regardless of what the
    personal data says. X must already be standardized with the SAME
    scaler the prior was fit with - the prior's weights only mean what
    they mean in that scaler's feature space.
    """
    D = X.shape[1]

    def nll_and_grad(params):
        w, b = params[:D], params[D]
        z = X @ w + b
        p = expit(z)
        eps = 1e-9
        nll = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.5 * lam * (np.sum((w - w_prior) ** 2) + (b - b_prior) ** 2)
        grad_w = X.T @ (p - y) + lam * (w - w_prior)
        grad_b = np.sum(p - y) + lam * (b - b_prior)
        return nll + reg, np.concatenate([grad_w, [grad_b]])

    x0 = np.concatenate([w_prior, [b_prior]])
    res = minimize(nll_and_grad, x0, jac=True, method='L-BFGS-B')
    return res.x[:D], res.x[D]


def _predict(X, w, b):
    return (expit(X @ w + b) >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _load_by_subject_task(trace_dir):
    import csv
    import re
    from collections import defaultdict
    from pathlib import Path

    NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)
    by_subject_task = defaultdict(lambda: defaultdict(list))   # subj -> task -> [row,...]
    for path in sorted(Path(trace_dir).glob('features_*.csv')):
        m = NAME_RE.search(path.stem)
        if not m:
            continue
        subj, task = f"s{int(m.group(1))}", int(m.group(2))
        for row in csv.DictReader(open(path)):
            try:
                by_subject_task[subj][task].append([float(row[n]) for n in SIGNALS])
            except (KeyError, ValueError):
                continue
    return by_subject_task


def _fit_prior(by_subject_task, held_out):
    """Population prior fit on every OTHER subject's frames - identical
    recipe to train_meta_learner.py's own LOSO fold."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    X_other, y_other = [], []
    for subj, task_rows in by_subject_task.items():
        if subj == held_out:
            continue
        for task, rows in task_rows.items():
            label = 0 if task == 1 else 1
            X_other.extend(rows)
            y_other.extend([label] * len(rows))
    X_other, y_other = np.array(X_other), np.array(y_other)

    sc = StandardScaler().fit(X_other)
    prior = LogisticRegression(max_iter=2000, class_weight='balanced')
    prior.fit(sc.transform(X_other), y_other)
    return sc, prior.coef_[0], prior.intercept_[0]


def _score_conditions(X_personal_sc, y_personal, X_eval, y_eval, w_prior, b_prior, lambdas):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    row = {'n_eval': len(y_eval)}
    row['population_only'] = accuracy_score(y_eval, _predict(X_eval, w_prior, b_prior))

    personal_model = LogisticRegression(max_iter=2000, class_weight='balanced')
    personal_model.fit(X_personal_sc, y_personal)
    row['personal_only'] = accuracy_score(y_eval, personal_model.predict(X_eval))

    row['warm_started'] = {}
    for lam in lambdas:
        w, b = fit_logreg_with_prior(X_personal_sc, y_personal, w_prior, b_prior, lam)
        row['warm_started'][lam] = accuracy_score(y_eval, _predict(X_eval, w, b))
    return row


def run_cross_task(trace_dir, personal_frames_per_task, lambdas, seed):
    """
    Personalize on T1 (relaxed) + T2 (speech-stress), evaluate on ALL of T3
    (arithmetic-stress) - a genuinely unseen TASK, not just unseen frames.
    """
    by_subject_task = _load_by_subject_task(trace_dir)
    subjects = sorted(by_subject_task, key=lambda s: int(s[1:]))
    rng = np.random.default_rng(seed)

    results = []
    for held_out in subjects:
        tasks = by_subject_task[held_out]
        if not all(t in tasks and len(tasks[t]) >= personal_frames_per_task for t in (1, 2, 3)):
            print(f"  [skip] {held_out}: missing/short task data")
            continue

        sc, w_prior, b_prior = _fit_prior(by_subject_task, held_out)

        t1, t2 = np.array(tasks[1]), np.array(tasks[2])
        t1_idx = rng.choice(len(t1), size=min(personal_frames_per_task, len(t1)), replace=False)
        t2_idx = rng.choice(len(t2), size=min(personal_frames_per_task, len(t2)), replace=False)
        X_personal_sc = sc.transform(np.vstack([t1[t1_idx], t2[t2_idx]]))
        y_personal = np.array([0] * len(t1_idx) + [1] * len(t2_idx))

        X_eval = sc.transform(np.array(tasks[3]))
        y_eval = np.ones(len(X_eval), dtype=int)   # T3 is entirely "stressed"

        row = _score_conditions(X_personal_sc, y_personal, X_eval, y_eval, w_prior, b_prior, lambdas)
        row['subject'] = held_out
        results.append(row)
        print(f"  {held_out}: pop_only={row['population_only']*100:5.1f}%  "
             f"personal_only={row['personal_only']*100:5.1f}%  "
             f"warm[best]={max(row['warm_started'].values())*100:5.1f}%  "
             f"(n_eval={row['n_eval']})")
    return results


def run_same_task(trace_dir, personal_frames_per_task, lambdas, seed):
    """
    Control condition: personalize on the FIRST personal_frames_per_task
    frames of T1 and T2, evaluate on the REMAINING (unseen) frames of the
    SAME two tasks. Isolates "does personalization help at all" from the
    cross-task run's "does it transfer to a DIFFERENT stressor" question -
    T3 is not used here at all.
    """
    by_subject_task = _load_by_subject_task(trace_dir)
    subjects = sorted(by_subject_task, key=lambda s: int(s[1:]))

    results = []
    for held_out in subjects:
        tasks = by_subject_task[held_out]
        if not all(t in tasks and len(tasks[t]) >= personal_frames_per_task + 20 for t in (1, 2)):
            print(f"  [skip] {held_out}: missing/short task data")
            continue

        sc, w_prior, b_prior = _fit_prior(by_subject_task, held_out)

        t1, t2 = np.array(tasks[1]), np.array(tasks[2])
        n1, n2 = personal_frames_per_task, personal_frames_per_task
        X_personal_sc = sc.transform(np.vstack([t1[:n1], t2[:n2]]))
        y_personal = np.array([0] * n1 + [1] * n2)

        X_eval = sc.transform(np.vstack([t1[n1:], t2[n2:]]))
        y_eval = np.array([0] * (len(t1) - n1) + [1] * (len(t2) - n2))

        row = _score_conditions(X_personal_sc, y_personal, X_eval, y_eval, w_prior, b_prior, lambdas)
        row['subject'] = held_out
        results.append(row)
        print(f"  {held_out}: pop_only={row['population_only']*100:5.1f}%  "
             f"personal_only={row['personal_only']*100:5.1f}%  "
             f"warm[best]={max(row['warm_started'].values())*100:5.1f}%  "
             f"(n_eval={row['n_eval']})")
    return results


def render(results, lambdas, label, frames_per_task):
    print("\n" + "=" * 78)
    print(f"SUMMARY: {label}")
    print("=" * 78)
    if not results:
        print("  no subjects had usable data for this condition")
        return

    pop = np.mean([r['population_only'] for r in results])
    per = np.mean([r['personal_only'] for r in results])
    print(f"  population_only (no personalization)         : {pop*100:5.1f}%")
    print(f"  personal_only   (scratch, today's approach)   : {per*100:5.1f}%")
    print()
    print("  warm_started, by lambda (0 = pure personal, large = pure population):")
    best_lam, best_acc = None, -1.0
    for lam in lambdas:
        accs = [r['warm_started'][lam] for r in results]
        mean_acc = np.mean(accs)
        if mean_acc > best_acc:
            best_acc, best_lam = mean_acc, lam
        print(f"    lambda={lam:6.2f}  mean acc={mean_acc*100:5.1f}%  "
             f"(n={len(accs)} subjects)")
    print()
    print(f"  best warm-started setting: lambda={best_lam} at {best_acc*100:.1f}%")
    print(f"  vs population_only {pop*100:.1f}%  ({'+' if best_acc>pop else ''}{(best_acc-pop)*100:.1f}pp)")
    print(f"  vs personal_only   {per*100:.1f}%  ({'+' if best_acc>per else ''}{(best_acc-per)*100:.1f}pp)")
    print()
    print(f"  Read honestly: {frames_per_task} frames/task standing in for a handful of")
    print("  real sessions. No lambda was chosen in advance - all are reported. A real")
    print("  deployment would need its own way to pick lambda without peeking at the")
    print("  eval set (e.g. cross-validated within the personal data itself).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace-dir', default='../datasets/feature_traces')
    ap.add_argument('--personal-frames-per-task', type=int, default=PERSONAL_FRAMES_PER_TASK)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    print("=" * 78)
    print("PROTOTYPE: warm-start personalization (offline validation, UBFC-Phys)")
    print("=" * 78)
    print(f"personal data simulated as {args.personal_frames_per_task} frames/task\n")

    print("-" * 78)
    print("CONDITION 1: CROSS-TASK - personalize on T1+T2 (relaxed/speech-stress),")
    print("evaluate on ALL of T3 (arithmetic-stress) - an unseen STRESSOR, not just")
    print("unseen frames.")
    print("-" * 78)
    cross_results = run_cross_task(args.trace_dir, args.personal_frames_per_task, LAMBDAS, args.seed)
    render(cross_results, LAMBDAS, "cross-task (personalize T1+T2, eval unseen T3)",
          args.personal_frames_per_task)

    print("\n" + "-" * 78)
    print("CONDITION 2: SAME-TASK CONTROL - personalize on the first N frames of")
    print("T1+T2, evaluate on the REMAINING frames of those SAME two tasks. T3 is")
    print("not used. Isolates whether personalization helps at all from whether it")
    print("transfers to a different stressor.")
    print("-" * 78)
    same_results = run_same_task(args.trace_dir, args.personal_frames_per_task, LAMBDAS, args.seed)
    render(same_results, LAMBDAS, "same-task control (personalize+eval on T1+T2 only)",
          args.personal_frames_per_task)
    return 0


if __name__ == '__main__':
    sys.exit(main())
