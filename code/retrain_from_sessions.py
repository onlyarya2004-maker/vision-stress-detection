"""
Self-retraining loop (Phase 3.3)
================================
The product's self-improving property: the fusion layer retrains itself from
the user's OWN sessions, labelled by the user's OWN end-of-session reports -
not by the system's predictions, which would be the model grading its own
homework.

Data flow (all already in place):
  live session -> per-frame signals stored in samples (schema v5)
  Stop         -> "how stressed did you actually feel?" (1-5) -> self_report

This script turns that accumulation into a personal fusion model:
  self_report >= 4  -> frames labelled stressed
  self_report <= 2  -> frames labelled relaxed
  self_report == 3  -> excluded (ambiguous by the user's own account)

Personalization is scoped PER TASK MODE, warm-started from the population
meta-learner - not one model trained across all sessions regardless of
context (2026-08-23 redesign)
--------------------------------------------------------------------------
prototype_warmstart_personalization.py validated this on UBFC-Phys (real
data, 10 subjects) before anything here changed: personalizing on one
stressor and predicting a DIFFERENT one made things WORSE than not
personalizing at all (population-only beat every warm-started setting).
Personalizing and predicting WITHIN the same context was a genuine,
reproducible improvement (+1.8pp over population-only, +5.3pp over
from-scratch personal-only, at lambda=10). task_mode
(Default/Studying/Hospital/Driving) is this app's own proxy for "context",
so training one model across all your sessions and using it everywhere is
close to the failure mode that experiment found - a model per task mode
instead, each warm-started from ../models/meta_learner.joblib when that
exists (train_meta_learner.py --save), falling back to today's plain
from-scratch behavior when it doesn't.

DEFAULT_LAMBDA=10.0 is a FIXED constant taken from that experiment's
aggregate result across 10 subjects - it is NOT tuned per-user here (no
nested cross-validation is implemented). Every run reports lambda=0
(from-scratch, the old behavior) and DEFAULT_LAMBDA (warm-started)
side-by-side via session-held-out evaluation, on THIS user's own data, so
a human can see whether warm-starting actually helped before trusting it -
same "read the numbers, then decide" discipline the rest of this script
already has.

MIN_LABELED_SESSIONS / MIN_PER_CLASS apply PER TASK MODE now, not summed
across modes - a real, deliberate consequence of scoping by context: it
takes more total sessions to activate ANY mode than the old aggregate
gate did, because they have to concentrate within one mode. That is the
honest cost of the approach actually working, per the experiment above.

Versioned output: models/personal_fusion_{mode}_v{N}.joblib - never
overwrites, so before/after-adaptation comparisons stay possible.

Honesty guards, all enforced, per task mode:
  * refuses to train until both classes exist (someone who only ever reports
    "3" or only "5" cannot be learned from yet)
  * refuses below a minimum session count - a model fitted on two sessions
    is noise wearing a lab coat
  * evaluates session-held-out (leave-one-session-out), never on its own
    training frames
  * activation is a PRINTED INSTRUCTION, not automatic. A human flips
    USE_LEARNED_FUSION / LEARNED_FUSION_MODEL_PATH after reading the numbers.

Run it whenever; it says clearly what it is still waiting for:
    python retrain_from_sessions.py
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import config
import session_history
from session_history import SIGNAL_COLUMNS

MIN_LABELED_SESSIONS = 6        # PER TASK MODE - see module docstring
MIN_PER_CLASS = 2               # PER TASK MODE
STRESSED_AT = 4                 # self_report >= this -> stressed
RELAXED_AT = 2                  # self_report <= this -> relaxed

# From prototype_warmstart_personalization.py's same-task validation (real
# UBFC-Phys data, 10 subjects, personalize-and-eval within one context):
# lambda=10 had the best mean accuracy there (75.5%, vs 73.7%
# population-only, vs 70.2% from-scratch personal-only). A fixed default,
# not per-user-tuned - see module docstring.
DEFAULT_LAMBDA = 10.0
DEFAULT_PRIOR_PATH = '../models/meta_learner.joblib'


# ---------------------------------------------------------------------------
# MAP logistic regression with a Gaussian prior - identical math to
# prototype_warmstart_personalization.py's fit_logreg_with_prior(), which
# was verified there against sklearn (lambda=0) and against the raw prior
# (huge lambda) before this file started relying on it.
# ---------------------------------------------------------------------------
def fit_logreg_with_prior(X, y, w_prior, b_prior, lam):
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


def _wrap_as_sklearn_model(w, b):
    """
    learned_fusion.py (the consumer of these files) expects blob['model']
    to be an sklearn-compatible object with .predict_proba()/
    .decision_function() - not a raw (w, b) pair. sklearn's
    LogisticRegression.predict_proba() only reads coef_/intercept_/
    classes_ at inference time, so a never-.fit()'d instance with those
    three attributes set by hand behaves identically to a real fitted one
    (verified against a real .fit() call, including a joblib round-trip,
    before this was relied on here). classes_=[0,1] matches this file's
    own label convention (0=relaxed, 1=stressed) exactly.
    """
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.coef_ = w.reshape(1, -1)
    model.intercept_ = np.array([b])
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = w.shape[0]
    return model


def load_population_prior(path=None):
    """
    Returns {'w':, 'b':, 'scaler':} or None. None means "no prior
    available" - callers must treat that as lambda=0 (today's from-scratch
    behavior), not an error; models/meta_learner.joblib is an optional,
    separately-produced artifact (train_meta_learner.py --save), not
    guaranteed to exist.

    Refuses (returns None) rather than guesses if the saved model's
    feature order doesn't match SIGNAL_COLUMNS - applying a prior's
    weights to a differently-ordered feature vector would silently score
    the wrong signal against the wrong learned direction.
    """
    path = Path(path or DEFAULT_PRIOR_PATH)
    if not path.exists():
        return None
    import joblib
    try:
        blob = joblib.load(path)
    except Exception as e:
        print(f"  WARNING: could not load population prior {path}: {e} - "
             f"proceeding without one.")
        return None
    features = list(blob.get('features', []))
    if features != list(SIGNAL_COLUMNS):
        print(f"  WARNING: {path.name}'s feature order {features} does not "
             f"match SIGNAL_COLUMNS {list(SIGNAL_COLUMNS)} - ignoring the "
             f"population prior rather than risk misapplied weights.")
        return None
    model = blob['model']
    return {'w': model.coef_[0], 'b': model.intercept_[0], 'scaler': blob['scaler']}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataset():
    """
    (X, y, session_ids, skipped_reasons) from labelled sessions, flattened
    across ALL task modes. Kept for backward compatibility (existing
    callers/tests) - build_dataset_by_task_mode() below is what
    personalization actually trains on now.
    """
    sessions = session_history.labeled_sessions(min_samples=10)
    X, y, groups, skipped = [], [], [], []
    for s in sessions:
        rep = s['self_report']
        if RELAXED_AT < rep < STRESSED_AT:
            skipped.append((s['id'], f"self_report={rep} (ambiguous)"))
            continue
        label = 1 if rep >= STRESSED_AT else 0
        usable = _load_signal_rows(s['id'])
        if usable is None:
            skipped.append((s['id'], "pre-v5 session (no per-signal data)"))
            continue
        for r in usable:
            X.append([float(v) for v in r])
            y.append(label)
            groups.append(s['id'])
    return np.array(X), np.array(y), np.array(groups), skipped


def _load_signal_rows(session_id):
    """Raw per-signal rows for one session, or None if too few usable
    frames (pre-v5 session, no signal columns)."""
    conn = session_history._get_conn()
    try:
        cols = ", ".join(SIGNAL_COLUMNS)
        raw = conn.execute(
            f"SELECT {cols} FROM samples WHERE session_id = ? ORDER BY id",
            (session_id,)).fetchall()
    finally:
        conn.close()
    usable = [r for r in raw if all(v is not None for v in r)]
    return usable if len(usable) >= 10 else None


def build_dataset_by_task_mode():
    """
    {task_mode: (X, y, groups)} - same label rules as build_dataset(), now
    partitioned by which context (Default/Studying/Hospital/Driving) each
    session was recorded under. Returns (by_mode, skipped_reasons).
    """
    sessions = session_history.labeled_sessions(min_samples=10)
    by_mode = defaultdict(lambda: {'X': [], 'y': [], 'groups': []})
    skipped = []
    for s in sessions:
        rep = s['self_report']
        if RELAXED_AT < rep < STRESSED_AT:
            skipped.append((s['id'], f"self_report={rep} (ambiguous)"))
            continue
        label = 1 if rep >= STRESSED_AT else 0
        usable = _load_signal_rows(s['id'])
        if usable is None:
            skipped.append((s['id'], "pre-v5 session (no per-signal data)"))
            continue
        mode = s.get('task_mode') or 'Default'
        d = by_mode[mode]
        for r in usable:
            d['X'].append([float(v) for v in r])
            d['y'].append(label)
            d['groups'].append(s['id'])
    return ({mode: (np.array(d['X']), np.array(d['y']), np.array(d['groups']))
            for mode, d in by_mode.items()}, skipped)


# ---------------------------------------------------------------------------
# Per-mode fit + session-held-out evaluation
# ---------------------------------------------------------------------------
def evaluate_lambda(X, y, groups, w_prior, b_prior, lam, scaler=None):
    """Session-held-out (LeaveOneGroupOut) frame accuracy at one lambda -
    w_prior/b_prior may be all-zero (plain MLE, lambda's value then moot)
    when there is no population prior.

    scaler: when given, transforms BOTH train and test folds instead of
    fitting a fresh StandardScaler per fold. This MUST be the same scaler
    w_prior/b_prior were fit in (train_meta_learner.py --save's scaler) -
    the prior's weights only mean what they mean in that scaler's
    coordinate space. Fitting a fresh scaler on this user's own (much
    smaller) sample instead would standardize X into DIFFERENT units than
    w_prior lives in, so `(lam/2)*||w - w_prior||^2` would regularize
    toward a direction that means nothing in the fitted w's own units -
    silently producing a worse-than-either-alternative model (see
    BUG_AUDIT_2026-08-23.md finding #1). Pass None only when there is
    truly no prior (w_prior/b_prior are then zero and lambda is moot)."""
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    oof = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc = scaler if scaler is not None else StandardScaler().fit(X[tr])
        w, b = fit_logreg_with_prior(sc.transform(X[tr]), y[tr], w_prior, b_prior, lam)
        oof[te] = _predict(sc.transform(X[te]), w, b)
    acc = float((oof == y).mean())
    sess_ids = np.unique(groups)
    sess_ok = sum(int((oof[groups == g] == y[groups == g]).mean() > 0.5) for g in sess_ids)
    return acc, sess_ok, len(sess_ids)


def process_mode(mode, X, y, groups, prior, min_sessions, models_dir, lam=DEFAULT_LAMBDA):
    sess_ids = np.unique(groups)
    n_stress = len({g for g, lab in zip(groups, y) if lab == 1})
    n_relax = len({g for g, lab in zip(groups, y) if lab == 0})
    print(f"\n{'-' * 74}\n{mode}\n{'-' * 74}")
    print(f"  {len(sess_ids)} labelled sessions ({n_relax} relaxed, {n_stress} stressed), "
         f"{len(X)} frames")

    if len(sess_ids) < min_sessions or min(n_stress, n_relax) < MIN_PER_CLASS:
        print(f"  NOT TRAINING YET. Needs >= {min_sessions} labelled sessions "
             f"with >= {MIN_PER_CLASS} per class, in {mode} mode specifically.")
        return

    # Zero prior == plain MLE regardless of lambda's value, so "no prior
    # available" and "lambda=0" naturally coincide - no special-casing needed.
    w_prior = prior['w'] if prior else np.zeros(X.shape[1])
    b_prior = prior['b'] if prior else 0.0
    # w_prior/b_prior only mean what they mean in the population scaler's
    # coordinate space - every fit below that involves them (lambda=0
    # included, for a true apples-to-apples comparison) MUST transform X
    # with that same scaler, not a freshly-fit one. See evaluate_lambda's
    # docstring and BUG_AUDIT_2026-08-23.md finding #1.
    prior_scaler = prior['scaler'] if prior else None

    acc0, sok0, sn = evaluate_lambda(X, y, groups, w_prior, b_prior, 0.0,
                                     scaler=prior_scaler)
    print(f"  lambda=0   (from-scratch, today's old behavior) : "
         f"{acc0*100:5.1f}% frame-level, {sok0}/{sn} sessions")

    if prior is None:
        print("  no population prior available (../models/meta_learner.joblib "
             "not found or unusable) - training from scratch only.")
        chosen_lam, chosen_acc = 0.0, acc0
    else:
        acc_w, sok_w, _ = evaluate_lambda(X, y, groups, w_prior, b_prior, lam,
                                          scaler=prior_scaler)
        print(f"  lambda={lam:<5.1f}(warm-started)"
             f"{'':16s}: {acc_w*100:5.1f}% frame-level, {sok_w}/{sn} sessions")
        pop_pred = _predict(prior['scaler'].transform(X), w_prior, b_prior)
        acc_pop = float((pop_pred == y).mean())
        print(f"  population-only (zero personal data)            : "
             f"{acc_pop*100:5.1f}% frame-level (not session-held-out - it never "
             f"trains on your data at all)")
        # NOT a selection among candidates - DEFAULT_LAMBDA is a fixed
        # constant from the warm-start experiment (see module docstring),
        # always used verbatim when a prior exists. "chosen" here means
        # "what gets saved below", not "what scored best above".
        chosen_lam, chosen_acc = lam, acc_w

    # ---- fit on everything at the chosen lambda, save versioned ----
    # When there's a prior, reuse ITS scaler (see above) so the saved
    # w/b/scaler triple is internally consistent AND consistent with the
    # session-held-out numbers just printed. Only fit a fresh scaler on
    # this user's own data in the true no-prior path, where there is no
    # foreign coordinate space to stay consistent with.
    if prior is not None:
        sc = prior_scaler
    else:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
    w, b = fit_logreg_with_prior(sc.transform(X), y, w_prior, b_prior, chosen_lam)

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    mode_slug = mode.lower()

    import joblib
    blob = {
        # Same schema train_meta_learner.py --save already produces
        # ('model'/'scaler'/'features') so learned_fusion.py - the only
        # consumer of this file - needs no changes to load either kind.
        'model': _wrap_as_sklearn_model(w, b), 'scaler': sc,
        'features': list(SIGNAL_COLUMNS),
        'task_mode': mode, 'lambda': chosen_lam,
        'trained_on_sessions': [int(g) for g in sess_ids],
        'loso_accuracy': chosen_acc,
        'warm_started': prior is not None,
    }
    # glob-then-max-then-write is a TOCTOU race under real concurrent
    # invocation (two runs could compute the same next version); claim the
    # filename atomically via exclusive create ('xb', OS-level O_EXCL) and
    # retry on collision, instead of trusting the max() alone.
    version = 1 + max(
        [int(p.stem.rsplit('_v', 1)[-1])
         for p in models_dir.glob(f'personal_fusion_{mode_slug}_v*.joblib')] or [0])
    while True:
        out = models_dir / f"personal_fusion_{mode_slug}_v{version}.joblib"
        try:
            fh = open(out, 'xb')
        except FileExistsError:
            version += 1
            continue
        with fh:
            joblib.dump(blob, fh)
        break
    print(f"  saved: {out.name}  (lambda={chosen_lam}, "
         f"{'warm-started' if prior is not None else 'from-scratch, no prior available'})")

    print("  learned weights (|coef|, sign = direction):")
    for name, c in sorted(zip(SIGNAL_COLUMNS, w), key=lambda nc: -abs(nc[1])):
        print(f"    {name:18s} {c:+.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-sessions', type=int, default=MIN_LABELED_SESSIONS,
                    help='per task mode, not summed across modes')
    ap.add_argument('--models-dir', default='../models')
    ap.add_argument('--prior', default=None,
                    help=f'population meta-learner path (default: {DEFAULT_PRIOR_PATH})')
    ap.add_argument('--lambda', type=float, default=DEFAULT_LAMBDA, dest='lam',
                    help='warm-start regularization strength - see module docstring')
    args = ap.parse_args()

    print("=" * 74)
    print("SELF-RETRAINING - personal fusion from your own labelled sessions,")
    print("scoped per task mode, warm-started from the population model")
    print("=" * 74)

    session_history.init_db()

    by_mode, skipped = build_dataset_by_task_mode()
    for sid, why in skipped:
        print(f"  [skip] session {sid}: {why}")

    prior = load_population_prior(args.prior)
    print(f"\npopulation prior: "
         f"{'loaded from ' + str(Path(args.prior or DEFAULT_PRIOR_PATH)) if prior else 'NOT AVAILABLE - training from scratch per mode'}")

    if not by_mode:
        print("\n  No labelled sessions with usable signal data yet. Keep using the")
        print("  app and answering the end-of-session question honestly - that is")
        print("  the entire mechanism.")
        return 0

    for mode in sorted(by_mode):
        X, y, groups = by_mode[mode]
        process_mode(mode, X, y, groups, prior, args.min_sessions, args.models_dir,
                    lam=args.lam)

    print(f"\n{'=' * 74}")
    print("TO ACTIVATE (deliberately manual - read the numbers above first):")
    print("  config.USE_LEARNED_FUSION = True")
    print("  config.LEARNED_FUSION_MODEL_PATH = '../models/personal_fusion_<mode>_vN.joblib'")
    print("  (pick the mode + version whose printed accuracy actually looked good -")
    print("   nothing above activates anything by itself)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
