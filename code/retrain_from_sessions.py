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

Versioned output: models/personal_fusion_v{N}.joblib - never overwrites, so
the before/after-adaptation comparison the plan calls for stays possible.

Honesty guards, all enforced:
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
from pathlib import Path

import numpy as np

import config
import session_history
from session_history import SIGNAL_COLUMNS

MIN_LABELED_SESSIONS = 6        # per class-bearing total, see guard below
MIN_PER_CLASS = 2               # sessions per class before training
STRESSED_AT = 4                 # self_report >= this -> stressed
RELAXED_AT = 2                  # self_report <= this -> relaxed


def build_dataset():
    """(X, y, session_ids, skipped_reasons) from labelled sessions."""
    sessions = session_history.labeled_sessions(min_samples=10)
    X, y, groups, skipped = [], [], [], []
    for s in sessions:
        rep = s['self_report']
        if RELAXED_AT < rep < STRESSED_AT:
            skipped.append((s['id'], f"self_report={rep} (ambiguous)"))
            continue
        label = 1 if rep >= STRESSED_AT else 0
        rows = session_history.load_session_samples(s['id'])
        # Per-signal values arrive via the raw row dicts; re-query with them.
        conn = session_history._get_conn()
        try:
            cols = ", ".join(SIGNAL_COLUMNS)
            raw = conn.execute(
                f"SELECT {cols} FROM samples WHERE session_id = ? ORDER BY id",
                (s['id'],)).fetchall()
        finally:
            conn.close()
        usable = [r for r in raw if all(v is not None for v in r)]
        if len(usable) < 10:
            skipped.append((s['id'], "pre-v5 session (no per-signal data)"))
            continue
        for r in usable:
            X.append([float(v) for v in r])
            y.append(label)
            groups.append(s['id'])
    return np.array(X), np.array(y), np.array(groups), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min-sessions', type=int, default=MIN_LABELED_SESSIONS)
    ap.add_argument('--models-dir', default='../models')
    args = ap.parse_args()

    print("=" * 74)
    print("SELF-RETRAINING - personal fusion from your own labelled sessions")
    print("=" * 74)

    # Run migrations first: this script must work even if the app has never
    # been launched since the schema gained self_report / signal columns.
    session_history.init_db()

    X, y, groups, skipped = build_dataset()
    for sid, why in skipped:
        print(f"  [skip] session {sid}: {why}")

    sess_ids = np.unique(groups) if len(groups) else []
    n_stress = len({g for g, lab in zip(groups, y) if lab == 1})
    n_relax = len({g for g, lab in zip(groups, y) if lab == 0})
    print(f"\n  usable sessions: {len(sess_ids)} "
          f"({n_relax} relaxed-labelled, {n_stress} stressed-labelled), "
          f"{len(X)} frames")

    # ---- honesty guards ----
    if len(sess_ids) < args.min_sessions or min(n_stress, n_relax) < MIN_PER_CLASS:
        print(f"\n  NOT TRAINING YET. Needs >= {args.min_sessions} labelled "
              f"sessions with >= {MIN_PER_CLASS} per class.")
        print("  Keep using the app and answering the end-of-session question")
        print("  honestly - that is the entire mechanism.")
        return 0

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    # ---- session-held-out evaluation ----
    oof = np.zeros_like(y)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, class_weight='balanced')
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    acc = float((oof == y).mean())

    # per-session verdicts (majority vote) - the honest denominator
    sess_ok = sum(
        int((oof[groups == g] == y[groups == g]).mean() > 0.5)
        for g in sess_ids)
    print(f"\n  leave-one-session-out: {acc*100:.1f}% frame-level, "
          f"{sess_ok}/{len(sess_ids)} sessions")

    # ---- fit on everything, save versioned ----
    sc = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(sc.transform(X), y)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    version = 1 + max(
        [int(p.stem.split('_v')[-1])
         for p in models_dir.glob('personal_fusion_v*.joblib')] or [0])
    out = models_dir / f"personal_fusion_v{version}.joblib"

    import joblib
    joblib.dump({'model': model, 'scaler': sc,
                 'features': list(SIGNAL_COLUMNS),
                 'trained_on_sessions': [int(g) for g in sess_ids],
                 'loso_accuracy': acc}, out)
    print(f"\n  saved: {out.name}")

    print("\n  learned weights (|coef|, sign = direction):")
    for name, c in sorted(zip(SIGNAL_COLUMNS, model.coef_[0]),
                          key=lambda nc: -abs(nc[1])):
        print(f"    {name:18s} {c:+.2f}")

    print("\n  TO ACTIVATE (deliberately manual - read the numbers first):")
    print(f"    config.USE_LEARNED_FUSION = True")
    print(f"    config.LEARNED_FUSION_MODEL_PATH = '../models/{out.name}'")
    return 0


if __name__ == '__main__':
    sys.exit(main())
