"""
Meta-learner retrain on the team's own recordings (Phase 3.2 follow-up)
========================================================================
train_meta_learner.py's LOSO study was only ever run on UBFC-Phys (someone
else's data). This script does the same LOSO-validated comparison
(hand-tuned weights vs. learned logistic regression) but on the team's own
labelled sessions in data/sessions.db - the exact data USE_LEARNED_FUSION
was kept off pending, per config.py's own comment.

Per-frame behavioral signals (eye_strain, blink_stress, brow_tension,
jaw_tension, head_instability, neck_strain, slouch) are stored in the
samples table (schema v5a) even though the exported evaluation_data/ CSVs
don't include them - this script reads directly from the database via
session_history.py, the same way retrain_from_sessions.py's personalization
pipeline already does, rather than the feature_traces/ CSV path
train_meta_learner.py uses for UBFC-Phys.

Label: the session's own self-report (1-5), NOT the recorded condition -
self-report is the ground truth this project treats as real (Phase 3.1).
Rating 3 is excluded as ambiguous, matching every other place self-report
is binarized. A session whose self-report ends up all-one-class after this
(e.g. S2/S3, see the printed warning) still contributes real held-out
frames to its own LOSO fold, but that fold isn't testing discrimination,
only recall for whichever class survived - reported honestly, not hidden.

This is a first look, not a decision: the result is printed and nothing is
switched on automatically, matching config.USE_LEARNED_FUSION's existing
manual-activation policy.

Usage
-----
    cd code && python retrain_meta_learner_from_team.py
"""

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

import config
import session_history
from learned_fusion import FEATURE_ORDER
from train_meta_learner import handtuned_predictions

STRESSED_MIN = 4
RELAXED_MAX = 2
NEUTRAL = 3

# S2/S3 were recorded under the old, unanchored self-report scale (points 2
# and 4 had no label text) - both raters have since said they reserve the
# top of the scale for extreme distress, so their "stressed" condition
# likely landed as a 2-3 despite a real, moderate stress response. Not a
# data error, a scale-calibration mismatch (see RECORDING_PROTOCOL.md's
# 2026-09-16 fix). Excluded here rather than trained on with a mislabelled
# stressed class - their raw sessions are NOT deleted, only left out of this
# fit, so they can be re-included once re-recorded under the fixed scale.
EXCLUDED_SUBJECTS = {'S2', 'S3'}


def load_team_frames(min_samples=10):
    """
    Returns (X, y, groups): per-frame behavioral feature rows, binary
    self-report labels (rating 3 excluded), and the subject name each row
    belongs to (for LOSO grouping).
    """
    sessions = session_history.labeled_sessions(min_samples=min_samples)
    users = {u['id']: u['name'] for u in session_history.list_users()}

    X, y, groups = [], [], []
    per_subject_labels = {}
    for sess in sessions:
        # Anonymous/legacy sessions (user_id NULL, pre-dating multi-user
        # profiles) are a different population entirely and would silently
        # get treated as their own "subject" via the f-string fallback
        # below - explicitly excluded rather than accidentally pooled in.
        if sess['user_id'] is None or sess['user_id'] not in users:
            continue
        if users[sess['user_id']] in EXCLUDED_SUBJECTS:
            continue
        rating = sess['self_report']
        if rating == NEUTRAL:
            continue
        label = 1 if rating >= STRESSED_MIN else 0
        subject = users[sess['user_id']]
        rows = session_history.load_session_signal_rows(sess['id'])
        for r in rows:
            X.append([r.get(name, 0.0) or 0.0 for name in FEATURE_ORDER])
            y.append(label)
            groups.append(subject)
        per_subject_labels.setdefault(subject, set()).add(label)

    return np.array(X), np.array(y), np.array(groups), per_subject_labels


def main():
    X, y, groups, per_subject_labels = load_team_frames()
    n_subjects = len(set(groups))

    print("=" * 78)
    print("META-LEARNER RETRAIN ON TEAM DATA (first look, not a decision)")
    print("=" * 78)
    print(f"  frames         : {len(y)}")
    print(f"  subjects       : {n_subjects} ({sorted(set(groups))})")
    print(f"  class balance  : {int((y == 0).sum())} relaxed / {int((y == 1).sum())} stressed")
    print()

    for subj, labels in sorted(per_subject_labels.items()):
        if len(labels) < 2:
            only = 'relaxed' if 0 in labels else 'stressed'
            print(f"  WARNING: {subj}'s labelled sessions are ALL '{only}' after excluding "
                  f"rating-3 - their LOSO fold tests recall only, not discrimination.")
    print()

    if n_subjects < 2:
        print("Need >= 2 subjects with labelled sessions for a LOSO comparison. Stopping.")
        return 1

    # ---- Hand-tuned baseline on this exact data ----
    # BUG FIX: this used to be a hand-rolled raw dot product (no
    # BEHAVIOR_AMPLIFICATION, no w.sum() normalization) thresholded at
    # this sample's own MEDIAN score - a data-fitted threshold, not the
    # live system's actual fixed cutoff (config.LOW_TO_MODERATE_CUTOFF =
    # 0.36). A median split is guaranteed ~50/50 by construction regardless
    # of real separability, so the "hand-tuned" number it produced was not
    # a real measurement of what the live fusion would score on this data.
    # Reuses train_meta_learner.py's own handtuned_predictions() - the
    # exact formula (weighted average / w.sum(), x1.2 amplification, clip
    # to [0,1], threshold at 0.36) the live app actually runs - instead of
    # a second, drifted reimplementation.
    pred_hand = handtuned_predictions(X, FEATURE_ORDER)
    acc_hand = accuracy_score(y, pred_hand)
    bal_hand = balanced_accuracy_score(y, pred_hand)
    f1_hand = f1_score(y, pred_hand, zero_division=0)

    # ---- Learned logistic regression, LOSO by subject ----
    logo = LeaveOneGroupOut()
    fold_acc, fold_bal, fold_f1 = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(set(y[train_idx])) < 2:
            continue  # can't fit a classifier with only one class in training
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        fold_acc.append(accuracy_score(y[test_idx], pred))
        fold_bal.append(balanced_accuracy_score(y[test_idx], pred))
        fold_f1.append(f1_score(y[test_idx], pred, zero_division=0))

    print("Hand-tuned weights (current live default, real 0.36 cutoff, same as production):")
    print(f"  accuracy: {acc_hand*100:.2f}%   balanced accuracy: {bal_hand*100:.2f}%   F1: {f1_hand:.3f}")
    print()
    print(f"Learned logistic regression (LOSO across {len(fold_acc)} usable folds):")
    if fold_acc:
        print(f"  accuracy: {np.mean(fold_acc)*100:.2f}%   "
              f"balanced accuracy: {np.mean(fold_bal)*100:.2f}%   "
              f"F1: {np.mean(fold_f1):.3f}")
    else:
        print("  no fold had both classes in its training set - cannot fit LOSO with this "
              "few subjects yet.")

    print()
    print("Full-data fit (all subjects pooled, NOT held out - for inspecting learned "
          "weights only, not a validation number):")
    clf_full = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X, y)
    for name, coef, hand_w in zip(FEATURE_ORDER, clf_full.coef_[0],
                                  [config.BEHAVIOR_WEIGHTS[n] for n in FEATURE_ORDER]):
        direction = "stress up" if coef > 0 else "INVERTED - stress down"
        print(f"  {name:18s} hand={hand_w:.2f}  learned_coef={coef:+.3f}  {direction}")

    print()
    print("This is a first look at n={} subjects - NOT enough to responsibly switch "
          "config.USE_LEARNED_FUSION to True yet. Re-run as more team recordings land."
          .format(n_subjects))
    return 0


if __name__ == '__main__':
    sys.exit(main())
