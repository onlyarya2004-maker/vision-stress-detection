"""
Temporal (GRU) retrain on UBFC-Phys instead of DaiSEE
======================================================
The DaiSEE-trained GRU (models/temporal_gru_test_report.txt) scored 45.74%,
below the 86.6% majority floor - and a threshold sweep against its real
held-out test predictions confirmed this is a genuine "didn't learn a
strong signal" problem, not a calibration one (best achievable accuracy at
any threshold just degenerates to the trivial majority classifier, 0%
recall on stressed).

Two independent reasons to expect UBFC-Phys does better as training data:

1. Volume: DaiSEE gave 8569 sixteen-frame clips total (train+val+test),
   each clip contributing exactly ONE window. UBFC-Phys's 30 continuous
   per-(subject,task) traces (datasets/feature_traces/, already computed,
   already used elsewhere in this project as evaluation ground truth) total
   123,728 frames - sliced into overlapping 16-frame windows this yields
   far more real training sequences.

2. Feature completeness: DaiSEE's processed_daisee/ crops are face-only,
   so neck_strain/slouch are structurally ~0 for nearly every frame in
   that training set (documented in temporal_gru_test_report.txt).
   Verified directly against UBFC-Phys: posture_detected=100%, neck_strain
   non-zero for 75.7% of frames, slouch for 70.3% - two of the ten input
   dimensions that were DEAD in the old training data carry real signal
   here.

Label: UBFC-Phys's own experimental design (T1=rest=relaxed=0,
T2=speech/T3=arithmetic=stressed=1) - the same ground truth this project
already trusts for the ablation study, the meta-learner LOSO comparison,
and the cross-subject accuracy headline number, not a new assumption
introduced here.

Reuses GRUStressModel/train_model/evaluate/verify_export/export_to_numpy
from train_temporal_model.py unchanged - only the DATA LOADING differs
(sliding windows over continuous per-subject traces instead of DaiSEE's
one-clip-one-window format), so the model, training procedure, and export/
verification rigor are identical to the DaiSEE run, making this a fair,
same-architecture comparison of training data alone.

This is a first look, not a decision: results are reported honestly and
exported to a SEPARATE file (models/temporal_gru_ubfc.npz), never
overwriting the live models/temporal_gru.npz. Switching the live model
requires manually renaming/copying, exactly like every other "manual
activation" gate in this project (USE_LEARNED_FUSION, W_RPPG, etc.).

MUST be run with .venv-train's python (PyTorch + CUDA):
    .venv-train\\Scripts\\python train_temporal_model_ubfc.py
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from temporal_model import TEMPORAL_FEATURE_ORDER, normalize_features
from train_temporal_model import (
    GRUStressModel, train_model, evaluate, verify_export, export_to_numpy,
    compute_summary_stats, run_baseline_ablation, _weights_dict,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_history  # noqa: E402

NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)
SEQ_LEN = 16


def load_trace(path):
    """(T, 10) float array in TEMPORAL_FEATURE_ORDER, or None if any
    required column is missing/unparseable for the whole file (skip
    rather than silently zero-fill an entire malformed trace)."""
    import csv
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    try:
        arr = np.array(
            [[float(r[name]) for name in TEMPORAL_FEATURE_ORDER] for r in rows],
            dtype=np.float64,
        )
    except (KeyError, ValueError):
        return None
    return arr


def windows_from_trace(X, label, seq_len=SEQ_LEN, stride=8):
    n = X.shape[0]
    windows, labels = [], []
    s = 0
    while s + seq_len <= n:
        windows.append(X[s:s + seq_len])
        labels.append(label)
        s += stride
    return windows, labels


def build_dataset(trace_dir, seq_len=SEQ_LEN, stride=8):
    """
    Returns (X, y, groups): (N, seq_len, 10) windows, binary labels
    (T1=0, T2/T3=1), and the subject id ('s1'..'s10') each window belongs
    to, for LOSO grouping - same subject split UBFC-Phys uses everywhere
    else in this project.
    """
    X_all, y_all, groups_all = [], [], []
    skipped = []
    for path in sorted(Path(trace_dir).glob('features_vid_*.csv')):
        m = NAME_RE.search(path.stem)
        if not m:
            continue
        subj, task = f"s{m.group(1)}", int(m.group(2))
        label = 0 if task == 1 else 1
        trace = load_trace(path)
        if trace is None or trace.shape[0] < seq_len:
            skipped.append(path.name)
            continue
        windows, labels = windows_from_trace(trace, label, seq_len, stride)
        X_all.extend(windows)
        y_all.extend(labels)
        groups_all.extend([subj] * len(windows))

    if skipped:
        print(f"  skipped (too short or malformed): {skipped}")
    return (np.array(X_all, dtype=np.float64), np.array(y_all, dtype=np.int64),
            np.array(groups_all))


def build_team_dataset(seq_len=SEQ_LEN, stride=8):
    """
    Sliding-window sequences from REAL webcam recordings in the session
    history DB (not UBFC-Phys) - only sessions recorded after the v7
    schema migration have the 3 extra columns GRU windows need, so this
    is currently just S5's two sessions (relaxed=60, stressed=61). Used
    ONLY as an extra, never-trained-on evaluation set: a genuine
    domain-transfer check (real webcam + this project's own feature
    engine, not UBFC-Phys's study cameras) that the UBFC LOSO loop above
    cannot provide by itself. Returns (X, y, subject) or (None, None, None)
    if no v7-complete session exists yet.
    """
    import sqlite3
    conn = sqlite3.connect(str(session_history.DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        sessions = conn.execute(
            "SELECT id, user_id, self_report FROM sessions "
            "WHERE self_report IS NOT NULL ORDER BY id"
        ).fetchall()
        users = {r['id']: r['name'] for r in conn.execute("SELECT id, name FROM users")}
    finally:
        conn.close()

    X_all, y_all, subj_all = [], [], []
    cols = TEMPORAL_FEATURE_ORDER
    for sess in sessions:
        # Only trust unambiguous self-reports (1-2 = relaxed, 4-5 =
        # stressed) - a "3" is exactly the calibration-ambiguity problem
        # RECORDING_PROTOCOL.md now documents, not a clean label.
        if sess['self_report'] == 3:
            continue
        label = 0 if sess['self_report'] <= 2 else 1
        rows = session_history.load_session_temporal_rows(sess['id'])
        if not rows or any(rows[0][c] is None for c in cols):
            continue  # pre-v7 session, missing ear/blink columns
        trace = np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)
        windows, labels = windows_from_trace(trace, label, seq_len, stride)
        X_all.extend(windows)
        y_all.extend(labels)
        subj_all.extend([users.get(sess['user_id'], f"user{sess['user_id']}")] * len(windows))

    if not X_all:
        return None, None, None
    return (np.array(X_all, dtype=np.float64), np.array(y_all, dtype=np.int64),
            np.array(subj_all))


def print_report(title, result):
    print(f"\n  {title}")
    print(f"    n                    : {result['n']}")
    print(f"    accuracy             : {result['accuracy']*100:.2f}%")
    print(f"    balanced accuracy    : {result['balanced_accuracy']*100:.2f}%")
    print(f"    precision (stressed) : {result['precision']:.3f}")
    print(f"    recall (stressed)    : {result['recall']:.3f}")
    print(f"    f1 (stressed)        : {result['f1']:.3f}")
    print(f"    majority-class floor : {result['majority_floor']*100:.2f}%")
    verdict = "ABOVE" if result['above_floor'] else "BELOW"
    print(f"    verdict              : {verdict} the majority-class floor")
    print(f"    confusion matrix (rows=true, cols=pred [not_stressed, stressed]):")
    print(f"      {result['confusion_matrix'][0].tolist()}")
    print(f"      {result['confusion_matrix'][1].tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace-dir', default='../datasets/feature_traces')
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--hidden-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--n-verify', type=int, default=20)
    ap.add_argument('--export-out', default='../models/temporal_gru_ubfc.npz')
    ap.add_argument('--report-out', default='../models/temporal_gru_ubfc_test_report.txt')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    print("\n" + "=" * 74)
    print("BUILDING SLIDING-WINDOW DATASET FROM UBFC-PHYS TRACES")
    print("=" * 74)
    X, y, groups = build_dataset(args.trace_dir, stride=args.stride)
    subjects = sorted(set(groups))
    print(f"  windows        : {len(y)}")
    print(f"  subjects       : {len(subjects)} ({subjects})")
    print(f"  class balance  : {int((y == 0).sum())} relaxed / {int((y == 1).sum())} stressed")

    rng = np.random.RandomState(args.seed)

    # ---- LOSO ----
    print("\n" + "=" * 74)
    print("LEAVE-ONE-SUBJECT-OUT EVALUATION")
    print("=" * 74)
    all_y_true, all_y_pred_prob = [], []
    gbt_fold_accs = []

    for held_out in subjects:
        train_mask = groups != held_out
        test_mask = ~train_mask
        Xtr_full, ytr_full = X[train_mask], y[train_mask]
        groups_full = groups[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        if len(set(ytr_full)) < 2 or len(set(yte)) < 1:
            print(f"  [{held_out}] skipped - training set missing a class")
            continue

        # BUG FIX: this used to be a random 85/15 split of the TRAINING
        # POOL's own frames - meaning "validation" frames came from the
        # exact same subjects the model trained on, just different moments.
        # Early stopping against that measured "has it memorized these 8
        # people," not "does it generalize to a new one" - and it showed:
        # a fully-converged run (val_balanced_acc up to 0.97) scored WORSE
        # on the real held-out subject (70.68%) than a barely-trained
        # 2-epoch smoke test did (79.43%). Fixed by holding out a SECOND,
        # genuinely different subject from training entirely and using
        # THEIR frames as the validation set - a real nested-LOSO proxy for
        # "will this work on someone new," which is what early stopping is
        # actually supposed to measure here.
        remaining = [s for s in subjects if s != held_out and s in set(groups_full)]
        val_subject = remaining[rng.randint(len(remaining))]
        val_mask = groups_full == val_subject
        tr_mask = ~val_mask
        Xtr, ytr = Xtr_full[tr_mask], ytr_full[tr_mask]
        Xval, yval = Xtr_full[val_mask], ytr_full[val_mask]

        feature_mean = Xtr.mean(axis=(0, 1))
        feature_std = Xtr.std(axis=(0, 1))

        print(f"\n  --- held out: {held_out} (train windows={len(Xtr)}, "
              f"val={len(Xval)}, test={len(Xte)}) ---")
        model, best_val = train_model(
            Xtr, ytr.astype(np.float32), Xval, yval,
            feature_mean, feature_std,
            epochs=args.epochs, hidden_size=args.hidden_size, lr=args.lr,
            patience=args.patience, batch_size=args.batch_size, device=device,
            log_path=None, seed=args.seed,
        )

        model.eval()
        Xte_n = torch.tensor(normalize_features(Xte, feature_mean, feature_std),
                             dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(Xte_n)).cpu().numpy()
        all_y_true.extend(yte.tolist())
        all_y_pred_prob.extend(probs.tolist())

        # GBT baseline on the same fold, same summary-stats representation
        # train_temporal_model.py already uses for its own GBT comparison.
        # Reported per-fold only (not pooled into a final aggregate) - it's
        # a secondary sanity comparison, the GRU is the primary result. A
        # blocked sklearn.ensemble DLL (see train_temporal_model.py's own
        # comment) must not take down the GRU run over a secondary metric.
        try:
            Xtr_stats = compute_summary_stats(Xtr_full)
            Xte_stats = compute_summary_stats(Xte)
            gbt_result = run_baseline_ablation(Xtr_stats, ytr_full, Xte_stats, yte)
            gbt_fold_accs.append(gbt_result['accuracy'])
            print(f"    [{held_out}] GBT this fold: acc={gbt_result['accuracy']*100:.1f}%  "
                  f"balanced_acc={gbt_result['balanced_accuracy']*100:.1f}%")
        except ImportError as e:
            print(f"    [{held_out}] GBT baseline unavailable in this environment ({e}) - skipped")

    all_y_true = np.array(all_y_true)
    all_y_pred_prob = np.array(all_y_pred_prob)
    gru_result = evaluate(all_y_true, all_y_pred_prob)
    print_report("GRU (feature-sequence temporal model, trained on UBFC-Phys)", gru_result)

    if gbt_fold_accs:
        print(f"\n  GBT baseline (per-fold mean, secondary comparison only): "
              f"{100*np.mean(gbt_fold_accs):.2f}%")

    old = {'accuracy': 0.4574, 'majority_floor': 0.8660}
    print(f"\n  OLD (DaiSEE-trained) GRU : {old['accuracy']*100:.2f}% accuracy, "
          f"BELOW its {old['majority_floor']*100:.2f}% floor")
    print(f"  NEW (UBFC-trained) GRU  : {gru_result['accuracy']*100:.2f}% accuracy, "
          f"{'ABOVE' if gru_result['above_floor'] else 'BELOW'} its "
          f"{gru_result['majority_floor']*100:.2f}% floor")

    # ---- Final export candidate: TRUE 3-way subject split ----
    print("\n" + "=" * 74)
    print("EXPORT CANDIDATE: train / early-stop-val / held-out-TEST subjects")
    print("=" * 74)
    # PREVIOUS BUG (found via ablation_study.py after activating
    # W_TEMPORAL=0.10): this export fit trained on 7 of 10 subjects and
    # pooled the other 3 for early-stopping validation ONLY - meaning when
    # ablation_study.py later scored this exact exported file across ALL
    # 10 subjects' cached traces, 7 of them were in-sample. That's why
    # removing the (genuinely trustworthy) behavioral branch appeared to
    # RAISE cross-subject accuracy from ~60% to 88-100% - the temporal
    # branch was recognizing people it had memorized, not detecting
    # stress. The LOSO loop above is unaffected by this (every fold's
    # test subject really is held out) and its ~76-77% remains an honest
    # estimate of the METHOD. But that loop retrains a fresh model per
    # fold and throws each one away - it never produces a single
    # deployable artifact. This section now fixes the deployable artifact
    # itself: a THIRD group of subjects is set aside before training even
    # starts and touched by nothing until the very end - not training,
    # not early-stopping. Its accuracy is the true, honest number for the
    # actual file that would ship.
    # round() rather than floor() so the held-out test set actually grows as
    # subjects are added (2026-09-16: 10 subjects gave floor(10/4)=2, but
    # floor(11/4) is STILL 2 - the whole point of processing an 11th subject
    # was to afford a 3rd genuinely held-out test subject, per this file's
    # own "n=2 is still thin" note; round(11/4)=3 while round(10/4)=2 keeps
    # the original 10-subject behavior unchanged).
    n_test_subjects = min(4, max(1, round(len(subjects) / 4)))
    # str(s) on each choice - rng.choice() over a list of Python strings
    # returns numpy.str_ elements, whose repr (np.str_('s9')) leaked
    # straight into the printed report and the HELD_OUT_TEST_SUBJECTS line
    # otherwise; plain str keeps both human- and machine-readable.
    export_test_subjects = [str(s) for s in rng.choice(subjects, size=n_test_subjects, replace=False)]
    remaining_subjects = [s for s in subjects if s not in export_test_subjects]
    n_val_subjects = min(3, max(1, len(remaining_subjects) // 3))
    export_val_subjects = [str(s) for s in rng.choice(remaining_subjects, size=n_val_subjects, replace=False)]
    train_subjects = [s for s in remaining_subjects if s not in export_val_subjects]

    test_mask = np.isin(groups, export_test_subjects)
    val_mask = np.isin(groups, export_val_subjects)
    tr_mask = np.isin(groups, train_subjects)
    print(f"  train subjects       (fit on these)      : {train_subjects}")
    print(f"  validation subjects  (early-stop only)    : {export_val_subjects}")
    print(f"  TEST subjects        (never touched until scoring): {export_test_subjects}")
    feature_mean = X[tr_mask].mean(axis=(0, 1))
    feature_std = X[tr_mask].std(axis=(0, 1))
    final_model, _ = train_model(
        X[tr_mask], y[tr_mask].astype(np.float32), X[val_mask], y[val_mask],
        feature_mean, feature_std,
        epochs=args.epochs, hidden_size=args.hidden_size, lr=args.lr,
        patience=args.patience, batch_size=args.batch_size, device=device,
        log_path=None, seed=args.seed,
    )

    final_model.eval()
    Xtest_n = torch.tensor(normalize_features(X[test_mask], feature_mean, feature_std),
                           dtype=torch.float32).to(device)
    with torch.no_grad():
        test_probs = torch.sigmoid(final_model(Xtest_n)).cpu().numpy()
    artifact_result = evaluate(y[test_mask], test_probs)
    print_report(f"EXPORTED ARTIFACT - true held-out test ({export_test_subjects})", artifact_result)

    # ---- Bonus: real webcam domain-transfer check (S5, never trained on) ----
    team_X, team_y, team_subj = build_team_dataset()
    team_result = None
    if team_X is not None:
        team_Xn = torch.tensor(normalize_features(team_X, feature_mean, feature_std),
                               dtype=torch.float32).to(device)
        with torch.no_grad():
            team_probs = torch.sigmoid(final_model(team_Xn)).cpu().numpy()
        team_result = evaluate(team_y, team_probs)
        team_subjects_used = sorted(set(team_subj.tolist()))
        print_report(f"REAL WEBCAM CHECK - team recordings {team_subjects_used} "
                     f"(never in UBFC training data)", team_result)
    else:
        print("\n  (no v7-complete team session available yet for the webcam check)")

    verify_idx = rng.choice(len(X), size=min(args.n_verify, len(X)), replace=False)
    verify_X_norm = normalize_features(X[verify_idx], feature_mean, feature_std)
    ok, verify_Y, detail = verify_export(final_model, _weights_dict(final_model, args.hidden_size),
                                        verify_X_norm, device)
    if not ok:
        print("EXPORT PARITY CHECK FAILED - not writing an artifact.")
        for line in detail:
            print(line)
        return 1

    weights = export_to_numpy(final_model, feature_mean, feature_std,
                              verify_X_norm, verify_Y, args.hidden_size)
    out_path = Path(args.export_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **weights)
    print(f"  export parity check: OK ({len(verify_idx)}/{len(verify_idx)} sequences)")
    print(f"  written: {out_path}  (NOT the live model - models/temporal_gru.npz is untouched)")

    report_lines = [
        "UBFC-PHYS TEMPORAL MODEL - v2 (true held-out test subjects)",
        "=" * 74,
        # Machine-parseable, not just prose - ablation_study.py's
        # load_held_out_test_subjects() reads this exact line so
        # run_all_evaluations.py can automatically re-run the leak-free
        # contamination check (run_ablation_study(subjects=...)) every time
        # it regenerates CANONICAL_RESULTS, instead of that check only ever
        # existing as a one-off manual command someone has to remember.
        f"HELD_OUT_TEST_SUBJECTS: {','.join(export_test_subjects)}",
        f"windows={len(y)}  subjects={len(subjects)}",
        f"class balance: {int((y==0).sum())} relaxed / {int((y==1).sum())} stressed",
        "",
        "LOSO GRU result (method-level estimate, fresh model per fold, all",
        "10 subjects each genuinely held out once, nothing deployable comes",
        "out of this loop - it is a sanity check on the APPROACH):",
        f"  accuracy={gru_result['accuracy']*100:.2f}%  "
        f"balanced_accuracy={gru_result['balanced_accuracy']*100:.2f}%  "
        f"precision={gru_result['precision']:.3f}  recall={gru_result['recall']:.3f}  "
        f"f1={gru_result['f1']:.3f}",
        f"  majority floor: {gru_result['majority_floor']*100:.2f}%  "
        f"verdict: {'ABOVE' if gru_result['above_floor'] else 'BELOW'}",
        "",
        f"EXPORTED ARTIFACT result (the actual file below, scored ONLY on "
        f"{export_test_subjects}, never seen during its training or its "
        f"early-stopping - this is the number that matters for deployment):",
        f"  accuracy={artifact_result['accuracy']*100:.2f}%  "
        f"balanced_accuracy={artifact_result['balanced_accuracy']*100:.2f}%  "
        f"precision={artifact_result['precision']:.3f}  recall={artifact_result['recall']:.3f}  "
        f"f1={artifact_result['f1']:.3f}",
        f"  majority floor: {artifact_result['majority_floor']*100:.2f}%  "
        f"verdict: {'ABOVE' if artifact_result['above_floor'] else 'BELOW'}",
        "",
    ]
    if team_result is not None:
        report_lines += [
            "REAL WEBCAM CHECK (team recordings, this project's own feature",
            "engine and camera pipeline, never in UBFC training data - the",
            "closest thing to a live-deployment test available today):",
            f"  accuracy={team_result['accuracy']*100:.2f}%  "
            f"balanced_accuracy={team_result['balanced_accuracy']*100:.2f}%  "
            f"precision={team_result['precision']:.3f}  recall={team_result['recall']:.3f}  "
            f"f1={team_result['f1']:.3f}",
            f"  majority floor: {team_result['majority_floor']*100:.2f}%  "
            f"verdict: {'ABOVE' if team_result['above_floor'] else 'BELOW'}",
            "",
        ]
    report_lines += [
        f"old DaiSEE-trained GRU: 45.74% accuracy, BELOW its 86.60% floor",
        "",
        "The export at models/temporal_gru_ubfc.npz is NOT the live model.",
        "Manually copy it to models/temporal_gru.npz and set config.W_TEMPORAL",
        "to a nonzero value to activate, after review.",
    ]
    Path(args.report_out).write_text("\n".join(report_lines) + "\n", encoding='utf-8')
    print(f"  report written: {args.report_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
