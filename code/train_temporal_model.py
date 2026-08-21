"""
Phase 2.1 temporal model training
==================================
Trains the GRU that replaces the retired pixel-LSTM (models/lstm_test_report.txt:
71.13% accuracy, BELOW the 86.6% majority-class floor - "frozen ImageNet
features carry no stress signal"). This one trains on BEHAVIORAL-FEATURE
sequences (BehavioralFeatureEngine.extract() output over the already-
extracted datasets/processed_daisee/ crops) instead of raw pixels.

MUST be run with .venv-train's python (has PyTorch + CUDA; the main venv
only has TensorFlow):

    .venv-train\\Scripts\\python train_temporal_model.py
    .venv-train\\Scripts\\python train_temporal_model.py --smoke-test

Ordering note: split validation, feature extraction, training, and the
export/parity check ALL run inside main() in that order, and the .npz is
only ever written after every parity comparison has passed - see
export_to_numpy()/verify_export() below. Model artifacts are saved before
the final diagnostic report is printed (same crash-safety ordering
train_meta_learner.py adopted after a real Windows cp1252 console crash
from non-ASCII characters in a printed table - this file's own tables are
plain-ASCII-only for the same reason).
"""

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import config
from behavioral_features import BehavioralFeatureEngine
from temporal_model import TEMPORAL_FEATURE_ORDER, gru_forward_numpy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

# DaiSEE clips are ~10s; daisee_preprocessing.py sampled 16 frames evenly
# across each one -> ~1.6Hz effective rate. Only affects blinks_per_minute's
# scale, which is normalized away before training regardless.
FRAME_RATE_HZ = 1.6


# ===========================================================================
# Split / leakage validation
# ===========================================================================
def build_subject_map(dataset_root):
    """
    Walk datasets/DaiSEE/DataSet/{Train,Validation,Test}/<SubjectID>/<ClipID>/
    once to build an authoritative clip_id -> subject_id mapping.

    Deliberately NOT a fixed-prefix-length heuristic: clip_id length varies
    (8-11 characters observed), because DaiSEE's own naming appends a
    variable-length per-clip suffix to a subject id, e.g. subject "500044"
    + clip "5000441001" (4-digit suffix) vs subject "907001" + clip
    "90700160" (2-digit suffix) - the subject prefix itself happens to be a
    consistent 6 digits in every sample checked, but this walks the real
    folder structure instead of trusting that pattern to hold everywhere.
    """
    mapping = {}
    for split_dir in ('Train', 'Validation', 'Test'):
        root = Path(dataset_root) / split_dir
        if not root.is_dir():
            continue
        for subject_dir in root.iterdir():
            if not subject_dir.is_dir():
                continue
            for clip_dir in subject_dir.iterdir():
                if clip_dir.is_dir():
                    mapping[clip_dir.name] = subject_dir.name
    return mapping


def validate_splits(index, dataset_root):
    """
    Asserts clip-level AND subject-level disjointness across train/val/test.
    Fails loudly (raises RuntimeError) on any leakage found - never silently
    drops/reassigns clips to "fix" it. DaiSEE's official split is designed
    to be subject-disjoint, so this is expected to pass; the point is to
    verify that, not assume it.
    """
    train_ids = {c[0] for c in index['train']}
    val_ids = {c[0] for c in index['val']}
    test_ids = {c[0] for c in index['test']}

    clip_overlaps = []
    if not train_ids.isdisjoint(val_ids):
        clip_overlaps.append(f"train/val clip overlap: {len(train_ids & val_ids)} clips")
    if not train_ids.isdisjoint(test_ids):
        clip_overlaps.append(f"train/test clip overlap: {len(train_ids & test_ids)} clips")
    if not val_ids.isdisjoint(test_ids):
        clip_overlaps.append(f"val/test clip overlap: {len(val_ids & test_ids)} clips")
    if clip_overlaps:
        raise RuntimeError("SPLIT VALIDATION FAILED (clip-level):\n  " + "\n  ".join(clip_overlaps))

    subject_map = build_subject_map(dataset_root)
    all_ids = train_ids | val_ids | test_ids
    missing = [cid for cid in all_ids if cid not in subject_map]
    if missing:
        print(f"  WARNING: {len(missing)}/{len(all_ids)} clip IDs not found under "
              f"{dataset_root} - subject-level leakage cannot be checked for these "
              f"(clip-level check above still covers them). Example IDs: {missing[:5]}")

    def subjects_of(ids):
        return {subject_map[cid] for cid in ids if cid in subject_map}

    train_subj, val_subj, test_subj = subjects_of(train_ids), subjects_of(val_ids), subjects_of(test_ids)
    subj_overlaps = []
    if not train_subj.isdisjoint(val_subj):
        subj_overlaps.append(f"train/val SUBJECT overlap: {sorted(train_subj & val_subj)}")
    if not train_subj.isdisjoint(test_subj):
        subj_overlaps.append(f"train/test SUBJECT overlap: {sorted(train_subj & test_subj)}")
    if not val_subj.isdisjoint(test_subj):
        subj_overlaps.append(f"val/test SUBJECT overlap: {sorted(val_subj & test_subj)}")
    if subj_overlaps:
        raise RuntimeError("SPLIT VALIDATION FAILED (subject-level):\n  " + "\n  ".join(subj_overlaps))

    print(f"  split validation OK: {len(train_ids)} train / {len(val_ids)} val / "
          f"{len(test_ids)} test clips - no clip-level or subject-level leakage "
          f"({len(train_subj)}/{len(val_subj)}/{len(test_subj)} distinct subjects)")


# ===========================================================================
# Feature extraction
# ===========================================================================
class FallbackTracker:
    """
    Per-split diagnostics for the carry-forward fallback (unreadable image,
    or BehavioralFeatureEngine.extract() found no face -> reuse the
    previous frame's vector, zeros if there is no previous frame yet). The
    fallback STRATEGY does not change based on these numbers - this is
    reporting only, so a reader of the final report can judge how much of
    the training data is degraded rather than discovering it silently.
    """
    def __init__(self):
        self.total_frames = 0
        self.none_count = 0
        self.unreadable_count = 0
        self.fallback_flags = []   # one bool per clip, in the order processed

    @property
    def total_fallback(self):
        return self.none_count + self.unreadable_count

    @property
    def fallback_pct(self):
        return 100.0 * self.total_fallback / self.total_frames if self.total_frames else 0.0

    @property
    def clips_with_fallback(self):
        return int(sum(self.fallback_flags))

    def fallback_clip_indices(self):
        return [i for i, f in enumerate(self.fallback_flags) if f]

    def summary_line(self, label):
        return (f"  [{label}] frames={self.total_frames}  none={self.none_count}  "
               f"unreadable={self.unreadable_count}  fallback={self.total_fallback} "
               f"({self.fallback_pct:.2f}%)  clips_affected={self.clips_with_fallback}/"
               f"{len(self.fallback_flags)}")


def extract_clip_features(frame_paths, tracker=None):
    """
    Returns (vec_seq, had_fallback). vec_seq is a (16, 10) float64 array,
    or None if the clip produced zero usable frames (no readable image and
    no successful extraction on any of its 16 frames - expected to be
    vanishingly rare).

    A FRESH BehavioralFeatureEngine per clip is deliberate: clips are
    independent short recordings, unlike a continuous live session, so
    blink-history/EAR-history state must not leak from one clip into the
    next. No calibration call - this is the documented, tested behavior of
    an uncalibrated engine (scores against BehavioralBaseline()'s generic
    defaults, not garbage; see tests/test_behavioral_features.py's
    _calibrated() helper for the same pattern), and DaiSEE has no
    calibration segment to use anyway.
    """
    clock = {'t': 0.0}
    engine = BehavioralFeatureEngine(time_source=lambda: clock['t'])
    vectors = []
    had_fallback = False
    try:
        for i, path in enumerate(frame_paths):
            clock['t'] = i / FRAME_RATE_HZ
            if tracker is not None:
                tracker.total_frames += 1

            img = cv2.imread(str(path))
            feats = None
            if img is None:
                if tracker is not None:
                    tracker.unreadable_count += 1
                had_fallback = True
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                feats = engine.extract(rgb)
                if feats is None:
                    if tracker is not None:
                        tracker.none_count += 1
                    had_fallback = True

            if feats is not None:
                vec = np.array(
                    [float(feats.get(name, 0.0)) for name in TEMPORAL_FEATURE_ORDER],
                    dtype=np.float64,
                )
            else:
                vec = vectors[-1].copy() if vectors else np.zeros(len(TEMPORAL_FEATURE_ORDER))
            vectors.append(vec)
    finally:
        engine.close()

    if not vectors:
        return None, had_fallback
    return np.stack(vectors), had_fallback


def build_feature_cache(index, split, cache_dir, force=False):
    """
    Returns (X, y, clip_ids, tracker). Caches to
    temporal_features_{split}.npz, including the fallback diagnostics, so a
    cache hit still carries the same diagnostic information a fresh build
    would - not just the arrays.
    """
    cache_path = Path(cache_dir) / f"temporal_features_{split}.npz"
    tracker = FallbackTracker()

    if cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=True)
        X, y, ids = data['X'], data['y'], list(data['clip_ids'])
        tracker.fallback_flags = list(bool(v) for v in data['fallback_flags'])
        tracker.total_frames = int(data['total_frames'])
        tracker.none_count = int(data['none_count'])
        tracker.unreadable_count = int(data['unreadable_count'])
        print(f"  [{split}] using cached features: {cache_path} ({X.shape[0]} clips)")
        return X, y, ids, tracker

    Xs, ys, ids = [], [], []
    clips = index[split]
    t0 = time.perf_counter()
    for i, (clip_id, label, frame_paths) in enumerate(clips):
        vec_seq, had_fallback = extract_clip_features(frame_paths, tracker)
        if vec_seq is None:
            print(f"  WARNING: clip {clip_id} produced zero usable frames - skipping")
            continue
        Xs.append(vec_seq)
        ys.append(label)
        ids.append(clip_id)
        tracker.fallback_flags.append(had_fallback)
        if (i + 1) % 500 == 0:
            print(f"  [{split}] {i + 1}/{len(clips)} clips ({time.perf_counter() - t0:.0f}s elapsed)")

    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    print(tracker.summary_line(split))

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path, X=X, y=y, clip_ids=np.array(ids),
        fallback_flags=np.array(tracker.fallback_flags),
        total_frames=tracker.total_frames, none_count=tracker.none_count,
        unreadable_count=tracker.unreadable_count,
    )
    print(f"  [{split}] cached {X.shape[0]} clips to {cache_path}")
    return X, y, ids, tracker


def compute_summary_stats(X):
    """
    (N, 16, 10) -> (N, 30). For each of the 10 feature dims: mean across
    the 16 frames, std across the 16 frames, and the slope of a
    least-squares linear fit against timestep t=0..15
    (np.polyfit(t, values, 1)[0]). Feeds the GBT baseline ablation - reuses
    the SAME cached (16,10) sequences the GRU trains on, no separate
    extraction pipeline.
    """
    N, T, D = X.shape
    t = np.arange(T)
    out = np.empty((N, D * 3), dtype=np.float64)
    for d in range(D):
        vals = X[:, :, d]
        mean = vals.mean(axis=1)
        std = vals.std(axis=1)
        slope = np.array([np.polyfit(t, vals[i], 1)[0] for i in range(N)])
        out[:, 3 * d:3 * d + 3] = np.stack([mean, std, slope], axis=1)
    return out


# ===========================================================================
# Model
# ===========================================================================
class GRUStressModel(nn.Module):
    def __init__(self, input_dim=10, hidden_size=32, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, num_layers=1, batch_first=True)
        # nn.GRU's own dropout= kwarg only applies BETWEEN stacked recurrent
        # layers - with num_layers=1 it is silently a no-op. Regularization
        # comes from this explicit Dropout instead.
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, 16, input_dim) -> logits (B,), NOT probabilities. Sigmoid
        # is applied by the caller (BCEWithLogitsLoss during training,
        # torch.sigmoid at eval time) - gru_forward_numpy mirrors this same
        # raw-logit contract so export-time parity can compare logits
        # directly rather than only the post-sigmoid probability, which
        # would mask small numeric drift.
        _, h_n = self.gru(x)
        return self.fc(self.dropout(h_n.squeeze(0))).squeeze(-1)


def _normalize(X, feature_mean, feature_std):
    safe_std = np.where(feature_std > 1e-8, feature_std, 1.0)
    return (X - feature_mean) / safe_std


def _weights_dict(model, hidden_size):
    """The exported/verified weight dict, pulled from the live PyTorch
    model - used identically by verify_export() and the final .npz write,
    so there is exactly one place this mapping is built."""
    sd = model.state_dict()
    return {
        'W_ih': sd['gru.weight_ih_l0'].detach().cpu().numpy().astype(np.float64),
        'W_hh': sd['gru.weight_hh_l0'].detach().cpu().numpy().astype(np.float64),
        'b_ih': sd['gru.bias_ih_l0'].detach().cpu().numpy().astype(np.float64),
        'b_hh': sd['gru.bias_hh_l0'].detach().cpu().numpy().astype(np.float64),
        'W_out': sd['fc.weight'].detach().cpu().numpy().astype(np.float64),
        'b_out': sd['fc.bias'].detach().cpu().numpy().astype(np.float64),
        'hidden_size': np.array(hidden_size),
    }


def train_model(Xtr, ytr, Xval, yval, feature_mean, feature_std, epochs, hidden_size,
                lr, patience, batch_size, device, log_path, seed):
    torch.manual_seed(seed)

    Xtr_n = torch.tensor(_normalize(Xtr, feature_mean, feature_std), dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xval_n = torch.tensor(_normalize(Xval, feature_mean, feature_std), dtype=torch.float32).to(device)
    yval_np = yval

    train_loader = DataLoader(TensorDataset(Xtr_n, ytr_t), batch_size=batch_size, shuffle=True)

    model = GRUStressModel(input_dim=Xtr.shape[2], hidden_size=hidden_size).to(device)
    n_pos = max(int(ytr.sum()), 1)
    n_neg = len(ytr) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -1.0
    best_state = None
    stale_epochs = 0
    log_rows = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xval_n)
            val_pred = (torch.sigmoid(val_logits).cpu().numpy() >= 0.5).astype(int)
        val_bal_acc = balanced_accuracy_score(yval_np, val_pred)

        log_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_balanced_acc': val_bal_acc})
        print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_balanced_acc={val_bal_acc:.4f}")

        # Model selection on validation BALANCED accuracy, never raw
        # accuracy - the old LSTM's 71% headline hid a 0.285 stressed
        # recall; selecting on raw accuracy would risk the same collapse.
        if val_bal_acc > best_score:
            best_score = val_bal_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"  early stopping at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(best_state)

    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    return model, best_score


def evaluate(y_true, y_pred_prob):
    """Full honest metric set, both models go through this identically."""
    y_pred = (y_pred_prob >= 0.5).astype(int)
    report = classification_report(
        y_true, y_pred, target_names=['not_stressed', 'stressed'],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    majority_floor = max(np.bincount(y_true, minlength=2)) / len(y_true)
    acc = report['accuracy']
    return {
        'n': len(y_true),
        'accuracy': acc,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': report['stressed']['precision'],
        'recall': report['stressed']['recall'],
        'f1': report['stressed']['f1-score'],
        'confusion_matrix': cm,
        'majority_floor': majority_floor,
        'above_floor': acc > majority_floor,
    }


def run_baseline_ablation(Xtr_stats, ytr, Xte_stats, yte):
    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(Xtr_stats, ytr)
    probs = clf.predict_proba(Xte_stats)[:, 1]
    return evaluate(yte, probs)


# ===========================================================================
# Export + parity verification
# ===========================================================================
def verify_export(model, weights, verify_X_norm, device):
    """
    Compares the real trained PyTorch model against gru_forward_numpy on
    EVERY sequence in verify_X_norm (already-normalized (K,16,10)),
    comparing RAW LOGITS (not sigmoid outputs). Returns (ok, verify_Y,
    detail_lines) - verify_Y (the torch logits) is what gets baked into the
    exported artifact for the live loader to re-check later. Writes
    nothing to disk itself.

    Runs the comparison in float64 on BOTH sides. gru_forward_numpy always
    computes in float64; comparing against the trained model's normal
    float32 forward pass produced small (~1e-5 to 1e-4) drift over the
    16-step recurrence that is NOT a gate-order/layout bug (an isolated
    cross-check against a fresh random-weight model matched to 1e-8 in
    float32-vs-float64 too - it only surfaced here because a *trained*
    model's weight magnitudes/hidden states compound float32 rounding
    differently). A dedicated float64 copy of the model is used for this
    check only - the original stays float32 for training/saving, since
    that's the normal, faster precision for GPU training and the shipped
    .pt checkpoint.
    """
    import copy
    model64 = copy.deepcopy(model).double().eval()

    verify_Y = np.empty(verify_X_norm.shape[0], dtype=np.float64)
    detail = []
    ok = True
    for i in range(verify_X_norm.shape[0]):
        x = verify_X_norm[i]
        with torch.no_grad():
            xt = torch.tensor(x, dtype=torch.float64).unsqueeze(0).to(device)
            torch_logit = float(model64(xt).item())
        numpy_logit = gru_forward_numpy(x.astype(np.float64), weights)
        verify_Y[i] = torch_logit
        diff = abs(torch_logit - numpy_logit)
        match = np.allclose(torch_logit, numpy_logit, atol=1e-5)
        ok = ok and match
        status = "OK" if match else "MISMATCH"
        detail.append(f"    seq {i:2d}: {status}  torch={torch_logit:+.6f}  "
                      f"numpy={numpy_logit:+.6f}  diff={diff:.2e}")
    return ok, verify_Y, detail


def export_to_numpy(model, feature_mean, feature_std, verify_X_norm, verify_Y, hidden_size):
    """Builds the export dict in memory only - the caller writes it to disk,
    and only after verify_export has confirmed every comparison passed."""
    weights = _weights_dict(model, hidden_size)
    weights['feature_mean'] = feature_mean.astype(np.float64)
    weights['feature_std'] = feature_std.astype(np.float64)
    weights['verify_X'] = verify_X_norm.astype(np.float64)
    weights['verify_Y'] = verify_Y.astype(np.float64)
    return weights


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--index', default='../datasets/processed_daisee/lstm_index.pkl')
    ap.add_argument('--dataset-root', default='../datasets/DaiSEE/DataSet')
    ap.add_argument('--feature-cache-dir', default='../datasets/processed_daisee')
    ap.add_argument('--rebuild-features', action='store_true')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--hidden-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--n-verify', type=int, default=20)
    ap.add_argument('--smoke-test', action='store_true',
                    help='small clip subset + 2 epochs, for pipeline sanity checking')
    ap.add_argument('--model-out', default='../models/temporal_gru.pt')
    ap.add_argument('--export-out', default='../models/temporal_gru.npz')
    ap.add_argument('--report-out', default='../models/temporal_gru_test_report.txt')
    ap.add_argument('--log-out', default='../models/temporal_gru_training_log.csv')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print("=" * 74)
    print("PHASE 2.1 TEMPORAL MODEL TRAINING")
    print("=" * 74)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    print("\nloading index + validating splits...")
    with open(args.index, 'rb') as f:
        index = pickle.load(f)
    validate_splits(index, args.dataset_root)

    if args.smoke_test:
        import random
        random.seed(args.seed)
        for split in ('train', 'val', 'test'):
            index[split] = random.sample(index[split], min(40, len(index[split])))
        args.epochs = 2
        print("SMOKE TEST: 40 clips/split, 2 epochs")

    print("\nbuilding/loading feature caches...")
    Xtr, ytr, ids_tr, tr_tracker = build_feature_cache(
        index, 'train', args.feature_cache_dir, force=args.rebuild_features or args.smoke_test)
    Xval, yval, ids_val, val_tracker = build_feature_cache(
        index, 'val', args.feature_cache_dir, force=args.rebuild_features or args.smoke_test)
    Xte, yte, ids_te, te_tracker = build_feature_cache(
        index, 'test', args.feature_cache_dir, force=args.rebuild_features or args.smoke_test)

    feature_mean = Xtr.reshape(-1, Xtr.shape[2]).mean(axis=0).astype(np.float64)
    feature_std = Xtr.reshape(-1, Xtr.shape[2]).std(axis=0).astype(np.float64)

    print(f"\ntrain={Xtr.shape[0]} val={Xval.shape[0]} test={Xte.shape[0]} clips")

    print("\n" + "-" * 74)
    print("TRAINING GRU")
    print("-" * 74)
    model, best_val_score = train_model(
        Xtr, ytr, Xval, yval, feature_mean, feature_std,
        epochs=args.epochs, hidden_size=args.hidden_size, lr=args.lr,
        patience=args.patience, batch_size=args.batch_size, device=device,
        log_path=args.log_out, seed=args.seed,
    )

    model.eval()
    with torch.no_grad():
        Xte_n = torch.tensor(_normalize(Xte, feature_mean, feature_std), dtype=torch.float32).to(device)
        gru_probs = torch.sigmoid(model(Xte_n)).cpu().numpy()
    gru_result = evaluate(yte, gru_probs)

    print("\n" + "-" * 74)
    print("GBT BASELINE (mean/std/slope summary stats, same cached sequences)")
    print("-" * 74)
    Xtr_stats = compute_summary_stats(Xtr)
    Xte_stats = compute_summary_stats(Xte)
    gbt_result = run_baseline_ablation(Xtr_stats, ytr, Xte_stats, yte)

    print("\n" + "-" * 74)
    print("EXPORT + PARITY VERIFICATION")
    print("-" * 74)

    n_verify = min(args.n_verify, Xval.shape[0])
    verify_idx = list(np.linspace(0, Xval.shape[0] - 1, n_verify, dtype=int))
    fallback_idx = val_tracker.fallback_clip_indices()
    if fallback_idx and fallback_idx[0] not in verify_idx:
        verify_idx[-1] = fallback_idx[0]
        print(f"  including 1 fallback-containing val clip in the verify set (val idx {fallback_idx[0]})")
    else:
        print(f"  no fallback-containing val clips found to include" if not fallback_idx
              else "  a fallback-containing val clip was already in the sampled verify set")
    verify_idx = sorted(set(verify_idx))

    verify_X_norm = _normalize(Xval[verify_idx], feature_mean, feature_std).astype(np.float64)
    weights_for_check = _weights_dict(model, args.hidden_size)
    ok, verify_Y, detail = verify_export(model, weights_for_check, verify_X_norm, device)

    for line in detail:
        print(line)

    if not ok:
        print("\nEXPORT FAILED: the numpy replica does not match the trained PyTorch")
        print("model on at least one verify sequence. No .npz written. This means the")
        print("GRU weight layout or gate order in temporal_model.py is wrong somewhere -")
        print("do not proceed until gru_forward_numpy is fixed and this check passes.")
        return 1

    print(f"\n  PASS: all {len(verify_idx)} verify sequences match at atol=1e-5")

    export_weights = export_to_numpy(model, feature_mean, feature_std, verify_X_norm, verify_Y, args.hidden_size)

    # Save BEFORE printing the final diagnostic report.
    Path(args.export_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.export_out, **export_weights)
    torch.save(model.state_dict(), args.model_out)
    print(f"  saved: {args.export_out}")
    print(f"  saved: {args.model_out}")

    # ---- report ----
    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 74)
    out("PHASE 2.1 TEMPORAL MODEL - TEST REPORT")
    out("=" * 74)
    out(f"generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"clips     : train={Xtr.shape[0]} val={Xval.shape[0]} test={Xte.shape[0]}")
    out(f"best val balanced accuracy during training: {best_val_score:.4f}")
    out("")
    out("KNOWN LIMITATION: datasets/processed_daisee/ crops are face-only")
    out("(FaceMesh bbox + 15% padding). neck_strain and slouch are structurally")
    out("~0 for nearly every frame in this training set, since posture scoring")
    out("requires both-shoulder visibility >= 0.7 that a face-only crop")
    out("essentially never has. Not redesigned around - re-decoding the 8231")
    out("original DaiSEE videos for 2 of 10 dims was judged not worth it.")
    out("")

    def report_model(name, result):
        out("-" * 74)
        out(name)
        out("-" * 74)
        out(f"  n                    : {result['n']}")
        out(f"  accuracy             : {result['accuracy'] * 100:.2f}%")
        out(f"  balanced accuracy    : {result['balanced_accuracy'] * 100:.2f}%")
        out(f"  precision (stressed) : {result['precision']:.3f}")
        out(f"  recall (stressed)    : {result['recall']:.3f}")
        out(f"  f1 (stressed)        : {result['f1']:.3f}")
        out(f"  majority-class floor : {result['majority_floor'] * 100:.2f}%")
        out(f"  verdict              : {'ABOVE' if result['above_floor'] else 'BELOW'} the majority-class floor")
        cm = result['confusion_matrix']
        out("  confusion matrix (rows=true, cols=pred [not_stressed, stressed]):")
        out(f"    {cm[0].tolist()}")
        out(f"    {cm[1].tolist()}")
        out("")

    report_model("GRU (feature-sequence temporal model)", gru_result)
    report_model("GBT baseline (mean/std/slope summary stats)", gbt_result)

    out("-" * 74)
    out("FALLBACK DIAGNOSTICS")
    out("-" * 74)
    out("Frames where extraction fell back to the previous frame's vector -")
    out("no face found, or an unreadable image. Fallback strategy itself is")
    out("unchanged by these numbers; this is reporting only.")
    out(tr_tracker.summary_line('train'))
    out(val_tracker.summary_line('val'))
    out(te_tracker.summary_line('test'))
    out("")

    out("=" * 74)
    out("COMPARISON TO THE OLD PIXEL-LSTM (models/lstm_test_report.txt)")
    out("=" * 74)
    out("  old pixel-LSTM : 71.13% accuracy, BELOW its 86.6% majority floor")
    out(f"  new GRU        : {gru_result['accuracy'] * 100:.2f}% accuracy, "
       f"{'ABOVE' if gru_result['above_floor'] else 'BELOW'} its "
       f"{gru_result['majority_floor'] * 100:.2f}% floor")
    out(f"  GBT baseline   : {gbt_result['accuracy'] * 100:.2f}% accuracy, "
       f"{'ABOVE' if gbt_result['above_floor'] else 'BELOW'} its "
       f"{gbt_result['majority_floor'] * 100:.2f}% floor")
    out("")
    out("This result is reported as-is, whichever way it went - no tuning was")
    out("performed specifically to cross the majority-class floor.")

    Path(args.report_out).write_text("\n".join(lines) + "\n", encoding='ascii', errors='replace')
    print(f"\n  written: {args.report_out}")

    print("\nDone. W_TEMPORAL stays 0.0 in config.py until a live smoke test")
    print("confirms this is safe to enable - decision is manual, not automatic.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
