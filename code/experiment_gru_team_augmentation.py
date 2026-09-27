"""
Hypothetical: does adding the team's own webcam recordings to GRU training help?
================================================================================
Question: if the GRU is trained on UBFC-Phys AND the team's recordings, does it
do better on a team subject it has NOT seen than a UBFC-only GRU does?

Design (leak-free, same recipe as train_temporal_model_ubfc.py):
  * Evaluation subjects = team subjects with BOTH classes among their
    v7-complete, self-report-labelled sessions (currently S5, S6).
  * For each such subject T, T's windows are held out entirely.
  * Condition A: train on UBFC only.
    Condition B(k): train on UBFC + every OTHER team subject's windows,
    with team windows replicated k times (team data is ~100x smaller than
    UBFC, so k=1 is swamped; k>1 is a hypothesis, not a proven setting).
  * Early-stopping validation subject = the same held-out UBFC subject in every
    condition of a given seed, so conditions differ ONLY in the team data.
  * Several seeds, because with ~40 test windows per subject one run is noise.

Writes nothing to models/. Report only. Run with .venv-train's python:
    .venv-train\\Scripts\\python experiment_gru_team_augmentation.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_model import normalize_features
from train_temporal_model import train_model, evaluate
from train_temporal_model_ubfc import build_dataset, build_team_dataset


def run_one(Xu, yu, gu, Xt, yt, st, test_subj, k, seed, device, args):
    rng = np.random.RandomState(seed)
    ubfc_subjects = sorted(set(gu))
    val_subject = ubfc_subjects[rng.randint(len(ubfc_subjects))]
    val = gu == val_subject
    Xtr, ytr = Xu[~val], yu[~val]
    Xval, yval = Xu[val], yu[val]

    if k > 0:
        other = st != test_subj
        if other.any():
            Xo, yo = Xt[other], yt[other]
            Xtr = np.concatenate([Xtr] + [Xo] * k)
            ytr = np.concatenate([ytr] + [yo] * k)

    mean, std = Xtr.mean(axis=(0, 1)), Xtr.std(axis=(0, 1))
    model, _ = train_model(
        Xtr, ytr.astype(np.float32), Xval, yval, mean, std,
        epochs=args.epochs, hidden_size=32, lr=1e-3, patience=10,
        batch_size=32, device=device, log_path=None, seed=seed,
    )
    model.eval()
    te = st == test_subj
    Xte = torch.tensor(normalize_features(Xt[te], mean, std), dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(Xte)).cpu().numpy()
    return yt[te], probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace-dir', default='../datasets/feature_traces')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--factors', type=int, nargs='+', default=[1, 20])
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    Xu, yu, gu = build_dataset(args.trace_dir)
    Xt, yt, st = build_team_dataset()
    if Xt is None:
        print("no usable team data"); return 1

    print("\nTEAM DATA AVAILABLE (v7-complete, self-report labelled, no '3' ratings)")
    for s in sorted(set(st)):
        m = st == s
        print(f"  {s:8s} windows={int(m.sum()):4d}  relaxed={int((yt[m]==0).sum()):4d}  stressed={int((yt[m]==1).sum()):4d}")
    print(f"  UBFC training pool: {len(yu)} windows, {len(set(gu))} subjects")

    eval_subjects = [s for s in sorted(set(st))
                     if len(set(yt[st == s].tolist())) == 2]
    print(f"\nHeld-out evaluation subjects (both classes present): {eval_subjects}")

    conditions = [('A: UBFC only', 0)] + [(f'B: UBFC + other team x{k}', k) for k in args.factors]
    results = {c: {'y': [], 'p': [], 'per': {}} for c, _ in conditions}
    for T in eval_subjects:
        for cname, k in conditions:
            ys, ps = [], []
            for seed in range(args.seeds):
                y, p = run_one(Xu, yu, gu, Xt, yt, st, T, k, seed, device, args)
                ys.append(y); ps.append(p)
                r = evaluate(y, p)
                results[cname]['per'].setdefault(T, []).append(r['balanced_accuracy'])
            results[cname]['y'].extend(np.concatenate(ys).tolist())
            results[cname]['p'].extend(np.concatenate(ps).tolist())
            print(f"  held-out {T}  {cname:28s} balanced acc per seed: "
                  + ", ".join(f"{b*100:.1f}%" for b in results[cname]['per'][T]))

    print("\n" + "=" * 78)
    print("SUMMARY (pooled over held-out team subjects and seeds)")
    print("=" * 78)
    print(f"  {'condition':32s}{'accuracy':>10}{'balanced':>10}{'recall':>8}{'floor':>8}")
    for cname, _ in conditions:
        r = evaluate(np.array(results[cname]['y']), np.array(results[cname]['p']))
        print(f"  {cname:32s}{r['accuracy']*100:>9.1f}%{r['balanced_accuracy']*100:>9.1f}%"
              f"{r['recall']:>8.2f}{r['majority_floor']*100:>7.1f}%")
    print("\nCAVEAT: test windows per subject are ~40 and overlap within a recording,")
    print("so differences of a few points are within noise. Report as hypothetical.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
