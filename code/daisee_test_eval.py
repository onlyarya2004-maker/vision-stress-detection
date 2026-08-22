"""
DaiSEE CNN ensemble - held-out test evaluation (Phase 4.1)
============================================================
Evaluates the two ALREADY-TRAINED DaiSEE stress classifiers (see
train_daisee_stress.py) on their held-out test split - loads the saved
.h5 files, never retrains.

Why this is NOT called from run_all_evaluations.py directly
-------------------------------------------------------------
Unlike the ablation table / EDA correlation / learned-fusion sections (cheap
numpy/scipy/sklearn over small cached CSVs, safe to recompute every run),
batched CNN inference over the full 28,531-image test split takes real time
even without training. Same two-tier pattern as the GRU/LSTM temporal
models: this script writes ../models/daisee_test_report.txt once, and
run_all_evaluations.py only reads and echoes that file - identical to how
it already handles models/lstm_test_report.txt and
models/temporal_gru_test_report.txt.

Reports three numbers, not one: MobileNetV2 alone, EfficientNetB0 alone,
and the ENSEMBLE (mean of both softmax outputs) - the ensemble is what the
live system and evaluate_video_file.py actually use
(evaluate_video_file.ensemble_predict()), so it's the number that belongs
in the README, not either backbone in isolation.

Usage
-----
    python daisee_test_eval.py
    python daisee_test_eval.py --data-root ../datasets/processed_daisee \
        --report-out ../models/daisee_test_report.txt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import config

CLASS_ORDER = ['not_stressed', 'stressed']   # index 1 == stressed, matches
                                              # train_daisee_stress.py exactly


def _load_model(path):
    import tensorflow as tf
    path = Path(path)
    if not path.exists():
        return None
    return tf.keras.models.load_model(str(path))


def predict_over_test_split(model, img_size, preprocess_fn, data_root, batch_size=32):
    """
    Batched inference over data_root/test/{not_stressed,stressed}/ via the
    SAME ImageDataGenerator/flow_from_directory pattern
    train_daisee_stress.py trains against, so this reproduces the real
    held-out split exactly rather than a re-derived approximation of it.
    shuffle=False makes file iteration order deterministic - required for
    aligning MobileNet's and EfficientNet's predictions row-for-row into
    the ensemble mean below.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_data = gen.flow_from_directory(
        Path(data_root) / 'test', target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='categorical',
        classes=CLASS_ORDER, shuffle=False,
    )
    probs = model.predict(test_data, verbose=0)
    return probs, np.array(test_data.classes)


def evaluate(y_true, probs, label):
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_recall_fscore_support, confusion_matrix,
    )
    preds = probs.argmax(axis=1)
    acc = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, labels=[1], zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    # Recomputed from the actual test-label counts every run, never a
    # reused/assumed number - same honesty rule the GRU/LSTM reports follow.
    majority_floor = max(np.bincount(y_true, minlength=2)) / len(y_true)
    return {
        'label': label, 'n': len(y_true), 'accuracy': acc,
        'balanced_accuracy': bal_acc, 'precision': float(precision[0]),
        'recall': float(recall[0]), 'f1': float(f1[0]),
        'confusion_matrix': cm, 'majority_floor': majority_floor,
        'above_floor': acc > majority_floor,
    }


def render_result(out, result):
    out("-" * 74)
    out(result['label'])
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default='../datasets/processed_daisee')
    ap.add_argument('--report-out', default='../models/daisee_test_report.txt')
    args = ap.parse_args()

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_pre
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

    t0 = time.perf_counter()
    print("loading models...")
    mb_model = _load_model(config.DAISEE_MOBILENET_PATH)
    eff_model = _load_model(config.DAISEE_EFFICIENTNET_PATH)
    if mb_model is None or eff_model is None:
        print("one or both .h5 models are missing - cannot evaluate. "
             "See README.md's Git LFS setup step.")
        return 1

    print("evaluating MobileNetV2 over the held-out test split...")
    mb_probs, y_true_mb = predict_over_test_split(
        mb_model, config.MOBILENET_INPUT_SIZE, mb_pre, args.data_root)
    print(f"  {len(y_true_mb)} images, {time.perf_counter() - t0:.0f}s elapsed")

    print("evaluating EfficientNetB0 over the held-out test split...")
    eff_probs, y_true_eff = predict_over_test_split(
        eff_model, config.EFFICIENTNET_INPUT_SIZE, eff_pre, args.data_root)
    print(f"  {len(y_true_eff)} images, {time.perf_counter() - t0:.0f}s elapsed total")

    # Both generators walk the same directory tree with shuffle=False, so
    # they must iterate files in identical order - a mismatch here would
    # mean the ensemble silently averages misaligned predictions.
    if not np.array_equal(y_true_mb, y_true_eff):
        print("ERROR: test split ordering mismatch between the two "
             "generators - refusing to compute an ensemble over "
             "misaligned predictions. No report written.")
        return 1
    y_true = y_true_mb

    ensemble_probs = np.mean([mb_probs, eff_probs], axis=0)

    mb_result = evaluate(y_true, mb_probs, "MobileNetV2 (alone)")
    eff_result = evaluate(y_true, eff_probs, "EfficientNetB0 (alone)")
    ens_result = evaluate(y_true, ensemble_probs,
                          "Ensemble (mean of both - what the live system uses)")

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 74)
    out("DAISEE CNN ENSEMBLE - HELD-OUT TEST REPORT")
    out("=" * 74)
    out(f"generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"data root : {Path(args.data_root).resolve()}")
    out(f"n images  : {len(y_true)} ({int(np.sum(y_true == 0))} not_stressed, "
       f"{int(np.sum(y_true == 1))} stressed)")
    out("")
    out("Never retrained here - loads the already-trained .h5 checkpoints and")
    out("runs inference only. Re-run train_daisee_stress.py to change the")
    out("underlying models; re-run this script afterward to refresh this report.")
    out("")

    render_result(out, mb_result)
    render_result(out, eff_result)
    render_result(out, ens_result)

    out("=" * 74)
    out(f"total time: {time.perf_counter() - t0:.0f}s")

    Path(args.report_out).write_text("\n".join(lines) + "\n", encoding='ascii', errors='replace')
    print(f"\nwritten: {args.report_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
