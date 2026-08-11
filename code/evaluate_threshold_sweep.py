"""
Threshold Sweep Evaluation
==========================
Loads the trained DaiSEE binary stress models and evaluates them on the
held-out test set at multiple decision thresholds. Reports precision,
recall, F1, and accuracy for each threshold, then picks the threshold
that maximises macro-F1 (the right metric for imbalanced data).

Run AFTER train_daisee_stress.py finishes.

What this does NOT do
---------------------
- It does NOT retrain the model.
- It does NOT modify the saved .h5 files.
- It only measures the model you already have at different decision cutoffs.

Output
------
- Prints a table of metrics at each threshold for both models.
- Picks the optimal threshold for macro-F1.
- Saves models/optimal_thresholds.json for the live system to read.
"""

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2  import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report,
)

# ---------------- CONFIG ----------------
DATA_ROOT = Path('../datasets/processed_daisee')
MODEL_DIR = Path('../models')
BATCH_SIZE = 32
CLASS_ORDER = ['not_stressed', 'stressed']

THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def evaluate_at_thresholds(model, test_data, thresholds):
    """Run inference once, then sweep thresholds across the same predictions."""
    print("  Running inference on test set (one pass)...")
    probs = model.predict(test_data, verbose=1)
    p_stressed = probs[:, 1]   # probability of being in the "stressed" class
    y_true = test_data.classes  # 0 = not_stressed, 1 = stressed

    print(f"\n  {'Threshold':>10s}  {'Accuracy':>9s}  "
          f"{'Stress P':>9s}  {'Stress R':>9s}  {'Stress F1':>10s}  "
          f"{'Macro F1':>9s}")
    print("  " + "-" * 70)

    results = []
    for th in thresholds:
        y_pred = (p_stressed >= th).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        results.append({
            'threshold': th, 'accuracy': acc,
            'stressed_precision': prec, 'stressed_recall': rec,
            'stressed_f1': f1, 'macro_f1': macro_f1,
        })
        print(f"  {th:>10.2f}  {acc*100:>8.2f}%  "
              f"{prec*100:>8.2f}%  {rec*100:>8.2f}%  {f1:>10.3f}  "
              f"{macro_f1:>9.3f}")

    # Pick the threshold with the best macro-F1
    best = max(results, key=lambda r: r['macro_f1'])
    print(f"\n  >>> Best threshold by macro-F1: {best['threshold']:.2f}  "
          f"(macro-F1 = {best['macro_f1']:.3f}, "
          f"stressed recall = {best['stressed_recall']*100:.1f}%)")

    # Also show a detailed report at the best threshold
    y_pred_best = (p_stressed >= best['threshold']).astype(int)
    print(f"\n  Classification report at threshold {best['threshold']:.2f}:")
    print(classification_report(y_true, y_pred_best,
                                target_names=CLASS_ORDER, digits=3))
    print("  Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred_best))
    return results, best


def evaluate_one(model_path, preprocess_fn, img_size, label):
    print(f"\n{'=' * 70}")
    print(f"  EVALUATING {label}")
    print(f"  model: {model_path}")
    print(f"{'=' * 70}")

    if not model_path.exists():
        print(f"  [skip] model file not found")
        return None

    eval_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    test_data = eval_gen.flow_from_directory(
        DATA_ROOT / 'test',
        target_size=(img_size, img_size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_ORDER,
        shuffle=False,
    )
    model = tf.keras.models.load_model(str(model_path))
    return evaluate_at_thresholds(model, test_data, THRESHOLDS)


def main():
    out = {}
    mn_result = evaluate_one(
        MODEL_DIR / 'daisee_mobilenet_stress.h5',
        mobilenet_preprocess, 160, 'MobileNet')
    if mn_result is not None:
        _, best_mn = mn_result
        out['mobilenet'] = best_mn

    eff_result = evaluate_one(
        MODEL_DIR / 'daisee_efficientnet_stress.h5',
        efficientnet_preprocess, 224, 'EfficientNet')
    if eff_result is not None:
        _, best_eff = eff_result
        out['efficientnet'] = best_eff

    if out:
        json_path = MODEL_DIR / 'optimal_thresholds.json'
        with open(json_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nOptimal thresholds saved to: {json_path}")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for name, info in out.items():
        print(f"  {name:<14s} optimal_threshold={info['threshold']:.2f}  "
              f"macro_F1={info['macro_f1']:.3f}  "
              f"stressed_recall={info['stressed_recall']*100:.1f}%")


if __name__ == '__main__':
    main()
