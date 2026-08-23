"""
DaiSEE CNN ensemble - shared model loading + inference
=========================================================
Used by both app.py (live) and evaluate_video_file.py (offline UBFC-Phys
processing), so the live and offline paths are provably running the exact
same inference code - the same principle rppg_live.py already follows for
its POS/CHROM methods (see that module's own docstring).

Why tf.function instead of Model.predict()
-------------------------------------------
Benchmarked on this machine (CPU-only TensorFlow - the RTX 5070 Ti is
confirmed working via .venv-train's PyTorch, just unused by this venv;
see PLAN_FOR_PP2.md's Phase 2.5 investigation): Model.predict() carries
heavy fixed per-call overhead built for batched inference, not one-off
live frames. A tf.function-wrapped direct call on the same model cut
MobileNet from 124ms to 29ms (4.3x) and EfficientNet from 153ms to 54ms
(2.8x) per call - roughly doubling the live capture loop's effective
frame rate with no threading or session-state changes.

Caching is each caller's own responsibility (build_pairs() always does
the real load+wrap work when called) - app.py already has
@st.cache_resource for this, and evaluate_video_file.py already has its
own module-level cache; duplicating either mechanism here would fight one
of them.
"""

from pathlib import Path

import cv2
import numpy as np

import config


def _make_infer_fn(model):
    """Wrap one already-loaded model in a tf.function, traced once on
    first call and reused after. None in, None out - keeps the "missing
    model" degrade path (fusion redistributes weight automatically)
    unchanged for callers."""
    if model is None:
        return None
    import tensorflow as tf

    @tf.function(reduce_retracing=True)
    def infer(x):
        return model(x, training=False)
    return infer


def _try_load(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(str(path))
    except Exception as e:
        print(f"  [daisee_ensemble] failed to load {path.name}: {e}")
        return None


def build_pairs(skip=False):
    """
    Returns [(infer_fn, input_size, preprocess_fn), ...] for MobileNet and
    EfficientNet, in that fixed order. infer_fn is None wherever that
    model is missing/failed to load or skip=True - ensemble_predict()
    below treats that exactly like a model that was never there.
    """
    if skip:
        return []
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_pre
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre

    mb = _try_load(config.DAISEE_MOBILENET_PATH)
    eff = _try_load(config.DAISEE_EFFICIENTNET_PATH)
    return [
        (_make_infer_fn(mb), config.MOBILENET_INPUT_SIZE, mb_pre),
        (_make_infer_fn(eff), config.EFFICIENTNET_INPUT_SIZE, eff_pre),
    ]


def prep(face_bgr, size, preprocess_fn):
    img = cv2.resize(face_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    img = preprocess_fn(img)
    return np.expand_dims(img, axis=0)


def ensemble_predict(face_bgr, pairs):
    """Mean of both models' softmax outputs - the live system's and the
    offline pipeline's actual production formula. None if every model in
    `pairs` is unavailable."""
    preds = []
    for infer_fn, size, preproc in pairs:
        if infer_fn is None:
            continue
        x = prep(face_bgr, size, preproc)
        preds.append(infer_fn(x).numpy()[0])
    return np.mean(preds, axis=0) if preds else None
