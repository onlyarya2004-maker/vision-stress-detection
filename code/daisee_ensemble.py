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

import threading
import time
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


# --- one shared, pre-warmed copy per process ---------------------------------
# Cold start used to cost the first visitor 60+ s of blank page: importing
# TensorFlow (~15 s), loading two H5 models, and tracing each tf.function on its
# first call. The weights are stateless, so ONE copy serves every session; it is
# loaded once, in the background, as soon as any session reaches the app (the
# login page does not need it), and each model is run once on a dummy face so the
# trace happens before a real user's first frame instead of during it.
_SHARED = None
_SHARED_LOCK = threading.Lock()
_BG_THREAD = None


def shared_pairs():
    """The process-wide model list (blocks while it is still loading)."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            pairs = build_pairs()
            try:
                dummy = np.full((224, 224, 3), 128, dtype=np.uint8)
                ensemble_predict(dummy, pairs)          # triggers the tf.function trace
            except Exception as e:                      # a warm-up failure must not break loading
                print(f"  [daisee_ensemble] warm-up skipped: {e}")
            _SHARED = pairs
        return _SHARED


def start_background_load():
    """Begin loading the shared models without blocking the caller. Safe and
    cheap to call on every script run: only the first call starts a thread."""
    global _BG_THREAD
    with _SHARED_LOCK:
        if _SHARED is not None or _BG_THREAD is not None:
            return
        _BG_THREAD = threading.Thread(target=shared_pairs, name='model-preload', daemon=True)
        _BG_THREAD.start()


def prep(face_bgr, size, preprocess_fn):
    img = cv2.resize(face_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    img = preprocess_fn(img)
    return np.expand_dims(img, axis=0)


# The loaded models are one shared process-wide copy (app.py caches them with
# @st.cache_resource) while each hosted user session runs on its own thread.
# Serialising the CNN call keeps two simultaneous users from oversubscribing
# the CPU's TensorFlow threads against each other (which makes BOTH slower than
# taking turns), and sidesteps any thread-safety doubt about a shared
# tf.function. The call is ~90 ms and runs on every 3rd frame, so even two
# users contend for only ~60% of one core's time - the lock costs nothing.
_PREDICT_LOCK = threading.Lock()
# Timing of the most recent call (ms spent waiting for the lock / running the
# models) - read by session_pipeline's STRESS_PROFILE stage timers.
LAST_CALL_MS = {'wait': 0.0, 'infer': 0.0, 'detail': []}


def ensemble_predict(face_bgr, pairs):
    """Mean of both models' softmax outputs - the live system's and the
    offline pipeline's actual production formula. None if every model in
    `pairs` is unavailable."""
    preds = []
    t0 = time.perf_counter()
    with _PREDICT_LOCK:
        t1 = time.perf_counter()
        parts = []
        for infer_fn, size, preproc in pairs:
            if infer_fn is None:
                continue
            ta = time.perf_counter()
            x = prep(face_bgr, size, preproc)
            tb = time.perf_counter()
            y = infer_fn(x)
            tc = time.perf_counter()
            preds.append(y.numpy()[0])
            td = time.perf_counter()
            parts.append((size, (tb - ta) * 1000, (tc - tb) * 1000, (td - tc) * 1000))
        t2 = time.perf_counter()
    LAST_CALL_MS['wait'], LAST_CALL_MS['infer'] = (t1 - t0) * 1000.0, (t2 - t1) * 1000.0
    LAST_CALL_MS['detail'] = parts
    return np.mean(preds, axis=0) if preds else None
