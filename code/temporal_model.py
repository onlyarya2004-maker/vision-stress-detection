"""
Temporal Model Inference Module (Phase 2.1)
============================================
Replaces the old pixel-based LSTM (see code/_archive/lstm_temporal.py,
code/train_lstm.py, models/lstm_test_report.txt for the retired negative
result: 71.13% test accuracy, BELOW the 86.6% majority-class floor - root
cause "frozen ImageNet features carry no stress signal").

This module wraps a small GRU trained on BEHAVIORAL-FEATURE sequences
(BehavioralFeatureEngine.extract() output) instead of raw pixels. Training
happens in PyTorch, in the separate .venv-train environment (see
train_temporal_model.py) - but this module, which the live Streamlit app
imports, is deliberately numpy-only. The live app's venv has TensorFlow,
not torch, and requirements-train.txt explicitly warns that mixing the two
stacks breaks the live system (numpy 1.26.4 for TF vs numpy 2.x for torch).
So the trained model is exported to a small .npz of raw weight arrays, and
gru_forward_numpy() below is a hand-written replica of PyTorch's nn.GRU
forward pass that runs on those arrays with zero new dependencies.

The single highest-risk detail in this file is getting the GRU weight
LAYOUT and GATE ORDER exactly right - PyTorch's nn.GRU packs weight_ih_l0/
weight_hh_l0 as three stacked (hidden, dim) blocks in the order
[reset, update, new]. Get this wrong and the model produces a plausible-
looking but silently-wrong number, not an exception. Two independent
guards exist against that: train_temporal_model.py's export step hard-
fails (refuses to write the .npz) unless gru_forward_numpy matches the
real trained PyTorch model on multiple real sequences; and every .npz
this module loads carries its own frozen verify_X/verify_Y pair (raw
logits) that __init__ re-checks on every load, not just once at export
time - a corrupted file, a stale path, or a future export bug can never
silently ship a wrong score.

Usage from app.py:
    from temporal_model import GRUTemporalInference

    temporal_inf = GRUTemporalInference()   # tries to load + self-verify, fails gracefully

    # In the main loop, AFTER behavioral_features = engine.extract(rgb) succeeds:
    if temporal_inf.is_available():
        temporal_inf.add_frame(behavioral_features)
        temporal_score = temporal_inf.predict()   # None until buffer fills, then [0,1]

If the model file is missing, fails to load, or fails self-verification,
is_available() returns False and the rest of the system runs unchanged.
"""

from collections import deque
from pathlib import Path

import numpy as np

import config

# The single source of truth for feature ordering - train_temporal_model.py
# imports this same constant so the training-time and live-inference-time
# vectors can never drift apart. First 7 entries deliberately match
# learned_fusion.FEATURE_ORDER (the same canonical, [0,1]-bounded signal
# vocabulary every other branch of this system uses); the trailing 3 are
# raw diagnostics the derived signals partially discard (absolute blink
# rate, eye asymmetry).
TEMPORAL_FEATURE_ORDER = [
    'eye_strain', 'blink_stress', 'brow_tension', 'jaw_tension',
    'head_instability', 'neck_strain', 'slouch',
    'ear', 'ear_asymmetry', 'blinks_per_minute',
]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gru_forward_numpy(x_seq, weights):
    """
    Hand-replica of PyTorch nn.GRU(input_dim, hidden, num_layers=1,
    batch_first=True) followed by Linear(hidden, 1) - i.e. exactly
    GRUStressModel.forward() in train_temporal_model.py, MINUS the final
    sigmoid (this returns the raw logit; see module docstring for why).

    x_seq:   (seq_len, input_dim) array, already normalized by the
             caller (feature_mean/feature_std baked into `weights`).
    weights: mapping with keys W_ih (3H,D), W_hh (3H,H), b_ih (3H,),
             b_hh (3H,), W_out (1,H), b_out (1,), hidden_size (scalar).
             Row blocks of W_ih/W_hh/b_ih/b_hh are ordered
             [reset, update, new] - PyTorch's own nn.GRU convention.

    Returns: float, the raw pre-sigmoid logit.
    """
    H = int(weights['hidden_size'])
    W_ih, W_hh = weights['W_ih'], weights['W_hh']
    b_ih, b_hh = weights['b_ih'], weights['b_hh']

    W_ir, W_iz, W_in = W_ih[0:H], W_ih[H:2 * H], W_ih[2 * H:3 * H]
    W_hr, W_hz, W_hn = W_hh[0:H], W_hh[H:2 * H], W_hh[2 * H:3 * H]
    b_ir, b_iz, b_in = b_ih[0:H], b_ih[H:2 * H], b_ih[2 * H:3 * H]
    b_hr, b_hz, b_hn = b_hh[0:H], b_hh[H:2 * H], b_hh[2 * H:3 * H]

    h = np.zeros(H, dtype=np.float64)
    for t in range(x_seq.shape[0]):
        x_t = x_seq[t]
        r = _sigmoid(W_ir @ x_t + b_ir + W_hr @ h + b_hr)
        z = _sigmoid(W_iz @ x_t + b_iz + W_hz @ h + b_hz)
        n = np.tanh(W_in @ x_t + b_in + r * (W_hn @ h + b_hn))
        h = (1.0 - z) * n + z * h

    W_out, b_out = weights['W_out'], weights['b_out']
    # W_out is (1, H) (straight from nn.Linear(H, 1).weight), so the matmul
    # produces a 1-element array, not a scalar - numpy 2.x no longer lets
    # float() implicitly squeeze that, so do it explicitly.
    return float(np.squeeze(W_out @ h + b_out))


def normalize_features(raw, mean, std):
    """
    Shared normalization formula - the ONLY place this epsilon/fallback
    logic is defined. train_temporal_model.py imports this same function
    (rather than re-implementing it) so the training-time and
    live-inference-time normalization can never silently drift apart.
    """
    safe_std = np.where(std > 1e-8, std, 1.0)
    return (raw - mean) / safe_std


def _verify_weights(weights):
    """
    Re-run the export-time parity check against every frozen sequence baked
    into this specific artifact. Returns (ok: bool, detail: str).
    """
    verify_X = weights.get('verify_X')
    verify_Y = weights.get('verify_Y')
    if verify_X is None or verify_Y is None:
        return False, "artifact has no verify_X/verify_Y baked in"
    if verify_X.shape[0] != verify_Y.shape[0]:
        return False, (f"verify_X/verify_Y length mismatch "
                       f"({verify_X.shape[0]} vs {verify_Y.shape[0]})")
    if verify_X.shape[0] == 0:
        # An empty verify set would otherwise "pass" vacuously (the loop
        # below just never runs) and ship a completely unverified export -
        # refuse to trust it instead.
        return False, "verify set is empty - refusing to trust an unverified export"
    for i in range(verify_X.shape[0]):
        got = gru_forward_numpy(verify_X[i], weights)
        want = float(verify_Y[i])
        if not np.allclose(got, want, atol=1e-5):
            return False, (f"sequence {i}/{verify_X.shape[0]}: "
                           f"expected logit {want:.6f}, got {got:.6f}")
    return True, f"{verify_X.shape[0]} sequences OK"


class GRUTemporalInference:
    """
    Sliding-window semantics (exact, by design):
      - Maintain a rolling buffer of the most recent TEMPORAL_SEQ_LEN=16
        feature vectors.
      - predict() returns None until 16 frames have been added.
      - The first prediction fires at frame 16 (window = frames 1-16).
      - Thereafter the window keeps rolling every frame, but the GRU is
        only actually re-run every TEMPORAL_PREDICT_EVERY_N_FRAMES=8
        frames - predict() returns the cached value in between. So the
        prediction schedule is: frame 16 (window 1-16), frame 24
        (window 9-24), frame 32 (window 17-32), and so on.
      - reset() clears the buffer, the frame counter, the cached
        prediction, AND the next-fire threshold - all scheduling state,
        not just the buffer.
    """

    def __init__(self, model_path=None):
        self.weights = None
        path = Path(model_path) if model_path else Path(config.TEMPORAL_MODEL_PATH)
        self._load(path)
        self.reset()

    def _load(self, path):
        if not path.exists():
            print(f"[temporal_model] model not found at {path} - "
                  f"temporal branch DISABLED (system will use single-frame fusion only)")
            return
        # Entire load+verify sequence lives in one try/except, matching the
        # pattern every sibling module uses (learned_fusion.py, rppg_live.py,
        # the old lstm_temporal.py): a malformed .npz - missing a key,
        # corrupted array, wrong dtype - must degrade to is_available()=False,
        # never crash app startup or a live frame.
        try:
            data = np.load(path, allow_pickle=False)
            weights = {k: data[k] for k in data.files}
            for key in ('feature_mean', 'feature_std'):
                if key not in weights:
                    raise KeyError(key)

            ok, detail = _verify_weights(weights)
            if not ok:
                print(f"[temporal_model] self-verification FAILED ({detail}) - "
                      f"temporal branch DISABLED. A corrupted file or a bad export "
                      f"must never silently serve a score.")
                return

            self.weights = weights
            print(f"[temporal_model] loaded {path}, self-verified ({detail})")
        except Exception as e:
            print(f"[temporal_model] load FAILED: {e} - temporal branch DISABLED")

    def is_available(self):
        return self.weights is not None

    def reset(self):
        self._buffer = deque(maxlen=config.TEMPORAL_SEQ_LEN)
        self._frame_count = 0
        self._last_prediction = None
        self._next_predict_at = config.TEMPORAL_SEQ_LEN

    def _to_vector(self, behavioral_features):
        raw = np.array(
            [float(behavioral_features.get(name, 0.0)) for name in TEMPORAL_FEATURE_ORDER],
            dtype=np.float64,
        )
        return normalize_features(raw, self.weights['feature_mean'], self.weights['feature_std'])

    def add_frame(self, behavioral_features):
        """behavioral_features: the dict BehavioralFeatureEngine.extract() returns."""
        if self.weights is None:
            return
        self._buffer.append(self._to_vector(behavioral_features))
        self._frame_count += 1

    def predict(self):
        """
        Returns P(stressed) in [0,1] once the buffer has 16 frames, else
        None. Re-runs the model only every TEMPORAL_PREDICT_EVERY_N_FRAMES
        frames; returns the cached value in between. See the class
        docstring for the exact fire schedule.
        """
        if self.weights is None or len(self._buffer) < config.TEMPORAL_SEQ_LEN:
            return None
        if self._frame_count >= self._next_predict_at:
            x_seq = np.stack(self._buffer)  # (16, 10), most recent 16 frames
            logit = gru_forward_numpy(x_seq, self.weights)
            self._last_prediction = float(_sigmoid(logit))
            self._next_predict_at = self._frame_count + config.TEMPORAL_PREDICT_EVERY_N_FRAMES
        return self._last_prediction

    def buffer_status(self):
        """For debugging / UI: how full is the buffer?"""
        return len(self._buffer), config.TEMPORAL_SEQ_LEN
