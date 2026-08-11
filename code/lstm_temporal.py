"""
LSTM Temporal Inference Module
==============================
Wraps the trained LSTM model for real-time use. Maintains an internal
sliding buffer of recent face frames; when the buffer is full, runs
the LSTM and emits a temporal stress score in [0, 1].

Usage from app.py:
    from lstm_temporal import LSTMTemporalInference

    lstm_inf = LSTMTemporalInference()       # tries to load model, fails gracefully

    # In the main loop:
    if lstm_inf.is_available():
        lstm_inf.add_frame(face_crop_bgr)
        lstm_score = lstm_inf.predict()      # None until buffer fills, then [0,1]

If the model file doesn't exist or fails to load, is_available() returns
False and the rest of the system runs unchanged - this is how we keep the
89.6% baseline safe.
"""

from collections import deque
from pathlib import Path
import numpy as np
import cv2

# Lazy import - only loaded if model exists, so app.py doesn't pay startup cost
# when LSTM isn't trained yet.

LSTM_MODEL_PATH = Path('../models/lstm_stress.h5')

# Must match training values
SEQ_LEN = 16
IMG_SIZE = 96
PREDICT_EVERY_N_FRAMES = 8   # Re-run LSTM every 8 frames of new data,
                              # not every frame (too slow for live use)


class LSTMTemporalInference:
    def __init__(self, model_path=None):
        """Try to load the LSTM model. If anything fails, mark as unavailable
        and we let the rest of the system continue without LSTM."""
        self.model = None
        self._buffer = deque(maxlen=SEQ_LEN)
        self._frames_since_predict = 0
        self._last_score = None
        self._preprocess_fn = None

        path = Path(model_path) if model_path else LSTM_MODEL_PATH
        if not path.exists():
            print(f"[lstm_temporal] LSTM model not found at {path} - "
                  f"LSTM branch DISABLED (system will use single-frame fusion only)")
            return

        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            self._preprocess_fn = preprocess_input
            self.model = tf.keras.models.load_model(str(path))
            print(f"[lstm_temporal] LSTM model loaded from {path}")
        except Exception as e:
            print(f"[lstm_temporal] LSTM model load FAILED: {e}")
            print(f"[lstm_temporal] LSTM branch DISABLED (system will use "
                  f"single-frame fusion only)")
            self.model = None

    def is_available(self):
        return self.model is not None

    def reset(self):
        """Clear the buffer. Call when starting a new session."""
        self._buffer.clear()
        self._frames_since_predict = 0
        self._last_score = None

    def add_frame(self, face_bgr):
        """Add a face crop (BGR, any size) to the rolling buffer."""
        if self.model is None or face_bgr is None:
            return
        # Resize and convert to RGB at insertion time
        img = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
        self._buffer.append(img)
        self._frames_since_predict += 1

    def predict(self):
        """
        Returns a stress probability in [0, 1] (P(stressed)) if the buffer
        is full and enough new frames have arrived since last prediction.
        Returns the cached last score otherwise (or None if no prediction
        has been made yet).
        """
        if self.model is None:
            return None
        if len(self._buffer) < SEQ_LEN:
            return None
        if self._frames_since_predict < PREDICT_EVERY_N_FRAMES and \
                self._last_score is not None:
            return self._last_score

        # Build the input tensor (1, 16, 96, 96, 3)
        seq = np.stack(list(self._buffer), axis=0)
        seq = self._preprocess_fn(seq)
        seq = np.expand_dims(seq, axis=0)

        probs = self.model.predict(seq, verbose=0)[0]
        # probs[0] = P(not_stressed), probs[1] = P(stressed)
        score = float(probs[1])
        self._last_score = score
        self._frames_since_predict = 0
        return score

    def buffer_status(self):
        """For debugging / UI: how full is the buffer?"""
        return len(self._buffer), SEQ_LEN
