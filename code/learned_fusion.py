"""
Learned fusion inference
========================
Loads the model fitted by train_meta_learner.py and scores live behavioural
features with it, as a drop-in alternative to the hand-picked weights.

Measured on 5 UBFC-Phys subjects, leave-one-subject-out:

    hand-tuned fusion     57.73%
    logistic regression   68.22%
    gradient boosting     75.19%

and the weights it learned bear little resemblance to the assumed ones:

    head_instability   0.05 -> 0.52   (the best signal, weighted least)
    brow_tension       0.30 -> 0.14   (weighted most)
    eye_strain         0.10 -> 0.00   (contributes nothing)
    neck_strain        0.05 -> negative coefficient

DEFAULT OFF, DELIBERATELY
-------------------------
config.USE_LEARNED_FUSION is False. The current model was trained entirely on
UBFC-Phys - a 1024x1024 industrial camera, a French university cohort, and a
speech/arithmetic stress protocol. None of that is a laptop webcam in
Bengaluru. Enabling it live would repeat exactly the mistake this whole
investigation uncovered: assuming a fitted-elsewhere constant transfers.

The honest sequence is: record the team under RECORDING_PROTOCOL.md, retrain
or fine-tune on that data, compare against hand-tuned on the SAME recordings,
and only then flip the default. Until then this is a validated capability
sitting behind a switch, not a claim.
"""

from pathlib import Path

import numpy as np

import config

# Feature order is part of the model contract - a different order silently
# produces nonsense rather than an error.
FEATURE_ORDER = ['eye_strain', 'blink_stress', 'brow_tension', 'jaw_tension',
                 'head_instability', 'neck_strain', 'slouch']


class LearnedFusion:
    """
    Wraps the persisted meta-learner. Mirrors LSTMTemporalInference's
    contract: if the model is missing or unloadable, is_available() returns
    False and the caller falls back to the hand-tuned path unchanged.
    """

    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.features = None
        self._path = Path(model_path or config.LEARNED_FUSION_MODEL_PATH)

        if not self._path.exists():
            print(f"[learned_fusion] no model at {self._path} - "
                  f"using hand-tuned weights")
            return
        try:
            import joblib
            blob = joblib.load(self._path)
            self.model = blob['model']
            self.scaler = blob['scaler']
            self.features = blob.get('features', FEATURE_ORDER)
            print(f"[learned_fusion] loaded {self._path.name} "
                  f"({len(self.features)} features)")
        except Exception as e:                      # pragma: no cover
            print(f"[learned_fusion] load FAILED ({e}) - "
                  f"using hand-tuned weights")
            self.model = None

    def is_available(self):
        return self.model is not None

    def score(self, features):
        """
        P(stressed) in [0, 1] from a behavioural feature dict, or None when
        the model is unavailable. Returned on the same scale as the
        hand-tuned behaviour score, so it substitutes cleanly.
        """
        if self.model is None:
            return None
        try:
            x = np.array([[float(features.get(n, 0.0))
                           for n in self.features]])
            if self.scaler is not None:
                x = self.scaler.transform(x)
            if hasattr(self.model, 'predict_proba'):
                return float(self.model.predict_proba(x)[0][1])
            # Fall back to a squashed decision function.
            d = float(self.model.decision_function(x)[0])
            return float(1.0 / (1.0 + np.exp(-d)))
        except Exception:                            # pragma: no cover
            return None

    def weights(self):
        """Learned coefficients keyed by signal, for display and reporting."""
        if self.model is None or not hasattr(self.model, 'coef_'):
            return None
        return dict(zip(self.features, self.model.coef_[0].tolist()))


_INSTANCE = None


def get_learned_fusion():
    """Process-wide singleton; loading joblib on every frame would be silly."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LearnedFusion()
    return _INSTANCE


def reset():
    """Drop the cached instance (tests, or after retraining)."""
    global _INSTANCE
    _INSTANCE = None
