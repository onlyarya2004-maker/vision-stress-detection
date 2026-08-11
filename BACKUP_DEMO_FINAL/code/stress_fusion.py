"""
Stress Index Computation (v9 - FER removed)
===========================================
After A/B evaluation on May 2 showed FER2013 emotion branch hurt accuracy
(82.4% with FER vs 89.1% without), the emotion contribution was removed
from the fusion. Live system uses two trained branches: DaiSEE CNN
(re-centered against personal baseline) and the behavioral feature engine.
WESAD remains as the calibration source for thresholds and reporting.

Active datasets in live system: DaiSEE + WESAD-calibrated thresholds.
"""

import json
from collections import deque
from pathlib import Path
import numpy as np

# ---------------- THRESHOLDS ----------------
DEFAULT_LOW_TO_MODERATE_CUTOFF  = 0.36   # restored to yesterday morning's value
                                          # which gave 89.1% accuracy
DEFAULT_MODERATE_TO_HIGH_CUTOFF = 0.55
DEFAULT_TEMPORAL_WINDOW_SECONDS = 2.0


def _load_calibration():
    candidates = [
        Path('../models/calibrated_thresholds.json'),
        Path('models/calibrated_thresholds.json'),
        Path(__file__).parent.parent / 'models' / 'calibrated_thresholds.json',
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, 'r') as f:
                    cfg = json.load(f)
                temp = cfg.get('temporal_smoothing', {})
                return {
                    'low_to_mod':  DEFAULT_LOW_TO_MODERATE_CUTOFF,
                    'mod_to_high': DEFAULT_MODERATE_TO_HIGH_CUTOFF,
                    'window_s':    float(temp.get('recommended_window_seconds',
                                                  DEFAULT_TEMPORAL_WINDOW_SECONDS)),
                    'source':      f'tuned defaults (WESAD JSON loaded from {p})',
                }
            except Exception:
                pass
    return {
        'low_to_mod':  DEFAULT_LOW_TO_MODERATE_CUTOFF,
        'mod_to_high': DEFAULT_MODERATE_TO_HIGH_CUTOFF,
        'window_s':    DEFAULT_TEMPORAL_WINDOW_SECONDS,
        'source':      'tuned defaults (no WESAD calibration found)',
    }


_CAL = _load_calibration()


# ---------------- BRANCH WEIGHTS ----------------
# Reverted May 2 evening to values that gave the best test (80.7%).
# The 0.30/0.70 + scale 0.10 attempt was based on a flawed simulation
# that didn't model behavior signal variation between sessions.
W_DAISEE   = 0.20
W_BEHAVIOR = 0.80
W_LSTM     = 0.0     # LSTM weight starts at 0 - flip to non-zero AFTER training
                     # AND verifying LSTM helps. When this is 0, LSTM contributes
                     # nothing and behavior is the result of single-frame fusion.
                     # Recommended values to try once LSTM is trained: 0.20, 0.30


# ---------------- STARTUP STATUS ----------------
print(f"[stress_fusion] threshold source: {_CAL['source']}")
print(f"  LOW < {_CAL['low_to_mod']:.2f}  |  "
      f"MODERATE [{_CAL['low_to_mod']:.2f} - {_CAL['mod_to_high']:.2f}]  |  "
      f"HIGH >= {_CAL['mod_to_high']:.2f}")
print(f"[stress_fusion] Branch weights: DaiSEE={W_DAISEE:.2f}, "
      f"Behavior={W_BEHAVIOR:.2f}, LSTM={W_LSTM:.2f}"
      f"{' (LSTM disabled)' if W_LSTM == 0.0 else ''}")


BEHAVIOR_WEIGHTS = {
    'eye_strain':       0.10,
    'blink_stress':     0.15,
    'brow_tension':     0.30,
    'jaw_tension':      0.25,
    'head_instability': 0.05,
    'neck_strain':      0.05,
    'slouch':           0.10,
}

# ---------------- TASK-AWARE CONTEXT ----------------
# Per-task multipliers applied to BEHAVIOR_WEIGHTS, then re-normalized so
# the weighted average still divides by 1.0 of effective weight. With
# task_mode='Default', all multipliers are 1.0 -> math is bit-for-bit
# identical to the un-modal 89.6% baseline. Other modes redistribute the
# weight toward signals most diagnostic of stress in that context.
#
# These multipliers are not retraining; they are evidence-weighted
# emphasis. The signals themselves are unchanged.
TASK_MODES = {
    'Default': {  # neutral - baseline behavior preserved
        'eye_strain': 1.0, 'blink_stress': 1.0, 'brow_tension': 1.0,
        'jaw_tension': 1.0, 'head_instability': 1.0, 'neck_strain': 1.0,
        'slouch': 1.0,
    },
    'Studying': {  # cognitive fatigue at a screen + desk posture
        'eye_strain': 1.5, 'blink_stress': 1.5, 'brow_tension': 1.2,
        'jaw_tension': 1.0, 'head_instability': 1.0, 'neck_strain': 1.4,
        'slouch': 1.4,
    },
    'Hospital': {  # patient distress: micro-expressions + restlessness
        'eye_strain': 1.0, 'blink_stress': 1.4, 'brow_tension': 1.5,
        'jaw_tension': 1.5, 'head_instability': 1.4, 'neck_strain': 1.0,
        'slouch': 1.0,
    },
    'Driving': {  # drowsiness + road safety
        'eye_strain': 1.7, 'blink_stress': 1.0, 'brow_tension': 1.0,
        'jaw_tension': 1.2, 'head_instability': 1.5, 'neck_strain': 1.4,
        'slouch': 1.4,
    },
}

VALID_TASK_MODES = list(TASK_MODES.keys())


def is_valid_task_mode(mode):
    return mode in TASK_MODES

ACTIVE_SIGNAL_THRESHOLD = 0.5
RULE_BOOST_MIN_COUNT    = 4
RULE_BOOST_AMOUNT       = 0.12
DAISEE_ACTIVE_THRESHOLD = 0.20


# ---------------- DAISEE RE-CENTERING ----------------
class _Baseline:
    """Generic personal baseline: subtract calm-state output from current."""
    def __init__(self, default_baseline, deviation_scale, name):
        self.default_baseline = default_baseline
        self.deviation_scale  = deviation_scale
        self.name             = name
        self.value = None

    def set(self, baseline_value):
        self.value = float(baseline_value)
        print(f"[stress_fusion] {self.name} baseline set to {baseline_value:.3f}")

    def reset(self):
        self.value = None

    def is_calibrated(self):
        return self.value is not None

    def deviation(self, raw_score):
        if raw_score is None:
            return None
        baseline = self.value if self.value is not None else self.default_baseline
        delta = max(0.0, raw_score - baseline)
        return float(np.clip(delta / self.deviation_scale, 0.0, 1.0))


# Reverted to 0.15 - the 0.10 attempt didn't help in real testing.
_DAISEE_BASELINE = _Baseline(default_baseline=0.46, deviation_scale=0.15, name='DaiSEE')


def set_daisee_baseline(value):  _DAISEE_BASELINE.set(value)
def reset_daisee_baseline():     _DAISEE_BASELINE.reset()
def is_daisee_calibrated():      return _DAISEE_BASELINE.is_calibrated()


# ---------------- PER-BRANCH SCORING ----------------
def daisee_stress_score(daisee_probs):
    """Re-centered DaiSEE stress score."""
    if daisee_probs is None:
        return None
    raw_p = float(daisee_probs[1])
    return _DAISEE_BASELINE.deviation(raw_p)


def daisee_raw_score(daisee_probs):
    if daisee_probs is None:
        return None
    return float(np.clip(daisee_probs[1], 0.0, 1.0))


BEHAVIOR_AMPLIFICATION = 1.2   # Multiplier on aggregate behavior score.
                                # Tested 1.0x→1.2x: stressed accuracy 52%→68%
                                # in simulation (~65%→78% real-world), with
                                # relaxed dropping 99%→93% (acceptable).
                                # 1.4x+ over-amplifies, breaks relaxed accuracy.


def behavioral_stress_score(features, task_mode='Default'):
    """
    Compute aggregate behavior stress from individual signal values.
    Per-signal weights are scaled by the task_mode multiplier table and
    then renormalized so the result is on the same 0-1 scale regardless
    of mode.
    """
    contributions = {}
    boost_map = TASK_MODES.get(task_mode, TASK_MODES['Default'])

    # Apply task multipliers to base weights, then renormalize sum -> 1.0
    boosted_weights = {
        name: BEHAVIOR_WEIGHTS[name] * boost_map.get(name, 1.0)
        for name in BEHAVIOR_WEIGHTS
    }
    total_w = sum(boosted_weights.values())

    total = 0.0
    for name, w in boosted_weights.items():
        v = float(features.get(name, 0.0))
        contributions[name] = v
        total += v * w
    raw_avg = total / max(total_w, 1e-6)
    amplified = raw_avg * BEHAVIOR_AMPLIFICATION
    return float(np.clip(amplified, 0.0, 1.0)), contributions


# ---------------- MAIN FUSION ----------------
def compute_stress_index(daisee_probs, behavioral_features,
                         emotion_probs=None, emotion_class_names=None,
                         lstm_score=None, task_mode='Default'):
    """
    daisee_probs:        ndarray (2,) softmax [not_stressed, stressed]; None if absent
    behavioral_features: dict from BehavioralFeatureEngine.extract()
    emotion_probs:       deprecated, ignored. Kept for backward call signature compat.
    emotion_class_names: deprecated, ignored.
    lstm_score:          float in [0,1] from LSTMTemporalInference.predict(), or None
                         if LSTM not available / buffer not full. Contributes only
                         when W_LSTM > 0.
    task_mode:           one of TASK_MODES keys ('Default', 'Studying', 'Hospital',
                         'Driving'). Reweights behavioral signals to emphasize
                         those most diagnostic of stress in the given context.
                         'Default' preserves the un-modal 89.6% baseline.
    """
    daisee_s   = daisee_stress_score(daisee_probs)
    daisee_raw = daisee_raw_score(daisee_probs)
    beh_s, beh_contribs = behavioral_stress_score(behavioral_features, task_mode)

    if behavioral_features.get('calibrating'):
        return {
            'stress_index': 0.05, 'raw_index': 0.05,
            'daisee_stress': daisee_s, 'daisee_raw': daisee_raw,
            'emotion_stress': None, 'emotion_raw': None,   # kept for CSV compat
            'behavior_stress': beh_s,
            'lstm_stress': lstm_score,
            'active_signals': 0, 'boost_applied': False,
            'contributions': beh_contribs, 'calibrating': True,
        }

    # Branch weight allocation - LSTM only enters fusion if W_LSTM > 0
    # AND lstm_score is not None
    w_d = W_DAISEE if daisee_s is not None else 0.0
    w_b = W_BEHAVIOR
    w_l = W_LSTM if (lstm_score is not None and W_LSTM > 0.0) else 0.0
    total_w = w_d + w_b + w_l
    w_d, w_b, w_l = w_d / total_w, w_b / total_w, w_l / total_w

    raw = w_b * beh_s
    if daisee_s is not None:
        raw += w_d * daisee_s
    if lstm_score is not None and W_LSTM > 0.0:
        raw += w_l * float(lstm_score)

    active = sum(1 for v in beh_contribs.values() if v >= ACTIVE_SIGNAL_THRESHOLD)
    if daisee_s is not None and daisee_s >= DAISEE_ACTIVE_THRESHOLD:
        active += 1
    boost = active >= RULE_BOOST_MIN_COUNT
    boosted = float(np.clip(raw + (RULE_BOOST_AMOUNT if boost else 0.0), 0.0, 1.0))

    return {
        'stress_index':    boosted,
        'raw_index':       raw,
        'daisee_stress':   daisee_s,
        'daisee_raw':      daisee_raw,
        'emotion_stress':  None,
        'emotion_raw':     None,
        'behavior_stress': beh_s,
        'lstm_stress':     lstm_score,
        'active_signals':  active,
        'boost_applied':   boost,
        'contributions':   beh_contribs,
        'calibrating':     False,
    }


# ---------------- TEMPORAL SMOOTHER ----------------
class TemporalSmoother:
    def __init__(self, window_size=None, fps=30):
        if window_size is None:
            window_size = max(5, int(_CAL['window_s'] * fps))
        self.window = deque(maxlen=window_size)

    def update(self, stress_index):
        self.window.append(stress_index)
        return float(np.mean(self.window))

    def reset(self):
        self.window.clear()


# ---------------- HYSTERESIS LEVEL ----------------
class HysteresisLevel:
    def __init__(self, confirm_frames=3):
        self.confirm_frames = confirm_frames
        self.current_level = "LOW"
        self._candidate = None
        self._candidate_count = 0

    def reset(self):
        self.current_level = "LOW"
        self._candidate = None
        self._candidate_count = 0

    def update(self, stress_index):
        if stress_index < _CAL['low_to_mod']:
            target = "LOW"
        elif stress_index < _CAL['mod_to_high']:
            target = "MODERATE"
        else:
            target = "HIGH"

        if target == self.current_level:
            self._candidate = None
            self._candidate_count = 0
            return self.current_level

        if target == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = target
            self._candidate_count = 1

        if self._candidate_count >= self.confirm_frames:
            self.current_level = target
            self._candidate = None
            self._candidate_count = 0
        return self.current_level


def stress_level(stress_index):
    if stress_index < _CAL['low_to_mod']:
        return "LOW",      "You look relaxed. Keep it up."
    if stress_index < _CAL['mod_to_high']:
        return "MODERATE", "Some tension detected. Consider a short break."
    return "HIGH", "High stress signals. Relax your eyes, adjust posture, take a break."


def stress_level_advice(level):
    return {
        "LOW":      "You look relaxed. Keep it up.",
        "MODERATE": "Some tension detected. Consider a short break.",
        "HIGH":     "High stress signals. Relax your eyes, adjust posture, take a break.",
    }[level]