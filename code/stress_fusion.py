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

import config

# ---------------- THRESHOLDS ----------------
# Sourced from config.py. These are empirically tuned values; the WESAD
# JSON's own 'vision_system_cutoffs' are deliberately NOT used - see the
# WESAD_CALIBRATION note in config.py for why (scale mismatch between a
# physiological classifier's probabilities and the vision stress index).
DEFAULT_LOW_TO_MODERATE_CUTOFF  = config.LOW_TO_MODERATE_CUTOFF
DEFAULT_MODERATE_TO_HIGH_CUTOFF = config.MODERATE_TO_HIGH_CUTOFF


def load_wesad_provenance():
    """
    Read calibrated_thresholds.json for REPORTING only.

    This used to be _load_calibration(), which read the file, threw away its
    cutoffs, kept a smoothing window that both runners then overrode, and
    printed a message implying the thresholds came from WESAD. Nothing about
    the live decision boundaries depends on this function; it exists so the
    startup banner and the evaluation harness can state what calibration
    artifact is present.

    Returns a dict with the WESAD LOSO figures, or None if absent.
    """
    candidates = [Path(p) for p in config.CALIBRATION_JSON_CANDIDATES]
    candidates.append(Path(__file__).parent.parent / 'models'
                      / 'calibrated_thresholds.json')
    for p in candidates:
        if p.exists():
            try:
                with open(p, 'r') as f:
                    cfg = json.load(f)
                loso = cfg.get('classifier_loso', {})
                return {
                    'path':      str(p),
                    'accuracy':  loso.get('accuracy_mean'),
                    'f1':        loso.get('f1_mean'),
                    'subjects':  len(cfg.get('metadata', {})
                                     .get('subjects_used', [])),
                }
            except Exception:
                pass
    return None


_WESAD = load_wesad_provenance()

# Kept for backward compatibility with any code reading _CAL directly.
_CAL = {
    'low_to_mod':  config.LOW_TO_MODERATE_CUTOFF,
    'mod_to_high': config.MODERATE_TO_HIGH_CUTOFF,
}


# ---------------- BRANCH WEIGHTS ----------------
W_DAISEE   = config.W_DAISEE
W_BEHAVIOR = config.W_BEHAVIOR
W_TEMPORAL = config.W_TEMPORAL
W_RPPG     = config.W_RPPG


# ---------------- STARTUP STATUS ----------------
print(f"[stress_fusion] {config.summary()}")
print(f"  LOW < {_CAL['low_to_mod']:.2f}  |  "
      f"MODERATE [{_CAL['low_to_mod']:.2f} - {_CAL['mod_to_high']:.2f}]  |  "
      f"HIGH >= {_CAL['mod_to_high']:.2f}")
if _WESAD and _WESAD.get('accuracy') is not None:
    print(f"[stress_fusion] WESAD reference present: "
          f"{_WESAD['accuracy']*100:.2f}% LOSO over {_WESAD['subjects']} "
          f"subjects (provenance only - does NOT set the cutoffs above)")
else:
    print("[stress_fusion] no WESAD calibration artifact found "
          "(cutoffs are unaffected either way)")
if W_TEMPORAL == 0.0:
    print("[stress_fusion] Temporal branch disabled (W_TEMPORAL = 0.0)")


BEHAVIOR_WEIGHTS = config.BEHAVIOR_WEIGHTS

# ---------------- TASK-AWARE CONTEXT ----------------
# Per-task multipliers applied to BEHAVIOR_WEIGHTS, then re-normalized so
# the weighted average still divides by 1.0 of effective weight. With
# task_mode='Default', all multipliers are 1.0 -> math is bit-for-bit
# identical to the un-modal baseline (verified in tests/test_stress_fusion.py
# against the archived pre-task-aware implementation).
TASK_MODES = config.TASK_MODES

VALID_TASK_MODES = list(TASK_MODES.keys())


def is_valid_task_mode(mode):
    return mode in TASK_MODES

ACTIVE_SIGNAL_THRESHOLD = config.ACTIVE_SIGNAL_THRESHOLD
RULE_BOOST_MIN_COUNT    = config.RULE_BOOST_MIN_COUNT
RULE_BOOST_AMOUNT       = config.RULE_BOOST_AMOUNT
DAISEE_ACTIVE_THRESHOLD = config.DAISEE_ACTIVE_THRESHOLD


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


_DAISEE_BASELINE = _Baseline(
    default_baseline=config.DAISEE_DEFAULT_BASELINE,
    deviation_scale=config.DAISEE_DEVIATION_SCALE,
    name='DaiSEE',
)


def set_daisee_baseline(value):  _DAISEE_BASELINE.set(value)
def reset_daisee_baseline():     _DAISEE_BASELINE.reset()
def is_daisee_calibrated():      return _DAISEE_BASELINE.is_calibrated()
def get_daisee_baseline():       return _DAISEE_BASELINE.value


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


BEHAVIOR_AMPLIFICATION = config.BEHAVIOR_AMPLIFICATION


def behavioral_stress_score(features, task_mode='Default'):
    """
    Compute aggregate behavior stress from individual signal values.
    Per-signal weights are scaled by the task_mode multiplier table and
    then renormalized so the result is on the same 0-1 scale regardless
    of mode.
    """
    contributions = {name: float(features.get(name, 0.0))
                     for name in BEHAVIOR_WEIGHTS}

    # Learned path, when enabled AND a model is present. Falls through to the
    # hand-tuned weighting otherwise, so behaviour is unchanged by default.
    # Contributions are still reported from the raw signals, so the UI and the
    # feature dumps look identical either way.
    if config.USE_LEARNED_FUSION:
        from learned_fusion import get_learned_fusion
        learned = get_learned_fusion()
        if learned.is_available():
            score = learned.score(features)
            if score is not None:
                return float(np.clip(score, 0.0, 1.0)), contributions

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
                         temporal_score=None, task_mode='Default',
                         rppg_score=None):
    """
    daisee_probs:        ndarray (2,) softmax [not_stressed, stressed]; None if absent
    behavioral_features: dict from BehavioralFeatureEngine.extract()
    emotion_probs:       deprecated, ignored. Kept for backward call signature compat.
    emotion_class_names: deprecated, ignored.
    temporal_score:      float in [0,1] from GRUTemporalInference.predict(), or None
                         if the temporal model is unavailable / buffer not full.
                         Contributes only when W_TEMPORAL > 0.
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
            'temporal_stress': temporal_score,
            'rppg_stress': rppg_score,
            'active_signals': 0, 'boost_applied': False,
            'contributions': beh_contribs, 'calibrating': True,
        }

    # Branch weight allocation - the temporal branch only enters fusion if
    # W_TEMPORAL > 0 AND temporal_score is not None
    w_d = W_DAISEE if daisee_s is not None else 0.0
    w_b = W_BEHAVIOR
    w_t = W_TEMPORAL if (temporal_score is not None and W_TEMPORAL > 0.0) else 0.0
    # rPPG abstains (None) whenever signal quality is poor - typically while
    # the user is moving or talking - so its weight is redistributed rather
    # than a motion artefact being fused in.
    w_r = W_RPPG if (rppg_score is not None and W_RPPG > 0.0) else 0.0
    # W_BEHAVIOR alone keeps total_w > 0 under the shipped defaults, but an
    # ablation run (config.py's whole reason to exist) can set it to 0 - if
    # that coincides with DaiSEE failing to load, this would otherwise be an
    # uncaught ZeroDivisionError on every single frame of the live capture
    # loop. Same guard as behavioral_stress_score()'s total_w a few lines up.
    total_w = max(w_d + w_b + w_t + w_r, 1e-6)
    w_d, w_b = w_d / total_w, w_b / total_w
    w_t, w_r = w_t / total_w, w_r / total_w

    raw = w_b * beh_s
    if daisee_s is not None:
        raw += w_d * daisee_s
    if temporal_score is not None and W_TEMPORAL > 0.0:
        raw += w_t * float(temporal_score)
    if rppg_score is not None and W_RPPG > 0.0:
        raw += w_r * float(rppg_score)

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
        'temporal_stress': temporal_score,
        'rppg_stress':     rppg_score,
        'active_signals':  active,
        'boost_applied':   boost,
        'contributions':   beh_contribs,
        'calibrating':     False,
    }


# ---------------- TEMPORAL SMOOTHER ----------------
class TemporalSmoother:
    def __init__(self, window_size=None, fps=30):
        # Previously defaulted to a window read from the WESAD JSON, which
        # both runners then overrode - dead code. The default is now the
        # value the evaluated path actually uses.
        if window_size is None:
            window_size = config.SMOOTHING_WINDOW_FRAMES
        self.window = deque(maxlen=max(1, int(window_size)))

    def update(self, stress_index):
        self.window.append(stress_index)
        return float(np.mean(self.window))

    def reset(self):
        self.window.clear()


# ---------------- HYSTERESIS LEVEL ----------------
class HysteresisLevel:
    def __init__(self, confirm_frames=config.HYSTERESIS_CONFIRM_FRAMES):
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