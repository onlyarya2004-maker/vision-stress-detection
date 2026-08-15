"""
Live rPPG branch tests.

Two properties matter most:

1. It ABSTAINS rather than guessing. On UBFC-Phys the speech task gave
   15.3 bpm error while the arithmetic task gave 0.3 bpm on the SAME subject
   - motion is the dominant error source, so the branch must return None when
   the signal is poor rather than fuse a motion artefact.

2. It changes nothing at W_RPPG = 0.0, the shipped default. The offline
   numbers come from a 1024x1024 MJPG camera; a compressed 640x480 webcam is
   unproven, and compression attacks exactly the sub-1% colour changes rPPG
   reads.
"""

import numpy as np
import pytest

import config
import stress_fusion
from rppg_live import LiveRPPG, spectral_quality


def _pulse_rgb(bpm, fs, seconds, noise=0.0, seed=0):
    """RGB trace carrying a known pulse, strongest in green as real skin is."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)
    base = np.array([140.0, 120.0, 110.0])
    rgb = base + np.outer(pulse, np.array([0.3, 1.0, 0.4]))
    if noise:
        rgb = rgb + rng.normal(0, noise, rgb.shape)
    return t, rgb


def _feed(est, bpm=72.0, fs=30.0, seconds=15.0, noise=0.1, t0=0.0, seed=0):
    """Push a synthetic pulse straight into the buffer, bypassing MediaPipe."""
    t, rgb = _pulse_rgb(bpm, fs, seconds, noise, seed)
    for i in range(len(t)):
        est._buf.append((t0 + t[i], rgb[i, 0], rgb[i, 1], rgb[i, 2]))
    est._frames_since_estimate = 10 ** 6      # force an estimate
    return t0 + t[-1]


# ---------------------------------------------------------------------------
# Default-off guarantee
# ---------------------------------------------------------------------------

def test_rppg_weight_is_zero_by_default():
    """Unproven on webcam hardware; must not silently influence results."""
    assert config.W_RPPG == 0.0


def test_fusion_unchanged_when_weight_is_zero():
    feats = {'eye_strain': 0.3, 'blink_stress': 0.2, 'brow_tension': 0.4,
             'jaw_tension': 0.3, 'head_instability': 0.5,
             'neck_strain': 0.1, 'slouch': 0.2, 'calibrating': False}
    without = stress_fusion.compute_stress_index(None, feats)
    withrppg = stress_fusion.compute_stress_index(None, feats, rppg_score=0.9)
    assert withrppg['stress_index'] == pytest.approx(without['stress_index'])
    assert withrppg['rppg_stress'] == 0.9        # still reported


def test_rppg_score_is_reported_even_at_zero_weight():
    """It must be observable so it can be validated before being trusted."""
    feats = {'brow_tension': 0.2, 'calibrating': False}
    r = stress_fusion.compute_stress_index(None, feats, rppg_score=0.42)
    assert r['rppg_stress'] == 0.42


def test_enabled_weight_actually_shifts_the_index(monkeypatch):
    monkeypatch.setattr(stress_fusion, 'W_RPPG', 0.30)
    feats = {'brow_tension': 0.1, 'calibrating': False}
    low = stress_fusion.compute_stress_index(None, feats, rppg_score=0.0)
    high = stress_fusion.compute_stress_index(None, feats, rppg_score=1.0)
    assert high['stress_index'] > low['stress_index']


def test_abstention_redistributes_weight(monkeypatch):
    """rppg_score=None must behave exactly as if the branch did not exist."""
    monkeypatch.setattr(stress_fusion, 'W_RPPG', 0.30)
    feats = {'brow_tension': 0.4, 'calibrating': False}
    abstained = stress_fusion.compute_stress_index(None, feats, rppg_score=None)

    monkeypatch.setattr(stress_fusion, 'W_RPPG', 0.0)
    absent = stress_fusion.compute_stress_index(None, feats, rppg_score=None)
    assert abstained['stress_index'] == pytest.approx(absent['stress_index'])


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bpm", [55.0, 72.0, 110.0])
def test_recovers_a_known_heart_rate(bpm):
    est = LiveRPPG()
    _feed(est, bpm=bpm, fs=30.0, seconds=15.0)
    hr, quality = est.estimate(force=True)
    assert hr is not None, f"no estimate at {bpm} bpm (quality {quality:.2f})"
    assert abs(hr - bpm) < 6.0


def test_too_little_data_yields_no_estimate():
    """Under RPPG_MIN_SECONDS it must abstain, not extrapolate."""
    est = LiveRPPG()
    _feed(est, seconds=3.0)
    hr, _ = est.estimate(force=True)
    assert hr is None


def test_noise_is_rejected_by_the_quality_gate():
    """Pure noise has no spectral peak; the branch must abstain."""
    rng = np.random.default_rng(3)
    est = LiveRPPG()
    for i in range(400):
        v = 120 + rng.normal(0, 4, 3)
        est._buf.append((i / 30.0, v[0], v[1], v[2]))
    est._frames_since_estimate = 10 ** 6
    assert est.stress_score() is None


def test_quality_is_higher_for_a_clean_pulse_than_for_noise():
    fs = 30.0
    _, clean = _pulse_rgb(72.0, fs, 15.0, noise=0.05)
    rng = np.random.default_rng(0)
    noisy = 120 + rng.normal(0, 5, (int(fs * 15), 3))
    from analyze_rppg import method_pos
    q_clean, _ = spectral_quality(method_pos(clean, fs), fs)
    q_noise, _ = spectral_quality(method_pos(noisy, fs), fs)
    assert q_clean > q_noise


# ---------------------------------------------------------------------------
# Resting baseline and scoring
# ---------------------------------------------------------------------------

def test_no_score_before_a_resting_baseline_exists():
    """Elevation is meaningless without a personal reference."""
    est = LiveRPPG()
    _feed(est, bpm=72.0, seconds=15.0)
    est.estimate(force=True)
    assert est.has_resting_baseline() is False
    assert est.stress_score() is None


def test_elevated_heart_rate_scores_above_resting():
    """
    The WESAD-derived scale: ~25 bpm above resting is a full-scale response
    (baseline 72.41 vs stress 97.49).
    """
    est = LiveRPPG()
    t = 0.0
    for i in range(config.RPPG_RESTING_SAMPLES):
        t = _feed(est, bpm=70.0, seconds=15.0, t0=t, seed=i)
        est.estimate(force=True)
    assert est.has_resting_baseline() is True
    assert est.stress_score() == pytest.approx(0.0, abs=0.15)

    est._buf.clear()
    _feed(est, bpm=95.0, seconds=15.0, t0=t, seed=99)
    score = est.stress_score()
    assert score is not None and score > 0.5, f"elevated HR scored {score}"


def test_reset_clears_everything():
    est = LiveRPPG()
    _feed(est, seconds=15.0)
    est.estimate(force=True)
    est.reset()
    assert len(est._buf) == 0
    assert est.is_available() is False
    assert est.has_resting_baseline() is False


def test_status_is_informative_at_each_stage():
    est = LiveRPPG()
    assert "no face" in est.status()
    _feed(est, seconds=3.0)
    assert "warming up" in est.status() or "low frame rate" in est.status()
