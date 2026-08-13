"""
rPPG method tests.

These validate the GREEN / CHROM / POS implementations against a SYNTHETIC
pulse of known frequency. That matters because on real data there is nothing
to check the algorithms against except the ground truth we are trying to
predict - if the implementation were subtly wrong, a poor result would be
indistinguishable from "rPPG does not work here". Here the right answer is
known by construction.
"""

import numpy as np
import pytest

from analyze_rppg import (
    METHODS, hr_from_signal, method_chrom, method_green, method_pos,
    windowed_hr,
)


def synthetic_rgb(bpm=72.0, fs=30.0, seconds=20.0, noise=0.0,
                  motion=0.0, seed=0):
    """
    Build an RGB trace containing a known pulse.

    Skin exhibits the pulse most strongly in green, weakly in red and blue.
    `motion` adds a slow common-mode drift across all channels - the specular
    component CHROM and POS are specifically designed to suppress, and which
    plain GREEN is not.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * (bpm / 60.0) * t)

    base = np.array([140.0, 120.0, 110.0])          # mean skin RGB
    gain = np.array([0.3, 1.0, 0.4])                # green carries the pulse
    rgb = base + np.outer(pulse, gain)

    if motion:
        drift = motion * np.sin(2 * np.pi * 0.15 * t)   # 0.15 Hz, below band
        rgb = rgb + drift[:, None] * base / 100.0
    if noise:
        rgb = rgb + rng.normal(0, noise, rgb.shape)
    return rgb


@pytest.mark.parametrize("bpm", [55.0, 72.0, 100.0, 140.0])
@pytest.mark.parametrize("name", sorted(METHODS))
def test_methods_recover_known_heart_rate(name, bpm):
    """Every method must recover a clean synthetic pulse within 5 bpm."""
    rgb = synthetic_rgb(bpm=bpm, fs=30.0, seconds=20.0, noise=0.2)
    sig = METHODS[name](rgb, 30.0)
    est = hr_from_signal(sig, 30.0)
    assert abs(est - bpm) < 5.0, f"{name}: got {est:.1f}, expected {bpm}"


@pytest.mark.parametrize("name", sorted(METHODS))
def test_methods_work_at_low_sample_rate(name):
    """
    UBFC-Phys subsampled to ~11-17 Hz still has to work. Nyquist at 11.7 Hz
    is 5.85 Hz (351 bpm), so the cardiac band is comfortably resolvable even
    though it is below the usual rPPG recommendation.
    """
    rgb = synthetic_rgb(bpm=75.0, fs=11.7, seconds=30.0, noise=0.2)
    est = hr_from_signal(METHODS[name](rgb, 11.7), 11.7)
    assert abs(est - 75.0) < 8.0, f"{name}: got {est:.1f} at 11.7 Hz"


def test_chrom_and_pos_beat_green_under_motion():
    """
    The reason CHROM and POS exist: they suppress the common-mode (motion)
    component that swamps the raw green channel. If this ever fails, the
    projection maths is wrong.
    """
    rgb = synthetic_rgb(bpm=72.0, fs=30.0, seconds=20.0,
                        noise=0.2, motion=8.0)
    err = {
        'green': abs(hr_from_signal(method_green(rgb, 30.0), 30.0) - 72.0),
        'chrom': abs(hr_from_signal(method_chrom(rgb, 30.0), 30.0) - 72.0),
        'pos':   abs(hr_from_signal(method_pos(rgb, 30.0), 30.0) - 72.0),
    }
    assert min(err['chrom'], err['pos']) <= err['green'] + 1e-6, (
        f"motion suppression failed: {err}")


def test_hr_estimate_stays_in_physiological_band():
    """Pure noise must not yield an out-of-band heart rate."""
    rng = np.random.default_rng(1)
    rgb = 120 + rng.normal(0, 3, (600, 3))
    for fn in METHODS.values():
        hr = hr_from_signal(fn(rgb, 30.0), 30.0)
        assert np.isnan(hr) or 42.0 <= hr <= 180.0


def test_windowed_hr_tracks_a_changing_rate():
    """A rate that rises partway through must be reflected in the windows."""
    fs = 30.0
    slow = synthetic_rgb(bpm=60.0, fs=fs, seconds=20.0, noise=0.1)
    fast = synthetic_rgb(bpm=110.0, fs=fs, seconds=20.0, noise=0.1, seed=2)
    rgb = np.vstack([slow, fast])

    times, hrs = windowed_hr(method_pos(rgb, fs), fs)
    valid = ~np.isnan(hrs)
    assert valid.sum() >= 4
    first = np.nanmean(hrs[times < 15])
    last = np.nanmean(hrs[times > 25])
    assert last > first + 15, f"did not track the change: {first} -> {last}"


def test_short_signal_returns_nan_not_garbage():
    """Too little data must be reported as unknown, not guessed at."""
    assert np.isnan(hr_from_signal(np.zeros(10), 30.0))
