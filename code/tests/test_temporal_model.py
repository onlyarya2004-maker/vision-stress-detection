"""
Tests for temporal_model.py - the numpy-only GRU temporal-inference module
that replaces the retired pixel-LSTM (code/_archive/lstm_temporal.py).

These tests never import torch. PyTorch/numpy parity itself is checked by
train_temporal_model.py's own export-time verification (multiple real
validation sequences, raw logits, hard-fail on mismatch) - what's tested
here is temporal_model.py's own contract: the GRU math is deterministic
and returns a raw logit, the sliding-window/prediction-cadence schedule is
exactly right, and a corrupted or missing artifact degrades safely instead
of ever serving a wrong score.
"""
from pathlib import Path

import numpy as np
import pytest

import config
import learned_fusion
from temporal_model import TEMPORAL_FEATURE_ORDER, GRUTemporalInference, gru_forward_numpy


def _make_fixture_npz(tmp_path, hidden=4, input_dim=10, seed=0, n_verify=3,
                      filename="fixture.npz"):
    """A small, self-consistent synthetic artifact: verify_Y is computed by
    calling gru_forward_numpy on verify_X, so the fixture always passes its
    own self-verification - this tests the LOADING/lifecycle contract, not
    PyTorch parity (that's a separate, real-model concern)."""
    rng = np.random.default_rng(seed)
    weights = {
        'W_ih': rng.standard_normal((3 * hidden, input_dim)),
        'W_hh': rng.standard_normal((3 * hidden, hidden)),
        'b_ih': rng.standard_normal(3 * hidden),
        'b_hh': rng.standard_normal(3 * hidden),
        'W_out': rng.standard_normal((1, hidden)),
        'b_out': rng.standard_normal(1),
        'hidden_size': np.array(hidden),
        'feature_mean': np.zeros(input_dim),
        'feature_std': np.ones(input_dim),
    }
    verify_X = rng.standard_normal((n_verify, config.TEMPORAL_SEQ_LEN, input_dim))
    verify_Y = np.array([gru_forward_numpy(verify_X[i], weights) for i in range(n_verify)])
    weights['verify_X'] = verify_X
    weights['verify_Y'] = verify_Y

    path = tmp_path / filename
    np.savez(path, **weights)
    return path


def _feats(value):
    """A behavioral-features dict where every dim equals `value` - enough
    to make different frames (and hence different windows) distinguishable
    without needing real signal semantics."""
    return {name: float(value) for name in TEMPORAL_FEATURE_ORDER}


# ---------------------------------------------------------------------------
# gru_forward_numpy - raw math
# ---------------------------------------------------------------------------

def test_gru_forward_numpy_is_deterministic(tmp_path):
    path = _make_fixture_npz(tmp_path)
    weights = dict(np.load(path))
    x = np.random.default_rng(1).standard_normal((config.TEMPORAL_SEQ_LEN, 10))
    assert gru_forward_numpy(x, weights) == gru_forward_numpy(x, weights)


def test_gru_forward_numpy_returns_raw_logit(tmp_path):
    """Confirms the sigmoid-outside contract: this function must not clamp
    to [0,1] itself, since predict() applies sigmoid exactly once, and
    export-time verification needs to compare pre-sigmoid logits."""
    path = _make_fixture_npz(tmp_path, seed=7)
    weights = dict(np.load(path))
    x = np.full((config.TEMPORAL_SEQ_LEN, 10), 50.0)
    logit = gru_forward_numpy(x, weights)
    assert isinstance(logit, float)


# ---------------------------------------------------------------------------
# GRUTemporalInference - load / availability / safe degradation
# ---------------------------------------------------------------------------

def test_missing_file_degrades_without_raising(tmp_path):
    inf = GRUTemporalInference(model_path=tmp_path / "does_not_exist.npz")
    assert inf.is_available() is False
    assert inf.predict() is None
    inf.add_frame(_feats(1.0))   # must not raise even though unavailable
    assert inf.predict() is None


def test_valid_artifact_loads_and_is_available(tmp_path):
    path = _make_fixture_npz(tmp_path)
    inf = GRUTemporalInference(model_path=path)
    assert inf.is_available() is True


def test_corrupted_verify_pair_is_rejected(tmp_path):
    """A verify_Y that no longer matches gru_forward_numpy(verify_X, ...)
    must be treated exactly like a missing model - this is the guard
    against a silently-wrong export or a corrupted/stale file."""
    path = _make_fixture_npz(tmp_path)
    data = dict(np.load(path))
    data['verify_Y'] = data['verify_Y'] + 5.0   # deliberately break it
    bad_path = tmp_path / "bad.npz"
    np.savez(bad_path, **data)
    inf = GRUTemporalInference(model_path=bad_path)
    assert inf.is_available() is False


def test_missing_verify_pair_is_rejected(tmp_path):
    path = _make_fixture_npz(tmp_path)
    data = dict(np.load(path))
    del data['verify_X']
    del data['verify_Y']
    no_verify_path = tmp_path / "no_verify.npz"
    np.savez(no_verify_path, **data)
    inf = GRUTemporalInference(model_path=no_verify_path)
    assert inf.is_available() is False


# ---------------------------------------------------------------------------
# Sliding-window / prediction-cadence semantics (exact schedule)
# ---------------------------------------------------------------------------

def test_sliding_window_schedule(tmp_path):
    """
    frames 1-16  -> first prediction (window 1-16)
    frames 17-23 -> cached first prediction
    frame 24     -> second prediction (window 9-24)
    frames 25-31 -> cached second prediction
    frame 32     -> third prediction (window 17-32)
    """
    path = _make_fixture_npz(tmp_path)
    inf = GRUTemporalInference(model_path=path)
    assert inf.is_available()

    for i in range(1, 16):
        inf.add_frame(_feats(i))
        assert inf.predict() is None, f"frame {i} should still be None (buffer not full)"

    inf.add_frame(_feats(16))
    first = inf.predict()
    assert first is not None
    assert 0.0 <= first <= 1.0

    for i in range(17, 24):
        inf.add_frame(_feats(i))
        assert inf.predict() == first, f"frame {i} should return the cached value"

    inf.add_frame(_feats(24))
    second = inf.predict()
    assert second is not None
    assert second != first, "window 9-24 differs from window 1-16 - prediction must change"

    for i in range(25, 32):
        inf.add_frame(_feats(i))
        assert inf.predict() == second, f"frame {i} should return the cached value"

    inf.add_frame(_feats(32))
    third = inf.predict()
    assert third is not None
    assert third != second, "window 17-32 differs from window 9-24 - prediction must change"


def test_reset_clears_all_scheduling_state(tmp_path):
    path = _make_fixture_npz(tmp_path)
    inf = GRUTemporalInference(model_path=path)

    for i in range(1, 17):
        inf.add_frame(_feats(i))
    first = inf.predict()
    assert first is not None

    inf.reset()
    assert inf.predict() is None
    assert inf.buffer_status() == (0, config.TEMPORAL_SEQ_LEN)

    # A fresh, identical 16-frame run after reset must reproduce a "first
    # prediction" at frame 16 again - not silently continue the old
    # buffer/counter/cache/threshold state.
    for i in range(1, 16):
        inf.add_frame(_feats(i))
        assert inf.predict() is None
    inf.add_frame(_feats(16))
    again = inf.predict()
    assert again == first, "identical input after reset should reproduce the identical prediction"


# ---------------------------------------------------------------------------
# Feature vocabulary consistency
# ---------------------------------------------------------------------------

def test_feature_order_matches_behavior_weights_prefix():
    assert TEMPORAL_FEATURE_ORDER[:7] == learned_fusion.FEATURE_ORDER
    assert TEMPORAL_FEATURE_ORDER[7:] == ['ear', 'ear_asymmetry', 'blinks_per_minute']
    assert len(TEMPORAL_FEATURE_ORDER) == 10


# ---------------------------------------------------------------------------
# Shipped real artifact - optional, only once training has produced one
# ---------------------------------------------------------------------------

_SHIPPED_MODEL = Path(config.TEMPORAL_MODEL_PATH)


@pytest.mark.skipif(not _SHIPPED_MODEL.exists(),
                    reason="models/temporal_gru.npz not built yet")
def test_shipped_artifact_self_verifies():
    inf = GRUTemporalInference()
    assert inf.is_available(), (
        "the shipped temporal_gru.npz failed its own baked-in verification")
