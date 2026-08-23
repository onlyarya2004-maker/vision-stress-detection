"""
daisee_ensemble.py tests.

Model loading and tf.function tracing need real .h5 checkpoints and
TensorFlow warm-up (slow, and the checkpoints aren't part of the repo -
see README.md's dataset/model section), so build_pairs()/_make_infer_fn()
aren't covered directly here. ensemble_predict()/prep() are pure enough to
test with fake infer_fn callables standing in for a loaded model, which is
what actually matters: that the averaging, the None-skipping, and the
input shape contract are correct.
"""

import numpy as np
import pytest

from daisee_ensemble import ensemble_predict, prep


class _FakeTensor:
    """Stands in for a tf.function's return value - just needs .numpy()."""
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


def _fake_infer(probs):
    """Returns a callable shaped like a tf.function: takes the prepped
    (1, size, size, 3) batch, ignores it, returns a fixed (1, 2) softmax."""
    def infer(x):
        assert x.shape[0] == 1, "ensemble_predict must call infer with a batch of 1"
        return _FakeTensor(np.array([probs]))
    return infer


def _identity_preprocess(img):
    return img


def test_prep_produces_a_batch_of_one():
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    x = prep(face, size=160, preprocess_fn=_identity_preprocess)
    assert x.shape == (1, 160, 160, 3)


def test_ensemble_predict_averages_both_models():
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    pairs = [
        (_fake_infer([0.2, 0.8]), 160, _identity_preprocess),
        (_fake_infer([0.4, 0.6]), 224, _identity_preprocess),
    ]
    result = ensemble_predict(face, pairs)
    assert result == pytest.approx([0.3, 0.7])


def test_ensemble_predict_skips_none_infer_fn():
    """A missing/failed model (infer_fn=None) must be skipped, not crash -
    matches the live degrade path where fusion redistributes weight."""
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    pairs = [
        (None, 160, _identity_preprocess),
        (_fake_infer([0.1, 0.9]), 224, _identity_preprocess),
    ]
    result = ensemble_predict(face, pairs)
    assert result == pytest.approx([0.1, 0.9])


def test_ensemble_predict_returns_none_when_every_model_missing():
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    pairs = [(None, 160, _identity_preprocess), (None, 224, _identity_preprocess)]
    assert ensemble_predict(face, pairs) is None


def test_ensemble_predict_empty_pairs_returns_none():
    face = np.zeros((300, 300, 3), dtype=np.uint8)
    assert ensemble_predict(face, []) is None
