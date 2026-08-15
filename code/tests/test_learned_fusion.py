"""
Learned-fusion integration tests.

The critical property is that this changes NOTHING unless deliberately
switched on. config.USE_LEARNED_FUSION is False by default because the
shipped model was trained entirely on UBFC-Phys - a different camera, cohort
and stress protocol - and enabling it for a laptop webcam would repeat the
very error this investigation uncovered.
"""

import numpy as np
import pytest

import config
import learned_fusion
import stress_fusion
from learned_fusion import FEATURE_ORDER, LearnedFusion


FEATURES = {'eye_strain': 0.3, 'blink_stress': 0.2, 'brow_tension': 0.5,
            'jaw_tension': 0.4, 'head_instability': 0.6,
            'neck_strain': 0.1, 'slouch': 0.2}


@pytest.fixture(autouse=True)
def _reset_singleton():
    learned_fusion.reset()
    yield
    learned_fusion.reset()


# ---------------------------------------------------------------------------
# Default-off guarantee
# ---------------------------------------------------------------------------

def test_learned_fusion_is_off_by_default():
    """Shipping it enabled would deploy a model fitted on someone else's
    camera, cohort and protocol."""
    assert config.USE_LEARNED_FUSION is False


def test_disabled_flag_leaves_scoring_untouched(monkeypatch):
    """With the flag off, output must equal the hand-tuned computation."""
    monkeypatch.setattr(config, 'USE_LEARNED_FUSION', False)
    score, contribs = stress_fusion.behavioral_stress_score(FEATURES)

    w = config.BEHAVIOR_WEIGHTS
    expected = sum(FEATURES[k] * w[k] for k in w) / sum(w.values())
    expected = min(1.0, expected * config.BEHAVIOR_AMPLIFICATION)
    assert score == pytest.approx(expected)
    assert set(contribs) == set(w)


def test_missing_model_degrades_to_handtuned(monkeypatch, tmp_path):
    """Enabled but with no model file: must fall back, not crash."""
    monkeypatch.setattr(config, 'USE_LEARNED_FUSION', True)
    monkeypatch.setattr(config, 'LEARNED_FUSION_MODEL_PATH',
                        str(tmp_path / 'nope.joblib'))
    learned_fusion.reset()

    score, _ = stress_fusion.behavioral_stress_score(FEATURES)
    monkeypatch.setattr(config, 'USE_LEARNED_FUSION', False)
    baseline, _ = stress_fusion.behavioral_stress_score(FEATURES)
    assert score == pytest.approx(baseline)


def test_unavailable_model_reports_itself():
    lf = LearnedFusion(model_path='definitely/not/here.joblib')
    assert lf.is_available() is False
    assert lf.score(FEATURES) is None
    assert lf.weights() is None


# ---------------------------------------------------------------------------
# Behaviour when a model IS present
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_model(tmp_path):
    """
    Train a tiny model where head_instability alone determines the label -
    the shape the real data showed, with that signal dominating.
    """
    joblib = pytest.importorskip('joblib')
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 300
    X = rng.random((n, len(FEATURE_ORDER)))
    y = (X[:, FEATURE_ORDER.index('head_instability')] > 0.5).astype(int)
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=1000).fit(sc.transform(X), y)

    path = tmp_path / 'meta.joblib'
    joblib.dump({'model': m, 'scaler': sc, 'features': list(FEATURE_ORDER)},
                path)
    return path


def test_loaded_model_scores_in_range(fitted_model):
    lf = LearnedFusion(model_path=str(fitted_model))
    assert lf.is_available() is True
    s = lf.score(FEATURES)
    assert s is not None and 0.0 <= s <= 1.0


def test_learned_score_tracks_the_signal_it_learned(fitted_model):
    """Raising the signal the model relies on must raise the score."""
    lf = LearnedFusion(model_path=str(fitted_model))
    low = lf.score(dict(FEATURES, head_instability=0.05))
    high = lf.score(dict(FEATURES, head_instability=0.95))
    assert high > low


def test_enabled_model_is_actually_used(monkeypatch, fitted_model):
    """With the flag on and a model present, output must diverge from the
    hand-tuned path - otherwise the wiring is inert."""
    monkeypatch.setattr(config, 'USE_LEARNED_FUSION', False)
    handtuned, _ = stress_fusion.behavioral_stress_score(FEATURES)

    monkeypatch.setattr(config, 'USE_LEARNED_FUSION', True)
    monkeypatch.setattr(config, 'LEARNED_FUSION_MODEL_PATH',
                        str(fitted_model))
    learned_fusion.reset()
    learned, contribs = stress_fusion.behavioral_stress_score(FEATURES)

    assert learned != pytest.approx(handtuned)
    # Contributions must still be the raw signals, so the UI is unaffected.
    assert set(contribs) == set(config.BEHAVIOR_WEIGHTS)
    assert contribs['brow_tension'] == pytest.approx(FEATURES['brow_tension'])


def test_weights_are_readable_for_reporting(fitted_model):
    """Interpretability is the point: coefficients must be inspectable."""
    lf = LearnedFusion(model_path=str(fitted_model))
    w = lf.weights()
    assert set(w) == set(FEATURE_ORDER)
    assert abs(w['head_instability']) > abs(w['eye_strain'])


def test_feature_order_is_pinned():
    """
    Order is part of the model contract. A silent reordering would feed
    values into the wrong coefficients and produce plausible nonsense.
    """
    assert FEATURE_ORDER == ['eye_strain', 'blink_stress', 'brow_tension',
                             'jaw_tension', 'head_instability',
                             'neck_strain', 'slouch']
    assert set(FEATURE_ORDER) == set(config.BEHAVIOR_WEIGHTS)
