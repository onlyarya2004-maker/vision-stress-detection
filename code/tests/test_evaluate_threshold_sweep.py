"""
evaluate_threshold_sweep.py tests.

Real model loading + real DaiSEE images excluded (matches
test_daisee_ensemble.py's established boundary) - evaluate_at_thresholds()
itself is pure metric computation over whatever probabilities/labels it's
given, so it's tested here against a stub model/test_data pair with
hand-known probabilities, never touching TensorFlow model loading. This
had zero coverage before, despite the threshold-selection logic being
real, non-trivial branching (max by macro-F1) - the fact that its JSON
output is currently unread elsewhere (BUG_AUDIT_2026-08-23.md) doesn't
make the computation itself exempt from being correct.
"""

import numpy as np
import pytest

from evaluate_threshold_sweep import evaluate_at_thresholds


class _StubModel:
    """Mimics model.predict()'s (N, 2) softmax-shaped output without
    loading anything - column 1 is P(stressed)."""
    def __init__(self, p_stressed):
        self._p = np.asarray(p_stressed)

    def predict(self, test_data, verbose=0):
        return np.stack([1 - self._p, self._p], axis=1)


class _StubTestData:
    def __init__(self, classes):
        self.classes = np.asarray(classes)


def test_sweep_at_a_threshold_that_perfectly_separates_classes():
    """4 samples: 2 true not_stressed with low P(stressed), 2 true
    stressed with high P(stressed) - at threshold 0.5, every prediction
    should be exactly correct (accuracy = 1.0, stressed_recall = 1.0)."""
    p_stressed = [0.1, 0.2, 0.8, 0.9]
    y_true =      [0,    0,   1,   1]
    model = _StubModel(p_stressed)
    data = _StubTestData(y_true)

    results, best = evaluate_at_thresholds(model, data, thresholds=[0.5])
    r = results[0]
    assert r['accuracy'] == pytest.approx(1.0)
    assert r['stressed_recall'] == pytest.approx(1.0)
    assert r['stressed_precision'] == pytest.approx(1.0)
    assert r['macro_f1'] == pytest.approx(1.0)


def test_higher_threshold_reduces_stressed_recall_hand_checked():
    """3 true-stressed samples with P(stressed) = 0.3, 0.6, 0.9. At
    threshold 0.5, 2/3 are caught (recall=0.667); at threshold 0.8, only
    1/3 (recall=0.333) - a strictly monotonic, hand-computable check that
    raising the threshold can only ever miss more of the positive class,
    never catch more."""
    p_stressed = [0.3, 0.6, 0.9]
    y_true = [1, 1, 1]
    model = _StubModel(p_stressed)
    data = _StubTestData(y_true)

    results, _ = evaluate_at_thresholds(model, data, thresholds=[0.5, 0.8])
    by_th = {r['threshold']: r for r in results}
    assert by_th[0.5]['stressed_recall'] == pytest.approx(2 / 3)
    assert by_th[0.8]['stressed_recall'] == pytest.approx(1 / 3)


def test_best_threshold_selection_picks_highest_macro_f1():
    """Construct probabilities where threshold 0.5 perfectly separates
    the classes (macro-F1 = 1.0) but 0.2 does not (everything predicted
    stressed) - best must be 0.5, not just the first or last threshold
    tried, proving the max()-by-macro_f1 selection actually discriminates
    rather than defaulting to list order."""
    p_stressed = [0.1, 0.2, 0.8, 0.9]
    y_true =      [0,    0,   1,   1]
    model = _StubModel(p_stressed)
    data = _StubTestData(y_true)

    results, best = evaluate_at_thresholds(model, data, thresholds=[0.05, 0.5, 0.95])
    assert best['threshold'] == 0.5
    assert best['macro_f1'] == pytest.approx(1.0)


def test_no_predicted_positives_gives_zero_precision_not_a_crash():
    """Threshold above every probability -> zero_division=0 must produce
    0.0, not raise or emit a warning-driven NaN."""
    p_stressed = [0.1, 0.2, 0.3]
    y_true = [1, 1, 0]
    model = _StubModel(p_stressed)
    data = _StubTestData(y_true)

    results, _ = evaluate_at_thresholds(model, data, thresholds=[0.99])
    r = results[0]
    assert r['stressed_precision'] == 0.0
    assert r['stressed_recall'] == 0.0
