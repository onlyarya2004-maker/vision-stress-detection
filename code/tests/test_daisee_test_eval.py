"""
daisee_test_eval.py tests.

evaluate() is pure (no TensorFlow/file I/O), so it's tested directly with
synthetic probability arrays and a hand-known answer. Model loading /
batched inference need real .h5 checkpoints and the DaiSEE dataset, neither
of which is part of the repo (see README.md's dataset section) - not
covered here, same boundary the GRU/LSTM training scripts already have.
"""

import numpy as np
import pytest

from daisee_test_eval import evaluate, render_result


def _probs(*stressed_probs):
    """One row per input: [P(not_stressed), P(stressed)]."""
    return np.array([[1.0 - p, p] for p in stressed_probs])


def test_evaluate_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    probs = _probs(0.1, 0.2, 0.9, 0.8)   # all correctly on the >=0.5 side
    result = evaluate(y_true, probs, 'test')
    assert result['accuracy'] == pytest.approx(1.0)
    assert result['balanced_accuracy'] == pytest.approx(1.0)
    assert result['precision'] == pytest.approx(1.0)
    assert result['recall'] == pytest.approx(1.0)
    assert result['f1'] == pytest.approx(1.0)


def test_evaluate_confusion_matrix_orientation():
    """rows=true, cols=pred, [not_stressed, stressed] order - a swapped
    orientation would silently invert every downstream reading."""
    y_true = np.array([0, 0, 1, 1])
    probs = _probs(0.1, 0.9, 0.1, 0.9)   # one false positive, one false negative
    result = evaluate(y_true, probs, 'test')
    cm = result['confusion_matrix']
    # true=0: 1 correctly LOW (row0,col0), 1 wrongly HIGH (row0,col1)
    assert cm[0].tolist() == [1, 1]
    # true=1: 1 wrongly LOW (row1,col0), 1 correctly HIGH (row1,col1)
    assert cm[1].tolist() == [1, 1]


def test_evaluate_majority_floor_matches_actual_class_counts():
    """The floor must come from THIS call's y_true, never a hardcoded or
    stale number - exactly the discipline the GRU/LSTM reports follow."""
    y_true = np.array([0] * 90 + [1] * 10)
    probs = _probs(*([0.05] * 90 + [0.5] * 10))  # everything predicted LOW
    result = evaluate(y_true, probs, 'test')
    assert result['majority_floor'] == pytest.approx(0.90)
    assert result['accuracy'] == pytest.approx(0.90)   # matches the floor exactly
    assert result['above_floor'] == False   # tie does not count as "above"


def test_evaluate_above_floor_verdict_direction():
    y_true = np.array([0] * 80 + [1] * 20)
    # Recovers most of the stressed class -> clears the 80% floor.
    probs = _probs(*([0.1] * 80 + [0.9] * 20))
    result = evaluate(y_true, probs, 'test')
    assert result['accuracy'] > result['majority_floor']
    assert result['above_floor'] == True


def test_evaluate_n_matches_input_length():
    y_true = np.array([0, 1, 0, 1, 0])
    probs = _probs(0.1, 0.9, 0.2, 0.8, 0.3)
    result = evaluate(y_true, probs, 'my-label')
    assert result['n'] == 5
    assert result['label'] == 'my-label'


def test_render_result_reports_all_key_fields():
    y_true = np.array([0, 0, 1, 1])
    probs = _probs(0.1, 0.2, 0.9, 0.8)
    result = evaluate(y_true, probs, 'MobileNetV2 (alone)')
    lines = []
    render_result(lambda s="": lines.append(s), result)
    text = "\n".join(lines)
    assert 'MobileNetV2 (alone)' in text
    assert 'accuracy' in text
    assert 'balanced accuracy' in text
    assert 'majority-class floor' in text
    assert 'ABOVE the majority-class floor' in text
    text.encode('ascii')  # plain-ASCII-only, same rule as the other reports
