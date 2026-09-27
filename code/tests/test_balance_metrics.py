"""balance_metrics(): hand-checked against counts worked out on paper."""
import pytest
from eval_scoring import balance_metrics


def test_perfect_classifier():
    m = balance_metrics(relaxed_n=10, relaxed_ok=10, stressed_n=30, stressed_ok=30)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)


def test_majority_only_classifier_is_exposed():
    # Everything predicted "stressed" on 90% stressed data: raw accuracy 90%,
    # but balanced accuracy 50% - the case this metric exists to catch.
    m = balance_metrics(relaxed_n=10, relaxed_ok=0, stressed_n=90, stressed_ok=90)
    assert m["majority_baseline"] == pytest.approx(0.90)
    assert m["majority_class"] == "stressed"
    assert m["balanced_accuracy"] == pytest.approx(0.50)


def test_hand_checked_mixed_case():
    # relaxed: 8/10 read LOW (TN=8, FP=2); stressed: 15/20 read M/H (TP=15, FN=5)
    m = balance_metrics(10, 8, 20, 15)
    assert m["balanced_accuracy"] == pytest.approx((0.8 + 0.75) / 2)
    assert m["stressed_precision"] == pytest.approx(15 / 17)
    f1_s = 2 * 15 / (2 * 15 + 2 + 5)
    f1_r = 2 * 8 / (2 * 8 + 5 + 2)
    assert m["macro_f1"] == pytest.approx((f1_s + f1_r) / 2)


def test_returns_none_when_a_class_is_missing():
    assert balance_metrics(0, 0, 10, 5) is None
    assert balance_metrics(10, 5, 0, 0) is None
    assert balance_metrics(0, 0, 0, 0) is None
