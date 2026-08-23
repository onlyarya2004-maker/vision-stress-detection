"""
Tests for the warm-start personalization prototype's core math.

fit_logreg_with_prior() is a MAP logistic regression with a Gaussian prior
centered at (w_prior, b_prior). Two properties must hold for the offline
validation's numbers to mean anything:

  1. lambda=0 must reduce to plain unregularized MLE (the prior should only
     matter when lam > 0 - a bug here would silently pull ALL conditions
     toward the prior, including the "personal_only" baseline it's compared
     against).
  2. Very large lambda must collapse to the prior regardless of what the
     data says (the "ignore personal data entirely" limit) - this is what
     lets the cross-task condition's warm_started[large lambda] correctly
     approach population_only.
"""

import numpy as np
import pytest

from prototype_warmstart_personalization import fit_logreg_with_prior, _predict


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    true_w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    p = 1 / (1 + np.exp(-(X @ true_w)))
    y = (p > rng.uniform(size=n)).astype(int)
    return X, y


def test_zero_lambda_matches_unregularized_sklearn():
    from sklearn.linear_model import LogisticRegression
    X, y = _synthetic()
    sk = LogisticRegression(C=1e12, max_iter=5000).fit(X, y)   # ~unregularized
    w, b = fit_logreg_with_prior(X, y, np.zeros(5), 0.0, lam=0.0)
    assert np.abs(w - sk.coef_[0]).max() < 1e-2
    assert abs(b - sk.intercept_[0]) < 1e-2


def test_huge_lambda_collapses_to_the_prior():
    X, y = _synthetic()
    w_prior = np.array([9.0, -9.0, 9.0, 9.0, -9.0])
    b_prior = 3.3
    w, b = fit_logreg_with_prior(X, y, w_prior, b_prior, lam=1e6)
    assert np.allclose(w, w_prior, atol=1e-2)
    assert np.isclose(b, b_prior, atol=1e-2)


def test_increasing_lambda_monotonically_approaches_prior():
    """Sanity check on the shape of the regularization path, not just its
    two extremes - a bug that only manifested at intermediate lambda
    (e.g. a sign error) wouldn't be caught by the two tests above alone."""
    X, y = _synthetic()
    w_prior = np.array([5.0, -5.0, 5.0, 5.0, -5.0])
    b_prior = 1.0
    dists = []
    for lam in (0.0, 1.0, 10.0, 100.0, 10000.0):
        w, b = fit_logreg_with_prior(X, y, w_prior, b_prior, lam)
        dists.append(np.linalg.norm(w - w_prior))
    assert all(d1 >= d2 - 1e-6 for d1, d2 in zip(dists, dists[1:])), (
        f"distance to prior did not shrink monotonically: {dists}")


def test_predict_matches_manual_sigmoid_threshold():
    X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    w = np.array([2.0, 0.0])
    b = 0.0
    # z = [2.0, -2.0, 0.0] -> sigmoid >= 0.5 iff z >= 0
    preds = _predict(X, w, b)
    assert list(preds) == [1, 0, 1]
