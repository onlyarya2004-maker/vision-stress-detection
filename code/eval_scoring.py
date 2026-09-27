"""
Shared scoring utilities for the evaluation harness
====================================================
Wilson confidence intervals, frame-level and session-level accuracy scoring.
Split out of run_all_evaluations.py so ablation_study.py can reuse the exact
same scoring code (not a reimplementation) without the two modules importing
each other.
"""

import statistics
from collections import Counter

STRESSED_LEVELS = ("MODERATE", "HIGH")

# ---------------------------------------------------------------------------
# Wilson score interval - degrades gracefully if statsmodels is absent.
# ---------------------------------------------------------------------------
try:
    from statsmodels.stats.proportion import proportion_confint

    def wilson(successes, n):
        if n == 0:
            return (0.0, 0.0)
        return proportion_confint(successes, n, alpha=0.05, method="wilson")
except ImportError:                                   # pragma: no cover
    import math

    def wilson(successes, n):
        """Closed-form Wilson interval, so the harness never hard-depends
        on statsmodels."""
        if n == 0:
            return (0.0, 0.0)
        z = 1.959963984540054                          # 95%
        p = successes / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
        return (max(0.0, centre - margin), min(1.0, centre + margin))


def pct(x):
    return f"{x * 100:.2f}%"


def ci_str(ci):
    return f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def frame_scores(relaxed_files, stressed_files):
    """
    Frame-level accuracy. relaxed_files/stressed_files: [(name, rows), ...]
    where each row is a dict with at least 'level' and 'stress_index'.
    Ground truth: relaxed recordings should read LOW, stressed recordings
    should read MOD/HIGH.
    """
    relaxed = [r for _, rows in relaxed_files for r in rows]
    stressed = [r for _, rows in stressed_files for r in rows]

    r_ok = sum(1 for r in relaxed if r["level"] == "LOW")
    s_ok = sum(1 for r in stressed if r["level"] in STRESSED_LEVELS)
    total = len(relaxed) + len(stressed)
    correct = r_ok + s_ok

    confusion = {
        "relaxed": Counter(r["level"] for r in relaxed),
        "stressed": Counter(r["level"] for r in stressed),
    }
    gap = None
    if relaxed and stressed:
        gap = (statistics.mean(r["stress_index"] for r in stressed)
               - statistics.mean(r["stress_index"] for r in relaxed))

    return {
        "n": total, "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "ci": wilson(correct, total),
        "relaxed_n": len(relaxed), "relaxed_ok": r_ok,
        "relaxed_recall": r_ok / len(relaxed) if relaxed else 0.0,
        "relaxed_ci": wilson(r_ok, len(relaxed)),
        "stressed_n": len(stressed), "stressed_ok": s_ok,
        "stressed_recall": s_ok / len(stressed) if stressed else 0.0,
        "stressed_ci": wilson(s_ok, len(stressed)),
        "confusion": confusion,
        "gap": gap,
        "relaxed_mean": statistics.mean(
            r["stress_index"] for r in relaxed) if relaxed else None,
        "stressed_mean": statistics.mean(
            r["stress_index"] for r in stressed) if stressed else None,
    }


def balance_metrics(relaxed_n, relaxed_ok, stressed_n, stressed_ok):
    """
    Class-balance-aware companions to raw accuracy, from the same four counts
    frame_scores()/session_scores() already produce: relaxed_ok = relaxed
    items read LOW (true negatives), stressed_ok = stressed items read
    MODERATE/HIGH (true positives).

    Why: raw accuracy on imbalanced data can look fine while the system is
    only good at the bigger class. Reports balanced accuracy (mean of the two
    per-class recalls), macro-F1, and the accuracy a trivial "always predict
    the majority class" classifier would score on the same data, so every
    accuracy figure has its honest floor printed next to it.
    """
    n = relaxed_n + stressed_n
    if n == 0 or relaxed_n == 0 or stressed_n == 0:
        return None
    tp, fn = stressed_ok, stressed_n - stressed_ok
    tn, fp = relaxed_ok, relaxed_n - relaxed_ok
    f1_stressed = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_relaxed = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else 0.0
    return {
        "balanced_accuracy": (tp / stressed_n + tn / relaxed_n) / 2,
        "macro_f1": (f1_stressed + f1_relaxed) / 2,
        "stressed_precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "majority_baseline": max(relaxed_n, stressed_n) / n,
        "majority_class": "stressed" if stressed_n >= relaxed_n else "relaxed",
    }


def session_scores(relaxed_files, stressed_files):
    """
    Session-level accuracy: each recording gets ONE verdict by majority vote
    over its frames. This is the honest denominator - 176 frames from 3
    recordings is 3 independent observations, not 176.
    """
    results = []
    for name, rows in relaxed_files:
        if not rows:
            continue
        low = sum(1 for r in rows if r["level"] == "LOW")
        results.append((name, "relaxed", low > len(rows) / 2))
    for name, rows in stressed_files:
        if not rows:
            continue
        hi = sum(1 for r in rows if r["level"] in STRESSED_LEVELS)
        results.append((name, "stressed", hi > len(rows) / 2))

    correct = sum(1 for _, _, ok in results if ok)
    n = len(results)
    return {
        "n": n, "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "ci": wilson(correct, n),
        "detail": results,
    }
