"""
Canonical evaluation harness
============================
Regenerates EVERY number this project reports, in one run, into one
timestamped file.

Why this exists
---------------
PP1's reported figures drifted until they could not be reproduced: the report
contained both 88.37% and 98.9% for the same configuration, the paper used a
WESAD accuracy (85.56%) that no artifact in the repo produces, and the
"n=202 baseline" folder no longer reproduces any of the three numbers claimed
for it. The root cause was numbers being copied by hand from ad-hoc runs.

Every figure that appears anywhere must come from this script.

What it reports beyond raw accuracy
-----------------------------------
* Wilson 95% confidence intervals. A bare "98.9%" from n=176 invites the
  question "how confident?" - the interval answers it.
* Session-level results alongside frame-level. The frames in a recording are
  heavily autocorrelated (a frame 0.5s after a correct frame is almost
  certainly also correct), so frame-level n overstates the evidence. Session
  counts are the honest denominator; frame counts are the familiar one.
  Both are reported. Neither is hidden.

Usage
-----
    cd code && python run_all_evaluations.py
    python run_all_evaluations.py --out ../evaluation_data/CANONICAL.txt
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Wilson score interval — degrades gracefully if statsmodels is absent.
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


STRESSED_LEVELS = ("MODERATE", "HIGH")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def load_csv(path):
    """Read one exported session CSV into a list of per-frame dicts."""
    rows = []
    with open(path, "r", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "stress_index": float(r["stress_index"]),
                    "level": r["level"],
                    "daisee": float(r.get("daisee", 0.0) or 0.0),
                    "behavior": float(r.get("behavior", 0.0) or 0.0),
                })
            except (KeyError, ValueError):
                continue          # skip malformed rows rather than die
    return rows


def collect(folder, relaxed_glob, stressed_glob):
    """Return (relaxed_files, stressed_files) as [(name, rows), ...]."""
    folder = Path(folder)
    relaxed = [(p.name, load_csv(p)) for p in sorted(folder.glob(relaxed_glob))]
    stressed = [(p.name, load_csv(p)) for p in sorted(folder.glob(stressed_glob))]
    return relaxed, stressed


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def frame_scores(relaxed_files, stressed_files):
    """
    Frame-level accuracy. Ground truth comes from the filename: relaxed
    recordings should read LOW, stressed recordings should read MOD/HIGH.
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


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def pct(x):
    return f"{x * 100:.2f}%"


def ci_str(ci):
    return f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"


def render_config(title, folder, relaxed_glob, stressed_glob, out):
    relaxed_files, stressed_files = collect(folder, relaxed_glob, stressed_glob)
    if not relaxed_files and not stressed_files:
        out.append(f"\n### {title}\n  (no matching CSVs in {folder})")
        return None

    fr = frame_scores(relaxed_files, stressed_files)
    se = session_scores(relaxed_files, stressed_files)

    out.append(f"\n### {title}")
    out.append(f"  source: {folder}  ({relaxed_glob} / {stressed_glob})")
    out.append("")
    out.append("  Per-file:")
    for name, rows in relaxed_files + stressed_files:
        if not rows:
            continue
        levels = Counter(r["level"] for r in rows)
        mean_si = statistics.mean(r["stress_index"] for r in rows)
        out.append(f"    {name:28s} n={len(rows):>4}  mean_index={mean_si:.3f}  "
                   f"L/M/H={levels['LOW']}/{levels['MODERATE']}/{levels['HIGH']}")

    out.append("")
    out.append(f"  FRAME-level   n={fr['n']}")
    out.append(f"    overall accuracy   {pct(fr['accuracy'])}  95% CI {ci_str(fr['ci'])}")
    out.append(f"    relaxed  -> LOW    {fr['relaxed_ok']}/{fr['relaxed_n']} "
               f"= {pct(fr['relaxed_recall'])}  CI {ci_str(fr['relaxed_ci'])}")
    out.append(f"    stressed -> M/H    {fr['stressed_ok']}/{fr['stressed_n']} "
               f"= {pct(fr['stressed_recall'])}  CI {ci_str(fr['stressed_ci'])}")

    out.append("")
    out.append(f"  SESSION-level n={se['n']}   (majority vote per recording)")
    out.append(f"    accuracy           {se['correct']}/{se['n']} "
               f"= {pct(se['accuracy'])}  95% CI {ci_str(se['ci'])}")
    out.append("    NOTE: the wide interval here is the honest one - frames "
               "within a recording")
    out.append("          are autocorrelated, so frame-level n overstates the "
               "evidence.")

    out.append("")
    out.append("  Confusion (frame-level):")
    out.append(f"    {'':10s}{'LOW':>8}{'MODERATE':>10}{'HIGH':>8}")
    for truth in ("relaxed", "stressed"):
        c = fr["confusion"][truth]
        out.append(f"    {truth:10s}{c['LOW']:>8}{c['MODERATE']:>10}{c['HIGH']:>8}")

    if fr["gap"] is not None:
        out.append("")
        out.append(f"  Discrimination gap: {fr['gap']:+.3f}   "
                   f"(relaxed mean {fr['relaxed_mean']:.3f}, "
                   f"stressed mean {fr['stressed_mean']:.3f})")
    return fr


def render_artifacts(root, out):
    """Echo the offline-stage numbers straight from their generated files,
    so nothing is transcribed by hand."""
    out.append("\n" + "=" * 78)
    out.append("OFFLINE STAGE ARTIFACTS (read from generated files, not retyped)")
    out.append("=" * 78)

    cal = root / "models" / "calibrated_thresholds.json"
    if cal.exists():
        with open(cal) as f:
            data = json.load(f)
        loso = data.get("classifier_loso", {})
        meta = data.get("metadata", {})
        out.append("\n### WESAD physiological calibration")
        out.append(f"  source: {cal.relative_to(root)}")
        out.append(f"  LOSO mean accuracy : {loso.get('accuracy_mean', 0) * 100:.2f}%")
        out.append(f"  LOSO mean F1       : {loso.get('f1_mean', 0):.3f}")
        out.append(f"  subjects           : {len(meta.get('subjects_used', []))}")
        out.append(f"  windows            : {meta.get('n_baseline_windows', 0)} baseline"
                   f" / {meta.get('n_stress_windows', 0)} stress")
        cuts = data.get("vision_system_cutoffs", {})
        out.append(f"  JSON cutoffs       : {cuts.get('low_to_moderate', 0):.3f}"
                   f" / {cuts.get('moderate_to_high', 0):.3f}")
        out.append("  WARNING: those JSON cutoffs are NOT what the live system uses.")
        out.append("           stress_fusion.py hardcodes 0.36 / 0.55. See PLAN_FOR_PP2.md.")
    else:
        out.append("\n### WESAD calibration: calibrated_thresholds.json not found")

    lstm = root / "models" / "lstm_test_report.txt"
    if lstm.exists():
        out.append("\n### CNN+LSTM temporal branch (DISABLED in the live system)")
        out.append(f"  source: {lstm.relative_to(root)}")
        for line in lstm.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(k in line for k in ("Test accuracy", "Test set size", "Test loss")):
                out.append(f"  {line.strip()}")
        out.append("  Majority-class floor for DaiSEE (~6.4:1) is 86.6% - the model")
        out.append("  is BELOW it, which is why W_LSTM = 0.0.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="..",
                    help="project root (default: .. when run from code/)")
    ap.add_argument("--out", default=None,
                    help="output file (default: <root>/evaluation_data/"
                         "CANONICAL_RESULTS_<date>.txt)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    eval_dir = root / "evaluation_data"

    out = []
    out.append("=" * 78)
    out.append("CANONICAL EVALUATION RESULTS")
    out.append("=" * 78)
    out.append(f"generated : {datetime.now().isoformat(timespec='seconds')}")
    out.append(f"root      : {root}")
    out.append("")
    out.append("Every number this project reports must come from this file.")
    out.append("Regenerate with:  cd code && python run_all_evaluations.py")

    out.append("\n" + "=" * 78)
    out.append("LIVE SYSTEM — END-TO-END EVALUATION")
    out.append("=" * 78)

    render_config("Default mode (current evaluation_data/)",
                  eval_dir, "relaxed_*.csv", "stressed_*.csv", out)
    render_config("Studying mode (baseline_v1/study*)",
                  eval_dir / "baseline_v1",
                  "studyrelaxed_*.csv", "studystressed_*.csv", out)
    render_config("Archived baseline_v1 non-study recordings",
                  eval_dir / "baseline_v1",
                  "relaxed_*.csv", "stressed_*.csv", out)

    render_artifacts(root, out)

    out.append("\n" + "=" * 78)
    out.append("READING THESE NUMBERS")
    out.append("=" * 78)
    out.append("* Frame-level accuracy is the familiar figure but its n is inflated:")
    out.append("  frames within one recording are highly autocorrelated.")
    out.append("* Session-level accuracy uses the honest denominator (recordings),")
    out.append("  which is why its confidence interval is much wider.")
    out.append("* All live recordings to date are SINGLE-SUBJECT. These measure")
    out.append("  within-subject behaviour, not cross-user generalisation.")
    out.append("  Multi-subject evaluation is Phase 1 of PLAN_FOR_PP2.md.")

    text = "\n".join(out)
    print(text)

    out_path = (Path(args.out) if args.out else
                eval_dir / f"CANONICAL_RESULTS_{datetime.now():%Y-%m-%d}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n\nwritten to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
