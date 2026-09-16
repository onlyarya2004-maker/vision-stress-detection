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

from app_naming import parse_filename
from eval_scoring import wilson, pct, ci_str, frame_scores, session_scores, STRESSED_LEVELS
# ablation_study imports eval_scoring too, not this module - importing it
# here (rather than the other way around) avoids a circular import while
# still keeping the scoring code itself in exactly one place.
from ablation_study import run_ablation_study, render as render_ablation_table
from correlate_physiology import run_correlation_study, render as render_correlation
from correlate_daisee_selfreport import (
    run_daisee_selfreport_study, render as render_daisee_selfreport,
)
from train_meta_learner import run_learned_fusion_study, render as render_learned_fusion


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


def collect(folder, relaxed_glob, stressed_glob, mode=None):
    """
    Return (relaxed_files, stressed_files) as [(name, rows), ...].

    `mode`, when given, filters to files whose canonical name (see
    app_naming.export_filename) actually encodes that task mode - e.g.
    mode="default" excludes a "..._studying_..." file even though it still
    matches relaxed_*/stressed_* (condition is mode-blind by design, so the
    glob alone cannot tell Default and Studying data apart). Files that
    don't parse as canonical names are handled by matches_mode() below in
    one of two different ways, depending on WHY parsing failed - see there.

    Without this filter, any Studying/Hospital/Driving-mode recording
    dropped into evaluation_data/ per RECORDING_PROTOCOL.md would silently
    count as Default-mode ground truth here - the exact "which files were
    really in this number" corruption this project's audit was written to
    stop happening again.
    """
    folder = Path(folder)

    def matches_mode(path):
        if mode is None:
            return True
        meta = parse_filename(path.name)
        if meta:
            return meta['mode'] == mode
        # parse_filename() returned None. That happens for two structurally
        # different reasons, and only one of them may silently count as
        # Default:
        #   - a genuinely legacy 2-part name (relaxed_1.csv) - predates
        #     per-mode recording, was always Default-mode data, and must
        #     count there (but NOWHERE ELSE, or every mode section would
        #     show identical legacy data).
        #   - a modern, canonical-LOOKING name with one corrupted field
        #     (e.g. a hand-edited subject ID with a stray underscore, per
        #     RECORDING_PROTOCOL.md's anticipated hand-renaming - see
        #     test_export_naming.py's relaxed_S_01_... case) - this is NOT
        #     legacy data and must NOT be silently folded into Default. It
        #     is excluded from every mode's count here; main()'s
        #     unclaimed-files check then reports it loudly instead of
        #     letting it vanish into whichever section happens to run
        #     with mode="default".
        # >=4 underscore-separated fields is the same "looks canonical"
        # signal parse_filename() itself keys off (its own len(parts) < 4
        # is the true-legacy escape hatch) - reused here rather than
        # re-derived, so the two functions can't drift out of sync on what
        # "looks modern" means.
        stem = path.name[:-4] if path.name.lower().endswith('.csv') else path.name
        is_true_legacy = len(stem.split('_')) < 4
        return is_true_legacy and mode == 'default'

    relaxed = [(p.name, load_csv(p)) for p in sorted(folder.glob(relaxed_glob))
               if matches_mode(p)]
    stressed = [(p.name, load_csv(p)) for p in sorted(folder.glob(stressed_glob))
                if matches_mode(p)]
    return relaxed, stressed


def find_unclaimed_files(folder, relaxed_glob, stressed_glob, modes):
    """
    Cross-check that every relaxed_*/stressed_* CSV in `folder` was claimed
    by exactly one of `modes`'s collect() calls. Returns a sorted list of
    filenames that were claimed by none of them.

    Guards against a repeat of this project's founding bug
    (RECORDING_PROTOCOL.md's relaaxed_2.csv incident - 48 valid PP1 samples
    silently excluded from every reported number for months because no
    glob matched the typo). A file can go unclaimed either by failing every
    glob outright (an old-style typo) or by looking canonical-but-corrupted
    (see collect()'s matches_mode() above) - either way it must be
    surfaced, not silently absent from every section's numbers.
    """
    folder = Path(folder)
    all_files = ({p.name for p in folder.glob(relaxed_glob)} |
                {p.name for p in folder.glob(stressed_glob)})

    claimed = set()
    for mode in modes:
        relaxed, stressed = collect(folder, relaxed_glob, stressed_glob, mode=mode)
        claimed.update(name for name, _ in relaxed + stressed)

    return sorted(all_files - claimed)


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
# frame_scores/session_scores/wilson/pct/ci_str/STRESSED_LEVELS now live in
# eval_scoring.py (imported above) - ablation_study.py needs the exact same
# scoring code, and splitting it out was cleaner than a circular import.


def render_config(title, folder, relaxed_glob, stressed_glob, out, mode=None):
    relaxed_files, stressed_files = collect(folder, relaxed_glob, stressed_glob, mode=mode)
    if not relaxed_files and not stressed_files:
        out.append(f"\n### {title}\n  (no matching CSVs in {folder})")
        return None, 0

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

    n_subjects = render_per_subject(relaxed_files, stressed_files, out)
    return fr, n_subjects


def render_per_subject(relaxed_files, stressed_files, out):
    """
    Break results down per subject, when filenames follow the canonical
    scheme (see RECORDING_PROTOCOL.md). Silent for PP1's single-subject
    data; becomes the headline table once Phase 1 multi-subject recordings
    land, which is the gap the papers admit to. relaxed_files/stressed_files
    are already mode-filtered by the caller's collect(mode=...), so no
    further mode handling is needed here.
    """
    by_subject = {}
    for name, rows in relaxed_files + stressed_files:
        meta = parse_filename(name)
        if not meta or not rows:
            continue
        entry = by_subject.setdefault(meta["subject"], {"relaxed": [], "stressed": []})
        entry[meta["condition"]].append((name, rows))

    if len(by_subject) < 2:
        return len(by_subject)          # nothing to compare against

    out.append("")
    out.append("  PER-SUBJECT breakdown:")
    out.append(f"    {'subject':10s}{'n':>6}{'accuracy':>11}{'95% CI':>20}")
    for subject in sorted(by_subject):
        e = by_subject[subject]
        fr = frame_scores(e["relaxed"], e["stressed"])
        if fr["n"] == 0:
            continue
        out.append(f"    {subject:10s}{fr['n']:>6}{pct(fr['accuracy']):>11}"
                   f"{ci_str(fr['ci']):>20}")
    out.append("    NOTE: cross-subject spread is the real generalisation test.")
    return len(by_subject)


def render_artifacts(root, out):
    """Echo the offline-stage numbers straight from their generated files,
    so nothing is transcribed by hand."""
    out.append("\n" + "=" * 78)
    out.append("OFFLINE STAGE ARTIFACTS (read from generated files, not retyped)")
    out.append("=" * 78)

    daisee = root / "models" / "daisee_test_report.txt"
    if daisee.exists():
        out.append("\n### DaiSEE CNN ensemble (held-out test split)")
        out.append(f"  source: {daisee.relative_to(root)}")
        for line in daisee.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped in ("MobileNetV2 (alone)", "EfficientNetB0 (alone)",
                            "Ensemble (mean of both - what the live system uses)"):
                out.append("")
                out.append(f"  {stripped}")
            elif stripped and " : " in stripped:
                out.append(f"  {stripped}")
        out.append("")
        out.append("  Regenerate with: python daisee_test_eval.py (loads the saved .h5")
        out.append("  checkpoints, runs inference only - takes a few minutes, not retrained")
        out.append("  here, so this section is NOT called from this script directly).")

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
        out.append("           config.py sets 0.36 / 0.55. See PLAN_FOR_PP2.md.")
    else:
        out.append("\n### WESAD calibration: calibrated_thresholds.json not found")

    lstm = root / "models" / "lstm_test_report.txt"
    if lstm.exists():
        out.append("\n### CNN+LSTM pixel temporal branch (RETIRED - code/_archive/lstm_temporal.py)")
        out.append(f"  source: {lstm.relative_to(root)}")
        for line in lstm.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(k in line for k in ("Test accuracy", "Test set size", "Test loss")):
                out.append(f"  {line.strip()}")
        out.append("  Majority-class floor for DaiSEE (~6.4:1) is 86.6% - the model")
        out.append("  was BELOW it. Replaced in Phase 2.1 by a feature-sequence GRU -")
        out.append("  see the next section.")

    gru = root / "models" / "temporal_gru_test_report.txt"
    if gru.exists():
        out.append("\n### Feature-sequence GRU temporal branch (Phase 2.1, DISABLED via W_TEMPORAL=0.0)")
        out.append(f"  source: {gru.relative_to(root)}")
        for line in gru.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            # "key : value" metric lines and the model-name section headers -
            # skip separators, blank lines, and the multi-line prose blocks
            # (limitation note, comparison footer) that don't fit this
            # summary echo. The GRU and GBT blocks share identical field
            # labels (n, accuracy, balanced accuracy, ...), so a blank line
            # is inserted before each header - without it the two blocks run
            # together with only the header text itself as a boundary,
            # inviting misattribution on a skim.
            if stripped in ("GRU (feature-sequence temporal model)",
                            "GBT baseline (mean/std/slope summary stats)"):
                out.append("")
                out.append(f"  {stripped}")
            elif stripped and " : " in stripped:
                out.append(f"  {stripped}")
            elif stripped.startswith("best val balanced accuracy during training:"):
                # Formatted without a space before its colon (unlike every
                # other "label : value" line above), so it doesn't match the
                # " : " check - this is the one honesty-critical metric the
                # project cares most about (it's what training selects the
                # checkpoint on, precisely to avoid the old LSTM's hidden
                # recall-collapse failure mode), so it gets an explicit case
                # rather than being silently dropped.
                out.append(f"  {stripped}")


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

    # mode="default" matters here, not just as documentation: relaxed_*/
    # stressed_* is condition-only and mode-blind by design (see
    # app_naming.export_filename), so without this filter a Studying/
    # Hospital/Driving recording dropped straight into evaluation_data/ per
    # RECORDING_PROTOCOL.md would silently count as Default-mode ground
    # truth. Legacy 2-part filenames (relaxed_1.csv) predate per-mode
    # recording and are kept as Default regardless - see collect()'s
    # docstring.
    _, _default_n_subjects = render_config(
        "Default mode (current evaluation_data/)",
        eval_dir, "relaxed_*.csv", "stressed_*.csv", out, mode="default")
    _current_modes = ["default"]
    for _mode, _label in (("studying", "Studying"), ("hospital", "Hospital"),
                          ("driving", "Driving")):
        render_config(f"{_label} mode (current evaluation_data/)",
                      eval_dir, "relaxed_*.csv", "stressed_*.csv", out, mode=_mode)
        _current_modes.append(_mode)

    # Cross-check every relaxed_*/stressed_* CSV in evaluation_data/ landed
    # in exactly one of the sections just rendered - see
    # find_unclaimed_files()'s docstring for why this exists at all.
    _unclaimed = find_unclaimed_files(eval_dir, "relaxed_*.csv", "stressed_*.csv",
                                      _current_modes)
    if _unclaimed:
        out.append("\n" + "!" * 78)
        out.append("UNCLAIMED FILES - excluded from EVERY mode section above")
        out.append("!" * 78)
        out.append("  These matched relaxed_*.csv/stressed_*.csv but were not claimed")
        out.append("  by any task-mode section - they contribute to NO reported number.")
        out.append("  This is exactly the failure mode that silently excluded 48 valid")
        out.append("  PP1 samples for months (relaaxed_2.csv, see RECORDING_PROTOCOL.md).")
        out.append("  Check each name below for a typo or a corrupted field:")
        for _name in _unclaimed:
            out.append(f"    {_name}")

    out.append("\n### Studying mode (archived baseline_v1/study*)")
    out.append("  CAUTION: RECORDING_PROTOCOL.md documents this folder as unreconciled")
    out.append("  provenance - do not cite without re-checking which files were really")
    out.append("  in the original run. Kept for reference only, not the current number;")
    out.append("  see the 'Studying mode (current evaluation_data/)' section above for that.")
    render_config("  (detail)", eval_dir / "baseline_v1",
                  "studyrelaxed_*.csv", "studystressed_*.csv", out)
    render_config("Archived baseline_v1 non-study recordings",
                  eval_dir / "baseline_v1",
                  "relaxed_*.csv", "stressed_*.csv", out)

    out.append("\n" + "=" * 78)
    out.append("EXTERNAL VALIDATION — UBFC-PHYS")
    out.append("=" * 78)
    out.append("Independent subjects with synchronised physiological ground")
    out.append("truth. Kept in its own folder so it can never pool with the")
    out.append("single-subject recordings above - mixing them would silently")
    out.append("corrupt both numbers.")
    out.append("  T1 rest -> relaxed | T2 speech, T3 arithmetic -> stressed")
    render_config("UBFC-Phys (cross-subject)",
                  eval_dir / "ubfc_phys",
                  "relaxed_*.csv", "stressed_*.csv", out)

    # Phase 4.2: each fusion branch removed once, on the same cross-subject
    # data. Recomputed fresh every run from datasets/feature_traces/ (not a
    # separately-generated report) - it's pure numpy over cached CSVs, same
    # venv as this script, so there is no reason for it to be anything but
    # regenerated here like every other number.
    render_ablation_table(run_ablation_study(root / "datasets" / "feature_traces"), out)

    # Phase 1.4 / 4.4: stress index vs EDA correlation, folded in the same
    # way - pure numpy/scipy over already-saved traces + UBFC-Phys ground
    # truth, no reason for this to live only in its own separate report.
    render_correlation(run_correlation_study(
        root / "datasets" / "feature_traces", root / "datasets" / "UBFC-Phys"), out)

    # Phase 6.1: does the CNN branch's own output track genuine self-reported
    # stress? Reads the live session_history.db directly (the only section
    # here that does) rather than a cached CSV/trace, since self-reports
    # accumulate through normal app use, not a dedicated recording session.
    render_daisee_selfreport(run_daisee_selfreport_study(), out)

    # Phase 4.1: the behavioral-signals-only LOSO comparison (hand-tuned vs
    # logistic regression vs gradient boosting) that the 55.19%/66.74%/
    # 77.06% figures in the README come from. Never writes
    # models/meta_learner.joblib - that stays a deliberate, separate --save
    # step (see train_meta_learner.py), not a side effect of regenerating
    # reported numbers.
    render_learned_fusion(run_learned_fusion_study(root / "datasets" / "feature_traces"), out)

    render_artifacts(root, out)

    out.append("\n" + "=" * 78)
    out.append("READING THESE NUMBERS")
    out.append("=" * 78)
    out.append("* Frame-level accuracy is the familiar figure but its n is inflated:")
    out.append("  frames within one recording are highly autocorrelated.")
    out.append("* Session-level accuracy uses the honest denominator (recordings),")
    out.append("  which is why its confidence interval is much wider.")
    if _default_n_subjects and _default_n_subjects >= 2:
        out.append(f"* Default-mode live recordings now span {_default_n_subjects} "
                   f"distinct subjects (see the PER-SUBJECT")
        out.append("  breakdown above) - this is real cross-subject evidence, not "
                   "within-subject only.")
        out.append("  Studying/Hospital/Driving modes still have no live recordings "
                   "yet, so those remain")
        out.append("  single-subject or empty until Phase 1's team recordings extend "
                   "to those modes.")
    else:
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
