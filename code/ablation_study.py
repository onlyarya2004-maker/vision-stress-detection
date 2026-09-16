"""
Ablation table (Phase 4.2)
==========================
Each fusion branch removed one at a time, on the same UBFC-Phys cross-subject
data the headline "hand-tuned fusion" number is measured on. Per the plan
("each branch removed one at a time (behavior / CNN / rPPG / temporal / rule
boost / smoothing) - this becomes the strongest results artifact"), this
answers "how much does each piece actually contribute," not just "does the
whole system work."

Why this reuses saved traces instead of re-running video
----------------------------------------------------------
UBFC-Phys videos are deleted after processing (see PLAN_FOR_PP2.md); only the
derived per-frame traces in datasets/feature_traces/ survive. Those traces
already contain the CALIBRATED per-branch scores (`daisee_stress`,
`behavior_stress`) that stress_fusion.compute_stress_index() itself would
produce - re-deriving them would mean re-implementing personal-baseline
calibration for no benefit. So this module starts one layer downstream: it
takes the two already-computed branch scores and replays ONLY the part that
differs per ablation - weight allocation, the rule boost, and the smoothing
window - using the REAL TemporalSmoother/HysteresisLevel classes from
stress_fusion.py (not reimplementations), so each condition is a faithful
answer to "what if this one config constant were different," not a
hand-rolled approximation.

The rPPG branch is a genuine no-op: W_RPPG=0.0, it's an unresolved
validation gap (see rppg_live.py), and the cached traces never captured
the raw pixel colors rPPG needs (a fundamentally different input from the
10 behavioral-feature columns below) - there's nothing to recompute it
from, so it's reported as a citation, not recomputed.

The GRU temporal branch is NOT a no-op, as of 2026-09-16 - fixed here.
Earlier reasoning wrongly assumed this needed the (deleted) UBFC-Phys
videos to fix, since the traces have no temporal_stress column. But the
traces DO already carry all 10 of temporal_model.TEMPORAL_FEATURE_ORDER's
raw per-frame values (the 7 behavioral signals plus ear/ear_asymmetry/
blinks_per_minute) - the same inputs the live GRU consumes. So this module
now runs the REAL GRUTemporalInference over each trace's own sequence of
those 10 values (compute_temporal_scores() below), exactly reproducing
what the live app would have computed frame-by-frame, with no video
needed at all.

Usage
-----
    python ablation_study.py
    python ablation_study.py --trace-dir ../datasets/feature_traces
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import config
from stress_fusion import TemporalSmoother, HysteresisLevel, BEHAVIOR_WEIGHTS
from eval_scoring import frame_scores, session_scores, pct, ci_str
from temporal_model import GRUTemporalInference, TEMPORAL_FEATURE_ORDER

# The 7 canonical signal columns, same vocabulary/order the rest of the
# system uses (learned_fusion.FEATURE_ORDER, TEMPORAL_FEATURE_ORDER's first
# 7). Read from BEHAVIOR_WEIGHTS rather than hard-coded so a future signal
# addition/removal there is picked up automatically.
SIGNALS = list(BEHAVIOR_WEIGHTS)

# The 3 extra raw columns TEMPORAL_FEATURE_ORDER needs beyond SIGNALS -
# already present in every feature_traces/*.csv file (evaluate_video_file.py
# has always written them), just never read by this module before.
TEMPORAL_EXTRA_COLUMNS = [c for c in TEMPORAL_FEATURE_ORDER if c not in SIGNALS]

# features_vid_s12_T3.csv -> subject 12, task 3. Same convention as
# train_meta_learner.py's NAME_RE, kept independent rather than imported
# since it's a two-line regex, not worth a cross-module dependency for.
NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
def load_trace(path):
    """
    One subject-task trace file -> list of per-frame dicts, sorted by t.

    daisee_stress is None on frames the CNN branch skipped (CNN_EVERY_N_FRAMES
    cadence) - carried as None, never interpolated/carried-forward, matching
    exactly what compute_stress_index() received live (evaluate_video_file.py
    passes daisee_probs=None on those frames, not a stale value).

    A file entirely missing TEMPORAL_EXTRA_COLUMNS (pre-dates this schema
    extension) is excluded with a LOUD warning naming it, not silently -
    every row would otherwise raise KeyError and vanish with nothing
    printed, silently shrinking the study's subject count with no signal
    that it happened.
    """
    rows = []
    skipped_rows = 0
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        missing = [c for c in TEMPORAL_EXTRA_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"  WARNING: {path.name} is missing column(s) {missing} "
                  f"(pre-dates the temporal-branch schema extension) - "
                  f"EXCLUDED from this study rather than defaulted.")
            return []
        for r in reader:
            try:
                row = {
                    't': float(r['t']),
                    'behavior_stress': float(r['behavior_stress']),
                    'daisee_stress': (
                        float(r['daisee_stress'])
                        if r.get('daisee_stress') not in (None, '') else None
                    ),
                }
                for s in SIGNALS:
                    row[s] = float(r[s])
                for s in TEMPORAL_EXTRA_COLUMNS:
                    row[s] = float(r[s])
                rows.append(row)
            except (KeyError, ValueError):
                skipped_rows += 1
                continue
    if skipped_rows:
        print(f"  WARNING: {path.name} - {skipped_rows} malformed row(s) skipped")
    rows.sort(key=lambda r: r['t'])
    return rows


def compute_temporal_scores(rows, inf=None):
    """
    Real GRUTemporalInference replayed over one trace's own frames, in
    order. Returns a list aligned 1:1 with `rows`, each entry the model's
    predict() output at that point (None until the 16-frame buffer fills,
    exactly the live sliding-window contract - see temporal_model.py).

    inf: an existing GRUTemporalInference to reuse (reset() between files -
    each UBFC-Phys clip is an independent recording, matching
    replay_condition()'s existing fresh-smoother-per-file convention).
    None (default) constructs one for this call alone. discover_traces()
    passes a shared instance across all its files so the model's own
    load + self-verify (np.load + a full forward pass over every baked-in
    verification sequence) pays once per study, not once per subject/task.

    Uses the actual live inference path, not a reimplementation: if the
    shipped models/temporal_gru.npz is ever swapped or fails its own
    self-verification, this reports None for every frame rather than a
    silently wrong number (is_available() gates the whole computation).
    """
    if inf is None:
        inf = GRUTemporalInference()
    else:
        inf.reset()
    if not inf.is_available():
        return [None] * len(rows)
    scores = []
    for row in rows:
        inf.add_frame({name: row[name] for name in TEMPORAL_FEATURE_ORDER})
        scores.append(inf.predict())
    return scores


def discover_traces(trace_dir, subjects=None):
    """subject -> {task: (filename, rows, temporal_scores)}, only files that
    parse cleanly. temporal_scores is computed once per file here (not per
    ablation condition) since the GRU's own output never changes across
    conditions - only whether it gets WEIGHTED into fused_frame() does.

    subjects: optional iterable restricting discovery to only these subject
    ids - files for any other subject are skipped BEFORE loading or
    scoring, not after. This matters because compute_temporal_scores() is
    real GRU inference, not free: filtering only after computing it for
    every subject (the original bug) paid full inference cost for subjects
    the caller was about to throw away anyway - defeating the point of a
    restricted, leak-free check being cheap to run.
    """
    by_subject = defaultdict(dict)
    inf = GRUTemporalInference()
    for path in sorted(Path(trace_dir).glob('features_*.csv')):
        m = NAME_RE.search(path.stem)
        if not m:
            continue
        subject, task = f"s{int(m.group(1))}", int(m.group(2))
        if subjects is not None and subject not in subjects:
            continue
        rows = load_trace(path)
        if rows:
            temporal_scores = compute_temporal_scores(rows, inf=inf)
            by_subject[subject][task] = (path.name, rows, temporal_scores)
    return by_subject


def load_held_out_test_subjects(report_path='../models/temporal_gru_test_report.txt'):
    """
    Parses the `HELD_OUT_TEST_SUBJECTS: s7,s9` line train_temporal_model_ubfc.py
    writes into the exported model's report - the subjects its final export
    fit never trained on or used for early-stopping, i.e. the only subjects
    a fair (leak-free) ablation check can honestly use. Returns a set, or
    None if the report doesn't exist or has no such line (e.g. the DaiSEE-
    trained report, which predates this concept entirely).

    Exists so run_all_evaluations.py can re-run the leak-free check
    automatically every time it regenerates CANONICAL_RESULTS, instead of
    that check only ever having been run once by hand - the exact
    'someone has to remember to manually re-verify this' gap that made the
    2026-09-16 contamination fix easy to silently regress.
    """
    path = Path(report_path)
    if not path.exists():
        return None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('HELD_OUT_TEST_SUBJECTS:'):
            value = line.split(':', 1)[1].strip()
            return set(value.split(',')) if value else None
    return None


# ---------------------------------------------------------------------------
# Per-frame fusion - exact replica of compute_stress_index()'s weight
# allocation + rule boost, parameterized so each ablation is "one config
# constant changed," not a different formula.
# ---------------------------------------------------------------------------
def fused_frame(row, w_daisee, w_behavior, rule_boost_amount,
                temporal_score=None, w_temporal=0.0):
    """
    temporal_score/w_temporal default to None/0.0, matching
    compute_stress_index()'s own contract exactly: the temporal branch
    only enters the weighted average when BOTH a real score is present
    AND its weight is nonzero (config.py's W_TEMPORAL == 0.0 default means
    passing a real score here still changes nothing, by the same
    construction as the live system).
    """
    daisee_s = row['daisee_stress']
    beh_s = row['behavior_stress']

    w_d = w_daisee if daisee_s is not None else 0.0
    w_b = w_behavior
    w_t = w_temporal if (temporal_score is not None and w_temporal > 0.0) else 0.0
    total_w = max(w_d + w_b + w_t, 1e-6)
    w_d, w_b, w_t = w_d / total_w, w_b / total_w, w_t / total_w

    raw = w_b * beh_s
    if daisee_s is not None:
        raw += w_d * daisee_s
    if temporal_score is not None and w_temporal > 0.0:
        raw += w_t * float(temporal_score)

    # Active-signal count for the rule boost - independent of the weights
    # above, matching compute_stress_index() exactly (a signal can trigger
    # the boost even on a frame where its branch has been weighted to zero;
    # that is how the real system behaves too, not an artifact of ablation).
    active = sum(1 for s in SIGNALS if row[s] >= config.ACTIVE_SIGNAL_THRESHOLD)
    if daisee_s is not None and daisee_s >= config.DAISEE_ACTIVE_THRESHOLD:
        active += 1
    boost = active >= config.RULE_BOOST_MIN_COUNT
    return float(np.clip(raw + (rule_boost_amount if boost else 0.0), 0.0, 1.0))


def _effective_fps(rows, default=30.0):
    """Derived from the trace's own timestamps rather than assumed, since
    UBFC-Phys videos (and whatever --sample-every they were processed with)
    are gone - this works from the surviving trace alone."""
    if len(rows) < 2:
        return default
    span = rows[-1]['t'] - rows[0]['t']
    return (len(rows) - 1) / span if span > 0 else default


def smoothing_window(rows, smoothing_enabled):
    """
    window=1 is not a special case - a maxlen=1 deque's mean is the value
    itself, so it IS "no smoothing," computed the same way every other
    window size is. This is the literal config-equivalent of disabling
    TemporalSmoother, not a separate code path.
    """
    if not smoothing_enabled:
        return 1
    live_span_s = config.SMOOTHING_WINDOW_FRAMES / 30.0
    return max(1, round(live_span_s * _effective_fps(rows)))


def replay_condition(rows, temporal_scores, *, w_daisee, w_behavior,
                     rule_boost_amount, smoothing_enabled, w_temporal=None):
    """
    Re-run one subject-task file's frames through fusion + smoothing +
    hysteresis under one ablation condition. Fresh TemporalSmoother/
    HysteresisLevel per file, matching evaluate_video_file.py's per-video
    reset (each UBFC-Phys T1/T2/T3 clip is an independent recording).

    w_temporal=None means "use the live config value" (config.W_TEMPORAL) -
    every condition except the dedicated temporal-branch ablation keeps
    this at the current config value, matching the file's own "one config
    constant changed" philosophy; only that one condition passes 0.0
    explicitly.
    """
    if w_temporal is None:
        w_temporal = config.W_TEMPORAL
    window = smoothing_window(rows, smoothing_enabled)
    smoother = TemporalSmoother(window_size=window)
    hyst = HysteresisLevel()
    out = []
    for row, t_score in zip(rows, temporal_scores):
        boosted = fused_frame(row, w_daisee, w_behavior, rule_boost_amount,
                              temporal_score=t_score, w_temporal=w_temporal)
        smoothed = smoother.update(boosted)
        level = hyst.update(smoothed)
        out.append({'stress_index': smoothed, 'level': level})
    return out


# ---------------------------------------------------------------------------
# The six named branches
# ---------------------------------------------------------------------------
CONDITIONS = {
    'Full system (current config)': dict(
        w_daisee=config.W_DAISEE, w_behavior=config.W_BEHAVIOR,
        rule_boost_amount=config.RULE_BOOST_AMOUNT, smoothing_enabled=True,
    ),
    '- Behavioral engine (W_BEHAVIOR=0)': dict(
        w_daisee=config.W_DAISEE, w_behavior=0.0,
        rule_boost_amount=config.RULE_BOOST_AMOUNT, smoothing_enabled=True,
    ),
    '- DaiSEE CNN (W_DAISEE=0)': dict(
        w_daisee=0.0, w_behavior=config.W_BEHAVIOR,
        rule_boost_amount=config.RULE_BOOST_AMOUNT, smoothing_enabled=True,
    ),
    '- Rule boost (RULE_BOOST_AMOUNT=0)': dict(
        w_daisee=config.W_DAISEE, w_behavior=config.W_BEHAVIOR,
        rule_boost_amount=0.0, smoothing_enabled=True,
    ),
    '- Temporal smoothing (window=1 frame)': dict(
        w_daisee=config.W_DAISEE, w_behavior=config.W_BEHAVIOR,
        rule_boost_amount=config.RULE_BOOST_AMOUNT, smoothing_enabled=False,
    ),
    '- GRU temporal branch (W_TEMPORAL=0)': dict(
        w_daisee=config.W_DAISEE, w_behavior=config.W_BEHAVIOR,
        rule_boost_amount=config.RULE_BOOST_AMOUNT, smoothing_enabled=True,
        w_temporal=0.0,
    ),
}

# rPPG remains a genuine no-op: W_RPPG=0.0, and unlike the temporal branch
# the cached traces never captured the raw pixel colors it would need -
# nothing to recompute it from without the (deleted) video.
NOOP_CONDITIONS = [
    ('- rPPG heart-rate branch',
     'already W_RPPG=0.0 - unproven on compressed webcam video (see '
     'rppg_live.py); removing it changes nothing by construction.'),
]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_ablation_study(trace_dir='../datasets/feature_traces', subjects=None):
    """
    Returns an ordered dict: condition name -> (frame_scores dict,
    session_scores dict), or None if no trace CSVs are found (e.g. the
    dataset-derived traces aren't present in this checkout).

    subjects: optional iterable restricting the study to only these subject
    ids (e.g. {'s7', 's9'}). Exists because this project's 10 UBFC-Phys
    subjects are the ONLY temporal-labeled dataset available, so they are
    necessarily both the GRU's training pool and this study's evaluation
    pool - any exported GRU trained on some of them will look artificially
    strong here unless the check is restricted to subjects it genuinely
    never trained on. None (default) uses every subject found, matching
    the original whole-dataset behavior for every branch that has no
    training-time leakage risk (behavioral engine, DaiSEE CNN, rule boost,
    smoothing).
    """
    by_subject = discover_traces(trace_dir, subjects=subjects)
    if not by_subject:
        return None

    results = {}
    for cond_name, params in CONDITIONS.items():
        relaxed_files, stressed_files = [], []
        for subject, tasks in sorted(by_subject.items()):
            for task, (fname, rows, temporal_scores) in sorted(tasks.items()):
                out_rows = replay_condition(rows, temporal_scores, **params)
                label = (relaxed_files if task == 1 else stressed_files)
                label.append((f"{subject}_{fname}", out_rows))
        fr = frame_scores(relaxed_files, stressed_files)
        se = session_scores(relaxed_files, stressed_files)
        results[cond_name] = (fr, se)
    return results


def render(results, out):
    out.append("\n### Ablation table (UBFC-Phys cross-subject, each branch removed once)")
    if results is None:
        out.append("  (no traces found under datasets/feature_traces/ - skipped)")
        return
    out.append("  Frame-level accuracy, session-level accuracy, and the drop from the")
    out.append("  full system, for each branch removed one at a time.")
    out.append("")

    baseline_fr, baseline_se = results['Full system (current config)']
    out.append(f"  {'condition':38s}{'frame acc':>12}{'session acc':>14}{'delta (pp)':>12}")
    out.append("  " + "-" * 76)
    for cond_name, (fr, se) in results.items():
        delta = (fr['accuracy'] - baseline_fr['accuracy']) * 100.0
        delta_str = "  (baseline)" if cond_name.startswith('Full system') else f"{delta:+.2f}"
        out.append(f"  {cond_name:38s}{pct(fr['accuracy']):>12}"
                   f"{pct(se['accuracy']):>14}{delta_str:>12}")
    for cond_name, note in NOOP_CONDITIONS:
        out.append(f"  {cond_name:38s}{'(no-op)':>12}{'(no-op)':>14}{'+0.00':>12}")
    out.append("")

    for cond_name, (fr, se) in results.items():
        out.append(f"  {cond_name}")
        out.append(f"    frame-level   n={fr['n']:<6}  acc {pct(fr['accuracy'])}  "
                   f"95% CI {ci_str(fr['ci'])}")
        out.append(f"    session-level n={se['n']:<6}  acc {pct(se['accuracy'])}  "
                   f"95% CI {ci_str(se['ci'])}")
    out.append("")
    for cond_name, note in NOOP_CONDITIONS:
        out.append(f"  {cond_name}: {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace-dir', default='../datasets/feature_traces')
    args = ap.parse_args()

    results = run_ablation_study(args.trace_dir)
    out = []
    render(results, out)
    print("\n".join(out))
    return 0 if results is not None else 1


if __name__ == '__main__':
    sys.exit(main())
