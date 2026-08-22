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

Two of the six named branches (rPPG, the GRU temporal model) are already at
W_RPPG=0.0 / W_TEMPORAL=0.0 in the shipped config - see rppg_live.py and
models/temporal_gru_test_report.txt for why. Ablating an already-zero-weight
branch is a no-op by construction, so those two are reported as citations
rather than recomputed.

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

# The 7 canonical signal columns, same vocabulary/order the rest of the
# system uses (learned_fusion.FEATURE_ORDER, TEMPORAL_FEATURE_ORDER's first
# 7). Read from BEHAVIOR_WEIGHTS rather than hard-coded so a future signal
# addition/removal there is picked up automatically.
SIGNALS = list(BEHAVIOR_WEIGHTS)

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
    """
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
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
                rows.append(row)
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda r: r['t'])
    return rows


def discover_traces(trace_dir):
    """subject -> {task: (filename, rows)}, only files that parse cleanly."""
    by_subject = defaultdict(dict)
    for path in sorted(Path(trace_dir).glob('features_*.csv')):
        m = NAME_RE.search(path.stem)
        if not m:
            continue
        subject, task = f"s{int(m.group(1))}", int(m.group(2))
        rows = load_trace(path)
        if rows:
            by_subject[subject][task] = (path.name, rows)
    return by_subject


# ---------------------------------------------------------------------------
# Per-frame fusion - exact replica of compute_stress_index()'s weight
# allocation + rule boost, parameterized so each ablation is "one config
# constant changed," not a different formula.
# ---------------------------------------------------------------------------
def fused_frame(row, w_daisee, w_behavior, rule_boost_amount):
    daisee_s = row['daisee_stress']
    beh_s = row['behavior_stress']

    w_d = w_daisee if daisee_s is not None else 0.0
    w_b = w_behavior
    total_w = max(w_d + w_b, 1e-6)
    w_d, w_b = w_d / total_w, w_b / total_w

    raw = w_b * beh_s
    if daisee_s is not None:
        raw += w_d * daisee_s

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


def replay_condition(rows, *, w_daisee, w_behavior, rule_boost_amount, smoothing_enabled):
    """
    Re-run one subject-task file's frames through fusion + smoothing +
    hysteresis under one ablation condition. Fresh TemporalSmoother/
    HysteresisLevel per file, matching evaluate_video_file.py's per-video
    reset (each UBFC-Phys T1/T2/T3 clip is an independent recording).
    """
    window = smoothing_window(rows, smoothing_enabled)
    smoother = TemporalSmoother(window_size=window)
    hyst = HysteresisLevel()
    out = []
    for row in rows:
        boosted = fused_frame(row, w_daisee, w_behavior, rule_boost_amount)
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
}

# Already zero-weight in the shipped config - ablating them is a no-op by
# construction. Listed for completeness (the plan names six branches), not
# recomputed.
NOOP_CONDITIONS = [
    ('- rPPG heart-rate branch',
     'already W_RPPG=0.0 - unproven on compressed webcam video (see '
     'rppg_live.py); removing it changes nothing by construction.'),
    ('- GRU temporal branch',
     'already W_TEMPORAL=0.0 - scored below the majority-class floor (see '
     'models/temporal_gru_test_report.txt); removing it changes nothing by '
     'construction.'),
]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_ablation_study(trace_dir='../datasets/feature_traces'):
    """
    Returns an ordered dict: condition name -> (frame_scores dict,
    session_scores dict), or None if no trace CSVs are found (e.g. the
    dataset-derived traces aren't present in this checkout).
    """
    by_subject = discover_traces(trace_dir)
    if not by_subject:
        return None

    results = {}
    for cond_name, params in CONDITIONS.items():
        relaxed_files, stressed_files = [], []
        for subject, tasks in sorted(by_subject.items()):
            for task, (fname, rows) in sorted(tasks.items()):
                out_rows = replay_condition(rows, **params)
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
