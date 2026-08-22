"""
Ablation study tests.

fused_frame()/smoothing_window() are pure functions tested directly against
hand-computed expectations. discover_traces()/run_ablation_study() are
tested against synthetic tmp_path CSVs (real UBFC-Phys traces aren't part of
the repo - see README.md's dataset section), engineered so a branch that
should matter provably changes the result, not just "doesn't crash."
"""

import csv

import pytest

import config
from ablation_study import (
    fused_frame, smoothing_window, replay_condition, load_trace,
    discover_traces, run_ablation_study, CONDITIONS, NAME_RE, SIGNALS,
)


def _row(daisee_stress=0.5, behavior_stress=0.5, **signal_overrides):
    row = {'daisee_stress': daisee_stress, 'behavior_stress': behavior_stress}
    for s in SIGNALS:
        row[s] = signal_overrides.get(s, 0.0)
    return row


# --------------------------------------------------------------------------
# fused_frame
# --------------------------------------------------------------------------

def test_fused_frame_blends_both_branches_by_weight():
    row = _row(daisee_stress=1.0, behavior_stress=0.0)
    out = fused_frame(row, w_daisee=0.20, w_behavior=0.80, rule_boost_amount=0.0)
    assert out == pytest.approx(0.20)


def test_fused_frame_redistributes_weight_when_daisee_missing():
    """daisee_stress=None (CNN skipped this frame) - behavior gets 100% of
    the weight for that frame, matching compute_stress_index() exactly."""
    row = _row(daisee_stress=None, behavior_stress=0.7)
    out = fused_frame(row, w_daisee=0.20, w_behavior=0.80, rule_boost_amount=0.0)
    assert out == pytest.approx(0.7)


def test_fused_frame_zero_daisee_weight_is_behavior_only():
    row = _row(daisee_stress=1.0, behavior_stress=0.3)
    out = fused_frame(row, w_daisee=0.0, w_behavior=0.80, rule_boost_amount=0.0)
    assert out == pytest.approx(0.3)


def test_fused_frame_zero_behavior_weight_is_daisee_only():
    row = _row(daisee_stress=0.3, behavior_stress=1.0)
    out = fused_frame(row, w_daisee=0.20, w_behavior=0.0, rule_boost_amount=0.0)
    assert out == pytest.approx(0.3)


def test_fused_frame_rule_boost_fires_above_active_signal_count():
    """4+ active signals (>= ACTIVE_SIGNAL_THRESHOLD) triggers the boost,
    same threshold/count compute_stress_index() uses."""
    active_signals = {s: 1.0 for s in SIGNALS[:4]}   # 4 signals >= threshold
    row = _row(daisee_stress=None, behavior_stress=0.0, **active_signals)
    boosted = fused_frame(row, w_daisee=0.20, w_behavior=0.80,
                          rule_boost_amount=config.RULE_BOOST_AMOUNT)
    unboosted = fused_frame(row, w_daisee=0.20, w_behavior=0.80, rule_boost_amount=0.0)
    assert boosted == pytest.approx(unboosted + config.RULE_BOOST_AMOUNT)


def test_fused_frame_rule_boost_does_not_fire_below_count():
    active_signals = {s: 1.0 for s in SIGNALS[:3]}   # only 3 - below RULE_BOOST_MIN_COUNT=4
    row = _row(daisee_stress=None, behavior_stress=0.0, **active_signals)
    boosted = fused_frame(row, w_daisee=0.20, w_behavior=0.80,
                          rule_boost_amount=config.RULE_BOOST_AMOUNT)
    unboosted = fused_frame(row, w_daisee=0.20, w_behavior=0.80, rule_boost_amount=0.0)
    assert boosted == pytest.approx(unboosted)


def test_fused_frame_daisee_deviation_counts_toward_active_signals():
    """A DaiSEE score above DAISEE_ACTIVE_THRESHOLD counts as the 4th active
    signal even with only 3 behavioral signals active."""
    active_signals = {s: 1.0 for s in SIGNALS[:3]}
    row = _row(daisee_stress=config.DAISEE_ACTIVE_THRESHOLD + 0.01,
              behavior_stress=0.0, **active_signals)
    boosted = fused_frame(row, w_daisee=0.20, w_behavior=0.80,
                          rule_boost_amount=config.RULE_BOOST_AMOUNT)
    unboosted = fused_frame(row, w_daisee=0.20, w_behavior=0.80, rule_boost_amount=0.0)
    assert boosted == pytest.approx(unboosted + config.RULE_BOOST_AMOUNT)


def test_fused_frame_output_always_clipped_to_unit_range():
    row = _row(daisee_stress=1.0, behavior_stress=1.0, **{s: 1.0 for s in SIGNALS})
    out = fused_frame(row, w_daisee=0.20, w_behavior=0.80,
                      rule_boost_amount=config.RULE_BOOST_AMOUNT)
    assert 0.0 <= out <= 1.0


# --------------------------------------------------------------------------
# smoothing_window
# --------------------------------------------------------------------------

def test_smoothing_window_disabled_is_always_one():
    rows = [{'t': i * 0.1} for i in range(100)]
    assert smoothing_window(rows, smoothing_enabled=False) == 1


def test_smoothing_window_enabled_holds_span_constant_at_effective_fps():
    # 10 fps effective rate -> window should be ~ (SMOOTHING_WINDOW_FRAMES/30)*10
    rows = [{'t': i * 0.1} for i in range(200)]   # 10 samples/sec
    expected = max(1, round((config.SMOOTHING_WINDOW_FRAMES / 30.0) * 10.0))
    assert smoothing_window(rows, smoothing_enabled=True) == expected


def test_smoothing_window_degenerate_single_row_falls_back_safely():
    assert smoothing_window([{'t': 0.0}], smoothing_enabled=True) >= 1


# --------------------------------------------------------------------------
# replay_condition
# --------------------------------------------------------------------------

def test_replay_condition_preserves_frame_count_and_range():
    rows = [_row(daisee_stress=0.5, behavior_stress=0.5) | {'t': i * 0.1}
           for i in range(50)]
    out = replay_condition(rows, **CONDITIONS['Full system (current config)'])
    assert len(out) == len(rows)
    assert all(0.0 <= r['stress_index'] <= 1.0 for r in out)
    assert all(r['level'] in ('LOW', 'MODERATE', 'HIGH') for r in out)


def test_replay_condition_smoothing_dampens_a_single_spike():
    """One frame spiking to 1.0 in an otherwise-quiet sequence should move
    the smoothed index much less than the unsmoothed one - this is the
    entire point of the smoothing branch existing."""
    quiet = [_row(daisee_stress=0.1, behavior_stress=0.1) | {'t': i * 0.1}
            for i in range(30)]
    spike = quiet + [_row(daisee_stress=1.0, behavior_stress=1.0) | {'t': 3.0}]

    smoothed_params = dict(CONDITIONS['Full system (current config)'])
    unsmoothed_params = dict(CONDITIONS['- Temporal smoothing (window=1 frame)'])

    smoothed_out = replay_condition(spike, **smoothed_params)
    unsmoothed_out = replay_condition(spike, **unsmoothed_params)

    assert smoothed_out[-1]['stress_index'] < unsmoothed_out[-1]['stress_index']


def test_replay_condition_is_stateless_across_independent_calls():
    """A fresh smoother/hysteresis per call - a second, unrelated file must
    not inherit state from the first (matching evaluate_video_file.py's
    per-video reset)."""
    hot = [_row(daisee_stress=0.9, behavior_stress=0.9) | {'t': i * 0.1}
          for i in range(60)]
    cold = [_row(daisee_stress=0.0, behavior_stress=0.0) | {'t': i * 0.1}
           for i in range(5)]

    replay_condition(hot, **CONDITIONS['Full system (current config)'])
    cold_out = replay_condition(cold, **CONDITIONS['Full system (current config)'])
    assert cold_out[0]['level'] == 'LOW'


# --------------------------------------------------------------------------
# Trace loading / discovery
# --------------------------------------------------------------------------

TRACE_HEADER = (
    ['t'] + SIGNALS
    + ['posture_detected', 'daisee_raw', 'daisee_stress', 'behavior_stress',
       'raw_index', 'stress_index_frame', 'stress_index_smoothed', 'level',
       'active_signals', 'boost_applied']
)


def _write_trace(path, n_frames, daisee_every=3):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=TRACE_HEADER)
        w.writeheader()
        for i in range(n_frames):
            row = {name: 0.1 for name in SIGNALS}
            row.update({
                't': i * 0.1, 'posture_detected': 1,
                'daisee_raw': 0.4 if i % daisee_every == 0 else '',
                'daisee_stress': 0.2 if i % daisee_every == 0 else '',
                'behavior_stress': 0.3, 'raw_index': 0.25,
                'stress_index_frame': 0.25, 'stress_index_smoothed': 0.25,
                'level': 'LOW', 'active_signals': 1, 'boost_applied': 0,
            })
            w.writerow(row)


def test_name_re_distinguishes_double_digit_subjects():
    m = NAME_RE.search('features_vid_s10_T1')
    assert m.group(1) == '10' and m.group(2) == '1'
    m2 = NAME_RE.search('features_vid_s1_T1')
    assert m2.group(1) == '1'


def test_load_trace_marks_skipped_cnn_frames_as_none(tmp_path):
    path = tmp_path / 'features_vid_s1_T1.csv'
    _write_trace(path, n_frames=6, daisee_every=3)
    rows = load_trace(path)
    assert len(rows) == 6
    assert rows[0]['daisee_stress'] == pytest.approx(0.2)   # i=0
    assert rows[1]['daisee_stress'] is None                 # i=1, skipped
    assert rows[2]['daisee_stress'] is None                 # i=2, skipped
    assert rows[3]['daisee_stress'] == pytest.approx(0.2)   # i=3


def test_discover_traces_groups_by_subject_and_task(tmp_path):
    _write_trace(tmp_path / 'features_vid_s1_T1.csv', 5)
    _write_trace(tmp_path / 'features_vid_s1_T2.csv', 5)
    _write_trace(tmp_path / 'features_vid_s10_T1.csv', 5)

    by_subject = discover_traces(tmp_path)
    assert set(by_subject.keys()) == {'s1', 's10'}
    assert set(by_subject['s1'].keys()) == {1, 2}
    assert set(by_subject['s10'].keys()) == {1}


# --------------------------------------------------------------------------
# run_ablation_study - end to end
# --------------------------------------------------------------------------

def test_run_ablation_study_returns_none_for_empty_directory(tmp_path):
    assert run_ablation_study(tmp_path) is None


def test_run_ablation_study_removing_a_perfectly_discriminative_branch_hurts_accuracy(tmp_path):
    """
    Engineer daisee_stress to PERFECTLY separate relaxed (T1) from stressed
    (T2/T3) - i.e. the DaiSEE branch alone would get 100% - while
    behavior_stress is identical noise on both. Full system should score
    near-perfectly; removing DaiSEE (the only informative branch) must drop
    accuracy substantially. This is a correctness check on the ablation
    math, not just a smoke test.
    """
    def write(path, n, daisee_val, behavior_val):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=TRACE_HEADER)
            w.writeheader()
            for i in range(n):
                # Below ACTIVE_SIGNAL_THRESHOLD (0.5) deliberately - at 0.5
                # exactly these would count as "active" and silently trigger
                # the rule boost on every frame, swamping the DaiSEE signal
                # this test is trying to isolate.
                row = {name: 0.0 for name in SIGNALS}
                row.update({
                    't': i * 0.1, 'posture_detected': 1,
                    'daisee_raw': daisee_val, 'daisee_stress': daisee_val,
                    'behavior_stress': behavior_val, 'raw_index': 0.0,
                    'stress_index_frame': 0.0, 'stress_index_smoothed': 0.0,
                    'level': 'LOW', 'active_signals': 0, 'boost_applied': 0,
                })
                w.writerow(row)

    # behavior_val=0.3 is chosen so behavior's own contribution
    # (W_BEHAVIOR=0.80 x 0.3 = 0.24) stays safely below the LOW/MODERATE
    # cutoff (0.36) on its own - only DaiSEE's swing (0.05 -> 0.95) decides
    # the label. A behavior_val large enough to cross 0.36 by itself (e.g.
    # 0.5) would make this scenario meaningless, since it would classify
    # both classes identically regardless of what DaiSEE says.
    for subj in ('s1', 's2', 's3'):
        write(tmp_path / f'features_vid_{subj}_T1.csv', 40, daisee_val=0.05, behavior_val=0.3)
        write(tmp_path / f'features_vid_{subj}_T2.csv', 40, daisee_val=0.95, behavior_val=0.3)

    results = run_ablation_study(tmp_path)
    assert results is not None

    full_fr, _ = results['Full system (current config)']
    no_cnn_fr, _ = results['- DaiSEE CNN (W_DAISEE=0)']

    assert full_fr['accuracy'] > 0.9
    assert no_cnn_fr['accuracy'] < full_fr['accuracy']


def test_run_ablation_study_covers_all_named_conditions(tmp_path):
    for subj in ('s1', 's2'):
        _write_trace(tmp_path / f'features_vid_{subj}_T1.csv', 40)
        _write_trace(tmp_path / f'features_vid_{subj}_T2.csv', 40)

    results = run_ablation_study(tmp_path)
    assert set(results.keys()) == set(CONDITIONS.keys())
    for fr, se in results.values():
        assert fr['n'] == 160   # 2 subjects x 2 files x 40 frames
        assert se['n'] == 4     # 2 subjects x 2 conditions
