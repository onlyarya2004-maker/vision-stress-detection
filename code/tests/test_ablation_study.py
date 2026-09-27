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
import stress_fusion
from ablation_study import (
    fused_frame, smoothing_window, replay_condition, load_trace,
    discover_traces, run_ablation_study, CONDITIONS, NAME_RE, SIGNALS,
    TEMPORAL_EXTRA_COLUMNS, compute_temporal_scores, load_held_out_test_subjects,
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
# fused_frame() vs the REAL stress_fusion.compute_stress_index() -
# equivalence regression test (BUG_AUDIT_2026-08-23.md finding #10)
# --------------------------------------------------------------------------
# fused_frame() above is a hand-written reimplementation of
# compute_stress_index()'s weight-allocation + rule-boost math, verified
# equal to the real function only ONCE, manually, per PLAN_FOR_PP2.md
# (2026-08-22) - not by any automated test. It would silently drift the
# moment compute_stress_index() changes. The tests below close that gap:
# same synthetic signal dict fed into BOTH functions, asserting the
# outputs agree.
#
# W_RPPG stays out of scope here (still 0.0, still no rPPG term in
# fused_frame() - see the module docstring for why: the cached traces
# never captured the raw pixel colors it needs). The temporal branch WAS
# in this same boat until 2026-09-16 - fused_frame() now has a real
# w_temporal/temporal_score term, so this comparison covers both
# W_TEMPORAL=0.0 (today's shipped default) AND a nonzero value (proving
# the fix itself, not just that it doesn't break the zero case).

FUSION_EQUIVALENCE_CASES = {
    "low values, no daisee": (
        {s: 0.05 for s in SIGNALS}, None,
    ),
    "moderate values, no daisee": (
        {s: 0.30 for s in SIGNALS}, None,
    ),
    "high values, no daisee (triggers rule boost)": (
        {s: 0.9 for s in SIGNALS}, None,
    ),
    "boundary near LOW/MODERATE cutoff (0.36), no daisee": (
        {s: 0.36 for s in SIGNALS}, None,
    ),
    "boundary near MODERATE/HIGH cutoff (0.55), no daisee": (
        {s: 0.55 for s in SIGNALS}, None,
    ),
    "3 active behavioral signals + active daisee (triggers rule boost)": (
        {'brow_tension': 0.6, 'jaw_tension': 0.6, 'slouch': 0.6,
         'eye_strain': 0.1, 'blink_stress': 0.1,
         'head_instability': 0.1, 'neck_strain': 0.1},
        # raw_p chosen so the re-centered deviation from
        # DAISEE_DEFAULT_BASELINE clears DAISEE_ACTIVE_THRESHOLD (0.20),
        # making this the 4th active signal alongside the 3 above.
        [0.0, config.DAISEE_DEFAULT_BASELINE + 0.10],
    ),
    "mixed asymmetric values, no daisee": (
        {'eye_strain': 0.83, 'blink_stress': 0.0, 'brow_tension': 0.47,
         'jaw_tension': 0.91, 'head_instability': 0.22,
         'neck_strain': 0.65, 'slouch': 0.14}, None,
    ),
}


@pytest.fixture
def _clean_daisee_baseline():
    """_DAISEE_BASELINE is process-wide singleton state (like
    test_stress_fusion.py's identical fixture) - reset before AND after so
    this can't leak into another test running before/after it."""
    stress_fusion.reset_daisee_baseline()
    yield
    stress_fusion.reset_daisee_baseline()


@pytest.mark.parametrize("name", sorted(FUSION_EQUIVALENCE_CASES))
def test_fused_frame_matches_real_compute_stress_index(name, _clean_daisee_baseline):
    assert config.W_RPPG == 0.0, (
        "premise: this equivalence check is only valid while rPPG cannot "
        "enter the fusion - fused_frame() has no rPPG term at all"
    )
    features, daisee_probs = FUSION_EQUIVALENCE_CASES[name]
    features = dict(features, calibrating=False)

    real = stress_fusion.compute_stress_index(daisee_probs, features, task_mode='Default')

    # fused_frame() never recomputes daisee_stress/behavior_stress itself -
    # it consumes already-computed branch scores (exactly as it does when
    # reading a saved trace CSV) - so feed it the REAL function's own
    # branch outputs and check only the downstream weight/boost math.
    row = {s: features[s] for s in SIGNALS}
    row['daisee_stress'] = real['daisee_stress']
    row['behavior_stress'] = real['behavior_stress']

    # temporal_score=None, w_temporal=0.0 explicitly - deliberately fixed
    # regardless of the live config.W_TEMPORAL value (0.10 as of
    # 2026-09-16), so this parametrized set stays pure DaiSEE/behavioral-
    # path regression coverage. See the two dedicated tests below for the
    # temporal branch's own equivalence at a nonzero weight.
    ablation_out = fused_frame(row, w_daisee=config.W_DAISEE, w_behavior=config.W_BEHAVIOR,
                               rule_boost_amount=config.RULE_BOOST_AMOUNT,
                               temporal_score=None, w_temporal=0.0)

    assert ablation_out == pytest.approx(real['stress_index'], abs=1e-9)


def test_fused_frame_matches_real_compute_stress_index_with_temporal_branch_active(
    monkeypatch, _clean_daisee_baseline,
):
    """Same equivalence check as above, but with the temporal branch given
    a real, nonzero weight and a real score - the actual case
    config.py's W_TEMPORAL comment says fused_frame() couldn't handle
    until this fix. Patches stress_fusion.W_TEMPORAL directly (the
    module-level constant compute_stress_index() closes over), not
    config.W_TEMPORAL, since stress_fusion.py copies it once at import
    time and never re-reads config afterward."""
    monkeypatch.setattr(stress_fusion, 'W_TEMPORAL', 0.10)
    features = dict({s: 0.30 for s in SIGNALS}, calibrating=False)
    temporal_score = 0.72

    real = stress_fusion.compute_stress_index(
        None, features, task_mode='Default', temporal_score=temporal_score,
    )

    row = {s: features[s] for s in SIGNALS}
    row['daisee_stress'] = real['daisee_stress']
    row['behavior_stress'] = real['behavior_stress']

    ablation_out = fused_frame(row, w_daisee=config.W_DAISEE, w_behavior=config.W_BEHAVIOR,
                               rule_boost_amount=config.RULE_BOOST_AMOUNT,
                               temporal_score=temporal_score, w_temporal=0.10)

    assert ablation_out == pytest.approx(real['stress_index'], abs=1e-9)
    # Sanity check this test actually exercises blending, not an
    # accidental no-op - a nonzero temporal weight with a high temporal
    # score must move the result above the temporal-free equivalent.
    without_temporal = fused_frame(row, w_daisee=config.W_DAISEE, w_behavior=config.W_BEHAVIOR,
                                   rule_boost_amount=config.RULE_BOOST_AMOUNT,
                                   temporal_score=None, w_temporal=0.0)
    assert ablation_out > without_temporal


def test_fused_frame_temporal_weight_is_ignored_when_score_is_none():
    """A nonzero w_temporal with no actual score (buffer not full yet,
    matching temporal_model.GRUTemporalInference.predict()'s real None
    contract) must not enter the weighted average - the same guard
    compute_stress_index() itself applies."""
    row = _row(daisee_stress=0.5, behavior_stress=0.5)
    with_none_score = fused_frame(row, w_daisee=0.20, w_behavior=0.80,
                                  rule_boost_amount=0.0,
                                  temporal_score=None, w_temporal=0.5)
    without_temporal_at_all = fused_frame(row, w_daisee=0.20, w_behavior=0.80,
                                          rule_boost_amount=0.0)
    assert with_none_score == pytest.approx(without_temporal_at_all)


# --------------------------------------------------------------------------
# compute_temporal_scores
# --------------------------------------------------------------------------

def _temporal_row(i):
    """A frame dict carrying all 10 TEMPORAL_FEATURE_ORDER values plus the
    other fields _row()/load_trace() expect - compute_temporal_scores()
    only reads the TEMPORAL_FEATURE_ORDER subset, so constant filler
    values are fine; only the frame count matters for this test."""
    row = {s: 0.1 for s in SIGNALS}
    row.update({name: 0.1 for name in TEMPORAL_EXTRA_COLUMNS})
    return row


def test_compute_temporal_scores_matches_the_live_sliding_window_contract():
    """Uses the actual shipped models/temporal_gru.npz (skipped gracefully
    below if unavailable) - None for the first 15 frames, a real float
    from frame 16 onward, exactly GRUTemporalInference.predict()'s own
    documented contract. This is what proves compute_temporal_scores()
    replays the REAL live inference path, not a reimplementation of it."""
    from temporal_model import GRUTemporalInference
    if not GRUTemporalInference().is_available():
        pytest.skip("no live temporal model artifact in this checkout")

    rows = [_temporal_row(i) for i in range(20)]
    scores = compute_temporal_scores(rows)

    assert len(scores) == len(rows)
    assert all(s is None for s in scores[:15])
    assert scores[15] is not None
    assert 0.0 <= scores[15] <= 1.0


def test_compute_temporal_scores_all_none_when_model_unavailable(monkeypatch):
    """A missing/corrupted model must degrade to None for every frame,
    never raise - matching GRUTemporalInference's own fail-safe contract,
    not a special case invented here."""
    import ablation_study

    class _Unavailable:
        def is_available(self):
            return False

    monkeypatch.setattr(ablation_study, 'GRUTemporalInference', lambda: _Unavailable())
    rows = [_temporal_row(i) for i in range(5)]
    assert compute_temporal_scores(rows) == [None] * 5


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
    out = replay_condition(rows, [None] * len(rows),
                           **CONDITIONS['Full system (current config)'])
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
    none_scores = [None] * len(spike)

    smoothed_out = replay_condition(spike, none_scores, **smoothed_params)
    unsmoothed_out = replay_condition(spike, none_scores, **unsmoothed_params)

    assert smoothed_out[-1]['stress_index'] < unsmoothed_out[-1]['stress_index']


def test_replay_condition_is_stateless_across_independent_calls():
    """A fresh smoother/hysteresis per call - a second, unrelated file must
    not inherit state from the first (matching evaluate_video_file.py's
    per-video reset)."""
    hot = [_row(daisee_stress=0.9, behavior_stress=0.9) | {'t': i * 0.1}
          for i in range(60)]
    cold = [_row(daisee_stress=0.0, behavior_stress=0.0) | {'t': i * 0.1}
           for i in range(5)]

    replay_condition(hot, [None] * len(hot), **CONDITIONS['Full system (current config)'])
    cold_out = replay_condition(cold, [None] * len(cold),
                                **CONDITIONS['Full system (current config)'])
    assert cold_out[0]['level'] == 'LOW'


# --------------------------------------------------------------------------
# Trace loading / discovery
# --------------------------------------------------------------------------

TRACE_HEADER = (
    ['t'] + SIGNALS + TEMPORAL_EXTRA_COLUMNS
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
            row.update({name: 0.0 for name in TEMPORAL_EXTRA_COLUMNS})
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


def test_load_trace_excludes_file_missing_temporal_columns_without_raising(tmp_path, capsys):
    """A pre-v7-schema trace file (no ear/ear_asymmetry/blinks_per_minute
    header) must be excluded with a printed warning, not raise or silently
    vanish with zero signal."""
    path = tmp_path / 'features_vid_s1_T1.csv'
    legacy_header = ['t'] + SIGNALS + [
        'posture_detected', 'daisee_raw', 'daisee_stress', 'behavior_stress',
        'raw_index', 'stress_index_frame', 'stress_index_smoothed', 'level',
        'active_signals', 'boost_applied',
    ]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=legacy_header)
        w.writeheader()
        row = {name: 0.1 for name in SIGNALS}
        row.update({'t': 0.0, 'posture_detected': 1, 'daisee_raw': 0.4,
                    'daisee_stress': 0.2, 'behavior_stress': 0.3, 'raw_index': 0.25,
                    'stress_index_frame': 0.25, 'stress_index_smoothed': 0.25,
                    'level': 'LOW', 'active_signals': 1, 'boost_applied': 0})
        w.writerow(row)

    rows = load_trace(path)
    assert rows == []
    assert 'features_vid_s1_T1.csv' in capsys.readouterr().out


def test_discover_traces_subjects_filter_skips_excluded_files_before_loading(tmp_path, capsys):
    """The subjects filter must exclude a file BEFORE load_trace()/
    compute_temporal_scores() ever touch it - proven here by giving the
    excluded subject a malformed file that load_trace() would otherwise
    print a warning for. If discover_traces still reads it, this test's
    capsys check fails."""
    _write_trace(tmp_path / 'features_vid_s1_T1.csv', 10)
    (tmp_path / 'features_vid_s2_T1.csv').write_text('not,a,valid,trace,header\n1,2,3,4,5\n')

    by_subject = discover_traces(tmp_path, subjects={'s1'})
    assert set(by_subject.keys()) == {'s1'}
    assert 'features_vid_s2_T1.csv' not in capsys.readouterr().out


def test_compute_temporal_scores_reused_instance_resets_between_files():
    """Passing a shared `inf` (the discover_traces() efficiency fix) must
    behave identically to a fresh instance per call - no leaked buffer
    state from a previous file's frames."""
    from temporal_model import GRUTemporalInference
    rows_a = [{name: 0.1 for name in SIGNALS + TEMPORAL_EXTRA_COLUMNS} for _ in range(20)]
    rows_b = [{name: 0.9 for name in SIGNALS + TEMPORAL_EXTRA_COLUMNS} for _ in range(20)]

    shared = GRUTemporalInference()
    if not shared.is_available():
        pytest.skip("no shipped temporal model artifact available")
    scores_a_shared = compute_temporal_scores(rows_a, inf=shared)
    scores_b_shared = compute_temporal_scores(rows_b, inf=shared)

    scores_a_fresh = compute_temporal_scores(rows_a)
    scores_b_fresh = compute_temporal_scores(rows_b)

    assert scores_a_shared == scores_a_fresh
    assert scores_b_shared == scores_b_fresh


def test_load_held_out_test_subjects_parses_report_line(tmp_path):
    report = tmp_path / 'report.txt'
    report.write_text("SOME HEADER\nHELD_OUT_TEST_SUBJECTS: s9,s7\nmore text\n")
    assert load_held_out_test_subjects(str(report)) == {'s9', 's7'}


def test_load_held_out_test_subjects_returns_none_when_missing_or_absent(tmp_path):
    assert load_held_out_test_subjects(str(tmp_path / 'does_not_exist.txt')) is None

    report = tmp_path / 'no_line.txt'
    report.write_text("just some other report text, no held-out line\n")
    assert load_held_out_test_subjects(str(report)) is None


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
                row.update({name: 0.0 for name in TEMPORAL_EXTRA_COLUMNS})
                row.update({
                    't': i * 0.1, 'posture_detected': 1,
                    'daisee_raw': daisee_val, 'daisee_stress': daisee_val,
                    'behavior_stress': behavior_val, 'raw_index': 0.0,
                    'stress_index_frame': 0.0, 'stress_index_smoothed': 0.0,
                    'level': 'LOW', 'active_signals': 0, 'boost_applied': 0,
                })
                w.writerow(row)

    # behavior_val=0.15 is chosen so behavior's own contribution
    # (W_BEHAVIOR=0.80 x 0.15 = 0.12) stays safely below the LOW/MODERATE
    # cutoff (0.24, was 0.36 until 2026-09-27) on its own - only DaiSEE's
    # swing (0.05 -> 0.95) decides the label. A behavior_val large enough to
    # cross 0.24 by itself would make this scenario meaningless, since it
    # would classify both classes identically regardless of what DaiSEE says.
    for subj in ('s1', 's2', 's3'):
        write(tmp_path / f'features_vid_{subj}_T1.csv', 40, daisee_val=0.05, behavior_val=0.15)
        write(tmp_path / f'features_vid_{subj}_T2.csv', 40, daisee_val=0.95, behavior_val=0.15)

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


def test_run_ablation_study_subjects_filter_restricts_to_named_subjects(tmp_path):
    for subj in ('s1', 's2', 's3'):
        _write_trace(tmp_path / f'features_vid_{subj}_T1.csv', 40)
        _write_trace(tmp_path / f'features_vid_{subj}_T2.csv', 40)

    results = run_ablation_study(tmp_path, subjects={'s2'})
    assert results is not None
    for fr, se in results.values():
        assert fr['n'] == 80    # only s2: 2 files x 40 frames
        assert se['n'] == 2     # only s2: 2 conditions


def test_run_ablation_study_subjects_filter_empty_intersection_returns_none(tmp_path):
    _write_trace(tmp_path / 'features_vid_s1_T1.csv', 40)
    _write_trace(tmp_path / 'features_vid_s1_T2.csv', 40)

    assert run_ablation_study(tmp_path, subjects={'s99'}) is None
