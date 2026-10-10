"""
Tests for session_pipeline.py (the one per-frame pipeline both capture modes
share) and the new frame_ops helpers.

No camera and no real face: FaceMesh/Pose are monkeypatched to return the
synthetic landmark sets from conftest (the same trick the behavioural-feature
tests use), and the CNN ensemble is stubbed. What these tests pin down is the
PLUMBING that hosting introduced - batch validation, the clock mapping, per-
user isolation, record throttling, pose cadence, calibration lifecycle - not
model accuracy (that is measured separately on real footage, see
experiment_mediapipe_tasks_parity.py).
"""
import base64
from datetime import datetime, timedelta
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import config
import daisee_ensemble
import frame_ops
import session_history
import session_pipeline as sp
from conftest import (
    face_mesh_result, make_face_landmarks, make_pose_landmarks, pose_result,
)
from intervention_agent import InterventionAgent
from session_pipeline import BatchGate, CalibrationRun, LiveRun, UserState


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _jpeg_b64(w=640, h=480, color=90):
    img = np.full((h, w, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


def _batch(epoch=1, seq=1, times=(0.1, 0.2), tz=None, jpeg=None):
    jpeg = jpeg or _jpeg_b64()
    value = {'epoch': epoch, 'seq': seq,
             'frames': [{'t': t, 'j': jpeg} for t in times]}
    if tz is not None:
        value['tz'] = tz
    return value


@pytest.fixture
def faked(monkeypatch):
    """A UserState whose MediaPipe graphs return a fixed synthetic face, with
    the CNN stubbed. Yields (state, calls) where calls counts pose runs."""
    state = UserState()
    calls = {'pose': 0, 'face': 0}
    face = face_mesh_result(make_face_landmarks())
    pose = pose_result(make_pose_landmarks())

    def fake_face(rgb):
        calls['face'] += 1
        return face

    def fake_pose(rgb):
        calls['pose'] += 1
        return pose

    monkeypatch.setattr(state.engine.face_mesh, 'process', fake_face)
    monkeypatch.setattr(state.engine.pose, 'process', fake_pose)
    monkeypatch.setattr(daisee_ensemble, 'ensemble_predict',
                        lambda crop, pairs: np.array([0.6, 0.4]))
    yield state, calls
    state.close()


def _frame():
    return np.full((480, 640, 3), 100, dtype=np.uint8)


def _live(state, **kw):
    agent = InterventionAgent(stats={}, mode_stats={}, task_mode='Default')
    return LiveRun(state, {'pairs': []}, agent, task_mode='Default',
                   agent_enabled=kw.pop('agent_enabled', False),
                   dt0=kw.pop('dt0', datetime(2026, 10, 10, 12, 0, 0)))


# ----------------------------------------------------------------------------
# frame_ops: geometry + decode
# ----------------------------------------------------------------------------
def test_prepare_processing_frame_passes_640x480_through_untouched():
    f = np.zeros((480, 640, 3), np.uint8)
    out, geom = frame_ops.prepare_processing_frame(f)
    assert out is f and geom == (0, 0, 640, 480)


def test_prepare_processing_frame_centre_crops_16_9_to_4_3():
    f = np.zeros((1080, 1920, 3), np.uint8)
    out, geom = frame_ops.prepare_processing_frame(f)
    assert out.shape == (480, 640, 3)
    assert geom == (240, 0, 1440, 1080)            # 1440x1080 is 4:3, centred


def test_prepare_processing_frame_trims_top_and_bottom_of_tall_frames():
    f = np.zeros((640, 480, 3), np.uint8)           # portrait
    out, geom = frame_ops.prepare_processing_frame(f)
    assert out.shape == (480, 640, 3)
    assert geom == (0, 140, 480, 360)


def test_bbox_round_trips_from_processed_frame_to_source_frame():
    geom = (240, 0, 1440, 1080)
    assert frame_ops.bbox_to_source((0, 0, 640, 480), geom) == (240, 0, 1680, 1080)
    assert frame_ops.bbox_to_source((320, 240, 640, 480), geom) == (960, 540, 1680, 1080)
    assert frame_ops.bbox_to_source(None, geom) is None


def test_decode_jpeg_rejects_garbage_oversize_and_tiny_images():
    good = base64.b64decode(_jpeg_b64())
    assert frame_ops.decode_jpeg(good).shape == (480, 640, 3)
    assert frame_ops.decode_jpeg(b'') is None
    assert frame_ops.decode_jpeg(None) is None
    assert frame_ops.decode_jpeg(b'not a jpeg at all') is None
    assert frame_ops.decode_jpeg(b'\x00' * (config.MAX_FRAME_BYTES + 1)) is None
    ok, tiny = cv2.imencode('.jpg', np.zeros((8, 8, 3), np.uint8))
    assert frame_ops.decode_jpeg(tiny.tobytes()) is None


# ----------------------------------------------------------------------------
# BatchGate: everything from a browser is untrusted
# ----------------------------------------------------------------------------
def test_gate_accepts_a_valid_batch_and_acks_it():
    g = BatchGate()
    frames, ack = g.accept(_batch(seq=1, times=(0.1, 0.2, 0.3), tz=330), epoch=1)
    assert [round(t, 1) for t, _ in frames] == [0.1, 0.2, 0.3]
    assert ack == 1 and frames[0][1].shape == (480, 640, 3)
    assert g.tz_offset_min == 330


def test_gate_user_now_applies_the_browsers_utc_offset():
    g = BatchGate()
    g.accept(_batch(tz=330), epoch=1)
    delta = g.user_now() - datetime.utcnow()
    assert abs(delta - timedelta(minutes=330)) < timedelta(seconds=5)


def test_gate_drops_redelivered_batches():
    """The component re-delivers its LAST value on every rerun it didn't
    cause - the same (epoch, seq) must never be processed twice."""
    g = BatchGate()
    v = _batch(seq=5)
    assert len(g.accept(v, 1)[0]) == 2
    frames, ack = g.accept(v, 1)
    assert frames == [] and ack == 5


def test_gate_ignores_batches_from_a_previous_epoch_and_resets_on_a_new_one():
    g = BatchGate()
    g.accept(_batch(epoch=1, seq=7), 1)
    frames, ack = g.accept(_batch(epoch=1, seq=8), 2)       # stale: Stop was pressed
    assert frames == [] and ack == 0
    frames, ack = g.accept(_batch(epoch=2, seq=1), 2)       # fresh run restarts at 1
    assert len(frames) == 2 and ack == 1


def test_gate_drops_out_of_order_and_repeated_timestamps():
    g = BatchGate()
    g.accept(_batch(seq=1, times=(1.0, 2.0)), 1)
    frames, _ = g.accept(_batch(seq=2, times=(1.5, 2.0, 2.5)), 1)
    assert [t for t, _ in frames] == [2.5]
    assert g.dropped >= 2


@pytest.mark.parametrize('bad_t', [True, float('nan'), float('inf'), '1.0', None])
def test_gate_rejects_non_finite_or_non_numeric_timestamps(bad_t):
    g = BatchGate()
    v = {'epoch': 1, 'seq': 1, 'frames': [{'t': bad_t, 'j': _jpeg_b64()}]}
    assert g.accept(v, 1)[0] == []


def test_gate_survives_hostile_payloads():
    g = BatchGate()
    huge = 'A' * (config.MAX_FRAME_BYTES * 2)
    for value in (None, 'x', 5, [], {'epoch': 1}, {'epoch': 'one', 'seq': 1},
                  {'epoch': 1, 'seq': 'x'}, {'epoch': 1, 'seq': 1, 'frames': 'abc'},
                  {'epoch': 1, 'seq': 2, 'frames': [None, 5, 'x', {}]},
                  {'epoch': 1, 'seq': 3, 'frames': [{'t': 1.0, 'j': '!!!not base64!!!'}]},
                  {'epoch': 1, 'seq': 4, 'frames': [{'t': 2.0, 'j': huge}]},
                  {'epoch': 1, 'seq': 5, 'frames': [{'t': 3.0, 'j': base64.b64encode(b'junk').decode()}]}):
        frames, _ = g.accept(value, 1)
        assert frames == []


def test_gate_keeps_only_the_newest_frames_when_the_server_is_behind():
    g = BatchGate()
    n = config.MAX_FRAMES_PER_BATCH
    times = [0.1 * (i + 1) for i in range(n)]
    frames, _ = g.accept(_batch(seq=1, times=times), 1)
    assert len(frames) == config.MAX_FRAMES_PROCESSED_PER_BATCH
    assert frames[-1][0] == pytest.approx(times[-1])        # newest kept
    assert g.dropped == n - config.MAX_FRAMES_PROCESSED_PER_BATCH


def test_gate_caps_an_oversized_batch_before_decoding():
    g = BatchGate()
    times = [0.01 * (i + 1) for i in range(config.MAX_FRAMES_PER_BATCH * 3)]
    frames, _ = g.accept(_batch(seq=1, times=times), 1)
    assert len(frames) <= config.MAX_FRAMES_PROCESSED_PER_BATCH


# ----------------------------------------------------------------------------
# UserState: per-user isolation
# ----------------------------------------------------------------------------
def test_two_user_states_never_share_baselines_or_buffers():
    a, b = UserState(), UserState()
    try:
        a.daisee_baseline.set(0.9)
        a.engine.baseline.ear_baseline = 0.123
        assert not b.daisee_baseline.is_calibrated()
        assert b.engine.baseline.ear_baseline != 0.123
        assert a.engine is not b.engine and a.rppg is not b.rppg
        assert a.temporal_inf is not b.temporal_inf
        assert a.daisee_baseline is not b.daisee_baseline
    finally:
        a.close(); b.close()


def test_apply_profile_round_trips_and_none_resets_to_defaults():
    a, b = UserState(), UserState()
    try:
        a.engine.baseline.ear_baseline = 0.31
        a.engine.baseline.calibrated = True
        a.daisee_baseline.set(0.7)
        payload = a.baseline_payload()
        b.apply_profile(payload)
        assert b.engine.baseline.ear_baseline == pytest.approx(0.31)
        assert b.daisee_baseline.value == pytest.approx(0.7) and b.fully_calibrated
        b.apply_profile(None)
        assert not b.fully_calibrated
        assert b.engine.baseline.ear_baseline != 0.31
    finally:
        a.close(); b.close()


# ----------------------------------------------------------------------------
# LiveRun
# ----------------------------------------------------------------------------
def _drive(run, n=60, fps=10.0, base=0.0):
    outs = []
    for i in range(n):
        outs.append(run.process(_frame(), base + i / fps))
    return outs


def test_live_run_is_independent_of_the_browsers_clock_origin(faked):
    """The browser's timestamps start near 0 (performance.now since the run
    began); the engine's anchors are wall-clock. Whatever origin the client
    uses, the scores must come out identical."""
    state, _ = faked
    run_a = _live(state)
    series_a = [o['smoothed'] for o in _drive(run_a, base=0.0)]

    state2 = UserState()
    try:
        state2.engine.face_mesh.process = state.engine.face_mesh.process
        state2.engine.pose.process = state.engine.pose.process
        run_b = _live(state2)
        series_b = [o['smoothed'] for o in _drive(run_b, base=98765.4)]
    finally:
        state2.close()
    np.testing.assert_allclose(series_a, series_b, atol=1e-9)


def test_live_run_maps_client_time_onto_one_server_clock(faked):
    state, _ = faked
    run = _live(state)
    _drive(run, n=41, fps=10.0, base=1000.0)               # 4.0 s of video
    assert run.elapsed() == pytest.approx(4.0, abs=1e-6)
    assert state.clock['t'] - run.t0 == pytest.approx(4.0, abs=1e-6)
    assert abs(run.t0 - sp.time.time()) < 60               # wall-clock domain


def test_live_run_throttles_history_rows_to_one_per_half_second(faked):
    state, _ = faked
    run = _live(state, dt0=datetime(2026, 10, 10, 9, 30, 0))
    records = [o['record'] for o in _drive(run, n=51, fps=10.0) if o['record']]
    assert 9 <= len(records) <= 11                         # ~5 s at 2 rows/s
    gaps = [(b['ts'] - a['ts']).total_seconds() for a, b in zip(records, records[1:])]
    assert all(g >= 0.5 - 1e-9 for g in gaps)
    assert records[0]['ts'] == datetime(2026, 10, 10, 9, 30, 0)       # run time zero
    r = records[0]
    for key in ('stress_index', 'level', 'behavior', 'ear', 'ear_asymmetry',
                'blinks_per_minute', *session_history.SIGNAL_COLUMNS):
        assert key in r
    assert isinstance(r['daisee'], float)


def test_live_run_records_none_not_zero_when_the_cnn_branch_has_no_score(faked, monkeypatch):
    """session_history's v7->v8 fix: "CNN said 0" and "CNN gave no score" must
    stay distinguishable in the stored rows."""
    state, _ = faked
    monkeypatch.setattr(daisee_ensemble, 'ensemble_predict', lambda crop, pairs: None)
    run = _live(state)
    records = [o['record'] for o in _drive(run, n=30) if o['record']]
    assert records and all(r['daisee'] is None for r in records)


def test_live_run_reports_no_face_without_feeding_the_models(faked, monkeypatch):
    state, calls = faked
    monkeypatch.setattr(state.engine.face_mesh, 'process',
                        lambda rgb: SimpleNamespace(multi_face_landmarks=[]))
    run = _live(state)
    out = run.process(_frame(), 0.0)
    assert out['no_face'] is True and 'geom' in out
    assert run.latest is None


def test_live_run_finish_is_safe_before_any_frame(faked):
    state, _ = faked
    run = _live(state)
    assert run.finish() == []
    assert run.agent_status() is None


# ----------------------------------------------------------------------------
# Pose cadence (the server-CPU lever)
# ----------------------------------------------------------------------------
def test_pose_runs_every_frame_by_default_exactly_like_the_validated_path(faked, monkeypatch):
    state, calls = faked
    monkeypatch.setattr(config, 'POSE_EVERY_N_FRAMES', 1)
    run = _live(state)
    _drive(run, n=9)
    assert calls['pose'] == 9


def test_pose_cadence_n_runs_pose_every_nth_frame_and_reuses_it(faked, monkeypatch):
    state, calls = faked
    monkeypatch.setattr(config, 'POSE_EVERY_N_FRAMES', 3)
    run = _live(state)
    outs = _drive(run, n=9)
    assert calls['pose'] == 3
    assert all(not o['no_face'] for o in outs)              # stale pose still scores


# ----------------------------------------------------------------------------
# CalibrationRun
# ----------------------------------------------------------------------------
def test_calibration_completes_after_the_configured_seconds_and_sets_baselines(faked):
    state, _ = faked
    run = CalibrationRun(state, {'pairs': []})
    assert not run.done and run.elapsed() == 0.0
    t, fps = 500.0, 10.0
    i = 0
    while not run.done:
        info = run.process(_frame(), t + i / fps)
        assert info['no_face'] is False and info['bbox'] is not None
        i += 1
        assert i < 400
    assert run.elapsed() >= config.CALIBRATION_SECONDS
    assert run.progress() == 1.0 and run.seconds_left() == 0.0
    face_ok, daisee_ok = run.finish()
    assert face_ok and daisee_ok and state.fully_calibrated


def test_calibration_abort_clears_the_engines_calibrating_flag(faked):
    state, _ = faked
    run = CalibrationRun(state, {'pairs': []})
    run.process(_frame(), 0.0)
    assert state.engine._calibrating is True
    run.abort()
    assert state.engine._calibrating is False


def test_calibration_without_a_face_collects_nothing_and_fails_cleanly(faked, monkeypatch):
    state, _ = faked
    monkeypatch.setattr(state.engine.face_mesh, 'process',
                        lambda rgb: SimpleNamespace(multi_face_landmarks=None))
    run = CalibrationRun(state, {'pairs': []})
    for i in range(250):
        info = run.process(_frame(), i / 10.0)
        assert info['no_face'] is True and info['bbox'] is None
    face_ok, daisee_ok = run.finish()
    assert not face_ok and not daisee_ok and not state.fully_calibrated


# ----------------------------------------------------------------------------
# Capacity guard
# ----------------------------------------------------------------------------
def test_active_count_tracks_live_sessions_and_frees_them_on_release():
    import gc
    gc.collect()                      # sessions left over from earlier tests must not skew the baseline
    base = UserState.active_count()
    a, b = UserState(), UserState()
    assert UserState.active_count() == base + 2
    a.close()
    del a, b
    gc.collect()
    assert UserState.active_count() == base            # released sessions stop counting


def test_a_dropped_session_is_freed_without_waiting_for_the_cyclic_gc():
    """UserState must not reference itself (the engine's clock closes over the
    dict, not self), otherwise a finished login would keep counting against the
    capacity cap until some later GC pass."""
    import gc
    gc.collect()                      # as above: start from a clean count
    gc.disable()
    try:
        base = UserState.active_count()
        s = UserState()
        assert UserState.active_count() == base + 1
        del s                                           # plain refcount release, no gc.collect()
        assert UserState.active_count() == base
    finally:
        gc.enable()
