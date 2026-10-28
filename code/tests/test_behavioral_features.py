"""
Behavioral feature engine tests.

These drive the real _extract_face() / _extract_posture() code paths with
synthetic landmarks (see conftest.py), so the three documented fixes are
verified as implemented rather than as described:

  1. smile detection      — a smile must not be scored as jaw tension
  2. pose-visibility gate — occluded shoulders must not produce garbage posture
  3. hand-near-face damp  — a hand by the face must damp posture, not spike it
"""

import pytest

import config
from conftest import (
    face_mesh_result, make_face_landmarks, make_pose_landmarks, pose_result,
)


def _calibrated(engine):
    """Put the engine in the post-calibration state with default baselines."""
    engine._calibrating = False
    engine.baseline.calibrated = True
    engine.baseline.ear_baseline = 0.28
    engine.baseline.brow_distance_norm = 0.18
    engine.baseline.lip_gap_norm = 0.012
    engine.baseline.lip_width_norm = 0.42
    return engine


def _face_features(engine, monkeypatch, blank_frame, **geometry):
    landmarks = make_face_landmarks(**geometry)
    monkeypatch.setattr(engine.face_mesh, "process",
                        lambda _frame: face_mesh_result(landmarks))
    feats, _bbox = engine._extract_face(blank_frame)
    return feats


# --------------------------------------------------------------------------
# 1. Smile handling
# --------------------------------------------------------------------------

def test_smile_is_not_scored_as_jaw_tension(engine, monkeypatch, blank_frame):
    """
    A smile widens the lips while keeping/increasing the vertical gap. The
    original formula read that as a grimace; the fix must score it as zero.
    """
    _calibrated(engine)
    feats = _face_features(engine, monkeypatch, blank_frame,
                           lip_width_norm=0.50,   # ~19% wider than baseline
                           lip_gap_norm=0.014)    # gap not reduced
    assert feats["jaw_tension"] == 0.0


def test_clenched_jaw_still_scores(engine, monkeypatch, blank_frame):
    """The smile fix must not suppress genuine clenching."""
    _calibrated(engine)
    feats = _face_features(engine, monkeypatch, blank_frame,
                           lip_width_norm=0.35,   # narrower — compression
                           lip_gap_norm=0.005)    # pressed together
    assert feats["jaw_tension"] > 0.0


def test_neutral_face_produces_no_tension(engine, monkeypatch, blank_frame):
    """Baseline geometry must score ~zero on every tension signal."""
    _calibrated(engine)
    feats = _face_features(engine, monkeypatch, blank_frame)  # all defaults
    assert feats["jaw_tension"] == pytest.approx(0.0, abs=1e-6)
    assert feats["brow_tension"] == pytest.approx(0.0, abs=1e-6)
    assert feats["eye_strain"] == pytest.approx(0.0, abs=1e-6)


def test_furrowed_brow_scores_tension(engine, monkeypatch, blank_frame):
    """Brows drawn together (smaller normalized distance) => tension > 0."""
    _calibrated(engine)
    feats = _face_features(engine, monkeypatch, blank_frame, brow_norm=0.15)
    assert feats["brow_tension"] > 0.0


# --------------------------------------------------------------------------
# 2. Pose visibility gate
# --------------------------------------------------------------------------

def test_posture_gated_when_shoulders_not_visible(engine, monkeypatch,
                                                  blank_frame):
    """
    Below the 0.7 visibility threshold, posture must report zeros and flag
    itself unavailable rather than emit values from unreliable coordinates.
    """
    landmarks = make_pose_landmarks(shoulder_visibility=0.4)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(landmarks))

    posture = engine._extract_posture(blank_frame, face_bbox=None)

    assert posture["neck_strain"] == 0.0
    assert posture["slouch"] == 0.0
    assert posture["posture_detected"] is False
    assert posture["pose_visibility_low"] is True


def test_posture_computed_when_shoulders_visible(engine, monkeypatch,
                                                 blank_frame):
    """Above the threshold, posture is computed and marked available."""
    landmarks = make_pose_landmarks(shoulder_visibility=0.95)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(landmarks))

    posture = engine._extract_posture(blank_frame, face_bbox=None)

    assert posture["posture_detected"] is True
    assert 0.0 <= posture["neck_strain"] <= 1.0
    assert 0.0 <= posture["slouch"] <= 1.0


def test_missing_pose_landmarks_degrade_safely(engine, monkeypatch,
                                               blank_frame):
    """No pose detected at all must not raise."""
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(None).__class__(
                            pose_landmarks=None))
    posture = engine._extract_posture(blank_frame, face_bbox=None)
    assert posture["posture_detected"] is False


# --------------------------------------------------------------------------
# 3. Hand-near-face damping
# --------------------------------------------------------------------------

def test_hand_near_face_damps_posture_scores(engine, monkeypatch,
                                             blank_frame):
    """
    A visible wrist inside the padded face box must scale posture down by
    0.3x — chin-resting and yawning are not stress postures.
    """
    face_bbox = (200.0, 150.0, 440.0, 400.0)   # pixels, 640x640 frame

    clear = make_pose_landmarks(shoulder_visibility=0.95)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(clear))
    undamped = engine._extract_posture(blank_frame, face_bbox=face_bbox)

    # Wrist parked in the middle of the face box, clearly visible.
    near = make_pose_landmarks(shoulder_visibility=0.95,
                               wrist_xy=(0.50, 0.43), wrist_visibility=0.9)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(near))
    damped = engine._extract_posture(blank_frame, face_bbox=face_bbox)

    assert damped["neck_strain"] == pytest.approx(
        undamped["neck_strain"] * 0.3)
    assert damped["slouch"] == pytest.approx(undamped["slouch"] * 0.3)


def test_low_visibility_wrist_does_not_trigger_damping(engine, monkeypatch,
                                                       blank_frame):
    """A wrist the model is unsure about (<0.5) must be ignored."""
    face_bbox = (200.0, 150.0, 440.0, 400.0)

    clear = make_pose_landmarks(shoulder_visibility=0.95)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(clear))
    undamped = engine._extract_posture(blank_frame, face_bbox=face_bbox)

    ghost = make_pose_landmarks(shoulder_visibility=0.95,
                                wrist_xy=(0.50, 0.43), wrist_visibility=0.2)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(ghost))
    result = engine._extract_posture(blank_frame, face_bbox=face_bbox)

    assert result["neck_strain"] == pytest.approx(undamped["neck_strain"])


def test_baseline_reset_clears_calibration(engine):
    """reset_baseline() must return the engine to the uncalibrated state."""
    _calibrated(engine)
    assert engine.baseline.calibrated is True
    engine.reset_baseline()
    assert engine.baseline.calibrated is False


# --------------------------------------------------------------------------
# 4. PERCLOS (Driving mode, Phase 2.3)
# --------------------------------------------------------------------------
# These need a controllable clock (real elapsed test time is a few
# milliseconds, far too short to exercise a 60s window), so they build their
# own engine with a synthetic time_source rather than using the shared
# `engine` fixture - same clock-dict pattern train_temporal_model.py uses.

def _clock_engine():
    from behavioral_features import BehavioralFeatureEngine
    clock = {'t': 0.0}
    eng = BehavioralFeatureEngine(time_source=lambda: clock['t'])
    return eng, clock


def _feed(engine, monkeypatch, blank_frame, ear):
    landmarks = make_face_landmarks(ear=ear)
    monkeypatch.setattr(engine.face_mesh, "process",
                        lambda _frame: face_mesh_result(landmarks))
    feats, _bbox = engine._extract_face(blank_frame)
    return feats


def test_perclos_zero_when_eyes_stay_open(monkeypatch, blank_frame):
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        feats = None
        for i in range(10):
            clock['t'] = i * 0.5
            feats = _feed(engine, monkeypatch, blank_frame, ear=0.28)  # baseline, open
        assert feats["perclos"] == pytest.approx(0.0)
    finally:
        engine.close()


def test_perclos_high_when_eyes_stay_closed(monkeypatch, blank_frame):
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        feats = None
        for i in range(10):
            clock['t'] = i * 0.5
            # <=20% of the 0.28 baseline = <=0.056 - well past "closed".
            feats = _feed(engine, monkeypatch, blank_frame, ear=0.02)
        assert feats["perclos"] == pytest.approx(1.0)
    finally:
        engine.close()


def test_perclos_reflects_mixed_open_closed_ratio(monkeypatch, blank_frame):
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        feats = None
        for i in range(10):
            clock['t'] = i * 0.5
            ear = 0.02 if i % 2 == 0 else 0.28   # alternating closed/open
            feats = _feed(engine, monkeypatch, blank_frame, ear=ear)
        assert feats["perclos"] == pytest.approx(0.5)
    finally:
        engine.close()


def test_perclos_window_evicts_old_samples(monkeypatch, blank_frame):
    """
    A closed stretch that has scrolled out of the 60s window must stop
    affecting PERCLOS - otherwise a single old drowsy moment would pin the
    metric high for the rest of the session.
    """
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        clock['t'] = 0.0
        for _ in range(5):
            _feed(engine, monkeypatch, blank_frame, ear=0.02)  # closed, long ago

        clock['t'] = 120.0  # 2 minutes later - the closed run is now stale
        feats = None
        for i in range(5):
            clock['t'] = 120.0 + i
            feats = _feed(engine, monkeypatch, blank_frame, ear=0.28)  # open
        assert feats["perclos"] == pytest.approx(0.0)
    finally:
        engine.close()


def test_perclos_present_during_calibration(monkeypatch, blank_frame):
    """PERCLOS is a raw diagnostic like ear/blinks_per_minute - computed
    during calibration too, unlike the zeroed derived stress signals."""
    engine, clock = _clock_engine()
    engine._calibrating = True
    try:
        clock['t'] = 0.0
        feats = _feed(engine, monkeypatch, blank_frame, ear=0.28)
        assert feats["calibrating"] is True
        assert "perclos" in feats
        assert 0.0 <= feats["perclos"] <= 1.0
    finally:
        engine.close()


def test_finish_calibration_clears_perclos_window(monkeypatch, blank_frame):
    """A stale pre-calibration window must not leak into calibrated scoring."""
    engine, clock = _clock_engine()
    engine.start_calibration()
    try:
        for i in range(config.MIN_CALIBRATION_SAMPLES + 5):
            clock['t'] = i * 0.1
            _feed(engine, monkeypatch, blank_frame, ear=0.02)  # closed throughout
        assert len(engine.state.perclos_samples) > 0
        engine.finish_calibration()
        assert len(engine.state.perclos_samples) == 0
    finally:
        engine.close()


# --------------------------------------------------------------------------
# 5. Blink-rate extrapolation warm-up (BUG_AUDIT_2026-08-23.md finding #3)
# --------------------------------------------------------------------------
# Reproduces the exact failure: extrapolating "blinks so far / elapsed" to a
# full-minute rate is wildly unstable when the window has only just reset -
# one blink 0.5s in used to read as 120/min, pinning blink_stress to its
# ceiling from ordinary blinking at the start of every session.

def _blink(engine, monkeypatch, blank_frame, closed):
    """One frame, eyes open (ear=0.28) or fully shut (ear=0.02)."""
    return _feed(engine, monkeypatch, blank_frame, ear=0.02 if closed else 0.28)


def test_one_blink_right_after_reset_does_not_spike_the_rate(monkeypatch, blank_frame):
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        clock['t'] = 0.0
        _blink(engine, monkeypatch, blank_frame, closed=True)   # blink registers
        clock['t'] = 0.5
        feats = _blink(engine, monkeypatch, blank_frame, closed=False)
        # Old formula: 1 blink / 0.5s elapsed -> 120/min. Fixed: too little
        # real time to trust an extrapolation from one blink, so it reports
        # the calibrated baseline (12/min default) instead of a spike.
        assert feats['blinks_per_minute'] == pytest.approx(
            engine.baseline.blink_rate_baseline)
    finally:
        engine.close()


def test_zero_blinks_reads_as_zero_even_during_warmup(monkeypatch, blank_frame):
    """No blinks observed yet is always correctly zero - there's nothing to
    over-extrapolate FROM, warm-up or not."""
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        clock['t'] = 0.0
        feats = None
        for i in range(5):
            clock['t'] = i * 0.1
            feats = _blink(engine, monkeypatch, blank_frame, closed=False)
        assert feats['blinks_per_minute'] == pytest.approx(0.0)
    finally:
        engine.close()


def test_blink_rate_converges_to_true_rate_after_warmup(monkeypatch, blank_frame):
    """Once real time has actually passed, the rate should reflect the
    genuinely-observed blink count again, not stay pinned to the baseline."""
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        clock['t'] = 0.0
        _blink(engine, monkeypatch, blank_frame, closed=True)
        clock['t'] = config.BLINK_RATE_LIVE_WARMUP_SECONDS + 1.0
        feats = _blink(engine, monkeypatch, blank_frame, closed=False)
        expected = 1 * (60.0 / (config.BLINK_RATE_LIVE_WARMUP_SECONDS + 1.0))
        assert feats['blinks_per_minute'] == pytest.approx(expected)
    finally:
        engine.close()


def test_finish_calibration_resets_the_blink_warmup_window(monkeypatch, blank_frame):
    """A blink recorded during calibration must not count toward the live
    window's warm-up clock once calibration finishes."""
    engine, clock = _clock_engine()
    engine.start_calibration()
    try:
        for i in range(config.MIN_CALIBRATION_SAMPLES + 5):
            clock['t'] = i * 0.1
            _blink(engine, monkeypatch, blank_frame, closed=False)
        clock['t'] += config.MIN_BLINK_CALIB_SECONDS
        engine.finish_calibration()

        clock['t'] += 0.5
        _blink(engine, monkeypatch, blank_frame, closed=True)
        clock['t'] += 0.5
        feats = _blink(engine, monkeypatch, blank_frame, closed=False)
        # Still within warm-up of the POST-calibration window (only 1s has
        # passed since finish_calibration), so this must be the baseline,
        # not an extrapolation from calibration-era elapsed time.
        assert feats['blinks_per_minute'] == pytest.approx(
            engine.baseline.blink_rate_baseline)
    finally:
        engine.close()


# --------------------------------------------------------------------------
# 6. Per-user state reset (BUG_AUDIT_2026-08-23.md finding #5)
# --------------------------------------------------------------------------

def test_reset_state_clears_rolling_detection_history(monkeypatch, blank_frame):
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        clock['t'] = 0.0
        _blink(engine, monkeypatch, blank_frame, closed=True)
        clock['t'] = 1.0
        _blink(engine, monkeypatch, blank_frame, closed=False)
        assert len(engine.state.blink_times) > 0
        assert len(engine.state.ear_history) > 0

        clock['t'] = 2.0
        engine.reset_state()

        assert len(engine.state.blink_times) == 0
        assert len(engine.state.ear_history) == 0
        assert len(engine.state.head_tilt_history) == 0
        assert len(engine.state.perclos_samples) == 0
        assert engine.state.blink_window_started_at == pytest.approx(2.0)
    finally:
        engine.close()


def test_reset_state_does_not_touch_the_calibrated_baseline(monkeypatch, blank_frame):
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        engine.reset_state()
        assert engine.baseline.calibrated is True
        assert engine.baseline.ear_baseline == pytest.approx(0.28)
    finally:
        engine.close()


def test_reset_baseline_also_clears_detection_state(monkeypatch, blank_frame):
    """'Reset baseline' must fully live up to its name - previously it left
    stale blink/EAR history behind despite blowing away the baseline."""
    engine, clock = _clock_engine()
    _calibrated(engine)
    try:
        clock['t'] = 0.0
        _blink(engine, monkeypatch, blank_frame, closed=True)
        assert len(engine.state.blink_times) > 0

        engine.reset_baseline()

        assert len(engine.state.blink_times) == 0
        assert engine.baseline.calibrated is False
    finally:
        engine.close()


def test_jaw_tension_is_capped_so_it_cannot_saturate(engine, monkeypatch, blank_frame):
    """Jaw tension used to peg at 1.0 on ordinary closed lips and pushed every
    score upward (config.JAW_TENSION_CAP); an extreme clench now tops out at the
    cap, while a mild one is untouched."""
    import config
    _calibrated(engine)
    extreme = _face_features(engine, monkeypatch, blank_frame,
                             lip_width_norm=0.20, lip_gap_norm=0.0)
    assert extreme["jaw_tension"] == pytest.approx(config.JAW_TENSION_CAP)
    mild = _face_features(engine, monkeypatch, blank_frame,
                          lip_width_norm=0.40, lip_gap_norm=0.011)
    assert 0.0 < mild["jaw_tension"] < config.JAW_TENSION_CAP
