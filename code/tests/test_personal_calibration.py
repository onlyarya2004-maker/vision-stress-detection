"""
Personal-calibration tests for posture and blink detection.

These cover the two fixes prompted by the first cross-subject run, where on
UBFC-Phys subject 1 (a RESTING subject) `slouch` sat at exactly 1.000 for
100% of frames and blink detection fired continuously.

Root cause in both cases was an absolute scale where a personal one was
needed:
  * posture had no baseline at all - the one branch calibration never touched
  * the blink cutoff was a fixed EAR of 0.21, but measured EAR was 0.159

Both fixes are designed so the UNCALIBRATED engine behaves exactly as before,
which the first test in each section pins down.
"""

import numpy as np
import pytest

import config
from conftest import (
    face_mesh_result, make_face_landmarks, make_pose_landmarks, pose_result,
)


# ===========================================================================
# Blink threshold
# ===========================================================================

def test_uncalibrated_blink_threshold_matches_the_old_constant(engine):
    """0.75 x the 0.28 default == 0.21, the previous hardcoded value."""
    assert engine.blink_threshold == pytest.approx(0.21)
    assert engine.blink_threshold == pytest.approx(config.EAR_BLINK_THRESHOLD)


def test_blink_threshold_follows_the_calibrated_baseline(engine):
    """A narrow-eyed subject gets a proportionally lower cutoff."""
    engine.baseline.ear_baseline = 0.159          # the UBFC-Phys s1 value
    assert engine.blink_threshold == pytest.approx(0.75 * 0.159, abs=1e-6)
    # Crucially, their resting eyes are no longer permanently "blinking".
    assert 0.159 > engine.blink_threshold


def test_open_eyes_are_not_counted_as_blinks_after_calibration(
        engine, monkeypatch, blank_frame):
    """
    Regression for the observed failure: EAR 0.159 against a fixed 0.21 cutoff
    registered every frame as a blink. With a calibrated baseline it must not.
    """
    engine._calibrating = False
    engine.baseline.calibrated = True
    engine.baseline.ear_baseline = 0.159

    lms = make_face_landmarks(ear=0.159)          # resting, eyes open
    monkeypatch.setattr(engine.face_mesh, "process",
                        lambda _f: face_mesh_result(lms))
    for _ in range(30):
        feats, _ = engine._extract_face(blank_frame)

    assert feats['blinks_per_minute'] < 5.0, (
        f"steady open eyes still read as blinking: "
        f"{feats['blinks_per_minute']:.1f}/min")


def test_genuine_blinks_are_still_detected(engine, monkeypatch, blank_frame):
    """The fix must not make the detector deaf."""
    engine._calibrating = False
    engine.baseline.calibrated = True
    engine.baseline.ear_baseline = 0.159

    open_lms = make_face_landmarks(ear=0.159)
    shut_lms = make_face_landmarks(ear=0.05)      # well below 0.75 x 0.159
    for i in range(40):
        lms = shut_lms if i % 10 == 0 else open_lms
        monkeypatch.setattr(engine.face_mesh, "process",
                            lambda _f, l=lms: face_mesh_result(l))
        feats, _ = engine._extract_face(blank_frame)

    assert feats['blinks_per_minute'] > 0.0


# ===========================================================================
# Posture calibration
# ===========================================================================

def _pose_at(engine, monkeypatch, blank_frame, nose_y, shoulder_y=0.70):
    """Score one frame with the nose at a given height."""
    lms = make_pose_landmarks(shoulder_visibility=0.95)
    lms[0].y = nose_y                       # NOSE_POSE
    lms[11].y = shoulder_y                  # LEFT_SHOULDER
    lms[12].y = shoulder_y                  # RIGHT_SHOULDER
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(lms))
    return engine._extract_posture(blank_frame, face_bbox=None)


def test_uncalibrated_posture_matches_the_old_absolute_formula(
        engine, monkeypatch, blank_frame):
    """
    Defaults reproduce the previous behaviour exactly:
        slouch = clip((1.0 - vertical_gap) / 0.6)
    """
    engine._calibrating = False
    lms = make_pose_landmarks(shoulder_visibility=0.95)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(lms))
    got = engine._extract_posture(blank_frame, face_bbox=None)

    # Recompute the old formula from the same landmark geometry.
    h = w = 640
    nose = np.array([0.50 * w, 0.35 * h])
    ls = np.array([0.35 * w, 0.70 * h])
    rs = np.array([0.65 * w, 0.70 * h])
    mid = (ls + rs) / 2
    width = np.linalg.norm(ls - rs) + 1e-6
    vertical_gap = (mid[1] - nose[1]) / width
    expected = float(np.clip((1.0 - vertical_gap) / 0.6, 0.0, 1.0))

    assert got['slouch'] == pytest.approx(expected, abs=1e-6)


def test_calibration_captures_posture_geometry(engine, monkeypatch,
                                               blank_frame):
    """Frames seen while calibrating must populate the posture baseline."""
    engine.start_calibration()
    face_lms = make_face_landmarks()
    monkeypatch.setattr(engine.face_mesh, "process",
                        lambda _f: face_mesh_result(face_lms))
    pose_lms = make_pose_landmarks(shoulder_visibility=0.95)
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(pose_lms))

    for _ in range(40):
        engine.extract(blank_frame)

    assert len(engine._posture_calib_buffer) >= config.MIN_POSTURE_CALIB_SAMPLES
    assert engine.finish_calibration() is True
    assert engine.baseline.posture_calibrated is True
    # The baseline must reflect the observed geometry, not the generic 1.0.
    assert engine.baseline.vertical_gap_baseline != pytest.approx(
        config.DEFAULT_VERTICAL_GAP_BASELINE)


def test_resting_posture_scores_near_zero_after_calibration(
        engine, monkeypatch, blank_frame):
    """
    THE regression test. Calibrate on a posture, then hold it: slouch must
    read ~0, not saturate. This is precisely what failed on UBFC-Phys, where
    an unfamiliar camera geometry pinned slouch at 1.000 for every frame.
    """
    face_lms = make_face_landmarks()
    monkeypatch.setattr(engine.face_mesh, "process",
                        lambda _f: face_mesh_result(face_lms))

    # A framing that saturates the OLD absolute formula.
    tight = make_pose_landmarks(shoulder_visibility=0.95)
    tight[0].y = 0.62          # nose close above the shoulders
    tight[11].y = 0.70
    tight[12].y = 0.70
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(tight))

    before = engine._extract_posture(blank_frame, face_bbox=None)
    assert before['slouch'] == pytest.approx(1.0), (
        "expected the old formula to saturate on this geometry")

    engine.start_calibration()
    for _ in range(40):
        engine.extract(blank_frame)
    engine.finish_calibration()

    after = engine._extract_posture(blank_frame, face_bbox=None)
    assert after['slouch'] < 0.05, (
        f"resting posture still scores {after['slouch']:.3f} after "
        f"calibrating on that very posture")


def test_actual_slouching_still_scores_after_calibration(
        engine, monkeypatch, blank_frame):
    """Personalising the baseline must not make the signal blind."""
    face_lms = make_face_landmarks()
    monkeypatch.setattr(engine.face_mesh, "process",
                        lambda _f: face_mesh_result(face_lms))

    upright = make_pose_landmarks(shoulder_visibility=0.95)
    upright[0].y = 0.30
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(upright))
    engine.start_calibration()
    for _ in range(40):
        engine.extract(blank_frame)
    engine.finish_calibration()

    # Now slump: the head drops towards the shoulders.
    slumped = _pose_at(engine, monkeypatch, blank_frame, nose_y=0.60)
    assert slumped['slouch'] > 0.15, (
        f"real slouching only scored {slumped['slouch']:.3f}")


def test_posture_not_calibrated_when_shoulders_never_visible(
        engine, monkeypatch, blank_frame):
    """
    Face calibration can succeed while posture calibration cannot. The flags
    are independent so the system knows which baselines it can trust.
    """
    face_lms = make_face_landmarks()
    monkeypatch.setattr(engine.face_mesh, "process",
                        lambda _f: face_mesh_result(face_lms))
    hidden = make_pose_landmarks(shoulder_visibility=0.2)   # below the gate
    monkeypatch.setattr(engine.pose, "process",
                        lambda _f: pose_result(hidden))

    engine.start_calibration()
    for _ in range(40):
        engine.extract(blank_frame)

    assert engine.finish_calibration() is True      # face still calibrated
    assert engine.baseline.posture_calibrated is False
    assert engine.baseline.vertical_gap_baseline == pytest.approx(
        config.DEFAULT_VERTICAL_GAP_BASELINE)
