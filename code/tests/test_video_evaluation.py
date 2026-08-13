"""
Offline video evaluation tests.

The important one is the media clock. Blink rate is derived from timestamps,
and a video file is consumed far faster than real time - with a wall clock,
every blink in a clip lands within a fraction of a second and
blinks_per_minute (which drives blink_stress, weight 0.15) becomes garbage.
These tests pin that behaviour down.
"""

import pytest

from conftest import face_mesh_result, make_face_landmarks
from evaluate_video_file import ubfc_meta


# --------------------------------------------------------------------------
# Injectable clock
# --------------------------------------------------------------------------

def test_engine_uses_injected_time_source(monkeypatch, blank_frame):
    """A supplied clock must be used instead of wall time."""
    from behavioral_features import BehavioralFeatureEngine

    fake = {'t': 100.0}
    eng = BehavioralFeatureEngine(time_source=lambda: fake['t'])
    try:
        landmarks = make_face_landmarks()
        monkeypatch.setattr(eng.face_mesh, "process",
                            lambda _f: face_mesh_result(landmarks))
        eng._calibrating = False
        eng.baseline.calibrated = True
        eng._extract_face(blank_frame)
        # The engine should have recorded activity against OUR clock.
        assert eng._time() == 100.0
    finally:
        eng.close()


def test_default_time_source_is_wall_clock():
    """Live webcam use must keep real time - the offline path is opt-in."""
    import time as time_mod
    from behavioral_features import BehavioralFeatureEngine

    eng = BehavioralFeatureEngine()
    try:
        assert abs(eng._time() - time_mod.time()) < 1.0
    finally:
        eng.close()


def test_media_clock_produces_sane_blink_rate(monkeypatch, blank_frame):
    """
    Simulate blinks at a known rate in MEDIA time and confirm the reported
    blinks-per-minute is plausible. With a wall clock this figure explodes,
    because all the frames are processed within milliseconds.
    """
    from behavioral_features import BehavioralFeatureEngine

    fps = 30.0
    frame = {'i': 0}
    eng = BehavioralFeatureEngine(time_source=lambda: frame['i'] / fps)
    try:
        eng._calibrating = False
        eng.baseline.calibrated = True

        open_lms = make_face_landmarks(ear=0.30)      # eyes open
        shut_lms = make_face_landmarks(ear=0.10)      # below 0.21 threshold

        # 30 seconds at 30fps, blinking once per second (~60 blinks/min)
        last = None
        for i in range(int(30 * fps)):
            frame['i'] = i
            blinking = (i % 30) == 0
            lms = shut_lms if blinking else open_lms
            monkeypatch.setattr(eng.face_mesh, "process",
                                lambda _f, l=lms: face_mesh_result(l))
            feats, _ = eng._extract_face(blank_frame)
            last = feats

        rate = last['blinks_per_minute']
        # ~60/min expected; allow a wide band, but it must not be absurd.
        assert 20 < rate < 200, f"implausible blink rate: {rate}"
    finally:
        eng.close()


# --------------------------------------------------------------------------
# UBFC-Phys filename mapping
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,subject,condition", [
    ("vid_s1_T1.avi",  "U01", "relaxed"),    # T1 = rest
    ("vid_s1_T2.avi",  "U01", "stressed"),   # T2 = speech
    ("vid_s1_T3.avi",  "U01", "stressed"),   # T3 = arithmetic
    ("vid_s42_T2.avi", "U42", "stressed"),
    ("s7_T1",          "U07", "relaxed"),
])
def test_ubfc_naming_maps_tasks_to_conditions(name, subject, condition):
    meta = ubfc_meta(name)
    assert meta is not None, f"failed to parse {name}"
    assert meta["subject"] == subject
    assert meta["condition"] == condition


@pytest.mark.parametrize("name", [
    "random_video.avi", "relaxed_S01_default_2026-08-12.csv", "IMG_1234.mp4",
])
def test_non_ubfc_names_are_rejected(name):
    """Anything not matching the scheme must return None, not a wrong guess."""
    assert ubfc_meta(name) is None
