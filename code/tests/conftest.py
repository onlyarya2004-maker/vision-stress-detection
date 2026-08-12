"""
Shared pytest fixtures.

The behavioral-feature tests need MediaPipe landmarks without a webcam, so
this module builds *synthetic* 478-point landmark sets with directly
controllable geometry (lip width, lip gap, brow distance, eye openness).
That lets the tests drive the real _extract_face() code path rather than a
reimplementation of it — which is the whole point of testing it.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# code/ must be importable — the modules import each other as flat siblings.
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

FRAME_W = FRAME_H = 640
N_LANDMARKS = 478          # refine_landmarks=True gives 478

# Landmark indices the engine actually reads (mirrors behavioral_features.py)
LEFT_EYE = [33, 160, 158, 133, 153, 144]     # p1,p2,p3,p4,p5,p6
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_INNER_BROW, RIGHT_INNER_BROW = 55, 285
UPPER_LIP, LOWER_LIP, LIP_LEFT, LIP_RIGHT = 13, 14, 61, 291
FACE_LEFT, FACE_RIGHT = 234, 454

# Face width is fixed so *_norm values map to predictable pixel distances.
_FACE_LEFT_X, _FACE_RIGHT_X = 0.2, 0.8
FACE_W_PX = (_FACE_RIGHT_X - _FACE_LEFT_X) * FRAME_W      # 384 px


def _lm(x, y, visibility=1.0):
    """One normalized landmark, shaped like MediaPipe's."""
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=visibility)


def make_face_landmarks(ear=0.28, brow_norm=0.18, lip_gap_norm=0.012,
                        lip_width_norm=0.42):
    """
    Build a synthetic 478-landmark face whose derived features equal the
    requested values. Defaults match BehavioralBaseline's defaults, i.e. a
    perfectly neutral face that should produce zero tension everywhere.
    """
    # Neutral filler so the face bounding box is sane.
    lms = [_lm(0.5, 0.5) for _ in range(N_LANDMARKS)]

    lms[FACE_LEFT] = _lm(_FACE_LEFT_X, 0.5)
    lms[FACE_RIGHT] = _lm(_FACE_RIGHT_X, 0.5)

    # Eyes. EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|); with both vertical pairs
    # equal to V and horizontal span W, EAR == V / W.
    eye_w_px = 40.0
    v_px = ear * eye_w_px
    for eye_idx, cx in ((LEFT_EYE, 0.35), (RIGHT_EYE, 0.65)):
        cy = 0.40
        half_w = (eye_w_px / 2) / FRAME_W
        half_v = (v_px / 2) / FRAME_H
        p1, p2, p3, p4, p5, p6 = eye_idx
        lms[p1] = _lm(cx - half_w, cy)          # outer corner
        lms[p4] = _lm(cx + half_w, cy)          # inner corner
        lms[p2] = _lm(cx - half_w / 2, cy - half_v)
        lms[p6] = _lm(cx - half_w / 2, cy + half_v)
        lms[p3] = _lm(cx + half_w / 2, cy - half_v)
        lms[p5] = _lm(cx + half_w / 2, cy + half_v)

    # Inner brows, separated horizontally by brow_norm * face_w.
    brow_px = brow_norm * FACE_W_PX
    half = (brow_px / 2) / FRAME_W
    lms[LEFT_INNER_BROW] = _lm(0.5 - half, 0.32)
    lms[RIGHT_INNER_BROW] = _lm(0.5 + half, 0.32)

    # Lips: vertical gap and horizontal width, both normalized by face_w.
    gap_px = lip_gap_norm * FACE_W_PX
    half_gap = (gap_px / 2) / FRAME_H
    lms[UPPER_LIP] = _lm(0.5, 0.70 - half_gap)
    lms[LOWER_LIP] = _lm(0.5, 0.70 + half_gap)

    width_px = lip_width_norm * FACE_W_PX
    half_w = (width_px / 2) / FRAME_W
    lms[LIP_LEFT] = _lm(0.5 - half_w, 0.70)
    lms[LIP_RIGHT] = _lm(0.5 + half_w, 0.70)

    return lms


def face_mesh_result(landmarks):
    """Mimic FaceMesh.process()'s return shape."""
    return SimpleNamespace(
        multi_face_landmarks=[SimpleNamespace(landmark=landmarks)]
    )


def make_pose_landmarks(shoulder_visibility=1.0, wrist_xy=None,
                        wrist_visibility=0.0):
    """
    33 pose landmarks. Shoulders are placed symmetrically below a centered
    nose; wrist_xy (normalized) optionally parks a wrist somewhere specific
    so the hand-near-face check can be exercised.
    """
    from behavioral_features import (
        LEFT_SHOULDER, RIGHT_SHOULDER, NOSE_POSE, LEFT_WRIST, RIGHT_WRIST,
    )
    lms = [_lm(0.5, 0.5, 0.0) for _ in range(33)]
    lms[NOSE_POSE] = _lm(0.50, 0.35, 1.0)
    lms[LEFT_SHOULDER] = _lm(0.35, 0.70, shoulder_visibility)
    lms[RIGHT_SHOULDER] = _lm(0.65, 0.70, shoulder_visibility)
    if wrist_xy is not None:
        lms[LEFT_WRIST] = _lm(wrist_xy[0], wrist_xy[1], wrist_visibility)
        lms[RIGHT_WRIST] = _lm(wrist_xy[0], wrist_xy[1], wrist_visibility)
    else:
        lms[LEFT_WRIST] = _lm(0.1, 0.95, wrist_visibility)
        lms[RIGHT_WRIST] = _lm(0.9, 0.95, wrist_visibility)
    return lms


def pose_result(landmarks):
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))


@pytest.fixture
def blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


@pytest.fixture
def engine():
    """
    A BehavioralFeatureEngine with its MediaPipe graphs closed afterwards.
    Tests monkeypatch .face_mesh.process / .pose.process, so no real
    inference runs.
    """
    from behavioral_features import BehavioralFeatureEngine
    eng = BehavioralFeatureEngine()
    yield eng
    eng.close()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """
    Point session_history at a throwaway database. Without this the tests
    would read and write the user's real ../data/sessions.db.
    """
    import session_history
    db = tmp_path / "test_sessions.db"
    monkeypatch.setattr(session_history, "DB_PATH", db)
    return db
