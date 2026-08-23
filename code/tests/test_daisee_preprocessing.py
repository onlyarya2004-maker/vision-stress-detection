"""
Tests for the pure label-scoring and crop-geometry helpers in
daisee_preprocessing.py. The heavy parts of this module (walking the real
DaiSEE dataset, running MediaPipe on real video frames) are out of scope -
these tests exercise only the deterministic math: the DaiSEE-labels -> 3
class stress score/bucket formula, filename/path indexing, and the crop
bounding-box math, using synthetic stand-ins instead of real images.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import daisee_preprocessing as dp


# --------------------------------------------------------------------------
# stress_score_from_daisee / stress_bucket
# --------------------------------------------------------------------------

def test_score_all_zero_labels():
    # Only the (1 - engagement/3) term contributes when everything else is 0.
    score = dp.stress_score_from_daisee(boredom=0, engagement=0, confusion=0, frustration=0)
    assert score == pytest.approx(0.15)


def test_score_all_max_labels():
    score = dp.stress_score_from_daisee(boredom=3, engagement=3, confusion=3, frustration=3)
    assert score == pytest.approx(0.85)


def test_score_weights_frustration_heaviest_then_confusion_then_boredom():
    fru_only = dp.stress_score_from_daisee(0, 0, 0, 3)
    con_only = dp.stress_score_from_daisee(0, 0, 3, 0)
    bor_only = dp.stress_score_from_daisee(3, 0, 0, 0)
    assert fru_only > con_only > bor_only > 0


def test_score_full_engagement_cancels_disengagement_term():
    score = dp.stress_score_from_daisee(boredom=0, engagement=3, confusion=0, frustration=0)
    assert score == pytest.approx(0.0)


def test_score_stays_within_unit_range():
    score = dp.stress_score_from_daisee(boredom=3, engagement=0, confusion=3, frustration=3)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("score,expected", [
    (0.0, 'low'), (0.32, 'low'), (0.329999, 'low'),
    (0.33, 'moderate'), (0.5, 'moderate'), (0.659999, 'moderate'),
    (0.66, 'high'), (0.9, 'high'), (1.0, 'high'),
])
def test_stress_bucket_thresholds(score, expected):
    assert dp.stress_bucket(score) == expected


# --------------------------------------------------------------------------
# build_clip_index
# --------------------------------------------------------------------------

def test_build_clip_index_maps_stem_to_path(tmp_path):
    (tmp_path / 'u1' / 'c1').mkdir(parents=True)
    (tmp_path / 'u2' / 'c2').mkdir(parents=True)
    avi = tmp_path / 'u1' / 'c1' / 'AAA111.avi'
    mp4 = tmp_path / 'u2' / 'c2' / 'BBB222.mp4'
    avi.write_bytes(b"")
    mp4.write_bytes(b"")

    index = dp.build_clip_index(tmp_path)

    assert index == {'AAA111': avi, 'BBB222': mp4}


def test_build_clip_index_missing_root_returns_empty(tmp_path):
    assert dp.build_clip_index(tmp_path / 'does_not_exist') == {}


# --------------------------------------------------------------------------
# crop_largest_face - synthetic landmarks stand in for real MediaPipe output
# --------------------------------------------------------------------------

class _FakeLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class _FakeFaceMesh:
    """Mimics mediapipe.FaceMesh's .process() return shape."""

    def __init__(self, landmarks):
        self._landmarks = landmarks

    def process(self, _rgb):
        if not self._landmarks:
            return SimpleNamespace(multi_face_landmarks=None)
        return SimpleNamespace(
            multi_face_landmarks=[SimpleNamespace(landmark=self._landmarks)]
        )


def _frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_crop_largest_face_returns_none_when_no_face_detected():
    assert dp.crop_largest_face(_frame(), _FakeFaceMesh(None)) is None


def test_crop_largest_face_resizes_to_img_size():
    landmarks = [_FakeLandmark(0.3, 0.2), _FakeLandmark(0.7, 0.6)]
    crop = dp.crop_largest_face(_frame(h=480, w=640), _FakeFaceMesh(landmarks))
    assert crop is not None
    assert crop.shape == (dp.IMG_SIZE, dp.IMG_SIZE, 3)


def test_crop_largest_face_rejects_degenerate_bbox():
    """All landmarks collapsed to one point -> bbox stays under the 20px
    floor even after padding, so no unusable crop should be produced."""
    landmarks = [_FakeLandmark(0.5, 0.5), _FakeLandmark(0.5, 0.5)]
    assert dp.crop_largest_face(_frame(), _FakeFaceMesh(landmarks)) is None


def test_crop_largest_face_clamps_padding_at_frame_edges():
    """Landmarks flush against the top-left corner must not push the padded
    bbox into negative indices."""
    landmarks = [_FakeLandmark(0.0, 0.0), _FakeLandmark(0.05, 0.1)]
    crop = dp.crop_largest_face(_frame(h=480, w=640), _FakeFaceMesh(landmarks))
    assert crop is not None
    assert crop.shape == (dp.IMG_SIZE, dp.IMG_SIZE, 3)


# --------------------------------------------------------------------------
# sample_frames - only the no-real-video-file edge case is testable here
# --------------------------------------------------------------------------

def test_sample_frames_missing_file_returns_empty_list(tmp_path):
    assert dp.sample_frames(tmp_path / 'missing.avi', 16) == []
