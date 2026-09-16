"""
crop_face() tests - covers all 4 independent copies of this function.

crop_face() is deliberately duplicated (not shared via import) across
app.py, stress_detector.py, benchmark_inference.py, and
evaluate_video_file.py, so each entry point stays free of a Streamlit
dependency (see benchmark_inference.py's own build notes). That
duplication had ZERO test coverage anywhere before this file - a bug or
drift introduced in any one copy's padding/clamping math would have gone
completely undetected, since this function feeds the exact pixels the CNN
branch sees. app.py's and benchmark_inference.py's copies return the
bounding box too (for UI overlay drawing / diagnostics) as a second tuple
element; stress_detector.py's and evaluate_video_file.py's copies return
the bare crop array only - both shapes are exercised here.

Uses conftest.py's make_face_landmarks()/face_mesh_result() synthetic
landmark builder (already proven out by test_behavioral_features.py) so
these drive the real landmark-to-pixel-box code path, not a
reimplementation of it.
"""

import numpy as np
import pytest

import app
import benchmark_inference
import evaluate_video_file
import stress_detector
from conftest import face_mesh_result, make_face_landmarks

FRAME_W = FRAME_H = 640

# (module, returns_bbox) - app.py's and benchmark_inference.py's crop_face
# return (crop, bbox); stress_detector.py's and evaluate_video_file.py's
# return the bare crop array.
IMPLS = [
    (app, True),
    (stress_detector, False),
    (benchmark_inference, True),
    (evaluate_video_file, False),
]


def _blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _crop_only(module, returns_bbox, frame, mesh_result):
    result = module.crop_face(frame, mesh_result)
    if returns_bbox:
        return result[0]
    return result


@pytest.mark.parametrize("module,returns_bbox", IMPLS)
def test_no_face_returns_none(module, returns_bbox):
    empty = face_mesh_result([])
    empty.multi_face_landmarks = []       # no detected face at all
    frame = _blank_frame()
    result = module.crop_face(frame, empty)
    if returns_bbox:
        assert result == (None, None)
    else:
        assert result is None


@pytest.mark.parametrize("module,returns_bbox", IMPLS)
def test_crop_covers_the_full_face_bounding_box(module, returns_bbox):
    """A face spanning known normalized coordinates must produce a crop
    whose shape matches the padded bounding box, computed independently
    here rather than by calling the module's own math back at itself."""
    landmarks = make_face_landmarks()
    mesh = face_mesh_result(landmarks)
    frame = _blank_frame()

    crop = _crop_only(module, returns_bbox, frame, mesh)
    assert crop is not None

    # Hand-compute the expected box the same way conftest placed FACE_LEFT/
    # FACE_RIGHT (0.2, 0.8 normalized) - other landmarks sit within this
    # x-range and at y=0.5 or narrower, so the face-left/right pair bounds
    # the box on x; y bounds come from the full landmark spread (min/max
    # over all 478 points, most of which sit at the y=0.5 filler value).
    xs = np.array([p.x for p in landmarks]) * FRAME_W
    ys = np.array([p.y for p in landmarks]) * FRAME_H
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    exp_x1, exp_y1 = max(0, x1 - pad), max(0, y1 - pad)
    exp_x2, exp_y2 = min(FRAME_W, x2 + pad), min(FRAME_H, y2 + pad)

    assert crop.shape[0] == exp_y2 - exp_y1
    assert crop.shape[1] == exp_x2 - exp_x1


@pytest.mark.parametrize("module,returns_bbox", IMPLS)
def test_bounding_box_clamps_to_frame_edges(module, returns_bbox):
    """A face pushed into the corner must not produce negative or
    out-of-frame crop coordinates - the padding+clamp order matters here:
    padding is applied first, then clamped, not the other way round."""
    landmarks = make_face_landmarks()
    # Shift every landmark toward the top-left corner so the padded box
    # would go negative without clamping.
    for lm in landmarks:
        lm.x = max(0.0, lm.x - 0.19)   # FACE_LEFT (0.2) -> ~0.01
        lm.y = max(0.0, lm.y - 0.45)   # most filler points (0.5) -> ~0.05
    mesh = face_mesh_result(landmarks)
    frame = _blank_frame()

    crop = _crop_only(module, returns_bbox, frame, mesh)
    assert crop is not None
    assert crop.shape[0] > 0 and crop.shape[1] > 0
    # A crop that safely exists at all, with the corner-pushed geometry
    # above, is only possible if x1/y1 were clamped to >= 0 rather than
    # left negative (which would silently produce a wrapped/garbage numpy
    # slice instead of raising).
    assert crop.shape[0] <= FRAME_H and crop.shape[1] <= FRAME_W


@pytest.mark.parametrize("module,returns_bbox", IMPLS)
def test_degenerate_zero_width_box_returns_none(module, returns_bbox):
    """All landmarks collapsed to a single point produces x2<=x1 - must
    return None rather than an empty/invalid numpy slice."""
    landmarks = make_face_landmarks()
    for lm in landmarks:
        lm.x = 0.5
        lm.y = 0.5
    mesh = face_mesh_result(landmarks)
    frame = _blank_frame()

    result = module.crop_face(frame, mesh)
    if returns_bbox:
        assert result == (None, None)
    else:
        assert result is None


def test_app_and_other_modules_agree_on_crop_pixels():
    """The four independently-maintained copies must still produce
    IDENTICAL crops for the same input - if one drifts (different padding
    formula, different rounding), this is the test that would catch it,
    since nothing else in the suite compares them against each other."""
    landmarks = make_face_landmarks()
    mesh = face_mesh_result(landmarks)
    frame = np.random.RandomState(0).randint(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)

    app_crop, _ = app.crop_face(frame, mesh)
    bench_crop, _ = benchmark_inference.crop_face(frame, mesh)
    other_crops = [
        stress_detector.crop_face(frame, mesh),
        bench_crop,
        evaluate_video_file.crop_face(frame, mesh),
    ]
    for crop in other_crops:
        assert crop.shape == app_crop.shape
        assert np.array_equal(crop, app_crop)
