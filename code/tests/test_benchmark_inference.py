"""
benchmark_inference.py tests.

Loading real MediaPipe graphs + the DaiSEE TensorFlow ensemble is slow and
belongs to a manual benchmark run, not the fast unit suite - same boundary
test_daisee_ensemble.py already draws around real model loading. What's
covered here is everything pure: the stats math, the effective-FPS
estimate, the report renderer, and the benchmark-frame builder (which does
need a real on-disk image, but no MediaPipe/TensorFlow).
"""

import cv2
import numpy as np
import pytest

from benchmark_inference import (
    _stats, _time_calls, build_benchmark_frame, effective_fps, render_result,
)


def test_time_calls_returns_one_duration_per_call():
    calls = []
    _time_calls(lambda: calls.append(1), 5)
    assert len(calls) == 5


def test_stats_computes_expected_summary():
    # Three calls of known, controllable duration via a tiny counter -
    # exact wall-clock timing isn't asserted (inherently noisy), only that
    # the aggregation math (mean/median/stdev/min/max, n) is self-consistent.
    times = [0.010, 0.020, 0.030]
    s = _stats(times)
    assert s['n'] == 3
    assert s['mean_ms'] == pytest.approx(20.0, abs=0.01)
    assert s['median_ms'] == pytest.approx(20.0, abs=0.01)
    assert s['min_ms'] == pytest.approx(10.0, abs=0.01)
    assert s['max_ms'] == pytest.approx(30.0, abs=0.01)
    assert s['stdev_ms'] > 0


def test_stats_handles_single_sample_without_crashing():
    s = _stats([0.005])
    assert s['n'] == 1
    assert s['stdev_ms'] == 0.0


def _fake_bench(facemesh_ms, crop_ms, extract_ms, cnn_ms, cnn_every_n=3):
    mk = lambda ms: {'n': 10, 'mean_ms': ms, 'median_ms': ms, 'stdev_ms': 0.0,
                     'min_ms': ms, 'max_ms': ms}
    return {
        'results': {
            'facemesh': mk(facemesh_ms), 'pose': mk(1.0),
            'crop_face': mk(crop_ms), 'behavioral_extract': mk(extract_ms),
            'daisee_ensemble': mk(cnn_ms) if cnn_ms is not None else None,
        },
        'n_cnn_loaded': 2 if cnn_ms is not None else 0,
        'frame_size': (640, 480), 'cnn_every_n_frames': cnn_every_n,
    }


def test_effective_fps_sums_every_frame_components_plus_amortized_cnn():
    bench = _fake_bench(facemesh_ms=10.0, crop_ms=1.0, extract_ms=20.0,
                        cnn_ms=90.0, cnn_every_n=3)
    per_frame_ms, fps = effective_fps(bench)
    # (10 + 1 + 20) + 90/3 = 61ms/frame
    assert per_frame_ms == pytest.approx(61.0)
    assert fps == pytest.approx(1000.0 / 61.0)


def test_effective_fps_excludes_pose_to_avoid_double_counting():
    """Pose runs INSIDE behavioral_extract (_extract_posture calls
    self.pose.process) - counting the standalone 'pose' timing on top
    would double-count it."""
    bench = _fake_bench(facemesh_ms=10.0, crop_ms=1.0, extract_ms=20.0, cnn_ms=None)
    per_frame_ms, _ = effective_fps(bench)
    assert per_frame_ms == pytest.approx(31.0)  # not 31 + pose's 1.0


def test_effective_fps_handles_no_cnn_loaded():
    bench = _fake_bench(facemesh_ms=10.0, crop_ms=1.0, extract_ms=20.0, cnn_ms=None)
    per_frame_ms, fps = effective_fps(bench)
    assert per_frame_ms == pytest.approx(31.0)
    assert fps > 0


def test_render_result_includes_every_component_and_the_fps_estimate():
    bench = _fake_bench(facemesh_ms=8.8, crop_ms=0.4, extract_ms=49.2, cnn_ms=89.5)
    report = render_result(bench)
    assert 'MediaPipe FaceMesh' in report
    assert 'MediaPipe Pose' in report
    assert 'daisee_ensemble.ensemble_predict' in report
    assert 'estimated effective FPS' in report
    assert '2/2' in report  # n_cnn_loaded


def test_render_result_handles_missing_daisee_models():
    bench = _fake_bench(facemesh_ms=8.8, crop_ms=0.4, extract_ms=49.2, cnn_ms=None)
    report = render_result(bench)
    assert 'skipped - no DaiSEE model loaded' in report


def test_build_benchmark_frame_has_the_configured_live_capture_size(tmp_path):
    config_size = __import__('config')
    sample_dir = tmp_path / "test" / "not_stressed"
    sample_dir.mkdir(parents=True)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(sample_dir / "sample.jpg"), img)

    frame = build_benchmark_frame(data_root=tmp_path)
    assert frame.shape == (config_size.FRAME_HEIGHT, config_size.FRAME_WIDTH, 3)


def test_build_benchmark_frame_raises_clearly_when_no_sample_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_benchmark_frame(data_root=tmp_path / "does_not_exist")
