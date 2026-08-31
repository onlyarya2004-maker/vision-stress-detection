"""
Inference performance benchmark (Phase 6.2)
==============================================
Measures real per-component and end-to-end per-frame cost of the live
capture pipeline on THIS machine: MediaPipe FaceMesh, MediaPipe Pose, the
DaiSEE CNN ensemble, and the effective end-to-end frame rate that implies -
answering the external-review request for FPS/latency numbers with a real,
regenerable artifact instead of the informal paragraph this project's
planning doc had before (see PLAN_FOR_PP2.md's "CNN inference speed"
entry, 2026-08-23).

No camera involved - runs entirely against a synthetic frame built from a
real, already-on-disk DaiSEE face crop (datasets/processed_daisee), so
MediaPipe has something genuine to detect rather than timing its
no-detection fast path. Standing project policy never touches
cv2.VideoCapture(0); this script never imports Streamlit or opens one.

Why this is NOT called from run_all_evaluations.py directly
-------------------------------------------------------------
Same two-tier pattern as daisee_test_eval.py / train_temporal_model.py:
loading MediaPipe + TensorFlow and running real inference takes real wall
time (tens of seconds), unlike the harness's cheap numpy/scipy sections.
This script writes ../models/inference_benchmark_report.txt once;
run_all_evaluations.py could echo it the same way it already echoes the
DaiSEE/GRU/LSTM report files, if that's ever wanted.

Usage
-----
    python benchmark_inference.py
    python benchmark_inference.py --iterations 200 --warmup 20
"""

import argparse
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

import config
import daisee_ensemble
from behavioral_features import BehavioralFeatureEngine

DEFAULT_ITERATIONS = 100
DEFAULT_CNN_ITERATIONS = 30     # CNN is far slower per-call; fewer reps is still stable
DEFAULT_WARMUP = 10             # discards tf.function's one-time tracing cost


def crop_face(frame_bgr, face_mesh_results):
    """Same small utility every non-Streamlit runner in this codebase
    keeps its own copy of (app.py, evaluate_video_file.py,
    stress_detector.py) - kept here too rather than importing app.py
    (and its Streamlit module-level side effects) into a benchmark
    script that should stay lightweight and camera-free."""
    if not face_mesh_results.multi_face_landmarks:
        return None, None
    h, w = frame_bgr.shape[:2]
    lm = face_mesh_results.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = int(config.FACE_CROP_PADDING * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def build_benchmark_frame(data_root=None):
    """
    A config.FRAME_WIDTH x config.FRAME_HEIGHT BGR frame with a real face
    roughly centred - built by pasting an actual DaiSEE face crop onto a
    blank canvas at live-capture resolution, rather than a synthetic
    landmark mock (this needs MediaPipe to run its REAL detection graph,
    not the mocked path the unit tests use) or a random-noise frame (which
    would time FaceMesh's no-detection fast path, not real inference).
    """
    data_root = Path(data_root or "../datasets/processed_daisee")
    candidates = sorted((data_root / "test" / "not_stressed").glob("*.jpg"))
    if not candidates:
        raise FileNotFoundError(
            f"No sample face crop found under {data_root} - pass --data-root "
            f"to a directory containing processed_daisee's test/not_stressed/*.jpg"
        )
    face = cv2.imread(str(candidates[0]))
    if face is None:
        raise RuntimeError(f"Could not read {candidates[0]}")

    canvas = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 30, dtype=np.uint8)
    # Paste at roughly 40% of frame height/width, matching where a face
    # typically sits in a real webcam frame - centred, not filling it.
    face_h = int(config.FRAME_HEIGHT * 0.55)
    face_w = int(face.shape[1] * (face_h / face.shape[0]))
    face_resized = cv2.resize(face, (face_w, face_h))
    y0 = (config.FRAME_HEIGHT - face_h) // 2
    x0 = (config.FRAME_WIDTH - face_w) // 2
    canvas[y0:y0 + face_h, x0:x0 + face_w] = face_resized
    return canvas


def _time_calls(fn, n):
    """Runs fn() n times, returns a list of per-call wall-clock seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _stats(times_sec):
    ms = [t * 1000.0 for t in times_sec]
    return {
        'n': len(ms), 'mean_ms': statistics.mean(ms),
        'median_ms': statistics.median(ms),
        'stdev_ms': statistics.stdev(ms) if len(ms) > 1 else 0.0,
        'min_ms': min(ms), 'max_ms': max(ms),
    }


def run_benchmark(iterations=DEFAULT_ITERATIONS, cnn_iterations=DEFAULT_CNN_ITERATIONS,
                  warmup=DEFAULT_WARMUP, data_root=None):
    """
    Returns a dict of component -> stats, plus a derived end-to-end
    estimate. Pure function (no printing) so it's independently testable
    and reusable - render_result() below does the printing.
    """
    frame = build_benchmark_frame(data_root)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    engine = BehavioralFeatureEngine()
    pairs = daisee_ensemble.build_pairs()
    n_cnn_loaded = sum(1 for infer_fn, _, _ in pairs if infer_fn is not None)

    try:
        # Warm-up: MediaPipe's first call and tf.function's first trace
        # both carry one-time costs that don't represent steady-state
        # per-frame cost - discard these before timing anything.
        for _ in range(warmup):
            mesh = engine.face_mesh.process(rgb)
            engine.pose.process(rgb)
            face_crop, _ = crop_face(frame, mesh)
            if face_crop is not None and n_cnn_loaded:
                daisee_ensemble.ensemble_predict(face_crop, pairs)

        results = {}

        results['facemesh'] = _stats(_time_calls(
            lambda: engine.face_mesh.process(rgb), iterations))
        results['pose'] = _stats(_time_calls(
            lambda: engine.pose.process(rgb), iterations))

        mesh = engine.face_mesh.process(rgb)
        face_crop, _ = crop_face(frame, mesh)
        results['crop_face'] = _stats(_time_calls(
            lambda: crop_face(frame, mesh), iterations))
        results['behavioral_extract'] = _stats(_time_calls(
            lambda: engine.extract(rgb, face_mesh_results=mesh), iterations))

        if n_cnn_loaded and face_crop is not None:
            results['daisee_ensemble'] = _stats(_time_calls(
                lambda: daisee_ensemble.ensemble_predict(face_crop, pairs),
                cnn_iterations))
        else:
            results['daisee_ensemble'] = None

        return {
            'results': results,
            'n_cnn_loaded': n_cnn_loaded,
            'frame_size': (config.FRAME_WIDTH, config.FRAME_HEIGHT),
            'cnn_every_n_frames': config.CNN_EVERY_N_FRAMES,
        }
    finally:
        engine.close()


def effective_fps(bench):
    """
    Steady-state per-frame cost as actually wired in app.py's capture
    loop: FaceMesh + crop_face + behavioral extraction run every frame;
    the CNN ensemble runs only every CNN_EVERY_N_FRAMES frames, so its
    cost is amortized across that cadence, not paid every frame. Pose is
    NOT in this sum - app.py's capture loop doesn't call engine.pose
    directly per frame; it runs inside behavioral_extract via
    _extract_posture(), so counting it separately here would double-count.
    """
    r = bench['results']
    per_frame_ms = (r['facemesh']['mean_ms'] + r['crop_face']['mean_ms']
                    + r['behavioral_extract']['mean_ms'])
    if r['daisee_ensemble'] is not None:
        per_frame_ms += r['daisee_ensemble']['mean_ms'] / bench['cnn_every_n_frames']
    return per_frame_ms, (1000.0 / per_frame_ms if per_frame_ms > 0 else 0.0)


def render_result(bench):
    lines = []
    lines.append("=" * 78)
    lines.append("INFERENCE PERFORMANCE BENCHMARK")
    lines.append("=" * 78)
    lines.append(f"generated : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"frame size: {bench['frame_size'][0]}x{bench['frame_size'][1]} "
                 f"(config.FRAME_WIDTH/FRAME_HEIGHT)")
    lines.append(f"DaiSEE models loaded: {bench['n_cnn_loaded']}/2")
    lines.append("")
    lines.append("No camera involved - real MediaPipe/TensorFlow inference against a")
    lines.append("synthetic frame built from a real on-disk DaiSEE face crop.")

    labels = {
        'facemesh': 'MediaPipe FaceMesh.process()',
        'pose': 'MediaPipe Pose.process()',
        'crop_face': 'crop_face() (bbox + array slice)',
        'behavioral_extract': 'BehavioralFeatureEngine.extract() (face+posture, reuses FaceMesh result)',
        'daisee_ensemble': 'daisee_ensemble.ensemble_predict() (MobileNet+EfficientNet, tf.function)',
    }
    for key, label in labels.items():
        s = bench['results'][key]
        lines.append("")
        lines.append(f"  {label}")
        if s is None:
            lines.append("    skipped - no DaiSEE model loaded")
            continue
        lines.append(f"    n={s['n']}  mean={s['mean_ms']:.1f}ms  "
                     f"median={s['median_ms']:.1f}ms  stdev={s['stdev_ms']:.1f}ms  "
                     f"min={s['min_ms']:.1f}ms  max={s['max_ms']:.1f}ms")

    per_frame_ms, fps = effective_fps(bench)
    lines.append("")
    lines.append("-" * 78)
    lines.append("EFFECTIVE END-TO-END ESTIMATE")
    lines.append("-" * 78)
    lines.append("  As actually wired in app.py's capture loop: FaceMesh + crop_face +")
    lines.append(f"  behavioral extraction run every frame; the DaiSEE ensemble runs every "
                 f"{bench['cnn_every_n_frames']}rd")
    lines.append("  frame and its cost is amortized across that cadence. Pose is not summed")
    lines.append("  separately - it runs inside behavioral extraction, not on its own.")
    lines.append(f"  estimated per-frame cost : {per_frame_ms:.1f}ms")
    lines.append(f"  estimated effective FPS  : {fps:.1f}")
    lines.append("")
    lines.append("  This is inference cost only - it excludes webcam capture/decode,")
    lines.append("  Streamlit re-render, and rPPG/rule-boost/smoothing, which are all")
    lines.append("  cheap numpy operations by comparison (sub-millisecond).")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS,
                    help='reps for the fast per-frame components')
    ap.add_argument('--cnn-iterations', type=int, default=DEFAULT_CNN_ITERATIONS,
                    help='reps for the slower CNN ensemble call')
    ap.add_argument('--warmup', type=int, default=DEFAULT_WARMUP,
                    help='discarded reps before timing (MediaPipe/tf.function warm-up)')
    ap.add_argument('--data-root', default=None,
                    help='processed_daisee root (default: ../datasets/processed_daisee)')
    ap.add_argument('--report-out', default='../models/inference_benchmark_report.txt')
    args = ap.parse_args()

    print("Loading engine + models (MediaPipe FaceMesh/Pose, DaiSEE ensemble)...")
    bench = run_benchmark(iterations=args.iterations, cnn_iterations=args.cnn_iterations,
                          warmup=args.warmup, data_root=args.data_root)
    report = render_result(bench)
    print(report)

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding='utf-8')
    print(f"\nWritten to {out_path}")


if __name__ == '__main__':
    main()
