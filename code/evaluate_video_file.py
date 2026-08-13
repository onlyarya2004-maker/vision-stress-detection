"""
Offline video evaluation
========================
Runs the full live pipeline over a VIDEO FILE instead of a webcam, and emits
the same CSV format the evaluation harness already consumes.

Why this exists
---------------
Every number in this project so far comes from one person sitting at one
laptop. UBFC-Phys provides 56 subjects with synchronised video AND
physiological ground truth (BVP + EDA), which is the missing piece: DaiSEE
has video with a derived label, WESAD has real physiology with no video.
This script is what lets the existing system be scored against that.

Two things differ from the live path, and both matter
-----------------------------------------------------
1. MEDIA TIME, NOT WALL TIME. Blink rate is computed from timestamps, and a
   file is processed far faster than real time. With a wall clock every blink
   in a clip lands within a fraction of a second and blinks_per_minute becomes
   nonsense. The engine's time source is overridden with frame_index / fps.

2. CALIBRATION COMES FROM THE VIDEO. The live system asks the user to sit
   calm for 6 seconds. Here the first N seconds of the clip are used instead,
   which is exactly right for UBFC-Phys: its T1 task IS a rest period.

UBFC-Phys mapping
-----------------
    T1 (rest)       -> calibrate on the first 6s, score the remainder as relaxed
    T2 (speech)     -> stressed
    T3 (arithmetic) -> stressed

Usage
-----
    # single file
    python evaluate_video_file.py --video path/to/vid_s1_T1.avi \
        --subject S01 --condition relaxed --mode default

    # whole directory, inferring condition from UBFC-Phys T1/T2/T3 naming
    python evaluate_video_file.py --dir ../datasets/UBFC-Phys/s1 --ubfc

Output goes to ../evaluation_data/ using the canonical naming scheme, so
run_all_evaluations.py picks it up with no further work.
"""

import argparse
import csv
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

import config
from app_naming import export_filename
from behavioral_features import BehavioralFeatureEngine
from stress_fusion import (
    compute_stress_index, TemporalSmoother, HysteresisLevel,
    set_daisee_baseline, reset_daisee_baseline,
)

# TensorFlow is imported lazily - the behavioural branch alone is useful and
# loading TF costs ~10s, which is wasteful when models are absent.
_TF_MODELS = None


def load_cnn_models(skip=False):
    """Load the DaiSEE ensemble once. Returns {} if unavailable or skipped."""
    global _TF_MODELS
    if skip:
        return {}
    if _TF_MODELS is not None:
        return _TF_MODELS
    paths = [Path(config.DAISEE_MOBILENET_PATH),
             Path(config.DAISEE_EFFICIENTNET_PATH)]
    if not any(p.exists() for p in paths):
        print("  [models] no .h5 files found - behavioural branch only "
              "(fusion redistributes DaiSEE weight automatically)")
        _TF_MODELS = {}
        return _TF_MODELS
    print("  [models] loading DaiSEE ensemble...", flush=True)
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input as mb_pre)
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as eff_pre)

    def _try(p):
        try:
            return tf.keras.models.load_model(str(p)) if p.exists() else None
        except Exception as e:
            print(f"  [models] failed to load {p.name}: {e}")
            return None

    _TF_MODELS = {
        'pairs': [
            (_try(paths[0]), config.MOBILENET_INPUT_SIZE, mb_pre),
            (_try(paths[1]), config.EFFICIENTNET_INPUT_SIZE, eff_pre),
        ]
    }
    return _TF_MODELS


def crop_face(frame_bgr, mesh_result):
    if not mesh_result.multi_face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    lm = mesh_result.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = int(config.FACE_CROP_PADDING * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def ensemble_predict(face_bgr, pairs):
    preds = []
    for model, size, preproc in pairs:
        if model is None:
            continue
        img = cv2.resize(face_bgr, (size, size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
        img = preproc(img)
        preds.append(model.predict(np.expand_dims(img, 0), verbose=0)[0])
    return np.mean(preds, axis=0) if preds else None


def process_video(video_path, calib_seconds=None, sample_every=1,
                  task_mode='Default', skip_cnn=False, max_seconds=None,
                  verbose=True):
    """
    Run the pipeline over one video.

    Returns (records, meta). records match the live history format, so
    history_to_csv-style output drops straight into the harness.
    """
    calib_seconds = (config.CALIBRATION_SECONDS if calib_seconds is None
                     else calib_seconds)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if verbose:
        print(f"  {Path(video_path).name}: {total} frames @ {fps:.1f}fps "
              f"({total / fps:.0f}s)")

    # --- media clock: the whole reason offline results are trustworthy ---
    clock = {'t': 0.0}
    engine = BehavioralFeatureEngine(time_source=lambda: clock['t'])

    models = load_cnn_models(skip=skip_cnn)
    pairs = models.get('pairs', [])

    smoother = TemporalSmoother(window_size=config.SMOOTHING_WINDOW_FRAMES)
    hyst = HysteresisLevel()
    reset_daisee_baseline()

    engine.start_calibration()
    calibrating = True
    daisee_calib = []

    records = []
    frame_idx = 0
    processed = 0
    no_face = 0
    t_start = time.perf_counter()
    base_ts = datetime(2026, 1, 1)      # synthetic, monotonic per-video

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        clock['t'] = frame_idx / fps
        if max_seconds and clock['t'] > max_seconds:
            break
        if frame_idx % sample_every:
            continue
        processed += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh = engine.face_mesh.process(rgb)
        face = crop_face(frame, mesh)
        if face is None:
            no_face += 1
            continue

        daisee_probs = None
        if pairs and processed % config.CNN_EVERY_N_FRAMES == 0:
            daisee_probs = ensemble_predict(face, pairs)

        # ---- calibration window ----
        if calibrating:
            engine.extract(rgb)                     # fills the calib buffer
            if daisee_probs is not None:
                daisee_calib.append(float(daisee_probs[1]))
            if clock['t'] >= calib_seconds:
                ok_face = engine.finish_calibration()
                ok_daisee = False
                if len(daisee_calib) >= config.MIN_DAISEE_CALIB_SAMPLES:
                    set_daisee_baseline(float(np.median(daisee_calib)))
                    ok_daisee = True
                calibrating = False
                if verbose:
                    print(f"    calibrated at {clock['t']:.1f}s "
                          f"(face={ok_face}, daisee={ok_daisee}, "
                          f"{len(engine._calib_buffer)} samples)")
                if not ok_face:
                    print("    WARNING: face calibration failed - too few "
                          "usable frames; results unreliable")
            continue

        feats = engine.extract(rgb)
        if feats is None:
            continue

        result = compute_stress_index(
            daisee_probs=daisee_probs, behavioral_features=feats,
            task_mode=task_mode,
        )
        smoothed = smoother.update(result['stress_index'])
        level = hyst.update(smoothed)

        records.append({
            'ts': base_ts + timedelta(seconds=clock['t']),
            'stress_index': smoothed,
            'level': level,
            'daisee': result.get('daisee_raw') or 0.0,
            'behavior': result['behavior_stress'],
        })

    cap.release()
    engine.close()

    elapsed = time.perf_counter() - t_start
    meta = {
        'video': str(video_path), 'fps': fps, 'frames': frame_idx,
        'processed': processed, 'no_face': no_face,
        'records': len(records), 'seconds': clock['t'],
        'wall_seconds': elapsed,
        'realtime_factor': (clock['t'] / elapsed) if elapsed else 0.0,
    }
    if verbose:
        print(f"    -> {len(records)} samples, {no_face} frames without a "
              f"face, {elapsed:.1f}s wall ({meta['realtime_factor']:.1f}x "
              f"realtime)")
    return records, meta


def write_csv(records, out_path):
    """Same columns as the live export, so the harness needs no changes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ts', 'stress_index', 'level',
                                          'daisee', 'emotion', 'behavior'])
        w.writeheader()
        for r in records:
            w.writerow({
                'ts': r['ts'].isoformat(),
                'stress_index': f"{r['stress_index']:.4f}",
                'level': r['level'],
                'daisee': f"{r['daisee']:.4f}",
                'emotion': "0.0000",          # retained for column compat
                'behavior': f"{r['behavior']:.4f}",
            })


UBFC_RE = re.compile(r'(?:vid_)?s(\d+)_T(\d)', re.IGNORECASE)


def ubfc_meta(path):
    """
    Parse UBFC-Phys naming: subject number and task.
      T1 = rest      -> relaxed
      T2 = speech    -> stressed
      T3 = arithmetic-> stressed
    """
    m = UBFC_RE.search(Path(path).stem)
    if not m:
        return None
    subject, task = int(m.group(1)), int(m.group(2))
    return {
        'subject': f"U{subject:02d}",
        'task': task,
        'condition': 'relaxed' if task == 1 else 'stressed',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', help='single video file')
    src.add_argument('--dir', help='directory of videos')
    ap.add_argument('--subject', help='subject ID, e.g. S01 (ignored with --ubfc)')
    ap.add_argument('--condition', choices=['relaxed', 'stressed'],
                    help='ground truth (ignored with --ubfc)')
    ap.add_argument('--mode', default='Default',
                    choices=['Default', 'Studying', 'Hospital', 'Driving'])
    ap.add_argument('--ubfc', action='store_true',
                    help='infer subject/condition from UBFC-Phys T1/T2/T3 names')
    ap.add_argument('--calib-seconds', type=float, default=None,
                    help=f'calibration window (default {config.CALIBRATION_SECONDS}s)')
    ap.add_argument('--sample-every', type=int, default=1,
                    help='process every Nth frame (speeds up long videos)')
    ap.add_argument('--max-seconds', type=float, default=None,
                    help='stop after N seconds of media (useful for smoke tests)')
    ap.add_argument('--skip-cnn', action='store_true',
                    help='behavioural branch only; skips loading TensorFlow')
    ap.add_argument('--out-dir', default='../evaluation_data',
                    help='where CSVs are written')
    args = ap.parse_args()

    videos = []
    if args.video:
        videos = [Path(args.video)]
    else:
        d = Path(args.dir)
        for ext in ('*.avi', '*.mp4', '*.mov', '*.mkv'):
            videos.extend(sorted(d.rglob(ext)))
    if not videos:
        print("no videos found")
        return 1

    if not args.ubfc and not (args.subject and args.condition):
        print("ERROR: --subject and --condition are required unless --ubfc "
              "is given.\n       Unlabelled output cannot be scored, so this "
              "refuses rather than\n       writing a file the harness will "
              "silently ignore.")
        return 2

    print(f"processing {len(videos)} video(s)\n")
    written = []
    for v in videos:
        if args.ubfc:
            meta = ubfc_meta(v)
            if not meta:
                print(f"  [skip] {v.name}: not UBFC-Phys naming")
                continue
            subject, condition = meta['subject'], meta['condition']
        else:
            subject, condition = args.subject, args.condition

        try:
            records, info = process_video(
                v, calib_seconds=args.calib_seconds,
                sample_every=args.sample_every, task_mode=args.mode,
                skip_cnn=args.skip_cnn, max_seconds=args.max_seconds,
            )
        except Exception as e:
            print(f"  [error] {v.name}: {e}")
            continue

        if not records:
            print(f"  [skip] {v.name}: no scorable frames "
                  f"(clip shorter than the calibration window?)")
            continue

        name = export_filename(subject, condition, args.mode,
                               when=datetime.now())
        # Keep per-video files distinct when a subject has several clips.
        out = Path(args.out_dir) / f"{Path(name).stem}_{v.stem}.csv"
        write_csv(records, out)
        written.append(out)
        print(f"    written: {out.name}\n")

    print(f"\n{len(written)} CSV(s) written to {Path(args.out_dir).resolve()}")
    if written:
        print("next:  cd code && python run_all_evaluations.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
