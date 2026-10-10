"""
Parity check: legacy MediaPipe Python solutions vs the Tasks API models
=======================================================================
Question: if landmark detection moves into the browser (MediaPipe Tasks
Vision JS - the hosting plan that replaces the retired WebRTC approach), do
the behavioral signals change? The live system was validated on the legacy
Python solutions (mp.solutions.face_mesh with refine_landmarks=True,
mp.solutions.pose with model_complexity=1). The browser can only run the
Tasks API models (face_landmarker.task, pose_landmarker_full.task).

Method: the SAME real frames (UBFC-Phys s11 T1 rest + T2 speech, still on
disk) go through the SAME BehavioralFeatureEngine code twice - once fed
legacy landmarks (engine A, exactly today's pipeline), once fed Tasks
landmarks through a legacy-shaped adapter (engine B). Each engine calibrates
on the same opening seconds of T1, as a real user would. The Python Tasks
API runs the identical .task model file and C++ graph the JS API compiles
to WASM, so engine B is the closest available proxy for browser output.

Pass bar (fixed before running): r >= 0.9 on the continuous face signals
and on the weighted behavior score, and the T1-vs-T2 separation preserved.

Usage:
    python experiment_mediapipe_tasks_parity.py \\
        --face-model PATH/face_landmarker.task \\
        --pose-model PATH/pose_landmarker_full.task
"""
import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config
from behavioral_features import BehavioralFeatureEngine
from stress_fusion import BEHAVIOR_WEIGHTS

SIGNALS = list(BEHAVIOR_WEIGHTS)
RAW = ['ear', 'ear_asymmetry']


# --------------------------------------------------------------------------
# Legacy-result-shaped adapters. BehavioralFeatureEngine reads
# result.multi_face_landmarks[0].landmark[i].x/.y and
# result.pose_landmarks.landmark[i].x/.y/.visibility - these reproduce exactly
# that shape from any landmark source, so the engine itself stays unchanged.
# --------------------------------------------------------------------------
class _LM:
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, x, y, z=0.0, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


class _LMList:
    __slots__ = ('landmark',)

    def __init__(self, lms):
        self.landmark = lms


class FaceMeshResultLike:
    def __init__(self, lms):
        self.multi_face_landmarks = [_LMList(lms)] if lms else None


class PoseResultLike:
    def __init__(self, lms):
        self.pose_landmarks = _LMList(lms) if lms else None


def tasks_face_to_legacy(res):
    if not res.face_landmarks:
        return FaceMeshResultLike(None)
    return FaceMeshResultLike([_LM(p.x, p.y, p.z) for p in res.face_landmarks[0]])


def tasks_pose_to_legacy(res):
    if not res.pose_landmarks:
        return PoseResultLike(None)
    return PoseResultLike([
        _LM(p.x, p.y, p.z, p.visibility if p.visibility is not None else 0.0)
        for p in res.pose_landmarks[0]])


def make_tasks_landmarkers(face_model, pose_model):
    # Passing the model bytes (model_asset_buffer) rather than a path:
    # MediaPipe's resource loader on Windows intermittently resolves even a
    # clean absolute model_asset_path relative to its own package dir
    # ("Unable to open file at .../site-packages/C:\..."). Reading the bytes
    # here sidesteps its path handling entirely.
    with open(os.path.abspath(face_model), 'rb') as f:
        face_bytes = f.read()
    with open(os.path.abspath(pose_model), 'rb') as f:
        pose_bytes = f.read()
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=face_bytes),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=pose_bytes),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return face, pose


def behavior_score(feats):
    return sum(BEHAVIOR_WEIGHTS[s] * feats.get(s, 0.0) for s in SIGNALS)


def run_video(path, task_label, eng_a, eng_b, clock_a, clock_b, face_model,
              pose_model, stride, calibrate_seconds, rows, lm_deltas,
              mode='tasks', jpeg_quality=80, size=None, pose_every=3):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose_cache = {'pose': None, 'n': 0}
    t_face, t_pose = (make_tasks_landmarkers(face_model, pose_model)
                      if mode == 'tasks' else (None, None))
    calibrating = calibrate_seconds > 0
    if calibrating:
        eng_a.start_calibration()
        eng_b.start_calibration()

    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue
            t = idx / fps
            native = frame
            if size:
                frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if calibrating and t >= calibrate_seconds:
                eng_a.finish_calibration()
                eng_b.finish_calibration()
                calibrating = False

            legacy_face = eng_a.face_mesh.process(rgb)
            clock_a['t'] = t
            clock_b['t'] = t
            fa = eng_a.extract(rgb, face_mesh_results=legacy_face)

            if mode == 'tasks':
                ts_ms = int(t * 1000)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                b_face = tasks_face_to_legacy(t_face.detect_for_video(mp_img, ts_ms))
                b_pose = tasks_pose_to_legacy(t_pose.detect_for_video(mp_img, ts_ms))
                fb = eng_b.extract(rgb, face_mesh_results=b_face, pose_results=b_pose)
            elif mode == 'pose-cadence':
                # The SAME legacy models and the SAME FaceMesh result; the only
                # difference is that Pose runs every `pose_every`th frame and
                # the last result is reused in between - the server-CPU lever
                # config.POSE_EVERY_N_FRAMES. Isolates what that costs the
                # posture signals (neck_strain, slouch).
                b_face = legacy_face
                if pose_cache['pose'] is None or pose_cache['n'] % pose_every == 0:
                    pose_cache['pose'] = eng_b.pose.process(rgb)
                pose_cache['n'] += 1
                fb = eng_b.extract(rgb, face_mesh_results=legacy_face,
                                   pose_results=pose_cache['pose'])
            elif mode == 'jpeg':
                # The SAME legacy models, fed the frame after a JPEG
                # encode/decode round trip - exactly what the server would
                # receive from the browser. Isolates compression alone.
                ok_enc, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                rgb_b = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                b_face = eng_b.face_mesh.process(rgb_b)
                fb = eng_b.extract(rgb_b, face_mesh_results=b_face)
            else:
                # 'resolution': the SAME legacy models on the full native
                # frame vs the downscaled one (A) - does processing at high
                # resolution change the signals the system was validated on
                # at 640x480?
                rgb_b = cv2.cvtColor(native, cv2.COLOR_BGR2RGB)
                b_face = eng_b.face_mesh.process(rgb_b)
                fb = eng_b.extract(rgb_b, face_mesh_results=b_face)

            if legacy_face.multi_face_landmarks and b_face.multi_face_landmarks:
                la = legacy_face.multi_face_landmarks[0].landmark
                lb = b_face.multi_face_landmarks[0].landmark
                h, w = rgb.shape[:2]
                for i in (33, 133, 159, 145, 362, 263, 55, 285, 13, 14, 61, 291, 234, 454):
                    lm_deltas.append(np.hypot((la[i].x - lb[i].x) * w, (la[i].y - lb[i].y) * h))

            rows.append({
                'task': task_label, 't': t, 'calibrating': calibrating,
                'a': fa, 'b': fb,
            })
            idx += 1
    finally:
        cap.release()
        if t_face is not None:
            t_face.close()
            t_pose.close()
        if calibrating:
            eng_a.finish_calibration()
            eng_b.finish_calibration()


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['tasks', 'jpeg', 'resolution', 'pose-cadence'],
                    default='tasks',
                    help="tasks: legacy vs Tasks-API models; jpeg: legacy on raw "
                         "frames vs legacy on JPEG-round-tripped frames; "
                         "resolution: legacy at --size vs legacy at native size; "
                         "pose-cadence: Pose every frame vs every --pose-every frames")
    ap.add_argument('--pose-every', type=int, default=3,
                    help='for --mode pose-cadence: run Pose every Nth frame')
    ap.add_argument('--face-model', help='required for --mode tasks')
    ap.add_argument('--pose-model', help='required for --mode tasks')
    ap.add_argument('--jpeg-quality', type=int, default=80)
    ap.add_argument('--size', type=int, default=None,
                    help='resize each (square) frame to SIZE x SIZE first')
    ap.add_argument('--subject', default='s11')
    ap.add_argument('--ubfc', default='../datasets/UBFC-Phys')
    ap.add_argument('--stride', type=int, default=2,
                    help='process every Nth frame (2 at 35fps ~= the GRU training pace)')
    args = ap.parse_args()
    if args.mode == 'tasks' and not (args.face_model and args.pose_model):
        ap.error('--mode tasks needs --face-model and --pose-model')

    clock_a, clock_b = {'t': 0.0}, {'t': 0.0}
    eng_a = BehavioralFeatureEngine(time_source=lambda: clock_a['t'])
    eng_b = BehavioralFeatureEngine(time_source=lambda: clock_b['t'])

    rows, lm_deltas = [], []
    base = f"{args.ubfc}/{args.subject}"
    for task, label, calib in (('T1', 'relaxed', config.CALIBRATION_SECONDS),
                               ('T2', 'stressed', 0)):
        path = f"{base}/vid_{args.subject}_{task}.avi"
        print(f"processing {path} ...", flush=True)
        if task != 'T1':
            eng_a.reset_state()
            eng_b.reset_state()
        run_video(path, label, eng_a, eng_b, clock_a, clock_b, args.face_model,
                  args.pose_model, args.stride, calib, rows, lm_deltas,
                  mode=args.mode, jpeg_quality=args.jpeg_quality, size=args.size,
                  pose_every=args.pose_every)

    scored = [r for r in rows if not r['calibrating']]
    both = [r for r in scored if r['a'] is not None and r['b'] is not None]
    only_a = sum(1 for r in scored if r['a'] is not None and r['b'] is None)
    only_b = sum(1 for r in scored if r['a'] is None and r['b'] is not None)

    print("\n" + "=" * 78)
    what = {'tasks': 'LEGACY vs TASKS-API MODELS',
            'jpeg': f'RAW vs JPEG q{args.jpeg_quality} (same legacy models)',
            'pose-cadence': f'POSE EVERY FRAME (A) vs EVERY {args.pose_every}rd/th FRAME (B)',
            'resolution': 'DOWNSCALED (A) vs NATIVE RESOLUTION (B), same legacy models'}[args.mode]
    print(f"{what} - UBFC-Phys {args.subject}, stride {args.stride}"
          f"{f', {args.size}px' if args.size else ''}")
    print("=" * 78)
    print(f"  scored frames: {len(scored)}   both detected: {len(both)}   "
          f"legacy only: {only_a}   tasks only: {only_b}")
    if lm_deltas:
        d = np.asarray(lm_deltas)
        print(f"  key-landmark position difference (px @1024): mean {d.mean():.2f}  "
              f"median {np.median(d):.2f}  p95 {np.percentile(d, 95):.2f}")

    print(f"\n  {'signal':18s}{'r':>8}{'MAD':>9}{'mean A':>9}{'mean B':>9}")
    print("  " + "-" * 52)
    for key in RAW + SIGNALS:
        xa = [r['a'].get(key, 0.0) for r in both]
        xb = [r['b'].get(key, 0.0) for r in both]
        print(f"  {key:18s}{pearson(xa, xb):>8.3f}{np.mean(np.abs(np.subtract(xa, xb))):>9.4f}"
              f"{np.mean(xa):>9.4f}{np.mean(xb):>9.4f}")
    ba = [behavior_score(r['a']) for r in both]
    bb = [behavior_score(r['b']) for r in both]
    print(f"  {'BEHAVIOR SCORE':18s}{pearson(ba, bb):>8.3f}"
          f"{np.mean(np.abs(np.subtract(ba, bb))):>9.4f}{np.mean(ba):>9.4f}{np.mean(bb):>9.4f}")

    print("\n  Relaxed (T1) vs stressed (T2) separation, weighted behavior score:")
    for name, key in (('legacy (A)', 'a'), ('tasks (B)', 'b')):
        rel = [behavior_score(r[key]) for r in both if r['task'] == 'relaxed']
        stre = [behavior_score(r[key]) for r in both if r['task'] == 'stressed']
        print(f"    {name:12s} relaxed {np.mean(rel):.4f}   stressed {np.mean(stre):.4f}   "
              f"gap {np.mean(stre) - np.mean(rel):+.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
