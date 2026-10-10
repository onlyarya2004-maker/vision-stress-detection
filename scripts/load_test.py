"""
How many simultaneous users can this machine / container serve?

Runs N independent "users" in threads, each with its OWN UserState and LiveRun
(exactly as the hosted app does - one per login session, sharing only the
CNN weights), feeding them frames from a real face video as fast as they can
be processed. Reports frames/second per user. The hosted app adds Streamlit's
page painting on top (~15% in measurements), so treat the numbers as the
pipeline's ceiling.

    python scripts/load_test.py --video datasets/UBFC-Phys/s11/vid_s11_T1.avi --users 2

To mimic the cloud container, run it inside Docker with the CPU you will rent:
    docker run --rm --cpus=2 -m 4g -v <repo>/scripts:/app/scripts \
        -v <repo>/datasets:/app/datasets stress-app \
        python ../scripts/load_test.py --video ../datasets/UBFC-Phys/s11/vid_s11_T1.avi --users 2

Targets: >= 8 fps per user (the validated live system ran ~11 fps).
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

ORIG_CWD = Path.cwd()
CODE_DIR = Path(__file__).resolve().parent.parent / 'code'
sys.path.insert(0, str(CODE_DIR))
# The app's model paths are relative to code/ (../models/...). Without this a run
# from any other directory silently loads NO models and reports absurdly good
# numbers (the CNN is the heaviest stage).
os.chdir(CODE_DIR)

import cv2                                   # noqa: E402

import config                                # noqa: E402
import daisee_ensemble                       # noqa: E402
from frame_ops import prepare_processing_frame   # noqa: E402
from intervention_agent import InterventionAgent   # noqa: E402
from session_pipeline import LiveRun, UserState    # noqa: E402


def load_frames(path, n, fps_out=10.0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step, nxt, idx, frames = src_fps / fps_out, 0.0, 0, []
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if idx >= nxt:
            h, w = f.shape[:2]
            if w < h * 4 / 3:      # square UBFC frames: letterbox, don't crop the forehead
                pad = int(round(h * 4 / 3)) - w
                f = cv2.copyMakeBorder(f, 0, 0, pad // 2, pad - pad // 2,
                                       cv2.BORDER_CONSTANT, value=(0, 0, 0))
            frames.append(prepare_processing_frame(cv2.flip(f, 1))[0])
            nxt += step
        idx += 1
    cap.release()
    return frames


def worker(i, frames, models, seconds, results, barrier):
    state = UserState()
    agent = InterventionAgent(stats={}, mode_stats={}, task_mode='Default')
    run = LiveRun(state, models, agent, task_mode='Default', agent_enabled=True)
    barrier.wait()
    done, t_end, k, busy = 0, time.perf_counter() + seconds, 0, 0.0
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        run.process(frames[k % len(frames)], k / 10.0)
        busy += time.perf_counter() - t0
        k += 1
        done += 1
    results[i] = (done, busy)
    state.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--users', type=int, default=2)
    ap.add_argument('--seconds', type=float, default=30.0)
    ap.add_argument('--pose-every', type=int, default=None,
                    help='override config.POSE_EVERY_N_FRAMES for this run')
    args = ap.parse_args()
    args.video = str(ORIG_CWD / args.video) if not Path(args.video).is_absolute() else args.video
    if args.pose_every:
        config.POSE_EVERY_N_FRAMES = args.pose_every

    frames = load_frames(args.video, 120)
    print(f"{len(frames)} frames of {frames[0].shape[1]}x{frames[0].shape[0]}; "
          f"{args.users} user(s); POSE_EVERY_N_FRAMES={config.POSE_EVERY_N_FRAMES}", flush=True)
    models = {'pairs': daisee_ensemble.build_pairs()}
    if not any(fn is not None for fn, _, _ in models['pairs']):
        sys.exit("CNN weights not found - numbers without the CNN would be meaningless")
    # Warm up TensorFlow/MediaPipe so the timed window measures steady state.
    warm = UserState()
    warm_run = LiveRun(warm, models, InterventionAgent(stats={}, mode_stats={},
                       task_mode='Default'), task_mode='Default', agent_enabled=False)
    for k in range(12):
        warm_run.process(frames[k], k / 10.0)
    warm.close()

    results = {}
    barrier = threading.Barrier(args.users)
    threads = [threading.Thread(target=worker, args=(i, frames, models, args.seconds,
                                                     results, barrier))
               for i in range(args.users)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    print(f"\n{'user':>6}{'frames':>9}{'fps':>8}{'ms/frame':>10}")
    total = 0
    for i in range(args.users):
        n, busy = results[i]
        total += n
        print(f"{i + 1:>6}{n:>9}{n / wall:>8.1f}{busy / n * 1000:>10.0f}")
    per_user = total / wall / args.users
    print(f"\nper-user average: {per_user:.1f} fps   total: {total / wall:.1f} fps")
    print("VERDICT:", "OK (>= 8 fps per user)" if per_user >= 8 else
          "TOO SLOW for this many users - add CPU, raise POSE_EVERY_N_FRAMES, or serve fewer users")


if __name__ == '__main__':
    main()
