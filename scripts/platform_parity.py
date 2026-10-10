"""
Does the Linux container reproduce the Windows numbers?

The accuracy figures in the report were produced on Windows with MediaPipe
0.10.14. The hosted image runs the SAME version on Linux (different compiled
binaries). This script runs the validated feature engine over the first N
frames of a video (no calibration - generic defaults - so the result depends
only on the landmark/feature code) and prints per-signal means plus the
weighted behaviour score. Run it on both platforms and compare:

    python scripts/platform_parity.py --video datasets/UBFC-Phys/s11/vid_s11_T1.avi
    docker run --rm -v <repo>\\datasets\\UBFC-Phys\\s11:/data:ro -v <repo>\\scripts:/app/scripts stress-app \\
        python ../scripts/platform_parity.py --video /data/vid_s11_T1.avi

Expect agreement to ~3 decimals (the TFLite models are identical).
"""
import argparse
import os
import sys
from pathlib import Path

ORIG_CWD = Path.cwd()
CODE_DIR = Path(__file__).resolve().parent.parent / 'code'
sys.path.insert(0, str(CODE_DIR))
os.chdir(CODE_DIR)

import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402

import config                                # noqa: E402
from behavioral_features import BehavioralFeatureEngine   # noqa: E402

SIGNALS = list(config.BEHAVIOR_WEIGHTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--video', required=True)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--stride', type=int, default=3)
    args = ap.parse_args()
    video = args.video if Path(args.video).is_absolute() else str(ORIG_CWD / args.video)

    clock = {'t': 0.0}
    eng = BehavioralFeatureEngine(time_source=lambda: clock['t'])
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, idx = [], 0
    while len(rows) < args.frames:
        ok, f = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            f = cv2.resize(f, (480, 480), interpolation=cv2.INTER_AREA)
            clock['t'] = idx / fps
            feats = eng.extract(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if feats is not None:
                rows.append(feats)
        idx += 1
    cap.release()
    print(f"platform: {sys.platform}   frames scored: {len(rows)}")
    print(f"{'signal':18s}{'mean':>9}{'std':>9}")
    for k in ['ear'] + SIGNALS:
        v = np.array([r.get(k, 0.0) for r in rows])
        print(f"{k:18s}{v.mean():>9.4f}{v.std():>9.4f}")
    beh = np.array([sum(config.BEHAVIOR_WEIGHTS[s] * r.get(s, 0.0) for s in SIGNALS) for r in rows])
    print(f"{'BEHAVIOUR SCORE':18s}{beh.mean():>9.4f}{beh.std():>9.4f}")


if __name__ == '__main__':
    main()
