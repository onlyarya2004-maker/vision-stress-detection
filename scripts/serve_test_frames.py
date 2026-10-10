"""
Serve a short video as a numbered-JPEG "fake camera" for end-to-end tests.

The browser_camera component normally reads the user's webcam. For automated
checks (where no camera is available - e.g. the Claude browser pane blocks
getUserMedia) it can instead read a folder of numbered JPEGs when the app is
started with STRESS_TEST_FRAMES_URL=http://localhost:<port>. This script
prepares that folder from any video file and serves it with the CORS header
the canvas needs.

    python scripts/serve_test_frames.py --video datasets/UBFC-Phys/s11/vid_s11_T1.avi \
        --seconds 20 --fps 10 --port 8765

Test data only. Nothing here is imported by the app.
"""
import argparse
import json
import shutil
import sys
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def log_message(self, *a):          # quiet
        pass


def pad_to_aspect(frame, aspect=4 / 3):
    """Letterbox to the given aspect (4:3 by default, or 16:9 to mimic a modern
    webcam) with black side bars. UBFC frames are square with the head near the
    top; the app's 4:3 centre crop of a square frame would trim the forehead
    off. A real camera has no such crop, so padding is the faithful stand-in
    (face geometry untouched). With 16:9 the app's own 4:3 side-crop applies,
    exactly as on a real 1080p webcam."""
    h, w = frame.shape[:2]
    target_w = int(round(h * aspect))
    if w >= target_w:
        return frame
    left = (target_w - w) // 2
    return cv2.copyMakeBorder(frame, 0, 0, left, target_w - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def extract(video, out_dir, seconds, fps, start=0.0, quality=95, aspect=4 / 3):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = src_fps / fps
    n_target = int(seconds * fps)
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    out_dir.mkdir(parents=True, exist_ok=True)
    n, next_take, idx = 0, 0.0, 0
    while n < n_target:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_take:
            if aspect:
                frame = pad_to_aspect(frame, aspect)
            cv2.imwrite(str(out_dir / f"{n:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            n += 1
            next_take += step
        idx += 1
    cap.release()
    (out_dir / "manifest.json").write_text(json.dumps({"n": n, "fps": fps}))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--start', type=float, default=0.0, help='seconds into the video')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--aspect', default='4:3', help="letterbox to this aspect: 4:3 or 16:9")
    args = ap.parse_args()
    num, den = args.aspect.split(':')
    aspect = float(num) / float(den)

    out_dir = Path(tempfile.mkdtemp(prefix='stress_test_frames_'))
    try:
        n = extract(Path(args.video), out_dir, args.seconds, args.fps, args.start, aspect=aspect)
        print(f"{n} frames from {args.video} -> {out_dir}", flush=True)
        handler = partial(CORSHandler, directory=str(out_dir))
        srv = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
        print(f"serving on http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
