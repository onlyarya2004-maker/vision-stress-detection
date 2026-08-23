"""
Real-time Cognitive Stress Monitoring (v3 - FER removed)
========================================================
Standalone OpenCV runner. Alternative to the Streamlit app for quick
demos without a browser.

After A/B evaluation showed FER2013 hurt live-system accuracy (82.4% vs
89.1% without FER), the emotion branch was removed. Active branches:

  Webcam ─► Frame Extraction ─► Face Detection (MediaPipe FaceMesh)
                                        │
                  ┌─────────────────────┴─────────────────────┐
                  ▼                                           ▼
        DaiSEE stress model                      Behavioral Feature Engine
        (re-centered against                     (EAR, blink, brow, jaw,
         personal baseline)                       posture, head tilt)
                  └─────────────────────┬─────────────────────┘
                                        ▼
                            Stress Index Computation
                    (Weighted multi-signal fusion + rule boost)
                                        │
                                        ▼
                             Temporal Analysis Module
                                        │
                                        ▼
                            Stress Dashboard (LOW/MOD/HIGH)

Calibration is done at startup: press 'c' to calibrate (sit calm for 6s),
then the live monitor runs. Press 'q' to quit, 'r' to reset baseline.
"""

import time
import cv2
import numpy as np

from behavioral_features import BehavioralFeatureEngine
import config
import daisee_ensemble
from stress_fusion import (
    compute_stress_index, TemporalSmoother, stress_level,
    set_daisee_baseline, reset_daisee_baseline, is_daisee_calibrated,
)

# ---------------- CONFIG ----------------
SMOOTHING_WINDOW    = config.SMOOTHING_WINDOW_FRAMES_OPENCV
CALIBRATION_SECONDS = config.CALIBRATION_SECONDS
CNN_EVERY_N_FRAMES  = config.CNN_EVERY_N_FRAMES


# ================ HELPERS ================
def crop_face(frame_bgr, face_mesh_results):
    if not face_mesh_results.multi_face_landmarks:
        return None
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
        return None
    return frame_bgr[y1:y2, x1:x2]


# ================ DASHBOARD RENDERING ================
def draw_dashboard(frame, smoothed_index, result, calibrating=False, calib_progress=0.0):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if calibrating:
        cv2.putText(frame, f"CALIBRATING ({calib_progress*100:.0f}%) - sit calm",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 200, 255), 2)
        cv2.putText(frame, "Don't pose, just breathe normally",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        bar_x, bar_y, bar_w, bar_h = 15, 90, w - 30, 18
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill = int(bar_w * calib_progress)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill, bar_y + bar_h), (0, 200, 255), -1)
        return

    level, advice = stress_level(smoothed_index)
    color = {'LOW': (0, 200, 0),
             'MODERATE': (0, 200, 255),
             'HIGH': (0, 0, 255)}[level]

    cv2.putText(frame, f"Stress: {level}  ({smoothed_index*100:.0f}%)",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

    subline = []
    if result.get('daisee_stress') is not None:
        subline.append(f"daisee={result['daisee_stress']:.2f}")
    subline.append(f"behavior={result['behavior_stress']:.2f}")
    cv2.putText(frame, "  ".join(subline),
                (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.putText(frame, advice, (15, 87),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    bar_x, bar_y, bar_w, bar_h = 15, 105, w - 30, 18
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
    fill = int(bar_w * smoothed_index)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill, bar_y + bar_h), color, -1)

    y0 = 145
    cv2.putText(frame, "Behavioral signals:", (15, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, (name, value) in enumerate(result['contributions'].items()):
        y = y0 + 25 + i * 22
        cv2.putText(frame, f"{name}: {value:.2f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.rectangle(frame, (210, y - 10),
                      (210 + int(150 * value), y),
                      (0, int(255 * (1 - value)), int(255 * value)), -1)

    cv2.putText(frame, "[c]alibrate  [r]eset baseline  [q]uit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (180, 180, 180), 1)


# ================ CALIBRATION ================
def run_calibration(cap, engine, pairs, window_name):
    """Run a 6-second calibration to learn the user's baseline."""
    print(f"\n[calibration] starting - sit calm for {CALIBRATION_SECONDS}s...")
    engine.start_calibration()
    daisee_calib_samples = []
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ = engine.extract(rgb)

        face_mesh_results = engine.face_mesh.process(rgb)
        face_crop = crop_face(frame, face_mesh_results)
        elapsed = time.time() - start

        if face_crop is not None:
            if int(elapsed * 30) % CNN_EVERY_N_FRAMES == 0:
                daisee_probs = daisee_ensemble.ensemble_predict(face_crop, pairs)
                if daisee_probs is not None:
                    daisee_calib_samples.append(float(daisee_probs[1]))

        progress = min(1.0, elapsed / CALIBRATION_SECONDS)
        draw_dashboard(frame, 0.0, {}, calibrating=True, calib_progress=progress)
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        if elapsed >= CALIBRATION_SECONDS:
            face_ok = engine.finish_calibration()
            daisee_ok = False
            if len(daisee_calib_samples) >= 5:
                set_daisee_baseline(float(np.median(daisee_calib_samples)))
                daisee_ok = True
            if face_ok and daisee_ok:
                print(f"[calibration] done. face baseline + daisee baseline set.")
                return True
            else:
                print(f"[calibration] partial. face_ok={face_ok}, daisee_ok={daisee_ok}")
                return face_ok or daisee_ok


# ================ MAIN ================
def main():
    print("Loading models...")
    # Delegates loading + inference to daisee_ensemble.py, the same module
    # app.py and evaluate_video_file.py use - so this runner's predictions
    # are provably identical to the live/offline paths, not a hand-kept
    # duplicate that can silently drift from them.
    pairs = daisee_ensemble.build_pairs()
    n_loaded = sum(1 for infer_fn, _, _ in pairs if infer_fn is not None)
    print(f"  loaded {n_loaded}/{len(pairs)} DaiSEE model(s)")
    if n_loaded == 0:
        raise RuntimeError(
            "No DaiSEE models could be loaded. Run train_daisee_stress.py first.")

    engine   = BehavioralFeatureEngine()
    smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    window_name = "Cognitive Stress Monitor"
    print("\nKeys: [c] calibrate  [r] reset baseline  [q] quit")
    print("Tip: press 'c' first to calibrate, then the live monitor runs.\n")

    last_daisee_probs = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_mesh_results = engine.face_mesh.process(rgb)
            face_crop = crop_face(frame, face_mesh_results)
            if face_crop is None:
                cv2.putText(frame, "No face detected",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                if key == ord('c'): run_calibration(cap, engine, pairs, window_name)
                if key == ord('r'):
                    engine.reset_baseline(); reset_daisee_baseline()
                    print("[baseline] reset.")
                frame_idx += 1
                continue

            if frame_idx % CNN_EVERY_N_FRAMES == 0:
                last_daisee_probs = daisee_ensemble.ensemble_predict(face_crop, pairs)

            behavioral_features = engine.extract(rgb)
            if behavioral_features is None:
                frame_idx += 1
                continue

            result = compute_stress_index(
                daisee_probs=last_daisee_probs,
                behavioral_features=behavioral_features,
            )
            smoothed = smoother.update(result['stress_index'])

            draw_dashboard(frame, smoothed, result)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('c'):
                smoother.reset()
                run_calibration(cap, engine, pairs, window_name)
            if key == ord('r'):
                engine.reset_baseline(); reset_daisee_baseline()
                smoother.reset()
                print("[baseline] reset.")
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()


if __name__ == "__main__":
    main()