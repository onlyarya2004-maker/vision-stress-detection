"""
DaiSEE preprocessing → stress-labeled frame dataset
----------------------------------------------------
What this script does
=====================
1. Reads the 3 DaiSEE label CSVs (TrainLabels.csv / ValidationLabels.csv / TestLabels.csv).
2. Walks the DaiSEE dataset directory to locate each clip's video file
   (DaiSEE's folder layout is Dataset/{Train,Validation,Test}/{UserID}/{ClipFolder}/{ClipID}.avi).
3. For each clip:
     - derives a 3-class stress label (LOW / MODERATE / HIGH)
       from the 4 DaiSEE affect labels using a weighted formula that
       prioritises Frustration and Confusion (true cognitive-stress proxies),
     - samples N frames uniformly along the 10 s clip,
     - detects a face in each frame with MediaPipe FaceMesh
       and saves a tight face crop.
4. Saves everything into:
     processed_daisee/
       train/{low,moderate,high}/<clipid>_f0.jpg, ...
       val/{low,moderate,high}/
       test/{low,moderate,high}/
   so it can be loaded with `ImageDataGenerator.flow_from_directory` directly.

Why frame-level and not full-video?
-----------------------------------
Frame-level classification is the pragmatic first pass: it reuses the
MobileNet / EfficientNet backbones we already have, trains in hours (not days),
and the temporal smoothing module at inference time recovers most of the
"video context" benefit. Your future-work slide already earmarks a proper
CNN+LSTM upgrade — when you're ready, this pre-extracted frame structure
will still be reusable (just group frames by clip).

Usage
-----
Edit DAISEE_ROOT and LABELS_DIR to match your paths, then:

    python daisee_preprocessing.py

Will take a few hours on 8k clips. Use `MAX_CLIPS_PER_SPLIT` for a smoke test.
"""

import os
import csv
import glob
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import Counter


# ---------------- CONFIG ----------------
# Point these at YOUR machine's layout.
DAISEE_ROOT = Path('../datasets/DaiSEE/DataSet')            # contains Train/ Validation/ Test/
LABELS_DIR  = Path('../datasets/DaiSEE/Labels')             # contains *Labels.csv
OUTPUT_ROOT = Path('../datasets/processed_daisee')          # where stress-labeled frames will go

FRAMES_PER_CLIP = 16      # uniformly sampled across each 10-second video
IMG_SIZE        = 224     # final saved crop size (matches EfficientNet input)
MAX_CLIPS_PER_SPLIT = None   # e.g. 50 for a quick smoke test; None = all clips
FACE_MARGIN     = 0.15    # 15 % padding around face bbox
MIN_FACE_SCORE  = 0.5

# ---------------- STRESS LABEL DERIVATION ----------------
def stress_score_from_daisee(boredom, engagement, confusion, frustration):
    """
    Combine the four DaiSEE ordinal labels (0..3 each) into a single
    stress score in [0, 1]. Weights follow the affect-computing literature:
    frustration is the strongest correlate of cognitive stress, confusion
    is second, disengagement contributes mildly, boredom contributes least.
    """
    s = (0.45 * (frustration / 3.0)
         + 0.30 * (confusion   / 3.0)
         + 0.15 * (1.0 - engagement / 3.0)
         + 0.10 * (boredom    / 3.0))
    return float(np.clip(s, 0.0, 1.0))


def stress_bucket(score):
    if score < 0.33:
        return 'low'
    if score < 0.66:
        return 'moderate'
    return 'high'


# ---------------- CLIP ↔ VIDEO FILE RESOLUTION ----------------
def build_clip_index(split_root):
    """
    Walk a DaiSEE split folder once and build a dict: clip_id_without_ext -> full video path.
    Avoids re-scanning the filesystem for every label row.
    """
    index = {}
    if not split_root.exists():
        print(f"  [warn] split folder missing: {split_root}")
        return index
    patterns = ['*.avi', '*.mp4']
    for ext in patterns:
        for video_path in split_root.rglob(ext):
            index[video_path.stem] = video_path
    return index


# ---------------- FACE CROP ----------------
def crop_largest_face(frame_bgr, face_mesh):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    lm = results.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = int(FACE_MARGIN * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 + 20 or y2 <= y1 + 20:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


# ---------------- FRAME SAMPLER ----------------
def sample_frames(video_path, n_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total < n_frames:
        # very short / corrupt — grab what we can
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, n_frames).astype(int).tolist()
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


# ---------------- MAIN PIPELINE ----------------
def process_split(split_name, csv_path, split_root, output_root, face_mesh):
    """
    split_name: 'train' | 'val' | 'test'
    csv_path  : DaiSEE label CSV for this split
    split_root: DaiSEE Dataset folder for this split (Train/ Validation/ Test/)
    output_root: OUTPUT_ROOT / split_name
    """
    print(f"\n=== Processing split: {split_name} ===")
    print(f"    labels: {csv_path}")
    print(f"    videos: {split_root}")

    # Make output directories up front
    for bucket in ['low', 'moderate', 'high']:
        (output_root / bucket).mkdir(parents=True, exist_ok=True)

    # Index clip_id -> video path
    clip_index = build_clip_index(split_root)
    print(f"    indexed {len(clip_index)} video files")

    # Read labels
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # The DaiSEE CSV header ends with a trailing space on "Frustration " — normalise.
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]

    if MAX_CLIPS_PER_SPLIT:
        rows = rows[:MAX_CLIPS_PER_SPLIT]

    bucket_counts = Counter()
    skipped_no_video = 0
    skipped_no_face  = 0
    total_saved      = 0

    for i, row in enumerate(rows):
        clip_with_ext = row['ClipID'].strip()
        clip_stem = os.path.splitext(clip_with_ext)[0]

        try:
            bor  = int(row['Boredom'])
            eng  = int(row['Engagement'])
            con  = int(row['Confusion'])
            fru  = int(row['Frustration'])
        except (KeyError, ValueError):
            continue

        score  = stress_score_from_daisee(bor, eng, con, fru)
        bucket = stress_bucket(score)

        video_path = clip_index.get(clip_stem)
        if video_path is None:
            skipped_no_video += 1
            continue

        frames = sample_frames(video_path, FRAMES_PER_CLIP)
        saved_this_clip = 0
        for j, frame in enumerate(frames):
            crop = crop_largest_face(frame, face_mesh)
            if crop is None:
                continue
            out_path = output_root / bucket / f"{clip_stem}_f{j:02d}.jpg"
            cv2.imwrite(str(out_path), crop)
            saved_this_clip += 1
            total_saved += 1

        if saved_this_clip == 0:
            skipped_no_face += 1
        bucket_counts[bucket] += 1

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(rows)} clips processed "
                  f"(saved={total_saved} frames, "
                  f"no_video={skipped_no_video}, no_face={skipped_no_face})")

    print(f"\n    Split '{split_name}' summary:")
    print(f"      clips by bucket:  {dict(bucket_counts)}")
    print(f"      frames saved:     {total_saved}")
    print(f"      clips w/o video:  {skipped_no_video}")
    print(f"      clips w/o face:   {skipped_no_face}")


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=MIN_FACE_SCORE,
    )

    # DaiSEE's folder names vs. its label filenames:
    # Train.txt / TrainLabels.csv  → DataSet/Train
    # Validation.txt / ValidationLabels.csv → DataSet/Validation
    # Test.txt / TestLabels.csv   → DataSet/Test
    split_map = [
        ('train', LABELS_DIR / 'TrainLabels.csv',      DAISEE_ROOT / 'Train'),
        ('val',   LABELS_DIR / 'ValidationLabels.csv', DAISEE_ROOT / 'Validation'),
        ('test',  LABELS_DIR / 'TestLabels.csv',       DAISEE_ROOT / 'Test'),
    ]

    for split_name, csv_path, split_root in split_map:
        if not csv_path.exists():
            print(f"[skip] label file not found: {csv_path}")
            continue

        process_split(
            split_name,
            csv_path,
            split_root,
            OUTPUT_ROOT / split_name,
            face_mesh,
        )

    face_mesh.close()
    print("\nDone. Output at:", OUTPUT_ROOT.resolve())


if __name__ == '__main__':
    main()