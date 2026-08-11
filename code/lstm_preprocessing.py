"""
LSTM Preprocessing
==================
Re-organizes the existing per-frame face crops in processed_daisee/ into
16-frame SEQUENCES, one sequence per DaiSEE clip.

The frame extraction was already done by daisee_preprocessing.py - that
script saved frames as `<clipid>_f00.jpg ... <clipid>_f15.jpg`. Here we
just BUILD AN INDEX that groups them by clip_id, so the LSTM training
script can load whole sequences efficiently.

Output: lstm_index.pkl - a pickled dict:
  {
    'train': [(clip_id, label, [path_f00, ..., path_f15]), ...],
    'val':   [...],
    'test':  [...],
  }

label: 0 = not_stressed, 1 = stressed (binary, same as DaiSEE rebucketing)

Why 16 frames per clip?
  DaiSEE clips are 10 seconds. We sampled 16 frames uniformly across each
  clip during preprocessing - so 16 frames = the full temporal context of
  a 10-second affective state, sampled at ~1.6 Hz.

Run AFTER:
  - daisee_preprocessing.py (already extracted the frames)
  - rebucket_to_binary.py  (already organized into not_stressed/stressed)

This is fast (~1-2 minutes for all splits) since we don't touch images.
"""

import re
import pickle
from pathlib import Path
from collections import defaultdict, Counter

# ---------------- CONFIG ----------------
PROCESSED_ROOT = Path('../datasets/processed_daisee')
OUTPUT_INDEX   = Path('../datasets/processed_daisee/lstm_index.pkl')

CLASS_TO_LABEL = {
    'not_stressed': 0,
    'stressed':     1,
}

# Frames per sequence. daisee_preprocessing.py used 16, but we tolerate
# clips that have fewer (videos that were too short for some frames).
TARGET_FRAMES = 16
MIN_FRAMES_TO_KEEP = 12   # if a clip has fewer frames than this, skip it

# Filename pattern: <clipid>_f<frame_idx>.jpg
# clipid can contain underscores so we're strict about the _f suffix.
FRAME_RE = re.compile(r'^(.+)_f(\d{1,3})\.jpg$', re.IGNORECASE)


def index_split(split_name):
    """
    Walk processed_daisee/<split>/<class>/*.jpg, group by clip_id, and
    return a list of (clip_id, label, sorted_frame_paths) tuples.
    """
    split_dir = PROCESSED_ROOT / split_name
    if not split_dir.exists():
        print(f"[skip] {split_dir} does not exist")
        return []

    print(f"\n=== Indexing split: {split_name} ===")
    # clip_id -> {'label': int, 'frames': {frame_idx: path}}
    clips = defaultdict(lambda: {'label': None, 'frames': {}})

    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"  [warn] {class_dir} missing")
            continue

        files = list(class_dir.glob('*.jpg'))
        print(f"  scanning {class_name}: {len(files)} files")

        for f in files:
            match = FRAME_RE.match(f.name)
            if not match:
                continue
            clip_id, frame_idx_str = match.group(1), match.group(2)
            frame_idx = int(frame_idx_str)
            entry = clips[clip_id]
            entry['label'] = label
            entry['frames'][frame_idx] = str(f)

    # Convert to list of tuples, sorted by frame index per clip
    sequences = []
    skipped_too_short = 0
    skipped_no_label = 0
    for clip_id, entry in clips.items():
        if entry['label'] is None:
            skipped_no_label += 1
            continue
        sorted_frames = [entry['frames'][i] for i in sorted(entry['frames'].keys())]
        if len(sorted_frames) < MIN_FRAMES_TO_KEEP:
            skipped_too_short += 1
            continue
        # If we have more than TARGET_FRAMES, sample evenly
        # If we have fewer, repeat the last frame to pad
        if len(sorted_frames) > TARGET_FRAMES:
            step = len(sorted_frames) / TARGET_FRAMES
            indices = [int(i * step) for i in range(TARGET_FRAMES)]
            sorted_frames = [sorted_frames[i] for i in indices]
        elif len(sorted_frames) < TARGET_FRAMES:
            while len(sorted_frames) < TARGET_FRAMES:
                sorted_frames.append(sorted_frames[-1])
        sequences.append((clip_id, entry['label'], sorted_frames))

    label_counts = Counter(s[1] for s in sequences)
    print(f"  total clips indexed: {len(sequences)}")
    print(f"    label distribution: not_stressed={label_counts[0]}, "
          f"stressed={label_counts[1]}")
    print(f"    skipped (too few frames): {skipped_too_short}")
    print(f"    skipped (no label):       {skipped_no_label}")
    return sequences


def main():
    if not PROCESSED_ROOT.exists():
        raise SystemExit(f"Cannot find {PROCESSED_ROOT.resolve()}. "
                         f"Run daisee_preprocessing.py first.")

    index = {}
    for split in ['train', 'val', 'test']:
        index[split] = index_split(split)

    with open(OUTPUT_INDEX, 'wb') as f:
        pickle.dump(index, f)

    print("\n" + "=" * 60)
    print("LSTM INDEX SUMMARY")
    print("=" * 60)
    for split, seqs in index.items():
        if seqs:
            labels = Counter(s[1] for s in seqs)
            print(f"  {split:<6} {len(seqs):>5} sequences  "
                  f"(not_stressed={labels[0]}, stressed={labels[1]})")
    print(f"\nIndex saved to: {OUTPUT_INDEX.resolve()}")
    print("Now run: python train_lstm.py")


if __name__ == '__main__':
    main()
