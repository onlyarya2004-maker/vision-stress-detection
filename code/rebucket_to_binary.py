"""
Re-bucket the existing processed_daisee frames from 3 classes
(low/moderate/high) into 2 classes (not_stressed/stressed).

Mapping:
    low       -> not_stressed
    moderate  -> stressed
    high      -> stressed

This is a quick one-time fix-up so we don't need to re-extract 137k face crops.
Just moves files between folders. Takes 1-2 minutes total.
"""

import shutil
from pathlib import Path

PROCESSED_ROOT = Path('../datasets/processed_daisee')

# split_name -> {old_bucket: new_bucket}
MAPPING = {
    'low':      'not_stressed',
    'moderate': 'stressed',
    'high':     'stressed',
}


def rebucket_split(split_name):
    split_dir = PROCESSED_ROOT / split_name
    if not split_dir.exists():
        print(f"[skip] {split_dir} doesn't exist")
        return

    print(f"\n=== Re-bucketing split: {split_name} ===")

    # Make new target directories
    for new_bucket in set(MAPPING.values()):
        (split_dir / new_bucket).mkdir(parents=True, exist_ok=True)

    moved_counts = {'not_stressed': 0, 'stressed': 0}

    for old_bucket, new_bucket in MAPPING.items():
        src_dir = split_dir / old_bucket
        dst_dir = split_dir / new_bucket
        if not src_dir.exists():
            continue

        files = list(src_dir.glob('*.jpg'))
        print(f"  moving {len(files):>6} files: {old_bucket} -> {new_bucket}")

        for f in files:
            target = dst_dir / f.name
            # If a file with the same name somehow already exists, prefix with old bucket
            if target.exists():
                target = dst_dir / f"{old_bucket}_{f.name}"
            shutil.move(str(f), str(target))
            moved_counts[new_bucket] += 1

        # Remove the now-empty old folder
        try:
            src_dir.rmdir()
        except OSError:
            pass  # not empty for some reason; leave it

    print(f"  Result: not_stressed={moved_counts['not_stressed']}, "
          f"stressed={moved_counts['stressed']}")


def main():
    if not PROCESSED_ROOT.exists():
        raise SystemExit(f"Cannot find {PROCESSED_ROOT.resolve()}")

    print(f"Processing root: {PROCESSED_ROOT.resolve()}")
    for split in ['train', 'val', 'test']:
        rebucket_split(split)

    print("\nDone. Verify with:")
    print(f"  dir {PROCESSED_ROOT.resolve()}\\train")
    print("You should now see 2 folders: not_stressed and stressed.")


if __name__ == '__main__':
    main()
