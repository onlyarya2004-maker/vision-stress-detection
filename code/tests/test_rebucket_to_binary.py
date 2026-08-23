"""
Tests for rebucket_to_binary.py's 3-class -> binary label mapping.

A bug in MAPPING silently mislabels every frame trained on downstream, so
this is the highest-value part of the offline pipeline to lock down: the
static dict itself, the file-moving logic that applies it, and a
cross-check that it actually covers every bucket daisee_preprocessing.py
can produce.
"""

import pytest

import daisee_preprocessing
import rebucket_to_binary


def test_mapping_covers_exactly_the_three_daisee_buckets():
    assert set(rebucket_to_binary.MAPPING.keys()) == {'low', 'moderate', 'high'}


def test_mapping_matches_documented_binary_scheme():
    assert rebucket_to_binary.MAPPING == {
        'low': 'not_stressed',
        'moderate': 'stressed',
        'high': 'stressed',
    }


def test_mapping_only_produces_the_two_binary_labels():
    assert set(rebucket_to_binary.MAPPING.values()) == {'not_stressed', 'stressed'}


def test_mapping_covers_every_bucket_stress_bucket_can_produce():
    """
    Every label stress_bucket() can emit must be a key MAPPING knows how to
    translate - otherwise some frames would silently end up in a folder
    nothing downstream ever reads (or rebucket_split() would KeyError).
    """
    for score in (0.0, 0.1, 0.32, 0.33, 0.5, 0.65, 0.66, 0.9, 1.0):
        bucket = daisee_preprocessing.stress_bucket(score)
        assert bucket in rebucket_to_binary.MAPPING


def _make_split(tmp_path, counts):
    """Populate tmp_path/train/<bucket>/*.jpg with `n` dummy files each."""
    split_dir = tmp_path / 'train'
    for bucket, n in counts.items():
        bucket_dir = split_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (bucket_dir / f"{bucket}_{i:03d}.jpg").write_bytes(b"fake")
    return split_dir


def test_rebucket_split_moves_files_into_binary_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(rebucket_to_binary, 'PROCESSED_ROOT', tmp_path)
    split_dir = _make_split(tmp_path, {'low': 3, 'moderate': 2, 'high': 4})

    rebucket_to_binary.rebucket_split('train')

    not_stressed = list((split_dir / 'not_stressed').glob('*.jpg'))
    stressed = list((split_dir / 'stressed').glob('*.jpg'))
    assert len(not_stressed) == 3
    assert len(stressed) == 6  # 2 moderate + 4 high


def test_rebucket_split_removes_now_empty_source_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(rebucket_to_binary, 'PROCESSED_ROOT', tmp_path)
    split_dir = _make_split(tmp_path, {'low': 1, 'moderate': 1, 'high': 1})

    rebucket_to_binary.rebucket_split('train')

    assert not (split_dir / 'low').exists()
    assert not (split_dir / 'moderate').exists()
    assert not (split_dir / 'high').exists()


def test_rebucket_split_handles_name_collision_with_bucket_prefix(tmp_path, monkeypatch):
    """moderate and high both map to 'stressed' - a same-named file from each
    must not silently overwrite the other."""
    monkeypatch.setattr(rebucket_to_binary, 'PROCESSED_ROOT', tmp_path)
    split_dir = tmp_path / 'train'
    (split_dir / 'moderate').mkdir(parents=True)
    (split_dir / 'high').mkdir(parents=True)
    (split_dir / 'moderate' / 'clip1.jpg').write_bytes(b"m")
    (split_dir / 'high' / 'clip1.jpg').write_bytes(b"h")

    rebucket_to_binary.rebucket_split('train')

    stressed_files = sorted(p.name for p in (split_dir / 'stressed').glob('*.jpg'))
    # 'moderate' is processed first (dict insertion order), so it keeps the
    # plain name; 'high' collides and gets prefixed rather than clobbering it.
    assert stressed_files == ['clip1.jpg', 'high_clip1.jpg']
    assert (split_dir / 'stressed' / 'clip1.jpg').read_bytes() == b"m"
    assert (split_dir / 'stressed' / 'high_clip1.jpg').read_bytes() == b"h"


def test_rebucket_split_skips_missing_split_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rebucket_to_binary, 'PROCESSED_ROOT', tmp_path)
    rebucket_to_binary.rebucket_split('does_not_exist')
    assert 'skip' in capsys.readouterr().out
