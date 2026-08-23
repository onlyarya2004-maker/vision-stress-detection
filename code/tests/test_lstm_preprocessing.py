"""
Tests for lstm_preprocessing.py's frame-grouping/subsampling/padding logic.

RETIRED code (kept only for provenance, per README) - this is a lighter
pass than the live DaiSEE pipeline gets, but index_split()'s per-clip
frame-count normalization (sample down / pad up to TARGET_FRAMES) is a
self-contained, deterministic function worth locking in.
"""

import pytest

import lstm_preprocessing as lp


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_clip_frames(split_dir, class_name, clip_id, n_frames):
    for i in range(n_frames):
        _touch(split_dir / class_name / f"{clip_id}_f{i:02d}.jpg")


def test_frame_regex_splits_clip_id_with_underscores():
    m = lp.FRAME_RE.match("subj_001_clip_02_f07.jpg")
    assert m is not None
    assert m.group(1) == "subj_001_clip_02"
    assert int(m.group(2)) == 7


def test_frame_regex_rejects_non_matching_names():
    assert lp.FRAME_RE.match("subj_001.jpg") is None


def test_index_split_passes_through_exact_target_length(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, 'PROCESSED_ROOT', tmp_path)
    split_dir = tmp_path / 'train'
    _make_clip_frames(split_dir, 'stressed', 'clipA', lp.TARGET_FRAMES)

    seqs = lp.index_split('train')

    assert len(seqs) == 1
    clip_id, label, frames = seqs[0]
    assert clip_id == 'clipA'
    assert label == 1
    assert len(frames) == lp.TARGET_FRAMES
    assert frames == sorted(frames)  # already in frame-index order


def test_index_split_pads_short_clips_by_repeating_last_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, 'PROCESSED_ROOT', tmp_path)
    split_dir = tmp_path / 'train'
    n = lp.MIN_FRAMES_TO_KEEP + 1   # >= min, < target
    _make_clip_frames(split_dir, 'not_stressed', 'clipB', n)

    seqs = lp.index_split('train')

    assert len(seqs) == 1
    _clip_id, label, frames = seqs[0]
    assert label == 0
    assert len(frames) == lp.TARGET_FRAMES
    pad_count = lp.TARGET_FRAMES - n
    assert frames[-pad_count:] == [frames[n - 1]] * pad_count
    assert len(set(frames[:n])) == n  # the real frames aren't touched


def test_index_split_drops_clips_below_min_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, 'PROCESSED_ROOT', tmp_path)
    split_dir = tmp_path / 'train'
    _make_clip_frames(split_dir, 'stressed', 'tooShort', lp.MIN_FRAMES_TO_KEEP - 1)

    assert lp.index_split('train') == []


def test_index_split_subsamples_long_clips_deterministically(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, 'PROCESSED_ROOT', tmp_path)
    split_dir = tmp_path / 'train'
    n = 20
    _make_clip_frames(split_dir, 'stressed', 'clipLong', n)

    seqs = lp.index_split('train')

    assert len(seqs) == 1
    _clip_id, _label, frames = seqs[0]
    assert len(frames) == lp.TARGET_FRAMES
    got_indices = [int(f.split('_f')[-1].replace('.jpg', '')) for f in frames]
    step = n / lp.TARGET_FRAMES
    expected_indices = [int(i * step) for i in range(lp.TARGET_FRAMES)]
    assert got_indices == expected_indices


def test_index_split_missing_split_dir_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, 'PROCESSED_ROOT', tmp_path)
    assert lp.index_split('does_not_exist') == []
