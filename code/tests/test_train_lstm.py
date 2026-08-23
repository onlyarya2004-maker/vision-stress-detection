"""
Tests for train_lstm.py.

RETIRED code (pixel-LSTM, superseded, kept only for provenance). Almost the
entire file is TensorFlow model-building/training - not unit-testable
without real sequence data and hours of runtime. SequenceDataLoader's batch
bookkeeping is the one self-contained, deterministic piece: __len__ is pure
arithmetic, and __getitem__'s "missing frame falls back to zeros" branch is
cheap to exercise without any real image files on disk.
"""

import train_lstm as tl


def _dummy_sequences(n, missing_dir):
    return [
        (f"clip{i}", i % 2, [str(missing_dir / f"clip{i}_f{j:02d}.jpg") for j in range(16)])
        for i in range(n)
    ]


def test_sequence_data_loader_len_is_ceil_of_batch_count(tmp_path):
    loader = tl.SequenceDataLoader(_dummy_sequences(10, tmp_path), batch_size=4, shuffle=False)
    assert len(loader) == 3  # ceil(10 / 4)


def test_sequence_data_loader_exact_multiple_batch_count(tmp_path):
    loader = tl.SequenceDataLoader(_dummy_sequences(12, tmp_path), batch_size=4, shuffle=False)
    assert len(loader) == 3


def test_sequence_data_loader_missing_frame_falls_back_to_zeros(tmp_path):
    """cv2.imread() returns None for a frame path that doesn't exist on disk;
    the loader must skip it (leaving zeros) instead of raising."""
    frame_paths = [str(tmp_path / f"clip0_f{j:02d}.jpg") for j in range(16)]
    seq = [("clip0", 1, frame_paths)]  # label 1 = "stressed"
    loader = tl.SequenceDataLoader(seq, batch_size=1, shuffle=False, augment=False)

    X, Y = loader[0]

    assert X.shape == (1, tl.SEQ_LEN, tl.IMG_SIZE, tl.IMG_SIZE, 3)
    assert Y.shape == (1, 2)
    assert Y[0].tolist() == [0.0, 1.0]  # label 1 ("stressed") one-hot
