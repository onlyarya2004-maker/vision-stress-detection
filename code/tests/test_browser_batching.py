"""
Browser-side batch trimming (browser_camera/frontend/index.html).

The server keeps only the newest MAX_FRAMES_PROCESSED_PER_BATCH snapshots of a
batch (session_pipeline.BatchGate). The browser used to upload its whole
buffer (up to 40 JPEGs) anyway; it now uploads only those newest frames.

What is pinned down here:
  * the shipped JavaScript helper keeps the newest N in chronological order
    (run with Node against the function text taken from index.html itself);
  * the send path actually uses it, with N from the server's own config;
  * the server processes EXACTLY the same frames whether it receives the old
    full batch or the new trimmed one - so models, timestamps, sampling and
    session semantics are unchanged; only the upload got smaller.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import config
from session_pipeline import BatchGate
from test_session_pipeline import _batch, _jpeg_b64

HTML = Path(__file__).resolve().parent.parent / 'browser_camera' / 'frontend' / 'index.html'
SRC = HTML.read_text(encoding='utf-8')
N = config.MAX_FRAMES_PROCESSED_PER_BATCH


def _js_function(name):
    m = re.search(r'function ' + name + r'\(buf, max\) \{.*?\n  \}', SRC, re.S)
    assert m, f"{name}() not found in index.html"
    return m.group(0)


def _run_newest_frames(cases):
    """Evaluate the real newestFrames() from index.html in Node."""
    node = shutil.which('node')
    if node is None:
        pytest.skip("Node.js is not installed")
    script = (_js_function('newestFrames') + '\n'
              + 'const cases = ' + json.dumps(cases) + ';\n'
              + 'const out = cases.map(c => { const before = JSON.stringify(c.buf);\n'
              + '  const r = newestFrames(c.buf, c.max);\n'
              + '  return {r: r, mutated: JSON.stringify(c.buf) !== before}; });\n'
              + 'process.stdout.write(JSON.stringify(out));')
    res = subprocess.run([node, '-e', script], capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def _frames(k):
    return [{'t': round(0.1 * (i + 1), 3), 'j': f'jpeg{i}'} for i in range(k)]


# ---------------------------------------------------------------------------
# the shipped JavaScript
# ---------------------------------------------------------------------------
def test_browser_keeps_only_the_newest_frames_in_chronological_order():
    buf = _frames(40)                                  # the old worst case: a full buffer
    (out,) = _run_newest_frames([{'buf': buf, 'max': N}])
    assert out['r'] == buf[-N:]                        # newest N, oldest first
    assert [f['t'] for f in out['r']] == sorted(f['t'] for f in out['r'])
    assert not out['mutated']


@pytest.mark.parametrize('k', [0, 1, 5, N])
def test_a_batch_within_the_limit_is_sent_unchanged(k):
    buf = _frames(k)
    (out,) = _run_newest_frames([{'buf': buf, 'max': N}])
    assert out['r'] == buf


@pytest.mark.parametrize('bad', [None, 0, -3, 'x', 2.9])
def test_a_missing_or_invalid_limit_falls_back_safely(bad):
    buf = _frames(20)
    (out,) = _run_newest_frames([{'buf': buf, 'max': bad}])
    expected = 2 if bad == 2.9 else 8                  # floor(2.9) = 2; junk -> 8
    assert out['r'] == buf[-expected:]


def test_the_send_path_trims_with_the_server_supplied_limit():
    send = re.search(r'function maybeSend\(\) \{.*?\n  \}', SRC, re.S).group(0)
    assert 'newestFrames(buffer.splice(0, buffer.length), cfg.max_frames)' in send
    assert re.search(r'let cfg = \{[^}]*max_frames: 8', SRC)       # safe default
    # Unchanged protocol fields: same keys, same per-frame shape.
    assert "ST.setValue({ epoch: epoch, seq: seq, frames: frames, tz:" in send
    assert "buffer.push({ t: +((performance.now() - tOrigin) / 1000).toFixed(3), j: j });" in SRC


def test_the_server_passes_its_own_processing_limit_to_the_browser(monkeypatch):
    import browser_camera
    seen = {}
    monkeypatch.setattr(browser_camera, '_component', lambda **kw: seen.update(kw))
    browser_camera.browser_camera(mode='live', epoch=1, ack=0)
    assert seen['cfg']['max_frames'] == config.MAX_FRAMES_PROCESSED_PER_BATCH
    # Cadence/format settings are untouched.
    assert seen['cfg']['fps'] == config.BROWSER_CAPTURE_FPS
    assert seen['cfg']['quality'] == config.BROWSER_JPEG_QUALITY
    assert seen['cfg']['batch_ms'] == config.BROWSER_BATCH_MS


# ---------------------------------------------------------------------------
# compatibility with the unchanged server endpoint
# ---------------------------------------------------------------------------
def test_server_processes_identical_frames_from_a_full_or_a_trimmed_batch():
    times = [round(0.1 * (i + 1), 3) for i in range(40)]
    old_gate, new_gate = BatchGate(), BatchGate()
    old_frames, old_ack = old_gate.accept(_batch(seq=1, times=times), 1)          # before
    new_frames, new_ack = new_gate.accept(_batch(seq=1, times=times[-N:]), 1)     # after
    assert [t for t, _ in old_frames] == [t for t, _ in new_frames] == times[-N:]
    assert old_ack == new_ack == 1
    assert old_gate.last_t == new_gate.last_t == times[-1]
    assert new_gate.dropped == 0                       # nothing wasted any more
    # The next batch continues identically for both.
    nxt = [round(times[-1] + 0.1 * (i + 1), 3) for i in range(5)]
    a, _ = old_gate.accept(_batch(seq=2, times=nxt), 1)
    b, _ = new_gate.accept(_batch(seq=2, times=nxt), 1)
    assert [t for t, _ in a] == [t for t, _ in b] == nxt


def test_server_accepts_a_full_trimmed_batch_in_order():
    times = [round(5.0 + 0.1 * i, 3) for i in range(N)]
    frames, ack = BatchGate().accept(_batch(seq=1, times=times, jpeg=_jpeg_b64()), 1)
    assert [t for t, _ in frames] == times and ack == 1
    assert all(f.shape == (480, 640, 3) for _, f in frames)
