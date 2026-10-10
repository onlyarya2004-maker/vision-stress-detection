"""
browser_camera - a Streamlit custom component that owns the camera in the
USER'S browser.

Why it exists: a hosted server has no camera, and streamlit-webrtc was tried
and rejected (it re-encodes the video and visibly degraded the picture). Here
the browser plays the camera locally at full resolution - no network hop, no
re-encoding, no lag in what the user sees - and, separately, posts small
640x480 JPEG snapshots (~10/s, batched every 0.5 s) to the server over
Streamlit's ordinary HTTPS/WebSocket connection. The server runs the exact
validated Python models on them (session_pipeline.py) and sends back the
label/colour/face-box to draw over the local video.

Protocol (see frontend/index.html):
  Python -> browser (args): mode ('idle'|'calibrate'|'live'), epoch (bumped on
      every Calibrate/Start/Stop so late frames from an old run are ignored),
      ack (last batch sequence the server processed - the browser sends the
      next batch only after this catches up: backpressure), overlay (label,
      colour, bbox in 640x480 processing coordinates, progress, note), cfg.
  Browser -> Python (return value, a dict): epoch, seq, frames
      [{t: seconds since the run began, j: base64 JPEG}], tz (UTC offset
      minutes), optional err (camera error code).

The returned value is the browser's LAST value and is re-delivered on every
rerun it did not cause; session_pipeline.BatchGate de-duplicates by
(epoch, seq). Call this from inside an @st.fragment so each batch reruns only
the fragment, not the whole page.
"""
import os
from pathlib import Path

import streamlit.components.v1 as components

import config

_FRONTEND = Path(__file__).parent / "frontend"
_component = components.declare_component("browser_camera", path=str(_FRONTEND))


def browser_camera(*, mode, epoch, ack, overlay=None, test_url="", key="browser_camera"):
    """Render the camera view. See the module docstring for the protocol."""
    cfg = {
        'fps': config.BROWSER_CAPTURE_FPS,
        'quality': config.BROWSER_JPEG_QUALITY,
        'batch_ms': config.BROWSER_BATCH_MS,
        'proc_w': config.PROCESS_WIDTH,
        'proc_h': config.PROCESS_HEIGHT,
        'test_url': test_url or "",
        # Cosmetic in-browser face box (downloads ~10 MB of MediaPipe WASM from a
        # CDN on first use). STRESS_LOCAL_FACE_BOX=0 turns it off, e.g. for a
        # venue network that blocks the CDN; the server's box is drawn instead.
        'local_box': os.environ.get('STRESS_LOCAL_FACE_BOX', '1') != '0',
    }
    return _component(mode=mode, epoch=int(epoch), ack=int(ack),
                      overlay=overlay or {}, cfg=cfg, key=key, default=None)
