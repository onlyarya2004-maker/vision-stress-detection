"""
Pure per-frame image helpers (processing geometry, JPEG decode, face crop,
on-frame annotation).

Split out of app.py so session_pipeline.py can use them without importing
app.py (app.py imports session_pipeline - importing back would be circular).
Pure OpenCV/numpy, no Streamlit. crop_face/annotate_frame are moved verbatim
from app.py (no behaviour change); the processing-geometry and JPEG-decode
helpers are new for the browser-capture path.
"""
import cv2
import numpy as np

import config


def prepare_processing_frame(frame_bgr):
    """
    Bring any camera frame to the one geometry the models are validated on:
    a centred 4:3 crop, resized to config.PROCESS_WIDTH x PROCESS_HEIGHT
    (640x480). Returns (processed_frame, geom) where geom = (x0, y0, cw, ch)
    is the crop rectangle in the SOURCE frame, so a bbox found on the
    processed frame can be drawn back on the full-resolution picture
    (bbox_to_source). A frame already at the target size passes through
    untouched, so the browser path (which encodes exactly 640x480) and the
    old 640x480 local path cost nothing here.
    """
    h, w = frame_bgr.shape[:2]
    pw, ph = config.PROCESS_WIDTH, config.PROCESS_HEIGHT
    if (w, h) == (pw, ph):
        return frame_bgr, (0, 0, w, h)
    target = pw / ph
    if w / h > target:                       # too wide: trim the sides
        cw, ch = int(round(h * target)), h
    else:                                    # too tall: trim top/bottom
        cw, ch = w, int(round(w / target))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    cropped = frame_bgr[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (pw, ph), interpolation=cv2.INTER_AREA), (x0, y0, cw, ch)


def bbox_to_source(bbox, geom):
    """Map an (x1, y1, x2, y2) box from the processed frame back into the
    source frame that prepare_processing_frame() was given. None passes
    through (no face)."""
    if bbox is None:
        return None
    x0, y0, cw, ch = geom
    sx, sy = cw / config.PROCESS_WIDTH, ch / config.PROCESS_HEIGHT
    x1, y1, x2, y2 = bbox
    return (int(x0 + x1 * sx), int(y0 + y1 * sy), int(x0 + x2 * sx), int(y0 + y2 * sy))


def decode_jpeg(data):
    """
    Decode one browser-supplied JPEG to a BGR frame, or None if it is
    unusable. Everything arriving from a browser is untrusted: refuse
    oversized payloads before touching the decoder, and anything that does
    not decode to a real image (or decodes to something absurdly large).
    """
    if not data or len(data) > config.MAX_FRAME_BYTES:
        return None
    try:
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if frame is None or frame.ndim != 3:
        return None
    h, w = frame.shape[:2]
    if w > 4096 or h > 4096 or w < 64 or h < 64:
        return None
    return frame


def crop_face(frame_bgr, face_mesh_results):
    if not face_mesh_results.multi_face_landmarks:
        return None, None
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
        return None, None
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def annotate_frame(frame_bgr, bbox, level, smoothed_index, calibrating=False, scale=1.0):
    """Draw the face box + label on a frame. scale=1.0 is the original 640x480
    look; the local camera path passes frame_width/640 so the drawing keeps
    the same apparent size on a 1080p picture."""
    if calibrating:
        color = (237, 158, 47); label = "CALIBRATING"     # amber, BGR
    else:
        color = {'LOW': (34, 197, 34), 'MODERATE': (0, 158, 245),
                 'HIGH': (68, 68, 239)}[level]              # BGR of green/amber/red
        label = f"{level} ({smoothed_index*100:.0f}%)"
    s = max(1.0, float(scale))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, max(1, round(2 * s)))
        corner = int(14 * s)
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y2, -1, -1)):
            cv2.line(frame_bgr, (cx, cy), (cx + dx * corner, cy), color, max(1, round(3 * s)))
            cv2.line(frame_bgr, (cx, cy), (cx, cy + dy * corner), color, max(1, round(3 * s)))
    cv2.putText(frame_bgr, label, (int(15 * s), int(30 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, color, max(1, round(2 * s)))
