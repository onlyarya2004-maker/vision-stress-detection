"""
CLIP "hints" for the context suggestion (see context_detector.py).

The scene model only knows ROOM types. This second model compares the frame
with sentences about people and objects ("a doctor wearing a white coat", "a
student writing in a notebook", "a person sitting inside a car" ...) and
reports how much of its probability falls on each group (Studying / Hospital /
Vehicle / Other). Runs locally through OpenCV's DNN (no new dependency);
model files live in models/context/clip/ (gitignored - too large for plain
Git - but under models/, which the Dockerfile copies) and are produced once by
scripts/convert_clip_to_onnx.py.

Hints are support, not a verdict: a zero-shot guess about what is in a webcam
picture. They are never shown as an accuracy.
"""
import os
import threading
from pathlib import Path

import cv2
import numpy as np

HINT_DIR = Path(os.environ.get('STRESS_AUTO_CONTEXT_CLIP_DIR')
                or Path(__file__).resolve().parent.parent / 'models' / 'context' / 'clip')
IMAGE_MODEL = 'clip_image_encoder.onnx'
TEXT_FILE = 'clip_text_embeds.npz'

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class HintsUnavailable(RuntimeError):
    """The CLIP hint model is missing or could not be loaded."""


def preprocess(frame_bgr):
    """CLIP's own recipe: shortest side to 224 (bicubic), centre crop, normalise."""
    h, w = frame_bgr.shape[:2]
    s = 224.0 / min(h, w)
    img = cv2.resize(frame_bgr, (max(224, round(w * s)), max(224, round(h * s))),
                     interpolation=cv2.INTER_CUBIC)
    y, x = (img.shape[0] - 224) // 2, (img.shape[1] - 224) // 2
    rgb = cv2.cvtColor(img[y:y + 224, x:x + 224], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None])


class ClipHints:
    def __init__(self, model_dir=None):
        d = Path(model_dir or HINT_DIR)
        if not (d / IMAGE_MODEL).is_file() or not (d / TEXT_FILE).is_file():
            raise HintsUnavailable(f"CLIP hint model not found in {d} (see models/context/README.md)")
        try:
            t = np.load(d / TEXT_FILE)
            self.text = t['text'].astype(np.float32)                  # (n, 512), unit length
            self.prompts = [str(p) for p in t['prompts']]
            self.groups = [str(g) for g in t['groups']]
            self.scale = float(t['logit_scale'])
            self.net = cv2.dnn.readNetFromONNX(str(d / IMAGE_MODEL))
        except Exception as e:
            raise HintsUnavailable(f"CLIP hint model failed to load: {e}") from e
        self._lock = threading.Lock()

    def predict(self, frame_bgr):
        """({group: probability}, [(sentence, probability)] top 2) for one frame."""
        blob = preprocess(frame_bgr)
        with self._lock:
            self.net.setInput(blob)
            emb = self.net.forward().reshape(-1)
        emb = emb / np.linalg.norm(emb)
        logits = self.scale * (self.text @ emb)
        z = np.exp(logits - logits.max())
        p = z / z.sum()
        groups = {}
        for g, v in zip(self.groups, p):
            groups[g] = groups.get(g, 0.0) + float(v)
        top = [(self.prompts[i].replace('a photo of ', ''), float(p[i]))
               for i in np.argsort(p)[::-1][:2]]
        return groups, top


_HINTS = None
_HINTS_LOCK = threading.Lock()


def get_hints():
    """Shared instance, loaded on first use; a failed load is not cached."""
    global _HINTS
    with _HINTS_LOCK:
        if _HINTS is None:
            _HINTS = ClipHints()
        return _HINTS
