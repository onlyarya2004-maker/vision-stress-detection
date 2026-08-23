"""
Live rPPG branch
================
Heart rate from the webcam, in real time, as a fourth fusion branch.

Why this branch matters more than the others
--------------------------------------------
The project's own WESAD calibration ranks heart rate as the single most
discriminative stress feature it measured:

    hr_bpm        baseline 72.41  stress 97.49   Cohen's d  +1.61
    scr_per_min   baseline 35.75  stress 24.10              -1.77
    eda_std       baseline  0.02  stress  0.11              +0.72

Larger than every behavioural signal the live system uses. Until now it was
unreachable without a chest strap - rPPG makes it a property of pixels the
camera already captures, which is the difference between "physiological-grade
thresholds derived offline" and actually sensing physiology at runtime.

Validated on UBFC-Phys, 15 traces vs Empatica E4 ground truth:

    GREEN 16.57 bpm   CHROM 9.12 bpm   POS 3.52 bpm   (mean absolute error)

DISABLED BY DEFAULT (config.W_RPPG = 0.0)
-----------------------------------------
Those numbers come from a 1024x1024 MJPG industrial camera at 35 fps. A
laptop webcam is 640x480, heavily compressed, often 30 fps under variable
indoor lighting - and compression artefacts attack precisely the sub-1%
colour fluctuations rPPG depends on. There is currently NO evidence it works
on that hardware.

So the heart rate is computed and DISPLAYED while contributing zero weight.
Watch it against a finger on your wrist for thirty seconds; if it tracks,
raise W_RPPG. That is a cheap, honest validation the user can perform
themselves, and it beats shipping a number nobody has checked.
"""

import time
from collections import deque

import numpy as np

import config

# POS/CHROM live in analyze_rppg; sharing them keeps the live path and the
# offline validation provably identical.
from analyze_rppg import HR_MAX_HZ, HR_MIN_HZ, method_pos

# FaceMesh indices for the skin regions (see evaluate_video_file for the
# offline counterpart).
INNER_BROWS = (55, 285)
NOSE_TIP = 1
CHIN = 152
FACE_LEFT, FACE_RIGHT = 234, 454

# How long without a fresh buffer append before status()/estimate() treat
# the data as stale rather than a frozen-but-still-confident old reading.
# Set well above RPPG_WINDOW_SECONDS (the rolling analysis window) so
# ordinary per-frame jitter never trips it - this only fires on a genuine
# gap (e.g. a caller that keeps calling add_frame() through an extended
# face-loss period with no real new data).
RPPG_STALE_AFTER_SECONDS = 3.0 * config.RPPG_WINDOW_SECONDS


def _patch_mean_rgb(frame_bgr, cx, cy, half):
    h, w = frame_bgr.shape[:2]
    x1, x2 = int(max(0, cx - half)), int(min(w, cx + half))
    y1, y2 = int(max(0, cy - half)), int(min(h, cy + half))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    p = frame_bgr[y1:y2, x1:x2]
    return float(p[:, :, 2].mean()), float(p[:, :, 1].mean()), float(p[:, :, 0].mean())


def extract_face_rgb(frame_bgr, mesh_result):
    """
    Mean RGB of forehead / cheeks / whole face, anchored to landmarks so the
    regions track head movement. Returns None when no face is present.
    """
    if not mesh_result or not mesh_result.multi_face_landmarks:
        return None
    h, w = frame_bgr.shape[:2]
    lm = mesh_result.multi_face_landmarks[0].landmark

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    brow_mid = (pt(INNER_BROWS[0]) + pt(INNER_BROWS[1])) / 2
    chin, nose = pt(CHIN), pt(NOSE_TIP)
    fl, fr = pt(FACE_LEFT), pt(FACE_RIGHT)
    face_w = float(np.linalg.norm(fr - fl)) + 1e-6
    face_h = float(np.linalg.norm(chin - brow_mid)) + 1e-6

    fh = brow_mid + np.array([0.0, -0.35 * face_h])
    forehead = _patch_mean_rgb(frame_bgr, fh[0], fh[1], 0.14 * face_w)
    cl = _patch_mean_rgb(frame_bgr, *((fl + nose) / 2), 0.09 * face_w)
    cr = _patch_mean_rgb(frame_bgr, *((fr + nose) / 2), 0.09 * face_w)

    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    face = _patch_mean_rgb(frame_bgr, xs.mean(), ys.mean(), 0.30 * face_w)

    if forehead is None or face is None:
        return None
    cheeks = [c for c in (cl, cr) if c is not None]
    cheek = tuple(float(np.mean([c[i] for c in cheeks])) for i in range(3)) \
        if cheeks else forehead
    return {'forehead': forehead, 'cheek': cheek, 'face': face}


def spectral_quality(sig, fs):
    """
    Signal quality as the share of in-band power sitting at the dominant
    peak. Motion smears the spectrum, so a clean pulse scores high and a
    subject who is talking or moving scores low.

    This matters: on UBFC-Phys the speech task (T2) gave 15.3 bpm error while
    the arithmetic task (T3) gave 0.3 bpm on the SAME subject. Motion is the
    dominant error source, and the branch must know when to abstain.
    """
    from scipy import signal as sps
    if len(sig) < int(fs * 4):
        return 0.0, None
    nper = min(len(sig), int(fs * 8))
    f, p = sps.welch(sig, fs=fs, nperseg=nper)
    band = (f >= HR_MIN_HZ) & (f <= HR_MAX_HZ)
    if not band.any() or p[band].sum() <= 0:
        return 0.0, None
    pb, fb = p[band], f[band]
    peak = int(np.argmax(pb))
    # Power within +/-0.2 Hz of the peak, relative to all in-band power.
    near = np.abs(fb - fb[peak]) <= 0.2
    return float(pb[near].sum() / pb.sum()), float(fb[peak] * 60.0)


class LiveRPPG:
    """
    Rolling-window rPPG estimator.

    Contract mirrors GRUTemporalInference and LearnedFusion: when it cannot
    produce a trustworthy number it says so (None) rather than guessing, and
    the fusion redistributes its weight.
    """

    def __init__(self, roi='cheek'):
        self.roi = roi
        self._buf = deque()                 # (t, r, g, b)
        self._resting_hr = None
        self._resting_samples = []
        self._last_hr = None
        self._last_quality = 0.0
        self._frames_since_estimate = 0
        self._last_update_wall = None       # time.time() of the last real
                                            # buffer append; None = never

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self._buf.clear()
        self._resting_hr = None
        self._resting_samples = []
        self._last_hr = None
        self._last_quality = 0.0
        self._frames_since_estimate = 0
        self._last_update_wall = None

    def is_available(self):
        """True once a trustworthy HR has been produced at least once."""
        return self._last_hr is not None

    def has_resting_baseline(self):
        return self._resting_hr is not None

    # -- ingestion ---------------------------------------------------------
    def add_frame(self, t, frame_bgr, mesh_result):
        """Feed one frame. Cheap: a couple of patch means."""
        rgbs = extract_face_rgb(frame_bgr, mesh_result)
        if rgbs is None:
            return
        r, g, b = rgbs[self.roi]
        self._buf.append((t, r, g, b))
        self._last_update_wall = time.time()
        self._frames_since_estimate += 1
        cutoff = t - config.RPPG_WINDOW_SECONDS
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    # -- estimation --------------------------------------------------------
    def _effective_fps(self):
        if len(self._buf) < 2:
            return 0.0
        span = self._buf[-1][0] - self._buf[0][0]
        return (len(self._buf) - 1) / span if span > 0 else 0.0

    def _is_stale(self):
        """
        True once too much real (wall-clock) time has passed since the last
        frame that actually updated the buffer. Protects against a caller
        that keeps invoking add_frame()/estimate()/status() through a
        face-loss gap with no fresh data - without this, the buffer's old
        contents still look like a perfectly clean pulse and would keep
        producing a frozen, stale reading forever. self._last_update_wall
        is None (check skipped) whenever the buffer was populated some
        other way than a real add_frame() call, e.g. in tests.
        """
        if self._last_update_wall is None:
            return False
        return (time.time() - self._last_update_wall) > RPPG_STALE_AFTER_SECONDS

    def estimate(self, force=False):
        """
        Returns (hr_bpm, quality) or (None, quality). Rate-limited: the FFT
        is far too costly to run on every frame.
        """
        if self._is_stale():
            self._last_hr = None
            self._last_quality = 0.0
            return None, 0.0
        if not force and self._frames_since_estimate < config.RPPG_ESTIMATE_EVERY_N_FRAMES:
            return self._last_hr, self._last_quality
        self._frames_since_estimate = 0

        fps = self._effective_fps()
        span = (self._buf[-1][0] - self._buf[0][0]) if len(self._buf) > 1 else 0.0
        if fps < config.RPPG_MIN_FPS or span < config.RPPG_MIN_SECONDS:
            return None, 0.0

        rgb = np.array([[r, g, b] for _, r, g, b in self._buf])
        try:
            sig = method_pos(rgb, fps)
        except Exception:
            return None, 0.0

        quality, hr = spectral_quality(sig, fps)
        self._last_quality = quality
        if hr is None or quality < config.RPPG_MIN_QUALITY:
            # Abstain rather than emit a motion artefact.
            return None, quality

        self._last_hr = hr
        # The first stretch of usable readings defines the resting rate.
        if self._resting_hr is None:
            self._resting_samples.append(hr)
            if len(self._resting_samples) >= config.RPPG_RESTING_SAMPLES:
                self._resting_hr = float(np.median(self._resting_samples))
        return hr, quality

    # -- scoring -----------------------------------------------------------
    def stress_score(self):
        """
        Elevation above the personal resting rate, mapped to [0, 1].

        The scale comes from WESAD rather than being invented: baseline
        72.41 bpm vs stress 97.49 bpm, so ~25 bpm of elevation corresponds to
        a full-scale physiological stress response.

        None until a resting baseline exists and the current reading is
        trustworthy - the branch abstains rather than guesses.
        """
        hr, quality = self.estimate()
        if hr is None or self._resting_hr is None:
            return None
        if quality < config.RPPG_MIN_QUALITY:
            return None
        delta = hr - self._resting_hr
        return float(np.clip(delta / config.RPPG_STRESS_SCALE_BPM, 0.0, 1.0))

    def status(self):
        """Human-readable state, for the dashboard."""
        if not self._buf:
            return "no face"
        if self._is_stale():
            return "no recent signal - stale"
        fps = self._effective_fps()
        span = self._buf[-1][0] - self._buf[0][0]
        if fps < config.RPPG_MIN_FPS:
            return f"low frame rate ({fps:.0f} fps)"
        if span < config.RPPG_MIN_SECONDS:
            return f"warming up ({span:.0f}/{config.RPPG_MIN_SECONDS:.0f}s)"
        if self._last_hr is None:
            return "signal too noisy - hold still"
        if self._resting_hr is None:
            n = len(self._resting_samples)
            return f"{self._last_hr:.0f} bpm (resting baseline {n}/{config.RPPG_RESTING_SAMPLES})"
        return f"{self._last_hr:.0f} bpm (rest {self._resting_hr:.0f})"


_INSTANCE = None


def get_live_rppg():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = LiveRPPG(roi=config.RPPG_ROI)
    return _INSTANCE


def reset_live_rppg():
    global _INSTANCE
    _INSTANCE = None
