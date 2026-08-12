"""
Behavioral Feature Engine (v5 - pose noise reduction)
=====================================================
Adds two fixes after observing that hand-near-face was inflating the
stress signal during testing:

  1. Pose features now require the visibility score from MediaPipe Pose
     for both shoulders to exceed 0.7. When shoulders aren't clearly
     visible (hand-near-face occlusion, partial frame, side view),
     posture features default to 0.0 instead of producing garbage values.

  2. New `_hand_near_face_factor()` damping: if the LEFT/RIGHT WRIST
     landmarks are detected near the face bounding box, neck_strain
     and slouch contributions are scaled down. A hand near the face
     is usually a yawn / chin-rest / glasses adjustment, not a
     stress posture.

The face features (EAR, brow, jaw, head tilt) are unchanged - they
were already working well. This file only touches the posture branch.
"""

from collections import deque
from dataclasses import dataclass, field
from time import time
import math
from statistics import mean

import numpy as np
import mediapipe as mp

import config

# ----- FaceMesh landmark indices -----
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_INNER_BROW  = 55
RIGHT_INNER_BROW = 285
UPPER_LIP = 13
LOWER_LIP = 14
LIP_LEFT  = 61
LIP_RIGHT = 291
FACE_LEFT  = 234
FACE_RIGHT = 454
NOSE_TIP = 1
CHIN = 152

# ----- Pose landmark indices -----
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
NOSE_POSE      = 0
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14

POSE_VISIBILITY_THRESHOLD = config.POSE_VISIBILITY_THRESHOLD  # below this we ignore the pose


def _p(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h])


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(eye_pts):
    A = _dist(eye_pts[1], eye_pts[5])
    B = _dist(eye_pts[2], eye_pts[4])
    C = _dist(eye_pts[0], eye_pts[3])
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


@dataclass
class BehavioralBaseline:
    ear_baseline:        float = config.DEFAULT_EAR_BASELINE
    brow_distance_norm:  float = config.DEFAULT_BROW_DISTANCE_NORM
    lip_gap_norm:        float = config.DEFAULT_LIP_GAP_NORM
    lip_width_norm:      float = config.DEFAULT_LIP_WIDTH_NORM
    head_tilt_baseline:  float = config.DEFAULT_HEAD_TILT_BASELINE
    blink_rate_baseline: float = config.DEFAULT_BLINK_RATE_BASELINE
    calibrated: bool = False


@dataclass
class BehavioralState:
    blink_times:        deque = field(default_factory=lambda: deque(maxlen=300))
    eye_closed:         bool = False
    last_ear:           float = 0.3
    ear_history:        deque = field(
        default_factory=lambda: deque(maxlen=config.EAR_HISTORY_FRAMES))
    head_tilt_history:  deque = field(
        default_factory=lambda: deque(maxlen=config.HEAD_TILT_HISTORY_FRAMES))


class BehavioralFeatureEngine:
    EAR_BLINK_THRESHOLD     = config.EAR_BLINK_THRESHOLD
    BLINK_WINDOW_SECONDS    = config.BLINK_WINDOW_SECONDS

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.state = BehavioralState()
        self.baseline = BehavioralBaseline()
        self._calibrating = False
        self._calib_buffer = []

    def close(self):
        self.face_mesh.close()
        self.pose.close()

    def start_calibration(self):
        self._calibrating = True
        self._calib_buffer = []

    def is_calibrating(self):
        return self._calibrating

    def calibration_progress(self, target_samples=120):
        return min(1.0, len(self._calib_buffer) / target_samples)

    def finish_calibration(self):
        self._calibrating = False
        if len(self._calib_buffer) < config.MIN_CALIBRATION_SAMPLES:
            return False
        ears        = [s['ear'] for s in self._calib_buffer]
        brows       = [s['brow_distance_norm'] for s in self._calib_buffer]
        lip_gaps    = [s['lip_gap_norm'] for s in self._calib_buffer]
        lip_widths  = [s['lip_width_norm'] for s in self._calib_buffer]
        head_tilts  = [s['head_tilt_deg'] for s in self._calib_buffer]
        self.baseline.ear_baseline       = float(mean(ears))
        self.baseline.brow_distance_norm = float(mean(brows))
        self.baseline.lip_gap_norm       = float(mean(lip_gaps))
        self.baseline.lip_width_norm     = float(mean(lip_widths))
        self.baseline.head_tilt_baseline = float(mean(head_tilts))
        self.baseline.calibrated = True
        self.state.blink_times.clear()
        self.state.ear_history.clear()
        self.state.head_tilt_history.clear()
        return True

    def reset_baseline(self):
        self.baseline = BehavioralBaseline()

    def _extract_face(self, rgb_frame):
        h, w = rgb_frame.shape[:2]
        result = self.face_mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None, None
        lm = result.multi_face_landmarks[0].landmark

        face_left  = _p(lm, FACE_LEFT,  w, h)
        face_right = _p(lm, FACE_RIGHT, w, h)
        face_w = _dist(face_left, face_right) + 1e-6

        # face bbox (used later for hand-near-face check)
        xs = np.array([p.x for p in lm]) * w
        ys = np.array([p.y for p in lm]) * h
        face_bbox = (float(xs.min()), float(ys.min()),
                     float(xs.max()), float(ys.max()))

        left_eye  = [_p(lm, i, w, h) for i in LEFT_EYE]
        right_eye = [_p(lm, i, w, h) for i in RIGHT_EYE]
        left_ear  = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        ear_asymmetry = abs(left_ear - right_ear)

        now = time()
        if ear < self.EAR_BLINK_THRESHOLD and not self.state.eye_closed:
            self.state.eye_closed = True
            self.state.blink_times.append(now)
        elif ear >= self.EAR_BLINK_THRESHOLD:
            self.state.eye_closed = False

        cutoff = now - self.BLINK_WINDOW_SECONDS
        while self.state.blink_times and self.state.blink_times[0] < cutoff:
            self.state.blink_times.popleft()
        window_elapsed = max(
            1.0,
            min(self.BLINK_WINDOW_SECONDS,
                now - (self.state.blink_times[0] if self.state.blink_times else now - 1)),
        )
        blinks_per_minute = len(self.state.blink_times) * (60.0 / window_elapsed)

        self.state.ear_history.append(ear)

        lb = _p(lm, LEFT_INNER_BROW, w, h)
        rb = _p(lm, RIGHT_INNER_BROW, w, h)
        brow_distance_norm = _dist(lb, rb) / face_w

        up = _p(lm, UPPER_LIP, w, h)
        lo = _p(lm, LOWER_LIP, w, h)
        ll = _p(lm, LIP_LEFT,  w, h)
        lr = _p(lm, LIP_RIGHT, w, h)
        lip_gap_norm   = _dist(up, lo) / face_w
        lip_width_norm = _dist(ll, lr) / face_w

        left_eye_center  = np.mean(left_eye,  axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        dx = right_eye_center[0] - left_eye_center[0]
        dy = right_eye_center[1] - left_eye_center[1]
        head_tilt_deg = abs(math.degrees(math.atan2(dy, dx)))
        self.state.head_tilt_history.append(head_tilt_deg)
        head_tilt_instability = float(np.std(self.state.head_tilt_history))

        if self._calibrating:
            self._calib_buffer.append({
                'ear':                ear,
                'brow_distance_norm': brow_distance_norm,
                'lip_gap_norm':       lip_gap_norm,
                'lip_width_norm':     lip_width_norm,
                'head_tilt_deg':      head_tilt_deg,
            })
            return ({
                'ear': ear, 'ear_asymmetry': ear_asymmetry,
                'blinks_per_minute': blinks_per_minute,
                'eye_strain': 0.0, 'blink_stress': 0.0,
                'brow_tension': 0.0, 'jaw_tension': 0.0,
                'head_instability': 0.0, 'calibrating': True,
            }, face_bbox)

        b = self.baseline
        ear_mean = float(np.mean(self.state.ear_history)) if self.state.ear_history else ear
        eye_strain = float(np.clip(
            abs(ear_mean - b.ear_baseline) / config.EYE_STRAIN_SCALE, 0.0, 1.0))

        blink_excess = blinks_per_minute - b.blink_rate_baseline
        blink_stress = float(np.clip(
            blink_excess / config.BLINK_EXCESS_SCALE, 0.0, 1.0))

        brow_drop = (b.brow_distance_norm - brow_distance_norm) / max(b.brow_distance_norm, 1e-6)
        brow_tension = float(np.clip(
            brow_drop / config.BROW_DROP_SCALE, 0.0, 1.0))

        # Jaw tension fix: a smile makes lips WIDER, which the old formula
        # confused with a grimace/clench. Now we treat narrower lips
        # (compression/clenching) as positive tension, but wider lips only
        # contribute when they're combined with reduced lip-gap (a grit-teeth
        # grimace looks like wide-thin lips simultaneously). A normal smile
        # has wide lips + larger lip-gap, which now correctly reads as low
        # tension.
        gap_change   = (b.lip_gap_norm   - lip_gap_norm)   / max(b.lip_gap_norm,   1e-6)
        width_change = (lip_width_norm   - b.lip_width_norm) / max(b.lip_width_norm, 1e-6)
        # Smile heuristic: if lips are wider AND lip_gap is roughly normal-or-bigger,
        # this is a smile - skip jaw tension entirely.
        is_smile = ((width_change > config.SMILE_WIDTH_INCREASE)
                    and (gap_change < config.SMILE_MAX_GAP_REDUCTION))
        if is_smile:
            jaw_tension = 0.0
        else:
            # Compression: lips narrower than baseline = clenching
            compression = max(0.0, -width_change)
            # Reduced gap: lips pressed together
            press = max(0.0, gap_change)
            jaw_tension = float(np.clip(
                (compression + press) / config.JAW_TENSION_SCALE, 0.0, 1.0))

        head_instability = float(np.clip(
            head_tilt_instability / config.HEAD_INSTABILITY_SCALE, 0.0, 1.0))

        return ({
            'ear': ear, 'ear_asymmetry': ear_asymmetry,
            'blinks_per_minute': blinks_per_minute,
            'eye_strain': eye_strain, 'blink_stress': blink_stress,
            'brow_tension': brow_tension, 'jaw_tension': jaw_tension,
            'head_instability': head_instability, 'calibrating': False,
        }, face_bbox)

    def _hand_near_face(self, lm, w, h, face_bbox):
        """Return True if either wrist landmark falls inside (or near) the
        face bounding box. This is the typical 'hand on chin', 'covering
        yawn', or 'adjusting glasses' pose - we want to damp pose-derived
        stress signals here."""
        if face_bbox is None:
            return False
        x1, y1, x2, y2 = face_bbox
        # Expand bbox slightly so a hand near the face still triggers
        pad = config.HAND_NEAR_FACE_PADDING * max(x2 - x1, y2 - y1)
        x1 -= pad; y1 -= pad; x2 += pad; y2 += pad

        for wrist_idx in (LEFT_WRIST, RIGHT_WRIST):
            wrist_lm = lm[wrist_idx]
            if wrist_lm.visibility < config.WRIST_VISIBILITY_THRESHOLD:
                continue
            wx, wy = wrist_lm.x * w, wrist_lm.y * h
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                return True
        return False

    def _extract_posture(self, rgb_frame, face_bbox):
        result = self.pose.process(rgb_frame)
        if not result.pose_landmarks:
            return {'neck_strain': 0.0, 'slouch': 0.0, 'posture_detected': False}

        lm = result.pose_landmarks.landmark
        h, w = rgb_frame.shape[:2]

        # Fix 1: require both shoulders clearly visible
        ls_vis = lm[LEFT_SHOULDER].visibility
        rs_vis = lm[RIGHT_SHOULDER].visibility
        if ls_vis < POSE_VISIBILITY_THRESHOLD or rs_vis < POSE_VISIBILITY_THRESHOLD:
            return {
                'neck_strain': 0.0, 'slouch': 0.0,
                'posture_detected': False,
                'pose_visibility_low': True,
            }

        ls   = _p(lm, LEFT_SHOULDER,  w, h)
        rs   = _p(lm, RIGHT_SHOULDER, w, h)
        nose = _p(lm, NOSE_POSE,      w, h)

        shoulder_mid   = (ls + rs) / 2.0
        shoulder_width = _dist(ls, rs) + 1e-6

        head_forward = abs(nose[0] - shoulder_mid[0]) / shoulder_width
        head_forward_stress = float(np.clip((head_forward - 0.1) / 0.25, 0.0, 1.0))

        vertical_gap = (shoulder_mid[1] - nose[1]) / shoulder_width
        slouch = float(np.clip((1.0 - vertical_gap) / 0.6, 0.0, 1.0))

        shoulder_tilt = abs(ls[1] - rs[1]) / shoulder_width
        shoulder_tilt_stress = float(np.clip(shoulder_tilt / 0.15, 0.0, 1.0))

        neck_strain = max(head_forward_stress, shoulder_tilt_stress)

        # Fix 2: damp pose stress when hand is near face
        if self._hand_near_face(lm, w, h, face_bbox):
            neck_strain *= config.HAND_NEAR_FACE_DAMPING
            slouch      *= config.HAND_NEAR_FACE_DAMPING

        return {
            'neck_strain': neck_strain,
            'slouch': slouch,
            'posture_detected': True,
            'pose_visibility_low': False,
        }

    def extract(self, rgb_frame):
        face_result = self._extract_face(rgb_frame)
        if face_result is None:
            return None
        face_feats, face_bbox = face_result
        if face_feats is None:
            return None
        posture_feats = self._extract_posture(rgb_frame, face_bbox)
        face_feats.update(posture_feats)
        return face_feats