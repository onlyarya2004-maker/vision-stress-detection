"""
One per-frame pipeline, shared by every capture path
====================================================
Two ways a frame can reach the server:
  - local:   cv2.VideoCapture on the machine running Streamlit (development,
             and the demo-day fallback)
  - browser: the browser_camera component (hosted deployment) - the browser
             shows the camera locally at full quality and sends compressed
             640x480 snapshots here for processing

Both call the SAME classes in this module, so the browser path runs the exact
per-frame math the local path was validated with - parity by construction,
not by keeping two copies in sync. Callers pass frames that are already
mirrored (the local path cv2.flip()s; the browser mirrors before encoding)
plus a capture timestamp in seconds. Nothing in here draws on frames or
touches Streamlit.

Why browser-side landmark detection was rejected: running MediaPipe in the
browser means the Tasks-API models, not the legacy Python solutions this
system was validated on. Measured on real UBFC-Phys s11 footage
(experiment_mediapipe_tasks_parity.py): jaw_tension r=0.18 and brow_tension
r=0.69 between the two - the two heaviest-weighted signals - and the weighted
behaviour score only r=0.49. So the browser sends pixels, and the validated
legacy models run here. (The same experiment's --mode jpeg shows the JPEG hop
itself is harmless: behaviour score r=0.955, identical mean.)

Time: every run maps the caller's capture clock onto ONE server-side clock
domain (wall-clock seconds), keeping the exact spacing between frames. The
feature engine's blink/PERCLOS windows and the GRU's pacing gate all read
that clock, so (a) frames arriving in half-second batches are not squashed
into one instant, and (b) the engine never sees time jump between a
construction-time anchor (wall-clock) and a browser-relative timestamp.

Also owns everything that must be per concurrent user. The old app kept the
feature engine, GRU buffer and rPPG buffer as process-wide singletons
(@st.cache_resource / module globals) and the DaiSEE baseline as a module
global - harmless for one local user, but on a hosted server two people
calibrating at once would overwrite each other's baselines. UserState holds
one private copy of each; app.py keeps one UserState per Streamlit session.
"""
import base64
import binascii
import dataclasses
import math
import os
import time
import weakref
from datetime import datetime, timedelta

import cv2
import numpy as np

import config
import daisee_ensemble
from behavioral_features import BehavioralBaseline, BehavioralFeatureEngine
from frame_ops import crop_face, decode_jpeg, prepare_processing_frame
from rppg_live import LiveRPPG
from stress_fusion import (
    compute_stress_index, TemporalSmoother, HysteresisLevel, new_daisee_baseline,
)
from temporal_model import GRUTemporalInference


def _close_solutions(*solutions):
    for s in solutions:
        try:
            s.close()
        except Exception:
            pass


class UserState:
    """Everything one concurrent user needs to keep private."""

    _live = weakref.WeakSet()       # every UserState still alive (capacity guard)

    def __init__(self):
        # Every engine time lookup reads this; each frame sets it to that
        # frame's mapped time before processing (see module docstring).
        self.clock = {'t': time.time()}
        clock = self.clock          # closure over the dict, NOT over self: a
                                    # self-reference would make a cycle that
                                    # only the cyclic GC frees, so a finished
                                    # session would keep counting as live
        self.engine = BehavioralFeatureEngine(time_source=lambda: clock['t'])
        self.daisee_baseline = new_daisee_baseline()
        self.temporal_inf = GRUTemporalInference()
        self.rppg = LiveRPPG(roi=config.RPPG_ROI)
        self._last_pose = None
        self._pose_age = 0
        # Release the two MediaPipe graphs (native memory) when this session's
        # state is garbage-collected - a hosted server outlives every login.
        self._finalizer = weakref.finalize(
            self, _close_solutions, self.engine.face_mesh, self.engine.pose)
        UserState._live.add(self)

    @classmethod
    def active_count(cls):
        """How many sessions currently hold an engine (each one is two
        MediaPipe graphs + a GRU, ~150-250 MB and a share of the CPU)."""
        return len(cls._live)

    def close(self):
        self._finalizer()

    @property
    def fully_calibrated(self):
        return self.engine.baseline.calibrated and self.daisee_baseline.is_calibrated()

    def apply_profile(self, saved):
        """Load a saved profile baseline (the store's load_user_baseline
        output), or generic defaults when saved is None. Clears rolling
        windows first so one user's samples never score against another's
        baseline (BUG_AUDIT_2026-08-23.md finding #5)."""
        self.engine.reset_state()
        if saved is not None:
            self.engine.baseline = BehavioralBaseline(**saved['behavioral'])
            if saved.get('daisee_baseline') is not None:
                self.daisee_baseline.set(saved['daisee_baseline'])
            else:
                self.daisee_baseline.reset()
        else:
            self.engine.reset_baseline()
            self.daisee_baseline.reset()

    def reset_baselines(self):
        self.engine.reset_baseline()
        self.daisee_baseline.reset()

    def baseline_payload(self):
        """What the store's save_user_baseline() persists."""
        return {'behavioral': dataclasses.asdict(self.engine.baseline),
                'daisee_baseline': self.daisee_baseline.value}

    # -- shared per-frame helpers ------------------------------------------
    def reset_pose_cache(self):
        self._last_pose = None
        self._pose_age = 0

    def pose_results_for(self, rgb):
        """None when Pose should simply run inside extract() as it always
        has (POSE_EVERY_N_FRAMES <= 1: bit-for-bit the validated behaviour);
        otherwise the cached result, refreshed every Nth frame."""
        n = config.POSE_EVERY_N_FRAMES
        if n <= 1:
            return None
        if self._last_pose is None or self._pose_age % n == 0:
            self._last_pose = self.engine.pose.process(rgb)
        self._pose_age += 1
        return self._last_pose


class _Run:
    """Clock mapping shared by CalibrationRun and LiveRun."""

    def __init__(self, state, models):
        self.state = state
        self.models = models
        self.started = False
        self.client0 = None      # first frame's caller-side timestamp
        self.t0 = None           # the same instant in the server clock domain
        self.t = None            # latest frame's server-domain time

    def _map(self, client_t):
        return self.t0 + (client_t - self.client0)

    def _start_clock(self, client_t, wall0=None):
        self.client0 = client_t
        self.t0 = time.time() if wall0 is None else wall0
        self.t = self.t0
        self.state.clock['t'] = self.t0

    def elapsed(self):
        return 0.0 if self.t is None else self.t - self.t0


class CalibrationRun(_Run):
    """One calibration attempt. Mirrors the validated calibration loop's
    per-frame body and its finish step exactly."""

    def __init__(self, state, models):
        super().__init__(state, models)
        self.daisee_samples = []

    def process(self, frame_bgr, client_t):
        """Returns {'no_face': bool, 'bbox': (x1,y1,x2,y2) or None,
        'geom': crop rectangle of the source frame}. bbox is in the 640x480
        processing frame (frame_ops.bbox_to_source maps it back)."""
        s = self.state
        if not self.started:
            self._start_clock(client_t)
            s.reset_pose_cache()
            s.engine.start_calibration()
            self.started = True
        self.t = self._map(client_t)
        s.clock['t'] = self.t

        proc, geom = prepare_processing_frame(frame_bgr)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        # One FaceMesh pass per frame, shared by extract() and crop_face().
        face_mesh_results = s.engine.face_mesh.process(rgb)
        s.engine.extract(rgb, face_mesh_results=face_mesh_results,
                         pose_results=s.pose_results_for(rgb))
        face_crop, bbox = crop_face(proc, face_mesh_results)
        if face_crop is not None:
            if (not self.daisee_samples or
                    int((self.t - self.t0) * 30) % config.CNN_EVERY_N_FRAMES == 0):
                probs = daisee_ensemble.ensemble_predict(face_crop, self.models['pairs'])
                if probs is not None:
                    self.daisee_samples.append(float(probs[1]))
        return {'no_face': face_crop is None, 'bbox': bbox, 'geom': geom}

    def progress(self):
        return min(1.0, self.elapsed() / config.CALIBRATION_SECONDS)

    def seconds_left(self):
        return max(0.0, config.CALIBRATION_SECONDS - self.elapsed())

    @property
    def done(self):
        return self.started and self.elapsed() >= config.CALIBRATION_SECONDS

    def finish(self):
        """(face_ok, daisee_ok) - same acceptance rules as before."""
        s = self.state
        if self.t is not None:
            s.clock['t'] = self.t
        face_ok = s.engine.finish_calibration()
        daisee_ok = False
        if len(self.daisee_samples) >= config.MIN_DAISEE_CALIB_SAMPLES:
            s.daisee_baseline.set(float(np.median(self.daisee_samples)))
            daisee_ok = True
        return face_ok, daisee_ok

    def abort(self):
        self.state.engine._calibrating = False


class LiveRun(_Run):
    """One live session (Start to Stop). process() is the validated capture
    loop's per-frame body, step for step and in the same order."""

    HISTORY_INTERVAL_S = 0.5

    def __init__(self, state, models, agent, *, task_mode, agent_enabled, dt0=None):
        super().__init__(state, models)
        self.agent = agent
        self.task_mode = task_mode
        self.agent_enabled = agent_enabled
        # Wall-clock datetime that run-time zero maps onto, for history 'ts'.
        # Local mode: the machine's clock. Browser mode: the app passes the
        # USER's local time (server clocks are UTC), so time-of-day analytics
        # reflect when the person actually recorded.
        self.dt0 = dt0 if dt0 is not None else datetime.now()
        self.smoother = TemporalSmoother(window_size=config.SMOOTHING_WINDOW_FRAMES)
        self.hyst = HysteresisLevel(confirm_frames=config.HYSTERESIS_CONFIRM_FRAMES)
        self.last_daisee_probs = None
        self.frame_idx = 0
        self.last_record_t = None
        self.latest = None          # last successful frame's result, for rendering
        self.last_smoothed = 0.0
        # STRESS_PROFILE=1: accumulate per-stage wall time (ms) so a slow
        # deployment can be diagnosed stage by stage. Off = no overhead.
        self.stage_ms = {} if os.environ.get('STRESS_PROFILE') else None

    def _lap(self, name, t_prev):
        """Record the time since t_prev under `name`; returns the new mark."""
        now = time.perf_counter()
        if self.stage_ms is not None:
            self.stage_ms[name] = self.stage_ms.get(name, 0.0) + (now - t_prev) * 1000.0
        return now

    def _begin(self, client_t):
        s = self.state
        self._start_clock(client_t)
        if s.temporal_inf.is_available():
            s.temporal_inf.reset()
        s.rppg.reset()
        s.reset_pose_cache()
        self.agent.start_session(self.t0)
        self.started = True

    def process(self, frame_bgr, client_t):
        """
        Returns {'no_face': True, 'geom': ...} when no face is usable,
        otherwise a dict with bbox, geom, level, smoothed, result,
        behavioral_features, intervention, hr_bpm, rppg_status and record (a
        history row, or None when throttled). Also stored as self.latest.
        """
        s = self.state
        if not self.started:
            self._begin(client_t)
        t = self._map(client_t)
        self.t = t
        s.clock['t'] = t

        mark = time.perf_counter()
        proc, geom = prepare_processing_frame(frame_bgr)
        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        mark = self._lap('prep', mark)

        face_mesh_results = s.engine.face_mesh.process(rgb)
        mark = self._lap('face_mesh', mark)
        face_crop, bbox = crop_face(proc, face_mesh_results)
        if face_crop is None:
            self.frame_idx += 1
            return {'no_face': True, 'geom': geom}

        if self.frame_idx % config.CNN_EVERY_N_FRAMES == 0:
            self.last_daisee_probs = daisee_ensemble.ensemble_predict(
                face_crop, self.models['pairs'])
            if self.stage_ms is not None:
                for k in ('wait', 'infer'):
                    self.stage_ms['cnn_' + k + '(per call)'] = (
                        self.stage_ms.get('cnn_' + k + '(per call)', 0.0)
                        + daisee_ensemble.LAST_CALL_MS[k])
                for size, prep_ms, call_ms, numpy_ms in daisee_ensemble.LAST_CALL_MS['detail']:
                    for what, v in (('prep', prep_ms), ('call', call_ms), ('numpy', numpy_ms)):
                        key = f'cnn{size}_{what}(per call)'
                        self.stage_ms[key] = self.stage_ms.get(key, 0.0) + v
                self.stage_ms['cnn_calls'] = self.stage_ms.get('cnn_calls', 0) + 1
        mark = self._lap('cnn', mark)

        # rPPG: computed and shown even at W_RPPG = 0 so it can be
        # sanity-checked against a wrist pulse before being trusted.
        s.rppg.add_frame(t, proc, face_mesh_results)
        rppg_score = s.rppg.stress_score()
        mark = self._lap('rppg', mark)

        behavioral_features = s.engine.extract(
            rgb, face_mesh_results=face_mesh_results,
            pose_results=s.pose_results_for(rgb))
        mark = self._lap('extract(pose)', mark)
        if behavioral_features is None:
            self.frame_idx += 1
            return {'no_face': True, 'geom': geom}

        temporal_score = None
        if s.temporal_inf.is_available():
            s.temporal_inf.add_frame(behavioral_features, timestamp=t)
            temporal_score = s.temporal_inf.predict()
        mark = self._lap('gru', mark)

        result = compute_stress_index(
            daisee_probs=self.last_daisee_probs,
            behavioral_features=behavioral_features,
            temporal_score=temporal_score,
            task_mode=self.task_mode,
            rppg_score=rppg_score,
            daisee_baseline=s.daisee_baseline,
        )
        smoothed = self.smoother.update(result['stress_index'])
        level = self.hyst.update(smoothed)
        self.last_smoothed = smoothed

        intervention = None
        if self.agent_enabled:
            _iv = self.agent.update(t, smoothed, result['contributions'])
            if _iv is not None and (t - _iv.t_shown) <= config.INTERVENTION_SHOW_SECONDS:
                intervention = _iv

        hr_bpm, _ = s.rppg.estimate()
        self._lap('fusion+agent+hr', mark)

        record = None
        if self.last_record_t is None or (t - self.last_record_t) >= self.HISTORY_INTERVAL_S:
            # daisee stays None (not 0.0) on frames the CNN didn't score -
            # see the session_history v7->v8 migration (2026-09-17).
            record = {
                'ts': self.dt0 + timedelta(seconds=t - self.t0),
                'stress_index': smoothed, 'level': level,
                'daisee': result.get('daisee_raw'),
                'behavior': result['behavior_stress'],
            }
            # Per-signal values: the self-retraining loop learns from these.
            record.update(result['contributions'])
            # The 3 raw values the GRU needs beyond the 7 contributions
            # (session_history schema v7).
            record['ear'] = behavioral_features.get('ear', 0.0)
            record['ear_asymmetry'] = behavioral_features.get('ear_asymmetry', 0.0)
            record['blinks_per_minute'] = behavioral_features.get('blinks_per_minute', 0.0)
            self.last_record_t = t

        self.frame_idx += 1
        out = {
            'no_face': False, 'bbox': bbox, 'geom': geom, 'level': level,
            'smoothed': smoothed, 'result': result,
            'behavioral_features': behavioral_features,
            'intervention': intervention, 'hr_bpm': hr_bpm,
            'rppg_status': s.rppg.status(), 'record': record,
        }
        self.latest = out
        return out

    def agent_status(self):
        return self.agent.status(self.t) if (self.agent_enabled and self.started) else None

    def finish(self, final_index=None):
        """The agent's measured outcomes, for the store's log_interventions."""
        if not self.started:
            return []
        idx = self.last_smoothed if final_index is None else final_index
        return self.agent.finish_session(self.t, idx)


# ---------------------------------------------------------------------------
# Browser batches
# ---------------------------------------------------------------------------
class BatchGate:
    """
    Validates the browser component's batches before anything is decoded or
    processed. The component re-delivers its LAST value on every rerun that
    it did not itself cause, and a batch can arrive after the user has
    already pressed Stop - so this tracks the epoch (bumped on every
    Calibrate/Start/Stop) and the last sequence number, and drops
    duplicates, stale epochs, out-of-order frames and oversized batches.
    Everything in a batch is untrusted input.
    """

    def __init__(self):
        self.epoch = None
        self.last_seq = 0
        self.last_t = -math.inf
        self.dropped = 0
        self.tz_offset_min = 0

    def reset(self, epoch):
        self.epoch = epoch
        self.last_seq = 0
        self.last_t = -math.inf

    def accept(self, value, epoch):
        """
        value: the component's returned dict (or None).
        Returns (frames, ack_seq): frames = [(client_t, bgr_frame), ...]
        oldest-first, at most MAX_FRAMES_PROCESSED_PER_BATCH newest ones;
        ack_seq = the sequence number to acknowledge back to the browser.
        """
        if self.epoch != epoch:
            self.reset(epoch)
        if not isinstance(value, dict):
            return [], self.last_seq
        try:
            if int(value.get('epoch', -1)) != int(epoch):
                return [], self.last_seq
            seq = int(value.get('seq', 0))
        except (TypeError, ValueError):
            return [], self.last_seq
        if seq <= self.last_seq:
            return [], self.last_seq               # duplicate redelivery
        self.last_seq = seq                         # ack even if nothing usable

        tz = value.get('tz')
        if isinstance(tz, (int, float)) and math.isfinite(tz) and -14 * 60 <= tz <= 14 * 60:
            self.tz_offset_min = int(tz)

        raw = value.get('frames')
        if not isinstance(raw, list):
            return [], self.last_seq
        raw = raw[-config.MAX_FRAMES_PER_BATCH:]

        frames = []
        for item in raw:
            if not isinstance(item, dict):
                self.dropped += 1
                continue
            t, j = item.get('t'), item.get('j')
            if (not isinstance(t, (int, float)) or isinstance(t, bool)
                    or not math.isfinite(t) or t <= self.last_t
                    or not isinstance(j, str)
                    or len(j) > config.MAX_FRAME_BYTES * 4 // 3 + 8):
                self.dropped += 1
                continue
            try:
                data = base64.b64decode(j, validate=True)
            except (binascii.Error, ValueError):
                self.dropped += 1
                continue
            frame = decode_jpeg(data)
            if frame is None:
                self.dropped += 1
                continue
            self.last_t = float(t)
            frames.append((float(t), frame))
        # Behind? Keep only the newest - never queue up latency.
        if len(frames) > config.MAX_FRAMES_PROCESSED_PER_BATCH:
            self.dropped += len(frames) - config.MAX_FRAMES_PROCESSED_PER_BATCH
            frames = frames[-config.MAX_FRAMES_PROCESSED_PER_BATCH:]
        return frames, self.last_seq

    def user_now(self):
        """The user's local wall-clock time, from the browser's UTC offset
        (the server clock is UTC)."""
        return datetime.utcnow() + timedelta(minutes=self.tz_offset_min)
