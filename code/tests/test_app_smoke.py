"""
Smoke tests that run the REAL app.py script under Streamlit's AppTest harness
(no browser, no camera). The browser-camera component itself is JavaScript and
is exercised separately in a real browser (scripts/serve_test_frames.py), but
everything on the Python side of the app wiring - login gate, per-session
state, button handlers, Stop/save ordering - runs here.

Why this exists: app.py was restructured for hosting (per-user state, a store
abstraction, a fragment-driven camera). A typo in any of those paths would
only otherwise show up when a person clicks the right button in a browser.
"""
from datetime import datetime, timedelta

import pytest
from streamlit.testing.v1 import AppTest

import config
import daisee_ensemble
import session_history
from session_history import SIGNAL_COLUMNS

APP = str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'app.py')


@pytest.fixture(autouse=True)
def never_open_the_real_camera(monkeypatch):
    """Standing project rule: the webcam is only ever opened for a real user
    session. AppTest executes app.py as a separate module instance, so patching
    app.open_local_camera from here does NOT reach it - patch cv2 itself, which
    IS shared. Anything that reaches the real VideoCapture fails the test."""
    import cv2

    def _forbidden(*a, **k):
        raise AssertionError("a test tried to open the REAL webcam")
    monkeypatch.setattr(cv2, 'VideoCapture', _forbidden)


@pytest.fixture(autouse=True)
def roomy_capacity(monkeypatch):
    """Finished AppTest sessions linger until GC; don't let leftovers from
    earlier tests trip the simultaneous-session cap in later ones."""
    monkeypatch.setattr(config, 'MAX_USER_SESSIONS', 10_000)


@pytest.fixture
def no_cnn(monkeypatch):
    """Skip loading the 50 MB CNN weights - these tests are about wiring."""
    monkeypatch.setattr(daisee_ensemble, 'build_pairs', lambda skip=False: [])


def _run(timeout=90):
    at = AppTest.from_file(APP, default_timeout=timeout)
    at.run()
    return at


def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def _history(n=12, start=None):
    start = start or datetime(2026, 10, 10, 9, 0, 0)
    rows = []
    for i in range(n):
        r = {'ts': start + timedelta(seconds=0.5 * i), 'stress_index': 0.3,
             'level': 'LOW', 'daisee': 0.5, 'behavior': 0.25}
        for c in SIGNAL_COLUMNS:
            r[c] = 0.1
        rows.append(r)
    return rows


# ---- hosted mode: nothing renders until login ---------------------------------
def test_hosted_mode_without_server_credentials_fails_closed(monkeypatch, no_cnn):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'supabase')
    monkeypatch.delenv('SUPABASE_URL', raising=False)
    monkeypatch.delenv('SUPABASE_ANON_KEY', raising=False)
    at = _run()
    assert not at.exception
    assert any('not configured for login' in e.value for e in at.error)
    assert not at.sidebar.button                       # no app UI behind the gate
    assert 'user_state' not in at.session_state        # no models/engine built for strangers


def test_hosted_mode_shows_login_form_when_configured(monkeypatch, no_cnn):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'supabase')
    monkeypatch.setenv('SUPABASE_URL', 'https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_ANON_KEY', 'sb_publishable_test')
    at = _run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ['Sign in', 'Create account']
    assert not at.sidebar.button


def test_hosted_mode_refuses_a_service_role_key(monkeypatch, no_cnn):
    import base64, json

    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip('=')
    secret = f"{b64({'alg': 'HS256'})}.{b64({'role': 'service_role'})}.x"
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'supabase')
    monkeypatch.setenv('SUPABASE_URL', 'https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_ANON_KEY', secret)
    at = _run()
    assert any('SECRET' in e.value for e in at.error)
    assert not at.tabs


# ---- local mode ---------------------------------------------------------------
@pytest.mark.parametrize('capture_mode', ['browser', 'local'])
def test_local_mode_idle_screen_renders(monkeypatch, temp_db, no_cnn, capture_mode):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', capture_mode)
    at = _run()
    assert not at.exception
    labels = [b.label for b in at.sidebar.button]
    for needed in ('Calibrate', 'Start', 'Stop', 'Reset baseline'):
        assert needed in labels
    assert _button(at, 'Stop').disabled
    # Each browser session owns its own engine/baseline state.
    assert at.session_state['user_state'] is not None


def test_calibrate_click_starts_a_new_camera_epoch(monkeypatch, temp_db, no_cnn):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    at = _run()
    epoch0 = at.session_state['cam_epoch']
    _button(at, 'Calibrate').click().run()
    assert not at.exception
    assert at.session_state['calibrating'] is True
    assert at.session_state['cam_epoch'] == epoch0 + 1
    assert not _button(at, 'Stop').disabled
    # Stop abandons it cleanly and starts yet another epoch (late frames from
    # the abandoned run must be ignored).
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert at.session_state['calibrating'] is False
    assert at.session_state['cam_epoch'] == epoch0 + 2
    assert at.session_state['calib_run'] is None


def test_stop_saves_the_session_and_queues_the_self_report(monkeypatch, temp_db, no_cnn):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    session_history.init_db()
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state['history'] = _history()
    at.session_state['history_saved'] = False
    at.session_state['running'] = True
    at.run()
    assert not at.exception
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert at.session_state['running'] is False
    assert session_history.session_count() == 1
    assert at.session_state['pending_self_report'] == at.session_state['last_saved_session_id']
    # Pressing Stop again (a stale click) must not write a duplicate row.
    assert session_history.session_count() == 1


# ---- local (cv2) camera mode: the demo-day fallback, driven by a fake camera ----
class FakeCapture:
    """Stands in for cv2.VideoCapture: n blank 1080p frames, then a read
    failure (which is how a yanked/blocked camera shows up)."""

    def __init__(self, n, opened=True):
        import numpy as np
        self.left = n
        self.opened = opened
        self.frame = np.full((1080, 1920, 3), 90, dtype=np.uint8)
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, *a):
        return True

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, self.frame.copy()

    def release(self):
        self.released = True


@pytest.fixture
def fake_face(monkeypatch):
    """Every UserState gets a synthetic face instead of running FaceMesh/Pose."""
    import session_pipeline as sp
    from conftest import (face_mesh_result, make_face_landmarks,
                          make_pose_landmarks, pose_result)
    face = face_mesh_result(make_face_landmarks())
    pose = pose_result(make_pose_landmarks())
    orig = sp.UserState.__init__

    def patched(self):
        orig(self)
        self.engine.face_mesh.process = lambda rgb: face
        self.engine.pose.process = lambda rgb: pose
    monkeypatch.setattr(sp.UserState, '__init__', patched)


def test_local_camera_calibration_runs_to_completion(monkeypatch, temp_db, no_cnn, fake_face):
    import cv2
    cap = FakeCapture(10_000)
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'local')
    monkeypatch.setattr(config, 'CALIBRATION_SECONDS', 2.0)
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a, **k: cap)
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state['calibrating'] = True
    at.run()
    assert not at.exception
    assert at.session_state['calibrating'] is False         # finished and returned to idle
    assert cap.released                                       # camera handle freed
    ustate = at.session_state['user_state']
    assert ustate.engine.baseline.calibrated                  # face baseline learned
    assert not ustate.daisee_baseline.is_calibrated()         # CNN is stubbed out in this test


def test_local_camera_loss_mid_session_saves_the_data_in_the_right_order(
        monkeypatch, temp_db, no_cnn, fake_face):
    """A yanked camera must end the session cleanly AND keep what was recorded
    (the old code logged the agent's outcomes against the NEXT session)."""
    import cv2
    session_history.init_db()
    cap = FakeCapture(60)
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'local')
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a, **k: cap)
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state['running'] = True
    at.run()
    assert not at.exception
    assert at.session_state['running'] is False
    assert cap.released
    assert session_history.session_count() == 1               # partial session was saved
    assert at.session_state['live_run'] is None


def test_local_camera_that_will_not_open_reports_cleanly(monkeypatch, temp_db, no_cnn):
    import cv2
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'local')
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a, **k: FakeCapture(0, opened=False))
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state['calibrating'] = True
    at.run()
    assert any('Could not open webcam' in e.value for e in at.error)
    assert at.session_state['calibrating'] is False


def test_new_sessions_are_refused_at_capacity(monkeypatch, temp_db, no_cnn):
    from session_pipeline import UserState
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    monkeypatch.setattr(config, 'MAX_USER_SESSIONS', UserState.active_count())   # full
    at = _run()
    assert not at.exception
    assert any('at capacity' in w.value for w in at.warning)
    assert 'user_state' not in at.session_state           # no engine was built
    assert not at.sidebar.button or all(b.label != 'Calibrate' for b in at.sidebar.button)


def test_reset_baseline_only_touches_this_sessions_state(monkeypatch, temp_db, no_cnn):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    a, b = _run(), _run()
    a.session_state['user_state'].daisee_baseline.set(0.8)
    assert not b.session_state['user_state'].daisee_baseline.is_calibrated()
    _button(a, 'Reset baseline').click().run()
    assert not a.exception
    assert not a.session_state['user_state'].daisee_baseline.is_calibrated()
