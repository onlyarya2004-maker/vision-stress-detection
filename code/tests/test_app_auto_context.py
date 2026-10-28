"""
The "Auto-detect context" toggle through the REAL app.py (Streamlit AppTest,
no browser, no camera, no scene model - a fake classifier stands in).

Pinned down here: off by default and absent in hosted mode; with it off the
Calibrate flow is unchanged; with it on a calibration produces a suggestion
that changes NOTHING until the user accepts it; manual choices win; Start
locks the context and the session is saved with the mode the user chose.
"""
import pytest
from streamlit.testing.v1 import AppTest

import config
import context_detector as cd
import daisee_ensemble
import session_history
from context_detector import ContextProbe, ContextSuggestion
from test_context_detector import FakeClassifier, probs_for

APP = str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'app.py')


@pytest.fixture(autouse=True)
def guards(monkeypatch):
    """Never the real webcam (project rule - patch cv2 itself, AppTest runs
    app.py as its own module), never the 50 MB CNN, local mode by default."""
    import cv2

    def _forbidden(*a, **k):
        raise AssertionError("a test tried to open the REAL webcam")
    monkeypatch.setattr(cv2, 'VideoCapture', _forbidden)
    monkeypatch.setattr(config, 'MAX_USER_SESSIONS', 10_000)
    monkeypatch.setattr(daisee_ensemble, 'build_pairs', lambda skip=False: [])
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    monkeypatch.delenv('STRESS_AUTO_CONTEXT', raising=False)


def _app(**state):
    at = AppTest.from_file(APP, default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception
    return at


def _toggle(at):
    return next(b for b in at.button if b.label == 'Auto-detect')


def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def _labels(at):
    return [b.label for b in at.sidebar.button]


def _suggestion(mode='Studying', current='Default', status='suggest'):
    return ContextSuggestion(status=status, current_mode=current, mode=mode,
                             message=f"Suggested context: {mode}.", agreement=0.9,
                             confidence=0.72, usable=10, sampled=10,
                             top_scenes=[('home_office', 0.5), ('office', 0.2)])


# ---- defaults / scope ---------------------------------------------------------
def test_toggle_exists_locally_and_is_off_by_default(temp_db):
    at = _app()
    assert at.session_state['auto_context'] is False
    assert at.session_state['context_probe'] is None


def test_toggle_can_be_hidden_locally(temp_db, monkeypatch):
    monkeypatch.setenv('STRESS_AUTO_CONTEXT', '0')
    at = _app()
    assert not any(b.label == 'Auto-detect' for b in at.button)


def test_hosted_mode_also_shows_the_button(temp_db):
    """Signed-in hosted dashboard, built the same way test_app_hosted does."""
    import httpx
    from types import SimpleNamespace
    from postgrest import SyncPostgrestClient
    from auth import AuthSession
    from test_session_store import FakeSupabase
    config.STORAGE_BACKEND = 'supabase'            # restored by the guards fixture
    server = FakeSupabase()
    pg = SyncPostgrestClient('http://sb.test/rest/v1',
                             http_client=httpx.Client(transport=httpx.MockTransport(server.handler)))
    sess = AuthSession(SimpleNamespace(
        table=pg.from_, auth=SimpleNamespace(get_session=lambda: SimpleNamespace(access_token='t'),
                                             sign_out=lambda: None)))
    sess._user = SimpleNamespace(id='uid-alice', email='alice@example.com')
    at = _app(auth_session=sess)
    assert 'Sign out' in _labels(at)               # really the hosted dashboard
    assert any(b.label == 'Auto-detect' for b in at.button)


# ---- toggle off: the old Calibrate flow -----------------------------------------
def test_calibrate_with_the_toggle_off_creates_no_probe(temp_db):
    at = _app()
    _button(at, 'Calibrate').click().run()
    assert not at.exception
    assert at.session_state['calibrating'] is True
    assert at.session_state['context_probe'] is None
    assert at.session_state['context_suggestion'] is None


# ---- toggle on --------------------------------------------------------------------
def test_calibrate_with_the_toggle_on_arms_a_probe_without_loading_the_model(temp_db):
    at = _app()
    _toggle(at).click().run()
    _button(at, 'Calibrate').click().run()
    assert not at.exception
    probe = at.session_state['context_probe']
    assert isinstance(probe, ContextProbe)
    assert probe._clf is None                      # lazy: nothing loaded until a frame arrives
    assert _toggle(at).disabled                    # can't flip it mid-calibration
    assert any('Checking your surroundings' in c.value for c in at.sidebar.caption)


def test_stop_during_calibration_discards_the_probe(temp_db):
    at = _app()
    _toggle(at).click().run()
    _button(at, 'Calibrate').click().run()
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert at.session_state['context_probe'] is None
    assert at.session_state['context_suggestion'] is None


def test_a_suggestion_changes_nothing_until_accepted(temp_db):
    at = _app(auto_context=True, context_suggestion=_suggestion())
    assert at.session_state['task_mode'] == 'Default'
    assert 'Use Studying' in _labels(at) and 'Keep Default' in _labels(at)
    assert any('Suggested context: Studying' in m.value for m in at.sidebar.markdown)
    assert any('not a validated accuracy' in c.value for c in at.sidebar.caption)
    _button(at, 'Use Studying').click().run()
    assert not at.exception
    assert at.session_state['task_mode'] == 'Studying'
    assert at.session_state['task_select'] == 'Studying'       # the selector follows
    assert at.session_state['context_suggestion'] is None


def test_keep_current_leaves_the_mode_alone(temp_db):
    at = _app(auto_context=True, context_suggestion=_suggestion(mode='Hospital'))
    _button(at, 'Keep Default').click().run()
    assert not at.exception
    assert at.session_state['task_mode'] == 'Default'
    assert at.session_state['context_suggestion'] is None
    assert 'Use Hospital' not in _labels(at)


def test_a_manual_choice_overrides_a_pending_suggestion(temp_db):
    at = _app(auto_context=True, context_suggestion=_suggestion(mode='Studying'))
    next(s for s in at.sidebar.selectbox if s.label == 'Context').set_value('Hospital').run()
    assert not at.exception
    assert at.session_state['task_mode'] == 'Hospital'
    assert at.session_state['context_suggestion'] is None
    assert 'Use Studying' not in _labels(at)


@pytest.mark.parametrize('status, shown', [('unclear', 'info'), ('manual_only', 'info'),
                                           ('unavailable', 'warning')])
def test_fallback_messages_offer_no_switch(temp_db, status, shown):
    sug = ContextSuggestion(status=status, current_mode='Default',
                            message=f"fallback message for {status}")
    at = _app(auto_context=True, context_suggestion=sug)
    assert any(f'fallback message for {status}' in m.value for m in getattr(at.sidebar, shown))
    assert not any(l.startswith('Use ') for l in _labels(at))
    _button(at, 'Dismiss').click().run()
    assert at.session_state['context_suggestion'] is None
    assert at.session_state['task_mode'] == 'Default'


def test_turning_the_toggle_off_drops_the_suggestion(temp_db):
    at = _app(auto_context=True, context_suggestion=_suggestion())
    _toggle(at).click().run()
    assert not at.exception
    assert at.session_state['context_suggestion'] is None
    assert 'Use Studying' not in _labels(at)


def test_start_locks_the_context_and_drops_an_unanswered_suggestion(temp_db):
    at = _app(auto_context=True, context_suggestion=_suggestion())
    _button(at, 'Start').click().run()
    assert not at.exception
    assert at.session_state['running'] is True
    assert at.session_state['task_mode'] == 'Default'           # never applied silently
    assert at.session_state['context_suggestion'] is None
    assert _toggle(at).disabled
    assert next(s for s in at.sidebar.selectbox if s.label == 'Context').disabled


# ---- end to end through the local (cv2) capture path, fake camera ----------------
class FakeCapture:
    def __init__(self, n, clock=None):
        import numpy as np
        self.left, self.clock = n, clock
        self.frame = np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        self.released = False

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        if self.clock is not None:
            self.clock['offset'] += self.clock['step']
        return True, self.frame.copy()

    def release(self):
        self.released = True


@pytest.fixture
def fake_face(monkeypatch):
    import session_pipeline as sp
    from conftest import face_mesh_result, make_face_landmarks, make_pose_landmarks, pose_result
    face, pose = face_mesh_result(make_face_landmarks()), pose_result(make_pose_landmarks())
    orig = sp.UserState.__init__

    def patched(self):
        orig(self)
        self.engine.face_mesh.process = lambda rgb: face
        self.engine.pose.process = lambda rgb: pose
    monkeypatch.setattr(sp.UserState, '__init__', patched)


def test_calibration_produces_a_suggestion_and_accepting_it_reaches_the_saved_session(
        monkeypatch, temp_db, fake_face):
    import cv2, time as _time
    session_history.init_db()
    # Virtual clock: one step per captured frame (as test_app_smoke does), so the
    # 20 s calibration spans a fixed number of frames on any machine.
    clock = {'offset': 0.0, 'step': 0.1}
    base = _time.time()
    monkeypatch.setattr(_time, 'time', lambda: base + clock['offset'])
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'local')
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a, **k: FakeCapture(10_000, clock))
    clf = FakeClassifier(probs_for(home_office=0.6, office=0.2))
    probe = ContextProbe(classifier_factory=lambda: clf)

    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state['auto_context'] = True
    at.session_state['calibrating'] = True
    at.session_state['context_probe'] = probe
    at.run()
    assert not at.exception
    assert at.session_state['calibrating'] is False
    assert clf.calls == cd.MAX_SAMPLES                          # ~one frame per 2 s, not every frame
    sug = at.session_state['context_suggestion']
    assert (sug.status, sug.mode, sug.current_mode) == ('suggest', 'Studying', 'Default')
    assert at.session_state['task_mode'] == 'Default'           # suggestion only
    at.run()                                                    # the rerun after calibration
    _button(at, 'Use Studying').click().run()
    assert at.session_state['task_mode'] == 'Studying'

    # Start: a short fake recording through the real LiveRun, then the camera
    # "fails", which ends and saves the session (test_app_smoke covers that path).
    monkeypatch.setattr(cv2, 'VideoCapture', lambda *a, **k: FakeCapture(40, clock))
    _button(at, 'Start').click().run()
    assert not at.exception
    assert at.session_state['running'] is False
    (row,) = session_history.load_recent_sessions(5)
    assert row['task_mode'] == 'Studying'                       # the mode the user accepted


# ---- header-row UI: the suggested context is highlighted --------------------------
def _all_markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_the_suggested_context_button_is_highlighted_and_labelled(temp_db):
    at = _app(auto_context=True, context_suggestion=_suggestion(mode='Studying'))
    md = _all_markdown(at)
    assert '.st-key-ctxbtn_Studying button' in md and 'SUGGESTED' in md
    assert at.session_state['task_mode'] == 'Default'            # highlight only, nothing switched
    assert md.count('SUGGESTED') == 1


def test_no_highlight_without_a_change_suggestion(temp_db):
    for sug in (None, _suggestion(mode='Default'),
                ContextSuggestion(status='unclear', current_mode='Default', message='x')):
        at = _app(auto_context=True, context_suggestion=sug)
        assert 'SUGGESTED' not in _all_markdown(at)
    at = _app(auto_context=False, context_suggestion=_suggestion(mode='Studying'))
    assert 'SUGGESTED' not in _all_markdown(at)                 # toggle off: nothing shown


def test_the_toggle_sits_in_the_header_row_not_the_sidebar(temp_db):
    at = _app()
    assert any(b.label == 'Auto-detect' for b in at.button)
    assert 'Auto-detect' not in _labels(at)



def test_clicking_the_button_turns_auto_detect_on_and_off_and_shows_it(temp_db):
    at = _app()
    assert at.session_state['auto_context'] is False
    assert _toggle(at).proto.type == 'secondary'              # off = plain button
    _toggle(at).click().run()
    assert at.session_state['auto_context'] is True
    assert _toggle(at).proto.type == 'primary'                # on = solid blue fill
    _toggle(at).click().run()
    assert at.session_state['auto_context'] is False
    assert _toggle(at).proto.type == 'secondary'
