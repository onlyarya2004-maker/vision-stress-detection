"""
The hosted flow end to end through the real app.py, with the database replaced
by the same mock-transport PostgREST client test_session_store.py uses.

What this pins down is the property the teacher asked for - one account can
never see another's data - at the level of the WHOLE APP rather than one
class: every database request the app makes for a signed-in user must be
scoped to that user, and the app must expose no way to pick another profile.
(The database's own Row Level Security is the second wall; scripts/rls_check.py
proves that one against a real project.)
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from postgrest import SyncPostgrestClient
from streamlit.testing.v1 import AppTest

import config
import daisee_ensemble
from auth import AuthSession
from session_history import SIGNAL_COLUMNS
from test_session_store import FakeSupabase

APP = str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'app.py')
UID = 'uid-alice'


@pytest.fixture(autouse=True)
def _guards(monkeypatch):
    import cv2
    monkeypatch.setattr(cv2, 'VideoCapture',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("real webcam")))
    monkeypatch.setattr(config, 'MAX_USER_SESSIONS', 10_000)
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'supabase')
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    monkeypatch.setattr(daisee_ensemble, 'build_pairs', lambda skip=False: [])


@pytest.fixture
def signed_in(temp_db):
    """(AppTest, FakeSupabase) with Alice already signed in. temp_db points the
    local SQLite module at a scratch file so the test can prove hosted mode
    never touches it (and can never touch the real ../data/sessions.db)."""
    server = FakeSupabase()
    pg = SyncPostgrestClient('http://sb.test/rest/v1',
                             http_client=httpx.Client(transport=httpx.MockTransport(server.handler)))
    gotrue = SimpleNamespace(get_session=lambda: SimpleNamespace(access_token='tok'),
                             sign_out=lambda: None)
    sess = AuthSession(SimpleNamespace(table=pg.from_, auth=gotrue))
    sess._user = SimpleNamespace(id=UID, email='alice@example.com')
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state['auth_session'] = sess
    return at, server


SESSION_ROW = {
    'id': 7, 'user_id': UID, 'started_at': '2026-10-09T18:30:00', 'ended_at': '2026-10-09T18:31:00',
    'duration_sec': 60.0, 'sample_count': 120, 'mean_stress': 0.31, 'max_stress': 0.5,
    'pct_low': 70.0, 'pct_moderate': 30.0, 'pct_high': 0.0, 'task_mode': 'Default',
    'self_report': None,
}


def _with_one_saved_session(server):
    server.on('GET', 'sessions', lambda r: httpx.Response(
        200, json=[SESSION_ROW], headers={'content-range': '0-0/1'}))
    server.on('POST', 'sessions', lambda r: [{'id': 42}])      # what Postgres returns for the new row


def _button(at, label):
    return next(b for b in at.sidebar.button if b.label == label)


def _history(n=10):
    start = datetime(2026, 10, 10, 9, 0, 0)
    rows = []
    for i in range(n):
        r = {'ts': start + timedelta(seconds=0.5 * i), 'stress_index': 0.3, 'level': 'LOW',
             'daisee': 0.5, 'behavior': 0.25}
        r.update({c: 0.1 for c in SIGNAL_COLUMNS})
        rows.append(r)
    return rows


def _params(req):
    return dict(req.url.params)


def test_signed_in_user_sees_their_account_and_no_profile_picker(signed_in):
    at, server = signed_in
    _with_one_saved_session(server)
    at.run()
    assert not at.exception
    labels = [b.label for b in at.sidebar.button]
    assert 'Sign out' in labels and 'Delete my data' in labels
    assert 'Add user' not in labels                      # no way to create/pick other profiles
    assert not any(s.label == 'User' for s in at.sidebar.selectbox)
    assert any('alice@example.com' in m.value for m in at.sidebar.markdown)
    assert at.session_state['active_user_id'] == UID


def test_every_query_the_app_makes_is_scoped_to_the_signed_in_user(signed_in):
    at, server = signed_in
    _with_one_saved_session(server)
    at.session_state['history'] = _history()
    at.session_state['history_saved'] = False
    at.session_state['running'] = True
    at.run()
    _button(at, 'Stop').click().run()                    # saves the session
    _button(at, 'Delete my data').click().run()          # and wipes only Alice's
    assert not at.exception

    assert server.requests, "the hosted run made no database calls at all"
    for req in server.requests:
        table = req.url.path.rsplit('/', 1)[-1]
        p = _params(req)
        if req.method == 'POST':
            body = req.content.decode()
            assert 'user_id' not in body, "owner must come from auth.uid(), not the client"
        elif table == 'profiles':
            assert p.get('id') == f'eq.{UID}'
        else:
            assert p.get('user_id') == f'eq.{UID}', f"unscoped {req.method} {req.url}"
    methods = {(r.method, r.url.path.rsplit('/', 1)[-1]) for r in server.requests}
    assert ('POST', 'sessions') in methods and ('POST', 'samples') in methods
    assert ('DELETE', 'sessions') in methods
    # Hosted mode never touches the local SQLite file at all.
    import session_history
    assert not session_history.DB_PATH.exists()


def test_calibrating_loads_the_accounts_saved_baseline_not_a_shared_one(signed_in):
    at, server = signed_in
    server.on('GET', 'profiles', lambda r: [{'baseline_json': {
        'behavioral': {'ear_baseline': 0.33, 'calibrated': True}, 'daisee_baseline': 0.61}}])
    at.run()
    assert not at.exception
    ustate = at.session_state['user_state']
    assert ustate.engine.baseline.ear_baseline == pytest.approx(0.33)
    assert ustate.daisee_baseline.value == pytest.approx(0.61)
    assert ustate.fully_calibrated


def test_sign_out_wipes_the_browser_sessions_state(signed_in):
    at, server = signed_in
    at.run()
    assert 'user_state' in at.session_state
    _button(at, 'Sign out').click().run()
    assert not at.exception
    assert 'user_state' not in at.session_state and 'auth_session' not in at.session_state
    at.run()                                              # the st.rerun() the handler requested
    assert not at.exception
    assert not at.sidebar.button                          # back at the login gate: no app UI
    assert 'user_state' not in at.session_state


# ---- the login gate (no session / expired session) ------------------------
def test_nothing_of_the_dashboard_renders_without_a_session(monkeypatch, temp_db):
    monkeypatch.setenv('SUPABASE_URL', 'https://abcdefghijklmnopqrst.supabase.co')
    monkeypatch.setenv('SUPABASE_PUBLISHABLE_KEY', 'sb_publishable_dummyDummyDummy0123')
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception
    assert not at.sidebar.button and not at.sidebar.markdown        # no app chrome at all
    labels = [b.label for b in at.button]
    assert 'Sign in' in labels and 'Create account' in labels
    for app_only in ('Start', 'Stop', 'Calibrate', 'Delete my data', 'Sign out'):
        assert app_only not in labels
    assert 'user_state' not in at.session_state                     # no camera/model state built
    import session_history
    assert not session_history.DB_PATH.exists()                     # and no local DB fallback


def test_an_expired_session_goes_back_to_the_login_page(signed_in, monkeypatch):
    at, server = signed_in
    monkeypatch.setenv('SUPABASE_URL', 'https://abcdefghijklmnopqrst.supabase.co')
    monkeypatch.setenv('SUPABASE_PUBLISHABLE_KEY', 'sb_publishable_dummyDummyDummy0123')
    at.session_state['auth_session'].client.auth.get_session = lambda: None   # refresh failed
    at.run()
    assert not at.exception
    assert 'auth_session' not in at.session_state
    assert any('expired' in w.value for w in at.warning)
    assert not at.sidebar.button
    assert server.requests == []                                    # no query with a dead token


def test_after_expiry_the_next_user_inherits_and_saves_nothing_of_the_previous_one(
        signed_in, monkeypatch):
    """Alice's session expires mid-use with an unsaved recording; Bob then signs
    in on the SAME tab. Bob must start from a blank slate, and nothing of
    Alice's may be written anywhere - not to her account, not to his."""
    at, server = signed_in
    monkeypatch.setenv('SUPABASE_URL', 'https://abcdefghijklmnopqrst.supabase.co')
    monkeypatch.setenv('SUPABASE_PUBLISHABLE_KEY', 'sb_publishable_dummyDummyDummy0123')

    at.run()                                                   # Alice's dashboard
    alice_state = at.session_state['user_state']
    at.session_state['history'] = _history()                   # unsaved recording
    at.session_state['history_saved'] = False

    # Her refresh token is revoked / the refresh fails.
    at.session_state['auth_session'].client.auth.get_session = lambda: None
    at.run()
    assert not at.exception
    assert any('expired' in w.value for w in at.warning)
    for key in ('auth_session', 'auth_client_session', 'user_state', 'history',
                'history_saved', 'active_user_id', 'active_user_name'):
        assert key not in at.session_state, f"{key} survived the expiry"

    # Bob signs in on the same tab.
    server.requests.clear()
    pg = SyncPostgrestClient('http://sb.test/rest/v1',
                             http_client=httpx.Client(transport=httpx.MockTransport(server.handler)))
    gotrue = SimpleNamespace(get_session=lambda: SimpleNamespace(access_token='tok-bob'),
                             sign_out=lambda *a, **k: None)
    bob = AuthSession(SimpleNamespace(table=pg.from_, auth=gotrue))
    bob._user = SimpleNamespace(id='uid-bob', email='bob@example.com')
    at.session_state['auth_session'] = bob
    at.run()
    assert not at.exception
    assert at.session_state['active_user_id'] == 'uid-bob'
    assert at.session_state['history'] == []                   # nothing inherited
    assert at.session_state['user_state'] is not alice_state   # fresh calibration state
    assert any('bob@example.com' in m.value for m in at.sidebar.markdown)

    # Nothing was saved for anyone, and every request was Bob's own.
    assert not any(r.method == 'POST' for r in server.requests), "a recording was saved"
    for req in server.requests:
        p = _params(req)
        table = req.url.path.rsplit('/', 1)[-1]
        assert (p.get('id') if table == 'profiles' else p.get('user_id')) == 'eq.uid-bob'
