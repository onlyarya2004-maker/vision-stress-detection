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
import json
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


# ---- end-of-session self-report (hosted) -----------------------------------
RATE_4 = "4 · noticeably stressed, tense but OK"


def _sessions_route(owned_ids=(42,)):
    """GET sessions: the ownership lookup answers only for ids this user owns
    (what RLS + the user filter return); every other read gets the one
    saved session."""
    def handler(r):
        p = _params(r)
        if p.get('select') == 'id' and p.get('id', '').startswith('eq.'):
            sid = int(p['id'][3:])
            return [{'id': sid}] if sid in owned_ids else []
        return httpx.Response(200, json=[SESSION_ROW], headers={'content-range': '0-0/1'})
    return handler


def _stopped_with_prompt(at, server, patch=lambda r: [{'id': 42, 'self_report': 4}]):
    server.on('GET', 'sessions', _sessions_route())
    server.on('POST', 'sessions', lambda r: [{'id': 42}])
    server.on('PATCH', 'sessions', patch)
    at.session_state['history'] = _history()
    at.session_state['history_saved'] = False
    at.session_state['running'] = True
    at.run()
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert at.session_state['pending_self_report'] == 42
    return next(b for b in at.button if b.label == RATE_4)


def test_hosted_rating_is_saved_to_supabase_not_sqlite(signed_in):
    at, server = signed_in
    _stopped_with_prompt(at, server).click().run()
    assert not at.exception
    (patch,) = server.of('PATCH', 'sessions')
    p = _params(patch)
    assert p['id'] == 'eq.42' and p['user_id'] == f'eq.{UID}'
    assert json.loads(patch.content) == {'self_report': 4}
    assert at.session_state['pending_self_report'] is None
    assert at.session_state['last_saved_self_report'] == 4
    assert any('rating was saved' in s.value for s in at.success)
    import session_history
    assert not session_history.DB_PATH.exists()          # the old bug wrote here


def test_hosted_failed_rating_write_is_not_reported_as_saved(signed_in):
    at, server = signed_in
    _stopped_with_prompt(at, server, patch=lambda r: []).click().run()   # 0 rows updated
    assert not at.exception
    assert not at.success
    assert any('NOT saved' in w.value for w in at.warning)
    assert at.session_state['pending_self_report'] == 42   # prompt stays: retry or Skip
    assert at.session_state['last_saved_self_report'] is None


def test_hosted_server_error_on_rating_is_reported_without_server_details(signed_in):
    at, server = signed_in
    boom = lambda r: httpx.Response(500, json={'message': 'internal detail XYZ'})
    _stopped_with_prompt(at, server, patch=boom).click().run()
    assert not at.exception and not at.success
    shown = ' '.join(w.value for w in at.warning)
    assert 'NOT saved' in shown and 'XYZ' not in shown


def test_hosted_rating_cannot_target_another_users_session(signed_in):
    """A session id that is not this user's (e.g. a tampered state value):
    the single update is filtered on THIS account's id (and RLS only lets it
    touch this account's rows), so it matches nothing, and the user is told
    the rating was not saved."""
    at, server = signed_in
    server.on('GET', 'sessions', _sessions_route(owned_ids=(42,)))
    server.on('PATCH', 'sessions', lambda r: [])            # no row of this user's has id 999
    at.run()
    at.session_state['pending_self_report'] = 999
    at.run()
    next(b for b in at.button if b.label == RATE_4).click().run()
    assert not at.exception
    for req in server.of('PATCH', 'sessions'):
        assert _params(req)['user_id'] == f'eq.{UID}' and _params(req)['id'] == 'eq.999'
    assert any('NOT saved' in w.value for w in at.warning) and not at.success


# ---- personalization status (hosted) ---------------------------------------
def _with_rated_history(server, user, relaxed=0.10, stressed=0.22,
                        baseline_at='2026-09-30T09:00:00'):
    """Six rated sessions whose GENERAL index reads this user too low (every
    stressed session under the 0.24 cutoff) - a case the leave-one-out check
    accepts, so personalization switches on."""
    labeled = [{'id': i, 'user_id': user, 'self_report': 1 if i <= 3 else 5,
                'sample_count': 120, 'task_mode': 'Default',
                'started_at': '2026-10-0%dT10:00:00' % i, 'ended_at': '2026-10-0%dT10:01:00' % i,
                'duration_sec': 60.0, 'mean_stress': relaxed if i <= 3 else stressed,
                'max_stress': 0.4,
                'pct_low': 50.0, 'pct_moderate': 50.0, 'pct_high': 0.0}
               for i in range(1, 7)]

    def sessions(r):
        p = _params(r)
        if p.get('user_id') != f'eq.{user}':
            return []                                   # RLS: nothing of anyone else's
        if p.get('self_report') == 'not.is.null':
            return labeled
        return httpx.Response(200, json=labeled[:1], headers={'content-range': '0-0/6'})

    def samples(r):
        p = _params(r)
        if p.get('user_id') != f'eq.{user}' or p.get('offset', '0') != '0':
            return []
        sid = int(p['session_id'][3:])
        return [{'stress_index': relaxed if sid <= 3 else stressed}] * 120
    server.on('GET', 'sessions', sessions)
    server.on('GET', 'samples', samples)
    # The account's saved baseline, stamped when it took effect (before the
    # six sessions above) - written by stamp_baseline_change() on calibration.
    server.on('GET', 'profiles', lambda r: [{'baseline_json': {
        'behavioral': {'ear_baseline': 0.30, 'calibrated': True}, 'daisee_baseline': 0.5,
        'baseline_set_at': baseline_at}}] if _params(r).get('id') == f'eq.{user}' else [])


def test_a_returning_user_with_enough_rated_sessions_sees_personal_calibration(signed_in):
    at, server = signed_in
    _with_rated_history(server, UID)
    at.run()
    assert not at.exception
    p = at.session_state['personalization']
    assert p.active and p.n_sessions == 6 and p.shift > 0
    assert any('Personal calibration on' in c.value for c in at.sidebar.caption)
    for req in server.requests:                               # still all Alice's
        table = req.url.path.rsplit('/', 1)[-1]
        if req.method == 'GET' and table in ('sessions', 'samples'):
            assert _params(req).get('user_id') == f'eq.{UID}'


def test_a_new_user_gets_the_general_model_and_is_told_why(signed_in):
    at, server = signed_in
    _with_one_saved_session(server)                            # unrated
    at.run()
    assert not at.exception
    p = at.session_state['personalization']
    assert not p.active and p.shift == 0.0
    # No calibration on record yet, so no session can be matched to a baseline.
    assert any('General model' in c.value and 'calibrate once' in c.value
               for c in at.sidebar.caption)


def test_another_users_rated_history_does_not_personalize_this_account(signed_in):
    """The database holds six rated sessions - all Bob's. Alice's account must
    stay on the general model."""
    at, server = signed_in
    _with_rated_history(server, 'uid-bob')
    at.run()
    assert not at.exception
    assert not at.session_state['personalization'].active


def test_switching_the_context_mode_refits_on_that_modes_sessions_only(signed_in):
    """All six rated sessions are Default-mode: personal calibration is on in
    Default and off after switching the Context selector to Studying."""
    at, server = signed_in
    _with_rated_history(server, UID)
    at.run()
    assert at.session_state['personalization'].active
    at.selectbox(key='task_select').select('Studying').run()
    assert not at.exception
    p = at.session_state['personalization']
    assert not p.active and p.shift == 0.0
    assert any('counting Studying sessions' in c.value for c in at.sidebar.caption)


def test_sessions_older_than_the_accounts_current_baseline_do_not_personalize(signed_in):
    """The account recalibrated after all six rated sessions: they were scored
    against the previous baseline, so none of them is pooled."""
    at, server = signed_in
    _with_rated_history(server, UID, baseline_at='2026-10-09T12:00:00')
    at.run()
    assert not at.exception
    p = at.session_state['personalization']
    assert not p.active and 'since your last calibration' in p.reason


# ---- recording persistence: every Stop outcome is visible and logged -----
# Regression for the silent-save defect: save_current_session() used to
# report a failure with st.warning() and the Stop handler then called
# st.rerun(), wiping it - a failed save looked exactly like nothing at all.
def _ready_to_stop(at, server, history=None, samples=lambda r: [], sessions_post=None):
    _with_one_saved_session(server)
    if sessions_post is not None:
        server.on('POST', 'sessions', sessions_post)
    server.on('POST', 'samples', samples)
    at.session_state['history'] = _history() if history is None else history
    at.session_state['history_saved'] = False
    at.session_state['running'] = True
    at.run()


def _infos(at):
    return [e.value for e in list(at.success) + list(at.warning) + list(at.info) + list(at.error)]


def test_a_stopped_recording_is_saved_with_all_its_samples_and_confirmed(signed_in, capfd):
    at, server = signed_in
    _ready_to_stop(at, server)
    _button(at, 'Stop').click().run()
    assert not at.exception
    (sess,) = server.of('POST', 'sessions')
    assert json.loads(sess.content)['sample_count'] == 10
    rows = [r for req in server.of('POST', 'samples') for r in json.loads(req.content)]
    assert len(rows) == 10 and all(r['session_id'] == 42 for r in rows)
    assert any('Recording saved (10 samples)' in m for m in _infos(at))
    assert at.session_state['pending_self_report'] == 42 and at.session_state['history_saved']
    out = capfd.readouterr().out
    assert '[save] session saved: 10 samples' in out
    assert 'alice' not in out and UID not in out            # no personal data in the log


def test_stop_with_no_processed_frames_says_nothing_was_saved(signed_in, capfd):
    at, server = signed_in
    _ready_to_stop(at, server, history=[])
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert server.of('POST', 'sessions') == [] and server.of('POST', 'samples') == []
    assert any('Nothing was saved' in m for m in _infos(at))
    assert at.session_state['pending_self_report'] is None
    assert '[save] nothing to save: 0 processed samples' in capfd.readouterr().out


def test_a_failed_sample_insert_is_rolled_back_and_the_recording_kept_for_retry(signed_in, capfd):
    at, server = signed_in
    boom = lambda r: httpx.Response(500, json={'message': 'internal detail XYZ'})
    _ready_to_stop(at, server, samples=boom)
    _button(at, 'Stop').click().run()
    assert not at.exception
    # The half-written session row was removed - by id AND owner.
    (rollback,) = server.of('DELETE', 'sessions')
    p = _params(rollback)
    assert p['id'] == 'eq.42' and p['user_id'] == f'eq.{UID}'
    # The user is told, persistently, and nothing claims success.
    assert any('NOT saved' in e.value for e in at.error)
    assert not at.success and at.session_state['pending_self_report'] is None
    assert 'XYZ' not in ' '.join(_infos(at))                  # no server detail leaked
    # The recording is still there, and cannot be overwritten by accident.
    assert len(at.session_state['history']) == 10 and at.session_state['save_failed']
    assert _button(at, 'Start').disabled and _button(at, 'Calibrate').disabled
    assert _button(at, 'Sign out').disabled
    assert '[save] FAILED to save a recording of 10 samples' in capfd.readouterr().out
    # Retry once the database is back: saved, rating prompt appears.
    server.on('POST', 'samples', lambda r: [])
    next(b for b in at.button if b.label == 'Retry saving').click().run()
    assert not at.exception
    assert any('Recording saved (10 samples)' in m for m in _infos(at))
    assert not at.session_state['save_failed'] and at.session_state['pending_self_report'] == 42
    assert len(server.of('POST', 'sessions')) == 2            # first attempt was rolled back
    assert not any('NOT saved' in e.value for e in at.error)


def test_a_rejected_session_insert_is_reported_not_hidden(signed_in, capfd):
    at, server = signed_in
    _ready_to_stop(at, server,
                   sessions_post=lambda r: httpx.Response(503, json={'message': 'down'}))
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert server.of('POST', 'samples') == []                 # nothing written after it
    assert any('NOT saved' in e.value for e in at.error) and not at.success
    assert at.session_state['save_failed'] and len(at.session_state['history']) == 10
    assert '[save] FAILED' in capfd.readouterr().out


def test_a_network_failure_during_save_is_reported(signed_in):
    at, server = signed_in

    def unreachable(r):
        raise httpx.ConnectError("no route to host")
    _ready_to_stop(at, server, sessions_post=unreachable)
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert any('NOT saved' in e.value and 'ConnectError' in e.value for e in at.error)
    assert at.session_state['save_failed']


def test_discarding_an_unsaved_recording_is_explicit_and_unblocks_start(signed_in, capfd):
    at, server = signed_in
    _ready_to_stop(at, server, samples=lambda r: httpx.Response(500, json={}))
    _button(at, 'Stop').click().run()
    next(b for b in at.button if b.label == 'Discard this recording').click().run()
    assert not at.exception
    assert at.session_state['history'] == [] and not at.session_state['save_failed']
    assert any('Recording discarded' in m for m in _infos(at))
    assert not _button(at, 'Start').disabled
    assert '[save] unsaved recording discarded by the user' in capfd.readouterr().out


def test_a_coaching_log_failure_does_not_claim_the_recording_failed(signed_in):
    at, server = signed_in
    _ready_to_stop(at, server)
    server.on('POST', 'interventions', lambda r: httpx.Response(500, json={}))
    at.session_state['pending_interventions'] = [{
        'ts': datetime(2026, 10, 10, 9, 1), 'kind': 'break', 'trigger_signal': None,
        'index_before': 0.6, 'index_after': 0.5, 'outcome': 'helped'}]
    _button(at, 'Stop').click().run()
    assert not at.exception
    msgs = _infos(at)
    assert any('Recording saved (10 samples)' in m and 'coaching outcome' in m for m in msgs)
    assert not any('NOT saved' in m for m in msgs)
    assert at.session_state['pending_self_report'] == 42 and not at.session_state['save_failed']


def test_a_second_stop_for_the_same_recording_does_not_duplicate_it(signed_in):
    at, server = signed_in
    _ready_to_stop(at, server)
    _button(at, 'Stop').click().run()
    at.session_state['running'] = True                        # a stale second Stop
    at.run()
    _button(at, 'Stop').click().run()
    assert not at.exception
    assert len(server.of('POST', 'sessions')) == 1
