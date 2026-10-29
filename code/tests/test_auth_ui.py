"""
The login page end to end through Streamlit's AppTest: the real
auth.render_login_page(), with Supabase replaced by a fake client (no network,
no real accounts).

Pinned down here:
  * the page offers Sign in / Create account with Confirm password - and no
    invite code and no third-party (Google) sign-in anywhere;
  * email/password sign-in works, and wrong credentials stay signed out;
  * registration sends only email + password, mismatched or short passwords
    never reach Supabase, and both Supabase outcomes are handled (signed
    straight in, or "confirm your email first");
  * sign-out clears the browser session.
"""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import auth
from test_auth import FakeGoTrue

SCRIPT = '''
import streamlit as st
import auth
sess = auth.current_session()
if sess is not None:
    st.markdown("SIGNED-IN:" + sess.user_id)
    if st.button("Sign out"):
        auth.sign_out_current()
        st.rerun()
else:
    auth.render_login_page(secrets={})
'''


@pytest.fixture
def env(monkeypatch):
    """Dummy (non-secret) credentials and a fake Supabase client."""
    monkeypatch.setenv('SUPABASE_URL', 'https://abcdefghijklmnopqrst.supabase.co')
    monkeypatch.setenv('SUPABASE_PUBLISHABLE_KEY', 'sb_publishable_dummyDummyDummy0123')
    gt = FakeGoTrue()
    monkeypatch.setattr(auth, 'make_client', lambda url, key: SimpleNamespace(auth=gt))
    return SimpleNamespace(gt=gt, mp=monkeypatch)


def _app():
    at = AppTest.from_string(SCRIPT, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(at):
    out = []
    for kind in ('markdown', 'caption', 'success', 'error', 'warning', 'info'):
        out += [str(e.value) for e in getattr(at, kind)]
    return out


def _signed_in(at):
    return any(str(m.value).startswith('SIGNED-IN:') for m in at.markdown)


def _btn(at, label):
    return next(b for b in at.button if b.label == label)


# ---- layout -------------------------------------------------------------------
def test_page_has_signin_and_signup_with_password_confirmation(env):
    at = _app()
    assert [t.label for t in at.tabs] == ['Sign in', 'Create account']
    assert [t.label for t in at.text_input] == ['Email', 'Password', 'Email', 'Password',
                                                'Confirm password']
    assert [b.label for b in at.button] == ['Sign in', 'Create account']
    assert any('Privacy' in t for t in _texts(at))


def test_no_invite_code_and_no_third_party_sign_in_anywhere(env):
    at = _app()
    everything = ' '.join(_texts(at) + [t.label for t in at.text_input]
                          + [b.label for b in at.button]).lower()
    for gone in ('invite', 'google', 'oauth'):
        assert gone not in everything
    assert at.get('link_button') == []


# ---- sign in -------------------------------------------------------------------------
def test_email_password_sign_in(env):
    at = _app()
    at.text_input(key='login_email').input('alice@example.com')
    at.text_input(key='login_pw').input('correct-horse-1')
    _btn(at, 'Sign in').click().run()
    assert _signed_in(at)


def test_wrong_password_shows_a_message_and_stays_signed_out(env):
    env.gt.fail = 'Invalid login credentials'
    at = _app()
    at.text_input(key='login_email').input('alice@example.com')
    at.text_input(key='login_pw').input('wrong-password')
    _btn(at, 'Sign in').click().run()
    assert not _signed_in(at)
    assert any('Wrong email or password' in e.value for e in at.error)


def test_unconfirmed_email_is_told_to_confirm_first(env):
    env.gt.fail = 'Email not confirmed'
    at = _app()
    at.text_input(key='login_email').input('alice@example.com')
    at.text_input(key='login_pw').input('correct-horse-1')
    _btn(at, 'Sign in').click().run()
    assert not _signed_in(at)
    assert any('confirm your email' in e.value for e in at.error)


# ---- create account ------------------------------------------------------------------
def _register(at, email, pw, confirm):
    at.text_input(key='signup_email').input(email)
    at.text_input(key='signup_pw').input(pw)
    at.text_input(key='signup_pw2').input(confirm)
    _btn(at, 'Create account').click().run()


def test_mismatched_passwords_never_reach_supabase(env):
    at = _app()
    _register(at, 'new@example.com', 'longenough1', 'longenough2')
    assert any("don't match" in e.value for e in at.error)
    assert not hasattr(env.gt, 'last_sign_up')


def test_short_passwords_never_reach_supabase(env):
    at = _app()
    _register(at, 'new@example.com', 'short', 'short')
    assert any('at least 8' in e.value for e in at.error)
    assert not hasattr(env.gt, 'last_sign_up')


def test_registration_needs_no_invite_code_and_says_to_confirm_the_email(env):
    env.mp.setenv('STRESS_SIGNUP_CODE', 'leftover')          # ignored now
    at = _app()
    _register(at, 'new@example.com', 'longenough1', 'longenough1')
    assert env.gt.last_sign_up == {'email': 'new@example.com', 'password': 'longenough1'}
    assert any('confirmation link' in s.value for s in at.success)
    assert not _signed_in(at)


def test_registration_on_an_auto_confirm_project_signs_straight_in(env):
    env.gt.session = SimpleNamespace(access_token='t')
    at = _app()
    _register(at, 'new@example.com', 'longenough1', 'longenough1')
    assert _signed_in(at)


def test_an_existing_address_gets_a_clear_message(env):
    env.gt.fail = 'User already registered'
    at = _app()
    _register(at, 'alice@example.com', 'longenough1', 'longenough1')
    assert any('already exists' in e.value for e in at.error)
    assert not _signed_in(at)


# ---- sign out ----------------------------------------------------------------------
def test_sign_out_returns_to_the_login_page(env):
    at = _app()
    at.text_input(key='login_email').input('alice@example.com')
    at.text_input(key='login_pw').input('correct-horse-1')
    _btn(at, 'Sign in').click().run()
    assert _signed_in(at)
    _btn(at, 'Sign out').click().run()
    assert not _signed_in(at)
    assert 'auth_session' not in at.session_state
    assert env.gt.sign_out_options == {'scope': 'local'}
    assert [t.label for t in at.tabs] == ['Sign in', 'Create account']


# ---- staying signed in across a page refresh --------------------------------------
# The browser side (session_keeper) is replaced by a dict that outlives each
# AppTest, the way localStorage outlives a page refresh. Each new AppTest is a
# refresh: a brand-new Streamlit session.
REFRESH_SCRIPT = '''
import streamlit as st
import auth
sess = auth.current_session()
if sess is not None:
    auth.remember_in_browser(sess)
    st.markdown("SIGNED-IN:" + sess.user_id)
    if st.button("Sign out"):
        auth.sign_out_current()
        st.rerun()
else:
    auth.render_login_page(secrets={})
'''


@pytest.fixture
def browser(env):
    import session_keeper
    storage = {}

    def fake_keeper(*, mode, nonce='', token='', key='session_keeper'):
        if mode == 'write':
            storage['token'] = token
            return None
        if mode == 'clear':
            storage.pop('token', None)
        return {'nonce': nonce, 'token': storage.get('token', '')}

    env.mp.setattr(session_keeper, 'session_keeper', fake_keeper)
    return storage


def _refresh():
    at = AppTest.from_string(REFRESH_SCRIPT, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _sign_in(at):
    at.text_input(key='login_email').input('alice@example.com')
    at.text_input(key='login_pw').input('correct-horse-1')
    _btn(at, 'Sign in').click().run()


def test_a_page_refresh_keeps_the_user_signed_in(env, browser):
    at = _refresh()
    assert not _signed_in(at)
    _sign_in(at)
    assert _signed_in(at)
    assert browser['token'] == 'rt-1'
    at2 = _refresh()                               # the user presses F5
    assert _signed_in(at2)
    assert browser['token'] == 'rt-2'              # rotated token saved back


def test_after_sign_out_a_refresh_stays_signed_out(env, browser):
    at = _refresh()
    _sign_in(at)
    _btn(at, 'Sign out').click().run()
    assert not _signed_in(at)
    assert 'token' not in browser                  # deleted from the browser
    assert not _signed_in(_refresh())


def test_a_dead_saved_token_shows_the_login_page_and_is_deleted(env, browser):
    browser['token'] = 'rt-5'
    env.gt.revoked = True
    at = _refresh()
    assert not _signed_in(at)
    assert [t.label for t in at.tabs] == ['Sign in', 'Create account']
    assert 'token' not in browser                  # deleted straight away
    assert not _signed_in(_refresh())              # and not tried again


def test_no_saved_token_just_shows_the_login_page(env, browser):
    at = _refresh()
    assert not _signed_in(at)
    assert [t.label for t in at.tabs] == ['Sign in', 'Create account']
