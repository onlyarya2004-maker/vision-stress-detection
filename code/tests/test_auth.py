"""
Tests for auth.py: key safety checks, credential loading, and AuthSession's
sign-in / sign-up / refresh behaviour against a fake supabase client.

The most important test here is the key check: the service_role key bypasses
Row Level Security completely, so pasting it into the hosted app by mistake
would silently turn "each user sees only their own data" into "every visitor
sees everything". load_credentials() must refuse it.
"""
import base64
import json
from types import SimpleNamespace

import pytest

import auth
from auth import AuthFailure, AuthSession


def _jwt(role):
    def b64(d):
        return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip('=')
    return f"{b64({'alg': 'HS256', 'typ': 'JWT'})}.{b64({'role': role, 'iss': 'supabase'})}.sig"


# ---- key / credential safety ---------------------------------------------
def test_service_role_jwt_is_refused():
    with pytest.raises(AuthFailure, match='SECRET'):
        auth.validate_public_key(_jwt('service_role'))


def test_new_style_secret_key_is_refused():
    with pytest.raises(AuthFailure, match='SECRET'):
        auth.validate_public_key('sb_secret_abc123')


@pytest.mark.parametrize('key', [_jwt('anon'), 'sb_publishable_abc123'])
def test_public_keys_are_accepted(key):
    assert auth.validate_public_key(key) == key


def test_missing_key_is_an_error():
    with pytest.raises(AuthFailure):
        auth.validate_public_key('')
    with pytest.raises(AuthFailure):
        auth.validate_public_key(None)


def test_load_credentials_prefers_the_environment_then_secrets(monkeypatch):
    monkeypatch.setenv('SUPABASE_URL', 'https://abc.supabase.co/')
    monkeypatch.setenv('SUPABASE_ANON_KEY', _jwt('anon'))
    url, key = auth.load_credentials()
    assert url == 'https://abc.supabase.co' and key == _jwt('anon')

    monkeypatch.delenv('SUPABASE_URL')
    monkeypatch.delenv('SUPABASE_ANON_KEY')
    with pytest.raises(AuthFailure, match='SUPABASE_URL'):
        auth.load_credentials()
    secrets = {'SUPABASE_URL': 'https://x.supabase.co', 'SUPABASE_ANON_KEY': 'sb_publishable_q'}
    assert auth.load_credentials(secrets) == ('https://x.supabase.co', 'sb_publishable_q')


def test_load_credentials_refuses_a_service_key_from_secrets_too(monkeypatch):
    monkeypatch.delenv('SUPABASE_URL', raising=False)
    monkeypatch.delenv('SUPABASE_ANON_KEY', raising=False)
    with pytest.raises(AuthFailure, match='SECRET'):
        auth.load_credentials({'SUPABASE_URL': 'https://x.supabase.co',
                               'SUPABASE_ANON_KEY': _jwt('service_role')})


# ---- AuthSession ----------------------------------------------------------
class FakeGoTrue:
    def __init__(self):
        self.session = None
        self.fail = None
        self.signed_out = False

    def sign_in_with_password(self, creds):
        if self.fail:
            raise Exception(self.fail)
        user = SimpleNamespace(id='uid-1', email=creds['email'])
        self.session = SimpleNamespace(access_token='tok')
        return SimpleNamespace(session=self.session, user=user)

    def sign_up(self, creds):
        if self.fail:
            raise Exception(self.fail)
        user = SimpleNamespace(id='uid-2', email=creds['email'])
        return SimpleNamespace(session=self.session, user=user)

    def get_session(self):
        return self.session

    def sign_out(self):
        self.signed_out = True
        self.session = None


@pytest.fixture
def sess():
    gt = FakeGoTrue()
    return AuthSession(SimpleNamespace(auth=gt)), gt


def test_sign_in_success_exposes_user_id_and_email(sess):
    s, _ = sess
    assert not s.is_signed_in
    s.sign_in('  a@b.com ', 'pw12345678')
    assert s.is_signed_in and s.user_id == 'uid-1' and s.email == 'a@b.com'


@pytest.mark.parametrize('raw, shown', [
    ('Invalid login credentials', 'Wrong email or password'),
    ('Email not confirmed', 'confirm your email'),
    ('User already registered', 'already exists'),
    ('Too many requests: rate limit exceeded', 'Too many attempts'),
])
def test_sign_in_failures_become_friendly_messages(sess, raw, shown):
    s, gt = sess
    gt.fail = raw
    with pytest.raises(AuthFailure, match=shown):
        s.sign_in('a@b.com', 'pw12345678')
    assert not s.is_signed_in


def test_sign_in_needs_both_fields(sess):
    s, _ = sess
    for e, p in (('', 'x'), ('a@b.com', ''), (None, None)):
        with pytest.raises(AuthFailure):
            s.sign_in(e, p)


def test_sign_up_rejects_short_passwords_before_calling_the_server(sess):
    s, gt = sess
    gt.fail = 'must not be reached'
    with pytest.raises(AuthFailure, match='8 characters'):
        s.sign_up('a@b.com', 'short')


def test_sign_up_reports_whether_email_confirmation_is_pending(sess):
    s, gt = sess
    assert s.sign_up('a@b.com', 'longenough1') == 'confirm_email'   # no session returned
    assert not s.is_signed_in
    gt.session = SimpleNamespace(access_token='t')                   # auto-confirm project
    assert s.sign_up('c@d.com', 'longenough1') == 'signed_in'
    assert s.is_signed_in and s.user_id == 'uid-2'


def test_ensure_fresh_passes_while_a_session_exists_and_fails_after_expiry(sess):
    s, gt = sess
    s.sign_in('a@b.com', 'pw12345678')
    s.ensure_fresh()                                    # no exception
    gt.session = None                                   # refresh failed / token revoked
    with pytest.raises(AuthFailure, match='expired'):
        s.ensure_fresh()
    assert not s.is_signed_in                           # UI will show the login page


def test_sign_out_clears_the_user_even_if_the_server_call_fails(sess):
    s, gt = sess
    s.sign_in('a@b.com', 'pw12345678')
    gt.sign_out = lambda: (_ for _ in ()).throw(RuntimeError('network'))
    s.sign_out()
    assert not s.is_signed_in and s.user_id is None


# ---- invite code for sign-ups ---------------------------------------------
def test_signup_is_open_when_no_invite_code_is_configured(monkeypatch):
    monkeypatch.delenv('STRESS_SIGNUP_CODE', raising=False)
    assert auth.signup_code_required() is None
    assert auth.signup_code_ok('anything', None) is True
    assert auth.signup_code_ok(None, None) is True


def test_invite_code_is_read_from_the_environment_then_secrets(monkeypatch):
    monkeypatch.setenv('STRESS_SIGNUP_CODE', 'bmsce2026')
    assert auth.signup_code_required() == 'bmsce2026'
    monkeypatch.delenv('STRESS_SIGNUP_CODE')
    assert auth.signup_code_required({'STRESS_SIGNUP_CODE': 'from-secrets'}) == 'from-secrets'


def test_invite_code_comparison_accepts_only_the_exact_code():
    assert auth.signup_code_ok('bmsce2026', 'bmsce2026')
    assert auth.signup_code_ok('  bmsce2026 ', 'bmsce2026')        # pasted whitespace is fine
    for wrong in ('', None, 'BMSCE2026', 'bmsce202', 'bmsce20266', 'x' * 500):
        assert not auth.signup_code_ok(wrong, 'bmsce2026')
