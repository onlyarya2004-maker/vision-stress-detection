"""
Login for the hosted app (Supabase Auth, email + password)
==========================================================
Two halves, kept apart so the logic is testable without a Streamlit runtime:

  AuthSession  - wraps ONE supabase client for ONE browser session.
  render_login_page() / ensure_login()  - the Streamlit UI around it.

Security rules this module enforces (they matter for the "one user can never
see another's data" requirement):

  * One supabase client PER Streamlit session, stored in st.session_state.
    NEVER @st.cache_resource: supabase-py keeps the signed-in user's token in
    the client's own headers, so a shared client would hand user A's
    credentials to user B's requests.
  * Only the PUBLIC key ever reaches this process. load_credentials() refuses
    a service_role / secret key outright - that key bypasses Row Level
    Security entirely and must live only on an admin's machine.
  * All data access then goes through session_store.SupabaseStore using the
    user's own JWT, so the database's RLS policies (supabase/schema.sql), not
    app code, decide what each person can see.
  * Tokens (access, refresh) stay inside the server-side client. They are
    never rendered, logged, or sent to the browser.

Registration is open (no invite code). What limits abuse is Supabase itself:
its password policy, its rate limits, and - when switched ON in the dashboard -
"Confirm email", which makes people prove they own the address they register.
The app handles both outcomes of sign-up (signed straight in, or "check your
email"). Data isolation never depends on any of this: RLS scopes every row to
the signed-in account.
"""
import base64
import hmac
import json
import os
import re

MIN_PASSWORD_LENGTH = 8
_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


class AuthFailure(Exception):
    """A login/sign-up problem with a message safe to show the user."""


def _jwt_role(key):
    """The `role` claim of a legacy-format Supabase key, or None."""
    parts = key.split('.')
    if len(parts) != 3:
        return None
    try:
        payload = parts[1] + '=' * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get('role')
    except Exception:
        return None


def validate_public_key(key):
    """Raise AuthFailure unless `key` is a public (anon/publishable) key."""
    if not key:
        raise AuthFailure("SUPABASE_ANON_KEY is not set.")
    if key.startswith('sb_secret_') or _jwt_role(key) == 'service_role':
        raise AuthFailure(
            "That is a SECRET (service_role) key. It bypasses all row-level "
            "security and must never be given to the app. Use the public "
            "'anon' / 'publishable' key instead.")
    return key


def _setting(name, secrets=None):
    """Environment first, then Streamlit secrets; None when unset."""
    v = os.environ.get(name)
    if v:
        return v
    if secrets is not None:
        try:
            return secrets.get(name)
        except Exception:
            return None
    return None


def load_credentials(secrets=None):
    """(url, public_key) from the environment, falling back to Streamlit
    secrets. Raises AuthFailure when missing or when given a secret key."""
    url = _setting('SUPABASE_URL', secrets)
    # Supabase renamed the public key from "anon" to "publishable"; accept both
    # names so a dashboard copy-paste works either way. The newer name wins.
    key = _setting('SUPABASE_PUBLISHABLE_KEY', secrets) or _setting('SUPABASE_ANON_KEY', secrets)
    if not url:
        raise AuthFailure("SUPABASE_URL is not set.")
    if not key:
        raise AuthFailure("Neither SUPABASE_PUBLISHABLE_KEY nor SUPABASE_ANON_KEY is set.")
    return url.rstrip('/'), validate_public_key(key)


def make_client(url, public_key):
    from supabase import create_client
    return create_client(url, public_key)


def _friendly(exc):
    text = str(exc)
    low = text.lower()
    if 'invalid login' in low or 'invalid credentials' in low:
        return "Wrong email or password."
    if 'not confirmed' in low:
        return "Please confirm your email first (check your inbox), then sign in."
    if 'already registered' in low or 'already exists' in low:
        return "An account with that email already exists - use Sign in."
    if 'signups not allowed' in low or 'signup is disabled' in low or 'signups are disabled' in low:
        return "New account registration is currently turned off."
    if 'password' in low and ('weak' in low or 'least' in low or 'short' in low):
        return f"Password too weak - use at least {MIN_PASSWORD_LENGTH} characters."
    if 'rate limit' in low or 'too many' in low:
        return "Too many attempts - wait a minute and try again."
    if 'invalid' in low and 'email' in low:
        return "Enter a valid email address."
    return "Something went wrong. Please try again."


def validate_registration(email, password, confirm):
    """Checks done before anything is sent to Supabase. Returns the cleaned
    email. Supabase still applies its own password policy afterwards."""
    email = (email or '').strip()
    if not email or not password:
        raise AuthFailure("Enter an email and a password.")
    if not _EMAIL_RE.match(email):
        raise AuthFailure("Enter a valid email address.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthFailure(f"Password too weak - use at least {MIN_PASSWORD_LENGTH} characters.")
    if not hmac.compare_digest((confirm or '').encode(), password.encode()):
        raise AuthFailure("The two passwords don't match.")
    return email


class AuthSession:
    """One user's authenticated link to Supabase."""

    def __init__(self, client):
        self.client = client
        self._user = None

    @property
    def is_signed_in(self):
        return self._user is not None

    @property
    def user_id(self):
        return str(self._user.id) if self._user else None

    @property
    def email(self):
        return getattr(self._user, 'email', None) if self._user else None

    def sign_in(self, email, password):
        email = (email or '').strip()
        if not email or not password:
            raise AuthFailure("Enter your email and password.")
        try:
            res = self.client.auth.sign_in_with_password(
                {"email": email, "password": password})
        except Exception as e:
            raise AuthFailure(_friendly(e))
        if not getattr(res, 'session', None) or not getattr(res, 'user', None):
            raise AuthFailure("Login failed.")
        self._user = res.user

    def sign_up(self, email, password):
        """'signed_in' when the project auto-confirms, 'confirm_email' when
        the user must click a link first."""
        email = (email or '').strip()
        if not email or not password:
            raise AuthFailure("Enter an email and a password.")
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AuthFailure(f"Password too weak - use at least {MIN_PASSWORD_LENGTH} characters.")
        try:
            res = self.client.auth.sign_up({"email": email, "password": password})
        except Exception as e:
            raise AuthFailure(_friendly(e))
        if getattr(res, 'session', None) and getattr(res, 'user', None):
            self._user = res.user
            return 'signed_in'
        return 'confirm_email'

    def sign_out(self):
        """Sign out THIS browser session only (scope 'local' revokes this
        session's refresh token; other devices stay signed in, as on any
        normal website)."""
        try:
            self.client.auth.sign_out({"scope": "local"})
        except TypeError:                       # clients without the options arg
            try:
                self.client.auth.sign_out()
            except Exception:
                pass
        except Exception:
            pass
        self._user = None

    def ensure_fresh(self):
        """Called before every database call. get_session() transparently
        refreshes an expiring access token (and supabase-py then swaps the
        new token into its request headers). Raises AuthFailure when the
        session is gone, so the UI can send the user back to the login."""
        try:
            session = self.client.auth.get_session()
        except Exception:
            session = None
        if session is None:
            self._user = None
            raise AuthFailure("Your session expired - please sign in again.")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
_PRIVACY_NOTE = (
    "**Privacy.** Once signed in, your browser shows a live camera preview on the "
    "main page (your browser's camera light will be on), but **no images leave your "
    "computer unless you press Calibrate or Start**. While a session runs, frames are "
    "analysed in memory on the server and are **never stored**. What is saved to your "
    "account is numbers only - stress score, signal values and timestamps. If you switch "
    "on **Auto-detect**, a few frames from your Calibrate step are also analysed in memory "
    "to suggest a context (Studying or Hospital); no image or description of your "
    "surroundings is saved. Each account can see only its own data; the database "
    "enforces that, not just the app."
)

_LOGIN_CSS = """
<style>
.auth-brand { text-align: center; margin: 2.2rem 0 1.2rem 0; }
.auth-brand .auth-title { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.01em; }
.auth-brand .auth-sub { color: #A3B1C6; font-size: 0.92rem; margin-top: 0.3rem; }
.auth-hint { color: #A3B1C6; font-size: 0.82rem; text-align: center; margin-top: 0.6rem; }
</style>
"""

def _streamlit_secrets():
    try:
        import streamlit as st
        return st.secrets
    except Exception:
        return None


def _brand(st, subtitle="Sign in to continue"):
    st.markdown(
        '<div class="auth-brand"><div class="auth-title">Vision-Based Behavioral '
        'Stress Detection</div><div class="auth-sub">Camera only &middot; no wearables '
        f'&middot; {subtitle}</div></div>', unsafe_allow_html=True)


def render_login_page(secrets=None):
    """Draw the sign-in / create-account page. On success stores the
    AuthSession in st.session_state['auth_session'] and reruns the script."""
    import streamlit as st

    secrets = secrets if secrets is not None else _streamlit_secrets()
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)

    try:
        url, key = load_credentials(secrets)
    except AuthFailure as e:
        _brand(st)
        st.error(f"The server is not configured for login: {e}")
        return

    def _session():
        if 'auth_client_session' not in st.session_state:
            st.session_state['auth_client_session'] = AuthSession(make_client(url, key))
        return st.session_state['auth_client_session']

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        _brand(st)
        notice = st.session_state.pop('auth_notice', None)
        if notice:
            st.warning(notice)
        with st.container(border=True):
            tab_in, tab_up = st.tabs(["Sign in", "Create account"])

            with tab_in:
                with st.form("login_form", border=False):
                    email = st.text_input("Email", key="login_email",
                                          placeholder="you@example.com", autocomplete="email")
                    pw = st.text_input("Password", type="password", key="login_pw",
                                       autocomplete="current-password")
                    go = st.form_submit_button("Sign in", type="primary", width='stretch')
                if go:
                    sess = _session()
                    try:
                        sess.sign_in(email, pw)
                    except AuthFailure as e:
                        st.error(str(e))
                    else:
                        st.session_state['auth_session'] = sess
                        st.rerun()
                st.markdown('<div class="auth-hint">New here? Open the '
                            '<b>Create account</b> tab.</div>', unsafe_allow_html=True)

            with tab_up:
                with st.form("signup_form", border=False):
                    email2 = st.text_input("Email", key="signup_email",
                                           placeholder="you@example.com", autocomplete="email")
                    pw2 = st.text_input("Password", type="password", key="signup_pw",
                                        help=f"At least {MIN_PASSWORD_LENGTH} characters.",
                                        autocomplete="new-password")
                    pw3 = st.text_input("Confirm password", type="password", key="signup_pw2",
                                        autocomplete="new-password")
                    go2 = st.form_submit_button("Create account", type="primary",
                                                width='stretch')
                if go2:
                    sess = _session()
                    try:
                        clean = validate_registration(email2, pw2, pw3)
                        outcome = sess.sign_up(clean, pw2)
                    except AuthFailure as e:
                        st.error(str(e))
                    else:
                        if outcome == 'signed_in':
                            st.session_state['auth_session'] = sess
                            st.rerun()
                        else:
                            st.success("Account created. We've sent you a confirmation link - "
                                       "open it, then come back here and sign in.")
                st.markdown('<div class="auth-hint">Already have an account? Open the '
                            '<b>Sign in</b> tab.</div>', unsafe_allow_html=True)

        st.caption(_PRIVACY_NOTE)


def current_session():
    """The signed-in AuthSession for this browser session, or None."""
    import streamlit as st
    sess = st.session_state.get('auth_session')
    return sess if (sess is not None and sess.is_signed_in) else None


def _clear_browser_session():
    # Drop EVERYTHING this browser session held (baselines, history, store):
    # the next person to use this tab must start from a blank slate.
    import streamlit as st
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def expire_current(message):
    """Forget a session whose tokens are no longer valid - with all of that
    user's in-memory state, exactly as a sign-out does, so whoever signs in
    next in this tab cannot inherit or save the previous user's calibration
    or unsaved recording - and leave a notice for the login page."""
    import streamlit as st
    _clear_browser_session()
    st.session_state['auth_notice'] = message


def sign_out_current():
    import streamlit as st
    sess = st.session_state.get('auth_session')
    if sess is not None:
        sess.sign_out()
    _clear_browser_session()
