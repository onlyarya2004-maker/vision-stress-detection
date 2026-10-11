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
"""
import base64
import json
import os


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


def load_credentials(secrets=None):
    """(url, public_key) from the environment, falling back to Streamlit
    secrets. Raises AuthFailure when missing or when given a secret key."""
    def get(name):
        v = os.environ.get(name)
        if v:
            return v
        if secrets is not None:
            try:
                return secrets.get(name)
            except Exception:
                return None
        return None

    url = get('SUPABASE_URL')
    # Supabase renamed the public key from "anon" to "publishable"; accept both
    # names so a dashboard copy-paste works either way. The newer name wins.
    key = get('SUPABASE_PUBLISHABLE_KEY') or get('SUPABASE_ANON_KEY')
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
    if 'password' in low and ('weak' in low or 'least' in low or 'short' in low):
        return "Password too weak - use at least 8 characters."
    if 'rate limit' in low or 'too many' in low:
        return "Too many attempts - wait a minute and try again."
    return text or "Login failed."


def signup_code_required(secrets=None):
    """The invite code new accounts must present, or None when sign-up is
    open. Set STRESS_SIGNUP_CODE on the server to stop strangers who find the
    public URL from creating accounts (the Supabase key never reaches the
    browser, so the app's own form is the only way in)."""
    v = os.environ.get('STRESS_SIGNUP_CODE')
    if not v and secrets is not None:
        try:
            v = secrets.get('STRESS_SIGNUP_CODE')
        except Exception:
            v = None
    return v or None


def signup_code_ok(entered, required):
    """Constant-time comparison; True whenever no code is required."""
    if not required:
        return True
    import hmac
    return hmac.compare_digest((entered or '').strip().encode(), required.strip().encode())


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
        if len(password) < 8:
            raise AuthFailure("Password too weak - use at least 8 characters.")
        try:
            res = self.client.auth.sign_up({"email": email, "password": password})
        except Exception as e:
            raise AuthFailure(_friendly(e))
        if getattr(res, 'session', None) and getattr(res, 'user', None):
            self._user = res.user
            return 'signed_in'
        return 'confirm_email'

    def sign_out(self):
        try:
            self.client.auth.sign_out()
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
    "account is numbers only - stress score, signal values and timestamps. Each account "
    "can see only its own data; the database enforces that, not just the app."
)


def render_login_page(secrets=None):
    """Draw the login/sign-up form. On success stores the AuthSession in
    st.session_state['auth_session'] and reruns the script."""
    import streamlit as st

    st.markdown("## Vision-Based Behavioral Stress Detection")
    st.caption("Camera only · no wearables · each account sees only its own data")

    try:
        url, key = load_credentials(secrets if secrets is not None else _streamlit_secrets())
    except AuthFailure as e:
        st.error(f"The server is not configured for login: {e}")
        return

    left, mid, right = st.columns([1, 2, 1])
    with mid:
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])

        def _session():
            if 'auth_client_session' not in st.session_state:
                st.session_state['auth_client_session'] = AuthSession(make_client(url, key))
            return st.session_state['auth_client_session']

        with tab_in:
            with st.form("login_form"):
                email = st.text_input("Email", key="login_email")
                pw = st.text_input("Password", type="password", key="login_pw")
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

        with tab_up:
            invite = signup_code_required(secrets if secrets is not None else _streamlit_secrets())
            with st.form("signup_form"):
                email2 = st.text_input("Email", key="signup_email")
                pw2 = st.text_input("Password (8+ characters)", type="password", key="signup_pw")
                code2 = (st.text_input("Invite code", key="signup_code") if invite else None)
                go2 = st.form_submit_button("Create account", width='stretch')
            if go2:
                sess = _session()
                try:
                    if not signup_code_ok(code2, invite):
                        raise AuthFailure("That invite code is not valid.")
                    outcome = sess.sign_up(email2, pw2)
                except AuthFailure as e:
                    st.error(str(e))
                else:
                    if outcome == 'signed_in':
                        st.session_state['auth_session'] = sess
                        st.rerun()
                    else:
                        st.success("Account created. Check your email for a confirmation "
                                   "link, then use the Sign in tab.")
        st.markdown(_PRIVACY_NOTE)


def _streamlit_secrets():
    try:
        import streamlit as st
        return st.secrets
    except Exception:
        return None


def current_session():
    """The signed-in AuthSession for this browser session, or None."""
    import streamlit as st
    sess = st.session_state.get('auth_session')
    return sess if (sess is not None and sess.is_signed_in) else None


def sign_out_current():
    import streamlit as st
    sess = st.session_state.get('auth_session')
    if sess is not None:
        sess.sign_out()
    # Drop EVERYTHING this browser session held (baselines, history, store):
    # the next person to use this tab must start from a blank slate.
    for k in list(st.session_state.keys()):
        del st.session_state[k]
