"""
session_keeper - a tiny Streamlit custom component that keeps the user signed
in across a page refresh.

A refresh starts a brand-new Streamlit session, so the server-side login in
st.session_state is gone. This component stores the Supabase refresh token of
the signed-in account in the browser's localStorage (on the app's own origin,
as Supabase's own JavaScript client does) and hands it back after a refresh,
so auth.py can restore the session with it.

Modes:
  'write' - store `token`; no answer, so it never triggers a rerun.
  'clear' - forget the stored token (sign-out, expired session), then answer.
  'read'  - answer {nonce, token} once per nonce ('' when nothing is stored).

The nonce ties an answer to the Python session that asked: a value the
browser re-delivers from before a sign-out (old nonce) is ignored.
"""
from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND = Path(__file__).parent / "frontend"
_component = components.declare_component("session_keeper", path=str(_FRONTEND))


def session_keeper(*, mode, nonce="", token="", key="session_keeper"):
    return _component(mode=mode, nonce=nonce, token=token or "", key=key, default=None)
