"""
Vision-Based Behavioral Stress Detection - Web UI (v8)
=======================================================
FER2013 emotion branch fully removed after A/B evaluation showed it hurt
live-system accuracy. Active branches: DaiSEE CNN ensemble (re-centered
against personal baseline) + behavioral feature engine + rPPG + AI
intervention agent. Calibrated against WESAD physiological reference.

UI follows UI_DESIGN_BRIEF.md / the "Fixed Sidebars" concept selected from
the generated mockups: fixed-height panels, no page scroll, near-black /
blue / orange palette, zero emoji.

Run:
    streamlit run app.py
"""

import io
import csv
import dataclasses
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2  import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from behavioral_features import BehavioralBaseline, BehavioralFeatureEngine
from stress_fusion import (
    compute_stress_index, TemporalSmoother, stress_level, stress_level_advice,
    HysteresisLevel,
    set_daisee_baseline, reset_daisee_baseline, is_daisee_calibrated,
    get_daisee_baseline,
    VALID_TASK_MODES,
)
from temporal_model import GRUTemporalInference
from app_naming import export_filename
from rppg_live import get_live_rppg, reset_live_rppg
from intervention_agent import InterventionAgent, CATALOGUE
import session_history
import config

# ---------------- CONFIG ----------------
# All values now live in config.py; these aliases keep the rest of the file
# readable and preserve the previous names.
DAISEE_MOBILENET_PATH    = config.DAISEE_MOBILENET_PATH
DAISEE_EFFICIENTNET_PATH = config.DAISEE_EFFICIENTNET_PATH

MOBILENET_SIZE      = config.MOBILENET_INPUT_SIZE
EFFICIENTNET_SIZE   = config.EFFICIENTNET_INPUT_SIZE
SMOOTHING_WINDOW    = config.SMOOTHING_WINDOW_FRAMES
CNN_EVERY_N_FRAMES  = config.CNN_EVERY_N_FRAMES
LIVE_GRAPH_WINDOW   = config.LIVE_GRAPH_WINDOW
CALIBRATION_SECONDS = config.CALIBRATION_SECONDS

GUIDANCE_TIPS = {
    'LOW': [
        "Maintain your current rhythm",
        "Keep up regular hydration",
        "Steady breathing keeps you focused",
    ],
    'MODERATE': [
        "Take a 2-3 minute break",
        "Relax your jaw and shoulders",
        "Look 20 ft away for 20 seconds",
        "Take a few slow deep breaths",
    ],
    'HIGH': [
        "Step away from the screen",
        "Drink water and stretch",
        "Practice slow box-breathing (4-4-4-4)",
        "Consider a 10-minute pause",
    ],
}

TASK_LABELS = {
    'Default':  ('Default', 'GEN'),
    'Studying': ('Studying', 'STU'),
    'Hospital': ('Hospital', 'MED'),
    'Driving':  ('Driving', 'DRV'),
}

AGENT_PHASES = ['warming', 'watching', 'showing', 'evaluating', 'cooldown']

st.set_page_config(
    page_title="Vision-Based Behavioral Stress Detection",
    page_icon="\U0001F9E0", layout="wide", initial_sidebar_state="expanded",
)


# =============================================================================
# STYLE
# =============================================================================
def inject_css():
    st.markdown("""
    <style>
      :root {
        --bg:#0B0F14; --panel:#10161D; --panel2:#141B23; --border:#1C2733;
        --border2:#2E3B4B; --text:#E6EDF3; --sub:#8B98A5; --dim:#5B6772;
        --blue:#2F6FED; --orange:#F5762B; --green:#22C55E; --amber:#F59E0B;
        --red:#EF4444; --indigo:#7C8CF8;
        --mono:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;
      }
      html, body, [data-testid="stAppViewContainer"] {
        font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
        background:
          linear-gradient(rgba(47,111,237,.035) 1px, transparent 1px) 0 0/28px 28px,
          linear-gradient(90deg, rgba(47,111,237,.035) 1px, transparent 1px) 0 0/28px 28px,
          var(--bg);
      }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
      [data-testid="stStatusWidget"], .stAppDeployButton { display:none !important; }
      [data-testid="stHeader"] { background:transparent; height:0; }
      .block-container { padding-top:0.9rem; padding-bottom:1rem; max-width:1350px; }
      section[data-testid="stSidebar"] { background:#0A0E13; border-right:1px solid var(--border); }
      section[data-testid="stSidebar"] .block-container { padding-top:1rem; }
      h3 { font-size:13px !important; font-weight:700 !important;
           letter-spacing:.08em; text-transform:uppercase; color:var(--sub) !important; }

      /* ---- native widgets restyled to match the console ---- */
      .stButton>button {
        background:var(--panel2); color:var(--text);
        border:1px solid var(--border2); border-radius:7px;
        font-size:12.5px; font-weight:700; letter-spacing:.02em;
        padding:.45rem 1rem; transition:border-color .15s, background .15s;
      }
      .stButton>button:hover { border-color:var(--blue); color:#fff; background:#182233; }
      .stButton>button[kind="primary"] { background:var(--blue); border-color:var(--blue); color:#fff; }
      .stButton>button[kind="primary"]:hover { background:#2861D6; border-color:#2861D6; }
      .stButton>button:disabled { opacity:.35; }
      div[data-testid="stSelectbox"] > div, .stTextInput input {
        background:var(--panel2) !important; border-color:var(--border2) !important;
        border-radius:6px !important; font-size:12.5px !important;
      }
      [data-testid="stExpander"] { background:var(--panel2); border:1px solid var(--border);
                                    border-radius:8px; }
      [data-testid="stExpander"] summary { font-size:10.5px; letter-spacing:.08em;
                                            text-transform:uppercase; color:var(--sub); font-weight:700; }
      hr { border-color:var(--border) !important; }

      /* Native bordered containers become our "panel" primitive. Streamlit
         1.56 gives st.container(border=True) no distinct testid - the
         border lands directly on a plain stVerticalBlock, indistinguishable
         by attribute from any other layout block on the page - so we key
         off content that's already the first thing inside every one of
         ours: a .label header, or one of the HUD/agent markers used below.
         The `> [data-testid="stElementContainer"]` hop is load-bearing: a
         bordered container is a stVerticalBlock wrapped in a further
         (unbordered) stLayoutWrapper > stVerticalBlock - without that
         direct-child hop, :has() also matches that outer wrapper, since
         :has() looks at descendants of any depth, not just children. */
      div[data-testid="stVerticalBlock"]:has(
        > [data-testid="stElementContainer"] .label,
        > [data-testid="stElementContainer"] .agent-wrap,
        > [data-testid="stElementContainer"] .hud-frame-marker,
        > [data-testid="stElementContainer"] .panel-marker
      ) {
        background:var(--panel); border:1px solid var(--border) !important;
        border-radius:10px;
      }

      .label { font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
               color:var(--sub); font-weight:700; margin-bottom:8px;
               display:flex; align-items:center; gap:7px; }
      .label::before { content:''; width:3px; height:11px; background:var(--blue);
                        border-radius:1px; display:inline-block; flex-shrink:0; }
      .mono { font-family:var(--mono); font-variant-numeric:tabular-nums; }

      /* ---- header ---- */
      .app-header { display:flex; align-items:center; gap:14px; padding:2px 4px 12px 4px;
                    flex-wrap:wrap; }
      .app-icon { width:36px; height:36px; border-radius:9px;
                  background:linear-gradient(135deg,#2F6FED 0%,#1B49B8 100%);
                  display:flex; align-items:center; justify-content:center; flex-shrink:0; }
      .app-title { color:var(--text); font-size:16px; font-weight:700; letter-spacing:-.01em; }
      .app-subtitle { color:var(--dim); font-size:11.5px; margin-top:1px; }
      .pill { display:inline-flex; align-items:center; gap:6px; font-size:11px;
              font-weight:700; letter-spacing:.07em; padding:5px 11px; border-radius:20px;
              border:1px solid; white-space:nowrap; }
      .pill.dot::before { content:''; width:7px; height:7px; border-radius:50%;
                          background:currentColor; box-shadow:0 0 8px currentColor;
                          animation:pulse 1.8s infinite; }
      .pill-live { color:var(--blue); border-color:#21335c; background:#132038; }
      .pill-offline { color:var(--dim); border-color:var(--border2); background:var(--panel2); }
      .pill-offline::before { display:none; }
      .pill-calibrating { color:var(--amber); border-color:#5c4416; background:#241a08; }
      .rec-tag { font-size:10px; font-weight:800; letter-spacing:.08em; color:var(--red);
                 background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.4);
                 padding:4px 9px; border-radius:6px; }
      .sess-title { font-size:12.5px; letter-spacing:.03em; color:var(--sub); font-weight:600;
                    font-family:var(--mono); font-variant-numeric:tabular-nums;
                    background:var(--panel2); border:1px solid var(--border);
                    padding:5px 11px; border-radius:6px; }
      .chips { display:flex; gap:6px; margin-left:auto; align-items:center; flex-wrap:wrap; }
      .ctx-label { font-size:9.5px; color:var(--dim); letter-spacing:.07em;
                   text-transform:uppercase; margin-right:2px; }
      .chip { font-size:10.5px; font-weight:700; letter-spacing:.03em; padding:5px 10px;
              border-radius:6px; background:var(--panel2); border:1px solid var(--border);
              color:var(--sub); }
      .chip.on { color:#fff; border-color:var(--blue); background:#132038; }
      .cal-pill { font-size:11px; font-weight:700; letter-spacing:.04em; display:inline-flex;
                  align-items:center; gap:5px; padding:5px 10px; border-radius:20px; border:1px solid; }
      .cal-pill.ok { color:var(--green); border-color:#1e4430; background:#0f2417; }
      .cal-pill.warn { color:var(--amber); border-color:#5c4416; background:#241a08; }

      /* ---- calibration HUD ----
         The primary, always-visible progress readout. Rendered above the
         video columns so it can never end up below the fold - the ring
         inside the video panel is a secondary flourish, not the only way
         to see progress. */
      .calib-hud { background:rgba(47,111,237,.06); border:1px solid rgba(47,111,237,.35);
                   border-radius:10px; padding:12px 16px; margin-bottom:10px; }
      .chud-row { display:flex; align-items:center; justify-content:space-between;
                  gap:12px; margin-bottom:9px; }
      .chud-label { font-size:11px; font-weight:800; letter-spacing:.09em; color:var(--blue);
                    display:flex; align-items:center; gap:8px; white-space:nowrap; }
      .chud-dot { width:7px; height:7px; border-radius:50%; background:var(--blue);
                  flex-shrink:0; box-shadow:0 0 8px var(--blue); animation:pulse 1.2s infinite; }
      .chud-time { font-family:var(--mono); font-size:22px; font-weight:700; color:var(--text);
                   letter-spacing:.02em; font-variant-numeric:tabular-nums; }
      .chud-bar { display:flex; gap:3px; margin-bottom:9px; }
      .chud-bar i { flex:1; height:6px; border-radius:1px; background:#1C2733; display:block; }
      .chud-bar i.on { background:var(--blue); box-shadow:0 0 6px rgba(47,111,237,.55); }
      .chud-note { color:var(--sub); font-size:11.5px; line-height:1.5; }
      .chud-note .warn { color:var(--amber); font-weight:600; }

      /* ---- video panel ---- */
      /* Viewfinder-style corner brackets on the two "hero" panels (video
         feed, agent core) - selected via :has() on content that's already
         unconditionally present, no extra markup needed for the agent
         panel since .agent-wrap always renders there. The video panel has
         no such always-present class (its placeholders start empty), so
         it gets one dedicated invisible marker div - see the Python call
         site. Brackets sit 8px inset so the rounded panel border never
         clips them. */
      .hud-frame-marker, .panel-marker { width:0; height:0; overflow:hidden; display:block; }
      div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hud-frame-marker),
      div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .agent-wrap) {
        position:relative;
      }
      div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hud-frame-marker)::before,
      div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .agent-wrap)::before {
        content:''; position:absolute; inset:8px; pointer-events:none; z-index:6; opacity:.5;
        background:
          linear-gradient(to right, var(--blue) 2px, transparent 0) top left / 14px 2px no-repeat,
          linear-gradient(to bottom, var(--blue) 2px, transparent 0) top left / 2px 14px no-repeat,
          linear-gradient(to left, var(--blue) 2px, transparent 0) top right / 14px 2px no-repeat,
          linear-gradient(to bottom, var(--blue) 2px, transparent 0) top right / 2px 14px no-repeat,
          linear-gradient(to right, var(--blue) 2px, transparent 0) bottom left / 14px 2px no-repeat,
          linear-gradient(to top, var(--blue) 2px, transparent 0) bottom left / 2px 14px no-repeat,
          linear-gradient(to left, var(--blue) 2px, transparent 0) bottom right / 14px 2px no-repeat,
          linear-gradient(to top, var(--blue) 2px, transparent 0) bottom right / 2px 14px no-repeat;
      }
      /* Slow ambient scan sweep - video feed only, always running, very
         subtle so it never fights with reading the numbers. */
      div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .hud-frame-marker)::after {
        content:''; position:absolute; left:8px; right:8px; top:8px; height:2px;
        pointer-events:none; z-index:6; opacity:0;
        background:linear-gradient(90deg, transparent, var(--blue), transparent);
        animation:scan-sweep 5s linear infinite;
      }
      @keyframes scan-sweep {
        0%   { top:8px; opacity:0; }
        8%   { opacity:.4; }
        92%  { opacity:.4; }
        100% { top:calc(100% - 10px); opacity:0; }
      }

      .vid-topbar { display:flex; gap:8px; padding:10px 12px 0 12px; }
      .hr-chip { background:var(--panel2); border:1px solid var(--border); border-radius:7px;
                 padding:5px 10px; font-size:11px; color:var(--sub); }
      .hr-chip .v { font-family:var(--mono); font-weight:700; font-size:13px; color:var(--text); }
      .level-block { margin:10px 12px 12px 12px; background:var(--panel2);
                     border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
      .index-row { display:flex; align-items:baseline; gap:12px; margin-bottom:8px; }
      .index-num { font-family:var(--mono); font-size:32px; font-weight:700; letter-spacing:-.02em; }
      .level-tag { font-size:12.5px; font-weight:800; letter-spacing:.09em; }
      .lvl-bar { height:7px; border-radius:5px; background:#0d1117; overflow:hidden;
                 border:1px solid var(--border); }
      .lvl-bar>i { display:block; height:100%; }

      .idle-view, .calib-view { display:flex; flex-direction:column; align-items:center;
                                justify-content:center; gap:14px; text-align:center;
                                padding:36px 20px; min-height:260px; }
      .idle-view .ot { color:var(--sub); font-size:13px; margin-top:2px; }
      .idle-view .os { color:var(--dim); font-size:11.5px; }
      .calib-ring { width:104px; height:104px; border-radius:50%; position:relative;
                    display:flex; align-items:center; justify-content:center; }
      .calib-ring::before { content:''; position:absolute; inset:6px; border-radius:50%;
                            background:#0d1117; }
      .calib-ring span { position:relative; font-family:var(--mono); font-size:19px;
                         font-weight:700; color:var(--text); }
      .calib-instr-mini { color:var(--sub); font-size:11.5px; margin-top:2px;
                          font-family:var(--mono); font-variant-numeric:tabular-nums; }

      .face-status { font-size:12px; color:var(--amber); padding:8px 12px; }

      /* ---- signals ---- */
      .sig { display:flex; align-items:center; gap:10px; padding:7px 2px;
             border-bottom:1px solid var(--border); }
      .sig:last-child { border-bottom:none; }
      .code { font-family:var(--mono); font-size:9.5px; font-weight:800;
              background:var(--panel2); border:1px solid var(--border); border-radius:4px;
              padding:3px 5px; width:32px; text-align:center; flex-shrink:0; }
      .sig-mid { flex:1; min-width:0; }
      .sig-name { font-size:11.5px; color:var(--text); font-weight:500; }
      .sig-desc { font-size:9.5px; color:var(--dim); }
      .sig-bar { height:4px; border-radius:3px; background:var(--panel2); margin-top:4px; overflow:hidden; }
      .sig-bar>i { display:block; height:100%; }
      .sig-val { font-family:var(--mono); font-size:12px; width:32px; text-align:right; flex-shrink:0; }

      /* ---- metrics ---- */
      .mtiles { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }
      .mtile { background:var(--panel2); border:1px solid var(--border); border-radius:8px;
               padding:10px; text-align:center; }
      .mtile .v { font-family:var(--mono); font-size:19px; font-weight:700; color:var(--text); }
      .mtile .l { font-size:9px; color:var(--dim); text-transform:uppercase;
                  letter-spacing:.06em; margin-top:3px; }

      /* ---- guidance ---- */
      .glist { font-size:11.5px; color:var(--sub); line-height:1.75; }
      .glist span { display:block; padding-left:14px; position:relative; }
      .glist span::before { content:'\\2013'; position:absolute; left:0; color:var(--orange); }
      .glist-sub { font-size:11px; font-weight:700; margin-bottom:6px; }

      /* ---- sidebar: session list ---- */
      .sess-row { display:flex; align-items:center; gap:8px; padding:8px 2px;
                  border-bottom:1px solid var(--border); font-size:11px; }
      .sess-row:last-child { border-bottom:none; }
      .sdot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
      .sinfo { flex:1; min-width:0; }
      .sinfo .t { color:var(--text); font-weight:600; }
      .sinfo .m { color:var(--dim); font-size:10px; margin-top:1px; }
      .tag { font-family:var(--mono); font-size:9px; font-weight:800; letter-spacing:.05em;
             background:var(--panel2); border:1px solid var(--border2); padding:2px 5px;
             border-radius:4px; color:var(--sub); vertical-align:1px; }
      .tag.on { color:var(--blue); border-color:#21335c; }
      .cal-check { display:inline-flex; align-items:center; gap:6px; font-size:11px;
                   font-weight:700; padding:5px 10px; border-radius:20px; margin-bottom:10px; }
      .cal-check.ok { color:var(--green); background:#0f2417; border:1px solid #1e4430; }
      .cal-check.warn { color:var(--amber); background:#241a08; border:1px solid #5c4416; }

      /* ---- agent panel ---- */
      .agent-wrap { padding:2px; }
      .agent-top { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
      .agent-name { font-weight:800; font-size:11.5px; letter-spacing:.04em; color:var(--text); }
      .state-pill { margin-left:auto; font-size:9.5px; font-weight:800; letter-spacing:.06em;
                    padding:3px 9px; border-radius:12px; border:1px solid; }
      .state-pill.active { color:var(--green); border-color:#1e4430; background:#0f2417; }
      .state-pill.standby { color:var(--dim); border-color:var(--border2); background:var(--panel2); }
      .agent-desc { font-size:10.5px; color:var(--sub); line-height:1.45; margin-bottom:10px; }

      /* Big circular power core. The marker div is invisible (0x0); it only
         exists so :has() can tell whether the button that follows it should
         render "on" (blue, spinning) or "off" (grey, still). See the Python
         call site for why this indirection is needed - Streamlit gives no
         other way to attach a per-instance CSS class to a real st.button. */
      .agent-core-marker { width:0; height:0; overflow:hidden; display:block; }
      div:has(.agent-core-marker) + div .stButton { display:flex; justify-content:center;
                                                    margin:8px 0 4px 0; }
      div:has(.agent-core-marker) + div .stButton button {
        position:relative; width:108px !important; height:108px !important;
        border-radius:50% !important; background:#0d1117 !important; border:none !important;
        color:#fff !important; font-weight:800 !important; font-size:12.5px !important;
        letter-spacing:.09em !important; padding:0 !important; z-index:1;
        transition:none !important;
      }
      div:has(.agent-core-marker) + div .stButton button::before {
        content:''; position:absolute; inset:-5px; border-radius:50%; z-index:-1;
      }
      div:has(.agent-core-marker.on) + div .stButton button::before {
        background:conic-gradient(from 0deg,var(--blue),#0d1117 40%,var(--blue) 80%,#0d1117);
        animation:spin 3s linear infinite;
      }
      div:has(.agent-core-marker.off) + div .stButton button::before {
        background:#1C2733; animation:none;
      }
      div:has(.agent-core-marker.off) + div .stButton button { color:var(--dim) !important; }
      div:has(.agent-core-marker) + div .stButton button:hover { border-color:transparent !important; }
      .agent-core-caption { text-align:center; font-size:10px; color:var(--dim);
                            line-height:1.5; margin:2px 0 10px 0; }
      .agent-core-caption b { color:var(--sub); }
      @keyframes spin { to { transform:rotate(360deg); } }
      @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }

      .mem { font-size:10px; color:var(--text); background:var(--panel2);
             border:1px solid var(--border); border-radius:6px; padding:6px 9px; margin-bottom:8px; }
      .mem b { color:var(--orange); }
      .phases { display:flex; gap:4px; flex-wrap:wrap; margin-bottom:6px; }
      .phase { font-size:8.5px; letter-spacing:.04em; font-weight:700; padding:4px 6px;
               border-radius:5px; background:var(--panel2); border:1px solid var(--border);
               color:var(--dim); text-transform:uppercase; }
      .phase.cur { background:#3a2312; border-color:var(--orange); color:var(--orange); }
      .phase-reason { font-size:10px; color:var(--sub); }

      .intervention-card { background:rgba(245,118,43,.08); border:1px solid rgba(245,118,43,.45);
                           border-radius:10px; padding:12px 16px; margin:0 0 10px 0; }
      .intervention-card .tag-lbl { color:var(--orange); font-size:10px; font-weight:800;
                                    letter-spacing:.09em; margin-bottom:3px; }
      .intervention-card .ttl { color:var(--text); font-weight:700; font-size:14px; }
      .intervention-card .body { color:var(--sub); font-size:12px; margin-top:4px; }

      .del-btn button { color:#F87171 !important; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# MODEL / ENGINE LOADING
# =============================================================================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load only the active live-system models (DaiSEE ensemble)."""
    def try_load(path):
        if not Path(path).exists():
            return None
        try:
            return tf.keras.models.load_model(path)
        except Exception:
            return None
    return {
        'daisee_mb':  try_load(DAISEE_MOBILENET_PATH),
        'daisee_eff': try_load(DAISEE_EFFICIENTNET_PATH),
    }


@st.cache_resource
def get_feature_engine():
    return BehavioralFeatureEngine()


@st.cache_resource(show_spinner="Loading temporal model (if available)...")
def get_temporal_inference():
    """Returns GRUTemporalInference - safe even if the model is unavailable
    or fails self-verification. is_available() returns False in that case
    and fusion ignores it."""
    return GRUTemporalInference()


# =============================================================================
# VISION HELPERS
# =============================================================================
def crop_face(frame_bgr, face_mesh_results):
    if not face_mesh_results.multi_face_landmarks:
        return None, None
    h, w = frame_bgr.shape[:2]
    lm = face_mesh_results.multi_face_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad = int(config.FACE_CROP_PADDING * max(x2 - x1, y2 - y1))
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None, None
    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


def prep(face_bgr, size, preprocess_fn):
    img = cv2.resize(face_bgr, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')
    img = preprocess_fn(img)
    return np.expand_dims(img, axis=0)


def ensemble_predict(face_bgr, model_pairs):
    preds = []
    for model, size, preproc in model_pairs:
        if model is None:
            continue
        p = model.predict(prep(face_bgr, size, preproc), verbose=0)[0]
        preds.append(p)
    if not preds:
        return None
    return np.mean(preds, axis=0)


def annotate_frame(frame_bgr, bbox, level, smoothed_index, calibrating=False):
    if calibrating:
        color = (237, 158, 47); label = "CALIBRATING"     # amber, BGR
    else:
        color = {'LOW': (34, 197, 34), 'MODERATE': (0, 158, 245),
                 'HIGH': (68, 68, 239)}[level]              # BGR of green/amber/red
        label = f"{level} ({smoothed_index*100:.0f}%)"
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        corner = 14
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y2, -1, -1)):
            cv2.line(frame_bgr, (cx, cy), (cx + dx * corner, cy), color, 3)
            cv2.line(frame_bgr, (cx, cy), (cx, cy + dy * corner), color, 3)
    cv2.putText(frame_bgr, label, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# =============================================================================
# RENDER HELPERS
# =============================================================================
LEVEL_COLORS = {'LOW': '#22C55E', 'MODERATE': '#F59E0B', 'HIGH': '#EF4444'}


def render_mini_graph(values, height=110, width=250):
    if len(values) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" '
            f'style="width:100%;height:{height}px;background:#141B23;border-radius:8px;">'
            f'<text x="{width/2}" y="{height/2}" font-size="11" fill="#5B6772" '
            f'text-anchor="middle">waiting for data...</text></svg>'
        )
    pad = 8
    n = len(values)
    def x_of(i): return pad + (i / max(n - 1, 1)) * (width - 2 * pad)
    def y_of(v): return height - pad - max(0, min(1, v)) * (height - 2 * pad)
    pts = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values))
    area_pts = (f"{pad},{height-pad} " + pts + f" {x_of(n-1):.1f},{height-pad}")
    return (
        f'<svg viewBox="0 0 {width} {height}" '
        f'style="width:100%;height:{height}px;background:#141B23;border-radius:8px;">'
        f'<polygon points="{area_pts}" fill="rgba(245,118,43,0.15)"/>'
        f'<polyline points="{pts}" fill="none" stroke="#F5762B" stroke-width="2"/>'
        f'</svg>'
    )


def render_full_timeline(history, height=150):
    """3-line timeline: stress_index, daisee, behavior."""
    if len(history) < 2:
        return ""
    width = 1000
    pad = 26
    n = len(history)
    def x_of(i): return pad + (i / max(n - 1, 1)) * (width - 2 * pad)
    def y_of(v): return height - pad - v * (height - 2 * pad)
    series = {
        'Stress index': '#F5762B',
        'CNN branch':   '#2F6FED',
        'Behavioral branch': '#7C8B99',
    }
    keymap = {'Stress index': 'stress_index', 'CNN branch': 'daisee',
              'Behavioral branch': 'behavior'}
    paths = []
    for label, color in series.items():
        key = keymap[label]
        pts = " ".join(f"{x_of(i):.1f},{y_of(rec[key]):.1f}"
                       for i, rec in enumerate(history))
        width_px = "2.5" if key == 'stress_index' else "1.6"
        paths.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="{width_px}"/>')
    gridlines = "".join(
        f'<line x1="0" y1="{height*f:.0f}" x2="{width}" y2="{height*f:.0f}" '
        f'stroke="#1C2733"/>' for f in (0.25, 0.5, 0.75))
    return f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" ' \
           f'style="width:100%;height:{height}px;display:block;">{gridlines}{"".join(paths)}</svg>'


TIMELINE_LEGEND = (
    '<div style="display:flex;gap:16px;font-size:10.5px;color:#8B98A5;margin-bottom:6px;">'
    '<span><i style="display:inline-block;width:10px;height:2px;background:#F5762B;'
    'margin-right:5px;vertical-align:middle;"></i>Stress index</span>'
    '<span><i style="display:inline-block;width:10px;height:2px;background:#2F6FED;'
    'margin-right:5px;vertical-align:middle;"></i>CNN branch</span>'
    '<span><i style="display:inline-block;width:10px;height:2px;background:#7C8B99;'
    'margin-right:5px;vertical-align:middle;"></i>Behavioral branch</span>'
    '</div>'
)


SIGNAL_DISPLAY = {
    'jaw_tension':      ("JAW", "Jaw tension",   "Clench / grimace"),
    'brow_tension':     ("BRW", "Brow tension",  "Furrow / contraction"),
    'slouch':           ("PST", "Posture",       "Spinal deviation"),
    'eye_strain':       ("EYE", "Eye strain",    "Openness + squint"),
    'neck_strain':      ("NCK", "Neck strain",   "Forward head position"),
    'blink_stress':     ("BLK", "Blink rate",    "Deviation from baseline"),
    'head_instability': ("MOV", "Head movement", "Fidget / restlessness"),
}


def _severity_color(value):
    if value >= 0.6:  return '#EF4444'
    if value >= 0.4:  return '#F5762B'
    if value >= 0.2:  return '#2F6FED'
    return '#3a4552'


def render_signal_rows(contributions):
    rows = []
    for name, value in sorted(contributions.items(), key=lambda kv: kv[1], reverse=True):
        code, label, detail = SIGNAL_DISPLAY.get(name, (name[:3].upper(), name, ""))
        color = _severity_color(value)
        pct = max(0, min(100, value * 100))
        rows.append(
            f'<div class="sig">'
            f'<span class="code" style="color:{color};border-color:{color}55;">{code}</span>'
            f'<div class="sig-mid"><div class="sig-name">{label}</div>'
            f'<div class="sig-desc">{detail}</div>'
            f'<div class="sig-bar"><i style="width:{pct:.0f}%;background:{color};"></i></div></div>'
            f'<span class="sig-val mono" style="color:{color};">{value:.2f}</span>'
            f'</div>'
        )
    return ''.join(rows)


def render_tiles(blink_rate, ear_value, hr_text="--", perclos_pct=None):
    """
    perclos_pct: 0-100, or None to omit the tile entirely (only Driving mode
    passes a value - PERCLOS is a drowsiness metric, not generally useful
    outside that context; see config.PERCLOS_CLOSURE_RATIO).
    """
    tiles = [
        f'<div class="mtile"><div class="v">{blink_rate:.0f}</div><div class="l">Blinks/min</div></div>',
        f'<div class="mtile"><div class="v">{ear_value:.2f}</div><div class="l">Eye open</div></div>',
        f'<div class="mtile"><div class="v">{hr_text}</div><div class="l">HR bpm</div></div>',
    ]
    if perclos_pct is not None:
        tiles.append(
            f'<div class="mtile"><div class="v">{perclos_pct:.0f}%</div><div class="l">PERCLOS</div></div>'
        )
    return (
        f'<div class="mtiles" style="grid-template-columns:repeat({len(tiles)},1fr);">'
        + ''.join(tiles) +
        f'</div>'
    )


def render_guidance(level, contributions=None):
    """
    When one signal clearly dominates, lead with the SAME specific advice the
    intervention agent would give ("unclench your jaw", not "reduce stress")
    - reusing intervention_agent.CATALOGUE rather than maintaining a second,
    vaguer copy of the same knowledge. Falls back to the generic per-level
    tips when nothing stands out, so idle/low-signal states still show
    something useful.
    """
    level_labels = {
        'LOW':      ('Stay relaxed', '#22C55E'),
        'MODERATE': ('Reduce stress', '#F59E0B'),
        'HIGH':     ('Act now', '#EF4444'),
    }
    sub_label, sub_color = level_labels[level]

    lead_html = ''
    if contributions:
        top_signal, top_value = max(contributions.items(), key=lambda kv: kv[1])
        if top_value >= config.INTERVENTION_SIGNAL_THRESHOLD and top_signal in CATALOGUE:
            entry = CATALOGUE[top_signal]
            lead_html = (
                f'<div style="background:rgba(245,118,43,.06);border:1px solid '
                f'rgba(245,118,43,.3);border-radius:8px;padding:8px 10px;margin-bottom:8px;">'
                f'<div style="color:#F5762B;font-size:9.5px;font-weight:800;'
                f'letter-spacing:.07em;margin-bottom:2px;">DRIVEN BY {top_signal.replace("_"," ").upper()}</div>'
                f'<div style="color:#E6EDF3;font-weight:700;font-size:12.5px;">{entry["title"]}</div>'
                f'<div style="color:#8B98A5;font-size:11px;margin-top:2px;">{entry["body"]}</div>'
                f'</div>'
            )

    items = ''.join(f'<span>{tip}</span>' for tip in GUIDANCE_TIPS[level])
    return (
        f'{lead_html}'
        f'<div class="glist-sub" style="color:{sub_color};">{sub_label}</div>'
        f'<div class="glist">{items}</div>'
    )


def render_video_topbar(hr_text, ear_value):
    return (
        f'<div class="vid-topbar">'
        f'<span class="hr-chip">HR <span class="v mono">{hr_text}</span> bpm</span>'
        f'<span class="hr-chip">Eye open <span class="v mono">{ear_value:.2f}</span></span>'
        f'</div>'
    )


def render_level_block(level, smoothed):
    color = LEVEL_COLORS[level]
    pct = max(0.0, min(1.0, smoothed)) * 100
    return (
        f'<div class="level-block">'
        f'<div class="index-row"><span class="index-num mono">{smoothed:.2f}</span>'
        f'<span class="level-tag" style="color:{color};">{level}</span></div>'
        f'<div class="lvl-bar"><i style="width:{pct:.0f}%;'
        f'background:linear-gradient(90deg,#22C55E,{color});"></i></div>'
        f'</div>'
    )


def render_calib_ring(progress, seconds_left):
    deg = max(0, min(100, progress * 100))
    return (
        f'<div class="calib-view">'
        f'<div class="calib-ring" style="background:conic-gradient(#2F6FED {deg:.0f}%, #1C2733 0);">'
        f'<span>{deg:.0f}%</span></div>'
        f'<div class="calib-instr-mini">Hold steady &mdash; {seconds_left:.0f}s remaining</div>'
        f'</div>'
    )


def render_calib_hud(progress, seconds_left):
    """
    The primary calibration readout - see the .calib-hud CSS comment for
    why this exists as a separate, always-above-the-fold element rather
    than relying on the ring inside the video panel.
    """
    n_segments = 28
    filled = int(round(n_segments * max(0.0, min(1.0, progress))))
    segs = ''.join(
        f'<i class="{"on" if i < filled else ""}"></i>' for i in range(n_segments)
    )
    mins, secs = divmod(max(0, int(round(seconds_left))), 60)
    return (
        f'<div class="calib-hud">'
        f'<div class="chud-row">'
        f'<div class="chud-label"><span class="chud-dot"></span>PERSONAL BASELINE ACQUISITION</div>'
        f'<div class="chud-time mono">{mins:02d}:{secs:02d}</div>'
        f'</div>'
        f'<div class="chud-bar">{segs}</div>'
        f'<div class="chud-note">Sit naturally, neutral expression, look at the camera &mdash; '
        f'<span class="warn">do not smile</span>: a smiling baseline skews the jaw and brow '
        f'signals for the whole session.</div>'
        f'</div>'
    )


IDLE_SVG = (
    '<svg viewBox="0 0 24 24" width="42" height="42" fill="none" stroke="#5B6772" '
    'stroke-width="1.6"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" '
    'height="14" rx="2"/></svg>'
)


def render_idle_view():
    return (
        f'<div class="idle-view">{IDLE_SVG}'
        f'<div class="ot">Camera off</div>'
        f'<div class="os">Press Calibrate, then Start, to begin a session</div></div>'
    )


def render_agent_core_caption(active):
    sub = "Monitoring all seven signals" if active else "Interventions and learning paused"
    tap = "Tap the core to power down" if active else "Tap the core to power up"
    return f'<div class="agent-core-caption"><b>{sub}</b><br>{tap}</div>'


def render_agent_phases(current_phase, reason):
    chips = ''.join(
        f'<span class="phase{" cur" if p == current_phase else ""}">{p}</span>'
        for p in AGENT_PHASES
    )
    return f'<div class="phases">{chips}</div><div class="phase-reason">{reason}</div>'


def render_intervention_card(title, body):
    return (
        f'<div class="intervention-card">'
        f'<div class="tag-lbl">AI INTERVENTION</div>'
        f'<div class="ttl">{title}</div>'
        f'<div class="body">{body}</div></div>'
    )


def history_to_csv(history):
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=['ts', 'stress_index', 'level', 'daisee', 'emotion', 'behavior'],
    )
    writer.writeheader()
    for rec in history:
        writer.writerow({
            'ts':           rec['ts'].isoformat(),
            'stress_index': f"{rec['stress_index']:.4f}",
            'level':        rec['level'],
            'daisee':       f"{rec['daisee']:.4f}",
            'emotion':      "0.0000",   # column kept for backward-compat with analyzer
            'behavior':     f"{rec['behavior']:.4f}",
        })
    return buf.getvalue()


def init_state():
    if 'running' not in st.session_state: st.session_state.running = False
    if 'calibrating' not in st.session_state: st.session_state.calibrating = False
    if 'history' not in st.session_state: st.session_state.history = []
    if 'session_start' not in st.session_state: st.session_state.session_start = None
    if 'viewing_session_id' not in st.session_state: st.session_state.viewing_session_id = None
    if 'viewing_session_label' not in st.session_state: st.session_state.viewing_session_label = None
    if 'task_mode' not in st.session_state: st.session_state.task_mode = 'Default'
    # A header context button sets this on the rerun it triggers, then this
    # block (which runs before the sidebar "Context" selectbox is
    # instantiated) applies it to both the real state and the selectbox's own
    # widget key. Setting task_select directly from the button handler would
    # raise StreamlitAPIException because that widget has already been
    # instantiated earlier in that same script run.
    if 'pending_task_mode' in st.session_state:
        _pending = st.session_state.pop('pending_task_mode')
        st.session_state.task_mode = _pending
        st.session_state.task_select = _pending
    if 'active_user_id' not in st.session_state: st.session_state.active_user_id = None
    if 'active_user_name' not in st.session_state: st.session_state.active_user_name = None
    # Same pattern as pending_task_mode above: "Add user" sets this on the
    # rerun it triggers (rather than the user_select widget key directly,
    # which raises StreamlitAPIException from inside that same script run
    # since the widget is instantiated later), and this block - which runs
    # before the sidebar "User" selectbox - applies it to both the real
    # state and the widget's own key.
    if 'pending_user_selection' in st.session_state:
        _pending_user = st.session_state.pop('pending_user_selection')
        st.session_state.active_user_id = _pending_user['id']
        st.session_state.active_user_name = _pending_user['name']
        st.session_state.user_select = _pending_user['name'] or '(anonymous)'
    if 'subject_id' not in st.session_state: st.session_state.subject_id = ''
    if 'condition' not in st.session_state: st.session_state.condition = 'unlabeled'
    if 'pending_self_report' not in st.session_state: st.session_state.pending_self_report = None
    if 'agent_enabled' not in st.session_state: st.session_state.agent_enabled = True
    if 'session_count' not in st.session_state: st.session_state.session_count = 0
    if 'db_initialized' not in st.session_state:
        session_history.init_db()
        st.session_state.db_initialized = True


def save_current_session():
    """
    Persist st.session_state.history to the DB and queue the self-report
    prompt. Shared by the Stop button and the mid-session camera-failure
    path, so a session ending abnormally is saved exactly like one ending
    normally rather than silently dropping its data.
    """
    if not st.session_state.history:
        return
    try:
        sid = session_history.save_session(
            st.session_state.history, task_mode=st.session_state.task_mode,
            user_id=st.session_state.active_user_id,
        )
        # Queue the self-report prompt for this session. The user's answer -
        # not the system's prediction - is the ground-truth label the
        # adaptation loop trains on.
        st.session_state.pending_self_report = sid
        ivlog = st.session_state.pop('pending_interventions', [])
        if ivlog:
            session_history.log_interventions(sid, ivlog)
    except Exception as e:
        st.warning(f"Could not save session to history DB: {e}")


def apply_user_profile(engine, user_id):
    """
    Swap the live (process-wide singleton) engine's baseline for the given
    user's saved one, or back to generic defaults if they have none yet /
    user_id is None (anonymous). This is the same mechanism the "Reset
    baseline" button already used - engine.baseline is just a dataclass,
    swapping it is cheap and safe between sessions (never call this while
    running/calibrating, same guard every other baseline-touching control
    in the sidebar already has).
    """
    saved = session_history.load_user_baseline(user_id) if user_id is not None else None
    if saved is not None:
        engine.baseline = BehavioralBaseline(**saved['behavioral'])
        if saved.get('daisee_baseline') is not None:
            set_daisee_baseline(saved['daisee_baseline'])
        else:
            reset_daisee_baseline()
    else:
        engine.reset_baseline()
        reset_daisee_baseline()


def save_active_user_baseline(engine):
    """Persist the engine's current baseline to the active user's profile,
    called right after a completed calibration. No-op for anonymous use
    (active_user_id is None) - unchanged, single-shared-baseline behavior."""
    user_id = st.session_state.active_user_id
    if user_id is None:
        return
    session_history.save_user_baseline(user_id, {
        'behavioral': dataclasses.asdict(engine.baseline),
        'daisee_baseline': get_daisee_baseline(),
    })


# =============================================================================
# MAIN
# =============================================================================
def main():
    init_state()
    inject_css()

    models = load_models()
    engine = get_feature_engine()
    temporal_inf = get_temporal_inference()

    is_fully_calibrated = engine.baseline.calibrated and is_daisee_calibrated()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown('<div class="label">Live stress</div>', unsafe_allow_html=True)
        sidebar_graph_placeholder = st.empty()
        sidebar_graph_placeholder.markdown(render_mini_graph([]), unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="label">Session control</div>', unsafe_allow_html=True)

        # ---------- User profile ----------
        # Phase 1 prerequisite for team recordings: each person gets their
        # own persisted calibration baseline instead of everyone sharing
        # the single process-wide one. Switching users loads that person's
        # saved baseline (or generic defaults for a brand-new profile) into
        # the live engine - the same swap "Reset baseline" already did.
        _users = session_history.list_users()
        _user_names = [u['name'] for u in _users]
        _user_options = ['(anonymous)'] + _user_names
        _current_user_name = st.session_state.active_user_name or '(anonymous)'
        _current_user_idx = (_user_options.index(_current_user_name)
                             if _current_user_name in _user_options else 0)
        _selected_user_name = st.selectbox(
            "User", options=_user_options, index=_current_user_idx,
            disabled=st.session_state.running or st.session_state.calibrating,
            help="Each user gets their own saved calibration baseline. "
                 "Switching loads that person's baseline (generic defaults "
                 "for a brand-new profile) - Calibrate afterward to refine "
                 "or replace it.",
            key="user_select",
        )
        if _selected_user_name != _current_user_name:
            if _selected_user_name == '(anonymous)':
                st.session_state.active_user_id = None
                st.session_state.active_user_name = None
            else:
                _match = next(u for u in _users if u['name'] == _selected_user_name)
                st.session_state.active_user_id = _match['id']
                st.session_state.active_user_name = _match['name']
            apply_user_profile(engine, st.session_state.active_user_id)
            st.rerun()

        # Deliberately NOT an st.expander: expanders reset to their
        # hardcoded `expanded=` value on every script rerun, and typing in
        # the text field below triggers one - the field would visually
        # collapse out from under the user before they could reach the
        # button. Found live while verifying this feature, not theoretical.
        _new_user_col, _add_user_col = st.columns([2, 1])
        with _new_user_col:
            _new_user_name = st.text_input(
                "New user", key="new_user_name", placeholder="e.g. Arjun",
                label_visibility="collapsed",
                disabled=st.session_state.running or st.session_state.calibrating,
            )
        with _add_user_col:
            if st.button("Add user", width='stretch',
                        disabled=st.session_state.running or st.session_state.calibrating):
                try:
                    new_id = session_history.create_user(_new_user_name)
                    apply_user_profile(engine, new_id)
                    st.session_state.pending_user_selection = {
                        'id': new_id, 'name': _new_user_name.strip(),
                    }
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        if is_fully_calibrated:
            st.markdown('<span class="cal-check ok">&#10003; CALIBRATED</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="cal-check warn">&#9679; NOT CALIBRATED</span>',
                        unsafe_allow_html=True)

        # ---------- Context selector ----------
        # Reweights behavioral signals to emphasize what matters most in
        # the chosen context. 'Default' preserves the un-modal baseline.
        _current_idx = (
            VALID_TASK_MODES.index(st.session_state.task_mode)
            if st.session_state.task_mode in VALID_TASK_MODES else 0
        )
        _selected_task = st.selectbox(
            "Context",
            options=VALID_TASK_MODES,
            format_func=lambda x: TASK_LABELS.get(x, (x, x))[0],
            index=_current_idx,
            disabled=st.session_state.running or st.session_state.calibrating,
            help="Reweights stress signals for the chosen context. 'Default' "
                 "is the un-modal baseline. Other modes emphasize signals "
                 "most diagnostic of stress in that context.",
            key="task_select",
        )
        if _selected_task != st.session_state.task_mode:
            st.session_state.task_mode = _selected_task

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Calibrate", width='stretch',
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help=f"Sit naturally for {CALIBRATION_SECONDS} seconds. "
                              f"System learns your personal baseline."):
                st.session_state.calibrating = True
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.rerun()
        with col_b:
            if st.button("Start", type="primary", width='stretch',
                         disabled=st.session_state.running or st.session_state.calibrating):
                st.session_state.running = True
                st.session_state.session_start = datetime.now()
                st.session_state.history = []
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.session_state.session_count += 1
                st.rerun()

        if st.button("Stop", width='stretch',
                     disabled=not (st.session_state.running or st.session_state.calibrating)):
            st.session_state.running = False
            st.session_state.calibrating = False
            engine._calibrating = False
            save_current_session()
            st.rerun()

        col_c, col_d = st.columns(2)
        with col_c:
            if st.button("Clear view", width='stretch',
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help="Clears the on-screen timeline. Saved sessions in the DB are not affected."):
                st.session_state.history = []
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.rerun()
        with col_d:
            if st.button("Reset baseline", width='stretch',
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help="Wipes the personal baseline back to generic defaults. "
                              "Stop first - resetting mid-session would silently "
                              "degrade the rest of that session's scoring."):
                engine.reset_baseline()
                reset_daisee_baseline()
                st.rerun()

        # ---------- Recording labels ----------
        # Set BEFORE recording; these are baked into the export filename so
        # provenance cannot be lost afterwards. See RECORDING_PROTOCOL.md.
        with st.expander("Recording labels", expanded=False):
            st.text_input(
                "Subject ID", key="subject_id", placeholder="S01",
                disabled=st.session_state.running or st.session_state.calibrating,
                help="Stable anonymous ID, e.g. S01. Never a real name.",
            )
            st.selectbox(
                "Condition", options=['unlabeled', 'relaxed', 'stressed'],
                key="condition",
                disabled=st.session_state.running or st.session_state.calibrating,
                help="Ground-truth label for this recording. Relaxed should "
                     "read LOW; stressed should read MODERATE or HIGH.",
            )
            _preview = export_filename(
                st.session_state.subject_id, st.session_state.condition,
                st.session_state.task_mode,
            )
            st.caption(f"Export filename: `{_preview}`")
            if _preview.startswith("stress_session_"):
                st.caption("Unlabelled — set both fields or this recording "
                           "will not be picked up by the evaluation harness.")

        # ---------- AI Intervention Agent ----------
        # A real switch: when off, the agent's decision loop does not run at
        # all. Disabled mid-session so the capture loop is not interrupted.
        st.divider()
        _on = st.session_state.agent_enabled
        try:
            _mem = InterventionAgent(stats=session_history.intervention_stats()).learned_summary()
        except Exception:
            _mem = "learning memory unavailable"
        with st.container(border=True):
            st.markdown(
                f'<div class="agent-wrap">'
                f'<div class="agent-top"><span class="agent-name">AI AGENT &mdash; ADAPTIVE COACH</span>'
                f'<span class="state-pill {"active" if _on else "standby"}">'
                f'{"ACTIVE" if _on else "STANDBY"}</span></div>'
                f'<div class="agent-desc">Decides when an interruption is worth it, targets the '
                f'signal driving the stress, then measures whether it helped and learns what '
                f'works for you. Runs entirely on this device.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # The big circle IS the toggle. A marker element immediately
            # precedes the real button so CSS can style that specific button
            # as a large spinning core (blue + rotating = on, grey + still =
            # off) without touching every other button on the page.
            st.markdown(f'<div class="agent-core-marker {"on" if _on else "off"}"></div>',
                       unsafe_allow_html=True)
            if st.button("ACTIVE" if _on else "STANDBY", key="agent_core_btn",
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help="Click to power the agent down" if _on else
                              "Click to power the agent up"):
                st.session_state.agent_enabled = not _on
                st.rerun()
            st.markdown(render_agent_core_caption(_on) + f'<div class="mem">{_mem}</div>',
                       unsafe_allow_html=True)

        # ---------- Past Sessions ----------
        st.divider()
        st.markdown('<div class="label">Past sessions</div>', unsafe_allow_html=True)
        try:
            recent = session_history.load_recent_sessions(n=10)
            total  = session_history.session_count()
        except Exception as e:
            recent, total = [], 0
            st.caption(f"Could not read history DB: {e}")

        if total == 0:
            st.caption("No saved sessions yet. Stop a session to save it.")
        else:
            st.caption(f"Showing last {len(recent)} of {total} sessions")
            for s in recent:
                start_label = s['started_at'][:16].replace('T', '  ')
                pct_h, pct_m, pct_l = s['pct_high'], s['pct_moderate'], s['pct_low']
                if pct_h >= 30:   dot_color = '#EF4444'
                elif pct_m >= 40: dot_color = '#F59E0B'
                else:             dot_color = '#22C55E'
                _task = s.get('task_mode', 'Default') or 'Default'
                _tag = TASK_LABELS.get(_task, (_task, 'GEN'))[1]

                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    is_active = (st.session_state.viewing_session_id == s['id'])
                    active_tag = ' <span class="tag on">VIEWING</span>' if is_active else ''
                    st.markdown(
                        f'<div class="sess-row">'
                        f'<span class="sdot" style="background:{dot_color}"></span>'
                        f'<div class="sinfo"><div class="t">{start_label} '
                        f'<span class="tag">{_tag}</span>{active_tag}</div>'
                        f'<div class="m">{s["duration_sec"]:.0f}s &middot; avg {s["mean_stress"]:.2f} '
                        f'&middot; L/M/H {pct_l:.0f}/{pct_m:.0f}/{pct_h:.0f}%</div></div></div>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("View", key=f"view_{s['id']}",
                                 disabled=st.session_state.running or st.session_state.calibrating,
                                 help="Load this session's timeline into the main view"):
                        samples = session_history.load_session_samples(s['id'])
                        if samples:
                            st.session_state.history = samples
                            st.session_state.viewing_session_id = s['id']
                            st.session_state.viewing_session_label = start_label
                        else:
                            st.session_state.empty_session_warning = (
                                f"Session from {start_label} was saved before per-frame "
                                f"recording was supported. Only the summary stats above "
                                f"are available."
                            )
                        st.rerun()

            st.markdown('<div class="del-btn">', unsafe_allow_html=True)
            if st.button("Delete all history", width='stretch', key="clear_db",
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help="Deletes every saved session from the local DB. Cannot be undone."):
                session_history.clear_all_sessions()
                st.session_state.history = []
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Header ----------
    if st.session_state.calibrating:
        pill_class, pill_label = "pill dot pill-calibrating", "CALIBRATING"
    elif st.session_state.running:
        pill_class, pill_label = "pill dot pill-live", "LIVE"
    else:
        pill_class, pill_label = "pill pill-offline", "OFFLINE"

    if st.session_state.running:
        sess_text = f"SESSION {st.session_state.session_count:02d} &middot; running"
    elif st.session_state.calibrating:
        sess_text = "Calibrating&hellip;"
    elif st.session_state.viewing_session_id:
        sess_text = f"Viewing {st.session_state.viewing_session_label}"
    else:
        sess_text = "No active session"

    _cal_html = ('<span class="cal-pill ok">&#10003; CALIBRATED</span>' if is_fully_calibrated
                else '<span class="cal-pill warn">&#9679; NOT CALIBRATED</span>')

    header_placeholder = st.empty()

    def render_header(session_text):
        rec = '<span class="rec-tag">&#9679; REC</span>' if st.session_state.running else ''
        header_placeholder.markdown(
            f'<div class="app-header">'
            f'<div class="app-icon"><svg viewBox="0 0 24 24" width="18" height="18" fill="none" '
            f'stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
            f'<polyline points="2 12 6 12 9 4 14 20 17 12 22 12"/></svg></div>'
            f'<div><div class="app-title">Vision-Based Behavioral Stress Detection</div>'
            f'<div class="app-subtitle">Camera only &middot; on-device &middot; no wearables</div></div>'
            f'<span class="{pill_class}">{pill_label}</span>{rec}'
            f'<span class="sess-title">{session_text}</span>'
            f'</div>', unsafe_allow_html=True)

    render_header(sess_text)

    # ---------- Context switcher (REAL buttons, not decorative HTML) ----------
    # This used to be a row of <span> chips rendered via st.markdown, which
    # cannot fire a Python callback - Streamlit has no way to route a click on
    # raw HTML back into the script. These are actual st.button widgets in the
    # same visual position, so clicking them actually changes task_mode.
    _ctx_row = st.columns([1, 1, 1, 1, 2.2])
    for _i, (_k, (_label, _tag)) in enumerate(TASK_LABELS.items()):
        _is_current = (_k == st.session_state.task_mode)
        if _ctx_row[_i].button(
            _label, key=f"ctxbtn_{_k}", width='stretch',
            type="primary" if _is_current else "secondary",
            disabled=st.session_state.running or st.session_state.calibrating,
            help=f"Switch to {_label} context",
        ):
            # Applied in init_state() on the next rerun, before the sidebar
            # selectbox is instantiated - see comment there.
            st.session_state.pending_task_mode = _k
            st.rerun()
    with _ctx_row[4]:
        st.markdown(f'<div style="text-align:right;padding-top:6px;">{_cal_html}</div>',
                   unsafe_allow_html=True)

    # Lives directly under the context row, above the video columns, so it
    # is never scrolled out of view - see the .calib-hud CSS comment.
    # Streamlit does not clear a placeholder just because a later rerun
    # skips writing to it - whatever was last displayed here persists
    # until something explicitly overwrites it - so the not-calibrating
    # branch must clear it itself, or the HUD freezes on screen forever
    # after the calibration that produced it ends.
    calib_hud_placeholder = st.empty()
    if st.session_state.calibrating:
        calib_hud_placeholder.markdown(
            render_calib_hud(0.0, CALIBRATION_SECONDS), unsafe_allow_html=True)
    else:
        calib_hud_placeholder.empty()

    col_video, col_right = st.columns([2, 1])

    with col_video:
        video_container = st.container(border=True)
        with video_container:
            # Invisible - only exists so CSS :has() can frame this specific
            # panel with corner brackets and the ambient scan sweep. See
            # the .hud-frame-marker CSS comment.
            st.markdown('<div class="hud-frame-marker"></div>', unsafe_allow_html=True)
            topbar_placeholder = st.empty()
            video_placeholder = st.empty()
            level_placeholder = st.empty()
            status_placeholder = st.empty()

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="label">Behavioral signals</div>', unsafe_allow_html=True)
            signals_placeholder = st.empty()
            signals_placeholder.markdown(
                '<div style="color:#5B6772;font-size:11.5px;padding:6px 0;">'
                'No session running.</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="label">Metrics</div>', unsafe_allow_html=True)
            tiles_placeholder = st.empty()
            tiles_placeholder.markdown(render_tiles(0, 0), unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="label">Guidance</div>', unsafe_allow_html=True)
            guidance_placeholder = st.empty()

    # ---------- Agent live panel (only meaningful while running) ----------
    agent_panel_placeholder = st.empty()
    intervention_placeholder = st.empty()

    # Empty-session warning (legacy v1 sessions with no per-frame data)
    if st.session_state.get('empty_session_warning'):
        st.warning(st.session_state.empty_session_warning)
        del st.session_state.empty_session_warning

    if st.session_state.viewing_session_id and not st.session_state.running:
        st.caption(f"Viewing saved session from {st.session_state.viewing_session_label}. "
                   f"Press Start or Calibrate to begin a new live session.")

    with st.container(border=True):
        st.markdown('<div class="label">Session timeline</div>', unsafe_allow_html=True)
        st.markdown(TIMELINE_LEGEND, unsafe_allow_html=True)
        timeline_placeholder = st.empty()
        summary_placeholder = st.empty()

    # ---------- CALIBRATION ----------
    if st.session_state.calibrating:
        video_placeholder.markdown('', unsafe_allow_html=True)
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            st.session_state.calibrating = False
            return

        engine.start_calibration()
        calib_start = time.time()
        daisee_calib_samples = []

        try:
            while st.session_state.calibrating:
                ok, frame = cap.read()
                if not ok:
                    # Symmetric with the isOpened() guard above this loop -
                    # without resetting the flag here too, a mid-calibration
                    # camera glitch leaves the app stuck believing it's
                    # still calibrating (Start/Calibrate disabled) after the
                    # camera handle is already gone, until Stop is clicked.
                    st.session_state.calibrating = False
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _ = engine.extract(rgb)

                face_mesh_results = engine.face_mesh.process(rgb)
                face_crop, bbox = crop_face(frame, face_mesh_results)

                if face_crop is not None:
                    if (len(daisee_calib_samples) == 0 or
                        int((time.time() - calib_start) * 30) % CNN_EVERY_N_FRAMES == 0):
                        daisee_probs = ensemble_predict(face_crop, [
                            (models['daisee_mb'],  MOBILENET_SIZE,    mobilenet_preprocess),
                            (models['daisee_eff'], EFFICIENTNET_SIZE, efficientnet_preprocess),
                        ])
                        if daisee_probs is not None:
                            daisee_calib_samples.append(float(daisee_probs[1]))

                annotate_frame(frame, bbox, "LOW", 0.0, calibrating=True)
                video_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", width='stretch',
                )
                elapsed = time.time() - calib_start
                progress = min(1.0, elapsed / CALIBRATION_SECONDS)
                seconds_left = max(0.0, CALIBRATION_SECONDS - elapsed)
                level_placeholder.markdown(
                    render_calib_ring(progress, seconds_left), unsafe_allow_html=True,
                )
                calib_hud_placeholder.markdown(
                    render_calib_hud(progress, seconds_left), unsafe_allow_html=True,
                )

                if elapsed >= CALIBRATION_SECONDS:
                    face_ok = engine.finish_calibration()
                    daisee_ok = False
                    if len(daisee_calib_samples) >= config.MIN_DAISEE_CALIB_SAMPLES:
                        set_daisee_baseline(float(np.median(daisee_calib_samples)))
                        daisee_ok = True
                    st.session_state.calibrating = False
                    if face_ok or daisee_ok:
                        save_active_user_baseline(engine)
                    if face_ok and daisee_ok:
                        st.success("Calibration complete. Press Start to begin.")
                    elif face_ok or daisee_ok:
                        st.warning("Partial calibration — try again with a clearer view.")
                    else:
                        st.error("Calibration failed.")
                    time.sleep(1.2)
                    st.rerun()
                time.sleep(0.01)
        finally:
            cap.release()
        return

    # ---------- CAPTURE ----------
    if st.session_state.running:
        if not engine.baseline.calibrated:
            st.warning("Personal baseline not set — Stop, then Calibrate.")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            st.session_state.running = False
            return

        smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)
        hyst = HysteresisLevel(confirm_frames=config.HYSTERESIS_CONFIRM_FRAMES)

        if temporal_inf.is_available():
            temporal_inf.reset()

        rppg = get_live_rppg()
        rppg.reset()

        # The agentic core: decides when to interrupt, targets the driving
        # signal, then measures whether the intervention worked. Seeded with
        # what has historically worked on this install.
        try:
            _known = session_history.intervention_stats()
        except Exception:
            _known = {}
        agent = InterventionAgent(stats=_known)
        agent.start_session(time.time())
        last_agent_status_update = 0.0

        last_daisee_probs = None
        frame_idx = 0
        last_timeline_update = 0.0
        last_sidebar_update  = 0.0
        last_header_update = 0.0
        session_started_at = time.time()

        try:
            while st.session_state.running:
                ok, frame = cap.read()
                if not ok:
                    status_placeholder.warning("Frame read failed")
                    # Symmetric with the isOpened() guard above this loop -
                    # without resetting the flag here too, a mid-session
                    # camera glitch leaves the app stuck believing a
                    # session is still running (Start/Calibrate disabled,
                    # no loop actually active) until Stop is clicked. Also
                    # save whatever the session accumulated so far, same as
                    # the Stop button - otherwise resetting the flag here
                    # would silently drop that data instead.
                    st.session_state.running = False
                    save_current_session()
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_mesh_results = engine.face_mesh.process(rgb)
                face_crop, bbox = crop_face(frame, face_mesh_results)

                if face_crop is None:
                    status_placeholder.markdown(
                        '<div class="face-status">No face detected — please face the camera</div>',
                        unsafe_allow_html=True)
                    video_placeholder.image(rgb, channels="RGB", width='stretch')
                    frame_idx += 1
                    continue
                status_placeholder.empty()

                if frame_idx % CNN_EVERY_N_FRAMES == 0:
                    last_daisee_probs = ensemble_predict(face_crop, [
                        (models['daisee_mb'],  MOBILENET_SIZE,    mobilenet_preprocess),
                        (models['daisee_eff'], EFFICIENTNET_SIZE, efficientnet_preprocess),
                    ])

                # rPPG: cheap per frame. Heart rate is computed and shown
                # even at W_RPPG = 0 so it can be sanity-checked against a
                # wrist pulse before being trusted.
                rppg.add_frame(time.time(), frame, face_mesh_results)
                rppg_score = rppg.stress_score()

                behavioral_features = engine.extract(rgb)
                if behavioral_features is None:
                    frame_idx += 1
                    continue

                # Must come AFTER behavioral_features exists - the temporal
                # model consumes the features dict, not a pixel crop (the
                # old LSTM took face_crop directly, which is why this used
                # to run earlier in the loop, before engine.extract() had
                # even been called yet).
                temporal_score = None
                if temporal_inf.is_available():
                    temporal_inf.add_frame(behavioral_features)
                    temporal_score = temporal_inf.predict()

                result = compute_stress_index(
                    daisee_probs=last_daisee_probs,
                    behavioral_features=behavioral_features,
                    temporal_score=temporal_score,
                    task_mode=st.session_state.task_mode,
                    rppg_score=rppg_score,
                )
                smoothed = smoother.update(result['stress_index'])
                level = hyst.update(smoothed)

                # ---- intervention agent: decide / act / measure ----
                _iv = None
                if st.session_state.agent_enabled:
                    _iv = agent.update(time.time(), smoothed, result['contributions'])
                if _iv is not None and (time.time() - _iv.t_shown) <= config.INTERVENTION_SHOW_SECONDS:
                    intervention_placeholder.markdown(
                        render_intervention_card(_iv.title, _iv.body), unsafe_allow_html=True)
                else:
                    intervention_placeholder.empty()

                annotate_frame(frame, bbox, level, smoothed)
                video_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", width='stretch',
                )

                _hr, _q = rppg.estimate()
                _hr_text = f"{_hr:.0f}" if _hr else "--"
                topbar_placeholder.markdown(
                    render_video_topbar(_hr_text, behavioral_features.get('ear', 0.0)),
                    unsafe_allow_html=True)
                level_placeholder.markdown(render_level_block(level, smoothed),
                                          unsafe_allow_html=True)
                signals_placeholder.markdown(render_signal_rows(result['contributions']),
                                            unsafe_allow_html=True)
                _perclos_pct = (behavioral_features.get('perclos', 0.0) * 100.0
                               if st.session_state.task_mode == 'Driving' else None)
                tiles_placeholder.markdown(
                    render_tiles(behavioral_features.get('blinks_per_minute', 0.0),
                                behavioral_features.get('ear', 0.0), _hr_text,
                                perclos_pct=_perclos_pct),
                    unsafe_allow_html=True)
                guidance_placeholder.markdown(render_guidance(level, result['contributions']),
                                             unsafe_allow_html=True)

                now = datetime.now()
                if (not st.session_state.history or
                    (now - st.session_state.history[-1]['ts']).total_seconds() >= 0.5):
                    daisee_record = result.get('daisee_raw') or 0.0
                    _rec = {
                        'ts': now, 'stress_index': smoothed, 'level': level,
                        'daisee': daisee_record, 'behavior': result['behavior_stress'],
                    }
                    # Per-signal values: without these the self-retraining
                    # loop has nothing to learn from.
                    _rec.update(result['contributions'])
                    st.session_state.history.append(_rec)

                now_ts = time.time()

                if now_ts - last_header_update > 1.0:
                    elapsed_s = int(now_ts - session_started_at)
                    render_header(
                        f"SESSION {st.session_state.session_count:02d} &middot; "
                        f"{elapsed_s // 60:02d}:{elapsed_s % 60:02d}"
                    )
                    last_header_update = now_ts

                if now_ts - last_agent_status_update > 1.0:
                    if st.session_state.agent_enabled:
                        st_ = agent.status(now_ts)
                        agent_panel_placeholder.markdown(
                            render_agent_phases(st_['phase'], st_['detail']),
                            unsafe_allow_html=True)
                    else:
                        agent_panel_placeholder.markdown(
                            '<div class="phase-reason" style="color:#5B6772;">'
                            'Agent is off — monitoring only, no interventions, no learning.</div>',
                            unsafe_allow_html=True)
                    last_agent_status_update = now_ts

                if now_ts - last_sidebar_update > 0.5 and st.session_state.history:
                    recent = [h['stress_index'] for h in st.session_state.history[-LIVE_GRAPH_WINDOW:]]
                    sidebar_graph_placeholder.markdown(render_mini_graph(recent), unsafe_allow_html=True)
                    last_sidebar_update = now_ts

                if now_ts - last_timeline_update > 1.0 and len(st.session_state.history) > 1:
                    timeline_placeholder.markdown(render_full_timeline(st.session_state.history),
                                                  unsafe_allow_html=True)
                    last_timeline_update = now_ts

                frame_idx += 1
                time.sleep(0.01)
        finally:
            cap.release()
            # Hand the agent's measured outcomes to the Stop handler, which
            # persists them alongside the session row.
            try:
                _final_idx = (st.session_state.history[-1]['stress_index']
                              if st.session_state.history else 0.0)
                st.session_state.pending_interventions = agent.finish_session(
                    time.time(), _final_idx)
            except Exception:
                pass
    else:
        video_placeholder.markdown(render_idle_view(), unsafe_allow_html=True)
        signals_placeholder.markdown(
            '<div style="color:#5B6772;font-size:11.5px;padding:6px 0;">'
            'No session running.</div>', unsafe_allow_html=True)
        guidance_placeholder.markdown(
            '<div style="color:#5B6772;font-size:11.5px;padding:6px 0;">'
            'Guidance appears once a session starts.</div>', unsafe_allow_html=True)
        agent_panel_placeholder.empty()

        # ---------- SELF-REPORT PROMPT (Phase 3 labeling loop) ----------
        if st.session_state.pending_self_report is not None:
            _sid = st.session_state.pending_self_report
            with st.container(border=True):
                st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
                st.markdown("**How stressed did you actually feel during that "
                            "session?** Your answer is stored with the session "
                            "and is what the system learns from — be honest, "
                            "not agreeable.")
                _cols = st.columns(6)
                _labels = ["1 · calm", "2", "3 · neutral", "4", "5 · very stressed"]
                for _i, _lab in enumerate(_labels, start=1):
                    if _cols[_i - 1].button(_lab, key=f"selfrep_{_i}"):
                        try:
                            session_history.set_self_report(_sid, _i)
                        except Exception as e:
                            st.warning(f"Could not save rating: {e}")
                        st.session_state.pending_self_report = None
                        st.rerun()
                if _cols[5].button("Skip", key="selfrep_skip"):
                    st.session_state.pending_self_report = None
                    st.rerun()

        if st.session_state.history:
            timeline_placeholder.markdown(render_full_timeline(st.session_state.history),
                                          unsafe_allow_html=True)
            sidebar_values = [h['stress_index'] for h in st.session_state.history]
            sidebar_graph_placeholder.markdown(
                render_mini_graph(sidebar_values[-LIVE_GRAPH_WINDOW:]), unsafe_allow_html=True)
            history = st.session_state.history
            duration = (history[-1]['ts'] - history[0]['ts']).total_seconds()
            counts = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0}
            for r in history:
                counts[r['level']] = counts.get(r['level'], 0) + 1
            with summary_placeholder.container():
                st.markdown(f'<span style="color:#8B98A5;font-size:12px;">Session length '
                           f'<b style="color:#E6EDF3;">{duration:.0f}s</b> &middot; samples '
                           f'<b style="color:#E6EDF3;">{len(history)}</b></span>',
                           unsafe_allow_html=True)
                cols = st.columns(3)
                for i, lvl in enumerate(['LOW', 'MODERATE', 'HIGH']):
                    pct = (counts[lvl] / len(history)) * 100 if history else 0
                    cols[i].metric(lvl, f"{pct:.0f}%")
                csv_text = history_to_csv(history)
                _fname = export_filename(
                    st.session_state.subject_id, st.session_state.condition,
                    st.session_state.task_mode, when=history[0]['ts'],
                )
                st.download_button("Download session CSV", csv_text, file_name=_fname,
                                   mime="text/csv")
                st.caption(f"Saves as `{_fname}` — move it into `evaluation_data/`, "
                           f"then re-run `python run_all_evaluations.py`.")
        else:
            level_placeholder.markdown('', unsafe_allow_html=True)
            tiles_placeholder.markdown(render_tiles(0, 0), unsafe_allow_html=True)
            guidance_placeholder.markdown(
                '<div style="color:#5B6772;font-size:11.5px;">'
                'Guidance appears once a session starts.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
