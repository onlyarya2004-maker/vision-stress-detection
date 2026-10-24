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

import html
import io
import csv
import os
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import streamlit as st

from stress_fusion import (
    stress_level, stress_level_advice,
    VALID_TASK_MODES,
)
from app_naming import export_filename
from intervention_agent import InterventionAgent, CATALOGUE
import daisee_ensemble
import personalization
import session_history
import session_report
import session_store
import auth
import config
# crop_face / annotate_frame used to be defined in this file; they moved to
# frame_ops.py so session_pipeline.py can share them. Re-exported here because
# existing tests (and anything else) call app.crop_face.
from frame_ops import crop_face, annotate_frame, bbox_to_source
from session_pipeline import UserState, CalibrationRun, LiveRun, BatchGate
from browser_camera import browser_camera

# ---------------- CONFIG ----------------
# All values now live in config.py; these aliases keep the rest of the file
# readable and preserve the previous names. DaiSEE model paths/input sizes
# moved into daisee_ensemble.py, which app.py no longer needs to know.
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

AGENT_PHASES = ['warming', 'watching', 'declining', 'showing', 'evaluating', 'cooldown']

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
      /* Dark theme, navy-and-black. Every colour below is a token; the few
         hard-coded hexes left in the render helpers mirror these same values.
         Shades follow one cool navy ramp (bg -> panel -> panel2 -> border) so
         surfaces read as deliberate elevation steps rather than arbitrary
         greys, and accents sit in the 300-400 brightness band, which is what
         stays legible on a dark ground (the 500-600 shades used for light
         backgrounds go muddy here). Text: --text 15:1, --sub 8:1, --dim 4.9:1
         against --bg. */
      :root {
        --bg:#0B1020; --panel:#111827; --panel2:#141D30; --border:#1F2A3C;
        --border2:#2C3A52; --text:#E8EDF5; --sub:#A3B1C6; --dim:#7486A0;
        --blue:#3B82F6; --orange:#FB923C; --green:#34D399; --amber:#FBBF24;
        --red:#F87171; --indigo:#818CF8;
        --track:#1C2535;                       /* empty part of bars/rings */
        --tint-blue:#13203A;   --edge-blue:#27406E;
        --tint-green:#0E2A20;  --edge-green:#1C5844;
        --tint-amber:#2A2008;  --edge-amber:#5C4718;
        --tint-orange:#2A1A0C;
        --mono:'SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace;
      }
      html, body, [data-testid="stAppViewContainer"] {
        font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
        color:var(--text);
        background:
          linear-gradient(rgba(59,130,246,.05) 1px, transparent 1px) 0 0/28px 28px,
          linear-gradient(90deg, rgba(59,130,246,.05) 1px, transparent 1px) 0 0/28px 28px,
          var(--bg);
      }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
      [data-testid="stStatusWidget"], .stAppDeployButton { display:none !important; }
      [data-testid="stHeader"] { background:transparent; height:0; }
      /* Width matters for picture quality: the camera panel gets ~2/3 of this,
         and the browser downscales the (up to 1080p) stream into exactly that
         many CSS pixels. The old 1350px cap + Streamlit's default 80px side
         padding left the video ~780px wide on a 1920 screen - a 2.5x downscale
         that visibly softened the picture. Processing geometry is unaffected:
         the models always get config.PROCESS_WIDTH x PROCESS_HEIGHT. */
      .block-container { padding-top:0.9rem; padding-bottom:1rem; max-width:1800px;
                         padding-left:2.5rem; padding-right:2.5rem; }
      section[data-testid="stSidebar"] { background:#070C18; border-right:1px solid var(--border); }
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
      .stButton>button:hover { border-color:var(--blue); color:#fff; background:#16233C; }
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
                  background:linear-gradient(135deg,#3B82F6 0%,#1D4ED8 100%);
                  display:flex; align-items:center; justify-content:center; flex-shrink:0; }
      .app-title { color:var(--text); font-size:16px; font-weight:700; letter-spacing:-.01em; }
      .app-subtitle { color:var(--dim); font-size:11.5px; margin-top:1px; }
      .pill { display:inline-flex; align-items:center; gap:6px; font-size:11px;
              font-weight:700; letter-spacing:.07em; padding:5px 11px; border-radius:20px;
              border:1px solid; white-space:nowrap; }
      .pill.dot::before { content:''; width:7px; height:7px; border-radius:50%;
                          background:currentColor; box-shadow:0 0 7px currentColor;
                          animation:pulse 1.8s infinite; }
      .pill-live { color:var(--blue); border-color:var(--edge-blue); background:var(--tint-blue); }
      .pill-offline { color:var(--dim); border-color:var(--border2); background:var(--panel2); }
      .pill-offline::before { display:none; }
      .pill-calibrating { color:var(--amber); border-color:var(--edge-amber); background:var(--tint-amber); }
      .rec-tag { font-size:10px; font-weight:800; letter-spacing:.08em; color:var(--red);
                 background:rgba(248,113,113,.10); border:1px solid rgba(248,113,113,.38);
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
      .chip.on { color:#fff; border-color:var(--blue); background:var(--tint-blue); }
      .cal-pill { font-size:11px; font-weight:700; letter-spacing:.04em; display:inline-flex;
                  align-items:center; gap:5px; padding:5px 10px; border-radius:20px; border:1px solid; }
      .cal-pill.ok { color:var(--green); border-color:var(--edge-green); background:var(--tint-green); }
      .cal-pill.warn { color:var(--amber); border-color:var(--edge-amber); background:var(--tint-amber); }

      /* ---- calibration HUD ----
         The primary, always-visible progress readout. Rendered above the
         video columns so it can never end up below the fold - the ring
         inside the video panel is a secondary flourish, not the only way
         to see progress. */
      .calib-hud { background:rgba(59,130,246,.07); border:1px solid rgba(59,130,246,.32);
                   border-radius:10px; padding:12px 16px; margin-bottom:10px; }
      .chud-row { display:flex; align-items:center; justify-content:space-between;
                  gap:12px; margin-bottom:9px; }
      .chud-label { font-size:11px; font-weight:800; letter-spacing:.09em; color:var(--blue);
                    display:flex; align-items:center; gap:8px; white-space:nowrap; }
      .chud-dot { width:7px; height:7px; border-radius:50%; background:var(--blue);
                  flex-shrink:0; box-shadow:0 0 7px var(--blue); animation:pulse 1.2s infinite; }
      .chud-time { font-family:var(--mono); font-size:22px; font-weight:700; color:var(--text);
                   letter-spacing:.02em; font-variant-numeric:tabular-nums; }
      .chud-bar { display:flex; gap:3px; margin-bottom:9px; }
      .chud-bar i { flex:1; height:6px; border-radius:1px; background:var(--track); display:block; }
      .chud-bar i.on { background:var(--blue); box-shadow:none; }
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
      .hr-chip .hr-uncertain { color:var(--dim); font-weight:600; }
      .level-block { margin:10px 12px 12px 12px; background:var(--panel2);
                     border:1px solid var(--border); border-radius:10px; padding:14px 16px; }
      .index-row { display:flex; align-items:baseline; gap:12px; margin-bottom:8px; }
      .index-num { font-family:var(--mono); font-size:32px; font-weight:700; letter-spacing:-.02em; }
      .level-tag { font-size:12.5px; font-weight:800; letter-spacing:.09em; }
      .lvl-bar { height:7px; border-radius:5px; background:var(--track); overflow:hidden;
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
                            background:var(--bg); }
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
      .tag.on { color:var(--blue); border-color:var(--edge-blue); }
      .cal-check { display:inline-flex; align-items:center; gap:6px; font-size:11px;
                   font-weight:700; padding:5px 10px; border-radius:20px; margin-bottom:10px; }
      .cal-check.ok { color:var(--green); background:var(--tint-green); border:1px solid var(--edge-green); }
      .cal-check.warn { color:var(--amber); background:var(--tint-amber); border:1px solid var(--edge-amber); }

      /* ---- agent panel ---- */
      .agent-wrap { padding:2px; }
      .agent-top { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
      .agent-name { font-weight:800; font-size:11.5px; letter-spacing:.04em; color:var(--text); }
      .state-pill { margin-left:auto; font-size:9.5px; font-weight:800; letter-spacing:.06em;
                    padding:3px 9px; border-radius:12px; border:1px solid; }
      .state-pill.active { color:var(--green); border-color:var(--edge-green); background:var(--tint-green); }
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
        border-radius:50% !important; background:#0D1424 !important; border:1px solid var(--border) !important;
        color:#fff !important; font-weight:800 !important; font-size:12.5px !important;
        letter-spacing:.09em !important; padding:0 !important; z-index:1;
        transition:none !important;
      }
      div:has(.agent-core-marker) + div .stButton button::before {
        content:''; position:absolute; inset:-5px; border-radius:50%; z-index:-1;
      }
      div:has(.agent-core-marker.on) + div .stButton button::before {
        background:conic-gradient(from 0deg,var(--blue),var(--track) 40%,var(--blue) 80%,var(--track));
        animation:spin 3s linear infinite;
      }
      div:has(.agent-core-marker.off) + div .stButton button::before {
        background:var(--track); animation:none;
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
      .phase.cur { background:var(--tint-orange); border-color:var(--orange); color:var(--orange); }
      .phase-reason { font-size:10px; color:var(--sub); }

      .intervention-card { background:rgba(251,146,60,.08); border:1px solid rgba(251,146,60,.38);
                           border-radius:10px; padding:12px 16px; margin:0 0 10px 0; }
      .intervention-card .tag-lbl { color:var(--orange); font-size:10px; font-weight:800;
                                    letter-spacing:.09em; margin-bottom:3px; }
      .intervention-card .ttl { color:var(--text); font-weight:700; font-size:14px; }
      .intervention-card .body { color:var(--sub); font-size:12px; margin-top:4px; }

      .del-btn button { color:var(--red) !important; }
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# MODEL / ENGINE LOADING
# =============================================================================
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    """Load only the active live-system models (DaiSEE ensemble). Shared
    with evaluate_video_file.py via daisee_ensemble.py, so live and
    offline inference are provably running the same code, each with its
    own caching (Streamlit's decorator here). The weights themselves come
    from daisee_ensemble.shared_pairs(), which main() starts loading in the
    background before login so a cold start isn't a minute of blank page."""
    return {'pairs': daisee_ensemble.shared_pairs()}


def get_user_state():
    """This browser session's private UserState (feature engine + baselines +
    GRU buffer + rPPG). It used to be three process-wide singletons
    (@st.cache_resource get_feature_engine / get_temporal_inference,
    rppg_live.get_live_rppg) plus a module-global DaiSEE baseline - fine for
    one local user, but on a hosted server two people calibrating at once
    would overwrite each other's baselines. The CNN models in load_models()
    stay shared: they are stateless."""
    if 'user_state' not in st.session_state:
        if UserState.active_count() >= config.MAX_USER_SESSIONS:
            st.warning("The demo server is at capacity right now "
                       f"({config.MAX_USER_SESSIONS} simultaneous sessions). "
                       "Please try again in a minute.")
            st.stop()
        with st.spinner("Preparing your session..."):
            st.session_state.user_state = UserState()
    return st.session_state.user_state


def get_store():
    """Storage backend for THIS browser session: the local SQLite module, or
    (hosted) a SupabaseStore bound to the signed-in user's own login."""
    if config.STORAGE_BACKEND != 'supabase':
        return session_history
    sess = auth.current_session()
    if sess is None:
        raise RuntimeError("not signed in")
    return session_store.SupabaseStore(sess)


# prep() / ensemble_predict() live in daisee_ensemble.py, shared with
# evaluate_video_file.py - see that module's docstring for why (and for
# the tf.function speed fix that motivated pulling this out).
# crop_face() / annotate_frame() live in frame_ops.py (imported above).


# =============================================================================
# RENDER HELPERS
# =============================================================================
LEVEL_COLORS = {'LOW': '#34D399', 'MODERATE': '#FBBF24', 'HIGH': '#F87171'}


def render_mini_graph(values, height=110, width=250):
    if len(values) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" '
            f'style="width:100%;height:{height}px;background:#121A2A;border-radius:8px;">'
            f'<text x="{width/2}" y="{height/2}" font-size="11" fill="#7486A0" '
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
        f'style="width:100%;height:{height}px;background:#121A2A;border-radius:8px;">'
        f'<polygon points="{area_pts}" fill="rgba(245,118,43,0.15)"/>'
        f'<polyline points="{pts}" fill="none" stroke="#FB923C" stroke-width="2"/>'
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
        'Stress index': '#FB923C',
        'CNN branch':   '#3B82F6',
        'Behavioral branch': '#7486A0',
    }
    keymap = {'Stress index': 'stress_index', 'CNN branch': 'daisee',
              'Behavioral branch': 'behavior'}
    paths = []
    for label, color in series.items():
        key = keymap[label]
        # 'daisee' can be None (CNN branch hasn't run yet this session, or
        # never runs if no .h5 models are present - session_history.py's
        # v7->v8 migration). Carry the last real value forward for display,
        # matching how the live fusion itself reuses the last CNN score
        # between evaluations (app.py's last_daisee_probs); 0.0 only until
        # a real value has ever been seen. 'stress_index'/'behavior' are
        # never None, so this is a no-op for those two series.
        last = 0.0
        vals = []
        for rec in history:
            v = rec[key]
            if v is not None:
                last = v
            vals.append(last)
        pts = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}"
                       for i, v in enumerate(vals))
        width_px = "2.5" if key == 'stress_index' else "1.6"
        paths.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                     f'stroke-width="{width_px}"/>')
    gridlines = "".join(
        f'<line x1="0" y1="{height*f:.0f}" x2="{width}" y2="{height*f:.0f}" '
        f'stroke="#1E293B"/>' for f in (0.25, 0.5, 0.75))
    return f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" ' \
           f'style="width:100%;height:{height}px;display:block;">{gridlines}{"".join(paths)}</svg>'


TIMELINE_LEGEND = (
    '<div style="display:flex;gap:16px;font-size:10.5px;color:#A3B1C6;margin-bottom:6px;">'
    '<span><i style="display:inline-block;width:10px;height:2px;background:#FB923C;'
    'margin-right:5px;vertical-align:middle;"></i>Stress index</span>'
    '<span><i style="display:inline-block;width:10px;height:2px;background:#3B82F6;'
    'margin-right:5px;vertical-align:middle;"></i>CNN branch</span>'
    '<span><i style="display:inline-block;width:10px;height:2px;background:#7486A0;'
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
    if value >= 0.6:  return '#F87171'
    if value >= 0.4:  return '#FB923C'
    if value >= 0.2:  return '#3B82F6'
    return '#3A4A61'


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
        'LOW':      ('Stay relaxed', '#34D399'),
        'MODERATE': ('Reduce stress', '#FBBF24'),
        'HIGH':     ('Act now', '#F87171'),
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
                f'<div style="color:#FB923C;font-size:9.5px;font-weight:800;'
                f'letter-spacing:.07em;margin-bottom:2px;">DRIVEN BY {top_signal.replace("_"," ").upper()}</div>'
                f'<div style="color:#E8EDF5;font-weight:700;font-size:12.5px;">{entry["title"]}</div>'
                f'<div style="color:#A3B1C6;font-size:11px;margin-top:2px;">{entry["body"]}</div>'
                f'</div>'
            )

    items = ''.join(f'<span>{tip}</span>' for tip in GUIDANCE_TIPS[level])
    return (
        f'{lead_html}'
        f'<div class="glist-sub" style="color:{sub_color};">{sub_label}</div>'
        f'<div class="glist">{items}</div>'
    )


def render_video_topbar(rppg_status, ear_value):
    """
    rppg_status: LiveRPPG.status()'s own text - "no face" / "warming up
    (8/20s)" / "signal too noisy - hold still" / "72 bpm (resting baseline
    3/8)" / "72 bpm (rest 68)". Showing this instead of a bare number (or
    "--" when unavailable) is the point: rPPG abstains internally whenever
    the estimate would be unreliable, and previously nothing surfaced WHY -
    a user watching a blank "--" HR reading had no way to tell "camera
    hasn't found your face yet" from "your face is fine, just hold still"
    from "still building your resting baseline". See PLAN_FOR_PP2.md 3.4.
    """
    # status() always leads with the bpm number ("72 bpm (rest 68)") once it
    # has one - every abstaining state ("no face", "warming up (8/20s)",
    # "signal too noisy - hold still") leads with a word instead. That's a
    # reliable enough signal to dim the reading rather than parsing state
    # out of the string a second time.
    hr_class = "v mono" if rppg_status[:1].isdigit() else "v mono hr-uncertain"
    return (
        f'<div class="vid-topbar">'
        f'<span class="hr-chip">HR <span class="{hr_class}">{rppg_status}</span></span>'
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
        f'background:linear-gradient(90deg,#34D399,{color});"></i></div>'
        f'</div>'
    )


def render_calib_ring(progress, seconds_left):
    deg = max(0, min(100, progress * 100))
    return (
        f'<div class="calib-view">'
        f'<div class="calib-ring" style="background:conic-gradient(#3B82F6 {deg:.0f}%, #1E293B 0);">'
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
    '<svg viewBox="0 0 24 24" width="42" height="42" fill="none" stroke="#7486A0" '
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
            'daisee':       '' if rec['daisee'] is None else f"{rec['daisee']:.4f}",
            'emotion':      "0.0000",   # column kept for backward-compat with analyzer
            'behavior':     f"{rec['behavior']:.4f}",
        })
    return buf.getvalue()


def init_state():
    if 'view' not in st.session_state: st.session_state.view = 'live'
    if 'running' not in st.session_state: st.session_state.running = False
    if 'calibrating' not in st.session_state: st.session_state.calibrating = False
    if 'history' not in st.session_state: st.session_state.history = []
    if 'session_start' not in st.session_state: st.session_state.session_start = None
    if 'viewing_session_id' not in st.session_state: st.session_state.viewing_session_id = None
    if 'viewing_session_label' not in st.session_state: st.session_state.viewing_session_label = None
    if 'viewing_session_task_mode' not in st.session_state: st.session_state.viewing_session_task_mode = None
    if 'viewing_session_self_report' not in st.session_state: st.session_state.viewing_session_self_report = None
    # Tracks the self-report for the CURRENTLY-DISPLAYED history when it
    # came from a live Stop/failure save, not from "View" - so answering
    # the self-report banner is reflected immediately in that same
    # session's own CSV/PDF export instead of only the next time it's
    # reopened via "View" (BUG_AUDIT_2026-08-23.md #8).
    if 'last_saved_session_id' not in st.session_state: st.session_state.last_saved_session_id = None
    if 'last_saved_self_report' not in st.session_state: st.session_state.last_saved_self_report = None
    # Makes save_current_session() idempotent per in-memory history: a
    # stale, already-disabled-on-the-next-rerun Stop button that still
    # gets clicked (or the mid-session camera-failure path, which calls
    # save_current_session() then reruns without clearing history so the
    # CSV/PDF export stays available) can no longer write a second,
    # duplicate DB row for the same captured frames (BUG_AUDIT_2026-08-23.md #2).
    if 'history_saved' not in st.session_state: st.session_state.history_saved = False
    # True while a stopped recording failed to save and is still held in
    # memory for Retry/Discard (see save_current_session()).
    if 'save_failed' not in st.session_state: st.session_state.save_failed = False
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
    # Browser-camera run bookkeeping (CAPTURE_MODE == 'browser'). cam_epoch is
    # bumped on every Calibrate/Start/Stop so the browser restarts its frame
    # sequence and any late batch from the previous run is ignored.
    if 'cam_epoch' not in st.session_state: st.session_state.cam_epoch = 0
    if 'cam_ack' not in st.session_state: st.session_state.cam_ack = 0
    if 'cam_overlay' not in st.session_state: st.session_state.cam_overlay = {}
    if 'batch_gate' not in st.session_state: st.session_state.batch_gate = BatchGate()
    if 'calib_run' not in st.session_state: st.session_state.calib_run = None
    if 'live_run' not in st.session_state: st.session_state.live_run = None
    if 'live_timers' not in st.session_state: st.session_state.live_timers = {}


def init_storage():
    """Create/migrate the DB once per browser session. Runs AFTER login in
    hosted mode (the store needs a signed-in user)."""
    if 'db_initialized' not in st.session_state:
        get_store().init_db()
        st.session_state.db_initialized = True


def ensure_login():
    """True when the app may proceed. Local mode has no login; hosted mode
    shows the login page until this browser session has signed in."""
    if config.STORAGE_BACKEND != 'supabase':
        return True
    sess = auth.current_session()
    if sess is not None:
        # Every rerun: a revoked or expired (and unrefreshable) session sends
        # the user back to the login page instead of failing deep inside a
        # database call. get_session() is local unless the token is expiring.
        try:
            sess.ensure_fresh()
        except auth.AuthFailure as e:
            auth.expire_current(str(e))
            sess = None
    if sess is None:
        auth.render_login_page()
        return False
    st.session_state.active_user_id = sess.user_id
    st.session_state.active_user_name = sess.email
    return True


def bump_cam_epoch():
    """New camera run: the browser resets its frame sequence/clock, and the
    server forgets the previous run's acknowledgement, overlay and gate."""
    st.session_state.cam_epoch += 1
    st.session_state.cam_ack = 0
    st.session_state.cam_overlay = {}


def end_live_run(ustate):
    """Stop-button handler for the camera side: hand the agent's measured
    outcomes to save_current_session() (the old blocking loop did this in its
    `finally`), abandon any half-done calibration, and start a fresh epoch."""
    run = st.session_state.live_run
    if run is not None:
        try:
            st.session_state.pending_interventions = run.finish()
        except Exception:
            pass
    cal = st.session_state.calib_run
    if cal is not None:
        cal.abort()
    ustate.engine._calibrating = False
    st.session_state.live_run = None
    st.session_state.calib_run = None
    st.session_state.live_timers = {}
    bump_cam_epoch()


def save_current_session():
    """
    Persist st.session_state.history to the DB and queue the self-report
    prompt. Shared by the Stop button and the mid-session camera-failure
    path, so a session ending abnormally is saved exactly like one ending
    normally rather than silently dropping its data.

    Idempotent per in-memory history: a second call before history is next
    reset (e.g. a stale Stop click after a camera-failure save already ran
    this run) is a no-op rather than a duplicate DB row - see
    history_saved in init_state() and BUG_AUDIT_2026-08-23.md finding #2.

    Every outcome is reported in a way that SURVIVES the st.rerun() both
    callers make right after this returns (a plain st.warning() here used
    to be wiped before the user could read it, so a failed save looked
    exactly like a successful one with no rating prompt), and is written to
    the server log as counts / error types only - never data, ids or error
    messages. A recording that fails to save stays in memory
    (save_failed) until the user retries or explicitly discards it.

    Returns 'saved', 'empty', 'failed' or 'already'.
    """
    ss = st.session_state
    if ss.history_saved:
        return 'already'
    if not ss.history:
        ss.save_notice = ('info', "Nothing was saved: no frames were processed during that "
                                  "recording (no face detected, or it was stopped before any "
                                  "result arrived). Keep your face in view and record for at "
                                  "least 30 seconds.")
        print("[save] nothing to save: 0 processed samples", flush=True)
        return 'empty'
    n = len(ss.history)
    store = get_store()
    try:
        sid = store.save_session(ss.history, task_mode=ss.task_mode,
                                 user_id=ss.active_user_id)
        if sid is None:
            raise RuntimeError("the database returned no session id")
    except Exception as e:
        ss.save_failed = True
        ss.save_error = type(e).__name__
        print(f"[save] FAILED to save a recording of {n} samples: {type(e).__name__}",
              flush=True)
        return 'failed'
    ss.history_saved = True
    ss.save_failed = False
    # Queue the self-report prompt for this session. The user's answer -
    # not the system's prediction - is the ground-truth label the
    # adaptation loop trains on.
    ss.pending_self_report = sid
    ss.last_saved_session_id = sid
    ss.last_saved_self_report = None
    ss.save_notice = ('success', f"Recording saved ({n} samples).")
    print(f"[save] session saved: {n} samples", flush=True)
    # Only taken once the session exists, so a failed save keeps them for
    # the retry. A failure HERE does not un-save the session - say so.
    ivlog = ss.pop('pending_interventions', [])
    if ivlog:
        try:
            store.log_interventions(sid, ivlog)
        except Exception as e:
            print(f"[save] session saved, but {len(ivlog)} coaching outcome(s) failed: "
                  f"{type(e).__name__}", flush=True)
            ss.save_notice = ('warning', f"Recording saved ({n} samples), but its "
                                         f"{len(ivlog)} coaching outcome(s) could not be saved.")
    return 'saved'


def discard_unsaved_recording():
    """The user's explicit choice after a failed save: drop the recording."""
    ss = st.session_state
    ss.history = []
    ss.history_saved = False
    ss.save_failed = False
    ss.pop('save_error', None)
    ss.pop('pending_interventions', None)
    ss.save_notice = ('info', "Recording discarded - nothing was saved.")
    print("[save] unsaved recording discarded by the user", flush=True)


def apply_user_profile(ustate, user_id):
    """
    Swap this session's baseline for the given user's saved one, or back to
    generic defaults if they have none yet / user_id is None (anonymous).
    The same mechanism the "Reset baseline" button uses - the baseline is
    just a dataclass, swapping it is cheap and safe between sessions (never
    call this while running/calibrating, same guard every other
    baseline-touching control in the sidebar already has).
    UserState.apply_profile clears the rolling blink/EAR/PERCLOS windows
    first, so one person's samples never score against another's baseline
    (BUG_AUDIT_2026-08-23.md finding #5).
    """
    saved = get_store().load_user_baseline(user_id) if user_id is not None else None
    ustate.apply_profile(saved)


def save_active_user_baseline(ustate):
    """Persist this session's baseline to the active user's profile, called
    right after a completed calibration. No-op for anonymous use
    (active_user_id is None) - unchanged, single-shared-baseline behavior."""
    user_id = st.session_state.active_user_id
    if user_id is None:
        return
    get_store().save_user_baseline(user_id, ustate.baseline_payload())


def user_local_now():
    """'Now' on the clock saved sessions' started_at uses: the browser's local
    time in browser mode (BatchGate - the same source LiveRun's dt0 comes
    from), the machine's own in local-camera mode. If the browser has not yet
    reported its offset, the LATEST possible local time (UTC+14): a session
    recorded before this moment can then never look newer than it."""
    if config.CAPTURE_MODE == 'browser':
        gate = st.session_state.get('batch_gate')
        if gate is not None and gate.tz_known:
            return gate.user_now()
        return datetime.utcnow() + timedelta(hours=14)
    return datetime.now()


def stamp_baseline_change(ustate):
    """The baseline just changed (calibration or reset): record when, so
    personalization only pools sessions scored against THIS baseline, then
    persist it like before."""
    ustate.baseline_set_at = user_local_now()
    st.session_state.pop('personalization', None)
    save_active_user_baseline(ustate)


# =============================================================================
# ANALYTICS (Phase 5.1) - longitudinal view over saved session history.
# Read-only, no camera - kept as a separate early-return branch in main()
# rather than touching the live capture loop at all.
# =============================================================================
def session_analytics_dataframe(store=None):
    """
    All saved sessions as a pandas DataFrame with derived time columns, or
    None if there are no sessions yet. Pure function of the store - no
    Streamlit dependency, so this is unit-testable on its own. `store`
    defaults to the local SQLite module; hosted mode passes the signed-in
    user's SupabaseStore (so the frame holds that user's sessions only).
    """
    import pandas as pd
    store = store if store is not None else session_history
    rows = store.load_recent_sessions(n=100000)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['started_at'] = pd.to_datetime(df['started_at'])
    df['date'] = df['started_at'].dt.date
    df['hour'] = df['started_at'].dt.hour
    df['weekday'] = df['started_at'].dt.day_name()
    return df.sort_values('started_at').reset_index(drop=True)


_WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                  'Saturday', 'Sunday']


def render_analytics_page(store=None):
    with st.sidebar:
        st.markdown('<div class="label">Analytics</div>', unsafe_allow_html=True)
        if st.button("Back to live view", width='stretch'):
            st.session_state.view = 'live'
            st.rerun()

    st.markdown("## Session analytics")
    if config.STORAGE_BACKEND == 'supabase':
        st.caption("Weekly trends and time-of-day patterns across your saved sessions. "
                  "Only your account can see this data.")
    else:
        st.caption("Weekly trends and time-of-day patterns across every saved session - "
                  "on-device only, nothing here leaves this machine.")

    df = session_analytics_dataframe(store)
    if df is None or df.empty:
        st.info("No saved sessions yet. Run a live session (Calibrate -> Start -> "
               "Stop) to start building history here.")
        return

    n = len(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", n)
    c2.metric("Avg stress index", f"{df['mean_stress'].mean():.2f}")
    c3.metric("Days active", df['date'].nunique())
    c4.metric("Total time", f"{df['duration_sec'].sum() / 60:.0f} min")

    st.markdown("#### Stress trend over time")
    st.caption(f"Based on {n} session{'s' if n != 1 else ''} - patterns sharpen with more.")
    st.line_chart(df.set_index('started_at')[['mean_stress', 'max_stress']])

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### By time of day")
        st.bar_chart(df.groupby('hour')['mean_stress'].mean())
    with col_b:
        st.markdown("#### By day of week")
        daily = df.groupby('weekday')['mean_stress'].mean().reindex(_WEEKDAY_ORDER).dropna()
        st.bar_chart(daily)

    st.markdown("#### By context")
    by_mode = df.groupby('task_mode')['mean_stress'].mean().sort_values(ascending=False)
    st.bar_chart(by_mode)

    labeled = df[df['self_report'].notna()].copy()
    st.markdown("#### System reading vs. your own report")
    if len(labeled) < 2:
        st.caption(f"{len(labeled)} of {n} sessions have a self-report so far - need at "
                  "least 2 to plot a trend. Answer the end-of-session question to build "
                  "this up (it's also what powers Phase 3's continual adaptation).")
    else:
        st.caption("Self-report (1-5) normalized to 0-1 so it shares an axis with the "
                  "system's 0-1 stress index. Both should move in the same direction if "
                  "the system is reading you correctly - they are NOT the same scale, "
                  "just plotted together for shape comparison.")
        labeled['self_report_normalized'] = (labeled['self_report'] - 1) / 4
        st.line_chart(labeled.set_index('started_at')[['self_report_normalized', 'mean_stress']])

    with st.expander("Raw session history"):
        st.dataframe(
            df[['started_at', 'duration_sec', 'task_mode', 'mean_stress', 'max_stress',
               'pct_low', 'pct_moderate', 'pct_high', 'self_report']],
            width='stretch',
        )


# =============================================================================
# LIVE SESSION PLUMBING - shared by the local-camera loops and the browser
# camera fragment, so both paint the UI identically.
# =============================================================================
class LivePanels:
    """The Streamlit placeholders a live session paints into (all created in
    main()'s layout). sidebar_graph is None in browser mode: a fragment is
    not allowed to write into st.sidebar, so that mini-graph only refreshes
    on full reruns there (the main session timeline updates live)."""

    def __init__(self, **elements):
        self.__dict__.update(elements)


NO_FACE_HTML = ('<div class="face-status">No face detected — '
                'please face the camera</div>')


def open_local_camera():
    """cv2 camera for CAPTURE_MODE == 'local', or None if it won't open."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def make_live_run(ustate, models, store, dt0=None):
    """Start a LiveRun: the agentic core, seeded with what has historically
    worked for THIS user (hosted) / install (local), and - once a task-mode
    bucket has enough evidence - what worked in this specific context. See
    InterventionAgent's own docstring for why both are passed."""
    try:
        known = store.intervention_stats()
        known_by_mode = store.intervention_stats_by_mode()
    except Exception:
        known, known_by_mode = {}, {}
    agent = InterventionAgent(stats=known, mode_stats=known_by_mode,
                              task_mode=st.session_state.task_mode)
    # Refit from the database at every Start: picks up ratings given since
    # (on this or another device) and can never include this new session.
    personal = current_personalization(store, ustate, refresh=True)
    # Server-log marker (no personal data) so a missing session can later be
    # told apart from a recording that never started.
    print(f"[live] recording started (personalization "
          f"{'on' if personal.active else 'off'})", flush=True)
    return LiveRun(ustate, models, agent, task_mode=st.session_state.task_mode,
                   agent_enabled=st.session_state.agent_enabled, dt0=dt0,
                   personalization=personal)


def current_personalization(store, ustate, refresh=False):
    """The signed-in user's personalization.Personalization for the CURRENT
    task mode and baseline (general model when disabled, not yet earned,
    incompatible history, or unreadable history). Kept in st.session_state,
    which sign-out/expiry wipe entirely, and refit whenever the user, mode or
    baseline changes; the per-session feature cache is keyed by user id."""
    if not personalization.enabled_for(config.STORAGE_BACKEND):
        return personalization.general("personalization is off for this storage mode.")
    uid = st.session_state.active_user_id
    key = (uid, st.session_state.task_mode, ustate.baseline_set_at)
    if (refresh or 'personalization' not in st.session_state
            or st.session_state.get('personalization_key') != key):
        caches = st.session_state.setdefault('personal_cache', {})
        st.session_state.personalization = personalization.load(
            store, uid, cache=caches.setdefault(uid, {}),
            task_mode=st.session_state.task_mode, baseline_set_at=ustate.baseline_set_at)
        st.session_state.personalization_key = key
    return st.session_state.personalization


def render_calibration_progress(panels, run):
    panels.level.markdown(
        render_calib_ring(run.progress(), run.seconds_left()), unsafe_allow_html=True)
    panels.calib_hud.markdown(
        render_calib_hud(run.progress(), run.seconds_left()), unsafe_allow_html=True)


def finish_calibration(ustate, run):
    """Shared calibration epilogue. Returns the (st-method-name, text)
    message for the user."""
    face_ok, daisee_ok = run.finish()
    st.session_state.calibrating = False
    if face_ok or daisee_ok:
        stamp_baseline_change(ustate)
    if face_ok and daisee_ok:
        return 'success', "Calibration complete. Press Start to begin."
    if face_ok or daisee_ok:
        return 'warning', "Partial calibration — try again with a clearer view."
    return 'error', "Calibration failed."


def record_history(out):
    """Append a processed frame's history row (the run throttles these to one
    per HISTORY_INTERVAL_S). Called for EVERY frame of a browser batch, not
    just the last one, so no row is skipped."""
    if not out.get('no_face') and out.get('record') is not None:
        st.session_state.history.append(out['record'])


def render_live_outputs(panels, out, run, update_sidebar=True):
    """Everything the old blocking capture loop painted after each frame:
    status line, intervention card, top bar, level, signals, tiles, guidance,
    plus the throttled header / agent panel / sidebar graph / timeline."""
    ss = st.session_state
    timers = ss.live_timers
    if out.get('no_face'):
        panels.status.markdown(NO_FACE_HTML, unsafe_allow_html=True)
        return
    panels.status.empty()

    iv = out['intervention']
    if iv is not None:
        panels.intervention.markdown(
            render_intervention_card(iv.title, iv.body), unsafe_allow_html=True)
    else:
        panels.intervention.empty()

    bf, result = out['behavioral_features'], out['result']
    level, smoothed = out['level'], out['smoothed']
    panels.topbar.markdown(
        render_video_topbar(out['rppg_status'], bf.get('ear', 0.0)),
        unsafe_allow_html=True)
    panels.level.markdown(render_level_block(level, smoothed), unsafe_allow_html=True)
    panels.signals.markdown(render_signal_rows(result['contributions']),
                            unsafe_allow_html=True)
    hr = out['hr_bpm']
    hr_text = f"{hr:.0f}" if hr else "--"
    perclos_pct = (bf.get('perclos', 0.0) * 100.0
                   if ss.task_mode == 'Driving' else None)
    panels.tiles.markdown(
        render_tiles(bf.get('blinks_per_minute', 0.0), bf.get('ear', 0.0), hr_text,
                     perclos_pct=perclos_pct),
        unsafe_allow_html=True)
    panels.guidance.markdown(render_guidance(level, result['contributions']),
                             unsafe_allow_html=True)

    now = time.time()
    if now - timers.get('header', 0.0) > 1.0:
        elapsed_s = int(run.elapsed())
        panels.header(f"SESSION {ss.session_count:02d} &middot; "
                      f"{elapsed_s // 60:02d}:{elapsed_s % 60:02d}")
        timers['header'] = now

    if now - timers.get('agent', 0.0) > 1.0:
        if ss.agent_enabled:
            st_ = run.agent_status()
            if st_ is not None:
                panels.agent_panel.markdown(
                    render_agent_phases(st_['phase'], st_['detail']),
                    unsafe_allow_html=True)
        else:
            panels.agent_panel.markdown(
                '<div class="phase-reason" style="color:#7486A0;">'
                'Agent is off — monitoring only, no interventions, no learning.</div>',
                unsafe_allow_html=True)
        timers['agent'] = now

    if (update_sidebar and panels.sidebar_graph is not None and ss.history
            and now - timers.get('sidebar', 0.0) > 0.5):
        recent = [h['stress_index'] for h in ss.history[-LIVE_GRAPH_WINDOW:]]
        panels.sidebar_graph.markdown(render_mini_graph(recent), unsafe_allow_html=True)
        timers['sidebar'] = now

    if len(ss.history) > 1 and now - timers.get('timeline', 0.0) > 1.0:
        panels.timeline.markdown(render_full_timeline(ss.history), unsafe_allow_html=True)
        timers['timeline'] = now


# ---- local (cv2) capture ----------------------------------------------------
def local_calibration(panels, video_placeholder, ustate, models):
    video_placeholder.markdown('', unsafe_allow_html=True)
    cap = open_local_camera()
    if cap is None:
        st.error("Could not open webcam.")
        st.session_state.calibrating = False
        return
    run = CalibrationRun(ustate, models)
    try:
        while st.session_state.calibrating:
            ok, frame = cap.read()
            if not ok:
                # Symmetric with the isOpened() guard above this loop -
                # without resetting the flag here too, a mid-calibration
                # camera glitch leaves the app stuck believing it's still
                # calibrating (Start/Calibrate disabled) after the camera
                # handle is already gone, until Stop is clicked.
                st.session_state.calibrating = False
                break
            frame = cv2.flip(frame, 1)
            info = run.process(frame, time.time())
            annotate_frame(frame, bbox_to_source(info['bbox'], info['geom']),
                           "LOW", 0.0, calibrating=True, scale=frame.shape[1] / 640.0)
            video_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
            render_calibration_progress(panels, run)
            if run.done:
                kind, text = finish_calibration(ustate, run)
                getattr(st, kind)(text)
                time.sleep(1.2)
                st.rerun()
            time.sleep(0.01)
    finally:
        cap.release()


def local_live(panels, video_placeholder, ustate, models, store):
    cap = open_local_camera()
    if cap is None:
        st.error("Could not open webcam.")
        st.session_state.running = False
        return
    run = make_live_run(ustate, models, store)
    st.session_state.live_run = run
    st.session_state.live_timers = {}
    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                panels.status.warning("Frame read failed")
                # Symmetric with the isOpened() guard above this loop -
                # without resetting the flag here too, a mid-session camera
                # glitch leaves the app stuck believing a session is still
                # running (Start/Calibrate disabled, no loop actually
                # active) until Stop is clicked. Also save whatever the
                # session accumulated so far, same as the Stop button -
                # otherwise resetting the flag here would silently drop that
                # data instead. end_live_run() first so the agent's measured
                # outcomes are queued BEFORE save_current_session() pops
                # them (the old order logged them against the next session).
                st.session_state.running = False
                end_live_run(ustate)
                save_current_session()
                # Rerun immediately so the sidebar's Stop button (rendered
                # earlier in this same script run while running was still
                # True) re-renders disabled - without this it stays live and
                # stale. history is deliberately NOT cleared (same as the Stop
                # button) so the CSV/PDF export is still available;
                # save_current_session() is idempotent per in-memory history
                # (BUG_AUDIT_2026-08-23.md #2).
                st.rerun()

            frame = cv2.flip(frame, 1)
            out = run.process(frame, time.time())
            if out['no_face']:
                panels.status.markdown(NO_FACE_HTML, unsafe_allow_html=True)
                video_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
                continue
            record_history(out)
            annotate_frame(frame, bbox_to_source(out['bbox'], out['geom']),
                           out['level'], out['smoothed'], scale=frame.shape[1] / 640.0)
            video_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width='stretch')
            render_live_outputs(panels, out, run)
            time.sleep(0.01)
    finally:
        cap.release()
        # Hand the agent's measured outcomes to the Stop handler, which
        # persists them alongside the session row - unless end_live_run()
        # already did (the camera-failure path above).
        if st.session_state.get('live_run') is run:
            try:
                st.session_state.pending_interventions = run.finish()
            except Exception:
                pass
            st.session_state.live_run = None


# ---- browser capture --------------------------------------------------------
def _calibration_step(frames, panels, ustate, models):
    ss = st.session_state
    run = ss.calib_run
    if run is None:
        run = ss.calib_run = CalibrationRun(ustate, models)
    last = None
    for t, frame in frames:
        last = run.process(frame, t)
        if run.done:
            break
    render_calibration_progress(panels, run)
    overlay = {'label': f"CALIBRATING  {run.seconds_left():.0f}s",
               'color': '#3B82F6', 'progress': run.progress(),
               'bbox': list(last['bbox']) if last and last['bbox'] else None}
    if last and last['no_face']:
        overlay['note'] = 'No face detected — please face the camera'
    if run.done:
        ss.flash = finish_calibration(ustate, run)
        ss.calib_run = None
        bump_cam_epoch()
        st.rerun()      # full rerun: sidebar buttons, CALIBRATED badge, flash
    return overlay


def _live_step(frames, panels, ustate, models, store, gate):
    ss = st.session_state
    run = ss.live_run
    if run is None:
        run = ss.live_run = make_live_run(ustate, models, store, dt0=gate.user_now())
    last = None
    t_models = time.perf_counter()
    for t, frame in frames:
        out = run.process(frame, t)
        record_history(out)
        last = out
    t_paint = time.perf_counter()
    render_live_outputs(panels, last, run, update_sidebar=False)
    if os.environ.get('STRESS_PROFILE'):
        print(f"[profile]   models {(t_paint - t_models) * 1000 / len(frames):.0f} ms/frame, "
              f"painting {(time.perf_counter() - t_paint) * 1000:.0f} ms/batch", flush=True)
        if run.stage_ms and run.frame_idx and run.frame_idx % 40 < len(frames):
            calls = max(1, run.stage_ms.get('cnn_calls', 1))
            print("[profile]   stage averages so far (ms): " + ", ".join(
                f"{k} {v / (calls if '(per call)' in k else run.frame_idx):.1f}"
                for k, v in run.stage_ms.items() if k != 'cnn_calls')
                + f"  [{calls} CNN calls over {run.frame_idx} frames]", flush=True)
            try:
                print("[profile]   tf.function tracing counts: "
                      + str([fn.experimental_get_tracing_count()
                             for fn, _, _ in models['pairs'] if fn is not None]), flush=True)
            except Exception:
                pass
    if last['no_face']:
        return {'label': '', 'color': '#FBBF24',
                'note': 'No face detected — please face the camera'}
    return {'label': f"{last['level']} ({last['smoothed'] * 100:.0f}%)",
            'color': LEVEL_COLORS[last['level']],
            'bbox': list(last['bbox']) if last['bbox'] else None}


@st.fragment
def camera_fragment(panels, ustate, models, store):
    """The browser camera. A fragment, so every batch of snapshots reruns only
    this function - not the whole page. Order matters: the component's
    incoming value is read from session_state BEFORE the component is
    rendered, so the acknowledgement and overlay computed from that batch go
    out in the same run (the browser sends its next batch only once it sees
    the ack). Verified: ~170 ms round trip, no stalls."""
    ss = st.session_state
    t_start = time.perf_counter()
    gate, epoch = ss.batch_gate, ss.cam_epoch
    incoming = ss.get('browser_cam')
    frames, ack = gate.accept(incoming, epoch)
    ss.cam_ack = ack
    t_decoded = time.perf_counter()

    mode = 'calibrate' if ss.calibrating else ('live' if ss.running else 'idle')
    overlay = ss.cam_overlay if mode != 'idle' else {}
    if isinstance(incoming, dict) and incoming.get('err') and mode != 'idle':
        panels.status.warning(
            "Camera unavailable — follow the message on the video, then press "
            "Stop and try again.")
    elif frames:
        if mode == 'calibrate':
            overlay = _calibration_step(frames, panels, ustate, models)
        elif mode == 'live':
            overlay = _live_step(frames, panels, ustate, models, store, gate)
    ss.cam_overlay = overlay
    t_done = time.perf_counter()
    browser_camera(mode=mode, epoch=epoch, ack=ack, overlay=overlay,
                   test_url=os.environ.get('STRESS_TEST_FRAMES_URL', ''),
                   key='browser_cam')
    if frames and os.environ.get('STRESS_PROFILE'):
        # Server-side cost of one batch: decode / (models + painting) per frame.
        print(f"[profile] {mode} batch of {len(frames)}: decode "
              f"{(t_decoded - t_start) * 1000:.0f} ms, process+paint "
              f"{(t_done - t_decoded) * 1000:.0f} ms "
              f"({(t_done - t_decoded) * 1000 / len(frames):.0f} ms/frame), "
              f"dropped so far {gate.dropped}", flush=True)

# =============================================================================
# MAIN
# =============================================================================
def apply_secret_overrides():
    """Let .streamlit/secrets.toml supply the hosted-mode switches, not just
    the Supabase credentials.

    config.py reads these from os.environ only - deliberately, because CLI
    tools (rls_check.py, load_test.py, the evaluation scripts) import it and
    must not drag Streamlit in. The gap that leaves: a secrets.toml holding
    only SUPABASE_* would start the app on local SQLite and never touch
    Supabase, with no error - it would simply look like nothing was saved.
    Environment variables still win, so Azure's container env is unaffected."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for env_name, cfg_name in (('STRESS_STORAGE_BACKEND', 'STORAGE_BACKEND'),
                               ('STRESS_CAPTURE_MODE', 'CAPTURE_MODE')):
        if os.environ.get(env_name):
            continue
        try:
            value = secrets.get(env_name)
        except Exception:
            value = None
        if value:
            setattr(config, cfg_name, value)


def main():
    apply_secret_overrides()
    init_state()
    inject_css()
    # Start loading the CNN weights now, in the background: the login page (and
    # the person typing their password) doesn't need them, and they will be
    # ready, already warmed up, by the time the first session starts.
    daisee_ensemble.start_background_load()

    # Hosted mode: nothing below runs until this browser session has signed in.
    if not ensure_login():
        return
    init_storage()
    store = get_store()

    if st.session_state.view == 'analytics':
        # Read-only history view - skip loading the CNN ensemble/engine
        # entirely, they're not needed here and cost real seconds to load.
        render_analytics_page(store)
        return

    models = load_models()
    ustate = get_user_state()
    engine = ustate.engine
    hosted = config.STORAGE_BACKEND == 'supabase'

    # Hosted: load the signed-in account's saved baseline once per login.
    if hosted and not st.session_state.get('profile_loaded'):
        apply_user_profile(ustate, st.session_state.active_user_id)
        st.session_state.profile_loaded = True

    is_fully_calibrated = ustate.fully_calibrated

    # Messages queued by the camera fragment just before it triggered a full
    # rerun (e.g. the calibration result) - shown once.
    _flash = st.session_state.pop('flash', None)
    if _flash:
        getattr(st, _flash[0])(_flash[1])

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown('<div class="label">Live stress</div>', unsafe_allow_html=True)
        sidebar_graph_placeholder = st.empty()
        sidebar_graph_placeholder.markdown(render_mini_graph([]), unsafe_allow_html=True)

        if st.button("View analytics", width='stretch',
                    disabled=st.session_state.running or st.session_state.calibrating,
                    help="Weekly trends and time-of-day patterns across saved sessions."):
            st.session_state.view = 'analytics'
            st.rerun()

        st.divider()
        st.markdown('<div class="label">Session control</div>', unsafe_allow_html=True)

        # ---------- User profile ----------
        # Phase 1 prerequisite for team recordings: each person gets their
        # own persisted calibration baseline instead of everyone sharing
        # the single process-wide one. Switching users loads that person's
        # saved baseline (or generic defaults for a brand-new profile) into
        # the live engine - the same swap "Reset baseline" already did.
        if hosted:
            # Hosted: the account IS the profile. No user dropdown - nobody
            # can pick another person's profile; the database only ever
            # returns the signed-in account's own rows.
            st.markdown(
                f'<div style="font-size:12px;color:#A3B1C6;margin-bottom:4px;">Signed in as</div>'
                f'<div style="font-size:13px;color:#E8EDF5;word-break:break-all;">'
                f'{html.escape(str(st.session_state.active_user_name))}</div>',
                unsafe_allow_html=True)
            if st.button("Sign out", width='stretch',
                         disabled=(st.session_state.running or st.session_state.calibrating
                                   or st.session_state.save_failed),
                         help=("Retry saving or discard the unsaved recording first."
                               if st.session_state.save_failed else None)):
                auth.sign_out_current()
                st.rerun()
        else:
            _users = store.list_users()
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
                apply_user_profile(ustate, st.session_state.active_user_id)
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
                        new_id = store.create_user(_new_user_name)
                        apply_user_profile(ustate, new_id)
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

        # A still-unanswered self-report from the PREVIOUS session blocks
        # starting a new one - Start/Stop don't clear pending_self_report,
        # so without this a second Stop would silently overwrite it with
        # the new session's id, losing the first session's self-report
        # opportunity with no warning (unlike the explicit, visible Skip
        # button). See BUG_AUDIT_2026-08-23.md #13.
        _report_pending = st.session_state.pending_self_report is not None
        # Likewise a recording that failed to save: starting over would
        # overwrite it, so Retry or Discard comes first.
        _unsaved = st.session_state.save_failed
        _blocked_help = ("Retry saving or discard the unsaved recording below first."
                         if _unsaved else
                         "Answer or skip the previous session's self-report "
                         "question below first." if _report_pending else None)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Calibrate", width='stretch',
                         disabled=(st.session_state.running or st.session_state.calibrating
                                  or _report_pending or _unsaved),
                         help=(_blocked_help if _blocked_help else
                              f"Sit naturally for {CALIBRATION_SECONDS} seconds. "
                              f"System learns your personal baseline.")):
                st.session_state.calibrating = True
                # The previous session's "rating saved / NOT saved" notice
                # belongs to that session - never show it after this one.
                st.session_state.pop('selfrep_notice', None)
                st.session_state.pop('save_notice', None)
                st.session_state.history = []
                st.session_state.history_saved = False
                st.session_state.last_saved_session_id = None
                st.session_state.last_saved_self_report = None
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.session_state.viewing_session_task_mode = None
                st.session_state.viewing_session_self_report = None
                st.session_state.calib_run = None
                bump_cam_epoch()
                st.rerun()
        with col_b:
            if st.button("Start", type="primary", width='stretch',
                         disabled=(st.session_state.running or st.session_state.calibrating
                                  or _report_pending or _unsaved),
                         help=_blocked_help):
                st.session_state.running = True
                st.session_state.session_start = datetime.now()
                st.session_state.pop('selfrep_notice', None)    # see Calibrate
                st.session_state.pop('save_notice', None)
                st.session_state.history = []
                st.session_state.history_saved = False
                st.session_state.last_saved_session_id = None
                st.session_state.last_saved_self_report = None
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.session_state.viewing_session_task_mode = None
                st.session_state.viewing_session_self_report = None
                st.session_state.session_count += 1
                st.session_state.live_run = None
                st.session_state.live_timers = {}
                st.session_state.pop('pending_interventions', None)
                bump_cam_epoch()
                st.rerun()

        if st.button("Stop", width='stretch',
                     disabled=not (st.session_state.running or st.session_state.calibrating)):
            st.session_state.running = False
            st.session_state.calibrating = False
            # Hands the agent's outcomes to save_current_session() (the old
            # blocking loop did this in its `finally`) and ends the camera run.
            end_live_run(ustate)
            save_current_session()
            st.rerun()

        col_c, col_d = st.columns(2)
        with col_c:
            if st.button("Clear view", width='stretch',
                         disabled=(st.session_state.running or st.session_state.calibrating
                                   or st.session_state.save_failed),
                         help="Clears the on-screen timeline. Saved sessions in the DB are not affected."):
                st.session_state.history = []
                st.session_state.history_saved = False
                st.session_state.last_saved_session_id = None
                st.session_state.last_saved_self_report = None
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.session_state.viewing_session_task_mode = None
                st.session_state.viewing_session_self_report = None
                st.rerun()
        with col_d:
            if st.button("Reset baseline", width='stretch',
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help="Wipes the personal baseline back to generic defaults. "
                              "Stop first - resetting mid-session would silently "
                              "degrade the rest of that session's scoring."):
                ustate.reset_baselines()
                # Without this, the reset is in-memory only: switching to
                # another user profile and back reloads the PRE-reset
                # baseline from the DB, silently undoing the reset for any
                # named user (no-op for anonymous - see the function's own
                # early return). Same call calibration success already
                # makes, kept consistent here (BUG_AUDIT_2026-08-23.md #7).
                stamp_baseline_change(ustate)
                st.rerun()

        # ---------- Personal calibration status ----------
        # Read-only status of personalization.py for THIS account. It changes
        # the score only from the next Start, never the session on screen.
        if personalization.enabled_for(config.STORAGE_BACKEND):
            try:
                _p = current_personalization(store, ustate)
            except Exception:
                _p = personalization.general("your history could not be loaded right now.")
            st.caption(_p.describe() + " The shared vision models are the same for everyone.")

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
            _mem = InterventionAgent(stats=store.intervention_stats()).learned_summary()
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
                f'works for you. '
                f'{"What it learns is kept per account - never shared with other users." if hosted else "Runs entirely on this device."}'
                f'</div>'
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
            recent = store.load_recent_sessions(n=10)
            total  = store.session_count()
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
                if pct_h >= 30:   dot_color = '#F87171'
                elif pct_m >= 40: dot_color = '#FBBF24'
                else:             dot_color = '#34D399'
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
                                 disabled=(st.session_state.running or st.session_state.calibrating
                                           or st.session_state.save_failed),
                                 help="Load this session's timeline into the main view"):
                        samples = store.load_session_samples(s['id'])
                        if samples:
                            st.session_state.history = samples
                            st.session_state.viewing_session_id = s['id']
                            st.session_state.viewing_session_label = start_label
                            st.session_state.viewing_session_task_mode = _task
                            st.session_state.viewing_session_self_report = s.get('self_report')
                        else:
                            # Clear whatever was previously being viewed too -
                            # otherwise this click appears to silently do
                            # nothing but show a warning, while an unrelated
                            # earlier session's timeline stays on screen as
                            # if it were the one just clicked
                            # (BUG_AUDIT_2026-08-23.md #14).
                            st.session_state.history = []
                            st.session_state.viewing_session_id = None
                            st.session_state.viewing_session_label = None
                            st.session_state.viewing_session_task_mode = None
                            st.session_state.viewing_session_self_report = None
                            st.session_state.empty_session_warning = (
                                f"Session from {start_label} was saved before per-frame "
                                f"recording was supported. Only the summary stats above "
                                f"are available."
                            )
                        st.rerun()

            st.markdown('<div class="del-btn">', unsafe_allow_html=True)
            if st.button("Delete my data" if hosted else "Delete all history",
                         width='stretch', key="clear_db",
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help=("Deletes every session saved under YOUR account. Other "
                               "accounts are not affected. Cannot be undone." if hosted else
                               "Deletes every saved session from the local DB. Cannot be undone.")):
                store.clear_all_sessions()
                st.session_state.pop('personalization', None)    # learned from deleted data
                st.session_state.pop('personal_cache', None)
                st.session_state.history = []
                st.session_state.history_saved = False
                st.session_state.last_saved_session_id = None
                st.session_state.last_saved_self_report = None
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.session_state.viewing_session_task_mode = None
                st.session_state.viewing_session_self_report = None
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
            if config.CAPTURE_MODE == 'browser':
                # The user's own browser shows the camera (browser_camera
                # component, mounted below); this container just reserves its
                # spot. Always the same position + key in every state so the
                # camera is never torn down and re-requested mid-session.
                cam_container = st.container()
                video_placeholder = None
            else:
                cam_container = None
                video_placeholder = st.empty()
            level_placeholder = st.empty()
            status_placeholder = st.empty()

    with col_right:
        with st.container(border=True):
            st.markdown('<div class="label">Behavioral signals</div>', unsafe_allow_html=True)
            signals_placeholder = st.empty()
            signals_placeholder.markdown(
                '<div style="color:#7486A0;font-size:11.5px;padding:6px 0;">'
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

    # ---------- CAPTURE (calibration + live) ----------
    panels = LivePanels(
        header=render_header, topbar=topbar_placeholder, level=level_placeholder,
        status=status_placeholder, signals=signals_placeholder,
        tiles=tiles_placeholder, guidance=guidance_placeholder,
        intervention=intervention_placeholder, agent_panel=agent_panel_placeholder,
        timeline=timeline_placeholder, calib_hud=calib_hud_placeholder,
        sidebar_graph=(sidebar_graph_placeholder
                       if config.CAPTURE_MODE != 'browser' else None),
    )
    if config.CAPTURE_MODE == 'browser':
        # Mounted in EVERY state (idle shows a live preview, calibrate/live
        # also stream snapshots) at the same position with the same key, so
        # the camera is never torn down and re-requested mid-session.
        with cam_container:
            camera_fragment(panels, ustate, models, store)
    elif st.session_state.calibrating:
        local_calibration(panels, video_placeholder, ustate, models)
        return

    if st.session_state.running:
        if not engine.baseline.calibrated:
            st.warning("Personal baseline not set — Stop, then Calibrate.")
        if config.CAPTURE_MODE != 'browser':
            local_live(panels, video_placeholder, ustate, models, store)
        # Browser mode: the camera fragment above drives the live session.
    elif st.session_state.calibrating:
        pass    # Browser mode: the camera fragment above drives calibration.
    else:
        if video_placeholder is not None:
            video_placeholder.markdown(render_idle_view(), unsafe_allow_html=True)
        signals_placeholder.markdown(
            '<div style="color:#7486A0;font-size:11.5px;padding:6px 0;">'
            'No session running.</div>', unsafe_allow_html=True)
        guidance_placeholder.markdown(
            '<div style="color:#7486A0;font-size:11.5px;padding:6px 0;">'
            'Guidance appears once a session starts.</div>', unsafe_allow_html=True)
        agent_panel_placeholder.empty()

        # ---------- SELF-REPORT PROMPT (Phase 3 labeling loop) ----------
        # Outcome of the last rating click, shown once. Kept across the
        # st.rerun() the click handler makes - a plain st.warning() there
        # vanished immediately, and a failure looked like nothing happened.
        # ---------- Recording save outcome ----------
        # A failed save stays on screen (not a one-run message) with a way
        # out, and blocks starting over so the recording cannot be lost by
        # accident. Other outcomes are shown once.
        if st.session_state.save_failed:
            with st.container(border=True):
                st.error(f"Your last recording was NOT saved "
                         f"({st.session_state.get('save_error', 'error')}). It is still held "
                         f"in this browser tab: retry, or discard it. Reloading or closing "
                         f"the tab will lose it.")
                _rc, _dc = st.columns(2)
                if _rc.button("Retry saving", key="retry_save", type="primary",
                              width='stretch'):
                    save_current_session()
                    st.rerun()
                if _dc.button("Discard this recording", key="discard_unsaved",
                              width='stretch'):
                    discard_unsaved_recording()
                    st.rerun()
        _save_notice = st.session_state.pop('save_notice', None)
        if _save_notice is not None:
            {'success': st.success, 'warning': st.warning,
             'info': st.info}.get(_save_notice[0], st.info)(_save_notice[1])
        _notice = st.session_state.pop('selfrep_notice', None)
        if _notice is not None:
            (st.success if _notice[0] == 'success' else st.warning)(_notice[1])
        if st.session_state.pending_self_report is not None:
            _sid = st.session_state.pending_self_report
            with st.container(border=True):
                st.markdown('<div class="panel-marker"></div>', unsafe_allow_html=True)
                st.markdown("**How stressed did you actually feel during that "
                            "session?** Your answer is stored with the session "
                            "and is what the system learns from — be honest, "
                            "not agreeable. Rate what YOU personally felt, not "
                            "how it compares to anyone else's stress.")
                _cols = st.columns(6)
                # Explicit anchors on every point, not just the endpoints -
                # "4" and "5" used to be unlabeled, and "5 · very stressed"
                # alone is genuinely ambiguous between "clearly stressed"
                # and "extreme distress." Different people anchor those
                # words differently (e.g. reserving "5" for crying-level
                # distress), which can make a real, moderate stress
                # response get rated a "3" - not a personal-report error,
                # a scale-wording gap. "4" now names the actual target
                # state most recordings should land in: noticeable,
                # uncomfortable, but not overwhelming.
                _labels = [
                    "1 · calm, relaxed",
                    "2 · mild, barely noticeable",
                    "3 · neutral / mixed, hard to tell",
                    "4 · noticeably stressed, tense but OK",
                    "5 · very stressed, overwhelmed",
                ]
                for _i, _lab in enumerate(_labels, start=1):
                    if _cols[_i - 1].button(_lab, key=f"selfrep_{_i}"):
                        # Through the storage backend (Supabase when hosted,
                        # SQLite locally) - it used to call the SQLite module
                        # directly, which has no tables on the hosted server.
                        # The store checks the session belongs to this user.
                        try:
                            get_store().set_self_report(
                                _sid, _i, user_id=st.session_state.active_user_id)
                        except (LookupError, ValueError) as e:
                            st.session_state.selfrep_notice = (
                                'warning', f"Your rating was NOT saved: {e}.")
                        except Exception:
                            # Server error text can carry request details;
                            # the user only needs to know it failed.
                            st.session_state.selfrep_notice = (
                                'warning', "Your rating was NOT saved: the database "
                                           "did not confirm it. Try again, or Skip.")
                        else:
                            # So the CSV/PDF export for THIS same session
                            # (still on screen, since Stop/failure don't
                            # clear history) reflects the answer right
                            # away instead of only after re-opening this
                            # session later via "View"
                            # (BUG_AUDIT_2026-08-23.md #8).
                            if _sid == st.session_state.last_saved_session_id:
                                st.session_state.last_saved_self_report = _i
                            st.session_state.pending_self_report = None
                            st.session_state.selfrep_notice = (
                                'success', "Thanks - your rating was saved with that session.")
                            st.session_state.pop('personalization', None)   # refit on next view
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
                st.markdown(f'<span style="color:#A3B1C6;font-size:12px;">Session length '
                           f'<b style="color:#E8EDF5;">{duration:.0f}s</b> &middot; samples '
                           f'<b style="color:#E8EDF5;">{len(history)}</b></span>',
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
                st.caption(f"Saves as `{_fname}` — drop it into "
                           f"`evaluation_data/incoming/`, run `python recording_intake.py`, "
                           f"then `python run_all_evaluations.py`.")

                if st.session_state.viewing_session_id:
                    _report_task_mode = (st.session_state.viewing_session_task_mode
                                         or st.session_state.task_mode)
                    _report_self_report = st.session_state.viewing_session_self_report
                    _report_label = st.session_state.viewing_session_label
                else:
                    _report_task_mode = st.session_state.task_mode
                    # last_saved_self_report is populated the moment the
                    # self-report banner above is answered for THIS
                    # session (matched by id) - falls back to None
                    # (unanswered/skipped) otherwise
                    # (BUG_AUDIT_2026-08-23.md #8).
                    _report_self_report = st.session_state.last_saved_self_report
                    _report_label = None
                pdf_bytes = session_report.build_session_report_pdf(
                    history, task_mode=_report_task_mode,
                    self_report=_report_self_report, session_label=_report_label,
                )
                st.download_button(
                    "Download PDF report", pdf_bytes,
                    file_name=f"stress_session_{history[0]['ts'].strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
        else:
            level_placeholder.markdown('', unsafe_allow_html=True)
            tiles_placeholder.markdown(render_tiles(0, 0), unsafe_allow_html=True)
            guidance_placeholder.markdown(
                '<div style="color:#7486A0;font-size:11.5px;">'
                'Guidance appears once a session starts.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
