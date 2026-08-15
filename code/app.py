"""
Vision-Based Behavioral Stress Detection - Web UI (v7)
======================================================
FER2013 emotion branch fully removed after A/B evaluation showed it hurt
live-system accuracy. Active branches: DaiSEE CNN ensemble (re-centered
against personal baseline) + behavioral feature engine. Calibrated
against WESAD physiological reference.

Run:
    streamlit run app.py
"""

import io
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2  import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from behavioral_features import BehavioralFeatureEngine
from stress_fusion import (
    compute_stress_index, TemporalSmoother, stress_level, stress_level_advice,
    HysteresisLevel,
    set_daisee_baseline, reset_daisee_baseline, is_daisee_calibrated,
    VALID_TASK_MODES,
)
from lstm_temporal import LSTMTemporalInference
from app_naming import export_filename
from rppg_live import get_live_rppg, reset_live_rppg
from intervention_agent import InterventionAgent
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

st.set_page_config(
    page_title="Vision-Based Behavioral Stress Detection",
    page_icon="🧠", layout="wide", initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown("""
    <style>
      :root {
        --bg:#0B0F14; --surface:#121820; --surface2:#171F29;
        --border:#232D3A; --border2:#2E3B4B;
        --text:#E6EBF2; --dim:#8A96A8; --faint:#5C6878;
        --blue:#2F6FED; --orange:#F5762B; --green:#2FBF71;
        --red:#E5484D; --amber:#F5A623; --indigo:#7C8CF8;
      }
      html, body, [data-testid="stAppViewContainer"] {
        font-family:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,sans-serif;
      }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
      [data-testid="stStatusWidget"], .stAppDeployButton { display:none !important; }
      [data-testid="stHeader"] { background:transparent; height:0; }
      .block-container { padding-top:1.1rem; max-width:1250px; }
      h3 { font-size:14px !important; font-weight:600 !important;
           letter-spacing:.06em; text-transform:uppercase; color:var(--dim) !important; }

      .stButton>button {
        background:var(--surface2); color:var(--text);
        border:1px solid var(--border2); border-radius:8px;
        font-size:13px; font-weight:600; letter-spacing:.01em;
        padding:.45rem 1rem; transition:border-color .15s, background .15s;
      }
      .stButton>button:hover { border-color:var(--blue); color:#fff; background:#1B2531; }
      .stButton>button[kind="primary"] { background:var(--blue); border-color:var(--blue); color:#fff; }
      .stButton>button[kind="primary"]:hover { background:#2861D6; border-color:#2861D6; }
      .stButton>button:disabled { opacity:.4; }

      .sb-label, .vbsd-section-label, .sidebar-graph-label {
        color:var(--faint); font-size:11px; font-weight:600;
        text-transform:uppercase; letter-spacing:.09em; margin:2px 0 8px 0;
      }

      .vbsd-header {
        background:linear-gradient(180deg,#101722 0%,#0D131C 100%);
        border:1px solid var(--border); border-radius:12px;
        padding:16px 22px; margin-bottom:14px;
        display:flex; justify-content:space-between; align-items:center;
      }
      .vbsd-header-left { display:flex; align-items:center; gap:14px; }
      .vbsd-icon {
        width:40px; height:40px; border-radius:10px;
        background:linear-gradient(135deg,#2F6FED 0%,#1B49B8 100%);
        display:flex; align-items:center; justify-content:center;
      }
      .vbsd-title { color:var(--text); font-size:17px; font-weight:650; letter-spacing:-.01em; }
      .vbsd-subtitle { color:var(--dim); font-size:12px; margin-top:2px; }

      .vbsd-badge {
        display:inline-flex; align-items:center; gap:7px;
        border:1px solid rgba(47,191,113,.35); color:var(--green);
        background:rgba(47,191,113,.07);
        padding:5px 12px; border-radius:999px;
        font-size:11px; font-weight:700; letter-spacing:.08em;
      }
      .vbsd-badge.offline { border-color:var(--border2); color:var(--faint); background:transparent; }
      .vbsd-badge.calibrating { border-color:rgba(47,111,237,.4); color:var(--blue); background:rgba(47,111,237,.07); }
      .vbsd-badge .dot { width:7px; height:7px; border-radius:50%; background:currentColor;
                         animation:pulse 2s infinite; }
      @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.35;} }

      .vbsd-live-tag {
        display:inline-block; background:rgba(229,72,77,.12); color:var(--red);
        border:1px solid rgba(229,72,77,.4);
        padding:3px 10px; border-radius:6px; font-size:11px;
        font-weight:700; letter-spacing:.08em; margin-bottom:6px;
      }
      .vbsd-live-tag.calibrating { background:rgba(47,111,237,.1); color:var(--blue);
                                   border-color:rgba(47,111,237,.4); }

      .vbsd-level-display { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
      .vbsd-level-dot { width:12px; height:12px; border-radius:50%; }
      .vbsd-level-text { font-size:26px; font-weight:750; letter-spacing:.01em; }
      .vbsd-stress-num { color:var(--dim); font-size:12px; margin-top:2px; }
      .vbsd-stress-num strong { font-size:13px; }

      .vbsd-feature-card {
        background:var(--surface); border:1px solid var(--border);
        border-left:3px solid; border-radius:8px;
        padding:8px 12px; margin-bottom:6px;
        display:flex; align-items:center; gap:10px;
      }
      .vbsd-feature-icon {
        width:34px; height:22px; border-radius:5px;
        display:flex; align-items:center; justify-content:center;
        font-size:9px; font-weight:800; letter-spacing:.06em;
        font-family:Consolas,monospace; flex-shrink:0;
      }
      .vbsd-feature-text { flex:1; }
      .vbsd-feature-text .fname { font-size:12.5px; font-weight:650; color:var(--text); }
      .vbsd-feature-text .fdetail { font-size:11px; color:var(--faint); }
      .vbsd-feature-val { font-family:Consolas,monospace; font-size:12px;
                          color:var(--dim); font-weight:600; }

      .vbsd-tile-row { display:flex; gap:8px; margin:10px 0; }
      .vbsd-tile {
        flex:1; background:var(--surface); border:1px solid var(--border);
        border-radius:8px; padding:12px 8px; text-align:center;
      }
      .vbsd-tile .v { font-size:22px; font-weight:700; color:var(--text);
                      font-variant-numeric:tabular-nums; }
      .vbsd-tile .l { font-size:9.5px; color:var(--faint); margin-top:3px;
                      text-transform:uppercase; letter-spacing:.07em; }

      .vbsd-guidance {
        background:var(--surface); border:1px solid var(--border);
        border-radius:10px; padding:14px; margin-top:8px;
      }
      .vbsd-guidance-title {
        color:var(--dim); font-size:11px; font-weight:700;
        text-transform:uppercase; letter-spacing:.09em; margin-bottom:8px;
      }
      .vbsd-guidance-sub { font-size:12.5px; font-weight:650; margin-bottom:6px; }
      .vbsd-guidance-list { color:var(--dim); font-size:12px; margin:0;
                            padding-left:0; list-style:none; }
      .vbsd-guidance-list li { padding:3px 0 3px 16px; position:relative; }
      .vbsd-guidance-list li::before { content:'\2013'; position:absolute; left:0;
                                       color:var(--blue); }

      .calib-banner {
        background:rgba(47,111,237,.06); border:1px solid rgba(47,111,237,.35);
        border-radius:10px; padding:14px 16px; margin-bottom:10px;
      }
      .cb-title { color:var(--blue); font-size:13px; font-weight:700; margin-bottom:4px; }
      .cb-body { color:var(--dim); font-size:12.5px; line-height:1.5; }

      .pill { display:inline-flex; align-items:center; gap:6px;
              font-size:11px; font-weight:650; letter-spacing:.04em;
              padding:3px 10px; border-radius:999px; border:1px solid; }
      .pill .dot { width:6px; height:6px; border-radius:50%; background:currentColor; }
      .pill.ok { color:var(--green); border-color:rgba(47,191,113,.35);
                 background:rgba(47,191,113,.06); }
      .pill.warn { color:var(--amber); border-color:rgba(245,166,35,.35);
                   background:rgba(245,166,35,.06); }

      .sdot { display:inline-block; width:8px; height:8px; border-radius:50%;
              margin-right:2px; }
      .tag { display:inline-block; font-size:9px; font-weight:750; letter-spacing:.08em;
             color:var(--faint); border:1px solid var(--border2);
             padding:1px 6px; border-radius:4px; vertical-align:1px; }
      .tag.active { color:var(--blue); border-color:rgba(47,111,237,.5); }

      .agent-panel {
        background:var(--surface); border:1px solid var(--border);
        border-radius:10px; padding:12px 14px; margin:6px 0 8px 0;
      }
      .agent-panel.on { border-color:rgba(47,111,237,.45);
        background:linear-gradient(180deg,rgba(47,111,237,.07),rgba(47,111,237,.02)); }
      .agent-head { display:flex; align-items:center; gap:8px; }
      .agent-led { width:9px; height:9px; border-radius:50%; background:var(--faint);
                   flex-shrink:0; }
      .agent-panel.on .agent-led { background:var(--blue);
                                   box-shadow:0 0 8px rgba(47,111,237,.9);
                                   animation:pulse 2s infinite; }
      .agent-name { color:var(--text); font-size:12px; font-weight:700; letter-spacing:.05em; }
      .agent-state { margin-left:auto; font-size:10px; font-weight:750;
                     letter-spacing:.08em; color:var(--faint); }
      .agent-panel.on .agent-state { color:var(--blue); }
      .agent-desc { color:var(--faint); font-size:11px; margin-top:6px; line-height:1.45; }
      .agent-mem { color:var(--dim); font-size:11px; margin-top:6px;
                   border-top:1px solid var(--border); padding-top:6px; }
    </style>
    """, unsafe_allow_html=True)


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


@st.cache_resource(show_spinner="Loading LSTM (if available)...")
def get_lstm_inference():
    """Returns LSTMTemporalInference - safe even if LSTM model not trained yet.
    is_available() returns False in that case and fusion ignores it."""
    return LSTMTemporalInference()


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
        color = (255, 200, 0); label = "CALIBRATING"
    else:
        color = {'LOW': (0, 200, 0), 'MODERATE': (0, 200, 255),
                 'HIGH': (0, 0, 255)}[level]
        label = f"{level} ({smoothed_index*100:.0f}%)"
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame_bgr, label, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def render_mini_graph(values, height=120, width=260):
    if len(values) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" '
            f'style="width:100%;height:{height}px;background:#121820;border-radius:8px;">'
            f'<text x="{width/2}" y="{height/2}" font-size="11" fill="#5C6878" '
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
        f'style="width:100%;height:{height}px;background:#121820;border-radius:8px;">'
        f'<polygon points="{area_pts}" fill="rgba(245,118,43,0.15)"/>'
        f'<polyline points="{pts}" fill="none" stroke="#F5762B" stroke-width="2"/>'
        f'</svg>'
    )


def render_full_timeline(history, height=240):
    """3-line timeline: stress_index, daisee, behavior. (No emotion line.)"""
    if len(history) < 2:
        return ""
    width = 700
    pad = 30
    n = len(history)
    def x_of(i): return pad + (i / max(n - 1, 1)) * (width - 2 * pad)
    def y_of(v): return height - pad - v * (height - 2 * pad)
    series = {
        'stress_index': '#F5762B',
        'daisee':       '#2F6FED',
        'behavior':     '#2FBF71',
    }
    paths = []
    for key, color in series.items():
        pts = " ".join(f"{x_of(i):.1f},{y_of(rec[key]):.1f}"
                       for i, rec in enumerate(history))
        paths.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    axes = (
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#232D3A" stroke-width="1"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#232D3A" stroke-width="1"/>'
        f'<text x="6" y="{pad+5}" font-size="10" fill="#8A96A8">1.0</text>'
        f'<text x="6" y="{height-pad+3}" font-size="10" fill="#8A96A8">0.0</text>'
    )
    legend = []
    for i, (key, color) in enumerate(series.items()):
        x = pad + i * 120
        legend.append(
            f'<rect x="{x}" y="6" width="10" height="10" fill="{color}"/>'
            f'<text x="{x+14}" y="15" font-size="11" fill="#B7C1CF">{key}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;'
        f'background:#121820;border-radius:10px;border:1px solid #232D3A;">'
        f'{axes}{"".join(paths)}{"".join(legend)}</svg>'
    )


SIGNAL_DISPLAY = {
    'eye_strain':       ("EYE", "#F5A623", "Eye strain",    "Sustained eyelid closure"),
    'blink_stress':     ("BLK", "#2FBF71", "Blink rate",    "Elevated blink frequency"),
    'brow_tension':     ("BRW", "#E5484D", "Brow tension",  "Furrowed brow"),
    'jaw_tension':      ("JAW", "#F5762B", "Jaw tension",   "Clench / grimace"),
    'head_instability': ("MOV", "#7C8CF8", "Head movement", "Restless motion"),
    'neck_strain':      ("NCK", "#7C8CF8", "Neck strain",   "Forward head posture"),
    'slouch':           ("PST", "#7C8CF8", "Posture",       "Slouching detected"),
}


def render_features(contributions, top_n=4):
    sorted_signals = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    cards_html = []
    for name, value in sorted_signals[:top_n]:
        icon, color, label, detail = SIGNAL_DISPLAY.get(name, ("•", "#232D3A", name, ""))
        if value >= 0.7:   border = "#E5484D"
        elif value >= 0.4: border = "#F5A623"
        elif value >= 0.2: border = "#2FBF71"
        else:              border = "#232D3A"
        cards_html.append(
            f'<div class="vbsd-feature-card" style="border-left-color:{border};">'
            f'<div class="vbsd-feature-icon" style="background:{color}1F;color:{color};">{icon}</div>'
            f'<div class="vbsd-feature-text">'
            f'  <div class="fname">{label}</div>'
            f'  <div class="fdetail">{detail}</div>'
            f'</div>'
            f'<div class="vbsd-feature-val">{value:.2f}</div>'
            f'</div>'
        )
    return ''.join(cards_html)


def render_tiles(blink_rate, ear_value, hr_text="--", hr_note=""):
    """Third tile is the rPPG heart rate. Shown even when W_RPPG = 0, so the
    reading can be checked against a wrist pulse before it is trusted."""
    return (
        f'<div class="vbsd-tile-row">'
        f'<div class="vbsd-tile"><div class="v">{blink_rate:.0f}</div>'
        f'<div class="l">Blinks / min</div></div>'
        f'<div class="vbsd-tile"><div class="v">{ear_value:.2f}</div>'
        f'<div class="l">Eye openness</div></div>'
        f'<div class="vbsd-tile"><div class="v">{hr_text}</div>'
        f'<div class="l">Heart rate{hr_note}</div></div>'
        f'</div>'
    )


def render_guidance(level):
    level_labels = {
        'LOW':      ('Stay relaxed:',          '#2FBF71'),
        'MODERATE': ('How to reduce stress:',  '#F5A623'),
        'HIGH':     ('Recommended right now:', '#E5484D'),
    }
    sub_label, sub_color = level_labels[level]
    items = ''.join(f'<li>{tip}</li>' for tip in GUIDANCE_TIPS[level])
    return (
        f'<div class="vbsd-guidance">'
        f'  <div class="vbsd-guidance-title">Guidance</div>'
        f'  <div class="vbsd-guidance-sub" style="color:{sub_color};">{sub_label}</div>'
        f'  <ul class="vbsd-guidance-list">{items}</ul>'
        f'</div>'
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
    if 'subject_id' not in st.session_state: st.session_state.subject_id = ''
    if 'condition' not in st.session_state: st.session_state.condition = 'unlabeled'
    if 'pending_self_report' not in st.session_state: st.session_state.pending_self_report = None
    if 'agent_enabled' not in st.session_state: st.session_state.agent_enabled = True
    if 'db_initialized' not in st.session_state:
        session_history.init_db()
        st.session_state.db_initialized = True


def main():
    init_state()
    inject_css()

    models = load_models()
    engine = get_feature_engine()
    lstm_inf = get_lstm_inference()

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown('<div class="sidebar-graph-label">LIVE STRESS</div>',
                    unsafe_allow_html=True)
        sidebar_graph_placeholder = st.empty()
        sidebar_graph_placeholder.markdown(render_mini_graph([]),
                                           unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="sb-label">Session control</div>', unsafe_allow_html=True)

        is_fully_calibrated = engine.baseline.calibrated and is_daisee_calibrated()
        if is_fully_calibrated:
            st.markdown('<span class="pill ok"><span class="dot"></span>Calibrated</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill warn"><span class="dot"></span>Not calibrated</span>', unsafe_allow_html=True)

        # ---------- Context selector ----------
        # Reweights behavioral signals to emphasize what matters most in
        # the chosen context. 'Default' preserves the un-modal 89.6% baseline.
        _task_labels = {
            'Default':  'Default (general)',
            'Studying': 'Studying',
            'Hospital': 'Hospital / medical',
            'Driving':  'Driving',
        }
        _current_idx = (
            VALID_TASK_MODES.index(st.session_state.task_mode)
            if st.session_state.task_mode in VALID_TASK_MODES else 0
        )
        _selected_task = st.selectbox(
            "Context",
            options=VALID_TASK_MODES,
            format_func=lambda x: _task_labels.get(x, x),
            index=_current_idx,
            disabled=st.session_state.running or st.session_state.calibrating,
            help="Reweights stress signals for the chosen context. "
                 "'Default' is the un-modal 89.6% baseline. Other modes "
                 "emphasize signals most diagnostic of stress in that context.",
            key="task_select",
        )
        if _selected_task != st.session_state.task_mode:
            st.session_state.task_mode = _selected_task

        # ---------- AI Intervention Agent ----------
        # A real switch: when off, the agent's decision loop does not run at
        # all. Disabled mid-session (like every other control) so the
        # capture loop is not interrupted.
        _on = st.session_state.agent_enabled
        try:
            from intervention_agent import InterventionAgent as _IA
            _mem = _IA(stats=session_history.intervention_stats()).learned_summary()
        except Exception:
            _mem = "learning memory unavailable"
        st.markdown(
            f'<div class="agent-panel {"on" if _on else ""}">'
            f'<div class="agent-head"><span class="agent-led"></span>'
            f'<span class="agent-name">AI INTERVENTION AGENT</span>'
            f'<span class="agent-state">{"ACTIVE" if _on else "STANDBY"}</span></div>'
            f'<div class="agent-desc">Watches all seven stress signals, decides when an '
            f'interruption is worth it, targets the signal driving the stress, then '
            f'measures whether it helped and learns what works for you. '
            f'Runs entirely on this device.</div>'
            f'<div class="agent-mem">{_mem}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Deactivate agent" if _on else "Activate agent",
                     width='stretch', key="agent_btn",
                     disabled=st.session_state.running or st.session_state.calibrating):
            st.session_state.agent_enabled = not _on
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
                st.rerun()

        if st.button("Stop", width='stretch',
                     disabled=not (st.session_state.running or st.session_state.calibrating)):
            st.session_state.running = False
            st.session_state.calibrating = False
            engine._calibrating = False
            # Persist the just-ended session to the local DB
            if st.session_state.history:
                try:
                    _sid = session_history.save_session(
                        st.session_state.history,
                        task_mode=st.session_state.task_mode,
                    )
                    # Queue the self-report prompt for this session. The
                    # user's answer - not the system's prediction - is the
                    # ground-truth label the adaptation loop trains on.
                    st.session_state.pending_self_report = _sid
                    _ivlog = st.session_state.pop('pending_interventions', [])
                    if _ivlog:
                        session_history.log_interventions(_sid, _ivlog)
                except Exception as e:
                    st.warning(f"Could not save session to history DB: {e}")
            st.rerun()

        if st.button("Clear view", width='stretch',
                     help="Clears the on-screen timeline. Saved sessions in the DB are not affected."):
            st.session_state.history = []
            st.session_state.viewing_session_id = None
            st.session_state.viewing_session_label = None
            st.rerun()

        if st.button("Reset baseline", width='stretch'):
            engine.reset_baseline()
            reset_daisee_baseline()
            st.rerun()

        # ---------- Past Sessions ----------
        st.divider()
        st.markdown('<div class="sb-label">Past sessions</div>', unsafe_allow_html=True)
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
                pct_h = s['pct_high']
                pct_m = s['pct_moderate']
                pct_l = s['pct_low']
                # Status dot reflects dominant state
                if pct_h >= 30:
                    _dc = '#E5484D'
                elif pct_m >= 40:
                    _dc = '#F5A623'
                else:
                    _dc = '#2FBF71'
                dot = f'<span class="sdot" style="background:{_dc}"></span>'
                _task = s.get('task_mode', 'Default') or 'Default'
                _task_icon = ('<span class="tag">'
                              + {'Default': 'GEN', 'Studying': 'STU',
                                 'Hospital': 'MED', 'Driving': 'DRV'}.get(_task, 'GEN')
                              + '</span>')

                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    is_active = (st.session_state.viewing_session_id == s['id'])
                    active_marker = ' <span class="tag active">VIEWING</span>' if is_active else ''
                    st.markdown(
                        f"{dot} **{start_label}**  ·  {s['duration_sec']:.0f}s  ·  {_task_icon}{active_marker}"
                        f"<div style='color:#8A96A8;font-size:11px;margin:-4px 0 6px 22px;'>"
                        f"avg {s['mean_stress']:.2f} · "
                        f"L/M/H {pct_l:.0f}/{pct_m:.0f}/{pct_h:.0f}%"
                        f"</div>",
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
                            # Legacy v1 session that has no per-frame data
                            st.session_state.empty_session_warning = (
                                f"Session from {start_label} was saved before per-frame "
                                f"recording was supported. Only the summary stats above "
                                f"are available."
                            )
                        st.rerun()

            if st.button("Delete all history", width='stretch', key="clear_db",
                         help="Deletes every saved session from the local DB. Cannot be undone."):
                session_history.clear_all_sessions()
                st.session_state.history = []
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.rerun()

    # ---------- Header ----------
    if st.session_state.calibrating:
        badge_class = "vbsd-badge calibrating"; badge_label = "CALIBRATING"
    elif st.session_state.running:
        badge_class = "vbsd-badge"; badge_label = "LIVE"
    else:
        badge_class = "vbsd-badge offline"; badge_label = "OFFLINE"

    st.markdown(
        f'<div class="vbsd-header">'
        f'  <div class="vbsd-header-left">'
        f'    <div class="vbsd-icon"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="2 12 6 12 9 4 14 20 17 12 22 12"/></svg></div>'
        f'    <div>'
        f'      <div class="vbsd-title">Vision-Based Behavioral Stress Detection</div>'
        f'      <div class="vbsd-subtitle">Real-time behavioral analysis &middot; camera only &middot; on-device</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="{badge_class}"><span class="dot"></span>{badge_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.calibrating:
        st.markdown(
            '<div class="calib-banner">'
            '<div class="cb-title">Personal baseline calibration</div>'
            f'<div class="cb-body">Sit naturally with a relaxed, neutral expression for '
            f'{CALIBRATION_SECONDS} seconds. Look at the camera and breathe normally '
            '— do not pose, and do not smile: a smiling baseline skews the jaw and '
            'brow signals for the whole session. The system is learning your calm '
            'state.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    col_features, col_video, col_metrics = st.columns([1, 2, 1])

    with col_video:
        live_tag_placeholder = st.empty()
        if st.session_state.calibrating:
            live_tag_placeholder.markdown(
                '<span class="vbsd-live-tag calibrating">CALIBRATING</span>',
                unsafe_allow_html=True)
        elif st.session_state.running:
            live_tag_placeholder.markdown(
                '<span class="vbsd-live-tag">&#9679; REC</span>',
                unsafe_allow_html=True)
        video_placeholder  = st.empty()
        status_placeholder = st.empty()

    with col_features:
        st.markdown('<div class="vbsd-section-label">Stress Level</div>',
                    unsafe_allow_html=True)
        level_placeholder    = st.empty()
        bar_placeholder      = st.empty()
        st.markdown('<div class="vbsd-section-label" style="margin-top:14px;">Detected Features</div>',
                    unsafe_allow_html=True)
        features_placeholder = st.empty()

    with col_metrics:
        st.markdown('<div class="vbsd-section-label">Metrics</div>',
                    unsafe_allow_html=True)
        tiles_placeholder    = st.empty()
        guidance_placeholder = st.empty()

    st.markdown("---")

    # Empty-session warning (legacy v1 sessions with no per-frame data)
    if st.session_state.get('empty_session_warning'):
        st.warning(st.session_state.empty_session_warning)
        del st.session_state.empty_session_warning

    # Viewing banner (when looking at a saved session)
    if st.session_state.viewing_session_id and not st.session_state.running:
        st.info(
            f"Viewing saved session from **{st.session_state.viewing_session_label}**. "
            f"Press Start or Calibrate to begin a new live session."
        )
        st.subheader("Saved session timeline")
    else:
        st.subheader("Session timeline")
    timeline_placeholder = st.empty()

    # ---------- CALIBRATION ----------
    if st.session_state.calibrating:
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
                bar_placeholder.progress(progress, text=f"Calibrating: {int(progress*100)}%")
                level_placeholder.markdown(
                    '<div style="color:#2F6FED;font-size:24px;font-weight:800;">Calibrating&hellip;</div>',
                    unsafe_allow_html=True,
                )

                if elapsed >= CALIBRATION_SECONDS:
                    face_ok = engine.finish_calibration()
                    daisee_ok = False
                    if len(daisee_calib_samples) >= config.MIN_DAISEE_CALIB_SAMPLES:
                        set_daisee_baseline(float(np.median(daisee_calib_samples)))
                        daisee_ok = True
                    st.session_state.calibrating = False
                    if face_ok and daisee_ok:
                        st.success("Calibration complete. Press Start to begin.")
                    elif face_ok or daisee_ok:
                        st.warning("Partial calibration — try again with a clearer view.")
                    else:
                        st.error("Calibration failed.")
                    time.sleep(1.5)
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

        # LSTM gets a fresh buffer per session (clears stale frames from previous session)
        if lstm_inf.is_available():
            lstm_inf.reset()

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
        intervention_placeholder = st.empty()
        agent_status_placeholder = st.empty()
        last_agent_status_update = 0.0

        last_daisee_probs = None
        frame_idx = 0
        last_timeline_update = 0.0
        last_sidebar_update  = 0.0

        try:
            while st.session_state.running:
                ok, frame = cap.read()
                if not ok:
                    status_placeholder.warning("Frame read failed")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_mesh_results = engine.face_mesh.process(rgb)
                face_crop, bbox = crop_face(frame, face_mesh_results)

                if face_crop is None:
                    status_placeholder.info("No face detected — please face the camera")
                    video_placeholder.image(rgb, channels="RGB", width='stretch')
                    frame_idx += 1
                    continue
                status_placeholder.empty()

                if frame_idx % CNN_EVERY_N_FRAMES == 0:
                    last_daisee_probs = ensemble_predict(face_crop, [
                        (models['daisee_mb'],  MOBILENET_SIZE,    mobilenet_preprocess),
                        (models['daisee_eff'], EFFICIENTNET_SIZE, efficientnet_preprocess),
                    ])

                # LSTM: feed every frame to the buffer; predict() internally
                # rate-limits how often the model actually runs.
                lstm_score = None
                if lstm_inf.is_available():
                    lstm_inf.add_frame(face_crop)
                    lstm_score = lstm_inf.predict()

                # rPPG: cheap per frame (a couple of patch means). The heart
                # rate is computed and shown even at W_RPPG = 0 so it can be
                # sanity-checked against a wrist pulse before being trusted.
                rppg.add_frame(time.time(), frame, face_mesh_results)
                rppg_score = rppg.stress_score()

                behavioral_features = engine.extract(rgb)
                if behavioral_features is None:
                    frame_idx += 1
                    continue

                result = compute_stress_index(
                    daisee_probs=last_daisee_probs,
                    behavioral_features=behavioral_features,
                    lstm_score=lstm_score,
                    task_mode=st.session_state.task_mode,
                    rppg_score=rppg_score,
                )
                smoothed = smoother.update(result['stress_index'])
                level = hyst.update(smoothed)
                advice = stress_level_advice(level)

                # ---- intervention agent: decide / act / measure ----
                _iv = None
                if st.session_state.agent_enabled:
                    _iv = agent.update(time.time(), smoothed,
                                       result['contributions'])
                if _iv is not None and (time.time() - _iv.t_shown) <= config.INTERVENTION_SHOW_SECONDS:
                    intervention_placeholder.markdown(
                        f'<div style="background:rgba(245,118,43,0.08);'
                        f'border:1px solid rgba(245,118,43,0.45);'
                        f'border-radius:10px;padding:12px 16px;margin:6px 0;">'
                        f'<div style="color:#F5762B;font-size:10px;font-weight:800;'
                        f'letter-spacing:.09em;margin-bottom:3px;">AI INTERVENTION</div>'
                        f'<div style="color:#E6EBF2;font-weight:650;font-size:14px;">'
                        f'{_iv.title}</div>'
                        f'<div style="color:#B7C1CF;font-size:12px;margin-top:4px;">'
                        f'{_iv.body}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    intervention_placeholder.empty()

                annotate_frame(frame, bbox, level, smoothed)
                video_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", width='stretch',
                )

                color_map = {'LOW': '#2FBF71', 'MODERATE': '#F5A623', 'HIGH': '#E5484D'}
                level_placeholder.markdown(
                    f'<div class="vbsd-level-display">'
                    f'  <div class="vbsd-level-dot" style="background:{color_map[level]};"></div>'
                    f'  <div class="vbsd-level-text" style="color:{color_map[level]};">{level}</div>'
                    f'</div>'
                    f'<div class="vbsd-stress-num">'
                    f'  Stress Index: <strong style="color:{color_map[level]};">{smoothed:.2f}</strong> / 1.00'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                bar_placeholder.progress(min(1.0, max(0.0, smoothed)))

                features_placeholder.markdown(
                    render_features(result['contributions']),
                    unsafe_allow_html=True,
                )
                _hr, _q = rppg.estimate()
                _hr_text = f"{_hr:.0f}" if _hr else "--"
                _hr_note = " (bpm)" if _hr else ""
                tiles_placeholder.markdown(
                    render_tiles(
                        blink_rate=behavioral_features.get('blinks_per_minute', 0.0),
                        ear_value=behavioral_features.get('ear', 0.0),
                        hr_text=_hr_text, hr_note=_hr_note,
                    ),
                    unsafe_allow_html=True,
                )
                guidance_placeholder.markdown(
                    render_guidance(level),
                    unsafe_allow_html=True,
                )

                now = datetime.now()
                if (not st.session_state.history or
                    (now - st.session_state.history[-1]['ts']).total_seconds() >= 0.5):
                    daisee_record = result.get('daisee_raw') or 0.0
                    _rec = {
                        'ts': now,
                        'stress_index': smoothed,
                        'level': level,
                        'daisee':   daisee_record,
                        'behavior': result['behavior_stress'],
                    }
                    # Per-signal values: without these the self-retraining
                    # loop has nothing to learn from.
                    _rec.update(result['contributions'])
                    st.session_state.history.append(_rec)

                now_ts = time.time()
                if now_ts - last_agent_status_update > 1.0:
                    if st.session_state.agent_enabled:
                        _st = agent.status(now_ts)
                        _colors = {'watching': ('#2FBF71', 'rgba(47,191,113,0.07)'),
                                   'warming': ('#7C8CF8', 'rgba(124,140,248,0.07)'),
                                   'showing': ('#F5762B', 'rgba(245,118,43,0.09)'),
                                   'evaluating': ('#F5762B', 'rgba(245,118,43,0.09)'),
                                   'cooldown': ('#8A96A8', 'rgba(138,150,168,0.07)')}
                        _fg, _bg = _colors.get(_st['phase'], ('#8A96A8', '#121820'))
                        agent_status_placeholder.markdown(
                            f'<div style="background:{_bg};border-radius:8px;'
                            f'padding:8px 12px;margin:4px 0;display:flex;'
                            f'align-items:center;gap:8px;">'
                            f'<span style="width:8px;height:8px;border-radius:50%;'
                            f'background:{_fg};animation:pulse 2s infinite;"></span>'
                            f'<span style="color:{_fg};font-size:12px;font-weight:700;">'
                            f'AI AGENT · {_st["phase"].upper()}</span>'
                            f'<span style="color:{_fg};font-size:11px;opacity:0.85;">'
                            f'{_st["detail"]}</span></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        agent_status_placeholder.markdown(
                            '<div style="background:#121820;border-radius:8px;'
                            'padding:8px 12px;margin:4px 0;">'
                            '<span style="color:#5C6878;font-size:12px;font-weight:700;">'
                            'AI AGENT · OFF</span>'
                            '<span style="color:#5C6878;font-size:11px;"> — '
                            'monitoring only, no interventions, no learning</span></div>',
                            unsafe_allow_html=True,
                        )
                    last_agent_status_update = now_ts

                if now_ts - last_sidebar_update > 0.5 and st.session_state.history:
                    recent = [h['stress_index']
                              for h in st.session_state.history[-LIVE_GRAPH_WINDOW:]]
                    sidebar_graph_placeholder.markdown(
                        render_mini_graph(recent), unsafe_allow_html=True,
                    )
                    last_sidebar_update = now_ts

                if now_ts - last_timeline_update > 1.0 and len(st.session_state.history) > 1:
                    timeline_placeholder.markdown(
                        render_full_timeline(st.session_state.history),
                        unsafe_allow_html=True,
                    )
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
        # ---------- SELF-REPORT PROMPT (Phase 3 labeling loop) ----------
        if st.session_state.pending_self_report is not None:
            _sid = st.session_state.pending_self_report
            with st.container(border=True):
                st.markdown("**How stressed did you actually feel during that "
                            "session?** Your answer is stored with the session "
                            "and is what the system learns from - be honest, "
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
            timeline_placeholder.markdown(
                render_full_timeline(st.session_state.history),
                unsafe_allow_html=True,
            )
            sidebar_values = [h['stress_index'] for h in st.session_state.history]
            sidebar_graph_placeholder.markdown(
                render_mini_graph(sidebar_values[-LIVE_GRAPH_WINDOW:]),
                unsafe_allow_html=True,
            )
            history = st.session_state.history
            duration = (history[-1]['ts'] - history[0]['ts']).total_seconds()
            counts = {'LOW': 0, 'MODERATE': 0, 'HIGH': 0}
            for r in history:
                counts[r['level']] = counts.get(r['level'], 0) + 1
            st.write(f"**Session length:** {duration:.0f} s, **samples:** {len(history)}")
            cols = st.columns(3)
            for i, lvl in enumerate(['LOW', 'MODERATE', 'HIGH']):
                pct = (counts[lvl] / len(history)) * 100 if history else 0
                cols[i].metric(f"{lvl}", f"{pct:.0f}%")
            csv_text = history_to_csv(history)
            _fname = export_filename(
                st.session_state.subject_id, st.session_state.condition,
                st.session_state.task_mode, when=history[0]['ts'],
            )
            st.download_button(
                f"Download session CSV",
                csv_text, file_name=_fname, mime="text/csv",
            )
            st.caption(f"Saves as `{_fname}` — move it into `evaluation_data/`, "
                       f"then re-run `python run_all_evaluations.py`.")
        else:
            st.info("Press Calibrate, then Start to begin a session.")


if __name__ == "__main__":
    main()