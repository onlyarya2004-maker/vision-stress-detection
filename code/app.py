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
      .vbsd-header {
        background: linear-gradient(180deg, #1a1f2e 0%, #151926 100%);
        border-radius: 12px; padding: 18px 24px; margin-bottom: 16px;
        border: 1px solid #2a3142; display: flex;
        justify-content: space-between; align-items: center;
      }
      .vbsd-header-left { display:flex; align-items:center; gap:14px; }
      .vbsd-icon {
        width:46px; height:46px; border-radius:10px;
        background:linear-gradient(135deg,#6366f1 0%,#a855f7 100%);
        display:flex; align-items:center; justify-content:center;
        font-size:24px;
      }
      .vbsd-title { color:#f1f5f9; font-size:22px; font-weight:700; margin:0; }
      .vbsd-subtitle { color:#94a3b8; font-size:12px; margin:4px 0 0 0; }
      .vbsd-badge {
        background:#064e3b; color:#6ee7b7; padding:6px 14px; border-radius:20px;
        font-size:12px; font-weight:600; letter-spacing:0.5px;
        display:inline-flex; align-items:center; gap:6px;
      }
      .vbsd-badge.offline { background:#44211f; color:#fca5a5; }
      .vbsd-badge.calibrating { background:#1e3a8a; color:#93c5fd; }
      .vbsd-badge .dot {
        width:8px; height:8px; border-radius:50%;
        background:currentColor; animation:pulse 2s infinite;
      }
      @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
      .vbsd-live-tag {
        display:inline-block; background:#dc2626; color:white;
        padding:4px 12px; border-radius:14px; font-size:11px;
        font-weight:700; letter-spacing:0.5px; margin-bottom:6px;
      }
      .vbsd-live-tag.calibrating { background:#2563eb; }
      .vbsd-section-label {
        color:#94a3b8; font-size:11px; text-transform:uppercase;
        letter-spacing:0.7px; margin-bottom:6px; font-weight:600;
      }
      .vbsd-level-display {
        display:flex; align-items:center; gap:12px; margin-bottom:8px;
      }
      .vbsd-level-dot { width:14px; height:14px; border-radius:50%; }
      .vbsd-level-text { font-size:30px; font-weight:800; letter-spacing:0.5px; }
      .vbsd-stress-num { color:#cbd5e1; font-size:13px; margin-top:4px; }
      .vbsd-stress-num strong { font-weight:700; font-size:14px; }
      .vbsd-feature-card {
        background:#1a1f2e; border-left:3px solid;
        padding:9px 14px; margin-bottom:7px; border-radius:6px;
        display:flex; align-items:center; gap:10px;
      }
      .vbsd-feature-icon {
        width:30px; height:30px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:14px; flex-shrink:0;
      }
      .vbsd-feature-text { flex:1; }
      .vbsd-feature-text .fname { font-size:13px; font-weight:700; color:#e2e8f0; }
      .vbsd-feature-text .fdetail { font-size:11px; color:#94a3b8; }
      .vbsd-feature-val {
        font-family:monospace; font-size:12px; color:#cbd5e1; font-weight:600;
      }
      .vbsd-tile-row { display:flex; gap:8px; margin:10px 0; }
      .vbsd-tile {
        flex:1; background:#1a1f2e; border-radius:8px; padding:12px;
        text-align:center; border:1px solid #2a3142;
      }
      .vbsd-tile .v { font-size:24px; font-weight:800; color:#f1f5f9; }
      .vbsd-tile .l { font-size:10px; color:#94a3b8; margin-top:2px;
                      text-transform:uppercase; letter-spacing:0.5px; }
      .vbsd-guidance {
        background:#0f1729; border:1px solid #1e3a8a;
        border-radius:10px; padding:14px; margin-top:8px;
      }
      .vbsd-guidance-title {
        color:#fbbf24; font-size:13px; font-weight:700;
        text-transform:uppercase; letter-spacing:0.5px;
        display:flex; align-items:center; gap:6px; margin-bottom:8px;
      }
      .vbsd-guidance-sub {
        color:#bfdbfe; font-size:13px; font-weight:600; margin-bottom:6px;
      }
      .vbsd-guidance-list { color:#cbd5e1; font-size:12px;
                            margin:0; padding-left:0; list-style:none; }
      .vbsd-guidance-list li { padding:3px 0 3px 18px; position:relative; }
      .vbsd-guidance-list li::before {
        content: '✓'; position:absolute; left:0; color:#3b82f6; font-weight:700;
      }
      .sidebar-graph-label {
        color:#94a3b8; font-size:11px; text-transform:uppercase;
        letter-spacing:0.7px; margin-bottom:4px; font-weight:600;
      }
      .calib-banner {
        background:#1e3a8a30; border:1px solid #3b82f6; border-radius:8px;
        padding:12px; margin-bottom:10px; color:#bfdbfe;
      }
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
            f'style="width:100%;height:{height}px;background:#1a1f2e;border-radius:8px;">'
            f'<text x="{width/2}" y="{height/2}" font-size="11" fill="#64748b" '
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
        f'style="width:100%;height:{height}px;background:#1a1f2e;border-radius:8px;">'
        f'<polygon points="{area_pts}" fill="#facc1530"/>'
        f'<polyline points="{pts}" fill="none" stroke="#facc15" stroke-width="2"/>'
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
        'stress_index': '#facc15',
        'daisee':       '#3b82f6',
        'behavior':     '#10b981',
    }
    paths = []
    for key, color in series.items():
        pts = " ".join(f"{x_of(i):.1f},{y_of(rec[key]):.1f}"
                       for i, rec in enumerate(history))
        paths.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    axes = (
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#475569" stroke-width="1"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#475569" stroke-width="1"/>'
        f'<text x="6" y="{pad+5}" font-size="10" fill="#94a3b8">1.0</text>'
        f'<text x="6" y="{height-pad+3}" font-size="10" fill="#94a3b8">0.0</text>'
    )
    legend = []
    for i, (key, color) in enumerate(series.items()):
        x = pad + i * 120
        legend.append(
            f'<rect x="{x}" y="6" width="10" height="10" fill="{color}"/>'
            f'<text x="{x+14}" y="15" font-size="11" fill="#cbd5e1">{key}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;'
        f'background:#1a1f2e;border-radius:10px;border:1px solid #2a3142;">'
        f'{axes}{"".join(paths)}{"".join(legend)}</svg>'
    )


SIGNAL_DISPLAY = {
    'eye_strain':       ("👁",  "#f59e0b", "Eye Strain",      "Sustained eyelid closure"),
    'blink_stress':     ("👁",  "#10b981", "High Blink Rate", "Elevated blink frequency"),
    'brow_tension':     ("😟",  "#ef4444", "Brow Tension",    "Furrowed brow detected"),
    'jaw_tension':      ("😬",  "#f59e0b", "Jaw Tension",     "Clenched/grimacing"),
    'head_instability': ("⟳",   "#a855f7", "Head Movement",   "Restless head motion"),
    'neck_strain':      ("🦴",  "#a855f7", "Neck Strain",     "Forward head posture"),
    'slouch':           ("🪑",  "#a855f7", "Slouching",       "Poor posture detected"),
}


def render_features(contributions, top_n=4):
    sorted_signals = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    cards_html = []
    for name, value in sorted_signals[:top_n]:
        icon, color, label, detail = SIGNAL_DISPLAY.get(name, ("•", "#475569", name, ""))
        if value >= 0.7:   border = "#ef4444"
        elif value >= 0.4: border = "#f59e0b"
        elif value >= 0.2: border = "#10b981"
        else:              border = "#475569"
        cards_html.append(
            f'<div class="vbsd-feature-card" style="border-left-color:{border};">'
            f'<div class="vbsd-feature-icon" style="background:{color}25;">{icon}</div>'
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
        'LOW':      ('Stay relaxed:',          '#10b981'),
        'MODERATE': ('How to reduce stress:',  '#f59e0b'),
        'HIGH':     ('Recommended right now:', '#ef4444'),
    }
    sub_label, sub_color = level_labels[level]
    items = ''.join(f'<li>{tip}</li>' for tip in GUIDANCE_TIPS[level])
    return (
        f'<div class="vbsd-guidance">'
        f'  <div class="vbsd-guidance-title">💡 Guidance</div>'
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
        st.markdown("### Session control")

        is_fully_calibrated = engine.baseline.calibrated and is_daisee_calibrated()
        if is_fully_calibrated:
            st.caption("✅ Calibrated")
        else:
            st.caption("⚠️ Not calibrated")

        # ---------- Context selector ----------
        # Reweights behavioral signals to emphasize what matters most in
        # the chosen context. 'Default' preserves the un-modal 89.6% baseline.
        _task_labels = {
            'Default':  '⚙️ Default (general)',
            'Studying': '📚 Studying',
            'Hospital': '🏥 Hospital / medical',
            'Driving':  '🚗 Driving',
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

        # ---------- Recording labels ----------
        # Set BEFORE recording; these are baked into the export filename so
        # provenance cannot be lost afterwards. See RECORDING_PROTOCOL.md.
        with st.expander("🏷️ Recording labels", expanded=False):
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
                st.caption("⚠️ Unlabelled — set both fields or this recording "
                           "will not be picked up by the evaluation harness.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎯 Calibrate", width='stretch',
                         disabled=st.session_state.running or st.session_state.calibrating,
                         help=f"Sit naturally for {CALIBRATION_SECONDS} seconds. "
                              f"System learns your personal baseline."):
                st.session_state.calibrating = True
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.rerun()
        with col_b:
            if st.button("▶ Start", width='stretch',
                         disabled=st.session_state.running or st.session_state.calibrating):
                st.session_state.running = True
                st.session_state.session_start = datetime.now()
                st.session_state.history = []
                st.session_state.viewing_session_id = None
                st.session_state.viewing_session_label = None
                st.rerun()

        if st.button("⏸ Stop", width='stretch',
                     disabled=not (st.session_state.running or st.session_state.calibrating)):
            st.session_state.running = False
            st.session_state.calibrating = False
            engine._calibrating = False
            # Persist the just-ended session to the local DB
            if st.session_state.history:
                try:
                    session_history.save_session(
                        st.session_state.history,
                        task_mode=st.session_state.task_mode,
                    )
                except Exception as e:
                    st.warning(f"Could not save session to history DB: {e}")
            st.rerun()

        if st.button("🗑 Clear current view", width='stretch',
                     help="Clears the on-screen timeline. Saved sessions in the DB are not affected."):
            st.session_state.history = []
            st.session_state.viewing_session_id = None
            st.session_state.viewing_session_label = None
            st.rerun()

        if st.button("🔄 Reset baseline", width='stretch'):
            engine.reset_baseline()
            reset_daisee_baseline()
            st.rerun()

        # ---------- Past Sessions ----------
        st.divider()
        st.markdown("### 📊 Past sessions")
        try:
            recent = session_history.load_recent_sessions(n=10)
            total  = session_history.session_count()
        except Exception as e:
            recent, total = [], 0
            st.caption(f"⚠️ Could not read history DB: {e}")

        if total == 0:
            st.caption("No saved sessions yet. Press ⏸ Stop after a session to save it.")
        else:
            st.caption(f"Showing last {len(recent)} of {total} sessions")
            for s in recent:
                start_label = s['started_at'][:16].replace('T', '  ')
                pct_h = s['pct_high']
                pct_m = s['pct_moderate']
                pct_l = s['pct_low']
                # Status dot reflects dominant state
                if pct_h >= 30:
                    dot = "🔴"
                elif pct_m >= 40:
                    dot = "🟡"
                else:
                    dot = "🟢"
                # Task mode icon (legacy v2 sessions default to 'Default')
                _task = s.get('task_mode', 'Default') or 'Default'
                _task_icon = {
                    'Default':  '⚙️',
                    'Studying': '📚',
                    'Hospital': '🏥',
                    'Driving':  '🚗',
                }.get(_task, '⚙️')

                col_info, col_btn = st.columns([3, 1])
                with col_info:
                    is_active = (st.session_state.viewing_session_id == s['id'])
                    active_marker = " ◀ viewing" if is_active else ""
                    st.markdown(
                        f"{dot} **{start_label}**  ·  {s['duration_sec']:.0f}s  ·  {_task_icon}{active_marker}"
                        f"<div style='color:#94a3b8;font-size:11px;margin:-4px 0 6px 22px;'>"
                        f"avg {s['mean_stress']:.2f} · "
                        f"L/M/H {pct_l:.0f}/{pct_m:.0f}/{pct_h:.0f}%"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("👁 View", key=f"view_{s['id']}",
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

            if st.button("🗑 Clear all saved history", width='stretch', key="clear_db",
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
        badge_class = "vbsd-badge"; badge_label = "CAMERA ACTIVE"
    else:
        badge_class = "vbsd-badge offline"; badge_label = "OFFLINE"

    st.markdown(
        f'<div class="vbsd-header">'
        f'  <div class="vbsd-header-left">'
        f'    <div class="vbsd-icon">🧠</div>'
        f'    <div>'
        f'      <div class="vbsd-title">Vision-Based Behavioral Stress Detection</div>'
        f'      <div class="vbsd-subtitle">Real-time analysis of facial expressions and posture</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="{badge_class}"><span class="dot"></span>{badge_label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.calibrating:
        st.markdown(
            '<div class="calib-banner">'
            '<b>📷 Personal Baseline Calibration</b><br>'
            f'Sit naturally with a relaxed expression for {CALIBRATION_SECONDS} seconds. '
            'Look at the camera, breathe normally, do not pose. '
            'The system is learning what calm looks like for YOU.'
            '</div>',
            unsafe_allow_html=True,
        )

    col_features, col_video, col_metrics = st.columns([1, 2, 1])

    with col_video:
        live_tag_placeholder = st.empty()
        if st.session_state.calibrating:
            live_tag_placeholder.markdown(
                '<span class="vbsd-live-tag calibrating">● CALIBRATING</span>',
                unsafe_allow_html=True)
        elif st.session_state.running:
            live_tag_placeholder.markdown(
                '<span class="vbsd-live-tag">● LIVE</span>',
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
            f"📂 Viewing saved session from **{st.session_state.viewing_session_label}**. "
            f"Press ▶ Start or 🎯 Calibrate to begin a new live session."
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
                    '<div style="color:#3b82f6;font-size:24px;font-weight:800;">📷 Calibrating...</div>',
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
                        st.success("✅ Calibrated. Click ▶ Start to begin.")
                    elif face_ok or daisee_ok:
                        st.warning("⚠️ Partial calibration. Try again with a clearer view.")
                    else:
                        st.error("❌ Calibration failed.")
                    time.sleep(1.5)
                    st.rerun()
                time.sleep(0.01)
        finally:
            cap.release()
        return

    # ---------- CAPTURE ----------
    if st.session_state.running:
        if not engine.baseline.calibrated:
            st.warning("⚠️ Personal baseline not set. Click ⏸ Stop then 🎯 Calibrate.")

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

                annotate_frame(frame, bbox, level, smoothed)
                video_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB", width='stretch',
                )

                color_map = {'LOW': '#10b981', 'MODERATE': '#f59e0b', 'HIGH': '#ef4444'}
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
                    st.session_state.history.append({
                        'ts': now,
                        'stress_index': smoothed,
                        'level': level,
                        'daisee':   daisee_record,
                        'behavior': result['behavior_stress'],
                    })

                now_ts = time.time()
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
    else:
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
                f"📥 Download session as CSV",
                csv_text, file_name=_fname, mime="text/csv",
            )
            st.caption(f"Saves as `{_fname}` — move it into `evaluation_data/`, "
                       f"then re-run `python run_all_evaluations.py`.")
        else:
            st.info("Press **🎯 Calibrate** first, then **▶ Start** to begin a session.")


if __name__ == "__main__":
    main()