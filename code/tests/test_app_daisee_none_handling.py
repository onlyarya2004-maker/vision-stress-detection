"""
Tests for two pure app.py functions' handling of daisee=None (2026-09-17
fix: the CNN branch not having scored a frame is no longer collapsed into
a fabricated 0.0 - see session_history.py's v7->v8 migration).

Like test_app_analytics.py's session_analytics_dataframe(), these two
functions have no st.* calls and import/run fine under Streamlit's bare
mode - verified before relying on it here. Not a general precedent for
testing app.py's UI code, just these two pure functions.
"""

from datetime import datetime, timedelta

from app import render_full_timeline, history_to_csv


def _history(n=5, daisee_at=None):
    """Synthetic history where `daisee_at` maps frame index -> value
    (None or a float); frames not listed default to 0.5."""
    daisee_at = daisee_at or {}
    start = datetime(2026, 8, 11, 10, 0, 0)
    return [
        {
            'ts': start + timedelta(seconds=i * 0.5),
            'stress_index': 0.3, 'level': 'LOW',
            'daisee': daisee_at.get(i, 0.5), 'behavior': 0.4,
        }
        for i in range(n)
    ]


def test_render_full_timeline_does_not_crash_on_none_daisee():
    """The old code did `v * (height - 2*pad)` directly on rec['daisee'] -
    a None there (CNN branch not scored yet) raised TypeError."""
    history = _history(5, daisee_at={0: None, 1: None, 2: 0.7})
    svg = render_full_timeline(history)
    assert '<svg' in svg
    assert svg.count('<polyline') == 3


def test_render_full_timeline_forward_fills_none_daisee_from_last_real_value():
    """None frames should hold the last real CNN score for display,
    mirroring how the live fusion itself reuses last_daisee_probs between
    CNN evaluations - not silently drawn as if the CNN said 0.0."""
    import re
    history = _history(4, daisee_at={0: 0.8, 1: None, 2: None, 3: 0.2})
    svg = render_full_timeline(history)
    polylines = re.findall(r'<polyline points="([^"]+)"', svg)
    # CNN branch is the second series drawn (Stress index, CNN branch, Behavioral branch)
    daisee_points = polylines[1].split(' ')
    ys = [float(p.split(',')[1]) for p in daisee_points]
    # frame 1 and 2 (None) must plot at the SAME y as frame 0's real 0.8,
    # not at y_of(0.0) - i.e. held flat, not dropped to the bottom.
    assert ys[1] == ys[0]
    assert ys[2] == ys[0]
    assert ys[3] != ys[0]   # frame 3's real 0.2 is a genuinely different value


def test_render_full_timeline_none_before_any_real_value_falls_back_to_zero():
    """If the CNN never ran yet (e.g. no .h5 models loaded all session),
    there is no 'last real value' to hold - falls back to 0.0, same as
    the old default, but only for display."""
    history = _history(3, daisee_at={0: None, 1: None, 2: None})
    svg = render_full_timeline(history)
    assert '<svg' in svg   # renders without raising


def test_history_to_csv_writes_empty_field_for_none_daisee():
    history = _history(3, daisee_at={1: None})
    csv_text = history_to_csv(history)
    lines = csv_text.strip().splitlines()
    assert lines[0] == 'ts,stress_index,level,daisee,emotion,behavior'
    assert lines[2].split(',')[3] == ''       # frame 1's daisee field is empty
    assert lines[1].split(',')[3] == '0.5000'  # frame 0's real value still formatted
