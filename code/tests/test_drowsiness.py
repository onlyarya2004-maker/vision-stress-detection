"""
Drowsiness warning (drowsiness.py) - Driving mode only, display only.

Pinned down: the timing rules of the monitor; that it exists only in Driving
mode; and, most importantly, that the stress score is IDENTICAL with the
warning running or not and that PERCLOS never appears in the fusion code.
"""
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import config
from drowsiness import DrowsinessMonitor
from intervention_agent import InterventionAgent
from session_pipeline import LiveRun
from test_session_pipeline import _frame, faked  # noqa: F401  (fixture)


def feed(mon, perclos, start, stop, step=0.5):
    out = None
    t = start
    while t <= stop:
        out = mon.update(t, perclos)
        t += step
    return out


def test_defaults_are_the_documented_starting_values():
    assert (config.DROWSY_PERCLOS_LEVEL, config.DROWSY_HOLD_SECONDS,
            config.DROWSY_CLEAR_SECONDS, config.DROWSY_MIN_SESSION_SECONDS) == (0.15, 30.0, 10.0, 30.0)


def test_no_warning_during_the_first_seconds_even_with_high_perclos():
    m = DrowsinessMonitor()
    assert feed(m, 1.0, 0.0, 29.5)['warning'] is False


def test_warns_only_after_perclos_stays_high_for_the_hold_time():
    m = DrowsinessMonitor()
    feed(m, 0.0, 0.0, 29.5)                          # warm-up, eyes open
    assert feed(m, 0.30, 30.0, 59.5)['warning'] is False      # 29.5 s high: not yet
    r = m.update(60.0, 0.30)
    assert r['warning'] is True and r['above_s'] == pytest.approx(30.0)


def test_a_dip_below_the_level_restarts_the_hold_timer():
    m = DrowsinessMonitor()
    feed(m, 0.0, 0.0, 29.5)
    feed(m, 0.30, 30.0, 50.0)
    m.update(50.5, 0.05)                              # eyes open again
    assert feed(m, 0.30, 51.0, 79.5)['warning'] is False      # timer restarted at 51
    assert m.update(81.0, 0.30)['warning'] is True


def test_the_warning_clears_after_the_clear_time_below_the_level():
    m = DrowsinessMonitor()
    feed(m, 0.0, 0.0, 29.5)
    feed(m, 0.30, 30.0, 60.0)
    assert m.warning
    assert feed(m, 0.05, 60.5, 69.5)['warning'] is True       # 9 s below: still warning
    assert m.update(71.0, 0.05)['warning'] is False


def test_perclos_exactly_at_the_level_counts_as_high():
    m = DrowsinessMonitor()
    feed(m, 0.0, 0.0, 29.5)
    assert feed(m, config.DROWSY_PERCLOS_LEVEL, 30.0, 60.0)['warning'] is True


# ---- wired into the live run, Driving only -------------------------------------
def _run(state, mode, perclos):
    orig = state.engine.extract
    state.engine.extract = lambda *a, **k: {**orig(*a, **k), 'perclos': perclos}
    agent = InterventionAgent(stats={}, mode_stats={}, task_mode=mode)
    return LiveRun(state, {'pairs': []}, agent, task_mode=mode, agent_enabled=False,
                   dt0=datetime(2026, 10, 10, 12, 0, 0))


def test_only_driving_mode_has_a_drowsiness_monitor(faked):
    state, _ = faked
    for mode in ('Default', 'Studying', 'Hospital'):
        run = _run(state, mode, 0.5)
        assert run.drowsiness is None
        assert run.process(_frame(), 0.0)['drowsy'] is None
    assert _run(state, 'Driving', 0.5).drowsiness is not None


def test_driving_run_raises_the_warning_after_sustained_closure(faked):
    state, _ = faked
    run = _run(state, 'Driving', 0.40)
    warned_at = None
    for i in range(140):                               # 70 s at 2 frames/s
        t = i * 0.5
        out = run.process(_frame(), t)
        if out['drowsy']['warning'] and warned_at is None:
            warned_at = t
    assert warned_at == pytest.approx(60.0, abs=0.6)   # 30 s warm-up + 30 s hold


def test_the_stress_score_is_identical_with_and_without_the_warning(faked):
    state, _ = faked
    with_mon = _run(state, 'Driving', 0.40)
    without = _run(state, 'Driving', 0.40)
    without.drowsiness = None
    a, b = [], []
    for i in range(60):
        t = i * 0.5
        oa, ob = with_mon.process(_frame(), t), without.process(_frame(), t)
        a.append((oa['smoothed'], oa['result']['stress_index']))
        b.append((ob['smoothed'], ob['result']['stress_index']))
        assert oa['level'] == ob['level']
    # both runs share one engine in this fixture, so allow float rounding noise only
    assert np.allclose(a, b, rtol=0, atol=1e-9)


def test_perclos_never_enters_the_stress_fusion():
    root = Path(__file__).resolve().parent.parent
    assert not re.search(r'perclos', (root / 'stress_fusion.py').read_text(encoding='utf-8'), re.I)
    assert 'perclos' not in config.BEHAVIOR_WEIGHTS
    from temporal_model import TEMPORAL_FEATURE_ORDER
    assert 'perclos' not in TEMPORAL_FEATURE_ORDER
