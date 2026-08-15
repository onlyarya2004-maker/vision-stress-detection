"""
Intervention agent tests.

The agent's contract: never nag, target the actual driver, measure honestly,
and stop suggesting what demonstrably does not work on this user.
"""

import pytest

import config
from intervention_agent import CATALOGUE, InterventionAgent


CALM = {s: 0.1 for s in ('eye_strain', 'blink_stress', 'brow_tension',
                         'jaw_tension', 'head_instability', 'neck_strain',
                         'slouch')}


def _agent(t0=0.0, stats=None):
    a = InterventionAgent(stats=stats or {})
    a.start_session(t0)
    return a


def warmed(t0=0.0):
    return t0 + config.INTERVENTION_WARMUP_SECONDS + 1


# ---------------------------------------------------------------------------
# Decide: when NOT to fire
# ---------------------------------------------------------------------------

def test_no_intervention_when_calm():
    a = _agent()
    assert a.update(warmed(), 0.20, CALM) is None


def test_no_intervention_during_warmup():
    """The smoother is still filling; interrupting at t=5s is noise-driven."""
    a = _agent()
    stressed = dict(CALM, jaw_tension=0.9)
    assert a.update(5.0, 0.80, stressed) is None


def test_cooldown_prevents_nagging():
    a = _agent()
    stressed = dict(CALM, jaw_tension=0.9)
    t = warmed()
    first = a.update(t, 0.80, stressed)
    assert first is not None
    # Card window passes, evaluation resolves, but cooldown still holds.
    t2 = t + config.INTERVENTION_EVAL_SECONDS + 1
    a.update(t2, 0.80, stressed)
    within_cooldown = t + config.INTERVENTION_COOLDOWN_SECONDS - 1
    if within_cooldown > t2:
        assert a.update(within_cooldown, 0.80, stressed) is None or \
            a._last_fired_at != within_cooldown


# ---------------------------------------------------------------------------
# Act: targeting
# ---------------------------------------------------------------------------

def test_targets_the_dominant_signal():
    a = _agent()
    stressed = dict(CALM, slouch=0.95, jaw_tension=0.55)
    iv = a.update(warmed(), 0.70, stressed)
    assert iv is not None
    assert iv.kind == CATALOGUE['slouch']['kind']
    assert iv.trigger_signal == 'slouch'


def test_general_break_when_no_single_driver():
    """High fused index but every signal below threshold -> generic break."""
    a = _agent()
    diffuse = {s: 0.4 for s in CALM}
    iv = a.update(warmed(), 0.75, diffuse)
    assert iv is not None
    assert iv.kind == CATALOGUE['general']['kind']
    assert iv.trigger_signal is None


# ---------------------------------------------------------------------------
# Measure + learn
# ---------------------------------------------------------------------------

def _run_outcome(index_after):
    a = _agent()
    stressed = dict(CALM, jaw_tension=0.9)
    t = warmed()
    iv = a.update(t, 0.80, stressed)
    assert iv is not None
    a.update(t + config.INTERVENTION_EVAL_SECONDS + 1, index_after, stressed)
    return a, iv.kind


def test_index_drop_is_recorded_as_helped():
    a, kind = _run_outcome(0.80 - config.INTERVENTION_HELPED_DROP - 0.02)
    assert a.stats[kind]['helped'] == 1


def test_no_drop_is_recorded_as_no_effect():
    a, kind = _run_outcome(0.80)
    assert a.stats[kind]['no_effect'] == 1


def test_success_rate_prefers_what_worked():
    a = _agent(stats={'unclench_jaw': {'helped': 8, 'no_effect': 1},
                      'reset_posture': {'helped': 1, 'no_effect': 8}})
    assert a.success_rate('unclench_jaw') > 0.7
    assert a.success_rate('reset_posture') < 0.3
    assert a.success_rate('never_seen') == pytest.approx(0.5)


def test_failing_intervention_is_retired_for_this_user():
    """
    An intervention that has repeatedly done nothing must give way to the
    general break rather than being suggested forever.
    """
    a = _agent(stats={'unclench_jaw': {'helped': 0, 'no_effect': 8}})
    stressed = dict(CALM, jaw_tension=0.95)
    iv = a.update(warmed(), 0.80, stressed)
    assert iv is not None
    assert iv.kind == CATALOGUE['general']['kind'], (
        f"retired intervention still suggested: {iv.kind}")


def test_finish_session_hands_over_the_log():
    a, _ = _run_outcome(0.60)
    log = a.finish_session(1000.0, 0.60)
    assert len(log) == 1
    assert log[0]['outcome'] in ('helped', 'no_effect')
    assert 'index_before' in log[0]
    # Log is handed over exactly once.
    assert a.finish_session(1001.0, 0.60) == []


def test_short_session_outcome_is_unknown_not_guessed():
    """Firing 10s before Stop cannot be scored; it must say so."""
    a = _agent()
    stressed = dict(CALM, brow_tension=0.9)
    t = warmed()
    a.update(t, 0.80, stressed)
    log = a.finish_session(t + 10, 0.79)
    assert len(log) == 1
    assert log[0]['outcome'] == 'unknown'
    assert log[0]['index_after'] is None


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

def test_intervention_log_roundtrips_through_db(temp_db):
    import session_history
    from datetime import datetime
    session_history.init_db()
    session_history.log_interventions(None, [{
        'ts': datetime(2026, 8, 16, 12, 0, 0), 'kind': 'unclench_jaw',
        'trigger_signal': 'jaw_tension',
        'index_before': 0.8, 'index_after': 0.6, 'outcome': 'helped',
    }, {
        'ts': datetime(2026, 8, 16, 12, 5, 0), 'kind': 'unclench_jaw',
        'trigger_signal': 'jaw_tension',
        'index_before': 0.7, 'index_after': 0.72, 'outcome': 'no_effect',
    }])
    stats = session_history.intervention_stats()
    assert stats['unclench_jaw'] == {'helped': 1, 'no_effect': 1}


# ---------------------------------------------------------------------------
# Visibility (the UI's window into the agent)
# ---------------------------------------------------------------------------

def test_status_reports_each_phase():
    a = _agent()
    assert a.status(5.0)['phase'] == 'warming'
    assert a.status(warmed())['phase'] == 'watching'

    stressed = dict(CALM, jaw_tension=0.9)
    t = warmed()
    a.update(t, 0.80, stressed)
    assert a.status(t + 1)['phase'] == 'showing'
    assert a.status(t + config.INTERVENTION_SHOW_SECONDS + 5)['phase'] == 'evaluating'

    a.update(t + config.INTERVENTION_EVAL_SECONDS + 1, 0.80, stressed)
    assert a.status(t + config.INTERVENTION_EVAL_SECONDS + 2)['phase'] == 'cooldown'


def test_learned_summary_is_honest_when_empty_and_when_not():
    a = _agent()
    assert 'no measured outcomes' in a.learned_summary()
    a2 = _agent(stats={'unclench_jaw': {'helped': 3, 'no_effect': 1}})
    s = a2.learned_summary()
    assert 'unclench jaw' in s and '3/4' in s
