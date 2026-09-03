"""
Intervention agent tests.

The agent's contract: never nag, target the actual driver, measure honestly,
and stop suggesting what demonstrably does not work on this user. Also
covers the 2026-08-31 additions: Thompson-Sampling-driven targeting,
Wilson-CI retirement, trend-aware hold-back, per-task-mode stats with a
safe fallback, multi-signal general-break competition, and explainability.
"""

import numpy as np
import pytest

import config
from intervention_agent import CATALOGUE, InterventionAgent


CALM = {s: 0.1 for s in ('eye_strain', 'blink_stress', 'brow_tension',
                         'jaw_tension', 'head_instability', 'neck_strain',
                         'slouch')}


def _agent(t0=0.0, stats=None, mode_stats=None, task_mode=None, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    a = InterventionAgent(stats=stats or {}, mode_stats=mode_stats or {},
                          task_mode=task_mode, rng=rng)
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


def test_unresolved_intervention_is_not_dropped_if_cooldown_is_short(monkeypatch):
    """
    Nothing normally prevents a second intervention starting before the
    first resolves except INTERVENTION_COOLDOWN_SECONDS (180) happening to
    exceed INTERVENTION_EVAL_SECONDS (150) - an invariant nothing asserts.
    If cooldown is ever retuned shorter than the eval window, the agent
    must still protect an unresolved intervention itself rather than
    silently losing its outcome measurement.
    """
    monkeypatch.setattr(config, 'INTERVENTION_COOLDOWN_SECONDS', 25.0)
    a = _agent()
    stressed = dict(CALM, jaw_tension=0.9)
    t = warmed()
    first = a.update(t, 0.80, stressed)
    assert first is not None
    first_kind = first.kind

    # Between SHOW_SECONDS (20) and EVAL_SECONDS (150): the card may be
    # hidden by the UI, but the first intervention has not resolved yet -
    # and the (shortened) cooldown has already elapsed, which is exactly
    # the scenario that used to fall through to the firing logic below.
    mid = t + config.INTERVENTION_SHOW_SECONDS + 10
    second = a.update(mid, 0.80, stressed)
    assert second is None, "a second intervention fired before the first resolved"
    assert a._active is not None and a._active.kind == first_kind, (
        "the first intervention's outcome was silently dropped")

    # It still resolves normally once its own eval window closes. The
    # smoothed index drops below INTERVENTION_MIN_INDEX here so this same
    # update() call can't immediately fire a replacement (that fall-through
    # firing when still-stressed is normal, separate behaviour - see
    # _run_outcome above); the point of this assertion is only that the
    # first intervention's outcome was actually recorded, not lost.
    a.update(t + config.INTERVENTION_EVAL_SECONDS + 1, 0.10, CALM)
    assert a._active is None
    total = a.stats[first_kind]['helped'] + a.stats[first_kind]['no_effect']
    assert total == 1, "first intervention's outcome was never recorded"


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
    general break rather than being suggested forever. Retirement is now
    Wilson-CI-based (95% confidence the true success rate is below
    INTERVENTION_RETIRE_BELOW=0.25) rather than a raw trial-count floor -
    a 0-success record needs ~12 trials before the CI's upper bound
    actually clears 0.25, so this uses 15 for a comfortable margin (a
    small run of bad luck at, say, 0/4 can no longer retire something
    prematurely, which the old point-estimate rule could).
    """
    a = _agent(stats={'unclench_jaw': {'helped': 0, 'no_effect': 15}})
    stressed = dict(CALM, jaw_tension=0.95)
    iv = a.update(warmed(), 0.80, stressed)
    assert iv is not None
    assert iv.kind == CATALOGUE['general']['kind'], (
        f"retired intervention still suggested: {iv.kind}")


def test_general_break_is_also_retired_when_it_has_also_failed():
    """
    If BOTH the targeted intervention and the general break have
    independently failed enough times (see the note above on why 15, not
    8, trials are needed under the Wilson-CI retirement rule), falling
    back from the retired targeted one must not skip re-checking the
    fallback itself - the agent must stop nagging entirely, not keep
    re-suggesting an equally-proven-useless general break forever.
    """
    a = _agent(stats={
        'unclench_jaw':  {'helped': 0, 'no_effect': 15},
        'short_break':   {'helped': 0, 'no_effect': 15},
    })
    stressed = dict(CALM, jaw_tension=0.95)
    assert a.update(warmed(), 0.80, stressed) is None


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


def test_intervention_stats_by_mode_joins_against_the_owning_session(temp_db):
    """
    No schema change was needed for per-task-mode tracking - the mode
    comes from a join against sessions.task_mode via session_id, which
    already existed. Two sessions in different modes, each with its own
    intervention outcome, must not bleed into each other's bucket.
    """
    import session_history
    from datetime import datetime, timedelta

    session_history.init_db()
    rows = [{'ts': datetime(2026, 8, 16, 9, 0, 0) + timedelta(seconds=i),
            'stress_index': 0.5, 'level': 'MODERATE', 'daisee': 0.5,
            'behavior': 0.5} for i in range(20)]
    driving_id = session_history.save_session(rows, task_mode='Driving')
    studying_id = session_history.save_session(rows, task_mode='Studying')

    session_history.log_interventions(driving_id, [{
        'ts': datetime(2026, 8, 16, 9, 1, 0), 'kind': 'unclench_jaw',
        'trigger_signal': 'jaw_tension',
        'index_before': 0.8, 'index_after': 0.6, 'outcome': 'helped',
    }])
    session_history.log_interventions(studying_id, [{
        'ts': datetime(2026, 8, 16, 9, 1, 0), 'kind': 'unclench_jaw',
        'trigger_signal': 'jaw_tension',
        'index_before': 0.8, 'index_after': 0.79, 'outcome': 'no_effect',
    }])

    by_mode = session_history.intervention_stats_by_mode()
    assert by_mode['Driving']['unclench_jaw'] == {'helped': 1, 'no_effect': 0}
    assert by_mode['Studying']['unclench_jaw'] == {'helped': 0, 'no_effect': 1}


def test_intervention_stats_by_mode_excludes_sessionless_entries(temp_db):
    """An intervention with no session_id has no known context - it must
    still count in the global intervention_stats(), but be absent from
    the mode-aware view rather than guessed into some default bucket."""
    import session_history
    from datetime import datetime

    session_history.init_db()
    session_history.log_interventions(None, [{
        'ts': datetime(2026, 8, 16, 9, 0, 0), 'kind': 'unclench_jaw',
        'trigger_signal': 'jaw_tension',
        'index_before': 0.8, 'index_after': 0.6, 'outcome': 'helped',
    }])
    assert session_history.intervention_stats()['unclench_jaw'] == \
        {'helped': 1, 'no_effect': 0}
    assert session_history.intervention_stats_by_mode() == {}


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


# ---------------------------------------------------------------------------
# Thompson Sampling targeting (2026-08-31)
# ---------------------------------------------------------------------------

def test_large_urgency_gap_always_wins_regardless_of_history():
    """A big, unambiguous urgency gap must never flip, even against a
    strongly-favourable success history for the LESS urgent signal -
    INTERVENTION_EXPLORE_WEIGHT is bounded specifically so exploration only
    matters for close calls, never overrides a clear driver."""
    stats = {CATALOGUE['jaw_tension']['kind']: {'helped': 0, 'no_effect': 20},
            CATALOGUE['slouch']['kind']: {'helped': 20, 'no_effect': 0}}
    for seed in range(10):
        a = _agent(stats=stats, seed=seed)
        stressed = dict(CALM, slouch=0.95, jaw_tension=0.55)
        iv = a.update(warmed(), 0.70, stressed)
        assert iv.trigger_signal == 'slouch', (
            f"seed {seed}: a 0.40 urgency gap was flipped by history")


def test_close_call_lets_history_decide_most_of_the_time():
    """When two signals are close in urgency, the one with a strong
    success history should win noticeably more often than its weaker
    rivals - real Thompson-Sampling exploration, not a fixed deterministic
    blend. A 0.03 urgency gap also invites the general break to compete
    (see INTERVENTION_MULTI_SIGNAL_GAP) - given a weak history too, it
    should lose out to jaw's strong one just like brow does."""
    stats = {CATALOGUE['jaw_tension']['kind']: {'helped': 15, 'no_effect': 1},
            CATALOGUE['brow_tension']['kind']: {'helped': 1, 'no_effect': 15},
            CATALOGUE['general']['kind']: {'helped': 1, 'no_effect': 15}}
    wins = 0
    trials = 60
    for seed in range(trials):
        a = _agent(stats=stats, seed=seed)
        stressed = dict(CALM, jaw_tension=0.55, brow_tension=0.52)
        iv = a.update(warmed(), 0.65, stressed)
        if iv.trigger_signal == 'jaw_tension':
            wins += 1
    assert wins > trials * 0.85, (
        f"the proven-effective signal only won {wins}/{trials} close calls")


def test_untested_intervention_occasionally_gets_tried():
    """The flip side of exploration: a kind with NO track record yet must
    still occasionally win against one with a decent-but-not-perfect
    history, or the agent can never learn about it."""
    stats = {CATALOGUE['jaw_tension']['kind']: {'helped': 3, 'no_effect': 2}}
    wins_for_untested = 0
    trials = 80
    for seed in range(trials):
        a = _agent(stats=stats, seed=seed)
        stressed = dict(CALM, jaw_tension=0.55, brow_tension=0.52)
        iv = a.update(warmed(), 0.65, stressed)
        if iv.trigger_signal == 'brow_tension':
            wins_for_untested += 1
    assert wins_for_untested > 0, (
        "an untested intervention never won a single close call in "
        f"{trials} trials - exploration isn't happening")


# ---------------------------------------------------------------------------
# Wilson-CI retirement (2026-08-31)
# ---------------------------------------------------------------------------

def test_a_small_run_of_bad_luck_does_not_prematurely_retire():
    """The old rule (raw success rate < 0.25 AND >= 4 trials) could retire
    something after just 4 unlucky outcomes. Wilson-CI retirement must not
    - a 0/4 record's confidence interval is still wide enough that we
    cannot be 95% sure the true rate is bad."""
    a = _agent(stats={'unclench_jaw': {'helped': 0, 'no_effect': 4}})
    stressed = dict(CALM, jaw_tension=0.95)
    iv = a.update(warmed(), 0.80, stressed)
    assert iv.kind == 'unclench_jaw', (
        "retired after only 4 outcomes - Wilson-CI retirement should need "
        "more evidence than that")


# ---------------------------------------------------------------------------
# Trend-aware hold-back (2026-08-31)
# ---------------------------------------------------------------------------

def test_holds_back_when_stress_is_already_declining():
    """
    A fresh agent's very FIRST eligible update() always fires (there's no
    trend history yet to judge by) - so this seeds _index_history directly
    with an older, higher reading rather than building it up through
    update() calls, to isolate the trend check itself from the unrelated
    fire/cooldown/resolve interactions a realistic multi-call sequence
    would otherwise tangle it with.
    """
    a = _agent()
    t = warmed()
    a._index_history.append((t - 20, 0.85))   # 20s ago, well above now
    stressed = dict(CALM, jaw_tension=0.9)
    iv = a.update(t, 0.60, stressed)           # still >= MIN_INDEX, but falling
    assert iv is None
    assert a.status(t)['phase'] == 'declining'


def test_still_fires_when_stress_is_rising_not_declining():
    """The trend check must not suppress firing outright - only when the
    index is genuinely falling."""
    a = _agent()
    t = warmed()
    a._index_history.append((t - 20, 0.50))    # lower 20s ago, rising since
    stressed = dict(CALM, jaw_tension=0.9)
    iv = a.update(t, 0.85, stressed)
    assert iv is not None


# ---------------------------------------------------------------------------
# Explainability (2026-08-31)
# ---------------------------------------------------------------------------

def test_status_detail_explains_the_targeted_choice():
    a = _agent(stats={'unclench_jaw': {'helped': 3, 'no_effect': 1}})
    stressed = dict(CALM, jaw_tension=0.9)
    t = warmed()
    a.update(t, 0.80, stressed)
    detail = a.status(t + 1)['detail']
    assert 'jaw tension' in detail
    assert '0.90' in detail
    assert '%' in detail


def test_status_detail_explains_the_general_break_choice():
    a = _agent()
    diffuse = {s: 0.4 for s in CALM}
    t = warmed()
    a.update(t, 0.75, diffuse)
    detail = a.status(t + 1)['detail']
    assert 'several signals elevated' in detail


# ---------------------------------------------------------------------------
# Per-task-mode stats with a safe fallback (2026-08-31)
# ---------------------------------------------------------------------------

def test_uses_mode_specific_stats_once_evidence_threshold_is_met():
    mode_stats = {'Driving': {'unclench_jaw':
                              {'helped': config.INTERVENTION_MODE_MIN_TRIALS, 'no_effect': 0}}}
    global_stats = {'unclench_jaw': {'helped': 0, 'no_effect': 20}}
    a = _agent(stats=global_stats, mode_stats=mode_stats, task_mode='Driving')
    assert a.success_rate('unclench_jaw') > 0.8, (
        "mode-specific evidence (perfect record) should dominate once it "
        "meets the trust threshold, not the poor global record")


def test_falls_back_to_global_stats_when_mode_bucket_is_thin():
    mode_stats = {'Driving': {'unclench_jaw': {'helped': 1, 'no_effect': 0}}}  # below threshold
    global_stats = {'unclench_jaw': {'helped': 0, 'no_effect': 20}}
    a = _agent(stats=global_stats, mode_stats=mode_stats, task_mode='Driving')
    assert a.success_rate('unclench_jaw') < 0.2, (
        "a thin mode-specific bucket should not override the well-"
        "established global record yet")


def test_mode_stats_ignored_when_no_task_mode_set():
    mode_stats = {'Driving': {'unclench_jaw':
                              {'helped': config.INTERVENTION_MODE_MIN_TRIALS, 'no_effect': 0}}}
    global_stats = {'unclench_jaw': {'helped': 0, 'no_effect': 20}}
    a = _agent(stats=global_stats, mode_stats=mode_stats, task_mode=None)
    assert a.success_rate('unclench_jaw') < 0.2


# ---------------------------------------------------------------------------
# Multi-signal general-break competition (2026-08-31)
# ---------------------------------------------------------------------------

def test_general_break_not_offered_when_one_signal_clearly_dominates():
    """Same scenario as test_targets_the_dominant_signal, re-checked across
    many seeds to confirm '__general__' never sneaks in as a winner when
    the urgency gap is large."""
    stats = {CATALOGUE['general']['kind']: {'helped': 20, 'no_effect': 0}}
    for seed in range(10):
        a = _agent(stats=stats, seed=seed)
        stressed = dict(CALM, slouch=0.95, jaw_tension=0.55)
        iv = a.update(warmed(), 0.70, stressed)
        assert iv.kind == CATALOGUE['slouch']['kind']


def test_general_break_can_win_a_close_call_with_a_strong_history():
    """Two close-together signals with a mediocre history each, against a
    general break with a strong one - the break should win noticeably
    more often than either specific signal."""
    stats = {
        CATALOGUE['jaw_tension']['kind']: {'helped': 1, 'no_effect': 1},
        CATALOGUE['brow_tension']['kind']: {'helped': 1, 'no_effect': 1},
        CATALOGUE['general']['kind']: {'helped': 20, 'no_effect': 0},
    }
    general_wins = 0
    trials = 60
    for seed in range(trials):
        a = _agent(stats=stats, seed=seed)
        stressed = dict(CALM, jaw_tension=0.55, brow_tension=0.52)
        iv = a.update(warmed(), 0.65, stressed)
        if iv.kind == CATALOGUE['general']['kind']:
            general_wins += 1
    assert general_wins > trials * 0.5, (
        f"general break only won {general_wins}/{trials} despite a much "
        "stronger track record than either close competitor")


def test_general_break_never_competes_when_only_one_signal_is_elevated():
    """With a single elevated signal, there's nothing to be 'close' to -
    the general break must not be offered as an alternative at all."""
    stats = {CATALOGUE['general']['kind']: {'helped': 20, 'no_effect': 0}}
    for seed in range(10):
        a = _agent(stats=stats, seed=seed)
        stressed = dict(CALM, jaw_tension=0.9)
        iv = a.update(warmed(), 0.80, stressed)
        assert iv.trigger_signal == 'jaw_tension'
