"""
Stage 2 of personalization.py: per-user signal weighting.

Stage 1 can only slide a user's whole reading. Stage 2 learns which of the
seven behaviour signals rise when THIS user feels stressed, and is used only
when held-out testing on the user's own sessions shows it clearly beats both
the general model and stage 1. Covered here:
  * fit()     - when it switches on, when it must stay off, bounds, privacy of
                the fit (pure function of one user's examples)
  * held-out  - prequential (fit on the past, predict the next session) on
                synthetic users: helps the user stage 1 cannot, never hurts
                a user with no signal
  * load()    - signal means fetched only when stage 2 can run; a store
                without them, or failing, leaves stage 1 working
  * LiveRun   - per-frame adjustment from this frame's signals, smoothed,
                bounded; saved records stay general
"""
import numpy as np
import pytest

import config
import personalization as P
import session_pipeline as sp
from personalization import SIGNALS, Personalization
from test_session_pipeline import _drive, _live, faked      # noqa: F401 (fixture)

CUT = config.LOW_TO_MODERATE_CUTOFF
POP_W = np.array([0.10, 0.15, 0.30, 0.25, 0.05, 0.05, 0.10])


def _user(kind, n=24, seed=0):
    """Synthetic sessions, OLDEST first. kind: 'hidden' (stress shows in
    posture signals the general score barely weighs), 'nosignal' (ratings
    unrelated to the face), 'offset' (general reads this person too high)."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.08, 0.20, size=7)
    sig = {'hidden': [4, 5, 6], 'offset': [2, 3], 'nosignal': []}[kind]
    off = 0.08 if kind == 'offset' else 0.0
    out = []
    for k in range(n):
        stressed = k % 2 == 1
        x = base + rng.normal(0, 0.04, size=7)
        if stressed:
            for j in sig:
                x[j] += 0.30
        x = np.clip(x, 0, 1)
        g = float(np.clip(0.9 * POP_W @ x + 0.06 + off + rng.normal(0, 0.02), 0, 1))
        out.append({'self_report': 5 if stressed else 1, 'n_index': 600, 'mean_index': g,
                    'signals': dict(zip(SIGNALS, map(float, x)))})
    return out


def _newest_first(oldest_first):
    return list(reversed(oldest_first))


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------
def test_hidden_signal_user_gets_signal_weighting_that_beats_stage1():
    ex = _newest_first(_user('hidden'))
    p = P.fit(ex)
    s1 = P._fit_stage1(ex)
    assert p.active and p.model == 'signals'
    assert p.loo_personal >= (s1.loo_personal if s1.active else p.loo_general) \
        + config.PERSONAL_SIGNAL_MIN_GAIN
    w = dict(p.weights)
    # it learned that THIS user's stress shows in posture, not in the brow
    assert min(w['head_instability'], w['neck_strain'], w['slouch']) > max(
        abs(w['brow_tension']), abs(w['eye_strain']))


def test_stage2_needs_more_sessions_than_stage1():
    ex = _newest_first(_user('hidden', n=config.PERSONAL_SIGNAL_MIN_SESSIONS - 1))
    assert P.fit(ex).model != 'signals'


def test_stage2_needs_enough_of_each_class():
    ex = _newest_first(_user('hidden'))
    calm = [e for e in ex if e['self_report'] == 1]
    stressed = [e for e in ex if e['self_report'] == 5][:config.PERSONAL_SIGNAL_MIN_PER_CLASS - 1]
    assert P.fit(calm + stressed).model != 'signals'


def test_no_signal_user_does_not_get_signal_weighting():
    for seed in range(5):
        assert P.fit(_newest_first(_user('nosignal', seed=seed))).model != 'signals'


def test_offset_user_keeps_the_simpler_stage1_shift():
    p = P.fit(_newest_first(_user('offset')))
    assert p.active and p.model == 'shift'


def test_without_signal_data_stage2_never_runs():
    ex = _newest_first(_user('hidden'))
    for e in ex:
        e.pop('signals')
    assert P.fit(ex).model != 'signals'


def test_switching_stage2_off_restores_stage1(monkeypatch):
    monkeypatch.setattr(config, 'PERSONAL_SIGNALS_ENABLED', False)
    assert P.fit(_newest_first(_user('hidden'))).model != 'signals'


def test_fit_is_deterministic():
    ex = _newest_first(_user('hidden'))
    assert P.fit(ex) == P.fit(ex)


def test_adjustment_is_bounded_and_missing_signals_count_as_calm():
    p = P.fit(_newest_first(_user('hidden')))
    lim = config.PERSONAL_MAX_ADJUST
    assert abs(p.adjustment({k: 1.0 for k in SIGNALS})) <= lim
    assert abs(p.adjustment({k: 0.0 for k in SIGNALS})) <= lim
    ref = dict(p.reference)
    assert p.adjustment({}) == pytest.approx(max(-lim, min(lim, p.shift)))
    assert p.adjustment(ref) == pytest.approx(max(-lim, min(lim, p.shift)))
    assert p.adjustment({k: float('nan') for k in SIGNALS}) == pytest.approx(
        max(-lim, min(lim, p.shift)))


def test_describe_names_the_users_main_signals():
    p = P.fit(_newest_first(_user('hidden')))
    text = p.describe()
    assert 'Personal calibration on' in text and 'confidence' in text
    assert any(n in text for n in ('slouch', 'neck strain', 'head instability'))


def test_an_inactive_result_adjusts_nothing():
    assert P.general('x').adjustment({k: 1.0 for k in SIGNALS}) == 0.0


# ---------------------------------------------------------------------------
# held-out: fit only on the past, predict the next session
# ---------------------------------------------------------------------------
def _prequential(sessions_oldest_first, use_stage2):
    rows = []
    for k, s in enumerate(sessions_oldest_first):
        hist = list(reversed(sessions_oldest_first[:k]))
        p = P.fit(hist) if use_stage2 else P._fit_stage1(hist)
        a = p.adjustment(s['signals']) if p.active else 0.0
        rows.append((s['self_report'] == 5, s['mean_index'] + a >= CUT))
    hit = {True: 0, False: 0}
    tot = {True: 0, False: 0}
    for y, yhat in rows:
        tot[y] += 1
        hit[y] += int(y == yhat)
    return 0.5 * (hit[True] / tot[True] + hit[False] / tot[False])


def test_on_future_sessions_stage2_helps_the_user_stage1_cannot():
    gains = [_prequential(_user('hidden', n=30, seed=s), True)
             - _prequential(_user('hidden', n=30, seed=s), False) for s in range(6)]
    assert np.mean(gains) > 0.03


def test_on_future_sessions_stage2_does_not_hurt_a_no_signal_user():
    diffs = [_prequential(_user('nosignal', n=30, seed=s), True)
             - _prequential(_user('nosignal', n=30, seed=s), False) for s in range(6)]
    assert np.mean(diffs) >= -0.01


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------
class SigStore:
    """One user's sessions with index and signal summaries; counts fetches."""

    def __init__(self, sessions_oldest_first, fail_signals=False):
        self.rows, self.stats, self.sig = [], {}, {}
        self.sig_calls = 0
        self.fail = fail_signals
        for i, s in enumerate(sessions_oldest_first, start=1):
            self.rows.append({'id': i, 'user_id': 'u1', 'self_report': s['self_report'],
                              'task_mode': 'Default', 'started_at': '2026-10-02T09:00:00'})
            self.stats[i] = (s['n_index'], s['mean_index'])
            self.sig[i] = s['signals']

    def labeled_sessions(self, min_samples=10):
        return list(self.rows)

    def session_index_stats(self, sid):
        return self.stats[sid]

    def session_signal_means(self, sid):
        self.sig_calls += 1
        if self.fail:
            raise RuntimeError('boom')
        return self.sig[sid]


def _load(store, cache=None):
    return P.load(store, 'u1', cache, task_mode='Default', baseline_set_at='2026-10-01T08:00:00')


def test_load_fetches_signals_and_runs_stage2():
    p = _load(SigStore(_user('hidden')))
    assert p.active and p.model == 'signals'


def test_load_skips_signal_fetches_when_stage2_cannot_run():
    store = SigStore(_user('hidden', n=config.PERSONAL_SIGNAL_MIN_SESSIONS - 1))
    _load(store)
    assert store.sig_calls == 0


def test_a_failing_signal_fetch_leaves_stage1_working(capsys):
    store = SigStore(_user('offset'), fail_signals=True)
    p = _load(store)
    assert p.active and p.model == 'shift'
    assert 'RuntimeError' in capsys.readouterr().out


def test_signal_summaries_are_cached():
    store, cache = SigStore(_user('hidden')), {}
    a = _load(store, cache)
    calls = store.sig_calls
    b = _load(store, cache)
    assert a == b and store.sig_calls == calls


# ---------------------------------------------------------------------------
# LiveRun
# ---------------------------------------------------------------------------
def _signals_personalization(d=0.0, beta=0.5):
    w = tuple((k, beta if k == 'slouch' else 0.0) for k in SIGNALS)
    ref = tuple((k, 0.0) for k in SIGNALS)
    return Personalization(d, True, 12, 6, 6, 'validated', 0.5, 0.9, model='signals',
                           weights=w, reference=ref, confidence='medium')


def test_live_signal_weighting_follows_the_frames_signals_and_records_stay_general(faked):
    state, _ = faked
    from intervention_agent import InterventionAgent
    agent = InterventionAgent(stats={}, mode_stats={}, task_mode='Default')
    p = _signals_personalization(d=0.02)
    run = sp.LiveRun(state, {'pairs': []}, agent, task_mode='Default', agent_enabled=False,
                     personalization=p)
    assert run.personal_signals is p and run.personal_shift == 0.0
    outs = _drive(run)
    assert outs
    for out in outs:
        lim = config.PERSONAL_MAX_ADJUST
        assert abs(out['personal_adjust']) <= lim + 1e-12
        assert out['smoothed'] == pytest.approx(
            float(np.clip(out['general_smoothed'] + out['personal_adjust'], 0, 1)))
        if out['record'] is not None:
            assert out['record']['stress_index'] == out['general_smoothed']
            assert out['record']['level'] == out['general_level']


def test_live_stage1_results_still_use_the_constant_shift(faked):
    state, _ = faked
    from intervention_agent import InterventionAgent
    agent = InterventionAgent(stats={}, mode_stats={}, task_mode='Default')
    p = Personalization(0.05, True, 6, 3, 3, 'validated', 0.5, 0.83)
    run = sp.LiveRun(state, {'pairs': []}, agent, task_mode='Default', agent_enabled=False,
                     personalization=p)
    assert run.personal_signals is None and run.personal_shift == pytest.approx(0.05)
    for out in _drive(run):
        assert out['smoothed'] - out['general_smoothed'] == pytest.approx(0.05, abs=1e-12) or \
            out['smoothed'] in (0.0, 1.0)
