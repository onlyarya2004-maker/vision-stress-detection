"""
Per-user personalization (personalization.py) - what a returning user gets
from their OWN rated sessions, and what nobody else ever gets from them.

The shift lives in the general model's SMOOTHED FINAL INDEX (the space the
0.24 cutoff was derived in). Layers covered, bottom up:
  * fit()     - the rule: minimums, labels, direction and exact magnitude,
                bounds, leave-one-session-out gate
  * held-out  - synthetic users: does it help on FUTURE sessions it never saw,
                and stay out of the way when it should?
  * load()    - reading one user's history through a store, with fallbacks
  * LiveRun   - applied after smoothing; stored values stay general; works
                with the CNN/GRU missing; clipping; its own hysteresis
  * app       - make_live_run hands the fitted shift to the run
  * hosted    - two users on the real SupabaseStore + PostgREST client
                against a fake database that only shows each their own rows
"""
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from postgrest import SyncPostgrestClient

import config
import daisee_ensemble
import personalization
import session_history
import session_pipeline as sp
from personalization import Personalization, _shift, fit, load
from session_store import SupabaseStore
from test_session_pipeline import _drive, _frame, _live, faked      # noqa: F401 (fixture)
from test_session_store import FakeSupabase

CUT = config.LOW_TO_MODERATE_CUTOFF
SINCE = datetime(2026, 10, 1, 8, 0, 0)       # current baseline took effect


def _ex(rating, mean, n=200):
    return {'self_report': rating, 'mean_index': mean, 'n_index': n}


def _sessions(calm, stressed):
    """Newest first: interleave so neither class is systematically newer."""
    out = []
    for i in range(max(len(calm), len(stressed))):
        if i < len(calm):
            out.append(_ex(1, calm[i]))
        if i < len(stressed):
            out.append(_ex(5, stressed[i]))
    return out


def _expected_shift(calm, stressed):
    n = len(calm) + len(stressed)
    mid = 0.5 * (sum(calm) / len(calm) + sum(stressed) / len(stressed))
    d = n / (n + config.PERSONAL_PRIOR_SESSIONS) * (CUT - mid)
    return max(-config.PERSONAL_MAX_SHIFT, min(config.PERSONAL_MAX_SHIFT, d))


# ---------------------------------------------------------------------------
# fit(): minimums and labels
# ---------------------------------------------------------------------------
READS_LOW = ([0.08, 0.10, 0.12], [0.19, 0.21, 0.22])     # stressed sessions read LOW
READS_HIGH = ([0.27, 0.29, 0.30], [0.40, 0.45, 0.50])    # calm sessions read MODERATE


def test_no_history_means_the_general_model():
    p = fit([])
    assert not p.active and p.shift == 0.0 and 'needs 6 rated sessions' in p.reason


def test_below_the_minimum_the_general_model_runs_unchanged():
    p = fit(_sessions(*READS_LOW)[:-1])                    # 5 sessions
    assert not p.active and p.shift == 0.0 and p.n_sessions == 5


def test_both_classes_are_required_not_just_a_count():
    p = fit(_sessions([0.1] * 10, [0.3]))
    assert not p.active and (p.n_relaxed, p.n_stressed) == (10, 1)


def test_unrated_ambiguous_and_invalid_ratings_are_not_labels():
    noise = ([_ex(None, 0.9)] * 5 + [_ex(3, 0.9)] * 5 +
             [_ex(0, 0.9), _ex(6, 0.9), _ex(True, 0.9), _ex('5', 0.9), _ex(4.0, 0.9)])
    clean = fit(_sessions(*READS_LOW))
    p = fit(noise + _sessions(*READS_LOW))
    assert p == clean and p.active


def test_short_or_broken_sessions_are_ignored():
    junk = [_ex(5, 0.9, n=10), _ex(1, float('nan')), _ex(5, float('inf')),
            _ex(1, -0.2), _ex(5, 1.7), _ex(1, None)]
    assert fit(junk + _sessions(*READS_LOW)) == fit(_sessions(*READS_LOW))
    assert not fit(junk).active


# ---------------------------------------------------------------------------
# fit(): direction and exact magnitude
# ---------------------------------------------------------------------------
def test_user_whose_stress_reads_too_low_gets_an_upward_shift_of_the_exact_size():
    calm, stressed = READS_LOW
    p = fit(_sessions(calm, stressed))
    assert p.active and p.shift > 0
    # mid = (0.100 + 0.2067)/2 = 0.1533; 6/12 * (0.24 - 0.1533) = +0.0433
    assert p.shift == pytest.approx(_expected_shift(calm, stressed))
    assert p.shift == pytest.approx(0.5 * (CUT - 0.5 * (0.10 + 0.62 / 3)))
    # The general cutoff missed every stressed session; validated on held-out folds:
    assert p.loo_general == pytest.approx(0.5)
    assert p.loo_personal == pytest.approx((1.0 + 2 / 3) / 2)


def test_user_whose_calm_reads_too_high_gets_a_downward_shift_of_the_exact_size():
    calm, stressed = READS_HIGH
    p = fit(_sessions(calm, stressed))
    assert p.active and p.shift < 0
    assert p.shift == pytest.approx(_expected_shift(calm, stressed))      # = -0.0642
    assert p.loo_general == pytest.approx(0.5) and p.loo_personal > p.loo_general


def test_after_the_shift_the_users_own_sessions_fall_on_the_right_side():
    for calm, stressed in (READS_LOW, READS_HIGH):
        d = fit(_sessions(calm, stressed)).shift
        assert all(m + d < CUT for m in calm)
        assert sum(m + d >= CUT for m in stressed) >= 2


def test_more_evidence_moves_further_toward_the_full_midpoint_shift():
    few = fit(_sessions([0.08, 0.10, 0.12], [0.19, 0.21, 0.22])).shift
    many = fit(_sessions([0.08, 0.10, 0.12] * 4, [0.19, 0.21, 0.22] * 4)).shift
    full = CUT - 0.5 * (0.10 + 0.62 / 3)
    assert 0 < few < many < full
    assert many == pytest.approx(24 / 30 * full)


# ---------------------------------------------------------------------------
# fit(): when it must stay out of the way
# ---------------------------------------------------------------------------
def test_no_shift_when_the_general_model_already_separates_the_user():
    p = fit(_sessions([0.10, 0.12, 0.15], [0.30, 0.35, 0.40]))
    assert not p.active and p.shift == 0.0
    assert p.loo_general == 1.0 and 'would not have agreed better' in p.reason


def test_no_shift_when_the_general_score_ranks_the_user_the_wrong_way():
    p = fit(_sessions([0.30, 0.32, 0.35], [0.15, 0.18, 0.20]))
    assert not p.active and 'not higher when you felt stressed' in p.reason


def test_no_shift_when_calm_and_stressed_overlap():
    p = fit(_sessions([0.20, 0.30, 0.25], [0.22, 0.28, 0.26]))
    assert not p.active and p.shift == 0.0


def test_a_fold_that_cannot_be_fitted_falls_back_to_the_general_model():
    """2 calm + 4 stressed: holding out a calm session leaves one calm, so
    that fold classifies its session with the GENERAL cutoff (shift 0)."""
    p = fit(_sessions([0.10, 0.12], [0.20, 0.21, 0.22, 0.23]))
    # calm: both correct either way; stressed held-out 0.20 -> 0.2355 (miss),
    # 0.21/0.22/0.23 -> hit. Balanced = (2/2 + 3/4) / 2.
    assert p.active and p.loo_personal == pytest.approx(0.875)
    assert p.loo_general == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fit(): bounds and stability
# ---------------------------------------------------------------------------
def test_the_shift_is_bounded_both_ways_and_can_be_active_at_the_bound():
    up = fit(_sessions([0.05] * 15, [0.16] * 15))       # unbounded would be +0.1125
    down = fit(_sessions([0.32] * 15, [0.45] * 15))     # unbounded would be -0.1208
    assert up.active and up.shift == pytest.approx(config.PERSONAL_MAX_SHIFT)
    assert down.active and down.shift == pytest.approx(-config.PERSONAL_MAX_SHIFT)


def test_shift_helper_bounds_extreme_inputs():
    pts = [(False, 0.0)] * 15 + [(True, 0.01)] * 15
    assert _shift(pts) == pytest.approx(config.PERSONAL_MAX_SHIFT)
    pts = [(False, 0.99)] * 15 + [(True, 1.0)] * 15
    assert _shift(pts) == pytest.approx(-config.PERSONAL_MAX_SHIFT)


def test_only_the_newest_sessions_are_used():
    newest = _sessions([0.08, 0.10, 0.12] * 5, [0.19, 0.21, 0.22] * 5)    # 30
    old = _sessions([0.9] * 40, [0.95] * 40)
    assert fit(newest + old) == fit(newest)
    assert fit(newest + old).n_sessions == config.PERSONAL_MAX_SESSIONS


def test_fit_is_deterministic():
    data = _sessions(*READS_HIGH)
    assert fit(data) == fit(list(data))


# ---------------------------------------------------------------------------
# held-out: future sessions the fit never saw (synthetic users)
# ---------------------------------------------------------------------------
def _user(rng, mu_calm, mu_stressed, sd, n):
    calm = np.clip(rng.normal(mu_calm, sd, n), 0, 1)
    stressed = np.clip(rng.normal(mu_stressed, sd, n), 0, 1)
    return list(calm), list(stressed)


def _future_bal_acc(calm, stressed, d):
    rec_c = np.mean([m + d < CUT for m in calm])
    rec_s = np.mean([m + d >= CUT for m in stressed])
    return 0.5 * (rec_c + rec_s)


def _simulate(mu_calm, mu_stressed, sd=0.04, history=6, seeds=40):
    active, gen, per = 0, [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        h_c, h_s = _user(rng, mu_calm, mu_stressed, sd, history)
        f_c, f_s = _user(rng, mu_calm, mu_stressed, sd, 200)        # never seen
        p = fit(_sessions(h_c, h_s))
        active += p.active
        gen.append(_future_bal_acc(f_c, f_s, 0.0))
        per.append(_future_bal_acc(f_c, f_s, p.shift))
    return active / seeds, float(np.mean(gen)), float(np.mean(per)), np.array(per) - np.array(gen)


def test_held_out_a_user_the_general_model_reads_too_high_is_helped():
    """Calm sessions sit above the cutoff for this user (the 'false alarm'
    case). On 200 future sessions per seed, the personalized cutoff agrees
    with the user's labels better than the general one."""
    rate, gen, per, diff = _simulate(0.29, 0.42)
    assert rate >= 0.8
    assert per > gen + 0.15
    assert (diff >= 0).mean() >= 0.9


def test_held_out_a_user_the_general_model_reads_too_low_is_helped():
    rate, gen, per, diff = _simulate(0.10, 0.22)
    assert rate >= 0.8 and per > gen + 0.15


def test_held_out_a_well_served_user_is_left_alone():
    rate, gen, per, diff = _simulate(0.12, 0.38)
    assert rate <= 0.1
    assert per >= gen - 1e-9


def test_held_out_no_signal_user_is_not_made_worse_on_average():
    """Calm and stressed look identical to the general model: nothing to
    learn. The gate should rarely fire, and must not cost accuracy overall."""
    rate, gen, per, diff = _simulate(0.25, 0.25, sd=0.06)
    assert rate <= 0.35
    assert per >= gen - 0.02


# ---------------------------------------------------------------------------
# load(): one user's history through a store
# ---------------------------------------------------------------------------
class ListStore:
    def __init__(self, sessions, means, fail=False):
        self.sessions, self.means, self.fail = sessions, means, fail
        self.stats_calls = []

    def labeled_sessions(self, min_samples=10):
        if self.fail:
            raise ConnectionError("offline secret-ish detail")
        return [dict(s) for s in self.sessions if s['sample_count'] >= min_samples]

    def session_index_stats(self, sid):
        self.stats_calls.append(sid)
        return self.means[sid]


def _rows(user, ratings, start_id=1, mode='Default', start=SINCE + timedelta(hours=1)):
    return [{'id': start_id + i, 'user_id': user, 'self_report': r, 'sample_count': 200,
             'task_mode': mode, 'started_at': (start + timedelta(hours=i)).isoformat()}
            for i, r in enumerate(ratings)]


def _store_for(calm, stressed, user='alice', start_id=1):
    ratings = [1] * len(calm) + [5] * len(stressed)
    rows = _rows(user, ratings, start_id)
    means = {r['id']: (200, m) for r, m in zip(rows, calm + stressed)}
    return rows, means


def test_load_uses_only_the_requested_users_rows():
    a_rows, a_means = _store_for(*READS_LOW)
    b_rows, b_means = _store_for([0.9] * 5, [0.95] * 5, user='bob', start_id=101)
    store = ListStore(a_rows + b_rows, {**a_means, **b_means})
    p = load(store, 'alice', baseline_set_at=SINCE)
    assert p.active and p.shift == pytest.approx(_expected_shift(*READS_LOW))
    assert all(sid < 100 for sid in store.stats_calls), "read another user's samples"


def test_load_does_not_fetch_samples_for_unusable_ratings():
    rows = _rows('alice', [None, 3, 3, 1, 5])
    store = ListStore(rows, {i: (200, 0.2) for i in range(1, 6)})
    load(store, 'alice', baseline_set_at=SINCE)
    assert sorted(store.stats_calls) == [4, 5]


def test_load_falls_back_and_logs_only_the_error_type(capsys):
    p = load(ListStore([], {}, fail=True), 'alice', baseline_set_at=SINCE)
    assert p == personalization.general("your history could not be loaded right now.")
    out = capsys.readouterr().out
    assert 'ConnectionError' in out and 'secret-ish' not in out


def test_empty_sessions_do_not_take_the_newest_slots():
    """30 newer rated sessions without usable samples must not crowd out the
    usable older ones."""
    empty = _rows('alice', [1, 5] * 15, start_id=100)
    rows, means = _store_for(*READS_LOW)
    means.update({r['id']: (0, None) for r in empty})
    p = load(ListStore(rows + empty, means), 'alice', baseline_set_at=SINCE)
    assert p.active and p.n_sessions == 6


def test_the_cache_is_reused_and_gives_the_same_answer():
    rows, means = _store_for(*READS_LOW)
    store, cache = ListStore(rows, means), {}
    first = load(store, 'alice', cache=cache, baseline_set_at=SINCE)
    calls = len(store.stats_calls)
    assert load(store, 'alice', cache=cache, baseline_set_at=SINCE) == first and len(store.stats_calls) == calls


def test_local_sqlite_backend_end_to_end(temp_db):
    """The same path on the local SQLite store, per local profile, reading
    the stored stress_index."""
    session_history.init_db()
    alice = session_history.create_user("Alice")
    bob = session_history.create_user("Bob")

    def save(user, rating, index):
        hist = [{'ts': sp.datetime(2026, 10, 1, 9, 0, 0), 'stress_index': index,
                 'level': 'LOW', 'daisee': None, 'behavior': 0.5} for _ in range(80)]
        sid = session_history.save_session(hist, user_id=user)
        if rating is not None:
            session_history.set_self_report(sid, rating, user_id=user)

    calm, stressed = READS_LOW
    for c, s in zip(calm, stressed):
        save(alice, 1, c)
        save(alice, 5, s)
        save(bob, None, 0.9)
    assert session_history.session_index_stats(1) == (80, pytest.approx(calm[0]))
    p_a = load(session_history, alice, baseline_set_at=SINCE)
    p_b = load(session_history, bob, baseline_set_at=SINCE)
    assert p_a.active and p_a.shift == pytest.approx(_expected_shift(calm, stressed))
    assert not p_b.active and p_b.shift == 0.0


def test_local_mode_does_not_personalize_unless_switched_on(monkeypatch):
    assert personalization.enabled_for('supabase')
    assert not personalization.enabled_for('sqlite')
    monkeypatch.setattr(config, 'PERSONALIZE_LOCAL_SQLITE', True)
    assert personalization.enabled_for('sqlite')
    monkeypatch.setattr(config, 'PERSONALIZATION_ENABLED', False)
    assert not personalization.enabled_for('supabase')


# ---------------------------------------------------------------------------
# LiveRun: applied after smoothing; stored values stay general
# ---------------------------------------------------------------------------
def _active(shift):
    return Personalization(shift, True, 6, 3, 3, "validated", 0.5, 0.83)


def _run(state, p=None):
    run = _live(state)
    if p is not None:
        run.personalization, run.personal_shift = p, p.shift
    return run


def test_without_personalization_the_output_is_the_general_reading(faked):
    state, _ = faked
    for out in _drive(_run(state)):
        assert out['smoothed'] == out['general_smoothed']
        assert out['level'] == out['general_level'] and out['personal_shift'] == 0.0


def test_inactive_personalization_changes_nothing(faked):
    state, _ = faked
    from intervention_agent import InterventionAgent
    agent = InterventionAgent(stats={}, mode_stats={}, task_mode='Default')
    run = sp.LiveRun(state, {'pairs': []}, agent, task_mode='Default', agent_enabled=False,
                     personalization=personalization.general("not yet"))
    assert run.personal_shift == 0.0
    for out in _drive(run):
        assert out['smoothed'] == out['general_smoothed']


def test_shift_is_exactly_additive_and_records_stay_general(faked):
    state, _ = faked
    outs = _drive(_run(state, _active(0.07)))
    for out in outs:
        assert out['smoothed'] - out['general_smoothed'] == pytest.approx(0.07, abs=1e-12)
        if out['record'] is not None:
            assert out['record']['stress_index'] == out['general_smoothed']
            assert out['record']['level'] == out['general_level']


@pytest.mark.parametrize('cnn,gru', [(False, True), (True, False), (False, False)])
def test_shift_is_the_same_whether_or_not_the_cnn_and_gru_produced_a_score(
        faked, monkeypatch, cnn, gru):
    """The point of working in the final-index space: unlike a behaviour-score
    offset, the effect does not depend on which fusion branches are present."""
    state, _ = faked
    if not cnn:
        monkeypatch.setattr(daisee_ensemble, 'ensemble_predict', lambda crop, pairs: None)
    if not gru:
        monkeypatch.setattr(state.temporal_inf, 'is_available', lambda: False)
    outs = _drive(_run(state, _active(-0.05)))
    assert all(o['result']['daisee_raw'] is None for o in outs) == (not cnn)
    for out in outs:
        expected = max(0.0, out['general_smoothed'] - 0.05)
        assert out['smoothed'] == pytest.approx(expected, abs=1e-12)


def _const_fusion(value):
    sig = {k: 0.0 for k in config.BEHAVIOR_WEIGHTS}
    return lambda *a, **k: {'stress_index': value, 'contributions': dict(sig),
                            'behavior_stress': 0.0, 'daisee_raw': None}


def test_the_personal_reading_is_clipped_to_the_score_range(faked, monkeypatch):
    state, _ = faked
    monkeypatch.setattr(sp, 'compute_stress_index', _const_fusion(0.97))
    top = _drive(_run(state, _active(0.10)), n=5)[-1]
    assert top['general_smoothed'] == pytest.approx(0.97) and top['smoothed'] == 1.0
    monkeypatch.setattr(sp, 'compute_stress_index', _const_fusion(0.02))
    low = _drive(_run(state, _active(-0.10)), n=5)[-1]
    assert low['smoothed'] == 0.0


def test_the_personal_level_has_its_own_hysteresis(faked, monkeypatch):
    """General 0.20 (LOW) + 0.08 = 0.28: the personal level turns MODERATE
    after the usual confirmation frames; the stored level stays LOW."""
    state, _ = faked
    monkeypatch.setattr(sp, 'compute_stress_index', _const_fusion(0.20))
    outs = _drive(_run(state, _active(0.08)), n=config.HYSTERESIS_CONFIRM_FRAMES + 5)
    assert outs[0]['level'] == 'LOW'                       # not instant
    assert outs[-1]['level'] == 'MODERATE' and outs[-1]['general_level'] == 'LOW'
    assert all(o['record'] is None or o['record']['level'] == 'LOW' for o in outs)


def test_the_coach_acts_on_the_personal_reading(faked, monkeypatch):
    state, _ = faked
    monkeypatch.setattr(sp, 'compute_stress_index', _const_fusion(0.20))
    run = _run(state, _active(0.08))
    run.agent_enabled = True
    seen = []
    monkeypatch.setattr(run.agent, 'update',
                        lambda t, idx, c, general_index=None: seen.append((idx, general_index)))
    _drive(run, n=10)
    assert seen and all(v == pytest.approx(0.28) and g == pytest.approx(0.20) for v, g in seen)


def test_fusion_inputs_and_shared_model_weights_are_untouched(faked, monkeypatch):
    """Same frames, with and without personalization: the fusion receives
    identical arguments (so the CNN/GRU saw identical inputs), and the GRU
    weights are bit-for-bit unchanged."""
    state, _ = faked
    gru_before = {k: np.array(v, copy=True) for k, v in state.temporal_inf.weights.items()}
    real, calls = sp.compute_stress_index, []

    def spy(**kw):
        calls.append({k: (None if v is None else repr(v)) for k, v in kw.items()
                      if k != 'daisee_baseline'})
        return real(**kw)
    monkeypatch.setattr(sp, 'compute_stress_index', spy)
    _drive(_run(state), n=20)
    general_calls, calls[:] = list(calls), []
    state.temporal_inf.reset()
    state.engine.reset_state()
    state2 = sp.UserState()
    try:
        state2.engine.face_mesh.process = state.engine.face_mesh.process
        state2.engine.pose.process = state.engine.pose.process
        _drive(_run(state2, _active(0.1)), n=20)
    finally:
        state2.close()
    assert len(calls) == len(general_calls) and calls == general_calls
    for k, v in state.temporal_inf.weights.items():
        np.testing.assert_array_equal(v, gru_before[k])


# ---------------------------------------------------------------------------
# app: make_live_run hands the fitted personalization to the run
# ---------------------------------------------------------------------------
class _State(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def test_make_live_run_applies_the_signed_in_users_validated_shift(faked, monkeypatch):
    import app
    state, _ = faked
    state.baseline_set_at = SINCE
    ss = _State(task_mode='Default', agent_enabled=False, active_user_id='alice')
    monkeypatch.setattr(app.st, 'session_state', ss)
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'supabase')
    rows, means = _store_for(*READS_LOW)
    run = app.make_live_run(state, {'pairs': []}, ListStore(rows, means))
    assert run.personal_shift > 0
    assert run.personal_shift == pytest.approx(_expected_shift(*READS_LOW))
    assert ss['personalization'].active

    ss2 = _State(task_mode='Default', agent_enabled=False, active_user_id='bob')
    monkeypatch.setattr(app.st, 'session_state', ss2)
    run_b = app.make_live_run(state, {'pairs': []}, ListStore(rows, means))
    assert run_b.personal_shift == 0.0                  # alice's rows are not bob's


# ---------------------------------------------------------------------------
# hosted: two users, a database that only shows each their own rows
# ---------------------------------------------------------------------------
class FakeDB:
    """sessions + samples for several users; every GET is answered only from
    rows whose user_id matches the request's filter (like RLS + the store's
    own filter), so a leak would show up as wrong numbers."""

    def __init__(self):
        self.sessions, self.samples = [], []
        self.server = FakeSupabase()
        self.server.on('GET', 'sessions', self._sessions)
        self.server.on('GET', 'samples', self._samples)

    def add(self, user, rating, index, n=120):
        sid = len(self.sessions) + 1
        self.sessions.append({'id': sid, 'user_id': user, 'self_report': rating,
                              'sample_count': n, 'task_mode': 'Default',
                              'started_at': (SINCE + timedelta(hours=sid)).isoformat()})
        self.samples += [{'session_id': sid, 'user_id': user, 'stress_index': index}] * n

    @staticmethod
    def _eq(p, key):
        v = p.get(key)
        return v[3:] if v and v.startswith('eq.') else None

    def _sessions(self, req):
        p = dict(req.url.params)
        rows = [s for s in self.sessions if s['user_id'] == self._eq(p, 'user_id')]
        if p.get('self_report') == 'not.is.null':
            rows = [s for s in rows if s['self_report'] is not None]
        if 'sample_count' in p:
            rows = [s for s in rows if s['sample_count'] >= int(p['sample_count'][4:])]
        return rows

    def _samples(self, req):
        p = dict(req.url.params)
        assert p.get('select') == 'stress_index'
        uid, sid = self._eq(p, 'user_id'), self._eq(p, 'session_id')
        rows = [{'stress_index': s['stress_index']} for s in self.samples
                if s['user_id'] == uid and str(s['session_id']) == sid]
        start = int(p.get('offset', 0))
        return rows[start:start + int(p.get('limit', 1000))]

    def store_for(self, user):
        http = httpx.Client(transport=httpx.MockTransport(self.server.handler))
        pg = SyncPostgrestClient('http://sb.test/rest/v1', http_client=http)
        auth = SimpleNamespace(user_id=user, client=SimpleNamespace(table=pg.from_),
                               ensure_fresh=lambda: None)
        return SupabaseStore(auth)


@pytest.fixture
def db():
    d = FakeDB()
    calm, stressed = READS_LOW
    for c, s in zip(calm, stressed):
        d.add('uid-alice', 1, c)
        d.add('uid-alice', 5, s)
    d.add('uid-alice', None, 0.99)                 # unrated: must not count
    d.add('uid-bob', 5, 0.05)
    d.add('uid-bob', 1, 0.90)
    return d


def test_hosted_alice_gets_her_shift_and_bob_the_general_model(db):
    p_a = load(db.store_for('uid-alice'), 'uid-alice', baseline_set_at=SINCE)
    p_b = load(db.store_for('uid-bob'), 'uid-bob', baseline_set_at=SINCE)
    assert p_a.active and p_a.n_sessions == 6
    assert p_a.shift == pytest.approx(_expected_shift(*READS_LOW))
    assert not p_b.active and p_b.shift == 0.0
    for req in db.server.requests:
        assert dict(req.url.params).get('user_id') in ('eq.uid-alice', 'eq.uid-bob')


def test_hosted_bobs_new_data_cannot_change_alices_result(db):
    before = load(db.store_for('uid-alice'), 'uid-alice', baseline_set_at=SINCE)
    for _ in range(10):
        db.add('uid-bob', 1, 0.30)
        db.add('uid-bob', 5, 0.42)
    assert load(db.store_for('uid-alice'), 'uid-alice', baseline_set_at=SINCE) == before
    assert load(db.store_for('uid-bob'), 'uid-bob', baseline_set_at=SINCE).active       # bob has his own now


def test_hosted_result_is_reproducible_from_a_fresh_client(db):
    first = load(db.store_for('uid-alice'), 'uid-alice', cache={}, baseline_set_at=SINCE)
    second = load(db.store_for('uid-alice'), 'uid-alice', cache={}, baseline_set_at=SINCE)
    assert first == second and first.active


def test_hosted_a_store_bound_to_bob_cannot_be_asked_for_alices_history(db):
    p = load(db.store_for('uid-bob'), 'uid-alice', baseline_set_at=SINCE)
    assert not p.active and p.n_sessions == 0


# ---------------------------------------------------------------------------
# compatibility: only sessions scored like the coming run are pooled
# ---------------------------------------------------------------------------
class NoCallStore:
    def __getattr__(self, name):
        raise AssertionError(f"history was read ({name}) although it cannot be used")


def _mixed(rows_a, means_a, rows_b, means_b):
    return ListStore(rows_a + rows_b, {**means_a, **means_b})


def test_sessions_from_another_task_mode_are_not_pooled():
    d_rows, d_means = _store_for(*READS_LOW)                         # Default
    s_rows, s_means = _store_for(*READS_HIGH, start_id=101)
    for r in s_rows:
        r['task_mode'] = 'Studying'
    store = _mixed(d_rows, d_means, s_rows, s_means)
    default = load(store, 'alice', task_mode='Default', baseline_set_at=SINCE)
    studying = load(store, 'alice', task_mode='Studying', baseline_set_at=SINCE)
    driving = load(store, 'alice', task_mode='Driving', baseline_set_at=SINCE)
    assert default.shift == pytest.approx(_expected_shift(*READS_LOW)) and default.shift > 0
    assert studying.shift == pytest.approx(_expected_shift(*READS_HIGH)) and studying.shift < 0
    assert not driving.active and 'counting Driving sessions' in driving.reason


def test_a_session_without_a_mode_counts_as_default():
    rows, means = _store_for(*READS_LOW)
    for r in rows:
        r['task_mode'] = None
    assert load(ListStore(rows, means), 'alice', baseline_set_at=SINCE).active


def test_sessions_from_before_the_current_baseline_are_not_pooled():
    new_rows, new_means = _store_for(*READS_LOW)
    old_rows, old_means = _store_for(*READS_HIGH, start_id=101)
    for i, r in enumerate(old_rows):
        r['started_at'] = (SINCE - timedelta(days=3, hours=i)).isoformat()
    p = load(_mixed(new_rows, new_means, old_rows, old_means), 'alice', baseline_set_at=SINCE)
    assert p.shift == pytest.approx(_expected_shift(*READS_LOW))
    # A recalibration after all of them: nothing compatible is left.
    later = SINCE + timedelta(days=30)
    q = load(_mixed(new_rows, new_means, old_rows, old_means), 'alice', baseline_set_at=later)
    assert not q.active and q.shift == 0.0 and 'since your last calibration' in q.reason


def test_unknown_baseline_time_disables_personalization_without_reading_history():
    p = load(NoCallStore(), 'alice', baseline_set_at=None)
    assert not p.active and p.shift == 0.0 and 'calibrate once' in p.reason
    assert not load(NoCallStore(), 'alice', baseline_set_at='not a time').active


def test_unreadable_or_other_clock_start_times_are_excluded():
    rows, means = _store_for(*READS_LOW)
    bad, bad_means = _store_for(*READS_HIGH, start_id=101)
    for r, v in zip(bad, ['garbage', None, '', '2026-10-02T10:00:00+00:00',
                          '2026-10-02T10:00:00Z', 12345]):
        r['started_at'] = v
    p = load(_mixed(rows, means, bad, bad_means), 'alice', baseline_set_at=SINCE)
    assert p.shift == pytest.approx(_expected_shift(*READS_LOW))


def test_parse_local_time():
    from personalization import parse_local_time
    assert parse_local_time('2026-10-02T10:00:00') == datetime(2026, 10, 2, 10)
    assert parse_local_time('2026-10-02T10:00:00.250000') == datetime(2026, 10, 2, 10, 0, 0, 250000)
    assert parse_local_time(datetime(2026, 1, 1)) == datetime(2026, 1, 1)
    for v in (None, '', 'garbage', '2026-10-02T10:00:00+05:30', 7):
        assert parse_local_time(v) is None


def test_baseline_time_round_trips_through_the_saved_profile(faked):
    state, _ = faked
    state.baseline_set_at = datetime(2026, 10, 3, 9, 30)
    payload = state.baseline_payload()
    assert payload['baseline_set_at'] == '2026-10-03T09:30:00'
    other = sp.UserState()
    try:
        other.apply_profile(payload)
        assert other.baseline_set_at == datetime(2026, 10, 3, 9, 30)
        legacy = {k: v for k, v in payload.items() if k != 'baseline_set_at'}
        other.apply_profile(legacy)                     # profile saved before this existed
        assert other.baseline_set_at is None
        other.baseline_set_at = datetime(2026, 10, 3)
        other.apply_profile(None)
        assert other.baseline_set_at is None
        other.baseline_set_at = datetime(2026, 10, 3)
        other.reset_baselines()
        assert other.baseline_set_at is None
    finally:
        other.close()


def test_batch_gate_knows_when_the_browser_reported_its_offset():
    from test_session_pipeline import _batch
    gate = sp.BatchGate()
    gate.accept(_batch(epoch=1, seq=1), 1)
    assert not gate.tz_known
    gate.accept(_batch(epoch=1, seq=2, times=(0.3, 0.4), tz=330), 1)
    assert gate.tz_known and gate.tz_offset_min == 330


def _app_with(monkeypatch, **state):
    import app
    ss = _State(**state)
    monkeypatch.setattr(app.st, 'session_state', ss)
    return app, ss


def test_user_local_now_uses_the_sessions_own_clock(monkeypatch):
    gate = sp.BatchGate()
    gate.tz_offset_min, gate.tz_known = 330, True
    app, _ = _app_with(monkeypatch, batch_gate=gate)
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    expect = datetime.utcnow() + timedelta(minutes=330)
    assert abs((app.user_local_now() - expect).total_seconds()) < 5

    gate.tz_known = False                       # offset not reported yet: latest possible
    assert app.user_local_now() >= datetime.utcnow() + timedelta(hours=14) - timedelta(seconds=5)

    monkeypatch.setattr(config, 'CAPTURE_MODE', 'local')
    assert abs((app.user_local_now() - datetime.now()).total_seconds()) < 5


def test_a_baseline_change_is_stamped_and_drops_the_cached_fit(faked, monkeypatch):
    state, _ = faked
    gate = sp.BatchGate()
    gate.tz_offset_min, gate.tz_known = 0, True
    app, ss = _app_with(monkeypatch, batch_gate=gate, active_user_id=None,
                        personalization=_active(0.05))
    monkeypatch.setattr(config, 'CAPTURE_MODE', 'browser')
    state.baseline_set_at = None
    app.stamp_baseline_change(state)
    assert state.baseline_set_at is not None
    assert abs((state.baseline_set_at - datetime.utcnow()).total_seconds()) < 5
    assert 'personalization' not in ss


def test_current_personalization_refits_when_the_mode_or_baseline_changes(faked, monkeypatch):
    state, _ = faked
    state.baseline_set_at = SINCE
    rows, means = _store_for(*READS_LOW)
    store = ListStore(rows, means)
    app, ss = _app_with(monkeypatch, task_mode='Default', active_user_id='alice')
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'supabase')
    assert app.current_personalization(store, state).active
    ss['task_mode'] = 'Studying'                          # no Studying sessions
    assert not app.current_personalization(store, state).active
    ss['task_mode'] = 'Default'
    assert app.current_personalization(store, state).active
    state.baseline_set_at = SINCE + timedelta(days=30)    # recalibrated since
    assert not app.current_personalization(store, state).active


def test_live_run_finish_hands_the_agent_the_general_reading(faked, monkeypatch):
    state, _ = faked
    monkeypatch.setattr(sp, 'compute_stress_index', _const_fusion(0.20))
    run = _run(state, _active(0.08))
    _drive(run, n=5)
    seen = {}

    def finish(t, idx, general_index=None):
        seen.update(idx=idx, gen=general_index)
        return []
    monkeypatch.setattr(run.agent, 'finish_session', finish)
    run.finish()
    assert seen['idx'] == pytest.approx(0.28) and seen['gen'] == pytest.approx(0.20)
