"""
Tests for session_store.SupabaseStore.

These run the REAL postgrest-py client (the library supabase-py uses for every
table call) against an httpx MockTransport, so what is asserted is the actual
HTTP the store would send to Supabase - URL, filters, bodies - not a Python-
level stub of it. That is the closest to "tested against Supabase" that is
possible without a live project (the live check is scripts/rls_check.py).

What matters most here is isolation: every read/delete carries an explicit
user_id filter (defence in depth behind the database's Row Level Security),
and a caller-supplied user id is never trusted as authority.
"""
import json
import math
from datetime import datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from postgrest import SyncPostgrestClient

import config
import session_history
import session_store
from session_store import SupabaseStore

UID = 'user-111'


class FakeSupabase:
    """Records every request; answers from per-(method, table) handlers."""

    def __init__(self):
        self.requests = []
        self.routes = {}

    def on(self, method, table, fn):
        self.routes[(method, table)] = fn

    def handler(self, request):
        self.requests.append(request)
        table = request.url.path.rsplit('/', 1)[-1]
        fn = self.routes.get((request.method, table))
        if fn is None:
            return httpx.Response(200, json=[])
        out = fn(request)
        if isinstance(out, httpx.Response):
            return out
        return httpx.Response(200, json=out)

    def of(self, method, table):
        return [r for r in self.requests
                if r.method == method and r.url.path.endswith('/' + table)]


class FakeAuth:
    user_id = UID

    def __init__(self, pg):
        self.client = SimpleNamespace(table=pg.from_)
        self.fresh_calls = 0

    def ensure_fresh(self):
        self.fresh_calls += 1


@pytest.fixture
def env():
    server = FakeSupabase()
    http = httpx.Client(transport=httpx.MockTransport(server.handler))
    pg = SyncPostgrestClient('http://sb.test/rest/v1', http_client=http)
    auth = FakeAuth(pg)
    return SimpleNamespace(server=server, store=SupabaseStore(auth), auth=auth)


def _history(n, start=datetime(2026, 10, 10, 9, 0, 0)):
    rows = []
    for i in range(n):
        r = {'ts': start + timedelta(seconds=0.5 * i), 'stress_index': 0.3,
             'level': 'LOW', 'daisee': 0.5 if i % 2 else None, 'behavior': 0.25}
        for c in session_history.SIGNAL_COLUMNS + session_history.TEMPORAL_EXTRA_COLUMNS:
            r[c] = 0.1
        rows.append(r)
    return rows


def _params(req):
    return dict(req.url.params)


# ----------------------------------------------------------------------------
# writes
# ----------------------------------------------------------------------------
def test_save_session_inserts_the_summary_then_samples_in_chunks(env):
    env.server.on('POST', 'sessions', lambda r: [{'id': 42}])
    hist = _history(1000)
    hist[3]['stress_index'] = float('nan')              # not valid JSON
    sid = env.store.save_session(hist, task_mode='Studying', user_id='ignored')
    assert sid == 42

    (sess_req,) = env.server.of('POST', 'sessions')
    body = json.loads(sess_req.content)
    assert body['task_mode'] == 'Studying' and body['sample_count'] == 1000
    assert body['started_at'] == '2026-10-10T09:00:00'
    assert 'user_id' not in body, "owner must come from auth.uid(), never the client"

    sample_reqs = env.server.of('POST', 'samples')
    sizes = [len(json.loads(r.content)) for r in sample_reqs]
    assert sum(sizes) == 1000 and max(sizes) <= 400 and len(sizes) == 3
    first = json.loads(sample_reqs[0].content)[0]
    assert first['session_id'] == 42 and first['ts'] == '2026-10-10T09:00:00'
    assert json.loads(sample_reqs[0].content)[3]['stress_index'] is None    # NaN -> null
    assert first['daisee'] is None and 'user_id' not in first


def test_save_session_removes_the_summary_row_if_samples_fail(env):
    env.server.on('POST', 'sessions', lambda r: [{'id': 9}])
    env.server.on('POST', 'samples', lambda r: httpx.Response(500, json={'message': 'boom'}))
    with pytest.raises(Exception):
        env.store.save_session(_history(5))
    deletes = env.server.of('DELETE', 'sessions')
    assert len(deletes) == 1 and _params(deletes[0])['id'] == 'eq.9'


def test_save_session_with_empty_history_does_nothing(env):
    assert env.store.save_session([]) is None
    assert env.server.requests == []


def test_save_user_baseline_always_writes_the_authenticated_users_row(env):
    env.store.save_user_baseline('someone-else', {'behavioral': {}, 'daisee_baseline': 0.4})
    (req,) = env.server.of('POST', 'profiles')
    body = json.loads(req.content)
    body = body[0] if isinstance(body, list) else body
    assert body['id'] == UID and body['baseline_json']['daisee_baseline'] == 0.4
    assert 'merge-duplicates' in req.headers.get('prefer', '')          # upsert


def test_set_self_report_is_validated_and_scoped_to_the_owner(env):
    env.server.on('GET', 'sessions', lambda r: [{'id': 5}])      # it is this user's session
    env.server.on('PATCH', 'sessions', lambda r: [{'id': 5, 'self_report': 3}])
    with pytest.raises(ValueError):
        env.store.set_self_report(5, 6)
    env.store.set_self_report(5, 3)
    (req,) = env.server.of('PATCH', 'sessions')
    assert _params(req)['id'] == 'eq.5' and _params(req)['user_id'] == f'eq.{UID}'
    assert json.loads(req.content) == {'self_report': 3}
    (check,) = env.server.of('GET', 'sessions')                  # ownership looked up first
    assert _params(check)['id'] == 'eq.5' and _params(check)['user_id'] == f'eq.{UID}'


@pytest.mark.parametrize('bad', [0, 6, -1, 4.5, 3.0, True, '4', [4]])
def test_set_self_report_rejects_anything_off_the_1_to_5_scale(env, bad):
    with pytest.raises(ValueError):
        env.store.set_self_report(5, bad)
    assert env.server.requests == [], "an invalid rating must not reach the database"


def test_set_self_report_refuses_someone_elses_session(env):
    """RLS hides other users' rows, so the ownership lookup comes back empty:
    nothing may be written, and the caller must be told."""
    env.server.on('GET', 'sessions', lambda r: [])
    with pytest.raises(LookupError):
        env.store.set_self_report(999, 4, user_id='someone-else')     # caller id ignored
    assert env.server.of('PATCH', 'sessions') == []
    (check,) = env.server.of('GET', 'sessions')
    assert _params(check)['user_id'] == f'eq.{UID}'


def test_set_self_report_reports_a_zero_row_update_as_a_failure(env):
    env.server.on('GET', 'sessions', lambda r: [{'id': 5}])
    env.server.on('PATCH', 'sessions', lambda r: [])               # matched nothing
    with pytest.raises(RuntimeError):
        env.store.set_self_report(5, 4)


def test_set_self_report_surfaces_server_errors(env):
    env.server.on('GET', 'sessions', lambda r: [{'id': 5}])
    env.server.on('PATCH', 'sessions', lambda r: httpx.Response(500, json={'message': 'boom'}))
    with pytest.raises(Exception):
        env.store.set_self_report(5, 4)


def test_session_index_stats_pages_one_column_of_this_users_session(env):
    page = [{'stress_index': 0.2}] * 1000

    def samples(r):
        return page if _params(r).get('offset', '0') == '0' else (
            [{'stress_index': 0.5}, {'stress_index': None}, {'stress_index': 0.5}])
    env.server.on('GET', 'samples', samples)
    n, mean = env.store.session_index_stats(42)
    assert n == 1002 and mean == pytest.approx((1000 * 0.2 + 2 * 0.5) / 1002)
    reqs = env.server.of('GET', 'samples')
    assert len(reqs) == 2                                    # paged past the 1000-row cap
    for req in reqs:
        p = _params(req)
        assert p['select'] == 'stress_index' and p['order'] == 'id.asc'
        assert p['session_id'] == 'eq.42' and p['user_id'] == f'eq.{UID}'


def test_session_index_stats_of_an_empty_session(env):
    assert env.store.session_index_stats(42) == (0, None)


def test_delete_my_data_only_deletes_the_callers_rows(env):
    env.store.clear_all_sessions()
    (req,) = env.server.of('DELETE', 'sessions')
    assert _params(req) == {'user_id': f'eq.{UID}'}          # never an unfiltered delete


def test_delete_session_is_scoped_to_the_owner(env):
    env.store.delete_session(7)
    (req,) = env.server.of('DELETE', 'sessions')
    assert _params(req) == {'id': 'eq.7', 'user_id': f'eq.{UID}'}


def test_log_interventions_inserts_rows_without_a_user_id(env):
    env.store.log_interventions(3, [{'ts': datetime(2026, 10, 10, 9, 5), 'kind': 'breathing',
                                     'trigger_signal': 'jaw_tension', 'index_before': 0.6,
                                     'index_after': None, 'outcome': 'pending'}])
    (req,) = env.server.of('POST', 'interventions')
    row = json.loads(req.content)[0]
    assert row['session_id'] == 3 and row['kind'] == 'breathing' and row['index_after'] is None
    assert 'user_id' not in row
    env.server.requests.clear()
    env.store.log_interventions(3, [])
    assert env.server.requests == []


# ----------------------------------------------------------------------------
# reads
# ----------------------------------------------------------------------------
def test_recent_sessions_and_count_are_per_user(env):
    env.server.on('GET', 'sessions', lambda r: httpx.Response(
        200, json=[{'id': 1}], headers={'content-range': '0-0/37'}))
    assert env.store.load_recent_sessions(10) == [{'id': 1}]
    assert env.store.session_count() == 37
    for req in env.server.of('GET', 'sessions'):
        assert _params(req)['user_id'] == f'eq.{UID}'
    first = env.server.of('GET', 'sessions')[0]
    assert _params(first)['order'] == 'id.desc' and _params(first)['limit'] == '10'


def test_session_samples_page_through_postgrests_1000_row_cap(env):
    """A 10-minute session stores ~1200 rows; Supabase returns at most 1000 per
    request, so an unpaged read would silently truncate every long session."""
    all_rows = [{'ts': (datetime(2026, 10, 10, 9) + timedelta(seconds=i)).isoformat(),
                 'stress_index': 0.3, 'level': 'LOW', 'daisee': None, 'behavior': 0.2}
                for i in range(1250)]

    def page(req):
        p = _params(req)
        off, lim = int(p.get('offset', 0)), int(p.get('limit', 1000))
        return all_rows[off:off + lim]

    env.server.on('GET', 'samples', page)
    out = env.store.load_session_samples(7)
    assert len(out) == 1250
    assert out[0]['ts'] == datetime(2026, 10, 10, 9) and out[0]['daisee'] is None
    assert out[-1]['ts'] == datetime(2026, 10, 10, 9) + timedelta(seconds=1249)
    reqs = env.server.of('GET', 'samples')
    assert len(reqs) == 2
    for r in reqs:
        assert _params(r)['session_id'] == 'eq.7' and _params(r)['user_id'] == f'eq.{UID}'


def test_labeled_sessions_query_matches_the_sqlite_semantics(env):
    env.store.labeled_sessions(min_samples=10)
    (req,) = env.server.of('GET', 'sessions')
    p = _params(req)
    assert p['self_report'] == 'not.is.null' and p['sample_count'] == 'gte.10'
    assert p['user_id'] == f'eq.{UID}'


def test_intervention_stats_aggregate_only_this_users_measured_outcomes(env):
    env.server.on('GET', 'interventions', lambda r: [
        {'session_id': 1, 'kind': 'breathing', 'outcome': 'helped'},
        {'session_id': 1, 'kind': 'breathing', 'outcome': 'no_effect'},
        {'session_id': 2, 'kind': 'breathing', 'outcome': 'helped'},
        {'session_id': 2, 'kind': 'stretch', 'outcome': 'no_effect'},
        {'session_id': 99, 'kind': 'stretch', 'outcome': 'helped'}])        # unknown session
    env.server.on('GET', 'sessions', lambda r: [
        {'id': 1, 'task_mode': 'Studying'}, {'id': 2, 'task_mode': 'Driving'}])
    assert env.store.intervention_stats() == {
        'breathing': {'helped': 2, 'no_effect': 1}, 'stretch': {'helped': 1, 'no_effect': 1}}
    by_mode = env.store.intervention_stats_by_mode()
    assert by_mode == {'Studying': {'breathing': {'helped': 1, 'no_effect': 1}},
                       'Driving': {'breathing': {'helped': 1, 'no_effect': 0},
                                   'stretch': {'helped': 0, 'no_effect': 1}}}
    p = _params(env.server.of('GET', 'interventions')[0])
    assert p['user_id'] == f'eq.{UID}' and p['outcome'] == 'in.(helped,no_effect)'


def test_load_user_baseline_returns_none_when_unset(env):
    assert env.store.load_user_baseline(UID) is None
    env.server.on('GET', 'profiles', lambda r: [{'baseline_json': {'daisee_baseline': 0.5}}])
    assert env.store.load_user_baseline(UID) == {'daisee_baseline': 0.5}
    assert _params(env.server.of('GET', 'profiles')[-1])['id'] == f'eq.{UID}'


def test_the_token_is_refreshed_before_every_database_call(env):
    env.store.session_count()
    env.store.load_recent_sessions(5)
    env.store.save_user_baseline(UID, {})
    assert env.auth.fresh_calls >= len(env.server.requests) == 3


# ----------------------------------------------------------------------------
# helpers / factory
# ----------------------------------------------------------------------------
def test_num_makes_values_json_safe():
    assert session_store._num(None) is None
    assert session_store._num(float('nan')) is None
    assert session_store._num(math.inf) is None
    assert session_store._num('x') is None
    assert session_store._num(3) == 3.0


def test_get_store_returns_the_sqlite_module_locally(monkeypatch):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'sqlite')
    assert session_store.get_store(config) is session_history


def test_get_store_refuses_to_run_hosted_without_a_signed_in_user(monkeypatch):
    monkeypatch.setattr(config, 'STORAGE_BACKEND', 'supabase')
    with pytest.raises(RuntimeError):
        session_store.get_store(config, None)
    with pytest.raises(RuntimeError):
        session_store.get_store(config, SimpleNamespace(is_signed_in=False))
    assert isinstance(
        session_store.get_store(config, SimpleNamespace(is_signed_in=True, user_id=UID)),
        SupabaseStore)
