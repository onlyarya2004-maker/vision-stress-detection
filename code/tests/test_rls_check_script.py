"""
Dry run of scripts/rls_check.py against an in-memory fake of Supabase.

The real script can only be pointed at a live project, which doesn't exist until
deployment day - so this makes sure that on the day it (a) runs end to end
without crashing, (b) reports ALL CHECKS PASSED against a backend that enforces
Row Level Security the way supabase/schema.sql does, (c) FAILS LOUDLY against a
backend whose RLS is switched off (a script that can't fail proves nothing), and
(d) leaves both accounts' real data exactly as it found it.

FakeSupabase implements just the PostgREST behaviours the script and
SupabaseStore use: filters (eq / in / gte / not.is.null), order/limit/offset,
insert with auth.uid() defaults, upsert, update, delete (with cascade) - and the
policy rules: a signed-in user sees/changes only their own rows, may only attach
samples/interventions to a session they own, and `anon` gets "permission denied".
"""
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from postgrest import SyncPostgrestClient

import auth

SCRIPT = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'rls_check.py'
USERS = {'alice@x.com': ('uid-alice', 'pwA'), 'bob@x.com': ('uid-bob', 'pwB')}


class FakeSupabase:
    OWNER_COL = {'sessions': 'user_id', 'samples': 'user_id',
                 'interventions': 'user_id', 'profiles': 'id'}

    def __init__(self, rls=True):
        self.rls = rls
        self.tables = {t: [] for t in self.OWNER_COL}
        self.seq = 0

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _matches(row, col, expr):
        val = row.get(col)
        if expr.startswith('eq.'):
            return str(val) == expr[3:]
        if expr == 'not.is.null':
            return val is not None
        if expr.startswith('in.('):
            return str(val) in expr[4:-1].split(',')
        if expr.startswith('gte.'):
            return val is not None and float(val) >= float(expr[4:])
        return True

    def _visible(self, table, uid):
        rows = self.tables[table]
        if not self.rls:
            return rows
        col = self.OWNER_COL[table]
        return [r for r in rows if r.get(col) == uid]

    def _filter(self, table, uid, params):
        rows = self._visible(table, uid)
        for col, expr in params.items():
            if col in ('select', 'order', 'limit', 'offset'):
                continue
            rows = [r for r in rows if self._matches(r, col, expr)]
        return rows

    def _can_attach(self, table, row, uid):
        if not self.rls or table not in ('samples', 'interventions'):
            return True
        sid = row.get('session_id')
        if sid is None and table == 'interventions':
            return True
        return any(s['id'] == sid and s['user_id'] == uid for s in self.tables['sessions'])

    # -- the HTTP surface ---------------------------------------------------
    def handle(self, request, uid):
        table = request.url.path.rsplit('/', 1)[-1]
        params = dict(request.url.params)
        if uid is None and self.rls:
            return httpx.Response(401, json={'code': '42501',
                                             'message': f'permission denied for table {table}'})
        m = request.method
        if m == 'GET':
            rows = self._filter(table, uid, params)
            for key in ('order',):
                if key in params:
                    col, _, direction = params[key].partition('.')
                    rows = sorted(rows, key=lambda r: r.get(col), reverse=(direction == 'desc'))
            total = len(rows)
            off, lim = int(params.get('offset', 0)), int(params.get('limit', 10 ** 9))
            rows = rows[off:off + lim]
            return httpx.Response(200, json=rows, headers={
                'content-range': f"0-{max(len(rows) - 1, 0)}/{total}"})
        if m == 'POST':
            body = json.loads(request.content)
            body = body if isinstance(body, list) else [body]
            col = self.OWNER_COL[table]
            out = []
            for row in body:
                row = dict(row)
                if table != 'profiles':
                    row.setdefault('user_id', uid)
                if self.rls and row.get(col) != uid:
                    return httpx.Response(403, json={'code': '42501', 'message': 'new row violates row-level security policy'})
                if not self._can_attach(table, row, uid):
                    return httpx.Response(403, json={'code': '42501', 'message': 'new row violates row-level security policy'})
                if table == 'profiles':
                    existing = [r for r in self.tables[table] if r['id'] == row['id']]
                    if existing:
                        existing[0].update(row)
                        out.append(existing[0])
                        continue
                else:
                    self.seq += 1
                    row['id'] = self.seq
                self.tables[table].append(row)
                out.append(row)
            return httpx.Response(201, json=out)
        if m == 'PATCH':
            for r in self._filter(table, uid, params):
                r.update(json.loads(request.content))
            return httpx.Response(200, json=[])
        if m == 'DELETE':
            doomed = self._filter(table, uid, params)
            self.tables[table] = [r for r in self.tables[table] if r not in doomed]
            if table == 'sessions':                     # ON DELETE CASCADE
                ids = {r['id'] for r in doomed}
                for child in ('samples', 'interventions'):
                    self.tables[child] = [r for r in self.tables[child] if r.get('session_id') not in ids]
            return httpx.Response(204)
        return httpx.Response(405)


class FakeGoTrue:
    def __init__(self, client):
        self.client = client

    def sign_in_with_password(self, creds):
        uid, pw = USERS[creds['email']]
        if creds['password'] != pw:
            raise Exception('Invalid login credentials')
        self.client.token = uid
        return SimpleNamespace(session=SimpleNamespace(access_token=uid),
                               user=SimpleNamespace(id=uid, email=creds['email']))

    def get_session(self):
        return SimpleNamespace(access_token=self.client.token) if self.client.token else None

    def sign_out(self):
        self.client.token = None


def make_fake_client_factory(server):
    def make_client(url, key):
        client = SimpleNamespace(token=None)
        http = httpx.Client(transport=httpx.MockTransport(
            lambda req: server.handle(req, client.token)))
        client.table = SyncPostgrestClient('http://sb.test/rest/v1', http_client=http).from_
        client.auth = FakeGoTrue(client)
        return client
    return make_client


def load_script():
    spec = importlib.util.spec_from_file_location('rls_check_script', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(monkeypatch, capsys, server):
    mod = load_script()
    mod.FAILS.clear()
    monkeypatch.setattr(auth, 'make_client', make_fake_client_factory(server))
    monkeypatch.setattr(sys, 'argv', [
        'rls_check.py', '--url', 'http://sb.test', '--key', 'sb_publishable_x',
        '--email-a', 'alice@x.com', '--password-a', 'pwA',
        '--email-b', 'bob@x.com', '--password-b', 'pwB'])
    code = 0
    try:
        mod.main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
    return code, capsys.readouterr().out


def test_script_passes_against_a_database_that_enforces_rls(monkeypatch, capsys):
    server = FakeSupabase(rls=True)
    code, out = run(monkeypatch, capsys, server)
    assert code == 0, out
    assert 'ALL CHECKS PASSED' in out and '[FAIL]' not in out
    assert out.count('[PASS]') >= 15


def test_script_fails_loudly_against_a_database_with_rls_switched_off(monkeypatch, capsys):
    server = FakeSupabase(rls=False)
    code, out = run(monkeypatch, capsys, server)
    assert code == 1
    assert '[FAIL]' in out and 'DO NOT DEMO' in out


def test_script_leaves_real_data_exactly_as_it_found_it(monkeypatch, capsys):
    server = FakeSupabase(rls=True)
    # Pre-existing REAL data for both accounts, including a saved baseline for A.
    server.tables['sessions'] += [
        {'id': 901, 'user_id': 'uid-alice', 'task_mode': 'Default', 'self_report': 4},
        {'id': 902, 'user_id': 'uid-bob', 'task_mode': 'Studying', 'self_report': None}]
    server.tables['samples'] += [{'id': 951, 'session_id': 901, 'user_id': 'uid-alice'}]
    server.tables['profiles'] += [{'id': 'uid-alice', 'baseline_json': {'daisee_baseline': 0.77}}]
    server.seq = 1000
    def snapshot():
        # profiles.updated_at is bookkeeping that every baseline save touches;
        # everything a person actually recorded or calibrated must be identical.
        t = {k: [dict(r) for r in v] for k, v in server.tables.items()}
        for r in t['profiles']:
            r.pop('updated_at', None)
        return json.dumps(t, sort_keys=True)

    before = snapshot()
    code, out = run(monkeypatch, capsys, server)
    assert code == 0, out
    assert snapshot() == before, "the script changed or deleted real account data"
