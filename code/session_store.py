"""
Hosted storage backend (Supabase / Postgres)
============================================
SupabaseStore exposes the same function names app.py already calls on the
local SQLite module (session_history.py), so the UI code is identical in both
modes: app.py does `store = get_store()` and calls store.save_session(...),
store.load_recent_sessions(...), and so on. In local mode get_store() simply
returns the session_history module itself.

Every call goes through the signed-in user's own supabase client, i.e. with
THEIR JWT, so the Row Level Security policies in supabase/schema.sql decide
what is visible. Each query also filters on user_id explicitly - defence in
depth, and it keeps the intent readable. Nothing here ever accepts a user id
from the caller as authority: the id comes from the authenticated session.

Differences from the SQLite module, all deliberate:
  * load_recent_sessions / session_count / intervention_stats* are PER USER
    (the SQLite versions are one shared log for the whole install).
  * clear_all_sessions() deletes only the caller's own data ("Delete my data").
  * list_users()/create_user() don't exist: accounts are Supabase Auth users.
  * PostgREST returns at most 1000 rows per request, and a 10-minute session
    stores ~1200 samples - every multi-row read therefore pages explicitly.
"""
import math
from datetime import datetime
from typing import Dict, List, Optional

from session_history import (
    SIGNAL_COLUMNS, TEMPORAL_EXTRA_COLUMNS, summarize_history, validate_self_report,
)

_PAGE = 1000          # PostgREST's default max rows per response
_INSERT_CHUNK = 400   # samples per insert request (keeps each request small)


def _num(x):
    """JSON-safe float or None (NaN/inf are not valid JSON)."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iso(dt):
    return dt.isoformat() if isinstance(dt, datetime) else str(dt)


class SupabaseStore:
    def __init__(self, auth_session):
        """auth_session: auth.AuthSession (signed in)."""
        self._auth = auth_session

    # -- plumbing -----------------------------------------------------------
    @property
    def user_id(self):
        return self._auth.user_id

    def _t(self, name):
        """Table handle, after making sure the access token is fresh."""
        self._auth.ensure_fresh()
        return self._auth.client.table(name)

    def _all(self, build):
        """Run build(start, end) -> executed-able query over successive
        pages until a short page comes back."""
        rows, start = [], 0
        while True:
            data = build(start, start + _PAGE - 1).execute().data or []
            rows.extend(data)
            if len(data) < _PAGE:
                return rows
            start += _PAGE

    # -- schema / users -----------------------------------------------------
    def init_db(self):
        """Schema is created once by supabase/schema.sql; nothing to do."""

    def list_users(self):
        return []

    def create_user(self, name):
        raise NotImplementedError("accounts are managed by Supabase Auth")

    def save_user_baseline(self, user_id, baseline: Dict):
        # `user_id` is ignored on purpose: the owner is always the
        # authenticated user, never a caller-supplied id.
        self._t('profiles').upsert(
            {'id': self.user_id, 'baseline_json': baseline,
             'updated_at': datetime.utcnow().isoformat()}).execute()

    def load_user_baseline(self, user_id) -> Optional[Dict]:
        rows = (self._t('profiles').select('baseline_json')
                .eq('id', self.user_id).limit(1).execute().data or [])
        return rows[0]['baseline_json'] if rows and rows[0].get('baseline_json') else None

    # -- sessions -----------------------------------------------------------
    def save_session(self, history: List[Dict], task_mode: str = 'Default',
                     user_id=None) -> Optional[int]:
        if not history:
            return None
        stats = summarize_history(history)
        sess = (self._t('sessions').insert({
            'started_at': _iso(history[0]['ts']),
            'ended_at': _iso(history[-1]['ts']),
            'duration_sec': _num(stats['duration_sec']),
            'sample_count': int(stats['sample_count']),
            'mean_stress': _num(stats['mean_stress']),
            'max_stress': _num(stats['max_stress']),
            'pct_low': _num(stats['pct_low']),
            'pct_moderate': _num(stats['pct_moderate']),
            'pct_high': _num(stats['pct_high']),
            'task_mode': str(task_mode),
        }).execute().data or [])
        session_id = int(sess[0]['id'])

        extra = SIGNAL_COLUMNS + TEMPORAL_EXTRA_COLUMNS
        rows = [{
            'session_id': session_id,
            'ts': _iso(r['ts']),
            'stress_index': _num(r['stress_index']),
            'level': str(r.get('level', 'LOW')),
            'daisee': _num(r.get('daisee')),
            'behavior': _num(r.get('behavior', 0.0)) or 0.0,
            **{c: _num(r.get(c)) for c in extra},
        } for r in history]
        try:
            for i in range(0, len(rows), _INSERT_CHUNK):
                self._t('samples').insert(rows[i:i + _INSERT_CHUNK]).execute()
        except Exception:
            # Don't leave a summary row that claims N samples it doesn't have.
            try:
                self._t('sessions').delete().eq('id', session_id).execute()
            except Exception:
                pass
            raise
        return session_id

    def set_self_report(self, session_id: int, rating: Optional[int], user_id=None):
        """Store the signed-in user's rating on one of THEIR sessions.

        `user_id` is ignored on purpose (owner = the authenticated user). The
        session is looked up as this user first, and the update must come
        back with exactly that row - RLS silently matches zero rows for
        someone else's id, and a zero-row update must never be reported as
        saved. Raises LookupError (not theirs / not found) or RuntimeError
        (the write did not take)."""
        rating = validate_self_report(rating)
        owned = (self._t('sessions').select('id')
                 .eq('id', session_id).eq('user_id', self.user_id)
                 .limit(1).execute().data or [])
        if not owned:
            raise LookupError("that session was not found for this account")
        updated = (self._t('sessions').update({'self_report': rating})
                   .eq('id', session_id).eq('user_id', self.user_id)
                   .execute().data or [])
        if len(updated) != 1:
            raise RuntimeError("the rating was not saved")

    def labeled_sessions(self, min_samples: int = 10) -> List[Dict]:
        return self._all(lambda a, b: (
            self._t('sessions').select('*').eq('user_id', self.user_id)
            .not_.is_('self_report', 'null').gte('sample_count', min_samples)
            .order('id').range(a, b)))

    def load_recent_sessions(self, n: int = 10) -> List[Dict]:
        n = int(n)
        if n <= _PAGE:
            return (self._t('sessions').select('*').eq('user_id', self.user_id)
                    .order('id', desc=True).limit(n).execute().data or [])
        return self._all(lambda a, b: (
            self._t('sessions').select('*').eq('user_id', self.user_id)
            .order('id', desc=True).range(a, min(b, n - 1))))[:n]

    def session_count(self) -> int:
        res = (self._t('sessions').select('id', count='exact')
               .eq('user_id', self.user_id).limit(1).execute())
        return int(res.count or 0)

    def load_session_samples(self, session_id: int) -> List[Dict]:
        rows = self._all(lambda a, b: (
            self._t('samples')
            .select('ts,stress_index,level,daisee,behavior')
            .eq('session_id', session_id).eq('user_id', self.user_id)
            .order('id').range(a, b)))
        return [{
            'ts': datetime.fromisoformat(r['ts']),
            'stress_index': float(r['stress_index']),
            'level': str(r['level']),
            'daisee': None if r['daisee'] is None else float(r['daisee']),
            'behavior': float(r['behavior']),
        } for r in rows]

    def session_index_stats(self, session_id: int):
        """(count, mean) of one of this user's sessions' stored stress_index
        (the general model's smoothed final index) - only that one column is
        fetched (PostgREST aggregates are off by default, so the mean is
        taken here)."""
        rows = self._all(lambda a, b: (
            self._t('samples').select('stress_index')
            .eq('session_id', session_id).eq('user_id', self.user_id)
            .order('id').range(a, b)))
        vals = [float(r['stress_index']) for r in rows
                if r.get('stress_index') is not None and math.isfinite(float(r['stress_index']))]
        return len(vals), (sum(vals) / len(vals) if vals else None)

    def load_session_signal_rows(self, session_id: int) -> List[Dict]:
        return self._all(lambda a, b: (
            self._t('samples').select(','.join(SIGNAL_COLUMNS))
            .eq('session_id', session_id).eq('user_id', self.user_id)
            .order('id').range(a, b)))

    def load_session_temporal_rows(self, session_id: int) -> List[Dict]:
        return self._all(lambda a, b: (
            self._t('samples').select(','.join(SIGNAL_COLUMNS + TEMPORAL_EXTRA_COLUMNS))
            .eq('session_id', session_id).eq('user_id', self.user_id)
            .order('id').range(a, b)))

    def delete_session(self, session_id: int):
        # ON DELETE CASCADE removes the session's samples and interventions.
        (self._t('sessions').delete().eq('id', session_id)
         .eq('user_id', self.user_id).execute())

    def clear_all_sessions(self):
        """Delete the CALLER'S sessions (and, by cascade, their samples and
        interventions). Never anyone else's - RLS and the user_id filter."""
        self._t('sessions').delete().eq('user_id', self.user_id).execute()

    # -- intervention agent -------------------------------------------------
    def log_interventions(self, session_id: Optional[int], records: List[Dict]):
        if not records:
            return
        self._t('interventions').insert([{
            'session_id': session_id,
            'ts': _iso(r['ts']),
            'kind': str(r['kind']),
            'trigger_signal': r.get('trigger_signal'),
            'index_before': _num(r['index_before']),
            'index_after': _num(r.get('index_after')),
            'outcome': str(r.get('outcome', 'pending')),
        } for r in records]).execute()

    def _measured_interventions(self):
        return self._all(lambda a, b: (
            self._t('interventions').select('session_id,kind,outcome')
            .eq('user_id', self.user_id).in_('outcome', ['helped', 'no_effect'])
            .order('id').range(a, b)))

    def intervention_stats(self) -> Dict[str, Dict[str, int]]:
        """Per-kind helped/failed counts for THIS user only - the agent
        learns what works for the person using it."""
        stats: Dict[str, Dict[str, int]] = {}
        for r in self._measured_interventions():
            d = stats.setdefault(r['kind'], {'helped': 0, 'no_effect': 0})
            d[r['outcome']] += 1
        return stats

    def intervention_stats_by_mode(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        modes = {s['id']: s.get('task_mode') or 'Default' for s in self._all(
            lambda a, b: (self._t('sessions').select('id,task_mode')
                          .eq('user_id', self.user_id).order('id').range(a, b)))}
        stats: Dict[str, Dict[str, Dict[str, int]]] = {}
        for r in self._measured_interventions():
            mode = modes.get(r['session_id'])
            if mode is None:           # unknown/deleted session: excluded, as in SQLite
                continue
            stats.setdefault(mode, {}).setdefault(
                r['kind'], {'helped': 0, 'no_effect': 0})[r['outcome']] += 1
        return stats


def get_store(config_module, auth_session=None):
    """The storage backend for this run: the SQLite module locally, a
    SupabaseStore bound to the signed-in user when hosted."""
    if config_module.STORAGE_BACKEND == 'supabase':
        if auth_session is None or not auth_session.is_signed_in:
            raise RuntimeError("hosted storage needs a signed-in user")
        return SupabaseStore(auth_session)
    import session_history
    return session_history
