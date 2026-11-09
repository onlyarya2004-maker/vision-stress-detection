"""
Measure what the per-user learner (code/personalization.py) actually buys.

Protocol ("prequential", the honest one for a learner that keeps learning):
for every user, walk through their rated sessions in time order; before each
session k, fit on sessions 1..k-1 ONLY (newest first, exactly as the live app
does), then predict session k. A session is predicted stressed when
mean_index + adjustment >= LOW_TO_MODERATE_CUTOFF. Three readers are compared
on the same sessions:

  general   the shared model alone (no learner)
  stage1    the original learner (one bounded shift)
  full      the current learner (stage 1, plus stage 2 signal weighting when
            it validates)

Reported: balanced accuracy over all predicted sessions, overall and per user
type. Sessions before a user has enough history are still counted (the
learner then returns the general model), so early sessions dilute the gain -
that is what a real user experiences.

Two data sources:
  --simulate N   N synthetic users per type (offset, hidden-signal,
                 well-served, no-signal). SYNTHETIC: shows how the learner
                 behaves on known patterns, NOT how accurate it is on people.
  --db PATH      a local session_history SQLite file (per user_id, sessions
                 with a self_report and per-frame signals). Real, but the
                 team's own history is far too small for firm conclusions.

Usage (from the repo root):
  python scripts/evaluate_personalization.py --simulate 200
  python scripts/evaluate_personalization.py --db data/sessions.db
"""
import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
import config                     # noqa: E402
import personalization as P       # noqa: E402

CUT = config.LOW_TO_MODERATE_CUTOFF
POP_W = np.array([0.10, 0.15, 0.30, 0.25, 0.05, 0.05, 0.10])   # Default-mode base weights


def readers(history_newest_first, session):
    """Adjustments of the three readers for `session`, fitted on history."""
    p_full = P.fit(history_newest_first)
    p_s1 = P._fit_stage1(history_newest_first)
    sig = session.get('signals') or {}
    return {'general': 0.0,
            'stage1': p_s1.adjustment(sig) if p_s1.active else 0.0,
            'full': p_full.adjustment(sig) if p_full.active else 0.0}, p_full


def balanced(rows):
    """rows: [(is_stressed, predicted_stressed)] -> balanced accuracy or None."""
    hit = {True: 0, False: 0}
    tot = {True: 0, False: 0}
    for y, yhat in rows:
        tot[y] += 1
        hit[y] += int(y == yhat)
    if not tot[True] or not tot[False]:
        return None
    return 0.5 * (hit[True] / tot[True] + hit[False] / tot[False])


def evaluate(users):
    """users: {user: (type, [session dicts oldest first])}."""
    preds = defaultdict(list)                  # (type, reader) -> rows
    activations = defaultdict(lambda: defaultdict(int))
    for user, (utype, sessions) in users.items():
        for k, sess in enumerate(sessions):
            lab = P._label(sess.get('self_report'))
            if lab is None or P._usable_mean(sess) is None:
                continue
            hist = list(reversed(sessions[:k]))          # newest first, like load()
            adj, p_full = readers(hist, sess)
            activations[utype][p_full.model if p_full.active else 'general'] += 1
            for name, a in adj.items():
                yhat = sess['mean_index'] + a >= CUT
                preds[(utype, name)].append((lab == 'stressed', yhat))
                preds[('ALL', name)].append((lab == 'stressed', yhat))
    return preds, activations


def report(preds, activations, title):
    print(f"\n{title}")
    print(f"{'user type':<14}{'sessions':>9}{'general':>10}{'stage 1':>10}{'full':>10}   learner state used")
    types = sorted({t for t, _ in preds if t != 'ALL'}) + ['ALL']
    for t in types:
        n = len(preds[(t, 'general')])
        cells = [balanced(preds[(t, r)]) for r in ('general', 'stage1', 'full')]
        cells = ''.join(f"{'-':>10}" if c is None else f"{c:>10.1%}" for c in cells)
        act = activations.get(t)
        mix = ', '.join(f"{k} {v}" for k, v in sorted(act.items())) if act else ''
        print(f"{t:<14}{n:>9}{cells}   {mix}")


# ---------------------------------------------------------------------------
def simulate(n_per_type, n_sessions=25, seed=7):
    rng = np.random.default_rng(seed)
    users = {}
    types = {
        # general model reads this person ~0.08 too high: a shift fixes it
        'offset':      dict(offset=+0.08, signature=[2, 3], effect=0.20),
        # stress shows in slouch / neck / head, which carry little weight in
        # the general score: the shift alone cannot fix it, signal weights can
        'hidden':      dict(offset=0.0, signature=[4, 5, 6], effect=0.30),
        # already well separated by the general model
        'well-served': dict(offset=0.0, signature=[2, 3], effect=0.30),
        # ratings unrelated to the face: nothing should help, nothing may hurt
        'no-signal':   dict(offset=0.0, signature=[], effect=0.0),
    }
    for tname, spec in types.items():
        for u in range(n_per_type):
            base = rng.uniform(0.05, 0.25, size=7)
            sessions = []
            for k in range(n_sessions):
                stressed = rng.random() < 0.5
                x = base + rng.normal(0, 0.05, size=7)
                if stressed:
                    for j in spec['signature']:
                        x[j] += spec['effect'] * rng.uniform(0.7, 1.3)
                x = np.clip(x, 0, 1)
                g = float(np.clip(0.9 * POP_W @ x + spec['offset'] + 0.06 + rng.normal(0, 0.025), 0, 1))
                rating = int(rng.choice([4, 5])) if stressed else int(rng.choice([1, 2]))
                sessions.append({'self_report': rating, 'n_index': 600, 'mean_index': g,
                                 'signals': dict(zip(P.SIGNALS, map(float, x)))})
            users[f'{tname}-{u}'] = (tname, sessions)
    return users


def from_db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(sessions)")}
    ucol = 'user_id' if 'user_id' in cols else None
    rows = conn.execute("SELECT * FROM sessions WHERE self_report IS NOT NULL ORDER BY id").fetchall()
    users = defaultdict(list)
    for r in rows:
        sid = r['id']
        n, mean = conn.execute("SELECT COUNT(stress_index), AVG(stress_index) FROM samples "
                               "WHERE session_id = ?", (sid,)).fetchone()
        sig = conn.execute("SELECT " + ", ".join(f"AVG({c})" for c in P.SIGNALS)
                           + " FROM samples WHERE session_id = ?", (sid,)).fetchone()
        signals = None if any(v is None for v in sig) else dict(zip(P.SIGNALS, map(float, sig)))
        key = r[ucol] if ucol else 'all'
        users[key].append({'self_report': r['self_report'], 'n_index': n or 0,
                           'mean_index': mean, 'signals': signals})
    conn.close()
    return {u: ('real', s) for u, s in users.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--simulate', type=int, metavar='N')
    g.add_argument('--db', metavar='PATH')
    ap.add_argument('--sessions', type=int, default=25)
    a = ap.parse_args()
    if a.simulate:
        users = simulate(a.simulate, a.sessions)
        title = (f"SYNTHETIC users ({a.simulate} per type, {a.sessions} sessions each) - "
                 f"behaviour on known patterns, not accuracy on people")
    else:
        users = from_db(a.db)
        title = f"Real sessions from {a.db} ({len(users)} user(s))"
    preds, acts = evaluate(users)
    report(preds, acts, title)
