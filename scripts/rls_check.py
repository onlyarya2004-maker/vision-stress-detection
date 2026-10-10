"""
Live proof that one account cannot see or touch another account's data.

Run this against the REAL Supabase project after supabase/schema.sql has been
applied. It signs in as two different users and has user B try, one by one, to
read / change / delete / write into what user A saved. Every attempt must
fail; the script prints PASS/FAIL per check and exits non-zero on any FAIL.
It is also a good thing to show the examiners.

    python scripts/rls_check.py --url https://xxxx.supabase.co --key <anon key> \
        --email-a alice@example.com --password-a ... \
        --email-b bob@example.com   --password-b ...

Both accounts must already exist (create them in the app, or in the Supabase
dashboard with "Auto Confirm User"). Best practice: use two THROWAWAY accounts
rather than the real demo accounts - although the script is careful: it only
ever touches rows it creates itself (one probe session for A), and it restores
A's saved calibration baseline to exactly what it was. Nothing of either
account's real data is deleted.

It only uses the PUBLIC key - the same one the hosted app has - so what it
proves is what a malicious visitor could or could not do.
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))

import auth                                   # noqa: E402
from session_store import SupabaseStore       # noqa: E402

FAILS = []


def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ''))
    if not ok:
        FAILS.append(name)


def login(url, key, email, password):
    s = auth.AuthSession(auth.make_client(url, key))
    s.sign_in(email, password)
    return s


def history(n=6):
    start = datetime(2026, 10, 10, 9, 0, 0)
    return [{'ts': start + timedelta(seconds=0.5 * i), 'stress_index': 0.3, 'level': 'LOW',
             'daisee': 0.5, 'behavior': 0.25} for i in range(n)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', required=True)
    ap.add_argument('--key', required=True, help='the PUBLIC anon/publishable key')
    ap.add_argument('--email-a', required=True)
    ap.add_argument('--password-a', required=True)
    ap.add_argument('--email-b', required=True)
    ap.add_argument('--password-b', required=True)
    args = ap.parse_args()

    auth.validate_public_key(args.key)
    A = login(args.url, args.key, args.email_a, args.password_a)
    B = login(args.url, args.key, args.email_b, args.password_b)
    print(f"A = {A.email} ({A.user_id})\nB = {B.email} ({B.user_id})")
    if A.user_id == B.user_id:
        sys.exit("A and B must be different accounts")
    sa, sb = SupabaseStore(A), SupabaseStore(B)
    raw_b = B.client
    created = []                                   # session ids this script created (A's)
    original_baseline = sa.load_user_baseline(A.user_id)
    probe_baseline = {'behavioral': {}, 'daisee_baseline': 0.4242}

    try:
        print("\nSetup: A saves one probe session, a baseline and an intervention")
        sid = sa.save_session(history(), task_mode='Studying')
        created.append(sid)
        sa.save_user_baseline(A.user_id, probe_baseline)
        sa.log_interventions(sid, [{'ts': datetime(2026, 10, 10, 9, 1), 'kind': 'breathing',
                                    'trigger_signal': 'jaw_tension', 'index_before': 0.6,
                                    'index_after': 0.4, 'outcome': 'helped'}])
        check("A can read back its own probe session",
              sid in [s['id'] for s in sa.load_recent_sessions(500)])
        check("A can read back its own probe samples", len(sa.load_session_samples(sid)) == 6)

        print("\nB tries to READ A's data")
        check("A's probe session is not in B's session list",
              sid not in [s['id'] for s in sb.load_recent_sessions(500)])
        got = raw_b.table('sessions').select('*').eq('user_id', A.user_id).execute().data
        check("B cannot list ANY session belonging to A", got == [], f"{len(got)} rows")
        got = raw_b.table('sessions').select('*').eq('id', sid).execute().data
        check("B cannot fetch A's probe session by id", got == [], f"{len(got)} rows")
        got = raw_b.table('samples').select('*').eq('session_id', sid).execute().data
        check("B cannot fetch A's samples by session id", got == [], f"{len(got)} rows")
        got = raw_b.table('profiles').select('*').eq('id', A.user_id).execute().data
        check("B cannot read A's calibration baseline", got == [], f"{len(got)} rows")
        got = raw_b.table('interventions').select('*').eq('user_id', A.user_id).execute().data
        check("B sees none of A's interventions", got == [], f"{len(got)} rows")

        print("\nB tries to CHANGE A's data")
        try:
            raw_b.table('sessions').update({'self_report': 5}).eq('id', sid).execute()
        except Exception:
            pass
        mine = [s for s in sa.load_recent_sessions(500) if s['id'] == sid]
        check("B's UPDATE on A's session changed nothing",
              len(mine) == 1 and mine[0].get('self_report') is None)
        try:
            raw_b.table('sessions').delete().eq('id', sid).execute()
        except Exception:
            pass
        check("B's DELETE on A's session removed nothing",
              sid in [s['id'] for s in sa.load_recent_sessions(500)])
        try:
            raw_b.table('profiles').update({'baseline_json': {'evil': True}}).eq('id', A.user_id).execute()
        except Exception:
            pass
        check("B could not overwrite A's baseline",
              (sa.load_user_baseline(A.user_id) or {}).get('daisee_baseline') == 0.4242)

        print("\nB tries to WRITE into A's session")
        try:
            raw_b.table('samples').insert({
                'session_id': sid, 'ts': '2026-10-10T09:00:09', 'stress_index': 0.9,
                'level': 'HIGH', 'behavior': 0.9}).execute()
            wrote = True
        except Exception:
            wrote = False
        check("B's INSERT of a sample into A's session is rejected", not wrote)
        check("A's probe session still has exactly its own 6 samples",
              len(sa.load_session_samples(sid)) == 6)
        try:
            raw_b.table('samples').insert({
                'session_id': sid, 'user_id': A.user_id, 'ts': '2026-10-10T09:00:09',
                'stress_index': 0.9, 'level': 'HIGH', 'behavior': 0.9}).execute()
            forged = True
        except Exception:
            forged = False
        check("B cannot forge rows claiming to be A's", not forged)

        print("\nAn anonymous visitor (public key, no login)")
        anon = auth.make_client(args.url, args.key)
        for table in ('sessions', 'samples', 'profiles', 'interventions'):
            try:
                rows = anon.table(table).select('*').execute().data
            except Exception:
                rows = []
            check(f"sees no rows in {table}", rows == [], f"{len(rows)} rows")
    finally:
        print("\nCleanup (only what this script created)")
        for session_id in created:
            try:
                sa.delete_session(session_id)
            except Exception as e:
                print(f"  cleanup warning (session {session_id}):", e)
        try:
            if original_baseline is not None:
                sa.save_user_baseline(A.user_id, original_baseline)     # exactly as it was
            else:
                A.client.table('profiles').update({'baseline_json': None}).eq('id', A.user_id).execute()
        except Exception as e:
            print("  cleanup warning (baseline):", e)
        A.sign_out()
        B.sign_out()

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED - DO NOT DEMO THIS DATABASE:")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED: users are isolated from each other.")


if __name__ == '__main__':
    main()
