-- =============================================================================
-- Hosted schema for the stress-detection app (Supabase / Postgres)
-- =============================================================================
-- Run ONCE in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to re-run on an empty project; it is not an upgrade script.
--
-- Mirrors the local SQLite schema in code/session_history.py (v8) so the same
-- data shapes flow through app.py in both modes. Differences:
--   * every row carries user_id (default auth.uid()) = the logged-in account
--   * timestamps are `timestamp` (no zone): they hold the USER'S local wall
--     clock, exactly like the SQLite TEXT columns did, so time-of-day
--     analytics show when the person actually recorded (the server is UTC)
--   * ON DELETE CASCADE everywhere, so "Delete my data" is one statement
--
-- ISOLATION MODEL (the point of this file): Row Level Security is switched on
-- for every table and the ONLY policies are "a signed-in user may touch rows
-- whose user_id is their own". The database itself enforces this - not the
-- app - so a bug in app code cannot leak another user's rows. The `anon` role
-- (anyone holding just the public key) gets no policy, i.e. no access at all.
-- The service_role key bypasses RLS and must NEVER be put in the app or the
-- container; it is for offline admin work (e.g. pooled model retraining) only.
-- =============================================================================

create table public.profiles (
    id             uuid primary key references auth.users (id) on delete cascade,
    baseline_json  jsonb,                       -- personal calibration baseline
    updated_at     timestamptz not null default now()
);

create table public.sessions (
    id            bigint generated always as identity primary key,
    user_id       uuid not null default auth.uid() references auth.users (id) on delete cascade,
    started_at    timestamp not null,
    ended_at      timestamp not null,
    duration_sec  double precision not null,
    sample_count  integer not null,
    mean_stress   double precision not null,
    max_stress    double precision not null,
    pct_low       double precision not null,
    pct_moderate  double precision not null,
    pct_high      double precision not null,
    task_mode     text not null default 'Default',
    self_report   smallint check (self_report between 1 and 5)
);
create index sessions_user_started on public.sessions (user_id, started_at desc);

create table public.samples (
    id                 bigint generated always as identity primary key,
    session_id         bigint not null references public.sessions (id) on delete cascade,
    user_id            uuid not null default auth.uid() references auth.users (id) on delete cascade,
    ts                 timestamp not null,
    stress_index       double precision not null,
    level              text not null,
    daisee             double precision,        -- NULL = the CNN did not run this frame (not "0")
    behavior           double precision not null,
    eye_strain         double precision,
    blink_stress       double precision,
    brow_tension       double precision,
    jaw_tension        double precision,
    head_instability   double precision,
    neck_strain        double precision,
    slouch             double precision,
    ear                double precision,
    ear_asymmetry      double precision,
    blinks_per_minute  double precision
);
create index samples_session on public.samples (session_id);
create index samples_user    on public.samples (user_id);

create table public.interventions (
    id              bigint generated always as identity primary key,
    session_id      bigint references public.sessions (id) on delete cascade,
    user_id         uuid not null default auth.uid() references auth.users (id) on delete cascade,
    ts              timestamp not null,
    kind            text not null,
    trigger_signal  text,
    index_before    double precision not null,
    index_after     double precision,
    outcome         text not null default 'pending'
);
create index interventions_user on public.interventions (user_id);

-- ---------------------------------------------------------------------------
-- Row Level Security
-- ---------------------------------------------------------------------------
alter table public.profiles      enable row level security;
alter table public.sessions      enable row level security;
alter table public.samples       enable row level security;
alter table public.interventions enable row level security;

-- The `anon` role must see nothing, even if a policy is added by mistake later.
revoke all on public.profiles, public.sessions, public.samples, public.interventions from anon;

create policy "own profile" on public.profiles
    for all to authenticated
    using      (id = (select auth.uid()))
    with check (id = (select auth.uid()));

create policy "own sessions" on public.sessions
    for all to authenticated
    using      (user_id = (select auth.uid()))
    with check (user_id = (select auth.uid()));

-- samples/interventions: besides owning the row, a user may only attach rows to
-- a session THEY own (the sub-select runs under RLS, so it only sees their own
-- sessions). Without this, a malicious client could write rows into someone
-- else's session id.
create policy "own samples" on public.samples
    for all to authenticated
    using      (user_id = (select auth.uid()))
    with check (user_id = (select auth.uid())
                and exists (select 1 from public.sessions s where s.id = session_id));

create policy "own interventions" on public.interventions
    for all to authenticated
    using      (user_id = (select auth.uid()))
    with check (user_id = (select auth.uid())
                and (session_id is null
                     or exists (select 1 from public.sessions s where s.id = session_id)));
