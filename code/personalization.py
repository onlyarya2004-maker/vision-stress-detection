"""
Per-user personalization (2026-10-15)
=====================================
What it is - and is not
-----------------------
This is NOT retraining. The shared DaiSEE CNNs, the UBFC-Phys GRU and the
fusion are loaded read-only and identical for every user. What a returning
user can get is ONE bounded number, fitted from their own rated sessions: a
shift added to the general model's smoothed final stress index.

Which score space, and why
--------------------------
The app's decision is made on the SMOOTHED FINAL INDEX: HysteresisLevel reads
LOW below config.LOW_TO_MODERATE_CUTOFF (0.24), and that cutoff was derived on
that very quantity (ROC / Youden's J, leave-one-subject-out -
evaluation_data/THRESHOLD_RESULTS_2026-09-22.txt). The same quantity is what
samples.stress_index stores. So the shift is estimated AND applied there:

  * estimate: per rated session, g = mean of its stored stress_index (the
    general reading - LiveRun always stores the general value, never the
    personal one, so this can never learn from its own output);
  * apply:    personal reading = clip(general smoothed index + shift, 0, 1),
    with its own hysteresis, AFTER the general pipeline has run unchanged.

TemporalSmoother is a moving average, so adding a constant to the smoothed
index is the same as adding it to every frame: a shift of d moves a session's
mean index by exactly d (except where the 0/1 clip binds). Nothing has to be
assumed about where the behaviour branch "should" sit - an earlier draft did
exactly that (a behaviour-score offset aimed at the final-index cutoff) and was
replaced because the behaviour weight in the fusion varies from 0.73 to 1.0
with CNN/GRU availability, so the cutoff does not transfer to that scale.

The rule
--------
Labels are the project's existing convention (retrain_from_sessions.py):
rating <= 2 relaxed, >= 4 stressed, 3 excluded. With mu_r / mu_s the means of
g over the user's relaxed / stressed sessions (newest 30, each >= 30 s):

  shift = n/(n + 6) * (cutoff - (mu_r + mu_s)/2), clipped to +/- 0.10

i.e. move the midpoint between the user's own calm and stressed sessions onto
the cutoff the app actually uses, shrunk toward "no change" when evidence is
thin. A session is then read as stressed when g + shift >= cutoff.

Switched on only when ALL hold:
  1. >= 6 eligible sessions with >= 2 relaxed and >= 2 stressed;
  2. mu_s > mu_r (the general score already ranks this user's stressed
     sessions higher on average - a constant shift cannot fix a wrong order);
  3. leave-one-session-out: refitting without each session in turn and
     classifying that held-out session gives a strictly higher balanced
     accuracy than the general cutoff does on the same sessions.
Otherwise the shift is 0.0 and the user gets the general model, bit for bit.
Only COMPATIBLE sessions are pooled (load()): same task mode as the coming run
(modes re-weight the behaviour signals) and recorded since the user's current
baseline took effect (UserState.baseline_set_at, stored inside baseline_json
on the sessions' own user-local clock). If that time is unknown - profiles
saved before it was recorded - nothing is pooled until the next calibration.

Passing (3) means "agreed better with this user's own past ratings", which is
evidence on a handful of sessions - not a demonstrated accuracy improvement.

Stage 2: which of YOUR signals rise when you are stressed (2026-11-09)
-------------------------------------------------------------------
The rule above can only slide a user's whole reading up or down; it cannot
help a user whose overall score does not rise when they feel stressed. Stage 2
looks at the seven behaviour signals of each rated session (their session
means, x) and fits, for this user only, a per-frame adjustment

  a(x) = d + sum_j beta_j * (x_j - r_j),     clipped to +/- PERSONAL_MAX_ADJUST

added to the general smoothed index, where r is the user's own calm-session
mean of each signal. It is a weighted logistic model in final-index units
(fixed slope PERSONAL_LOGIT_SLOPE: a session is stressed when g + a(x) >=
cutoff), fitted as a MAP estimate with Gaussian priors that pull d and every
beta towards 0 - with little evidence it stays close to "no change". Newer
sessions count more (half-life PERSONAL_RECENCY_HALF_LIFE sessions) and so do
clear ratings (1 or 5 count more than 2 or 4).

Stage 2 needs more evidence than stage 1 (PERSONAL_SIGNAL_MIN_SESSIONS, with
PERSONAL_SIGNAL_MIN_PER_CLASS of each class) and is used ONLY if its
leave-one-session-out balanced accuracy beats both the general model and stage
1 by at least PERSONAL_SIGNAL_MIN_GAIN. Otherwise stage 1's result stands
(including "general model"). Live, a(x) is computed from the same seven
signals every frame and smoothed over the same window as the index.
"""
import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

import config

# Same order and names as the samples table's signal columns
# (session_history.SIGNAL_COLUMNS) and the fusion's contributions dict.
SIGNALS = ('eye_strain', 'blink_stress', 'brow_tension', 'jaw_tension',
           'head_instability', 'neck_strain', 'slouch')


def parse_local_time(value):
    """A stored user-local timestamp (ISO text, no zone - how started_at and
    baseline_set_at are written) as a naive datetime, or None when missing,
    unparseable or zone-aware (a different clock: not comparable)."""
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
    return None if dt.tzinfo is not None else dt


@dataclass(frozen=True)
class Personalization:
    shift: float           # added to the general smoothed index; 0.0 = general model
    active: bool
    n_sessions: int        # eligible sessions used for the fit
    n_relaxed: int
    n_stressed: int
    reason: str            # one line for the UI / logs (no personal data)
    loo_general: Optional[float] = None    # session-level balanced accuracy,
    loo_personal: Optional[float] = None   # leave-one-session-out
    model: str = 'shift'                   # 'shift' (stage 1) or 'signals' (stage 2)
    weights: Optional[tuple] = None        # stage 2: ((signal, beta), ...)
    reference: Optional[tuple] = None      # stage 2: ((signal, calm mean), ...)
    confidence: str = ''                   # 'low' | 'medium' | 'high' when active

    def adjustment(self, contributions) -> float:
        """The amount added to the general smoothed index for one frame's
        signal values (a dict like the fusion's 'contributions'). Stage 1:
        the constant shift. Stage 2: d + sum beta_j (x_j - r_j), bounded.
        A missing or non-finite signal counts as the user's calm level."""
        if not self.active:
            return 0.0
        if self.model != 'signals' or not self.weights:
            return float(self.shift)
        ref = dict(self.reference or ())
        a = float(self.shift)
        for name, beta in self.weights:
            v = (contributions or {}).get(name)
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                a += float(beta) * (v - ref.get(name, 0.0))
        lim = config.PERSONAL_MAX_ADJUST
        return max(-lim, min(lim, a)) if math.isfinite(a) else 0.0

    def describe(self) -> str:
        conf = f"; confidence {self.confidence}" if self.confidence else ""
        if self.active and self.model == 'signals':
            top = sorted(self.weights or (), key=lambda kv: -abs(kv[1]))[:2]
            names = " and ".join(k.replace('_', ' ') for k, b in top if abs(b) > 1e-3)
            return (f"Personal calibration on: your live reading is adjusted using the "
                    f"signals that rise when YOU feel stressed"
                    f"{' (mainly ' + names + ')' if names else ''}, learned from "
                    f"{self.n_sessions} of your rated sessions (held-out agreement with "
                    f"your ratings {self.loo_personal:.0%} vs {self.loo_general:.0%} for "
                    f"the general model{conf}). Saved history keeps the general reading.")
        if self.active:
            return (f"Personal calibration on: your live reading is shifted by "
                    f"{self.shift:+.3f}, learned from {self.n_sessions} of your rated "
                    f"sessions (held-out agreement with your ratings "
                    f"{self.loo_personal:.0%} vs {self.loo_general:.0%} for the general "
                    f"model{conf}). Saved history keeps the general reading.")
        return f"General model - {self.reason}"


def general(reason: str, n=0, n_relaxed=0, n_stressed=0,
            loo_general=None, loo_personal=None) -> Personalization:
    return Personalization(0.0, False, n, n_relaxed, n_stressed, reason,
                           loo_general, loo_personal)


def enabled_for(storage_backend: str) -> bool:
    if not config.PERSONALIZATION_ENABLED:
        return False
    return storage_backend == 'supabase' or config.PERSONALIZE_LOCAL_SQLITE


def _label(rating):
    """'relaxed' / 'stressed' / None (unrated, ambiguous or invalid)."""
    if isinstance(rating, bool) or not isinstance(rating, int):
        return None
    if not 1 <= rating <= 5:
        return None
    if rating <= config.PERSONAL_RELAXED_AT:
        return 'relaxed'
    if rating >= config.PERSONAL_STRESSED_AT:
        return 'stressed'
    return None


def _usable_mean(ex):
    """The session's mean general index if it can be used, else None."""
    m = ex.get('mean_index')
    if m is None:
        return None
    m = float(m)
    if not math.isfinite(m) or not 0.0 <= m <= 1.0:
        return None
    if int(ex.get('n_index') or 0) < config.PERSONAL_MIN_SAMPLES_PER_SESSION:
        return None
    return m


def _points(examples):
    """[(is_stressed, mean_index)] - eligible sessions, newest first, capped."""
    pts = []
    for ex in examples:
        lab = _label(ex.get('self_report'))
        m = _usable_mean(ex)
        if lab is None or m is None:
            continue
        pts.append((lab == 'stressed', m))
        if len(pts) >= config.PERSONAL_MAX_SESSIONS:
            break
    return pts


def _shift(pts):
    """The shrunk, bounded midpoint shift for these points, or None when they
    cannot support one (a class under the minimum, or wrong ordering)."""
    rel = [m for s, m in pts if not s]
    stz = [m for s, m in pts if s]
    if len(rel) < config.PERSONAL_MIN_PER_CLASS or len(stz) < config.PERSONAL_MIN_PER_CLASS:
        return None
    mu_r, mu_s = sum(rel) / len(rel), sum(stz) / len(stz)
    if mu_s <= mu_r:
        return None
    n = len(pts)
    d = n / (n + config.PERSONAL_PRIOR_SESSIONS) * (
        config.LOW_TO_MODERATE_CUTOFF - 0.5 * (mu_r + mu_s))
    return max(-config.PERSONAL_MAX_SHIFT, min(config.PERSONAL_MAX_SHIFT, d))


def _balanced_accuracy(pts, shifts):
    """Session-level: stressed when mean + shift >= cutoff (HysteresisLevel's
    LOW boundary). Mean of the two per-class recalls."""
    cut = config.LOW_TO_MODERATE_CUTOFF
    hit = {True: 0, False: 0}
    tot = {True: 0, False: 0}
    for (stressed, m), d in zip(pts, shifts):
        tot[stressed] += 1
        hit[stressed] += int((m + d >= cut) == stressed)
    return 0.5 * (hit[True] / tot[True] + hit[False] / tot[False])


def fit(examples) -> Personalization:
    """
    examples: iterable of dicts with 'self_report', 'n_index' and
    'mean_index' (one per saved session of ONE user, newest first), and
    optionally 'signals' ({signal: session mean}) for stage 2.
    Pure function - no I/O - so it is easy to test and reason about.
    """
    examples = list(examples)
    p1 = _fit_stage1(examples)
    p2 = _fit_stage2(examples, p1)
    return p2 if p2 is not None else p1


def _confidence(n, loo_personal, loo_general):
    margin = (loo_personal or 0.0) - (loo_general or 0.0)
    if n >= 15 and margin >= 0.15:
        return 'high'
    if n >= 10 and margin >= 0.10:
        return 'medium'
    return 'low'


def _fit_stage1(examples) -> Personalization:
    pts = _points(examples)
    n = len(pts)
    n_s = sum(1 for s, _ in pts if s)
    n_r = n - n_s
    need, per = config.PERSONAL_MIN_LABELED_SESSIONS, config.PERSONAL_MIN_PER_CLASS
    if n < need or n_r < per or n_s < per:
        return general(
            f"needs {need} rated sessions incl. {per} calm (1-2) and {per} stressed (4-5); "
            f"you have {n} ({n_r} calm, {n_s} stressed).", n, n_r, n_s)

    full = _shift(pts)
    if full is None:
        return general("on your rated sessions the general score is not higher when you "
                       "felt stressed, so a constant shift cannot help.", n, n_r, n_s)

    # Leave-one-session-out: each held-out session is classified with a shift
    # fitted WITHOUT it; a fold that cannot support a fit falls back to the
    # general model (shift 0) for that session - never to the full-data shift.
    loo = []
    for i in range(n):
        d = _shift(pts[:i] + pts[i + 1:])
        loo.append(0.0 if d is None else d)
    ba_general = _balanced_accuracy(pts, [0.0] * n)
    ba_personal = _balanced_accuracy(pts, loo)
    if not ba_personal > ba_general:
        return general(
            f"tested on your own sessions (leave-one-out) it would not have agreed better "
            f"with your ratings ({ba_personal:.0%} vs {ba_general:.0%}), so the general "
            f"model is kept.", n, n_r, n_s, ba_general, ba_personal)
    return Personalization(float(full), True, n, n_r, n_s, "validated",
                           ba_general, ba_personal, model='shift',
                           confidence=_confidence(n, ba_personal, ba_general))


# ---------------------------------------------------------------------------
# Stage 2: per-user signal weighting
# ---------------------------------------------------------------------------
def _signal_vector(ex):
    """The session's seven signal means as an array, or None if unusable."""
    sig = ex.get('signals')
    if not isinstance(sig, dict):
        return None
    out = []
    for name in SIGNALS:
        try:
            v = float(sig.get(name))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(v) or not -0.01 <= v <= 1.01:
            return None
        out.append(min(1.0, max(0.0, v)))
    return np.array(out, dtype=float)


def _points2(examples):
    """[(is_stressed, mean_index, signals, weight)] for stage 2 - the same
    eligible sessions as stage 1 (newest first, capped), each weighted by
    recency and by how clear its rating is."""
    pts = []
    for ex in examples:
        lab = _label(ex.get('self_report'))
        m = _usable_mean(ex)
        if lab is None or m is None:
            continue
        x = _signal_vector(ex)
        if x is None:
            continue
        rank = len(pts)                                # 0 = newest
        w = (0.5 ** (rank / config.PERSONAL_RECENCY_HALF_LIFE)
             * config.PERSONAL_RATING_WEIGHT.get(int(ex['self_report']), 0.6))
        pts.append((lab == 'stressed', m, x, w))
        if len(pts) >= config.PERSONAL_MAX_SESSIONS:
            break
    return pts


def _fit_signals(pts):
    """MAP fit of (d, beta) on these points, or None when a class is under
    the stage-2 minimum. Returns (d, beta array, reference array)."""
    y = np.array([1.0 if s else 0.0 for s, _, _, _ in pts])
    if (y.sum() < config.PERSONAL_SIGNAL_MIN_PER_CLASS
            or (1 - y).sum() < config.PERSONAL_SIGNAL_MIN_PER_CLASS):
        return None
    g = np.array([m for _, m, _, _ in pts])
    X = np.vstack([x for _, _, x, _ in pts])
    w = np.array([wt for _, _, _, wt in pts])
    calm = y == 0
    ref = (w[calm, None] * X[calm]).sum(axis=0) / w[calm].sum()
    Z = np.hstack([np.ones((len(pts), 1)), X - ref])           # [1, x - r]
    k = config.PERSONAL_LOGIT_SLOPE
    off = k * (g - config.LOW_TO_MODERATE_CUTOFF)
    prec = np.array([1.0 / config.PERSONAL_SHIFT_PRIOR_SD ** 2]
                    + [1.0 / config.PERSONAL_SIGNAL_PRIOR_SD ** 2] * len(SIGNALS))
    theta = np.zeros(Z.shape[1])
    for _ in range(50):                                        # Newton (convex)
        eta = off + k * (Z @ theta)
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -40, 40)))
        grad = k * (Z.T @ (w * (p - y))) + prec * theta
        hess = k * k * (Z.T * (w * p * (1 - p))) @ Z + np.diag(prec)
        step = np.linalg.solve(hess, grad)
        theta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    if not np.all(np.isfinite(theta)):
        return None
    return float(theta[0]), theta[1:], ref


def _adjust(d, beta, ref, x):
    a = d + float(beta @ (x - ref))
    lim = config.PERSONAL_MAX_ADJUST
    return max(-lim, min(lim, a))


def _fit_stage2(examples, p1):
    """A stage-2 Personalization if it clearly beats the general model and
    stage 1 on held-out sessions, else None (stage 1's result stands)."""
    if not config.PERSONAL_SIGNALS_ENABLED:
        return None
    pts = _points2(examples)
    n = len(pts)
    n_s = sum(1 for s, _, _, _ in pts if s)
    n_r = n - n_s
    per = config.PERSONAL_SIGNAL_MIN_PER_CLASS
    if n < config.PERSONAL_SIGNAL_MIN_SESSIONS or n_s < per or n_r < per:
        return None
    full = _fit_signals(pts)
    if full is None:
        return None
    # Leave-one-session-out, as in stage 1: the held-out session is scored
    # with (d, beta, r) fitted WITHOUT it; an unfittable fold scores it with
    # the general model.
    loo_adj = []
    for i in range(n):
        f = _fit_signals(pts[:i] + pts[i + 1:])
        loo_adj.append(0.0 if f is None else _adjust(f[0], f[1], f[2], pts[i][2]))
    pairs = [(s, m) for s, m, _, _ in pts]
    ba_general = _balanced_accuracy(pairs, [0.0] * n)
    ba_signals = _balanced_accuracy(pairs, loo_adj)
    best_simpler = ba_general
    if p1.active and p1.loo_personal is not None:
        best_simpler = max(best_simpler, p1.loo_personal)
    if not (ba_signals > ba_general
            and ba_signals >= best_simpler + config.PERSONAL_SIGNAL_MIN_GAIN):
        return None
    d, beta, ref = full
    return Personalization(
        float(max(-config.PERSONAL_MAX_ADJUST, min(config.PERSONAL_MAX_ADJUST, d))),
        True, n, n_r, n_s, "validated (signal weighting)", ba_general, ba_signals,
        model='signals',
        weights=tuple((name, float(b)) for name, b in zip(SIGNALS, beta)),
        reference=tuple((name, float(r)) for name, r in zip(SIGNALS, ref)),
        confidence=_confidence(n, ba_signals, ba_general))


def _compatible(row, task_mode, since):
    """Was this session scored the way the coming run will be? Same task mode
    (the mode re-weights the behaviour signals, so scores from different modes
    are not on one scale) and recorded no earlier than the current baseline
    took effect (an older baseline scores the same face differently). A
    session whose start time cannot be read is treated as incompatible."""
    if (row.get('task_mode') or 'Default') != task_mode:
        return False
    started = parse_local_time(row.get('started_at'))
    return started is not None and started >= since


def _row_index_stats(row):
    """(count, mean) of the session's general index straight from its
    sessions-table row - save_session() stores sample_count and mean_stress
    from the very same summarize_history() over the very same stress_index
    values the samples table holds - or None when the row lacks them (older
    rows, or a store that does not return them), in which case the samples
    are read as before."""
    try:
        n = int(row.get('sample_count'))
        m = float(row.get('mean_stress'))
    except (TypeError, ValueError):
        return None
    if n <= 0 or not math.isfinite(m):
        return None
    return n, m


def _attach_signals(store, examples, cache):
    """Add each example's signal means (stage 2) when there are enough
    sessions for stage 2 to run and the store can provide them. Any problem
    here only disables stage 2 - stage 1 still runs on the index means."""
    if (not config.PERSONAL_SIGNALS_ENABLED
            or len(examples) < config.PERSONAL_SIGNAL_MIN_SESSIONS
            or not hasattr(store, 'session_signal_means')):
        return
    try:
        for ex in examples:
            sid = ex['_sid']
            key = ('signals', sid)
            if cache is not None and key in cache:
                ex['signals'] = cache[key]
                continue
            sig = store.session_signal_means(sid)
            ex['signals'] = sig
            if cache is not None:
                cache[key] = sig
    except Exception as e:
        print(f"[personalization] signal history unavailable ({type(e).__name__}); "
              f"stage 1 only")
        for ex in examples:
            ex.pop('signals', None)


def load(store, user_id, cache=None, *, task_mode='Default',
         baseline_set_at=None) -> Personalization:
    """
    Fit the personalization for `user_id` from `store` (a SupabaseStore bound
    to that signed-in user, or the local session_history module). Any storage
    problem falls back to the general model - never to someone else's data.

    Only COMPATIBLE sessions are pooled: same `task_mode` as the coming run,
    started at or after `baseline_set_at` (when the user's current baseline
    took effect, on the same user-local clock as started_at). With no known
    baseline time - profiles saved before it was recorded, or never calibrated
    - compatibility cannot be established and the general model is used.

    cache: optional dict {session_id: (n, mean)}. Saved samples never change,
    so a session's summary can be reused within one browser session; the
    caller keeps one dict per user.
    """
    since = parse_local_time(baseline_set_at)
    if since is None:
        return general("calibrate once so sessions can be matched to your current "
                       "baseline; sessions recorded after that count.")
    scope = f" (counting {task_mode} sessions since your last calibration)"
    try:
        rows = store.labeled_sessions(min_samples=config.PERSONAL_MIN_SAMPLES_PER_SESSION)
        rows = [r for r in rows if r.get('user_id') == user_id]
        rows.sort(key=lambda r: r['id'], reverse=True)          # newest first
        examples = []
        for r in rows:
            if _label(r.get('self_report')) is None or not _compatible(r, task_mode, since):
                continue                                         # skip the fetch
            sid = r['id']
            row_stats = _row_index_stats(r)
            if row_stats is not None:
                n_i, mean_i = row_stats           # no request: the session row has it
            elif cache is not None and sid in cache:
                n_i, mean_i = cache[sid]
            else:
                n_i, mean_i = store.session_index_stats(sid)
                if cache is not None:
                    cache[sid] = (n_i, mean_i)
            ex = {'self_report': r.get('self_report'), 'n_index': n_i, 'mean_index': mean_i,
                  '_sid': sid}
            if _usable_mean(ex) is None:
                continue                  # an unusable session must not take a slot
            examples.append(ex)
            if len(examples) >= config.PERSONAL_MAX_SESSIONS:
                break
        _attach_signals(store, examples, cache)
    except Exception as e:
        # Type only - never the message, which could carry request details.
        print(f"[personalization] history unavailable ({type(e).__name__}); general model")
        return general("your history could not be loaded right now.")
    p = fit(examples)
    return p if p.active else dataclasses.replace(p, reason=p.reason.rstrip('.') + scope + '.')
