"""
Intervention Agent
==================
The product's agentic core: during a live session it PERCEIVES the signal
state, DECIDES whether interrupting the user is warranted, ACTS with an
intervention targeted at whatever is actually driving the stress, then
MEASURES whether it worked and LEARNS which interventions help this user.

This replaces the static guidance panel's behaviour ("stress is HIGH, here
are the same three tips as always") with a closed loop:

    signals -> decide (worth interrupting? what for?) -> targeted action
            -> watch the next few minutes -> helped / no effect
            -> prefer what works, retire what does not

Everything is on-device and interpretable: the decision rules are readable,
the learning is a Thompson-Sampling bandit over helped/failed counts per
intervention (Beta(helped+1, no_effect+1) - the same Laplace-smoothed prior
success_rate() always used, now sampled from instead of only averaged), and
the whole action log lives in the local SQLite database.

Deliberate non-goals: no LLM, no cloud, no personality. The agent's job is
timing and targeting, not conversation.

2026-08-31 additions (see PLAN_FOR_PP2.md's status snapshot for the full
reasoning): Thompson Sampling replaces a fixed 0.7/0.3 blend constant for
ranking candidates; retirement uses a Wilson confidence interval
(eval_scoring.wilson()) instead of a raw trial-count floor; the agent holds
back if the index is already falling on its own; per-(task_mode,
intervention) success is tracked with a fallback to the global stat until a
context bucket has enough evidence; the general break can compete against a
specific signal when no single one clearly dominates; and status() explains
its own choice instead of only reporting a phase name.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

import config
from eval_scoring import wilson

# ---------------------------------------------------------------------------
# The intervention catalogue.
# Each entry is keyed by the signal that triggers it; 'general' fires when
# stress is high but no single signal dominates. Text is imperative and
# specific - "unclench your jaw" beats "try to relax".
# ---------------------------------------------------------------------------
CATALOGUE = {
    'jaw_tension': {
        'kind': 'unclench_jaw',
        'title': 'Unclench your jaw',
        'body': 'Your jaw has been tight for a while. Let your teeth part '
                'slightly and rest your tongue on the roof of your mouth.',
    },
    'brow_tension': {
        'kind': 'relax_brow',
        'title': 'Soften your brow',
        'body': 'Your brow is furrowed. Raise your eyebrows once, let them '
                'drop, and let your forehead go slack.',
    },
    'slouch': {
        'kind': 'reset_posture',
        'title': 'Reset your posture',
        'body': 'You have been sinking toward the screen. Sit back, stack '
                'your head over your shoulders, drop your shoulders.',
    },
    'neck_strain': {
        'kind': 'neck_release',
        'title': 'Release your neck',
        'body': 'Your head is craning forward. Tuck your chin gently and '
                'roll your shoulders back once.',
    },
    'eye_strain': {
        'kind': 'eyes_20_20_20',
        'title': 'Rest your eyes',
        'body': 'Sustained eye strain. Look at something ~20 feet away for '
                '20 seconds. Blink slowly a few times.',
    },
    'blink_stress': {
        'kind': 'slow_blinks',
        'title': 'Slow your blinking',
        'body': 'Your blink rate is elevated. Close your eyes for two '
                'seconds, twice. Breathe out longer than you breathe in.',
    },
    'head_instability': {
        'kind': 'settle_breathing',
        'title': 'Settle for a moment',
        'body': 'You are moving restlessly. Plant both feet, one slow '
                'breath in for 4, out for 6.',
    },
    'general': {
        'kind': 'short_break',
        'title': 'Take a two-minute break',
        'body': 'Several stress signals are elevated together. Step away '
                'from the screen for two minutes - this is the point where '
                'a break actually pays for itself.',
    },
}


@dataclass
class ActiveIntervention:
    kind: str
    title: str
    body: str
    trigger_signal: Optional[str]
    t_shown: float
    index_before: float
    resolved: bool = False
    # Explainability (see status()/_explain()): what the picked signal's
    # raw value was, and the point-estimate success rate the choice was
    # made against, both captured at fire time rather than recomputed
    # later against whatever the stats happen to be by then.
    trigger_value: Optional[float] = None
    success_rate_at_fire: float = 0.5


@dataclass
class InterventionAgent:
    """
    Drive with update() once per recorded frame. Returns the currently
    active intervention (for the UI) or None.

    Learning: a Thompson-Sampling bandit over helped/failed counts per
    kind (see _sample_success()), seeded from the database at construction
    (session_history.intervention_stats/intervention_stats_by_mode).
    Interventions that keep failing for THIS user are deprioritised below
    ones that keep working, with real exploration (not a fixed blend) when
    the choice is close; sustained, statistically-confident failure
    retires a kind for the session (see _retired()).

    mode_stats/task_mode (both optional): per-(task_mode, kind) outcome
    counts and the CURRENT session's mode. When given, success_rate()/
    _sample_success()/_retired() prefer the mode-specific bucket once it
    has config.INTERVENTION_MODE_MIN_TRIALS outcomes, falling back to the
    global, mode-blind stat otherwise - context matters (the same lesson
    this project's warm-start personalization research found), but
    slicing by context from the very first session would just starve
    every bucket of evidence.
    """
    stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    mode_stats: Dict[str, Dict[str, Dict[str, int]]] = field(default_factory=dict)
    task_mode: Optional[str] = None
    rng: object = None

    _active: Optional[ActiveIntervention] = None
    _last_fired_at: float = -1e9
    _log: List[Dict] = field(default_factory=list)
    _session_start: float = 0.0
    _index_history: deque = field(default_factory=deque)

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng()

    # -- lifecycle ----------------------------------------------------------
    def start_session(self, t: float):
        self._active = None
        self._last_fired_at = -1e9
        self._log = []
        self._session_start = t
        self._index_history = deque()

    def _outcome_counts(self, kind: str):
        """
        (helped, no_effect) counts for `kind` - the per-task-mode bucket if
        it has enough evidence (config.INTERVENTION_MODE_MIN_TRIALS),
        otherwise the global, mode-blind bucket. Single source of truth
        for success_rate()/_sample_success()/_retired(), so all three
        agree on which evidence is actually being trusted right now.
        """
        if self.task_mode is not None:
            m = self.mode_stats.get(self.task_mode, {}).get(kind)
            if m is not None:
                h, f = m.get('helped', 0), m.get('no_effect', 0)
                if h + f >= config.INTERVENTION_MODE_MIN_TRIALS:
                    return h, f
        g = self.stats.get(kind, {})
        return g.get('helped', 0), g.get('no_effect', 0)

    def success_rate(self, kind: str) -> float:
        """Laplace-smoothed success estimate; unknown kinds start at 0.5.
        This is the MEAN of the Beta(helped+1, no_effect+1) posterior -
        see _sample_success() for the stochastic draw from that same
        distribution that the bandit actually ranks candidates with."""
        helped, failed = self._outcome_counts(kind)
        return (helped + 1) / (helped + failed + 2)

    def _sample_success(self, kind: str) -> float:
        """Thompson-Sampling draw from Beta(helped+1, no_effect+1) - a
        genuine stochastic sample from the same posterior success_rate()
        only averages. An intervention with little evidence yet (wide
        posterior) can still occasionally outscore the current best-looking
        one, which is what lets the agent keep exploring instead of
        permanently locking onto whichever kind happened to look best
        first. See config.INTERVENTION_EXPLORE_WEIGHT for how much this
        can move a ranking."""
        helped, failed = self._outcome_counts(kind)
        return float(self.rng.beta(helped + 1, failed + 1))

    def _is_declining(self) -> bool:
        """True if the smoothed index has been falling meaningfully over
        the trend window - likely resolving on its own, so firing now
        would be an unnecessary interruption (the same "don't poison
        outcome data by nagging" concern the cooldown exists for, from a
        different angle)."""
        if len(self._index_history) < 2:
            return False
        t0, i0 = self._index_history[0]
        t1, i1 = self._index_history[-1]
        if t1 - t0 < config.INTERVENTION_TREND_MIN_SPAN_SECONDS:
            return False
        return (i0 - i1) >= config.INTERVENTION_TREND_DECLINE_DROP

    # -- the loop -----------------------------------------------------------
    def update(self, t: float, smoothed_index: float,
               contributions: Dict[str, float]) -> Optional[ActiveIntervention]:
        # Trend tracking - kept current on every call, resolving or not,
        # so _is_declining() always reflects real recent history rather
        # than only whatever happened to be seen while nothing was active.
        self._index_history.append((t, smoothed_index))
        cutoff = t - config.INTERVENTION_TREND_WINDOW_SECONDS
        while self._index_history and self._index_history[0][0] < cutoff:
            self._index_history.popleft()

        # 1. Resolve an outstanding intervention once its window elapses.
        if self._active and not self._active.resolved:
            if t - self._active.t_shown >= config.INTERVENTION_EVAL_SECONDS:
                self._resolve(t, smoothed_index)
            elif t - self._active.t_shown >= config.INTERVENTION_SHOW_SECONDS:
                # Card may be hidden by UI, but the eval window hasn't
                # closed yet - do not fall through to the targeting/firing
                # logic below and silently start a second intervention
                # before this one's outcome has been measured. Enforced
                # here rather than relying on INTERVENTION_COOLDOWN_SECONDS
                # happening to exceed INTERVENTION_EVAL_SECONDS.
                return None
            else:
                return self._active     # still showing

        # 2. Decide whether firing is warranted at all.
        if smoothed_index < config.INTERVENTION_MIN_INDEX:
            return None
        if t - self._session_start < config.INTERVENTION_WARMUP_SECONDS:
            return None                 # smoother still filling / settling in
        if t - self._last_fired_at < config.INTERVENTION_COOLDOWN_SECONDS:
            return None                 # do not nag
        if self._is_declining():
            return None                 # already resolving on its own

        # 3. Target: signals above threshold are candidates, scored by
        #    urgency (raw value) with a bounded Thompson-Sampling nudge
        #    from what's actually worked before - a real difference in
        #    urgency always wins outright (see INTERVENTION_EXPLORE_WEIGHT's
        #    provenance comment); only a close call lets history decide,
        #    with genuine exploration rather than a fixed blend.
        candidates = [
            (name, val) for name, val in contributions.items()
            if name in CATALOGUE and val >= config.INTERVENTION_SIGNAL_THRESHOLD
        ]

        # When several signals are elevated with no single clear driver,
        # let the general break compete on equal footing instead of always
        # deferring to whichever one happens to be a few points higher.
        if len(candidates) >= 2:
            by_value = sorted(candidates, key=lambda nv: nv[1], reverse=True)
            top_gap = by_value[0][1] - by_value[1][1]
            if top_gap < config.INTERVENTION_MULTI_SIGNAL_GAP:
                candidates = candidates + [('__general__', smoothed_index)]

        def _kind_of(name):
            return CATALOGUE['general']['kind'] if name == '__general__' \
                else CATALOGUE[name]['kind']

        def _score(nv):
            name, val = nv
            explore = self._sample_success(_kind_of(name)) - 0.5
            return val + config.INTERVENTION_EXPLORE_WEIGHT * explore

        if candidates:
            candidates.sort(key=_score, reverse=True)
            signal, trigger_value = candidates[0]
            if signal == '__general__':
                signal, entry = None, CATALOGUE['general']
            else:
                entry = CATALOGUE[signal]
        else:
            signal, trigger_value, entry = None, None, CATALOGUE['general']

        # 4. Retire interventions we're statistically confident don't work
        # on this user (Wilson upper bound below the threshold - see
        # config.INTERVENTION_RETIRE_BELOW). Falling back to 'general' must
        # not skip re-checking IT against the same rule - otherwise a user
        # for whom BOTH the targeted intervention AND the general break
        # have independently failed keeps getting shown the general break
        # forever, contradicting the "even the break does nothing; stop
        # nagging" guarantee below (that branch only fired when 'general'
        # was the first pick, never when reached via this fallback).
        def _retired(kind):
            helped, failed = self._outcome_counts(kind)
            total = helped + failed
            if total == 0:
                return False
            _, hi = wilson(helped, total)
            return hi < config.INTERVENTION_RETIRE_BELOW

        if _retired(entry['kind']):
            fallback = CATALOGUE['general']
            if entry['kind'] != fallback['kind'] and not _retired(fallback['kind']):
                entry, signal, trigger_value = fallback, None, smoothed_index
            else:
                return None             # even the break does nothing; stop nagging

        # 5. Act.
        self._active = ActiveIntervention(
            kind=entry['kind'], title=entry['title'], body=entry['body'],
            trigger_signal=signal, t_shown=t, index_before=smoothed_index,
            trigger_value=trigger_value,
            success_rate_at_fire=self.success_rate(entry['kind']),
        )
        self._last_fired_at = t
        return self._active

    def _resolve(self, t: float, smoothed_index: float):
        """Measure: did the index actually drop after the intervention?"""
        a = self._active
        drop = a.index_before - smoothed_index
        outcome = ('helped' if drop >= config.INTERVENTION_HELPED_DROP
                   else 'no_effect')
        self.stats.setdefault(a.kind, {'helped': 0, 'no_effect': 0})
        self.stats[a.kind][outcome] = self.stats[a.kind].get(outcome, 0) + 1
        self._log.append({
            'ts': datetime.now(), 'kind': a.kind,
            'trigger_signal': a.trigger_signal,
            'index_before': a.index_before, 'index_after': smoothed_index,
            'outcome': outcome,
        })
        a.resolved = True
        self._active = None

    # -- visibility ---------------------------------------------------------
    def _explain(self, iv: ActiveIntervention) -> str:
        """Why THIS was picked, not just what - the point of surfacing it
        at all is so the agent's choice is auditable, not a black box."""
        pct = f"{iv.success_rate_at_fire * 100:.0f}%"
        if iv.trigger_signal:
            return (f"{iv.trigger_signal.replace('_', ' ')} at "
                    f"{iv.trigger_value:.2f} (elevated) - {pct} historical success")
        return f"several signals elevated together - {pct} historical success"

    def status(self, t: float) -> Dict[str, str]:
        """
        Live state for the UI, so the agent is VISIBLY working even while it
        correctly decides to do nothing. Phases:
          warming    - settling in, will not interrupt yet
          watching   - monitoring, ready to act
          declining  - stress is elevated but already falling; holding back
          showing    - an intervention card is up
          evaluating - measuring whether the intervention worked
          cooldown   - deliberately holding back to avoid nagging
        """
        if self._active and not self._active.resolved:
            since = t - self._active.t_shown
            if since < config.INTERVENTION_SHOW_SECONDS:
                return {'phase': 'showing',
                        'detail': f"intervening: {self._active.title} - "
                                  f"{self._explain(self._active)}"}
            left = max(0, config.INTERVENTION_EVAL_SECONDS - since)
            return {'phase': 'evaluating',
                    'detail': f"measuring effect of "
                              f"“{self._active.title}” "
                              f"({left:.0f}s left)"}
        if t - self._session_start < config.INTERVENTION_WARMUP_SECONDS:
            left = config.INTERVENTION_WARMUP_SECONDS - (t - self._session_start)
            return {'phase': 'warming',
                    'detail': f"settling in ({left:.0f}s)"}
        since_fire = t - self._last_fired_at
        if since_fire < config.INTERVENTION_COOLDOWN_SECONDS:
            left = config.INTERVENTION_COOLDOWN_SECONDS - since_fire
            return {'phase': 'cooldown',
                    'detail': f"holding back ({left:.0f}s) - "
                              f"nagging poisons outcome data"}
        if self._is_declining():
            return {'phase': 'declining',
                    'detail': 'stress elevated but already dropping - '
                              'holding back, likely resolving on its own'}
        return {'phase': 'watching',
                'detail': 'monitoring all seven signals'}

    def learned_summary(self) -> str:
        """One line of what the agent has learned, for the UI."""
        measured = [(k, v.get('helped', 0), v.get('no_effect', 0))
                    for k, v in self.stats.items()
                    if v.get('helped', 0) + v.get('no_effect', 0) > 0]
        if not measured:
            return "no measured outcomes yet - it learns as you use it"
        measured.sort(key=lambda kv: -self.success_rate(kv[0]))
        k, h, f = measured[0]
        total = sum(h + f for _, h, f in measured)
        return (f"{total} outcome(s) measured; most effective so far: "
                f"{k.replace('_', ' ')} ({h}/{h + f} helped)")

    # -- session end --------------------------------------------------------
    def finish_session(self, t: float, smoothed_index: float) -> List[Dict]:
        """
        Resolve anything still pending with the data available, then hand the
        log over for persistence (session_history.log_interventions).
        """
        if self._active and not self._active.resolved:
            if t - self._active.t_shown >= config.INTERVENTION_MIN_EVAL_SECONDS:
                self._resolve(t, smoothed_index)
            else:
                a = self._active
                self._log.append({
                    'ts': datetime.now(), 'kind': a.kind,
                    'trigger_signal': a.trigger_signal,
                    'index_before': a.index_before, 'index_after': None,
                    'outcome': 'unknown',
                })
                self._active = None
        log, self._log = self._log, []
        return log
