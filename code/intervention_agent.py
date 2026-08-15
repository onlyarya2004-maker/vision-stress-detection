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
the learning is helped/failed counts per intervention (Laplace-smoothed),
and the whole action log lives in the local SQLite database.

Deliberate non-goals: no LLM, no cloud, no personality. The agent's job is
timing and targeting, not conversation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import config

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


@dataclass
class InterventionAgent:
    """
    Drive with update() once per recorded frame. Returns the currently
    active intervention (for the UI) or None.

    Learning: helped/failed counts per kind, seeded from the database at
    construction (session_history.intervention_stats), Laplace-smoothed into
    a success estimate. Interventions that keep failing for THIS user are
    deprioritised below ones that keep working; sustained failure retires a
    kind for the session.
    """
    stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    _active: Optional[ActiveIntervention] = None
    _last_fired_at: float = -1e9
    _log: List[Dict] = field(default_factory=list)
    _session_start: float = 0.0

    # -- lifecycle ----------------------------------------------------------
    def start_session(self, t: float):
        self._active = None
        self._last_fired_at = -1e9
        self._log = []
        self._session_start = t

    def success_rate(self, kind: str) -> float:
        """Laplace-smoothed success estimate; unknown kinds start at 0.5."""
        s = self.stats.get(kind, {})
        helped = s.get('helped', 0)
        failed = s.get('no_effect', 0)
        return (helped + 1) / (helped + failed + 2)

    # -- the loop -----------------------------------------------------------
    def update(self, t: float, smoothed_index: float,
               contributions: Dict[str, float]) -> Optional[ActiveIntervention]:
        # 1. Resolve an outstanding intervention once its window elapses.
        if self._active and not self._active.resolved:
            if t - self._active.t_shown >= config.INTERVENTION_EVAL_SECONDS:
                self._resolve(t, smoothed_index)
            elif t - self._active.t_shown >= config.INTERVENTION_SHOW_SECONDS:
                pass          # card may be hidden by UI; window still runs
            else:
                return self._active     # still showing

        # 2. Decide whether firing is warranted at all.
        if smoothed_index < config.INTERVENTION_MIN_INDEX:
            return None
        if t - self._session_start < config.INTERVENTION_WARMUP_SECONDS:
            return None                 # smoother still filling / settling in
        if t - self._last_fired_at < config.INTERVENTION_COOLDOWN_SECONDS:
            return None                 # do not nag

        # 3. Target: the highest signal above threshold wins; ties broken by
        #    what has actually WORKED for this user before.
        candidates = [
            (name, val) for name, val in contributions.items()
            if name in CATALOGUE and val >= config.INTERVENTION_SIGNAL_THRESHOLD
        ]
        if candidates:
            candidates.sort(
                key=lambda nv: (nv[1] * 0.7
                                + self.success_rate(CATALOGUE[nv[0]]['kind']) * 0.3),
                reverse=True)
            signal, _ = candidates[0]
            entry = CATALOGUE[signal]
        else:
            signal, entry = None, CATALOGUE['general']

        # 4. Retire interventions that demonstrably do not work on this user.
        if self.success_rate(entry['kind']) < config.INTERVENTION_RETIRE_BELOW \
                and self._total_outcomes(entry['kind']) >= config.INTERVENTION_RETIRE_MIN_TRIALS:
            fallback = CATALOGUE['general']
            if entry['kind'] != fallback['kind']:
                entry, signal = fallback, None
            else:
                return None             # even the break does nothing; stop nagging

        # 5. Act.
        self._active = ActiveIntervention(
            kind=entry['kind'], title=entry['title'], body=entry['body'],
            trigger_signal=signal, t_shown=t, index_before=smoothed_index,
        )
        self._last_fired_at = t
        return self._active

    def _total_outcomes(self, kind: str) -> int:
        s = self.stats.get(kind, {})
        return s.get('helped', 0) + s.get('no_effect', 0)

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
    def status(self, t: float) -> Dict[str, str]:
        """
        Live state for the UI, so the agent is VISIBLY working even while it
        correctly decides to do nothing. Phases:
          warming    - settling in, will not interrupt yet
          watching   - monitoring, ready to act
          showing    - an intervention card is up
          evaluating - measuring whether the intervention worked
          cooldown   - deliberately holding back to avoid nagging
        """
        if self._active and not self._active.resolved:
            since = t - self._active.t_shown
            if since < config.INTERVENTION_SHOW_SECONDS:
                return {'phase': 'showing',
                        'detail': f"intervening: {self._active.title}"}
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
