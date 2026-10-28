"""
Drowsiness warning (Driving mode only)
======================================
PERCLOS measures eyelid closure, i.e. drowsiness - not stress - so it is kept
OUT of the stress index (stress_fusion.py never sees it). This is a separate,
display-only warning: when PERCLOS stays at or above config.DROWSY_PERCLOS_LEVEL
for config.DROWSY_HOLD_SECONDS in a row, a banner is raised; it clears once
PERCLOS has stayed below the level for config.DROWSY_CLEAR_SECONDS.

Honest scope: PERCLOS here is approximated from eye-aspect-ratio on a laptop
webcam, the 0.15 level and the 30 s hold are starting values taken from the
drowsiness literature, not tuned or validated on Driving recordings (there are
none), and this is NOT a certified safety device. Nothing is stored and the
stress score, its weights and thresholds are unchanged.
"""
import config


class DrowsinessMonitor:
    """Feed it every frame's (time, PERCLOS); it reports whether to warn.

    The first DROWSY_MIN_SESSION_SECONDS of a run are never judged: PERCLOS is a
    rolling 60 s share, and over the first few frames it is mostly noise (one
    long blink would read as 100%).
    """

    def __init__(self, level=None, hold_s=None, clear_s=None, min_session_s=None):
        self.level = config.DROWSY_PERCLOS_LEVEL if level is None else level
        self.hold_s = config.DROWSY_HOLD_SECONDS if hold_s is None else hold_s
        self.clear_s = config.DROWSY_CLEAR_SECONDS if clear_s is None else clear_s
        self.min_session_s = (config.DROWSY_MIN_SESSION_SECONDS
                              if min_session_s is None else min_session_s)
        self._t_start = None
        self._above_since = None
        self._below_since = None
        self.warning = False

    def update(self, t, perclos):
        """Returns {'warning': bool, 'perclos': float, 'above_s': float}."""
        t, perclos = float(t), float(perclos)
        if self._t_start is None:
            self._t_start = t
        above_s = 0.0
        if t - self._t_start >= self.min_session_s:
            if perclos >= self.level:
                self._below_since = None
                if self._above_since is None:
                    self._above_since = t
                above_s = t - self._above_since
                if above_s >= self.hold_s:
                    self.warning = True
            else:
                self._above_since = None
                if self._below_since is None:
                    self._below_since = t
                if self.warning and t - self._below_since >= self.clear_s:
                    self.warning = False
        return {'warning': self.warning, 'perclos': perclos, 'above_s': above_s}
