"""
Automatic context SUGGESTION (local-only, opt-in)
=================================================
While the user calibrates, a handful of camera frames are classified by a
pretrained scene model (Places365 ResNet18, run on this computer through
OpenCV's DNN module - already a dependency, so nothing new is installed) and
the result is offered as a suggestion for the Context selector. The user
accepts or keeps the current mode; nothing is ever switched automatically and
nothing changes during a recording.

Scope and honesty:
  - A scene label says where the camera is, not what the person is doing. A
    desk with a laptop is not proof of studying, a hospital room is not proof
    of being a patient, a car interior is not proof of driving. So this only
    SUGGESTS, abstains whenever the frames disagree, and the user always
    confirms. Driving used to be manual-only, on the grounds that a car
    interior does not prove the person is driving; it is suggested since
    2026-11-09, because a driver stopped at a red light sees exactly the same
    stationary car interior, so the scene cannot tell "driving" from "sitting
    in a parked car" either way - the user's confirmation is what decides.
  - The confidence shown is the scene model's own output plus frame agreement.
    It is not a validated accuracy for choosing application modes.
  - The stress pipeline is untouched: an accepted suggestion just sets
    st.session_state.task_mode, the same value the manual selector sets.

Privacy: frames are classified in memory and dropped. An Observation keeps
only per-mode probability sums and the top scene labels - never pixels.

Where it runs: wherever the app runs, locally AND hosted. Everything is
local to the server process - no external API. The converted model lives in
<repo>/models/context/places365/ (see models/context/README.md for its source,
licence and the one-time conversion). Its files are gitignored (too large for
plain Git) but sit under models/, which the Dockerfile already copies, so a
build from a checkout that has them ships them; a build without them simply
reports "context detection is unavailable" and everything else still works.
STRESS_AUTO_CONTEXT=0 switches the feature off without a rebuild.
"""
import os
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

MODEL_DIR = Path(os.environ.get('STRESS_AUTO_CONTEXT_MODEL_DIR')
                 or Path(__file__).resolve().parent.parent / 'models' / 'context' / 'places365')
MODEL_FILE = 'resnet18_places365.onnx'
CATEGORIES_FILE = 'categories_places365.txt'
N_CATEGORIES = 365

# Explicit Places365 scene -> application mode map. Anything not listed counts
# towards none of the modes. Deliberately NOT mapped, because they say too
# little about the task: bedroom, dorm_room, living_room, conference_room,
# waiting_room, hospital (a building exterior), cockpit/bus/train interiors
# (passengers), veterinarians_office.
SCENE_TO_MODE = {
    'Studying': ('classroom', 'lecture_room', 'library/indoor', 'home_office',
                 'office', 'office_cubicles', 'computer_room'),
    'Hospital': ('hospital_room', 'nursing_home'),      # operating_room/vet office get NO credit:
                                                        # a white close-up looked like them (2026-10-28)
    'Driving': ('car_interior',),
}
# Modes that are recognised but never suggested. Empty since 2026-11-09:
# Driving is suggested like the others (a driver waiting at a red light looks
# exactly like someone sitting in a parked car, so refusing to suggest it
# helped nobody); the user still confirms every suggestion.
MANUAL_ONLY_MODES = ()

# Sampling during calibration (config.CALIBRATION_SECONDS = 20 s -> 10 frames).
SAMPLE_INTERVAL_S = 2.0
MAX_SAMPLES = 10
MIN_USABLE_SAMPLES = 5

# Conservative decision thresholds - starting values, not tuned optima.
# Per mode: a wrong Hospital suggestion costs far more than a missed one (on
# 2026-10-28 a close-up of a white notebook scored 0.58 as "hospital" in 9/10
# frames), so Hospital needs much stronger evidence than Studying.
FRAME_VOTE_MIN_MASS = {'Studying': 0.65, 'Hospital': 0.75, 'Driving': 0.65}
MIN_MEAN_MASS = {'Studying': 0.65, 'Hospital': 0.75, 'Driving': 0.65}
FRAME_DEFAULT_MAX_MASS = 0.15  # a frame votes Default only if no mapped mode
                               # reaches this (clearly something else)
MIN_AGREEMENT = 0.70           # share of usable frames that must agree

# With CLIP hints a frame's evidence for a mode is a blend of the scene model's
# probability and CLIP's hint probability. All three gates must hold; the
# scene floor stops a hint alone from naming a place the scene model finds no
# trace of (Hospital), but is 0 for Studying, which happens in any room.
HINT_WEIGHT = 0.5
COMBINED_MIN = {'Studying': 0.45, 'Hospital': 0.60, 'Driving': 0.55}
HINT_MIN = {'Studying': 0.40, 'Hospital': 0.50, 'Driving': 0.50}
SCENE_FLOOR = {'Studying': 0.0, 'Hospital': 0.15, 'Driving': 0.10}
HINT_GROUP = {'Studying': 'Studying', 'Hospital': 'Hospital', 'Driving': 'Vehicle'}

# Frame quality gates (grey levels 0-255).
MIN_BRIGHTNESS = 40.0
MIN_CONTRAST = 12.0
MAX_WHITE_SHARE = 0.40         # share of bright, unsaturated pixels (a page or
                               # wall filling the view, or glare): the scene
                               # model reads these as clinical rooms

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet stats, as
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)    # Places365 was trained


def feature_available():
    """The Auto-detect button is offered locally and hosted; set
    STRESS_AUTO_CONTEXT=0 to hide it (e.g. on a server without the models)."""
    return os.environ.get('STRESS_AUTO_CONTEXT', '1') != '0'


# ---------------------------------------------------------------------------
# Scene model
# ---------------------------------------------------------------------------
class ContextModelUnavailable(RuntimeError):
    """The scene model files are missing or could not be loaded."""


def load_categories(path):
    """categories_places365.txt lines look like '/l/library/indoor 212'."""
    labels = [None] * N_CATEGORIES
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        name, idx = line.rsplit(' ', 1)
        labels[int(idx)] = name[3:]                 # drop the '/l/' prefix
    if any(l is None for l in labels):
        raise ValueError(f"{path} does not list all {N_CATEGORIES} categories")
    return labels


def preprocess(frame_bgr):
    """BGR frame -> 1x3x224x224 float blob, as the official Places365 demo
    prepares images (resize to 224x224, ImageNet normalisation)."""
    img = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _MEAN) / _STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = np.exp(z - z.max())
    return z / z.sum()


class SceneClassifier:
    def __init__(self, model_dir=None):
        model_dir = Path(model_dir or MODEL_DIR)
        onnx_path, cats_path = model_dir / MODEL_FILE, model_dir / CATEGORIES_FILE
        if not onnx_path.is_file() or not cats_path.is_file():
            raise ContextModelUnavailable(
                f"scene model not found in {model_dir} (see models/context/README.md)")
        try:
            self.labels = load_categories(cats_path)
            self.net = cv2.dnn.readNetFromONNX(str(onnx_path))
        except Exception as e:
            raise ContextModelUnavailable(f"scene model failed to load: {e}") from e
        self._lock = threading.Lock()          # one cv2.dnn.Net, many sessions

    def predict(self, frame_bgr):
        """Scene probabilities (365,) for one BGR frame."""
        blob = preprocess(frame_bgr)
        with self._lock:
            self.net.setInput(blob)
            logits = self.net.forward()
        return softmax(logits.reshape(-1))


_CLASSIFIER = None
_CLASSIFIER_LOCK = threading.Lock()


def get_classifier():
    """The shared classifier, loaded on first use. A failed load is not
    cached, so downloading the model later works without a restart."""
    global _CLASSIFIER
    with _CLASSIFIER_LOCK:
        if _CLASSIFIER is None:
            _CLASSIFIER = SceneClassifier()
        return _CLASSIFIER


# ---------------------------------------------------------------------------
# Per-frame summary and voting
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """What is kept from one sampled frame - numbers and labels, no pixels."""
    masses: Dict[str, float] = field(default_factory=dict)  # mode -> summed prob
    top: List[Tuple[str, float]] = field(default_factory=list)
    invalid: Optional[str] = None          # why the frame was unusable
    hints: Optional[Dict[str, float]] = None      # CLIP group -> probability (None = no hints)
    hint_top: List[Tuple[str, float]] = field(default_factory=list)


def evidence(obs, mode):
    """Strength of the evidence for `mode` in one frame, and the threshold it
    has to reach: scene+hint blend when hints exist, scene alone otherwise."""
    if obs.hints is None:
        return obs.masses[mode], MIN_MEAN_MASS[mode]
    hint = obs.hints.get(HINT_GROUP[mode], 0.0)
    return (1 - HINT_WEIGHT) * obs.masses[mode] + HINT_WEIGHT * hint, COMBINED_MIN[mode]


def frame_quality(frame_bgr):
    """None when usable, else a short reason."""
    if frame_bgr is None or getattr(frame_bgr, 'size', 0) == 0:
        return 'no camera frame'
    grey = cv2.cvtColor(cv2.resize(frame_bgr, (160, 120)), cv2.COLOR_BGR2GRAY)
    if float(grey.mean()) < MIN_BRIGHTNESS:
        return 'too dark'
    if float(grey.std()) < MIN_CONTRAST:
        return 'camera covered or view blank'
    hsv = cv2.cvtColor(cv2.resize(frame_bgr, (160, 120)), cv2.COLOR_BGR2HSV)
    if float(((hsv[..., 2] > 170) & (hsv[..., 1] < 40)).mean()) > MAX_WHITE_SHARE:
        return 'view mostly white (page, wall or glare)'
    return None


def summarise(probs, labels, top_k=5):
    index = {name: i for i, name in enumerate(labels)}
    masses = {mode: float(sum(probs[index[s]] for s in scenes))
              for mode, scenes in SCENE_TO_MODE.items()}
    order = np.argsort(probs)[::-1][:top_k]
    return Observation(masses=masses, top=[(labels[i], float(probs[i])) for i in order])


def frame_vote(obs):
    """A mode name, 'Default', or 'unclear' for one usable observation."""
    if obs.hints is None:                          # scene model only
        passing = [(m - FRAME_VOTE_MIN_MASS[mode], mode) for mode, m in obs.masses.items()
                   if m >= FRAME_VOTE_MIN_MASS[mode]]
        top_mass = max(obs.masses.values())
    else:
        passing = []
        for mode in obs.masses:
            score, need = evidence(obs, mode)
            if (score >= need and obs.hints.get(HINT_GROUP[mode], 0.0) >= HINT_MIN[mode]
                    and obs.masses[mode] >= SCENE_FLOOR[mode]):
                passing.append((score - need, mode))
        top_mass = max(evidence(obs, m)[0] for m in obs.masses)
    if passing:
        return max(passing)[1]
    if top_mass < FRAME_DEFAULT_MAX_MASS:
        return 'Default'
    return 'unclear'


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
@dataclass
class ContextSuggestion:
    status: str                    # 'suggest' | 'unclear' | 'manual_only' | 'unavailable'
    current_mode: str
    message: str
    mode: Optional[str] = None     # set only when status == 'suggest'
    agreement: float = 0.0         # share of usable frames voting for the result
    confidence: float = 0.0        # mean scene-model probability of the result
    usable: int = 0
    sampled: int = 0
    top_scenes: List[Tuple[str, float]] = field(default_factory=list)
    top_hints: List[Tuple[str, float]] = field(default_factory=list)

    @property
    def is_change(self):
        return self.status == 'suggest' and self.mode != self.current_mode

    def log_line(self):
        """One server-log line: labels and numbers only, no personal data."""
        return (f"[context] {self.status} mode={self.mode} current={self.current_mode} "
                f"agreement={self.agreement:.2f} confidence={self.confidence:.2f} "
                f"usable={self.usable}/{self.sampled}")


def _top_scenes(usable, k=3):
    totals = Counter()
    for o in usable:
        for name, p in o.top:
            totals[name] += p
    return [(name, s / len(usable)) for name, s in totals.most_common(k)]


def _top_hints(usable, k=2):
    totals = Counter()
    for o in usable:
        for name, p in o.hint_top:
            totals[name] += p
    return [(name, s / len(usable)) for name, s in totals.most_common(k)]


def decide(observations, current_mode):
    """Combine the sampled frames into one conservative suggestion."""
    sampled = len(observations)
    usable = [o for o in observations if o.invalid is None]
    base = dict(current_mode=current_mode, usable=len(usable), sampled=sampled)
    keep = f"keeping {current_mode}"

    if len(usable) < MIN_USABLE_SAMPLES:
        reasons = Counter(o.invalid for o in observations if o.invalid)
        if sampled and all(r.startswith('model error') for r in reasons.elements()):
            return ContextSuggestion(
                status='unavailable', message=f"Context detection failed on every frame - {keep}.",
                **base)
        why = (f"{reasons.most_common(1)[0][0]} in most frames" if reasons
               else "too few camera frames were analysed")
        return ContextSuggestion(status='unclear', message=f"Scene unclear ({why}) - {keep}.",
                                 **base)

    votes = Counter(frame_vote(o) for o in usable)
    ranked = votes.most_common()
    winner, n = ranked[0]
    tied = len(ranked) > 1 and ranked[1][1] == n
    agreement = n / len(usable)
    top = _top_scenes(usable)
    if winner == 'Default':
        confidence = 1.0 - float(np.mean([max(evidence(o, m)[0] for m in o.masses)
                                          for o in usable]))
    elif winner in SCENE_TO_MODE:
        # mean over the frames that voted for it: disagreement is judged by
        # MIN_AGREEMENT, strength of the evidence by this
        confidence = float(np.mean([evidence(o, winner)[0] for o in usable
                                    if frame_vote(o) == winner]))
    else:
        confidence = 0.0
    base.update(agreement=agreement, confidence=confidence, top_scenes=top,
                top_hints=_top_hints(usable))

    if winner == 'unclear' or tied or agreement < MIN_AGREEMENT or (
            winner != 'Default' and confidence < evidence(usable[0], winner)[1]):
        return ContextSuggestion(
            status='unclear',
            message=f"The camera view does not clearly match a supported context - {keep}.",
            **base)
    if winner in MANUAL_ONLY_MODES:
        return ContextSuggestion(
            status='manual_only',
            message=(f"This looks like a {winner.lower()}-related scene, but {winner} is never "
                     f"chosen automatically - select it yourself if it applies. {keep[0].upper()}"
                     f"{keep[1:]}."),
            **base)
    return ContextSuggestion(status='suggest', mode=winner,
                             message=f"Suggested context: {winner}.", **base)


# ---------------------------------------------------------------------------
# Calibration-time sampler
# ---------------------------------------------------------------------------
class ContextProbe:
    """Samples at most MAX_SAMPLES frames, one per SAMPLE_INTERVAL_S of the
    caller's clock. offer() never raises - context detection must never be
    able to break a calibration."""

    def __init__(self, classifier_factory=None, interval_s=SAMPLE_INTERVAL_S,
                 max_samples=MAX_SAMPLES, hint_factory=None):
        self._factory = classifier_factory or get_classifier
        self._hint_factory = hint_factory       # None = scene model only
        self._hints = None
        self.hints_unavailable = None           # why hints are off (scene model still works)
        self.interval_s = interval_s
        self.max_samples = max_samples
        self._clf = None
        self._last_t = None
        self.unavailable = None            # reason the model could not be used
        self.observations = []

    def offer(self, frame_bgr, t):
        """Classify this frame if a sample is due. Returns True if it was used."""
        if self.unavailable is not None or len(self.observations) >= self.max_samples:
            return False
        if self._last_t is not None and t - self._last_t < self.interval_s:
            return False
        self._last_t = t
        if self._clf is None:
            try:
                self._clf = self._factory()
            except Exception as e:
                self.unavailable = str(e) or type(e).__name__
                return False
        if (self._hint_factory is not None and self._hints is None
                and self.hints_unavailable is None):
            try:
                self._hints = self._hint_factory()
            except Exception as e:              # hints are optional support
                self.hints_unavailable = str(e) or type(e).__name__
        try:
            reason = frame_quality(frame_bgr)
            if reason is not None:
                self.observations.append(Observation(invalid=reason))
            else:
                o = summarise(self._clf.predict(frame_bgr), self._clf.labels)
                if self._hints is not None:
                    try:
                        o.hints, o.hint_top = self._hints.predict(frame_bgr)
                    except Exception:
                        o.hints, o.hint_top = None, []
                self.observations.append(o)
        except Exception as e:
            self.observations.append(Observation(invalid=f"model error ({type(e).__name__})"))
        return True

    def result(self, current_mode):
        if self.unavailable is not None:
            return ContextSuggestion(
                status='unavailable', current_mode=current_mode,
                message=(f"Context detection is unavailable ({self.unavailable}) - keeping "
                         f"{current_mode}. Manual selection still works."))
        return decide(self.observations, current_mode)
