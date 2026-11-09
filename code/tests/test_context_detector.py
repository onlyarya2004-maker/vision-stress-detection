"""
context_detector.py - the local-only context SUGGESTION.

Most tests use a fake classifier, so the mapping, the thresholds, the
frame-agreement rule and every failure path are checked independently of the
real model. The last block runs the real Places365 model through OpenCV when
it is present in local_models/places365/ (it is gitignored, so those tests
skip on a fresh clone).
"""
from pathlib import Path

import numpy as np
import pytest

import config
import context_detector as cd
from context_detector import ContextProbe, ContextSuggestion, Observation

MODEL_PRESENT = (cd.MODEL_DIR / cd.MODEL_FILE).is_file() and (cd.MODEL_DIR / cd.CATEGORIES_FILE).is_file()
needs_model = pytest.mark.skipif(not MODEL_PRESENT, reason="Places365 model not downloaded")

MAPPED = [s for scenes in cd.SCENE_TO_MODE.values() for s in scenes]
LABELS = [f'scene_{i}' for i in range(cd.N_CATEGORIES)]
for _i, _name in enumerate(MAPPED + ['bedroom', 'kitchen', 'living_room']):
    LABELS[_i] = _name


def probs_for(**mass):
    """365 probabilities: the named scenes get the given mass, the rest is
    spread evenly over filler scenes no mode maps to."""
    p = np.zeros(cd.N_CATEGORIES)
    for name, v in mass.items():
        p[LABELS.index(name.replace('__', '/'))] = v
    filler = [i for i, l in enumerate(LABELS) if l.startswith('scene_')]
    p[filler] += (1.0 - sum(mass.values())) / len(filler)
    return p


class FakeClassifier:
    labels = LABELS

    def __init__(self, *seq):
        self.seq, self.calls = list(seq), 0

    def predict(self, frame):
        p = self.seq[min(self.calls, len(self.seq) - 1)]
        self.calls += 1
        if isinstance(p, Exception):
            raise p
        return p


def textured_frame(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (480, 640, 3), dtype=np.uint8)


def obs(studying=0.0, hospital=0.0, driving=0.0):
    return Observation(masses={'Studying': studying, 'Hospital': hospital, 'Driving': driving})


def run_probe(clf, frames=None, n=10, dt=2.0):
    probe = ContextProbe(classifier_factory=lambda: clf)
    for i in range(n):
        probe.offer(textured_frame(i) if frames is None else frames[i], i * dt)
    return probe


# ---------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------
def test_mapping_targets_only_real_application_modes():
    assert set(cd.SCENE_TO_MODE) <= set(config.TASK_MODES)
    assert 'Default' not in cd.SCENE_TO_MODE        # Default = "none of the mapped scenes"
    assert len(MAPPED) == len(set(MAPPED)), "a scene may map to one mode only"


def test_driving_is_recognised_and_suggestable():
    assert cd.SCENE_TO_MODE['Driving'] == ('car_interior',)
    assert 'Driving' not in cd.MANUAL_ONLY_MODES


def test_task_ambiguous_scenes_are_deliberately_unmapped():
    for scene in ('bedroom', 'dorm_room', 'living_room', 'conference_room', 'waiting_room',
                  'hospital', 'cockpit', 'bus_interior', 'train_interior'):
        assert scene not in MAPPED


@needs_model
def test_every_mapped_scene_exists_in_the_places365_label_file():
    labels = cd.load_categories(cd.MODEL_DIR / cd.CATEGORIES_FILE)
    assert len(labels) == 365
    assert set(MAPPED) <= set(labels)


def test_summarise_sums_each_modes_scene_probabilities():
    o = cd.summarise(probs_for(home_office=0.4, office=0.2, hospital_room=0.1, car_interior=0.05),
                     LABELS)
    assert o.masses['Studying'] == pytest.approx(0.6)
    assert o.masses['Hospital'] == pytest.approx(0.1)
    assert o.masses['Driving'] == pytest.approx(0.05)
    assert o.top[0] == ('home_office', pytest.approx(0.4))
    assert o.invalid is None


# ---------------------------------------------------------------------------
# thresholds and abstention
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('masses, vote', [
    (dict(studying=0.65), 'Studying'),               # at the threshold: votes
    (dict(studying=0.64), 'unclear'),                # just below: abstains
    (dict(hospital=0.80), 'Hospital'),
    (dict(hospital=0.60), 'unclear'),                # Hospital needs 0.75, not 0.65
    (dict(driving=0.70), 'Driving'),
    (dict(studying=0.14, hospital=0.10), 'Default'), # nothing mapped is plausible
    (dict(studying=0.15), 'unclear'),                # weak but not negligible
])
def test_frame_vote_thresholds(masses, vote):
    assert cd.frame_vote(obs(**masses)) == vote


def test_unanimous_frames_give_a_suggestion():
    s = cd.decide([obs(studying=0.7)] * 10, 'Default')
    assert (s.status, s.mode, s.agreement, s.usable) == ('suggest', 'Studying', 1.0, 10)
    assert s.confidence == pytest.approx(0.7)
    assert s.is_change


def test_seven_of_ten_agreeing_frames_is_enough_six_is_not():
    seven = [obs(hospital=0.8)] * 7 + [obs(studying=0.3)] * 3
    six = [obs(hospital=0.8)] * 6 + [obs(studying=0.3)] * 4
    assert cd.decide(seven, 'Default').status == 'suggest'
    assert cd.decide(six, 'Default').status == 'unclear'


def test_a_tie_between_two_contexts_is_unclear():
    s = cd.decide([obs(studying=0.9)] * 5 + [obs(hospital=0.9)] * 5, 'Default')
    assert s.status == 'unclear' and s.mode is None


def test_low_mean_probability_blocks_a_suggestion_even_with_agreement(monkeypatch):
    monkeypatch.setattr(cd, 'FRAME_VOTE_MIN_MASS', {'Studying': 0.30, 'Hospital': 0.30, 'Driving': 0.30})
    s = cd.decide([obs(studying=0.35)] * 10, 'Default')          # weak frames may vote...
    assert s.status == 'unclear'                                 # ...the mean gate still holds


def test_hospital_needs_stronger_evidence_than_studying():
    assert cd.decide([obs(studying=0.70)] * 10, 'Default').mode == 'Studying'
    assert cd.decide([obs(hospital=0.70)] * 10, 'Default').status == 'unclear'
    assert cd.decide([obs(hospital=0.80)] * 10, 'Default').mode == 'Hospital'


def test_the_white_notebook_regression_is_not_a_hospital():
    # 2026-10-28 live frame: hospital 0.58 in 9/10 frames suggested Hospital.
    s = cd.decide([obs(hospital=0.58)] * 9 + [obs(hospital=0.20)], 'Default')
    assert s.status == 'unclear' and s.mode is None


def test_operating_room_and_vet_office_give_hospital_no_credit():
    o = cd.summarise(probs_for(hospital_room=0.1), LABELS)
    assert 'operating_room' not in cd.SCENE_TO_MODE['Hospital']
    assert o.masses['Hospital'] == pytest.approx(0.1)


def test_a_mostly_white_view_is_unusable():
    page = np.full((480, 640, 3), 235, np.uint8)
    page[:, :200] = np.random.default_rng(0).integers(0, 120, (480, 200, 3), dtype=np.uint8)
    assert 'mostly white' in cd.frame_quality(page)
    assert cd.frame_quality(textured_frame()) is None


def test_a_clearly_unsupported_scene_suggests_default():
    s = cd.decide([obs(studying=0.05)] * 10, 'Studying')
    assert (s.status, s.mode) == ('suggest', 'Default')
    assert s.is_change and s.confidence == pytest.approx(0.95)


def test_suggesting_the_current_mode_is_not_a_change():
    s = cd.decide([obs(studying=0.9)] * 10, 'Studying')
    assert s.status == 'suggest' and not s.is_change


def test_a_clear_car_interior_suggests_driving():
    # a driver stopped at a red light sees the same still car interior as a
    # parked one, so a clear car interior is suggested; the user confirms
    s = cd.decide([obs(driving=0.9)] * 10, 'Default')
    assert (s.status, s.mode) == ('suggest', 'Driving')
    assert s.is_change


def test_a_weak_car_interior_is_still_not_suggested():
    s = cd.decide([obs(driving=0.40)] * 10, 'Default')
    assert s.status != 'suggest' or s.mode != 'Driving'


def test_a_manual_only_mode_is_still_honoured_if_configured(monkeypatch):
    monkeypatch.setattr(cd, 'MANUAL_ONLY_MODES', ('Driving',))
    s = cd.decide([obs(driving=0.9)] * 10, 'Default')
    assert s.status == 'manual_only' and s.mode is None and not s.is_change


def test_too_few_usable_frames_is_unclear_and_names_the_reason():
    frames = [Observation(invalid='too dark')] * 6 + [obs(studying=0.9)] * 4
    s = cd.decide(frames, 'Hospital')
    assert s.status == 'unclear' and s.mode is None
    assert 'too dark' in s.message and 'keeping Hospital' in s.message


def test_no_frames_at_all_is_unclear():
    s = cd.decide([], 'Default')
    assert s.status == 'unclear' and 'too few camera frames' in s.message


# ---------------------------------------------------------------------------
# the calibration-time probe
# ---------------------------------------------------------------------------
def test_probe_samples_one_frame_per_interval_up_to_the_cap():
    clf = FakeClassifier(probs_for(home_office=0.8))
    probe = ContextProbe(classifier_factory=lambda: clf)
    used = [probe.offer(textured_frame(), i * 0.1) for i in range(300)]   # 30 s at 10 fps
    assert sum(used) == cd.MAX_SAMPLES == clf.calls
    assert [i for i, u in enumerate(used) if u][:3] == [0, 20, 40]          # every 2.0 s
    s = probe.result('Default')
    assert (s.status, s.mode, s.sampled) == ('suggest', 'Studying', 10)


def test_probe_loads_the_model_lazily_on_its_first_sample():
    made = []
    probe = ContextProbe(classifier_factory=lambda: made.append(1) or FakeClassifier(probs_for()))
    assert made == []                                   # creating a probe loads nothing
    probe.offer(textured_frame(), 0.0)
    probe.offer(textured_frame(), 5.0)
    assert made == [1]                                  # loaded once, then reused


def test_missing_model_makes_the_probe_unavailable_without_raising():
    def missing():
        raise cd.ContextModelUnavailable("scene model not found in X")
    probe = ContextProbe(classifier_factory=missing)
    for i in range(10):
        assert probe.offer(textured_frame(), i * 2.0) is False
    s = probe.result('Studying')
    assert s.status == 'unavailable' and s.mode is None
    assert 'not found' in s.message and 'keeping Studying' in s.message


def test_inference_errors_are_contained_and_reported():
    probe = run_probe(FakeClassifier(RuntimeError("boom")))
    assert all(o.invalid == 'model error (RuntimeError)' for o in probe.observations)
    assert probe.result('Default').status == 'unavailable'


def test_occasional_inference_errors_still_allow_a_decision():
    clf = FakeClassifier(*([probs_for(hospital_room=0.9)] * 8 + [RuntimeError()] * 2))
    s = run_probe(clf).result('Default')
    assert (s.status, s.mode, s.usable, s.sampled) == ('suggest', 'Hospital', 8, 10)


def test_dark_or_covered_frames_are_skipped_before_inference():
    clf = FakeClassifier(probs_for(home_office=0.9))
    dark = [np.full((480, 640, 3), 10, np.uint8)] * 10
    s = run_probe(clf, frames=dark).result('Default')
    assert clf.calls == 0
    assert s.status == 'unclear' and 'too dark' in s.message
    flat = [np.full((480, 640, 3), 128, np.uint8)] * 10
    assert 'camera covered' in run_probe(clf, frames=flat).result('Default').message


def test_observations_keep_no_pixels():
    probe = run_probe(FakeClassifier(probs_for(home_office=0.8)))
    for o in probe.observations:
        assert set(vars(o)) == {'masses', 'top', 'invalid', 'hints', 'hint_top'}
        assert all(isinstance(v, float) for v in o.masses.values())
        assert all(isinstance(n, str) and isinstance(p, float) for n, p in o.top)


def test_scene_classifier_reports_missing_files(tmp_path):
    with pytest.raises(cd.ContextModelUnavailable, match='not found'):
        cd.SceneClassifier(tmp_path)


def test_a_failed_model_load_is_not_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(cd, '_CLASSIFIER', None)
    monkeypatch.setattr(cd, 'MODEL_DIR', tmp_path)
    with pytest.raises(cd.ContextModelUnavailable):
        cd.get_classifier()
    assert cd._CLASSIFIER is None                     # a later download is picked up


def test_feature_is_available_locally_and_hosted_and_can_be_switched_off(monkeypatch):
    monkeypatch.delenv('STRESS_AUTO_CONTEXT', raising=False)
    for backend in ('sqlite', 'supabase'):
        monkeypatch.setattr(config, 'STORAGE_BACKEND', backend)
        assert cd.feature_available()
    monkeypatch.setenv('STRESS_AUTO_CONTEXT', '0')
    assert not cd.feature_available()


def test_default_model_locations_are_under_models_so_the_image_ships_them():
    root = Path(__file__).resolve().parent.parent.parent
    import context_hints as ch
    assert cd.MODEL_DIR == root / 'models' / 'context' / 'places365'
    assert ch.HINT_DIR == root / 'models' / 'context' / 'clip'


def test_log_line_carries_labels_and_numbers_only():
    s = ContextSuggestion(status='suggest', current_mode='Default', message='m', mode='Studying',
                          agreement=0.8, confidence=0.71, usable=9, sampled=10)
    assert s.log_line() == ("[context] suggest mode=Studying current=Default "
                            "agreement=0.80 confidence=0.71 usable=9/10")


# ---------------------------------------------------------------------------
# the real model (skipped when it has not been downloaded)
# ---------------------------------------------------------------------------
@needs_model
def test_opencv_runtime_matches_the_pytorch_reference():
    import cv2
    ref = np.load(cd.MODEL_DIR / 'parity_reference.npz')
    net = cv2.dnn.readNetFromONNX(str(cd.MODEL_DIR / cd.MODEL_FILE))
    net.setInput(ref['inputs'])
    out = net.forward()
    assert np.abs(out - ref['logits']).max() < 1e-3
    assert (out.argmax(1) == ref['logits'].argmax(1)).all()


@needs_model
def test_real_model_returns_a_probability_over_365_scenes():
    clf = cd.SceneClassifier()
    p = clf.predict(textured_frame())
    assert p.shape == (365,) and p.sum() == pytest.approx(1.0) and (p >= 0).all()
    assert cd.get_classifier() is cd.get_classifier()          # one shared instance


@needs_model
def test_no_webcam_frame_is_ever_written_to_disk(monkeypatch, tmp_path):
    import cv2
    from PIL import Image

    def forbidden(*a, **k):
        raise AssertionError("context detection tried to write an image")
    monkeypatch.setattr(cv2, 'imwrite', forbidden)
    monkeypatch.setattr(Image.Image, 'save', forbidden)
    monkeypatch.setattr(np, 'save', forbidden)
    monkeypatch.chdir(tmp_path)
    probe = ContextProbe()                                       # the real shared model
    for i in range(10):
        probe.offer(textured_frame(i), i * 2.0)
    assert probe.result('Default').sampled == 10
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# CLIP hints (people / objects) on top of the scene model
# ---------------------------------------------------------------------------
def hobs(scene_s=0.0, scene_h=0.0, scene_d=0.0, studying=0.0, hospital=0.0, vehicle=0.0):
    o = obs(studying=scene_s, hospital=scene_h, driving=scene_d)
    o.hints = {'Studying': studying, 'Hospital': hospital, 'Vehicle': vehicle, 'Other': 0.0}
    return o


def test_a_student_hint_suggests_studying_even_in_a_neutral_room():
    # the 2026-10-28 live frames: notebook on a sofa, scene model sees nothing
    s = cd.decide([hobs(studying=0.9)] * 10, 'Default')
    assert (s.status, s.mode) == ('suggest', 'Studying')


def test_hospital_needs_both_a_scene_trace_and_a_strong_hint():
    assert cd.decide([hobs(hospital=0.9)] * 10, 'Default').status == 'unclear'          # hint alone
    assert cd.decide([hobs(scene_h=0.30, hospital=0.3)] * 10, 'Default').status == 'unclear'  # scene alone
    s = cd.decide([hobs(scene_h=0.40, hospital=0.9)] * 10, 'Default')
    assert (s.status, s.mode) == ('suggest', 'Hospital')


def test_a_vehicle_hint_with_a_car_scene_suggests_driving():
    s = cd.decide([hobs(scene_d=0.6, vehicle=0.9)] * 10, 'Default')
    assert (s.status, s.mode) == ('suggest', 'Driving')


def test_a_vehicle_hint_alone_does_not_suggest_driving():
    # the scene floor: CLIP saying "car" is not enough without any car-interior trace
    s = cd.decide([hobs(scene_d=0.0, vehicle=0.9)] * 10, 'Default')
    assert s.mode != 'Driving'


def test_without_hints_the_scene_rules_apply_unchanged():
    assert cd.frame_vote(obs(studying=0.70)) == 'Studying'
    assert cd.frame_vote(obs(hospital=0.60)) == 'unclear'


def test_probe_attaches_hints_and_survives_a_missing_hint_model():
    class Hints:
        def predict(self, f):
            return {'Studying': 0.9, 'Hospital': 0.0, 'Vehicle': 0.0, 'Other': 0.1}, [('a student writing', 0.8)]
    clf = FakeClassifier(probs_for())
    probe = ContextProbe(classifier_factory=lambda: clf, interval_s=0.0, hint_factory=Hints)
    for i in range(10):
        probe.offer(textured_frame(i), float(i))
    s = probe.result('Default')
    assert (s.status, s.mode) == ('suggest', 'Studying') and s.top_hints[0][0] == 'a student writing'

    def missing():
        raise ValueError("no clip")
    probe = ContextProbe(classifier_factory=lambda: FakeClassifier(probs_for(home_office=0.8)),
                         interval_s=0.0, hint_factory=missing)
    for i in range(10):
        probe.offer(textured_frame(i), float(i))
    assert probe.hints_unavailable == 'no clip'
    assert probe.result('Default').mode == 'Studying'          # scene-only fallback still works


HINTS_PRESENT = (Path(__file__).resolve().parent.parent.parent / 'models' / 'context' / 'clip' / 'clip_image_encoder.onnx').is_file()


@pytest.mark.skipif(not HINTS_PRESENT, reason="CLIP hint model not converted")
def test_clip_runtime_matches_pytorch_and_hint_probabilities_are_valid():
    import context_hints as ch
    ref = np.load(ch.HINT_DIR / 'parity_reference.npz')
    h = ch.ClipHints()
    h.net.setInput(ref['inputs'][:1])
    assert np.abs(h.net.forward() - ref['embeds'][:1]).max() < 1e-3
    groups, top = h.predict(textured_frame())
    assert sum(groups.values()) == pytest.approx(1.0) and set(groups) == {'Studying', 'Hospital', 'Vehicle', 'Other'}
    assert len(top) == 2


def test_hints_missing_files_raise_a_clear_error(tmp_path):
    import context_hints as ch
    with pytest.raises(ch.HintsUnavailable, match='not found'):
        ch.ClipHints(tmp_path)
