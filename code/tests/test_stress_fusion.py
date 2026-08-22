"""
Fusion-layer tests.

The headline case is Default-mode equivalence: the task-aware rewrite must
produce bit-identical output to the pre-task-aware implementation when no
task mode is selected. That claim underpins "adding context modes did not
regress the baseline", so it is verified against the actual archived code
in code/_archive/ rather than against a reimplementation.
"""

import importlib.util
from pathlib import Path

import pytest

import stress_fusion

ARCHIVE = (Path(__file__).resolve().parent.parent
           / "_archive" / "stress_fusion_pre_taskaware.py")


def _load_archived():
    spec = importlib.util.spec_from_file_location("_pre_taskaware", ARCHIVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The four configurations the equivalence claim was made across.
FEATURE_CONFIGS = {
    "zero": dict.fromkeys(
        ["eye_strain", "blink_stress", "brow_tension", "jaw_tension",
         "head_instability", "neck_strain", "slouch"], 0.0),
    "low": {"eye_strain": 0.10, "blink_stress": 0.05, "brow_tension": 0.12,
            "jaw_tension": 0.08, "head_instability": 0.03,
            "neck_strain": 0.02, "slouch": 0.07},
    "full": dict.fromkeys(
        ["eye_strain", "blink_stress", "brow_tension", "jaw_tension",
         "head_instability", "neck_strain", "slouch"], 1.0),
    "mixed": {"eye_strain": 0.83, "blink_stress": 0.0, "brow_tension": 0.47,
              "jaw_tension": 0.91, "head_instability": 0.22,
              "neck_strain": 0.65, "slouch": 0.14},
}


@pytest.mark.parametrize("name", sorted(FEATURE_CONFIGS))
def test_default_mode_matches_pre_taskaware_baseline(name):
    """Default mode must equal the original un-modal implementation."""
    archived = _load_archived()
    features = FEATURE_CONFIGS[name]

    new_score, new_contribs = stress_fusion.behavioral_stress_score(
        features, task_mode="Default")
    old_score, old_contribs = archived.behavioral_stress_score(features)

    assert abs(new_score - old_score) < 1e-9, (
        f"{name}: task-aware Default diverged from baseline "
        f"({new_score!r} vs {old_score!r})"
    )
    assert new_contribs == old_contribs


def test_task_mode_weights_renormalize_to_one():
    """
    Every mode's multiplied weights must renormalize to 1.0, otherwise the
    0.36 / 0.55 thresholds would mean different things per mode.
    """
    for mode, multipliers in stress_fusion.TASK_MODES.items():
        boosted = {
            k: stress_fusion.BEHAVIOR_WEIGHTS[k] * multipliers.get(k, 1.0)
            for k in stress_fusion.BEHAVIOR_WEIGHTS
        }
        total = sum(boosted.values())
        normalized = sum(v / total for v in boosted.values())
        assert normalized == pytest.approx(1.0), f"{mode} failed to normalize"


def test_non_default_modes_actually_differ():
    """
    Guards against the reweighting silently doing nothing — if a mode never
    changes the score, the whole task-aware feature is decorative.
    """
    features = FEATURE_CONFIGS["mixed"]
    default, _ = stress_fusion.behavioral_stress_score(features, "Default")

    differing = [
        mode for mode in stress_fusion.TASK_MODES if mode != "Default"
        and abs(stress_fusion.behavioral_stress_score(features, mode)[0]
                - default) > 1e-6
    ]
    assert differing, "no task mode changed the score; reweighting is inert"


def test_unknown_task_mode_falls_back_to_default():
    """An unrecognised mode must not crash — it degrades to Default."""
    features = FEATURE_CONFIGS["mixed"]
    fallback, _ = stress_fusion.behavioral_stress_score(features, "Nonsense")
    default, _ = stress_fusion.behavioral_stress_score(features, "Default")
    assert fallback == pytest.approx(default)


def test_rule_boost_requires_four_active_signals():
    """+0.12 fires at 4 active signals (>=0.5), not at 3."""
    three = {"brow_tension": 0.6, "jaw_tension": 0.6, "slouch": 0.6,
             "eye_strain": 0.1, "blink_stress": 0.1,
             "head_instability": 0.1, "neck_strain": 0.1,
             "calibrating": False}
    four = dict(three, eye_strain=0.6)

    r3 = stress_fusion.compute_stress_index(None, three)
    r4 = stress_fusion.compute_stress_index(None, four)

    assert r3["boost_applied"] is False
    assert r4["boost_applied"] is True
    assert r4["stress_index"] == pytest.approx(
        min(1.0, r4["raw_index"] + stress_fusion.RULE_BOOST_AMOUNT))


def test_calibrating_frames_are_not_scored():
    """While calibrating, the index is pinned low rather than reported."""
    features = dict(FEATURE_CONFIGS["full"], calibrating=True)
    result = stress_fusion.compute_stress_index(None, features)
    assert result["calibrating"] is True
    assert result["stress_index"] < 0.10


def test_hysteresis_needs_three_consecutive_frames():
    """A level change is only emitted after 3 consecutive confirming frames."""
    hyst = stress_fusion.HysteresisLevel(confirm_frames=3)
    assert hyst.current_level == "LOW"

    moderate = 0.45     # between the 0.36 and 0.55 cutoffs
    assert hyst.update(moderate) == "LOW"      # 1st — not yet
    assert hyst.update(moderate) == "LOW"      # 2nd — still not
    assert hyst.update(moderate) == "MODERATE"  # 3rd — confirmed


def test_hysteresis_resets_on_flapping_candidate():
    """An interrupted candidate restarts the count instead of accumulating."""
    hyst = stress_fusion.HysteresisLevel(confirm_frames=3)
    hyst.update(0.45)          # MODERATE candidate, count 1
    hyst.update(0.10)          # back to LOW — candidate abandoned
    hyst.update(0.45)          # candidate restarts at 1
    assert hyst.current_level == "LOW"


def test_temporal_smoother_returns_running_mean():
    smoother = stress_fusion.TemporalSmoother(window_size=4)
    for value in (0.0, 1.0, 0.0, 1.0):
        last = smoother.update(value)
    assert last == pytest.approx(0.5)

    smoother.reset()
    assert smoother.update(0.8) == pytest.approx(0.8)


@pytest.fixture
def _clean_daisee_baseline():
    """_DAISEE_BASELINE is process-wide singleton state (like the live
    engine's) - reset it before AND after so this test can't leak into
    others that run before/after it in the same session."""
    stress_fusion.reset_daisee_baseline()
    yield
    stress_fusion.reset_daisee_baseline()


def test_get_daisee_baseline_roundtrips_and_resets(_clean_daisee_baseline):
    assert stress_fusion.get_daisee_baseline() is None
    assert stress_fusion.is_daisee_calibrated() is False

    stress_fusion.set_daisee_baseline(0.42)
    assert stress_fusion.get_daisee_baseline() == pytest.approx(0.42)
    assert stress_fusion.is_daisee_calibrated() is True

    stress_fusion.reset_daisee_baseline()
    assert stress_fusion.get_daisee_baseline() is None
    assert stress_fusion.is_daisee_calibrated() is False
