"""
Export filename tests.

The naming scheme is load-bearing: run_all_evaluations.py finds recordings by
globbing relaxed_*/stressed_*, so a name that does not match means a recording
silently contributes nothing. That is not hypothetical - 'relaaxed_2.csv' sat
in evaluation_data/ excluded from every PP1 number until it was found in audit.
"""

from datetime import datetime

import pytest

from app_naming import export_filename


WHEN = datetime(2026, 8, 12, 14, 30, 0)


def test_canonical_name_format():
    assert export_filename("S01", "relaxed", "Default", WHEN) == \
        "relaxed_S01_default_2026-08-12.csv"
    assert export_filename("S02", "stressed", "Studying", WHEN) == \
        "stressed_S02_studying_2026-08-12.csv"


@pytest.mark.parametrize("condition", ["relaxed", "stressed"])
def test_names_match_the_harness_globs(condition):
    """The generated name MUST match the glob the harness searches with."""
    import fnmatch
    name = export_filename("S07", condition, "Hospital", WHEN)
    assert fnmatch.fnmatch(name, f"{condition}_*.csv")


def test_case_and_whitespace_are_normalised():
    assert export_filename(" S01 ", "RELAXED", "DRIVING", WHEN) == \
        "relaxed_S01_driving_2026-08-12.csv"


@pytest.mark.parametrize("subject,condition", [
    ("", "relaxed"),            # no subject
    ("S01", "unlabeled"),       # condition not set
    ("S01", ""),                # condition blank
    (None, None),               # nothing set at all
])
def test_unlabelled_exports_fall_back_visibly(subject, condition):
    """
    An unlabelled export must be obviously unlabelled rather than quietly
    mislabelled - and must NOT match the harness globs, so it cannot sneak
    into an evaluation as though it were ground truth.
    """
    import fnmatch
    name = export_filename(subject, condition, "Default", WHEN)
    assert name.startswith("stress_session_")
    assert not fnmatch.fnmatch(name, "relaxed_*.csv")
    assert not fnmatch.fnmatch(name, "stressed_*.csv")


def test_arbitrary_condition_is_rejected():
    """Only the two ground-truth labels produce a harness-visible name."""
    name = export_filename("S01", "kind-of-stressed", "Default", WHEN)
    assert name.startswith("stress_session_")


def test_subject_ids_with_spaces_are_collapsed():
    """A pasted 'S 01' must not produce a name with a space in it."""
    name = export_filename("S 01", "relaxed", "Default", WHEN)
    assert " " not in name
    assert name == "relaxed_S01_default_2026-08-12.csv"
