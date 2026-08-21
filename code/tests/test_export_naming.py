"""
Export filename tests.

The naming scheme is load-bearing: run_all_evaluations.py finds recordings by
globbing relaxed_*/stressed_*, so a name that does not match means a recording
silently contributes nothing. That is not hypothetical - 'relaaxed_2.csv' sat
in evaluation_data/ excluded from every PP1 number until it was found in audit.
"""

from datetime import datetime

import pytest

from app_naming import export_filename, parse_filename


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


def test_subject_ids_with_underscores_are_stripped():
    """
    A subject/mode containing '_' would otherwise produce a filename
    parse_filename can't invert (extra split points shift every field
    after condition) - stripped at the source so the app's own export
    path can never generate a name it can't later parse back correctly.
    """
    name = export_filename("S_01", "relaxed", "Default", WHEN)
    assert name == "relaxed_S01_default_2026-08-12.csv"


def test_parse_filename_round_trips_export_filename():
    for subject, condition, mode in [
        ("S01", "relaxed", "Default"), ("S07", "stressed", "Hospital"),
    ]:
        name = export_filename(subject, condition, mode, WHEN)
        meta = parse_filename(name)
        assert meta == {
            'condition': condition, 'subject': subject,
            'mode': mode.lower(), 'date': '2026-08-12',
        }


def test_parse_filename_rejects_names_it_cannot_invert():
    """
    A hand-renamed or foreign file with an extra underscore must be
    rejected outright (None), not silently mis-parsed into wrong fields -
    this used to accept 5-part names by truncating/shifting everything
    after condition, corrupting the per-subject breakdown table with no
    error raised anywhere.
    """
    assert parse_filename("relaxed_S_01_default_2026-08-12.csv") is None
    assert parse_filename("relaxed_S01_default_2026-08-12.csv") is not None


def test_parse_filename_tolerates_the_ubfc_traceability_suffix():
    """
    evaluate_video_file.py legitimately appends "_vid_s{N}_T{K}" to the
    canonical name for UBFC-Phys traceability - a real 7-part name that
    must still parse. A too-strict "exactly 4 parts" guard (an earlier,
    broken version of this fix) silently dropped the entire per-subject
    breakdown for UBFC-Phys data by rejecting every one of these.
    """
    meta = parse_filename("relaxed_U01_default_2026-08-15_vid_s1_T1.csv")
    assert meta == {'condition': 'relaxed', 'subject': 'U01',
                    'mode': 'default', 'date': '2026-08-15'}


def test_parse_filename_rejects_non_canonical_names():
    assert parse_filename("relaxed_1.csv") is None
    assert parse_filename("stress_session_2026-08-12_143000.csv") is None
    assert parse_filename("unlabelled_S01_default_2026-08-12.csv") is None
