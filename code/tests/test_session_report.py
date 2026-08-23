"""
Tests for session_report.py (Phase 5.2 PDF export) and
session_history.summarize_history().

The PDF itself renders through matplotlib's PdfPages - not worth asserting
on exact pixel/text content, but the contract that matters IS checkable:
valid PDF bytes come out, the right numbers went into computing the
summary stats that get printed onto the page, and edge cases (empty
history, no self-report) don't crash.
"""

from datetime import datetime, timedelta

import pytest

from session_history import summarize_history
from session_report import build_session_report_pdf


def _history(n=10, base=0.3, levels=None, start=None):
    start = start or datetime(2026, 8, 23, 9, 0, 0)
    levels = levels or ['LOW'] * n
    return [
        {
            'ts': start + timedelta(seconds=i),
            'stress_index': base + i * 0.01,
            'level': levels[i % len(levels)],
            'daisee': 0.5,
            'behavior': base,
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------
# summarize_history()
# --------------------------------------------------------------------------

def test_summarize_history_computes_duration_and_sample_count():
    h = _history(n=5, start=datetime(2026, 8, 23, 9, 0, 0))
    stats = summarize_history(h)
    assert stats['sample_count'] == 5
    assert stats['duration_sec'] == pytest.approx(4.0)  # 5 samples, 1s apart


def test_summarize_history_mean_and_max_stress():
    h = [
        {'ts': datetime(2026, 8, 23, 9, 0, i), 'stress_index': v, 'level': 'LOW'}
        for i, v in enumerate([0.1, 0.5, 0.9])
    ]
    stats = summarize_history(h)
    assert stats['mean_stress'] == pytest.approx(0.5)
    assert stats['max_stress'] == pytest.approx(0.9)


def test_summarize_history_level_percentages():
    h = _history(n=4, levels=['LOW', 'LOW', 'MODERATE', 'HIGH'])
    stats = summarize_history(h)
    assert stats['pct_low'] == pytest.approx(50.0)
    assert stats['pct_moderate'] == pytest.approx(25.0)
    assert stats['pct_high'] == pytest.approx(25.0)


def test_summarize_history_rejects_empty_list():
    with pytest.raises(ValueError):
        summarize_history([])


# --------------------------------------------------------------------------
# build_session_report_pdf()
# --------------------------------------------------------------------------

def test_build_session_report_pdf_returns_valid_pdf_bytes():
    pdf = build_session_report_pdf(_history())
    assert pdf.startswith(b'%PDF')
    assert len(pdf) > 1000


def test_build_session_report_pdf_rejects_empty_history():
    with pytest.raises(ValueError):
        build_session_report_pdf([])


def test_build_session_report_pdf_works_without_self_report():
    pdf = build_session_report_pdf(_history(), self_report=None)
    assert pdf.startswith(b'%PDF')


def test_build_session_report_pdf_works_with_self_report():
    pdf = build_session_report_pdf(_history(), self_report=4)
    assert pdf.startswith(b'%PDF')


def test_build_session_report_pdf_accepts_custom_label_and_task_mode():
    pdf = build_session_report_pdf(
        _history(), task_mode='Driving', session_label='Custom label')
    assert pdf.startswith(b'%PDF')


def test_build_session_report_pdf_handles_single_sample():
    """Duration is 0 for a single frame - must not divide by zero anywhere
    in the rendering path."""
    h = _history(n=1)
    pdf = build_session_report_pdf(h)
    assert pdf.startswith(b'%PDF')
