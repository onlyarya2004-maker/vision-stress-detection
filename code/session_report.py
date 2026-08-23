"""
Session PDF report (Phase 5.2).

Renders a saved session into a downloadable PDF - metadata + summary stats
on one page, the stress-index timeline on a second. Uses matplotlib's
PdfPages backend directly: matplotlib was already installed for the
evaluation harness's plots, so this needed no new dependency (no reportlab/
fpdf). Pure function of the same per-frame `history` list app.py already
keeps in st.session_state.history and persists via session_history.py, so
it works identically for a just-finished live session and one reloaded
from the history DB - no camera, no live state, on-device only.
"""

import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from session_history import summarize_history

LEVEL_COLORS = {'LOW': '#22C55E', 'MODERATE': '#F59E0B', 'HIGH': '#EF4444'}
_PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches


def build_session_report_pdf(history, task_mode='Default', self_report=None,
                             session_label=None) -> bytes:
    """
    A 2-page PDF for one session. `history`: list of {'ts' (datetime),
    'stress_index' (float), 'level' (str), ...} records, oldest first -
    the same shape session_history.load_session_samples() and
    st.session_state.history already use.
    """
    if not history:
        raise ValueError("history must not be empty")

    stats = summarize_history(history)
    label = session_label or history[0]['ts'].strftime('%Y-%m-%d %H:%M')

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _add_summary_page(pdf, label, task_mode, self_report, stats)
        _add_timeline_page(pdf, label, history)
    return buf.getvalue()


def _add_summary_page(pdf, label, task_mode, self_report, stats):
    fig = plt.figure(figsize=_PAGE_SIZE)
    fig.suptitle("Stress Session Report", fontsize=18, fontweight='bold', y=0.95)

    lines = [
        f"Session:  {label}",
        f"Context:  {task_mode}",
        f"Duration: {stats['duration_sec']:.0f}s    Samples: {stats['sample_count']}",
        "",
        f"Mean stress index: {stats['mean_stress']:.2f}",
        f"Max stress index:  {stats['max_stress']:.2f}",
        "",
        f"Time Low:      {stats['pct_low']:.0f}%",
        f"Time Moderate: {stats['pct_moderate']:.0f}%",
        f"Time High:     {stats['pct_high']:.0f}%",
    ]
    if self_report is not None:
        lines += ["", f"Self-reported stress (1-5): {self_report}"]
    else:
        lines += ["", "Self-reported stress: not answered"]

    fig.text(0.1, 0.85, "\n".join(lines), fontsize=11, va='top', family='monospace')
    fig.text(0.1, 0.04,
             "Generated on-device from local session history - no video frames are "
             "ever stored, only these derived scalars.",
             fontsize=8, color='gray', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _add_timeline_page(pdf, label, history):
    fig, ax = plt.subplots(figsize=_PAGE_SIZE)
    t0 = history[0]['ts']
    xs = [(r['ts'] - t0).total_seconds() for r in history]
    ys = [r['stress_index'] for r in history]
    colors = [LEVEL_COLORS.get(r.get('level'), '#888888') for r in history]

    ax.plot(xs, ys, color='#3B82F6', linewidth=1, zorder=2)
    ax.scatter(xs, ys, c=colors, s=8, zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stress index")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Stress index timeline - {label}")
    ax.grid(True, alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)
