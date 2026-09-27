"""
Stress index vs physiology (Phase 1.4 / 1.5)
=============================================
Correlates the system's continuous stress index against two UBFC-Phys ground
truth physiological channels: EDA (electrodermal activity, the classic
sympathetic-arousal signal) and heart rate derived from BVP
(photoplethysmogram).

This asks a harder question than the accuracy tables: not "does the index
clear a threshold on stressed clips", but "does it MOVE WITH measured
physiology". Two levels, because they can disagree:

  WITHIN-task  - per (subject, task): Pearson/Spearman between the smoothed
                 index and the physiological signal over the 3 minutes.
                 Honest caveat printed with the results: both EDA and heart
                 rate drift slowly and are autocorrelated, so within-task r
                 is indicative, not inferential.

  BETWEEN-task - per subject: do the task MEANS move together? If EDA/heart
                 rate rises from rest to stressor, does the index rise too?
                 This is the claim the system actually makes, and with 3
                 tasks per subject it reduces to direction agreement.

Inputs are all already on disk (video long deleted):
  datasets/feature_traces/features_vid_s{N}_T{K}.csv        stress index over time
  datasets/UBFC-Phys/s{N}/s{N}/eda_s{N}_T{K}.csv (or s{N}/) EDA @ 4 Hz
  datasets/UBFC-Phys/s{N}/s{N}/bvp_s{N}_T{K}.csv (or s{N}/) BVP @ 64 Hz

Both physiological file lookups tolerate two directory layouts - most
subjects' files sit under a doubled s{N}/s{N}/ folder, but s11 (added later,
unzipped differently) sits directly under s11/. The original EDA lookup only
tried the doubled path, so s11 silently reported "no EDA file" even though
the file exists - _find_physio_file() below tries both, which fixes that gap
as a side effect of adding the heart-rate study.

Heart rate is derived from raw BVP via a bandpass filter (0.7-3.5 Hz, i.e.
42-210 bpm) followed by peak detection - there is no hardware HR channel in
UBFC-Phys, only the raw PPG waveform. This is noisier than a real HR monitor
and can misfire on weak/noisy segments; physiologically impossible intervals
(outside 30-220 bpm) are dropped rather than passed through as correlation
noise, and that is the full extent of the cleanup - no smoothing beyond what
the bandpass filter already does, so a bad correlation is not being hidden
by extra massaging.

Phase 4.4/4.5: run_correlation_study()/render() and
run_hr_correlation_study()/render_hr() below are called directly from
run_all_evaluations.py, so both are folded into the one regenerated
canonical file rather than living only in their own standalone report.
Still runnable on their own too (same functions, same output) - see Usage.

Usage
-----
    cd code && python correlate_physiology.py
    python correlate_physiology.py --traces ../datasets/feature_traces \
        --ubfc ../datasets/UBFC-Phys
    python correlate_physiology.py --out ../evaluation_data/CORRELATION.txt
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal, stats

EDA_FS = 4.0                      # Empatica E4 EDA sample rate
BVP_FS = 64.0                     # Empatica E4 BVP sample rate
NAME_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)


def load_index(path):
    """(t, smoothed stress index) from a feature trace, sorted by t.

    np.interp() in align() silently misbehaves on unsorted input - the same
    schema's loader in ablation_study.py (load_trace()) already sorts for
    exactly this reason. Currently safe here too (the only writer produces
    monotonic rows), but sorting explicitly rather than relying on that
    holding forever costs nothing.
    """
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            rows.append((float(r['t']), float(r['stress_index_smoothed'])))
        except (KeyError, ValueError):
            continue
    rows.sort(key=lambda row: row[0])
    if not rows:
        return np.array([]), np.array([])
    t, x = zip(*rows)
    return np.array(t), np.array(x)


def load_eda(path):
    """
    One EDA reading per line. Skips any line that doesn't parse (a header
    row, a dropout marker, a truncated download) rather than crashing -
    matching load_index()'s pattern. A single bad EDA file used to abort
    the whole run and lose every already-computed subject's results with
    it, since output was only ever written once, at the very end.
    """
    vals = []
    for l in open(path):
        l = l.strip()
        if not l:
            continue
        try:
            vals.append(float(l))
        except ValueError:
            continue
    return np.array(vals)


# BVP files use the exact same "one raw reading per line" format as EDA -
# load_eda()'s parsing (including skipping dropout markers/blank lines) is
# format-generic despite its name, so it is reused rather than duplicated.
# Aliased under its own name here purely so call sites read honestly.
load_bvp = load_eda


def _find_physio_file(ubfc_dir, subj, task, prefix):
    """
    Locate a UBFC-Phys physiological file, trying the doubled s{N}/s{N}/
    layout (most subjects) then the flat s{N}/ layout (s11) - see module
    docstring. Returns None if neither exists.
    """
    ubfc_dir = Path(ubfc_dir)
    name = f"{prefix}_s{subj}_T{task}.csv"
    nested = ubfc_dir / f"s{subj}" / f"s{subj}" / name
    if nested.exists():
        return nested
    flat = ubfc_dir / f"s{subj}" / name
    if flat.exists():
        return flat
    return None


def bvp_to_heart_rate(bvp, fs=BVP_FS):
    """
    Convert a raw BVP (photoplethysmogram) trace into an instantaneous
    heart-rate series via bandpass filtering + peak detection.

    Returns (t_hr, hr_bpm), timestamped at the later peak of each
    consecutive pair - irregularly spaced, unlike EDA's fixed 4Hz clock,
    which is why alignment against it needs align_hr() rather than align().

    Too short to filter reliably, or too few peaks to form any interval,
    returns empty arrays rather than raising - matching load_index()'s and
    load_eda()'s "skip, don't crash" convention so one bad recording can't
    abort a whole run.
    """
    bvp = np.asarray(bvp, dtype=float)
    if len(bvp) < fs * 5:
        return np.array([]), np.array([])
    nyq = fs / 2.0
    b, a = signal.butter(3, [0.7 / nyq, 3.5 / nyq], btype='band')
    filtered = signal.filtfilt(b, a, bvp)
    min_distance = max(1, int(fs * 60.0 / 220.0))   # refractory at 220bpm cap
    peaks, _ = signal.find_peaks(filtered, distance=min_distance)
    if len(peaks) < 3:
        return np.array([]), np.array([])
    peak_times = peaks / fs
    ibi = np.diff(peak_times)                        # inter-beat intervals, s
    with np.errstate(divide='ignore'):
        hr = 60.0 / ibi
    t_hr = peak_times[1:]
    valid = (hr >= 30) & (hr <= 220)                  # drop impossible beats
    return t_hr[valid], hr[valid]


def align_hr(t_idx, x_idx, t_hr, hr):
    """
    Resample the stress index onto the heart-rate trace's own (irregular,
    peak-driven) timestamps, trimmed to the overlapping window - the same
    overlap-then-interpolate shape as align(), adapted for a non-uniform
    clock instead of EDA's fixed 4Hz grid.
    """
    if len(hr) < 10:
        return None, None
    lo, hi = max(t_idx[0], t_hr[0]), min(t_idx[-1], t_hr[-1])
    keep = (t_hr >= lo) & (t_hr <= hi)
    if keep.sum() < 10:
        return None, None
    x_on_hr = np.interp(t_hr[keep], t_idx, x_idx)
    return x_on_hr, hr[keep]


def align(t_idx, x_idx, eda):
    """
    Resample the stress index onto the EDA clock (4 Hz). The index trace
    starts after the calibration window, so EDA is trimmed to the overlap.
    """
    if len(eda) == 0:
        return None, None
    t_eda = np.arange(len(eda)) / EDA_FS
    lo, hi = max(t_idx[0], t_eda[0]), min(t_idx[-1], t_eda[-1])
    keep = (t_eda >= lo) & (t_eda <= hi)
    if keep.sum() < 20:
        return None, None
    x_on_eda = np.interp(t_eda[keep], t_idx, x_idx)
    return x_on_eda, eda[keep]


# ---------------------------------------------------------------------------
# Computation - separated from rendering so run_all_evaluations.py can call
# this directly and fold the result into the one canonical file, instead of
# this living only in its own separately-generated report.
# ---------------------------------------------------------------------------
def run_correlation_study(traces_dir, ubfc_dir):
    """
    Returns a results dict, or None if no feature traces are found (e.g. the
    dataset-derived traces aren't present in this checkout - same contract
    as ablation_study.run_ablation_study()).

    per_trace entries only exist for traces that produced SOME output line
    (no-EDA-file, NaN-dropout, or a real correlation) - a trace skipped for
    being too short or having no clock overlap with its EDA file produces no
    entry, matching the original script's silent `continue`.
    """
    traces = sorted(Path(traces_dir).glob('features_*.csv'))
    if not traces:
        return None

    per_trace = []
    per_subject = {}
    for tr in traces:
        m = NAME_RE.search(tr.stem)
        if not m:
            continue
        subj, task = int(m.group(1)), int(m.group(2))
        label = f"s{subj}_T{task}"
        eda_path = _find_physio_file(ubfc_dir, subj, task, 'eda')
        if eda_path is None:
            per_trace.append({'label': label, 'status': 'no_eda'})
            continue

        t_idx, x_idx = load_index(tr)
        if len(t_idx) < 20:
            continue
        x, e = align(t_idx, x_idx, load_eda(eda_path))
        if x is None:
            continue

        # NaN std (an unparsed dropout marker survived as a literal "nan")
        # must not look identical to genuinely-zero variance, so it's
        # called out explicitly instead of falling through to the same
        # blank-looking "N/A" the zero-variance case would print.
        sx, se = np.std(x), np.std(e)
        if np.isnan(sx) or np.isnan(se):
            per_trace.append({'label': label, 'status': 'nan', 'n': len(x)})
            continue
        pr = stats.pearsonr(x, e)[0] if sx > 0 and se > 0 else np.nan
        sr = stats.spearmanr(x, e)[0] if sx > 0 and se > 0 else np.nan
        per_trace.append({'label': label, 'status': 'ok', 'n': len(x),
                          'pearson': float(pr), 'spearman': float(sr)})

        d = per_subject.setdefault(subj, {})
        d[task] = {'idx_mean': float(np.mean(x)), 'eda_mean': float(np.mean(e))}

    # ---- between-task: does the index move the way EDA moves? ----
    agreement_rows = []
    agree = total = 0
    for subj in sorted(per_subject):
        d = per_subject[subj]
        if 1 not in d or not ({2, 3} & set(d)):
            continue
        eda_rest = d[1]['eda_mean']
        idx_rest = d[1]['idx_mean']
        eda_stress = float(np.mean([d[t]['eda_mean'] for t in (2, 3) if t in d]))
        idx_stress = float(np.mean([d[t]['idx_mean'] for t in (2, 3) if t in d]))
        same = (eda_stress - eda_rest) * (idx_stress - idx_rest) > 0
        agree += int(same)
        total += 1
        agreement_rows.append({
            'subject': subj, 'eda_rest': eda_rest, 'eda_stress': eda_stress,
            'idx_rest': idx_rest, 'idx_stress': idx_stress, 'agree': same,
        })

    return {
        'traces_dir': Path(traces_dir).resolve(),
        'ubfc_dir': Path(ubfc_dir).resolve(),
        'per_trace': per_trace,
        'agreement_rows': agreement_rows,
        'agree': agree,
        'total': total,
    }


def run_hr_correlation_study(traces_dir, ubfc_dir):
    """
    Same study as run_correlation_study(), against heart rate (derived from
    BVP via bvp_to_heart_rate()) instead of EDA. Structurally identical
    (within-task r, between-task direction agreement) so the two studies'
    output reads the same way side by side; kept as a separate function
    rather than parameterising run_correlation_study() over signal type,
    since the two signals need different loaders (fixed-rate load_eda() +
    align() vs derived, irregularly-timestamped bvp_to_heart_rate() +
    align_hr()) and forcing them through one code path would obscure that
    difference rather than simplify anything.
    """
    traces = sorted(Path(traces_dir).glob('features_*.csv'))
    if not traces:
        return None

    per_trace = []
    per_subject = {}
    for tr in traces:
        m = NAME_RE.search(tr.stem)
        if not m:
            continue
        subj, task = int(m.group(1)), int(m.group(2))
        label = f"s{subj}_T{task}"
        bvp_path = _find_physio_file(ubfc_dir, subj, task, 'bvp')
        if bvp_path is None:
            per_trace.append({'label': label, 'status': 'no_bvp'})
            continue

        t_idx, x_idx = load_index(tr)
        if len(t_idx) < 20:
            continue
        t_hr, hr = bvp_to_heart_rate(load_bvp(bvp_path))
        if len(hr) == 0:
            per_trace.append({'label': label, 'status': 'no_peaks'})
            continue
        x, h = align_hr(t_idx, x_idx, t_hr, hr)
        if x is None:
            continue

        sx, sh = np.std(x), np.std(h)
        if np.isnan(sx) or np.isnan(sh):
            per_trace.append({'label': label, 'status': 'nan', 'n': len(x)})
            continue
        pr = stats.pearsonr(x, h)[0] if sx > 0 and sh > 0 else np.nan
        sr = stats.spearmanr(x, h)[0] if sx > 0 and sh > 0 else np.nan
        per_trace.append({'label': label, 'status': 'ok', 'n': len(x),
                          'pearson': float(pr), 'spearman': float(sr)})

        d = per_subject.setdefault(subj, {})
        d[task] = {'idx_mean': float(np.mean(x)), 'hr_mean': float(np.mean(h))}

    # ---- between-task: does the index move the way heart rate moves? ----
    agreement_rows = []
    agree = total = 0
    for subj in sorted(per_subject):
        d = per_subject[subj]
        if 1 not in d or not ({2, 3} & set(d)):
            continue
        hr_rest = d[1]['hr_mean']
        idx_rest = d[1]['idx_mean']
        hr_stress = float(np.mean([d[t]['hr_mean'] for t in (2, 3) if t in d]))
        idx_stress = float(np.mean([d[t]['idx_mean'] for t in (2, 3) if t in d]))
        same = (hr_stress - hr_rest) * (idx_stress - idx_rest) > 0
        agree += int(same)
        total += 1
        agreement_rows.append({
            'subject': subj, 'hr_rest': hr_rest, 'hr_stress': hr_stress,
            'idx_rest': idx_rest, 'idx_stress': idx_stress, 'agree': same,
        })

    return {
        'traces_dir': Path(traces_dir).resolve(),
        'ubfc_dir': Path(ubfc_dir).resolve(),
        'per_trace': per_trace,
        'agreement_rows': agreement_rows,
        'agree': agree,
        'total': total,
    }


def render(results, out, generated_line=None):
    """
    generated_line: a "generated : <timestamp>" line to include, for
    standalone use (main() below). Omitted when called from
    run_all_evaluations.py, which already stamps one timestamp for the
    whole canonical file - no other section repeats its own.
    """
    out.append("\n" + "=" * 74)
    out.append("STRESS INDEX vs EDA  (UBFC-Phys ground truth)")
    out.append("=" * 74)
    if results is None:
        out.append("  (no feature traces found under datasets/feature_traces/ - skipped)")
        return

    if generated_line:
        out.append(f"  {generated_line}")
    out.append(f"  traces : {results['traces_dir']}")
    out.append(f"  ubfc   : {results['ubfc_dir']}")
    out.append("")
    out.append("Within-task r is indicative only - EDA drifts and both series are")
    out.append("autocorrelated. The between-task direction test is the real claim.")
    out.append("")

    out.append(f"  {'trace':22s}{'n':>6}{'Pearson':>10}{'Spearman':>10}")
    out.append("  " + "-" * 50)
    for row in results['per_trace']:
        label = row['label']
        if row['status'] == 'no_eda':
            out.append(f"  {label:<22} (no EDA file)")
        elif row['status'] == 'nan':
            out.append(f"  {label:<22}{row['n']:>6}  (NaN in EDA - dropout marker?)")
        else:
            out.append(f"  {label:<22}{row['n']:>6}{row['pearson']:>+10.2f}"
                       f"{row['spearman']:>+10.2f}")

    out.append("")
    out.append("  BETWEEN-TASK (rest T1 -> stressors T2/T3): direction agreement")
    out.append(f"  {'subject':10s}{'EDA rest':>10}{'EDA stress':>12}"
               f"{'idx rest':>10}{'idx stress':>12}   agree?")
    out.append("  " + "-" * 62)
    for row in results['agreement_rows']:
        out.append(f"  s{row['subject']:<9}{row['eda_rest']:>10.3f}{row['eda_stress']:>12.3f}"
                   f"{row['idx_rest']:>10.3f}{row['idx_stress']:>12.3f}   "
                   f"{'YES' if row['agree'] else 'NO'}")

    if results['total']:
        out.append(f"\n  direction agreement: {results['agree']}/{results['total']} subjects")
        out.append("  (chance is ~50%; this is the number that grows meaningful as")
        out.append("   more subjects are processed)")


def render_hr(results, out, generated_line=None):
    """render()'s counterpart for the heart-rate study - see run_hr_correlation_study()."""
    out.append("\n" + "=" * 74)
    out.append("STRESS INDEX vs HEART RATE (from BVP)  (UBFC-Phys ground truth)")
    out.append("=" * 74)
    if results is None:
        out.append("  (no feature traces found under datasets/feature_traces/ - skipped)")
        return

    if generated_line:
        out.append(f"  {generated_line}")
    out.append(f"  traces : {results['traces_dir']}")
    out.append(f"  ubfc   : {results['ubfc_dir']}")
    out.append("")
    out.append("Heart rate is derived from raw BVP via bandpass filtering + peak")
    out.append("detection (no hardware HR channel exists in UBFC-Phys) - noisier than")
    out.append("a real HR monitor. Within-task r is indicative only - both series")
    out.append("drift and are autocorrelated. The between-task direction test is the")
    out.append("real claim.")
    out.append("")

    out.append(f"  {'trace':22s}{'n':>6}{'Pearson':>10}{'Spearman':>10}")
    out.append("  " + "-" * 50)
    for row in results['per_trace']:
        label = row['label']
        if row['status'] == 'no_bvp':
            out.append(f"  {label:<22} (no BVP file)")
        elif row['status'] == 'no_peaks':
            out.append(f"  {label:<22} (too few detected heartbeats)")
        elif row['status'] == 'nan':
            out.append(f"  {label:<22}{row['n']:>6}  (NaN in heart rate)")
        else:
            out.append(f"  {label:<22}{row['n']:>6}{row['pearson']:>+10.2f}"
                       f"{row['spearman']:>+10.2f}")

    out.append("")
    out.append("  BETWEEN-TASK (rest T1 -> stressors T2/T3): direction agreement")
    out.append(f"  {'subject':10s}{'HR rest':>10}{'HR stress':>12}"
               f"{'idx rest':>10}{'idx stress':>12}   agree?")
    out.append("  " + "-" * 62)
    for row in results['agreement_rows']:
        out.append(f"  s{row['subject']:<9}{row['hr_rest']:>10.1f}{row['hr_stress']:>12.1f}"
                   f"{row['idx_rest']:>10.3f}{row['idx_stress']:>12.3f}   "
                   f"{'YES' if row['agree'] else 'NO'}")

    if results['total']:
        out.append(f"\n  direction agreement: {results['agree']}/{results['total']} subjects")
        out.append("  (chance is ~50%; this is the number that grows meaningful as")
        out.append("   more subjects are processed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--traces', default='../datasets/feature_traces')
    ap.add_argument('--ubfc', default='../datasets/UBFC-Phys')
    ap.add_argument('--root', default='..',
                    help="project root (default: .. when run from code/)")
    ap.add_argument('--out', default=None,
                    help="output file (default: <root>/evaluation_data/"
                         "CORRELATION_RESULTS_<date>.txt)")
    args = ap.parse_args()

    results = run_correlation_study(args.traces, args.ubfc)
    hr_results = run_hr_correlation_study(args.traces, args.ubfc)
    if results is None and hr_results is None:
        print("no feature traces found")
        return 1

    out = []
    generated_line = f"generated : {datetime.now().isoformat(timespec='seconds')}"
    render(results, out, generated_line=generated_line)
    render_hr(hr_results, out, generated_line=generated_line)
    text = "\n".join(out)
    print(text)

    root = Path(args.root).resolve()
    out_path = (Path(args.out) if args.out else
                root / "evaluation_data" / f"CORRELATION_RESULTS_{datetime.now():%Y-%m-%d}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n\nwritten to: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
