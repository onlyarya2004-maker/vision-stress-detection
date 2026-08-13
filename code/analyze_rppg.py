"""
rPPG analysis and validation
============================
Recovers heart rate from the per-frame face-RGB traces produced by
evaluate_video_file.py, and scores it against ground truth when available.

Why this matters for the project
--------------------------------
The system's whole pitch is "physiological-grade monitoring with no sensors".
Until now that rested on WESAD-derived thresholds applied offline. rPPG makes
it literal: heart rate measured from the same webcam pixels, at runtime.

That matters because heart rate is the single most discriminative feature in
the project's own WESAD analysis - Cohen's d = +1.61, larger than every
behavioural signal - and it is currently the one strong signal the live
system cannot sense at all.

Methods
-------
GREEN  the raw green channel. Simplest, most motion-sensitive. Baseline only.
CHROM  de Haan & Jeanne (2013). Chrominance projection that cancels much of
       the specular (motion) component.
POS    Wang et al. (2017). Projects onto a plane orthogonal to the skin-tone
       direction. Generally the strongest of the three; the default here.

Usage
-----
    # single trace, with ground truth
    python analyze_rppg.py --trace ../datasets/rppg_traces/rppg_vid_s1_T1.csv \
        --bvp ../datasets/UBFC-Phys/s1/s1/bvp_s1_T1.csv

    # every trace, auto-matching UBFC-Phys ground truth
    python analyze_rppg.py --dir ../datasets/rppg_traces --ubfc-root ../datasets/UBFC-Phys
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sps

# Physiologically plausible band: 42-180 bpm.
HR_MIN_HZ, HR_MAX_HZ = 0.7, 3.0
BVP_FS = 64.0            # Empatica E4 BVP sample rate
WINDOW_SECONDS = 12.0    # HR estimation window
WINDOW_STEP = 2.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_trace(path, roi='cheek'):
    """Load an rPPG RGB trace. Returns (t, rgb[N,3], fs)."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    t = np.array([float(r['t']) for r in rows])
    rgb = np.array([[float(r[f'{roi}_r']), float(r[f'{roi}_g']),
                     float(r[f'{roi}_b'])] for r in rows])
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("non-monotonic timestamps")
    return t, rgb, 1.0 / dt


def load_bvp(path):
    return np.array([float(l) for l in open(path) if l.strip()])


# ---------------------------------------------------------------------------
# rPPG methods.  Each maps an RGB trace to a 1-D pulse signal.
# ---------------------------------------------------------------------------
def _bandpass(x, fs):
    hi = min(HR_MAX_HZ, 0.99 * fs / 2)
    if hi <= HR_MIN_HZ:
        return x - np.mean(x)
    b, a = sps.butter(3, [HR_MIN_HZ / (fs / 2), hi / (fs / 2)], btype='band')
    return sps.filtfilt(b, a, x)


def method_green(rgb, fs):
    return _bandpass(rgb[:, 1] - rgb[:, 1].mean(), fs)


def method_chrom(rgb, fs):
    """de Haan & Jeanne (2013)."""
    mean = rgb.mean(axis=0) + 1e-9
    rn, gn, bn = (rgb / mean).T
    xs = 3 * rn - 2 * gn
    ys = 1.5 * rn + gn - 1.5 * bn
    xf, yf = _bandpass(xs, fs), _bandpass(ys, fs)
    alpha = np.std(xf) / (np.std(yf) + 1e-9)
    return xf - alpha * yf


def method_pos(rgb, fs):
    """
    Wang et al. (2017), Plane-Orthogonal-to-Skin, overlap-added over a
    ~1.6s sliding window as in the original formulation.
    """
    n = len(rgb)
    win = max(4, int(1.6 * fs))
    out = np.zeros(n)
    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for i in range(0, n - win + 1):
        block = rgb[i:i + win]
        mu = block.mean(axis=0) + 1e-9
        Cn = block / mu
        S = P @ Cn.T                        # (2, win)
        h = S[0] + (np.std(S[0]) / (np.std(S[1]) + 1e-9)) * S[1]
        out[i:i + win] += h - h.mean()      # overlap-add
    return _bandpass(out, fs)


METHODS = {'green': method_green, 'chrom': method_chrom, 'pos': method_pos}


# ---------------------------------------------------------------------------
# Heart-rate estimation
# ---------------------------------------------------------------------------
def hr_from_signal(x, fs):
    """Dominant frequency in the cardiac band, via Welch PSD, in bpm."""
    if len(x) < int(fs * 4):
        return np.nan
    nper = min(len(x), int(fs * WINDOW_SECONDS))
    f, p = sps.welch(x, fs=fs, nperseg=nper)
    band = (f >= HR_MIN_HZ) & (f <= HR_MAX_HZ)
    if not band.any():
        return np.nan
    return float(f[band][np.argmax(p[band])] * 60.0)


def windowed_hr(x, fs, window_s=WINDOW_SECONDS, step_s=WINDOW_STEP):
    """HR over sliding windows -> (centre_times, bpm)."""
    win, step = int(window_s * fs), max(1, int(step_s * fs))
    times, hrs = [], []
    for start in range(0, max(1, len(x) - win + 1), step):
        seg = x[start:start + win]
        if len(seg) < win:
            break
        hrs.append(hr_from_signal(seg, fs))
        times.append((start + win / 2) / fs)
    return np.array(times), np.array(hrs)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def analyse(trace_path, bvp_path=None, roi='cheek', verbose=True):
    t, rgb, fs = load_trace(trace_path, roi=roi)
    duration = t[-1] - t[0]
    result = {'trace': Path(trace_path).name, 'fs': fs,
              'duration': duration, 'roi': roi, 'methods': {}}

    if verbose:
        print(f"\n{Path(trace_path).name}")
        print(f"  {len(t)} samples, {duration:.1f}s, fs={fs:.1f} Hz, roi={roi}")
        if fs < 15:
            print(f"  NOTE: {fs:.1f} Hz is low for rPPG (20-30 Hz preferred). "
                  f"Re-run extraction with a smaller --sample-every for a "
                  f"cleaner pulse.")

    ref_hr, ref_series = None, None
    if bvp_path and Path(bvp_path).exists():
        bvp = load_bvp(bvp_path)
        bvp = bvp[:int(duration * BVP_FS)] if duration else bvp
        ref_hr = hr_from_signal(_bandpass(bvp - bvp.mean(), BVP_FS), BVP_FS)
        _, ref_series = windowed_hr(_bandpass(bvp - bvp.mean(), BVP_FS), BVP_FS)
        result['reference_hr'] = ref_hr
        if verbose:
            print(f"  ground truth (BVP @ {BVP_FS:.0f}Hz): {ref_hr:.1f} bpm")

    for name, fn in METHODS.items():
        sig = fn(rgb, fs)
        hr = hr_from_signal(sig, fs)
        entry = {'hr': hr}
        if ref_hr is not None and not np.isnan(hr):
            entry['abs_error'] = abs(hr - ref_hr)
        # windowed agreement, when both series exist
        if ref_series is not None:
            _, est_series = windowed_hr(sig, fs)
            m = min(len(est_series), len(ref_series))
            if m >= 3:
                a, b = est_series[:m], ref_series[:m]
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() >= 3:
                    entry['windowed_mae'] = float(np.mean(np.abs(a[ok] - b[ok])))
                    if np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                        entry['windowed_corr'] = float(
                            np.corrcoef(a[ok], b[ok])[0, 1])
        result['methods'][name] = entry

        if verbose:
            line = f"  {name.upper():6s} {hr:6.1f} bpm"
            if 'abs_error' in entry:
                line += f"   err {entry['abs_error']:5.1f}"
            if 'windowed_mae' in entry:
                line += f"   windowed MAE {entry['windowed_mae']:5.1f}"
            if 'windowed_corr' in entry:
                line += f"   r={entry['windowed_corr']:+.2f}"
            print(line)

    return result


UBFC_RE = re.compile(r's(\d+)_T(\d)', re.IGNORECASE)


def find_bvp(trace_name, ubfc_root):
    """Locate the BVP file matching an rPPG trace filename."""
    m = UBFC_RE.search(trace_name)
    if not m or not ubfc_root:
        return None
    s, t = m.group(1), m.group(2)
    for cand in Path(ubfc_root).rglob(f"bvp_s{s}_T{t}.csv"):
        return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--trace', help='single rPPG trace CSV')
    src.add_argument('--dir', help='directory of traces')
    ap.add_argument('--bvp', help='ground-truth BVP CSV (with --trace)')
    ap.add_argument('--ubfc-root', help='UBFC-Phys root, to auto-match BVP files')
    ap.add_argument('--roi', default='cheek',
                    choices=['cheek', 'forehead', 'face'])
    args = ap.parse_args()

    traces = ([Path(args.trace)] if args.trace
              else sorted(Path(args.dir).glob('rppg_*.csv')))
    if not traces:
        print("no traces found")
        return 1

    results = []
    for tr in traces:
        bvp = args.bvp if args.trace else find_bvp(tr.name, args.ubfc_root)
        if not args.trace and args.ubfc_root and bvp is None:
            print(f"\n{tr.name}\n  (no matching BVP found)")
        try:
            results.append(analyse(tr, bvp, roi=args.roi))
        except Exception as e:
            print(f"\n{tr.name}\n  ERROR: {e}")

    scored = [r for r in results if 'reference_hr' in r]
    if len(scored) > 1:
        print("\n" + "=" * 62)
        print("SUMMARY vs BVP ground truth")
        print("=" * 62)
        print(f"  {'method':8s}{'mean |err| bpm':>18s}{'traces':>10s}")
        for name in METHODS:
            errs = [r['methods'][name]['abs_error'] for r in scored
                    if 'abs_error' in r['methods'][name]]
            if errs:
                print(f"  {name.upper():8s}{np.mean(errs):>18.2f}{len(errs):>10d}")
        print("\n  Lower is better. This is the number that justifies adding")
        print("  an rPPG branch to the live fusion in Phase 2.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
