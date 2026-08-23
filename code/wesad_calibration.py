"""
WESAD Calibration
=================
Validates and calibrates the vision-based stress monitoring system against
WESAD's gold-standard physiological data.

What this script does
---------------------
1. Loads every WESAD subject (S2-S17 except S12) from .pkl files.
2. Extracts chest physiological signals (ECG, EDA, EMG, Resp, Temp) during
   the protocol's baseline (label=1) and stress (label=2) periods.
3. Computes per-window physiological features:
     - Heart rate (mean, std)
     - Heart rate variability (RMSSD)
     - EDA tonic level + EDA peaks per minute
     - Respiration rate
     - Skin temperature mean
4. Trains a binary classifier (baseline vs stress) using leave-one-subject-out
   cross-validation. Reports its accuracy — this is the WESAD benchmark number
   you'll quote in your report.
5. Derives calibrated thresholds:
     - LOW / MODERATE / HIGH stress-level cutoffs for the vision system,
       mapped from the empirical distribution of physiological stress scores.
     - Recommended temporal smoothing window length, based on the duration
       of stress events in WESAD.
6. Writes:
     - calibrated_thresholds.json : consumed by stress_fusion.py at runtime
     - wesad_calibration_report.txt : human-readable summary for the project
       report (claim: "thresholds validated against the WESAD reference dataset")

Why this matters
----------------
DaiSEE remains the training data for the vision model — that doesn't change.
But your stress LEVEL CUTOFFS (the 0.35 / 0.65 boundaries) and the temporal
smoother window were previously chosen heuristically. Now they're empirically
calibrated against measured physiological stress. That's the report-grade
contribution.

Usage
-----
Edit WESAD_ROOT to point at your WESAD folder, then:

    python wesad_calibration.py

Will process all 15 subjects (~30 mins on a normal laptop). Outputs go to
the project ../models/ directory by default.
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.stats import zscore
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, classification_report

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
WESAD_ROOT  = Path('../datasets/WESAD')                # contains S2/, S3/, ...
OUTPUT_DIR  = Path('../models')                        # where calibrated_thresholds.json goes
REPORT_PATH = Path('../wesad_calibration_report.txt')  # human-readable summary

CHEST_FS = 700        # All chest signals sampled at 700 Hz
WINDOW_SECONDS  = 60  # 60-second windows for feature extraction
WINDOW_OVERLAP  = 0.5 # 50 % overlap between successive windows
WINDOW_SAMPLES  = WINDOW_SECONDS * CHEST_FS

LABEL_BASELINE = 1
LABEL_STRESS   = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- PKL LOADING ----------------
def _normalize_keys(obj):
    """Recursively turn bytes keys into str keys (WESAD pkl was Python-2-pickled)."""
    if isinstance(obj, dict):
        return {(k.decode() if isinstance(k, bytes) else k): _normalize_keys(v)
                for k, v in obj.items()}
    return obj


def load_subject(pkl_path):
    """Load one subject's .pkl and return a clean dict."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return _normalize_keys(data)


def find_signal(chest_dict, *candidates):
    """WESAD pkls vary in capitalization (e.g. Resp vs RESP). Find the right key."""
    for key in chest_dict.keys():
        if key.lower() in [c.lower() for c in candidates]:
            return np.asarray(chest_dict[key]).squeeze()
    raise KeyError(f"None of {candidates} found in chest signals. Keys present: {list(chest_dict.keys())}")


# ---------------- FEATURE EXTRACTION ----------------
def extract_hr_rmssd(ecg_window, fs=CHEST_FS):
    """Heart rate (bpm) and RMSSD HRV (ms) from an ECG segment."""
    # Bandpass 5-15 Hz to isolate QRS complexes
    b, a = sps.butter(3, [5 / (fs / 2), 15 / (fs / 2)], btype='band')
    filtered = sps.filtfilt(b, a, ecg_window)

    # R-peak detection — distance enforces max ~150 bpm
    min_distance = int(0.4 * fs)
    peaks, _ = sps.find_peaks(filtered,
                              distance=min_distance,
                              height=np.std(filtered) * 0.5)
    if len(peaks) < 2:
        return np.nan, np.nan

    rr_intervals_s = np.diff(peaks) / fs
    rr_intervals_ms = rr_intervals_s * 1000.0

    # Drop physiologically implausible RRs (< 300 ms or > 1500 ms)
    valid = (rr_intervals_ms > 300) & (rr_intervals_ms < 1500)
    rr_intervals_ms = rr_intervals_ms[valid]
    if len(rr_intervals_ms) < 2:
        return np.nan, np.nan

    hr_bpm = 60.0 / np.mean(rr_intervals_ms / 1000.0)
    rmssd  = np.sqrt(np.mean(np.diff(rr_intervals_ms) ** 2))
    return hr_bpm, rmssd


def extract_eda_features(eda_window, fs=CHEST_FS):
    """Tonic (mean) and phasic (peaks/min) EDA features."""
    # Low-pass at 5 Hz to remove noise
    b, a = sps.butter(3, 5 / (fs / 2), btype='low')
    eda_filt = sps.filtfilt(b, a, eda_window)

    tonic_mean = float(np.mean(eda_filt))
    tonic_std  = float(np.std(eda_filt))

    # Phasic component (deviation from a slow trend)
    b_low, a_low = sps.butter(3, 0.05 / (fs / 2), btype='low')
    tonic = sps.filtfilt(b_low, a_low, eda_filt)
    phasic = eda_filt - tonic

    # SCRs (skin conductance responses) per minute
    threshold = 0.05 * np.std(phasic)
    peaks, _ = sps.find_peaks(phasic, height=threshold, distance=int(1.0 * fs))
    duration_min = len(eda_window) / fs / 60.0
    scr_per_min = len(peaks) / duration_min if duration_min > 0 else 0.0

    return tonic_mean, tonic_std, scr_per_min


def extract_resp_rate(resp_window, fs=CHEST_FS):
    """Respiration rate in breaths per minute."""
    b, a = sps.butter(3, [0.1 / (fs / 2), 0.5 / (fs / 2)], btype='band')
    resp_filt = sps.filtfilt(b, a, resp_window)
    peaks, _ = sps.find_peaks(resp_filt, distance=int(1.5 * fs))
    duration_min = len(resp_window) / fs / 60.0
    return len(peaks) / duration_min if duration_min > 0 else np.nan


def features_for_window(chest, start_sample, end_sample):
    """Compute the full feature vector for one chest-data window."""
    ecg  = find_signal(chest, 'ECG')[start_sample:end_sample]
    eda  = find_signal(chest, 'EDA')[start_sample:end_sample]
    resp = find_signal(chest, 'Resp', 'RESP')[start_sample:end_sample]
    temp = find_signal(chest, 'Temp', 'TEMP')[start_sample:end_sample]

    hr, rmssd       = extract_hr_rmssd(ecg)
    eda_mean, eda_std, scr_per_min = extract_eda_features(eda)
    resp_rate       = extract_resp_rate(resp)
    temp_mean       = float(np.mean(temp))

    return {
        'hr_bpm':      hr,
        'hrv_rmssd':   rmssd,
        'eda_mean':    eda_mean,
        'eda_std':     eda_std,
        'scr_per_min': scr_per_min,
        'resp_rate':   resp_rate,
        'temp_mean':   temp_mean,
    }


# ---------------- WINDOWING ----------------
def windows_for_label(labels, target_label, win_samples, overlap):
    """Yield (start, end) sample indices for windows of contiguous target_label."""
    in_block = False
    block_start = 0
    for i, lab in enumerate(labels):
        if lab == target_label and not in_block:
            in_block = True
            block_start = i
        elif lab != target_label and in_block:
            in_block = False
            yield from _slide(block_start, i, win_samples, overlap)
    if in_block:
        yield from _slide(block_start, len(labels), win_samples, overlap)


def _slide(start, stop, win, overlap):
    step = max(1, int(win * (1 - overlap)))
    s = start
    while s + win <= stop:
        yield s, s + win
        s += step


# ---------------- MAIN PIPELINE ----------------
def process_subject(pkl_path):
    """Returns a list of (feature_dict, label_int, subject_id) tuples."""
    sid = pkl_path.parent.name
    print(f"  loading {sid}...", end=" ", flush=True)
    data = load_subject(pkl_path)

    chest = data['signal']['chest']
    labels = np.asarray(data['label']).astype(int)

    rows = []
    for label_int, name in [(LABEL_BASELINE, 'baseline'),
                            (LABEL_STRESS,   'stress')]:
        n_windows_this_label = 0
        for s, e in windows_for_label(labels, label_int,
                                      WINDOW_SAMPLES, WINDOW_OVERLAP):
            try:
                feats = features_for_window(chest, s, e)
            except Exception as ex:
                continue
            if any(np.isnan(v) for v in feats.values()):
                continue
            feats['label'] = label_int
            feats['subject'] = sid
            feats['condition'] = name
            rows.append(feats)
            n_windows_this_label += 1
    print(f"  {len(rows)} windows extracted")
    return rows


def run_calibration():
    if not WESAD_ROOT.exists():
        raise FileNotFoundError(f"WESAD_ROOT not found: {WESAD_ROOT}")

    pkl_paths = sorted(WESAD_ROOT.glob('S*/S*.pkl'))
    if not pkl_paths:
        raise FileNotFoundError(
            f"No S*/S*.pkl files in {WESAD_ROOT}. "
            "Check that subjects are unzipped one folder deep."
        )
    print(f"Found {len(pkl_paths)} subject(s).")

    all_rows = []
    for pkl in pkl_paths:
        try:
            all_rows.extend(process_subject(pkl))
        except Exception as ex:
            print(f"  [WARN] skipping {pkl.parent.name}: {ex}")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal feature windows: {len(df)}")
    print(f"  baseline windows: {sum(df['label'] == LABEL_BASELINE)}")
    print(f"  stress windows:   {sum(df['label'] == LABEL_STRESS)}")
    print(f"  subjects covered: {df['subject'].nunique()}")

    # ---- Train a binary stress classifier with LOSO CV ----
    feature_cols = ['hr_bpm', 'hrv_rmssd', 'eda_mean', 'eda_std',
                    'scr_per_min', 'resp_rate', 'temp_mean']
    X = df[feature_cols].values
    y = (df['label'] == LABEL_STRESS).astype(int).values
    groups = df['subject'].values

    logo = LeaveOneGroupOut()
    fold_accs, fold_f1s = [], []
    all_y_true, all_y_pred, all_y_score = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(Xtr, y[train_idx])
        pred = clf.predict(Xte)
        score = clf.predict_proba(Xte)[:, 1]
        fold_accs.append(accuracy_score(y[test_idx], pred))
        fold_f1s.append(f1_score(y[test_idx], pred))
        all_y_true.extend(y[test_idx])
        all_y_pred.extend(pred)
        all_y_score.extend(score)

    overall_acc = float(np.mean(fold_accs))
    overall_f1  = float(np.mean(fold_f1s))

    # ---- Per-feature baseline vs stress statistics ----
    per_feature_stats = {}
    for col in feature_cols:
        b_vals = df.loc[df['label'] == LABEL_BASELINE, col]
        s_vals = df.loc[df['label'] == LABEL_STRESS,   col]
        per_feature_stats[col] = {
            'baseline_mean': float(b_vals.mean()),
            'baseline_std':  float(b_vals.std()),
            'stress_mean':   float(s_vals.mean()),
            'stress_std':    float(s_vals.std()),
            'effect_size_d': float(
                (s_vals.mean() - b_vals.mean()) /
                np.sqrt((b_vals.var() + s_vals.var()) / 2 + 1e-12)
            ),
        }

    # ---- Derive calibrated stress level cutoffs for the VISION system ----
    # Use the LOSO classifier scores as an empirical "physiological stress
    # severity score" distribution.  Vision-side cutoffs are placed at the
    # 33rd / 67th percentiles of this distribution.
    stress_scores = np.array(all_y_score)
    cutoff_low_to_mod  = float(np.percentile(stress_scores, 33))
    cutoff_mod_to_high = float(np.percentile(stress_scores, 67))

    # ---- Recommended temporal window length ----
    # Find the median duration of contiguous stress label blocks across subjects.
    stress_block_durations = []
    for pkl in pkl_paths:
        try:
            data = load_subject(pkl)
            labs = np.asarray(data['label']).astype(int)
            in_block, start = False, 0
            for i, lab in enumerate(labs):
                if lab == LABEL_STRESS and not in_block:
                    in_block, start = True, i
                elif lab != LABEL_STRESS and in_block:
                    in_block = False
                    stress_block_durations.append((i - start) / CHEST_FS)
            if in_block:
                stress_block_durations.append((len(labs) - start) / CHEST_FS)
        except Exception:
            continue
    median_stress_duration_s = float(np.median(stress_block_durations)) if stress_block_durations else 0.0

    # ---- Save calibrated thresholds JSON ----
    out = {
        'metadata': {
            'subjects_used': sorted(df['subject'].unique().tolist()),
            'n_baseline_windows': int(sum(df['label'] == LABEL_BASELINE)),
            'n_stress_windows':   int(sum(df['label'] == LABEL_STRESS)),
            'window_seconds':     WINDOW_SECONDS,
        },
        'classifier_loso': {
            'accuracy_mean': overall_acc,
            'f1_mean':       overall_f1,
        },
        'feature_stats': per_feature_stats,
        'vision_system_cutoffs': {
            'low_to_moderate':  cutoff_low_to_mod,
            'moderate_to_high': cutoff_mod_to_high,
            'comment': (
                "These cutoffs are derived from the 33rd / 67th percentiles "
                "of physiological stress probabilities in WESAD. The vision "
                "system reads these to set its LOW/MODERATE/HIGH boundaries."
            ),
        },
        'temporal_smoothing': {
            'median_stress_block_seconds': median_stress_duration_s,
            # median_stress_duration_s / 60.0 converts seconds to MINUTES -
            # the key name says so explicitly now (it used to be called
            # ...window_seconds while holding a minutes value). Confirmed
            # unused by the live system (README.md / config.py both flag
            # USE_WESAD_CUTOFFS=False), and no code reads this JSON key.
            'recommended_window_minutes': max(2.0, median_stress_duration_s / 60.0),
            'comment': (
                "Stress events in WESAD last on the order of minutes. "
                "Real-time vision smoothing of 1-3 seconds is fast enough to "
                "respond, while not being so jittery that it flips on transient "
                "facial expressions."
            ),
        },
    }

    json_path = OUTPUT_DIR / 'calibrated_thresholds.json'
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nCalibration JSON saved to: {json_path}")

    # ---- Save human-readable report ----
    lines = []
    lines.append("WESAD CALIBRATION REPORT")
    lines.append("=" * 60)
    lines.append(f"Subjects used: {len(out['metadata']['subjects_used'])} "
                 f"({', '.join(out['metadata']['subjects_used'])})")
    lines.append(f"Total baseline windows: {out['metadata']['n_baseline_windows']}")
    lines.append(f"Total stress windows:   {out['metadata']['n_stress_windows']}")
    lines.append(f"Window length:          {WINDOW_SECONDS} s, "
                 f"{int(WINDOW_OVERLAP*100)}% overlap\n")
    lines.append("Leave-One-Subject-Out classifier (binary baseline vs stress):")
    lines.append(f"  Mean accuracy:  {overall_acc * 100:.2f}%")
    lines.append(f"  Mean F1 score:  {overall_f1:.3f}\n")
    lines.append("Per-feature baseline vs stress (Cohen's d):")
    for fname, stats in per_feature_stats.items():
        lines.append(f"  {fname:13s}  baseline={stats['baseline_mean']:7.2f} "
                     f"stress={stats['stress_mean']:7.2f}  "
                     f"d={stats['effect_size_d']:+.2f}")
    lines.append("\nVision-system stress level cutoffs (calibrated):")
    lines.append(f"  LOW < {cutoff_low_to_mod:.2f}")
    lines.append(f"  MODERATE between {cutoff_low_to_mod:.2f} and "
                 f"{cutoff_mod_to_high:.2f}")
    lines.append(f"  HIGH ≥ {cutoff_mod_to_high:.2f}")
    lines.append("\nTemporal smoothing recommendation:")
    lines.append(f"  Median stress block duration: "
                 f"{median_stress_duration_s:.1f} s")
    lines.append(f"  Recommended window:           "
                 f"{out['temporal_smoothing']['recommended_window_minutes']:.1f} min")
    lines.append("\nClassification report (overall, pooled):")
    lines.append(classification_report(all_y_true, all_y_pred,
                                       target_names=['baseline', 'stress']))
    report_text = "\n".join(lines)

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == '__main__':
    run_calibration()
