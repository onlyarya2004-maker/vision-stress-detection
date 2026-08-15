# code/ — module map

> The full project README (setup, datasets, results, privacy) is at the repo root:
> [`../README.md`](../README.md). This file documents what each module in `code/` does and
> which switches control the system. It is kept in sync with the code — if you change a
> default, change it here too.

## Live system

| Module | Role |
|---|---|
| `app.py` | Streamlit dashboard — main entry point (`streamlit run app.py` from this folder) |
| `config.py` | **Every tunable constant, with provenance.** Start here before changing behaviour |
| `behavioral_features.py` | MediaPipe FaceMesh + Pose engine, 7 signals, personal calibration (face, posture, blink threshold, blink rate) |
| `stress_fusion.py` | Branch fusion, task modes, rule boost, smoothing, hysteresis |
| `rppg_live.py` | Heart rate from the camera (POS), quality-gated, abstains when noisy |
| `learned_fusion.py` | Data-fitted fusion weights (meta-learner inference) |
| `lstm_temporal.py` | LSTM inference wrapper (disabled — see below) |
| `session_history.py` | SQLite persistence: sessions, per-frame samples, self-reports |
| `app_naming.py` | Canonical evaluation-CSV naming + parsing |
| `stress_detector.py` | OpenCV-only runner (no Streamlit) |

## The switches (all in `config.py`)

| Switch | Default | Meaning |
|---|---|---|
| `W_DAISEE / W_BEHAVIOR` | 0.20 / 0.80 | Hand-tuned branch weights (PP1 values) |
| `W_LSTM` | **0.0** | LSTM disabled: scored 71.1% vs an 86.6% majority-class floor |
| `W_RPPG` | **0.0** | rPPG validated offline (POS 3.52 bpm on UBFC-Phys) but unproven on compressed webcam video. HR is still computed and displayed; raise to 0.15–0.25 only after it tracks reality on your hardware |
| `USE_LEARNED_FUSION` | **False** | Meta-learner beats hand-tuned by 10–17 points on UBFC-Phys (LOSO), but was trained on that camera/cohort. Enable only after retraining on recordings made under `../RECORDING_PROTOCOL.md` |
| `USE_WESAD_CUTOFFS` | **False** | The LOW/MOD/HIGH cutoffs (0.36/0.55) are **empirically tuned, not WESAD-derived**. The JSON's cutoffs are percentiles of a physiological classifier's probabilities — a different scale — and are deliberately unused. `calibrated_thresholds.json` is read for provenance/reporting only |

The temporal smoother is a fixed **60 frames (~2 s at 30 fps)** from `config.py`. The
11.1 s window recommended inside the WESAD JSON is **not** used (it describes stress-block
duration in WESAD's protocol, not per-frame vision noise).

## Training & preprocessing (offline)

| Module | Role |
|---|---|
| `daisee_preprocessing.py` → `rebucket_to_binary.py` → `train_daisee_stress.py` | DaiSEE face crops → binary labels → CNN ensemble (.h5) |
| `wesad_calibration.py` | WESAD LOSO (91.86%) → `calibrated_thresholds.json` (provenance) |
| `lstm_preprocessing.py` / `train_lstm.py` | The failed pixel-LSTM (kept as the documented negative result) |
| `train_meta_learner.py` | Learned fusion weights, LOSO over subjects → `../models/meta_learner.joblib` |

## Evaluation & analysis

| Module | Role |
|---|---|
| `run_all_evaluations.py` | **Regenerates every reported number** into one timestamped file. Nothing is ever hand-transcribed |
| `evaluate_video_file.py` | Full pipeline over video files (UBFC-Phys etc.): stress CSVs + per-frame feature traces + rPPG RGB traces, then the video is disposable |
| `diagnose_features.py` | Per-signal discriminability (Cohen's d, saturation, inversion); can diff two runs |
| `analyze_rppg.py` | GREEN/CHROM/POS heart-rate recovery vs BVP ground truth |
| `correlate_physiology.py` | Stress index ↔ EDA correlation per subject/task |
| `analyze_sessions.py` | Quick single-folder accuracy check (superseded by the harness) |
| `evaluate_threshold_sweep.py` | DaiSEE decision-threshold sweep |

## Tests

```bash
python -m pytest tests -q        # from code/, or: python -m pytest code/tests from the root
```

113+ tests. The load-bearing ones: Default-mode fusion equals the archived pre-task-aware
implementation **exactly** (diff 0.00e+00); uncalibrated posture/blink behaviour is
byte-identical to PP1; rPPG methods recover a synthetic pulse of known frequency; every
default-off switch above leaves output unchanged.

## History

`_archive/` holds pre-SQLite and pre-task-aware snapshots, kept for the equivalence tests
and provenance. Do not edit them — tests compare against them.
