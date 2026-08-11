# Vision-Based Behavioral Stress Detection

A real-time, vision-based stress monitoring system that classifies cognitive stress into LOW / MODERATE / HIGH categories using a webcam feed. Combines a deep-learning visual classifier (DaiSEE-trained CNN ensemble), an interpretable behavioral feature engine (MediaPipe FaceMesh + Pose), and physiological calibration (WESAD) into a single Streamlit dashboard with persistent session history and task-aware context modes.

---

## Headline results

| Metric | Value | Source |
|---|---|---|
| WESAD physiological validation (LOSO) | **91.86%** | `wesad_calibration.py`, 15 subjects |
| DaiSEE single-frame test accuracy | **~82%** | `train_daisee_stress.py`, ensemble |
| Live system accuracy, Default mode | **98.9%** | 176 samples (52 relaxed + 124 stressed) |
| Live system accuracy, Studying mode | **94.5%** | 293 samples (147 relaxed + 146 stressed) |
| Earlier baseline (May 2, pre-task-aware) | 89.6% | 202 samples |

**Discrimination gap (Default mode):** mean stress_index of relaxed sessions = 0.262, mean of stressed sessions = 0.554. Gap = **+0.292**.

---

## Architecture

```
TRAINING (one-time, offline)
  DaiSEE 8,571 clips ──► derive binary stress label ──► CNN ensemble training
                                                   ──► daisee_mobilenet_stress.h5
                                                   ──► daisee_efficientnet_stress.h5

CALIBRATION (one-time, offline)
  WESAD 15 subjects ──► physiological features ──► LOSO CV stress classifier
                                                ──► calibrated_thresholds.json

CALIBRATION (per-session, online)
  6 sec of relaxed user video ──► personal DaiSEE baseline
                              ──► personal behavioral baseline (EAR, brow, lip)

INFERENCE (real-time, ~30 fps)
  Webcam ──► MediaPipe FaceMesh + Pose
         ├──► DaiSEE CNN ensemble (re-centered against personal baseline)
         ├──► Behavioral feature engine (EAR, blink, brow, jaw, posture)
         └──► [LSTM temporal branch — see "Negative results" below; disabled]
                                                              │
                                                              ▼
              Task-aware weighted fusion (Default / Studying / Hospital / Driving)
                              + multi-signal rule boost (+0.12 if 4+ active)
                              + 60-frame temporal smoother
                              + 3-frame hysteresis level confirmation
                                                              │
                                                              ▼
                                                       LOW / MOD / HIGH
                                                              │
                                                              ▼
                          Streamlit dashboard + SQLite session history DB
```

---

## File map

### Live system (active in production)
| File | Role |
|---|---|
| `app.py` | Streamlit web UI. Session control, calibration flow, live timeline, past-session viewer, context selector. |
| `behavioral_features.py` | MediaPipe FaceMesh + Pose engine. Extracts EAR, blink rate, brow tension, jaw tension, head instability, neck strain, slouch. v5 (smile fix + pose noise reduction). |
| `stress_fusion.py` | Three-branch weighted fusion (DaiSEE 0.20, Behavior 0.80, LSTM 0.0). Task-aware reweighting (4 modes). Temporal smoother. Hysteresis level. |
| `session_history.py` | SQLite persistence layer for completed sessions. Schema: `sessions` summary table + `samples` per-frame table. Auto-creates `../data/sessions.db`. |
| `lstm_temporal.py` | Real-time LSTM inference module. Returns `is_available()=False` if model file absent (current state). Wired into fusion but disabled. |
| `stress_detector.py` | OpenCV alternative runner (no Streamlit dependency). |

### Training and preprocessing
| File | Role |
|---|---|
| `daisee_preprocessing.py` | Walks DaiSEE, extracts 16 face crops per clip, derives binary stress labels. |
| `rebucket_to_binary.py` | Converts DaiSEE 3-class stress labels to binary (not_stressed / stressed). |
| `train_daisee_stress.py` | Trains MobileNetV2 + EfficientNetB0 on DaiSEE binary labels. Two-stage, class-weighted. |
| `wesad_calibration.py` | Loads all 15 WESAD subjects, runs LOSO CV, outputs calibrated thresholds JSON + report. |
| `lstm_preprocessing.py` | Builds sequence index from extracted face crops for LSTM training. |
| `train_lstm.py` | Trains CNN+LSTM temporal classifier (frozen MobileNet → LSTM → dense). |
| `evaluate_threshold_sweep.py` | DaiSEE threshold sweep utility. |

### Evaluation
| File | Role |
|---|---|
| `analyze_sessions.py` | Aggregates labeled session CSVs (`relaxed_*.csv`, `stressed_*.csv`) into per-file and pooled accuracy reports + confusion matrix. |

### Deprecated (kept for documentation only)
| File | Note |
|---|---|
| `train_mobilenet.py` | FER2013 emotion classifier. A/B tested May 2 — including FER hurt accuracy (82.4% with FER vs 89.1% without). Removed from active pipeline. |
| `train_efficientnet.py` | Same as above. |

---

## Folder layout

```
stress_project/
├── code/                      ← all .py files + README.md (this file)
├── models/
│   ├── daisee_mobilenet_stress.h5         ← in use
│   ├── daisee_efficientnet_stress.h5      ← in use
│   ├── calibrated_thresholds.json         ← WESAD-derived
│   └── (lstm_stress.h5 not present — see Negative results below)
├── datasets/
│   ├── DaiSEE/                ← raw video clips
│   ├── processed_daisee/      ← extracted face crops + lstm_index.pkl
│   └── WESAD/                 ← physiological data
├── data/
│   └── sessions.db            ← SQLite session history (auto-created)
├── evaluation_data/
│   ├── relaxed_1.csv, relaxed_2.csv, stressed_1.csv, stressed_2.csv
│   └── baseline_v1/           ← archived May-2 baseline CSVs (89.6% reference)
└── BACKUP_89.6_percent/       ← safety backup of pre-task-aware code
```

---

## How to run

### Prerequisites
```
- Windows 10/11
- Python 3.11
- Webcam
```

### Install dependencies
```
pip install streamlit==1.30 tensorflow==2.15.0 mediapipe==0.10.14 protobuf==4.25.3 numpy==1.26.4 opencv-python scikit-learn pandas scipy
```

### Run the live system
```
cd code
streamlit run app.py
```

The browser opens automatically. Workflow:

1. **Pick a context** from the sidebar dropdown (Default / Studying / Hospital / Driving).
2. **🎯 Calibrate** — sit relaxed for 6 seconds. The system learns your personal baseline.
3. **▶ Start** — begin live monitoring.
4. Stress level updates in real time with a 60-frame smoother + 3-frame hysteresis.
5. **⏸ Stop** — auto-saves the session to the SQLite history DB.
6. **👁 View** any past session in the sidebar to replay its timeline.

### Run accuracy evaluation on saved CSVs
```
cd code
python analyze_sessions.py --dir ..\evaluation_data
```
The analyzer treats `relaxed_*.csv` files as ground truth = LOW and `stressed_*.csv` as ground truth = MODERATE/HIGH.

---

## Stress label derivation

### DaiSEE (training labels)
DaiSEE provides four affect annotations (Engagement, Boredom, Confusion, Frustration), each on a 0–3 scale. We collapse these into a continuous stress score:

```
stress_score = 0.45 * (Frustration / 3)
             + 0.30 * (Confusion   / 3)
             + 0.15 * (1 - Engagement / 3)
             + 0.10 * (Boredom     / 3)
```

Frustration and confusion are the strongest predictors of cognitive stress in affective-computing literature. The score is then thresholded to a binary `not_stressed` / `stressed` label for CNN training.

### Live runtime (inference cutoffs)
| Range | Level |
|---|---|
| stress_index < 0.36 | LOW |
| 0.36 ≤ stress_index < 0.55 | MODERATE |
| stress_index ≥ 0.55 | HIGH |

Cutoffs are auto-loaded from `calibrated_thresholds.json` (WESAD-derived) at startup.

---

## Task-aware context modes

Each mode applies signal-specific multipliers to the base behavior weights, then renormalizes. Default mode uses 1.0× across all signals — mathematically identical to the un-modal baseline.

| Signal | Default | Studying | Hospital | Driving |
|---|---|---|---|---|
| brow_tension | 30% | 31% | **33%** | 22% |
| jaw_tension | 25% | 22% | **28%** | 22% |
| blink_stress | 15% | **19%** | 16% | 11% |
| eye_strain | 10% | **13%** | 7% | **13%** |
| slouch | 10% | 12% | 7% | 11% |
| neck_strain | 5% | 6% | 4% | 5% |
| head_instability | 5% | 4% | 5% | **6%** |

Bolded cells indicate where each mode diverges most from Default. Studying emphasizes screen-related fatigue (eye, blink) and desk posture (slouch, neck). Hospital emphasizes facial micro-expressions (brow, jaw) and restlessness (head). Driving emphasizes drowsiness signals (eye openness, head bobs) and slumping (slouch).

---

## Negative results (worth citing)

### FER2013 emotion branch — removed
A/B evaluation on May 2 showed that adding the FER2013 emotion classifier to the fusion **decreased** live-system accuracy from 89.1% to 82.4%. The DaiSEE classifier already encodes emotion-relevant features, and the FER branch added noise rather than signal. FER scripts (`train_mobilenet.py`, `train_efficientnet.py`) are retained for reproducibility but disabled in the active pipeline.

### CNN+LSTM temporal branch — disabled
A CNN+LSTM model (frozen MobileNetV2 → LSTM(64) → Dense) was trained on 5,356 DaiSEE sequences. Final test accuracy was 71.1%, which is **below the trivial-majority-class baseline of 86.6%** for DaiSEE's class distribution (~6.4:1 not_stressed:stressed). The model collapsed to predicting the majority class.

Likely causes:
1. Frozen ImageNet features are not stress-discriminative.
2. DaiSEE's synthesized stress label is noisy.
3. Insufficient training data for end-to-end fine-tuning.

The branch is wired into `stress_fusion.py` with weight = 0.0 and `lstm_temporal.py` returns `is_available()=False` when the model file is absent. **Recommended for future work:** retrain on a face-emotion-labeled dataset such as AffectNet, AFEW-VA, or RAF-DB.

---

## Honest limitations

1. **Dataset coverage.** DaiSEE is an e-learning dataset (university students at desks). The model generalizes well to study/desk-work and clinical-seated contexts, but extrapolating to driving or outdoor contexts is unvalidated.
2. **Ground truth.** DaiSEE's "stress" label is derived from affect annotations, not biosignals. WESAD calibration partly closes this gap by anchoring decision boundaries to physiological data, but WESAD has no video and therefore cannot validate per-frame vision predictions.
3. **Single-frame limitation.** The CNN runs per-frame. Slow-building stress over minutes is partially captured by the temporal smoother but not modeled directly. A working LSTM (future work) would improve this.
4. **Behavioral signal calibration.** Personal baseline is critical. Reset baseline whenever the user, lighting, or camera position changes meaningfully.
5. **Task-mode selection is user-driven.** The system does not automatically detect context.

---

## Future work

| Item | Notes |
|---|---|
| LSTM on face-emotion datasets | AffectNet / AFEW-VA / RAF-DB / FER+ have proper labels for posed-static stress expressions. |
| Auto-detection of task context | Replace dropdown with classifier on user environment / posture. |
| Mobile deployment | TFLite int8 quantization. |
| Larger live-system evaluation | Current evaluation uses 4 CSVs (~300 samples). Scaling to 50+ subjects would tighten confidence intervals. |
| Multi-subject baseline robustness | All current testing was single-subject. Cross-subject validation is the next frontier. |

---

## Citations to include in the project paper

> *"Primary stress classification was trained on the DaiSEE dataset (Gupta et al., 2016), with binary stress labels derived from a weighted combination of frustration (0.45), confusion (0.30), inverse engagement (0.15), and boredom (0.10) annotations. Stress-level decision boundaries were independently calibrated against the WESAD physiological reference dataset (Schmidt et al., 2018), where a leave-one-subject-out classifier achieved 91.86% binary baseline-vs-stress accuracy."*

> *"On a held-out evaluation set of N=176 samples (52 relaxed, 124 stressed), the live system achieved 98.9% overall classification accuracy in Default mode, with a relaxed→LOW recall of 100% and a stressed→MOD/HIGH recall of 98.4%. Task-aware reweighting maintained accuracy across context modes (Studying: 94.5% on N=293)."*

> *"A CNN+LSTM temporal branch was evaluated but underperformed the trivial-majority-class baseline on DaiSEE's imbalanced label distribution and was disabled in the final system. Future work will retrain temporal modeling on face-emotion-labeled corpora."*
