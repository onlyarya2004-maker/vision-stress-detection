# code/ — module map

> The full project README (setup, datasets, results, privacy) is at the repo root:
> [`../README.md`](../README.md). This file documents what each module in `code/` does and
> which switches control the system. It is kept in sync with the code — if you change a
> default, change it here too.

## Live system

| Module | Role |
|---|---|
| `app.py` | Streamlit dashboard — main entry point (`streamlit run app.py` from this folder). Also holds the Phase 5.1 analytics view (`session_analytics_dataframe()` + `render_analytics_page()`) — an in-app `st.session_state.view` toggle reached via the sidebar's "View analytics" button, not a separate page, to avoid duplicating the custom sidebar styling or touching the live capture loop |
| `daisee_ensemble.py` | DaiSEE CNN model loading + inference, shared by `app.py` and `evaluate_video_file.py` so live and offline paths run provably identical code. `tf.function`-wrapped instead of `Model.predict()` — ~3.4x faster per call, same predictions |
| `config.py` | **Every tunable constant, with provenance.** Start here before changing behaviour |
| `behavioral_features.py` | MediaPipe FaceMesh + Pose engine, 7 signals, personal calibration (face, posture, blink threshold, blink rate), PERCLOS (Driving mode, displayed only — see below) |
| `stress_fusion.py` | Branch fusion, task modes, rule boost, smoothing, hysteresis |
| `rppg_live.py` | Heart rate from the camera (POS), quality-gated, abstains when noisy. `status()` drives the live HR topbar's text, dimmed while abstaining |
| `learned_fusion.py` | Data-fitted fusion weights (meta-learner inference) |
| `intervention_agent.py` | The agentic core: perceive → decide → act → measure → learn, on-device, no LLM/cloud (deliberate non-goal, stated in the module docstring). Targeting is Thompson-Sampling over a Beta(helped+1, no_effect+1) posterior per intervention kind (real exploration, not a fixed blend), bounded so a clear urgency gap always wins outright; retirement uses a Wilson confidence interval (`eval_scoring.wilson()`) instead of a raw trial-count floor; holds back if the index is already falling on its own; success is tracked per-(task_mode, kind) with a fallback to the global stat until a context bucket has enough evidence (`config.INTERVENTION_MODE_MIN_TRIALS`); the general break can compete against a specific signal when none clearly dominates; `status()` explains its own choice, not just its phase |
| `temporal_model.py` | GRU temporal inference wrapper (disabled — see below). NumPy-only by design — the model trains in PyTorch (`.venv-train`) but this module has to load cleanly in the live app's TensorFlow venv, so a hand-verified NumPy replica of the GRU forward pass runs live instead of torch itself |
| `session_history.py` | SQLite persistence: sessions, per-frame samples, self-reports, user profiles + per-user calibration baselines (Phase 1 prerequisite for team recordings) |
| `session_report.py` | Phase 5.2: renders one saved session into a downloadable 2-page PDF (summary stats + timeline) via matplotlib's `PdfPages` — no separate PDF library. `session_history.summarize_history()` supplies the same duration/mean/max/pct numbers `save_session()` persists, so the report and the DB row can never drift apart |
| `session_pipeline.py` | **The one per-frame pipeline both capture modes share** (local cv2 camera and the browser camera call the same `CalibrationRun`/`LiveRun`, so they cannot drift apart). Also `UserState` — one private engine + baselines + GRU buffer + rPPG buffer per login session (these used to be process-wide singletons) — and `BatchGate`, which validates every batch a browser sends (epoch/sequence de-duplication, size limits, out-of-order frames) |
| `browser_camera/` | Streamlit custom component (plain HTML/JS, no build step) that owns the camera **in the user's browser**: shows it locally at full resolution, mirrors + 4:3-crops to 640×480, posts JPEG snapshots (~10/s, batched, with an acknowledgement-based backpressure), draws the label/face box the server returns |
| `frame_ops.py` | Pure image helpers: `prepare_processing_frame` (every model sees centre-cropped 640×480, the geometry it was validated on), `decode_jpeg` (untrusted-input limits), `crop_face`, `annotate_frame` |
| `session_store.py` | Hosted storage backend: `SupabaseStore` has the same function names as `session_history.py`, per user, paged reads (PostgREST returns ≤1000 rows/request), explicit `user_id` filters on top of the database's Row Level Security |
| `auth.py` | Supabase email/password login for the hosted app; one client per browser session (never cached across users); refuses a service-role key |
| `app_naming.py` | Canonical evaluation-CSV naming + parsing |
| `recording_intake.py` | Validates freshly-exported session CSVs against `RECORDING_PROTOCOL.md` (filename, condition, mode, 60s minimum) at a staging folder (`../evaluation_data/incoming/`) before they reach `evaluation_data/live_recordings/` - the pipeline-entry-point guard against a repeat of the `relaaxed_2.csv` incident |
| `stress_detector.py` | OpenCV-only runner (no Streamlit) — uses `daisee_ensemble.py` for CNN inference, same as `app.py`/`evaluate_video_file.py`, so all three runners are provably identical rather than three hand-kept copies |

## The switches (all in `config.py`)

| Switch | Default | Meaning |
|---|---|---|
| `W_DAISEE / W_BEHAVIOR` | 0.20 / 0.80 | Hand-tuned branch weights (PP1 values) |
| `W_TEMPORAL` | **0.0** | Temporal branch disabled: the Phase 2.1 feature-sequence GRU that replaced the old pixel-LSTM also scored below the majority-class floor (45.74% vs 86.60%; balanced accuracy 54.17%, barely above chance). See `../models/temporal_gru_test_report.txt`. Separately, even setting accuracy aside: the model trains on 16 frames spanning ~10s, but live `add_frame()` is called every processed webcam frame with no pacing, so the live window spans well under 2s — a distribution shift that must be fixed before this is ever raised above 0.0 (see the comment in `config.py`) |
| `W_RPPG` | **0.0** | rPPG validated offline (POS 3.52 bpm on UBFC-Phys) but unproven on compressed webcam video. HR is still computed and displayed; raise to 0.15–0.25 only after it tracks reality on your hardware |
| `USE_LEARNED_FUSION` | **False** | Meta-learner beats hand-tuned by 10–17 points on UBFC-Phys (LOSO), but was trained on that camera/cohort. Enable only after retraining on recordings made under `../RECORDING_PROTOCOL.md` |
| `USE_WESAD_CUTOFFS` | **False** | The LOW/MOD/HIGH cutoffs (0.24/0.55) are **empirically tuned, not WESAD-derived**. 0.24 replaces the original 0.36 as of 2026-09-27 — see `config.py`'s `LOW_TO_MODERATE_CUTOFF` comment and `derive_thresholds.py`. The JSON's cutoffs are percentiles of a physiological classifier's probabilities — a different scale — and are deliberately unused. `calibrated_thresholds.json` is read for provenance/reporting only |

The temporal smoother is a fixed **60 frames (~2 s at 30 fps)** from `config.py`. The
11.1 s window recommended inside the WESAD JSON is **not** used (it describes stress-block
duration in WESAD's protocol, not per-frame vision noise).

**PERCLOS** (Phase 2.3): the standard drowsiness metric (% of the last 60s with the eye
≥80% closed), computed from EAR the same way `blinks_per_minute` already is. Always
computed, but only *displayed* — as a fourth metric tile — when `task_mode == 'Driving'`;
it does not carry a `W_*` switch because it doesn't feed the fusion score at all, same
"shown but not fused in" treatment as heart rate. See `config.PERCLOS_CLOSURE_RATIO`.

## Training & preprocessing (offline)

| Module | Role |
|---|---|
| `daisee_preprocessing.py` → `rebucket_to_binary.py` → `train_daisee_stress.py` | DaiSEE face crops → binary labels → CNN ensemble (.h5) |
| `daisee_test_eval.py` | Phase 4.1: loads the saved `.h5` ensemble and re-evaluates it on the held-out test split (inference only, never retrains) → `../models/daisee_test_report.txt`, echoed by `run_all_evaluations.py`. Not called from the harness directly — a few minutes of CPU inference over 28,531 images, unlike the cheap ablation/correlation/learned-fusion sections |
| `wesad_calibration.py` | WESAD LOSO (91.86%) → `calibrated_thresholds.json` (provenance) |
| `lstm_preprocessing.py` / `train_lstm.py` | The failed pixel-LSTM, retired to `_archive/lstm_temporal.py` (kept as the documented negative result) |
| `train_temporal_model.py` | Phase 2.1 GRU: DaiSEE `processed_daisee/` crops → behavioral-feature sequences → GRU + GBT-baseline ablation → `../models/temporal_gru.npz`. Run only via `.venv-train\Scripts\python`; also below the majority floor (kept as a second documented negative result) |
| `train_meta_learner.py` | Learned fusion weights, LOSO over subjects → `../models/meta_learner.joblib` via `--save`. The LOSO comparison itself (Phase 4.1) is also called from `run_all_evaluations.py`, which never writes the model file — only `--save` does |
| `prototype_warmstart_personalization.py` | Research script (not wired into anything): validates on UBFC-Phys whether initializing a personal fusion model from the population meta-learner beats training from scratch under low personal data. See `../models/warmstart_personalization_prototype_report.txt` and `PLAN_FOR_PP2.md` |
| `retrain_from_sessions.py` | Phase 3.3: your own labelled sessions → a personal fusion model, **scoped per task mode** and warm-started from `../models/meta_learner.joblib` when present (the prototype's finding, integrated — see `PLAN_FOR_PP2.md`) → `../models/personal_fusion_<mode>_vN.joblib`. Manual activation only |

## Evaluation & analysis

| Module | Role |
|---|---|
| `run_all_evaluations.py` | **Regenerates every reported number** into one timestamped file. Nothing is ever hand-transcribed |
| `eval_scoring.py` | Shared frame-/session-level accuracy scoring + Wilson CIs, used by both `run_all_evaluations.py` and `ablation_study.py` (split out to avoid a circular import) |
| `ablation_study.py` | Phase 4.2: each fusion branch removed once on UBFC-Phys cross-subject data, replaying the real `TemporalSmoother`/`HysteresisLevel` over saved `feature_traces/`. Called from `run_all_evaluations.py`, not a separate report file |
| `evaluate_video_file.py` | Full pipeline over video files (UBFC-Phys etc.): stress CSVs + per-frame feature traces + rPPG RGB traces, then the video is disposable |
| `diagnose_features.py` | Per-signal discriminability (Cohen's d, saturation, inversion); can diff two runs |
| `analyze_rppg.py` | GREEN/CHROM/POS heart-rate recovery vs BVP ground truth |
| `correlate_physiology.py` | Phase 4.4: stress index ↔ EDA correlation per subject/task, within- and between-task. Called from `run_all_evaluations.py` (same compute/render split as `ablation_study.py`), still standalone-runnable |
| `analyze_sessions.py` | Quick single-folder accuracy check (superseded by the harness) |
| `evaluate_threshold_sweep.py` | DaiSEE decision-threshold sweep |
| `benchmark_inference.py` | Phase 6.2: real, regenerable FPS/latency numbers for FaceMesh/Pose/behavioral extraction/the DaiSEE ensemble and the effective end-to-end frame rate they imply → `../models/inference_benchmark_report.txt`. No camera — benchmarks against a synthetic frame built from a real on-disk DaiSEE face crop |

## Tests

```bash
python -m pytest tests -q        # from code/, or: python -m pytest code/tests from the root
```

474+ tests. The load-bearing ones: Default-mode fusion equals the archived pre-task-aware
implementation **exactly** (diff 0.00e+00); uncalibrated posture/blink behaviour is
byte-identical to PP1; rPPG methods recover a synthetic pulse of known frequency; every
default-off switch above leaves output unchanged.

## History

`_archive/` holds pre-SQLite and pre-task-aware snapshots, kept for the equivalence tests
and provenance. Do not edit them — tests compare against them.
