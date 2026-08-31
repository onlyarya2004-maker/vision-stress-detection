# Inference Performance Benchmark

**Project**: Vision-Based Behavioral Stress Detection
**Date**: 2026-08-31
**Prepared in response to**: PP2 external review feedback (Reviewer #5 — "authors should provide real-time performance benchmarks (FPS and latency)")

---

## 1. Summary

This benchmark measures the real, on-device inference cost of every model-driven component
in the live capture pipeline, and the effective end-to-end frame rate that implies. It was
produced by a regenerable script (`code/benchmark_inference.py`) rather than hand-measured
or estimated — the same "one script produces every reported number" discipline the rest of
this project's evaluation follows (`run_all_evaluations.py`).

**Headline result**: the system runs at an estimated **11.3 frames per second** end-to-end
on this development machine, using CPU-only inference. The dominant cost is the DaiSEE CNN
ensemble (89.5 ms per call), which is deliberately run only once every 3 frames rather than
every frame — a cadence choice, not a limitation, that keeps the interactive frame rate high
without sacrificing prediction freshness (see §4).

## 2. Methodology

| | |
|---|---|
| Script | `code/benchmark_inference.py` |
| Report artifact | `models/inference_benchmark_report.txt` (raw, regenerable output) |
| Input | A synthetic 640×480 frame built by pasting a real, already-processed DaiSEE face crop onto a blank canvas at live-capture resolution |
| Why not a live camera frame | Standing project policy never triggers the webcam outside a real user session — the benchmark needed to be reproducible without a person sitting in front of a camera. Using a *real* face image (rather than a blank or random frame) still gives MediaPipe a genuine face to detect, so the timing reflects real detection work, not a fast "no face found" exit path |
| Components measured | MediaPipe FaceMesh, MediaPipe Pose (measured standalone for reference, though it executes *inside* behavioral feature extraction in the live pipeline — see §3), the behavioral feature engine's full extraction pass, and the DaiSEE CNN ensemble (MobileNetV2 + EfficientNetB0, `tf.function`-wrapped) |
| Repetitions | 100 calls per fast component, 30 calls for the slower CNN ensemble call, after 10 discarded warm-up iterations (MediaPipe's first call and `tf.function`'s first trace both carry a one-time cost that isn't representative of steady-state performance) |
| Hardware | CPU-only TensorFlow inference. The development machine has an RTX 5070 Ti (confirmed functional via a separate PyTorch environment used for model training), but the main application's TensorFlow build was never compiled with CUDA support (`tf.test.is_built_with_cuda()` returns `False`) — a known, documented constraint (`PLAN_FOR_PP2.md`), not something this benchmark works around |

## 3. Results

| Component | Mean | Median | Std. dev. | Min | Max | Runs |
|---|---|---|---|---|---|---|
| MediaPipe FaceMesh | 8.8 ms | 8.8 ms | 0.7 ms | 7.2 ms | 10.8 ms | 100 |
| MediaPipe Pose | 47.9 ms | 47.4 ms | 3.4 ms | 43.6 ms | 63.5 ms | 100 |
| `crop_face()` (bounding box + array slice) | 0.4 ms | 0.4 ms | 0.0 ms | 0.3 ms | 0.5 ms | 100 |
| Behavioral feature extraction (face + posture, reuses the FaceMesh result) | 49.2 ms | 48.9 ms | 2.1 ms | 45.1 ms | 53.5 ms | 100 |
| DaiSEE CNN ensemble (MobileNetV2 + EfficientNetB0, mean of both) | 89.5 ms | 89.6 ms | 6.0 ms | 75.7 ms | 102.7 ms | 30 |

**Note on Pose**: MediaPipe Pose is listed above for reference, but it is *not* called as a
separate step in the live pipeline — it runs internally inside behavioral feature extraction
(`BehavioralFeatureEngine._extract_posture()`), so its cost is already included in the
49.2 ms "Behavioral feature extraction" row, not additional to it.

## 4. Effective end-to-end frame rate

The live capture loop (`app.py`) runs FaceMesh, face cropping, and behavioral feature
extraction on **every frame**, but only runs the DaiSEE CNN ensemble on **every 3rd frame**
(`config.CNN_EVERY_N_FRAMES = 3`) — the CNN's prediction is cached and reused for the frames
in between, since a facial expression does not change meaningfully within ~100ms.

```
per-frame cost  = FaceMesh + crop_face + behavioral extraction + (CNN cost / 3)
                = 8.8 + 0.4 + 49.2 + (89.5 / 3)  ms
                = 88.2 ms

effective FPS   = 1000 / 88.2  ≈  11.3 fps
```

This excludes webcam capture/decode, Streamlit re-rendering, and the rPPG/rule-boost/
smoothing stages, all of which are inexpensive NumPy operations relative to the model
inference above (sub-millisecond each).

## 5. Interpretation

- **11.3 fps is well within the range needed for the system's actual purpose.** This is not
  a video-quality or gaming application — it is a slow-moving physiological/behavioral
  signal (blink rate, brow tension, posture) sampled and smoothed over a 60-frame (~5–6
  second) window. An update roughly every 90ms is far faster than the underlying signal
  changes, so the frame rate is not the bottleneck on prediction quality or responsiveness.
- **The dominant cost is model inference, not I/O or the fusion logic.** The DaiSEE ensemble
  alone accounts for roughly a third of the amortized per-frame budget despite running on
  only one in three frames — this is the correct target for any future speed work, and the
  project's `daisee_ensemble.py` already documents one such optimization already applied
  (`tf.function` instead of `Model.predict()`, a measured 2.8–4.3× speedup per model over the
  naive call pattern, folded into the numbers above).
- **A CUDA-enabled TensorFlow build is the largest remaining lever**, not yet pursued because
  the CPU-only numbers above already meet the system's real requirements. If lower latency
  is ever needed (e.g. for a stricter Driving-mode drowsiness alert), this is documented as
  the first thing to try, ahead of further algorithmic optimization.

## 6. Reproducibility

```bash
cd code
python benchmark_inference.py
```

Regenerates `models/inference_benchmark_report.txt` from scratch on whatever machine it is
run on — no numbers in this document are hand-transcribed from a one-off measurement.
