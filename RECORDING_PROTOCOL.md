# Recording Protocol

How to record an evaluation session so the result is usable, comparable, and
still reconstructable in six months.

**Follow this exactly.** PP1's `evaluation_data/baseline_v1/` folder is the
cautionary tale: it is cited in the report as "n=202, 89.6%", but it now
contains 468 pooled rows and reproduces 83.76%. Nobody can tell which files
were in the original run, who recorded them, or under what settings — so that
number is permanently unciteable. Everything below exists to stop that
happening again.

---

## Before you start

**Environment** — keep it constant within a subject, and write down anything unusual.

- Indoor, seated, facing the camera, upper body visible (shoulders must be in frame —
  the posture branch gates itself off below 0.7 shoulder visibility and silently
  contributes nothing).
- Consistent lighting. No window directly behind you.
- Camera roughly at eye level, ~50–70 cm away.
- Close other webcam apps. The system needs exclusive camera access.

**Session hygiene**

- One condition per recording. Never switch from relaxed to stressed mid-recording —
  the label applies to the whole file.
- Aim for **60 seconds minimum**. The 60-frame smoother needs ~2 s to fill, so the
  first ~2 s of every recording is unreliable by design. Shorter recordings let that
  startup artifact dominate.
- **Re-calibrate for every recording.** Do not reuse a baseline across conditions:
  the personal baseline is the thing being tested, and carrying one over from a
  relaxed session into a stressed one leaks information between them.

---

## The procedure

For each recording:

1. **Set the labels** — open **🏷️ Recording labels** in the sidebar and set
   *Subject ID* (e.g. `S01`) and *Condition* (`relaxed` or `stressed`).
   These are baked into the export filename; getting them right here is the whole point.
2. **Pick the context mode** — Default / Studying / Hospital / Driving.
3. **🎯 Calibrate** — sit still, relaxed, neutral expression, 20 seconds.
   Do not pose, do not smile, breathe normally. Confirm the sidebar shows **✅ Calibrated**.
4. **▶ Start**, then perform the condition (below) for at least 60 seconds.
5. **⏸ Stop** — this auto-saves to the session database.
6. **📥 Download session as CSV** — the filename is generated for you. Save it into
   `evaluation_data/incoming/` (see "After recording" below — `recording_intake.py`
   validates it from there, it does not go straight into `evaluation_data/`).
7. **Restart Streamlit before the next recording.** This clears the cached baseline so
   the next calibration genuinely starts fresh.

### What "relaxed" means

Sit normally and do nothing in particular. Breathe steadily, neutral or lightly
pleasant expression, upright but not rigid. Read something undemanding if sitting
still feels artificial. **Do not deliberately relax** — over-performing calm is as
much a confound as over-performing stress.

### What "stressed" means

Perform the behaviours the system actually measures, rather than "feeling" stressed:

- furrow the brow (`brow_tension`, the highest-weighted signal at 0.30)
- clench or grit the jaw (`jaw_tension`, 0.25) — **not a smile**, which is explicitly
  detected and excluded
- blink more frequently (`blink_stress`, 0.15)
- slouch forward and let the neck crane (`slouch` + `neck_strain`)
- shift and fidget (`head_instability`)

A genuine stressor (mental arithmetic under time pressure, a difficult task) is better
science than acting, and should be preferred once you have baseline numbers.
Whichever you use, **record it in the log** — the two are not interchangeable evidence.

---

## Naming

The app generates this automatically. If you ever rename by hand, match it exactly:

```
{condition}_{subject}_{mode}_{YYYY-MM-DD}.csv

relaxed_S01_default_2026-08-12.csv
stressed_S01_studying_2026-08-12.csv
```

Condition comes **first** deliberately, so the existing `relaxed_*.csv` /
`stressed_*.csv` globs in `run_all_evaluations.py` keep working unchanged.

Rules:

- Condition is exactly `relaxed` or `stressed` — nothing else parses.
- Subject IDs are stable and anonymous: `S01`, `S02`, … Never real names. One person
  keeps the same ID forever.
- **Check the spelling.** `relaaxed_2.csv` sat in `evaluation_data/` for months
  contributing nothing, because the glob never matched it. 48 valid samples, silently
  excluded from every number PP1 ever reported.

---

## After recording

1. Drop the CSVs into `evaluation_data/incoming/`, then run:
   ```bash
   cd code && python recording_intake.py
   ```
   This validates filename, condition, mode, and the 60-second minimum before
   moving each file into `evaluation_data/` - a blind manual copy is exactly
   how `relaaxed_2.csv` got in unnoticed. Anything flagged stays in
   `incoming/` with the specific reason printed; fix it (usually a re-export)
   and run the command again. Add `--dry-run` to check without moving
   anything.
2. Regenerate every number:
   ```bash
   cd code && python run_all_evaluations.py
   ```
3. Commit the CSVs **and** the regenerated results file together, so the data and the
   numbers it produces are always in the same commit.
4. Add a line to the log below.

---

## Recording log

Fill this in as you go. It is the only place that captures things the filename cannot.

| Date | Subject | Condition | Mode | Duration | Stressor used | Notes |
|---|---|---|---|---|---|---|
| 2026-05-03 | S01 | relaxed | default | ~52 s | — | PP1 `relaxed_1.csv` |
| 2026-05-03 | S01 | relaxed | default | ~48 s | — | PP1 `relaxed_2.csv` (was misnamed `relaaxed_2`) |
| 2026-05-03 | S01 | stressed | default | ~62 s | acted | PP1 `stressed_1.csv` |
| 2026-05-03 | S01 | stressed | default | ~62 s | acted | PP1 `stressed_2.csv` |
| | | | | | | |

---

## Checklist

- [ ] Subject ID and Condition set in the sidebar **before** starting
- [ ] Calibrated for **this** recording, not carried over
- [ ] Sidebar shows ✅ Calibrated
- [ ] Shoulders visible in frame
- [ ] At least 60 seconds
- [ ] One condition only, start to finish
- [ ] CSV downloaded into `evaluation_data/incoming/`
- [ ] `recording_intake.py` run and the file landed in `evaluation_data/` clean
      (nothing flagged)
- [ ] Streamlit restarted before the next recording
- [ ] `run_all_evaluations.py` re-run and committed alongside the new CSVs
- [ ] Row added to the recording log

---

## Multi-subject sessions (Phase 1)

Cross-subject validation is the headline gap in PP1 — every reported number comes from
one person. When recording others:

- Get **verbal consent** and explain what is captured: no video or images are ever
  stored, only derived numbers (stress index, level, per-branch scores).
- Assign a fresh anonymous ID. Keep any name↔ID mapping **out of the repository**.
- Record **both** conditions per subject, or that subject contributes nothing to an
  accuracy figure.
- Same protocol for everyone. Resist tweaking settings between subjects — that is
  precisely what makes results incomparable.
- Target ≥ 4 subjects × 2 conditions. Session-level *n* is what the confidence
  intervals actually rest on, and right now it is 4 recordings from 1 person.
