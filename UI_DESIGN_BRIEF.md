# UI Design Brief — Vision-Based Behavioral Stress Detection

> Paste this whole document into a design tool or AI assistant to generate UI
> mockups. It describes the real, working product — every feature listed here
> exists in code today. Nothing is aspirational.

## What the product is

A desktop web app that monitors a person's **cognitive stress in real time using
only their webcam** — no wearables, no sensors, nothing touching the user. It
watches the face and upper body, fuses several AI signals, and classifies stress
as **LOW / MODERATE / HIGH**, continuously, about twice per second.

Its differentiating feature is an **on-device AI agent**: it doesn't just
display a number — it decides when the user is stressed enough to be worth
interrupting, delivers a targeted micro-intervention ("unclench your jaw",
"reset your posture"), then **measures whether that intervention actually
lowered the user's stress**, and learns over time which interventions work for
this specific person. The system also retrains its own decision layer from the
user's end-of-session self-reports.

Audience for the design: it must read as a **shippable commercial product** —
think clinical instrument / mission-control console — not a student prototype.

## The user journey (design each state)

1. **IDLE / OFFLINE** — app open, camera off. User sees: status "OFFLINE",
   past-session history, the AI agent panel (standby), a prompt to calibrate.
2. **CALIBRATING (20 seconds)** — camera on. The system learns this person's
   neutral face, posture, eye openness and blink rate. UI shows live video, a
   progress indicator, and instructions: *sit naturally, neutral expression,
   do not smile*.
3. **LIVE MONITORING** — the main event. Live video with a face bounding box
   tinted by current level (green/amber/red). Stress index, level, per-signal
   breakdown, heart rate, charts — all updating in real time. The AI agent's
   status is always visible.
4. **INTERVENTION MOMENT** — the agent fires: a prominent but non-modal card
   appears ("AI INTERVENTION — Unclench your jaw" + one sentence of guidance),
   stays ~20 s, then the agent visibly enters "evaluating" (measuring whether
   the index drops over the next 2½ minutes), then "cooldown".
5. **SESSION END** — summary screen: duration, sample count, % time in each
   level, full timeline chart, CSV export. Plus the **self-report prompt**:
   "How stressed did you actually feel? 1–5 / Skip" — the answer trains the
   system.
6. **HISTORY REVIEW** — pick any past session from the sidebar and replay its
   full timeline.

## Complete feature inventory

### Live monitoring core
- **Live webcam feed** with face bounding box, tinted by stress level
- **Stress index**: continuous 0.00–1.00, smoothed; large level readout
  LOW (green) / MODERATE (amber) / HIGH (red) with a progress bar
- **Seven behavioral signals**, each 0–1, shown as ranked cards with a short
  code, name, description, and live value:
  EYE eye strain · BLK blink rate · BRW brow tension · JAW jaw tension ·
  MOV head movement · NCK neck strain · PST posture/slouch
- **Metric tiles**: Blinks/min · Eye openness · **Heart rate (bpm)** —
  heart rate is measured *from the camera* (remote photoplethysmography),
  shows "--" while the 10 s signal window fills
- **Guidance panel**: level-appropriate advice list

### The AI agent (must feel like a first-class presence, not a toggle)
- **Agent panel**: name, glowing status LED, state ACTIVE / STANDBY, a
  one-paragraph description, and its **learning memory** ("12 outcomes
  measured; most effective: unclench jaw, 7/9 helped")
- **Activate / Deactivate control** — a real switch; when off, the agent
  provably does nothing and the UI says so
- **Live agent status strip** during sessions, cycling through phases with
  reasons: WARMING ("settling in, 32 s") → WATCHING ("monitoring all seven
  signals") → SHOWING (intervention up) → EVALUATING ("measuring effect,
  78 s left") → COOLDOWN ("holding back — nagging poisons outcome data")
- **Intervention cards**: targeted, signal-specific, orange-accented

### Context & personalization
- **Context modes**: Default / Studying / Hospital / Driving — reweights the
  signals for the situation
- **Personal calibration** (20 s) — everything is scored relative to *this
  user's* neutral state
- **Self-report loop**: 1–5 rating after each session; the system's fusion
  layer retrains itself from these (versioned, on-device)

### Charts & history
- **Sidebar sparkline**: last 60 s of stress index
- **Session timeline chart**: three series — stress index (orange, primary),
  CNN branch (blue), behavioral branch (slate) — on a 0–1 axis
- **Session summary stats**: length, samples, % LOW / MODERATE / HIGH
- **Past sessions list**: state dot, date/time, duration, mode tag
  (GEN/STU/MED/DRV), View to replay; delete-all with confirmation
- **CSV export** with research-grade auto-naming (subject / condition fields
  in a "Recording labels" section, used for formal evaluations)

### Status & trust elements
- Global status pill: OFFLINE / CALIBRATING / LIVE
- Calibrated / Not-calibrated indicator
- Privacy note: all processing on-device; no video is ever stored — only
  derived numbers

## Design direction

- **Palette**: near-black base (#0B0F14 family), **blue** as the primary/brand
  accent (#2F6FED), **orange** exclusively for stress/alerts/interventions
  (#F5762B), green/amber/red strictly for LOW/MOD/HIGH semantics. No purple
  gradients, no neon rainbow.
- **Tone**: precision instrument. Dense but ordered. Tabular numerals for all
  metrics. Uppercase micro-labels with letterspacing for section headers.
- **Absolutely no emojis.** Icons are line-style SVG or monospace code badges.
- **Layout**: desktop-first, ~1440px. Suggested: left rail (session controls,
  agent panel, history), large center stage (video + level), right column
  (signals, metrics, guidance), full-width timeline below. But propose better
  arrangements freely.
- The **video feed and the stress level are the heroes**; the agent panel is
  the signature element and deserves visual prominence when it acts.
- Motion: subtle — pulsing status LEDs, smooth chart updates, an intervention
  card that slides in and demands a glance without blocking anything.

## Constraints (for realistic mockups)

- Currently rendered with Streamlit (Python), so favor card/column layouts
  over exotic overlays; assume real implementation will approximate, not
  pixel-match.
- Everything runs locally on a laptop; there is no cloud, no accounts, no
  login screen needed.
- Single user per machine for now; no profile switcher required.

## Deliverables wanted from the designer/AI

1. IDLE state (calibrate prompt, history, agent standby)
2. LIVE monitoring state (the money shot)
3. INTERVENTION moment (agent card active, agent status = SHOWING)
4. SESSION END (summary + self-report)
Desktop, dark theme only.
