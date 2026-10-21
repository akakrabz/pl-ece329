# ECE 329 — PrairieLearn course (practice questions)

A PrairieLearn course directory for self-study: **52 randomized "v3" questions** covering Lectures 1–30,
one course instance and **eight assessments** (seven practice sets and a timed, repeatable Midterm 2 practice exam).
File formats are checked against PrairieLearn's own JSON schemas (vendored in `tools/schemas/`, October 2026)
and its element documentation.

- **13 original questions** (September): five modelled on FA26 Exam 1 (and the Lecture 3 flux challenge), four on
  HW4 / Lecture 10 material, four on magnetostatics (Lectures 12–15).
- **19 concept questions, one per lecture (L1–L19)** (October): multiple choice with feedback on every wrong
  choice, select-all statement pools, matching, and one or two numerical parts each.
- **11 concept questions, one per lecture (L20–L30)** (October 20): Poynting's theorem, phasors, damped waves,
  good conductors, polarization, reflection, standing waves, line parameters, bounce diagrams, pulses on lines, line
  junctions — every bounce-diagram key checked against an independent line simulation.
- **9 exam-style Midterm 2 questions** (October): multi-part problems modelled on past ECE 329 Exam 2 problem
  families, re-parameterized (new numbers and at least one changed detail; no answer-key text), plus a quick-fire
  set of six short multiple-choice items drawn from twelve generators.

```
pl-ece329/
├── infoCourse.json                      topics; tags (lec01…lec19, midterm2, exam-style, topic tags); "Practice" set
├── courseInstances/Fa26/
│   ├── infoCourseInstance.json
│   └── assessments/
│       ├── exam1-practice/              Practice 1: the five Exam 1 families
│       ├── dielectrics-practice/        Practice 2: HW4-style (Lectures 8–10)
│       ├── magnetostatics-practice/     Practice 3: Lectures 12–15
│       ├── concepts-lectures-1-11/      Practice 4: one concept question per lecture, L1–L11
│       ├── concepts-lectures-12-19/     Practice 5: one concept question per lecture, L12–L19
│       ├── midterm2-exam-style/         Practice 6: the nine exam-style Midterm 2 problems, untimed
│       ├── concepts-lectures-20-30/     Practice 7: one concept question per lecture, L20–L30
│       └── midterm2-practice-exam/      Practice Exam 1: timed (75 min), 5 problems drawn from pools, repeatable
├── questions/
│   ├── electrostatics/                  Lectures 1–11
│   ├── magnetostatics/                  Lectures 12–15
│   ├── waves/                           Lectures 16–26
│   ├── transmission-lines/              Lectures 27–30
│   └── midterm2/                        exam-style problems, Lectures 11–19
└── tools/
    ├── test_questions.py                offline checker (see below)
    ├── checks/<question>.py             independent physics check for each new question
    ├── render_question.py               print one question rendered for one seed (proofreading)
    ├── mustache_lite.py, katex_check.js helpers for the checker
    └── schemas/                         PrairieLearn's JSON schemas (see SOURCE.txt)
```

### The new questions

| Lecture | Question | What it practises |
|---|---|---|
| 1–2 | `electrostatics/lorentz-force-work` | F = q(E + v×B) components, power F·v, what the magnetic force can't do, velocity selector |
| 2 | `electrostatics/coulomb-null-point` | where E = 0 for two charges, field by superposition, scaling |
| 3 | `electrostatics/gauss-symmetry` | matching distributions to field shapes, when Gauss's law fails, ball + shell (Q r³/a³) |
| 4 | `electrostatics/div-curl-classification` | div and curl of a linear field; could it be E? B? Stokes; continuity |
| 5 | `electrostatics/potential-work-superposition` | V by superposition, work by an agent, potential-energy sign, potential facts |
| 6 | `electrostatics/magnetic-boundary-conditions` | H across current sheets (n̂ from 2 into 1), J_s from a jump, which law gives which condition |
| 7 | `electrostatics/pn-junction-poisson` | depletion widths, peak field, built-in potential, Poisson facts |
| 8 | `electrostatics/conductor-shell-relaxation` | induced charges on a shell, field outside, relaxation time ε/σ |
| 8–9 | `electrostatics/polarization-bound-charge` | what ∇·P tells you: ρ_b, ρ_sb, total bound charge, E = −P/ε₀ in an electret |
| 10 | `electrostatics/capacitor-plate-separation` | C, energy, force, work, battery bookkeeping at fixed Q or fixed V |
| 11 | `electrostatics/drude-lorentz-models` | σ = Nq²τ/m, mobility, drift speed, AC σ(ω), Lorentz χe sign |
| 12 | `magnetostatics/biot-savart-arcs` | B at the centre of arc-and-radius loops; why radial segments contribute nothing |
| 13 | `magnetostatics/ampere-symmetry-toroid` | matching current distributions to H shapes, toroid field and flux, vector potential |
| 14 | `magnetostatics/rotating-rod-emf` | ½BωL², which end is positive, current and mechanical power |
| 15 | `magnetostatics/mutual-inductance-rl` | M of a pick-up coil, induced emf, reciprocity, RL switch-on |
| 16 | `waves/displacement-current-pec` | ∂D/∂t in a charging capacitor, H between the plates, J_s = n̂×H on a perfect conductor |
| 17 | `waves/magnetization-materials` | J_sM = M×n̂, B and H in a magnet, dia/para/ferro, μ interface |
| 18 | `waves/plane-wave-reading` | direction and speed from the argument, ε_r, H = k̂×E/η, where the pulse is |
| 19 | `waves/sheet-radiation-poynting` | v_p, β, λ; E and H from a cosine current sheet; instantaneous S |
| 12–13 | `midterm2/current-slabs-ampere` | H from two current slabs and a sheet (Summer 2019 HE2 #2a family) |
| 13, 15 | `midterm2/vector-potential-sheet` | B, J_s or J, and E from a given A (Summer 2019 HE2 #2b family) |
| 13–14 | `midterm2/nested-solenoids-lenz` | cancelling the core field, Lenz, emf of a coaxial loop (Summer 2017/2020 HE2 family) |
| 14 | `midterm2/moving-loop-gradient` | loop moving through a graded field: orientation, flux, emf, current (Summer 2018/2020 HE2 family) |
| 14 | `midterm2/voltmeters-changing-flux` | three voltmeters that disagree around changing flux (Summer 2017/2018 HE2 #3 family) |
| 13, 15, 17 | `midterm2/parallel-plate-line-inductance` | J_s, B, flux, L and energy of a magnetically filled shorted line (SP18 Exam 2 #3 family) |
| 18 | `midterm2/pulse-on-the-move` | a pulse at another time and place, ε_r, H (SP18 Exam 2 #4 family) |
| 19 | `midterm2/sheet-launches-waves` | a pulsed current sheet: E and H on both sides, timing, snapshot (SP18 Exam 2 #5 family) |
| 11–19 | `midterm2/quickfire-mc` | six short multiple-choice items from twelve generators (SP18 Exam 2 #1 style) |
| 20 | `waves/poynting-power-ledger` | Poynting's theorem statements; a resistive wire heated through its side; ⟨S⟩ through a tilted window |
| 21 | `waves/phasor-conversion-power` | time domain ↔ phasor matching; H̃ and ⟨S⟩ from Ẽ; direction from a template; plasma β, v_p |
| 22 | `waves/damped-wave-decoding` | α, β, λ, δ; η = jωμ/γ; σ and ε_r from γ/η; the other field's amplitude and lag |
| 23 | `waves/good-conductor-skin-depth` | classifying media by loss tangent; δ, \|η\|; dB and % of power left; 45° lag |
| 24 | `waves/polarization-handedness` | linear/circular/elliptical and handedness; ⟨S⟩ of a circular wave; plasma v_p, v_g |
| 25 | `waves/reflection-interface` | Γ, τ, reflected/transmitted fields, power fractions; PEC J_s |
| 26 | `waves/standing-wave-mirror` | E and H nulls, peak fields, reflected hand, J_s, radiation pressure |
| 27 | `transmission-lines/line-parameters-waves` | 𝓛, 𝓒, v, Z₀; V/I of a pulse; instantaneous power; rebuilt lines |
| 28 | `transmission-lines/bounce-step-response` | injection, Γ_L, Γ_g, V(z,t), I(z,t), DC limit |
| 29 | `transmission-lines/pulse-snapshot-record` | snapshots and records of a pulse and its echo; overlap |
| 30 | `transmission-lines/two-line-junction` | junction Γ and τ both ways, first echoes, junction voltage, matching resistor |

Each question is `info.json` + `question.html` + `server.py`. All randomization happens in `generate(data)`, and
`server.py` imports only `random`, `math`, `fractions` and `numpy`, so the questions need nothing beyond the
PrairieLearn runtime. Select-all parts use `partial-credit="each-answer"`; most statement pools are larger than
what one variant shows.

## Install

Copy the contents of this folder into your PrairieLearn course repository (or point a local PrairieLearn at it):

```bash
# local docker PrairieLearn, from https://docs.prairielearn.com/installing/
docker run -it --rm -p 3000:3000 -v "$PWD":/course prairielearn/prairielearn
```

Then open http://localhost:3000, load the course, and open each question's preview. Sync errors to expect if you
merge into an existing course: duplicate UUIDs (regenerate with `python3 -c "import uuid;print(uuid.uuid4())"`)
and undeclared topics/tags (add them to your `infoCourse.json` or change the questions' `topic`/`tags`).

The practice exam is an `Exam`-type assessment with `multipleInstance` (start a fresh instance any time) and a
75-minute timer (`durationMinutes` in `accessControl`); change the duration there if your midterm is longer or
shorter.

## Test without PrairieLearn

```bash
python3 tools/test_questions.py 30                       # quick; numpy only
python3 tools/test_questions.py 300 --only gauss-symmetry,coulomb-null-point
python3 tools/test_questions.py 60 --katex path/to/katex.min.js   # also compile every equation (needs node)
python3 tools/render_question.py gauss-symmetry 7        # see one variant as PrairieLearn's template pass would
```

For every question and seed the checker:

- runs `generate()`, renders `question.html` with a small Mustache renderer, and fails on any unset name;
- checks every multiple-choice block (exactly one correct, distinct choices), select-all pool (enough true and
  false statements for `number-answers`), matching block and numeric answer (finite);
- rejects unknown or deprecated element attributes, undeclared tags/topics, duplicate UUIDs, assessment
  references to missing questions, and (with `pip install jsonschema`) anything PrairieLearn's schemas reject;
- **recomputes every answer independently** — the original questions' checks are in the script, the new ones in
  `tools/checks/<question>.py` (brute-force Coulomb/Biot–Savart sums, quadrature, finite-difference div/curl,
  `np.cross` for every direction, time stepping), including the key of every multiple-choice and the truth of every
  select-all statement.

Every new question was also re-solved from its rendered statement by an independent reviewer; their findings are
summarised in the October entry below.

## Notes on the design

- **Problem 1 family** generates the field *from* a potential, so it is conservative by construction; a field with
  random polynomial components almost never is, and then "find V" has no answer.
- **Dielectric interface** picks $E_{1z}$ so that $E_{2z}$ is an integer (via the gcd of the two $\epsilon_r$).
- **Flux MC** builds its distractors from the actual mistakes (both charges counted +, closed-surface answer,
  flipped normal, swapped above/below) and de-duplicates them against the correct value.
- **Concept questions**: wrong choices encode real misconceptions and carry `feedback` explaining the slip;
  statements are worded to be unambiguously true or false in every variant; choices never print a number that
  answers another part.
- **Moving loop in a graded field** uses the curl-free field B = B₀[(1 + x/L) ẑ + (z/L) x̂]-type form (the statement
  says why); on the loop's plane it equals the exams' single-component field.
- Points: Homework-type with `autoPoints`/`maxAutoPoints` (e.g. 5/15) so a student can bank three correct variants;
  `triesPerVariant` 2–3 so the hidden hints (`show-after-submission`) have a chance to appear. The practice exam
  uses declining `autoPoints` lists ([20, 14, 8]).
- Hints use `pl-hidden-hints` (the only place `pl-hint` is allowed); `order="fixed"` replaces the deprecated
  `fixed-order`; `accessControl`/`publishing` are the current (non-deprecated) access mechanisms.

## Changelog

- **October 20, 2026** — 11 concept questions for Lectures 20–30 and the Practice 7 assessment; new tags (lec20–lec30,
  exam3, topic tags); checker now rejects a `params` key that collides with a choice element's `answers-name`
  (the cause of the "duplicate params variable name" error fixed in 13 questions on October 18). Independent review of
  the 11: 2 blockers (ambiguous select-all statements), 5 majors (tolerances, answers accepted for the wrong reason,
  feedback wording), all fixed; no wrong numeric key.

- **October 2026** — 28 new questions (19 per-lecture concept questions, 9 exam-style Midterm 2 questions), four
  new assessments, a stronger checker (rendering, KaTeX, schemas, per-question check modules). Independent review
  of all 28 found no wrong numeric answer; it fixed one select-all statement that was not clearly true in every
  variant, several choices that gave away other parts' answers, tolerances in three questions that were too tight for
  a student carrying three significant figures, and wording.
- **September 2026** — the first 13 questions and three practice sets.
