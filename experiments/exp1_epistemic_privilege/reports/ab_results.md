# Injection-method A/B — consolidated results

Durable record of the robustness check in `experiment.md` (§"Robustness: does the leak depend on the
injection method?"). Same per-model dose (`--no-calibrate`), only the injection method varies. Raw bundles:
`runs/_ab/*.pt` (released on HF; see `runs/_ab/README.md`). Reproduce with `analysis/analyze_v2.py`,
`analysis/nameability_ab.py`, `analysis/char_tilt.py`.

## R1 — symbol-counter balanced accuracy at the strong dose (chance 0.5, shuffle floor ~0.59)

| model | strong dose | all-position | generation-only | original (published) |
|-------|------|------|------|------|
| 1.5B | s60 | 0.853 | 0.819 | 0.83 |
| 3B   | s60 | 0.794 | 0.772 | 0.79 |
| 7B   | s93 | 0.740 | 0.699* | 0.73 |

All-position reproduces the original within run-to-run noise (Δ ≤ 0.02). Generation-only barely differs
(Δ ≤ 0.04) and never approaches the floor. s0 controls sit at chance below floor (0.51 / 0.53 / 0.49).
*7B generation-only at s93 is *under-dosed* (see nameability) — the dose-matched row below settles it.

## Arm-A nameability — median own-concept rank (lower = concept more strongly instantiated)

| model | dose | all-position | generation-only |
|-------|------|------|------|
| 1.5B | s60 | 3 | 4 |
| 3B   | s60 | 23 | 20 |
| 7B   | s93 | 27 | 68 |

Dose is matched at 1.5B/3B (clean method test). At 7B the same effmag under-injects generation-only (`effmag`
is a per-position knob; fewer positions ⇒ less total perturbation, an effect that grows with scale).

## 7B dose-matched (generation-only, effmag raised to equalize nameability)

| dose (effmag) | arm-A rank | R1 |
|------|------|------|
| 93 (original) | 68 | 0.699 |
| 124 | 47 | 0.826 |
| 140 | 38 | **0.858** |

At near-matched nameability (rank 38 vs all-position's 27), generation-only 7B R1 = **0.858 — above**
all-position's 0.740. The 7B difference at equal effmag was the *dose*, not the method.

## char-tilt — mean total-variation of the character histogram from the s0 control

| model | all-position | generation-only | ratio (all : gen) |
|-------|------|------|------|
| 1.5B | 0.281 | 0.260 | 1.08× |
| 3B   | 0.165 | 0.157 | 1.05× |
| 7B   | 0.110 | 0.116 | 0.94× |

The raw character signal the reader exploits is preserved under generation-only (within ±8%, and *higher* at
7B). Both methods show the tilt shrinking with scale — the published "weakens with scale" trend, intact.

## Conclusion

The leak — *random-looking output leaks the injected concept, recoverable by a pure symbol-counter* — is
**not** an artifact of soaking the prompt in the concept. It reproduces under the stricter generation-only
steering at matched dose, across all three scales. The robustness check hardens the headline finding.
