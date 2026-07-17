# Pre-registration: does the injected channel close, plateau, or grow at 14B?

**Registered before any 14B data is collected.** Question: the *natural* channel closes by 7B, but the
*injected* one does not close cleanly — the distribution readout falls with scale (dist@T12 2.64 → 2.24 →
0.89), the transcript fingerprint is FLAT from 3B → 7B (char@full 2.06 → 0.72 → 0.77), and exp1's one
dose-matched 7B check read *higher* than the unmatched run. Three live hypotheses our 3 points cannot
separate: the injected channel (1) keeps closing, (2) plateaus (steering permanently output-detectable),
(3) grows at matched dose. One scale point decides among them directionally.

## Design (frozen)

- **Model:** Qwen2.5-14B-Instruct (registry slug `qwen2.5-14b`), the next rung of the same family.
- **Collection:** `src/covert_collect.py`, generation-only injection (`--inject gen`), `orig` variant,
  word-free filter, target 30 clean streams/concept/strength — the exact exp1 protocol. Smoke first
  (measures `resid_norm`, pins it in the registry); then the full run with the **on-box auto-calibrated**
  eff_mags. The calibration + nameability/capability gates are the established cross-scale dose-matching
  mechanism: strengths are tuned per model until the concept is nameable (arm-A rank ≤ 50) with capability
  retained (clean fraction ≥ 0.70 × own baseline) — so the comparison is at matched *relative* dose, the
  same criterion that produced the 1.5B/3B/7B points.
- **Analysis:** `analysis/dose_titration.py --slug qwen2.5-14b` — identical protocol to the 1.5B/7B
  titrations (accepted streams, common-N 24/class, seeds 0/1/2, per-channel best-decoder nested-CV, bits
  currency). No new analysis code.
- **Baselines:** the 7B cells from `dose_titration_qwen2.5-7b.json` (same capture format, same protocol,
  computed before this prereg was frozen — values filled in below when that local run completes; they are
  baselines, not outcomes).

## Primary measures (frozen)

At the **stronger calibrated operating point** (the s60-equivalent, matching how the 1.5B/3B/7B headline
cells were taken), pooled over seeds 0/1/2:

- **P1 — transcript channel:** `char@full` (the accumulating fingerprint; the safety-relevant
  "output-detectability of steering" number).
- **P2 — distribution channel:** `dist@T12` (the fast channel).

Let σ = the per-cell seed sd. Verdicts per channel, 14B vs 7B:

- **CLOSING:** 14B < 0.5 × 7B *and* the drop exceeds 2σ (combined).
- **PLATEAU:** 14B within [0.5×, 1.5×] of 7B *and* 14B ≥ 0.2 bits (still clearly readable).
- **GROWING:** 14B > 1.5 × 7B *and* the rise exceeds 2σ (combined).
- Otherwise **MIXED/INCONCLUSIVE** (e.g. 14B < 0.2 bits but the drop is within noise), reported as such.

The headline verdict is P1's (the transcript channel — the question "would a much larger model still leak
an injected concept *into its text*"). P2 is reported alongside. If the channels disagree (e.g. P2 closing,
P1 plateau — the 3B→7B pattern), that disagreement IS the result and is reported as such, not averaged.

## Validity gates (frozen — no verdict without them)

1. **Injection took:** the calibration nameability gate passes (arm-A own-concept rank ≤ 50) and clean
   fraction ≥ 0.70 × the model's own uninjected baseline. Fail ⇒ "under-injected, no verdict," not
   "closing."
2. **Negative control:** strength-0 cells read ≈ 0 (|bits| < 0.1 for every reader). Fail ⇒ pipeline bug,
   no verdict.
3. **Enough long streams:** char@full requires the realistic-cohort protocol (min_len 8); if accepted
   stream count per class < 24, use the max common-N and report it.

## Named calls (committed now, before data)

1. **P1 (char@full) lands PLATEAU** — between 0.4 and 1.2 bits: steering stays output-detectable at 14B.
   This is the bet the 3B→7B flatness (0.72 → 0.77) motivates.
2. **P2 (dist@T12) continues down but stays > 0.3 bits** — the distribution readout keeps shrinking
   with scale without reaching the natural floor.
3. The medium dose (s40-equivalent) again shows the channel decoupling: dist saturated near the strong
   dose, char well below it.

Being wrong on these calls is fine and will be reported plainly (as with the confusion-geometry prereg,
where the named calls failed and were reported as failed).

## Interpretation locked in advance

- CLOSING (both channels) ⇒ the injected channel follows the natural one with a lag; the "steered models
  betray the steer" claim is a small-model phenomenon; extrapolation worry decreases.
- PLATEAU on P1 ⇒ **activation steering remains detectable from output text alone at every scale tested**
  — the standalone audit fact strengthens, and the writeup's 7B caveat becomes a positive claim.
- GROWING on either ⇒ adverse-direction scaling at matched relative dose; this becomes the lead follow-up
  result and the writeup's scale framing must be rewritten before publication.
- MIXED ⇒ report per-channel; no synthesis beyond what the cells support.

---

## Amendment 1 — 2026-07-08, before any 14B data collected

Forced by the completed 7B protocol-matched baseline (`dose_titration_qwen2.5-7b.json`) and a clean-agent
science review that inspected the calibration code; **no 14B data exists**. The frozen body above is left
unedited; the original named calls will be scored as written.

1. **Calibration criterion corrected.** The frozen design described the on-box auto-tuner as targeting the
   rank criterion; in fact `calibrate_effmags` targets **nats lifts (CAL_TARGET_NATS = 4.0/7.0)** with rank
   only logged — and at 7B that criterion selected effmags **19/32**
   (`runs/qwen2.5-7b/results/calibration_trace.json`), far below both the norm-scaled 62/93 and the
   rank-tuned 124/140 the baseline capture actually used. Running the 14B collect with the default tuner
   would therefore measure a *criterion switch*, not scale. The 14B strengths will instead be chosen by
   the **baseline's criterion**: an on-box mini-sweep (explicit `--effmags` list, `--no-calibrate`) to
   arm-A own-concept rank ≤ 50 with clean fraction ≥ 0.70 × the model's own s0 baseline — the procedure
   that produced 7B's s124. The nats trace is recorded as diagnostics only.
2. **Baseline cell pinned.** P1 baseline = 7B **s124 char@full = 1.90 (seed sd 0.13)**; P2 = **s124
   dist@T12 = 2.51 (sd 0.04)**. The s140 cells (2.36 / 2.86) are reported but excluded as verdict
   baseline: s140's clean fraction (430/736 = 0.584 = 0.65 × the s0 baseline 0.893) fails validity gate
   1's capability floor. The 14B primary cell is likewise the strongest strength passing both gates.
3. **Verdict bands instantiated** (rules unchanged): CLOSING: 14B char@full < 0.95 and the drop > 2σ;
   PLATEAU: within [0.95, 2.84] and above the permutation-null band; GROWING: > 2.84 and the rise > 2σ.
4. **Gate 2 replaced.** The frozen |bits| < 0.1 two-sided control gate is already violated by a known,
   benign pipeline mode: the nested-CV capacity grid has no predict-the-prior arm, so under a true null it
   can miscalibrate *downward* (7B s0 dist@12 = −0.302 ± 0.077; same mode visible in published natural-7B
   `emb` cells). New gate: s0 cells must read **≤ +0.1 bits (one-sided)**, and a ≥ 20-shuffle
   label-permutation null on the s0 pool must bracket the observed s0 value (confirming the negative
   reading is decoder miscalibration, not leakage — check: `analysis/perm_null_check.py`). Any **positive**
   verdict cell must exceed its permutation-null 95th percentile.
5. **Named calls scored as originally written.** Under the pinned baseline, call 1's 0.4–1.2 band lies in
   CLOSING territory; if 14B lands there, call 1 is scored **wrong** and the verdict follows rule 3.
6. **Language.** "Nameability-matched" is downgraded to **"criterion-passing (gate-calibrated)"**
   throughout: the strong-arm ranks are 4 / 20 / 47 at 1.5B / 3B / 7B — gated (≤ 50), not equated — and
   the residual mismatch runs *conservative* for a non-closure finding (the biggest model, measured at its
   weakest criterion-passing instantiation, still leaks ≈ 1.5B-level bits).
7. **Context correction the baseline forced.** The gate-passing scale curve is **non-monotone, not
   declining**: dist@T12 2.64 → 2.24 → 2.51 and char@full 2.06 → 0.72 → 1.90 across 1.5B / 3B / 7B (3B
   from the published s60 cells, rank 20). The published 7B collapse (0.89 / 0.77) is a dose artifact of
   the norm-scaled s93 bundle (its rank was 68 — under the criterion); the 1.5B→3B char dip persists at
   criterion-passing dose and is NOT explained away. The 14B point is collected against this corrected
   picture.

### Errata to Amendment 1 (pre-launch, verification review; no 14B data exists)

- **§5 precision:** call 1's 0.4–1.2 band *overlaps* CLOSING below 0.95; a 14B P1 result in [0.95, 1.2]
  scores call 1 **right** (PLATEAU). Only a result in [0.4, 0.95) scores it wrong-in-CLOSING-territory.
- **§4 null pools:** the s0 gate uses the s0 pool's permutation null; each **positive verdict cell** must
  exceed a permutation null computed on **its own strength pool** (`perm_null_check.py --slug qwen2.5-14b
  --strength <lvl>`).
- **§1 diagnostics:** the nats trace exists because the *smoke* runs with default (nats) calibration; the
  sweep and full runs use explicit `--effmags` + `--no-calibrate`.
- **Cell selection:** `check_injection.py` prints a verdict against the *absolute* 0.70 clean floor; the
  gate here is *relative* (≥ 0.70 × own s0 clean). Apply the relative rule from the raw clean fractions
  when selecting the 14B primary cell (they diverge if 14B's s0 clean ≠ ~1.0). Sweep-time clean estimates
  ran high vs full-run clean at 7B — choose the strong point with margin; if the strong point fails gates
  at full-run data, fall back to the next strength; if none passes, "under-injected, no verdict" (frozen
  gate 1).
