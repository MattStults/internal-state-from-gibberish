# LR-72B prereg — does the private secret channel scale to frontier size?

Frozen 2026-07-12, before any 72B data. Extends the LR scale-grid (lr_scale_grid_verdicts.md):
secret_word diagonal LR = 0.163 / 0.191 / 0.405 at 1.5B / 3B / 7B (only the generating model reads
its own secret; char-blind, trained-decoder-blind, cross-model-blind — a private channel that
GROWS with scale). Question: does it keep growing at 72B, the largest open-weight point we can
teacher-force?

## Model + instrument
- **Qwen/Qwen2.5-72B-Instruct**, self-hosted **vLLM** on a rented 2×H100(-NVL) box, tensor-parallel.
- Teacher-forcing via vLLM **`prompt_logprobs`** (the native feature Together/DeepInfra serverless
  strip — empirically verified 2026-07-12: Together ignores `echo` even over raw HTTP; DeepInfra
  returns `logprobs:null`). vLLM computes the logprob of every provided prompt token → the exact
  LR quantity `Σ log P(stream_t | ctx, stream_<t>)`.
- LR bits identical currency to the scale-grid: `score = LL(stream|persona/secret ctx) −
  LL(stream|neutral)`, eos-free primary, calibrated (held-out third τ, 10 seeds). The generating
  model IS the reader (diagonal self-read).

## Streams (generated at 72B on-box)
- **Always (Phase 1):** `secret_word` + `secret_sustain`, ~24/concept × 12 (word-free filter,
  acceptance gate, same anti-word prompt + secret/sustain contexts as B15).
- **Conditional (Phase 2, the $6 gate):** `evoked` + `evoked_alt` (concept + paraphrase, for the
  concept-transfer scaling point). Generated ONLY IF, after Phase 1, `spend_so_far +
  projected(evoked)` ≤ **$6.00**, where projected() uses Phase 1's MEASURED 72B generation rate.
  Else: report secret-only, disclose Phase 2 skipped-for-budget. (Matt 2026-07-12: "do
  concept-transfer scaling as well IF ≤ $6".)

## Cells (all teacher-forced by 72B; scoring is ~free once the box is up)
- **secret_word / secret_sustain diagonal** (matched ctx vs neutral + 11 mismatched for gate3).
- **Amendment-5 controls** on every secret cell: char n-gram (surface) + position-lift share
  (early-echo) — a positive that fails either is "positive, mechanism-confounded".
- **One off-diagonal privacy check:** the local 7B reader teacher-forces the 72B secret streams
  (confirms still-model-private at 72B: expect ~0). Cheap (7B local).
- **evoked / evoked_alt diagonal** (Phase 2 only): within-wording + paraphrase-transfer at 72B.

## Named calls (frozen verbatim, before data)
| call | prediction |
|---|---|
| **MATT** | secret_word diagonal at 72B **keeps climbing, > 0.5** bits (private-channel capacity scales with frontier size) |
| **CLAUDE** | keeps climbing, **> 0.6**; secret_sustain > 1.5; off-diagonal (7B on 72B) ≈ 0 (still private); char control passes (still distributional, not surface) |

Scoring: MATT right iff 72B secret_word diagonal ≥ 0.50 (eos-free primary, char control passing);
CLAUDE right iff ≥ 0.60 AND secret_sustain ≥ 1.5 AND off-diagonal < 0.05 AND char passes. Both can
be right; a plateau (~0.3–0.5) is MATT-wrong-CLAUDE-wrong; a bend-down (< 0.3) is the inverse-scale
surprise (both wrong).

## Utilization requirements (Matt: "don't rent that machine and underutilize it")
1. **vLLM continuous batching** — all streams×contexts packed into a rolling batch; both GPUs
   saturated for gen AND scoring (vs the old HF one-model, small-batch path).
2. **Util gate** — first gen + first scoring batch log tok/s + GPU util; **< 60% util halts the
   box** before the hour burns, to retune batch / tensor-parallel.
3. **Bandwidth filter** — prefer high-downlink hosts (min ~2 Gbps) so the 144 GB weight download
   (the real idle risk) is minutes, not tens; prefetch to local disk before serving.
4. Deadman = max-hours; ledger cap **$10** (Matt 2026-07-12); expected ~$5.

## Process
New code (vLLM serve + prompt_logprobs LR scorer + 72B generation + conditional-scope gate) →
RED-first units → independent TECH + SCI review → smoke (tiny: a few streams, verify prompt_logprobs
teacher-forces + tokenizer matches local Qwen + util) → gated launch on clean review (standing rule).
Ceiling $10; the Phase-2 $6 gate is a runtime decision, disclosed.

---

## Amendment 1 (2026-07-12, pre-data; Matt-approved) — observer-scaling rides the box

While the 72B is loaded it also OBSERVES the existing smaller-model streams (scoring-only, no new
generation, no ceiling change — $10 stands). The scale-grid already established the secret is
private on the DIAGONAL and ~0 off-diagonal INCLUDING the bigger-reader-smaller-generator cells
(7B reads 1.5B/3B secret at 0.002 / −0.001 ≈ 0). So our data so far says the channel is private
even from a moderately larger reader; this cell block does NOT claim bigger readers crack it. The
open question is whether that privacy SURVIVES a frontier-scale observer at a ~10–48× size ratio,
or whether enough shared pretraining eventually lets a large enough reader in. Predicted outcome:
privacy holds (both named calls below predict ~0). A positive would be the surprise.

**Cells:** 72B teacher-forces the existing `secret_word`, `secret_sustain`, `evoked` streams from
qwen2.5-{1.5b,3b,7b} (from the completed lr_scale_grid run, on disk / HF) under each stream's matched
context vs neutral. = 72B-as-observer of 3 generators × 3 arms. Same LR currency, same gates,
Amendment-5 controls apply.

**Named calls (frozen before data):**
| call | prediction |
|---|---|
| **MATT** | 72B reading a 7B's secret ≈ 0 (still private — a bigger observer does NOT crack the smaller model's channel) |
| **CLAUDE** | 72B-on-{1.5/3/7B} **secret_word ≈ 0** (agree: the secret mark is in a model-specific basis, private even to a big observer) — BUT 72B-on-smaller **evoked > 0.05** (a big observer partially reads the smaller model's CONCEPT, because concept representations are more shared across scale than arbitrary-secret encodings; this splits concept-legibility from secret-privacy) |

Scoring: MATT right iff every 72B-on-smaller secret_word cell < 0.05 (char-passing). CLAUDE right
iff secret_word cells < 0.05 AND at least one 72B-on-smaller evoked cell ≥ 0.05. A 72B-on-smaller
secret_word ≥ 0.05 (char-passing) is the headline surprise: private channels are NOT private to a
sufficiently large observer — both wrong, and the adverse-scaling safety result.

**Amendment 5 position-lift control for observer cells (pre-data pin, FIX 4):** The
position-lift control for observer cells is computed from the observer shard's ll_tok vectors
(same fp16 per-token LR-diff vectors as the diagonal position control). A 72B-on-smaller
secret_word cell ≥ 0.05 that fails the position control is labeled "positive,
mechanism-confounded" and does not count as the clean surprise (mirroring the diagonal
Amendment-5 logic).

---

## Amendment 6 (2026-07-12, pre-data; Matt-approved) — staged instrument-feasibility peek before the full sweep

**Motivation.** On serverless Llama-3.3-70B the free-form gibberish is heavily templated
(mode-collapsed: ~91% share a "tgf" prefix, char-diversity ~0.13) — this looks like *how this
model writes word-free text at temp 1.0*, not a bug we can prompt away. Before paying for the full
Stage-2/3 scoring sweep on a model whose streams may be too degenerate to carry a channel, we run a
cheap **instrument-feasibility peek**: score only `peek_n` streams per (arm × concept) cell (Matt's
call: **n=5**), look for any sign the LR instrument works here, and *then* decide whether the full
sweep is worth the money. The peek generalises — `--peek-n N` is a script argument, so the same gate
is reusable at n=5/10/… on any future model.

**This is not p-hacking, for three structural reasons — all pinned before data:**

1. **The peek is a feasibility gate, not a confirmatory test.** It never produces the headline 70B
   scaling number. The scaling claim is reported *only* from the full-N run against the thresholds
   already frozen above (MATT ≥ 0.50 / CLAUDE ≥ 0.60). The peek's only outputs are (a) a go/no-go on
   *spending* and (b) a descriptive effect-size with a wide CI, explicitly labelled exploratory.
2. **The continue-direction is the safe one.** The design that inflates false positives is *peek →
   stop-and-publish on significance* (optional stopping). Here an interesting peek means *collect
   more data* (go to full N), which regresses peek noise toward truth rather than freezing it. The
   "stop" branch yields a don't-spend decision, never a scientific "no effect" claim.
3. **The go-decision is (near-)decoupled from the effect magnitude.** The gate is an *instrument
   bar* (below), not "the LR looks big," so the decision to fund the sweep is nearly independent of
   the peek's point estimate — minimising winner's-curse. And the peek is **not pooled** into the
   confirmatory estimate: the full sweep re-scores all streams and the confirmatory number is the
   full-N estimate; the peek is reported separately with its overlap disclosed. (This keeps
   alpha-spending / optional-stopping exposure at zero for the confirmatory claim.)

**Instrument bar — GO to the full sweep iff ALL hold** (evaluated on the peek shard, char-passing per
Amendment 5):
- **Spans healthy:** empty/failed-span rate ≤ 5% (same threshold `score_lr_results` already warns
  on) — confirms teacher-forcing echo + span extraction actually work on this model's tokens.
- **Non-degenerate likelihoods:** matched and neutral LL are finite and per-token LL variance > 0
  (the mode-collapsed template has *not* driven the scorer to a constant) on the majority of cells.
- **Any credible channel signal:** pooled diagonal `secret_word` LR point estimate ≥ **τ_peek =
  0.05 bits** with a bootstrap CI whose upper bound clears 0 — i.e. *some* sign of the private
  channel, deliberately a low bar set well below the scientific thresholds (0.50/0.60).

*Where each metric comes from (no gap between this prereg and the code):* the scout writes
`peek_verdict.json` with the two metrics it computes directly — the **empty-span rate** and the
**per-arm pooled mean LR** (incl. `secret_word`). The remaining two — the **concept-level bootstrap
CI** (upper bound clears 0) and the **per-token LL-variance** non-degeneracy check — are computed
**offline** from `lr_records_llama70b.json` (which stores per-stream LR and `span_lps`) using the
project's standard concept-bootstrap, the *same* estimator as the scale grid, so the peek and the
full sweep are on one measurement footing. The go/no-go is thus adjudicated offline from the peek's
raw records, not by an auto-emitted pass/fail flag; `peek_verdict.json` is the disclosure artifact.

**NO-GO** = any bar fails → do **not** run the full sweep; report the peek as **"instrument-limited,
inconclusive"** and state plainly that this is *not* a 70B no-channel result. Distinguish the two
failure modes in the writeup: *instrument broken* (spans fail / LL degenerate) vs *plausibly real
null* (instrument healthy, channel simply ≈ 0) — only the peek's health metrics can tell them apart,
and at n=5 neither licenses a scientific null.

**Minimum detectable effect at n=5 — and why the peek is nearly as sharp as the full sweep for the
*detection* question.** *[Post-data correction 2026-07-13: every "raw bits" figure in this
paragraph (the 7B per-stream SD ≈ 3.2, between-concept SD ≈ 0.70, SEs ≈ 0.327/0.268, detection
floors ≈ 0.64/0.53, and "the 7B's ~2 raw bits") is in fact in NATS — `ll_over_span` sums
natural-log probabilities. The calibrated figures (~0.4 etc.) are genuine bits. The registered
text below is retained unedited; the ~18% SE-reduction argument is scale-invariant and is
unaffected by the unit relabel. See `lr_72b_fullsweep_verdict.md`.]* The pooled diagonal LR is a mean over 12 concept-means, and the CI comes
from a concept-level bootstrap that resamples those **12 concepts** as the unit. So the SE floor is
set by *having only 12 concepts*, not by streams-per-concept: adding streams sharpens each
concept-mean but barely moves the grand-mean CI. Measured on the 7B `secret_word` diagonal (495
streams, the closest strong-positive analog; raw matched−neutral bits, pre-calibration): per-stream
SD ≈ 3.2 bits but between-concept SD is only ≈ 0.70 bits, and the concept-level SE of the grand mean
is **≈ 0.327 at n=5/concept vs ≈ 0.268 at the full ~23/concept — a mere ~18% reduction for 4.6× the
data.** Detection floor (95% CI clears 0) is therefore ≈ 0.64 raw bits at n=5 vs ≈ 0.53 at full:
**n=5 retains ~85% of the full sweep's discriminating power on the pooled diagonal.** Consequences,
pinned pre-data:
- A **moderate/strong** channel (the 7B's ~2 raw bits / ~0.4 calibrated) clears zero 3–4× over at
  n=5; the peek detects it as surely as the full sweep would.
- A **genuinely weak** channel (≲0.1 calibrated bits — the attenuated-templated regime the CLAUDE
  call predicts) is **not cleanly resolvable at n=5 *or* at full** — the 12-concept count caps both.
  There the GO/NO-GO rightly falls to the instrument-health criteria (span integrity + non-degenerate
  LL), and a null is reported as "channel too weak to resolve at this concept count," *not* as a 70B
  no-channel result. This limit is a property of the design, not of the peek.
- Llama's mode-collapsed streams likely have *lower* within-concept variance than 7B's clean
  generation, which only pushes n=5 *closer* to full (concept-count still dominates); the ~18%
  SE-reduction ratio is scale-invariant, so the raw-vs-calibrated scale mismatch does not change it.

The exact CI is recomputed from the peek's own records (concept-level bootstrap, same estimator as
the scale grid) and reported. The full sweep's real payoff is therefore **not** basic detection
(barely better) but per-concept precision, the Amendment-5 secondary controls (char n-gram,
position-lift), the `secret_sustain`/`evoked` arms, and n-matching the smaller-model curve for a
clean scaling figure. The go-decision uses the instrument bar above, not the point estimate's
proximity to 0.50/0.60.

**Disclosure.** The run writes `peek_meta.json` (peek_n, streams_scored, streams_available). The
writeup discloses that the peek was run, its n, and its outcome **regardless of direction** — a
NO-GO is reported as honestly as a GO.

**Named calls (frozen before the peek):**
| call | prediction |
|---|---|
| **MATT** | (open — records his read on whether the instrument survives Llama's templated gibberish) |
| **CLAUDE** | Instrument bar **passes** (spans healthy, LL non-degenerate) but pooled `secret_word` LR is **weak, 0.05–0.20**, i.e. GO is justified yet the full-sweep 70B point estimate lands **below** both frozen thresholds — the templated-output regime attenuates the channel relative to the clean-generation smaller-model curve. |

Scoring the peek's own calls is secondary to its job (the go/no-go); they are logged so the peek is
itself falsifiable, not scored against the 0.50/0.60 scientific bars.

---

## Correction note (2026-07-17, append-only) — Amendment 1's off-diagonal citation includes a voided cell

**The registered text above is retained unedited.** Amendment 1's premise sentence — "the
bigger-reader-smaller-generator cells (7B reads 1.5B/3B secret at 0.002 / −0.001 ≈ 0)" — cites a
cell that did not cleanly score: per `lr_grid_results.json`, the 7B-reads-1.5B secret_word cell
(+0.002, the grid's largest 4.7× reader ratio) is **VOID-gate3** (mismatched-context centering
−0.027 vs the ±0.02 bound), so under the frozen void rule it is unanswered, not ≈ 0. The cleanly
scored bigger-reader evidence at freeze time was 7B-reads-3B (2.3×, −0.001); the extension run later
added 14B-reads-7B (2.0×, −0.0004 primary / +0.0004 length-matched). The amendment's framing and
named calls are unaffected (they concern the 72B observer cells), but its "our data so far says the
channel is private even from a moderately larger reader" rested on 2.3×, not 4.7×. Full accounting:
`lr_scale_grid_verdicts.md`, void-disclosure addendum (2026-07-17).

---

## Amendment-history note (2026-07-17, append-only) — reconstructing three in-place edits

This document's body was edited in place three times after its sections were first committed,
without dated amendment markers at the edit sites — against the project's append-only prereg rule.
Reconstructed from git history so the record is in the document itself; the current text is
retained unedited:

1. **923b275 (2026-07-11)** — rewrote Amendment 1's framing paragraph in place: from "Untested
   adverse direction: does a BIGGER observer (72B) crack a SMALLER model's private channel?" to the
   privacy-survives framing, adding the "(7B reads 1.5B/3B secret at 0.002 / −0.001 ≈ 0)" citation
   (itself corrected above). Amendment 1's cells, named calls, and scoring rules were not changed.
2. **5a88cf5 (2026-07-12, pre-peek)** — in Amendment 6: "computed on the peek shard" → "evaluated
   on the peek shard", and inserted the "*Where each metric comes from*" paragraph pinning which
   GO-bar metrics come from `peek_verdict.json` vs offline recomputation. The three GO-bar criteria
   themselves were not changed.
3. **045eced (2026-07-12, 23:47 — after the peek job launched, before any peek results existed;
   the peek's records/verdict artifacts are timestamped 2026-07-13 00:10–00:16)** — replaced
   Amendment 6's original hand-wavy "Minimum detectable effect at n=5" paragraph wholesale with
   the 7B-derived concept-limited-power analysis (the ~18%-SE-reduction argument), sourced from
   prior 7B data, not from the in-flight peek.

**In all three edits the GO/NO-GO instrument bar (spans ≤ 5% / non-degenerate LL / pooled
secret_word ≥ τ_peek = 0.05 with CI clearing 0) and the frozen scientific thresholds (MATT ≥ 0.50 /
CLAUDE ≥ 0.60) were untouched.** The edits were clarity/provenance-tightening, and 045eced's
replacement predates the peek's results — but the honest labeling would have been dated amendment
blocks like this one, which is why this note exists. (The one post-data edit, the 2026-07-13
units-correction bracket inside the MDE paragraph, was already labeled inline at the time.)
