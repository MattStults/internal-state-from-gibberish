# Output monitorability of a steered concept: predicting the token budget to recover it, and how that scales

*The forward design — the proposed follow-up to [the experiment](../exp1_epistemic_privilege/reports/experiment.md). That experiment found that a
surface reader recovers an injected concept from word-free output (weakening with scale) while the
"introspective naming" metric is null once de-confounded — a result about **readers**, not about whether the
concept is present. This turns that into a quantitative, predict-then-demonstrate question: how many output
tokens does recovery cost, can we predict it, and how does the cost scale.*

## Hypothesis and Prediction

An injected concept is recoverable from a model's word-free output, but **recovery has a measurable token
cost that is predictable and reader-relative**: the per-token signal a *sampled-token* reader sees is small —
the concept lives in the output *distribution* and the state — so recovery takes many tokens, and that budget
is forecastable. Falsifiable:
1. **The budget is predictable.** The number of output tokens N to recover the concept to a fixed accuracy is
   predictable from the measured per-token distribution shift, and verified by drawing ~N tokens and checking
   recovery crosses threshold there (the headline — a predict-then-demonstrate loop).
2. **The loss is at sampling, not propagation.** A reader with logit-*distribution* access recovers the
   concept where a *sampled-token* reader needs far more tokens — the signal survives to the output
   distribution but is lost when it is sampled into discrete tokens.
3. **The budget grows with scale.** N rises with model size — claimable only at *matched dose* (below); at
   n=3 sizes it is directional, not a fitted law.

Tested across the Qwen2.5 (and, once in-regime, Qwen3) scale series.

### Relevant work
- **Softmax bottleneck / low-rank logits** — outputs lie in the `d`-dim image of the unembedding `W_U`;
  confirmed empirically *and tested on nonsense prompts* (our regime): Golowich, Liu & Shetty
  ([arXiv:2510.24966](https://arxiv.org/abs/2510.24966), ICLR 2026 Oral).
- **Inversion from outputs** — PILS losslessly linear-compresses next-token distributions to invert hidden
  prompts ([arXiv:2506.17090](https://arxiv.org/abs/2506.17090)); Carlini et al. recover the unembedding from
  black-box logits ([arXiv:2403.06634](https://arxiv.org/abs/2403.06634), ICML 2024 Best Paper). No verified
  source performs the direct *steering-vector*-from-logits inverse — a gap this work sits in.
- **Reader/prompt-relativity** — multi-prompt evaluation and prompt sensitivity (PromptEval
  [arXiv:2405.17202](https://arxiv.org/abs/2405.17202); Sclar et al. FormatSpread; Mizrahi et al.).
- **Precursor** — the introspection experiment ([experiment.md](../exp1_epistemic_privilege/reports/experiment.md), this repo); Subliminal Learning
  ([arXiv:2507.14805](https://arxiv.org/abs/2507.14805)).

## Experiment (proposed)

Same apparatus as the experiment (inject a concept vector at the introspection layer; emit word-free streams under the
anti-word prompt; concept-matched control throughout). The change is in the *measurement*: a token-budget law
with a predict-then-verify loop, a reader ladder to set the budget's currency, and a writing-vs-reading split.

**Token-budget law (the headline).**
- per reader, per scale: the **recovery curve** — recovery vs **total emitted tokens** T (= number of streams
  × tokens/stream; the grid cut positions give the length axis, subsampling streams gives the count axis).
  The free de-risk already shows R1's cost grows with scale: ~8 → 12 → 14 streams/concept to reach 0.70 at
  1.5/3/7B.
- **Predict T** from the per-token distribution shift (information leaked per token), then **demonstrate** by
  drawing ~T tokens and checking recovery crosses threshold there. *Within a model this runs offline on
  existing data with no dose issue — the cleanest predict-then-verify.* Across scale (predict 7B's budget from
  the smaller models) it is the stretch, and is only defensible at matched dose.
- **Bits-ladder:** run **1/2/3-bit recovery tasks** — 2/4/8-way groupings of the 12 concepts (the full set is
  log₂12 ≈ 3.58 bits; a 4-bit/16-way task needs a larger concept set, i.e. new collection). If per-bit cost is
  additive, the budget extrapolates to recovering **more bits of concept identity** — *which* concept among a
  larger set, **not** the steering vector itself (that nonlinear inverse is descoped): `T(n bits) ≈ n × (cost
  per bit)`. Additivity is **not assumed** — validated by checking 2-bit cost ≈ 2 × 1-bit on held-out concept
  pairs first (the 12 concepts are correlated).

**Reader ladder** — each reader is a *lower bound* on recoverability, in bits; the budget's currency is the
best-decoder lower bound, **not** a true ceiling (no upper bound on `I(concept; output)` is available at this
sample size — see Cannot establish):
- **R1 surface** — character n-gram histogram (token-access, weakest).
- **R_emb** — token-embedding / `W_U`-projection of the emitted tokens (does the concept bias token *choice*
  in semantic space, where a symbol-counter is blind). The rung the experiment was missing.
- **R3 prefill** — feed the stream through the model, read the concept logprob (full-model, but a *fixed*
  readout direction; a constrained lower bound — retained from the experiment).
- **R_learn** — a learned, cross-validated decoder on the model's read (optimal readout direction; the
  tightest practical lower bound). `bits = log₂K − cross-entropy` of this reader is the budget's currency.

**Writing vs reading split** — *where* is the concept lost on the way to output:
- **Propagation:** `‖Δh_final‖ / strength` — does the layer-L injection survive to the output layer?
  (`Δh_final = h_final_A − h_final_B` per cut, from the saved final-layer residual.)
- **Post-softmax detectability:** `W_U` is full column rank, so `δz = W_U·Δh_final` carries the shift to the
  logits ~losslessly *algebraically* — so the real question is whether that shift is **statistically
  detectable** after the softmax over the ~150k vocab and the sampling step. Measured as a
  **distribution-access** reader vs a **sampled-token** reader (identical except for the sampling step). The
  gap *is* the writing/sampling loss — the clean discriminator for "writing vs reading." (We do **not** claim
  a separate algebraic-bottleneck loss; that is ~zero by construction.)

**Captures (already saved).** The collector already persists the **final-layer residual `h_final`** (clean +
injected), the **injected vector `v̂`**, and the **per-step generation distribution** (`gen_topk`, top-K
next-token logprobs for accepted streams) — so the budget law, the reader ladder, propagation, and the
detectability split are **all computable offline on the existing bundles**, no new captures.

**Process (offline-first → matched-dose stretch):**
- **Phase 0 — offline, $0, existing Qwen2.5 data:** the reader ladder (incl. R_emb and R_learn on the saved
  top-K logits), the within-model recovery curve + predict-T-then-verify, the bits-ladder, propagation, and
  the distribution-vs-sampled detectability split. Tests reader-relativity, the prediction loop, and the cost
  story with no new runs.
- **Phase 1 — matched-dose clean scale series (the only new spend):** re-collect all models at a dose matched
  on arm-A nameability *and* residual-norm (the controlled-A/B method, already built), so the cross-scale
  budget forecast is defensible; add Qwen3 once in-regime. Gated on Phase 0 actually showing the prediction
  loop and the sampling gap.

**Expected.** The token budget is predictable from the per-token shift and verified by drawing it; the loss
localizes to sampling (a distribution-reader recovers where the token-reader needs far more tokens); the
bits-ladder is additive; and — at matched dose — the budget's scale-direction is upward.

## What is the strongest claim this experiment could support? What can it not establish?

**Strongest claim.** A predict-then-demonstrate law for output monitorability: for a given model and output
style, the number of output tokens needed to recover an injected concept is **predictable from the per-token
distribution shift and verified by drawing that budget**; the loss localizes to **sampling, not propagation**;
so **token-level output monitoring is weaker than logit/activation monitoring by a measurable margin** — and,
*at matched dose*, that margin's scale-direction.

**Cannot establish.**
- Model **predisposition/capability to *use*** the channel.
- That **no stronger reader exists** — the budget's currency is a best-decoder *lower bound* on
  `I(concept; output)`; no upper bound (true ceiling) is obtainable at this sample size.
- A **fitted scale law** — with n=3 sizes the cross-scale budget is directional; a law needs the matched-dose
  series (Phase 1) and more sizes.
- **Generalization** beyond the word-free regime / these concepts / output formats.
- The **layer-L steering vector itself** (vs `Δh_final`) — recovering it needs inverting the nonlinear
  L→final stack; a stretch goal, not a claim.

## Confounders and how we'd rule them out

- **Dose not held constant across scale** (the headline's sharpest threat). The existing series injects a
  different effect size per model (7B at `{0,62,93}` vs `{0,40,60}`), so a budget-vs-size claim conflates
  monitorability scaling with *how much we injected*. Rule out: run the budget law at **matched dose** — arm-A
  nameability **and** residual-norm matched (the controlled-A/B method) — and report whether the scale
  ordering survives both matchings. Until then the scale claim is directional only.
- **Reader/decoder capacity scales with sample size.** R_learn and the recovery curve compare across scales
  whose accepted-N, class balance, and (for Qwen3) tokenizer differ — capacity differences would inflate the
  tightest lower bound. Rule out: **freeze** PCA dims + regularization across all readers/scales, **subsample
  to a common N** per concept, report the per-featurization **shuffle floor** (already ~0.59). Tokenizer/vocab
  differences are an explicit confounder for any cross-family (Qwen2.5→Qwen3) bits-per-token comparison.
- **Writing vs reading loss** (is the signal absent from the tokens, or present-but-our-reader-is-weak?).
  Rule out: the **distribution-access** vs **sampled-token** reader gap (differ only by sampling), plus the
  best-decoder lower bound as the strongest reader we have.
- **Control choice** (the experiment's lesson): an un-injected control conflates concept signal with generic
  injection-induced word-suppression. Rule out: **concept-matched control** throughout.
- **Prompt as instrument / prompt-sensitivity.** The system prompt elicits the word-free regime; it is not the
  object of study. Select it on **yield (clean%)**, blind to the leakage outcome (so it's not p-hacking), and
  run a **prompt-invariance check** (does the result hold across ≥2 valid prompts on one model per family).
  Report baseline-clean as a covariate.
- **Bits non-uniformity and non-independence** (some concepts/directions leak more; concepts are correlated).
  Rule out: **average the bits-ladder over several concept sets / principled random directions**, and validate
  **additivity** empirically (2-bit ≈ 2 × 1-bit on held-out pairs) before extrapolating; else present the
  bits-ladder as within-range interpolation only.
- **Selection bias from the word-free filter** (does *which* stream survives correlate with the concept,
  biasing the readout?). Check per-concept acceptance rates; if acceptance correlates with concept, reweight
  or report the budget conditional on acceptance. The sharpest residual risk.

## What would make us abandon or pivot

- **The distribution-access reader *also* fails** → the leakage genuinely vanishes (propagation washout), and
  "monitoring is weaker than logit access" collapses. Pivot to characterizing the washout.
- **The budget is flat across scale at matched dose** → no scale story (the de-risk argues against this, but it
  is three points; more sizes needed). The within-model prediction loop still stands as the result.
- **The per-token signal does not predict T** → the readout is not information-limited (our readers are
  sub-optimal). Pivot to stronger readers.

## Why does this matter for safety?

Monitorability of steered or colluding model instances is **quantifiable and forecastable**: token-level
output oversight is *measurably* weaker than logit/activation oversight, and — at matched dose — the gap's
scale-direction is adverse, so for a fixed sample budget, output-only monitoring degrades as models grow. A
steered behavior can be present in the state and in the output distribution yet require a (scale-growing)
token budget to catch from emitted tokens. This bears directly on secret collusion between agents,
steganographic channels, and information passed around semantic filters into next-generation training data.
The actionable implication: safe multi-agent / oversight setups need either a **sample budget that scales with
model size**, or **logit/activation-level access** — output token monitoring alone is not enough, and the
predict-then-demonstrate loop here is a way to *size* how much is not enough.
