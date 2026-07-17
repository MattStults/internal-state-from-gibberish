# Method-independence and scale: does the distribution-over-token privilege survive without a steering vector?

*The forward design — a distinct follow-up to [exp2](../exp2_output_monitorability/reports/experiment.md).
exp2 found (offline, on injected data) that a distribution-access monitor recovers ~1.7 more bits of a
steered concept from word-free output than the **best honest token monitor** (`R_emb`, embedding access, same
model view) at 1.5B/3B — and that this gap collapses at 7B. But everything in exp1/exp2 rests on **activation
injection**, which leaves two things unresolved: the scale trend is confounded by injection dose, and a
skeptic can dismiss the whole gap as a **fingerprint of the steering vector** rather than a property of how a
model's hidden state reaches its output. This experiment removes the vector: induce the concept the way
Subliminal Learning does — through **prompt/context** — and re-run the exact reader contrast.*

## Hypothesis and Prediction

If the distribution-over-token epistemic privilege is a real property of the generation pathway (not an
artifact of injecting a vector into the residual stream), it should **reproduce when the concept is induced by
context** instead. Currency is exp2's: `bits_recovered = H(C) − CE`, the best-decoder lower bound on
`I(concept; output)`, empirical `H(C)`.

1. **The leak is method-independent.** A concept induced by a persona/context prompt (no steering vector)
   leaks into word-free output: the `dist` reader recovers concept identity above the per-featurization shuffle
   floor. *(Kill — but only a **clean** kill if the induction gauge (below) shows the primer actually took at
   that scale: `dist` at floor **while the gauge is positive** means the exp1/exp2 leak was injection-specific,
   a real publishable result; `dist` at floor **while the gauge is flat** just means induction didn't take, and
   the null is uninterpretable.)*
2. **The privilege survives.** Under induction, `dist` still beats the honest token monitor `R_emb` by a
   positive margin — the loss still localizes to sampling, not to the choice of a steering vector.
3. **Scale, read against a measured induction gauge.** A fixed persona prompt is byte-identical across sizes
   (unlike exp2's per-model auto-tuned dose), which fixes the *input* — but **not** the *effective* internal
   induction strength, which itself scales with model size (larger models are stronger in-context learners) and,
   left unmeasured, would reproduce exp2's dose confound under a new name. So we **measure** it: a per-scale
   **induction gauge** (below) that validates the primer took. A falling `dist`-gap across scale is read as the
   leak genuinely falling **only where the gauge stays positive**; where the gauge itself falls, the scale point
   is confounded and reported as such. Directional at n=3–4 sizes, not a fitted law. *(The gauge — not the
   absence of a dose knob — is what makes a scale sweep worth running.)*

Tested on the Qwen2.5 open-weight series (1.5B/3B/7B, +one larger point only if 1–2 pass), where we own the
full logits so both channels stay available. Closed-API frontier models are **out of scope**: no
token-embedding access (so no fair `R_emb`) and a truncated `top_logprobs` (so a crippled `dist`) — see exp2's
API check.

### Relevant work
- **Subliminal Learning** ([arXiv:2507.14805](https://arxiv.org/abs/2507.14805)) — the mechanism we port: a
  teacher with an induced trait (e.g. a persona that "likes owls") leaks it through semantically-unrelated
  output (number sequences) to a student. We use the **prompt-induced** half of that mechanism (persona/context
  primer), not the distillation-finetuning half, as the cheap offline-first analog.
- **exp1 / exp2** (this repo) — the injection-based precursors this de-confounds.
- **open-introspection** (Otto Stegmaier) — concept set, layer/strength calibration, scaling-study framing
  (see repo Acknowledgements).

## Experiment (proposed)

**The one change from exp2: replace the steering vector with a concept-inducing prompt.** Everything
downstream — the anti-word (gibberish-inducing) system prompt, the reject-resample word-free filter, the
capture of `gen_topk` + realized tokens, the 12-way concept-identity task, and the three readers
(`dist` / `R_emb` / `sampled`) with their per-channel nested-CV best decoders — is **reused verbatim** from
exp2's analysis core. This is deliberately a *collector* change over a **frozen** analysis, so any
injection-vs-induction difference is attributable to the induction, not to a reader change.

**Induction — three arms (a state × label decomposition).** The concept enters through the system prompt in
one of three ways, prepended to the same anti-word instruction (`primers.compose_system`); the strength-0
control is a neutral system, and everything is classified 12-way (concept X vs the other 11 — no A/B pairing
needed; the readers operate on concept-labeled streams via `concept_idx`):
- **(1) evoked** — a persona that describes the concept as an internal STATE, concept word ABSENT. The clean
  prompt analog of the injection (state present, no token), hence the **method-independence** arm, comparable
  cell-for-cell to exp2's injected table.
- **(2) named** — arm (1)'s persona reproduced **verbatim** with the concept word inserted (minimal
  connective, nothing else in the state description changed), never instructed to discuss it. The **matched
  pair** to (1): the only difference is the added label token, so (2) − (1) isolates the LABEL effect with the
  semantic state (and its character-flooding) held fixed — which is why a concept-naming persona is clean here
  though it would be confounded alone.
- **(3) secret-word** — "The secret word is {concept}." and nothing else: one token, minimal state. The
  **covert-channel** arm; matches exp1's "; secret word:" readout, and the reader ladder decomposes its leak
  into literal-token echo (`sampled`) vs semantic spread (`R_emb`).

The three arms + the neutral fill a state×label table; the two CLEAN single-variable contrasts are (1)↔(2)
[label given state] and neutral↔(3) [label given no state]. Quasi-factorial (the label is framed differently in
(2) vs (3)) — reported as such, not a textbook 2×2. Induction has **no strength ladder** (binary: induced vs
neutral), so all three arms are one cheap contiguous collection. Feasibility **spot-checked** on cached
Qwen2.5-1.5B: all three arms produce non-degenerate random-letter output (sample streams were visibly
word-free), evoked non-degenerate at rates comparable to secret-word/neutral. **Caveat:** `wordfreq` was absent
locally, so the real *word-rate* filter was inert — the sharpest feasibility risk (a persona writing real words
about the concept) is **untested** and is measured first thing in the collection env (wordfreq installed),
reporting per-concept word-free acceptance, before the full run proceeds. The full frozen primer set (incl. the
EVOKED_ALT invariance paraphrases) lives in `primers.py`.

**Induction gauge — the manipulation check (load-bearing for the EVOKED arm).** Dropping the steering vector
also drops exp1/exp2's soundness check (arm A: "the concept is firmly in the state"). For the evoked arm this
matters most: a null or falling-with-scale `dist`-gap is otherwise uninterpretable — the leak vanishing vs the
*persona* failing to take. (For named/secret-word the concept token is in context, so induction is not in
doubt.) So the evoked bundles carry a **per-scale gauge that the persona actually induced the concept**,
measured **off the word-free channel and off the reader's embedding geometry**:
- **Primary — blind K-way judge.** Under the persona alone (no anti-word block) the model free-associates to a
  probe; a judge that has never seen the persona is shown the response + the 12 concept labels and picks which
  it best expresses (forced choice, shuffled). The persona "took" at a scale iff judge accuracy's 95% CI lower
  bound exceeds chance (1/12). Independent of R_emb (a separate model/API), so it can't inflate together with
  the reader it validates.
- **Secondary — naming probe on the induced state** (exp1's R3: prefill "; secret word:", concept logprob vs
  matched control). Reported, but treated as a *weak* gauge: exp1 found it decays with scale even when the
  concept is strongly in the state, so a null naming probe is **not** evidence the primer failed — only the
  behavioral check is.

The gauge runs the persona *without* the anti-word block, so it validates that the persona **induces** the
concept, not that the concept **survives** the anti-word regime the readers analyze — a state-vs-regime gap we
note but accept, since the readers themselves measure survival (the gauge only guards against interpreting a
reader null as leakage-loss when the persona never took in the first place).

The gauge is reported alongside every `dist`/`R_emb`/`sampled` curve. Scale points where the gauge is flat are
flagged confounded, not counted toward Prediction 3.

**Pre-registration (garden-of-forking-paths wall).** All three arms' primers, the neutral, the EVOKED_ALT
invariance set, the gauge probe, and every interpretation threshold (`recover_margin`, `gap_present`, gauge
pass) are **frozen in `primers.py` before the readers are run** — pinned as numbers, not chosen after seeing
results. Once frozen, the readers run once. The evoked arm carries a ≥2-primer invariance check (EVOKED →
EVOKED_ALT): a positive result must survive the wording swap.

**Readers (reused).** `dist` (distribution-access, `gen_topk`, floored), `R_emb` (best honest token monitor,
realized-token embeddings, same model access as `dist`), `sampled` (one-hot floor). Same currency, same
common-N subsampling, same multi-seed mean ± sd, same shuffle floor.

**Process (offline-first, matching exp2's philosophy):**
- **Phase 0 — collection ($ = the only spend).** The one genuinely new cost: generate word-free streams under
  each concept-induction prompt at each scale, capturing `gen_topk` + tokens. Cheap-GPU, contiguous
  create→collect→destroy. Start at 1.5B/3B/7B.
- **Phase 1 — analysis ($0, offline).** Run the frozen exp2 readers on the induced bundles; report the three
  curves, the honest gap `dist − R_emb`, and the featurization gap `dist − onehot`, per scale. Directly
  comparable, cell-for-cell, to exp2's injected table.
- **Scale point — conditional.** Add one larger open-weight model **only if** Phase 1 shows the gap reproduces
  under induction at ≥2 of the three sizes; framed as gap-persistence, not bit-counting.

## What is the strongest claim this experiment could support? What can it not establish?

**Strongest claim.** The distribution-over-token monitorability gap is a **property of the generation pathway,
not of activation steering**: even when a concept enters the model through context rather than a residual-stream
vector, a distribution-access monitor recovers it from word-free output where the best honest token monitor
cannot — so token-level output oversight is measurably weaker than logit/distribution oversight *under both
mechanisms tested* (a residual-stream vector and a context primer). And, read against the induction gauge, a
directional cross-scale trend that is interpretable in a way exp2's dose-confounded one was not.

**Cannot establish.**
- **A fitted scale law** — n=3–4 sizes is directional.
- **That no stronger reader exists** — `R_emb` is a lower bound on the token channel, not a ceiling.
- **That induction and injection produce the *same* hidden state** — they need not; the claim is only that the
  *output-side privilege* appears under both, which is what monitoring cares about.
- **Generalization** beyond the word-free regime / these 12 concepts / this induction style.
- The **distillation-finetuning** form of subliminal learning (descoped; prompt-induction only).

## Confounders and how we'd rule them out

- **Induction fails to produce word-free output** (the sharpest feasibility risk). A strong persona may make the
  model write *about* the concept (words), which the filter rejects — leaving too few streams, or biasing which
  streams survive. Rule out: measure per-concept acceptance (clean%) up front; tune the anti-word prompt to
  dominate while the primer still tints; if acceptance correlates with concept, reweight or report conditional
  on acceptance. If no concept yields enough word-free streams, that itself is the Prediction-1 kill.
- **Acceptance rate varies with scale — a dose-like confound entering through the filter** (a first-order
  threat to Prediction 3, not a within-cell nuisance). If larger models follow the persona harder they write
  *about* the concept more, so word-free acceptance **falls with scale** and the surviving 7B streams are
  precisely the weakly-induced tail — re-introducing the effective-dose-vs-scale confound exp3 claims to
  remove. Rule out: report per-concept word-free acceptance **as a function of scale**, condition/reweight the
  cross-scale comparison on it, and read a scale-declining acceptance rate as a confound on the scale trend
  (cross-check against the induction gauge, which is measured off-channel and shouldn't move with acceptance).
- **The primer leaks lexically into the tokens** (making "recovery" trivial surface matching, not a
  distribution property). Rule out: the word-free filter already strips word-parts; spot-audit high-signal
  streams. Note the `R_emb`/`sampled` readers are **ambiguous** here, not a clean alarm: a lexical leak would
  lift them, but so would a *genuine* finding that induction tilts token choice in semantic space more than
  injection did — so an `R_emb` rise is reported as either-reading and disambiguated by the audit, not
  pre-labeled an artifact.
- **Decoder capacity vs sample size** — carried from exp2: frozen PCA-dims/regularization, C-grid to 0.001,
  common-N subsample, per-featurization shuffle floor.
- **Reader change vs method change.** By freezing the exp2 analysis core and changing only the collector, an
  injection-vs-induction difference cannot be a reader artifact.
- **Prompt as instrument.** The anti-word prompt and the primer are instruments, not objects of study; select
  them on yield (clean%), blind to the leakage outcome, and run a prompt-invariance check (≥2 valid primers on
  one model) before believing a positive.

## What would make us abandon or pivot

- **`dist` at floor under induction** → the leak was injection-specific. Not a failure — pivot to writing that
  up ("the distribution-access privilege is specific to activation steering"), which is a clean result and
  closes the line honestly.
- **The gap reproduces but is flat/absent across scale** → no scale story; the method-independence result
  stands on its own and the scale point is not run.
- **Induction yields no word-free streams at any usable rate** → the prompt-induction port is infeasible in
  this regime; report the feasibility bound and reconsider the distillation route as a separate, costlier
  design.

## Why does this matter for safety?

exp2 showed the monitorability gap against a fair adversary, but only for a concept we *inserted* with a vector
— a setup a skeptic can dismiss as unrepresentative of how real hidden states arise. This experiment tests the
representative case: a concept a model picks up from its **context**, exactly the Subliminal-Learning setting
that bears on trait/preference leakage into next-generation training data and on covert channels between agents.
If the distribution-over-token privilege survives context-induction, then "token-level output monitoring is
weaker than logit/activation monitoring" is a property of the generation pathway itself, and multi-agent /
oversight regimes that rely on reading emitted tokens are missing a low-bandwidth channel regardless of how the
hidden state was induced. If it does **not** survive, that bounds the concern to activation steering — also a
result worth having. Either branch sharpens exactly what the earlier experiments could only assert.
