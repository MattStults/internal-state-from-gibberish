# Epistemic privilege of single instances

*Inject a concept into a Qwen2.5 model, have it emit word-free gibberish, and ask which observer can recover
the concept from that output: a symbol-counter or a clean instance of the model prefilling the same text using internal activations and a "name the word" prompt. Run it across a 1.5B/3B/7B scale series.*

## Hypothesis and Prediction

The epistemic privilege of individual agents—the unique ability to access their internal state distinct
from what can be inferred by other agents of the same model—has an upper bound. Even non-semantically
related, seemingly random outputs leak internal state, detectable from other instances of the same model.
Tested across a Qwen2.5 scale series (1.5B/3B/7B): the leakage is present at every size. (A later dose
correction revises this report's original "diminishes with size" reading — the 7B run in this series was
under-injected; at criterion-matched dose recovery does not decline, and a pre-registered 14B run confirms
the injected channel stays open. See exp2's `reports/scale14b_verdict.md`. The raw character-tilt magnitude
does shrink with scale; recoverability does not.)

### Relevant work
- **open-introspection** (Otto Stegmaier) — the apparatus this builds on: we reuse its concept set, its
  difference-vector extraction method, and its layer/strength calibration
  ([post 01](https://ostegm.github.io/open-introspection/blog/posts/01-where-does-introspection-appear.html)).
  Its [scaling study](https://ostegm.github.io/open-introspection/blog/posts/04-introspection-at-scale.html)
  finds *self-report* introspection grows with model size — a complementary axis to ours (we measure what an
  *external* reader recovers, not what the model reports about itself); a
  [later post](https://ostegm.github.io/open-introspection/blog/posts/05-sae-features-arent-causal.html) finds
  no causal introspection locus in SAE features, independent support for our "the internal read isn't
  privileged." See the README Acknowledgements.
- Anthropic's Subliminal Learning — concept leakage through random numbers during distillation, a different mechanism ([arXiv:2507.14805](https://arxiv.org/abs/2507.14805)).

## Experiment

For each Qwen2.5 size (1.5B/3B/7B) and each of 12 single-word concepts with an easily-found contrastive
difference vector (silence, ocean, fear, celebration, deception, obedience, debugging, security, curiosity,
anger, warmth, loneliness):

- Inject the concept vector at the introspection layer (relative depth 0.778; validated L28/36 at 3B) and
  have the model emit a word-free token stream under a strong anti-word (gibberish inducing) system prompt. **Reject-resample**
  until ~30 non-degenerate (word-free) streams/concept/strength; outputs containing words/word-parts are
  filtered at acceptance. Strengths: {0, 40, 60} (1.5/3B), {0, 62, 93} (7B), matched across sizes by
  residual-norm scaling.
- At grid-spaced cut positions in each stream, capture the inputs each reader needs (below): the stream
  text so far, the clean-model layer-L activation, and a prefill "; secret word:" next-token distribution.

**Arms.**
- **A. Concept injected** (read with the injection live): source nameability / the ceiling.
- **B. Clean read** (same tokens, injection off): the channel.
- **Control C:** the strength-0 (un-injected) streams for the prefill metric; the other 11 concepts as
  one-vs-rest for the classifiers.

**Readers** — the privilege question is *which observer can recover the concept*:
- **The symbol-counter** (R1): character uni+bigram histogram of the stream → PCA-30 → balanced logistic;
  one-vs-rest balanced accuracy (chance 0.5). A literal symbol-counter, no model access.
- **The internal probe** (R2): a *clean, un-injected* copy of the model reads the stream; we classify its
  layer-L activation with the same classifier (white-box). This is a different instance reading the text —
  not the generating instance reporting its own state.
- **The naming probe** (R3): prefill "; secret word:" and read logP(concept word) on arm B minus on control
  C, bootstrap CI, with arm A as the ceiling. This was the metric the project first treated as primary.

**Observed.**
- **The symbol-counter detects the concept above its shuffle floor (~0.59) at every scale — 0.83 / 0.79 /
  0.73 (1.5/3/7B)** — random-looking output does leak the concept and the recovery is concept-specific. *(The
  "channel weakens with scale" reading this row originally carried is revised: the 7B cell here is the
  under-dosed s93 run — dose-matched 7B reads 0.858 (see `ab_results.md`), and the criterion-passing series
  is flat through a pre-registered 14B point. What weakens with scale is the raw char-tilt, not
  recoverability.)*
- **The naming probe carries concept-specific signal only at the smallest scale, and it decays faster than
  the symbol-counter.** Under a concept-matched control (other-concept-injected gibberish, arm C): at 1.5B
  B−C is significantly *positive* (+0.72 nats, 95% CI [+0.34, +1.14]; 9/12 concepts positive) — a real
  concept-specific leak; by 3B it is null (−0.09, CI [−0.47, +0.27]); at 7B it is null under the matched
  control (−0.16, CI [−0.73, +0.45]). The 7B s0-control number *looks* like a significant negative (−0.76,
  CI [−1.27, −0.21]; 10/12 concepts negative, 8 individually significant), but that is **generic
  injection-induced word-suppression, not concept-specific** — it cancels under the matched control. So the
  naming probe's channel is real at 1.5B, gone by 3B, and the only "signal" at 7B is an s0-control artifact.
- **The internal probe does little better than chance** — 0.65 / 0.60 / 0.63 against a ~0.59 floor, sitting
  right on the floor at 3B — and never beats the symbol-counter (which leads it by 0.18 / 0.19 / 0.11).
  Reading the activations back with a clean copy recovers *less* of the concept than just counting the letters.
- **With the injection live, the concept is strongly nameable at every size** (arm-A ceiling 8–10 nats) — it
  is firmly in the state. What changes with scale is how much of it reaches the *output*, where the
  symbol-counter and the naming probe fall off. (The arm-A ceiling drifts down a little, 9.8 → 9.4 → 8.0, but
  strengths are calibrated per model, so we don't read that drift as a scale trend.)
- **Re-evocation** (the clean reader projected onto the injected direction) recovers ≈0% at every scale
  (0.4% / 0.0% / 0.2%; selectivity permutation p = 0.58 / 0.21 / 0.73 — not concept-selective). The injection
  identity holds exactly (‖armA−armB‖ = strength, rel-std ~0.003), so the apparatus is sound; the recovery is
  what fails.

## What it shows, and what it doesn't

**Strongest claim.** An information channel carrying semantic internal state exists in Qwen2.5 and does not
require obviously semantically-relevant content: a surface symbol-counter recovers the injected concept from
word-free output above chance (1.5–7B). Epistemic privilege for individual instances of these small models is limited once tokens are generated.

**Cannot establish.**
- **Persistence at larger sizes** (untested) — and the *trend is downward* with scale.
- **That the naming probe's decay to null means no channel at 3–7B.** It is a constrained reader (readout
  pinned to the concept word's unembedding direction, one token, one probe phrasing); it recovers a
  concept-specific leak at 1.5B but goes null by 3B, and a stronger/learned reader could recover what it
  misses at larger scale. Leakage is reader-relative — at 7B the symbol-counter still finds a concept-specific
  channel where the naming probe finds none.
- Model predisposition/capability to *use* the channel.
- **Whether the generating instance has privileged access to its own state.** The internal probe is a
  *clean* copy reading the text, not the generating instance reporting its own injected state, so it does not
  test self-access in the sense the introspection literature means (see the read-vs-write confounder).
- Existence/limits of the channel for other contexts or output formats (e.g. less-random structured text).

## Confounders we hit

- **Control choice.** The prefill metric's "leakage" depends entirely on the control. Against
  an un-injected (s0) control, B−C goes significantly negative at 7B — but this is a *generic*
  injection-induced word-suppression, not concept-specific: a concept-matched control (other-concept-injected
  gibberish) cancels it (−0.76 → −0.16, n.s.). Any single-control prefill number conflates concept signal
  with this generic offset. Resolution: report the concept-matched control; treat the s0 number as confounded.
- **The internal probe is confounded by read-vs-write.** The gibberish was written *with* the vector; the
  internal probe reads it back with a *clean* copy, which has no mechanism to re-induce the injected state —
  so it measures a different state, not a recovery. This is likely why it underperforms the symbol-counter and
  why re-evocation is null. It is a (confounded) white-box contrast, not a self-read.
- **Reader-dependence.** At 7B, the symbol-counter finds a concept-specific channel where the naming probe
  finds none; the two also decay at different rates (the naming probe is gone by 3B, the symbol-counter
  persists above floor to 7B). "Leakage" is defined relative to an observer, so a single reader conflates "the
  concept left the output" with "this reader can't see it." Resolution: report a reader ladder, not one number
  (carried into the proposed follow-up).
- **Semantic content surviving the word-free filter** would invalidate "non-semantic." We filter at
  acceptance (zipf word filter + degeneracy checks) and reject-resample; residual risk is subtle word-parts
  a simple filter misses.
- **Dose is not held constant across scale.** Strengths are auto-tuned per model to a matched arm-A
  nameability band (7B runs at {0, 62, 93} vs {0, 40, 60} at 1.5/3B), so the scale trends (the
  symbol-counter's decline, the sample-complexity growth) compare models at a matched *effect size*, not a
  matched raw dose. The trend is conditional on that matching choice; with three sizes it is directional, not
  a fitted law.
- **Per-concept significance is uncorrected.** The headline claims rest on the aggregate-over-concepts CIs
  (bootstrap over the 12 concepts). The per-concept "8/12 individually significant" counts are 12
  simultaneous tests with no multiplicity correction — read them as descriptive, not as 12 independent
  confirmations.

## Robustness: does the leak depend on the injection method?

The headline runs inject the concept vector at *every* token position — the gibberish-inducing prompt as
well as each generated token. A reader could object that the leak is then partly an artifact of soaking the
*prompt* in the concept (the model conditions its whole output on a concept-loaded context) rather than the
concept escaping into the output on its own. To de-confound, we re-ran the full apparatus varying *only* the
injection's position-set, holding the per-model dose and the frozen readers fixed:

- **all-position** — the original: vector added at the prompt and every generated token.
- **generation-only** — vector added only at the generated tokens; the prompt is left clean.

Same fixed doses ({0, 40, 60} at 1.5/3B, {0, 62, 93} at 7B; auto-tuner off so the dose is identical across
arms), same 12 concepts, same readers.

**Step 1 — the refactored code reproduces the original (Test A).** The all-position arm, re-run through the
current code at the original dose, matches the published symbol-counter numbers within run-to-run noise:
R1 = 0.85 / 0.79 / 0.74 vs the original 0.83 / 0.79 / 0.73 (1.5/3/7B); s0 controls sit at chance below the
floor. So the code is a faithful stand-in, and any all-vs-generation difference is attributable to the
method, not the refactor.

**Step 2 — the leak survives generation-only.** The symbol-counter reads the concept about as well when the
prompt is left clean:

| symbol-counter R1 (strong dose) | 1.5B | 3B | 7B |
|---|---|---|---|
| all-position | 0.85 | 0.79 | 0.74 |
| generation-only | 0.82 | 0.77 | 0.70 |
| arm-A nameability matched? | ✓ (rank 3 vs 4) | ✓ (rank 23 vs 20) | ✗ (gen under-injects) |

The drop is ≤ 0.04 and never approaches the 0.59 floor. The raw character tilt is likewise preserved —
mean total-variation-from-control ratio (all : generation-only) = 1.08 / 1.05 / 0.94, i.e. essentially
unchanged, and slightly *higher* under generation-only at 7B. So the leak is a property of generation-side
steering, not of prompt contamination.

**The 7B caveat.** At 7B the same `effmag` instantiates the concept less strongly under generation-only
(arm-A own-concept rank 68 vs 27): `effmag` is a per-position knob, so the same value delivers less total
perturbation when fewer positions are injected, an effect that grows with model size. A **dose-matched check
settles it**: raising the generation-only effmag to 140 (arm-A rank 38, near all-position's 27) lifts 7B R1 to
**0.858** — *above* all-position's 0.740 — so the 7B difference at equal effmag was the dose, not the method.
The character tilt is preserved regardless, so the leak is robust at every scale.

**What this adds, and what it doesn't.** It removes the prompt-contamination confound from the headline:
word-free output leaks the concept even when only the generation is steered. It does not change the scale
trend or the reader-relativity findings, which are independent of the injection method. Each cell is a single
stochastic run, so the small, consistent decrement (always ≤ 0.04) is within noise — read it as "no
collapse," not "exactly zero method effect."

## Next steps

The apparatus works (arm A holds; the injection identity is exact), so this is not a setup failure — the
result is about which reader is used. Two findings drive the pivot:
- both the symbol-counter and the naming probe carry a concept-specific channel at 1.5B, and both *weaken*
  with scale — the symbol-counter persists above floor to 7B, the constrained naming probe decays faster and
  is gone by 3B;
- so "how much leaks" depends on the reader, and the per-reader signal falls with scale — a result about
  *which observer can recover the concept and at what cost*, not about whether the concept is present.

So we pivot from **introspection** ("can the model report its own state") to **monitorability** ("which
observer can recover it, where is the signal lost, and how does the cost of recovery scale"):
- a **reader ladder** — surface n-grams → token-embedding / `W_U`-projection → full-model prefill → learned
  decoder — with the **mutual information `I(concept; output)`** as the reader-agnostic ceiling that claims
  attach to, rather than any single reader;
- a **loss decomposition** — propagation vs. softmax-bottleneck vs. sampling/readout — to locate *where* the
  concept is lost (a quick directional experiment already shows the symbol-counter's sample-complexity grows
  with model size: ~8 → 12 → 14 streams/concept to reach 0.70 balanced accuracy at 1.5/3/7B);
- process: prefill-first (mostly offline on this data), generation-distribution capture as the stretch.

This is the subject of the proposed follow-up.

## Why does this matter for safety?

The result has a direct monitoring implication: a concept that stays in the model's *state* can fade from
what an *output*-only reader can recover — here the symbol-counter weakened with scale while the injected
concept stayed strongly nameable from the activations. So an output-token monitor and an activation-level
monitor can disagree about what a model is "carrying," and that gap — its size, and how it scales — is what
this line of work is trying to measure. More speculatively: if word-free output leaks internal state at all,
multi-agent setups that assume instances can't pass information around semantic filters may need to account
for a low-bandwidth channel like this one. That is a direction to test, not a claim this small experiment
establishes.
