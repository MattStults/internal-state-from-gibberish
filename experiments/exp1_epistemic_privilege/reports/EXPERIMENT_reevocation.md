# Experiment: does the clean reader *re-evoke* the injected concept — or just carry classifiable letters?

**Status:** run on Qwen2.5 1.5B/3B/7B. Result: re-evocation ≈0% at every scale (0.4 / 0.0 / 0.2%), not
concept-selective (selectivity permutation p = 0.58 / 0.21 / 0.73) — the clean reader carries classifiable
*letters*, not the injected *direction*. See `runs/<slug>/results/reevocation_results.json`,
`analysis/analyze_reevocation.py`, and the "Re-evocation" finding in `reports/experiment.md`.
**One-line question:** when the clean, un-injected reader reads concept-*c* gibberish, does its internal
state move along the *injected concept direction* **v_c** — selectively for *c* — and how large is that
"re-evocation" relative to the live injection and to the lexical "name-the-word" channel?

---

## 1. Why this, and why it's different from everything we've measured

Everything so far measured **decodability**: can you tell which concept was injected from the output?
Answer: yes, but it's surface/character-bounded, a counter matches the model, and (distinctness test)
the activations add nothing beyond the character histogram *for classification*.

**Decodability is not re-evocation.** In this data the concept and its letter-signature are perfectly
confounded, so "activations add nothing beyond characters" cannot distinguish *"the activation is raw
letter counts"* from *"the activation is a genuine concept representation those letters evoked."* The
introspection-relevant question — does reading the gibberish make the reader **represent the concept** —
is therefore untouched by our classification tests. This experiment measures it directly by projecting
the reader's residual onto the **actual injected direction** (a *semantic* direction from "Tell me about
X."), not onto a letter statistic.

We are **not** claiming re-evocation would be letter-*independent* (the channel is public; the cause is
the letters). We are asking whether a concept-aligned representation forms at all, how selective it is,
and how big it is — the quantity that bears on introspection, and the one the saved steering primitives
exist to make measurable.

## 2. Hypotheses (pre-registered)

- **H_reevoke** — reading concept-*c* gibberish raises the clean reader's projection onto **v_c** above
  baseline, **selectively** (diagonal of the 12×12 cross-projection > off-diagonal), CI-separated.
- **H_null** — no selective re-evocation (diagonal ≈ off-diagonal ≈ random-direction floor).
- **H_magnitude** (the interesting one, and our prior) — even if re-evocation is present, it is **small**
  relative to (a) the live injection (arm A) and (b) the lexical "name-the-word" uplift (prefill B−C):
  *"the word communicates more than the experience."*

## 3. Inputs — mostly reuse; the GPU part is tiny

- **Reuse `covert_collect.pt`** (no regeneration): clean-reader (**arm B**) and source (**arm A**)
  layer-28 residuals at grid cuts {2,4,8,16,32,64,127}, 12 concepts × strengths {0,40,60}, ~30 clean
  streams each.
- **GPU (tiny, required):** re-derive the 12 injected vectors **v_c** with `K.concept_vector_blog(model,
  tok, concept, base, layer=C.G2_LAYER)` — deterministic blog method, fixed `C.BASELINE_WORDS`, layer
  28 — and **save** them plus the per-strength injected magnitude **α** (the `alpha` from `gen_clean`),
  the baseline list, and the layer, to `concept_vectors.pt`. Seconds of compute. (Derivation is
  deterministic, so it reproduces the originally-injected vectors up to bf16 noise ≪ ‖v‖; we couldn't
  bit-verify because the originals weren't saved — which is exactly why the rule now saves them.)

## 4. Measurements

Let r = the clean reader's (arm B) layer-28 residual at the endpoint cut of a stream; v̂_j = v_j/‖v_j‖.

1. **Cross-projection matrix** `M[i,j] = mean over concept-i streams of cos(r, v̂_j)`, **baseline-
   subtracted** (subtract the mean cos(·, v̂_j) over s0/un-injected streams). Diagonal `M[i,i]` =
   re-evocation of the injected concept; off-diagonal = cross-talk. **Primary metric is cosine**
   (scale-free, comparable across concepts; avoids α-estimation noise).
2. **Selectivity** — is `M[i,i]` the max of its row and column? **Permutation test** (shuffle the
   concept→stream assignment, recompute diagonal dominance) for significance; **bootstrap CI** on each
   diagonal entry (resample streams within concept).
3. **Magnitude as a fraction of the injection** — the *un-normalized* projection ⟨r, v̂_c⟩ minus
   baseline, divided by **α_c** (the injected magnitude). "Reading re-evokes X% of what we pushed in."
   Cross-check the denominator empirically: **arm A** (injected reader) minus baseline ≈ α (it should,
   since the hook adds α·v at this layer) — this both sanity-checks the projection machinery and gives an
   empirical "full-injection" reference.
4. **Per-token curve** — re-evocation (diagonal cosine) vs cut t: does it front-load/decay like the
   classifier, or build?
5. **Word vs experience** — relative uplift of re-evocation (cosine over baseline) vs relative uplift of
   the lexical channel (prefill probability ratio `exp(B−C)`), both as dimensionless relative uplifts
   (cross-quantity, flagged). Tests H_magnitude.

## 5. Controls (preempts)

- **Random-direction floor** — project onto K random unit vectors; gives the cosine noise floor a
  "concept" must clear. (Guards against "any direction looks slightly evoked.")
- **Two baselines** — s0 (un-injected) streams *and* the off-diagonal; report against both.
- **Arm-A ceiling / machinery check** — arm A projected onto v_c must be large; if it isn't, the
  projection or the re-derived vector is wrong. **Verify the capture/hook ordering**: the saved arm-A
  residual must be *post-injection* for arm A−baseline ≈ α to hold; if `Capture` grabs pre-hook, arm A
  is not a clean ceiling — note and adjust.
- **Vector sanity** — cos(v_i, v_j) matrix: the v's should be largely distinct (near-orthogonal); if two
  concepts' vectors are highly aligned, their re-evocation can't be told apart — report it.

## 6. Decision rule (pre-registered)

- **Re-evocation present** iff mean diagonal cosine (baseline-subtracted) CI excludes 0 **and** clears
  the random-direction floor **and** diagonal-dominance permutation p < 0.05.
- **Magnitude** — report re-evocation as %α and as the ratio to the lexical uplift; **H_magnitude
  supported** if re-evocation% ≪ lexical%.
- Report per concept *and* aggregate; correct for the 12 diagonal tests (FDR or report CIs).

## 7. Confounds explicitly accepted / out of scope

- The projection is still a function of the letters — **acknowledged**; v_c being a *semantic* direction
  plus the selectivity + random-direction controls make a positive result "a concept-aligned
  representation forms," not "raw letter overlap." We do **not** claim letter-independence.
- One layer (28) only — the saved residuals are layer 28; a multi-layer capture is a future extension.
- No cross-model, no non-linear read — separate experiments.

## 8. Compute & deliverables

- **GPU** `derive_vectors.py`: load `C.MODEL` via `K.load_model`, derive + save the 12 v_c and α to
  `concept_vectors.pt`. < 5 min, ≈ $0.20, Vast harness, destroy on done.
- **Offline** `analyze_reevocation.py`: all projections/figures from `covert_collect.pt`. CPU, no model.
  Outputs `reevocation_results.json` + figures (cross-projection heatmap, per-concept re-evocation, the
  per-cut curve, word-vs-experience bar). Plus a one-paragraph report addendum.

---

## Review revisions (incorporated 2026-06-27, after clean-subagent review)

The review was GO-WITH-CHANGES; the fixes below are now the binding design. One of them simplifies the
whole experiment.

0. **The injected direction is recoverable offline — the GPU step is now optional.** Injection adds
   `alpha·v` *only* at layer 28, and arms A and B read identical tokens, so **`armA − armB = alpha·v`
   exactly at every captured position** (verified: hook registered before `Capture`, so A is
   post-injection; layers 0–27 identical). Therefore `v̂_c = normalize(armA − armB)` is the **exact
   injected direction**, recovered offline from `covert_collect.pt` — better than re-deriving. The blog
   re-derivation (`derive_vectors.py`) is demoted to an **optional cross-check** (does the blog method
   reproduce the injected direction? `cos ≥ 0.99`) and to keep the exact steering primitives saved
   for future runs. The core analysis is **CPU-only, no model**.
1. **Units fix.** `alpha = strength/‖v‖`, hook adds `alpha·v = strength·v̂`, so movement along `v̂` under
   injection is **`strength`** (40/60), *not* the scalar `alpha`. Denominator for %-injection is
   `strength`. Gate: `‖armA − armB‖ ≈ strength` and it is **constant across streams** (rel-std ≈ 0) —
   both asserted before any result is trusted.
2. **Selectivity is the primary test; random-direction floor demoted.** Random unit vectors
   underestimate the null for these high-variance directions. Primary null = **off-diagonal** (other
   real concept directions, matched subspace). Statistic (pre-registered): `mean(diag) − mean(offdiag)`
   of the baseline-subtracted, mean-`v̂`-removed 12×12 matrix; permutation = derangement of the
   own-direction assignment.
3. **Remove the shared component.** Project out `v̄ = normalize(mean_c v̂_c)` (the generic
   "Tell-me-about-X" direction) before the cross-projection, so the diagonal isn't inflated by a shared
   concept direction.
4. **Letter control as *context*, not a "beyond-letters" gate.** Regress the per-stream projection on the
   character histogram and report the variance it explains. A clean reader's state is *necessarily* a
   function of the letters, so we do **not** claim letter-independence; a positive selective result is
   reported as *"the letter-induced state lands selectively along the injected concept's semantic
   direction,"* the introspection-relevant geometric fact, not proof of introspection.
5. **Cut-match the baseline.** s0 baseline projection computed at the **same grid cut** as the concept
   streams (residual geometry differs by position).
6. **Pre-registered:** primary operating point **s60** (s40 = replication); primary cut **8**; one
   selectivity statistic + permutation scheme (above); per-concept CIs by stream bootstrap; FDR over 12.
7. **Word vs experience — commensurable version.** Both channels as a **fraction of the arm-A ceiling on
   their own channel**: re-evocation% = `⟨armB−base, v̂⟩ / strength` (= `/⟨armA−armB, v̂⟩`); lexical% =
   `(armB naming uplift)/(armA naming uplift)` from the saved logprobs. Same "fraction of full injection
   recovered in the clean reader" → comparable. (Replaces the `exp(B−C)` relative-uplift comparison.)
8. **Arm A is an algebraic identity, not empirical validation** (`armA = armB + strength·v̂`); used as the
   repro gate and the denominator, not as evidence a representation forms.
