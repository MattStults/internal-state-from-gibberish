# Spec — Powered covert-channel measurement (G2-v2)

## Goal
Strengthen **Thesis A** (covert channel: a clean, un-injected instance recovers the injected
concept from *word-free* gibberish), instrumented so we can later test **Thesis B / privilege**
(is the channel same-model-privileged or readable by any model). No elaborate output task — keep
flat gibberish so the surface baseline stays well-defined.

## What changed vs the pilot (and why)
- **All blog concepts** (K-way), not just silence/ocean. A concept is just a word (blog-faithful
  vector = `"Tell me about {word}."` − baseline mean), so adding concepts is free.
- **Rank-shift read** replaces the brittle 2-way softmax (which saturated/inverted and made the two
  margins arithmetically identical). Rank/logprob are open-vocab → K independent, interpretable,
  prior-controlled signals from K concepts. Matches the read we already validated in G0b.
- **Hardened surface baseline** — position-aware + repetition-robust, so "beyond surface" means
  beating a *strong* token-level reader, not the position-blind strawman that handed s60 a fake win.
- **Per-token recognition curves** — measure at every stream position, not just the end, to
  distinguish "established immediately (first-token/surface)" from "accumulates (distributed)".
- **Repetition suppression at generation** — kill the mode-collapse loops that made both metrics
  measure repetition instead of concept. Documented as part of the task definition.

---

## 1. Concepts (K-way)
`CONCEPTS = [silence, ocean, fear, celebration]` (extend to the blog's full list if provided).
Chance baselines scale with K (probe chance = 1/K).

## 2. Generation (the streams)
- Natural decoding: **letters allowed**, no constrained decoder, no word-ban.
- System prompt = `STRONG_SYSTEM` (the validated anti-word prompt; word_rate 0.0 at strength 0).
- Per concept c: inject blog-faithful vector at `L28`, `strength = eff_mag / ‖v‖`.
- **NO repetition suppression.** Decision (b): suppression is an uninterpretable perturbation that's
  entangled with the injection (collapse appears mainly under injection -> suppressing it may erase an
  effect of the vector). Keep generation natural; **measure** degeneracy instead of preventing it
  (see #3). If injection drives collapse, that is a finding, not noise.
- `eff_mag` grid: **CONFIRM** (focused {0, 40, 60, 80} vs wider {0,20,40,60,80,100}).
- `N` streams per concept per strength: **CONFIRM** (target ~50 for tight CIs).
- Save token ids (`.pt`) + decoded text for offline re-analysis.

## 3. Compliance / degeneracy as a MEASURED COVARIATE (reason about it, don't hide it)
Per stream, continuous scores (not just flags):
- `word_rate` — real-word runs / alpha runs (hardened: CamelCase-split, ≥3 chars).
- `repetition` — `1 − distinct_trigrams/total_trigrams` (or compression ratio); how looped it is.
- `non_latin_frac`, `single_char_spacing_frac`.
These are analysis variables, not gates. Consequences for every read (A/B/C):
- **Report twice:** on ALL streams, and on the NON-DEGENERATE subset. A channel that exists only in
  collapsed streams is structural/surface, not the distributed channel we'd claim.
- **Confound check:** test whether `repetition` (and the others) **correlate with concept**. If
  silence-streams collapse more than ocean-streams, the probe may be reading "how looped is this"
  rather than the concept — surface it instead of being fooled by it.

## Three reads (how the streams are interrogated)
Read **A (surface baseline)** and read **B (activation probe)** are the SAME classifier on different
features; the head-to-head **B-vs-A** comparison is what produces the act/surf/paired-CI table.
Read **C (rank-shift)** is the independent prefill read. So: A and B are a contest, C stands alone.

## 4. Reads A & B — activation probe vs hardened surface baseline (the B-vs-A contest)
- **Activation features:** one forward pass over `[prompt + stream]` → layer-28 residual at *every*
  position (cached for the curve, below).
- **Hardened surface baseline (strong R0):** token-level features that a determined token-counter
  could compute — unigram+bigram histogram **+ first-N token identities (position-aware) + distinct-
  token presence/log-counts (repetition-robust)**. Beating this is a real "beyond surface" claim.
- Both feature sets → **PCA-30 (capacity-matched)** → K-way logistic regression, repeated stratified
  CV (PCA fit train-only). Metric = net detection vs 1/K chance.
- **Comparison:** stratified paired bootstrap CI on `act_netdet − surf_netdet`; activation wins iff
  CI lower bound > 0. Run pre- and post-word-ablation.
- **Per-token curve:** from the single cached forward pass, train the probe on activations at each
  position t → `recognition_act(t)`. Save the curve (+ CI), not raw activations.

## 5. Read C — prefill rank-shift (per-token, KV-cached)
- **Primary statistic:** full-vocab **rank** of each concept word's first token after
  `[prompt + stream[:t] + "; secret word:"]`. **Backup:** absolute logprob of the concept word.
- **Per-token via KV cache (cheap, full-resolution):** encode `[prompt + stream]` once; for each cut
  t, `cache.crop(prompt_len + t)`, replay the ~4 suffix tokens, read ranks of all K concept words
  off the final logits (K lookups in one distribution). Per cut ≈ suffix length, not a re-encode.
- **Prior control (the actual signal):** `rank_shift(w) = median rank(w | other-concepts' streams)
  − median rank(w | w's own streams)`. The model's baseline fondness for w sits in both terms and
  cancels; the shift is the injection effect. Bootstrap CI; w passes if CI clear of 0.
- **Per-token curve:** `recognition_rank(t)` per concept → graph immediate-vs-accumulating.

## 6. Artifacts (kept small)
- `streams.pt` — token ids per (concept, strength).
- `streams_decoded.txt` — text + word_rate/degeneracy tags.
- `curves.json` — per (concept, strength): `recognition_act(t)`, `recognition_rank(t)` per concept,
  with CIs. (scalars per position → KB, graphable offline.)
- `results.json` — per (concept, strength): word_rate, degeneracy_rate, act/surf net-det + paired CI
  (endpoint), rank_shift per concept + CI, agreement flags.

## 7. Interpretation / gate
- **Covert channel (A) confirmed at a (strength) point** iff: streams clean (word_rate≈0, low
  degeneracy) AND the injected concept's `rank_shift` CI > 0 (reader recovers it), AND ideally the
  activation probe also beats the hardened surface baseline (CI > 0).
- **Curve shape:** flat-high from t≈1 → first-token / surface, low bandwidth. Rising with t →
  distributed channel (the more interesting, higher-bandwidth result).
- **Cross-readout agreement required:** activation and rank reads must point at the *same* strength
  with the *same* sign. (Their disagreement in the pilot was the underpowered-noise signature.)

## 8. Privilege hook (Thesis B — later)
The rank read is the most same-model-privileged (uses the model's own word associations). Follow-up:
add a **foreign-model probe (R1)** on the same tokens. R2≈R1>R0 = any model can read it (not
privileged); R2>R1>R0 = same-model privilege. Out of scope for this run; the design leaves room.

## Open decisions to confirm before build/spend
1. `eff_mag` grid — focused {0,40,60,80} or wider {0,20,40,60,80,100}?
2. `N` per concept (~50?).
3. Full concept list (the 4 we have, or the blog's complete set?).
4. ~~Repetition suppression~~ — DECIDED (b): no suppression; degeneracy measured (#3).
5. K-way activation probe: keep silence-vs-rest binary framing too, or pure K-way?
