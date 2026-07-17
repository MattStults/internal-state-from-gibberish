# Pre-registration: does the leak's confusion structure follow representational geometry?

**Registered before computing any confusion matrix.** Question: is the word-free leak an idiosyncratic
per-concept fingerprint, or does it encode the concept's *position in representation space*? The random
bits-ladder (coarse groupings recover ~0) cannot distinguish these; the decoder confusion matrix can.

## Independent predictor (touches no stream / no decoder output)
Cosine similarity of the **exact injected concept vectors** (`inject_vectors` in
`runs/_ab/qwen2.5-1.5b-gen.pt`), the directions that produced the injected 1.5B streams. Concept order =
`bundle["concepts"]`. Cosine range −0.261 .. +0.813, mean +0.418.

## Hypothesis
The leak is a low-rank shadow of the injected direction, so the decoder confuses concept pairs **in
proportion to their injected-vector cosine**.

## Primary test (falsifiable, could go either way)
Mantel correlation between the 12×12 injected-vector cosine matrix and the 12×12 leak-confusion matrix
(off-diagonals), on the **injected 1.5B** cell (`dist` reader, top budget T=12, held-out CV predictions,
pooled over seeds 0/1/2), permutation p from row/col shuffles.
- **SUPPORTED:** r ≥ 0.30 and p < 0.05.
- **REFUTED:** r ≈ 0 (|r| < 0.15) — leak is idiosyncratic, not geometric.
- **INCONCLUSIVE:** in between / p ≥ 0.05 (12 concepts = 66 pairs; underpowered is a real outcome, reported
  as such, NOT upgraded to supported).

## Named calls (eyeball-able, committed now)
1. **fear ↔ anger is the single most-confused pair** (cosine +0.813).
2. **ocean is the least-confused / most-distinctly-recovered concept** — it is the affective outlier
   (negative cosine with nearly everything; nearest neighbor only −0.011).
3. The negative-affect set **{fear, anger, loneliness, curiosity, silence}** confuses *within itself* more
   than with the rest.
4. **{deception, obedience, security}** form a secondary confusable cluster (control/social).

## Secondary
Repeat on the pooled **evoked+named 1.5B** cell (lower signal); consistency of the sign/structure across
injected vs evoked is corroborating (not required).

## Interpretation locked in advance
- SUPPORTED ⇒ the output leak is a projection of the concept direction: low-rank, a monitor recovers *region
  of concept-space*, and leak-confusability is predictable a priori from representation geometry.
- REFUTED ⇒ fine-grained idiosyncratic per-concept code; no exploitable low-dimensional structure.
