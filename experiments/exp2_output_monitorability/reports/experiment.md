# Output monitorability of a steered concept: the token budget to recover it

*The offline (Phase 0) result for [the proposal](../proposed-followup.md), run on the existing Qwen2.5
`_ab` bundles. Three readers of the **same** word-free streams — a distribution-access reader, an
embedding-based best-token-monitor, and a one-hot sampled-token floor — measured in **bits of concept
identity recovered vs output tokens T**. It answers: does the concept survive to the output distribution
but get lost at sampling, and does the best honest token monitor close the gap? Qwen2.5 1.5B/3B/7B.*

## Hypothesis and Prediction

From the proposal, three falsifiable predictions (currency: `bits_recovered = H(C) − CE`, the best-decoder
lower bound on `I(concept; output)`; empirical `H(C)` over the realized label counts, not `log₂K`):

1. **The budget is predictable** — the number of output tokens to recover the concept to a fixed accuracy is
   forecastable from the per-token distribution shift and verified by drawing that budget (predict-then-demonstrate).
2. **The loss is at sampling, not propagation** — a reader with logit-*distribution* access recovers the
   concept where a *sampled-token* reader needs far more tokens.
3. **The budget grows with scale** — recovery cost rises with model size (claimable only at matched dose).

**Verdict up front.** (2) is **supported at 1.5B/3B and the strongest thing this run establishes** — *within a
short read* (the full-stream update below shows it is a monitoring-*latency* gap, not a channel-capacity one) —
crucially, it survives replacing the weak one-hot floor with a fair embedding reader. (1) the loop is
**operational and self-gating** but the concept is either recovered almost immediately (distribution) or just
past the 12-token capture (token), so no clean within-capture crossing was demonstrated. (3) is **dose-confounded
and punted** — this is the sharpest threat the proposal named, and it is why Experiment 3 exists.

> **Full-stream update (re-analysis, `publish-prep`).** Everything below is stated *at the 12-token capture*,
> and the headline for (2) does **not** survive the full stream. Extending all readers to the whole stream and
> adding a fourth — `char`, a bare character uni/bigram histogram of the realized text, *no model access* —
> reverses it: the token/transcript channel is not thin, it is **slow**. On a fixed injected-1.5B cohort
> (streams ≥64 tokens, n=13/class, mean±sd over 3 seeds), bits at T = 12 / 24 / 48 / 64 / full are `dist`
> 2.15±.07 / 2.29±.21 / 2.06±.44 / 1.92±.37 / 1.53±.24 (fast, front-loaded, then dilutes under mean-pooling —
> and noisy late), `R_emb` 0.72 / 1.46 / 1.67 / 1.62 / 1.73, `char` 0.47 / 0.99 / 1.66 / 1.84 / **2.37±.16**.
> `char` catches `R_emb` by ≈48 tokens and passes it, and matches then exceeds `dist` by the full stream. Two
> caveats keep this honest: (a) `char` beating `dist` at full is a **reader**-level result — our `dist` reader
> mean-pools and the per-step distribution sits *upstream* of the tokens, so a better-aggregated `dist` reader
> should weakly dominate any transcript reader; the table proves the token channel is **not thin**, not that
> the transcript is a richer channel. (b) The 2.37 is the fixed ≥64 cohort; over *all* streams `char` at
> full recovers **2.06 / 0.72 / 0.77** bits at 1.5B / 3B / 7B (source: `reports/full_stream_convergence.json`,
> `analysis/full_stream_convergence.py`). So `R_emb ≈ floor` (the "central result" below) is a **budget artifact
> of the short capture**: distribution access buys *speed* (≈10× fewer tokens for the same bits), **not
> exclusivity**. The genuinely distribution-*only* result holds only for **natural** induction (Experiment 3),
> where even `char` at full stream floors (and stays floored on the length-controlled ≥64 cohort). Read the T=12
> verdict below as a statement about *monitoring latency*, not channel capacity.

> **Scale update (dose titration + pre-registered 14B run).** This report's scale trend (dist@T12
> 2.64 → 2.24 → 0.89) and its Prediction-3 punt are now RESOLVED, against the trend: the published 7B point
> came from the norm-scaled `_ab` bundle (s93, arm-A rank 68 — under the ≤ 50 criterion). Re-running the
> identical bits protocol on the criterion-passing captures gives dist@T12 **2.64 / 2.24 / 2.51** and
> char@full **2.06 / 0.72 / 1.90** at 1.5B / 3B / 7B — flat-to-non-monotone, not declining; the 7B "collapse"
> was a dose artifact (the 1.5B→3B char dip is real and unexplained). A pre-registered 14B collection
> (`scale14b_prereg.md`: frozen thresholds, validity gates, named calls, dated pre-data amendment) extends the
> curve: primary cell dist@T12 = **1.78 ± 0.26**, char@full = **1.24 ± 0.17** → verdict **PLATEAU** on both
> channels (`scale14b_verdict.md`; all gates pass, positive cells exceed own-pool permutation nulls, named
> calls scored 1/3 as written). The injected channel does not close through 14B; the closing-with-scale result
> belongs to the natural regime only. Evidence: `dose_titration{,_qwen2.5-3b,_qwen2.5-7b,_qwen2.5-14b}.json`,
> `perm_null_check_*.json`.

> **Confound-closing update.** The regime-vs-dose question this report left open is settled by the
> pre-registered E1–E5 suite (`confound_closing_prereg.md` → `confound_closing_verdicts.md`): weak
> per-token injection FLOORS on the transcript even where its distribution signal exceeds natural's
> (E1); strong prompt-only injection floors too (E3) — **dose × persistence jointly** gate the
> transcript channel; direct state trajectories (E4 + the free-association gauge measurement) show the
> persona installs ~no state along the concept direction (~0.05σ in-task, ~0.4σ free, vs injection's
> 12σ). The 'genuinely weak injection' named below as the missing experiment has been run.

## Experiment

Offline, $0, on the accepted strong-dose streams already collected for exp1's A/B robustness run (the
`runs/_ab/qwen2.5-{1.5b,3b,7b}-gen.pt` bundles: generation-only injection, `gen_topk` = per-step top-64
next-token logprobs, realized token ids, and the `concept_idx` label). The **12-way** concept-identity task
(H(C) = 3.585 bits). Common N = 288 (24/class) subsampled across models so cross-scale comparison is at
matched sample size; mean ± sd over 3 seeds (subsample × CV).

**Three readers of the identical streams** — the only difference between them is what they see of each
generation step:

- **`dist`** (distribution-access) — mean, over the first T steps, of the probability the model put on each
  vocab token (`gen_topk`, dense, uniform-tail-floored to close the top-64 boundary). The full pre-sampling head.
- **`R_emb`** (the best token monitor — *the rung exp1 was missing*) — mean-pool the model's own token
  embeddings over the first T **realized** tokens. **Same model access as `dist`**, so `dist − R_emb` isolates
  the *sampling step*, not an access gap. This is the honest strong token-level monitor.
- **`sampled`** (one-hot) — the realized token ids as a sparse one-hot bag; the weak floor and the literal
  Prediction-#2 contrast.

Each reader gets a **per-channel nested-CV best decoder** (inner `GridSearchCV` on `neg_log_loss` picks
capacity — dense: PCA-dims × logistic-C down to 0.001 so weak/thin signal floors near 0 rather than going
negative; sparse: TF-IDF → logistic; outer `cross_val_predict` gives held-out proba). The per-model recovery
curve is `bits_recovered(T)` for T ∈ {2,3,4,5,6,8,10,12}; a linear rate is fit on the early window and
**nulled if r² < 0.8** (the fit has reached saturation and T* would be inflated).

## Observed

Bits recovered at the top budget T = 12 (mean ± sd over 3 seeds), and the two gaps:

| model | `dist` (full head) | `R_emb` (best token monitor) | `sampled` (one-hot floor) | honest gap `dist−R_emb` | featurization gap `dist−onehot` |
|---|---|---|---|---|---|
| 1.5B | 2.64 ± 0.18 | 0.96 ± 0.18 | 0.75 ± 0.07 | **1.68** | 1.89 |
| 3B   | 2.24 ± 0.03 | 0.45 ± 0.02 | 0.45 ± 0.01 | **1.79** | 1.80 |
| 7B   | 0.89 ± 0.05 | 0.47 ± 0.03 | 0.45 ± 0.02 | **0.42** | 0.44 |

(Full curves and errorbars: `reports/budget_curves.png`; all numbers: `reports/budget_results.json`.)

- **The distribution reader out-reads the token readers *at 12 tokens* — and it isn't a featurization artifact
  (but see the latency caveat below).** `R_emb` (embedding access, same model view as `dist`) recovers essentially
  what the one-hot floor does (0.96 vs 0.75 at 1.5B; 0.45 vs 0.45 at 3B; 0.47 vs 0.45 at 7B). Giving the token monitor
  the model's own embedding space **barely helps** — so the honest gap (1.68 / 1.79 / 0.42) is only marginally
  below the featurization-matched gap (1.89 / 1.80 / 0.44). *At 12 tokens* the token channel looks thin and the
  distribution-vs-token gap is real — but the full-stream update above shows this is a **latency** gap, not a
  capacity one: given the whole transcript the token/`char` readers catch up and overtake. Read as "distribution
  monitoring is faster," not "the token channel is thin."
- **The distribution channel itself shrinks with scale** — 2.64 → 2.24 → **0.89** bits. The concept-identity
  content recoverable from the pre-sampling head nearly collapses at 7B, taking the gap down with it
  (1.7 b → 0.4 b).
- **The strong channel saturates; only the weak channels have a "rate."** `dist` reaches its plateau by
  T ≈ 4 at 1.5B/3B (rate correctly nulled, r² = 0.77 / 0.66). The clean per-token rates belong to the weak
  readers: `R_emb` 0.088 b/tok (r² 0.94) at 1.5B, 0.057 (r² 0.85) at 3B; `sampled` 0.049 (r² 0.89) at 3B. Their
  forecast budgets land just past the capture (T* ≈ 13–22 → "T* > budgets"), so recovery is *predicted* but not
  *demonstrated-within-capture* at these lengths — the honest read of Prediction 1.
- **7B `dist` is weak and noisy** — negative bits at T = 2–3 (−0.25, −0.19), a fit that flags itself
  `verified = False`. So the 7B collapse is real in this data but not clean.

## What it shows, and what it doesn't

**Strongest claim (at a short read).** On word-free Qwen2.5 output at 1.5B–3B, *within a 12-token read* a
distribution-access monitor recovers ~1.7 more bits of injected-concept identity than the best token-level
monitor (embedding access, same model view), and that margin is not an artifact of a weak one-hot reader. But
this is a **monitoring-latency** result — distribution access reads the concept ≈10× faster — **not** an
exclusive-channel one: given the full transcript the token and `char` readers recover as much or more (full-stream
update). The genuinely token-*invisible* regime is natural induction, not injection (Experiment 3).

**Cannot establish (beyond the proposal's standing list).**
- **A scale law for the budget (Prediction 3).** The gap is ~flat at 1.5B→3B then collapses at 7B, but the
  series injects a different effect size per model, so budget-vs-size conflates monitorability scaling with
  *how much we injected*. **This is unresolved and dose-confounded** — the collapse could be the vector failing
  to take at 7B (7B `dist` noise is consistent with that), not the concept genuinely leaking less. Punted to
  Experiment 3.
- **A demonstrated within-capture budget crossing (Prediction 1).** The loop is built and self-gating (nulls
  saturated fits, bootstrap-verifies T*), but at these stream lengths the strong reader saturates before a
  budget bites and the weak readers cross just past T = 12. Longer captures, or the bits-ladder, would show it.
- **That no stronger token reader exists.** `R_emb` is a strong lower bound on the token channel, not a ceiling;
  a learned decoder on richer token features could recover more. The claim is a lower-bound gap.

## Confounders we hit

- **Dose not held across scale** (the headline threat, unresolved here). The `_ab` bundles inject a per-model
  effect size; the 7B collapse is entangled with under-injection. We do **not** claim the scale trend from this
  run — it is the explicit motivation for Experiment 3's dose-controlled / injection-free design.
- **Top-64 truncation → presence/absence feature.** `gen_topk` is the top 64 logprobs; a token dropping out of
  the top-64 window would swing its probability to 0, which the downstream scaler would amplify into a pure
  truncation feature and inflate `dist` at low T. Ruled out with a **uniform-tail floor** (absent tokens get the
  per-step `(1 − Σ top-K)/(V − K)` mass); the reported `dist` is floored. (An early unfloored run showed the
  artifact directly: T = 2 dropped 2.48 → 0.74 once floored.)
- **Decoder capacity vs sample size / negative bits.** A high-capacity dense decoder overfits a thin signal into
  *invalid negative* bits. Ruled out by the per-channel nested CV with C down to 0.001 (weak signal floors at
  ~0, not below) and common-N subsampling, so cross-scale bits are capacity-matched.
- **One-hot overstates token weakness** (the reader we most worried about). At T = 12 the honest (`R_emb`) and
  featurization-matched gaps nearly coincide, so the one-hot floor wasn't overstating weakness *at that budget*.
  But the full-stream update shows the deeper answer is yes: at T = 12 *every* token reader badly understates
  the channel — given the whole transcript `R_emb` and `char` recover several times more (the signal is there,
  it just accrues slowly). The T = 12 coincidence is early-budget, not a channel ceiling.

## Next steps → Experiment 3

The two things this run cannot settle are the same thing: everything here rests on **activation injection**, and
(a) the scale trend is confounded by injection dose, while (b) a skeptic can dismiss the whole distribution-vs-token
gap as a *fingerprint of the steering vector* rather than a property of how hidden state reaches output. Both are
answered by removing the vector. Experiment 3 re-runs the reader contrast with the concept induced by
**prompt/context** (the Subliminal-Learning mechanism) instead of a steering vector, on open-weight models where
we own the logits — testing whether the distribution-over-token privilege is **method-independent**, and only
then whether it persists across scale. It is a distinct experiment with its own writeup and review.

## Why does this matter for safety?

The exp1 finding was qualitative — an output-token monitor and an activation monitor can disagree about what a
model is carrying. This run **quantifies that disagreement**: within a short read, a token monitor recovers
~1.7 fewer bits of a steered concept from word-free output than a distribution-access monitor. The full-stream
update reframes *what kind* of disagreement this is — a **latency** gap, not a capacity one: distribution access
lets a monitor read a steered concept an order of magnitude faster, but the token/transcript channel is not
blind to injection, only slow. The safety-relevant *exclusive*-channel claim — a concept that reaches the
distribution and **never** the transcript — belongs to natural, unforced induction, and that is Experiment 3.
