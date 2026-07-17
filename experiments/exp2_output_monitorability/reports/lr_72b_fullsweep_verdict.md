# LR-70B full-sweep verdict — no detectable concept-identity channel under this instrument

Run: Llama-3.3-70B-Instruct-Turbo (Together serverless batch), full sweep 810 streams × 13
contexts (matched + neutral + 11 mismatched), 2026-07-13.
Official adapter: `analysis/lr_70b_scout_offline.py` (RED-first tested,
`tests/test_lr_70b_scout_offline.py`) — parses `runs/llama70b_scout/lr_raw_batch_output.jsonl`
with the scout's own span finder (`_find_stream_span_lps`, imported), builds the per-arm
(n × 12) context matrices, and scores them through the CERTIFIED calibrator
(`lr_reader_offline.evaluate_cell`, held-out-third τ, 61-pt grid, 10 seeds — the same function
object as the whole scale grid). Machine-readable results:
`reports/lr_72b_fullsweep_results.json`. **This supersedes the n=5 peek's GO verdict**
(`lr_72b_peek_verdict.md`, corrected in place).

Data hygiene: 10,529 raw lines parsed; 91 empty-span extractions (0.86%, concentrated in 7
streams) + 1 batch-error request (`lr:secret_word:obedience:695:anger`, in
`lr_raw_batch_errors.jsonl`); streams missing any of the 13 contexts dropped ⇒ n = 270/278/254
per arm (of 276/278/256). Per-stream diagonal cross-checks against
`lr_records_llama70b.json` exactly (max |Δ| = 0.0 on all arms). Note: the records file's
matched+neutral-only `secret_word` mean is 1.627 nats over its 271 records; the matrix-complete
set (n = 270) gives 1.605 — same data, one high-LR stream fewer.

## Headline table (calibrated bits = log₂12 − CE, the SAME currency as the grid's 0.163/0.191/0.405)

| arm | n | raw diag (nats)* | raw diag (bits)* | generic lift (nats) | lift % of diag | centered diag (nats), 95% CI† | gate-3 mm nats/tok (bound 0.02) | **calibrated bits** | top-1 (chance 8.3%) |
|---|---|---|---|---|---|---|---|---|---|
| secret_word | 270 | 1.605 | 2.316 | 1.389 | 86.5% | 0.189 [−0.64, 0.99] — no | 0.0119 pass‡ | **−0.002 ± 0.011** | 5.5% |
| secret_sustain | 278 | 1.896 | 2.735 | 1.391 | 73.4% | 0.492 [−0.21, 1.11] — no | 0.0120 pass‡ | **+0.013 ± 0.006** | 11.2% |
| evoked | 254 | 4.534 | 6.541 | 4.037 | 89.0% | 0.409 [−1.25, 2.09] — no | **0.0283 FAIL → VOIDED** | +0.000 ± 0.009 | 12.2% |
| *7B reference (Qwen2.5-7B, scale grid)* | 495 | ~2.14 | ~3.09 | — | — | — | pass | **0.405** | **25.1%** |

\* raw diag = matched−neutral summed span log-probs. `ll_over_span` sums **natural-log** probs:
these are **NATS** (every "raw bits" label in the earlier 70B docs was a mislabel; the bits
column divides by ln 2). Calibrated values are genuine bits.
† 12-concept bootstrap of the mean of concept means, 10k resamples, seed 20260713. "no" = the
CI does not clear 0.
‡ Numerically within the 0.02 nats/token grid bound, but the +1.39-nat generic lift is
structural — the pass is not evidence of a centered instrument (see decomposition).

**Length-truncation robustness (calibrated bits at span[:K])** — the null is not a
long-stream artifact:

| arm | K=16 | K=32 | K=64 | full |
|---|---|---|---|---|
| secret_word | −0.000 ± 0.006 | −0.003 ± 0.016 | +0.000 ± 0.009 | −0.002 ± 0.011 |
| secret_sustain | +0.022 ± 0.011 | +0.026 ± 0.010 | +0.016 ± 0.005 | +0.013 ± 0.006 |
| evoked | −0.000 ± 0.009 | −0.003 ± 0.006 | +0.004 ± 0.004 | +0.000 ± 0.009 |

## Generic-lift decomposition: the raw diagonal was ~73–89% context lift

Mean mismatched−neutral is 1.389 / 1.391 / 4.037 nats — i.e. these streams are more likely
under *any* concept context than under neutral, which says nothing about *which* concept. That
is 86.5% / 73.4% / 89.0% of the raw diagonal. The concept-specific residue (centered diagonal,
diag − mean mismatch) does not clear zero in any arm under the 12-concept bootstrap, and the
12-way calibration — which cancels per-stream constants — reads **~0 bits with top-1 at/below
chance in all three arms**. The model cannot identify which of 12 concepts it was carrying from
its own word-free output, as measured by this instrument.

## Gate 3 (mismatched centering, grid bound 0.02 nats/token)

- `evoked`: mismatched per-token median **0.0283 > 0.02 → cell VOIDED** under the grid's own
  gate. Its raw 4.53-nat "signal" was overwhelmingly generic lift.
- `secret_word` (0.0119) and `secret_sustain` (0.0120) pass numerically, but the positive
  mismatched lift is structural (every context beats neutral), so gate-3 "pass" here should not
  be read as a certified-clean instrument — the small-model grid's teacher-forcing had no such
  offset.

## Why the peek said GO (and why that was wrong)

The Amendment-6 instrument bar's channel criterion tested **matched−neutral > 0** — a contrast
that generic context lift alone satisfies. The discriminating contrast (matched vs
**mismatched**) was computable at peek time (all 13 contexts were collected) but was set aside;
centered, no arm clears zero. The peek's headline "the model-private channel is clearly present
at 70B" is **retracted** — only generic lift was demonstrated. See the dated correction in
`lr_72b_peek_verdict.md`.

## Instrument confounds — why this is an instrument-qualified null

Four unmatched conditions separate this run from the preregistered instrument; the design
cannot separate "channel absent" from "instrument attenuated":

- **(a) Model-family switch.** The prereg (`reports/lr_72b_prereg.md`) froze thresholds and
  named calls for **Qwen2.5-72B on self-hosted vLLM**; this run is **Llama-3.3-70B on Together
  serverless**. Nothing on the Qwen scaling curve (0.163/0.191/0.405 at 1.5/3/7B) is
  within-family with this point.
- **(b) Generation mode collapse.** ~90% of the 810 streams share a near-identical "tgf"/"tgx"
  prefix template — little entropy for a concept mark to live in. Amendment 6 pre-committed to
  calling this regime "instrument-limited, inconclusive."
- **(c) Scoring-format mismatch.** Streams were generated via `/v1/chat/completions` (real
  Llama chat template + special tokens), but the LR echo scored a plain-text
  `{system}\n\nHuman: {gen_prompt}\n\nAssistant: {stream}` concatenation
  (`harness/run_llama70b_scout.py::_build_echo_prompt_text`) — a framing the model was never
  conditioned on. The small-model grid teacher-forced **token-identically** to generation.
  Supporting evidence that this matters: a char n-gram classifier recovers the concept from the
  evoked stream *text* at **12.9% ± 2.7%** vs 8.3% chance, while the LR instrument reads 0.000
  bits on the same arm — surface-visible information the instrument misses.
- **(d) Quantized serving.** The Turbo serverless endpoint serves a quantized model; the
  small-model grid scored full-precision weights.

**Conclusion (frozen wording): no detectable concept-identity channel at Llama-3.3-70B under
this instrument — an instrument-qualified null, NOT a clean no-channel result and NOT a
scoreable point on the Qwen scaling curve.**

## Amendment-5 controls: NOT computable on this run

The Amendment-5 char-surface and position-lift controls run through the certified readers,
which require per-stream token ids / `gen_topk` (and per-token LR vectors for the position
control). Every stream in `streams_llama70b.json` has `token_ids = null` and no
`gen_token_logprobs` (generated before the capture fix, 430dcd9), so **neither control can be
fed to the certified code on this run**. Disclosed as not-computable — not silently passed
(`am5_controls` in the results JSON). (The 12.9% char n-gram figure above is a diagnostic
classifier on stream text, not the certified Amendment-5 control.)

## Named-calls disposition (MATT ≥ 0.50 / CLAUDE ≥ 0.60 + sustain ≥ 1.5 + off-diag < 0.05 + char pass)

Two readings, both stated:

1. **If the Llama scout were accepted as the 72B test:** calibrated `secret_word` = −0.002
   bits ⇒ **MATT wrong** (< 0.50) and **CLAUDE wrong** (< 0.60; sustain +0.013 < 1.5). Note
   CLAUDE's off-diagonal clause ("7B reads the 70B streams" < 0.05) is **unscoreable** — that
   cell was never collected in this run — and the char-pass precondition is not computable
   (above), so even this reading is on a degraded scoring basis.
2. **The prereg targeted Qwen2.5-72B on self-hosted vLLM** — a different model family,
   serving stack, and (per confound (c)) a different instrument. The recommended disposition is
   therefore: **not scored on this run; pending a within-family point.**

**DECIDED (Matt, 2026-07-13): reading 2 — a disposition recorded after the calibrated null
was known (post-data); the grounds are the two instrument facts above.** The prereg was for
the Qwen family, period; the
frozen MATT/CLAUDE calls are **not scored on this run** and remain pending a within-family
point (or formal descoping if the Qwen-72B test is never funded). Recorded alongside, as an
informal unscored opinion (not a registered call): Matt would guess the same ≈0 outcome for
Llama-70B, but declines to score any prediction against a measurement he considers
instrument-invalid.

**Discriminator named calls (frozen 2026-07-13, before the faithful-echo re-score runs):**

| call | prediction for the template-faithful re-score of these same 810 streams |
|---|---|
| MATT | stays ≈0 — the echo-format mismatch was not the binding confound |
| CLAUDE | stays ≈0 (calibrated `secret_word` < 0.05 bits; ~65% credence). Mode collapse is the dominant suspect given the ~90% template share. The ~35% tail — signal appearing — would be a cross-family frontier positive AND an instrument lesson. |

The Amendment-6 CLAUDE peek-call ("weak, 0.05–0.20 calibrated") also does not score cleanly:
the measured calibrated value (~0) is below the band, but under the same instrument
qualification as everything else here.

## What would actually resolve it

- **Cheaper:** re-score these same 810 streams on Together using the *actual rendered Llama
  chat prompt* (token-identical to generation), ~$2.4, no regeneration — removes confound (c).
- **Preregistered path:** Qwen2.5-72B on self-hosted vLLM with token-identical teacher-forcing
  — removes (a), (c), (d) and yields a scoreable within-family point for the named calls.
- Either way, confound (b) (mode collapse) may still cap what is carriable; report accordingly.

## Paper-hygiene fixes made alongside this

- `lr_72b_peek_verdict.md` — dated CORRECTION appended (GO-bar contrast wrong; nats/bits
  conversions; "falsifies the CLAUDE named call in raw terms" retracted).
- `lr_72b_prereg.md` — Amendment-6 MDE paragraph carries a dated post-data correction: its
  "raw bits" figures are nats (prereg text otherwise untouched).
- `HANDOFF.md` — review-update section + nats/bits labels fixed; §6 gen-logprobs capture claim
  corrected (all 810 streams have `token_ids = null`, no `gen_token_logprobs`); §4 named-call
  scoring superseded by the disposition section above.

---

## Faithful-template discriminator result (2026-07-14 addendum; batch 1c4faf7a, ~$2)

The re-score of the SAME 810 streams under the real Llama-3.3 chat template
(`harness/rescore_echo.py` config A; validation gate GO — special tokens round-trip as single
tokens). 803 records (7 primary empty spans, 0.86% — identical to the original run; mismatch
empties 77/8,910). Scored through the certified `evaluate_cell`; data in
`runs/rescore_llama70b/` (records + meta committed; raw JSONL on disk).

| arm | plain-text calib (bits) | **faithful calib (bits)** | top-1 | gate-3 mm/tok | centered diag CI (nats) |
|---|---|---|---|---|---|
| secret_word | −0.002 ± 0.011 | **−0.011 ± 0.018** | 9.0% | +0.0074 pass | [−0.03, +0.50] — no |
| secret_sustain | +0.013 ± 0.006 | **+0.244 ± 0.032** | 16.1% | +0.0177 pass (near bound) | [+1.29, +2.39] — **yes** |
| evoked | 0.000 ± 0.009 | **+0.143 ± 0.037** | 15.2% | +0.0141 pass | [+0.97, +1.87] — **yes** |

**Reading.** The scout's plain-text echo format destroyed real signal in two of three arms:
a self-read channel EXISTS at Llama-3.3-70B for sustained-secret (0.244 bits) and evoked
(0.143 bits) conditioning, cross-family and despite ~90% template collapse. The bare
`secret_word` channel remains undetected — that null is now **format-exonerated** but still
mode-collapse/quantization-qualified.

**Discriminator named calls scored.** Frozen rule (calibrated `secret_word` < 0.05): both
MATT and CLAUDE **right on the letter** (−0.011). The stated rationale — "the echo-format
mismatch was NOT the binding confound" — is **wrong for sustain/evoked**; disclosed as
letter-right, rationale-half-wrong.

**Newly open cell.** The 70B sustain/evoked marks now have untested PRIVACY (no off-diagonal
was ever collected). A Qwen-7B-reads-Llama-70B-streams block is cheap and could ride the 14B
box session — requires a disclosed post-freeze amendment to `lr_scale_extend_prereg.md`
(Matt decides).

*Caveats: computed by the review path using the same certified function objects (an official
adapter pass over `rescore_lr_records.json` should confirm before publication); sustain's
gate-3 (0.0177) sits close to the 0.02 bound; "centered diag" is in nats.*

*CONFIRMED 2026-07-16: the official adapter pass (`analysis/confirm_rescore.py`, matrices
rebuilt from `rescore_lr_records.json`'s `context_lls`, scored through the certified
`lr_reader_offline.evaluate_cell`) reproduces the review-path numbers — secret_word
−0.011 ± 0.018, secret_sustain +0.244 ± 0.032, evoked +0.143 ± 0.037 (max |Δ| = 0.0003,
n = 271/278/254, 0 dropped). Machine-readable:
`reports/lr_72b_rescore_confirmation.json`. The publication precondition above is met.*
