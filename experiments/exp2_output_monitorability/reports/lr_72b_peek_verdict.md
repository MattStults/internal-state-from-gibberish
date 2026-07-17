> ⚠️ **RETRACTED / CORRECTED 2026-07-13.** This peek verdict is WRONG on its headline. The GO was
> computed on **matched−neutral** (raw nats, mislabeled "bits"), which is satisfied by *generic context
> lift* — NOT concept-specific signal. The full sweep's certified 12-way calibration is **~0 bits in all
> three arms** (secret_word −0.001, top-1 5.5%); ~87% of the raw diagonal is generic lift. The claim "the
> model-private channel is clearly present at 70B" is **withdrawn** — only generic lift was shown. See
> `lr_72b_fullsweep_verdict.md` for the corrected result and the instrument confounds. Retained below for
> the record; all "bits" on raw numbers should read "nats".

# LR-72B Amendment-6 instrument peek — verdict: ~~**GO**~~ **RETRACTED (see banner above)**

Ran 2026-07-13. Instrument-feasibility peek on **Llama-3.3-70B** (Together serverless batch),
n=5 streams/(arm×concept) = 180 of 810 available streams scored. Raw teacher-forcing LR
(matched−neutral, summed per stream, uncalibrated). This is the go/no-go peek, **not** the
confirmatory sweep. See prereg Amendment 6.

## Instrument bar — all three criteria pass

| criterion | threshold | result | pass |
|---|---|---|---|
| empty/failed-span rate | ≤ 5% | 0.56% (1/180) | ✓ |
| per-token LL non-degenerate | var > 0 | 0/179 degenerate; per-stream var median 4.2, min 0.93 | ✓ |
| `secret_word` LR ≥ τ_peek, CI clears 0 | ≥ 0.05, lower bound > 0 | grand 1.013, 95% CI [0.498, 1.526] | ✓ |

Concept-level bootstrap (12 concepts resampled, 10k, seed 20260713):

| arm | grand LR (raw bits) | 95% CI | clears 0 | n |
|---|---|---|---|---|
| secret_word | 1.013 | [0.498, 1.526] | yes | 59 |
| secret_sustain | 1.783 | [0.086, 3.607] | yes | 60 |
| evoked | 6.502 | [4.207, 8.839] | yes | 60 |

Data: `runs/llama70b_scout/{peek_verdict.json, lr_records_llama70b.json}`.

## Verdict: **GO** to the full sweep.

The teacher-forcing instrument works on Llama's templated word-free output (spans extract, LL is
non-degenerate) and the model-private channel is clearly present at 70B — the mode-collapsed
gibberish did **not** kill it. This falsifies the CLAUDE named call's "weak, 0.05–0.20" prediction
*in raw terms* (raw secret_word = 1.01 bits, CI lower bound 0.50).

## Critical caveat for reading these numbers

These are **raw** matched−neutral bits. The scale-grid headline (0.163 / 0.191 / 0.405 at
1.5B/3B/7B) is **calibrated** (held-out third τ, eos-free). They are **not** directly comparable:
for 7B, raw ≈ 2.14 compressed to calibrated 0.405 (≈5.3×). A similar factor would map Llama's 1.01
raw to ≈0.19 calibrated — near the top of the predicted band — but the calibration factor is
model-specific and must be measured, not assumed. **The calibrated secret_word diagonal — the number
that answers "does the private channel keep climbing at 70B?", is comparable to 0.405, and scores the
named calls — comes only from the full sweep** (proper calibration, char/position controls,
sustain/evoked at full n, n-matched to the smaller-model curve). The peek settles feasibility, not
magnitude.

The mismatched-context scores needed for calibration *were* collected in the peek (2340 = 180×13),
so a calibrated peek preview is computable on request; kept out of this verdict to preserve the
peek/confirmatory line.

---

## CORRECTION — 2026-07-13 (post-full-sweep review; appended, original text above unchanged)

Authoritative numbers: `analysis/lr_70b_scout_offline.py` →
`reports/lr_72b_fullsweep_results.json`; full writeup `lr_72b_fullsweep_verdict.md`. Where the
banner at top and this section differ (e.g. −0.001 vs −0.002), this section and the results
JSON govern.

**1. The GO bar tested the wrong contrast.** The third instrument criterion was
`secret_word` **matched−neutral** with a CI clearing 0. *Generic context lift* — the stream
being more likely under **any** concept context than under neutral — satisfies that on its own,
and the full sweep shows it is ~73–89% of the raw diagonal (mean mismatched−neutral = 1.389 /
1.391 / 4.037 nats per arm). The discriminating contrast is matched vs **mismatched**; the
mismatched contexts were collected at peek time and would have shown this. Centered on
mismatched, no arm's CI clears 0.

**2. The headline is withdrawn.** "The model-private channel is clearly present at 70B" was
wrong. The full sweep's certified 12-way calibration reads **≈0 bits in every arm**
(secret_word −0.002 ± 0.011, top-1 5.5%; secret_sustain +0.013 ± 0.006, 11.2%; evoked
+0.000 ± 0.009, 12.2%; chance 8.3%). Only generic lift was demonstrated. (This is an
instrument-qualified null, not a clean no-channel result — see the confounds in
`lr_72b_fullsweep_verdict.md`.)

**3. Every "raw bits" figure in this document is actually NATS** (`ll_over_span` sums
natural-log probabilities). Conversions (÷ ln 2):

| quantity | stated ("raw bits") | actually | in bits |
|---|---|---|---|
| peek secret_word | 1.013 | 1.013 nats | 1.462 |
| peek secret_sustain | 1.783 | 1.783 nats | 2.572 |
| peek evoked | 6.502 | 6.502 nats | 9.380 |
| peek secret_word 95% CI | [0.498, 1.526] | nats | [0.718, 2.202] |
| 7B "raw ≈ 2.14" | 2.14 | 2.14 nats | ≈3.09 |

Calibrated values quoted in project docs (0.163/0.191/0.405) are genuine bits and unaffected.

**4. RETRACTED: "This falsifies the CLAUDE named call's 'weak, 0.05–0.20' prediction *in raw
terms*."** That sentence compared a raw matched−neutral sum (in nats, and dominated by generic
lift) against a **calibrated-bits** prediction — different units *and* different contrasts, so
it could not falsify anything. On the number that actually scores the call, the calibrated
secret_word diagonal is ≈0, i.e. *below* CLAUDE's 0.05–0.20 band, not above it — and per the
full-sweep verdict the named calls are recommended **not scored on this run** at all (prereg
froze them for Qwen2.5-72B/vLLM; Matt's decision).
