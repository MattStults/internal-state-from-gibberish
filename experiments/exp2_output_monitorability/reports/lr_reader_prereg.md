# LR-reader prereg — likelihood-ratio readout of word-free streams (1.5B)

Registered 2026-07-09, BEFORE any GPU spend. Follow-up to the confound-closing run (see
`confound_closing_verdicts.md`): the trained readers say the natural regime's readable bits are
wording residue and the injected regime's bits are state; this experiment asks the same question with
a **model-likelihood instrument instead of a trained reader** — is a saved word-free stream *more
likely* under the persona context that (nominally) produced it, teacher-forced under the generating
model itself?

## Question

For each saved stream, compute the teacher-forced log-likelihood of the stream's tokens under each of
25 reconstructed collection contexts, and read the concept out of the likelihood profile. In
particular (Matt's question): are even the **injected** streams (persona never in context at
generation time — the concept entered as a residual-stream vector) more likely under the relevant
natural personas and their paraphrases?

## Instrument

- Model: `qwen2.5-1.5b` (Qwen/Qwen2.5-1.5B-Instruct), bf16, ONE model, this run only.
- Inputs (HF dataset `ErrareHumanumEst/internal-state-from-gibberish`, pulled on-box like box_gauge S0):
  - `qwen2.5-1.5b-gen.pt` → `runs/qwen2.5-1.5b/data/covert_collect.pt` (exp1 capture, variant `orig`)
  - `exp3/bundles/qwen2.5-1.5b-evoked.pt`, `exp3/bundles/qwen2.5-1.5b-evoked_alt.pt`

### Stream sets (all accepted streams; `len(tokens) >= 2`; report exact ns)

| set | source | selection | expected n |
|---|---|---|---|
| `injected` | exp1 1.5B capture | `accepted`, strength == max (s60) | ~435 (31–44/concept) |
| `evoked` | `qwen2.5-1.5b-evoked.pt` | `accepted`, strength 1 (concept) | ~707 |
| `evoked_alt` | `qwen2.5-1.5b-evoked_alt.pt` | `accepted`, strength 1 | ~703 |
| neutral gate streams | both bundles, strength 0 (`concept == "neutral"`) | `accepted` | ~58 + ~62 |

No sub-sampling: **use all accepted streams** (the per-concept balanced cap does not apply here; the
readout is per-stream and ns are reported per concept).

### Contexts (25) — built EXACTLY like collection

For each concept `c` in the 12 frozen concepts (`primers.CONCEPTS` == `config.COVERT_CONCEPTS`):

- **wording-A** (12): `primers_v2.compose_system(c, C.STRONG_SYSTEM, arm="evoked")`
- **wording-B** (12): `primers_v2.compose_system(c, C.STRONG_SYSTEM, arm="evoked_alt")`
- **neutral** (1): `primers_v2.compose_system(None, C.STRONG_SYSTEM, arm="evoked")` (the shared
  length-matched NEUTRAL persona; identical for both persona arms, hence 25 contexts not 26)

Each context is tokenized as at collection: `common.chat_ids(tok, C.GEN_PROMPT, system=<above>)`
(chat template, generation prompt appended). Stream tokens append directly after the generation
prompt, exactly as the saved generated tokens did.

### Likelihood

`LL(stream | ctx) = Σ_t log p(tok_t | ctx, tok_<t)`, teacher-forced, no sampling, no injection hook
anywhere (the LR reader is an *observer* — it never sees the vector): token 0 is predicted at the
last context position; log-softmax computed in float32 over the bf16 logits; sum over the stream's
tokens.

Efficiency: per context, the context KV cache is prefilled once and reused across a batch of streams
(right-padded, masked). **On-box self-check** (registered): the first batch is scored both via
KV-reuse and via full-concat forward; if max |ΔLL|/T > 0.02 nats/token the run falls back to the
concat path for everything (prints `LR_SELFCHECK_FALLBACK`), else prints `LR_SELFCHECK_OK`.

### Scores

`score(stream, c; W) = LL(stream | ctx_W_c) − LL(stream | ctx_neutral)` for wording W ∈ {A, B}.
(Note: the neutral subtraction is constant across `c` for a given stream, so the softmax readout
below is invariant to it — neutral anchors the sign conventions and the sanity gates.)
Per-token score = score / T is used wherever a length-free quantity is needed (gates, descriptives).

## Readout (per stream-set × context-set cell)

Cells (stream-set × context-wording):

| cell | role |
|---|---|
| evoked × A | within-wording ceiling |
| evoked × B | paraphrase transfer at likelihood level |
| alt × A | paraphrase transfer at likelihood level |
| alt × B | secondary within-wording ceiling (descriptive) |
| injected × A | Matt's question (natural personas) |
| injected × B | Matt's question (paraphrase personas) |

Per cell: posterior over the 12 concepts = `softmax(score_c / τ)` with **one scalar temperature τ
per cell**, fitted on a held-out calibration third and evaluated on the remaining two-thirds:

- Split: per seed, streams stratified by true concept, shuffled; `floor(n_c/3)` (min 1) per concept
  → calibration, rest → eval. Seeds 0–9 (10 splits); report mean ± sd over seeds.
- τ fit: grid search minimizing mean CE of the true concept on the calibration split; grid = 61
  log-spaced points, `10^-2 … 10^4` (scores are LL sums, O(1–100) nats).
- **bits = H(C) − CE = log2(12) − mean(−log2 p(true concept))** on the eval split (calibrated).
- **raw top-1** = argmax_c score accuracy (temperature-free), reported on the eval split (and on the
  full set, descriptive). Chance = 1/12 ≈ 8.3%; H(C) ≈ 3.585 bits.

## Gates (must pass for the calls to be scored; report regardless)

1. **Neutral-context sanity**: neutral streams (both bundles, s0) have no concept — their per-token
   scores must be centered ~0. Pass iff, per context set, |median over neutral streams of the
   per-stream mean-over-12-concepts per-token score| ≤ max(0.02 nats/token, 25% of the evoked×A
   matched-minus-mismatched per-token median gap).
2. **Mismatched-concept centering**: for concept streams in each cell, the median per-token score
   over *mismatched* concepts must satisfy the same bound (mismatched personas ≈ neutral).
3. **n**: all accepted streams used; per-concept ns reported; each seed must leave ≥ 6 eval streams
   per concept in every scored cell (expected: min concept n = 31 → cal 10, eval 21 ✓).

If gate 1 or 2 fails, cells are still reported but the named calls are scored with a flagged caveat
(the "positive" direction claim loses its neutral anchor; the softmax readout itself is unaffected).

## Named calls (registered verbatim, before any data)

- **MATT**: "even the injected streams are more likely under the relevant natural and paraphrase
  personas" — i.e. injected×A and injected×B both positive (top-1 > chance 8.3%, bits > 0.1).
  *Scoring*: mean-over-seeds calibrated bits > 0.1 AND mean eval top-1 > 0.083, in BOTH injected
  cells → right; else wrong.
- **ASSISTANT (Claude)**: positive but small on injected×matching — bits in (0.05, 0.5), well below
  evoked×A; evoked×B stays wording-tied-low (< 0.15 bits calibrated) consistent with the transfer
  test. *Scoring*: (a) injected×A and injected×B mean bits both in (0.05, 0.5) AND ≤ half of
  evoked×A mean bits ("well below"), AND (b) evoked×B mean bits < 0.15 → right; all of (a) or (b)
  failing → wrong; mixed → partially right, reported as such.

## Run plan & budget

- `src/lr_reader.py` on one RTX3090-tier box via `box_lr.py` + `harness/run_lr.py` (clones of the
  gauge pair: S0 HF-pull, heartbeat through silent downloads, `HF_HUB_DISABLE_XET=1`, generous
  `run_to`). 9 atomic shards `$INTRO_RUN_DIR/lr/qwen2.5-1.5b_<streamset>_<ctxset>.pt`
  (streamset ∈ {injected, evoked, evoked_alt} × ctxset ∈ {N, A, B}), LR_SKIP resume.
- Scoring is OFFLINE (this file's readout, `analysis/lr_reader.py`) →
  `reports/lr_reader_results.json` + an "LR-reader addendum" in `confound_closing_verdicts.md`.
- Budget: expected $0.15–0.40 (~20–40 min); ledger `runs/confound-ledger.json`, gate max_spend $5,
  self-imposed stop after 2 failed box attempts (~$0.5).
- ~48k stream×context scorings ≈ 1.9k batched forwards of ≤ 328 tokens at 1.5B — comfortably inside
  the window.

## Caveat note — 2026-07-09 (attempt 5 in flight; construction unchanged, analysis-time caveats)

1. **Top-p sampling warp.** Streams were sampled at `temperature=1.0, top_p=0.98`
   (`covert_collect.py`), but the LR reader scores them under the FULL (unwarped) softmax. LL is
   therefore the base model's likelihood of the tokens, not the truncated sampling distribution
   they were actually drawn from. The warp is identical across all 25 contexts for a given
   stream, so the per-stream contrasts the readout uses (`LL_c − LL_neutral`, softmax over
   concepts) are only mildly exposed — but absolute LLs and any cross-stream/cross-set use of
   them inherit it.
2. **eos-in-LL.** Saved streams keep the trailing `<|im_end|>` (id 151645) when generation hit
   eos, at rates that DIFFER BY STREAM SET (40.7–68.1%) — that final-token LL term is a
   construction artifact confounded with the measured cell contrasts. The shards store summed
   LLs only, so the term cannot be subtracted offline; a rerun would be needed to excise it.
   Analysis must therefore report the per-set eos-termination rates (computable from the
   bundles) next to the cells, and treat small cross-set bit differences as potentially within
   this artifact's range. (The level-2 elicited experiment strips the trailing eos pre-splice —
   see `elicited_report_prereg.md`'s 2026-07-09 amendment; the two levels differ here by
   construction.)
