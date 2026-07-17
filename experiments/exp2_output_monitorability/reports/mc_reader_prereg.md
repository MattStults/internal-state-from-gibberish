# MC-letter elicited reader prereg — triptych rows 1–2 upgraded to a letter-logprob readout

Registered 2026-07-10, BEFORE any code that could run on GPU and BEFORE any GPU spend (task #22).
This is a **readout upgrade**, not a new collection: it re-scores the SAME fixed 1.5B stream pool as
the level-2 elicited self-report (`elicited_report_prereg.md`), replacing the free-answer /
capitalization-variant closed-set readout of triptych rows 1–2 with a **multiple-choice letter
readout** — the reader is offered the 12 concepts as lettered options `(a)…(l)` and we read the
log-probability mass over the 12 single letter tokens at a forced `Answer: (` position. The
free-answer elicited reader (`src/elicit_reader.py`) and the LR reader (`src/lr_reader.py`) are
frozen; their reports stand. This experiment adds a parallel, cleaner-currency readout and is scored
independently, then joined against the char n-gram and LR baselines.

## Why the upgrade (question)

The free-answer closed-set readout reads per-concept mass as the logsumexp of `{tok(c)[0],
tok(c.capitalize())[0]}` — the concept's own first sub-word token(s). Those tokens differ in length,
frequency, and prior across concepts, so the readout mixes "which concept did the reader recover"
with "which concept-word does the reader's LM prior favour as a first token." A **letter** readout
removes that confound: all 12 answer tokens are single ASCII letters `a…l` of equal token length and
comparable prior, read at ONE fixed position (`Answer: (`), and the concept→letter binding is
**counterbalanced** by a cyclic Latin square so no concept is tied to any letter. The question is the
same as level 2 — does an elicited self-report recover the injected/evoked concept from a word-free
stream, and does it **rise with reader scale** — measured through a cleaner instrument, plus a
**cross-family** reader to separate genuine readout from same-model surface matching.

## Streams (identical to `elicited_report_prereg.md`; one fixed pool)

From the 1.5B collection, pulled on-box (HF dataset `ErrareHumanumEst/internal-state-from-gibberish`):

- `qwen2.5-1.5b-gen.pt` → `runs/qwen2.5-1.5b/data/covert_collect.pt` (exp1 capture, variant `orig`)
- `exp3/bundles/qwen2.5-1.5b-evoked.pt` → `runs/_ind/qwen2.5-1.5b/data/qwen2.5-1.5b-evoked.pt`

| set | source | selection | cap | expected n |
|---|---|---|---|---|
| `injected` | exp1 capture | accepted, strength == max (s60), len ≥ 2 | 17/concept, ascending gidx | 204 |
| `evoked` | evoked bundle | accepted, strength 1, len ≥ 2 | 17/concept, ascending gidx | 204 |

**Task-#22 scope:** rows 1–2 = the injected (exp1 1.5B capture, s60) and evoked (`runs/_ind` 1.5B,
strength-1) streams. The s0 controls (`injected_s0`, `evoked_s0`) ride along **as registered gates**
using the SAME selection rules `elicit_reader.select_streams` already implements (injected_s0 =
accepted s0; evoked_s0 = neutral, uncapped) — they carry no MC named call but their information-floor
/ concentration gates below must pass. Selection is reused verbatim from the frozen
`src/elicit_reader.select_streams` (deterministic ascending-gidx cap, no RNG).

## Readers (scales × families) — the FROZEN cap N

| slug | family | tier | role |
|---|---|---|---|
| `qwen2.5-1.5b` | Qwen2.5 | RTX3090 | same-family reader = the stream's own generator |
| `qwen2.5-3b`   | Qwen2.5 | RTX3090 | same-family, scale rung |
| `qwen2.5-7b`   | Qwen2.5 | RTX3090 | same-family, scale rung |
| `qwen2.5-14b`  | Qwen2.5 | RTX A6000 | same-family top rung — **run 2, only when the 14B supply sampler frees the tier** |
| `qwen3-1.7b`   | **Qwen3** | RTX3090 | **cross-family reader** |

**Frozen scale cap N (registered, not to be changed post-hoc):** `qwen2.5-1.5b`, `qwen2.5-3b`,
`qwen2.5-7b`, `qwen3-1.7b` on run 1 (RTX3090); `qwen2.5-14b` on run 2 (RTX A6000). No other readers.

### Cross-family reader: **Qwen3-1.7B** (`Qwen/Qwen3-1.7B`), and why

- **Different family, genuinely different reader.** Qwen3-1.7B is a distinct pretraining corpus and
  post-training from the Qwen2.5-Instruct generator of the streams — the "same-model surface
  matching" alternative (that the reader merely re-recognises its own sampling idiosyncrasies) does
  not apply to it. It is the concrete different-family reader for the same-model-vs-cross-model
  contrast.
- **Byte-identical base vocabulary → the saved 1.5B stream token ids splice in verbatim.** Verified
  from the local tokenizer files: Qwen3-1.7B and Qwen2.5-1.5B-Instruct share the SAME 151 643-entry
  base vocabulary (ids 0…151 642 map to identical sub-word strings; spot-checked every 997th id,
  0 mismatches) and the SAME `<|im_end|>` = id 151645 as eos. So a saved Qwen2.5 stream token id
  decodes to the same sub-word under Qwen3, and the eos-strip / letter-token machinery is identical.
  (Qwen3's `tokenizer.json` serialises BPE merges in a different *format*, so a naive `merges ==`
  check is False — but the id↔string map, which is all the splice and the letter read depend on, is
  identical. The on-box tokenizer-compat gate below enforces this per run, so a wrong reader FATALs
  rather than silently mis-reads.)
- **Available locally + already in the config registry** (`config.MODELS["qwen3-1.7b"]`,
  `Qwen/Qwen3-1.7B`, cached in `~/.cache/huggingface`), so no new download or registry edit.
- **Thinking mode off.** Qwen3 defaults to a `<think>` block; the reader builds ids with
  `enable_thinking=False` (the `common.chat_ids` / `_template_ids` fallback already handles this for
  Qwen2.5, which ignores the kwarg), so the forced `Answer: (` splice sits at the answer position
  with no think-block contamination. The greedy-capped-CoT variant (below) is the controlled way the
  reader may "think" — bounded, not the native unbounded block.

**On-box tokenizer-compat gate (per reader, FATAL on failure):** for each of the 12 concepts,
`tok(" " + c)` first-token id must equal the capture's saved `first_ids` (the passive readout's
currency, unchanged from level 2); AND each MC letter option token (below) must be a single token
under the reader. A reader failing either is FATAL (no data), never a silent skip.

## Design (task #22, verbatim): the full cell grid

Every cell is (**stream** × **reader** × **framing** × **reasoning** × **model-relation**), scored
over 12 Latin-square orderings:

- **streams** {`injected` (exp1 1.5B s60 capture), `evoked` (`runs/_ind` 1.5B strength-1)}
  (+ `injected_s0`, `evoked_s0` as gate-only sets).
- **readers** {`qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b`} now; {`qwen2.5-14b`} when the 14B supply
  sampler frees the A6000 tier; and the cross-family {`qwen3-1.7b`}.
- **framing** {**passive-MC**, **elicited-MC**}:
  - *elicited-MC* — the level-2 elicited chat: system (reconstructed generation context) → user
    `GEN_PROMPT` → **assistant = the saved stream ids VERBATIM** (eos-stripped) → user elicitation
    turn, with the elicitation turn presenting the 12 concepts as an MC list `(a) …\n(b) …\n…\n(l) …`
    (order = the Latin-square permutation) and instructing a single-letter answer.
  - *passive-MC* — the legacy passive continuation framing carried to MC: system → user `GEN_PROMPT`
    → **assistant = the saved stream ids VERBATIM** (eos-stripped), then a user turn that ONLY
    presents the same lettered MC list + the single-letter instruction (no elicitation "the previous
    turn was prefilled…" wording — the passive contrast at MC currency). This is the (passive-MC vs
    elicited-MC) contrast at every scale, the MC analogue of level 2's (1)-vs-(3).
- **reasoning** {**no-CoT direct**, **greedy-capped-CoT**}:
  - *no-CoT direct* — one forward; read the answer letter at the forced `Answer: (` position (below).
  - *greedy-capped-CoT* — greedy generation (`do_sample=False`, temperature 0), **cap = 256 new
    tokens** (FROZEN), then splice the **uniform forced-answer** string `\nAnswer: (` after the
    generated CoT and read the letter logprobs at the `(`-follows position (one further forward over
    the CoT + forced splice). **Truncation flag:** a stream whose CoT reaches the 256-token cap
    without emitting eos is flagged `truncated=True` (diagnostic + stratification below).
- **model-relation** {**same-model**, **cross-model**}: same-model = reader family == Qwen2.5 (the
  stream generator's family); cross-model = the Qwen3-1.7B reader. (A property of the reader slug, not
  a separate axis to sweep — it labels each reader cell.)

## The letter readout (the currency)

**Uniform forced-answer read.** After the framing (and, for CoT, after the ≤256-token generated
reasoning), the string `\nAnswer: (` is spliced as forced text; the answer letter is read at the
position immediately after `(`. The 12 answer tokens are the single letters `a, b, c, d, e, f, g, h,
i, j, k, l` in the reader's tokenization of `(a`, `(b`, … (message-mid form, so the letter directly
follows `(`). **On-box asserts (FATAL):** each of the 12 letters is a single token after `(`; no two
letters share an id. Read = float32 log-softmax at the answer position; per-letter logprob; **no
capitalization-variant set, no logsumexp over multiple sub-words** — one token per option (this is
the whole point of the letter upgrade vs the elicited closed-set variant sets).

**Cyclic Latin-square counterbalancing (registered).** The concept→letter binding is rotated so no
concept is fixed to any letter. Ordering `k` (k = 0…11) maps concept at config index `i` to letter
index `(i + k) mod 12`; equivalently, the MC list for ordering `k` lists concepts in the order
`concepts[(0−k) mod 12], concepts[(1−k) mod 12], …` so that letter `(a)` = concept `concepts[(−k) mod
12]`. The 12 orderings form a cyclic Latin square: **each concept appears in each letter slot exactly
once across the 12 orderings**, and each letter carries each concept exactly once. Every (stream,
reader, framing, reasoning) cell is read under all 12 orderings; the per-stream per-concept logprob is
**averaged over the 12 orderings** (in logit space at the answer position, then combined offline) so
any residual letter-token prior cancels. For CoT, the generated reasoning is produced ONCE per
(stream, framing) under a fixed reference ordering (ordering 0) — the Latin square is applied only to
the letter-read splice, so the 12 orderings share one CoT and differ only in the `(a)…(l)` mapping
and the forced-answer read (12 cheap reads, one generation).

**bits & top-1 & held-out-third temperature calibration (parity with LR).** Per (stream-set × reader
× framing × reasoning) cell, build the score matrix `S[stream, concept]` = the Latin-square-averaged
answer-position logprob for that concept. Then, EXACTLY as `lr_reader_offline` /
`analysis/lr_reader_offline.py`:

- posterior = `softmax(S / τ)` with ONE scalar temperature τ per cell, **fitted on a stratified
  held-out third** (per seed, per concept `floor(n_c/3)` (min 1) rows → calibration, rest → eval;
  τ grid = 61 log-spaced points `10⁻²…10⁴`; grid-min calibration CE), evaluated on the eval
  two-thirds; seeds 0–9, report mean ± sd;
- **bits = H(C) − CE = log2(12) − mean(−log2 p(true concept))** on the eval split (calibrated);
- **top-1** = raw argmax accuracy over `S` (temperature-free), on the eval split (and full-set,
  descriptive). Chance = 1/12 ≈ 8.3%; H(C) = log2(12) ≈ 3.585 bits.

This is currency-parity with the LR reader (calibrated bits), so the MC cells JOIN the LR cells and
the char n-gram baseline directly (unlike the level-2 elicited closed-set bits, which were raw
renormalized posteriors — see that prereg's `lr_join` non-parity note).

## Required diagnostics (registered, reported per cell regardless of verdicts)

1. **Truncation rate** per (reader-scale, condition): fraction of CoT streams that hit the 256-token
   cap without eos. Reported for every CoT cell.
2. **Bits stratified truncated-vs-concluded:** within each CoT cell, bits (and top-1) computed
   separately over streams with `truncated=True` vs `truncated=False` (each substratum calibrated
   on its own held-out third if n ≥ its gate; else reported raw with a low-n flag). This isolates
   whether the CoT readout's information rides on the concluded reasoning or survives truncation.
3. **Answer-position mass-on-12-letters fraction:** per stream, the total (un-renormalized)
   probability the reader puts on the 12 letter tokens at the `Answer: (` position — how much of the
   answer distribution the MC options capture (the MC analogue of level-2 coverage). Reported
   mean per cell.
4. **CoT drift / repetition quality:** the generated CoT text and token ids are SAVED per (stream,
   reader, framing) so an OFFLINE quality scorer computes, per CoT cell: mean generated length,
   fraction reaching cap, a repetition score (max fraction of the CoT covered by a single repeated
   token 3-gram — the exp3 looping signature), and the fraction whose CoT mentions any concept word
   verbatim (a "leaked the answer in the reasoning" diagnostic). No verdict rides on these; they
   contextualise the CoT cells.

## Baselines joined in scoring

Both are joined into `mc_reader_results.json` for direction/ordering comparison (calibrated-bits
currency; magnitudes comparable to LR since MC bits are also held-out-temperature-calibrated):

- **char n-gram reader** — the level-1 char n-gram cells (from the exp2 reader pipeline
  `reports/full_stream_convergence.json` / the committed char-reader results), the surface-text
  baseline every readout is measured against.
- **LR reader** — `reports/lr_reader_results.json` (per-cell calibrated bits + top-1), the level-1
  likelihood instrument. The MC cells are compared to LR cells cell-by-cell (injected-MC vs
  injected×A/B LR; evoked-MC vs evoked×A LR).

## Gates (report regardless; named calls scored with a caveat flag if a gate fails)

1. **injected_s0 information floor:** injected_s0 elicited-MC no-CoT bits ≤ 0.1 at every reader scale
   (un-injected streams' nominal labels carry no letter-readable information).
2. **evoked_s0 concentration:** on evoked_s0 (no true label), the mean elicited-MC no-CoT posterior's
   largest entry ≤ 1/6 (2× uniform) at every scale.
3. **letter-mass coverage:** answer-position mass-on-12-letters < 0.05 at a scale is flagged (the
   reader isn't answering in the MC format there) — no hard gate, reported.
4. **On-box asserts (any failure = FATAL, no data):** template prefix property for the MC splice;
   12 distinct single-letter answer tokens after `(`; passive `first_ids` match the capture; the
   cross-family reader's tokenizer-compat gate (per-concept `first_ids` + letter-token check).
5. **Latin-square balance (unit-tested, not on-box):** the 12 orderings form a cyclic Latin square —
   each concept in each letter slot exactly once.

## Named calls (registered verbatim, before any data)

- **MATT (design owner):** elicited-MC works, **especially on injected** streams, and **rises with
  reader scale**.
  *Scoring rule:* elicited-MC (no-CoT, same-family) injected bits — (a) non-decreasing across
  1.5b→3b→7b within a −0.05-bit tolerance per step (and, when the 14B run lands, bits(14b) = max of
  the same-family rung), AND (b) bits(7b) − bits(1.5b) ≥ 0.1, AND (c) at ≥ 1 same-family scale
  injected elicited-MC bits ≥ 0.5 AND top-1 ≥ 0.3. All of a–c → **right**; exactly two → **partial**;
  else **wrong**. ("especially injected" cross-check: at the scale where (c) fires, injected bits >
  evoked bits.)
- **CLAUDE (assistant):** **same-model ≈ cross-model** on the MC readout — the reader is
  surface-matching the stream, not privileged self-introspection — so Qwen3-1.7B (cross-family)
  reads injected-MC comparably to the same-scale Qwen2.5 same-family reader; AND **injected-MC may
  beat the LR reader's ~0 injected bits** (the MC elicitation extracts injected-stream information
  the likelihood instrument could not).
  *Scoring rule, two parts:* (i) **surface-matching**: |bits(qwen3-1.7b injected elicited-MC no-CoT)
  − bits(qwen2.5-1.5b injected elicited-MC no-CoT)| ≤ 0.15 bits AND both > chance-equivalent (bits >
  0.1) — cross-family reads comparably → the surface-matching part is **right**; if the same-family
  reader clears bits > 0.3 while the cross-family reader floors (< 0.1), surface-matching is
  **wrong** (evidence of same-model privilege). (ii) **injected-MC beats LR**: the best same-family
  injected elicited-MC bits (over scales) exceeds the LR `injectedxA` calibrated `bits_mean` by
  ≥ 0.1 → **right**; within ±0.1 → **tie**; below → **wrong**. Overall verdict reported as the pair
  (surface-matching, beats-LR).

## Crash-class guards (these bugs were burned already; enforced by unit tests)

- **(a) Progress marker must not substring-collide with the box's done/ready/fatal markers or the
  labkit FATAL tuple.** The reader prints `MC_SHARD_SAVED …` (NOT `MC_DONE_*`). A class-level test
  (extending `tests/test_marker_guard.py`'s scan) asserts every `src/*_reader.py`'s print-statement
  literals contain none of its box's ready/done markers nor labkit's FATAL substrings
  (`CUDA out of memory`, `CUDA error`, …). Box markers: `MC_READY` / `MC_DONE` / `MC_FATAL`.
- **(b) Trailing-eos strip before splice.** Saved streams keep collection's trailing `<|im_end|>`
  (id 151645) at set-correlated rates (40.7–68.1%); the reader strips AT MOST ONE trailing eos before
  splicing (reusing `elicit_reader.strip_one_eos`), RED-first tested, so the spliced ids are
  token-exact to a real conversation render. `eos_stripped` recorded per record.
- **(c) Prefill KV legacy-tuple handling + registered self-check fallback**, mirroring `lr_reader`:
  the CoT forced-answer read reuses the CoT-prefill KV; a legacy-tuple vs Cache object is accepted
  either way, and a first-batch self-check (KV-reuse vs concat, tol 0.02 nats/token) falls the whole
  run back to the concat path on mismatch (`MC_SELFCHECK_FALLBACK`), and a mid-run KV raise falls
  back by exception TYPE only (no raw message, which could carry a FATAL substring).

## Run plan & budget (BUILD ONLY for this task — do NOT launch)

- `src/mc_reader.py` (GPU) via `box_mc.py` + `harness/run_mc.py` — clones of the gauge/elicit
  driver pair (S0 HF-pull of the `*.pt` inputs, 2-min heartbeat through silent downloads,
  `HF_HUB_DISABLE_XET=1`, generous `run_to`, markers `MC_READY`/`MC_DONE`/`MC_FATAL`). Atomic shards
  `$INTRO_RUN_DIR/mc/<reader>_<streamset>_<framing>_<reasoning>.pt`, `MC_SKIP` resume, status
  `runs/mc-status.json`, ledger `runs/confound-ledger.json`.
- **Two driver invocations, each `--dry` first:** run 1 = RTX3090, readers
  {qwen2.5-1.5b, qwen2.5-3b, qwen2.5-7b, qwen3-1.7b}; run 2 = "RTX A6000", reader {qwen2.5-14b}
  (`--max-hours 2`, min-vram 40000, disk floor 56 — the 14B collect precedent). Run 2 waits for the
  14B supply sampler to free the tier; run 1 does NOT touch `runs/lr-*` or `runs/elicit-14b-*`.
- Scoring is OFFLINE: `analysis/mc_offline.py` → `reports/mc_reader_results.json` (Latin-square
  averaging → posterior → held-out-temperature-calibrated bits + top-1 + all diagnostics + char/LR
  join).
- Budget: expected $0.8–2.0 total; ledger `runs/confound-ledger.json` (cap $5, shared line);
  self-imposed stop after 2 failed box attempts per tier. **This task builds and unit-tests only; no
  box is launched.**

## Amendments log

- 2026-07-10 — registered at build time, before any GPU-runnable code. Cross-family reader chosen:
  Qwen3-1.7B (byte-identical base vocab to Qwen2.5-1.5B, verified locally; already in the registry).
  Frozen: cap N (the 5 readers above), CoT cap = 256 tokens temp 0, forced-answer string `\nAnswer:
  (`, 12 cyclic Latin-square orderings, held-out-third τ-calibration (10 seeds, 61-pt grid), all gate
  thresholds (injected_s0 ≤ 0.1, evoked_s0 ≤ 1/6, coverage flag < 0.05), and both named-call scoring
  rules above.

---

## Amendment 1 (pre-data, methodological review) — 2026-07-10

**Status: PRE-DATA.** Registered 2026-07-10, still BEFORE any GPU-runnable code has produced a
single MC shard (no `runs/mc_box/mc/*.pt` exists). No re-collection: this refines scoring/reporting
rules and named-call interpretation while the frozen body above is untouched. Because NO MC data
exists yet, named-call *interpretation* and *arbitration surface* can be sharpened here without
p-hacking; the two named-call scoring *bodies* in "Named calls" above (MATT rule a–c; CLAUDE rule
(i)/(ii) thresholds) are FROZEN and unchanged — this amendment adds the surrounding controls and
honest-bound qualifiers they are read against.

### B1 — the char n-gram reader is the PRIMARY surface-matching discriminant

The Qwen3-1.7B cross-family reader shares Qwen2.5's tokenizer and overlapping pretraining, so a
"same ≈ cross" MC result cannot by itself distinguish "no self-privilege" from "shared substrate."
Therefore:

- **(a)** The **char n-gram reader** (level-1 surface-text reader,
  `full_stream_convergence.json` → `analyses.convergence_injected_1p5b.readers.char.full.mean`) is
  registered as the **PRIMARY** surface-matching control. The decisive surface signature is **MC
  injected bits ≈ char-reader injected bits** — both recover the concept from surface lexical
  overlap. Qwen3-1.7B is **SECONDARY**: only its *asymmetric* branch (same-family reads while the
  cross-family reader floors, `< 0.1`) is clean evidence of same-model privilege; a symmetric
  "same ≈ cross" is consistent with shared substrate and does NOT establish surface-matching on its
  own — the char comparison arbitrates that.
  - *Currency + access-regime caveat (registered).* The MC-vs-char comparison is bits-vs-bits: both
    are H(C)−CE (the char reader's `.full` cell is likewise calibrated bits), so the magnitude
    comparison is legitimate. But the instruments differ in **access regime** — char@full reads the
    **whole stream's surface text** with n-gram statistics, whereas MC is a **single constrained
    letter read** over 12 offered options. char@full is therefore a *generous* surface ceiling, so
    "**MC bits ≲ char bits ⟹ surface-explicable**" is the **conservative** direction of the test
    (the direction the named call turns on) and is robust to the access difference; the only claim
    the access gap could threaten — MC bits *exceeding* a full-stream surface ceiling — would be
    strong evidence, not a false positive. We report the char cell as this conservative ceiling, not
    as an access-matched twin of the MC read.
- **(b)** The CLAUDE named call is **re-scoped**: "same ≈ cross" now means **"NOT same-model-SPECIFIC
  privilege,"** NOT "surface-matching established." Surface-matching is arbitrated by the MC-vs-char
  comparison, not by the cross-family reader.
- **(c)** `mc_offline` now computes the scored **MC-vs-char delta** for the primary named-call cell
  (`char_injected_bits`, `mc_vs_char_delta`; surfaced in `char_join.injected_char_bits` and
  `named_calls.claude.mc_vs_char`).

### B2 — bind the scale claim to the non-truncated (`concluded`) substratum

Truncation is scale-correlated, so a scale conclusion drawn from greedy-capped-CoT cells could ride
on differential truncation rather than reader scale. **Binding rule (not merely a diagnostic):** any
scale conclusion drawn from greedy-capped-CoT cells MUST reproduce within the `concluded`
(non-truncated) substratum (`mc_offline.cot_strata` already stratifies truncated-vs-concluded). The
**MATT primary call is scored on no-CoT `direct` cells** — confirmed in code
(`b(r, "injected", "elicited", "direct")`), so it already dodges truncation entirely. Additionally,
any raw-τ (uncalibrated, low-n) stratum is now **flagged** (`cot_strata[*].calib ==
"uncalibrated_raw_tau"`, via `calib_status`) so it is never compared across scales as if calibrated.

### B3 — passive-MC re-scope (Matt approved)

- **(a)** The passive-vs-elicited contrast is relabelled honestly: **passive-MC** = the MC letter
  format (options shown) + a MINIMAL match instruction with **NO "a concept was installed/injected"
  preamble**; **elicited-MC** = the same MC format + the full elicitation preamble. Their difference
  isolates the **PREAMBLE effect with format held fixed** — the clean comparable contrast.
  Verified in `src/mc_reader.py`: `PASSIVE_INSTRUCTION` = *"Which concept best matches the previous
  turn's output? Answer with a single letter."* — no "installed/injected/prefilled" preamble
  (unchanged; no fix needed).
- **(b)** The **legacy continuation probe** (`; secret word:`-style, open-vocab first-token renorm
  over the 12 concept first-tokens — the exp1 naming-probe / prior `elicit_offline` `passive`
  readout) rides as a **SEPARATE "raw unprompted leak" row** (`legacy_continuation_leak` in the
  results), reported explicitly as CONTEXT and **NOT subtracted into the MC / workspace tax**. It is
  a different currency (open-vocab continuation vs constrained MC letters); its low
  mass-concentration is a real finding, not normalized away. The prior elicit run already produced
  these numbers (`elicited_report_results.json` `passive` bits, e.g. injected `qwen2.5-1.5b` =
  −1.400 bits, top-1 0.167) — these are **cited** rather than recollected; they may also be
  collected fresh per-scale in this run's passive continuation cell if cheap.

### SHOULD-FIX

- **Honest "beats LR" claim.** If injected MC bits > 0 while LR injected bits ≈ 0, that is
  **upper-bounded by "identifiable-in-context by a reader offered the options"** and is **NOT a
  richer channel than LR** unless the char AND cross-family readers are **BOTH at floor (`< 0.1`) on
  injected**. `mc_offline` now ties the beats-LR interpretation to those controls
  (`named_calls.claude.richer_channel_than_lr` + `beats_lr_interpretation`); the frozen `beats_lr`
  threshold verdict (right/tie/wrong) is unchanged, this only qualifies its interpretation.
- **Low-n stratum calibration-status flag** surfaced in the stratified CoT output (`cot_strata[*].calib`; see B2).
- **Letter-position residual-bias diagnostic:** mean per-letter argmax rate on the label-free
  `evoked_s0` set, flagged if any letter's argmax rate `> 2 × 1/12`
  (`mc_offline.letter_position_bias`, reported under `readers.<r>.evoked_s0.*.letter_position_bias`).
- **Multiplicity guard (registered):** everything OUTSIDE the two named calls (MATT, CLAUDE) and the
  registered gates is **DESCRIPTIVE / hypothesis-generating** (`results.scope_note`).

### NITS

- The s0 cells outside `elicited_direct` (i.e. passive and/or CoT s0 cells) are retained only as
  registered gate/diagnostic inputs (injected_s0 information-floor gate on `elicited_direct`;
  evoked_s0 concentration + letter-position bias); they carry no named call and are descriptive.
- `mentions_concept` stem match tightened conceptually: it flags a "leaked the answer in the
  reasoning" only when a CoT word shares a `≥ min(5, len(concept))`-char prefix with a concept
  (unchanged behaviour; documented here as the intended, deliberately conservative stem rule — a
  diagnostic, never a verdict input).

### What Amendment 1 changed (prereg vs code)

- **Prereg only:** B1(a)/(b) re-scope of the CLAUDE call and char-as-primary control; B2 binding
  rule text; B3(a) relabelling; multiplicity guard; NITS.
- **Code (`analysis/mc_offline.py`), all RED-first tested:** B1(c) `char_injected_bits` +
  `mc_vs_char_delta` + `char_join.injected_char_bits`; B2 `calib_status` flag on `cot_strata`; B3(b)
  `legacy_leak_block`; SHOULD-FIX honest beats-LR bound + `letter_position_bias` on `evoked_s0` +
  `scope_note`. No change to `src/mc_reader.py` (passive instruction verified clean; the tech-fixed
  `letter_token_ids` and the frozen named-call scoring bodies are untouched).
