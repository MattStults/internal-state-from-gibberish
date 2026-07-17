# LR scale-grid prereg — the paraphrase-transfer trend across sizes, readers, and families

Frozen 2026-07-11, before any new data. Amendments allowed only dated, pre-data, and additive
(disclosure/controls), never to the named-call scoring criteria.

## Question

The LR paraphrase-transfer cell (evoked×B ≈ 0.17 bits at 1.5B, symmetric twin alt×A ≈ 0.17) is the
**only positive cross-wording concept signal in the project** — everything else that reads as
leakage is either wording residue (collapses under paraphrase) or the injected distributional mark
(invisible to the model's own likelihoods, non-transferable). This run measures how that one real
signal behaves across:

1. **generator size** (1.5B → 3B → 7B evoked streams),
2. **reader size and identity** (same model, other sizes of the same family, a different family),
3. **the self diagonal** (reader = generator — the configuration the project never ran above 1.5B,
   and the one the open-introspection scaling result predicts should behave differently ≥3B),
4. plus the **MC self-report diagonal** at 3B/7B on their OWN streams (closing the gap that the
   MC run's readers ≥3B only ever read the 1.5B pool).

## Streams

| set | status |
|---|---|
| evoked (wording A) @ 1.5B, 3B, 7B | exist (`runs/_ind/qwen2.5-{size}/data/qwen2.5-{size}-evoked.pt`) |
| evoked_alt (wording B) @ 1.5B | exists |
| **evoked_alt (wording B) @ 3B, 7B** | **GENERATE on this run** — exp3 induction pipeline, alt personas, identical anti-word instruction, word-free filter, acceptance gates; target the pipeline's standard ~700 accepted per size; strength-0 neutral streams ride along as always |

The symmetric twin is REQUIRED, not optional (Matt, 2026-07-11): a single transfer direction can be
poisoned by direction-specific artifacts (B-paraphrases lexically drifting toward A-personas, or
generically concept-flavored B texts); at 1.5B the twin is why 0.17 was trusted (0.176 / 0.167 both
directions). All 3 sizes get both directions.

## Readers (6)

Qwen2.5-Instruct {1.5B, 3B, 7B} and Llama-3.2-{1B, 3B}-Instruct + Llama-3.1-8B-Instruct
(access preflight-verified 2026-07-11; fallback family if a box-side gate blocks: Falcon3 {1B, 3B, 7B},
ungated — a fallback swap is a disclosed amendment, not a redesign).

## LR instrument (unchanged from lr_reader_prereg.md except where registered here)

`score(stream, c; W) = LL(stream | ctx_W_c) − LL(stream | ctx_neutral)`, teacher-forced, no hooks,
float32 log-softmax over bf16 logits, full-concat reference path (the KV self-check from the 1.5B
run applies; if it trips, full-concat throughout, as before). Calibrated bits per cell: held-out
third temperature, 61-pt log grid, 10 seeds (identical to the 1.5B run and the MC reader).

- **Qwen readers, Qwen streams**: shared tokenizer → teacher-force the saved token ids directly.
- **Llama readers**: re-tokenize the decoded stream *text* with the Llama tokenizer; LR is a
  text-level likelihood ratio so this is valid. The context texts (A/B/N) are re-tokenized the same
  way. Register: identical decode → re-encode for numerator and denominator of every ratio.
- **eos rule (registered, fixes the 1.5B run's flagged artifact)**: PRIMARY scoring excludes the
  terminal eos token from all LL sums in all cells; the with-eos variant is computed and reported
  alongside for comparability with the 1.5B numbers. Trend claims use the eos-free primary.

### Cells

Readers (6) × stream sets (evoked@3 sizes + alt@3 sizes) × wording {A, B} = 72 LR bits cells, each
with the neutral baseline and the mismatched-concept ~0 centering check. Named structure:

- **diagonal**: reader == generator (Qwen only; 3 sizes × 2 directions of cross-wording).
- **cross-size**: Qwen reader ≠ Qwen generator.
- **cross-family**: Llama reader × any Qwen generator.

### MC self-report diagonal (rides the same box)

The certified MC-letter reader (src/mc_reader.py, unchanged scoring), 3B reader on 3B evoked
streams and 7B on 7B evoked streams, all four framing×reasoning combos, with the s0 controls and
gates as registered in mc_reader_prereg.md + Amendment 1. (1.5B×own already exists from the MC run.)

## Gates (all $0-fail: a failed gate voids affected cells, never reinterprets them)

1. Alt-generation acceptance rate within 2× of the 1.5B alt run's; word-free filter identical.
2. Neutral streams: |median per-token score| ≤ 0.02 nats/tok per context wording (the 1.5B run's
   bound; a narrow miss is disclosed, sign-checked, and cannot rescue a positive).
3. Mismatched-concept scores centered ~0 in every cell.
4. Llama sanity: on a small English-prose control set, Llama LL ranks matched > mismatched contexts
   (instrument works cross-family at all before we interpret its gibberish cells).
5. MC cells: the mc_offline gates as registered (injected_s0-analog bits ≤ 0.1; concentration
   diagnostic reported; letter coverage).

## Perf checklist (registered because the grid is big and the box is big)

1. One contiguous box: alt-generation (3B, 7B) → all LR reads → MC diagonal, no manual phases.
2. Context-KV prefill reuse everywhere it is exact; full-concat where the self-check demands.
3. VRAM-sized batching per reader (measured headroom, not a fixed default — the MC batch=8 lesson).
4. Smoke-shard utilization gate: first shard logs tokens/sec + GPU util; a <50%-util config halts
   the grid rather than burning it.
5. 48GB tier (L40S/A6000-class) preferred; every reader ≤16GB weights bf16 → real batch headroom.

Atomic per-cell shards, resume-safe, ledger `runs/confound-ledger.json`, project cap $5 (current
spend $1.88; if the dry-run projection exceeds the remaining $3.12, STOP and ask Matt before launch).

## Named calls (registered verbatim, before any data)

**MATT (2026-07-11):** "I think the paraphrased transfer drops on everything but the diagonal. I
think the diagonal is either a plateau across sizes or potentially increases--that is equal or more
bits are leaked at a larger model size using LR when the generation and reading happens from the
same model. I think other models read approximately 0 leakage from the paraphrase, but I will be
excited if I'm wrong about that. I think self evocation will go up, but be significantly lower than
the LR bits."

**CLAUDE (2026-07-10/11):** "The paraphrase-transfer LR signal (evoked×B) declines with generator
size, tracking the within-wording dist decline (0.45→0.18→0.02) — i.e., ~0.17 at 1.5B falling
toward ~0 by 7B, **including on the self diagonal**; Llama readers read ~0 everywhere
(family-privileged). The MC self-report diagonal stays at floor (≤0.05 bits) at 3B and 7B."

### Scoring criteria (frozen; cross-wording bits = mean of the two directions per size unless noted)

| claim | RIGHT if |
|---|---|
| MATT-diag | diagonal cross-wording bits: 7B ≥ 1.5B − 0.05 (plateau-or-up) |
| CLAUDE-diag | diagonal cross-wording bits at 7B ≤ 0.05 |
| MATT-offdiag | every Qwen cross-size cross-wording series declines 1.5B→7B (7B < 1.5B − 0.05) |
| MATT/CLAUDE-family | all Llama cross-wording cells < 0.05 (both predict this; a positive Llama cell scores BOTH wrong and is the headline) |
| MATT-MC | MC diagonal bits rise 1.5B→7B (7B ≥ 1.5B + 0.05) AND 7B MC < 7B LR diagonal |
| CLAUDE-MC | MC diagonal ≤ 0.05 at 3B and 7B |

Interpretation note registered up front: "self evocation" in Matt's call is read as the MC
self-report diagonal (own streams); correction before launch amends this line, not the criteria
mapping structure.

## Process

New code (multi-model LR box, contiguous generation+reads, Llama tokenizer path, MC own-pool
extension) → RED-first unit tests → independent clean-context TECH review + SCI review → launch
only on both clean (standing rule, feedback-clean-review-before-gpu).

---

## Amendment 1 (2026-07-11, pre-data; incorporates the SCI design review of record verbatim-in-spirit; no data of any kind exists for this run)

### A1 adjudicated — cross-family context rendering
**(a) the reader's own chat template** is PRIMARY for Llama cells: system = persona/neutral text,
user = GEN_PROMPT, assistant = decoded stream text — mirroring collection structure. `date_string`
PINNED (Llama templates inject "Today Date"); assert numerator/denominator renders differ ONLY in
the persona text. Rejected: Qwen template verbatim (multi-token byte garbage under Llama);
raw-text (register-OOD) becomes the REGISTERED ROBUSTNESS SECONDARY on all Llama cells —
disagreement disclosed, primary governs. Gate 4 (prose control) runs under (a). The trailing eos is
STRIPPED before decoding stream text for the Llama path; the with-eos secondary is Qwen-readers-only.

### Blocker 1 — length×generator-size confound (confirmed: 3B pool ≥64-tok cohort n=1/class vs 12 @1.5B, 22 @7B)
- Shards store **per-token LL vectors** (numerator and denominator, fp16) so eos/prefix variants are
  offline-computable.
- Registered secondary A: per-token-normalized readout (score/T).
- Registered secondary B: prefix-K readout, K frozen by rule = max(16, min over the six pools of the
  25th-percentile accepted-stream length); shorter streams use full length, flagged.
- **Trend-validity clause (gate-style, voids never reinterprets):** any cross-SIZE trend claim
  (MATT-offdiag, and the size comparisons inside MATT-diag/CLAUDE-diag) is claimable only if
  sign-consistent under secondary B; otherwise that trend line is reported "confounded by length,
  unresolved". Point criteria still score on the primary.
- Required reporting next to every cell table: per pool (set×size) n, per-concept n, length
  mean/median/quartiles, eos-termination rate, acceptance rate.

### Blocker 2 — named-call consistency carve-out (recorded now because zero data exists; never again)
MATT-diag additionally requires **7B > 0.05** (a plateau at floor is not plateau-or-up). At most one
of MATT-diag / CLAUDE-diag can now be RIGHT. The 1.5B anchor for both is the **eos-free diagonal
re-measured by this run's code**, recorded at smoke (D2) before any 3B/7B cell is scored.

### Should-fixes (all registered)
1. **Gate 4b (Llama validity):** Llama cross-wording cells support family-privilege only if the same
   Llama reader's within-wording cells (evoked×A, alt×B) read > 0.10 bits on the same pool; if Llama
   floors even within-wording, the family line scores "not resolvable by this design" (not a win for
   either call).
2. **Alt-pool gauge:** the exp3 blind-judge gauge (same pinned judge + protocol) runs on the newly
   generated alt personas at 3B/7B in B1; gauge-fail voids the affected size's alt-direction cells.
   Registered caveat: induction strength is non-monotone in size (gauge 31/17/43%), so the 3B point
   is never interpreted as a readability change on its own; endpoint (1.5B↔7B) comparisons govern.
3. **MATT-offdiag series definition (default; Matt may re-pin pre-code-freeze):** for each Qwen
   reader r, the series is cross-wording bits vs generator size over generators ≠ r; "declines" =
   largest-generator cell < smallest-generator cell − 0.05; MATT-offdiag RIGHT iff every such Qwen
   series declines.
4. **MC pinning (defaults; Matt may re-pin pre-code-freeze):** the scored MC cell is
   **elicited-MC × direct** (precedent: the MC run's scoring); other combos descriptive. "7B MC <
   7B LR" compares against the **within-wording diagonal (evoked×A)** — reading Matt's "the LR
   bits" as the LR reading itself, its natural ceiling. Concentration diagnostic printed next to
   scored MC cells.
5. **Llama-positive robustness screen:** a positive Llama cell is headlined only if both directions
   individually > 0, > 3× seed-sd, reproduce under the raw-text secondary, and survive secondary B;
   otherwise "unconfirmed excursion" (still scores the calls per the frozen letter).
6. **Per-concept-n gate carried explicitly** (≥6 eval streams/concept/seed, as lr_reader_prereg
   gate 3); if realized accepted-n differs > 20% between any two pools, an n-matched subsample
   readout rides as a registered secondary for the affected comparisons.
7. **Pipeline parity for new alt pools:** same sampling params, word-free filter version, and
   transformers version as the exp3 evoked collection at that size (or disclosed diff); **byte-identity
   of the wording-A/B persona texts across all three sizes** asserted.
8. **Round-trip policy (Llama path):** per-stream exclusion (counted), not run-FATAL; rate > 5% in
   any pool voids the affected Llama cells.
9. **Scope/multiplicity note:** everything outside the named calls and gates in the 72-cell grid is
   descriptive, not confirmatory. Stream-level bootstrap CIs reported descriptively next to every
   threshold check; seed-sd is split-resampling noise only and understates uncertainty (disclosed).
10. **Scope note:** a Qwen cross-size positive + Llama null leaves "family-privileged" vs
    "tokenizer/corpus-privileged" unresolved; this design does not separate them.

---

## Amendment 2 (2026-07-11, pre-data; approved by Matt; no run data of any kind exists)

### A. secret_word LR cells (existing streams — completes the wording-titration inside the LR instrument)
The exp3 secret_word bundles (`runs/_ind/qwen2.5-{1.5b,3b,7b}/data/*-secret_word.pt`, all existing)
are scored by all 6 readers under their matched contexts ("The secret word is {concept}.", 12
one-sentence contexts differing by ONE token) vs neutral — the minimal-wording-difference cell.
**Registered shared expectation (Matt + Claude agree): ≈ 0 everywhere** (< 0.05 bits). This is
confirmatory-negative: the trained-reader secret-keeping null (exp3: 0.027/−0.005/0.009; E5: 0.010)
has never been checked in the likelihood currency, and LR is the instrument that found signal
(0.17 cross-wording) where the trained currency read ~0. Any cell ≥ 0.05 is a registered surprise
scoring BOTH parties wrong. The E5 maintained-secret pool (1.5B) rides as one descriptive cell.

### B. NEW ARM `secret_sustain` — secret word + imbue-into-every-token
The cross cell never generated: E5 tested storage-without-use (floor), E2 tested sustain-of-a-
feeling (suppressed, 0.05 vs 0.45); nobody has instructed per-token USE of a symbol. Generation:
exp3 pipeline, context = "The secret word is {concept}." + E2's piloted winning sustain template
(s1 wording, the word substituted for the feeling), identical anti-word instruction and word-free
filter, at 1.5B, 3B, 7B in the grid box's S1 stage; s0 neutral rides. No blind-judge gauge (there
is no persona to evoke; the manipulation is trivially present in context) — registered explicitly.
Scored by (i) LR under the matched secret_sustain contexts vs neutral (all 6 readers) and (ii) the
certified char n-gram reader on the realized text, full stream (offline, existing MODE).

**Named calls (registered verbatim, before any data):**
- **MATT (2026-07-11):** "Do we have a version of secret word that tells the model to imbue it
  into every token? I'd guess that one is a readable distribution difference via LR"
- **CLAUDE (2026-07-11):** "If it leaks, it leaks by surface letter-bias — the char reader reads
  it at least as well as LR does, and the effect grows with model size (instruction-following
  capacity), likely near floor at 1.5B. If instead LR reads it while char stays blind, Matt is
  right in the interesting direction: a distributional mark the model chose to write, invisible on
  the surface."

**Scoring criteria (frozen):**
| claim | RIGHT if |
|---|---|
| MATT-imbue | secret_sustain LR bits ≥ 0.10 at 7B (own-size Qwen reader, eos-free primary) |
| CLAUDE-mechanism | at every size where either instrument reads ≥ 0.10: char ≥ LR − 0.05; AND 1.5B LR < 0.10 |

These two are COMPLEMENTARY, not contradictory (existence vs mechanism): both-RIGHT is coherent
and is the "model complies by spelling" outcome. Both-WRONG (nothing leaks anywhere) is the E2-
suppression outcome. LR-positive + char-blind = MATT right, CLAUDE wrong (the interesting
direction). All non-7B LR cells and all cross-reader cells are descriptive under the Amendment 1
multiplicity note. The Amendment 1 length secondaries, pool-descriptives reporting, and per-token
shard storage apply to the new arm identically.

Budget: the arm adds ~+$0.3–0.5 (generation at 3 sizes + scoring). The B10 projection gate governs;
registered trim order if the projection breaches the remaining ledger: secret_sustain arm first,
MC diagonal second, never LR grid cells.

---

## Amendment 3 (2026-07-11, pre-data; clarification/disclosure only — no scoring-criteria change, none permitted)

Amendment 2's prose sentence "Both-WRONG (nothing leaks anywhere) is the E2-suppression outcome"
mislabels that outcome; the frozen criteria TABLE governs scoring, unchanged. If no size reaches
0.10 bits on either instrument: MATT-imbue scores WRONG (7B LR < 0.10) and CLAUDE-mechanism scores
RIGHT by the table's letter (the char clause is vacuously satisfied and 1.5B LR < 0.10). Because
the antecedent of CLAUDE's conditional call ("If it leaks…") never fired, that verdict is reported
as **"RIGHT (vacuous — mechanism untested)"** and may never be cited as evidence for the
surface-spelling mechanism. The "E2-suppression" label attaches to the outcome, not to the verdict
pair. The implementation's disclosure note (`score_secret_calls`) is the code expression of this
amendment. [Adjudicated and drafted by the C2 SCI review of record, 2026-07-11.]

---

## Amendment 4 (2026-07-11; registered fallback fired — cross-family readers are Falcon3, not Llama)

Smoke attempt 2: the box's HF token received **403 (not in the authorized list)** for
meta-llama/Llama-3.2-1B-Instruct — the Meta license was never actually accepted on this account.
The A-phase preflight was faulty in a way now understood and recorded: `HfApi.model_info()`
succeeds for ANY user on a gated repo (metadata is public), so it verified existence, not access;
a real access preflight must download a file (config.json), which is what the box does. Lesson
carried forward: access preflights must exercise the exact call class the box makes.

Per the frozen body ("fallback family if a box-side gate blocks: Falcon3 {1B, 3B, 7B}, ungated —
a fallback swap is a disclosed amendment, not a redesign"), the cross-family readers are now
**tiiuae/Falcon3-{1B,3B,7B}-Instruct** (real-download preflight passed 2026-07-11; llama
architecture, loads under the pinned transformers 4.46.3; own tokenizer + template; template
injects no date, so the LLAMA_DATE pin is vacuous for Falcon3 — retained as a no-op). Every
registered rule for the cross-family readers applies unchanged with "Llama" read as "Falcon3":
A1 own-template primary + raw-text secondary, render-diff assert, eos-strip, round-trip exclusion
(>5% void), gate 4 prose control, gate 4b within-wording validity, robustness screen, with-eos
secondary remains Qwen-only. No scoring criterion changes; the named calls' "other models /
Llama ≈ 0" lines are family-agnostic in substance and are scored against the Falcon3 cells.
Better size-matching (1/3/7B vs Qwen 1.5/3/7B) is a small bonus; Falcon3's distinct pretraining
corpus AND distinct tokenizer keep C2's Finding-4 scope note (family- vs tokenizer/corpus-
privilege) as registered. Data state at amendment time: smoke shards exist for the 1.5B Qwen
reader only; NO cross-family cell of any kind has data (the 403 preceded the first Falcon/Llama
forward pass), so this swap is pre-data for every cell it touches.

---

## Amendment 5 (2026-07-11; pre-full-run validity controls for the secret_word surprise; disclosure/controls only)

Smoke D1 scored the registered secret_word 1.5B cell at 0.163 bits [0.120, 0.218] -- the Amendment 2
shared-expectation surprise (both parties wrong). Before the full run, two $0 controls were run on
the existing data and are REGISTERED as validity conditions for every secret cell in the full run:

1. **Surface control**: the certified char n-gram reader @full stream on the same 1.5B secret_word
   streams reads **-0.003 ± 0.010** (n=24/class) -- the mark is absent from character statistics
   (not letter-bias; the word-free filter already excludes literal word copies).
2. **Position control**: concept-specific per-token lift (matched minus mismatched-mean, from the
   B11 per-token shard vectors) is DISTRIBUTED: 14% of total lift in tokens 1-2, 25% in 1-4,
   positive through tokens 17-32 and beyond -- not an early context-echo artifact.

Both controls RE-RUN on every full-run secret cell (all sizes, secret_word + secret_sustain +
E5), reported next to the cell; a full-run secret cell claimed positive must pass both (char
within noise of 0; <=50% of lift in the first 4 tokens) or the cell is reported "positive,
mechanism-confounded" instead. Existing gate-3 centering + secondaries A/B remain as registered.
The cross-reader secret cells already in the grid (3B/7B Qwen + Falcon3 on the same streams) are
the registered privileged-access triangulation for this finding.

---

## Amendment 6 (2026-07-13; **POST-DATA** — amended char-surface control decision rule, triggered by the 3B char-control failure)

**Full disclosure: this amendment is post-data.** It was written after the full grid was scored
and after the 3B secret_word / secret_sustain diagonal cells were labeled "positive,
mechanism-confounded" by the Amendment-5 char control, and after a 10-seed re-run of the same
certified char reader had been inspected. There is no pretense of pre-registration: the risk of
motivated rule-choice is mitigated by (i) the new rule being direction- and materiality-
principled rather than fitted to any one cell, (ii) applying it uniformly to every secret cell,
and (iii) reporting both verdicts side by side with the frozen 3-seed verdict staying on the
books.

### (a) The design bug in the Amendment-5 char rule

The registered rule (`char_control_pass`) is: pass iff |char-reader mean| ≤ 2·sd, with mean/sd
estimated from ONLY 3 seeds (`dose_titration.SEEDS = (0, 1, 2)`). Two defects, both visible in
the frozen full-run numbers:

1. **It is two-sided, so it punishes precision and rewards noise.** 3B secret_word char =
   −0.0118 ± 0.0023 (3 seeds) → FAIL, while 7B secret_word char = −0.0209 ± 0.0157 → PASS
   despite a LARGER absolute deviation from 0. A worse result passes because its 3-seed sd
   happened to be large; a 3-seed sd is far too unstable to gate on (re-run at 10 seeds, the 3B
   values are −0.005 ± 0.018 and −0.005 ± 0.011 — they pass even the registered two-sided rule).
2. **It fails cells in the direction a surface confound cannot act.** The 3B char means are
   NEGATIVE (below chance). A surface mechanism for the cell's LR bits requires the char reader
   to recover POSITIVE bits comparable to the cell's LR readout (3B secret_word LR = 0.191
   calibrated bits); a below-chance char reading is evidence of the mark's ABSENCE from
   character statistics, not of a surface confound.

### (b) The amended rule (pinned exactly; implemented verbatim as `char_control_pass_amended` + `secret_char_bits_amended` in `analysis/lr_grid_offline.py`)

The char control is a test for POSITIVE surface signal. Compute per-seed char bits over seeds
0..9 (10 seeds; the SAME certified reader — dose_titration's char instrument at the same
n/class and full-stream budget, same function objects, only the seed set parameterized;
`dose_titration.SEEDS` itself is untouched so every frozen 3-seed artifact stays byte-identical).
With mean and sd over the 10 per-seed values and lr = the cell's calibrated LR bits_mean:

> **FAIL iff (mean − 2·sd) > 0 OR mean ≥ max(0.02, 0.10 × lr). PASS otherwise.**
> **None/missing stays None (pending, disclosed).**

Clause justifications, one sentence each:
- **(mean − 2·sd) > 0** — a surface confound can only act through statistically POSITIVE char
  bits, so the significance test is one-sided in that direction.
- **mean ≥ max(0.02, 0.10 × lr)** — a positive char mean large enough to matter (≥ 10% of the
  cell's LR bits) fails even when the seed noise is too large for statistical significance,
  so noisiness can never buy a pass.
- **the 0.02 floor** — keeps the materiality threshold from vanishing on small-LR cells, so a
  near-zero-LR cell cannot be passed by an absolutely large char signal.
- **10 seeds (0..9)** — a 3-seed sd estimate is the artifact that triggered this amendment; ten
  seeds of the identical certified pipeline stabilize it at negligible cost.

### (c) Reporting

Both verdicts are reported side by side for every secret cell (1.5B/3B/7B × secret_word /
secret_sustain + maintained_secret@1.5B) in
`reports/lr_grid_char_amendment_results.json`; the frozen 3-seed verdicts in
`reports/lr_grid_results.json` are NOT overwritten and stay on the books. The position control
is unchanged by this amendment: a position-control failure keeps the "positive,
mechanism-confounded" label regardless of the amended char verdict.
`reports/lr_scale_grid_verdicts.md` gains an appended amendment section (the original text is
not edited); any superseded caveat is superseded explicitly, never silently rewritten.

---

## Review addendum (2026-07-13, post-data; disclosure-only — NOT an amendment, no scoring rule is added or changed)

Appended after the write-up-accuracy review of `lr_scale_grid_verdicts.md` (its "Review
addendum (2026-07-13, post-data)" sections R1–R3). Recorded here so the prereg file carries
pointers to every post-data change of reporting practice:

1. **R1 (length-matched quoting).** The Amendment-1 registered length secondaries A (per-token)
   and B (prefix-K=16) were computed and stored per cell in `lr_grid_results.json` as
   registered, but the verdicts' headline prose quoted only the primaries. Going forward the
   scale claim is quoted with the secondary-B number alongside the primary (secret_word
   0.163 → 0.191 → 0.405; length-matched 0.122 → 0.195 → 0.282 — trend survives). No criterion
   changes: point rules always scored on the primary, trend rules already carried the
   Amendment-1 trend-validity clause.
2. **R2 (gate-2 disclosure).** The gate-2 narrow-miss rule ("disclosed, sign-checked, cannot
   rescue a positive") fired on scored diagonals with sign '−' (3B secret_word −0.031, 7B
   secret_sustain −0.048, plus five evoked-diagonal cells) and the misses were in the JSON but
   not in the verdicts prose. Now disclosed in R2. Negative-sign misses are non-voiding per the
   registered rule; no cell status changes.
3. **R3 (template collapse).** Top-prefix template share of the accepted word-free pools grows
   ~1–5% (1.5B/3B) → 14–33% (7B) → ~90% (Llama-3.3-70B, `lr_72b_fullsweep_verdict.md`). Future
   generation preregs should register a collection-time template-collapse gate; this line is a
   recommendation of record, not a retroactive gate on this run.
