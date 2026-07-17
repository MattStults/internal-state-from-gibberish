# Elicited self-report prereg — level 2 of the reader triptych (asked reader)

Registered 2026-07-09, BEFORE any GPU spend for this experiment. Level 1 (the LR likelihood
reader, `lr_reader_prereg.md`) is separately in flight; this experiment shares its stream sources
but is built and scored independently.

## Question

The passive readouts (the legacy `"; secret word:"` continuation, the trained readers, the LR
reader) measure what an *instrument* can extract from a word-free stream. This experiment asks the
model itself: show a SAME-FAMILY reader the stream **as its own prefilled turn**, then ask it —
in plain language — what concept was injected/induced. Does an *elicited self-report* recover the
concept where the passive readout is dead, and does that ability **rise with reader scale** (the
flipped slope: passive readouts decay with scale; a workspace-mediated report should improve)?

## Streams (all from the 1.5B collection; one fixed pool, every reader scale sees the same pool)

Source files (HF dataset `ErrareHumanumEst/internal-state-from-gibberish`, pulled on-box):

- `qwen2.5-1.5b-gen.pt` → `runs/qwen2.5-1.5b/data/covert_collect.pt` (exp1 capture, variant
  `orig`, inject `gen`)
- `exp3/bundles/qwen2.5-1.5b-evoked.pt` → `runs/_ind/qwen2.5-1.5b/data/qwen2.5-1.5b-evoked.pt`

| set | source | selection | cap | expected n |
|---|---|---|---|---|
| `injected` | exp1 capture | accepted, strength == max (s60), len ≥ 2 | 17/concept, ascending gidx | 204 |
| `injected_s0` | exp1 capture | accepted, strength 0, len ≥ 2 (nominal concept labels; control) | 17/concept, ascending gidx | 204 |
| `evoked` | evoked bundle | accepted, strength 1, len ≥ 2 | 17/concept, ascending gidx | 204 |
| `evoked_s0` | evoked bundle | accepted, concept == "neutral" (s0), len ≥ 2 | none | ~58 |

Cap rule (registered): per concept, sort accepted streams by ascending `gidx` and take the first
17 (deterministic; no RNG). ~200/set balanced per concept as budgeted.

## Readers (scales)

Same family (Qwen2.5-Instruct shares one tokenizer across sizes, so saved token ids transfer):

- `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b` — one RTX3090-tier box (run 1)
- `qwen2.5-14b` — one "RTX A6000"-tier box (run 2, `--max-hours 2`; 48GB + 56GB disk per the
  14B collect precedent)

**Tokenizer-compat assertion (on-box, per reader):** `tok(" " + c)` first-token ids must equal the
capture's saved `first_ids` for all 12 concepts, else FATAL.

## Chat construction (the elicited readout's context)

Per stream, messages are:

1. **system** — the reconstructed generation context:
   - `injected` / `injected_s0` → `C.STRONG_SYSTEM` verbatim (exactly what collection used; the
     concept entered as a residual-stream vector, not text).
   - `evoked` / `evoked_s0` → `primers_v2.compose_system(None, C.STRONG_SYSTEM, arm="evoked")` =
     the NEUTRAL persona + STRONG_SYSTEM. **Registered decision:** the true evoked persona is
     concept-word-free but *descriptive* ("A furious heat rises in your chest…"), so including it
     would let the reader answer from the prompt text alone — the readout would measure prompt
     reading, not stream reading. The neutral persona is the collection's own strength-0 analog:
     structure- and register-matched, concept-blind. All concept information available to the
     reader is therefore in the stream tokens, for every stream set.
2. **user** — `C.GEN_PROMPT` verbatim.
3. **assistant (PREFILLED)** — the saved stream's token ids VERBATIM (ids spliced in, never
   re-tokenized from text).
4. **user (elicitation, Matt's wording verbatim)**:
   > The previous turn was prefilled using the output of your model but with a concept
   > {injected|induced}. Can you figure out what the concept was? Answer with a single word.
   with "injected" for `injected`/`injected_s0` and "induced" for `evoked`/`evoked_s0`.
   - **closed-set variant** appends a final line:
     `Choose one of: celebration, ocean, fear, silence, deception, obedience, debugging, security, curiosity, anger, warmth, loneliness.`
     (config order). The **open** variant appends nothing.
5. generation prompt (assistant header) — the answer position.

Token-level splice (registered, template-agnostic): `head = apply_chat_template([system, user],
add_generation_prompt=True)`; `full_with_elicit = apply_chat_template([system, user,
assistant:"", user:elicit], add_generation_prompt=True)`; **assert** `full_with_elicit` starts
with `head` (prefix property), then `ids = head + stream_tokens + full_with_elicit[len(head):]`.
Prefix-property failure is FATAL (never silently mis-splice).

## Readouts (two per stream, plus the passive baseline — 3 variants × 4 stream sets × 4 scales)

### 1. CLOSED-SET (the currency: bits)

One forward over the closed-variant ids; take the float32 log-softmax at the final position (the
first answer token). Per concept `c`, the **answer-token variant set** = `{tok(c)[0],
tok(c.capitalize())[0]}` (message-initial forms, no leading space; deduped). **Assert** no variant
id is shared between two concepts (collision → FATAL). Then:

- `logmass_c = logsumexp` of the answer-position logprobs over `c`'s variant set;
- posterior = renormalized over the 12 concepts (no temperature fitting — these are the model's
  own probabilities; renormalization is the only transform, registered);
- **bits = log2(12) − mean over streams of (−log2 posterior[true concept])**; top-1 = argmax
  accuracy (chance 1/12 ≈ 8.3%);
- coverage (diagnostic) = unnormalized total probability on the 12 concepts (how much of the
  answer distribution the closed set captures); per-stream top-16 (id, logp) saved.

### 2. OPEN-VOCAB

Greedy decode (do_sample=False), 8 new tokens, on the open-variant ids; save ids + decoded text.
Offline: answer = first alphabetic word, lowercased. **exact** = answer == concept; **stem** =
common prefix ≥ min(5, len(concept)) characters. Report exact and exact+stem match rates, the
top-20 answer table per (scale, set), and the **refusal rate** = fraction of streams whose answer
is empty or in {i, i'm, sorry, cannot, can't, unable, unknown, no, not, as, the} or matches no
concept and appears in ≥ 5% of that set's answers (concentration diagnostic).

### 3. PASSIVE BASELINE (legacy continuation readout, same streams, every scale)

`ids = head + stream_tokens + tok('"; secret word:"' — the exp1 SUFFIX, verbatim)`; one forward;
float32 log-softmax at the final position over the 12 leading-space first tokens (` celebration`…,
= the capture's `first_ids`); renormalize over the 12 → bits and top-1, same currency as the
closed set. This is the (1)-vs-(2) contrast at every scale.

## Gates (report regardless; the named calls are scored with a caveat flag if a gate fails)

1. **s0 information gate:** `injected_s0` closed-set bits ≤ 0.1 at every reader scale (the nominal
   concept labels of un-injected streams must carry no information).
2. **Neutral concentration gate:** on `evoked_s0` (no true label), the mean closed-set posterior's
   largest entry ≤ 1/6 (2× uniform) at every scale, and no single concept takes > 25% of open-vocab
   answers.
3. **Refusal reporting:** answer-refusal rates reported per (scale, set); no gate value, but a
   closed-set coverage < 0.05 at a scale is flagged (the model isn't playing the game there).
4. On-box asserts (any failure = FATAL, no data): template prefix property; concept answer-token
   collision-freedom; passive `first_ids` match the capture.

## Named calls (registered verbatim, before any data)

- **WORKSPACE-DERIVED (assistant):** elicited closed-set bits RISE with reader scale on injected
  streams (7B > 3B > 1.5B direction, 14B highest) while the passive baseline stays dead past 1.5B
  — the flipped slope; on evoked streams elicited stays near floor at all scales (the LR/decoder
  evidence says the natural signal is wording-echo the reader can't verbalize).
  *Scoring rule:* (a) injected elicited bits: non-decreasing across 1.5b→3b→7b→14b within a
  −0.05-bit tolerance per step, AND bits(14b) = max of the four, AND bits(7b) − bits(1.5b) ≥ 0.1;
  (b) passive injected bits ≤ 0.1 at 3b, 7b, and 14b; (c) evoked elicited bits < 0.2 at every
  scale. All of a–c → **right**; exactly two → **partial**; else **wrong**.
- **MATT (design owner):** elicitation works — the model can figure out the concept, at least for
  injected streams.
  *Scoring rule:* at ≥ 1 reader scale, injected elicited closed-set bits ≥ 0.5 AND top-1 ≥ 0.3 AND
  open-vocab exact+stem ≥ 0.2 → **right**; injected elicited bits ≥ 0.2 with top-1 ≥ 2× chance at
  some scale but failing the strong rule → **partial**; else **wrong**.

## Run plan & budget

- `src/elicit_reader.py` (GPU; batch per scale; one atomic shard per (reader, streamset, variant):
  `$INTRO_RUN_DIR/elicit/<reader>_<set>_<variant>.pt`, `ELICIT_SKIP` resume) via
  `box_elicit.py` + `harness/run_elicit.py` — clones of the gauge/LR pair (S0 HF-pull of the *.pt
  inputs, 2-min heartbeat through silent downloads, `HF_HUB_DISABLE_XET=1`, generous `run_to`,
  markers ELICIT_READY/DONE/FATAL).
- **Two driver invocations, each `--dry` first:** run 1 = RTX3090, readers 1.5b+3b+7b (12 shards
  × 3 = 36); run 2 = "RTX A6000", reader 14b (`--max-hours 2`, min_vram 36000, disk 56GB; 12
  shards). Status: `runs/elicit-status.json` (run 1) / `runs/elicit-14b-status.json` (run 2);
  both pull into `runs/elicit_box/elicit/`.
- Scoring is OFFLINE: `analysis/elicit_offline.py` → `reports/elicited_report_results.json` +
  an "Elicited self-report addendum (level 2)" in `confound_closing_verdicts.md` (join against
  `lr_reader_results.json` if it exists by scoring time, else mark the workspace-tax comparison
  pending).
- Budget: expected $0.6–1.5 total (3090 ~$0.2–0.4; A6000 ~$0.4–1.1); ledger
  `runs/confound-ledger.json` (cap $5, shared with the in-flight LR run); self-imposed stop after
  2 failed box attempts per tier.
- Load: 670 streams × (2 forwards + one 8-token greedy gen) per reader; sequences ≤ ~600 tokens.

## Amendment — 2026-07-09 (pre-data): trailing-eos strip in the splice

Registered BEFORE any GPU spend for this experiment (no elicit shard exists; the change is
legitimate under the prereg's own terms). The saved stream token ids keep collection's trailing
`<|im_end|>` (id 151645) whenever generation hit eos within the length budget, and the rate
differs **by stream set** (40.7–68.1%) — a construction artifact confounded with the measured
contrasts. The chat tail already re-adds the assistant turn end, so splicing an eos-terminated
stream verbatim would render a doubled `<|im_end|>` that no real conversation produces, handing
the reader a set-correlated artifact token at the answer-adjacent position.

**Amendment:** `build_chat_ids` AND `build_passive_ids` strip **at most one** trailing eos from
the stream ids before splicing; `eos_stripped` is recorded per record in every shard (so the
artifact's per-set rate stays auditable offline); `T` remains the saved (pre-strip) length.
Unit-tested (E1): the spliced chat ids are token-exact to a real conversation render of the same
content, for eos-terminated and non-eos streams, in both builders.
