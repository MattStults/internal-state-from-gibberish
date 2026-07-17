# Related-work positioning: is this result "deeply understood"?

*Written 2026-07-14 against the repo state on `publish-prep`. Sources: this repo's frozen verdicts
(`lr_scale_grid_verdicts.md` incl. Amendment-6 + review addenda, `lr_72b_fullsweep_verdict.md` incl.
the 2026-07-14 faithful-template addendum, `NOTE_injection_LR_comparability.md`,
`exp3_induction_and_scale/reports/experiment.md`, `mc_reader_verdicts.md`, `experiment.md` (exp2),
`runs/lit-search-2026-07-09.md`), plus web verification of the five core external papers. Paper
details for arXiv:2311.13647, 2405.15012, 2410.03768, 2310.18512, 2507.14805, 2506.17090,
2510.15511 are from training knowledge cross-checked against search results; 2606.09135
(post-cutoff) is from its arXiv abstract via search. One brief-supplied result — cross-model lift
separation ≈ 0 (detection-vs-identification, measured 2026-07-14) — could not be located in any
frozen repo artifact; rows using it are flagged.*

## The papers, one line each

- **Mächtle et al. 2026, arXiv:2606.09135 "Steganography Without Modification"** — a *sender*
  encodes a payload in the PRNG **seed** before generation; the receiver reconstructs the
  seed-dependent token-probability intervals (which requires the generating model, and in the
  strong mode the prompt) and recovers the payload by seed search. Deliberate, engineered channel;
  the text and the output distribution are *unmodified*.
- **Morris et al. 2023, arXiv:2311.13647 "Language Model Inversion" (logit2prompt)** — a trained
  T5-base inverter reads the victim's **next-token probability vector** and reconstructs the hidden
  prompt (Llama-2-7B: BLEU 59, token-F1 78, ~27% exact match; trained on the ~2M-prompt
  Instructions-2M corpus). External trained reader of the *distribution*, not of sampled text.
- **Zhang, Morris & Shmatikov 2024, arXiv:2405.15012 "Extracting Prompts by Inverting LLM
  Outputs" (output2prompt)** — a T5-base (222M-param) inverter with sparse encoding reads **64
  sampled text outputs** per target prompt (no logits, benign queries only), trained on
  Instructions-2M-scale data; recovers user and system prompts (cosine-sim ≈ 0.8–0.9+),
  **zero-shot transfer across victim models**.
- **Mathew et al. 2024, arXiv:2410.03768 "Hidden in Plain Text"** — steganographic **collusion**
  emerges un-asked-for under RL optimization pressure (GBRL and ICRL elicitation); the resulting
  encodings can evade paraphrasing and oversight. Optimized, goal-directed encoding in natural text.
- **Roger & Greenblatt 2023, arXiv:2310.18512 "Preventing Language Models From Hiding Their
  Reasoning"** — models can be *fine-tuned* to carry encoded reasoning in natural text; paraphrasing
  is evaluated as the defense and bounds the residual channel capacity. Deliberate/trained encoding
  with a shared codebook.

Adjacent (from the 2026-07-09 lit sweep + training knowledge): **Cloud et al. 2025,
arXiv:2507.14805 (Subliminal Learning)** — traits transmit through semantically unrelated data
(number sequences) via *fine-tuning*, and only between models sharing an initialization;
**Holtzman & West 2026, arXiv:2605.10794 ("Can You Keep a Secret?")** — a text-only detector
recovers a hidden secret *thematically* from unconstrained writing, up to 79% above chance, and
that leakage **grows with scale**; **PILS, arXiv:2506.17090** — stronger LM inversion from
compactly-represented logprob *sequences* (2–3.5× exact-match gains over logit2prompt);
**arXiv:2510.15511** — hidden **activations** are injective/invertible (white-box, not output);
**arXiv:2511.21399** (steering detection from the steered model's own activations) and
**arXiv:2606.06315** (deliberate text-carried steering watermark).

## 1. Claim-by-claim table

Verdict key: **COVERED** = a knowledgeable reviewer would say the literature already establishes
this; **PARTIAL** = an adjacent result exists that a reviewer will cite, but it differs in a
load-bearing way; **UNTOUCHED** = no paper found that measures this.

| # | Project claim (repo evidence) | 2606.09135 seed-stego | 2311.13647 / 2405.15012 / PILS inversion | 2410.03768 / 2310.18512 stego | SL 2507.14805 / H&W 2605.10794 | Verdict |
|---|---|---|---|---|---|---|
| 0 | *Umbrella:* model outputs / distributions carry recoverable information about the conditioning prompt at all | yes (channel exists in sampling) | **yes — this is their whole result** | yes (after optimization) | yes (semantic channel; distillation channel) | **COVERED** — never present this as the finding |
| 1 | **Emergent conditioning-mark in word-free text**: merely *telling* the model a secret makes it write a concept-identifying mark into forced letter-gibberish, no instruction to communicate, no optimization (LR diag 0.163/0.191/0.405 bits, top-1 25% vs 8.3%) | no — payload is planted in the seed by an external sender | no — they invert *natural* outputs / logit vectors of the prompt itself; never semantics-stripped output | no — encoding is trained/RL-optimized, goal-directed | **partial** — SL: passive trait marks in unrelated data, but only readable via *fine-tuning* a same-init student, no per-stream readout; H&W: passive secret leakage, but via the *semantic* channel word-free-ness removes | **PARTIAL** |
| 2 | **Self-privileged readability**: the mark is read by the generating model's *own likelihoods* (12-way calibrated LR), with no training and no shared codebook | **partial** — decoding also requires the generating model (interval reconstruction), but for an *engineered seed*, not model state | no — their readers are *external trained* inverters; if anything they show the opposite polarity (external readers succeed on distributions/text) | no — decoder is the colluding agent / codebook holder | **partial** — SL's same-initialization requirement is the gradient-space analog of "only the generator reads it" | **PARTIAL** |
| 3 | **Char-blindness** of the natural mark: char n-gram readers ≈ 0 on the same streams where LR reads 0.16–1.11 bits (Amendment-6 10-seed controls; caveat: 7B sustain char ≈ +0.04, ~4% of the cell, disclosed) | not tested | not tested (no surface-reader baselines on constrained text) | opposite regime (their encodings *are* surface text) | H&W's reader is surface-semantic and *succeeds* — on unconstrained text | **UNTOUCHED** in this setting |
| 4 | **Cross-model / cross-family blindness incl. larger readers**: same-family siblings read ≈ 0 off-diagonal where cleanly scored [CORRECTED 2026-07-17 — the original "up to 4.7× larger" cited a VOID-gate3 cell; the largest cleanly-scored bigger-reader ratios are 2.3× (grid) and 2.0× (extension); 7 of 18 grid secret_word cells are instrument-void — see the grid verdicts' void-disclosure addendum]; Falcon ≈ 0 (but Falcon null is instrument-unresolvable — it can't read Qwen gibberish even within-wording, caveat 4) | no | **cuts against** — output2prompt transfers zero-shot *across* victim models (on natural outputs) | no | **partial** — SL fails cross-init/cross-family; but different-size Qwen siblings are also different inits, so SL *predicts* our off-diagonal null rather than having measured it | **PARTIAL** |
| 5 | **Trained-decoder blindness**: the exp3 trained `dist`/`char`/`R_emb` decoders read secret_word ≈ 0 (0.027/−0.005/0.009 bits) where own-likelihood LR reads 0.163–0.405 | no | **the attack exists and is untested here** — output2prompt-class inverters (222M params, ~2M-prompt training, 64 streams/target) vastly exceed our reader budget (nested-CV logistic/PCA, **288 streams total, 24/class, single-stream input, top-64 logprobs at T=12**) | 2310.18512's paraphrase-capacity analysis is about deliberate channels | no | **PARTIAL — covered as an attack, not as a result; our null does not yet survive it** (Objection a) |
| 6 | **Capacity grows with scale**: diagonal 0.163→0.191→0.405 (length-matched K=16: 0.122→0.195→0.282, monotone survives) | no | no scale claims for this channel | no | **partial** — H&W leakage grows with scale, but through the semantic channel; our own *evoked* (wording-residue) channel *shrinks* with scale — the growth claim is specific to the self-read distributional mark | **PARTIAL** |
| 7 | **Sustain amplification**: "imbue it into every keystroke" → 1.109 bits at 7B (0.88 length-matched), ~8× the 1.5B cell; storage-without-use (maintenance arm) stays at floor | no | no | adjacent in spirit (instructed use opens channels) but never measured as likelihood-readable bits in word-free text | no — and note our *evoked+sustain* variant went *down* (0.05 vs 0.45), the opposite sign, disclosed | **UNTOUCHED** |
| 8 | **Regime split vs injection**: injected/steered marks are *public* (char@full ≈ 1.2–2.1 bits at 1.5B–14B, PLATEAU at 14B, dose-controlled) and ≈ 0 to own-likelihood LR; prompted secret marks are *private* (LR-only, char ≈ 0). A double dissociation on the same word-free substrate | no | no | no (single regime each) | no | **UNTOUCHED** — no paper contrasts conditioning *routes* by reader access |
| 9 | **Detection-vs-identification asymmetry**: generator's own likelihoods show +1.39-nat generic lift (detects "some conditioning") at 70B even where identification fails; cross-model lift separation ≈ 0 (*brief-supplied 2026-07-14 measurement — no frozen repo artifact located; do not publish without one*) | no | no | monitors in the collusion lit detect *content*, not conditioning-per-se | 2511.21399 detects steering from *internal activations*; 2512.03816 tracks model changes via logprobs, not concepts | **PARTIAL** (and evidentially thin on our side) |
| 10 | **Cross-family frontier positive through mode collapse**: Llama-3.3-70B faithful-template self-read = 0.244 bits (sustain) / 0.143 (evoked) through ~90% template collapse; bare secret_word null (format-exonerated, mode-collapse/quantization-qualified) [CORRECTED 2026-07-17 — "70B privacy untested" is stale: the extension run's Amendment-1 rider had Qwen 1.5B–14B read the same 810 Llama streams; every cleanly-scored cell ≈ 0 (−0.002…+0.002 bits), though 3 of 8 confirmatory rider cells are VOID-gate3 (`lr_extend_results.json`); no *larger*-than-70B reader has been tested] | no | no | no | no | **UNTOUCHED** |
| 11 | **Instrument format-sensitivity**: plain-text echo vs token-faithful chat template flips 70B sustain from +0.013 to +0.244 bits — the teacher-forcing format is part of the instrument | no | adjacent lore — inversion work is careful about logprob representation (PILS's gains are representational), and eval-template sensitivity is known methodology folklore | no | no | **PARTIAL** (as methodology; the specific demonstration is ours) |
| 12 | *Boundary row (not claimed):* deliberate encoding into this channel by a motivated model | **covered** (seed channel) | — | **covered** (2310.18512, 2410.03768, TrojanStego 2505.20118, 2606.09411) | — | **COVERED** — keep the blog's "passive only; deliberate is untested *in this word-free setting*" wording (lit-sweep item 4) |

**Counts (rows 0–12): 2 covered · 7 partial · 4 untouched.** The two covered rows are the umbrella
phenomenon and the deliberate-encoding boundary — neither is what the project claims as its result.

## 2. Delta statement

> Prior work shows that LLM outputs and logits leak their prompts to *trained external inverters*
> (arXiv:2311.13647, 2405.15012, 2506.17090), that *deliberately planted* covert payloads in
> unmodified sampling are decodable with generating-model access (arXiv:2606.09135), and that
> hidden traits or secrets escape through *semantic* text channels or *shared-initialization
> distillation* (arXiv:2605.10794, 2507.14805). What none of these measure is an **unengineered
> mark that a merely-conditioned model writes into semantics-stripped output, whose readability is
> split by reader access**: calibrated per-stream bits recoverable by the generating model's own
> likelihoods with no training and no codebook (0.16→0.41 bits growing with scale, 1.11 under a
> sustain instruction, 0.24 cross-family at 70B through 90% mode collapse) while char n-gram
> readers, same-family siblings up to 2.3× larger (larger reader ratios came back instrument-void
> — unanswered, not ≈ 0), other families, and our (small-budget) trained
> decoders read ≈ 0 — and whose polarity *inverts* under activation steering, where the mark
> becomes public to a model-free character reader at every scale tested. The unit of novelty is
> this access-conditioned regime map — which conditioning route writes which kind of mark, and who
> can read it — not the existence of output-side leakage.

## 3. Reviewer objections, hardest first

**(a) "output2prompt shows trained inverters recover prompts from text outputs alone; your
trained-decoder cell is far weaker than their attack — so 'trained-decoder blindness' is
unsupported."** Correct, and quantifiably so. Their inverter: T5-base, **222M parameters**,
trained on an Instructions-2M-scale corpus of prompt→output pairs, reading **64 sampled outputs
per target**, and it transfers zero-shot across victim models. Our trained-decoder cells
(exp3, `experiment.md`): nested-CV **logistic regression** (PCA×C grid for `dist`, TF-IDF for
`char`), **288 streams total (24 per class)**, **single-stream input**, top-64 logprobs truncated
at T=12 for `dist`. That is ~3 orders of magnitude fewer parameters, ~4 orders less training
signal, and 1/64th the per-target evidence. Worse for us: the victim is *open-weight*, so a real
attacker's training budget is unbounded — they can query Qwen2.5-7B for as many labeled word-free
streams as they like. Until the §4 attack is run, the claim must be worded "invisible to the
readers we trained (shallow, few-shot)" — never "invisible to trained decoders."

**(b) "Your privacy nulls are absence-of-evidence at small reader-training budgets."** Partially
conceded in the repo already, and a reviewer will stack four instances: (i) the Falcon null is
explicitly *instrument-unresolvable* — Falcon reads ≈ 0 even within-wording, so it "can't engage
Qwen gibberish at all" (grid caveat 4); (ii) the off-diagonal Qwen cells are zero-shot likelihood
reads, not trained readers — no one tried to *train* a 7B-based reader on 1.5B streams; (iii) the
70B sustain/evoked marks have **no privacy measurement at all** (off-diagonal never collected —
flagged "newly open cell" in the 2026-07-14 addendum) [stale as of 2026-07-17: the extension
rider ran this — see the row-10 correction; the instance now reads "zero-shot likelihood reads
only, 3 of 8 confirmatory cells void"]; (iv) the 7B sustain char control shows a
likely-real ~0.04-bit surface component (Amendment-6 note), so "char-blind" is 96%-blind at the
strongest cell, not 100%. The defensible claim shape is a *lower-bound gap* ("no reader below
budget X reads it; the generator reads it for free"), not a universal negative.

**(c) "This is 12-way prompt identification self-scored by the generator — the inversion
literature already established that outputs identify their prompts, and your own paraphrase-
transfer test says the readable signal is wording-tied. So you've built a weak, closed-world
prompt-inverter."** [CORRECTED 2026-07-14 — the original draft claimed "transfer ≈ −0.08 bits
(failure)", which matches no certified artifact. The certified numbers
(`lr_scale_grid_verdicts.md`): within-wording evoked diagonal 0.604/0.542/0.791 bits at
1.5B/3B/7B vs cross-wording (evoked read under the evoked_alt paraphrase) 0.166/0.073/0.129 —
transfer is ~4–6× WEAKER but **persists above the 0.05 floor at 7B**; the registered CLAUDE-diag
call predicted collapse and was scored WRONG on exactly this.] The sharpest surviving version:
the readable natural-regime signal is *predominantly* wording-tied (a 4–6× drop under
paraphrase), and secret_word arms differ by exactly one word — so much of the LR readout behaves
like prompt-fingerprinting. Response available: say "conditioning-identity (predominantly
wording-tied; a smaller wording-independent component survives paraphrase)" rather than
"concept" —
and note the §7a run-(2) injection test is the wording-FREE version of this question: the
injected mark involves no tokens at all, so a positive self-read under re-injection is a
conditioning mark that *cannot* be a wording fingerprint (reported as its own quantity per
`NOTE_injection_LR_comparability.md`).
the finding that survives is *who can read the fingerprint* — Morris/Zhang readers are external
and trained, ours is the generator's raw likelihood, and the same external-reader class that
succeeds on natural outputs reads ≈ 0 here (pending objection a). Also note Morris et al. read
the *full next-token probability vector*; our privacy claims are about *sampled word-free text* —
the blog must never imply the distribution itself is private (our own exp2 `dist` decoder pulls
2.64 bits from injected distributions).

**(d) "The 70B result is one arm, one family, quantized serving, gate-3 near the bound (0.0177 vs
0.02), through 90% mode collapse, and scored by a review-path script."** All true and all already
disclosed in `lr_72b_fullsweep_verdict.md`; the fullsweep verdict itself models the right posture
(instrument-qualified claims). Publishing the 70B positive requires the official-adapter
confirmation pass named in its own caveat line.

**(e) "Detection-vs-identification is asserted from a measurement with no frozen artifact."** I
could not find the 2026-07-14 cross-model-lift-separation measurement in any committed report or
run artifact. Either freeze it (prereg-style note + JSON) or drop the row from the blog.

## 4. Minimal experiment set the literature demands

**E-A. The output2prompt-style trained-inverter attack on the existing word-free streams (the one
that closes objection (a)).**

- **Threat model** (matches Zhang et al., adapted to open weights): attacker knows the 12 candidate
  secrets and the generation prompt template, can query the victim model freely, sees only sampled
  word-free *text* at attack time (no logits — that is the regime where our privilege claim lives).
- **Training data**: synthesize with the victim itself — generate ~2,000 fresh word-free streams
  per concept per arm (secret_word, secret_sustain) at Qwen2.5-7B (and 1.5B for the cheap dry run),
  same reject-resample word-free filter, **fresh seeds disjoint from the frozen pools**. 12 × 2,000
  × ~128 tokens ≈ 3M tokens/arm — a few hours of batched vLLM on one RTX 4090/6000Ada (~$2–4 at
  current Vast rates).
- **Inverter**: ByT5-small (~300M) or T5-base (222M, parity with output2prompt) encoder +
  12-way classification head (byte-level tokenizer strongly preferred — the streams are
  character gibberish, exactly where subword tokenizers shred signal). Two input formats,
  both reported: (i) single-stream (parity with our LR reader — same per-stream currency);
  (ii) **K=64 concatenated streams per prediction** (parity with output2prompt's 64-output
  evidence budget; select K streams per label at eval time by resampling the frozen pool).
- **Split hygiene (label leakage)**: labels are the 12 concepts, so no concept-held-out split
  exists by construction — this is closed-world identification, same as the LR task (that is the
  *fair* comparison; say so). Leakage controls: train only on freshly generated streams, test only
  on the **frozen pools** (`runs/lr_grid_box/_ind/`, `runs/_ind/`) so no generation batch, seed, or
  collection session is shared; report per-seed; for the wording-generalization version, train on
  `evoked`, test on `evoked_alt` (the repo's existing transfer protocol).
- **Metric**: calibrated bits = log₂12 − CE with the same held-out-third τ calibration as
  `evaluate_cell`, plus top-1 — same currency as 0.163/0.191/0.405, directly comparable.
- **Compute**: fine-tune 300M on ≤48k streams: 1–3 h on one 24GB GPU (fits a 4090; ~$1), or
  overnight on an M-series laptop (MPS). Total ≤ $10, ~1 day wall-clock.
- **Readout**: inverter ≈ 0 while LR reads 0.28–0.88 (length-matched) ⇒ the privilege survives an
  output2prompt-class budget and the headline is safe. Inverter > 0 ⇒ downgrade "self-privileged"
  to "cheap-reader-invisible"; the regime map (row 8) survives either way.

**E-B. 70B privacy off-diagonal** (closes the row-10 gap, already spec'd in the 2026-07-14
addendum): Qwen-7B likelihood-reads the 803 Llama-70B faithful-template streams; rides the
planned 14B box session; needs the disclosed post-freeze amendment to `lr_scale_extend_prereg.md`.
~$1–2. [Since run as that amendment's rider (readers 1.5B–14B, not just 7B) — see the row-10
correction above.]

**E-C. Freeze the detection-vs-identification artifact** (closes objection (e)): re-run the
cross-model lift-separation computation from the existing 13-context matrices
(`lr_raw_batch_output.jsonl`, grid shards — no new collection needed, $0) and commit a short
verdict note with the JSON.

**E-D (cheap, optional). Stronger surface reader on the sustain-7B cell**: char n-grams n≤5 +
byte-BPE features, 10 seeds, on the existing pool — either kills or sizes the ~0.04-bit surface
residue before a reviewer does ($0, CPU).

## 5. Frank paragraph

"The result is deeply understood" is **false as stated, but the adjacent half-truth is real and
must shape the writing.** What is deeply understood: *that* outputs and logits carry prompt
information (the whole inversion line), *that* covert channels through unmodified sampling decode
only with generating-model access (the seed-stego paper — but for a payload a sender planted, not
for model state), and *that* models can be trained or pressured into deliberate text-carried codes
(Roger & Greenblatt; Hidden in Plain Text). None of those papers measure what this repo measured:
a conditioning mark that *emerges* in semantics-stripped output with no instruction to communicate
and no optimization, whose readability is *access-split* (generator's own likelihoods: 0.16→0.41
bits, growing with scale, 1.11 sustained, 0.24 cross-family at 70B; every non-self reader we
fielded: ≈ 0), and whose polarity *flips* between prompt-conditioning (private mark) and
activation steering (public mark, 1.2–2 bits to a no-access character reader, non-closing through
14B). The two papers Matt found are the *nearest neighbors that make the framing precise*, not
scoops: seed-stego shares the decode primitive but plants the message; inversion shares the
information source but uses trained external readers on natural outputs — and it is exactly the
inversion line that makes our trained-decoder cell look underpowered (objection a) until E-A runs.
The bulletproof headline, quotable today with the length-matched numbers and the "readers we
fielded" qualifier: **"The same word-free gibberish carries two different marks: steer the model
and any character-counter can identify the concept at every scale we tested; merely *tell* the
model a secret and the mark that appears is one that — at every reader budget we tested — only the
generating model's own likelihoods can read, and it grows with scale."** After E-A (a ~$10,
one-day experiment), "at every reader budget we tested" can be upgraded to "even to an
output2prompt-class trained inverter" — or honestly downgraded; either outcome is publishable
because the regime split, the scale trend, and the 70B cross-family positive do not depend on it.

---

*Sources (external):* [arXiv:2606.09135](https://arxiv.org/abs/2606.09135) ·
[arXiv:2311.13647](https://arxiv.org/abs/2311.13647) ·
[arXiv:2405.15012](https://arxiv.org/abs/2405.15012) ·
[arXiv:2410.03768](https://arxiv.org/abs/2410.03768) ·
[arXiv:2310.18512](https://arxiv.org/abs/2310.18512) ·
[arXiv:2507.14805](https://arxiv.org/abs/2507.14805) ·
[arXiv:2605.10794](https://arxiv.org/abs/2605.10794) ·
[arXiv:2506.17090](https://arxiv.org/abs/2506.17090) ·
[arXiv:2510.15511](https://arxiv.org/abs/2510.15511)
