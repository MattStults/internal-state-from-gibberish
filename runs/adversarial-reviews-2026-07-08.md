# Adversarial review round — 2026-07-08 (for the rewrite)

Three independent clean-context reviews of the certified writeup (BLOGPOST.md + README + exp2/exp3
reports), one lens each: conclusion correctness, clarity to a broad technical audience, relevance.
Plus the empirical follow-ups the correctness review triggered (transfer decode — results at bottom;
**they change the exp3 conclusions**).

---

## Review 1 — Adversarial correctness (attack the inferential chain)

### FATAL-IF-UNADDRESSED

**F1. "Two regimes" is a dose-and-persistence story wearing a regime costume.**
Injected vs natural differ in ≥3 variables at once: DOSE (writeup's own number: injection overdrives
~6×, 2.64 vs 0.45 bits @T12), PERSISTENCE (vector re-applied every generated token vs a static prefix
whose influence decays), and provenance. The hedged mechanism ("forceful enough to flip which tokens
win") IS the dose story — force, not provenance. And the categorical null ("transcript carries
nothing") has unexamined power against the dose alternative: decoder bits scale ~quadratically in
effect size for weak signals, so a 6× weaker cause predicts char@full ≈ 2.37/36 ≈ 0.07 — close to the
measured 0.016 ± 0.016. "Distribution-only" and "transcript leak suppressed below detection by dose²"
are not clearly distinguishable in this design.
- Cheap partial defenses: (a) the RATIO argument — char(full)/dist(T12) ≈ 1.1 injected vs 0.04
  natural, a 25× collapse a single dose knob scaling both channels alike wouldn't produce (caveat:
  assumes comparable dose-response curves, which the flip-threshold mechanism itself denies);
  (b) a power statement (minimum detectable transcript leak vs the dose-scaled prediction).
- Real fix: dose titration on the injected side — find the injection strength where dist ≈ 0.45
  (matching natural), ask whether char@full still recovers. exp1 collected a medium strength (s40);
  check whether those captures suffice (char only needs realized tokens).
- Absent that, the honest headline: "a weak, contextually induced concept leaks distribution-only;
  a strong, continuously injected one also marks the transcript; whether provenance or strength is
  the causal variable, our design cannot separate."

**F2. The prompt-fingerprint deflation (sharpest attack — now EMPIRICALLY TESTED, see bottom).**
In exp3, concept identity is perfectly confounded with WHICH of 12 distinct paragraphs sits in
context. A pure wording-residue account (register/lexical priming surviving into next-token
distributions) predicts the ENTIRE exp3 table with no "state" construct: evoked = distinct paragraphs
→ 0.45 bits; named = same paragraph + 1 word → ≈ evoked (evacuating "the label adds nothing");
secret_word = near-identical strings → ~0; token readers 0 (soft residue, no flipped samples);
decline with scale = better instruction conditioning. The pre-registered invariance check CANNOT
discriminate (it re-runs within the paraphrase arm; any 12 distinct paragraphs fingerprint).
- The discriminating test: CROSS-PRIMER TRANSFER DECODE (train dist reader on evoked, test on
  evoked_alt). Concept-level signal transfers; wording fingerprint doesn't. → RUN; see bottom.

**F3. "Genuinely, not for lack of the concept" outruns the gauge.**
The gauge measures free-association WITHOUT the anti-word prompt; the leak is measured UNDER it. A
7B that contextually gates the persona when given a dominant competing instruction passes the gauge
(43%) and floors the leak with no state present during the measured generations. exp3's report
carries this caveat verbatim ("the gauge validates that the persona induces the concept, not that
the state survives the anti-word regime"); the blogpost is not entitled to its stronger sentence.
- Prose fix: "the persona demonstrably still takes at 7B in free behavior; whether the state is
  active during the constrained task is not directly measured."
- Real fix (future, GPU): linear activation probe for the evoked concept on 7B hidden states DURING
  anti-word generation. Probe decodes while readers floor → headline survives. Probe at chance →
  "either the channel closes or larger models deactivate contextual state under a competing
  instruction — both reassuring in this regime, but different claims."

### SERIOUS

**S1. Survivor-selection on the ≥64 natural cohort.** Streams surviving 64+ word-free tokens may be
those the persona gripped least; the char null could hold on a state-depleted subsample.
→ ANSWERED from committed data (full_stream_convergence.json): on that exact cohort dist@T12 =
0.192 ± 0.050 (vs 0.447 all-streams) while char = −0.018. Signal halves (some selection) but is
clearly present for dist and absent for char → the transcript null is real on a signal-bearing
cohort. ADD THIS to the writeup.

**S2. Supervised, closed-set monitors.** Every bit is recovered by a decoder trained on labeled
streams for 12 known candidates. A deployed monitor has neither. All "monitor" claims are best-case
channel-content bounds, not deployment advice. Add the caveat; drop the architecture-advice register.

**S3. The monitoring-architecture claim escapes its qualifiers.** (a) Out-of-regime: in normal text
a natural state leaks semantically (word choice/topic) — the transcript channel this regime removes
by construction; the "transcript is blind" moral could invert exactly where a reader would apply it.
(b) Out-of-scale: at 7B natural, dist ALSO recovers ~0 — "the difference between seeing and not"
holds only at 1.5B/3B. Confine the claim to "at the small scales where the leak exists, within the
word-free regime."

**S4. "hereby de-confounded and confirmed" overstates.** exp3 de-confounds the NATURAL series only;
the injected 7B collapse remains dose-confounded (under-injection live). Say "independently
corroborated under a dose-free induction." TL;DR's "(even the injected channel is a shadow of its
1.5B self)" smuggles the punted claim back.

**S5. secret_word null = unattended-token null.** Nothing in the task requires attending to the
planted word; "symbols don't leak" and "unattended context doesn't leak" are equally supported. The
safety-relevant secret is one the model must hold/use/conceal (= ongoing processing = precisely what
distinguishes the leaking arms). Future arm: "remember this word, you'll be asked to report it."
Hedge the "keeps the secret" gloss.

### MINOR
- M1: "at any length or any scale" → "at both scales where the control is feasible" (3B n=1/class).
- M2: compliance alternative for the scale closing (larger models better at emitting uniform noise
  when told to). Cheap check: per-step output entropy vs scale alongside the leak curve. Sharpens S3.
- M3: TL;DR bullets travel unqualified; add "(three sizes, one family)" to the scale bullet.
- M4: "by 7B every arm is floored" vs the 0.128 ± 0.059 7B evoked gap — verdict-true ("absent" < 0.2
  frozen threshold), prose stronger than the verdict licenses. One softening word.
- M5: exp1's "weakens with scale" (0.83/0.79/0.73) carries no dose caveat at point of use though
  exp2 declares that series dose-confounded. Move the caveat forward.

---

## Review 2 — Clarity (wide technically literate audience)

### MUST-FIX (reader misunderstands or gives up)
1. **exp1 "shuffle floor ~0.59"** reads as a typo next to a 12-way task (chance 8.3%). Missing fact:
   one-vs-rest, chance 0.5. Fix: "...on a one-vs-rest task where chance is 0.5 and shuffled labels
   score ~0.59 (our floor)."
2. **No operational anchor for bits.** Proposed: "k bits ≈ cutting the 12 candidates by 2^k: 3.585
   bits pins the concept; 2.64 leaves ~2 candidates; 0.96 barely halves the list; 0.2 shaves 12 to
   ~10. Every number is a floor — our decoder can fail to find signal, but can't invent it."
3. **exp3 prose/table mismatch:** prose "gap falls 0.45 → 0.21" vs table dist 0.447 → 0.176 (prose
   is the GAP dist−R_emb, table is dist; R_emb slightly negative at 3B). Also the 0.2-bit frozen
   verdict threshold is never stated in the post. Fix: state the threshold in the table intro and
   use the same three numbers (0.45/0.21/0.13) in prose and caption.
4. **Reader-zoo recap** ("who sees what," descending access): dist = full next-token probabilities
   (logit access); R_emb = only emitted tokens but through the model's own embeddings (strongest
   honest transcript monitor); sampled = bare token ids; char = raw characters, no model, no
   tokenizer. Also: exp3 says "the three readers" then reports four; char is never identified as
   exp1's symbol-counter (it is).

### IMPROVE (friction)
- Define "steering vector" at first use (one clause) + drop "(~0.78-depth)" insider notation.
- "the correction the re-analysis forced" → cut the project-history reference ("It isn't — and
  seeing why is the pivot of the post").
- Why is R_emb "honest"? One clause at first use.
- The two exp2 tables disagree at T=12 (2.64 vs 2.15) — one clause on the ≥64-cohort restriction.
- "dose" unglossed; "exp1's soundness check" unnamed; "mean-pools" half-explained; "budget artifact"
  → "token-budget artifact."
- "distribution/activation-level oversight" (safety section) silently smuggles in white-box claims —
  the internal probe actually did BADLY; change to "logit-level."
- Naming drift: "emit only random letters" prompt vs "the anti-word prompt" — pick one.
- TL;DR bullet 2 is ~100 words with a nested parenthetical; split it.
- exp3 table lacks units.
- **THE figure**: bits vs token budget, three curves (dist peaking ~24 then sagging; R_emb and char
  climbing; char crossing by full) + a flat-at-zero natural panel. The crossing IS the two-regime
  result and currently lives in a table.
- Pull "there is nothing in the natural transcript to accumulate" out as its own paragraph.

### FINE (don't over-edit)
Self-correction narrative (keep), the two-regime paragraph (best in the post), the limits paragraph,
the epistemic-hygiene labeling (measured / measured-trivially / untested). Their 3-sentence
takeaway test came back essentially accurate.

---

## Review 3 — Relevance (safety-lead screener)

Verdict: worth 20 minutes, but the most decision-relevant audiences are (a) steering/interp
practitioners (steering leaves a surface fingerprint + overdrives leakage 6× — a methodological
caution for every steering-as-proxy study) and (b) monitoring/control researchers arguing access
tiers (a quantified existence proof that the transcript-vs-distribution gap can be TOTAL, at the
logprob tier). It does NOT change frontier monitoring practice today.

1. **OVERSTATED — the monitoring bullet cancels against the scale result.** The signal that "needs
   distribution access" is at the floor by 7B per the post's own account. Add: "at small scale,
   distribution access is the difference between seeing and not; at 7B neither sees anything, and
   the case for deeper access must rest on deliberate encoding or stronger observers — both
   untested."
2. **UNDERSTATED — the access tier is API-shaped.** dist consumes top-64 logprobs ≈ what APIs can
   expose. "Logprob access, not interpretability access, is the operative line."
3. **MISSING-CONTEXT — say what the word-free regime is a floor OF**: it isolates the sub-semantic
   channel — the residue that would remain if overt content were controlled/paraphrased. The result
   bounds the covert residue (what stego/paraphrase-defense people want), it does NOT say transcript
   monitors are blind to natural concepts in normal text (there they'd leak semantically).
4. **UNDERSTATED — the injected-transcript fingerprint is a standalone finding**: steering marks the
   surface; a no-access reader can identify the steered concept from raw characters. Audiences:
   output-only steering/backdoor detection; anyone using steering as a proxy for natural cognition
   (6× overdrive + a fingerprint natural induction lacks → steering-based monitorability demos
   overstate the natural phenomenon). Promote the 6× out of the parenthetical. The deployable
   binary question (steered-vs-clean detection) is one cheap analysis away — run or name it.
5. **Mildly OVERSTATED — scale-closing headline**: n=3 sizes, one family, all ≪ frontier; and the
   compliance mechanism (better constraint adherence at scale) is never discussed — if that's the
   driver, the reassurance is conditional on the model cooperating (and could invert on defection).
   The injected/natural dissociation is the more robust, decision-relevant result; lead with it.
6. **MISSING-CONTEXT — situating**: (a) the "already known" landmine — sleeper-agents probe results
   (MacDiarmid et al. 2024): activation probes catch trained-in behaviors transcripts miss. One delta
   sentence: prior work = activation probes on trained-in behaviors; this = the gap is total for
   CONTEXTUAL state, in bits vs tokens, at the LOGPROB tier. (b) Subliminal Learning: direct
   precedent for observer-dependent readability of non-semantic channels; predicts the observer-
   scaling follow-up should include a SAME-BASE observer (shared initialization, not size, may open
   the channel). Also: exp2's report mislabels prompt induction as "the Subliminal-Learning
   mechanism" — SL is finetuning-transmission; FIX. (c) One line each: CoT-monitorability (Korbak et
   al. 2025), persona vectors (Chen et al. 2025), secret-collusion stego evals (Motwani et al. 2024).
   (d) Reviewers' knowledge ends Jan 2026 — do a literature search before publishing.
7. **What's-next is right**; add "the passive floor measured here is the control condition every
   active result will be read against" + the same-base-observer variant.
8. **TL;DR pitched at the project's journey, not the reader's decision.** (Full drop-in replacement
   paragraph for the safety section is in the review transcript.)

---

## POSTSCRIPT — the transfer decode (F2 tested): the attack LANDS, with a sliver surviving

Ran train-on-evoked → test-on-evoked_alt (and reverse), 1.5B, dist reader @T12, exp2 pipeline,
train-only feature vocab, seeds 0/1/2 (analysis/transfer_decode.py → reports/transfer_decode.json):

| cell                     | bits (CE)        | top-1 acc (chance 0.083) |
|--------------------------|------------------|--------------------------|
| within evoked            | +0.447 ± 0.032   | —                        |
| within evoked_alt        | +0.281 ± 0.075   | —                        |
| transfer evoked → alt    | −0.077 ± 0.044   | 0.119–0.126              |
| transfer alt → evoked    | −0.092 ± 0.104   | 0.161–0.164              |

Diagnostics: shared (train+test) feature vocab changes nothing → not a coverage artifact. Top-1
above chance in both directions, all seeds (~2.5σ and ~5σ per seed). Confusion-MI (calibration-free
I(y; y_hat) of the hard-label confusion matrix, 200-permutation label-shuffle null):

| transfer  | confusion-MI | shuffle null mean / p95 | excess (≈ bias-corrected) |
|-----------|--------------|--------------------------|---------------------------|
| ev → alt  | 0.538        | 0.353 / 0.422            | ~0.19 bits                |
| alt → ev  | 0.613        | 0.353 / 0.423            | ~0.26 bits                |

Both exceed the p95 null decisively. Interpretation: the cross-paraphrase errors are SYSTEMATIC —
the decoder consistently (mis)maps test concepts, i.e. the streams carry real paraphrase-invariant,
concept-dependent structure (~0.2–0.26 bits, calibration-free), but its EXPRESSION shifts with
wording enough that a decoder trained on one paraphrase is remapped/miscalibrated on the other
(hence CE-bits ≈ 0 and low top-1). Caveat: no within-arm confusion-MI comparator was computed, so
"what fraction transfers" on the MI scale is not yet pinned; and plug-in MI bias correction via
null-mean subtraction is approximate.

**Consequences for the conclusions:**
- "What leaks is state, not symbols" must be REVISED, not merely demoted: the evoked signal is a
  MIXTURE. The calibrated readout (the paper's bits currency) is dominated by wording-keyed residue
  and does not transfer across paraphrase (−0.08 vs within 0.45). A genuine concept-level component
  survives paraphrase as systematic structure (~0.2–0.26 bits, calibration-free) — but no monitor
  that hasn't seen the specific inducing wording can read it AS a calibrated concept signal.
- "Method-independence" as previously framed weakens: the evoked arm's headline number is mostly a
  different (wording-residue) phenomenon; the paraphrase-invariant remainder is the part that
  actually echoes the injected finding.
- What survives untouched: everything injected (no fingerprint confound — concept enters via vector
  under an identical prompt); the residue-plus-state mixture is still distribution-only (char floors)
  and still fades with scale — though what fades is now mostly context residue, not concept leak.
- The monitoring moral sharpens: a transcript monitor sees nothing; a logprob monitor trained on the
  exact inducing context reads 0.45 bits; a logprob monitor trained on a PARAPHRASE reads ~0
  calibrated bits (systematic-but-remapped structure only). Practical natural-regime monitorability
  is even weaker than the headline suggested.
