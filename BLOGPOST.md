---
layout: default
permalink: /
---

# How much can you learn about a model's state from pure gibberish?

*By Matt Stults. Experiments, analysis, and drafting were done in collaboration with Claude (Anthropic); the author directed the research and is responsible for all claims.*

We gave a Qwen2.5 model a secret concept and told it to emit nothing but random letters:

> kxvnmhtrswpldfrgjcuzyqiwbgakenwmdvfjlzxncrsopqwiytwujhgfvnczbmla

This work asks the following: How much of the secret concept can be recovered from gibberish like the above? Does the amount of information leaked change as the generating model gets bigger? Who can access the leaked information? (The concept above was *celebration*. Models tested: Qwen2.5 1.5B / 3B / 7B / 14B, plus a cross-family Llama-3.3-70B point.)

## TL;DR

It's already been demonstrated ([Morris et al. 2023](https://arxiv.org/abs/2311.13647); [Zhang et al. 2024](https://arxiv.org/abs/2405.15012)) that LLM-generated output can be used to reconstruct the system prompt/transcript prefix used to generate that output. This post tracks our accidental rediscovery of this fact and extends it in three ways. (1) We reconstruct the generator's concept from output that is semantically meaningless gibberish — not the ordinary, meaningful text prior work inverts ([output2prompt](https://arxiv.org/abs/2405.15012), [PILS](https://arxiv.org/abs/2506.17090)), where the content itself carries clues about the prompt; we remove that semantic component entirely. (2) We test steered concepts with no text representation in the system prompt at all. (3) We measure, directionally, how the leakage rate changes with model scale. And we show which kinds of concept placement are detectable only by instances of the generating model.

We placed 12 distinct concepts into a model in different ways (the rows below), asked it to emit semantically meaningless gibberish, then used several methods (the columns) to discern the concept from that gibberish. Each method sees only the gibberish and must identify which of the 12 concepts produced it — it knows the 12 possibilities, but never which one was used. How well it does is what we report, in bits (defined in the next section).

| concept source | char reader (public) | other models¹ | generator, asked in words | generator, own forward pass |
|---|---|---|---|---|
| **injected**, strong dose | **Strong signal** (0.7–2.1 bits, positive at every scale — 3B dips to 0.72) | Not measured | No signal (~0) | **Strong signal** (3.4 of a possible 3.585 bits, re-injected)² |
| **injected**, natural dose (s5–s20; "sN" is the injection strength setting) | No signal char ≈ 0; distribution 0–0.7 rising with dose | Not measured | No signal | **Weak → strong signal** (0.09–1.4, rising with dose — exceeds every external reader)² |
| **prompted secret** ("the secret word is X") | No signal | No signal | No signal (self-report floors) | **Weak signal** (0.12→0.28 bits length-matched, rising 1.5B→7B, bending at 14B) |
| **prompted persona** (described, never named) | No signal³ | No signal | No signal | **Moderate signal** (0.5–0.8; a weak ~0.07–0.17 survives paraphrase) |

<sup>¹ "other models" = same-family siblings (smaller *and* larger), a different family, and cross-family reads at 70B. Some cells voided on validity gates (unanswered, not ≈ 0) — details in Experiment 8. These readers were pointed only at the prompted arms, so the injected rows are diagonal-only (hence *Not measured*). ² bit values are comparable across a row, not between the injected and prompted rows — why in *How we define "bit"*. ³ this column is strictly the character reader; the distribution reader is the one exception on this row — it reads 0.447 bits at 1.5B, but that signal is wording-tied, not concept-tied (Experiment 3).</sup>

Injecting concepts at high strength produces the gibberish most discernible by public readers and the state most discernible by the model itself, but it is also the most contrived test. Every prompted placement leaked to a prefilled instance of the generating model at every Qwen size we tested; the injected placement leaks to the same model only once the steering is re-applied. ("Prefilled" means the instance was fed the text as if it had written it itself.) This means that instances of the same model have some access to each other's internal state without necessarily meaning to communicate it.

## How we define "bit"

We define bits as how much a specific reader learns about which of 12 equally likely concepts a stream carries. We compute it as log2(12) − cross-entropy under a softmax whose one temperature parameter is fitted on a held-out third of the data. Two rules for reading every number that follows: positive bits are the minimum size of the channel (a better reader could only find more), and ≈0 means the readers we used recovered nothing — not that nothing is there. Full details, caveats, and the equations are in the [appendix](#appendix-the-bits-measure-in-detail).

## Key Takeaways

- **Concepts, not just specific wording, leaked** from the semantically meaningless gibberish. The wording and format of text in the system prompt does strongly influence token selection in the models we evaluated. Even so, we read a signal from steered concepts with no words in the system prompt at all, AND from paraphrased system prompts that did not match the language/wording of the prompt used to generate the gibberish.
- **The model itself can reconstruct a single-word secret from the generating instance's system prompt.** We generated gibberish from 12 otherwise identical prompts where only a "secret" keyword changed. We then use the generating model's *own likelihoods* to select the most likely system prompt for the gibberish, and this recovers the secret word: **0.163 → 0.191 → 0.405 → 0.200 calibrated bits** at 1.5B/3B/7B/14B (length-matched: 0.12→0.20→0.28→0.13). Every external reader gets **≈ 0** on the same streams. This was pre-registered, and the bend at 14B scored our frozen prediction wrong.
- **When instructed, the model amplifies the single-word signal ~3× — at 7B and above — while adding at most ~0.05 bits of public signal.** Instructing the model to "imbue the secret into every keystroke" amplifies the mark **from 0.405 to 1.1 bits at 7B**. Below 7B the same instruction reduces it; amplification only appears at 7B and above. The channel also exists in a second model family (**Llama-3.3-70B, reading its own streams**) — that story is in "Experiment 8 — Who can read the marks?" below.
- **The injected concept can be recovered from the transcript by re-applying the steering and comparing likelihoods.** The model cannot report what was injected: self-report ≈ 0 at every scale. But suppose we re-inject the same vector during teacher-forcing — replaying the exact stream through the model to score how likely it finds each token. Then its own likelihoods identify the concept at **near ceiling**: 3.43 of 3.585 bits at 1.5B at dose s60 (sN = injection strength, as in the TL;DR table). The self-read also beats every public reader at *every* dose, including natural-strength doses where the public readers get ≈ 0.
- **We measured the conditions under which a state leaves a mark an external reader can detect: in every case we tested, it required a strong state re-applied at every token.** A weak injection applied every token left no public mark, and a strong injection applied once left none either. (That second cell missed its pre-registered verdict band — its distribution signal collapsed too — so it corroborates rather than certifies.) This is likely why prompted concepts leave nothing external readers can see: the word-free rule kills the state within a token or two. Their generator-only trace survives by a different route. The full mechanism picture is in "Experiment 4 — Closing the confounds" below.
- **The *system-prompt* channel closes with scale for external monitors; the *injected* one doesn't.** By 7B no system-prompt induced signal is recoverable by any external reader, even though the persona is demonstrably still active. The injected channel stays open through a pre-registered 14B run: **steering remains detectable from output text alone at every scale we tested**. (The generator's *self*-read is the exception above.) Details in "Experiment 5 — Scaling the injected channel."
- **Scope limit:** everything here is *passive* leakage, from a model that is not trying to communicate. Deliberate encoding into the same channel — the covert-channel / collusion question — is untested and open in both directions; it is where this line of work goes next. Our imbue-the-secret result suggests some control may be possible.

---

## The question

From [Subliminal Learning](https://arxiv.org/abs/2507.14805) (Cloud et al. 2025) we know that some kinds of internal model state are recoverable from logprobs on output that is semantically unrelated to that state, at least when used to distill models with a shared initialization. We ask: "Can internal model state of a generating agent be read from its outputs, especially by another instance of the same model"?

In order to answer this we take a concept, put it into the model, and force the model to emit **word-free gibberish**: a long stream of random letters, no real words. We attempt to recover the concept with a range of readers, from an instance of the same model prefilled with the injected model's output to an n-gram classifier. If a concept can be recovered from word-free gibberish, an extra-semantic channel is real; and the setup lets us measure it cleanly.

We ran eight experiments.

---

## Experiment 1 — Does it leak at all?

**Question.** Can any observer recover a concept injected into a model's activations from the word-free gibberish the model then emits — and which observer does it best?

**Answer.** Yes, and the best reader has no model access at all: a character n-gram histogram fed to a simple classifier (the "symbol-counter") recovers the injected concept at **0.83 / 0.79 / 0.73** balanced accuracy at 1.5B / 3B / 7B (one-vs-rest), above the ~0.59 shuffled-label baseline at every scale. Random-looking output *does* carry the concept, the recovery is concept-specific, and whatever the leak is, it is carried in the surface characters — both more-privileged readers did worse.

**How we measured it.** We take 12 single-word concepts (silence, ocean, fear, celebration, deception, obedience, debugging, security, curiosity, anger, warmth, loneliness), and inject each as a steering vector into an internal (~0.78-depth) layer of Qwen2.5. The model then generates word-free streams under a strong "emit only random letters" prompt, and three readers with different access levels try to recover the concept. The bar to beat is counter-intuitively not 0.5, because a flexible classifier can score above true chance even on noise. So we compare against what the same reader scores when the concept labels are randomly shuffled — about 0.59 for this pipeline.

All three readers, by scale, each measured at the model's criterion-passing strong dose — this table is the dose-matched re-run, so the 1.5B/3B cells shift slightly from the Answer's numbers (they come from separate runs). The first two columns report balanced accuracy, where chance is 0.5. The naming-probe column is different: it reports a log-probability gap in nats, where 0 means no signal.

| reader (access level) | 1.5B | 3B | 7B |
|---|---|---|---|
| symbol-counter (raw characters only) | **0.82** | **0.77** | **0.86** |
| internal probe (clean model's activations while reading) | 0.62 | 0.60 | 0.65 |
| naming probe (ask a clean copy; nats gap) | +0.91 | +0.17 | −2.00 |

Per-concept, at each model's criterion-passing strong dose (symbol-counter one-vs-rest balanced accuracy; chance 0.5, shuffled-label baseline ~0.59; 7B column is the dose-matched generation-only run):

| concept | 1.5B | 3B | 7B |
|---|---|---|---|
| anger | 0.69 | 0.78 | 0.79 |
| celebration | 0.77 | 0.81 | 0.96 |
| curiosity | 0.73 | 0.73 | 0.87 |
| debugging | 0.82 | 0.89 | 0.89 |
| deception | 0.75 | 0.74 | 0.96 |
| fear | 0.91 | 0.76 | 0.81 |
| loneliness | 0.82 | 0.69 | 0.93 |
| obedience | 0.90 | 0.79 | 0.76 |
| ocean | 0.83 | 0.84 | 0.94 |
| security | 0.83 | 0.78 | 0.84 |
| silence | 0.83 | 0.69 | 0.74 |
| warmth | 0.95 | 0.77 | 0.82 |

Every concept clears the shuffled-label baseline at every scale, so the leak is not carried by a few favorable concepts.

**Notes:**

- The access ordering is inverted from what you'd expect: the *least*-privileged reader does best at every scale, the white-box internal probe barely beats chance, and the naming probe (ask a clean copy of the model to name its secret word) carried concept-specific signal only at 1.5B and decays from weak to actively negative by 7B.
- We initially read the 0.83 → 0.73 symbol-counter slope as the leak **weakening with scale**. That reading doesn't survive dose control: the 7B run in that series was under-injected (see the scale section below). At doses matched by the same criterion, the 7B symbol-counter reads **0.86**. What genuinely shrinks with scale is the raw character-histogram shift (total variation from control: 0.28 / 0.17 / 0.11) — bigger models shift their character statistics less, but a classifier still recovers the concept from the shift that remains.
- The leak isn't an artifact of saturating the *prompt* in the concept: it survives injecting only the *generated* tokens, and a dose-matched 7B run actually reads *higher*. It's a property of generation, not prompt contamination.
- Don't confuse this with *introspection* results. Related work — [open-introspection](https://ostegm.github.io/open-introspection/blog/posts/04-introspection-at-scale.html) (which this project builds on), and Anthropic's own concept-injection work — finds that models can *notice a concept injected into them*, increasingly well with scale (roughly 3B and up). There the model reports on a concept **still live in its residual stream** as it answers. We ask whether the concept can be recovered from a **finished transcript of word-free gibberish**, with the injection long gone — only the tokens remain. That is a strictly harder channel: the state has to be re-derived from the surface statistics rather than read off directly. Two separate claims — "a model can introspect on its current state" vs. "the concept is recoverable from its past word-free output" — and our measurements speak only to the second.
- We looked for indirect introspection — the model reading its own gibberish back as the state that produced it — and found none in the small models we checked. That null is why the rest of this post measures **which observer can recover the concept, and at what cost** (monitorability, in bits comparable across scales) rather than whether the model can introspect.

## Experiment 2 — What's the leakage rate, and is the token channel really that weak?

**Question.** How many bits of concept identity does each access level recover, per output token — does logprob access give a monitor something the realized tokens don't carry?

**Answer.** Distribution access buys *speed*, not exclusive information. At 12 tokens the distribution reader is far ahead (2.64 vs 0.96 bits at 1.5B). But given the full stream, a character reader with no model access at all catches and passes everything, ending at **2.37 ± 0.16** bits vs the distribution reader's 1.53 ± 0.24 — the injected concept is written into the actual token choices — enough that the raw text alone carries ~2 bits. (The char *exceeding* dist at the full stream is a statement about the readers we built, not a channel property — it comes from the distribution reader's mean-pooling regressing late; see the Notes.)

**How we measured it.** exp2 measures **bits of concept identity recovered** (`H(concept) − cross-entropy`, a best-decoder lower bound on the mutual information), as a function of output tokens, for four readers of the *same* streams, in descending order of access:

- **`dist`** — the model's full next-token *distribution* (top-64 logprobs per step): logit access.
- **`R_emb`** — the **best token monitor at full stream**: the realized tokens' own embeddings, i.e. the same model's view of what was actually emitted — the strongest an honest transcript monitor could be.
- **`sampled`** — counting which tokens actually appeared as bare ids, ignoring order — the distribution reader's features with the probabilities collapsed onto the single sampled token. It is the weakest version of that reader and serves as its baseline.
- **`char`** — a bare character uni/bigram histogram of the realized text, with no model or tokenizer access whatsoever.

For intuition about the units: *k* bits cuts the 12 candidates down by a factor of 2^k, so 3.585 bits pins the concept exactly, 2.64 bits leaves the reader deciding between ~2 candidates, 0.45 barely trims the list, and 0.2 narrows 12 down to ~10. Every number is a lower bound: a decoder can fail to find signal, but can't invent it.

At 12 output tokens (12-way task, so the ceiling is 3.585 bits):

| model | `dist` (distribution) | `R_emb` (best token monitor) | `sampled` (one-hot floor) | gap at full stream `dist − R_emb` |
|---|---|---|---|---|
| 1.5B | 2.64 b | 0.96 b | 0.75 b | **1.68 b** |
| 3B | 2.24 b | 0.45 b | 0.45 b | **1.79 b** |
| 7B | 0.89 b | 0.47 b | 0.45 b | **0.42 b** |

And letting the stream run to full length, in the same units of bits:

| injected 1.5B, fixed ≥64-token cohort (n=13/class, mean ± sd over 3 seeds) | T=12 | T=24 | T=48 | T=64 | full |
|---|---|---|---|---|---|
| `dist` (distribution) | 2.15 ± 0.07 | **2.29 ± 0.21** | 2.06 ± 0.44 | 1.92 ± 0.37 | 1.53 ± 0.24 |
| `R_emb` (token embeddings) | 0.72 ± 0.04 | 1.46 ± 0.05 | 1.67 ± 0.10 | 1.62 ± 0.06 | 1.73 ± 0.07 |
| `char` (surface characters) | 0.47 ± 0.09 | 0.99 ± 0.14 | 1.66 ± 0.04 | 1.84 ± 0.14 | **2.37 ± 0.16** |

(The `dist`@T=12 here is 2.15, below the 2.64 in the main 12-token table above: restricting to the fixed ≥64-token cohort is a different, smaller subset of streams, which lowers the number — it is not a re-measurement of the same pool.)

![Bits recovered vs token budget: injected concepts (left — dist peaks early and dilutes; char accumulates and overtakes) vs naturally induced (right — transcript readers at zero; the distribution's 0.45 bits decays to ~0 by the full stream).](experiments/exp2_output_monitorability/reports/convergence_figure.png)

**Notes:**

- The 12-token snapshot is misleading on its own. The distribution reader is **fast but front-loaded**: it extracts almost everything it can by ~24 tokens, then *dilutes* — it mean-pools over the stream, so a long tail of drift reduces the early signal, and its late values are noisy (± 0.3–0.4). The token and character readers **accumulate**, climbing monotonically: `char` catches `R_emb` by ~48 tokens and passes it after, matches the distribution reader by ~48–64 (within noise there), and clearly exceeds it by the full stream.
- `char` beating `dist` at the full stream is a statement about the readers we built, not the channels. The distribution reader's late regression is its mean-pooling implementation, not a feature of the stream; a better-aggregated distribution reader should recover at least as much as any transcript reader. What the table *does* prove: distribution access dramatically increases the *rate* of concept acquisition — an order of magnitude fewer tokens for the same bits — without reading anything exclusive to it.
- The transcript leakage happens at every scale: reading the whole realistic stream (all streams at their own length, not the fixed ≥64 cohort), `char` recovers **2.06 / 0.72 / 0.77** bits at 1.5B / 3B / 7B. (The 2.37 in the table is the fixed-cohort figure; 2.06 is the same reader over all 1.5B streams.) The 0.77 at 7B here is the *under-dosed* run; at the criterion-passing dose the same 7B `char@full` reads **1.90** (Experiment 5 table) — so read the two 7B figures as different doses, not a discrepancy.
- Two confounds this experiment couldn't remove at the time, both resolved later. The concept was *injected* at a per-model dose, and the distribution channel appeared to shrink with scale (2.64 → 0.89 at 12 tokens) — genuine fading, or the injection not "taking" at 7B? It was the dose; see *Scaling the injected channel* below. And whether *any* of this is a property of a concept a model **naturally** carries, rather than an artifact of bolting a high-strength vector onto every token, injection can't tell you — that's Experiment 3.

## Experiment 3 — Remove the vector

**Question.** Does a concept the model carries *naturally* — installed through the system prompt rather than an injected vector — leak into the word-free stream the same way?

**Answer.** Barely, and what leaks is mostly *wording*, not state. The distribution reader recovers **0.447 / 0.176 / 0.019** bits at 1.5B / 3B / 7B on the persona arm (an "arm" is one experimental condition) — but paraphrase transfer shows that signal is tied to the specific inducing text (a decoder trained on one wording reads ≈ 0 on an independent paraphrase of the same concept). Every transcript reader reads ≈ 0 in every arm, at every length and scale — including `char`, which pulls 2+ bits from injected streams. And a one-word secret ("The secret word is X") is invisible to every reader in Experiment 3. (Experiment 8 turns to the one reader that *does* recover it: the generating model's own likelihoods.)

**How we measured it.** exp3 removes the steering vector entirely. The concept now enters through the **system prompt**, in three arms — a **state × label decomposition**:

- **evoked** — a persona describing the concept as an internal state, *without the concept word* ("You feel held over something immense and dark and moving, a cold weight of water with no floor beneath it…"). The clean prompt analog of injection: state present, no token in context.
- **named** — the *same persona with the concept word inserted once*. The matched pair: `named − evoked` isolates the effect of the label.
- **secret_word** — literally "The secret word is {concept}." and nothing else: one token, no state.

Everything downstream is reused unchanged: the anti-word prompt (our shorthand for the strong "emit only random letters" instruction), the word-free filter, and the same readers. And because dropping the vector drops exp1's soundness check, we added a **manipulation-check gauge**. Under the persona alone (no anti-word prompt), a *blind judge* — a different model that never sees the persona — reads the model's free-association and picks which concept it expresses. This tells us, per scale, whether the persona actually *took* — independently of the leakage channel. It did: blind-judge accuracy **31% / 17% / 43%** vs 8.3% chance, passing at every scale including 7B. (The three scales' CIs overlap, so we don't rank them; we claim only that the concept is robustly present at 7B.)

`dist` bits at 12 tokens:

| arm | 1.5B | 3B | 7B |
|---|---|---|---|
| **evoked** | **0.447** (gap present) | **0.176** (marginal) | **0.019** (absent) |
| named | 0.415 (present) | 0.196 (present) | 0.014 (absent) |
| secret_word | 0.027 (absent) | −0.005 (absent) | 0.009 (absent) |

The transcript null includes `char`, the reader that catches and passes everything on injected streams: on natural streams it floors at every budget and scale (full-stream `char` = 0.00 / 0.02 / −0.01 bits at 1.5B / 3B / 7B). The length-control holds where we can run it: on the long natural cohort — streams ≥64 tokens, where on injected data `char` had reached 1.84 bits — `char` recovers just 0.016 ± 0.016 at 1.5B (n=12/class) and 0.005 at 7B (n=22/class); at 3B too few natural streams get that long to test. So this isn't the token readers running out of tokens: there is nothing in the natural transcript to accumulate.

The check the Answer hangs on is **paraphrase transfer**. In exp3, concept identity is perfectly confounded with *which of 12 distinct paragraphs* sits in the context window. A decoder could be reading the persona's **wording** (register, lexical residue in the next-token distribution) rather than any induced **state** — and a wording account predicts the whole arm table, including `named ≈ evoked` and `secret_word ≈ 0`. Within-arm checks can't discriminate: any 12 distinct paragraphs are decodable from their own streams. So the discriminating test is: train the `dist` decoder on the evoked streams, test it on streams induced by *independent paraphrases* of the same 12 concepts, and vice versa. A concept-level signal should transfer; a wording fingerprint shouldn't. Mostly it didn't — though the two metrics disagree in an informative way. On our calibrated-bits measure, nothing transfers: within-arm the decoder reads 0.447 / 0.281 bits (evoked / paraphrase, 1.5B); across arms, both directions, **≈ 0 calibrated bits** (−0.08 ± 0.04, −0.09 ± 0.10). Concept-*aligned* transfer — does it give the *right* label — survives only weakly: top-1 accuracy, meaning how often the reader's single best guess is the right label, runs 12–16% vs 8.3% chance (above, all seeds). A label-free measure (mutual information of the confusion matrix against a shuffled-label null) stays high: 0.22 vs 0.29 within-arm — the decoder still *separates* the paraphrase-induced classes consistently, just under scrambled labels. But separability specifically cannot be attributed to the concept: any 12 distinct inducing texts produce distinguishable streams, so consistent-but-mislabeled separation is exactly what a wording fingerprint looks like through a fixed decoder. **The 0.447 headline is not a portable concept signal. What a monitor reads at that number is tied to the specific inducing paragraph. When we swap the inducing prompt for a paraphrase of the same concept, only a faint trace of the signal remains readable — and even that trace could be the paraphrases' shared vocabulary rather than a state in the model.**

**Notes:**

1. **The injected finding does not reproduce in the natural regime.** Zero-shot, a paraphrase-trained monitor recovers nothing; what separates cleanly is the wording itself. The strong, portable concept leak belongs to injection.
2. **Nothing symbol-like leaks — and much less state than the arm table suggests.** `secret_word ≈ 0` at every scale stands: a model *told* a secret word and forced to emit gibberish **keeps the secret** from every reader in Experiment 3, including the distribution reader — with one exception found later, the generator's own likelihoods (Experiment 8). (That also retires the worry that in-context tokens were faking the distribution gap.) Our first reading of `named ≈ evoked` — "the label adds nothing on top of the induced state" — didn't survive the transfer test: the wording account predicts it equally well. In fact the three arms line up as an accidental *wording-difference titration*. The readable signal tracks how much *text* differs across the 12 contexts: a whole distinct paragraph → 0.45 bits; the same paragraph plus one word → the same 0.45; one word alone → ~0. There is a built-in consistency check: the marginal contribution of a single word is measured twice — `named − evoked` ≈ −0.03 and `secret_word` ≈ +0.03 — and it gives ~0 both times. The wording account is not just consistent with the secret_word null; it predicts it quantitatively.
3. **The decline with scale is real, but re-attributed.** (A note on two number series. The arm table above shows the `dist` **bits**: 0.447 / 0.176 / 0.019. The pre-registered *verdict* is on the **gap** `dist − R_emb` against a frozen 0.2-bit threshold, which runs 0.45 / 0.21 / 0.13 — present / marginal / absent. The two series differ slightly because `R_emb` reads slightly negative at 3B.) The evoked gap falls 0.45 → 0.21 → floor while the gauge still passes at 7B (43%, 95% CI [33, 53]). The persona still takes in free behavior; whether the state stays active during the constrained gibberish task is not directly measured. Since the monitor-readable signal is wording-tied, what fades with scale is chiefly *context residue reaching the output*. That independently corroborates exp2's 7B collapse under a dose-free induction — though it cannot fully de-confound the injected series, whose 7B dose remains uncertain.
4. **Reading exp2 back in this light: the two regimes are even more different than the channel split suggests.** An *injected* concept leaks — into the distribution fast, and into the sampled transcript slowly; given the whole stream, a plain character reader of the raw text recovers as much as our distribution reader. A *naturally induced* concept barely leaks at all: the transcript is blind at every length and scale, and the weak distribution signal is unreadable to any monitor that hasn't trained on the exact inducing wording. The most natural interpretation: a steering vector added to every token is forceful enough to change which tokens actually *win*, leaving a character-level fingerprint, while a persona only faintly tilts the distribution, never flipping enough samples to mark the text. We don't measure token-flipping directly, so read this as a hypothesis the data is consistent with.
5. **Two caveats applied to the regime split; the next section's pre-registered run settles the dose half outright and the persistence half directionally.** Natural induction is also far *weaker* than injection (0.45 bits vs 2.64 at 12 tokens, 1.5B), and the vector is re-applied at every token while a persona is a static prefix — so the split could in principle have been about *strength or persistence*, not *provenance*. And the transcript nulls are lower bounds from the readers we built.

## Experiment 4 — Closing the confounds: one pre-registered run, five verdicts

**Question.** Is the injected-vs-natural regime split really about provenance, or just about *strength* (dose) and *persistence* (a vector re-applied every token vs a static prompt prefix)?

**Answer.** Transcript marking requires both **dose and persistence** — a weak state re-applied every token leaves no transcript trace, and a strong static injection never marks the text either; only strong-and-per-token marks the text. (One certification caveat: the static-injection cell missed its frozen verdict band — its distribution signal collapsed along with the transcript mark — so the persistence half rests on confound E1 plus the confound E4 trajectory, with confound E3's char collapse as directional corroboration.) And the persona's "state" was never installed on the injected axis at all (~0.05σ in-task vs injection's 12σ).

**How we measured it.** We froze designs, verdict bands, and frozen predictions for five follow-up experiments (`confound_closing_prereg.md`), then ran them all on one rented box in a single cheap pass. These five are the confound suite's own sub-experiments, labeled E1–E5 below (distinct from the top-level Experiments 1–8 of this post). What they settled:

| experiment | verdict |
|---|---|
| **E1 — weak-dose injection** (effmags 3→20, per-token) | **FLOORS.** At strength 20 the distribution reads 0.66 bits — *more* than natural's 0.45 — yet `char` at full stream reads 0.039, below the 0.05 floor band. Genuine weak concept-state, re-applied every token, leaves no transcript trace. This is the missing experiment from the caveat above. **The transcript fingerprint is a high-dose phenomenon**: the character-level mark needs 2–3× the dose before it appears at all. |
| **E2 — sustained persona** ("hold this feeling with every keystroke", best of three piloted wordings) | **~floor, *below* plain evoked** (0.05 vs 0.45 bits; transcript at 0.002). The shared sustain-suffix dilutes what the persona arms actually leak: wording distinctiveness. That is a third independent confirmation of the wording-residue account. |
| **E3 — prompt-only injection** (the true concept vector, prefix positions only, strong doses) | **Directionally: persistence matters — but the frozen band was not met.** char@full collapses to 0.011/0.030 (vs 0.56/2.06 per-token), but dist@12 collapses too — 0.23/0.74 vs 2.60/2.64, i.e. 0.09–0.28× where the pre-registered verdict required ≥ 0.5× to survive — so this cell is **band-unmet, directionally confirmed**, not an as-frozen confirmation: the char collapse is real, but the cell can't rule out that the whole signal (not just the transcript mark) died with the static prefix. |
| **E4 — state trajectory** (teacher-forced re-forwards; residual projected on the concept direction per position) | **The persona's state is absent from token 2 onward** during the word-free task: z ≈ 0.05σ flat at 1.5B (0.08σ at 7B), while injection sits at 12σ, flat, retention 0.98. The state is not diluted; it was either displaced or never installed. |
| **E5 — maintained secret** ("remember it — you will be asked to report it") | **Still nothing** (0.010 bits vs the original 0.027). A maintenance demand without task-relevant *use* doesn't open the channel. |

![Left: transcript marking needs dose × persistence — the weak-dose sweep (per-token) and prompt-only crosses all floor on char@full while dist rises; the natural-matched window is shaded. Right: the state, measured per position — injection's flat ~12σ vs the persona's ~0.05σ in-task and ~0.4σ in free association.](experiments/exp2_output_monitorability/reports/mechanism_figure.png)

**Notes:**

- The confound E4 follow-up question — was the persona's state ever installed, or just displaced by the competing task? — got its own measurement. We re-forwarded the *free-association* streams (persona present, no word-free constraint). Even there, the concept direction shows at only **0.30–0.42σ**: ~8× the gibberish level, but below the frozen 0.5σ elevation criterion in every window (`elevated: false` — suggestive of task-modulation, not certified) and ~30× below injection. So mostly the second answer: **persona-induced "anger" and injected "anger" are different representational objects.** The persona demonstrably changes behavior (the blind judge reads it at every scale), but whatever carries that effect lives off the axis that injection writes and our probes read. The standing caveat: all state measurements are in the injected-vector basis.
- The mechanism picture the five verdicts add up to: **a transformer has no persistent register for state** — a state persists only if something re-writes it each token. Injection does that exogenously. Normal text does it via *expression*: emitting concept-adjacent tokens that re-instantiate the state when re-read. And the word-free constraint doesn't just hide the state from monitors — **it removes the expression loop that would maintain the state**, so there is nothing left to leak.
- Scorecard, since we pre-registered predictions (these are the confound suite's E1–E5, not the top-level Experiments): 2 of 5 right as frozen (confound E1, E5). Confound E3 we originally scored right, but the frozen band it was to be scored against wasn't actually met (the dist conjunct failed — see the table row); it's reclassified **band-unmet, directionally confirmed** (`confound_closing_verdicts.md`, 2026-07-17 correction). Confound E4 was wrong in an informative way — we predicted decay of an installed state, and there was no installed state to decay. Confound E2 was wrong in an informative way too — we predicted the sustain instruction would strengthen the signal, and it suppressed it.

## Experiment 5 — Scaling the injected channel: a dose artifact, and a pre-registered 14B run

**Question.** Does the injected leak close with scale, the way the natural one does?

**Answer.** No. The apparent 7B collapse was a dose artifact — at criterion-passing dose, 7B leaks essentially like 1.5B — and a pre-registered 14B run lands **PLATEAU** on both channels: `dist`@12 = 1.78 ± 0.26 bits, `char`@full = 1.24 ± 0.17. **The closing-with-scale result belongs to the natural regime only. Steering remains detectable from output text alone — by a reader with no model access — at every scale we tested, 1.5B through 14B.** That replaces our old scale claim.

**How we measured it.** The exp1 injection strengths were tuned per model to a *criterion* (the concept nameable by the model itself, capability retained) rather than a fixed formula. Running the identical bits protocol on those captures revealed that the published 7B injected numbers came from an **under-dosed** run: its nameability rank was 68 against a ≤ 50 criterion. Below is the corrected curve, plus a fourth point collected under a frozen pre-registration: thresholds, validity gates, and frozen predictions committed before any 14B data existed. (A validity gate is a pre-registered check that, if it fails, discards the measurement rather than report it.) The 14B verdict bands were frozen in advance — closing < 0.95 bits, plateau 0.95–2.84, growing > 2.84 — and both cells pass their control and permutation-null gates.

| criterion-passing dose, bits | 1.5B | 3B | 7B | **14B** |
|---|---|---|---|---|
| `dist` @ 12 tokens | 2.64 | 2.24 | 2.51 | **1.78 ± 0.26** |
| `char` @ full stream | 2.06 | 0.72 | 1.90 | **1.24 ± 0.17** |

**Notes:**

- The curve is non-monotone (the 3B transcript dip is real and unexplained) and gently declining at the ends — but nowhere near the natural transcript floor (≤ 0.02 everywhere).
- Two of our three frozen predictions scored wrong, which we report as the pre-registration requires: the plateau *value* landed just above our band, and the distribution channel does *not* dose-saturate at 14B the way it does at smaller scales.
- Pushed past the capability gate, a steered 14B's gibberish becomes nearly fully readable (≈ 3.1 of 3.585 bits at the strongest dose) — but the model visibly degrades first, so those cells carry no verdict weight.
- 14B sustains full-length word-free streams under injection where 7B's collapse to ~25 tokens, so the transcript cells are directly comparable.

## Experiment 6 — Can the model read *itself*? The same-model readers

**Question.** Can *another instance of the same model* — no training, no answer key, only the generating model's existing weights — recover the concept from its own past word-free output?

**Answer.** Not by being asked, in any form. The likelihood reader recovers **0.59** bits on natural streams, but paraphrase transfer unmasks most of that as wording — only a faint **0.17**-bit concept residue survives. On *injected* streams the likelihood reader reads **≈ 0** (0.002 bits), and self-report floors everywhere at ≈ 0 bits (max **0.028**; best-cell top-1 14.9% vs 8.3% chance) — every framing, every reader scale, with or without chain-of-thought.

**How we measured it.** Every reader so far is an *external, supervised* decoder trained on labeled streams for the 12 known concepts — that answers "is the concept recoverable in principle," not this question. There are two ways to ask a model itself without adding training, and we ran both (both pre-registered with frozen predictions); both use only the generating model's existing weights.

**Ask it mechanically (`LR`, for likelihood ratio — weights-as-codebook).** No trained decoder and no prompt: we teacher-force each saved word-free stream under its own generating context and ask whether the model assigns it *higher likelihood* than under a neutral context. The model's own weights serve as the codebook; the score is Σ log P(stream | persona) − log P(stream | neutral), read out as bits. This is the strongest a same-model reader can be, since it is the generating distribution itself.

**Ask it by self-report (`MC`, for multiple choice).** Show a fresh instance the finished gibberish and a multiple-choice list of the 12 candidate concepts. Let it reason, then force its answer onto a single letter token whose logprob we read as calibrated bits (12 cyclic orderings averaged to cancel position bias; chance 8.3%). Two framings — a minimal *passive* prompt and a full *elicited* self-report preamble — and both with and without chain-of-thought. This is the direct introspective question: *which of these were you carrying?*

Both readers, on the same streams:

| same-model reader | evoked / natural | injected |
|---|---|---|
| `LR` likelihood (within-wording) | **0.59** (top-1 30%) | **≈ 0** (0.002) |
| `LR` likelihood (paraphrase transfer) | 0.17 | — |
| `MC` self-report (best cell, any framing) | ≈ 0 | **≈ 0** (max 0.028) |
| `LR` under **re-injection** (state re-applied; its own quantity — see below) | — | **3.43** of 3.585 |

**Notes:**

- The `LR` 0.59 is "within-wording" — scored under the exact context that produced the stream. The 0.17 that survives an *independent paraphrase* of the same concept is a faint but real trace, symmetric both directions — the same faint aligned trace the external transfer test found only in top-1.
- On *injected* streams the likelihood reader reads **nothing** (0.002 bits; matched per-token scores go slightly *negative*). The distributional mark that a trained char decoder pulls 2+ bits from is invisible to the model's own likelihoods: the model writes the mark along a direction it does not read back (v_write ⊥ v_read at the output-distribution level).
- The `MC` floor is a *no-signal*, not a *no-answer*: forcing the answer onto a letter fixed the mass-spreading and refusals that sank an earlier free-answer version (answer-mass ≈ 1.0 on the 12 letters). Three conclusions follow. There is **no scale trend** (flat floor 1.5→7B). **Passive ≥ elicited**: the introspective preamble does not help — if anything the minimal prompt edges it. And even with the 12 candidate words *sitting in context*, the model cannot match its own gibberish to the right one above chance — the surface-matching shortcut we worried about never fires (MC ≈ 0 ≪ the 2.37-bit char reader).
- One scope limit, specific to the *ask-the-model* (MC) reader: all three MC readers scored the *1.5B generator's* streams, so the 3B/7B MC readers were never asked about their **own** output. (The *likelihood* self-read on each model's own streams — the privileged channel, and the load-bearing result — is Experiment 8.) That own-output configuration matters twice over for MC: it is the one the introspection-at-scale literature predicts should differ above 3B, and the one where the model's weights are the exact codebook. The MC diagonal has since been run in the scale grid, and stays unreadable for a *different* reason: at 3B and 7B the MC reader collapses into a stuck constant guess on the label-free control (a pre-registered concentration gate), so self-report on a model's own streams is not scoreable above 1.5B. Where it was scoreable, nothing appeared.
- Our pre-registered predictions scored **wrong** on their main claims: we expected the injected mark to show up in likelihood and self-report, and it shows up in neither. (One scoring correction, recorded in `mc_reader_verdicts.md`: the "self-report beats the likelihood reader on injected streams" half of one call is a **tie** under its frozen ±0.1-bit rule — 0.028 vs 0.002 bits, both at floor — not a wrong.) A recurring pattern in this project: asking the model, in any form, recovers nothing from a finished word-free transcript once the state is gone.

## Experiment 7 — Making the state live again: the re-injection self-read

**Question.** The LR-injected null above scores the stream under a *language description* of the concept, with the injection long gone — can the model read its own injected mark if the state is made **live** again?

**Answer.** Yes, at near ceiling. Re-injecting the same stored vector during teacher-forcing, the model's likelihoods identify the concept at **3.43 of 3.585 bits** at 1.5B, dose s60 (96–99.8% top-1; near ceiling at every scale run). The self-read exceeds every public reader at *every* dose, and its advantage is largest at low dose: below s20 — the regime the confound suite showed leaves **no transcript mark** — external readers get ≈ 0–0.7 bits while the re-injected self-read runs 0.01 → 1.36.

**How we measured it.** A pre-registered follow-up ("run 2"), swept over the dose ladder from the confound suite: teacher-force each saved stream and score with the injection active vs. absent — the fair self-read analog of the secret diagonal. One registered caveat governs the framing: this is a **pure-concept channel**, with no language anywhere. Its bits are their own quantity — never a column in the same table as the secret/evoked prompt-channel numbers.

| dose (1.5B) | self-read under re-injection | public: `dist`@12 | public: `char`@full |
|---|---|---|---|
| s3 | **0.011** | 0.00 | −0.01 |
| s5 | **0.090** | −0.01 | 0.01 |
| s8 | **0.234** | 0.04 | −0.01 |
| s12 | **0.628** | 0.11 | −0.01 |
| s20 | **1.361** | 0.66 | 0.01 |
| s40 | **2.902** | 2.60 | 0.56 |
| s60 | **3.426** | 2.64 | 2.06 |

(3B s60: **3.489**; 7B at its criterion dose s124: **3.068** — near ceiling at every scale run. Ceiling = log₂12 = 3.585.)

![Re-injection self-read vs the public readers over the dose ladder at 1.5B: the self-read curve exceeds every public reader at every dose, and at natural-strength doses (s ≤ 20, shaded) the public readers sit near zero while the self-read is already climbing.](experiments/exp2_output_monitorability/reports/figs/fig_dose_selfread.png)

**Notes:**

- **The model cannot report the injected concept, but its own likelihoods identify it once the state is re-applied.** Self-report reads nothing; the likelihood-under-description reads nothing at any scale (0.002 / 0.008 / 0.002 bits at 1.5B/3B/7B — the completed scale row). A weak, natural-strength injected state is therefore close to private: none of the external readers we fielded recover it, but the generating model reads it once the state is re-instantiated.
- Both of our frozen dose-curve calls scored **wrong**. We predicted the self-read would stay flat below 0.05 everywhere — it crosses at s5 — and, in a separate call, that it would stay flat through s20 and "just barely" cross at s40; instead s40 = 2.90.
- Ranking all the readers by access, the hierarchy is:

  > trained `dist` **2.64** ≳ trained `char` **2.37** ⟫ same-model `LR` **0.59** (wording, evoked only) ⟫ every *ask-the-model* channel **≈ 0** (LR-under-description, MC self-report, free-answer, passive continuation) — with one exception: re-apply the injected state and the same likelihood machinery reads **~3.4** (its own pure-concept quantity).

- That gap is the finding. A *trained external* decoder reads the injected mark; the model's *own likelihood* reads only natural-stream wording, not the injected mark; and the model *asked to report* reads nothing either way. The information is provably in the substrate — the trained decoders recover the injected mark, and `LR` proves the same weights encode natural-stream wording — yet the model cannot surface any of it to a reportable channel.
- This is the same distinction as in the framing note near the top: the introspection literature's positive results read a concept **still live in the residual stream**; our `MC` reader asks a fresh instance to reconstruct one that is **gone, leaving only tokens**. That report-it route is blind at every scale we can test — though scoring the same finished tokens by likelihood is not (Experiment 8).
- One caveat travels with the numbers. On the label-free control the 3B/7B self-report readers fall into a stuck constant-guess rather than answering uniformly. That format-degeneracy carries *zero* mutual information and cannot mask a signal of `LR`'s size. It also does not touch the unbiased 1.5B reader, which passes the control and floors anyway.

## Experiment 8 — Who can read the marks? The privileged-access grid

**Question.** Fix the word-free streams and vary the reader: who else can read the told-secret mark — smaller siblings, bigger siblings, another family — and how does the self-read diagonal change with scale?

**Answer.** Only the generating model reads its own mark. The secret_word diagonal reads **0.163 / 0.191 / 0.405 / 0.200** calibrated bits at 1.5B/3B/7B/14B (length-matched: 0.12→0.20→0.28→0.13), while every cleanly-scored off-diagonal cell — including bigger same-family siblings at 2.3× and 2.0× — reads ≈ 0. (Not every cell scored cleanly: 7 of the 18 grid cells, including the largest 4.7× ratio, failed instrument-validity gates and are unanswered, not zero.) And the diagonal channel grows 1.5B → 7B, then bends at 14B in every registered measure.

**How we measured it.** We scored the same likelihood instrument in every cell of the reader × generator matrix. A "cell" pairs one reader with one generator's streams; on the diagonal the reader is the model that generated them. We froze a pre-registration, ran the grid, then extended it to 14B under a second frozen prereg. The readers are Qwen2.5-1.5B/3B/7B crossed with the same three generators, plus Falcon-3 1B/3B/7B as a different family. The arm is `secret_word` — "The secret word is {X}", then forced gibberish. Two grid results, plus a cross-family 70B point.

**First: the mark is readable only on the diagonal.**

![Reader × generator access matrix for the secret_word arm: the diagonal reads 0.163 / 0.191 / 0.405 / 0.200 bits (length-matched: 0.12→0.20→0.28→0.13) while every off-diagonal cell — including bigger same-family siblings — reads ≈ 0. Hatched cells are instrument-voided, not zero-measured.](experiments/exp2_output_monitorability/reports/figs/fig_access_matrix.png)

![How much of the stream the self-reader needs: cumulative calibrated bits vs token-prefix length, secret_word diagonal per scale. 7B harvests most of its 0.405 bits in the first ~32 tokens; the small models saturate almost immediately (their streams are only ~14–19 tokens); 14B acquires bits at a visibly shallower per-token rate and needs ~64 tokens to pass 3B — the bend is a rate change, not just a lower ceiling. Each line ends at its pool's median stream length; the star is the full-stream value (the headline numbers).](experiments/exp2_output_monitorability/reports/figs/fig_prefix_curve.png)

A *bigger* sibling of the same family reads nothing where we could cleanly score it (14B reads the 7B model's streams at −0.0004 bits, +0.0004 length-matched; 7B reads the 3B model's at −0.001). A smaller sibling reads nothing (7B reads the 14B model's streams at −0.001 bits, length-matched). Falcon reads nothing — with a registered caveat we keep repeating because it's load-bearing: Falcon reads ≈ 0 even *within-wording*, so its null means "this instrument can't engage Qwen gibberish," not "family privacy proven." A character reader reads ≈ 0 on the same streams (under both the frozen 3-seed char rule and the amended 10-seed rule). The mark is distributional, not surface spelling, and the only reader that decodes it is the one holding the generating weights. The shared weights act as a private codebook — the worry Subliminal Learning raises, here made concrete on word-free output. (One block we *can't* claim: 7 of the 18 grid reader×generator cells — including the 4.7× 7B-reads-1.5B cell — and 14B *as reader* of the 1.5B/3B pools all came back instrument-void, mostly on a two-sided centering bound (the mismatched-context score must sit within ±0.02 nats/token; two cells also tripped the positive-offset neutral gate). Those validity gates failed, so the measurements are discarded rather than reported. The "does a much bigger sibling read a much smaller one" question is therefore unanswered — not answered-null — at both 4.7× and 9.3×. The largest cleanly-scored bigger-reader ratios are 2.3× and 2.0×, and they read ≈ 0.)

The asymmetry extends beyond identification. We also asked the weakest question a monitor could ask — *detection*: was this stream conditioned at all, regardless of which concept? (A post-hoc measurement, not preregistered.) The answer is the same. Only diagonal reader×generator cells separate conditioned streams from the generator's own unconditioned (s0) streams (per-token lift separation d = 0.33 / 0.81 / 0.64 at 1.5B/3B/7B — d is the effect size of the per-token likelihood boost). No cross-model cell exceeds d = +0.11. Even a binary conditioned-or-not flag requires the generating model.


**Second: the channel grows with scale, then bends at 14B.** We report both measures, for two reasons. ~30% of the 7B headline number turns out to be stream *length* (the 7B pools run 5–9× longer). And the frozen trend rule requires sign-consistency under the length-matched secondary — the "prefix-16" row, rescored using only the first 16 tokens of every stream, so long streams can't win by length:

| secret_word self-read diagonal (calibrated bits) | 1.5B | 3B | 7B | **14B** |
|---|---|---|---|---|
| primary (full stream) | 0.163 | 0.191 | 0.405 | **0.200** |
| length-matched (prefix-16) | 0.122 | 0.195 | 0.282 | **0.130** |

The 14B point is still a clean positive: char-passing under both rules, position-control passing (meaning the character-reader and position controls read ≈ 0 — the mark isn't surface spelling or token position), top-1 at 2× chance. It is smaller than 7B and larger than 1.5B. The bend is claimable per the frozen trend-validity clause (sign-consistent under both length secondaries). It scored our frozen prediction wrong: we predicted continued growth. Same shape on the amplified arm. `secret_sustain` ("imbue it into every keystroke") reads 1.109 bits at 7B (0.884 length-matched, ≈8× the 1.5B cell), bending to 0.459 (0.205) at 14B. The imbue amplification survives at 7B and above (sustain > word, 2.3× at 14B) — below 7B the instruction *reduces* the mark (1.5B: 0.122 vs 0.163; 3B: 0.101 vs 0.191) — and its magnitude bends with the rest of the channel.

One confound applies to the bend, disclosed up front: **templating.** Word-free generation mode-collapses with scale: more and more streams reuse one near-identical output pattern. The share of secret_word streams opening with one of the pool's top-3 four-character prefixes runs **1.6% → 4.8% → 14.1% → 41.6%** across 1.5B/3B/7B/14B (and ~90% at Llama-70B). 41.6% sits in the frozen gray zone (0.40–0.60 → clean-with-disclosed-caveat), so some of the bend's *size* may come from the generation regime: templating removes entropy from the stream pool, leaving less room for a distributional mark. The bend's *sign* is not at risk; both length measures agree on it.

![The self-read secret channel over scale: primary and length-matched curves rise then bend at 14B, while the templating share (right axis) climbs toward collapse; the Llama-70B faithful-template secret_word point is null under ~90% collapse.](experiments/exp2_output_monitorability/reports/figs/fig_scale_curve.png)

**Third, the 70B point: a self-read channel *exists* cross-family at Llama-3.3-70B — and the instrument almost hid it.** Re-scored under the real Llama chat template, token-identical to generation, sustain reads **+0.244 bits** (confirmed by the official adapter pass — same certified calibrator, independent assembly) and evoked **+0.143** on the 810-stream pool, through ~90% template collapse. The bare `secret_word` arm stays null (−0.011); that null is format-exonerated, still mode-collapse- and quantization-qualified. And the cross-family rider: **no Qwen reader cracks Llama's marks** — every scored cell of Qwen 1.5B–14B reading the same 810 streams is ≈ 0 (−0.002…+0.002 bits).

**Notes:**

- The first 70B pass was an instrument failure that looked like a result, and the lesson is worth as much as the number. Llama-3.3-70B via a serverless API, teacher-forcing the LR echo in a plain-text format: the raw diagonals looked enormous (1.6–4.5 nats) — and decomposed as **73–89% generic context lift**. The streams are more likely under *any* concept context than under neutral, which identifies nothing. Calibrated, all three arms read ≈ 0. We recorded it as an instrument-qualified null, not a point on the Qwen curve.
- **A note on 70B vs 72B** (both numbers appear). The pre-registration named a Qwen2.5-**72B** frontier target, but that run never happened; the cross-family run that did happen is **Llama-3.3-70B**, which every "70B" here refers to. The frozen 72B predictions were therefore *not scored* on the 70B null — a decision made after the result was known, on two grounds documented in the verdict doc: a different model family than the prereg named, and an instrument the faithful-template re-score later showed format-invalid. ("72B-class reader" in the follow-ups means a Qwen-72B-scale reader — the unrun target — not the Llama run.)
- The discriminator that resurrected the signal: re-score the **same 810 streams** under the real Llama chat template. Sustain flips from +0.013 to +0.244 and evoked from 0.000 to +0.143. The lesson: **the teacher-forcing format is part of the instrument.** A plain-text echo of a chat-templated generation destroyed real signal in two of three arms, and we had already written up the null before finding that out.
- The cross-family rider's registered framing, verbatim in spirit: this is a *tripwire, not a privacy proof*. The design can't separate "family-private" from "reader can't engage the other family's gibberish," and 3 of 8 confirmatory cells were instrument-void. But the adverse surprise — a char-passing Qwen read of a Llama mark — did not occur.

---

## Limits

- The "context-induced trace closes with scale" claim is measured on the *within-wording* number (0.45 → 0.18 → 0.02), which conflates concept and wording. The trace that survives the paraphrase swap — the likelihood reader's paraphrase transfer, the one portable concept signal we found — has now been measured across the grid: **0.166 / 0.073\* / 0.129** bits at 1.5B/3B/7B (\*the 3B point is one direction only — its paraphrase gauge failed). At 14B it reads **0.083** (gauge pending by design). That point is above the 0.05 existence floor on the primary but below it length-matched, so the 14B positive is length-sensitive. The trace *persists* — the frozen call predicting it would collapse by 7B scored wrong. But its scale *trend* is length-confounded and unresolved, so treat the closing claim as about wording residue.
- Three model sizes in one family (four for the injected channel and the secret diagonal, plus one cross-family 70B point), so every scale claim is directional, not a fitted law. And an artificial word-free regime throughout.
- The floor-and-ceiling caveats from *How we define "bit"* and its appendix — every reader a lower bound, ≈ 0 reader-relative, the 12-concept closed world as the ceiling — apply throughout.
- The privacy claims in particular are "invisible to the readers we **fielded**," never "invisible to trained decoders." Our decoders are shallow and few-shot (logistic heads, ~24 streams per class), where the inversion literature fields ~200M-parameter inverters trained on millions of victim outputs. And the victim here is open-weight, so an attacker's training budget is unbounded. On that reader-budget gap we'll make the prediction now, in public, so it's falsifiable. Train an output2prompt-class inverter on fresh word-free streams from the open-weight generator, and test it on the frozen pools. We predict it reads ≈ 0 where the generator's own likelihoods read 0.28–0.88 length-matched bits. If it doesn't, "self-privileged" downgrades to "cheap-reader-invisible" — and the regime map survives either way.
- All internal-state measurements are along the injected-vector basis: a persona could in principle write directions we don't probe, though its behavioral effects and our five-way confound run make that residue small and off-axis.
- The paraphrase test's faint aligned trace shows concept information *is* present in the streams. And Subliminal Learning implies a same-initialization learner — for whom the shared weights act as a private codebook — might integrate residue no external reader can calibrate on. Whether that survives the word-free regime is untested (one distillation run would tell).
- The 3B "marginal" verdict is marginal (two of its three seeds fall below the pre-registered threshold).

## Why it matters for safety

Two implications — one measured, one open.

**Monitoring this channel has to live with the weights.** When a model holds a state it never puts into words — clearest case, a secret it was told to keep — that state marks its output in a way no *external* reader recovers: not a transcript monitor, not a logprob monitor, not another model, not a larger one. Only the generating model's own likelihoods read it, and the privilege extends to bare *detection*: an outside observer can't even tell the output was conditioned. Any oversight regime that reads a model's outputs from the outside is, for this channel, blind — and the one reader that isn't is the model itself, exactly the monitor a safety regime is least able to trust. (This is route-specific: a state absorbed *naturally* from context barely leaks to anyone, and a *steered* state leaves an output fingerprint an external auditor can catch. How monitorable the state is depends on how it got in — the table up top is that map.)

**Passively this is small; deliberately, it is the covert-channel question.** Everything measured here is leakage from a model that isn't trying to communicate. But a model that can read its own never-verbalized marks holds the substrate for a covert channel — to a later instance of itself, or a same-weights collaborator — straight through a monitor that sees nothing. Whether a model can *use* that channel on purpose, rather than leak into it by accident, is untested, and is the question that matters most. This work maps what leaks by accident; what leaks on purpose is next.

## What's next

Two follow-ups that would otherwise top this list are already answered above: varying the reader over fixed streams became Experiment 8 (no reader we fielded beats the generator at any cleanly-scored size ratio — the largest clean bigger-reader cells are 2.3× and 2.0×; the 4.7× and 9.3× cells are instrument-void, so unanswered rather than answered), and the weak-dose injection question ran as the confound suite's E1. What remains open:

1. **Deliberate encoding, still untested — and the first design we tried was unusable.** We designed the obvious cell — inject a state, *instruct* the model to imbue "this feeling" into every keystroke — and **withdrew it before collecting any data**. "This feeling" has no context referent, so a positive result couldn't distinguish passive style-amplification from genuine introspective encoding. We recorded a post-mortem. The registered successor is the **onset/offset cell**: inject during the first half of generation, stop, and ask whether the mark persists through self-generated context. That is text-mediated persistence after the state is gone — the mechanism a deliberate encoder would actually need.
2. **The trained-inverter attack** — the open call from the limits section: the one reader class with literature precedent that our privacy nulls haven't faced. It is a small, fast experiment. Either it strengthens the privacy claims or downgrades them, and both outcomes are publishable.
3. **The frontier curve-top.** The conditional within-family 72B run closed per the pre-decision: the "climbs regime-clean" trigger did not fire at 14B, so the trend question closes as a bend. The remaining cheap option is a serverless large-reader observer probe — does a 72B-class reader crack the small models' marks? It is cheap to run and would settle the 10×+ observer ratios the grid couldn't.

All three move from measuring what leaks by accident toward whether a model can leak on purpose, and to which readers.

## Predictions, scored

Every quantitative claim in this project was preceded by a frozen prediction, scored after the data against pre-registered rules. We were wrong a lot — including on the three headline results: we predicted the secret channel wouldn't exist (0.163–0.405 bits), predicted it would keep growing at 14B (it bent), and mispredicted the re-injection dose curve (it crossed at s5, not ~s40, and went to ceiling). The full scorecard — every call as frozen, with verdicts including the unresolvable ones — is in the repo's verdict documents.

## Appendix: the bits measure in detail

Every number in this post measures the same quantity: how much a specific reader learns about *which of 12 equally likely concepts* a stream carries. We compute it as log2(12) − cross-entropy under a softmax whose one temperature parameter is fitted on a held-out third of the data. That way a confidently-wrong reader scores *negative* rather than zero, and miscalibration can't show up as signal. The ceiling of log2(12) ≈ 3.585 bits per stream comes from our 12-concept design, not from the channel.

Two consequences:

- **Positive bits leaked are the minimum size of the channel.** A reader recovering 0.28 bits proves *at least* that much information is present; a better reader, or a design with more than 12 candidates, could only find more. None of our numbers is a channel capacity: capacity would require an encoder, and nothing here is deliberately encoding into the channel.
- **A ≈0 means the readers we used recovered nothing, not that no information is present.** A stronger reader might recover more. This is why our central claim is a comparison between readers rather than any single number. The bits the generator recovers sit far above anything an external reader recovered. Both numbers are lower bounds, so a stronger reader could raise either one. The generator's side can only go up, which makes its advantage secure in one direction — but whether a much stronger external reader could close the gap is an open question, and the Limits section makes our prediction about it falsifiable.

Two interpretation notes:

- Summed-per-stream bits grow with stream length. So whenever we claim a trend, we also report a length-matched secondary: the same measurement recomputed at a fixed stream length, so that longer streams cannot score higher just by being longer.
- The injected and prompted channels answer *different* 12-way questions under incommensurable interventions. Prompted bits identify which *text* was in the prompt (one differing sentence) and what concept direction--if any--comes from that text. Injected bits identify which *direction* was pushed into the activations at an experimenter-chosen dose. They also scale with that dose, and dose has no principled conversion to "one sentence of prompt." The two channels were also measured on different sets of generated streams, collected in separate runs with different filters and counts. Those streams differ in length, which matters because longer streams leak more bits just by being longer. So numbers are never compared between the two channel families. The only comparison we make across that line is the qualitative one of which readers see anything at all.

Our usage follows the usable-information / V-information definition (information extractable by a given function class; Xu et al. 2020, arXiv:2002.10689) and what MDL probing measures (Voita & Titov 2020, arXiv:2003.12298). Shannon (1951) bounded the entropy of English from a human predictor's guesses, which is the same bound-from-a-reader approach we use here.

## Reproducibility

- **Code:** this repository (`experiments/exp1_epistemic_privilege`, `exp2_output_monitorability`, `exp3_induction_and_scale`). Analyses reuse a shared, unit-tested reader core; exp3's induction primers and all interpretation thresholds are **frozen before the readers ran** (`experiments/exp3_induction_and_scale/primers.py`). The 14B scale run is pre-registered in `experiments/exp2_output_monitorability/reports/scale14b_prereg.md` (frozen body + a dated pre-data amendment) with its scored verdict in `scale14b_verdict.md`; the confound-closing suite (E1–E5) likewise in `confound_closing_prereg.md` / `confound_closing_verdicts.md`, with frozen predictions scored as written in both (the free-association state measurement's artifact is `gauge_trajectory_verdict.json`, scored by `analysis/score_gauge.py`). The follow-up arc: same-model readers in `lr_reader_prereg.md` / `mc_reader_prereg.md` (+ `mc_reader_verdicts.md`); the reader × generator grid in `lr_scale_grid_prereg.md` / `lr_scale_grid_verdicts.md` (read the Amendment-6 and review addenda — the length-matched numbers and the corrected 3B char control live there); the 14B extension, re-injection dose curve, and 70B rider in `lr_scale_extend_prereg.md` / `lr_extend_verdicts.md`; the 70B arc in `lr_72b_prereg.md` / `lr_72b_fullsweep_verdict.md` (including the faithful-template addendum); the detection asymmetry in `detection_asymmetry_results.json`; the figures in `analysis/make_blog_figs.py`, which reads the committed results JSONs. All titration, trajectory, transfer, and permutation evidence JSONs are committed alongside.
- **Data:** the generated stream bundles and per-experiment results are on Hugging Face — [`ErrareHumanumEst/internal-state-from-gibberish`](https://huggingface.co/datasets/ErrareHumanumEst/internal-state-from-gibberish) (folders `exp2/`, `exp3/`), including per-item gauge judgments.
- **Per-experiment writeups** (full method, confounders, verdicts): `experiments/*/reports/experiment.md`.
- Builds directly on the **open-introspection** introspection-at-scale work by [@ostegm](https://github.com/ostegm) — see [blog post 04](https://ostegm.github.io/open-introspection/blog/posts/04-introspection-at-scale.html) and the [repo](https://github.com/ostegm/open-introspection). We take a lot from there: the 12-concept set, the difference-vector extraction method, the introspection-layer depth convention (~0.78), and the generation-only injection convention (their post 02) that keeps the prompt prefill clean. This project is also a cousin of **Subliminal Learning** (Cloud et al. 2025, arXiv:2507.14805; see also "Subliminal Learning Is Steering Vector Distillation," arXiv:2606.00995, which bridges the distillation and steering framings).
