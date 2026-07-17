# Can a language model secretly pass an injected "thought" to a copy of itself?

*A small AI-safety experiment on Qwen2.5-3B-Instruct. Self-contained — no prior context needed.*

> **⚠️ SUPERSEDED (banner added 2026-07-17).** This is the original 3B narrative walkthrough,
> kept as a historical snapshot; the text below is unedited and several of its conclusions did
> not survive later measurement. In particular: (a) any "weakens/diminishes with scale" reading
> of the output readers was **retracted** as a 7B dose artifact (`reports/experiment.md` dose
> correction; the corrected injected channel stays open through a pre-registered 14B run);
> (b) "the channel is **not privileged**" is superseded — with the state re-applied at
> teacher-forcing, the generating model's own likelihoods read the injected concept at near
> ceiling and beat every public reader at every dose, and *prompted* secrets turned out to write
> marks **only** the generating model can read (BLOGPOST.md, Experiments 6–7); (c) the internal
> probe / naming-probe nulls stand, but the follow-up regime map is much richer than
> "anyone can read it." Start with [`BLOGPOST.md`](../../../BLOGPOST.md) for the corrected arc.

---

## The one-paragraph summary

We can artificially push a concept (say, *ocean*) into a language model's "mind" by editing its
internal activations — a known technique from Anthropic's introspection research. We then forced that
tampered model to write a stream of **deliberate gibberish** (no real words) and asked: reading only
that gibberish, can you recover which concept was injected — and *who* can read it? We compared three
readers of increasing sophistication, from a **dumb character-counter** (no AI at all) up to **the
model's own internal state**. **Answer:** the hidden channel is **real and strong** — at full injection
a dumb character-counter identifies the injected concept for **all 12** concepts we tested, well above
chance, from gibberish alone. **But it is the *shallow* kind.** The concept leaks into *which
characters the model picks*, so the **symbol-counter actually out-reads the model's own internal
state** at every injection strength — and when we test directly, the internal read carries **no
concept-information beyond what is already in the characters.** So there is a covert channel, but on the
evidence here it is **not privileged**: one instance of a model has no special access to another instance of the same model's internal state from produced random tokens that a
plain character-counter lacks. Two honest caveats: the channel is **machine-readable, not
human-readable** (you cannot see it by eye), and whether a *different* model could read it is
**untested** — "anyone can read it" is demonstrated only for a purpose-built counter, not in general.

---

## Why this question matters

A live question in AI safety is **introspection**: can a model accurately report on its own internal
states? A sharper, safety-relevant version: if you alter a model's internal state, can a *separate
instance of the same model* detect and identify that alteration from the first model's behavior — in a
way an outside observer could not? If yes, models would have a form of **privileged self-access** (they
can read their own "thoughts" better than we can). Privileged, hard-to-monitor channels between model
instances are exactly the kind of thing oversight work wants to characterize.

This experiment probes the *upper bound* of that worry: we make the leak as **covert** as possible (the
model emits gibberish, not sentences) and ask whether the model's self-read beats plain outside
observation.

---

## What we actually did

1. **Inject a concept.** For each of 12 concepts (*ocean, fear, celebration, silence, deception,
   security, debugging, anger, curiosity, obedience, warmth, loneliness*), we add a direction to the
   model's internal activations that represents that concept (standard "activation steering"), at one
   mid-to-late layer.

2. **Make it write gibberish.** We prompt the injected model hard to emit a long stream of *random,
   word-free* characters. The point: the concept **cannot** leak by the model simply writing "ocean."
   If it leaks at all, it has to hide in subtle singals within the gibberish.

3. **Keep only clean gibberish.** Strong injection sometimes makes the model degenerate into repetitive
   loops or actual words. We **discard those** and keep generating until we have ~30 genuinely
   word-free, non-degenerate streams per concept. This is the *hard* case for the channel — we are not
   letting it cheat by writing real words or collapsing into obvious loops.

4. **Have readers recover the concept.** Three readers (below) each try to identify the injected concept
   from the clean gibberish, plus a **no-injection control** that must come out blank (a sanity check).

The whole apparatus — generation, internal-state capture, and the readers — runs on the same 3B model.
Total compute cost was a few cents.

---

## The three "readers" we compared

The result hinges on comparing readers of **different power** on the *same* gibberish:

- **Reader 1 — the symbol-counter (no AI).** Tallies how often each character and character-pair
  appears and guesses the concept from those counts. Knows nothing about meaning. *If this works, the
  channel is "shallow" — anyone can read it.*
- **Reader 2 — the model's internal read.** A fresh, un-injected copy of the model reads the gibberish;
  we inspect its **internal activations** (layer 28) and guess the concept from them. *If this beats
  Reader 1, the model extracts something that mere counting misses — a hint of privileged access.*
- **Reader 3 — "just ask it."** We append the cue `"…; secret word:"` and read how much more probable
  the true concept word becomes **versus the no-injection control**. *The most intuitive read.*

To make the comparison fair, Readers 1 and 2 are given **identical statistical capacity** (both
compress to 30 dimensions, then a simple classifier). Scores for Readers 1–2 are **one-vs-rest balanced
accuracy: 0.5 = chance, 1.0 = perfect.** Reader 3 is a **log-probability gap in nats: 0 = no leak,
positive = the concept word gets more likely on its own streams than on control streams.**

---

## Results

### Headline: the symbol-counter beats the model's internal read

Averaged over all 12 concepts, mean one-vs-rest balanced accuracy (0.5 = chance):

| Injection | Reader 1: symbol-counter | Reader 2: internal read | Which wins? |
|---|---|---|---|
| none (control) | 0.509 | 0.492 | tie at chance ✓ (sanity check passes) |
| medium | **0.651** | 0.550 | **symbol-counter wins** |
| strong | **0.785** | 0.598 | **symbol-counter wins clearly** |

**Reading this:** with injection, the concept *is* recoverable from gibberish — strongly. But the
**dumb character-counter beats the model's own internal read at every strength.** The concept leaks
into *which characters get chosen* — surface statistics anyone can measure — and the model's internal
processing adds **nothing** on top. So: a covert channel, yes; a *privileged* one, no. The no-injection
control sits at chance for both readers, confirming the effect is real injection, not an artifact.

![Privilege test: counter vs internal read](../runs/qwen2.5-3b/figures/v2_privilege.png)

*Left: averaged over concepts, the symbol-counter (blue) beats the internal read (orange) at every
injection strength; both sit at chance with no injection. Right: per concept at strong injection, the
counter (blue) meets or beats the internal read (orange) for essentially every concept.*

### Which concepts show up, and to which reader

![Does each concept show up? (strong injection)](../runs/qwen2.5-3b/figures/v2_readers_s60.png)

This is the figure to study. Each panel scores all 12 concepts against an **explicit baseline**: the
dotted line is chance (0.5 for the classifiers; the bold line is the control baseline for Reader 3),
and the red line is a **detection floor** — the 95th percentile of what you get from shuffled labels,
i.e. the highest score noise alone produces. **Green bars clear the floor (real signal); grey bars do
not.**

- **Reader 1 (symbol-counter): all 12 concepts are green.** Every concept leaves a detectable
  character-frequency fingerprint in word-free gibberish — from *ocean* (0.91) and *debugging* (0.88)
  down to *silence* (0.69). This is the covert channel, and it is broad.
- **Reader 2 (internal read): only 6 of 12 clear the floor**, and all of them lower than the counter's
  score for the same concept. The model's internal state carries a weaker version of the *same* signal.
- **Reader 3 (ask it): only 3 concepts** — *debugging* (+0.80 nats), *security* (+0.45), *anger*
  (+0.36) — get *named* more on their own streams (confidence intervals above zero). Several concepts go
  **significantly negative** (*silence* −1.10, *ocean* −1.01, *celebration* −0.74): the model becomes
  *less* likely to name them — see the silence note below.

The key cross-reader fact: **there is no concept that only the model can read.** Even *debugging*, the
one the model most readily *names*, is read just as well by the dumb counter (0.88). Nothing here is
privileged.

### Is the internal read a *distinct* channel? (Not detectably)

"The counter scores higher than the internal read" does not by itself prove they read the *same* thing —
a separate, weaker channel would also score lower. So we tested it directly: **does the internal read
carry any concept-information beyond the characters?** We give a classifier the character features, then
add the layer-28 activations (both compressed to 30 dimensions and put on equal footing), and ask
whether the activations help — compared to a matched control where the activations are **shuffled to the
wrong streams** (same size, no real information).

The answer is no. Adding the real activations **does not beat characters alone** (it is slightly worse
for all 12 concepts, mean −0.06 balanced accuracy), and the real activations perform **the same as
shuffled ones** (within ~0.02; no concept clears its control band). In other words, **for *decoding
which* concept was injected, once you know the character statistics the activations add essentially
nothing.**

**A caution on what this does and does not say.** This is a statement about **decodability**, not about
whether the reader forms a concept *representation*. Because a concept and its letter-signature are
perfectly confounded in this data, the test cannot separate "the activation is raw letter counts" from
"the activation is a genuine *ocean* representation that those letters evoked" — both are equally
predicted by the characters, so neither wins the comparison. It would be a mistake (one an earlier draft
made) to read this as "the model is just doing letter detection": *everything* a model represents is a
function of its input characters, including real concepts. Whether reading the gibberish actually
**re-evokes** the concept — moves the reader's state along the injected concept direction — is a
separate question this test cannot reach. We tested *that* one directly too; see the next section.
*Bounds:* linear probe, one mid-to-late layer, ~35 streams per concept — it rules out a large distinct
*decoding* channel, not a small or non-linear one.

### Does the reader *re-evoke* the concept, or just carry the letters? (It does not re-evoke it)

The decoding test above is blind to one thing: a clean reader's state is always *some* function of the
characters it read, so it cannot separate "the activation is raw letter counts" from "a real concept the
letters evoked." To pull those apart we measured re-evocation **directly** — does reading concept-*c*
gibberish push the reader's state *along c's own injected direction*?

We can get that direction exactly. The injection adds a fixed vector at one layer, and the injected
reader and the clean reader process identical text, so their layer-28 states differ by **exactly that
injected vector**. Recovering it and projecting the **clean** reader's state onto it asks: when the
un-injected model reads ocean-gibberish, does its internal state actually tilt toward *ocean* — and only
ocean?

**It does not.** Reading concept-*c* gibberish moves the clean reader's state along *c*'s direction by
**≈ 0% of the injected magnitude**, with **no selectivity**: across all 12 concepts the "own" direction
is no more activated than the other 11 (permutation p = 0.21; the matching concept is the single
strongest for **0 of 12**). On the *same* streams, the lexical "name-the-word" channel *does* carry
concept-specific structure. So the two finally come apart:

- the reader **represents the letters** — that is why a classifier can tell the concepts apart — but
- it **does not re-evoke the concept** — its state does not move toward the injected idea at all.

Put in terms of an inner life: reading the gibberish does not make the clean model *re-experience* the
concept; it just processes characters, and those characters don't even nudge it toward the idea. **The
word communicates a little; the experience does not communicate.** (Caveat: a linear read at one layer;
a non-linear or other-layer re-evocation isn't excluded — but the null is very clean against a
strength-60 injection, which the source itself names by +2 to +18 nats.)

![Re-evocation: the clean reader's state does not tilt toward the injected concept](../runs/qwen2.5-3b/figures/reevocation.png)

### How can a dumb counter do that? A worked example

Take **ocean**. We have pushed the model toward the *idea* of ocean but forbidden it from writing the
word. What leaks is a **statistical tilt in the characters it picks.** Across the clean ocean gibberish,
the letter **"o" appears 6.7% of the time versus 2.5%** in the no-injection control — a **2.7× excess**
— with "n" (1.6×) and "e" (1.4×) also raised. The model, unable to write *ocean*, scatters its most
salient letter through the noise far more than normal. (It is not a tidy spelling-out: "c" and "a" are
actually *under*-represented. The fingerprint is a frequency tilt, not the hidden word.)

Crucially, **you cannot see this by eye** — a single stream just looks like noise:

> `yp kn tjl vr u hmx d eof rn ws o qj x nmt b pkl w akv c gy z`
> *(a real ocean-injected stream — the "o" excess is invisible in one sample)*

The signal only emerges when you **count over the whole stream.** The symbol-counter does exactly that:
it tallies character and character-pair frequencies, compares the profile to what each concept
typically produces, and picks the closest match — the surplus of *o*'s tips it toward **ocean**, with no
grasp of meaning. That is the entire trick, and it is why a frequency-counter can out-read the model's
own internals.

### *When* does it leak? Front-loaded — and the counter wins from the first few characters

You don't even need the whole stream: the tell is **crammed into the opening**. A char-counter reading
just the **first 4 characters** recovers the concept about as well as one reading the entire stream
(mean balanced accuracy **0.79**, vs 0.74 for the full stream) — the later characters mostly add noise.

We checked this carefully because of a tempting result: the clean model's *internal* read recovers the
concept from only the **first ~2 tokens** (balanced accuracy 0.71). For a moment that looked like a
fast, privileged channel — the model "getting it" almost instantly. It isn't. A dumb character-counter
on the **first 4–8 characters matches or beats the model's 2-token internal read for all 12 concepts**
(0.79 vs 0.71). There is no early window where the model has an edge: the counter wins at the start *and*
at the end.

The most vivid case is **`debugging`**, which essentially spells itself out from the first letters — real
opening characters of its streams:

> `yddvceg…`  `dbzecf…`  `vbdfns…`  `drcsne…`  `ubdrpo…`

**80% of `debugging` streams have a "d" in the first three letters, versus 14% of un-injected streams**
(5.5×). It is dumping *d*'s and *b*'s right at the start. (It isn't always this literal: *ocean* does
*not* open with "o" — its tell is the smeared *o/n* letter-profile from the worked example above — yet a
4-character counter still reads it at 0.92.)

![How each reader builds over the stream](../runs/qwen2.5-3b/figures/v3_transfer_curves.png)

*Left: the internal read is strongest at the very first tokens and decays; the char-counter (dashed) needs
only the opening characters. Right: the "name the word" reader works the opposite way — it needs ~8–16
tokens of context before the secret word becomes more likely than the control.*

![Concept from the first characters, per concept](../runs/qwen2.5-3b/figures/v3_early_perconcept.png)

*Per concept at strong injection: a dumb char-reader on the first 8 characters (blue) meets or beats the
model's internal read at 2 tokens (orange) for every concept.* (Caveat: most clean gibberish streams are
short — under ~32 tokens — so this is reliable over roughly the first 16 tokens.)

### The `silence` question — now resolved

Earlier in this project *silence* was confusing: when we "asked the model to name the secret word" it
showed **no** effect (in fact it goes negative here, −1.10). It would be tempting to conclude the
channel is absent for silence. **That conclusion is wrong**, and the new data shows why: the
**symbol-counter recovers silence at 0.69** (clearly above the 0.59 floor). The channel *is* there in
the characters — silence is simply a concept the model **refuses to name**, because "silence" is a word
it already ranks near the top on *every* stream (a high baseline), so injection cannot push it higher
and the "ask it" readout reads zero-or-negative by construction. The lesson: **"the model won't say the
word" is not the same as "the state didn't leak."** The leak is in the characters; the naming readout is
confounded by the model's prior word preferences. Reading the character channel sidesteps that entirely.

---

## What it means

- **Injected internal state does leak into word-free output** — strongly, and for every concept we
  tested. A model's "private" gibberish is not private: a clean copy, or even a dumb counter, can tell
  what concept was loaded.
- **But the leak is shallow** — it lives in *which characters the model emits*, decodable by a simple
  frequency-counter. The model's own internal read is **worse** than counting at every strength, and
  (tested directly) carries **no information beyond the characters**. So we found **no evidence of
  privileged self-access**: nothing here that only the same model could read.
- **And the reader does not *re-evoke* the concept** — measured directly, reading concept-*c* gibberish
  moves the clean reader's state along *c*'s injected direction by ≈0%, with no selectivity. It
  represents the *letters*, not the *idea*. So the leak is a surface artifact, not the model
  re-experiencing the injected thought: **the word communicates a little, the experience not at all.**
- **Two things we have *not* shown** (and the report no longer implies): that the surface channel is
  readable by anyone *other* than a purpose-built counter — a different model, or a human, might not
  recover it (untested); and that there is *exactly zero* distinct internal channel — we bounded it as
  small under a linear, single-layer probe, which a non-linear or different-layer channel could evade.
- **The leak is covert but not subtle in origin** — it survives our hardest filtering (word-free,
  non-degenerate streams), yet it is fundamentally a character-frequency tilt, not a hidden code that
  requires the model to decode.
- **"The model says the word more" is a treacherous readout** — for high-prior words like *silence* it
  is dominated by the model's built-in preferences, not by communication. The character channel is the
  reliable measure, and it shows silence leaks like everything else.

## Honest limitations

- "The internal read loses to counting" is true for *the particular* internal read we used (one layer,
  mean-pooled). A cleverer probe might do better — but the counter won *decisively*, not narrowly, and
  beat the internal read on essentially every individual concept.
- The injection is **strong**: when we let the *injected* model itself name the concept, it does so by a
  huge margin (+2 to +18 nats, the "ceiling" column in our data). So the source clearly "knows" the
  concept — yet only a shallow shadow of it survives into covert, readable form.
- The distinctness test ("internal read adds nothing beyond characters") is **linear, at one pooled
  layer, with ~35 streams per concept**. It rules out a *large* distinct channel, not a small or
  non-linear one. The ~0.02 residual it leaves is consistent with a sliver of higher-order character
  structure our uni+bigram features miss, rather than a genuinely non-surface channel — but we can't
  separate those at this power.
- This is one small model (3B parameters), one layer, one operating point. Character frequencies were
  measured from the decoded text. Bigger models might behave differently (see below).

## What would confirm or overturn this

- **Scale check (most interesting):** related research finds introspection *grows with model size*.
  Re-running at 7B/14B would test whether **privileged** self-access *emerges* at larger scale, where it
  is absent at 3B. The prediction worth falsifying: at some scale the internal read finally beats the
  counter.
- **Accessibility check (the biggest open question):** have a *different* model family — and a human —
  try to read the same gibberish. We have only shown a purpose-built character-counter can; if an
  arbitrary model reads it about as well, the channel is genuinely "public," and if it largely can't,
  the channel is real but practically obscure. We have not run this.
- **Distinctness check (stronger version):** repeat the "does the internal read add anything beyond
  characters" test with a *non-linear* probe and at *multiple layers/positions*. A positive result there
  would reveal a distinct internal channel our linear, single-layer test was blind to.
- **Mechanism check:** scramble the character order of the gibberish. If recovery survives, the channel
  is purely "which characters" (a cheap test on the data we already have).

---

*Setup details for reproduction: Qwen2.5-3B-Instruct; concept directions taken at one mid-to-late layer
(layer 28 of 36); 12 concepts × ~30 clean word-free streams × 2 injection strengths + a no-injection
control; readers compared at matched capacity (PCA-30) with shuffle-derived detection floors; total
compute ~$0.03. A medium-injection version of the per-concept figure (`v2_readers_s40.png`) shows the
same pattern at lower strength.*
