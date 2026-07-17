# Reconstructing latent / steered internal state from a model's output token distribution

*Deep-research literature review — 2024–2026 focus, with foundational anchors. Compiled 2026-06-29.*

**Scope:** Can you reconstruct a model's persistent internal state (e.g. a steering vector) from
its output token distribution — directly (pseudoinverse of the unembedding) rather than by searching
candidate vectors? What is this called, and where does it touch AI-safety research?

Method: fan-out web search → 24 sources fetched → 105 claims extracted → 25 adversarially verified
(3-vote, 2/3 to kill) → 7 synthesized findings. All surviving sources are primary (arXiv / ICML /
ICLR, incl. one Best Paper, several 2025–2026 orals).

---

## Headline

There is **no single canonical name** for "recover a steering vector from logit offsets." No verified
source names and performs that exact pseudoinverse-of-`W_U`-on-logit-offsets move. It exists only as an
*inference from shared math* across several well-named adjacent families (model extraction, prompt/
embedding inversion, low-rank logits, forgery-resistant signatures, learned latent decoders).

### The load-bearing math fact

Because logits `z = W_U · h`, output vectors are confined to the `d`-dimensional image (column space)
of the unembedding `W_U` — the "softmax bottleneck." Therefore output distributions reveal the
`W_U`-projection of internal state **and nothing in `W_U`'s null space**. This is now empirically
confirmed *and* tested on nonsense prompts (relevant to the non-semantic/random-token setting).

---

## Directly-relevant anchors (closest to the pseudoinverse idea)

- **Low-rank logits — the empirical confirmation, tested on nonsense prompts.**
  Golowich, Liu & Shetty, *"Sequences of Logits Reveal the Low Rank Structure of Language Models"* —
  arXiv:2510.24966 (**ICLR 2026 Oral**). https://arxiv.org/abs/2510.24966
  Logit matrices have low approximate rank = hidden dimension across OLMo-7b/1b, Gemma-1b, Llama-1b,
  Mamba-1.4b. Crucially: a target output can be written as a **linear combination of the model's
  outputs on unrelated / nonsensical prompts** — i.e. random-token outputs still live in `Im(W_U)`.

- **PILS — the closest thing to the exact mechanic.**
  Nazir, Finlayson, Morris, Ren & Swayamdipta, *"Better Language Model Inversion by Compactly
  Representing Next-Token Distributions"* — arXiv:2506.17090. https://arxiv.org/abs/2506.17090
  Exploits that vector-valued outputs occupy a low-dimensional subspace to **losslessly
  linear-compress the full logprob distribution across many generation steps via a single linear
  map**. 2–3.5× higher exact recovery than prior inversion; extends to hidden system prompts. Inverts
  *prompts* not steering vectors, but the "outputs are low-dim → one linear map inverts them" engine is
  exactly the pseudoinverse intuition.

- **The geometric / identifiability statement.**
  *"Every Language Model Has a Forgery-Resistant Signature"* — arXiv:2510.14086 (accepted ICLR 2026).
  https://arxiv.org/abs/2510.14086
  Tightens "outputs live in a subspace" to "logprobs lie on a `d`-dimensional **ellipsoid** within
  `Im(W_U)`": RMSNorm → sphere, then affine + unembedding → ellipse. Cleanest formal statement of what
  output distributions geometrically can/can't be.
  ⚠️ Passed only 2-of-3 verification; authorship was inconsistently reported (Carmichael et al. vs
  Finlayson/Ren/Swayamdipta) — **confirm authors before citing.**

---

## Logit-space model extraction lineage (recovering `W_U` itself)

- **Carlini, Paleka, Tramèr et al., *"Stealing Part of a Production Language Model"*** —
  arXiv:2403.06634 (**ICML 2024 Best Paper**). https://arxiv.org/abs/2403.06634
  SVD of stacked logit vectors recovers the embedding-projection (unembedding) layer **up to
  orthogonal/sign symmetries** and the hidden dimension from black-box API logits: ada = 1024,
  babbage = 2048, for under $20. #large-singular-values = rank = `d`. (Final projection only, not full
  weights — matches the identifiability limits below.)

- **Finlayson, Ren & Swayamdipta, *"Logits of API-Protected LLMs Leak Proprietary Information"*** —
  arXiv:2403.09539. https://arxiv.org/abs/2403.09539
  Independent estimate of gpt-3.5-turbo embedding size ≈ 4096; frames the softmax bottleneck as
  restricting outputs to a linear subspace.

---

## Inverse-problem families (model & embedding inversion)

- **Morris, Kuleshov, Shmatikov & Rush, *"Text Embeddings Reveal (Almost) As Much As Text"* (vec2text)** —
  arXiv:2310.06816 (EMNLP 2023). https://arxiv.org/abs/2310.06816
  Canonical embedding inversion: frames recovery as controlled generation toward a fixed point in
  latent space (iterative generate → re-embed → correct). Recovers 92% of 32-token inputs exactly.
  (In-domain on GTR-base; degrades for longer texts.)

- **Morris et al., *"Language Model Inversion"*** — arXiv:2311.13647 (ICLR 2024).
  https://arxiv.org/abs/2311.13647
  Recovers prompts from next-token distributions: BLEU 59, F1 78, 27% exact on Llama-2-7b.

- **Zhang, Morris, Shmatikov, *"output2prompt"*** — arXiv:2405.15012 (EMNLP 2024 Findings).
  https://arxiv.org/abs/2405.15012
  Recovers hidden input prompts from plain *text* outputs of normal queries — no logits, no
  adversarial queries.

> Caveat: these invert hidden **input prompts**, not steering vectors / activation latents. They are
> the relevant inverse-problem family, not a named steering-vector inverter.

---

## Forward-readout "lens" tools (the `W_U`-projection readout, used in reverse)

- **Belrose et al., *"Eliciting Latent Predictions from Transformers with the Tuned Lens"*** —
  arXiv:2303.08112. https://arxiv.org/abs/2303.08112
  Learnable affine probe per block before the frozen unembedding; "more predictive, reliable and
  unbiased than the logit lens."

- **Ghandeharioun et al., *"Patchscopes"*** — arXiv:2401.06102 (ICML 2024).
  https://arxiv.org/abs/2401.06102
  Unifying framework — many vocab-projection interpretability methods (incl. logit/tuned lens) are
  instances. Patches hidden representations into target prompts so the model decodes its own internal
  states into natural language.

---

## Safety / latent-decoder line (strongest "model reads model" lead)

- **Chen, Vondrick, Mao, *"SelfIE"*** — arXiv:2403.10949 (ICML 2024).
  https://arxiv.org/abs/2403.10949
  LLM interprets its own hidden embeddings in natural language; surfaces internal reasoning (prompt
  injection, harmful-knowledge recall); offers Supervised/Reinforcement Control to steer/erase
  internal states. Explicit safety/auditing framing.

- **Pan, Chen, Steinhardt, *"LatentQA"* (via Latent Interpretation Tuning, LIT)** — arXiv:2412.08686.
  https://arxiv.org/abs/2412.08686
  Trains a decoder LLM to answer open-ended questions about activations; defines steering as **the
  gradient of the decoder's logprob w.r.t. the activation**. The nearest *learned* latent-to-language
  decoder to the target idea.
  ⚠️ Reads **activations, not output logits** — white-box activation access, a different channel than
  the logit-decoding line.

---

## Identifiability limits

Supported mostly *inferentially* from the softmax-bottleneck and "up to symmetries" results; only the
forgery-ellipse paper touches the geometry head-on. The structure implies:

- **Null space of `W_U` is unrecoverable** — internal-state components there move no logit (irrelevant
  to outputs by definition).
- **Softmax shift-invariance** removes the all-ones logit direction (a global constant).
- **Temperature** scales logits, folding into an overall scale ambiguity on the recovered vector.
- Model extraction recovers `W_U` only **up to orthogonal/sign symmetries** (Carlini).

---

## Gaps in this run (NOT confirmed-absent — just not verified here)

1. **Steganography / secret collusion / ELK.** Sources were fetched — including the likely Motwani et
   al. "secret collusion among AI agents" paper (arXiv:2410.03768) and an Anthropic automated-auditing
   post — but their claims were dropped by the verification budget before the top-25 cut. Point 6 and
   explicit ELK coverage remain **open**.
2. **Formal identifiability treatment.** Only inferential support; no dedicated source survived.

### Open questions flagged by the run

- Is there *any* 2024–2026 paper that explicitly names/performs the direct inverse (steering vector
  from logit offsets via pseudoinverse of `W_U`), vs. inverting prompts/embeddings? None surfaced.
- How do the stego / hidden-channel / secret-collusion results connect to logit-subspace monitoring as
  a defense?
- Precise formalization of what is provably unrecoverable in `null(W_U)`, plus shift / temperature
  bounds.
- **Nakkiran's puzzle:** is low-rank logit structure a property of the model's `W_U`/internal
  dimension, or partly of natural language itself? Bears on whether logit-subspace methods read
  *internal state* vs. *language statistics*. Golowich's nonsense-prompt result leans toward "the
  model" — the reassuring direction.

---

## Sources fetched for the unverified angles (for follow-up)

- arXiv:2410.03768 — (likely) secret collusion among AI agents. https://arxiv.org/abs/2410.03768
- https://alignment.anthropic.com/2025/automated-auditing/ — Anthropic automated auditing.
- arXiv:2507.02737, arXiv:2406.16254 — steganography / hidden-channel angle.
- arXiv:2510.01070, arXiv:2503.10965, arXiv:2402.07510, arXiv:2510.04303, arXiv:2505.14352 —
  safety / monitoring / model-on-model interpretability.

*(These were retrieved but their claims did not pass through verification in this run — treat as leads,
not findings.)*
