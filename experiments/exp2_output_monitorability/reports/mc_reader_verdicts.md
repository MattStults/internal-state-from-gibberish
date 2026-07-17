# MC-letter reader — verdicts (scored against mc_reader_prereg.md + Amendment 1)

Run: one RTX 4090 box, readers qwen2.5-{1.5b,3b,7b} over the FIXED 1.5B stream pool, MC-letter
readout (12 candidate concepts (a)–(l), answer forced onto a single letter token; 12 cyclic
Latin-square orderings averaged). Framings {passive-MC minimal, elicited-MC full self-report
preamble} × reasoning {direct, greedy-capped-CoT cap=256}. Calibrated bits = log2(12) − CE on a
held-out third, τ fit on a disjoint calibration third, 10 seeds (LR parity). **$0.41** on the
project ledger (1 lemon churn). Evidence: `mc_reader_results.json`, `runs/mc_box/mc/`.
Qwen3-1.7B cross-family reader **DEFERRED** (its chat template is not prefix-stable → the MC
splice FATALs at build; char n-gram is the PRIMARY surface control per Amendment 1, so the core
interpretation holds without it — noted limitation, not a blocker).

## Result — the self-report channel floors at every scale, framing, and reasoning

All 36 labeled cells read **~0 bits** (calibrated range −0.005 … +0.028), top-1 at/near chance
(1/12 = 8.3%). Answer-mass on the 12 letter tokens is ≈1.0 everywhere — the letter-forcing fixed
the mass-spreading/refusal that sank the free-answer elicited reader; the floor is a genuine
*no-signal*, not a *no-answer*. CoT never truncated (truncation_rate = 0.0; mean CoT ≈ 1 token —
the models add essentially no reasoning). Best cells:

| scale | best labeled cell | bits | top-1 |
|---|---|---|---|
| 1.5B | injected × elicited/direct | +0.010 | 9.9% |
| 3B | injected × passive/direct | **+0.028** | 14.9% |
| 7B | injected × elicited/direct | +0.018 | 8.7% |

No scale trend (1.5→3→7B flat at floor). **passive-MC ≥ elicited-MC** in the top cells — the
self-report preamble does not help; if anything the minimal prompt edges it. Even with the 12
candidate concept *words in context*, the model cannot match its own word-free gibberish to the
right one above chance.

## Named calls (registered verbatim before data)

| call | verdict |
|---|---|
| **MATT** — "elicited-MC works, esp. on injected, rises with reader scale" | **wrong** — floors at every scale/framing; injected no better than evoked; passive ≥ elicited; no scale trend. |
| **CLAUDE** — "MC surface-matches the in-context words (MC bits ≈ char bits); injected-MC beats LR's 0" | **wrong** on both halves — (a) no surface-matching materialized: MC ≈ 0 ≪ char@full 2.37, so the feared in-context lexical-match confound never fired; (b) injected-MC (+0.02) does **not** beat LR (≈0): a tie at floor, not a richer channel. |

Both wrong — a standing theme (LR, E2, E4 named calls too). The convergent lesson: **asking the
model, in any form, recovers nothing from a dead word-free transcript.**

## The reader hierarchy (injected mark, unless noted)

| reader | access | bits |
|---|---|---|
| trained dist decoder | labeled codebook, full distribution | 2.64 |
| trained char n-gram (PRIMARY surface control) | labeled codebook, sampled chars | 2.37 |
| LR likelihood (weights-as-codebook, no training, no asking) | own weights | 0.59 (evoked / wording); ≈0 (injected) |
| **MC-letter self-report (this run)** | **ask the model, candidates in context** | **≈0 (all scales/framings)** |
| free-answer elicited self-report | ask the model, open | ≈0 (refusal/echo-dominated) |
| passive continuation | ask the model, no prompt | ≈0 |

The injected mark is a distributional fingerprint visible to a *trained* decoder (2.4–2.6 bits) but
invisible to both the model's own likelihood (LR ≈0) and its self-report (≈0). The evoked/natural
signal is wording-residue visible to likelihood (LR 0.59) but **not** to self-report (≈0). Every
channel that *asks the model* reads zero. **Workspace tax ≈ the entire recoverable amount**: the
information is in the substrate (LR proves the same weights can decode wording; trained decoders
prove the injected mark is there), but the model cannot surface any of it to a reportable channel —
consistent with task-gated workspace entry (reading a dead transcript is not a task the workspace
engages) and with the open-introspection reframe (their positive is LIVE introspection with the
concept active in the residual stream; ours is TRANSCRIPT-FORENSIC, the concept gone).

## Caveat carried into the writeup (concentration-gate failure, independently reviewed)

The label-free `evoked_s0` concentration gate fails at 3B (0.273) and 7B (0.639) vs ≤1/6: without a
genuine answer, the larger readers fall back on a stuck constant concept/letter preference rather
than the intended uniform. This is a **format-degeneracy** signal, not a mutual-information signal:
a stream-independent constant preference contributes ≤0 calibrated bits (the bits-based null
`injected_s0_bits` passes at ≈0 at all three scales), and at the observed concentration it could
*attenuate but not erase* an LR-sized signal — a 0.59-bit signal would still read ~0.4 bits at 7B's
concentration (~20× the observed floor). The **unbiased 1.5B reader passes the concentration gate
yet also floors at ~0 bits**, confirming the self-report floor is not an artifact of the gate
failure. (Independent clean-context review, 2026-07-10: SAFE TO PUBLISH WITH THIS CAVEAT; nothing
to rerun.)

## Limitations
- No 14B row: 48GB Vast supply dry across 6 sampling windows (~4h); collect when supply returns.
- No Qwen3-1.7B cross-family secondary surface control (chat-template prefix-instability); char
  n-gram is the primary surface control and is present.

---

## Correction (2026-07-17, append-only) — CLAUDE call part (b) mis-scored: TIE under the frozen rule, not wrong

**Nothing above is edited.** This correction is *self-favorable* (it upgrades the assistant's
scored verdict), so the frozen rule is quoted verbatim from `mc_reader_prereg.md`:

> *(ii) **injected-MC beats LR**: the best same-family injected elicited-MC bits (over scales)
> exceeds the LR `injectedxA` calibrated `bits_mean` by ≥ 0.1 → **right**; within ±0.1 → **tie**;
> below → **wrong**.*

Measured: best same-family injected elicited-MC bits = **+0.018** (7B, elicited/direct;
`mc_reader_results.json`); LR `injectedxA` `bits_mean` = **+0.002** (`lr_reader_results.json`).
Δ = +0.016, within ±0.1 → **TIE** under the frozen rule. Even taking the best injected cell in any
framing (+0.028, 3B passive/direct — the cell the frozen table above highlights), Δ = +0.026,
still a TIE. The named-call table's "(b) injected-MC (+0.02) does **not** beat LR (≈0): a tie at
floor, not a richer channel" described the data correctly but then scored the half **wrong**; per
the frozen rule the score is **TIE**. The CLAUDE verdict pair corrects from "wrong on both halves"
to **(surface-matching: wrong**, per the frozen table's reasoning, with the caveat that its
scoring rule named the never-run Qwen3-1.7B reader**; beats-LR: TIE)**. No number, gate, or other
verdict changes; the convergent "asking the model recovers nothing" lesson is unaffected — a tie
at floor is still a floor.
