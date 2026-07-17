# exp2 — Output monitorability: predicting the token budget to recover a concept

The follow-up to [exp1](../exp1_epistemic_privilege/). exp1 showed word-free output leaks an injected concept
and that recovery is *reader-relative* (its original "weakens with scale" reading was later retracted as a
dose artifact). exp2 turns that into a **predict-then-demonstrate**
question: **how many output tokens does it take to recover *which* concept was injected, can we predict that
budget from the per-token leakage, and how does the budget scale with model size?**

- **The design:** [`proposed-followup.md`](proposed-followup.md) — hypothesis, the token-budget law
  (predict-then-verify), the writing-vs-reading split, confounders, and what it can/can't establish.
- **Background lit/math:** [`research_logit_state_reconstruction.md`](research_logit_state_reconstruction.md)
  — the logit→state reconstruction line (PILS/Carlini), which the design *descopes* (we recover concept
  identity, not the steering vector).

## Status: complete

Ran **offline on the existing `runs/_ab/` bundles** (the exp1 injection-method release) — no new collection.
The result, as corrected by the full-stream re-analysis (the update blocks at the top of
[`reports/experiment.md`](reports/experiment.md)): measured in *bits of concept identity recovered*
(`H(concept) − cross-entropy`, a best-decoder lower bound on I(concept; output)), a distribution-access
monitor recovers far more than the best honest **token** monitor *at the original 12-token capture* — but
that is a monitoring-**latency** gap, not a capacity one. Given the full stream the transcript reader
reopens the gap and then some: the char reader (exp1's symbol-counter, scored here in bits) accumulates to
**2.37 ± 0.16** bits on the fixed ≥64-token cohort, catching and passing the diluting distribution reader
(1.53 ± 0.24). Distribution access buys *speed* (~10× fewer tokens for the same bits), not exclusivity; the
genuinely distribution-only result belongs to the **natural** induction regime (exp3). Full result,
confounders, and verdicts: [`reports/experiment.md`](reports/experiment.md).

## Layout

| Path | What |
|------|------|
| `analysis/` | the bits currency (`info.py` + concept bootstrap), per-channel nested-CV best decoders (`reader.py`), the four readers — `dist` (next-token distribution), `R_emb` (realized-token embeddings), `sampled` (one-hot floor), `char` (transcript symbol-counter) — and the token-budget curves + full-stream + bits-ladder cross-checks (`run_budget.py`), on the shared `runs/_ab/` bundles. |
| `tests/` | offline unit tests (RED-first), CPU sklearn only. |
| `reports/` | `experiment.md` (tracked); generated `budget_results.json` / `budget_curves.png` live on HF. |

## Reproduce (offline, from the released streams)

```bash
.venv/bin/python experiments/exp2_output_monitorability/analysis/run_budget.py \
    runs/_ab/qwen2.5-1.5b-gen.pt runs/_ab/qwen2.5-3b-gen.pt runs/_ab/qwen2.5-7b-gen.pt
```

Writes `reports/budget_results.json` (+ `budget_curves.png`). This is heavy nested-CV — run it on a rented box,
**not** a laptop, via `harness/run_reanalysis.py` (gated through **experimentfactory** `authorized_run`). The
`char` reader needs `transformers` (tokenizer) and each model's embedding matrix (`load_embed_matrix`, extracted
from HF into `artifacts/`).

### Released-data schema
Each `runs/_ab/*-gen.pt` bundle: `{streams, strengths, concepts, model, inject}`; each accepted stream carries
`gen_topk` (per-step top-64 `{ids, logp}`), `tokens` (realized ids), and `concept_idx`. `budget_results.json`
keys per model: `readers.{dist,emb,sampled,char}.{bits_mean,bits_sd}` (per budget T), `best_monitor_gap_bits`
(dist−R_emb), `dist_minus_char_gap_bits`, `bootstrap_ci` (concept-level CI on top-budget bits + gaps),
`full_stream` (each stream at its own length), and `ladder` (K-way = log₂K-bit calibration).
