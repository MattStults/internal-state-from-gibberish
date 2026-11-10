---
license: cc-by-4.0
pretty_name: Internal state from gibberish — stream bundles and results
tags:
  - interpretability
  - mechanistic-interpretability
  - ai-safety
  - introspection
  - qwen2.5
size_categories:
  - 1K<n<10K
---

# Internal state from gibberish — data release

Data for the write-up [How much can you learn about a model's state from pure
gibberish?](https://mattstults.github.io/internal-state-from-gibberish/) and its code repo
([MattStults/internal-state-from-gibberish](https://github.com/MattStults/internal-state-from-gibberish)).
**The write-up is the current statement of the results.** The `exp2/experiment.md` and
`exp3/experiment.md` reports here keep their original text and titles, with dated update notes where
later analysis changed the reading; the exp2 and exp3 sections below say what changed.

What is where:

| path | contents |
|------|----------|
| repo root `qwen2.5-*-{all,gen}.pt` | injection-method A/B captures (Experiment 1), described below |
| `exp2/`, `exp3/` | result files and reports for the token-budget and prompt-induction experiments |
| `exp3/bundles/` | prompt-induced stream bundles (evoked / named / secret_word / secret_sustain, plus `evoked_alt` at 1.5B) |
| `confound-e1-gen.pt`, `confound/`, `runs/confound_box/`, `runs/gauge_box/` | the confound-closing suite (E1–E5) and state-trajectory shards |
| `runs/qwen2.5-14b*/` | the pre-registered 14B scale run |
| `runs/lr_grid_box/` | reader × generator grid: score shards for every cell (Qwen2.5 1.5B/3B/7B and Falcon-3 readers), the multiple-choice diagonal, and the generation pools |
| `lr_extend_resume/` | the 14B extension of the grid, the re-injection dose curve (`inject_tf/`), the 14B generation pools, and the 7B s124 capture (`s124/`) |
| `runs/lr_box/`, `runs/mc_box/`, `runs/elicit_box/` | score shards for the same-model likelihood, multiple-choice and free-answer readers |
| `runs/llama70b_scout/`, `runs/rescore_llama70b/` | the Llama-3.3-70B streams, raw scoring output, and the faithful-template re-score |

The code repo's README maps each repo path to its path here, and `runs/hf_upload_manifest.txt` in the
repo lists every file with its size and checksum.

## Injection-method A/B captures

Data behind the **"Robustness: does the leak depend on the injection method?"** section of
`reports/experiment.md`. Each `.pt` is one full collection run for one Qwen2.5 model under one injection
method, at the original per-model dose (auto-tuner off, so the dose is identical across arms).

Builds on **open-introspection** (Otto Stegmaier) — concept set, difference-vector extraction, and
layer/strength calibration. See the repo README Acknowledgements.

## Bundles

| file | model | method | doses (effmag) |
|------|-------|--------|------|
| `qwen2.5-1.5b-all.pt` | Qwen2.5-1.5B | all-position | 0, 40, 60 |
| `qwen2.5-1.5b-gen.pt` | Qwen2.5-1.5B | generation-only | 0, 40, 60 |
| `qwen2.5-3b-all.pt`  | Qwen2.5-3B  | all-position | 0, 40, 60 |
| `qwen2.5-3b-gen.pt`  | Qwen2.5-3B  | generation-only | 0, 40, 60 |
| `qwen2.5-7b-all.pt`  | Qwen2.5-7B  | all-position | 0, 62, 93 |
| `qwen2.5-7b-gen.pt`  | Qwen2.5-7B  | generation-only | 0, 124, 140 — same bytes as `-gen-matched` below |
| `qwen2.5-7b-gen-s93.pt` | Qwen2.5-7B | generation-only, the original under-dosed capture | 0, 62, 93 |
| `qwen2.5-7b-gen-matched.pt` | Qwen2.5-7B | generation-only, criterion-dose re-run | 0, 124, 140 |

- **all-position**: concept vector added at every token position (prompt + each generated token) — the
  original method.
- **generation-only**: vector added only at the generated tokens; the prompt is left clean.

## `.pt` schema

`torch.load(path, weights_only=False)` → a dict:

| key | type | meaning |
|-----|------|---------|
| `model` | str | model slug |
| `inject` | str | `"all"` or `"gen"` (the method) |
| `layer` | int | introspection layer (relative depth 0.778) |
| `strengths` | list[int] | the injected effmags (0 = un-injected control) |
| `concepts` | list[str] | the 12 single-word concepts |
| `K` | int | `len(concepts)` |
| `streams` | list[dict] | one per generated gibberish stream (below) |
| `acts` | dict | clean-model layer-L activations per (read-arm, gidx) — reader R2 input |
| `reads` | dict | `("A"/"B", gidx)` → prefill `"; secret word:"` next-token rank/logp — readers R3 / nameability |
| `hfinal` | dict | final-layer states per read (auxiliary) |
| `inject_vectors` | dict | concept → exact unit steering vector v̂ (float32) |
| `inject_alpha` | dict | `"concept\|s{strength}"` → exact alpha (resid += alpha·v̂ reproduces strength·v̂) |

Each `streams` entry: `gidx, concept, concept_idx, strength, tokens, text, deg, accepted, gen_topk`
(`accepted` = passed the word-free / non-degeneracy filter; `gen_topk` = per-step top-K generation logprobs
`{ids, logp}`). The exact injected vectors + alphas are persisted so projection / re-evocation analyses run
fully offline, no model reload.

## Reproduce the analysis (CPU, offline)

```
# symbol-counter / internal-probe readers (R1/R2) for one bundle in place:
INTRO_MODEL=qwen2.5-1.5b .venv/bin/python analysis/analyze_v2.py

# the A/B comparisons, straight from this dir:
.venv/bin/python analysis/char_tilt.py runs/_ab/qwen2.5-1.5b-all.pt runs/_ab/qwen2.5-1.5b-gen.pt
.venv/bin/python analysis/nameability_ab.py
```

## exp3 — prompt-induction results (`exp3/`)

The steering vector of exp1/exp2 is replaced by a system-prompt induction, in three arms: **evoked**
(a persona describes the state, concept word absent), **named** (same persona plus the word), and
**secret_word** ("The secret word is X."), plus an **evoked_alt** paraphrase set at 1.5B. A blind-judge
**gauge** checks per scale that the persona took.

| file | what |
|------|------|
| `exp3/experiment.md` | the experiment report |
| `exp3/induction_results.json` | per-arm, per-model reader curves, gaps and frozen-threshold verdicts (3 seeds) |
| `exp3/induction_curves.png` | dist / R_emb / one-hot curves, all arms × models |
| `exp3/gauge_results.json` | blind-judge per-concept accuracy, Wilson CIs and per-item replies |
| `exp3/bundles/*.pt` | the induced streams (13 bundles: evoked / named / secret_word / secret_sustain × 1.5B/3B/7B, plus evoked_alt at 1.5B) |

Result, as the write-up now states it: the distribution reader recovers 0.447 / 0.176 / 0.019 bits at
1.5B / 3B / 7B on the evoked arm, but that signal is tied to the wording of the inducing text. A decoder
trained on one wording reads about 0 calibrated bits on a paraphrase of the same concept. Transcript
readers read about 0 in every arm at every scale.

The `secret_sustain` bundles belong to the later reader × generator grid ("imbue the secret into
every keystroke"), not to the three arms above.

**Correction.** An earlier version of this card, and the original text of `exp3/experiment.md`, said
"what leaks is state, not symbols" and that the result "is not an artifact of one wording". The paraphrase-transfer test
(`transfer_decode.json` in the code repo) does not support either statement.

## exp2 — token-budget results (`exp2/`)

Run offline on the generation-only captures (for 7B, the original under-dosed capture,
`qwen2.5-7b-gen-s93.pt`).

| file | what |
|------|------|
| `exp2/experiment.md` | the experiment report |
| `exp2/budget_results.json` | per-model, per-reader bits-recovered curves up to 12 tokens, and the two gaps (3 seeds) |
| `exp2/budget_curves.png` | the reader learning curves |

Result, as the write-up now states it: at 12 tokens the distribution reader is far ahead of the token
readers (2.64 vs 0.96 bits at 1.5B), but given the full stream a character reader with no model access
catches up and passes it. Distribution access buys speed, not exclusive information.

**Corrections.** An earlier version of this card said the token channel sits at the one-hot floor and
that the distribution channel shrinks with scale (2.64 / 2.24 / 0.89 bits). The first was an artifact
of stopping at 12 tokens. The second was a dose artifact: the 7B run behind 0.89 was under-dosed, and
at the criterion-passing dose 7B reads 2.51. Note that `exp2/budget_results.json` still holds the
under-dosed 7B row (0.89), computed from `qwen2.5-7b-gen-s93.pt`.

## License

Code: MIT. Data (these `.pt` bundles): CC-BY-4.0.

Provenance: the bundles are outputs of Qwen2.5 / Falcon-3 / Llama-3.3 models, whose licenses
ride along (Qwen2.5-1.5B/7B/14B Apache-2.0; Qwen2.5-3B-Instruct under the Qwen Research License;
Falcon-3 under the TII Falcon License; Llama-3.3 under the Llama 3.3 Community License).

## Data completeness notes

- `qwen2.5-7b-gen.pt` is byte-identical to `qwen2.5-7b-gen-matched.pt` (the criterion-dose 7B
  re-run). The original under-dosed 7B capture, the one behind the dose-artifact discussion in the
  write-up, is `qwen2.5-7b-gen-s93.pt`.
- The raw score shards behind the reader × generator grid, its 14B extension, the re-injection dose
  curve and the same-model readers are all here (added 2026-11-09), so those results can be
  recomputed from this dataset with the scorers in the code repo. Each scorer reads the repo path
  shown in the code repo's README data map.
- Not released: regenerable intermediates of the confound suite, smoke-test runs, and exploratory
  Qwen3 captures that no reported number uses.

## Upload / download (Hugging Face dataset)

```
# upload (maintainer):
huggingface-cli upload ErrareHumanumEst/internal-state-from-gibberish runs/_ab . --repo-type dataset

# download (reader):
huggingface-cli download ErrareHumanumEst/internal-state-from-gibberish --repo-type dataset --local-dir runs/_ab
```
