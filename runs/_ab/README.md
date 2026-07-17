---
license: cc-by-4.0
pretty_name: Introspection-leakage injection-method A/B
tags:
  - interpretability
  - mechanistic-interpretability
  - ai-safety
  - introspection
  - qwen2.5
size_categories:
  - 1K<n<10K
---

# Injection-method A/B data (introspection-leakage robustness check)

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
| `qwen2.5-7b-gen.pt`  | Qwen2.5-7B  | generation-only | 0, 62, 93 |
| `qwen2.5-7b-gen-matched.pt` | Qwen2.5-7B | generation-only, nameability-matched | 0, 124, 140 |

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

## License

Code: MIT. Data (these `.pt` bundles): CC-BY-4.0.

Provenance: the bundles are outputs of Qwen2.5 / Falcon-3 / Llama-3.3 models, whose licenses
ride along (Qwen2.5-1.5B/7B/14B Apache-2.0; Qwen2.5-3B-Instruct under the Qwen Research License;
Falcon-3 under the TII Falcon License; Llama-3.3 under the Llama 3.3 Community License).

## Data completeness notes

- `qwen2.5-7b-gen.pt` here is byte-identical to `qwen2.5-7b-gen-matched.pt` (the criterion-dose
  7B re-run). The superseded original 7B `-gen` capture — the one behind the dose-artifact
  discussion in the writeup — is not uploaded; it is available on request.
- The raw per-cell grid capture shards (Falcon-3 cross-family cells, prose controls, and `_raw`
  variants — 230 of 278) are not released here. The derived per-cell score JSONs
  (`lr_grid_results.json` and the committed `reports/*.json`) ARE in the code repo, so every
  reported number reproduces from those.

## Upload / download (Hugging Face dataset)

```
# upload (maintainer):
huggingface-cli upload ErrareHumanumEst/internal-state-from-gibberish runs/_ab . --repo-type dataset

# download (reader):
huggingface-cli download ErrareHumanumEst/internal-state-from-gibberish --repo-type dataset --local-dir runs/_ab
```
