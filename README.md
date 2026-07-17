# Introspection-leakage lab

Can a monitor read a concept a model is "carrying" from its **word-free** output — and does that channel
grow or shrink with scale? A small lab on Qwen2.5 (1.5B / 3B / 7B, plus a pre-registered 14B scale point
for the injected channel), with a shared measurement core and one experiment per question. **Start with
[`BLOGPOST.md`](BLOGPOST.md)** for the combined arc.

## Experiments

| Experiment | Question | Status |
|---|---|---|
| [`exp1_epistemic_privilege/`](experiments/exp1_epistemic_privilege/) | Does word-free output leak an injected concept, and which observer recovers it? | **complete** — a char symbol-counter recovers it at every scale (0.82 / 0.77 / 0.86 bal-acc at criterion-matched dose; the published 0.83 → 0.73 "weakening" was a 7B dose artifact); robust to the injection method |
| [`exp2_output_monitorability/`](experiments/exp2_output_monitorability/) | What's the price (bits recovered vs output tokens), and is the token channel really that weak? | **complete** — distribution access buys **speed, not exclusivity**: it reads the injected concept in ~12 tokens, but given the full transcript even a char-histogram reader catches up (`R_emb ≈ floor` was a 12-token artifact). A pre-registered 14B run lands **PLATEAU**: the injected channel does not close with scale (char@full 2.06 / 0.72 / 1.90 / 1.24 across 1.5B→14B). A second pre-registered suite settles the mechanism: transcript marking needs **dose × persistence jointly**, and prompted concepts install ~no state along the concept direction (0.05σ during the task, 0.4σ free, vs injection's 12σ) — see `reports/scale14b_{prereg,verdict}.md`, `reports/confound_closing_{prereg,verdicts}.md` |
| [`exp3_induction_and_scale/`](experiments/exp3_induction_and_scale/) | Remove the steering vector — is the leak method-independent, and what does scale do? | **complete** — the prompt-persona signal is **wording-tied** (a paraphrase-transfer test reads ~0 calibrated bits across wordings; the "state, not symbols" reading is downgraded to open), transcripts carry nothing at any length/scale, and the readable context residue closes by 7B while a blind-judge gauge shows the persona still takes |

Each experiment folder owns its `README.md`, `analysis/`, `reports/`, and `tests/`. They all import the
shared core below; none of them fork it (extend it via flags — e.g. `--inject {all,gen}`).

## Shared core (used by every experiment)

| Path | What's in it |
|------|--------------|
| `src/` | GPU-side measurement: `covert_collect.py` (inject → generate word-free streams → read each arm), `common.py` (model load, injection hook, capture), `config.py` (model registry + knobs), `baseline_clean.py`, `derive_vectors.py`. |
| `harness/` | `run_labkit.py` — the GPU driver (provisions a remote box, runs the collect via the pinned **`labkit`** package). `LABKIT.md` documents it; `labkit-feature-request.md` collects harness spend-blockers. |
| `runs/<slug>/` | Per-model bundles (the shared data lake): `results/*.json` (committed), `figures/`, `streams/`, and `data/covert_collect.pt` (raw captures — **not** in git; see *Data*). `runs/_ab/` holds the injection-method A/B release + its dataset card. |

Model slugs: `qwen2.5-1.5b`, `qwen2.5-3b`, `qwen2.5-7b` (the committed scale series), plus exploratory
`qwen3-*`. The 12 concepts: silence, ocean, fear, celebration, deception, obedience, debugging, security,
curiosity, anger, warmth, loneliness.

## Environments

Two separate venvs, by design — the analysis box never needs the GPU stack, and the driver never needs torch:

```bash
# Analysis (CPU): everything under experiments/*/analysis and tests
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# GPU driver (only to launch a real collect; requires the private labkit package)
python3 -m venv .venv-driver && .venv-driver/bin/pip install -r requirements-driver.txt
```

GPU spend is gated through **experimentfactory** (`authorized_run` above `labkit`) — see each experiment's
driver. The `Makefile` has the common tasks (`make test`, `make analyze MODEL=…`, `make collect …`).

## Data

The stream bundles and per-experiment result artifacts are released on Hugging Face
([`ErrareHumanumEst/internal-state-from-gibberish`](https://huggingface.co/datasets/ErrareHumanumEst/internal-state-from-gibberish),
CC-BY-4.0), organized by experiment under `exp2/` and `exp3/` (the reader curves, the induction results, the
blind-judge gauge JSON, and the stream bundles):

```python
from huggingface_hub import hf_hub_download
hf_hub_download("ErrareHumanumEst/internal-state-from-gibberish", "exp3/induction_results.json",
                repo_type="dataset")
```

They contain only the model's own (word-free) output streams and captured distributions/activations — no
credentials or personal data. See each experiment's `README.md` for the bundle + result-JSON schema.

### Data layout: repo path → HF path

The analyses read data at the repo-relative paths below; the dataset holds the actual bytes. Early
uploads used short names; everything uploaded on 2026-07-17 mirrors its repo-relative path exactly
(md5s for those items are in `runs/hf_upload_manifest.txt`).

| artifact (repo path the code reads) | HF path |
|---|---|
| injection A/B captures `runs/_ab/qwen2.5-{1.5b,3b,7b}-{all,gen}.pt` | `qwen2.5-<size>-{all,gen}.pt` (repo root); `qwen2.5-7b-gen-matched.pt` is the criterion-dose 7B re-run |
| per-model captures `runs/qwen2.5-{1.5b,3b,7b}/data/covert_collect.pt` | identical bytes to the root `-gen`/`-all` captures (md5-verified; manifest note) |
| 14B scale-run captures `runs/qwen2.5-14b{,-sweep,-smoke1}/data/covert_collect.pt` | same paths |
| confound E1 weak-dose capture `runs/confound-e1/data/covert_collect.pt` | `confound-e1-gen.pt` |
| confound E3 prompt-only capture `runs/confound-e3/data/covert_collect.pt` | `runs/confound_box/e3_prompt/data/covert_collect.pt` |
| E2 sustained bundles + pilot `runs/confound_box/e2_{full,pilot}/data/*.pt` | same paths |
| E4 + gauge trajectory shards `runs/confound_box/e4_traj/trajectory/*.pt`, `runs/gauge_box/gauge_traj/trajectory/*.pt` | same paths |
| E5 maintained-secret bundle `runs/confound_box/e5_secret/data/qwen2.5-1.5b-maintained_secret.pt` | `confound/bundles/qwen2.5-1.5b-maintained_secret.pt` |
| exp3 induction bundles `runs/_ind/<slug>/data/<slug>-<arm>.pt` | `exp3/bundles/<slug>-<arm>.pt` |
| exp2/exp3 generated results (curves, gauge, budget) | `exp2/*`, `exp3/*` |
| LR-grid/extension score shards + 14B generation pools | `lr_extend_resume/**` |
| **70B chain** — generated streams `runs/llama70b_scout/streams_llama70b.json`; full-sweep raw batch output `runs/llama70b_scout/lr_raw_batch_output.jsonl`; scored records `runs/llama70b_scout/lr_records_llama70b.json`; faithful-template rescore `runs/rescore_llama70b/{rescore_lr_records.json, rescore_raw_file-*.jsonl, rescore_meta.json, rescore_validation.json}` | same paths (the small 70B records/streams are also committed in-repo; the official-adapter confirmation is `experiments/exp2_output_monitorability/reports/lr_72b_rescore_confirmation.json`, in git) |

## Acknowledgements

Builds directly on the **open-introspection** introspection-at-scale work by
[@ostegm](https://github.com/ostegm) — see [blog post 04](https://ostegm.github.io/open-introspection/blog/posts/04-introspection-at-scale.html)
and the [repo](https://github.com/ostegm/open-introspection). The 12-concept set and the difference-vector
extraction method come from there.

## License

- **Code** — MIT, see [`LICENSE`](LICENSE). **Data** (stream bundles, `runs/` results, `reports/*.json`) — CC-BY-4.0, see [`LICENSE-DATA`](LICENSE-DATA).
- The bundles are outputs of Qwen2.5 / Falcon-3 / Llama-3.3 models, whose licenses ride along
  (Qwen2.5-1.5B/7B/14B Apache-2.0; Qwen2.5-3B-Instruct under the Qwen Research License; Falcon-3
  under the TII Falcon License; Llama-3.3 under the Llama 3.3 Community License). The dataset
  cards record provenance.
