# exp3 — Method-independence and scale: the readable signal is wording-tied (the "state, not symbols" reading was overturned)

The follow-up to [exp2](../exp2_output_monitorability/). exp1/exp2 established, with an **injected** steering
vector, that a distribution-access monitor recovers concept identity from word-free output far faster than
the best honest token monitor (at full stream the transcript catches up — exp2's correction) — but everything
rode on activation steering, and the scale trend was confounded
by injection dose. **exp3 removes the vector.** The concept now enters through the **system prompt**, and a
pre-registered blind-judge gauge checks per scale that the persona actually took.

Result, as corrected by the paraphrase-transfer re-analysis (the update blocks at the top of
[`reports/experiment.md`](reports/experiment.md)): the phenomenon reproduces without the vector, but the
monitor-readable signal is **wording-tied**, not a portable concept signal — a decoder trained on one wording
reads ≈ 0 calibrated bits on independent paraphrases of the same 12 concepts, with only a **faint
concept-aligned trace** surviving transfer (top-1 12–16% vs 8.3% chance). The original "what leaks is state,
not symbols" conclusion is **overturned/downgraded to open**: a wording-residue account predicts the whole arm
table (`named ≈ evoked` and the secret_word null included), and the E4 trajectory measurement later found ~no
state installed along the concept direction during the task. What stands: transcripts carry nothing at any
length/scale, the bare "secret word" leaks nothing to any external reader, and the readable context residue
closes by 7B even as the gauge shows the persona still takes. Full result, confounders, and the
pre-registered verdicts + corrections: [`reports/experiment.md`](reports/experiment.md).

## The three arms (+ neutral baseline), frozen in [`primers.py`](primers.py)

- **evoked** — a persona describing the concept as an internal *state*, concept word absent (the clean-prompt
  analog of injection).
- **named** — the same persona with the concept word inserted once (`named − evoked` isolates the label).
- **secret_word** — literally "The secret word is X." and nothing else (one token, no state).
- **evoked_alt** — frozen paraphrases of `evoked` (1.5B) for the invariance check.

## Layout

| Path | What |
|------|------|
| `primers.py` | the frozen 3-arm primers + the pre-registered thresholds (`PREREG`) — frozen before the readers ran. |
| `collect_induction.py` | GPU collector: generate the induced word-free streams per arm (reuses `covert_collect` generation). |
| `run_induction.py` | analysis: exp2's frozen readers per arm + the per-seed gap, bootstrap CI, full-stream, `evoked` bits-ladder, and the frozen-threshold verdicts. |
| `gauge_judge.py` | the manipulation-check gauge — a pinned blind judge (`claude-haiku-4-5-20251001`) reads the persona's free-association and picks the concept. |
| `box_analyze.py` | on-box entrypoint (extract embeds from HF, run exp3 + exp2 analysis into `out/`). |
| `tests/` | offline unit tests (RED-first), CPU only. |
| `reports/` | `experiment.md` (tracked); generated `induction_results.json` / `gauge_results.json` / `induction_curves.png` live on HF. |

## Reproduce (offline, from the released bundles)

```bash
# reader analysis (heavy nested-CV -> rented box, not a laptop):
.venv/bin/python experiments/exp3_induction_and_scale/run_induction.py   # reads runs/_ind/<model>/data/<model>-<arm>.pt
# the blind-judge gauge (API; needs ANTHROPIC_API_KEY):
.venv/bin/python experiments/exp3_induction_and_scale/gauge_judge.py
```

Both exp3 and exp2 run together on a box via `harness/run_reanalysis.py` (gated through **experimentfactory**
`authorized_run`). The readers need `transformers` (char tokenizer) + each model's embedding matrix
(extracted from HF into `artifacts/`).

### Released-data schema
Each `runs/_ind/<model>/data/<model>-<arm>.pt` bundle uses the exp2 stream schema (see
[`../exp2_output_monitorability/README.md`](../exp2_output_monitorability/README.md)): accepted streams with
`gen_topk`, `tokens`, `concept_idx`; strengths `[0, 1]` (0 = neutral baseline, 1 = induced arm).
`induction_results.json` is a list of per-(model, arm) summaries: `readers.{dist,emb,sampled,char}.bits_mean/bits_sd`
(per budget), `gap_dist_emb` / `gap_dist_char` (mean, sd, per-seed), `bootstrap_ci`, `full_stream`, `ladder`
(evoked only), and `verdicts` against `primers.PREREG`. `gauge_results.json` holds the per-concept blind-judge
tallies, Wilson-95 CIs, and every raw judge reply.
