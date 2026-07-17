# exp1 — Epistemic privilege of single instances

Inject a concept difference-vector (e.g. *ocean*, *fear*, *deception*) into a Qwen2.5 model at an
introspection layer, have it emit **word-free gibberish**, and ask **which observer can recover the injected
concept from that output** — a literal symbol-counter, the model's own internal read, or a prefill
"name the secret word" probe. Run across 1.5B / 3B / 7B.

**Headline.** A dumb character n-gram counter recovers the injected concept from word-free output above its
shuffle floor at every scale (original series **0.83 / 0.79 / 0.73**, 1.5→7B): random-looking output *does*
leak the concept. The original "weakens with scale" reading of that series is **retracted** — the 7B run was
under-injected, and at criterion-matched dose the symbol-counter reads **0.86** at 7B (dose correction at the
top of [`reports/experiment.md`](reports/experiment.md); exp2's `scale14b_verdict.md` extends the corrected,
non-closing curve to 14B). What genuinely shrinks with scale is the raw character-histogram shift, not
recoverability. A clean copy's internal read does not beat the symbol-counter; the prefill probe carries
concept-specific signal only at 1.5B. So what an *output*-only monitor catches can diverge from what is *in
the state*. The leak is **robust to the injection method** (all-position vs generation-only; see the
robustness section).

- **The writeup:** [`reports/experiment.md`](reports/experiment.md)
- **Long-form narrative:** [`reports/REPORT.md`](reports/REPORT.md) — a 3B walkthrough (blog draft)
- **Specs / proposals:** [`reports/specs/`](reports/specs/)

## Layout

| Path | What |
|------|------|
| `analysis/` | Offline **CPU** analysis of the collected bundles. `_paths.py` (run resolver), `_features.py` (canonical R1 featurizer + OvR scorer), `analyze_v2.py` (the three readers), `char_tilt.py` + `nameability_ab.py` (the injection-method A/B), `concept_matched_control.py`, `derisk_learning_curve.py`, `verify_v1_numbers.py`, … |
| `reports/` | The writeups + `reports/specs/`. |
| `tests/` | Offline unit tests (floor tuner, sweep accounting, injection hook, driver wiring). |

The GPU collector, harness, and data lake are the repo's shared core (`../../src`, `../../harness`,
`../../runs`) — see the root README.

## Reproduce (CPU, from the repo root)

```bash
make test                                   # the offline unit tests
make analyze MODEL=qwen2.5-3b               # the full analysis suite for one model (needs its .pt)

# headline reader table — committed results JSON only, no GPU, no .pt:
.venv/bin/python experiments/exp1_epistemic_privilege/analysis/verify_v1_numbers.py

# the injection-method A/B (needs runs/_ab/*.pt):
.venv/bin/python experiments/exp1_epistemic_privilege/analysis/char_tilt.py \
    runs/_ab/qwen2.5-1.5b-all.pt runs/_ab/qwen2.5-1.5b-gen.pt
.venv/bin/python experiments/exp1_epistemic_privilege/analysis/nameability_ab.py
```

Select a model with `INTRO_MODEL=<slug>`; it routes all I/O to `runs/<slug>/`.

## Extending

- **A new reader:** add an `analysis/*.py` that does `import _paths as P` and loads
  `P.DATA / "covert_collect.pt"`; reuse `_features.char_features` / `ovr_bacc` for an apples-to-apples R1.
- **A new model:** one line in `MODELS` (`../../src/config.py`); `INTRO_MODEL` routes I/O. Collecting still
  needs a GPU + labkit.
