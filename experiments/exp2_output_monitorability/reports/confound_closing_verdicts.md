# Confound-closing run — verdicts (scored against confound_closing_prereg.md)

Run: one RTX3090 box, all stages S0–S5 contiguous, no intervention, **$0.30 total** (+$0.035 lemon
churn on attempt 1). Pilot winner: `s1` (acceptance 0.81, retention gate passed). Evidence:
`dose_titration_confound-e{1,3}.json`, `e4_trajectory_verdict.json`, `e2_e5_verdict.json`,
`runs/confound_box/` (pilot_verdict.json, bundles, trajectory shards, S5 budget_results).

| exp | verdict | named call |
|---|---|---|
| E1 weak-dose | **FLOORS** — dist@12 rises 0.00/−0.01/0.04/0.11/0.66 (s3→s20), bracketing the natural window at s12–s20; char@full ≤ 0.039 at every weak dose (< 0.05 band) | **right** |
| E2 sustained persona | **~floor, BELOW evoked** — within 0.051±0.022 / alt 0.042±0.015 (vs evoked 0.447); char@full +0.002±0.002; acceptance 0.86; transfer −0.09 | **wrong** (informative) |
| E3 prompt-only | **persistence-driver CONFIRMED** — dist@12 0.23/0.74 (vs gen-only 2.60/2.64), char@full 0.011/0.030 (vs 0.56/2.06) at s40/s60 | **right** |
| E4 trajectory | **state ABSENT from t=2** — evoked z ≈ 0.05σ flat (1.5B), 0.08σ (7B); injected 12σ flat (retention 0.98). F3: **not sustained** at 7B — the anti-word regime displaces the persona (or personas never write this basis) | **wrong** (the un-named third option; dilution presupposed an installed state) |
| E5 maintained secret | **absent** — 0.010±0.003 (original secret_word 0.027) | **right** |

Controls: every s0 cell ≤ +0.04 one-sided ✓. Sustained per-cut z (pilot, s1): 0.15/0.07/0.24/0.31/
0.19/0.37/0.09 across t=2..127 — a real but flickering ~0.2σ trace, no trend; ~50× below injection.

## Synthesis

**Transcript marking requires dose × persistence jointly**: weak-per-token floors (E1), strong-static
floors (E3); only strong-and-per-token marks. **Context induction installs ~no state along the injected
concept direction under a competing instruction** (E4) — even when explicitly told to imbue every token
(pilot: ~0.2σ vs injection's 12σ). The natural regime's readable 0.45 bits is **wording residue**, now
confirmed three independent ways: paraphrase-transfer (calibrated bits don't cross wordings), E4 (no
concept-state present to read), and E2 (a shared suffix across personas *suppresses* the signal —
diluting wording distinctiveness reduces the "leak"). E5: storage without use doesn't leak.

Writeup framing (Matt's, upgraded by the data): not a *fleeting* concept but a **displaced** one — and
"pure output may suffice to read internal state, but only state that is actually present and strongly,
persistently written; a contextual concept under a competing instruction is neither." Caveat to carry:
all state measurements are in the injected-vector basis; a persona could write a different direction
(v_read≠v_write) — the standing basis-dependence limit.

## LR-reader addendum (2026-07-09, scored against lr_reader_prereg.md)

Same question, different instrument: instead of a *trained reader*, teacher-forced likelihood under
the generating model itself — is each saved word-free stream more likely under the persona context
that (nominally) produced it? One RTX3090 box at 1.5B, all 25 reconstructed collection contexts ×
all accepted streams (injected s60 n=435, evoked n=707+58 neutral, evoked_alt n=699+62 neutral,
per-concept 31–63). Registered KV self-check tripped (0.037 > 0.02 nats/tok) → the run used the
full-concat reference path throughout, as pre-registered. Evidence: `lr_reader_results.json`,
`runs/lr_box/`. Spend: $0.20 total on the project ledger ($0.12 successful run incl. lemon churn; $0.08 across four
false starts — two vast ssh/rsync flakes, one cache-format crash, one done-marker substring collision,
each fixed with a RED-first regression test).

| cell | bits (mean±sd, 10 seeds) | top-1 (chance 8.3%) | role |
|---|---|---|---|
| evoked × A | **0.590 ± 0.031** | 29.6% | within-wording ceiling |
| evoked × B | 0.176 ± 0.016 | 16.4% | paraphrase transfer |
| alt × A | 0.167 ± 0.021 | 18.7% | paraphrase transfer |
| alt × B | 0.565 ± 0.022 | 27.5% | alt within-wording ceiling |
| injected × A | **0.002 ± 0.002** | 10.5% | Matt's question |
| injected × B | −0.001 ± 0.002 | 9.2% | Matt's question (paraphrase) |

Gates: 9/10 pass; one narrow fail (neutral streams under wording-B contexts: median −0.035
nats/token vs the ±0.02 bound — a mild register mismatch, *negative* sign, so it cannot manufacture
the positive effects scored below; flagged per prereg). Mismatched-concept scores centered ~0 in all
six cells ✓. Prereg caveat-note disclosures (2026-07-09 amendment): scores are full-softmax LLs of
top-p-0.98-sampled tokens (warp identical across the 25 contexts per stream); eos-termination rates
differ by set — injected 43.2%, evoked 66.1%, evoked_alt 56.5% — so small **cross-set** bit
differences sit inside the eos-in-LL artifact's range. Neither caveat can produce the injected-cell
floor (a ~0.6-bit within-wording effect vs a ~0.00-bit injected effect is far outside both).

**Named calls (registered verbatim before data):**

| call | verdict |
|---|---|
| MATT — "even the injected streams are more likely under the relevant natural and paraphrase personas" (injected×A and ×B: bits > 0.1, top-1 > 8.3%) | **wrong** — injected cells are at floor (0.002 / −0.001 bits); matched per-token scores are actually slightly *negative* (−0.014 nats/tok: personas make injected gibberish marginally *less* likely than neutral) |
| ASSISTANT — positive but small on injected×matching (bits ∈ (0.05, 0.5), well below evoked×A); evoked×B wording-tied-low (< 0.15 bits) | **wrong** on both halves — (a) injected is not small-positive but zero; (b) evoked×B = 0.176 ≥ 0.15: modest *real* paraphrase transfer exists at the likelihood level (~30% of the 0.59 ceiling, symmetric with alt×A 0.167) |

Reading: the likelihood instrument sharpens the story on both ends. The injected channel's readable
2.6 bits (trained dist-reader) is a distributional mark that does **not** look like "text generated
under the matching persona" — the generating model's own likelihood cannot see it at all. And the
natural regime's signal is mostly wording residue (0.59 within- vs 0.17 cross-wording), but the
cross-wording 0.17 bits is small-real here, slightly richer than the trained-reader transfer test
suggested — consistent with likelihoods catching shared persona *register* that a reader trained on
one wording does not transfer.

---

## Correction (2026-07-17, append-only; Matt-decided, self-adverse) — E3 re-scored: band-unmet, directionally confirmed

**Nothing above is edited.** Re-reading the frozen E3 band (`confound_closing_prereg.md`) against
`dose_titration_confound-e3.json` shows the table's "persistence-driver CONFIRMED" was mis-scored.
The frozen band is a conjunction: *char@full(prompt-only) < 0.5 × gen-only at both doses* **while**
*dist@T12(prompt-only) ≥ 0.5 × gen-only's*. The char half is met decisively (0.011/0.030 vs
0.56/2.06 → 0.020×/0.015×). The dist half is **not**: 0.229/0.737 vs 2.60/2.64 → **0.088× / 0.28×**,
both below the 0.5× floor. Prompt-only injection didn't just lose the transcript mark — it lost most
of the distribution signal too, which the frozen wording does not permit for a CONFIRMED verdict:
the conjunct existed precisely so the char collapse couldn't be attributed to the whole signal
dying. (The refutation band — char@full ≈ gen-only, ≥ 0.75× — is equally unmet.) The correct
as-frozen verdict is **band-unmet, directionally confirmed**: the char collapse is real and points
the persistence way, but this cell alone cannot certify it, because the effective dose reaching T12
wasn't matched.

Accounting that follows (per "Named calls are bets, scored as written"):

- The E3 table row's verdict "persistence-driver CONFIRMED — right" is corrected to **"band-unmet,
  directionally confirmed — out-of-band"** (the prediction's content largely happened, but the band
  it was to be scored against was not met, so it cannot score as *right*).
- Named-call accounting: **2 of 5 right as-frozen (E1, E5)**; E3 directionally consistent but
  out-of-band; E2 and E4 wrong-but-informative, unchanged.
- The synthesis's "strong-static floors (E3)" stands as a directional observation; the *certified*
  persistence story now rests on E1 (weak per-token floors while dist rises into the natural window)
  plus the E4 trajectory contrast (per-token injection holds ~12σ flat across the stream while the
  static-prefix persona's state is absent from token 2), with E3's char collapse as directional
  corroboration.

---

## Permutation-null addendum (2026-07-17, append-only) — the missing E1/E3 positive-cell nulls, now run

**Nothing above is edited.** The prereg's shared validity gates require every positive verdict cell
to exceed its own-pool ≥ 20-shuffle label-permutation null p95. The titration run reported the s0
controls but the shuffled-label nulls for the positive dist@T12 cells were never actually computed —
a gap flagged in review. Closed post-data with the identical reader path
(`analysis/perm_null_confounds.py`, the `perm_null_check.cell_bits` protocol: common-N 24/class,
seed 0 observed and 20 shuffles, dist@T12), artifact `reports/perm_null_confound_e1_e3.json`:

| cell | observed (true labels, seed 0) | null mean ± sd | null p95 | exceeds p95 |
|---|---|---|---|---|
| E1 s12 | +0.139 | −0.014 ± 0.017 | +0.021 | **yes** |
| E1 s20 | +0.694 | −0.010 ± 0.015 | +0.016 | **yes** |
| E3 s40 | +0.200 | −0.013 ± 0.009 | +0.004 | **yes** |
| E3 s60 | +0.739 | −0.030 ± 0.019 | −0.006 | **yes** |

All four positive cells clear their own-pool null p95 decisively (the nearest margin, E1 s12, is
+0.118 above its p95). The observed values are the seed-0 single-seed protocol (per
`perm_null_check.py`), so they differ slightly from the titration tables' 3-seed means
(e.g. E1 s12: 0.139 seed-0 vs 0.113 ± 0.017 3-seed); same reader, same pools. The E1/E3 dist
positives are therefore label-linked signal, not decoder miscalibration; no verdict changes.
