# Scale-14B verdict: PLATEAU — the injected channel does not close at 14B

Scored against [`scale14b_prereg.md`](scale14b_prereg.md) (frozen body + Amendment 1 + errata, all
committed before any 14B data). Collection: Qwen2.5-14B-Instruct, `covert_collect.py --inject gen`,
explicit `--effmags 101,152,190` + `--no-calibrate` (rank-criterion mini-sweep selection per Amendment
§1), word-free filter, 3,240 streams. Pipeline: smoke ($0.075, resid_norm 283.26) → rank mini-sweep
($0.047) → full collect ($0.53, util 84.5%). Total GPU spend **$0.65**. Analysis:
`dose_titration.py --slug qwen2.5-14b` → `dose_titration_qwen2.5-14b.json`; nulls →
`perm_null_check_qwen2.5-14b_s0.json`, `..._s101.json`, `..._s101_char100000.json`.

## Gates (all pass)

1. **Injection took / capability (full-run data, relative floor 0.70 × own s0 = 0.548):**
   s101 clean 426/640 = 0.666 (**0.85× — PASS**); s152 0.480 (0.61× — FAIL); s190 0.293 (0.37× — FAIL).
   Sweep-time estimates ran high exactly as the errata warned; the pre-specified fallback applies:
   **primary cell = s101** (sweep arm-A rank 8 ≤ 50). The failed-gate cells are reported below but carry
   no verdict weight.
2. **Control (amended gate):** s0 dist@12 = −0.094 ± 0.052 (≤ +0.1 one-sided ✓); permutation null on the
   s0 pool brackets the observed seed-0 value (−0.140 inside [−0.141, −0.000] ✓ — the known benign
   negative-miscalibration mode, milder than 7B's).
3. **Positive cells vs own-pool nulls:** dist@12 s101 observed +2.060 vs null p95 −0.003 ✓;
   char@full s101 observed +1.359 vs null p95 +0.005 ✓. Both exceeded by orders of magnitude.
4. **Common-N:** n = 24/class at every verdict cell (min class counts ≥ 30). The failed-gate s190
   cell only reaches 14/class (min class count 14) and is reported at that max common-N per gate 3.

## Verdict cells (mean ± sd over seeds 0/1/2) and bands

| | 14B s101 | 7B s124 baseline | band result |
|---|---|---|---|
| **P1 char@full** | **1.237 ± 0.167** | 1.896 ± 0.127 | **PLATEAU** — inside [0.95, 2.84] |
| **P2 dist@T12** | **1.775 ± 0.264** | 2.513 ± 0.037 | **PLATEAU** — 0.71× baseline, inside [0.5×, 1.5×] |

P2's decline (−0.74) exceeds 2σ (≈0.27 combined) — a *significant modest decline*, far from the < 0.5×
CLOSING bar. P1's decline (−0.66, combined σ ≈ 0.21) is likewise significant-but-plateau. Headline
(P1, per prereg): **PLATEAU. A no-access character reader recovers ~1.2 bits of a steered concept from
raw transcript characters at 14B. Activation steering remains detectable from output text alone at
every scale tested (1.5B → 14B).**

The criterion-passing scale curve, now four points:

| | 1.5B (s60) | 3B (s60) | 7B (s124) | 14B (s101) |
|---|---|---|---|---|
| dist@T12 | 2.64 | 2.24 | 2.51 | 1.78 |
| char@full | 2.06 | 0.72 | 1.90 | 1.24 |

Non-monotone, gently declining at the endpoints, nowhere near the natural regime's floor (≤ 0.02 at
every scale). The regime split survives a fourth scale point.

## Named calls, scored as originally written

1. "P1 lands PLATEAU — between 0.4 and 1.2 bits": **wrong as written** — the verdict *is* PLATEAU but
   P1 = 1.237 falls 0.037 above the stated band (within one seed-sd of the boundary; scored wrong
   regardless). The mechanism half was right, the numeric band was not.
2. "P2 continues down but stays > 0.3": **right** (2.51 → 1.78).
3. "Medium dose shows dist saturated near the strong dose, char well below": **wrong** — at 14B the
   dist channel is *not* dose-saturated (1.78 → 2.76 → 3.14 across s101/s152/s190); the 1.5B/7B
   saturation pattern does not recur. Scored 1/3. (The confusion-geometry prereg precedent applies:
   named calls are bets, reported as they land.)

## Caveats and non-verdict observations

- **Comparability:** the 14B primary cell sits at sweep rank ~8 — a *stronger* nameability
  instantiation than 7B's rank-47 baseline (closer to 1.5B's rank 4). The mismatch direction here cuts
  *toward* robustness of the plateau: measured at a stronger instantiation, 14B still leaks somewhat
  less than 7B — the plateau is not an artifact of over-dosing the new point. Rank equating across
  scales remains unsolved (§6 language: criterion-passing, not matched).
- **Failed-gate cells (reported, no verdict weight):** s152 char@full = 2.67, dist = 2.76; s190
  char@full = 2.84, dist = 3.14 (near the 3.585-bit ceiling) — heavily steered 14B is close to
  fully output-readable, but at 0.61×/0.37× capability it is partly incapacitated. The capability
  cliff arrives at a lower relative dose than the sweep estimated.
- **Stream lengths:** unlike 7B (median 23–25 under injection), 14B sustains full-length (median 128)
  word-free streams at every dose — the char@full cells are long-read cells, directly comparable to
  1.5B's, so the plateau is not a stream-length artifact.
- One dose grid, one family, one seed-set; all readers remain lower bounds.
