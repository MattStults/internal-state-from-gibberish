# LR scale-grid EXTENSION — verdicts (scored against lr_scale_extend_prereg.md, FROZEN 2026-07-13, + Amendments 1–4)

Run: one RTX 6000Ada, two box attempts (**$1.87** fatal at S4 — stale-HF-7B-capture vintage —
+ **$0.61** resumed completion via the cross-box resume built in `dd6fc93`), five smoke-class
launches **$0.85**, campaign total **$3.33** against the **$12** authorization (ledger entries
44908052→44945847; cumulative project ledger $9.21). 95 shards, `outcome=ok` (`LRX_DONE`).
Scope per the frozen Decisions block: **14B-only, 32B descoped** (the 32B named-call clauses
are VOID per the written contingency; both calls score on the 14B clauses alone — disclosed).
Evidence: `reports/lr_extend_results.json` (authoritative), `runs/lr_extend_box/`,
`runs/lr_extend_shakedowns.json`, `runs/confound-ledger.json`.

Instrument: **reproduced**. The 1.5B eos-free D2 anchor re-measured by this run = **0.166483**
vs reference 0.16648 (delta 0.0000, tol 0.01, `anchor.ok=true`; cells evokedxB 0.1783 /
evoked_altxA 0.1546). The 1.5B secret_word regression cell = **0.1628** vs the grid's 0.163.
No anchor-voided cells. Prefix-K rule re-computed over this run's 8 pools: **K = 16**
(unchanged from the grid).

## Headline — the secret_word diagonal BENDS at 14B

Secret_word diagonal, calibrated bits, all three registered currencies (primary / secondary-A
per-token / secondary-B prefix-16 — the frozen named-call currency):

| size | primary | secondary A | **secondary B (K=16)** | top-1 | n |
|---|---|---|---|---|---|
| 1.5B | 0.163 | 0.125 | 0.122 | — | 688 |
| 3B | 0.191 | 0.183 | 0.195 | — | 757 |
| 7B | 0.405 | 0.267 | 0.282 | 25.1% | 495 |
| **14B** | **0.200** | **0.144** | **0.130** | 16.6% | 481 |

(1.5B–7B rows: the grid run, `lr_grid_results.json`, quoted per the R1 addendum discipline;
14B row: this run, cell `qwen2.5-14b|qwen2.5-14b|secret_word`, primary 0.1999 ± 0.0194,
ci95 [0.135, 0.271], gates 2/3 pass, position control pass 21.2%.)

The curve rises 1.5B→7B and **turns over at 14B in every currency**: primary 0.163 → 0.191 →
0.405 → **0.200**; prefix-16 0.122 → 0.195 → 0.282 → **0.130**; per-token 0.125 → 0.183 →
0.267 → **0.144**. Per the registered trend-validity clause the bend is **sign-consistent
under BOTH length secondaries → the trend is claimable**: the within-family secret_word
diagonal does not keep growing; it bends down at 14B. The channel remains a clean positive
at 14B (0.13–0.20 bits, top-1 2× chance) — smaller than 7B, larger than 1.5B.

secret_sustain diagonal at 14B: **0.459** primary / 0.259 A / 0.205 B (top-1 22.9%, n=479,
gates 2/3 pass, position 7.1% pass) vs 7B's 1.109 / 0.688 / 0.884 — same bend, same sign
under both secondaries. The imbue-per-token amplification survives (sustain > word at 14B,
2.3× on the primary) but its magnitude bends with the rest of the channel.

## Controls and regime status on the 14B diagonals

- **Char surface control (both registered rules)**: 14B secret_word — frozen 3-seed
  −0.011 ± 0.010 pass, amended 10-seed +0.0006 ± 0.0137 **pass**; 14B secret_sustain — frozen
  +0.003 ± 0.011 pass, amended +0.0036 ± 0.0122 **pass**. **The bend is not a surface story**;
  both 14B diagonals are clean under both char rules and the position control.
- **Acceptance (gate 1, reported not gated)**: secret_word **83.5%** vs 7B ref 79.3%;
  secret_sustain **80.9%** vs 67.0% — both within-2×. No thin cells on the diagonals
  (min eval/concept 24).
- **Templating regime (Amendment 4.2, frozen 0.40/0.60)**: 14B secret_word top-3
  4-char-prefix share = **0.4158** (n=481; top prefixes 'qx z' 117, 'qxzj' 45, 'qz x' 38) —
  **gray zone (0.40–0.60) → CLEAN-WITH-DISCLOSED-CAVEAT** per the frozen default. (The
  scorer's `regime_status` string still carries a pre-freeze "PROPOSED" note; Amendment 4.2
  registered exactly these constants, so the label stands.) The measured templating curve is
  monotone in scale — 0.016 → 0.048 → 0.141 → **0.416** (→ ~0.90 at Llama-70B, where the
  faithful-template secret_word read null) — rising template collapse alongside the
  rise-then-bend of the channel. The gray-zone caveat travels with the bend claim: some of
  the 14B bend may be regime (entropy squeezed out of the pool), not channel.
- **Pre-decided branch (Decisions block)**: 14B **bends regime-clean → the trend question
  CLOSES as a bend**. The conditional-72B commitment, already SUSPENDED by Amendment 4.4,
  stays closed — the "climbs regime-clean" trigger did not fire.

## Named-call verdicts, Part A (frozen Q1 table, 2026-07-13, before any 14B data; currency = prefix-16 secret_word diagonal)

| owner | call (frozen rule, 32B clauses VOID) | verdict | evidence |
|---|---|---|---|
| **MATT** | keeps climbing: 14B > 0.282 ∧ char-passing ∧ every off-diagonal < 0.05 | **WRONG** | 14B prefix-16 = **0.130** < 0.282 — the diagonal conjunct is FALSE. The scorer emits "pending" because the privacy conjunct has gate-voided/null cells (see below), but false ∧ anything = false: the call cannot be rescued by any resolution of the pending cells. Scored WRONG with that logic disclosed. |
| **CLAUDE** | climbs sublinearly: 14B in [0.30, 0.55] ∧ off-diagonals < 0.05 ∧ char | **WRONG** | 0.130 is below the band's low side. Templating share 0.416 < the 0.60 collapse bar, so no regime-limited escape (that escape was 32B-only anyway, and is VOID). Same false-conjunct logic as MATT applies to the scorer's "pending". |
| CLAUDE side-call (descriptive) | secret_word top-3 prefix share > 20% at 14B | **RIGHT** | 41.6% |

The prereg's own contingency line — "a plateau/bend at 14B scores both wrong" — applies
verbatim. Both owners predicted continued growth; the channel bent. (Registered discipline
holds: at most one owner could have been RIGHT on the band; in the event, neither was.)

## Privacy / off-diagonal block

Old-reads-new (the confirmatory privacy direction), prefix-16 bits (primary in parens):

| cell | bits | status |
|---|---|---|
| 7B reads 14B secret_word | **−0.001** (+0.002) | scored, < 0.05 |
| 7B reads 14B secret_sustain | **−0.004** (−0.003) | scored, < 0.05 |
| 3B reads 14B secret_word / sustain | −0.001 / −0.001 (−0.002 / −0.001) | scored, < 0.05 (descriptive) |
| 1.5B reads 14B secret_sustain | −0.003 (−0.002) | scored, < 0.05 (descriptive) |
| 1.5B reads 14B secret_word | (−0.001) | VOID-gate3 (marginal: mm 0.0217 vs 0.02) |

**Every scored cell ≈ 0** — the privileged-access signature holds in the direction that was
scoreable: no smaller sibling reads the 14B model's secret.

New-reads-old (14B as READER of the existing 1.5B/3B/7B pools) — **largely instrument-VOID**:
5 of 6 cells fail gate 3 (mismatched-concept centering 0.020–0.083 nats/tok > the 0.02 bound),
two also carry gate-2 positive-sign misses — a positive instrument offset of the same class as
the 70B run's generic lift, milder. The one surviving cell, 14B reads 7B secret_word:
**+0.0004** prefix-16 (−0.0004 primary), < 0.05. Pre-void point estimates in the voided cells
were all ≈ 0 (−0.002…+0.014 primary) — consistent with privacy, but per the registered void
semantics they are **not scored**. Consequence: the **9.3× observer question (14B-reads-1.5B)
is VOID-instrument, not answered**. The largest *scored* privacy ratio this run adds is
7B-reads-14B (0.5×) plus the single 14B-reads-7B cell (2×).

## Amendment-1 rider — 70B cross-family tripwire: not tripped where scoreable

Readers qwen2.5-{1.5b,3b,7b,14b} × the 810 Llama-70B streams (confirmatory: secret_sustain +
evoked, bound < 0.05; secret_word descriptive — its 70B diagonal is null, a privacy claim
there is vacuous):

| reader | secret_sustain | evoked | secret_word (descr.) |
|---|---|---|---|
| 1.5B | −0.000 ok | −0.000 ok | VOID-gate3 |
| 3B | −0.000 ok | +0.002 ok | VOID-gate3 |
| 7B | −0.002 ok | VOID-gate3 | −0.002 ok |
| 14B | VOID-gate3 | VOID-gate3 | VOID-gate3 |

Every **scored** cell ≈ 0 (−0.002…+0.002; pre-void values in the voided cells −0.003…+0.007).
Verdict, per the Amendment-3 pre-pinned framing: **tripwire not tripped where scoreable; 3 of
8 confirmatory cells (and 3 of 4 descriptive) instrument-void (gate-3), privacy_ok=None** —
which is why the scorer's `privacy_verdict` reads PENDING. The registered caveat verbatim:
this design cannot separate "family-private" from "reader cannot engage the other family's
gibberish" (the grid's Falcon caveat 4) — a null is "tripwire not tripped," **never "privacy
holds."** The adverse surprise (a char-passing Qwen read of a Llama mark ≥ 0.05) did not occur.

## Evoked / evoked_alt at 14B (descriptive; gauge PENDING-BY-DESIGN per Amendment 4.1)

Cross-wording diagonal (D2 construction): evokedxB **0.0975** ± 0.016, evoked_altxA
**0.069** ± 0.026, mean **0.083** — above the 0.05 existence floor on the primary, so the
positive **self-certifies** under Amendment 4.1's gauge-pending rule (transfer proves the
induction took; no "unresolved (ungauged)" branch needed). Within-cell length secondaries
(registered alongside): evokedxB A 0.036 / B 0.028; evoked_altxA A 0.016 / B 0.055 — the
length-matched means (A 0.026, B 0.041) sit **below** the 0.05 floor, so the existence
positive is length-sensitive; disclosed, and no cross-scale trend is claimed (the grid verdict
"confounded by length, unresolved" stands). Reference points: 1.5B mean 0.166, 3B 0.073
(B-direction only), 7B 0.129.

## Part B — the §7a injection-LR closures (small-scale-only; registered scope)

### Run (1) — injected streams under the system-prompt persona LR (NOTE measurement #3; descriptive)

| cell | injected×A | injected×B |
|---|---|---|
| 1.5B (certified grid ref) | 0.002 | −0.001 |
| **3B smax=s60** | **0.0079** | −0.0028 |
| 7B s124 (primary, Q4 pin) | 0.0023 | 0.0018 |
| 7B smax=s140 (descriptive) | −0.0000 | −0.0003 |

All ≈ 0, top-1 ≈ chance (3B×A top-1 10.2% vs 8.3%). **Self-blindness-to-language holds with
scale**: a model carrying an injected concept reads nothing from a *language description* of
that concept, at 1.5B, 3B and 7B. The scale row is complete.

### Run (2) + Amendment-2a — self-read under ACTIVE re-injection (NOTE measurement #1; OWN quantity)

Registered framing verbatim: these bits are a **pure-concept channel** ("re-injection
self-legibility"), never comparable to or tabulated with the secret/evoked language-channel
LR. Calibrated bits, ceiling log₂12 = 3.585; public-curve reference (dose titration,
1.5B: dist@T12 / char@full) alongside for contrast only:

| dose (1.5B) | self-read (re-inject) | top-1 | public dist@T12 | public char@full |
|---|---|---|---|---|
| s3 | **0.011** | 13.5% | 0.000 | −0.013 |
| s5 | **0.090** | 13.4% | −0.013 | 0.009 |
| s8 | **0.234** | 21.7% | 0.036 | −0.006 |
| s12 | **0.628** | 32.4% | 0.113 | −0.009 |
| s20 | **1.361** | 49.7% | 0.655 | 0.010 |
| s40 | **2.902** | 85.0% | 2.600 | 0.556 |
| s60 | **3.426** | 96.3% | 2.640 | 2.056 |
| 3B s60 | **3.489** | 98.0% | — | — |
| 7B s124 | **3.068** | 99.8% | — | — |
| 7B s140 (descr.) | **2.810** | 99.8% | — | — |

The injected mark is **near-ceiling legible to the model's own re-injected forward pass**
(3.43 / 3.49 / 3.07 of 3.585 bits at the strong doses, top-1 96–99.8%), and the self-read
curve **exceeds the public read at every dose measured** — decisively so at s ≤ s20, where
the public transcript/distribution reads are ≈ 0–0.7 bits while self-read is already
0.01–1.36. At natural-ish strengths the injected state is *private-ish*: the model can
re-recognize its own marked stream while external readers get little. (s0 centering
reported: −0.43 nats/tok at 1.5B s60 — the certified 12-way calibration is differential and
cancels the common mode.)

### Named-call verdicts, Amendment 2 (frozen pre-data; 2b clauses VOID per Amendment 3)

| owner | call | verdict | evidence |
|---|---|---|---|
| **CLAUDE** (2a clause) | self-read flat < 0.05 at every dose s3–s60 ("self-blindness is about access, not signal strength") | **WRONG** | fails from s5 up (0.090, then monotone to 3.43). The scorer agrees: `FAILS on dose_robust_self_blindness_2a`. |
| **MATT** (Amendment 4.3) | < 0.05 through s20, first crosses at s40, s40 < 0.15 ("just barely") | **WRONG** | crossed at s5 (0.090 ≥ 0.05) and s40 = 2.90 ≫ 0.15. Scored manually here: the scorer's Amendment-2 MATT slot predates Amendment 4.3 and still reads PENDING — disclosed. |

Both owners bet on self-blindness surviving re-injection in some form; instead **run (2) is
the strongest self-read in the project** (within its own channel). Combined with run (1):
injection is public-but-self-blind-to-*language*, yet near-ceiling self-legible to the
re-injected forward pass — the missing cell is filled, and it is full.

### 2b — instruction-amplification cell: WITHDRAWN

Withdrawn pre-data by Amendment 3 (feature creep; post-mortem recorded there). Reported as
such: `expressed_2b_public` / `expressed_2b_self_read` = None by design; CLAUDE's 2b clauses
VOIDED-unscored. The follow-up queue (onset/offset persistence first) is registered in
Amendment 3.

## Ops record

- **Reviews**: two pre-launch reviews (TECH + SCI), consolidated verdict **LAUNCH-WITH-FIXES**,
  **17 fixes** applied and RED-first tested (`f345c05`; `tests/test_lr_extend_fixes.py`).
- **Incident 1 — premature-done marker**: smoke wrote LRX_DONE from a self-reinvocation child;
  fixed with the marker guard (`b1e5658`) before any full launch.
- **Incident 2 — stale capture vintage**: full attempt 1 died FATAL at S4 ($1.87) on a
  stale-HF-7B-capture vintage; fixed with an S0 capture-level assert + cross-box resume
  (`dd6fc93`); attempt 2 resumed and completed for $0.61.
- **Spend**: five smoke-class launches $0.85 + $1.87 + $0.61 = **$3.33 total vs the $12 cap**
  (28%). Shakedown registry: `runs/lr_extend_shakedowns.json` (data revisions f345c05,
  dd6fc93).

## Caveats carried

1. **Gray-zone templating (41.6%)** — the bend is regime-clean by the frozen rule but sits in
   the disclosed-caveat band; rising template collapse is a live alternative contributor to
   the bend's size (not its sign, which both length secondaries confirm).
2. **The 14B-as-reader off-diagonal block is instrument-void** (gate-3 offset) — the 9.3×
   observer question is unanswered, not answered-null. The cheap serverless 72B/large-reader
   observer probe remains the open item (Amendment 4.4).
3. **Rider nulls are tripwire-nulls** (engagement caveat verbatim above), and 3/8 confirmatory
   rider cells are void.
4. **Run-(2) bits are their own quantity** (pure-concept channel); any table or figure that
   places them beside secret/evoked LR numbers must carry the NOTE's caveat.
5. Bootstrap CIs descriptive (fixed-τ approximation); seed-sd understates uncertainty
   (registered disclosure).
