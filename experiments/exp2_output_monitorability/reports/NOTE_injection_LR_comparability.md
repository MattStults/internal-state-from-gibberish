# Note: injection-LR comparability (TABLED 2026-07-12, un-table after the 72B questions)

Matt's correction to an earlier (wrong) framing:

1. The FAIR injection analog to the secret_word diagonal is to **inject DURING scoring**
   (vector active vs neutral), NOT the inject-off/text-persona version currently built (B1).
   `E_{s~P(·|V)}[LL(s|V) - LL(s|neutral)] = KL(P_injected || P_neutral)` — same structure as the
   secret diagonal's `KL(P_secret || P_neutral)`. It is NOT tautological, for the SAME reason the
   secret diagonal isn't (it's an effect-size, not circular). Claude's earlier "re-injection is
   near-tautological" was an inconsistent standard — corrected.

2. But even the fair (inject-during-scoring) version is NOT directly comparable to secret/evoked LR:
   secret/evoked are scored under a SYSTEM PROMPT = **language + concept** channel; injection with
   the vector is **pure concept** (no language). Different channels; bits don't line up.

3. B1 as currently built (injected streams, vector OFF, scored under a text persona description) is a
   THIRD measurement again ("injected stream under a language description of the concept" ~0.002 @1.5B).

Consequence: B1 (LR-inject at 3B/7B) is interesting on its own but must NOT be framed as a clean
comparison to the secret diagonal. Decide the exact injection-LR variant(s) to run AFTER the 72B
work. Do not build/run B1 now.
