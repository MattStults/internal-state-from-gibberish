# LR scale-grid — build & launch checklist

Working artifact for the lr_scale_grid_prereg.md run. Every item is checked off ONLY by an
independent verifier (Fable subagent or Matt), never by the agent that did the work — the
2026-07-09 slip (launched on an amendment agent's self-report of "SCI-CLEARED") is the reason this
rule exists. Reviews required: **TECH (Fable)**, **SCI (Fable)**, **COMPLETENESS (Fable, verifies
this checklist's claimed-done items against reality)** at each phase gate, plus a **HOLISTIC
re-review after smoke, before the full run**.

Past failures this checklist encodes (each burned at least one attempt/launch): letter-token-id
merge bug (B1), transformers-version-vs-model-arch env mismatch, chat-template prefix-instability,
done-marker substring collision, KV legacy-tuple crash, disk-full mid-download, hf-xet silent
stall, batch=8 wasting 2/3 of a card, eos-in-LL artifact, self-report-vs-independent-review slip,
rsync-255 flake handling, under-dosed 7B invalidating a scale claim.

## Phase A — design (prereg frozen)
- [x] Prereg frozen with both named calls verbatim + scoring criteria (committed)
- [x] HF gated-repo access preflight for all 3 Llama readers ($0, local)
- [x] **A1 ADJUDICATED (SCI review of record, 2026-07-11): (a) reader's own chat template**,
      raw-text robustness secondary, date_string pinned, render-diff assert, eos stripped before
      decode. Registered as prereg Amendment 1.
- [x] A2 SCI review (Fable, clean context) DONE 2026-07-11: FINDINGS (2 blocking, 8 should-fix),
      ALL incorporated as prereg Amendment 1 (length confound → per-token shards + secondaries +
      trend-validity clause; named-call consistency carve-out; gate 4b; alt gauge; pinning
      defaults; robustness screen; parity asserts). Verifier = the review itself.
- [x] A3 Matt confirms the "self evocation = MC diagonal" interpretation line in the prereg
      (confirmed by Matt directly, 2026-07-11)

## Phase B — build (each unit: RED test first, fails for the named reason, then green)
- [x] B1 Alt-stream generation at 3B/7B: exp3 induction pipeline, alt personas, config-level
      change; acceptance + word-free gates identical; neutral (s0) streams ride along
- [x] B2 Multi-model LR box: readers × stream-sets × wordings grid; atomic per-cell shards;
      resume-safe; done/fatal markers **collision-checked** (no marker a substring of another)
- [x] B3 Qwen cross-size path: shared-tokenizer assert before teacher-forcing saved ids
- [x] B4 Llama path: decode → re-tokenize; **round-trip check on real gibberish samples**
      (decode(encode(text)) == text or FATAL); identical treatment numerator/denominator;
      per-reader chat-template prefix-stability guard where a template is used (Qwen3 lesson)
- [x] B5 eos rule: eos-free primary + with-eos secondary, both computed, RED test on a synthetic
      stream where they differ
- [x] B6 MC diagonal extension: own-pool selection for 3B/7B (mc_reader scoring code UNCHANGED —
      only pool wiring is new; byte-identical scoring assert in the test)
- [x] B7 deps_for per reader family validated for BOX env (transformers 4.46.3 loads Llama 3.1/3.2
      — verify on-box-equivalent import, not just local; the qwen3 deps bug was ENV not code)
- [x] B8 Perf: VRAM-sized batching per reader; context-KV prefill reuse where exact; smoke-shard
      util gate (<50% util halts the grid); HF_HUB_DISABLE_XET; disk floor = Σ weights + image + slack
- [x] B9 Driver: gated (experimentfactory), ledger-capped, deadman self-destruct, status heartbeat,
      monitor armed at launch; rsync-255 treated as retryable infra flake
- [x] B10 Spend projection from smoke timing. **Matt (2026-07-11): authorized up to $7.50 for this
      run, post-smoke, and delegated the E-phase GO (do not block on him).** So: projection ≤ $7.50
      → proceed on a clean E verdict without waiting for Matt; projection > $7.50 → STOP (that
      magnitude signals a structural problem, not a budget question) and apply the registered trim
      order (secret_sustain → MC diagonal → never LR cells) only with a disclosed recomputation.
- [x] B11 (Amendment 1, Blocker 1): shards store per-token LL vectors (numerator+denominator,
      fp16); offline recomputation test proves eos-free/with-eos/prefix-K all derivable from shards
- [x] B12 (Amendment 1): B1 additions — blind-judge gauge on alt personas at 3B/7B; env/pipeline
      parity asserts vs the exp3 evoked collection; byte-identity assert on wording-A/B persona
      texts across sizes
- [x] B13 (Amendment 1): B4 additions — date_string pinned; render-diff assert (renders differ only
      in persona text); per-stream round-trip exclusion (counted, >5% voids Llama cells, not FATAL)
- [x] B14 (Amendment 1): offline scorer implements secondaries A (per-token) and B (prefix-K,
      frozen rule) + pool-descriptives block printed next to every cell table + gate 4b wiring
- [x] B15 (Amendment 2, Matt-approved): (a) secret_word LR cells — existing bundles at 3 sizes,
      matched one-sentence contexts vs neutral, all 6 readers, shared-expectation ≈0 wiring +
      E5 descriptive cell; (b) secret_sustain arm — S1 generation at 1.5/3/7B (secret_word context
      + E2 piloted s1 sustain template, word substituted; NO gauge, registered), LR + char scoring,
      both frozen calls wired (MATT-imbue: 7B LR ≥ 0.10; CLAUDE-mechanism: char ≥ LR − 0.05
      wherever either ≥ 0.10, AND 1.5B LR < 0.10); registered trim order in the B10 projection

*(B1–B15 checked off 2026-07-11 on the authority of the C3 COMPLETENESS review of record —
691 tests re-run independently across 29 suites, 0 fail; RED trails, sha-pins, checkoff
independence, and the E5 HF upload all verified against the repo, not transcripts.)*

## Phase C — pre-launch reviews (all Fable, clean context, on the actual diff)
- [x] C1 TECH review DONE 2026-07-11: FINDINGS (0 blocking, 3 should-fix) — scoped swap, TF
      alignment, env, markers, lifecycle all verified; findings fixed and re-review-certified (C4)
- [x] C2 SCI review DONE 2026-07-11: FINDINGS (4 blocking, 3 should-fix) — measurement core
      verified faithful, all 12 pins blessed; Blocker 4 closed as prereg Amendment 3 (reviewer's
      draft verbatim); remaining findings fixed and re-review-certified (C4)
- [x] C3 COMPLETENESS review DONE 2026-07-11: **CLEAN** — 691/691 across 29 suites re-run
      independently; RED trails, sha-pins, checkoff independence, HF fetch paths all verified
      against the repo. Three non-blocking notes handed to C1/C2 (stale E5 comment; code-pinned
      values for SCI to ratify; mc_reader orchestration filters disclosed).
- [x] C4 CLOSED 2026-07-11: independent re-review of the 9-commit fix diff — **"C4: CLEAN — all
      findings closed"** (742/742 across 29 suites re-run by the reviewer; certified modules
      untouched; no scope creep). Notes carried forward: N1 run the smoke scoring command from repo
      root; N2 the F3 verdicts doc must apply Amendment 3's "RIGHT (vacuous — mechanism untested)"
      label verbatim.

## Phase D — smoke (cheapest real box, ~minutes)
- [x] D1 DONE 2026-07-11 (attempt 5, RTX 6000Ada, ~$0.25): full slice ran end-to-end incl.
      Falcon3 template+raw+prose, secret_word cell, MC diagonal shard; pulled + offline-scored to
      lr_grid_smoke_results.json. (Attempts 1-4: wordfreq attr fix $0.01; meta-llama 403 →
      Amendment 4 Falcon3 swap ~$0.03; 2× no_offer $0.)
- [x] D2 DONE: evoked×A eos-free 0.604 (certified 0.590 ✓); eos-free 1.5B diagonal anchor
      recorded = 0.166; alt×B 0.526 (cert 0.565 ✓); gate 2 narrow miss NEGATIVE sign (matches the
      certified run); Falcon3 prose control ran (family line correctly not_resolvable at 1B per
      gate 4b). REGISTERED SURPRISE: secret_word×SW = 0.163 [0.120,0.218] → both_wrong; two $0
      validity controls run + registered as Amendment 5 (char-surface −0.003; position-resolved
      lift distributed, 25% in first 4 tokens) — both PASS.
- [x] D3 DONE: qwen1.5b 5998 tok/s @74% util, falcon3-1b 13101 tok/s @98% (floor 50%); no batch
      change required for launch (larger readers meet the util gate mid-run per C1 note).
- [x] D4 DONE 2026-07-11: projection $16.17 @$1.00/hr ($11.97 @$0.74) fired the registered STOP
      (> the $7.50 authorization). **Matt (2026-07-11): authorization raised to $25 for this run,
      full scope, no trims.** 16.2h total → driver launches with max-hours 20, max-spend $25;
      single contiguous box (resume-safe shards cover host-failure risk).

## Phase E — HOLISTIC re-review (Fable, clean context) — after smoke, before the real run
- [x] E1 ROUND 1 DONE 2026-07-11: **"E: NO-GO (provider dead-man clamps to 6 hours ... the box
      self-kills ~6h into a 16.2h run)"** — VastProvider.default_deadman_s=6h min()-caps the
      driver's 20h request; no driver ever passed the kwarg; smoke (15min) structurally could not
      expose it. ALL other checks passed (anchors, gates, secret controls recomputed from disk,
      Falcon3 swap, disk/spend arithmetic). Fix: one driver line + RED test → scoped re-review →
      repeat E on the fix alone. Conditions attached: (1) launch flags explicit + --dry eyeball +
      B10 constant → $25; (2) Amendment 5 controls must be IMPLEMENTED in the scorer before F3
      claims any secret cell; (3) F2 ordering below; (4) test hermeticity (smoke_json explicit in
      tests); (5) host-death after ~10h leaves no headroom for a full second attempt under $25 —
      accepted, ask Matt only if it fires.
- [x] E2 GO recorded 2026-07-11: scoped E-repeat review of the fix diff (03e1d0a..8374afa) —
      **"E-REPEAT: GO"** — deadman traced end-to-end (min(73800,72600)=20.17h ≥ 16.2h run;
      pre-fix control re-derived at exactly 6.0h), $25 gate boundary-tested, Amendment 5 controls
      AST-verified non-invasive + real-shard share 0.2522 reproduced, hermeticity tests-only,
      no scope creep, 760/760 in-tree, dry run exits 0. Mandated launch command:
      run_lr_grid.py --max-dph 1.00 --max-spend 25 --max-hours 20. Carried note for F3:
      a PENDING Amendment-5 control is NOT-PASSED — no secret cell may be claimed positive on
      pending controls.

## Phase F — full run
- [ ] F1 Launch on 48GB tier per prereg perf checklist; contiguous generation → LR grid → MC
      diagonal; monitor armed
- [ ] F2 Pull → run gauge_judge_alt on the new 3B/7B alt bundles → gates evaluated FIRST →
      offline scoring (Amendment 5 char + position controls reported next to EVERY secret cell)
      → verdicts; voided cells marked before any interpretation (E1 conditions 2+3)
- [ ] F3 Verdicts doc: both named calls scored against the frozen criteria, verbatim quotes
- [ ] F4 BLOGPOST integration (incl. the pending scope-correction: MC run's ≥3B readers read the
      1.5B pool — superseded by the diagonal cells from this run)
