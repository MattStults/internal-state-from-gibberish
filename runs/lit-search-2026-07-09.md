# Pre-publish literature search — verdicts (2026-07-09 sweep, full report in session transcript)

**No claim is scooped.** Hard requirements before publishing:

1. **Steering-detection claim** (safety section): narrow wording to "from output text alone, by a reader
   with NO model access, identifying WHICH concept was steered" — the generic "steering is detectable"
   is no longer novel. CITE: Steering Awareness arXiv:2511.21399 (detection from within the steered
   model's own activations, Nov 2025) and LLM Self-Recognition arXiv:2606.06315 (deliberate text-carried
   steering watermark, Jun 2026) as nearest neighbors. Also relevant: 2604.09839 (white-box
   non-surjectivity), 2512.03816 (logprob tracking detects model changes, not concepts).
2. **secret_word / wording-tied discussion**: CITE Holtzman & West, "Can You Keep a Secret?"
   arXiv:2605.10794 (May 2026) — text-only detector recovers a hidden secret THEMATICALLY from
   unconstrained writing, up to 79% above chance, leakage GROWS with scale. Reviewer-collision likely if
   uncited. Guard the TL;DR secret-word sentence with: consistent with the semantic channel
   (topic/imagery) being the carrier in natural text, which the word-free constraint removes. Their
   grows-with-scale is a striking foil to our natural-channel-closes — worth one contrast sentence.
   Also CITE PILS arXiv:2506.17090 (+ Morris et al. 2311.13647): prompt WORDING recoverable verbatim
   from logprobs — prior evidence for wording-tied signal.
3. **Attribution fixes**: "Anthropic's Subliminal Learning" → **Cloud et al. (2025)**, arXiv:2507.14805
   (Anthropic Fellows / Truthful AI — not an Anthropic paper). Motwani et al. 2024 final title =
   "Secret Collusion among AI Agents: Multi-Agent Deception via Steganography" (NeurIPS 2024).
   MacDiarmid et al. sleeper-agent probes = Anthropic Alignment Science blog post, not arXiv.
   Korbak et al. 2025 = arXiv:2507.11473 ✓. Persona Vectors Chen et al. 2507.21509 ✓.
4. **Soften** "deliberate encoding is untested" → "untested in this passive/word-free setting"
   (TrojanStego 2505.20118 + 2606.09411 show TRAINED steganographic exfiltration works). Cite
   2601.22818 (embedding-space stego) at the capacity remark; 2601.21112 (monitor information access)
   at the monitoring-architecture reading.
5. **For E4/trajectory writeup later**: frame vs The Assistant Axis 2601.10387 (conversation-level
   persona drift via projections) as "token-level, under constrained output"; measurement primitive =
   Persona Vectors' projection-over-tokens.
6. **Read before publishing**: arXiv:2606.00995 "Subliminal Learning Is Steering Vector Distillation" —
   directly bridges our injected regime and the SL framing in the intro; also 2606.22019 (channel
   location constrains auditability of SL).
