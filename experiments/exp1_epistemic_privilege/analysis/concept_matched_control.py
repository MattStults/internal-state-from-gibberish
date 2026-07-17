"""Is arm B-C's negativity GENERIC (injection suppresses word-naming) or CONCEPT-SPECIFIC?
Recompute the prefill metric with a concept-MATCHED control (other-concept-injected gibberish, the original
arm C) instead of the s0 control. If the negative VANISHES, it was generic (both arms carry the same
injection-induced suppression, which cancels). If it PERSISTS, there is a real concept-specific anti-leak.

own_c     = mean logP(word c | arm-B read) over streams injected with c     (strong strength)
ctrl_s0_c = mean logP(word c | arm-B read) over s0 (un-injected) streams    [original control]
ctrl_oth_c= mean logP(word c | arm-B read) over streams injected with OTHER concepts (strong) [matched]

Run:  .venv/bin/python analysis/concept_matched_control.py
"""
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent


def endpoint(reads, gidx):
    r = reads.get(("B", gidx))
    return r[max(r.keys())] if r else None


for slug in ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]:
    d = torch.load(REPO / "runs" / slug / "data" / "covert_collect.pt", map_location="cpu", weights_only=False)
    S = d["streams"]; reads = d["reads"]; K = len(d["concepts"])
    strong = max(d["strengths"])

    def logp_col(streams, c):  # logP(word c) on arm-B reads of these streams
        vals = [endpoint(reads, r["gidx"])["logp"][c] for r in streams if r["accepted"] and endpoint(reads, r["gidx"]) is not None]
        return np.array([float(v) for v in vals])

    bmc_s0, bmc_matched = [], []
    for c in range(K):
        own = logp_col([r for r in S if r["concept_idx"] == c and r["strength"] == strong], c)
        s0 = logp_col([r for r in S if r["strength"] == 0], c)
        other = logp_col([r for r in S if r["concept_idx"] != c and r["strength"] == strong], c)
        bmc_s0.append(own.mean() - s0.mean())
        bmc_matched.append(own.mean() - other.mean())
    bmc_s0 = np.array(bmc_s0); bmc_matched = np.array(bmc_matched)

    def agg(x):
        rng = np.random.default_rng(0)
        boot = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return x.mean(), lo, hi, int((x < 0).sum())

    m0, l0, h0, n0 = agg(bmc_s0)
    mm, lm, hm, nm = agg(bmc_matched)
    print(f"\n=== {slug} (s{strong}) ===")
    print(f"  s0 control     : mean B-C = {m0:+.3f}  CI [{l0:+.3f},{h0:+.3f}]  ({n0}/{K} concepts negative)  "
          f"-> {'SIG NEG' if h0 < 0 else 'spans 0'}")
    print(f"  matched control: mean B-C'= {mm:+.3f}  CI [{lm:+.3f},{hm:+.3f}]  ({nm}/{K} concepts negative)  "
          f"-> {'SIG NEG' if hm < 0 else ('SIG POS' if lm > 0 else 'spans 0')}")
    verdict = ("GENERIC suppression (s0-control negative cancels under a concept-matched control)"
               if (h0 < 0 and lm <= 0 <= hm) else
               "CONCEPT-SPECIFIC anti-leak (persists under matched control)" if hm < 0 else
               "concept-specific POSITIVE leak (matched control is positive)" if lm > 0 else "mixed/unclear")
    print(f"  VERDICT: {verdict}")
