"""Is arm B - C (the prefill metric) significantly NEGATIVE, and is it concept-specific?
Pulls per-concept B-C + bootstrap CIs from covert_v2_results.json (no recompute).
Run:  .venv/bin/python analysis/check_bmc_significance.py"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

for slug in ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]:
    try:
        v2 = json.loads((REPO / "runs" / slug / "results" / "covert_v2_results.json").read_text())
    except FileNotFoundError:
        print(f"\n=== {slug} === (no covert_v2_results.json committed — skipping)")
        continue
    strong = str(max(int(k) for k in v2["strengths"]))
    pf = v2["strengths"][strong]["prefill"]
    rows = [(c, v["BmC"], v["lo"], v["hi"]) for c, v in pf.items()]
    bmc = np.array([r[1] for r in rows])
    sig_neg = [(c, b, lo, hi) for (c, b, lo, hi) in rows if hi < 0]      # CI entirely below 0
    sig_pos = [(c, b, lo, hi) for (c, b, lo, hi) in rows if lo > 0]      # CI entirely above 0
    spans = [(c, b, lo, hi) for (c, b, lo, hi) in rows if lo <= 0 <= hi]
    # aggregate: bootstrap the MEAN over concepts (resample concepts)
    rng = np.random.default_rng(0)
    boot = np.array([bmc[rng.integers(0, len(bmc), len(bmc))].mean() for _ in range(5000)])
    mlo, mhi = np.percentile(boot, [2.5, 97.5])
    print(f"\n=== {slug} (s{strong}) ===")
    print(f"  mean B-C = {bmc.mean():+.3f} nats,  bootstrap-over-concepts 95% CI = [{mlo:+.3f}, {mhi:+.3f}]"
          f"  -> {'MEAN SIG. NEGATIVE' if mhi < 0 else ('mean sig. positive' if mlo > 0 else 'spans 0')}")
    print(f"  per-concept CIs: {len(sig_neg)} sig-NEG (hi<0), {len(sig_pos)} sig-pos (lo>0), {len(spans)} span 0  (of 12)")
    if sig_neg:
        print("   sig-negative concepts: " + ", ".join(f"{c}({b:+.2f})" for c, b, lo, hi in sorted(sig_neg, key=lambda x: x[1])))
    sel_p = json.loads((REPO / "runs" / slug / "results" / "reevocation_results.json").read_text()).get("selectivity_p")
    print(f"  concept-specific? selectivity p (diag>offdiag) = {sel_p:.3f}  "
          f"-> {'concept-selective' if sel_p < 0.05 else 'NOT concept-selective (looks generic, not concept-targeted)'}")
