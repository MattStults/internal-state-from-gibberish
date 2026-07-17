"""Pull the v1-writeup numbers straight from the committed analysis outputs (no recompute, no GPU).
Run:  .venv/bin/python analysis/verify_v1_numbers.py"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
MODELS = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]

print(f"{'model':>14} {'strong':>7} {'R1 surf':>8} {'floor':>6} {'R2 act':>7} {'R1-R2':>6} "
      f"{'R3 B-C':>8} {'R3 ceil':>8} {'reevoc':>20}")
for slug in MODELS:
    rdir = REPO / "runs" / slug / "results"
    try:
        v2 = json.loads((rdir / "covert_v2_results.json").read_text())
    except FileNotFoundError:
        print(f"{slug:>14}  (no covert_v2_results.json committed — skipping)")
        continue
    strong = str(max(int(k) for k in v2["strengths"]))
    e = v2["strengths"][strong]
    r1 = e["surface"]["mean"]; floor = e["surface"]["thr95"]; r2 = e["activation"]["mean"]
    pf = e.get("prefill", {})
    bmc = float(np.mean([v["BmC"] for v in pf.values()])) if pf else float("nan")
    ceil = float(np.mean([v["ceiling"] for v in pf.values() if v.get("ceiling") is not None])) if pf else float("nan")
    # reevocation: print whatever summary it carries (structure unknown -> show top-level)
    try:
        re = json.loads((rdir / "reevocation_results.json").read_text())
        if isinstance(re, dict):
            # try common keys; else summarize
            keys = ["p", "pval", "p_value", "aligned", "cos", "mean_cos", "verdict"]
            hit = {k: re[k] for k in keys if k in re}
            re_str = str(hit) if hit else f"keys={list(re)[:4]}"
        else:
            re_str = str(re)[:20]
    except FileNotFoundError:
        re_str = "(none)"
    print(f"{slug:>14} {strong:>7} {r1:>8.3f} {floor:>6.3f} {r2:>7.3f} {r1 - r2:>6.3f} "
          f"{bmc:>8.3f} {ceil:>8.3f} {re_str:>20}")

print("\n(R1=surface mean OvR balanced acc; R2=activation mean; R3 B-C=mean over concepts of arm-B minus "
      "control logP(concept word); R3 ceil=mean arm-A nameability over the same control.)")
