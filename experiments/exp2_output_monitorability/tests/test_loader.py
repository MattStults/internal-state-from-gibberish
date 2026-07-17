"""RED-first unit test for filter_streams (exp2). No model, no GPU (synthetic bundle dict).
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_loader.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from loader import filter_streams  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

def st(**kw):
    return kw

bundle = dict(
    strengths=[0, 40, 60], concepts=["a", "b"], inject="gen", model="m",
    streams=[
        st(accepted=True, strength=60, concept_idx=0, tokens=[1, 2, 3], gen_topk=[{"ids": [1], "logp": [0.0]}]),
        st(accepted=False, strength=60, concept_idx=1, tokens=[4], gen_topk=[{"ids": [4], "logp": [0.0]}]),   # rejected
        st(accepted=True, strength=0, concept_idx=1, tokens=[5], gen_topk=[{"ids": [5], "logp": [0.0]}]),      # control dose
        st(accepted=True, strength=60, concept_idx=1, tokens=[6, 7], gen_topk=None),                           # no gen_topk
        st(accepted=True, strength=60, concept_idx=1, tokens=[8, 9], gen_topk=[{"ids": [8], "logp": [0.0]}]),
    ])

res = filter_streams(bundle)

# (1) keep only accepted + strong-dose + gen_topk-present streams (streams 0 and 4)
check("keeps accepted strong w/ gen_topk", len(res["streams"]) == 2)

# (2) concept labels + order preserved
check("concept_idx preserved", [s["concept_idx"] for s in res["streams"]] == [0, 1])

# (3) tokens normalized to an int array
check("tokens -> int array", isinstance(res["streams"][0]["tokens"], np.ndarray)
      and res["streams"][0]["tokens"].tolist() == [1, 2, 3])

# (4) gen_topk carried through
check("gen_topk carried", res["streams"][1]["gen_topk"] == [{"ids": [8], "logp": [0.0]}])

# (5) strength defaults to the max dose; meta passed through
check("strong = max strength + meta", res["strength"] == 60 and res["concepts"] == ["a", "b"] and res["inject"] == "gen")

# (6) explicit strength selects a different dose (only the s0 accepted stream)
check("explicit strength", len(filter_streams(bundle, strength=0)["streams"]) == 1)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
