"""RED-first unit test for kl_bits_topk (exp2). No model, no GPU.

Strategy: for a small vocab, brute-force the FULL distributions (top-K probs + a uniform tail) and compute
KL directly, then assert kl_bits_topk matches exactly. Run:
  .venv/bin/python experiments/exp2_output_monitorability/tests/test_dists.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from dists import kl_bits_topk  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))


def full_from_topk(ids, probs, V):
    fill = (1.0 - sum(probs)) / (V - len(ids)) if V > len(ids) else 0.0
    f = np.full(V, fill)
    for i, p in zip(ids, probs):
        f[i] = p
    return f

def full_kl_bits(p, q):
    return float(np.sum(p * np.log2(p / q)))


# (1) identical distributions -> 0
ids = [0, 1, 2, 3]; lp = np.log([0.4, 0.3, 0.2, 0.1])
check("identical -> 0", abs(kl_bits_topk(ids, lp, ids, lp, 4)) < 1e-9)

# (2) full vocab (no tail), exact KL vs brute force
p_pr, q_pr = [0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]
got = kl_bits_topk([0, 1, 2, 3], np.log(p_pr), [0, 1, 2, 3], np.log(q_pr), 4)
ref = full_kl_bits(full_from_topk([0, 1, 2, 3], p_pr, 4), full_from_topk([0, 1, 2, 3], q_pr, 4))
check("full V=4 exact", abs(got - ref) < 1e-9)

# (3) asymmetric: KL(p||q) != KL(q||p)
rev = kl_bits_topk([0, 1, 2, 3], np.log(q_pr), [0, 1, 2, 3], np.log(p_pr), 4)
check("asymmetric", abs(got - rev) > 1e-3)

# (4) tail + DISJOINT top-K ids, exact vs brute (the real gen_topk case)
V = 10
p_ids, p_pr = [0, 1, 2], [0.5, 0.2, 0.1]      # tail 0.2 over 7
q_ids, q_pr = [0, 3, 4], [0.4, 0.1, 0.1]      # tail 0.4 over 7
got = kl_bits_topk(p_ids, np.log(p_pr), q_ids, np.log(q_pr), V)
ref = full_kl_bits(full_from_topk(p_ids, p_pr, V), full_from_topk(q_ids, q_pr, V))
check("tail+disjoint exact", abs(got - ref) < 1e-9)

# (5) non-negative on a generic pair
check("non-negative", kl_bits_topk([0, 1], np.log([0.6, 0.2]), [0, 1], np.log([0.3, 0.3]), 50) >= 0)

# (6) floor branch: q has NO tail (covers all mass) but p puts mass where q can't -> large but FINITE
#     (true KL is +inf; the floor caps it deliberately). Real top-K gen_topk never hits this (tail always > 0).
big = kl_bits_topk([2, 3], np.log([0.5, 0.3]), [0, 1], np.log([0.6, 0.4]), 10)
check("q-zero-tail -> large but finite", np.isfinite(big) and 100 < big < 1e6)

# (7) duplicate ids must be rejected (else tail mass/count is silently wrong)
try:
    kl_bits_topk([0, 0, 1], np.log([0.4, 0.4, 0.2]), [0, 1], np.log([0.6, 0.4]), 10)
    check("rejects duplicate ids", False)
except (AssertionError, ValueError):
    check("rejects duplicate ids", True)

# (8) ids >= vocab_size must be rejected
try:
    kl_bits_topk([0, 99], np.log([0.6, 0.4]), [0, 1], np.log([0.6, 0.4]), 10)
    check("rejects ids >= vocab_size", False)
except (AssertionError, ValueError):
    check("rejects ids >= vocab_size", True)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
