"""RED-first unit test for the bits-recovered information currency (exp2). No model, no GPU.

bits_recovered(y_true, proba) = log2(K) - mean cross-entropy (bits) = best-decoder lower bound on
I(concept; output). Run:  .venv/bin/python experiments/exp2_output_monitorability/tests/test_info.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from info import bits_recovered  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

rng = np.random.RandomState(0)
K, n = 12, 600
y = rng.randint(0, K, n)

def emp_entropy(labels, k):
    c = np.bincount(labels, minlength=k) / len(labels)
    return float(-np.sum(c[c > 0] * np.log2(c[c > 0])))

# (1) a perfect reader recovers H(C) bits (the empirical label entropy, ~log2(K) for balanced y)
P = np.full((n, K), 1e-12); P[np.arange(n), y] = 1.0; P /= P.sum(1, keepdims=True)
check("perfect reader -> H(C)", abs(bits_recovered(y, P) - emp_entropy(y, K)) < 1e-6)

# (2) the uniform prior recovers 0 bits
check("uniform -> 0 bits", abs(bits_recovered(y, np.full((n, K), 1.0 / K))) < 0.02)

# (3) binary task, 0.75 on the true class -> H(C) - (-log2 0.75)
y2 = rng.randint(0, 2, n)
P2 = np.zeros((n, 2)); P2[np.arange(n), y2] = 0.75; P2[np.arange(n), 1 - y2] = 0.25
expect = emp_entropy(y2, 2) + math.log2(0.75)
check(f"binary 0.75 -> H(C){math.log2(0.75):+.3f}", abs(bits_recovered(y2, P2) - expect) < 1e-6)

# (4) a confidently-wrong reader is anti-informative (< 0 bits)
W = np.full((n, K), 1e-12); W[np.arange(n), (y + 1) % K] = 1.0; W /= W.sum(1, keepdims=True)
check("confidently wrong -> negative", bits_recovered(y, W) < 0)

# (5) monotone: a sharper-correct reader recovers more than a softer-correct one
soft = np.full((n, K), (1 - 0.5) / (K - 1)); soft[np.arange(n), y] = 0.5
sharp = np.full((n, K), (1 - 0.9) / (K - 1)); sharp[np.arange(n), y] = 0.9
check("sharper-correct recovers more", bits_recovered(y, sharp) > bits_recovered(y, soft) > 0)

# (6) [review BLOCK] imbalanced prior + a marginal-echo reader (zero per-instance info) must read ~0 bits,
#     NOT log2(K)-H(C). This is the bug: the bound is I >= H(C) - CE, not log2(K) - CE.
y_imb = np.concatenate([np.zeros(540, int), rng.randint(0, K, 60)])    # ~90% class 0
freq = np.bincount(y_imb, minlength=K) / len(y_imb)
P_echo = np.tile(freq, (len(y_imb), 1))                                # every row = the empirical marginal
check("imbalanced marginal-echo -> ~0 (no phantom bits)", abs(bits_recovered(y_imb, P_echo)) < 0.02)

# (7) unnormalized rows must be rejected (an invalid bound otherwise)
try:
    bits_recovered(np.array([0, 1]), np.array([[2.0, 3.0], [1.0, 1.0]]))
    check("rejects unnormalized proba", False)
except (AssertionError, ValueError):
    check("rejects unnormalized proba", True)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
