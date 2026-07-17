"""RED-first unit test for rate_bits_per_token (exp2). No model, no GPU.

Synthetic: construct the reader's predicted probabilities so that bits_recovered(T) is EXACTLY linear in the
token budget T with a known slope delta (p_true(T) = 2^(delta*T)/K makes CE = log2 K - delta*T, so
bits = H(C) - CE = delta*T + const). The estimator must recover delta. Teeth: an uninformative reader ->
rate 0, and labels shuffled against a true-y reader must NOT yield a positive rate.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_rate.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from rate import rate_bits_per_token  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

rng = np.random.RandomState(0)
K, n = 12, 400
y = rng.randint(0, K, n)
budgets = [2, 4, 8, 16, 32]


def proba_linear(labels, K, T, delta):
    """probs s.t. bits_recovered(labels, .) = delta*T + const (const cancels in the slope)."""
    m = len(labels)
    p_true = 2.0 ** (delta * T) / K
    P = np.full((m, K), (1.0 - p_true) / (K - 1))
    P[np.arange(m), labels] = p_true
    return P


def proba_saturating(labels, K, T, delta, H):
    """probs s.t. bits_recovered = H*(1 - exp(-(delta/H)*T)): initial slope delta, saturating at H."""
    m = len(labels)
    p_true = 2.0 ** (-(H * math.exp(-(delta / H) * T)))
    P = np.full((m, K), (1.0 - p_true) / (K - 1))
    P[np.arange(m), labels] = p_true
    return P


# (1) recovers the known per-token slope
delta = 0.08
pbb = {T: proba_linear(y, K, T, delta) for T in budgets}
check(f"recovers slope {delta}", abs(rate_bits_per_token(y, pbb, budgets).rate - delta) < 1e-6)

# (2) an uninformative (uniform) reader -> rate ~ 0
pbb_unif = {T: np.full((n, K), 1.0 / K) for T in budgets}
check("uninformative -> ~0 rate", abs(rate_bits_per_token(y, pbb_unif, budgets).rate) < 1e-6)

# (3) monotone: a stronger per-token signal -> a larger rate
pbb2 = {T: proba_linear(y, K, T, 0.15) for T in budgets}
check("monotone in delta", rate_bits_per_token(y, pbb2, budgets).rate > rate_bits_per_token(y, pbb, budgets).rate > 0)

# (4) linear case: windowing doesn't break it
check("window keeps linear slope", abs(rate_bits_per_token(y, pbb, budgets, window=(2, 8)).rate - delta) < 1e-6)

# (5) teeth: labels shuffled against a true-y reader must NOT read a positive rate
y_sh = y.copy(); rng.shuffle(y_sh)
check("shuffled labels -> not positive", rate_bits_per_token(y_sh, pbb, budgets).rate < 0.01)

# (6) [review] SATURATION: a full-range fit under-estimates the low-budget rate; windowing recovers it and r2
#     drops on the bending window -- makes `window` load-bearing and a bad window loud, not silent.
H = math.log2(K)
sat = {T: proba_saturating(y, K, T, 0.10, H) for T in budgets}       # [2..32] reach saturation
full = rate_bits_per_token(y, sat, budgets)
early = rate_bits_per_token(y, sat, budgets, window=(2, 8))
check("saturation: full-range under-estimates", full.rate < 0.8 * 0.10)
check("saturation: window recovers slope + beats full", abs(early.rate - 0.10) < 0.03 and early.rate > full.rate)
check("saturation: r2 higher on the pre-saturation window", early.r2 > full.r2)

# (7) budgets absent from proba_by_budget are rejected
try:
    rate_bits_per_token(y, pbb, [2, 999])
    check("rejects missing budget", False)
except KeyError:
    check("rejects missing budget", True)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
