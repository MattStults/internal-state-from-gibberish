"""RED-first unit test for reader_proba_by_budget (exp2). No model, no GPU (CPU sklearn only).

Synthetic: K classes with distinct feature means; the class separation GROWS with the token budget T. A valid
held-out reader must (a) return proper (n,K) distributions, (b) recover MORE bits as T grows, and (c) collapse
to ~0 bits under shuffled labels (no leakage / overfitting through CV).
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_reader.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from info import bits_recovered  # noqa: E402
from reader import reader_proba_by_budget  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

rng = np.random.RandomState(0)
K, nper, d = 4, 60, 8
mu = rng.randn(K, d) * 3.0                       # distinct per-class means
y = np.repeat(np.arange(K), nper)
budgets = [1, 4, 16]
signal = {1: 0.1, 4: 0.5, 16: 2.0}               # separability grows with the budget


def feats_at(sig):
    return np.vstack([mu[c] * sig + rng.randn(nper, d) for c in range(K)])

feats = {T: feats_at(signal[T]) for T in budgets}
proba = reader_proba_by_budget(feats, y, budgets, pca_dims=8, folds=5)

# (1) proper distributions (so info/rate can consume them)
check("proba is (n,K) rows-sum-1", all(proba[T].shape == (len(y), K) and abs(proba[T].sum(1) - 1).max() < 1e-6
                                       for T in budgets))

# (2) recovers MORE bits as the budget grows
bits = [bits_recovered(y, proba[T]) for T in budgets]
check("bits increases with budget", bits[0] < bits[1] < bits[2])

# (3) separable at high budget -> recovers a real chunk of the log2(K)=2 bits
check("recovers at high budget", bits[2] > 1.0)

# (4) shuffle floor: labels shuffled -> held-out reader recovers ~0 (CV blocks memorization)
y_sh = y.copy(); rng.shuffle(y_sh)
proba_sh = reader_proba_by_budget(feats, y_sh, budgets, pca_dims=8, folds=5)
check("shuffle -> ~0 bits", bits_recovered(y_sh, proba_sh[16]) < 0.3)

# (5) [review] column j really is P(class=j): make class 2 trivially separable, assert its mass lands in col 2
Xco = rng.randn(K * nper, d) * 0.1                # classes 0,1,3 overlap near origin
Xco[y == 2] += mu[2] * 20.0                       # class 2 far away -> trivially separable
pco = reader_proba_by_budget({0: Xco}, y, [0], pca_dims=8, folds=5)[0]
check("column j == P(class=j)", pco[y == 2][:, 2].mean() > 0.8 and pco[y == 2][:, [0, 1, 3]].max() < 0.3)

# (6) [review] non-contiguous labels must be rejected (the column-alignment guard)
y_gap = y.copy(); y_gap[y_gap == 3] = 4           # labels {0,1,2,4}, not 0..K-1
try:
    reader_proba_by_budget({0: feats[16]}, y_gap, [0], pca_dims=8, folds=5)
    check("rejects non-contiguous labels", False)
except (AssertionError, ValueError):
    check("rejects non-contiguous labels", True)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
