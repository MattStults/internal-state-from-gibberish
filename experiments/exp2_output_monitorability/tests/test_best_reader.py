"""RED-first unit test for best_reader_proba_by_budget (exp2). CPU sklearn only.

Validates the fix for the runner review's CONFOUND: a channel-appropriate pipeline + nested-CV capacity gives
a VALID (>=0) held-out lower bound on BOTH dense (distribution) and sparse (one-hot token) features -- where
the old fixed StandardScaler->PCA pipeline overfit sparse counts into negative (invalid) bits.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_best_reader.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from info import bits_recovered  # noqa: E402
from reader import best_reader_proba_by_budget, reader_proba_by_budget  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

rng = np.random.RandomState(0)
K, nper = 4, 60
y = np.repeat(np.arange(K), nper)

# dense continuous features (distribution-like)
mu = rng.randn(K, 8) * 3.0
Xd = np.vstack([mu[c] + rng.randn(nper, 8) for c in range(K)])

# sparse near-binary COUNT features (one-hot-token-like): each class has signature tokens + noise
d = 100
sig = {c: rng.choice(d, 3, replace=False) for c in range(K)}
Xs = np.zeros((K * nper, d))
for i, c in enumerate(y):
    for tok in sig[c]:
        if rng.rand() < 0.7:
            Xs[i, tok] += 1.0
    for tok in rng.choice(d, 2, replace=False):
        Xs[i, tok] += 1.0

# (1) dense channel recovers real bits, valid distributions
pd = best_reader_proba_by_budget({0: Xd}, y, [0], kind="dense", seed=0)[0]
check("dense recovers", bits_recovered(y, pd) > 0.5)
check("proba (n,K) rows-sum-1", pd.shape == (len(y), K) and abs(pd.sum(1) - 1).max() < 1e-6)

# (2) sparse channel recovers a VALID (>=0) lower bound on sparse counts (the fix)
ps = best_reader_proba_by_budget({0: Xs}, y, [0], kind="sparse", seed=0)[0]
check("sparse recovers, valid >= 0", bits_recovered(y, ps) > 0.3)

# (3) shuffle floor: no memorization through nested CV
y_sh = y.copy(); rng.shuffle(y_sh)
ps_sh = best_reader_proba_by_budget({0: Xs}, y_sh, [0], kind="sparse", seed=0)[0]
check("sparse shuffle -> ~0", bits_recovered(y_sh, ps_sh) < 0.3)

# (4) unknown kind rejected
try:
    best_reader_proba_by_budget({0: Xd}, y, [0], kind="nope")
    check("rejects unknown kind", False)
except ValueError:
    check("rejects unknown kind", True)

# (5) [review] the confound head-to-head: on HIGH-dim sparse counts, the OLD StandardScaler->PCA pipeline
#     overfits (invalid, often < 0), while the new sparse best-reader stays a VALID (>=0) higher lower bound.
dd = 600
sig2 = {c: rng.choice(dd, 2, replace=False) for c in range(K)}
Xs2 = np.zeros((K * nper, dd))
for i, c in enumerate(y):
    for tok in sig2[c]:
        if rng.rand() < 0.6:
            Xs2[i, tok] += 1.0
    for tok in rng.choice(dd, 4, replace=False):     # heavy noise -> StandardScaler blows up rare columns
        Xs2[i, tok] += 1.0
old = bits_recovered(y, reader_proba_by_budget({0: Xs2}, y, [0])[0])
new = bits_recovered(y, best_reader_proba_by_budget({0: Xs2}, y, [0], kind="sparse")[0])
check(f"new sparse valid+beats old (old={old:.2f} new={new:.2f})", new >= 0.0 and new > old)

# (6) [review] REPRODUCIBILITY: at d_model-scale features the dense PCA uses randomized SVD; an unseeded
#     PCA jitters the recovered bits run-to-run (~3e-3 at the real 288x3584 shape -- material at 3 dp, and it
#     rides R_emb = the headline dist-R_emb gap). Same seed -> identical held-out proba. (d>500 + enough rows
#     that n_components < 0.8*min(train) => randomized solver; RED before random_state=seed was threaded in.)
dd3 = 800
mu3 = rng.randn(K, dd3) * 0.3
Xr = np.vstack([mu3[c] + rng.randn(30, dd3) for c in range(K)])
yr = np.repeat(np.arange(K), 30)
pr1 = best_reader_proba_by_budget({0: Xr}, yr, [0], kind="dense", seed=0)[0]
pr2 = best_reader_proba_by_budget({0: Xr}, yr, [0], kind="dense", seed=0)[0]
check("dense same-seed reproducible (randomized-SVD width)", np.array_equal(pr1, pr2))

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
