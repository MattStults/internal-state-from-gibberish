"""RED-first unit test for pooled_dist_features (exp2). No model, no GPU.

Synthetic gen_topk steps ({ids, natural-log logp}); assert the featurizer returns, per vocab token, the mean
probability the model put on it over the first `budget` steps.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_features.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from features import embed_features, pooled_dist_features  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

vocab = {5: 0, 9: 1, 3: 2}                        # token id -> feature column

# (1) mean prob per vocab token, all steps
steps = [{"ids": np.array([5, 9]), "logp": np.log([0.7, 0.3])} for _ in range(10)]
check("mean prob per vocab token", np.allclose(pooled_dist_features(steps, 10, vocab), [0.7, 0.3, 0.0]))

# (2) budget truncates to the first T steps (steps 0-1 are token 5; later steps are token 9)
steps2 = [{"ids": np.array([5]), "logp": np.log([1.0])}] * 2 + [{"ids": np.array([9]), "logp": np.log([1.0])}] * 8
check("budget truncates", np.allclose(pooled_dist_features(steps2, 2, vocab), [1.0, 0.0, 0.0]))
check("full budget sees later steps", np.allclose(pooled_dist_features(steps2, 10, vocab), [0.2, 0.8, 0.0]))

# (3) off-vocab tokens are ignored (not every top-K token is a feature)
steps3 = [{"ids": np.array([5, 99]), "logp": np.log([0.6, 0.4])}]
check("off-vocab ignored", np.allclose(pooled_dist_features(steps3, 1, vocab), [0.6, 0.0, 0.0]))

# (4) a token absent from a step's top-K contributes 0 that step (averaged in)
steps4 = [{"ids": np.array([5]), "logp": np.log([1.0])}, {"ids": np.array([9]), "logp": np.log([1.0])}]
check("absent-in-step -> 0 that step", np.allclose(pooled_dist_features(steps4, 2, vocab), [0.5, 0.5, 0.0]))

# (5) [review] uniform-tail floor: with vocab_size, an ABSENT vocab token gets the per-step tail prob
#     (1 - sum_topK)/(V - K), not 0 -- closes the top-K boundary discontinuity for Prediction #2.
V = 100
stepsF = [{"ids": np.array([5]), "logp": np.log([0.6])}]        # top-1 0.6, tail 0.4 over V-1=99
ff = pooled_dist_features(stepsF, 1, {5: 0, 9: 1}, vocab_size=V)
check("floor: absent -> uniform tail", np.isclose(ff[1], 0.4 / 99) and np.isclose(ff[0], 0.6))
check("no floor default: absent -> 0", np.isclose(pooled_dist_features(stepsF, 1, {5: 0, 9: 1})[1], 0.0))

# --- embed_features (R_emb best-token-monitor): mean-pool token embeddings over the first T realized tokens ---
E = np.arange(12.0).reshape(4, 3)                 # 4-token vocab, d=3: row t = [3t, 3t+1, 3t+2]
# (6) mean of the first T tokens' embedding rows
check("mean-pool over budget", np.allclose(embed_features([1, 3], 2, E), (E[1] + E[3]) / 2))
check("budget truncates embed", np.allclose(embed_features([1, 3, 2], 1, E), E[1]))
# (7) out-of-vocab / empty handled
check("oob token skipped", np.allclose(embed_features([1, 99], 2, E), E[1]))
check("empty -> zeros", np.allclose(embed_features([], 4, E), np.zeros(3)))

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
