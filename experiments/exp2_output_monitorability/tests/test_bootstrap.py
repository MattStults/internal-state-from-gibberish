"""RED-first unit test for the concept-level bootstrap CI (info.concept_bootstrap_ci + the run_budget pooling
helper). This replaces the 3-seed sd, whose 2 dof over overlapping subsamples understated uncertainty; the
generalization unit is the CONCEPT. Pure numpy on synthetic proba -- instant, no reader, no data.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_bootstrap.py
"""
import os
import sys

import numpy as np

AN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
sys.path.insert(0, AN)
from info import bits_recovered, concept_bootstrap_ci  # noqa: E402
from run_budget import bootstrap_ci_from_per           # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

K, per = 6, 12
y = np.repeat(np.arange(K), per)
n = len(y)

# dist: confident-correct proba, but with PER-CONCEPT heterogeneity (some concepts recovered better than
# others) so the concept bootstrap has real between-concept variance to capture. emb: uniform (~0 bits).
conf = {c: 0.40 + 0.55 * (c / (K - 1)) for c in range(K)}   # correct-mass ranges 0.40..0.95 across concepts
Pd = np.zeros((n, K))
for i, c in enumerate(y):
    Pd[i] = (1 - conf[c]) / (K - 1)
    Pd[i, c] = conf[c]
Pe = np.full((n, K), 1.0 / K)

ci = concept_bootstrap_ci(y, {"dist": Pd, "emb": Pe},
                          gaps=[("dist_minus_emb", "dist", "emb")], n_boot=500, seed=0)

# (1) structure: a (lo, hi) per mode + per gap
check("bits_ci has both modes", set(ci["bits_ci"]) == {"dist", "emb"})
check("gap_ci has the gap", "dist_minus_emb" in ci["gap_ci"])
check("lo <= hi", all(lo <= hi for lo, hi in list(ci["bits_ci"].values()) + list(ci["gap_ci"].values())))

# (2) the CI brackets the point estimate on the full sample
pt_dist = bits_recovered(y, Pd)
lo, hi = ci["bits_ci"]["dist"]
check(f"dist CI brackets point est ({lo:.2f}<= {pt_dist:.2f} <={hi:.2f})", lo <= pt_dist <= hi)

# (3) signal vs null: dist CI strictly above 0; uniform-emb CI contains ~0; gap CI strictly above 0
check("dist CI lo > 0 (real signal)", ci["bits_ci"]["dist"][0] > 0.1)
check("emb CI brackets 0 (uniform reader, fixed-H)", ci["bits_ci"]["emb"][0] <= 0.0 <= ci["bits_ci"]["emb"][1])
check("dist-emb gap CI lo > 0", ci["gap_ci"]["dist_minus_emb"][0] > 0.1)

# (4) reproducible at fixed seed; wider CI is not degenerate (lo != hi for the signal)
ci2 = concept_bootstrap_ci(y, {"dist": Pd, "emb": Pe}, gaps=[("dist_minus_emb", "dist", "emb")], n_boot=500, seed=0)
check("same seed reproducible", ci2["bits_ci"]["dist"] == ci["bits_ci"]["dist"])
check("non-degenerate interval", ci["bits_ci"]["dist"][1] - ci["bits_ci"]["dist"][0] > 1e-6)

# (5) pooling helper: concatenates per-seed (y, top_proba) and returns a CI dict; None when proba absent
per = [{"y": y, "top_T": 12, "top_proba": {"dist": Pd, "emb": Pe}} for _ in range(3)]
pooled = bootstrap_ci_from_per(per, n_boot=200, seed=0)
check("pooling helper returns CI + top_T", pooled is not None and pooled["top_T"] == 12
      and pooled["gap_ci"]["dist_minus_emb"][0] > 0.1)
check("pooling helper None without proba", bootstrap_ci_from_per([{"bits": {}}]) is None)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
