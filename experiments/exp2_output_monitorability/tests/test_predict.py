"""RED-first unit test for predict_budget + verify_recovery (exp2). No model, no GPU.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_predict.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from predict import predict_budget, verify_recovery  # noqa: E402
from rate import RateFit  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

def raises(fn):
    try:
        fn(); return False
    except (ValueError, ZeroDivisionError):
        return True

# (1) T* through the origin; (2) intercept used
check("T* through origin", math.isclose(predict_budget(RateFit(0.1, 0.0, 1.0), 1.0), 10.0))
check("T* uses intercept", math.isclose(predict_budget(RateFit(0.1, 0.2, 1.0), 1.0), 8.0))

# (3) non-positive rate; (4) [review] non-finite rate slips a bare `rate<=0` -> must raise
check("non-positive rate raises", raises(lambda: predict_budget(RateFit(0.0, 0.0, 1.0), 1.0))
      and raises(lambda: predict_budget(RateFit(-0.05, 0.0, 1.0), 1.0)))
check("non-finite rate raises", raises(lambda: predict_budget(RateFit(float("nan"), 0.0, 1.0), 1.0))
      and raises(lambda: predict_budget(RateFit(float("inf"), 0.0, 1.0), 1.0)))

# (5) [review] low r2 (saturating window) inflates T* -> must raise, not ship silently
check("low r2 raises", raises(lambda: predict_budget(RateFit(0.1, 0.0, 0.5), 1.0)))

# (6) [review] required <= intercept is a degenerate fit (recovery at T=0), not a real 0-token budget
check("required<=intercept raises", raises(lambda: predict_budget(RateFit(0.1, 1.5, 1.0), 1.0)))

# (7) negative required; (8) [review] required above the reachable ceiling H(C)
check("negative required raises", raises(lambda: predict_budget(RateFit(0.1, 0.0, 1.0), -0.5)))
check("above ceiling raises", raises(lambda: predict_budget(RateFit(0.1, 0.0, 1.0), 3.0, ceiling=2.0)))

# (9) verify_recovery via bootstrap lower bound: recovered -> True, uniform -> False
rng = np.random.RandomState(0)
K, n = 4, 200
y = rng.randint(0, K, n)
P_good = np.full((n, K), 1e-9); P_good[np.arange(n), y] = 1.0; P_good /= P_good.sum(1, keepdims=True)
check("verify: recovered -> True", verify_recovery(y, P_good, 1.0) is True)
check("verify: uniform -> False", verify_recovery(y, np.full((n, K), 1.0 / K), 1.0) is False)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
