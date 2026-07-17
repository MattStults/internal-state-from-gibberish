"""exp2: the predict-then-demonstrate budget.

predict_budget inverts the fitted per-token rate to T* = the token budget needed to recover `required_bits`
about the concept: T* = (required_bits - intercept) / rate (the intercept matters -- the rate line fit on an
early window does not pass through the origin). verify_recovery then checks that a reader given ~T* tokens
actually crosses the threshold -- the out-of-sample confirmation that makes it a prediction, not a curve fit.

`required_bits` is in bits_recovered (empirical H(C) - CE) currency; the caller converts any accuracy
threshold to bits using the SAME empirical H(C).
"""
import math

import numpy as np

from info import bits_recovered


def predict_budget(rate_fit, required_bits, min_r2=0.9, ceiling=None):
    """rate_fit: a rate.RateFit(rate, intercept, r2) from an early/pre-saturation window. Returns the token
    budget T* = (required_bits - intercept) / rate. Raises on the ways a WRONG T* would otherwise ship
    silently: a low-r2 (saturating) window that inflates T* [min_r2]; a non-finite or non-positive rate;
    required_bits above the reachable ceiling H(C) [pass ceiling]; required_bits <= the fit intercept (a
    degenerate fit claiming recovery at T=0 -- suspect window/leakage); negative required_bits."""
    rate, intercept, r2 = rate_fit.rate, rate_fit.intercept, rate_fit.r2
    if required_bits < 0:
        raise ValueError(f"required_bits {required_bits} < 0")
    if ceiling is not None and required_bits > ceiling:
        raise ValueError(f"required_bits {required_bits} > ceiling H(C)={ceiling}: unreachable")
    if not (math.isfinite(rate) and math.isfinite(intercept)):
        raise ValueError(f"rate_fit has non-finite rate/intercept ({rate}, {intercept})")
    if rate <= 0.0:
        raise ValueError(f"rate {rate} <= 0: target unreachable (reader not information-limited / window past saturation)")
    if r2 < min_r2:
        raise ValueError(f"rate_fit.r2 {r2:.3f} < min_r2 {min_r2}: fit window reached saturation, T* would be "
                         "inflated -- refit on an earlier window")
    if required_bits <= intercept:
        raise ValueError(f"required_bits {required_bits} <= fit intercept {intercept:.3f}: degenerate fit "
                         "(claims recovery at T=0) -- check the window / leakage")
    return float((required_bits - intercept) / rate)


def verify_recovery(y_true, proba_at_budget, required_bits, n_boot=1000, alpha=0.05, seed=0):
    """The out-of-sample check: does a reader given ~T* tokens actually recover >= required_bits? Uses a
    paired bootstrap over streams (not a bare point threshold on one noisy draw): returns True iff the
    lower (alpha) percentile of the bootstrapped bits_recovered is >= required_bits, so a pass means the
    crossing is real, not noise. proba_at_budget: held-out predicted concept distributions at the budget."""
    y = np.asarray(y_true)
    P = np.asarray(proba_at_budget)
    n = len(y)
    rng = np.random.RandomState(seed)
    boots = np.array([bits_recovered(y[i], P[i]) for i in (rng.randint(0, n, n) for _ in range(n_boot))])
    return bool(np.percentile(boots, 100 * alpha) >= required_bits)
