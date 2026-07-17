"""exp2: the per-token evidence rate = the slope of bits_recovered(T) vs total emitted tokens T.

This is the rate the predict-then-verify loop consumes: T* = required_bits / rate. It is currency-consistent
with info.bits_recovered BY CONSTRUCTION -- per the methodology review, recovery is reader-defined, so the
rate that forecasts T* must be the SAME reader's realized bits/token, not a channel-KL quantity (that would
over-predict monitorability and the demonstration would fail). Reader-agnostic: takes the reader's held-out
predicted concept distributions at each token budget; the reader itself is a later unit.
"""
from collections import namedtuple

import numpy as np

from info import bits_recovered

# rate: bits/token (the slope the loop consumes). intercept: bits at T=0 of the fitted line -- the T*-unit
# must use (required_bits - intercept)/rate, NOT required_bits/rate, when the window is away from the origin.
# r2: linear-fit quality; bits_recovered(T) SATURATES at H(C), so a window that reaches saturation bends and
# r2 drops (and the rate under-estimates the low-budget slope) -- a materially-below-1 r2 flags a bad window.
RateFit = namedtuple("RateFit", ["rate", "intercept", "r2"])


def rate_bits_per_token(y_true, proba_by_budget, budgets, window=None):
    """y_true: (n,) held-out concept labels. proba_by_budget[T]: (n, K) predicted concept distributions using
    T emitted tokens. budgets: token budgets to fit over. window=(lo, hi): restrict to an early/PRE-SATURATION
    regime (required for validity, since the curve saturates at H(C)); default all budgets. Returns
    RateFit(rate, intercept, r2) -- the least-squares fit of bits_recovered(T) vs T. Check r2 (or that the
    rate is stable as you extend the window) to confirm the window is pre-saturation."""
    budgets = [T for T in budgets if (window is None or window[0] <= T <= window[1])]
    if len(budgets) < 2:
        raise ValueError("need >= 2 budgets in the window to fit a rate")
    missing = [T for T in budgets if T not in proba_by_budget]
    if missing:
        raise KeyError(f"budgets absent from proba_by_budget: {missing}")
    T = np.asarray(budgets, dtype=float)
    b = np.array([bits_recovered(y_true, proba_by_budget[t]) for t in budgets], dtype=float)
    slope, intercept = np.polyfit(T, b, 1)
    ss_res = float(np.sum((b - (slope * T + intercept)) ** 2))
    ss_tot = float(np.sum((b - b.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return RateFit(float(slope), float(intercept), float(r2))
