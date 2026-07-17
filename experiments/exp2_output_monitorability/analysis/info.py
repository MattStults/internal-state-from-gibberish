"""exp2 information currency.

bits_recovered(y_true, proba) = log2(K) - mean cross-entropy of a reader's predicted concept distribution,
in bits. This is the best-decoder LOWER BOUND on the mutual information I(concept; reader output): how many
bits about which-of-K concepts the reader extracts. It is the currency the token-budget law is denominated
in (bits recovered vs tokens). >0 = informative; ~0 = no better than the uniform prior; <0 = miscalibrated
(worse than the prior). No upper bound on I is obtainable at this sample size -- see proposed-followup.md.
"""
import numpy as np


def bits_recovered(y_true, proba):
    """y_true: (n,) int concept labels in [0, K); proba: (n, K) predicted concept distributions whose rows
    sum to 1 (enforced). Returns H(C) - mean cross-entropy (bits): the best-decoder lower bound on
    I(concept; output), since I = H(C) - H(C|X) and CE >= H(C|X). H(C) is the EMPIRICAL label entropy of
    y_true -- using a uniform-prior log2(K) would OVERSTATE the bound under class imbalance (a marginal-echo
    reader, zero per-instance info, would then read > 0 bits). Caller supplies held-out / calibrated
    probabilities so CE is a valid bound; this can't be enforced in-function. The 1e-12 floor caps a
    confidently-wrong example's penalty at ~39.9 bits (a regularizer)."""
    y = np.asarray(y_true, dtype=int)
    P = np.asarray(proba, dtype=float)
    if P.ndim != 2 or len(y) != len(P) or len(y) == 0:
        raise ValueError("proba must be (n, K) with n == len(y_true) > 0")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError("proba rows must sum to 1 (pass normalized probabilities, not logits/counts)")
    n, K = P.shape
    p_true = np.clip(P[np.arange(n), y], 1e-12, 1.0)         # mass the reader put on the true concept
    ce_bits = float(-np.mean(np.log2(p_true)))              # mean cross-entropy, bits
    pc = np.bincount(y, minlength=K) / n                    # empirical class prior P(C)
    H = float(-np.sum(pc[pc > 0] * np.log2(pc[pc > 0])))    # H(C), bits
    return H - ce_bits


def concept_bootstrap_ci(y, proba_by_mode, gaps=(), n_boot=2000, seed=0, alpha=0.05):
    """Concept-level bootstrap CI for bits (and dist-based gaps) at one budget. The unit of generalization is
    the CONCEPT, not the subsample seed: resample the unique concept labels WITH REPLACEMENT, gather all the
    (pooled) stream predictions for the resampled concepts with multiplicity, and recompute bits per mode +
    each gap on that resample. This is the honest uncertainty the 3-seed sd understated (2 dof over heavily
    overlapping subsamples; the real unit is the 12 concepts -- as exp1 already bootstrapped).
      y: (n,) pooled int labels. proba_by_mode: {mode: (n,K)} pooled held-out proba. gaps: iterable of
      (name, mode_a, mode_b) -> CI of bits[a]-bits[b]. Returns {'bits_ci': {mode:(lo,hi)}, 'gap_ci':
      {name:(lo,hi)}, 'n_boot': n_boot} at the (100*alpha/2, 100*(1-alpha/2)) percentiles."""
    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    idx_by_c = {int(c): np.where(y == c)[0] for c in classes}
    n, K = len(y), int(y.max()) + 1
    pc = np.bincount(y, minlength=K) / n                    # H(C) is a property of the 12-way TASK: hold it FIXED
    H_full = float(-np.sum(pc[pc > 0] * np.log2(pc[pc > 0])))  # at the full-task entropy and bootstrap only CE,
    #   else resampling concepts shrinks the entropy term and biases bits below the point estimate. H cancels
    #   in every gap, so gap CIs are unaffected either way.
    def _bits(P, idx, ys):
        p_true = np.clip(P[idx][np.arange(len(idx)), ys], 1e-12, 1.0)
        return H_full - float(-np.mean(np.log2(p_true)))
    rng = np.random.RandomState(seed)
    boot = {m: [] for m in proba_by_mode}
    gboot = {name: [] for name, _, _ in gaps}
    for _ in range(n_boot):
        cs = rng.choice(classes, size=len(classes), replace=True)
        idx = np.concatenate([idx_by_c[int(c)] for c in cs])
        ys = y[idx]
        bm = {m: _bits(P, idx, ys) for m, P in proba_by_mode.items()}
        for m in boot:
            boot[m].append(bm[m])
        for name, a, b in gaps:
            gboot[name].append(bm[a] - bm[b])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci = {m: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for m, v in boot.items()}
    gci = {name: (float(np.percentile(v, lo)), float(np.percentile(v, hi))) for name, v in gboot.items()}
    return {"bits_ci": ci, "gap_ci": gci, "n_boot": n_boot}
