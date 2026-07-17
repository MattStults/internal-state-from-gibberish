"""exp2 distribution utilities.

kl_bits_topk: KL(p || q) in bits between two next-token distributions given as top-K {ids, logp}
(the form `gen_topk` saves). The mass outside each distribution's top-K (1 - sum of its top-K probs) is
spread UNIFORMLY over the remaining vocab. This is the distribution-access primitive the per-token leakage
rate is built from. Note: collapsing each tail to uniform can only LOSE divergence, so this UNDER-estimates
the true full-vocab KL -- treat it as a lower bound, consistent with the followup's best-decoder-lower-bound
framing of the whole budget.
"""
import math

import numpy as np


def kl_bits_topk(p_ids, p_logp, q_ids, q_logp, vocab_size):
    """p_ids/q_ids: top-K token ids; p_logp/q_logp: their natural-log log-probs (gen_topk form).
    Returns KL(p || q) in bits, with each distribution's off-top-K mass spread uniformly over the rest of the
    vocab. Reference q-mass is floored (a token p assigns mass to but q's tail can't cover -> finite, large)."""
    V = int(vocab_size)
    p_ids = np.asarray(p_ids); q_ids = np.asarray(q_ids)
    for ids in (p_ids, q_ids):                        # a duplicate id silently corrupts tail mass + count
        if len(np.unique(ids)) != len(ids):
            raise ValueError("top-K ids must be unique")
        if len(ids) and int(np.max(ids)) >= V:
            raise ValueError("top-K ids must be < vocab_size")
    p_top = np.exp(np.asarray(p_logp, dtype=float))
    q_top = np.exp(np.asarray(q_logp, dtype=float))
    pmap = dict(zip(p_ids.tolist(), p_top.tolist()))
    qmap = dict(zip(q_ids.tolist(), q_top.tolist()))
    p_unif = max(1.0 - p_top.sum(), 0.0) / (V - len(pmap)) if V > len(pmap) else 0.0
    q_unif = max(1.0 - q_top.sum(), 0.0) / (V - len(qmap)) if V > len(qmap) else 0.0
    floor = 1e-300
    kl = 0.0
    for x in set(pmap) | set(qmap):                 # explicit terms for any token in either top-K
        px = pmap.get(x, p_unif)
        if px > 0.0:
            qx = max(qmap.get(x, q_unif), floor)
            kl += px * math.log2(px / qx)
    n_bulk = V - len(set(pmap) | set(qmap))          # tokens in neither top-K: one closed-form term
    if n_bulk > 0 and p_unif > 0.0:
        kl += n_bulk * p_unif * math.log2(p_unif / max(q_unif, floor))
    return float(kl)
