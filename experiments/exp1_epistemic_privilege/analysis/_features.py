"""Canonical R1 surface featurizer + one-vs-rest balanced-accuracy classifier.

These define reader R1 (the "symbol-counter") and the shared OvR scoring used across the
offline analyses, so every script that claims to use "the same R1" provably does. Import
from here rather than re-pasting:

    from _features import char_features, ovr_bacc

Note on the variants left out on purpose (do NOT route these through this module):
  - analyze_reevocation.char_feats uses RAW counts (no log1p) for a context-only letter control.
  - analyze_v3_curves.ovr_bacc uses a small-N-robust dynamic fold count + NaN guard for per-cut eval.
Both are intentionally different recipes; consolidating them would change their numbers.
"""
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


def char_features(texts, uni_top=120, big_top=500):
    """Literal symbol-counter: char unigram + char bigram log-count histogram, normalized."""
    uni = Counter(ch for t in texts for ch in t.lower())
    big = Counter((t.lower()[i], t.lower()[i + 1]) for t in texts for i in range(len(t) - 1))
    uv = {c: i for i, (c, _) in enumerate(uni.most_common(uni_top))}
    bv = {b: i for i, (b, _) in enumerate(big.most_common(big_top))}
    rows = []
    for t in texts:
        tl = t.lower()
        h = np.zeros(len(uv) + len(bv))
        for ch in tl:
            if ch in uv:
                h[uv[ch]] += 1
        for i in range(len(tl) - 1):
            b = (tl[i], tl[i + 1])
            if b in bv:
                h[len(uv) + bv[b]] += 1
        h = np.log1p(h)
        if h.sum() > 0:
            h = h / h.sum()
        rows.append(h)
    return np.array(rows)


def ovr_bacc(X, yb, repeats=5, seed0=0):
    """CV one-vs-rest balanced accuracy for a binary label vector yb. PCA-30 -> balanced logistic."""
    accs = []
    for seed in range(repeats):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed0 + seed)
        pred = np.zeros_like(yb)
        for tr, te in skf.split(X, yb):
            k = int(min(30, len(tr) - 1, X.shape[1]))
            p = PCA(n_components=k).fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(p.transform(X[tr]), yb[tr])
            pred[te] = clf.predict(p.transform(X[te]))
        accs.append(balanced_accuracy_score(yb, pred))
    return float(np.mean(accs))
