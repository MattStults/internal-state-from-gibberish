"""exp2 data-prep utilities (label-blind): common-N subsampling and feature-vocabulary building.

These run in the runner BEFORE the reader, so cross-scale/cross-condition comparisons hold capacity + N
fixed: common_n_subsample equalizes streams per concept (more streams => tighter decoder => more recovered
bits), and the vocab must be built label-blind (never by class-discriminativeness) or the reader leaks.
"""
import numpy as np


def common_n_subsample(y, n=None, seed=0):
    """y: (m,) labels. Returns sorted indices selecting exactly n streams per class (n defaults to the
    smallest class count), sampled without replacement under `seed`. Raises if any class has < n. Use its
    output to index features + labels so every concept contributes the same N to the reader."""
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("y is empty")
    classes = np.unique(y)
    if n is None:
        n = int(min((y == c).sum() for c in classes))
    rng = np.random.RandomState(seed)
    picked = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        if len(idx_c) < n:
            raise ValueError(f"class {c} has {len(idx_c)} streams < n={n}")
        picked.append(rng.choice(idx_c, size=n, replace=False))
    return np.sort(np.concatenate(picked))


def build_vocab_index(all_ids, max_vocab=None, min_count=1):
    """all_ids: a flat sequence of every top-K token id across the corpus (repetition = frequency).
    Returns {token_id: column} for the max_vocab most FREQUENT ids with count >= min_count (label-blind --
    never by class), columns 0..M-1 in descending-frequency order (ties by id asc).

    Only the corpus-MARGINAL frequency is pooled -- never class signal -- so sharing one vocab across the
    reader's CV folds does NOT leak class information into the held-out bits (a benign, intentional exception
    to per-fold refitting). And one shared vocab is REQUIRED for the distribution-vs-sampled (Prediction #2)
    and cross-scale comparisons to be fair -- differing vocabs would confound them. Pass max_vocab at real
    (~150k) vocab scale to avoid a huge sparse feature matrix; min_count drops rare noise columns. NOTE: token
    ids are not comparable across tokenizers -- keep per-tokenizer vocabs (Qwen2.5 vs Qwen3)."""
    ids = np.asarray(list(all_ids), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("all_ids is empty")
    uniq, counts = np.unique(ids, return_counts=True)
    keep = counts >= min_count
    uniq, counts = uniq[keep], counts[keep]
    order = np.lexsort((uniq, -counts))                 # primary: -count (freq desc); tie-break: id asc
    if max_vocab is not None:
        order = order[:max_vocab]
    return {int(uniq[i]): col for col, i in enumerate(order)}


def group_labels(y, n_groups, seed=0):
    """Remap the concepts in y into n_groups contiguous groups (0..n_groups-1), ~balanced, for the bits-ladder
    (n_groups = 2^k is a k-bit recovery task). Every original concept maps to exactly one group; deterministic
    under seed (vary seed to average over random groupings). Raises if n_groups > the number of concepts."""
    y = np.asarray(y)
    classes = np.unique(y)
    if n_groups > len(classes):
        raise ValueError(f"n_groups {n_groups} > number of concepts {len(classes)}")
    shuffled = np.random.RandomState(seed).permutation(classes)
    group_of = {}
    for gi, chunk in enumerate(np.array_split(shuffled, n_groups)):
        for c in chunk:
            group_of[int(c)] = gi
    return np.array([group_of[int(v)] for v in y])
