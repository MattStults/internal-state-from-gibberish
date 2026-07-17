"""exp2 feature extractors: turn one stream's per-step gen_topk into a fixed-dim feature vector at budget T.

pooled_dist_features (distribution-access reader): the mean, over the first T generation steps, of the
probability mass the model put on each token in a fixed feature vocabulary. This is the token-distribution
analog of exp1's R1 char histogram -- it asks which tokens the injected concept tends to elevate, before
sampling. Off-vocab and off-top-K tokens contribute 0. The runner builds vocab_index from the corpus.

char_features_from_tokens (transcript reader): exp1's R1 "symbol-counter" (char uni+bigram histogram) scored
in exp2's bits currency. It is a pure TRANSCRIPT reader -- LESS model access than R_emb (needs only the
decoded string), so its recovered bits directly test the "distribution-only / no transcript-reading monitor
sees anything" claim that R_emb alone cannot settle. It reuses exp1's exact char_features so the reader is
PROVABLY the same one exp1 headlined (0.73-0.83 one-vs-rest bal-acc on the full streams).
"""
import os
import sys

import numpy as np

# exp1's canonical char featurizer -- import (not copy) so the transcript reader is provably identical to R1.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "exp1_epistemic_privilege", "analysis"))
from _features import char_features as _exp1_char_features  # noqa: E402


def char_features_from_tokens(streams, budget, tokenizer):
    """Decode each stream's first `budget` realized tokens to text, then exp1's char uni+bigram log-count
    histogram. streams: [{tokens: int ids, ...}]. tokenizer: anything with .decode(list_of_int_ids)->str.
    Returns (n_streams, d) dense matrix with a corpus-derived char vocab (same per-call behavior as exp1's R1).
    Fewer tokens => fewer characters => a thinner histogram, so this is budget-matched to the other readers."""
    texts = [tokenizer.decode([int(t) for t in s["tokens"][:budget]]) for s in streams]
    return _exp1_char_features(texts)


def pooled_dist_features(steps, budget, vocab_index, vocab_size=None):
    """steps: list of {'ids': top-K token ids, 'logp': their natural-log log-probs} for ONE stream, in order.
    budget: use the first `budget` steps. vocab_index: {token_id: feature column}. Returns a
    (len(vocab_index),) vector: the mean over the used steps of the probability the model put on each vocab
    token. off-vocab tokens are ignored.

    vocab_size: if None (default), a token absent from a step's top-K contributes 0 that step. If given, the
    absent token instead gets that step's UNIFORM-TAIL prob (1 - sum top-K)/(vocab_size - K) -- this closes the
    top-K boundary discontinuity (rank-64 vs rank-65 no longer swings prob->0), which the downstream
    StandardScaler would otherwise amplify into a presence/absence (truncation) feature. Prediction #2's
    distribution-vs-sampled gap must be confirmed under BOTH modes (the runner runs both)."""
    acc = np.zeros(len(vocab_index))
    used = steps[:budget]
    for s in used:
        ids = np.asarray(s["ids"])
        top = np.exp(np.asarray(s["logp"], dtype=float))
        if vocab_size is not None:
            tail = max(1.0 - float(top.sum()), 0.0) / max(int(vocab_size) - len(ids), 1)
            present = {int(t) for t in ids.tolist()}
            for tid, col in vocab_index.items():
                if int(tid) not in present:
                    acc[col] += tail
        for tid, p in zip(ids.tolist(), top.tolist()):
            col = vocab_index.get(int(tid))
            if col is not None:
                acc[col] += float(p)
    return acc / max(len(used), 1)


def embed_features(token_ids, budget, embed_matrix):
    """R_emb (the best token monitor): featurize the first `budget` REALIZED tokens by mean-pooling their model
    embeddings. token_ids: realized token id sequence for one stream. embed_matrix: (vocab, d) token->embedding
    (the model's input embedding / W_U row). Returns a (d,) dense vector. Same MODEL ACCESS as the distribution
    reader, so the dist-vs-this gap isolates the sampling step (not an access gap) -- unlike a surface char
    reader. A token id >= vocab or an empty budget is skipped."""
    d = embed_matrix.shape[1]
    ids = [int(t) for t in np.asarray(token_ids)[:budget] if 0 <= int(t) < embed_matrix.shape[0]]
    if not ids:
        return np.zeros(d)
    return np.asarray(embed_matrix)[ids].mean(axis=0)
