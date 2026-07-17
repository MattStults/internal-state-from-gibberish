"""Cross-primer TRANSFER decode (1.5B): does the evoked `dist` signal carry the CONCEPT or the WORDING?

The deflationary reading of exp3 ("prompt-fingerprint"): the decoder recovers *which of 12 distinct
paragraphs* sits in context (wording/register residue), not a carried concept state -- which would also
predict named~evoked and secret_word~0. The within-arm invariance check can't discriminate (any 12
distinct paragraphs fingerprint). This can: train the dist reader on `evoked` streams, test on
`evoked_alt` (independent paraphrases of the same 12 concepts; different wording, same states).

  - Concept-level signal TRANSFERS across paraphrases  -> transfer bits ~ within-arm bits.
  - Wording-level fingerprint does NOT                 -> transfer bits ~ 0.

Reports both directions (evoked->alt, alt->evoked) + each arm's within-arm CV bits on the same
subsamples, seeds 0/1/2 -> mean+/-sd. Uses the exp2 dist pipeline (same capacity grid, inner-CV
capacity selection on TRAIN ONLY; the test arm is never touched during fitting -- a strictly
held-out paraphrase set, stronger than CV). CPU-only, run cores-capped. Writes
reports/transfer_decode.json.
"""
import json
import os
import sys

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "exp2_output_monitorability", "analysis"))
import run_budget as RB                                    # noqa: E402
from loader import load_ab_streams                         # noqa: E402
from prep import common_n_subsample, build_vocab_index     # noqa: E402
from reader import _channel_pipeline                       # noqa: E402
from info import bits_recovered                            # noqa: E402
from reader import best_reader_proba_by_budget             # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
T = 12
SEEDS = (0, 1, 2)
N_PER_CLASS = 24
V = 151936


def _vocab(bundle, idx):
    """The dist featurizer projects per-step top-64 mass onto a learned token index. Built from the
    given streams only -- for transfer, TRAIN only, so the test arm never shapes the feature space."""
    ss = [bundle["streams"][i] for i in idx]
    ids = ([int(t) for s in ss for st in s["gen_topk"] for t in st["ids"]] +
           [int(t) for s in ss for t in s["tokens"][:T]])
    return build_vocab_index(ids, max_vocab=300, min_count=2)


def _dist_features(bundle, idx, vocab):
    ss = [bundle["streams"][i] for i in idx]
    return np.asarray(RB._features(ss, T, vocab, "dist", vocab_size=V), dtype=float)


def _subsample(bundle, seed, min_len=T):
    keep = [i for i, s in enumerate(bundle["streams"]) if len(s["gen_topk"]) >= min_len]
    y = np.array([bundle["streams"][i]["concept_idx"] for i in keep])
    sub = common_n_subsample(y, n=N_PER_CLASS, seed=seed)
    idx = [keep[j] for j in sub]
    return idx, y[sub]


def _confusion_mi_bits(y_true, y_pred, k=12):
    """Label-free association: I(y; y_hat) of the hard-label confusion matrix, in bits. CE-based
    bits_recovered punishes cross-domain miscalibration; plug-in confusion MI is immune to that, at
    the cost of only seeing the argmax. Small-n plug-in MI biases upward -> report with the shuffle null.

    INTERPRETATION CAUTION (transfer cells): confusion MI measures SEPARABILITY, not concept content.
    Any 12 distinct inducing texts produce distinguishable streams, and a fixed decoder projects each
    to a consistent (possibly wrong) label -- so high transfer MI with chance-level accuracy is exactly
    the wording-fingerprint signature, NOT evidence that the concept transfers. Concept-ALIGNED transfer
    is what top1_acc / bits_ce measure (the decoder must assign the RIGHT label)."""
    c = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        c[t, p] += 1
    p = c / c.sum()
    pi, pj = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log2(p / (pi @ pj)), 0.0)
    return float(terms.sum())


def transfer_bits(train_b, test_b, seed):
    """Fit the exp2 dist best-decoder on the TRAIN arm (capacity by inner CV on train only; feature
    vocab from train only), then score the test arm subsample. No test contamination. Returns CE-based
    bits (the currency; calibration-sensitive), top-1 accuracy, and confusion-MI bits (calibration-free)."""
    tr_idx, y_tr = _subsample(train_b, seed)
    te_idx, y_te = _subsample(test_b, seed)
    vocab = _vocab(train_b, tr_idx)
    X_tr, X_te = _dist_features(train_b, tr_idx, vocab), _dist_features(test_b, te_idx, vocab)
    n_train_eff = int(len(y_tr) * 2 / 3)                   # inner 3-fold train size, matches reader.py
    pipe, grid = _channel_pipeline("dense", X_tr.shape[1], n_train_eff, seed=seed)
    gs = GridSearchCV(pipe, grid, scoring="neg_log_loss", n_jobs=1,
                      cv=StratifiedKFold(3, shuffle=True, random_state=seed))
    gs.fit(X_tr, y_tr)
    P = gs.predict_proba(X_te)
    yhat = P.argmax(1)
    # label-shuffle null for the plug-in MI's small-n upward bias (same n, same marginals)
    rng = np.random.default_rng(seed)
    null = [_confusion_mi_bits(rng.permutation(y_te), yhat) for _ in range(200)]
    return {"bits_ce": float(bits_recovered(y_te, P)),
            "top1_acc": float((yhat == y_te).mean()), "chance_acc": 1 / 12,
            "confusion_mi_bits": _confusion_mi_bits(y_te, yhat),
            "confusion_mi_shuffle_null_mean": float(np.mean(null)),
            "confusion_mi_shuffle_null_p95": float(np.quantile(null, 0.95))}


def within_bits(bundle, seed):
    """Within-arm nested-CV metrics on the same subsample -- the apples-to-apples COMPARATOR for the
    transfer cells: same bits_ce / top1 / confusion-MI trio, so 'what fraction transfers' is a
    like-for-like ratio on each scale (in particular MI-excess(transfer) / MI-excess(within))."""
    idx, y = _subsample(bundle, seed)
    X = _dist_features(bundle, idx, _vocab(bundle, idx))
    P = best_reader_proba_by_budget({T: X}, y, [T], kind="dense", folds=5, seed=seed, n_jobs=1)[T]
    yhat = P.argmax(1)
    rng = np.random.default_rng(seed)
    null = [_confusion_mi_bits(rng.permutation(y), yhat) for _ in range(200)]
    return {"bits_ce": float(bits_recovered(y, P)),
            "top1_acc": float((yhat == y).mean()), "chance_acc": 1 / 12,
            "confusion_mi_bits": _confusion_mi_bits(y, yhat),
            "confusion_mi_shuffle_null_mean": float(np.mean(null)),
            "confusion_mi_shuffle_null_p95": float(np.quantile(null, 0.95))}


def main():
    ev = load_ab_streams(os.path.join(REPO, "runs/_ind/qwen2.5-1.5b/data/qwen2.5-1.5b-evoked.pt"))
    al = load_ab_streams(os.path.join(REPO, "runs/_ind/qwen2.5-1.5b/data/qwen2.5-1.5b-evoked_alt.pt"))
    assert ev["concepts"] == al["concepts"], "concept order differs between arms -- labels not comparable"

    out = {"model": "qwen2.5-1.5b", "budget_T": T, "n_per_class": N_PER_CLASS, "seeds": list(SEEDS),
           "question": "concept-state (transfers across paraphrase) vs prompt-wording fingerprint (does not)"}
    cells = (("within_evoked", lambda s: within_bits(ev, s)),
             ("within_evoked_alt", lambda s: within_bits(al, s)),
             ("transfer_evoked_to_alt", lambda s: transfer_bits(ev, al, s)),
             ("transfer_alt_to_evoked", lambda s: transfer_bits(al, ev, s)))
    for name, fn in cells:
        per = [fn(s) for s in SEEDS]
        cell = {k: {"mean": float(np.mean([p[k] for p in per])), "sd": float(np.std([p[k] for p in per])),
                    "per_seed": [p[k] for p in per]}
                for k in per[0]}
        out[name] = cell
        print(f"{name:24} bits_ce={cell['bits_ce']['mean']:+.3f}  "
              f"top1={cell['top1_acc']['mean']:.3f} (chance 0.083)  "
              f"confusionMI={cell['confusion_mi_bits']['mean']:.3f} "
              f"(shuffle null {cell['confusion_mi_shuffle_null_mean']['mean']:.3f}, "
              f"p95 {cell['confusion_mi_shuffle_null_p95']['mean']:.3f})", flush=True)
    # separability retention across paraphrase, on the MI scale. NOT a concept-transfer fraction --
    # see _confusion_mi_bits caution: transfer MI credits consistent-but-mislabeled separation, which a
    # pure wording fingerprint also produces. Concept-aligned transfer = the bits_ce / top1_acc rows.
    ex = {n: out[n]["confusion_mi_bits"]["mean"] - out[n]["confusion_mi_shuffle_null_mean"]["mean"]
          for n, _ in cells}
    out["mi_excess_bits"] = ex
    within = np.mean([ex["within_evoked"], ex["within_evoked_alt"]])
    transfer = np.mean([ex["transfer_evoked_to_alt"], ex["transfer_alt_to_evoked"]])
    out["separability_retention_across_paraphrase"] = float(transfer / within) if within > 0 else None
    out["interpretation_note"] = ("transfer confusion-MI = label-free SEPARABILITY through the trained "
                                  "decoder; it cannot distinguish concept from wording (any 12 distinct "
                                  "texts separate). Concept-aligned transfer is bits_ce/top1_acc: ~0 "
                                  "calibrated bits, top-1 barely above chance.")
    print(f"MI-excess within={within:.3f}  transfer={transfer:.3f}  "
          f"separability retention={out['separability_retention_across_paraphrase']:.2f} "
          f"(NOT concept transfer -- see interpretation_note)", flush=True)

    dst = os.path.join(HERE, "..", "reports", "transfer_decode.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.abspath(dst)}", flush=True)


if __name__ == "__main__":
    main()
