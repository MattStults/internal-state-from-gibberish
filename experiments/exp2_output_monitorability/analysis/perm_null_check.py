"""Label-permutation null for the s0 (uninjected control) bits cells.

Motivation (science review of the 14B prereg): the 7B s0 control reads dist@12 = -0.302 +/- 0.077 --
strongly negative where the pipeline is supposed to floor near 0. Hypothesized mechanism: the nested-CV
capacity grid has no predict-the-prior arm, so on structured-but-label-irrelevant features the inner CV
picks overfit capacity that generalizes WORSE than uniform => a negative-bits miscalibration mode under a
true null (conservative for positive cells, but it breaks any |bits|<0.1 sanity gate and makes small
readings undecidable). Test: shuffle concept labels within the s0 pool and recompute the identical
dist@T12 cell N times. If the shuffled-label null reproduces the observed negative band, the mode is
confirmed benign decoder miscalibration, NOT leakage or a pipeline bug; the prereg gate becomes
"observed s0 must sit INSIDE the permutation null band" and positive cells must exceed the null p95.

Local, cores-capped, ~15 min. Writes reports/perm_null_check.json.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_budget as RB                                    # noqa: E402
from prep import common_n_subsample, build_vocab_index     # noqa: E402
from reader import best_reader_proba_by_budget             # noqa: E402
from info import bits_recovered                            # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
T = 12


def cell_bits(streams, y, tok, seed, mode="dist", budget=T):
    """One reader fit on the given labels -- the exact dose_titration protocol, one seed.
    mode/budget select the cell: dist@T12 (default) or char@full (budget=100000)."""
    idx = common_n_subsample(y, n=24, seed=seed)
    ss = [streams[i] for i in idx]
    yy = y[idx]
    ids = ([int(t) for s in ss for st in s["gen_topk"] for t in st["ids"]] +
           [int(t) for s in ss for t in s["tokens"][:budget]])
    vocab = build_vocab_index(ids, max_vocab=300, min_count=2)
    vs = {"dist": 151936, "char": None}[mode]
    X = RB._features(ss, budget, vocab, mode, vocab_size=vs, embed=None, tokenizer=tok)
    P = best_reader_proba_by_budget({budget: X}, yy, [budget], kind=RB.KIND[mode], folds=5,
                                    seed=seed, n_jobs=1)[budget]
    return float(bits_recovered(yy, P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="qwen2.5-7b")
    ap.add_argument("--n-shuffles", type=int, default=20)
    ap.add_argument("--strength", type=int, default=0, help="pool to test (0 = the null control)")
    ap.add_argument("--mode", default="dist", choices=["dist", "char"])
    ap.add_argument("--budget", type=int, default=T, help="token budget (100000 = full stream)")
    args = ap.parse_args()
    d = torch.load(os.path.join(REPO, "runs", args.slug, "data", "covert_collect.pt"),
                   map_location="cpu", weights_only=False)
    tok = RB._load_tokenizer(d["model"])
    min_len = args.budget if args.budget <= 128 else 8
    pool = [s for s in d["streams"] if s["strength"] == args.strength and s["accepted"]
            and len(s["gen_topk"]) >= min_len]
    y = np.array([s["concept_idx"] for s in pool])
    print(f"{args.slug} s{args.strength} {args.mode}@{args.budget}: pool={len(pool)}", flush=True)

    observed = cell_bits(pool, y, tok, seed=0, mode=args.mode, budget=args.budget)
    print(f"observed (true labels, seed 0) = {observed:+.3f}", flush=True)
    null = []
    for k in range(args.n_shuffles):
        rng = np.random.default_rng(1000 + k)
        null.append(cell_bits(pool, rng.permutation(y), tok, seed=0, mode=args.mode, budget=args.budget))
        print(f"  shuffle {k}: {null[-1]:+.3f}", flush=True)

    out = {"slug": args.slug, "strength": args.strength, "pool": len(pool),
           "mode": args.mode, "budget": args.budget, "T": T,
           "observed_true_labels_seed0": observed, "null_per_shuffle": null,
           "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
           "null_p5": float(np.quantile(null, 0.05)), "null_p95": float(np.quantile(null, 0.95)),
           "observed_inside_null_band": bool(np.quantile(null, 0.05) <= observed <= np.quantile(null, 0.95))}
    suffix = "" if (args.mode == "dist" and args.budget == T) else f"_{args.mode}{args.budget}"
    dst = os.path.join(HERE, "..", "reports", f"perm_null_check_{args.slug}_s{args.strength}{suffix}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"null mean={out['null_mean']:+.3f} sd={out['null_sd']:.3f} "
          f"band=[{out['null_p5']:+.3f}, {out['null_p95']:+.3f}] "
          f"observed inside band: {out['observed_inside_null_band']}", flush=True)
    print(f"wrote {os.path.abspath(dst)}", flush=True)


if __name__ == "__main__":
    main()
