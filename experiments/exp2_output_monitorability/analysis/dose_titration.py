"""Dose titration (1.5B): is the injected/natural channel split a REGIME effect or a DOSE effect?

The two-regime reading (injected concepts mark the transcript, natural ones don't) is confounded:
injection is also ~6x STRONGER (dist@T12 2.64 vs 0.45) and re-applied every token. If decoder bits
scale superlinearly in effect size, a weak-enough injection would ALSO leave no readable transcript
trace -- making "regime" a proxy for "dose". exp1's capture grid already contains the titration:
generation-only injection at strengths {0, 40, 60} on the same 12 concepts, with per-step top-64
logprobs AND realized tokens, through the same word-free filter.

For each strength: dist@T12 (the fast channel), char@T12, and char at the full stream (the
accumulating transcript channel), in the exp2 bits currency (per-channel best-decoder nested-CV,
seeds 0/1/2 -> mean+/-sd; common-N per class within each strength).

Readout:
  - strength 0  = negative control (everything ~0 or the pipeline is broken).
  - strength 40 = the medium dose. If dist@T12 lands near the NATURAL level (~0.45) the dose is
    matched to a realistic contextual cause; then char@full decides: recovers -> the transcript
    fingerprint survives at natural-strength injection (REGIME story holds); floors -> the
    transcript channel needs a strong dose (DOSE story; the two-regime headline must be reworded).
  - strength 60 = should approximate the exp2 strong-dose numbers (different capture batch).

CPU-only; run cores-capped. Writes reports/dose_titration.json.
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
SEEDS = (0, 1, 2)
FULL = 100000


def _bits(streams, tok, mode, budget, n, min_len):
    per = []
    for seed in SEEDS:
        pool = [s for s in streams if len(s["gen_topk"]) >= min_len]
        y = np.array([s["concept_idx"] for s in pool])
        idx = common_n_subsample(y, n=n, seed=seed)
        ss = [pool[i] for i in idx]
        y = y[idx]
        ids = ([int(t) for s in ss for st in s["gen_topk"] for t in st["ids"]] +
               [int(t) for s in ss for t in s["tokens"][:budget]])
        vocab = build_vocab_index(ids, max_vocab=300, min_count=2)
        vs = {"dist": 151936, "char": None}[mode]
        X = RB._features(ss, budget, vocab, mode, vocab_size=vs, embed=None, tokenizer=tok)
        P = best_reader_proba_by_budget({budget: X}, y, [budget], kind=RB.KIND[mode], folds=5,
                                        seed=seed, n_jobs=1)[budget]
        per.append(float(bits_recovered(y, P)))
    return {"mean": float(np.mean(per)), "sd": float(np.std(per)), "per_seed": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="qwen2.5-1.5b",
                    help="model slug; reads runs/<slug>/data/covert_collect.pt, writes reports/dose_titration_<slug>.json"
                         " (the original 1.5B run kept its unsuffixed dose_titration.json)")
    args = ap.parse_args()
    capture = os.path.join(REPO, "runs", args.slug, "data", "covert_collect.pt")
    d = torch.load(capture, map_location="cpu", weights_only=False)
    tok = RB._load_tokenizer(d["model"])
    out = {"model": d["model"], "capture": os.path.relpath(capture, REPO), "inject": d.get("inject"),
           "strengths": d["strengths"], "seeds": list(SEEDS),
           "question": "regime vs dose: does a weak (natural-level) injection still mark the transcript?",
           "cells": {}}
    for lvl in sorted({s["strength"] for s in d["streams"]}):
        acc = [s for s in d["streams"] if s["strength"] == lvl and s["accepted"]]
        cnt = np.bincount([s["concept_idx"] for s in acc], minlength=12)
        n12 = int(min(24, min(np.bincount([s["concept_idx"] for s in acc if len(s["gen_topk"]) >= 12], minlength=12))))
        nfull = int(min(24, min(np.bincount([s["concept_idx"] for s in acc if len(s["gen_topk"]) >= 8], minlength=12))))
        if min(n12, nfull) < 5:                        # stream-level CV needs >= folds(5) per class
            out["cells"][str(lvl)] = {"n_accepted": len(acc), "min_per_class": int(cnt.min()),
                                      "skipped": f"min common-N {min(n12, nfull)} < 5 CV folds"}
            print(f"strength {lvl:>2}: SKIPPED (n={min(n12, nfull)}/cls < 5 folds)", flush=True)
            continue
        L = [len(s["gen_topk"]) for s in acc]
        cell = {"n_accepted": len(acc), "min_per_class": int(cnt.min()),
                "len_median": int(np.median(L)), "len_p25": int(np.percentile(L, 25)),
                "len_p75": int(np.percentile(L, 75)),
                "n_T12": n12, "n_full": nfull,
                "dist_T12": _bits(acc, tok, "dist", 12, n12, 12),
                "char_T12": _bits(acc, tok, "char", 12, n12, 12),
                "char_full": _bits(acc, tok, "char", FULL, nfull, 8)}
        out["cells"][str(lvl)] = cell
        print(f"strength {lvl:>2}: n={len(acc)} len_med={cell['len_median']:>3} | "
              f"dist@12={cell['dist_T12']['mean']:+.3f}±{cell['dist_T12']['sd']:.3f} "
              f"char@12={cell['char_T12']['mean']:+.3f}±{cell['char_T12']['sd']:.3f} "
              f"char@full={cell['char_full']['mean']:+.3f}±{cell['char_full']['sd']:.3f}", flush=True)

    name = "dose_titration.json" if args.slug == "qwen2.5-1.5b" else f"dose_titration_{args.slug}.json"
    dst = os.path.join(HERE, "..", "reports", name)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.abspath(dst)}", flush=True)


if __name__ == "__main__":
    main()
