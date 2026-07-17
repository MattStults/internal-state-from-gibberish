"""Label-permutation nulls for the E1/E3 confound-suite POSITIVE cells (review item M2).

The confound-closing prereg's shared validity gates require every positive verdict cell to exceed
its own-pool >=20-shuffle label-permutation null p95. The E1/E3 titration run
(`dose_titration_confound-e{1,3}.json`) reported the s0 controls but never ran the shuffled-label
nulls for the positive dist@T12 cells. This closes that gap, post-data, using the identical reader
path as `dose_titration.py` / `perm_null_check.py` (common-N 24/class, nested-CV best decoder,
seed 0 for the observed cell and for every shuffle -- only the labels change).

Cells: E1 (gen-only weak-dose, runs/confound-e1) dist@T12 at s12 and s20;
       E3 (prompt-only, runs/confound-e3) dist@T12 at s40 and s60.

CPU-only, cores-capped. Writes reports/perm_null_confound_e1_e3.json.
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_budget as RB                                    # noqa: E402
from perm_null_check import cell_bits                      # noqa: E402  the exact one-cell protocol

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
T = 12
N_SHUFFLES = 20
CELLS = [                                # (run dir, strength) -- the positive dist@T12 cells
    ("confound-e1", 12),
    ("confound-e1", 20),
    ("confound-e3", 40),
    ("confound-e3", 60),
]


def main():
    out = {"question": "do the E1/E3 positive dist@T12 cells exceed their own-pool "
                       ">=20-shuffle label-permutation null p95 (confound prereg shared gate)?",
           "protocol": "perm_null_check.cell_bits (dose_titration reader path), mode=dist, "
                       f"budget=T{T}, common-N 24/class, seed 0, {N_SHUFFLES} shuffles/cell",
           "cells": {}}
    tok = None
    for run, lvl in CELLS:
        d = torch.load(os.path.join(REPO, "runs", run, "data", "covert_collect.pt"),
                       map_location="cpu", weights_only=False)
        if tok is None:
            tok = RB._load_tokenizer(d["model"])
        pool = [s for s in d["streams"] if s["strength"] == lvl and s["accepted"]
                and len(s["gen_topk"]) >= T]
        y = np.array([s["concept_idx"] for s in pool])
        print(f"{run} s{lvl} dist@T{T}: pool={len(pool)}", flush=True)
        observed = cell_bits(pool, y, tok, seed=0, mode="dist", budget=T)
        print(f"  observed (true labels, seed 0) = {observed:+.3f}", flush=True)
        null = []
        for k in range(N_SHUFFLES):
            rng = np.random.default_rng(1000 + k)
            null.append(cell_bits(pool, rng.permutation(y), tok, seed=0, mode="dist", budget=T))
            print(f"  shuffle {k}: {null[-1]:+.3f}", flush=True)
        p95 = float(np.quantile(null, 0.95))
        out["cells"][f"{run}_s{lvl}"] = {
            "run": run, "strength": lvl, "pool": len(pool), "mode": "dist", "budget": T,
            "observed_true_labels_seed0": observed, "null_per_shuffle": null,
            "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "null_p95": p95, "exceeds_null_p95": bool(observed > p95)}
        print(f"  null mean={np.mean(null):+.3f} sd={np.std(null):.3f} p95={p95:+.3f} "
              f"exceeds: {observed > p95}", flush=True)

    dst = os.path.join(HERE, "..", "reports", "perm_null_confound_e1_e3.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {os.path.abspath(dst)}", flush=True)


if __name__ == "__main__":
    main()
