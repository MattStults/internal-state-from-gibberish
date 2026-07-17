"""FREE DE-RISK (no GPU, existing data): does the SAMPLE COMPLEXITY of the surface reader grow with model
size? Hypothesis: the concept is in the model's state at all scales, but each emitted token leaks
LESS concept-signal in bigger models -- so a token-only reader needs MORE samples to recover it as the
model grows ("the gap grows with scale").

Test: trace R1's recovery accuracy as a function of the number of streams/concept used, at the STRONG
injection, for qwen2.5-1.5b/3b/7b -- all from data we already have. If the bigger model's learning curve
sits lower / rises slower (needs more streams for the same accuracy), the hypothesis holds and the
sample-complexity reframe is justified before capturing anything new.

R1 is replicated EXACTLY from analyze_v2.py (char uni+bigram histogram -> PCA-30 -> balanced logistic,
one-vs-rest balanced accuracy, chance=0.5). Run:  .venv/bin/python analysis/derisk_learning_curve.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from _features import char_features, ovr_bacc   # canonical R1 featurizer + OvR balanced accuracy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
MODELS = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
NS = [5, 8, 12, 16, 20, 25, 30]   # streams/concept to sub-sample
DRAWS = 6                          # random sub-samples per N (for a mean +/- spread)
THRESHOLD = 0.70                   # "samples to reach this accuracy" readout


def mean_ovr(texts, y, K):
    """R1 metric: mean over the K concepts of the one-vs-rest balanced accuracy on char features.
    repeats=3 (trimmed from analyze_v2's 5 for speed across the learning-curve sub-samples)."""
    X = char_features(texts)
    return float(np.mean([ovr_bacc(X, (y == c).astype(int), repeats=3) for c in range(K)]))


def learning_curve(slug, rng):
    d = torch.load(REPO / "runs" / slug / "data" / "covert_collect.pt", map_location="cpu", weights_only=False)
    S = d["streams"]; concepts = d["concepts"]; K = len(concepts)
    strong = max(r["strength"] for r in S)
    by_c = {c: [r for r in S if r["concept_idx"] == c and r["strength"] == strong and r["accepted"]]
            for c in range(K)}
    avail = min(len(v) for v in by_c.values())
    curve = {}
    for N in NS:
        if N > avail:
            break
        accs = []
        for _ in range(DRAWS):
            rows = [r for c in range(K) for r in [by_c[c][i] for i in rng.choice(len(by_c[c]), N, replace=False)]]
            y = np.array([r["concept_idx"] for r in rows])
            accs.append(mean_ovr([r["text"] for r in rows], y, K))
        curve[N] = (float(np.mean(accs)), float(np.std(accs)))
    return strong, avail, curve


def samples_to_threshold(curve):
    """Smallest N whose mean accuracy >= THRESHOLD (linear interp), else None (never reached)."""
    pts = sorted(curve.items())
    for (n0, (a0, _)), (n1, (a1, _)) in zip(pts, pts[1:]):
        if a0 < THRESHOLD <= a1:
            return round(n0 + (n1 - n0) * (THRESHOLD - a0) / (a1 - a0), 1)
    if pts and pts[0][1][0] >= THRESHOLD:
        return pts[0][0]
    return None


def main():
    rng = np.random.default_rng(0)
    out = {}
    plt.figure(figsize=(7, 5))
    for slug in MODELS:
        strong, avail, curve = learning_curve(slug, rng)
        out[slug] = dict(strong=strong, avail=avail, curve=curve, n_to_70=samples_to_threshold(curve))
        ns = sorted(curve)
        means = [curve[n][0] for n in ns]; stds = [curve[n][1] for n in ns]
        plt.errorbar(ns, means, yerr=stds, marker="o", capsize=3, label=f"{slug} (s{strong})")
        print(f"\n{slug}: strong=s{strong}, avail={avail}/concept")
        for n in ns:
            print(f"  N={n:>2}/concept: R1 bacc = {curve[n][0]:.3f} +/- {curve[n][1]:.3f}")
        print(f"  -> streams/concept to reach {THRESHOLD}: {out[slug]['n_to_70']}")
    plt.axhline(0.5, ls=":", c="gray", label="chance (0.5)")
    plt.axhline(THRESHOLD, ls="--", c="red", alpha=0.5)
    plt.xlabel("streams per concept (samples)"); plt.ylabel("R1 mean one-vs-rest balanced accuracy")
    plt.title("Sample complexity of the surface reader vs model size\n(does the bigger model need more samples?)")
    plt.legend(); plt.tight_layout()
    fig = REPO / "runs" / "derisk_learning_curve.png"
    plt.savefig(fig, dpi=130)
    with open(REPO / "runs" / "derisk_learning_curve.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {fig}")
    print("\nVERDICT key: if n_to_70 grows 1.5b -> 3b -> 7b (or the bigger model's curve sits lower / never"
          " reaches 0.70), sample complexity grows with scale -> the reframe is justified.")


if __name__ == "__main__":
    main()
