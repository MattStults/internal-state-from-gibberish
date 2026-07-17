"""Test B (triangulation): the RAW concept tilt in the generated gibberish, per condition.

Almost every reader (R1, R2, R3's B-C, re-evocation) reads the CLEAN stream, so the injection METHOD can
affect them only through the generated gibberish itself. This measures that directly: per concept, the
character-frequency shift between its strong-injected streams and the s0 (uninjected) streams -- the
mechanistic source of the leak. Run it on each `--inject` condition (all-position vs generation-only,
same dose) to see whether all-position simply produces MORE concept-tilted gibberish.

Metric: total-variation distance between a concept's injected char distribution and the s0 char
distribution (0 = no tilt = no leak; higher = more letters shifted by the injection). Plus the single
most-excess character (the 'o'-for-ocean signature).

Run:  .venv/bin/python analysis/char_tilt.py runs/qwen2.5-1.5b-all/data/covert_collect.pt \\
                                              runs/qwen2.5-1.5b/data/covert_collect.pt
"""
import sys
from collections import Counter

import numpy as np
import torch


def char_dist(texts, vocab):
    c = Counter(ch for t in texts for ch in t.lower() if ch in vocab)
    tot = sum(c.values()) or 1
    return np.array([c[ch] / tot for ch in vocab])


def measure(pt):
    d = torch.load(pt, map_location="cpu", weights_only=False)
    S, concepts = d["streams"], d["concepts"]
    strong, inj = max(d["strengths"]), d.get("inject", "?")
    vocab = sorted({ch for r in S if r["accepted"] for ch in r["text"].lower() if ch.isalpha()})
    base = char_dist([r["text"] for r in S if r["strength"] == 0 and r["accepted"]], vocab)
    rows = []
    for ci, c in enumerate(concepts):
        texts = [r["text"] for r in S if r["concept_idx"] == ci and r["strength"] == strong and r["accepted"]]
        p = char_dist(texts, vocab)
        tv = 0.5 * float(np.sum(np.abs(p - base)))      # tilt = TV distance of the char histogram from s0
        j = int(np.argmax(p - base))                    # the single most-excess character
        rows.append((c, tv, vocab[j], float(p[j]), float(base[j]), len(texts)))
    return dict(inject=inj, strong=strong, rows=rows, mean_tilt=float(np.mean([r[1] for r in rows])))


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: char_tilt.py <run.pt> [other.pt]   (pass the two --inject conditions to compare)")
        raise SystemExit(2)
    results = {pt: measure(pt) for pt in paths}
    for pt, r in results.items():
        print(f"\n=== {pt}  (inject={r['inject']}, s{r['strong']})  mean char-tilt (TV from s0) = {r['mean_tilt']:.3f} ===")
        for c, tv, ch, pi, b0, n in sorted(r["rows"], key=lambda x: -x[1]):
            print(f"  {c:12s} tilt={tv:.3f}  top '{ch}' {100*pi:.1f}% vs s0 {100*b0:.1f}%  (n={n})")
    if len(paths) == 2:
        a, b = (results[p] for p in paths)
        print(f"\n=== COMPARISON: inject={a['inject']} vs inject={b['inject']} ===")
        print(f"  mean char-tilt: {a['inject']}={a['mean_tilt']:.3f}  vs  {b['inject']}={b['mean_tilt']:.3f}  "
              f"(ratio {a['mean_tilt']/(b['mean_tilt']+1e-9):.2f}x)")
        print("  -> if all-position's tilt is much larger, the reader difference is a real GENERATION effect "
              "(steering the prompt makes the gibberish more concept-tilted), not a code/measurement bug.")


if __name__ == "__main__":
    main()
