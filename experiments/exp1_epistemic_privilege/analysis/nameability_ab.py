"""Arm-A nameability covariate for the all-position vs generation-only A/B.

The injection METHOD changes the readers (R1 etc.) only through the generated gibberish. But before reading
"R1 collapsed" as a mechanism finding, we must know whether the concept was even still INSTANTIATED under
gen-only. Arm A reads the concept with the injection live, so its own-concept rank/logp is exactly that
"is the concept present & nameable right now" covariate -- independent of whether it leaks to the surface.

  rank 1 / high logp  = strongly nameable (concept fully active)
  rank >> 50 / low logp = barely instantiated (gen-only just under-injected -> a DOSE story, not mechanism)

Reads the archived bundles runs/_ab/<slug>-{all,gen}.pt directly (no file-swapping).
Run:  .venv/bin/python analysis/nameability_ab.py
"""
import numpy as np
import torch


def arm_a_nameability(pt):
    d = torch.load(pt, map_location="cpu", weights_only=False)
    S, reads = d["streams"], d["reads"]
    out = {}
    for s in [x for x in d["strengths"] if x > 0]:
        ranks, logps = [], []
        for r in S:
            if r["strength"] != s or not r["accepted"]:
                continue
            key = ("A", r["gidx"])
            if key in reads:
                rd = reads[key]
                t = max(rd)                      # suffix (naming) position
                ci = r["concept_idx"]
                ranks.append(int(rd[t]["rank"][ci]))
                logps.append(float(rd[t]["logp"][ci]))
        if ranks:
            out[s] = (int(np.median(ranks)), float(np.median(logps)), len(ranks))
    return out


def main():
    models = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"]
    for m in models:
        print(f"\n=== {m}: arm-A nameability (median own-concept rank / logp) ===")
        a = arm_a_nameability(f"runs/_ab/{m}-all.pt")
        g = arm_a_nameability(f"runs/_ab/{m}-gen.pt")
        strs = sorted(set(a) | set(g))
        print(f"  {'strength':>8} | {'all-pos rank':>12} {'logp':>7} | {'gen-only rank':>13} {'logp':>7}")
        for s in strs:
            ar, al, an = a.get(s, (None, None, 0))
            gr, gl, gn = g.get(s, (None, None, 0))
            af = f"{ar:>12} {al:>7.2f}" if ar is not None else f"{'--':>12} {'--':>7}"
            gf = f"{gr:>13} {gl:>7.2f}" if gr is not None else f"{'--':>13} {'--':>7}"
            print(f"  {s:>8} | {af} | {gf}")
        print("  (rank low + logp high = concept strongly instantiated; if gen-only rank is huge it's a DOSE story)")


if __name__ == "__main__":
    main()
