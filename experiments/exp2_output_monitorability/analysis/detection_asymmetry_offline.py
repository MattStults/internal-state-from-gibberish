"""Freeze the detection-vs-identification asymmetry measurement (positioning memo E-C).

Question: can a reader tell that a stream was generated under SOME conditioning at all
(detection), separately from identifying WHICH concept (identification)? Detection signal =
the mean per-token lift of a stream's LL under concept contexts vs the arm-neutral context.
A reader detects conditioning iff conditioned (strength-1) streams separate from the
generator's own unconditioned (s0 "neutral") streams on this statistic.

Measured here for every available reader x generator pair on the secret_word grid shards
(runs/lr_grid_box/lr_grid/<reader>__<gen>__secret_word_{SW,N}.pt). First computed ad hoc
2026-07-14 during the post-70B review (conversation-only); this script is the frozen,
re-runnable artifact. Writes reports/detection_asymmetry_results.json.

Reading (from the 2026-07-14 run): only the DIAGONAL separates (e.g. 7B on its own streams:
conditioned +0.010 nats/tok vs neutral-stream -0.012); every cross-model pair shows no
separation or the wrong sign. Even detection is generator-privileged in this regime.

CPU-only. Run: .venv/bin/python experiments/exp2_output_monitorability/analysis/detection_asymmetry_offline.py
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GRID = os.path.join(REPO, "runs", "lr_grid_box", "lr_grid")
OUT = os.path.join(HERE, "..", "reports", "detection_asymmetry_results.json")

READERS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "falcon3-1b", "falcon3-3b", "falcon3-7b")
GENS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")


def pair_stats(reader, gen, arm="secret_word", ctx="SW"):
    p = os.path.join(GRID, f"{reader}__{gen}__{arm}_{ctx}.pt")
    pn = os.path.join(GRID, f"{reader}__{gen}__{arm}_N.pt")
    if not (os.path.exists(p) and os.path.exists(pn)):
        return None
    sh = torch.load(p, map_location="cpu", weights_only=False)
    shn = torch.load(pn, map_location="cpu", weights_only=False)
    concepts = list(sh["contexts"])
    lln = {r["gidx"]: r["ll"]["neutral"] for r in shn["records"]}
    cond, neu = [], []
    for r in sh["records"]:
        if r["gidx"] not in lln:
            continue
        lift = np.mean([float(r["ll"][c]) - float(lln[r["gidx"]]) for c in concepts])
        pt = lift / max(r.get("T_noeos", r["T"]), 1)
        if r["concept"] in concepts and r.get("strength", 1) == 1:
            cond.append(pt)
        elif r["concept"] == "neutral":
            neu.append(pt)
    if not cond or not neu:
        return None
    cond, neu = np.asarray(cond), np.asarray(neu)
    # separation in units of the pooled sd (a crude effect size; descriptive)
    pooled = np.sqrt((cond.var() + neu.var()) / 2)
    return dict(
        cond_pt_mean=float(cond.mean()), cond_pt_sd=float(cond.std()),
        neutral_pt_mean=float(neu.mean()), neutral_pt_sd=float(neu.std()),
        separation_pt=float(cond.mean() - neu.mean()),
        separation_d=float((cond.mean() - neu.mean()) / pooled) if pooled > 0 else None,
        n_cond=int(len(cond)), n_neutral=int(len(neu)),
        diagonal=bool(reader == gen),
    )


def main():
    cells = {}
    for reader in READERS:
        for gen in GENS:
            st = pair_stats(reader, gen)
            if st is not None:
                cells[f"{reader}__{gen}"] = st
    results = dict(
        question=("detection (was this stream conditioned at all?) vs identification "
                  "(which concept?): per-token mean-over-12-contexts lift vs the arm "
                  "neutral, conditioned streams vs the generator's s0 streams"),
        arm="secret_word", statistic="nats/token, eos-free length",
        first_measured="2026-07-14 (post-70B review); frozen by this script",
        cells=cells,
        reading=("only diagonal (reader==generator) cells separate conditioned from "
                 "unconditioned streams; cross-model cells show ~no or wrong-sign "
                 "separation -- detection, like identification, is generator-privileged "
                 "on word-free secret_word streams"),
    )
    with open(os.path.abspath(OUT), "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {os.path.abspath(OUT)}")
    print(f"{'reader on gen':34s} {'cond/tok':>10s} {'s0/tok':>10s} {'sep':>9s} {'d':>7s}")
    for k, c in cells.items():
        print(f"{k:34s} {c['cond_pt_mean']:+10.4f} {c['neutral_pt_mean']:+10.4f} "
              f"{c['separation_pt']:+9.4f} {c['separation_d'] if c['separation_d'] is None else round(c['separation_d'], 2):>7}")


if __name__ == "__main__":
    main()
