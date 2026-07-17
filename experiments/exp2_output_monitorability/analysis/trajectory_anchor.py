"""E4 local anchor ($0): injected + s0 concept-vector trajectories from the exp1 capture's STORED acts.
Arm A = read with injection hook LIVE (generation-time state, teacher-forced); s0 pool standardizes.
z(t) = own-concept minus mean-other projection, standardized vs the s0 pool at the same cut. Validates
the E4 projection methodology before any GPU spend: PREDICTION (prereg E4) injected z(t) stays >=80%
of its early value (mean t<=8) at t=64. Writes reports/trajectory_anchor.json."""
import json, os, sys
import numpy as np, torch
SLUG = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5-1.5b"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
cap = torch.load(os.path.join(REPO, "runs/" + SLUG + "/data/covert_collect.pt"),
                 map_location="cpu", weights_only=False)
V = {c: v / np.linalg.norm(v) for c, v in cap["inject_vectors"].items()}
names = sorted(V); vmat = np.stack([V[c] for c in names])
smax = max(s["strength"] for s in cap["streams"])
cuts = sorted({t for d in cap["acts"].values() for t in d})

def zrows(strength, arm):
    rows = {t: [] for t in cuts}
    for s in cap["streams"]:
        if s["strength"] != strength or not s["accepted"]: continue
        a = cap["acts"].get((arm, s["gidx"]))
        if not a: continue
        for t, h in a.items():
            p = vmat @ (h / np.linalg.norm(h))
            if s["strength"] == 0:
                rows[t].append(p)                      # s0: keep full projection vector (pool stats)
            else:
                i = names.index(s["concept"])
                rows[t].append(p[i] - np.mean(np.delete(p, i)))
    return rows

s0 = zrows(0, "B")
mu = {t: np.mean(np.concatenate(s0[t])) if s0[t] else 0 for t in cuts}
sd = {t: np.std(np.concatenate(s0[t])) + 1e-9 for t in cuts}
out = {"cuts": cuts, "strength": smax, "arms": {}}
for arm in ("A", "B"):
    inj = zrows(smax, arm)
    z = {t: float((np.mean(inj[t]) - mu[t]) / sd[t]) for t in cuts if inj[t]}
    out["arms"][arm] = {"z_by_cut": z, "n": {t: len(inj[t]) for t in cuts}}
    early = np.mean([z[t] for t in z if t <= 8]); late = z.get(64) or z.get(max(z))
    out["arms"][arm]["early"] = float(early); out["arms"][arm]["late_ratio"] = float(late / early) if early else None
    print(f"arm {arm} (injected s{smax}): z by cut", {t: round(z[t],2) for t in sorted(z)},
          f"late/early={out['arms'][arm]['late_ratio']:.2f}" if early else "", flush=True)
dst = os.path.join(REPO, "experiments/exp2_output_monitorability/reports/trajectory_anchor_" + SLUG + ".json")
json.dump(out, open(dst, "w"), indent=2); print("wrote", dst)
