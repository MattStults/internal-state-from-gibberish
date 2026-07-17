"""Verification gate — is the injection 'apples to apples' with the 3B reference for the active model?

The eff_mag norm-scaling is only a SEED; THIS is the real check. For each (scaled) eff_mag it reports the
SOURCE nameability (arm-A own-concept rank + logp — how well the injected model itself names the concept,
independent of model size) and capability retention (clean / non-degenerate fraction), and PASSes iff
rank <= NAMEABILITY_MAX_RANK and clean >= CAPABILITY_MIN_CLEAN. The 3B reference (rank ~20-32, clean
~72-85%) is printed so the MATCH QUALITY (not just pass/fail) is visible. Run after a smoke, before the
full collect:  INTRO_MODEL=<slug> python3 analysis/check_injection.py
"""
import _paths as P            # adds ../src to sys.path
import config as C
import numpy as np
import torch

d = torch.load(P.DATA / "covert_collect.pt", map_location="cpu", weights_only=False)
S = d["streams"]
NM, CAP = C.NAMEABILITY_MAX_RANK, C.CAPABILITY_MIN_CLEAN
rn = d.get("resid_norm") or C.MODELS[C.ACTIVE].get("resid_norm")   # .pt field, else registry (older bundles)
ref = C.REF_NORM
scale = f"{rn/ref:.2f}x" if rn else "N/A (resid_norm unmeasured)"
print(f"model={d.get('model', C.ACTIVE)}  layer={d['layer']}  resid_norm={rn}  (3B ref={ref})  this-run strengths={d['strengths']}")
print(f"norm scale vs 3B = {scale}  -> eff_mags scaled {[0]+C.BASE_EFFMAGS} -> {C.strengths()}")
# 3B reference rank/clean per BASE eff_mag (from the 3B FULL run). Each scaled eff_mag in C.strengths()
# came from a base in BASE_EFFMAGS (62<-40, 93<-60), so we compare to the SAME base point on 3B.
REF_3B = {40: (32, 0.85), 60: (21, 0.72)}
_nz = sorted([s for s in d["strengths"] if s > 0])
base_of = {_nz[i]: b for i, b in enumerate(sorted(C.BASE_EFFMAGS)) if i < len(_nz)}  # map by order
print(f"GATE: arm-A own-concept rank <= {NM} AND clean >= {CAP:.2f}; also flag OVER-injection (rank << 3B's)\n")
print(f"{'eff_mag':>7} {'<-3B base':>9} {'rank(med)':>9} {'3B rank':>8} {'gap':>6} {'logp':>7} {'clean%':>7} {'n':>4} {'verdict':>16}")

allpass, any_strength = True, False
for s in [x for x in d["strengths"] if x > 0]:
    any_strength = True
    ranks, logps, clean, tot = [], [], 0, 0
    for r in S:
        if r["strength"] != s:
            continue
        tot += 1; clean += r["accepted"]
        if r["accepted"] and ("A", r["gidx"]) in d["reads"]:
            rd = d["reads"][("A", r["gidx"])]; t = max(rd); ci = r["concept_idx"]
            ranks.append(int(rd[t]["rank"][ci])); logps.append(float(rd[t]["logp"][ci]))
    if not ranks:
        print(f"{s:>7}  (no accepted injected streams with arm-A reads)"); allpass = False; continue
    mr = int(np.median(ranks)); ml = float(np.median(logps)); cf = clean / tot if tot else 0.0
    base = base_of.get(s); ref_rank = REF_3B.get(base, (None, None))[0]
    gap = f"{mr/ref_rank:.1f}x" if ref_rank else "?"
    ok = (mr <= NM) and (cf >= CAP)
    over = mr < 5 or (ref_rank and mr < ref_rank / 4)        # saturated / much stronger than 3B
    v = "FAIL" if not ok else ("PASS(over-inj?)" if over else "PASS")
    allpass = allpass and ok
    print(f"{s:>7} {str(base):>9} {mr:>9} {str(ref_rank):>8} {gap:>6} {ml:>7.2f} {cf*100:>6.0f}% {len(ranks):>4} {v:>16}")

print("\n3B reference is the 3B FULL run (rank~32@40, ~21@60, clean~85/72%); this gate may run on a SMOKE")
print("(arm-A rank is read at the suffix position so it's ~length-invariant; match QUALITY from smoke is noisy).")
verdict = ("GREEN — injection in the matched regime; proceed to the full collect"
           if (allpass and any_strength) else
           "OFF — nudge resid_norm/eff_mag and re-smoke (rank too high = under-injected; clean too low = incapacitating)")
print("VERDICT:", verdict)
print("NOTE before cross-model CONCLUSIONS (not the smoke): two-sided rank band vs 3B; validate the 0.778")
print("depth transfer (layer sweep on a non-3B size); report generation-side dose; Qwen3 tokenizer comparability.")
