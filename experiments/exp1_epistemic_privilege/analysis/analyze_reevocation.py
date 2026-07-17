"""Re-evocation analysis (OFFLINE, no GPU, no model) — does the clean reader's state move along the
INJECTED concept direction when it reads that concept's gibberish, selectively, and how big vs the
lexical 'name-the-word' channel?

KEY IDENTITY (review): injection adds alpha*v ONLY at layer 28 and both arms read the same tokens, so
  armA - armB = alpha*v  EXACTLY at every captured position.
=> v_hat_c = normalize(armA - armB) is the *exact* injected direction (recovered offline), and
   ||armA - armB|| = alpha*||v|| = strength (40/60) is the movement along v_hat under injection.
We project the clean reader (arm B) onto v_hat_c.  re-evocation% = <armB - baseline, v_hat>/strength.

Pre-registered (see EXPERIMENT_reevocation.md): primary strength s60, primary cut 8, selectivity stat
mean(diag)-mean(offdiag) on baseline-subtracted + mean-v-removed matrix, derangement permutation.
"""
import json
import _paths as P
from collections import Counter
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = torch.load(P.DATA / "covert_collect.pt", map_location="cpu", weights_only=False)
S = d["streams"]; concepts = d["concepts"]; K = len(concepts); GRID = d["grid"]
STRENGTH = max(d["strengths"]); PRIMARY_CUT = 8
rng = np.random.default_rng(0)

def act(arm, g, t):
    r = d["acts"].get((arm, g))
    return None if (not r or t not in r) else np.asarray(r[t], dtype=np.float64)

def streams_of(c, strength, accepted=True):
    return [r for r in S if r["concept_idx"] == c and r["strength"] == strength and r["accepted"] == accepted]

# ---------------------------------------------------------------- 1. recover v_hat from armA-armB + GATE
vhat = {}; gate = {}
for c in range(K):
    diffs = []
    for r in streams_of(c, STRENGTH):
        a, b = act("A", r["gidx"], PRIMARY_CUT), act("B", r["gidx"], PRIMARY_CUT)
        if a is not None and b is not None:
            diffs.append(a - b)
    diffs = np.array(diffs)
    m = diffs.mean(0)
    rel_std = float(np.linalg.norm(diffs.std(0)) / (np.linalg.norm(m) + 1e-9))
    vhat[c] = m / np.linalg.norm(m)
    gate[c] = dict(norm=float(np.linalg.norm(m)), rel_std=rel_std, n=len(diffs))
VH = np.array([vhat[c] for c in range(K)])                      # [K, D]
vbar = VH.mean(0); vbar = vbar / np.linalg.norm(vbar)           # shared "Tell-me-about-X" direction
VHp = VH - np.outer(VH @ vbar, vbar)                            # mean-v removed
VHp = VHp / np.linalg.norm(VHp, axis=1, keepdims=True)

print("=== GATE: armA-armB should be the constant vector strength*v_hat (||.||~%d, rel_std~0) ===" % STRENGTH)
for c in range(K):
    print(f"  {concepts[c]:12s} ||armA-armB||={gate[c]['norm']:6.2f} (target {STRENGTH})  "
          f"rel_std={gate[c]['rel_std']:.3f}  n={gate[c]['n']}")
norms = np.array([gate[c]["norm"] for c in range(K)]); rsds = np.array([gate[c]["rel_std"] for c in range(K)])
gate_ok = bool(np.all(np.abs(norms - STRENGTH) < 0.15 * STRENGTH) and np.all(rsds < 0.05))
print(f"GATE {'PASS' if gate_ok else 'FAIL'}: |norm-{STRENGTH}|<15% for all={np.all(np.abs(norms-STRENGTH)<0.15*STRENGTH)}, "
      f"rel_std<0.05 for all={np.all(rsds<0.05)}")
cosvv = VH @ VH.T
print(f"vector distinctness: mean |cos(v_i,v_j)| off-diag = {np.abs(cosvv[~np.eye(K,dtype=bool)]).mean():.3f}")

# ---------------------------------------------------------------- 2. baseline (s0), cut-matched
def s0_acts(t):
    return np.array([act("B", r["gidx"], t) for r in S
                     if r["strength"] == 0 and r["accepted"] and act("B", r["gidx"], t) is not None])
baseP = {}      # baseP[t] = [K] mean s0 projection onto each v_hat (mean-v removed) at cut t
baseP_raw = {}  # onto raw v_hat (for %-injection magnitude)
for t in GRID:
    X0 = s0_acts(t)
    if len(X0) >= 30:
        baseP[t] = (X0 @ VHp.T).mean(0)
        baseP_raw[t] = (X0 @ VH.T).mean(0)

# ---------------------------------------------------------------- 3. per-cut re-evocation curve (diag)
def concept_acts(c, t):
    return np.array([act("B", r["gidx"], t) for r in streams_of(c, STRENGTH) if act("B", r["gidx"], t) is not None])
curve_t, curve_reevoc, curve_pct = [], [], []
for t in GRID:
    if t not in baseP:
        continue
    diag_pct = []
    for c in range(K):
        Xc = concept_acts(c, t)
        if len(Xc) < 20:
            diag_pct = None; break
        proj_raw = (Xc @ VH[c])                      # onto raw v_hat_c
        reevoc = proj_raw.mean() - baseP_raw[t][c]   # baseline-subtracted, cut-matched
        diag_pct.append(reevoc / STRENGTH)           # fraction of injection
    if diag_pct is not None:
        curve_t.append(t); curve_pct.append(float(np.mean(diag_pct)))
print("\n=== per-cut re-evocation (mean over concepts, as %% of injection) ===")
for t, p in zip(curve_t, curve_pct):
    print(f"  cut {t:3d}: {100*p:+.1f}% of injection")

# ---------------------------------------------------------------- 4. cross-projection matrix @ primary cut
t = PRIMARY_CUT
M = np.full((K, K), np.nan)              # mean-v-removed, baseline-subtracted (for selectivity)
Mpct = np.full((K, K), np.nan)          # raw, /strength (for magnitude)
boot = {}
for i in range(K):
    Xi = concept_acts(i, t)
    M[i] = (Xi @ VHp.T).mean(0) - baseP[t]
    Mpct[i] = ((Xi @ VH.T).mean(0) - baseP_raw[t]) / STRENGTH
    # bootstrap CI on the diagonal %-injection
    bs = [ ((Xi[rng.integers(0, len(Xi), len(Xi))] @ VH[i]).mean() - baseP_raw[t][i]) / STRENGTH
           for _ in range(2000) ]
    boot[i] = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

diag = np.diag(M); offdiag = M[~np.eye(K, dtype=bool)]
stat = float(diag.mean() - offdiag.mean())
# derangement permutation null
def derange(n):
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)): return p
perm = [float(np.mean([M[i, sg[i]] for i in range(K)]) - offdiag.mean()) for sg in (derange(K) for _ in range(5000))]
pval = float((np.sum(np.array(perm) >= stat) + 1) / (len(perm) + 1))
diag_is_rowmax = int(np.sum([np.argmax(M[i]) == i for i in range(K)]))
print(f"\n=== selectivity @cut{t}: mean(diag)-mean(offdiag) = {stat:+.4f}  perm p = {pval:.4f}  "
      f"(diag is row-max for {diag_is_rowmax}/{K} concepts) ===")

# ---------------------------------------------------------------- 5. word vs experience (both /armA ceiling)
def reads(arm, g, t):
    r = d["reads"].get((arm, g))
    return None if (not r or t not in r) else r[t]
# s0 naming baseline per concept word
s0r = [reads("B", r["gidx"], t) for r in S if r["strength"] == 0 and r["accepted"] and reads("B", r["gidx"], t)]
base_logp = np.array([x["logp"] for x in s0r]).mean(0)      # [K]
reevoc_pct = {}; lexical_pct = {}
for c in range(K):
    rs = streams_of(c, STRENGTH)
    lpB = np.array([reads("B", r["gidx"], t)["logp"][c] for r in rs if reads("B", r["gidx"], t)])
    lpA = np.array([reads("A", r["gidx"], t)["logp"][c] for r in rs if reads("A", r["gidx"], t)])
    upB, upA = lpB.mean() - base_logp[c], lpA.mean() - base_logp[c]
    lexical_pct[c] = float(upB / upA) if upA > 1e-6 else np.nan
    reevoc_pct[c] = float(Mpct[c, c])
mean_reevoc = float(np.nanmean(list(reevoc_pct.values())))
mean_lex = float(np.nanmean(list(lexical_pct.values())))
print(f"\n=== word vs experience (fraction of arm-A ceiling recovered in the clean reader) ===")
print(f"{'concept':12s} {'re-evoc%':>9} {'[95% CI]':>16} {'lexical%':>9}")
for c in range(K):
    print(f"{concepts[c]:12s} {100*reevoc_pct[c]:+8.1f}% [{100*boot[c][0]:+5.1f},{100*boot[c][1]:+5.1f}] {100*lexical_pct[c]:+8.1f}%")
print(f"MEAN  re-evocation={100*mean_reevoc:+.1f}%   lexical(naming)={100*mean_lex:+.1f}%")

# ---------------------------------------------------------------- 6. letter control (context, not a gate)
def char_feats(texts, ut=80, bt=300):
    uni = Counter(ch for tt in texts for ch in tt.lower()); big = Counter((tt.lower()[i], tt.lower()[i+1]) for tt in texts for i in range(len(tt)-1))
    uv = {c: i for i, (c, _) in enumerate(uni.most_common(ut))}; bv = {b: i for i, (b, _) in enumerate(big.most_common(bt))}
    R = []
    for tt in texts:
        tl = tt.lower(); h = np.zeros(len(uv)+len(bv))
        for ch in tl:
            if ch in uv: h[uv[ch]] += 1
        for i in range(len(tl)-1):
            b = (tl[i], tl[i+1])
            if b in bv: h[len(uv)+bv[b]] += 1
        s = h.sum(); R.append(h/s if s else h)
    return np.array(R)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
allc = [r for c in range(K) for r in streams_of(c, STRENGTH) if act("B", r["gidx"], t) is not None]
proj_own = np.array([(act("B", r["gidx"], t) @ VH[r["concept_idx"]]) for r in allc])
Xtxt = char_feats([r["text"] for r in allc])
pred = cross_val_predict(Ridge(alpha=1.0), Xtxt, proj_own, cv=5)
r2 = float(1 - np.var(proj_own - pred) / np.var(proj_own))
print(f"\nletter control: char-histogram predicts {100*max(0,r2):.0f}% of the variance in the v_hat projection "
      f"(context only; a clean reader's state is necessarily a function of the letters)")

json.dump(dict(strength=STRENGTH, cut=PRIMARY_CUT, gate_ok=gate_ok, gate=gate,
               selectivity_stat=stat, selectivity_p=pval, diag_row_max=diag_is_rowmax,
               reevoc_pct={concepts[c]: reevoc_pct[c] for c in range(K)},
               reevoc_ci={concepts[c]: boot[c] for c in range(K)},
               lexical_pct={concepts[c]: lexical_pct[c] for c in range(K)},
               mean_reevoc_pct=mean_reevoc, mean_lexical_pct=mean_lex,
               curve_cuts=curve_t, curve_pct=curve_pct, letter_r2=r2),
          open(P.RESULTS / "reevocation_results.json", "w"), indent=2, default=float)
print("\nwrote reevocation_results.json")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
im = ax[0].imshow(Mpct * 100, cmap="RdBu_r", vmin=-100*np.nanmax(np.abs(np.diag(Mpct))), vmax=100*np.nanmax(np.abs(np.diag(Mpct))))
ax[0].set_xticks(range(K)); ax[0].set_xticklabels(concepts, rotation=90, fontsize=8)
ax[0].set_yticks(range(K)); ax[0].set_yticklabels(concepts, fontsize=8)
ax[0].set_xlabel("projected onto v̂_j"); ax[0].set_ylabel("reading concept-i gibberish")
ax[0].set_title(f"Cross-projection (% of injection), cut {t}\ndiagonal = re-evocation; perm p={pval:.3f}")
fig.colorbar(im, ax=ax[0], fraction=0.046)
order = sorted(range(K), key=lambda c: -reevoc_pct[c]); names = [concepts[c] for c in order]
yp = np.arange(K)
ax[1].barh(yp - 0.2, [100*reevoc_pct[c] for c in order], 0.4, color="#ff7f0e", label="re-evocation % (state→concept dir)",
           xerr=[[100*(reevoc_pct[c]-boot[c][0]) for c in order], [100*(boot[c][1]-reevoc_pct[c]) for c in order]],
           error_kw=dict(lw=.7, ecolor="#444"))
ax[1].barh(yp + 0.2, [100*lexical_pct[c] for c in order], 0.4, color="#2ca02c", label="lexical % (name-the-word)")
ax[1].axvline(0, color="k", lw=1); ax[1].set_yticks(yp); ax[1].set_yticklabels(names, fontsize=8); ax[1].invert_yaxis()
ax[1].set_xlabel("% of arm-A (full-injection) ceiling recovered in the clean reader")
ax[1].set_title("Word vs experience"); ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(P.FIGURES / "reevocation.png", dpi=140); print("wrote reevocation.png")
