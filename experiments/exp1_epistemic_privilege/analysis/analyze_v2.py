"""Offline analysis of covert_collect.pt -> covert_v2_results.json + baseline-anchored figures.
Three readers, baseline-explicit, capacity-matched where they're classifiers:
  R1 surface   : CHARACTER uni+bigram histogram from the stream text (a literal symbol-counter),
                 PCA-30 -> logistic; per-concept one-vs-rest BALANCED ACCURACY (chance 0.5).
  R2 activation: clean reader layer-28 vector (endpoint cut), PCA-30 -> logistic; same OvR balanced acc.
  R3 prefill   : logP(secret word) on own clean streams MINUS on un-injected (s0) streams [B - C], boot CI.
Baselines explicit: classifiers -> chance=0.5 line + empirical label-shuffle 95% detection floor
(balanced accuracy is robust to the single-class collapse that poisoned plain recall); prefill -> 0-line
+ 95% bootstrap CI. Pure CPU/sklearn (no GPU, no model)."""
import json
import _paths as P
from _features import char_features, ovr_bacc   # canonical R1 featurizer + OvR balanced accuracy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = torch.load(P.DATA / "covert_collect.pt", map_location="cpu", weights_only=False)
S = d["streams"]; concepts = d["concepts"]; K = len(concepts)

# ----------------------------------------------------------------- features
def endpoint(store, arm, g):
    r = store.get((arm, g))
    if not r:
        return None
    return r[max(r.keys())]

def per_concept_bacc(X, y):
    return {concepts[c]: ovr_bacc(X, (y == c).astype(int)) for c in range(K)}

def shuffle_floor(X, y, n_perm=30):
    """Pooled OvR balanced-acc under label permutation -> Bonferroni family-wise-5% detection floor + null
    mean. Two corrections vs the naive version: (1) the null uses the SAME repeats (5) as the plotted
    per-concept statistic -- a repeats=1 null has higher variance, so its percentile is the wrong floor for a
    repeats=5 statistic; (2) the percentile is Bonferroni-corrected for the K concepts tested per panel
    (1 - 0.05/K), so a green bar is a family-wise-5% claim, not ~0.6 expected false greens per 12-bar panel."""
    rng = np.random.default_rng(0)
    pool = []
    for j in range(n_perm):
        yp = rng.permutation(y)
        for c in range(K):
            pool.append(ovr_bacc(X, (yp == c).astype(int), repeats=5, seed0=j))
    return float(np.percentile(pool, 100 * (1 - 0.05 / K))), float(np.mean(pool))

# ----------------------------------------------------------------- prefill B - C
def prefill_BminusC(strength, n_boot=2000):
    s0 = [r for r in S if r["strength"] == 0 and r["accepted"]]
    ctrl = np.array([endpoint(d["reads"], "B", r["gidx"])["logp"] for r in s0])  # (n0,K)
    rng = np.random.default_rng(0)
    out = {}
    for c in range(K):
        own_rows = [r for r in S if r["concept_idx"] == c and r["strength"] == strength and r["accepted"]]
        own = np.array([endpoint(d["reads"], "B", r["gidx"])["logp"][c] for r in own_rows])
        cc = ctrl[:, c]
        diff = own.mean() - cc.mean()
        bs = np.array([own[rng.integers(0, len(own), len(own))].mean()
                       - cc[rng.integers(0, len(cc), len(cc))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        # ceiling: arm A (injection-live reader) on its own streams, same control
        a_own = [endpoint(d["reads"], "A", r["gidx"]) for r in own_rows]
        a_own = np.array([x["logp"][c] for x in a_own if x is not None])
        ceil = float(a_own.mean() - cc.mean()) if len(a_own) else None
        out[concepts[c]] = dict(BmC=float(diff), lo=float(lo), hi=float(hi),
                                n_own=len(own), ceiling=ceil)
    return out

# ----------------------------------------------------------------- run all strengths
CHANCE = 0.5
results = {"layer": d["layer"], "K": K, "concepts": concepts, "chance": CHANCE, "strengths": {}}
for s in d["strengths"]:
    clean = [r for r in S if r["strength"] == s and r["accepted"]]
    y = np.array([r["concept_idx"] for r in clean])
    Xs = char_features([r["text"] for r in clean])
    Xa = np.array([endpoint(d["acts"], "B", r["gidx"]) for r in clean])
    bacc_s = per_concept_bacc(Xs, y)
    bacc_a = per_concept_bacc(Xa, y)
    thr_s, mean_s = shuffle_floor(Xs, y)
    thr_a, mean_a = shuffle_floor(Xa, y)
    entry = dict(n=len(y),
                 surface=dict(bacc=bacc_s, mean=float(np.mean(list(bacc_s.values()))), thr95=thr_s, null_mean=mean_s),
                 activation=dict(bacc=bacc_a, mean=float(np.mean(list(bacc_a.values()))), thr95=thr_a, null_mean=mean_a))
    if s != 0:
        entry["prefill"] = prefill_BminusC(s)
    results["strengths"][str(s)] = entry
    print(f"s{s}: n={len(y)}  surface mean-bacc={entry['surface']['mean']:.3f} (floor={thr_s:.3f})  "
          f"activation mean-bacc={entry['activation']['mean']:.3f} (floor={thr_a:.3f})")

json.dump(results, open(P.RESULTS / "covert_v2_results.json", "w"), indent=2)
print("wrote covert_v2_results.json")

# ================================================================= FIGURES
ORDER = sorted(range(K), key=lambda c: -results["strengths"][str(max(d["strengths"]))]["activation"]["bacc"][concepts[c]])
names = [concepts[c] for c in ORDER]

def bar_panel(ax, vals, thr, chance, title, xlabel, cis=None, zero_line=False):
    yp = np.arange(len(names))
    if zero_line:
        above = [(lo > 0) for (lo, hi) in cis]
    else:
        above = [v > thr for v in vals]
    colors = ["#2ca02c" if a else "#c7c7c7" for a in above]
    if cis is not None:
        err = np.array([[v - lo for v, (lo, hi) in zip(vals, cis)],
                        [hi - v for v, (lo, hi) in zip(vals, cis)]])
        ax.barh(yp, vals, color=colors, xerr=err, error_kw=dict(lw=.8, ecolor="#444"))
    else:
        ax.barh(yp, vals, color=colors)
    ax.set_yticks(yp); ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    if zero_line:
        ax.axvline(0, color="k", lw=1.4, label="control baseline (no injection)")
        ax.legend(fontsize=7.5, loc="lower right")
    else:
        ax.axvline(chance, color="#888", ls=":", lw=1.4, label=f"chance = {chance:.2f}")
        ax.axvline(thr, color="#d62728", ls="-", lw=1.6, label=f"detection floor (Bonferroni 5%/K shuffle) = {thr:.2f}")
        ax.set_xlim(0.4, 1.0)
        ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title(title, fontsize=11); ax.set_xlabel(xlabel, fontsize=9)

# ---- Figure 1: three readers, baseline-explicit (strong + medium injection) ----
for s in [str(x) for x in sorted([z for z in d["strengths"] if z > 0], reverse=True)]:
    E = results["strengths"][s]
    fig, ax = plt.subplots(1, 3, figsize=(17, 6.4))
    fig.suptitle(f"Does each concept show up in word-free gibberish?  (injection strength {s}, "
                 f"clean streams, 12 concepts)\nGreen = clears its baseline (real signal); grey = does not.",
                 fontsize=13)
    bar_panel(ax[0], [E["surface"]["bacc"][n] for n in names], E["surface"]["thr95"], CHANCE,
              "R1  Symbol-counter (counts characters)\none-vs-rest balanced accuracy",
              "balanced accuracy (0.5 = chance, 1.0 = perfect)")
    bar_panel(ax[1], [E["activation"]["bacc"][n] for n in names], E["activation"]["thr95"], CHANCE,
              "R2  Model's internal read (layer-28)\none-vs-rest balanced accuracy",
              "balanced accuracy (0.5 = chance, 1.0 = perfect)")
    pf = E["prefill"]
    bar_panel(ax[2], [pf[n]["BmC"] for n in names], 0, 0,
              "R3  Ask the model to name it\nlogP(word | own stream) − logP(word | control)",
              "log-prob advantage (nats); >0 = leaks", cis=[(pf[n]["lo"], pf[n]["hi"]) for n in names],
              zero_line=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(P.FIGURES / f"v2_readers_s{s}.png", dpi=140)
    print(f"wrote v2_readers_s{s}.png")

# ---- Figure 2: privilege headline — counter vs internal read ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
strs = [str(s) for s in d["strengths"]]
xs = np.arange(len(strs)); w = 0.36
sv = [results["strengths"][s]["surface"]["mean"] for s in strs]
av = [results["strengths"][s]["activation"]["mean"] for s in strs]
ax[0].bar(xs - w/2, sv, w, label="R1 symbol-counter (chars)", color="#1f77b4")
ax[0].bar(xs + w/2, av, w, label="R2 internal read (layer-28)", color="#ff7f0e")
ax[0].axhline(CHANCE, color="#888", ls=":", lw=1.4, label="chance = 0.50")
ax[0].set_xticks(xs); ax[0].set_xticklabels(["none (s0)", "medium (s40)", "strong (s60)"])
ax[0].set_ylim(0.45, max(max(sv), max(av)) + 0.05)
ax[0].set_ylabel("mean one-vs-rest balanced accuracy")
ax[0].set_title("Privilege test (averaged over 12 concepts):\ndoes the internal read beat a symbol-counter?")
ax[0].legend(fontsize=8.5)
# per-concept paired at s60
E = results["strengths"][str(max(d["strengths"]))]
yp = np.arange(len(names))
ax[1].barh(yp - 0.2, [E["surface"]["bacc"][n] for n in names], 0.4, label="R1 counter (chars)", color="#1f77b4")
ax[1].barh(yp + 0.2, [E["activation"]["bacc"][n] for n in names], 0.4, label="R2 internal", color="#ff7f0e")
ax[1].axvline(CHANCE, color="#888", ls=":", lw=1.4, label="chance=0.50")
ax[1].set_xlim(0.4, 1.0)
ax[1].set_yticks(yp); ax[1].set_yticklabels(names, fontsize=9); ax[1].invert_yaxis()
ax[1].set_xlabel("balanced accuracy"); ax[1].set_title("Per concept (strong injection): counter vs internal read")
ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(P.FIGURES / "v2_privilege.png", dpi=140)
print("wrote v2_privilege.png")
