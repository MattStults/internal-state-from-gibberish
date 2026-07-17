"""(3) Per-token recovery curves: does the concept transfer rate differ across readers as more of the
stream is read? Uses the grid-cut activations + prefill logprobs already saved in covert_collect.pt
(cuts {2,4,8,16,32,64,127}). Fully offline.
  R2 activation : one-vs-rest balanced accuracy from the layer-28 vector AT each cut
  R3 prefill    : logP(word|own)-logP(word|s0 control)  AND  mean rank of true word among 12, AT each cut
  R1 surface    : TOKEN-level histogram over the first t tokens (proxy; the true CHAR counter would need
                  the tokenizer to truncate text per cut -> see save-steering-primitives rule). Char
                  counter's full-stream value (0.785 at s60) shown as a reference dot.
Char surface is stronger than this token proxy (ocean is a letter effect), so R1 here is a lower bound."""
import json
import _paths as P
from collections import Counter
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = torch.load(P.DATA / "covert_collect.pt", map_location="cpu", weights_only=False)
S = d["streams"]; concepts = d["concepts"]; K = len(concepts); GRID = d["grid"]
CHAR_ENDPOINT = 0.785  # R1 char-counter full-stream balanced acc at s60 (covert_v2_results.json)

def ovr_bacc(X, yb, repeats=3):
    npos = int(yb.sum())
    if npos < 5 or (len(yb) - npos) < 5:   # too few of a class for stable CV -> skip
        return np.nan
    folds = int(min(5, npos))
    accs = []
    for seed in range(repeats):
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        pred = np.zeros_like(yb)
        for tr, te in skf.split(X, yb):
            k = int(min(30, len(tr) - 1, X.shape[1]))
            p = PCA(k).fit(X[tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(p.transform(X[tr]), yb[tr])
            pred[te] = clf.predict(p.transform(X[te]))
        accs.append(balanced_accuracy_score(yb, pred))
    return float(np.mean(accs))

def mean_bacc(X, y):
    vals = [ovr_bacc(X, (y == c).astype(int)) for c in range(K)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)), len(vals)

def has_cut(store, arm, g, t):
    r = store.get((arm, g))
    return bool(r) and (t in r)

STR = max(d["strengths"])
ts, act, bmc, ns, nc = [], [], [], [], []
for t in GRID:
    clean = [r for r in S if r["strength"] == STR and r["accepted"] and has_cut(d["acts"], "B", r["gidx"], t)]
    if len(clean) < 80:          # need enough streams reaching this token for a stable read
        continue
    y = np.array([r["concept_idx"] for r in clean])
    Xa = np.array([d["acts"][("B", r["gidx"])][t] for r in clean])
    ba, nconc = mean_bacc(Xa, y)          # R2 internal read: signal in the residual AT position t
    ctrl = [r for r in S if r["strength"] == 0 and r["accepted"] and has_cut(d["reads"], "B", r["gidx"], t)]
    ctrl_lp = np.array([d["reads"][("B", r["gidx"])][t]["logp"] for r in ctrl])
    bm = []
    for c in range(K):                    # R3 prefill: cumulative over first t tokens
        own = [r for r in clean if r["concept_idx"] == c]
        ol = np.array([d["reads"][("B", r["gidx"])][t]["logp"][c] for r in own])
        bm.append(ol.mean() - ctrl_lp[:, c].mean())
    ts.append(t); act.append(ba); bmc.append(float(np.mean(bm))); ns.append(len(clean)); nc.append(nconc)
    print(f"t={t:3d} n={len(clean):3d} ({nconc}/12 concepts)  act={ba:.3f}  prefill_BmC={np.mean(bm):+.3f}")

json.dump({"strength": STR, "tokens": ts, "n": ns, "n_concepts": nc, "act": act,
           "prefill_BmC": bmc, "char_endpoint": CHAR_ENDPOINT}, open(P.RESULTS / "covert_v3_curves.json", "w"), indent=2)
print("wrote covert_v3_curves.json")

# keep only cuts where (nearly) all 12 concepts still have streams -> comparable across t.
# (most clean gibberish is short: beyond ~16 tokens only a biased 5/12-concept subset survives.)
comp = [i for i in range(len(ts)) if nc[i] >= 11 and not np.isnan(bmc[i])]
ts, act, bmc = [ts[i] for i in comp], [act[i] for i in comp], [bmc[i] for i in comp]
print(f"plotting comparable cuts (>=11/12 concepts): {ts}")

# ---- figure: transfer rate over the stream, per reader ----
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
fig.suptitle(f"Transfer rate over the stream: the readers peak at DIFFERENT points "
             f"(strong injection, s{STR})", fontsize=13)
a = ax[0]
a.plot(ts, act, "-s", color="#ff7f0e", ms=8, label="R2 internal read (residual at token t)")
a.axhline(0.5, color="#888", ls=":", lw=1.3, label="chance = 0.5")
a.axhline(CHAR_ENDPOINT, color="#1f77b4", ls="--", lw=1.5,
          label=f"R1 char-counter, FULL stream = {CHAR_ENDPOINT} (cumulative)")
a.annotate("front-loaded: strongest at the\nfirst tokens, decays as random\ncontent accumulates",
           xy=(ts[0], act[0]), xytext=(ts[len(ts)//2], 0.72), fontsize=8.5,
           arrowprops=dict(arrowstyle="->", color="#ff7f0e"))
a.set_xscale("log", base=2); a.set_xticks(ts); a.set_xticklabels(ts)
a.set_xlabel("tokens read"); a.set_ylabel("mean one-vs-rest balanced accuracy")
a.set_title("Internal read vs char-counter (0.5 = chance)"); a.legend(fontsize=8.5, loc="upper right")
a.set_ylim(0.45, 0.95)
b = ax[1]
b.plot(ts, bmc, "-o", color="#2ca02c", ms=8, label="R3 prefill: logP(word|own) − logP(word|control)")
b.axhline(0, color="k", lw=1)
b.annotate("builds the OTHER way: needs ~8–16\ntokens of context before the word\nbecomes more likely than control",
           xy=(ts[min(3, len(ts)-1)], bmc[min(3, len(ts)-1)]), xytext=(ts[0], min(bmc) - 0.05),
           fontsize=8.5, arrowprops=dict(arrowstyle="->", color="#2ca02c"))
b.set_xscale("log", base=2); b.set_xticks(ts); b.set_xticklabels(ts)
b.set_xlabel("tokens read"); b.set_ylabel("log-prob advantage (nats)")
b.set_title("'Ask it to name the word' reader (0 = no leak)"); b.legend(fontsize=8.5, loc="lower right")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(P.FIGURES / "v3_transfer_curves.png", dpi=140)
print("wrote v3_transfer_curves.png")
