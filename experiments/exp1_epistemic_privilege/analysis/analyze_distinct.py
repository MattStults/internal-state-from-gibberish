"""Is there a DISTINCT channel? -> does the layer-28 internal read carry concept-information BEYOND the
character statistics? Incremental test (s60, clean streams, one-vs-rest per concept):
  bacc(char+act) vs bacc(char + act_shuffled-to-wrong-streams), matched capacity (PCA-30 per block).
A positive, beyond-null gap => activations see concept info the character-counter does not (a channel
NOT explained by surface characters). Pure CPU/sklearn."""
import numpy as np, torch
import _paths as P
from _features import char_features   # canonical R1 featurizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

d = torch.load(P.DATA / "covert_collect.pt", map_location="cpu", weights_only=False)
S = d["streams"]; concepts = d["concepts"]; K = len(concepts)

def endpoint(arm, g):
    r = d["acts"].get((arm, g))
    return None if not r else r[max(r.keys())]

def _pca_tt(Xtr, Xte, k):
    p = PCA(n_components=int(min(k, len(Xtr)-1, Xtr.shape[1]))).fit(Xtr)
    return p.transform(Xtr), p.transform(Xte)

def bacc_blocks(blocks, yb, seed=0):
    """blocks: list of feature matrices; each PCA-30'd train-only, concatenated, STANDARDIZED (so no
    block dominates by raw magnitude), balanced logistic CV."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.zeros_like(yb)
    for tr, te in skf.split(blocks[0], yb):
        Ztr, Zte = [], []
        for X in blocks:
            a, b = _pca_tt(X[tr], X[te], 30); Ztr.append(a); Zte.append(b)
        Ztr, Zte = np.hstack(Ztr), np.hstack(Zte)
        sc = StandardScaler().fit(Ztr)
        Ztr, Zte = sc.transform(Ztr), sc.transform(Zte)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Ztr, yb[tr])
        pred[te] = clf.predict(Zte)
    return balanced_accuracy_score(yb, pred)

def mean_seeds(blocks, yb, seeds=5):
    return float(np.mean([bacc_blocks(blocks, yb, s) for s in range(seeds)]))

s = max(d["strengths"])
clean = [r for r in S if r["strength"] == s and r["accepted"]]
y = np.array([r["concept_idx"] for r in clean])
C = char_features([r["text"] for r in clean])
A = np.array([endpoint("B", r["gidx"]) for r in clean])
rng = np.random.default_rng(0)

print(f"s{s}: n={len(y)}  (char dim={C.shape[1]}, act dim={A.shape[1]})  [blocks standardized]")
print(f"{'concept':12s} {'char':>5} {'act':>5} {'C+A':>5} {'null(C+Ashuf)':>13} {'C+A - char':>11}")
incrs = []
for c in range(K):
    yb = (y == c).astype(int)
    b_char = mean_seeds([C], yb)
    b_act = mean_seeds([A], yb)
    b_both = mean_seeds([C, A], yb)
    nulls = [mean_seeds([C, A[rng.permutation(len(A))]], yb) for _ in range(10)]   # seeds=5 (default), matching b_both's variance
    nmean, nhi = float(np.mean(nulls)), float(np.percentile(nulls, 95))
    incr = b_both - b_char            # does adding REAL activations beat char alone?
    incrs.append(incr)
    # significant distinct channel: C+A beats BOTH char-alone and the shuffled-A capacity null
    star = "*" if (b_both > nhi and incr > 0.01) else " "
    print(f"{concepts[c]:12s} {b_char:5.2f} {b_act:5.2f} {b_both:5.2f} {nmean:13.2f} {incr:+10.2f} {star}")
print(f"\nMEAN (C+A) - char = {np.mean(incrs):+.3f}   "
      f"[char-alone mean={np.mean([mean_seeds([C],(y==c).astype(int)) for c in range(K)]):.3f}]")
print("null(C+Ashuf) should sit ~char-alone now; star => activations add concept info BEYOND characters")
