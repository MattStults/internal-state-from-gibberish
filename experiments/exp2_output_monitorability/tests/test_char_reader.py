"""RED-first unit test for the transcript CHAR reader added to exp2's bits pipeline (the fix for the
'distribution-only' headline: exp1's char n-gram was never scored in bits). CPU sklearn only, mock tokenizer,
synthetic streams -- no network, no real bundles, no embed matrices. Sub-second.

RED rationale: before this change `mode='char'` did not exist -- _features had no char branch and MODES was
(dist, emb, sampled), so char features + char bits could not be produced at all.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_char_reader.py
"""
import os
import sys

import numpy as np

AN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
sys.path.insert(0, AN)
from features import char_features_from_tokens  # noqa: E402
from info import bits_recovered                 # noqa: E402
from reader import best_reader_proba_by_budget  # noqa: E402
import run_budget as RB                          # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))


class FakeTok:
    """id -> one of 5 chars by residue: makes the decoded transcript class-separable via char histogram."""
    def decode(self, ids):
        return "".join(chr(97 + (int(i) % 5)) for i in ids)


rng = np.random.RandomState(0)
K, per, budget = 5, 30, 12
tok = FakeTok()
streams, y = [], []
for c in range(K):
    for _ in range(per):
        # mostly residue c (-> char chr(97+c) dominates the histogram) + a little noise
        toks = [c + 5 * rng.randint(0, 40) if rng.rand() < 0.8 else rng.randint(0, 200) for _ in range(20)]
        streams.append({"tokens": np.array(toks, dtype=np.int64)})
        y.append(c)
y = np.array(y)

# (1) char featurizer decodes tokens[:budget] and returns a dense (n, d) histogram matrix
X = char_features_from_tokens(streams, budget, tok)
check("char features shape (n, d>0)", X.shape[0] == len(streams) and X.ndim == 2 and X.shape[1] > 0)
check("budget slices tokens", char_features_from_tokens([streams[0]], 3, tok).shape[0] == 1)

# (2) the char reader recovers real bits on separable transcripts (dense best-decoder)
p = best_reader_proba_by_budget({budget: X}, y, [budget], kind="dense", seed=0)[budget]
check("char recovers separable signal", bits_recovered(y, p) > 0.5)

# (3) shuffle floor: no memorization through nested CV
ysh = y.copy(); rng.shuffle(ysh)
psh = best_reader_proba_by_budget({budget: X}, ysh, [budget], kind="dense", seed=0)[budget]
check("char shuffle -> ~0", bits_recovered(ysh, psh) < 0.3)

# (4) wiring: run_budget knows char, routes it to the dense pipeline, and _features dispatches to char
check("MODES includes char", "char" in RB.MODES)
check("char is dense", RB.KIND["char"] == "dense")
Xw = RB._features(streams, budget, vocab=None, mode="char", tokenizer=tok)
check("_features(mode=char) == char_features_from_tokens", np.array_equal(Xw, X))

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
