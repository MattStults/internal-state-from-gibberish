"""RED-first unit test for common_n_subsample (exp2). No model, no GPU.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_prep.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from prep import build_vocab_index, common_n_subsample, group_labels  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

# imbalanced: class 0 has 50, class 1 has 30, class 2 has 40
y = np.array([0] * 50 + [1] * 30 + [2] * 40)

# (1) default n = min class count (30); exactly n per class
idx = common_n_subsample(y, seed=0)
counts = np.bincount(y[idx], minlength=3)
check("default n = min count, balanced", list(counts) == [30, 30, 30])

# (2) explicit n per class
idx2 = common_n_subsample(y, n=20, seed=0)
check("explicit n per class", list(np.bincount(y[idx2], minlength=3)) == [20, 20, 20])

# (3) indices are valid + unique + a subset of the input
check("indices valid/unique subset", len(set(idx2.tolist())) == len(idx2) and idx2.max() < len(y))

# (4) deterministic under a fixed seed
check("deterministic", np.array_equal(common_n_subsample(y, n=20, seed=0), common_n_subsample(y, n=20, seed=0)))

# (5) different seeds pick different members (but same counts)
d = common_n_subsample(y, n=20, seed=1)
check("seed changes selection", not np.array_equal(np.sort(idx2), np.sort(d)) and
      list(np.bincount(y[d], minlength=3)) == [20, 20, 20])

# (6) n larger than a class -> rejected
try:
    common_n_subsample(y, n=45)
    check("rejects n > a class count", False)
except (ValueError, AssertionError):
    check("rejects n > a class count", True)

# --- build_vocab_index: label-blind feature vocab by corpus frequency ---
ids = [5, 5, 5, 9, 9, 3, 7, 7, 7, 7]                 # counts: 7:4, 5:3, 9:2, 3:1

# (7) top-max_vocab by frequency, columns 0..M-1 in frequency order
vi = build_vocab_index(ids, max_vocab=2)
check("top-2 by frequency, ordered", set(vi.keys()) == {7, 5} and vi[7] == 0 and vi[5] == 1)

# (8) no cap -> all tokens, contiguous columns 0..M-1
vi_all = build_vocab_index(ids)
check("all tokens contiguous cols", set(vi_all.values()) == set(range(4)) and set(vi_all.keys()) == {3, 5, 7, 9})

# (9) deterministic (ties broken stably)
check("deterministic", build_vocab_index(ids, max_vocab=3) == build_vocab_index(ids, max_vocab=3))

# (10) [review] tie-break by id ascending when counts are EQUAL (never exercised before)
check("tie-break id ascending", build_vocab_index([1, 1, 2, 2, 9, 9]) == {1: 0, 2: 1, 9: 2})

# (11) [review] n=None retains the ENTIRE minimum class (the reason the function exists)
idxm = common_n_subsample(y, seed=0)
check("min class fully retained", set(np.where(y == 1)[0].tolist()).issubset(set(idxm.tolist())))

# (12) [review] min_count drops rare (noise) tokens
check("min_count drops rare", set(build_vocab_index([7, 7, 7, 5, 5, 3], min_count=2).keys()) == {7, 5})

# (13) [review] degenerate inputs raise clearly
for bad_call in (lambda: build_vocab_index([]), lambda: common_n_subsample(np.array([], dtype=int))):
    try:
        bad_call(); check("degenerate input raises", False)
    except (ValueError, AssertionError):
        check("degenerate input raises", True)

# --- group_labels: bits-ladder remap of the 12 concepts into 2^k contiguous groups ---
y12 = np.repeat(np.arange(12), 5)                    # 12 concepts

# (14) 1-bit task: 2 contiguous groups, ~balanced (6 concepts each)
g2 = group_labels(y12, 2, seed=0)
check("2 groups contiguous", set(np.unique(g2).tolist()) == {0, 1})
sizes2 = sorted(np.bincount([g2[np.where(y12 == c)[0][0]] for c in range(12)]).tolist())
check("2 groups balanced (6/6 concepts)", sizes2 == [6, 6])

# (15) 2-bit task: 4 groups, 3 concepts each
g4 = group_labels(y12, 4, seed=0)
sizes4 = sorted(np.bincount([g4[np.where(y12 == c)[0][0]] for c in range(12)]).tolist())
check("4 groups, 3 concepts each", set(np.unique(g4).tolist()) == {0, 1, 2, 3} and sizes4 == [3, 3, 3, 3])

# (16) same original concept -> same group (consistent remap)
check("consistent per concept", all(len(set(g4[np.where(y12 == c)[0]].tolist())) == 1 for c in range(12)))

# (17) deterministic under seed; different seed -> different grouping
check("deterministic", np.array_equal(group_labels(y12, 4, seed=0), group_labels(y12, 4, seed=0)))
check("seed changes grouping", not np.array_equal(group_labels(y12, 4, seed=0), group_labels(y12, 4, seed=1)))

# (18) more groups than concepts -> rejected
try:
    group_labels(np.repeat(np.arange(3), 5), 4)
    check("rejects n_groups > n_concepts", False)
except (ValueError, AssertionError):
    check("rejects n_groups > n_concepts", True)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
