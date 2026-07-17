"""Integration test for full_stream_bits + ladder_bits on a TINY synthetic bundle (mock tokenizer, tiny embed,
no network, no real weights). Exercises the real bundle->bits path with the new min_len / full-budget /
n_groups wiring. Sub-second.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_full_stream_ladder.py
"""
import os
import sys
import tempfile

import numpy as np

AN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
sys.path.insert(0, AN)
import torch                       # noqa: E402
import run_budget as RB           # noqa: E402


class FakeTok:
    """class-c tokens live in [c*15, c*15+15) -> decode maps each to char 'a'+c: a clean separable transcript."""
    def decode(self, ids):
        return "".join(chr(97 + int(i) // 15) for i in ids)


RB._load_tokenizer = lambda slug: FakeTok()   # avoid HF download in the test

rng = np.random.RandomState(0)
VOCAB, K, PER, d = 60, 4, 30, 8
gvec = rng.randn(K, d)
embed = np.vstack([gvec[t // 15] + 0.3 * rng.randn(d) for t in range(VOCAB)])   # (60, 8), class-separable

streams = []
for c in range(K):
    for _ in range(PER):
        L = int(rng.randint(8, 20))                               # VARIABLE length -> tests full-stream pooling
        toks = rng.randint(c * 15, c * 15 + 15, size=L)           # class-specific token range
        gt = [{"ids": np.array([int(t), (int(t) + 1) % VOCAB]),
               "logp": np.array([np.log(0.7), np.log(0.3)])} for t in toks]
        streams.append({"tokens": toks.astype(np.int64), "gen_topk": gt,
                        "concept_idx": c, "accepted": True, "strength": 1})
bundle = {"streams": streams, "strengths": [0, 1], "concepts": [f"c{c}" for c in range(K)],
          "model": "testmodel", "inject": "gen"}

tmp = tempfile.mkdtemp()
bp = os.path.join(tmp, "b.pt")
torch.save(bundle, bp)
np.save(os.path.join(tmp, "testmodel_embed.npy"), embed)
os.environ["INTRO_EMBED_DIR"] = tmp

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

# --- full-stream: variable length, all four readers, char recovers separable signal, bootstrap CI w/ char gap
fs = RB.full_stream_bits(bp, seeds=(0, 1), min_len=4, n=24)
check("full-stream has all 4 readers", set(fs["readers"]) >= {"dist", "emb", "sampled", "char"})
check("full-stream char recovers (>0.3 b)", fs["readers"]["char"]["bits_mean"] > 0.3)
check("full-stream dist recovers (>0.3 b)", fs["readers"]["dist"]["bits_mean"] > 0.3)
check("full-stream bootstrap CI + dist-char gap", fs["bootstrap_ci"] is not None
      and "dist_minus_char" in fs["bootstrap_ci"]["gap_ci"])

# --- ladder: 2-way = 1 bit, 4-way = 2 bits; bits <= H; recovered bits rise with task entropy (calibration)
lad = RB.ladder_bits(bp, seeds=(0, 1), budgets=(2, 4, 6, 8), groups=(2, 4), n=24)
check("ladder task_bits 1.0 & 2.0", abs(lad["2way"]["task_bits"] - 1.0) < 1e-9
      and abs(lad["4way"]["task_bits"] - 2.0) < 1e-9)
check("ladder bits <= H (2way)", lad["2way"]["readers"]["dist"]["bits_mean"] <= 1.0 + 1e-6)
check("ladder bits <= H (4way)", lad["4way"]["readers"]["dist"]["bits_mean"] <= 2.0 + 1e-6)
check("ladder recovered bits rise with K", lad["4way"]["readers"]["dist"]["bits_mean"]
      > lad["2way"]["readers"]["dist"]["bits_mean"])

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
