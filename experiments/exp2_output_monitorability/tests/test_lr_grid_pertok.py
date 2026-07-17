"""RED-first unit tests for scale-grid unit B11 (prereg Amendment 1, Blocker 1): shards store
PER-TOKEN LL vectors (numerator and denominator, fp16) alongside the certified sums, so the
eos-free / with-eos / prefix-K readouts are all OFFLINE-computable from a shard. No model, no GPU.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_lr_grid_pertok.py

P1  score_batch_dual fills a caller-supplied per-token out-dict [B, Tmax] riding the SAME single
    forward as the two certified sums (never a second model pass).
P2  the per-token values are the certified numerics' own: summed over each stream's with-eos /
    eos-free length they reproduce ll_eos / ll exactly (fp32, before the fp16 shard cast), and the
    capture goes through LR.ll_from_logits itself (no reimplemented numerics in lr_grid).
P3  record wiring: grid_records carries ll_tok {} + T_noeos; record_toks stores fp16 vectors of
    each stream's TRUE (unpadded, with-eos) length.
P4  offline recomputation from a shard: eos-free, with-eos and prefix-K sums all derive from the
    stored fp16 vectors within fp16 tolerance of the certified fp32 sums (Blocker 1's requirement).
P5  the raising-KV fallback path still yields a per-token capture consistent with the returned
    sums (concat re-entry overwrites the stash, matching lr_grid's documented stash semantics).
P6  main() persists the capture: record_toks wired in the scoring loop; the saved shard dict
    names ll_tok.
"""
import contextlib
import inspect
import io
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


import lr_grid as G      # noqa: E402
import lr_reader as LR   # noqa: E402

Vv = 16                  # > max synthetic token id


class _FlatModel:
    """Uniform logits: every true-token logprob is exactly -ln(V)."""
    def __init__(self):
        self.calls = 0

    def __call__(self, ids, attention_mask=None, **kw):
        self.calls += 1
        class _O:            # noqa: E306
            pass
        o = _O()
        o.logits = torch.zeros((ids.shape[0], ids.shape[1], Vv))
        return o


TOKS = [[5, 6, 9], [5, 6]]                       # stream 0 ends in eos=9
ctx = torch.ones((1, 3), dtype=torch.long)
batch, lens = LR.pad_tokens(TOKS, pad_id=0)
lens_ne = G.noeos_lens(TOKS, eos_id=9)
ln = float(np.log(Vv))

# ---------------------------------------------------------------- P1/P2: same-forward capture
try:
    m = _FlatModel()
    tokout = {}
    ll_free, ll_eos, use_kv = G.score_batch_dual(m, ctx, None, None, batch, lens, lens_ne,
                                                 use_kv=False, pertok=tokout)
    check("P1 pertok out-dict filled with a [B, Tmax] per-token LL matrix",
          "ll" in tokout and tuple(tokout["ll"].shape) == (2, 3), f"got {tokout.keys()}")
    check("P1 ONE forward for sums AND per-token capture", m.calls == 1, f"{m.calls} calls")
    pt = tokout["ll"]
    check("P2 per-token values are the certified -ln(V) at real positions, 0 at padding",
          abs(float(pt[0, 0]) + ln) < 1e-6 and abs(float(pt[0, 2]) + ln) < 1e-6
          and abs(float(pt[1, 2])) < 1e-9, f"pt = {pt.tolist()}")
    check("P2 with-eos sums reproduce ll_eos exactly (fp32, pre-cast)",
          all(abs(float(pt[i, :lens[i]].sum()) - float(ll_eos[i])) < 1e-5 for i in range(2)))
    check("P2 eos-free sums reproduce ll exactly (fp32, pre-cast)",
          all(abs(float(pt[i, :lens_ne[i]].sum()) - float(ll_free[i])) < 1e-5 for i in range(2)))
except Exception as e:
    check("P1/P2 score_batch_dual per-token capture", False, f"raised {type(e).__name__}: {e}")

# P2b: the capture is LR.ll_from_logits' own output (source-level: lr_grid still defines no
# log-probability numerics of its own -- guarded again here so B11 can't have snuck one in).
gsrc = open(os.path.join(REPO, "src", "lr_grid.py")).read()
check("P2 no reimplemented numerics (no log-softmax/logsumexp/gather in lr_grid source)",
      "log_softmax" not in gsrc and "logsumexp" not in gsrc and ".gather(" not in gsrc)

# ---------------------------------------------------------------- P3: record wiring
try:
    streams = [dict(gidx=7, concept="fear", strength=1, tokens=[5, 6, 9]),
               dict(gidx=8, concept="neutral", strength=0, tokens=[5, 6])]
    recs = G.grid_records(streams, eos_id=9)
    check("P3 grid_records: ll_tok {} + T_noeos per record",
          recs[0].get("ll_tok") == {} and recs[0].get("T_noeos") == 2
          and recs[1].get("T_noeos") == 2, f"recs[0] keys = {sorted(recs[0])}")
    check("P3 grid_records without eos_id keeps T_noeos = T (backward-compatible default)",
          G.grid_records(streams)[0].get("T_noeos") == 3)
    pt = torch.tensor([[-1.0, -2.0, -3.0], [-4.0, -5.0, 0.0]])
    G.record_toks(recs, 0, "fear", pt, lens=[3, 2])
    v0, v1 = recs[0]["ll_tok"]["fear"], recs[1]["ll_tok"]["fear"]
    check("P3 record_toks stores fp16 vectors of TRUE with-eos length",
          isinstance(v0, np.ndarray) and v0.dtype == np.float16 and v0.shape == (3,)
          and v1.shape == (2,), f"dtypes/shapes: {v0.dtype} {v0.shape} {v1.shape}")
    check("P3 stored values match the matrix rows",
          abs(float(v0[2]) + 3.0) < 1e-2 and abs(float(v1[1]) + 5.0) < 1e-2)
except Exception as e:
    check("P3 record wiring", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- P4: offline recomputation
try:
    m = _FlatModel()
    tokout = {}
    ll_free, ll_eos, _ = G.score_batch_dual(m, ctx, None, None, batch, lens, lens_ne,
                                            use_kv=False, pertok=tokout)
    recs = G.grid_records([dict(gidx=0, concept="fear", strength=1, tokens=TOKS[0]),
                           dict(gidx=1, concept="ocean", strength=1, tokens=TOKS[1])], eos_id=9)
    G.record_lls(recs, 0, "fear", ll_free, ll_eos)
    G.record_toks(recs, 0, "fear", tokout["ll"], lens=lens)
    TOL = 0.01                                     # fp16 cast: ~1e-3/token at these magnitudes
    ok_eos = ok_free = ok_prefix = True
    for r in recs:
        vec = r["ll_tok"]["fear"].astype(np.float64)
        ok_eos &= abs(vec.sum() - r["ll_eos"]["fear"]) < TOL
        ok_free &= abs(vec[: r["T_noeos"]].sum() - r["ll"]["fear"]) < TOL
        K = 2                                      # prefix-K on the eos-free stream
        ok_prefix &= abs(vec[: min(K, r["T_noeos"])].sum() - (-2 * ln)) < TOL
    check("P4 with-eos sum derivable from the stored fp16 vector", ok_eos)
    check("P4 eos-free sum derivable (vec[:T_noeos])", ok_free)
    check("P4 prefix-K sum derivable (vec[:min(K, T_noeos)])", ok_prefix)
except Exception as e:
    check("P4 offline recomputation", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- P5: KV-fallback consistency
try:
    _orig_kv = LR.score_batch_kv

    def _kv_raises(*a, **k):
        raise RuntimeError("synthetic KV failure")

    LR.score_batch_kv = _kv_raises
    try:
        m2 = _FlatModel()
        tokout = {}
        with contextlib.redirect_stdout(io.StringIO()):
            ll_free2, ll_eos2, uk2 = G.score_batch_dual(m2, ctx, None, None, batch, lens,
                                                        lens_ne, use_kv=True, pertok=tokout)
    finally:
        LR.score_batch_kv = _orig_kv
    pt = tokout["ll"]
    check("P5 KV fallback: per-token capture matches the returned (concat) sums",
          uk2 is False
          and all(abs(float(pt[i, :lens[i]].sum()) - float(ll_eos2[i])) < 1e-5 for i in range(2)))
except Exception as e:
    check("P5 KV-fallback per-token capture", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- P7 (TECH-SF3): alignment
# Uniform logits cannot catch an off-by-one or wrong-gather (every position scores -ln V
# regardless of alignment). DISTINCT per-(batch, position, vocab) logits pin the exact gather:
# on the concat path the prediction for stream token t comes from full-logits position
# plen-1+t, so pertok[b, t] == log_softmax(logits[b, plen-1+t])[targets[b, t]] elementwise.
try:
    class _PosModel:
        """Deterministic, DISTINCT logits at every (batch, position, vocab) coordinate."""
        def __init__(self):
            self.logits = None

        def __call__(self, ids, attention_mask=None, **kw):
            B, T = ids.shape
            self.logits = (torch.sin(torch.arange(B * T * Vv, dtype=torch.float32) * 0.7)
                           .reshape(B, T, Vv) * 3.0)
            class _O:            # noqa: E306
                pass
            o = _O()
            o.logits = self.logits
            return o

    mp = _PosModel()
    tokout = {}
    ll_free3, ll_eos3, _ = G.score_batch_dual(mp, ctx, None, None, batch, lens, lens_ne,
                                              use_kv=False, pertok=tokout)
    pt = tokout["ll"]
    plen = ctx.shape[1]
    lp = torch.log_softmax(mp.logits.float(), dim=-1)
    worst = max(abs(float(pt[b, t]) - float(lp[b, plen - 1 + t, int(batch[b, t])]))
                for b in range(batch.shape[0]) for t in range(lens[b]))
    check("P7 pertok[b,t] == logsoftmax(logits[b, plen-1+t])[targets[b,t]] elementwise "
          "(distinct logits catch off-by-one / wrong-gather that uniform logits cannot)",
          worst < 1e-5, f"worst |dev| = {worst:.6f}")
    check("P7 the distinct-logit per-token sums still reconcile with the returned with-eos lls",
          all(abs(float(pt[i, :lens[i]].sum()) - float(ll_eos3[i])) < 1e-4 for i in range(2)))
    shifted_ok = all(
        abs(float(pt[b, t]) - float(lp[b, plen + t, int(batch[b, t])])) < 1e-5
        for b in range(batch.shape[0])
        for t in range(min(lens[b], mp.logits.shape[1] - plen - 1)))
    check("P7 a one-position-shifted expectation DISAGREES (the check has discriminating "
          "power against the off-by-one bug class)", not shifted_ok)
except Exception as e:
    check("P7 distinct-logit alignment", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- P6: main() persists ll_tok
msrc = inspect.getsource(G.main)
check("P6 main wires record_toks in the scoring loop", "record_toks(" in msrc)
check("P6 main passes the pertok out-dict into score_batch_dual", "pertok=" in msrc)
check("P6 the saved shard names ll_tok (offline scorer's contract)", "ll_tok" in msrc)

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
