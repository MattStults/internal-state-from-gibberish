"""RED-first unit tests for the LR-reader (src/lr_reader.py + analysis/lr_reader.py).
No model, no GPU -- synthetic logits and stub tokenizers.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_lr_reader.py

C1  context construction: wording-A = primers_v2.compose_system(c, STRONG_SYSTEM, arm="evoked"),
    wording-B = arm="evoked_alt", EXACTLY as at collection (prereg: lr_reader_prereg.md).
C2  neutral context: concept None/"neutral" -> compose_system(None, ..., arm="evoked") for BOTH
    context sets (the shared NEUTRAL persona; 25 contexts, not 26).
C3  the user message is C.GEN_PROMPT (chat_ids path), never GAUGE_PROBE.
S1  stream selection: injected = accepted exp1 streams at max strength only; evoked/evoked_alt =
    accepted strength-1 concept streams PLUS the s0 neutral streams (sanity-gate pool); len>=2.
S2  token dtype robustness: list[int], np.ndarray and 1-D torch tensors all pad into the same batch.
L1  teacher-forced LL arithmetic on synthetic logits: pred_logits[:, t] predicts targets[:, t];
    LL = sum of float32 log-softmax picks over the stream's true length.
L2  right-padding is inert: garbage in padded logit/target positions cannot move a shorter
    stream's LL.
R1  temperature fit + bits readout (analysis): cleanly separable scores -> bits near log2(12);
    uninformative (all-equal) scores -> bits ~= 0; top-1 matches argmax accuracy.
R2  stratified thirds split: per-concept floor(n/3) (min 1) calibration rows, disjoint from eval,
    together covering everything.
K1  attempt-6 gates: the LR_SELFCHECK_FALLBACK print withholds raw exception text (a CUDA OOM
    message would trip labkit's FATAL substring match); a KV batch that raises POST-selfcheck
    falls back to the concat path with a collision-safe print instead of killing the box.
R3  lr_reader_offline min_eval_per_concept is the min over ALL split seeds, not the last seed's.
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis"))

import config as C             # noqa: E402
import primers_v2 as P         # noqa: E402

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


try:
    import lr_reader as LR         # src/lr_reader.py (GPU module; imported CPU-side for the pure fns)
except Exception as e:
    LR = None
    check("import src/lr_reader.py", False, f"{type(e).__name__}: {e}")

try:
    import lr_reader_offline as AN  # analysis/lr_reader.py is shadowed by src on sys.path -> module
except Exception:                   # name registered as analysis/lr_reader_offline.py
    AN = None

if AN is None:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "lr_reader_offline",
            os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis",
                         "lr_reader_offline.py"))
        AN = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(AN)
    except Exception as e:
        AN = None
        check("import analysis/lr_reader_offline.py", False, f"{type(e).__name__}: {e}")


class RecTok:
    """Records chat messages; returns fixed ids (K.chat_ids contract)."""
    def __init__(self):
        self.msgs = None

    def apply_chat_template(self, msgs, **kw):
        self.msgs = msgs
        return torch.ones((1, 4), dtype=torch.long)


# ---------------------------------------------------------------- C1/C2: context construction
if LR is not None:
    try:
        check("C1 wording-A system == compose_system(c, STRONG_SYSTEM, arm='evoked')",
              LR.context_system("A", "fear") == P.compose_system("fear", C.STRONG_SYSTEM, arm="evoked"))
        check("C1 wording-B system == compose_system(c, STRONG_SYSTEM, arm='evoked_alt')",
              LR.context_system("B", "fear") == P.compose_system("fear", C.STRONG_SYSTEM,
                                                                 arm="evoked_alt"))
        check("C1 A and B differ for a real concept",
              LR.context_system("A", "ocean") != LR.context_system("B", "ocean"))
        neutral = P.compose_system(None, C.STRONG_SYSTEM, arm="evoked")
        check("C2 neutral context (None) == compose_system(None, ..., 'evoked'), both sets",
              LR.context_system("A", None) == neutral and LR.context_system("B", None) == neutral)
        check("C2 concept string 'neutral' maps to the same neutral context",
              LR.context_system("A", "neutral") == neutral)
    except Exception as e:
        check("C1/C2 context_system", False, f"raised {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- C3: user message
    rec = RecTok()
    try:
        ids = LR.ctx_ids(rec, "A", "fear", "cpu")
        check("C3 ctx_ids returns [1, plen] ids", ids.shape == (1, 4))
        check("C3 user message is GEN_PROMPT",
              rec.msgs[-1]["role"] == "user" and rec.msgs[-1]["content"] == C.GEN_PROMPT)
        check("C3 system message is the wording-A context",
              rec.msgs[0]["role"] == "system"
              and rec.msgs[0]["content"] == LR.context_system("A", "fear"))
    except Exception as e:
        check("C3 ctx_ids builds GEN_PROMPT chat", False, f"raised {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- S1: stream selection
    cap = dict(streams=[
        dict(gidx=0, concept="fear", strength=60, accepted=True, tokens=torch.tensor([1, 2, 3])),
        dict(gidx=1, concept="fear", strength=60, accepted=False, tokens=torch.tensor([1, 2])),
        dict(gidx=2, concept="fear", strength=40, accepted=True, tokens=torch.tensor([1, 2])),
        dict(gidx=3, concept="ocean", strength=0, accepted=True, tokens=torch.tensor([1, 2])),
        dict(gidx=4, concept="ocean", strength=60, accepted=True, tokens=torch.tensor([9])),
    ])
    try:
        got = LR.select_streams(cap, "injected")
        check("S1 injected: accepted smax, len>=2 only", [s["gidx"] for s in got] == [0],
              f"gidx = {[s.get('gidx') for s in got]}")
    except Exception as e:
        check("S1 injected: accepted smax, len>=2 only", False, f"raised {type(e).__name__}: {e}")

    bun = dict(streams=[
        dict(gidx=0, concept="fear", strength=1, accepted=True, tokens=np.array([1, 2, 3])),
        dict(gidx=1, concept="neutral", strength=0, accepted=True, tokens=np.array([1, 2])),
        dict(gidx=2, concept="fear", strength=1, accepted=False, tokens=np.array([1, 2])),
        dict(gidx=3, concept="anger", strength=1, accepted=True, tokens=np.array([7])),
    ])
    try:
        got = LR.select_streams(bun, "evoked")
        check("S1 evoked: accepted concept streams + s0 neutral, len>=2",
              [s["gidx"] for s in got] == [0, 1], f"gidx = {[s.get('gidx') for s in got]}")
    except Exception as e:
        check("S1 evoked: accepted concept streams + s0 neutral, len>=2", False,
              f"raised {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- S2: batch padding
    try:
        batch, lens = LR.pad_tokens([[5, 6, 7], np.array([8, 9]), torch.tensor([1, 2, 3])], pad_id=0)
        check("S2 pad_tokens -> [B, Tmax] long + true lengths",
              batch.shape == (3, 3) and batch.dtype == torch.long and lens == [3, 2, 3])
        check("S2 mixed dtypes pad identically",
              batch[0].tolist() == [5, 6, 7] and batch[1].tolist() == [8, 9, 0]
              and batch[2].tolist() == [1, 2, 3])
    except Exception as e:
        check("S2 pad_tokens -> [B, Tmax] long + true lengths", False,
              f"raised {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- L1/L2: LL arithmetic
    V = 5
    pred = torch.full((2, 3, V), -1e9)
    # stream 0 (len 3): tokens [1, 2, 0]; give each true token logit 0 vs -1e9 -> logprob ~ 0
    for t, tokid in enumerate([1, 2, 0]):
        pred[0, t, tokid] = 0.0
    # stream 1 (len 2): tokens [3, 4]; at t=0 make a 2-way tie (true vs other) -> logprob = -ln 2
    pred[1, 0, 3] = 0.0
    pred[1, 0, 0] = 0.0
    pred[1, 1, 4] = 0.0
    targets = torch.tensor([[1, 2, 0], [3, 4, 0]])
    try:
        ll = LR.ll_from_logits(pred, targets, [3, 2])
        check("L1 ll_from_logits returns float32 [B]",
              ll.shape == (2,) and ll.dtype == torch.float32)
        check("L1 certain stream: LL ~= 0", abs(float(ll[0])) < 1e-4, f"ll[0] = {float(ll[0])}")
        check("L1 2-way-tie first token: LL ~= -ln 2",
              abs(float(ll[1]) + np.log(2.0)) < 1e-4, f"ll[1] = {float(ll[1])}")
        pred_garbage = pred.clone()
        pred_garbage[1, 2, :] = torch.randn(V) * 50           # padded position of the len-2 stream
        targets_g = targets.clone()
        targets_g[1, 2] = 4
        ll_g = LR.ll_from_logits(pred_garbage, targets_g, [3, 2])
        check("L2 padded positions are inert", abs(float(ll_g[1]) - float(ll[1])) < 1e-6)
    except Exception as e:
        check("L1 ll_from_logits", False, f"raised {type(e).__name__}: {e}")

    # ------------------------------------------------------------ P1/P2: prefill cache formats
    # Box attempt 2 died: on the box stack model(...).past_key_values is a LEGACY TUPLE, not a
    # Cache object -- prefill must accept both (tuple passthrough; Cache -> to_legacy_cache()).
    class _Out:
        def __init__(self, pk):
            self.past_key_values = pk
            self.logits = torch.zeros((1, 4, 7), dtype=torch.bfloat16)

    class _StubModel:
        def __init__(self, pk):
            self._pk = pk

        def __call__(self, ids, **kw):
            return _Out(self._pk)

    legacy = ((torch.zeros(1, 2, 4, 3), torch.zeros(1, 2, 4, 3)),)

    class _CacheObj:
        def to_legacy_cache(self):
            return legacy

    try:
        kv, first = LR.prefill(_StubModel(legacy), torch.ones((1, 4), dtype=torch.long))
        check("P1 prefill accepts legacy-tuple past_key_values (box stack)",
              kv is legacy and first.shape == (1, 7) and first.dtype == torch.float32)
    except Exception as e:
        check("P1 prefill accepts legacy-tuple past_key_values (box stack)", False,
              f"raised {type(e).__name__}: {e}")
    try:
        kv2, _ = LR.prefill(_StubModel(_CacheObj()), torch.ones((1, 4), dtype=torch.long))
        check("P2 prefill converts Cache objects via to_legacy_cache", kv2 is legacy)
    except Exception as e:
        check("P2 prefill converts Cache objects via to_legacy_cache", False,
              f"raised {type(e).__name__}: {e}")

    # ------------------------------------------------------------ K1: attempt-6 fallback gates
    with open(os.path.join(REPO, "src", "lr_reader.py")) as f:
        check("K1 LR_SELFCHECK_FALLBACK print withholds raw exception text (no '{e}')",
              "{e}" not in f.read(),
              "a raw exception message can carry a labkit FATAL substring (e.g. CUDA OOM)")
    if not hasattr(LR, "score_batch"):
        check("K1 post-selfcheck KV-batch fallback seam exists", False,
              "src/lr_reader.py has no score_batch -- a KV path raising mid-run kills the box")
    else:
        import contextlib
        import io
        _orig_kv, _orig_cc = LR.score_batch_kv, LR.score_batch_concat
        try:
            calls = dict(kv=0, cc=0)

            def _kv_raises(*a, **k):
                calls["kv"] += 1
                raise RuntimeError("CUDA error: device-side assert triggered (synthetic)")

            def _cc(*a, **k):
                calls["cc"] += 1
                return "CC_LL"

            LR.score_batch_kv, LR.score_batch_concat = _kv_raises, _cc
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ll, use_kv = LR.score_batch(None, None, None, None, None, None, True)
            printed = buf.getvalue()
            check("K1 a raising KV batch falls back to the concat path (returns use_kv=False)",
                  ll == "CC_LL" and use_kv is False and calls == dict(kv=1, cc=1),
                  f"ll={ll!r} use_kv={use_kv!r} calls={calls}")
            bad = ("LR_DONE", "LR_READY", "CUDA error", "CUDA out of memory",
                   "Traceback (most recent call last)", "ModuleNotFoundError",
                   "torch.cuda.OutOfMemoryError")
            check("K1 fallback print is collision-safe (type name only, no marker/FATAL substring)",
                  "RuntimeError" in printed and not any(s in printed for s in bad),
                  repr(printed))
            with contextlib.redirect_stdout(io.StringIO()):
                ll2, uk2 = LR.score_batch(None, None, None, None, None, None, False)
            check("K1 concat mode passes straight through",
                  ll2 == "CC_LL" and uk2 is False and calls["kv"] == 1)
        except Exception as e:
            check("K1 score_batch fallback", False, f"raised {type(e).__name__}: {e}")
        finally:
            LR.score_batch_kv, LR.score_batch_concat = _orig_kv, _orig_cc

# ---------------------------------------------------------------- R1: temperature fit + bits
if AN is not None:
    rng = np.random.default_rng(0)
    K = 12
    n = 240
    y = np.arange(n) % K
    S_perfect = np.where(np.eye(K)[y].astype(bool), 40.0, -40.0)      # separable, LL-sum scale
    S_flat = np.zeros((n, K))
    try:
        tau = AN.fit_temperature(S_perfect, y, grid=AN.TAU_GRID)
        bits, top1 = AN.bits_top1(S_perfect, y, tau)
        check("R1 separable scores -> bits ~= log2(12)",
              abs(bits - np.log2(K)) < 0.05, f"bits = {bits:.3f}")
        check("R1 separable scores -> top-1 == 1.0", top1 == 1.0, f"top1 = {top1}")
        tau_f = AN.fit_temperature(S_flat, y, grid=AN.TAU_GRID)
        bits_f, top1_f = AN.bits_top1(S_flat, y, tau_f)
        check("R1 flat scores -> bits ~= 0", abs(bits_f) < 1e-6, f"bits = {bits_f}")
        S_half = S_perfect.copy()
        S_half[: n // 2] = 0.0                                        # half the rows uninformative
        _, top1_h = AN.bits_top1(S_half, y, 1.0)
        check("R1 top-1 == argmax accuracy",
              abs(top1_h - (0.5 + 0.5 / K)) < 0.06, f"top1 = {top1_h}")
    except Exception as e:
        check("R1 fit_temperature/bits_top1", False, f"raised {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- R2: stratified thirds
    try:
        y2 = np.repeat(np.arange(K), 9)                               # 9 per concept
        cal, ev = AN.split_thirds(y2, seed=3)
        cal, ev = np.asarray(cal), np.asarray(ev)
        check("R2 split covers all rows, disjoint",
              len(set(cal) & set(ev)) == 0 and len(cal) + len(ev) == len(y2))
        per = [int((y2[cal] == k).sum()) for k in range(K)]
        check("R2 floor(n/3) calibration rows per concept", per == [3] * K, f"per-concept {per}")
        cal_b, _ = AN.split_thirds(y2, seed=4)
        check("R2 seed changes the split", sorted(cal_b.tolist()) != sorted(cal.tolist()))
        y3 = np.array([0, 0, 1])                                      # floor(2/3)=0, floor(1/3)=0
        cal3, _ = AN.split_thirds(y3, seed=0)
        per3 = [int((y3[np.asarray(cal3)] == k).sum()) for k in (0, 1)]
        check("R2 min 1 calibration row per concept", per3 == [1, 1], f"per-concept {per3}")
    except Exception as e:
        check("R2 split_thirds", False, f"raised {type(e).__name__}: {e}")

    # ------------------------------------------------------------ R3: min_eval over ALL seeds
    try:
        y4 = np.repeat(np.arange(2), 6)                               # 2 concepts x 6 streams
        S4 = np.where(np.eye(2)[y4].astype(bool), 5.0, -5.0)
        _orig_split = AN.split_thirds

        def _skewed(y, seed):
            # seed 0 leaves only ONE eval row for concept 0; every later seed leaves 4
            cal = np.array([0, 1, 2, 3, 4, 6, 7]) if seed == 0 else np.array([0, 1, 6, 7])
            return cal, np.setdiff1d(np.arange(len(y)), cal)

        AN.split_thirds = _skewed
        try:
            cell = AN.evaluate_cell(S4, y4, seeds=range(3))
        finally:
            AN.split_thirds = _orig_split
        check("R3 min_eval_per_concept is the min over ALL seeds, not the last seed's",
              cell["min_eval_per_concept"] == 1,
              f"min_eval={cell['min_eval_per_concept']} (a last-seed-only value would be 4)")
    except Exception as e:
        check("R3 min_eval over all seeds", False, f"raised {type(e).__name__}: {e}")

    # ------------------------------------------------------------ R4: dead pre-assignment gone
    with open(os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis",
                           "lr_reader_offline.py")) as f:
        check("R4 dead 'concepts = None' pre-assignment removed (main reassigns unconditionally)",
              "concepts = None" not in f.read())

# ---------------------------------------------------------------- M1: done-marker collision
# Box attempt 4's lesson: labkit matches the done marker as a SUBSTRING of log lines, so the
# per-shard progress line "LR_DONE_SHARD ..." fired the "LR_DONE" done marker after the FIRST
# shard -- the box was declared done, pulled and torn down 1/9th of the way in. Only box_lr.py
# (the marker owner, final line) may ever emit that substring.
with open(os.path.join(REPO, "src", "lr_reader.py")) as f:
    check("M1 src/lr_reader.py never emits the done-marker substring 'LR_DONE'",
          "LR_DONE" not in f.read())

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
