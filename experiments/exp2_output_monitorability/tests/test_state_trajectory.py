"""RED-first unit tests for src/state_trajectory.py (launch-review blockers B1-B3 + balanced cap).
No model, no GPU -- synthetic vectors/tokens and a stub tokenizer.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_state_trajectory.py

B1  neutral streams (collect_induction records concept="neutral") must re-forward under the arm's
    strength-0 baseline system (compose_system(None, ...)), not KeyError inside primers.
B2  the injected arm's hook must receive the RAW saved vector with the saved alpha
    (alpha = strength/||v_raw||, so ||alpha*v|| == strength) -- NOT the unit-normalized projection
    vector (that under-doses ~40x: ||delta|| ~= 1.5 instead of 60).
B3  saved stream tokens may be list[int] (collect_induction) OR 1-D torch tensors (exp1 captures);
    both must produce the identical [1, T] long tensor (torch.tensor([tensor]) raises TypeError).
CAP the per-arm stream cap must be balanced per concept (bundles are concept-major, so a global
    prefix cap starves late-listed concepts entirely).
G1  the gauge arm ("gauge:<base>") must build its context from compose_gauge_system (induction text
    ALONE, no anti-word block) -- routing it through compose_system would re-forward the gauge texts
    under a system prompt they were never generated under.
G2  neutral gauge records (concept="neutral") must map to compose_gauge_system(None, ...).
G3  the gauge user message is GAUGE_PROBE (free association), NOT C.GEN_PROMPT (the word-free stream
    prompt); non-gauge arms must keep GEN_PROMPT.
G4  gauge streams are synthesized from bundle["gauge"] texts with synthetic sequential gidx, and
    re-tokenized WITHOUT special tokens (they must append after the chat template exactly like
    generated tokens do).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

import config as C            # noqa: E402
import primers_v2 as P        # noqa: E402
import state_trajectory as ST  # noqa: E402

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


class StubTok:
    """Minimal tokenizer stub for K.chat_ids: returns a fixed [1,4] id tensor."""
    def apply_chat_template(self, msgs, **kw):
        return torch.ones((1, 4), dtype=torch.long)


# ---------------------------------------------------------------- B1: neutral concept in _ctx_ids
try:
    ids = ST._ctx_ids(StubTok(), "pilot:sustained_s1", "neutral", "cpu")
    check("B1 _ctx_ids(concept='neutral', arm='pilot:sustained_s1') does not raise",
          ids.shape == (1, 4))
except Exception as e:
    check("B1 _ctx_ids(concept='neutral', arm='pilot:sustained_s1') does not raise", False,
          f"raised {type(e).__name__}: {e}")

want_neutral = P.compose_system(None, C.STRONG_SYSTEM, arm="sustained_s1")
try:
    got = ST.system_text("pilot:sustained_s1", "neutral")
    check("B1 system_text('pilot:...','neutral') == compose_system(None, ...)", got == want_neutral)
    check("B1 system_text passes real concepts through",
          ST.system_text("evoked", "fear") == P.compose_system("fear", C.STRONG_SYSTEM, arm="evoked"))
    check("B1 injected/s0 arms use the word-free system regardless of concept",
          ST.system_text("injected", "fear") == C.STRONG_SYSTEM
          and ST.system_text("s0", "neutral") == C.STRONG_SYSTEM)
except Exception as e:
    check("B1 system_text('pilot:...','neutral') == compose_system(None, ...)", False,
          f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- B2: injected hook dose
d = 100
v_raw = np.full(d, 4.05, dtype=np.float32)                 # ||v_raw|| = 40.5
raw_norm = float(np.linalg.norm(v_raw))
inject_vectors = {"fear": v_raw}
inject_alpha = {"fear|s60": 60.0 / raw_norm}               # covert_collect: alpha = strength/||v_raw||
try:
    v, alpha = ST.injection_input(inject_vectors, inject_alpha, "fear", 60)
except AttributeError:
    # current code path (state_trajectory.py:59-62): main() unit-normalizes inject_vectors,
    # trajectory() feeds that normalized vector to the hook with the saved alpha.
    v = torch.tensor(v_raw / raw_norm, dtype=torch.float32)
    alpha = inject_alpha["fear|s60"]
delivered = float(torch.linalg.vector_norm(v.float() * alpha))
check("B2 hook receives ||alpha*v|| == 60 (saved strength), not ~1.5",
      abs(delivered - 60.0) < 1e-3, f"delivered ||alpha*v|| = {delivered:.3f}")

# ---------------------------------------------------------------- B3: token dtype robustness
def _toks(tokens):
    fn = getattr(ST, "stream_token_ids", None)
    if fn is not None:
        return fn(tokens, device="cpu")
    return torch.tensor([tokens], device="cpu")            # current code path (state_trajectory.py:55)

t_list = _toks([5, 6, 7])
check("B3 list[int] tokens -> [1,T] long tensor",
      t_list.shape == (1, 3) and t_list.dtype == torch.long)
try:
    t_tensor = _toks(torch.tensor([5, 6, 7]))
    check("B3 1-D torch-tensor tokens (exp1 capture) accepted",
          t_tensor.shape == (1, 3) and t_tensor.dtype == torch.long)
    check("B3 list and tensor tokens produce identical ids", torch.equal(t_list, t_tensor))
except Exception as e:
    check("B3 1-D torch-tensor tokens (exp1 capture) accepted", False,
          f"raised {type(e).__name__}: {e}")
    check("B3 list and tensor tokens produce identical ids", False, "unreachable (raise above)")

t_np = _toks(np.array([5, 6, 7]))
check("B3 np-array tokens (collect_induction) accepted", torch.equal(t_list, t_np))

# ---------------------------------------------------------------- CAP: per-concept balanced cap
concepts13 = [f"c{i}" for i in range(12)] + ["neutral"]    # 12 concepts + neutral, concept-major
pool = [dict(concept=c, gidx=i) for c in concepts13 for i in range(30)]
fn = getattr(ST, "balanced_pool", None)
capped = fn(pool, 25) if fn is not None else pool[:300]    # current code path (pool[:max_streams])
seen = {}
for s in capped:
    seen[s["concept"]] = seen.get(s["concept"], 0) + 1
check("CAP all 13 classes (12 concepts + neutral) survive the cap",
      set(seen) == set(concepts13), f"covered {len(seen)}/13: {sorted(seen)}")
check("CAP per-class count == 25", all(n == 25 for n in seen.values()), f"counts {seen}")

# ---------------------------------------------------------------- G1/G2: gauge arm system routing
try:
    got = ST.system_text("gauge:evoked", "fear")
    check("G1 system_text('gauge:evoked','fear') == compose_gauge_system('fear', arm='evoked')",
          got == P.compose_gauge_system("fear", arm="evoked"))
    check("G1 gauge system has NO anti-word block",
          C.STRONG_SYSTEM not in got and "RANDOM LETTERS" not in got)
except Exception as e:
    check("G1 system_text('gauge:evoked','fear') == compose_gauge_system('fear', arm='evoked')",
          False, f"raised {type(e).__name__}: {e}")
    check("G1 gauge system has NO anti-word block", False, "unreachable (raise above)")

try:
    check("G2 system_text('gauge:evoked','neutral') == compose_gauge_system(None, ...)",
          ST.system_text("gauge:evoked", "neutral") == P.compose_gauge_system(None, arm="evoked"))
except Exception as e:
    check("G2 system_text('gauge:evoked','neutral') == compose_gauge_system(None, ...)",
          False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- G3: gauge user message = GAUGE_PROBE
import primers as P0  # noqa: E402  (frozen exp3 prereg module; GAUGE_PROBE lives here)


class RecTok:
    """Records the chat messages so the test can inspect the USER message _ctx_ids builds."""
    def __init__(self):
        self.msgs = None

    def apply_chat_template(self, msgs, **kw):
        self.msgs = msgs
        return torch.ones((1, 4), dtype=torch.long)


rec = RecTok()
try:
    ST._ctx_ids(rec, "gauge:evoked", "fear", "cpu")
    user_msg = rec.msgs[-1]
    check("G3 gauge user message is GAUGE_PROBE, not GEN_PROMPT",
          user_msg["role"] == "user" and user_msg["content"] == P0.GAUGE_PROBE
          and user_msg["content"] != C.GEN_PROMPT,
          f"user content = {user_msg['content'][:60]!r}")
except Exception as e:
    check("G3 gauge user message is GAUGE_PROBE, not GEN_PROMPT", False,
          f"raised {type(e).__name__}: {e}")
rec2 = RecTok()
ST._ctx_ids(rec2, "evoked", "fear", "cpu")
check("G3 non-gauge arms keep GEN_PROMPT", rec2.msgs[-1]["content"] == C.GEN_PROMPT)

# ---------------------------------------------------------------- G4: gauge pool + re-tokenization
fake_bundle = {"gauge": {"fear": ["aa bb", "cc dd"], "neutral": ["ee ff"], "anger": ["gg hh"]}}
fn = getattr(ST, "gauge_pool", None)
if fn is None:
    check("G4 gauge_pool builds records from bundle['gauge']", False, "ST.gauge_pool missing")
    check("G4 gauge gidx synthetic + sequential", False, "ST.gauge_pool missing")
else:
    gp = fn(fake_bundle)
    check("G4 gauge_pool builds records from bundle['gauge']",
          len(gp) == 4 and {s["concept"] for s in gp} == {"fear", "neutral", "anger"}
          and all("text" in s for s in gp))
    check("G4 gauge gidx synthetic + sequential", [s["gidx"] for s in gp] == list(range(4)),
          f"gidx = {[s.get('gidx') for s in gp]}")


class RetokTok:
    """Records the add_special_tokens kwarg; returns fixed ids."""
    def __init__(self):
        self.kwargs = None

    def __call__(self, text, **kw):
        self.kwargs = kw
        return {"input_ids": [11, 12, 13]}


fn = getattr(ST, "retokenized", None)
if fn is None:
    check("G4 retokenized passes add_special_tokens=False", False, "ST.retokenized missing")
    check("G4 retokenized returns list[int] ids", False, "ST.retokenized missing")
else:
    rt = RetokTok()
    ids = fn(rt, "aa bb")
    check("G4 retokenized passes add_special_tokens=False",
          rt.kwargs.get("add_special_tokens") is False, f"kwargs = {rt.kwargs}")
    check("G4 retokenized returns list[int] ids",
          ids == [11, 12, 13] and all(isinstance(i, int) for i in ids))

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
