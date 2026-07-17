"""RED-first unit tests for src/inject_tf_lr.py -- the 7a run-(2) inject-during-teacher-forcing
scorer (prereg Part B of reports/lr_scale_extend_prereg.md, DRAFT). No GPU: the model and the
hook seam are FAKES injected through the module's seams; the numerics under test are the
certified lr_reader/lr_grid function objects running on CPU tensors.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_inject_tf_lr.py

I1  select_injected: default = smax pool (lr_reader.select_streams parity); --strength pins a
    level; an absent level raises.
I2  neutral_streams returns the s0 pool only.
I3  steering_primitives: the capture's OWN stored vectors/alphas, keyed per concept; a missing
    vector or alpha key raises (never silently re-derived).
I4  the REAL HookSeam registers common._injection_hook on model.model.layers[layer] with
    prompt_len wiring: positions >= prompt_len get + alpha*vec, prompt positions stay clean
    (the generation-time 'gen' convention).
I5  score_pool: the neutral label registers NO hook; every concept label registers exactly one
    hook and removes it (even if scoring raises); records carry ll / ll_eos / ll_tok per label.
I6  eos rule: a stream ending in the tokenizer eos loses exactly that position from the eos-free
    PRIMARY (ll) while the with-eos SECONDARY (ll_eos) keeps it; T_noeos = T - 1.
I7  the injected-vs-neutral difference flows: with a fake model whose logits depend on the hook
    state, LL(label c) != LL(neutral), and assert_hook_live passes; with a DEAD seam (no state
    change) assert_hook_live raises.
I8  shard_paths carry the injection level (s124/s140 passes never collide); run_slug writes both
    shards atomically, resumes on existing shards, and the shard meta carries the NOTE #2
    pure-concept-channel comparability sentence.
I9  marker safety: the module's source contains no box READY/DONE/FATAL marker substring.
I10 Amendment-2 (2a) dose wiring: available_doses splits scoreable from degraded levels (a
    missing alpha/vector or an absent level degrades to not-scored, NEVER regenerated);
    run_doses scores every scoreable level in ONE model pass-set, writes the dose_plan
    disclosure JSON, embeds it in every scored shard's meta, and writes NO shard for a
    degraded dose.
I11 Amendment-2 (2b) bundle overrides: a bundle carrying streamset='expressed' names its
    shards expressed_TF*_s<lvl> (never colliding with the 2a injected shards) and a stored
    system_text overrides the PROMPT_VARIANTS context (token-identical to the expressed
    generation).
I12 include_s0=False keeps the s0 centering pool OUT of both shards (the disclosed
    Amendment-2 dose-pass trim).
"""
import contextlib
import io
import os
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


import common as K            # noqa: E402
import inject_tf_lr as ITF    # noqa: E402
import lr_reader as LR        # noqa: E402

CONCEPTS = ["celebration", "ocean", "fear"]
EOS = 2
D = 4          # hidden size of the fake vectors
V = 7          # fake vocab


def fake_bundle():
    """A covert_collect-shaped capture: strengths {0, 40, 60}, 3 concepts, stored steering
    primitives, streams incl. one ending in eos."""
    streams, gidx = [], 0
    for s in (0, 40, 60):
        for ci, c in enumerate(CONCEPTS):
            for j in range(2):
                toks = [1, 3, 4] if j == 0 else [5, 1, EOS]     # second stream ends in eos
                streams.append(dict(gidx=gidx, concept=c, concept_idx=ci, strength=s,
                                    tokens=np.asarray(toks), text="x", accepted=True))
                gidx += 1
    iv = {c: np.full(D, 0.1 * (i + 1), dtype=np.float32) for i, c in enumerate(CONCEPTS)}
    ia = {f"{c}|s{s}": float(s) / 100.0 for c in CONCEPTS for s in (40, 60)}
    return dict(streams=streams, concepts=list(CONCEPTS), strengths=[0, 40, 60],
                inject_vectors=iv, inject_alpha=ia, layer=5, model="fake", inject="gen",
                variant="orig")


B = fake_bundle()

# ---------------------------------------------------------------- I1/I2: selection
try:
    sel = ITF.select_injected(B)
    check("I1 default selection = smax (s60) pool, parity with lr_reader.select_streams",
          len(sel) == 6 and all(s["strength"] == 60 for s in sel)
          and [s["gidx"] for s in sel] == [s["gidx"] for s in LR.select_streams(B, "injected")])
    s40 = ITF.select_injected(B, strength=40)
    check("I1 --strength pins a level", len(s40) == 6 and all(s["strength"] == 40 for s in s40))
    try:
        ITF.select_injected(B, strength=99)
        check("I1 absent level raises", False, "no exception")
    except ValueError:
        check("I1 absent level raises", True)
    s0 = ITF.neutral_streams(B)
    check("I2 neutral_streams = the s0 pool only",
          len(s0) == 6 and all(s["strength"] == 0 for s in s0))
except Exception as e:
    check("I1/I2 selection", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I3: steering primitives
try:
    vecs, alphas = ITF.steering_primitives(B, 60)
    check("I3 vectors/alphas per concept from the capture's own primitives",
          set(vecs) == set(CONCEPTS) and abs(alphas["ocean"] - 0.6) < 1e-9
          and torch.is_tensor(vecs["fear"]) and vecs["fear"].dtype == torch.float32)
    bad = dict(B, inject_alpha={})
    try:
        ITF.steering_primitives(bad, 60)
        check("I3 missing alpha key raises (never re-derived)", False, "no exception")
    except KeyError:
        check("I3 missing alpha key raises (never re-derived)", True)
except Exception as e:
    check("I3 primitives", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I4: the REAL HookSeam
try:
    captured = {}

    class FakeLayer:
        def register_forward_hook(self, hook):
            captured["hook"] = hook
            h = types.SimpleNamespace(removed=False)
            h.remove = lambda: setattr(h, "removed", True)
            captured["handle"] = h
            return h

    fake_model = types.SimpleNamespace(model=types.SimpleNamespace(layers={5: FakeLayer()}))
    vec = torch.ones(D)
    handle = ITF.HookSeam.register(fake_model, 5, vec, 0.5, prompt_len=3)
    hs = torch.zeros(1, 6, D)
    out = captured["hook"](None, None, hs)
    check("I4 HookSeam wires _injection_hook: stream positions get +alpha*vec, prompt clean",
          torch.allclose(out[0, :3], torch.zeros(3, D))
          and torch.allclose(out[0, 3:], torch.full((3, D), 0.5)))
    handle.remove()
    check("I4 handle removes", captured["handle"].removed)
except Exception as e:
    check("I4 HookSeam", False, f"raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------- fake model + seam for I5-I8
class FakeModel:
    """Callable like the HF model on the concat path: logits [B, T, V]; a fake 'injection'
    state (bias) shifts the logit of token id 0, so hooked and unhooked LLs differ."""
    def __init__(self):
        self.device = "cpu"
        self.bias = 0.0

    def __call__(self, ids, attention_mask=None, **kw):
        Bn, T = ids.shape
        logits = torch.zeros(Bn, T, V)
        logits[..., 1] = self.bias          # shift some token's logit when 'injected'
        return types.SimpleNamespace(logits=logits)


class FakeSeam:
    """Hook seam that mutates the fake model's state; records register/remove pairing."""
    log = []

    @staticmethod
    def register(model, layer, vec, alpha, prompt_len):
        model.bias = float(alpha) * 10.0
        FakeSeam.log.append(("on", layer, prompt_len))
        h = types.SimpleNamespace()
        def _rm():
            model.bias = 0.0
            FakeSeam.log.append(("off", layer, prompt_len))
        h.remove = _rm
        return h


class DeadSeam(FakeSeam):
    @staticmethod
    def register(model, layer, vec, alpha, prompt_len):
        h = types.SimpleNamespace()
        h.remove = lambda: None
        return h


class FakeTok:
    eos_token_id = EOS

    def apply_chat_template(self, msgs, **kw):
        return torch.ones(1, 3, dtype=torch.long)


# ---------------------------------------------------------------- I5: score_pool hook discipline
try:
    FakeSeam.log = []
    model = FakeModel()
    ctx = torch.ones(1, 3, dtype=torch.long)
    streams = ITF.select_injected(B)
    hooks = {"neutral": None, "ocean": (torch.ones(D), 0.6)}
    recs = ITF.score_pool(model, ctx, streams, batch_n=4, eos_id=EOS, label_hooks=hooks,
                          hook_seam=FakeSeam, layer=5)
    ons = [e for e in FakeSeam.log if e[0] == "on"]
    offs = [e for e in FakeSeam.log if e[0] == "off"]
    check("I5 exactly one hook registered+removed (concept label only, neutral none)",
          len(ons) == 1 and len(offs) == 1 and ons[0][2] == 3, f"log={FakeSeam.log}")
    check("I5 records carry ll/ll_eos/ll_tok per label",
          all(set(r["ll"]) == {"neutral", "ocean"} and set(r["ll_tok"]) == {"neutral", "ocean"}
              for r in recs))
    check("I5 injected label LL differs from neutral (the vector acted)",
          all(abs(r["ll"]["ocean"] - r["ll"]["neutral"]) > 1e-6 for r in recs))
    check("I5 model state restored after the hooked label", model.bias == 0.0)
except Exception as e:
    check("I5 score_pool", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I6: eos rule
try:
    eos_rec = [r for r in recs if r["T_noeos"] == r["T"] - 1]
    noeos_rec = [r for r in recs if r["T_noeos"] == r["T"]]
    check("I6 T_noeos drops exactly the terminal eos", len(eos_rec) == 3 and len(noeos_rec) == 3)
    r = eos_rec[0]
    ll_tok = np.asarray(r["ll_tok"]["neutral"], dtype=np.float64)
    check("I6 ll (eos-free) = ll_eos minus the final token's LL, from the same forward",
          abs((r["ll_eos"]["neutral"] - r["ll"]["neutral"]) - float(ll_tok[-1])) < 1e-2
          and abs(sum(ll_tok) - r["ll_eos"]["neutral"]) < 1e-2)  # ll_tok is fp16 STORAGE
          # (Amendment 1's registered dtype); the fp32 sums in ll/ll_eos are the source of truth
except Exception as e:
    check("I6 eos rule", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I7: hook-live gate
try:
    model = FakeModel()
    probe = streams[0]
    d = ITF.assert_hook_live(model, ctx, probe, torch.ones(D), 0.6, 5, EOS, hook_seam=FakeSeam)
    check("I7 assert_hook_live passes on a live seam (|dLL| > 0)", d > 0)
    try:
        ITF.assert_hook_live(model, ctx, probe, torch.ones(D), 0.6, 5, EOS, hook_seam=DeadSeam)
        check("I7 a DEAD seam raises", False, "no exception")
    except RuntimeError as e:
        check("I7 a DEAD seam raises", "DEAD" in str(e))
except Exception as e:
    check("I7 hook-live gate", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I8: shards + resume + meta
try:
    pa = ITF.shard_paths("/tmp/o", "qwen2.5-7b", 124)
    pb = ITF.shard_paths("/tmp/o", "qwen2.5-7b", 140)
    check("I8 shard names carry the level; s124/s140 never collide",
          "_s124" in pa[0] and "_s140" in pb[0] and pa[0] != pb[0] and "TFV" in pa[0]
          and "TFN" in pa[1])
    with tempfile.TemporaryDirectory() as td:
        model = FakeModel()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pv, pn = ITF.run_slug(model, FakeTok(), B, "fake-slug", td, batch_n=4,
                                  hook_seam=FakeSeam)
        check("I8 run_slug writes both shards", os.path.exists(pv) and os.path.exists(pn))
        sv = torch.load(pv, map_location="cpu", weights_only=False)
        sn = torch.load(pn, map_location="cpu", weights_only=False)
        check("I8 TFV carries the concept labels, TFN only neutral",
              sv["contexts"] == CONCEPTS and sn["contexts"] == ["neutral"])
        check("I8 s0 centering pool rides (streams at level + s0)",
              len(sv["records"]) == 12
              and sum(1 for r in sv["records"] if r["strength"] == 0) == 6)
        check("I8 meta carries the NOTE #2 pure-concept comparability sentence",
              "PURE-CONCEPT" in sv["score"].upper() or "pure-concept" in sv["score"])
        with contextlib.redirect_stdout(buf):
            ITF.run_slug(model, FakeTok(), B, "fake-slug", td, batch_n=4, hook_seam=FakeSeam)
        check("I8 resume skips existing shards", "ITF_SKIP" in buf.getvalue())
except Exception as e:
    check("I8 shards/run_slug", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I10: dose wiring (2a)
try:
    scored, missing = ITF.available_doses(B, [40, 60, 50, 99])
    check("I10 available_doses: stored-primitive levels scoreable; a level without alphas or "
          "streams degrades to not-scored",
          scored == [40, 60] and [m["level"] for m in missing] == [50, 99]
          and all(m.get("reason") for m in missing))
    with tempfile.TemporaryDirectory() as td:
        model = FakeModel()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            paths, plan = ITF.run_doses(model, FakeTok(), B, "fake-slug", td,
                                        [40, 60, 50], batch_n=4, hook_seam=FakeSeam)
        check("I10 run_doses writes shard pairs for the scoreable levels only",
              len(paths) == 2
              and all(os.path.exists(p) for pair in paths for p in pair)
              and not os.path.exists(os.path.join(
                  td, "fake-slug__fake-slug__injected_TFV_s50.pt")))
        check("I10 the dose_plan disclosure: scored + not_scored (degrade, never regenerate)",
              plan["scored"] == [40, 60]
              and [m["level"] for m in plan["not_scored"]] == [50]
              and "never regenerated" in plan["note"].lower())
        djs = [f for f in os.listdir(td) if f.startswith("dose_plan_") and f.endswith(".json")]
        with open(os.path.join(td, djs[0])) as fh:
            import json as _json
            ondisk = _json.load(fh)
        check("I10 the disclosure JSON lands next to the shards and round-trips the plan",
              len(djs) == 1 and ondisk["scored"] == [40, 60]
              and [m["level"] for m in ondisk["not_scored"]] == [50])
        sv = torch.load(paths[0][0], map_location="cpu", weights_only=False)
        check("I10 every scored shard's meta embeds the dose_plan (the prereg's 'disclosed in "
              "the shard meta')",
              sv.get("dose_plan", {}).get("scored") == [40, 60]
              and [m["level"] for m in sv["dose_plan"]["not_scored"]] == [50])
except Exception as e:
    check("I10 dose wiring", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I11: expressed overrides (2b)
try:
    XB = dict(fake_bundle(), streamset="expressed", system_text="EXPRESSED SYSTEM TEXT")

    class CtxTok(FakeTok):
        def __init__(self):
            self.seen = []
        def apply_chat_template(self, msgs, **kw):
            self.seen.append(msgs)
            return torch.ones(1, 3, dtype=torch.long)

    with tempfile.TemporaryDirectory() as td:
        model = FakeModel()
        ct = CtxTok()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pv, pn = ITF.run_slug(model, ct, XB, "fake-slug", td, batch_n=4,
                                  strength=60, hook_seam=FakeSeam, include_s0=False)
        check("I11 expressed bundle -> expressed_TF* shard names (no 2a collision)",
              os.path.basename(pv) == "fake-slug__fake-slug__expressed_TFV_s60.pt"
              and os.path.basename(pn) == "fake-slug__fake-slug__expressed_TFN_s60.pt")
        sysmsgs = [m[0]["content"] for m in ct.seen if m and m[0].get("role") == "system"]
        check("I11 a stored system_text overrides PROMPT_VARIANTS (token-identical context to "
              "the expressed generation)",
              sysmsgs and all(s == "EXPRESSED SYSTEM TEXT" for s in sysmsgs))
        sv = torch.load(pv, map_location="cpu", weights_only=False)
        check("I11 shard meta carries the streamset", sv.get("streamset") == "expressed")
except Exception as e:
    check("I11 expressed overrides", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I12: include_s0=False
try:
    with tempfile.TemporaryDirectory() as td:
        model = FakeModel()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pv, pn = ITF.run_slug(model, FakeTok(), B, "fake-slug", td, batch_n=4,
                                  strength=60, hook_seam=FakeSeam, include_s0=False)
        sv = torch.load(pv, map_location="cpu", weights_only=False)
        sn = torch.load(pn, map_location="cpu", weights_only=False)
        check("I12 include_s0=False keeps the s0 pool out of BOTH shards (disclosed dose-pass "
              "trim)",
              len(sv["records"]) == 6 and len(sn["records"]) == 6
              and not any(int(r["strength"]) == 0 for r in sv["records"] + sn["records"]))
except Exception as e:
    check("I12 include_s0", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- I9: marker safety
try:
    src = open(os.path.join(REPO, "src", "inject_tf_lr.py")).read()
    BAD = ("LRG_READY", "LRG_DONE", "LRG_FATAL", "LRX_READY", "LRX_DONE", "LRX_FATAL",
           "LR_READY", "LR_DONE", "LR_FATAL", "MC_READY", "MC_DONE", "MC_FATAL",
           "LR72_READY", "LR72_DONE", "LR72_FATAL", "COLLECT_DONE", "COLLECT_FATAL")
    check("I9 module source carries no box marker substring",
          not any(m in src for m in BAD))
except Exception as e:
    check("I9 marker safety", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
