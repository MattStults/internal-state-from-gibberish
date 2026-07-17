"""RED-first unit tests for src/expressed_collect.py -- the Amendment-2 (2b) expressed-injection
generation cell (prereg reports/lr_scale_extend_prereg.md Amendment 2, 2026-07-14: 1.5B, doses
{s20, s60}, concept vector injected DURING generation + the sustain-s1 suffix in its ORIGINAL
"this feeling" wording composed with the word-free STRONG_SYSTEM). No GPU: generation and model
are FAKES injected through the module's seams; the reuse targets (covert_collect.gen_clean,
primers_v2.SUSTAIN_SUFFIXES) are checked as the SAME objects, never re-typed.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_expressed_2b.py

X1  prompt composition BYTE-test: compose_expressed_system() == SUSTAIN_SUFFIXES['s1'] + the
    "\\n\\n" join + STRONG_SYSTEM (primers.compose_system's frozen convention: primer first,
    anti-word block LAST); the suffix is primers_v2's own VERBATIM object -- "this feeling"
    present, the primers_v3 secret_word-substituted derivative ABSENT; no concept word
    anywhere (asserted at compose time, a wordy strong system raises).
X2  stored_primitives: the capture's OWN (vector, alpha) with the scaling convention VERIFIED
    (alpha == strength/||v||, covert_collect's write); a drifted alpha raises; a missing
    vector/alpha key raises (never silently re-derived).
X3  assert_vectors_match: the two source captures must carry identical concept vectors
    (deterministic extraction, same model/layer); a mismatch refuses.
X4  collect() generation seam: the DEFAULT generator is covert_collect.gen_clean (the same
    function object -- reuse, not reimplementation); a fake gen_fn receives inject_mode='gen',
    the composed system, and per-dose (vector, alpha) from the DOSE'S OWN source capture
    (s20 from the e1 capture, s60 from the main capture).
X5  bundle schema: capture-schema keys (streams incl. rejected, concepts, strengths=doses,
    inject_vectors, inject_alpha, layer, inject='gen', model), plus streamset='expressed' +
    system_text (so the run-(2) self-read scores under the token-identical context) + gen_topk
    riding each accepted stream; per-(dose, concept) cell shards RESUME (no regeneration).
X6  covert_collect.gen_clean inject_override: the override's (vector, alpha) drive the hook
    (generation-only convention: prompt positions clean), concept_vector_blog is NEVER called;
    strength 0 stays uninjected.
X7  cfg: full = 12 concepts x {20, 60} x target 24, tokens 128, gen_topk 64 (the Amendment-2
    cell); smoke = a tiny s20 slice.
X8  marker safety: no box READY/DONE/FATAL marker substring in the module source.
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


import config as C            # noqa: E402
import covert_collect as CC   # noqa: E402
import expressed_collect as XC  # noqa: E402
import primers_v2 as PV2      # noqa: E402
import primers_v3 as PV3      # noqa: E402

D = 6


def src_bundle(level, concepts, vecs=None):
    """A covert_collect-shaped source capture carrying stored steering primitives at `level`
    with the REAL alpha convention (alpha = strength/||v||)."""
    vecs = vecs or {c: np.full(D, 0.1 * (i + 1), dtype=np.float32)
                    for i, c in enumerate(concepts)}
    ia = {f"{c}|s{level}": float(level) / float(np.linalg.norm(v))
          for c, v in vecs.items()}
    return dict(inject_vectors=dict(vecs), inject_alpha=ia, layer=5, model="qwen2.5-1.5b",
                inject="gen", variant="orig", concepts=list(concepts),
                strengths=[0, level])


CONCEPTS = ["ocean", "fear"]

# ---------------------------------------------------------------- X1: composition byte-test
try:
    sysm = XC.compose_expressed_system()
    check("X1 BYTE identity: SUSTAIN_SUFFIXES['s1'] + '\\n\\n' + STRONG_SYSTEM (primer first, "
          "anti-word block last -- the frozen primers convention)",
          sysm == PV2.SUSTAIN_SUFFIXES["s1"] + "\n\n" + C.STRONG_SYSTEM)
    check("X1 the ORIGINAL 'this feeling' wording, verbatim from primers_v2 (never retyped)",
          sysm.startswith(PV2.SUSTAIN_SUFFIXES["s1"]) and "this feeling" in sysm)
    check("X1 NOT the secret_word-substituted primers_v3 derivative",
          "the secret word" not in sysm and PV3.SECRET_SUSTAIN_SUFFIX not in sysm)
    check("X1 no concept word anywhere in the composed system",
          not any(c.lower() in sysm.lower() for c in C.COVERT_CONCEPTS))
    try:
        XC.compose_expressed_system("think about the ocean")
        check("X1 a concept-wordy strong system RAISES at compose time", False, "no exception")
    except RuntimeError:
        check("X1 a concept-wordy strong system RAISES at compose time", True)
    check("X1 module pins suffix s1 + the 1.5B slug + doses {20, 60}",
          XC.SUFFIX_KEY == "s1" and XC.SLUG == "qwen2.5-1.5b" and XC.DOSES == (20, 60))
except Exception as e:
    check("X1 composition", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- X2: stored primitives
try:
    b20 = src_bundle(20, CONCEPTS)
    v, a = XC.stored_primitives(b20, "ocean", 20)
    want = 20.0 / float(np.linalg.norm(b20["inject_vectors"]["ocean"]))
    check("X2 returns the capture's own vector + alpha, convention verified",
          np.allclose(v, b20["inject_vectors"]["ocean"]) and abs(a - want) < 1e-9)
    bad = dict(b20, inject_alpha={"ocean|s20": 2.0 * want,
                                  "fear|s20": b20["inject_alpha"]["fear|s20"]})
    try:
        XC.stored_primitives(bad, "ocean", 20)
        check("X2 a drifted alpha (2x the convention) RAISES", False, "no exception")
    except RuntimeError:
        check("X2 a drifted alpha (2x the convention) RAISES", True)
    for broken, what in ((dict(b20, inject_vectors={}), "vector"),
                         (dict(b20, inject_alpha={}), "alpha")):
        try:
            XC.stored_primitives(broken, "ocean", 20)
            check(f"X2 a missing {what} key RAISES (never re-derived)", False, "no exception")
        except KeyError:
            check(f"X2 a missing {what} key RAISES (never re-derived)", True)
except Exception as e:
    check("X2 primitives", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- X3: cross-capture identity
try:
    b20 = src_bundle(20, CONCEPTS)
    b60 = src_bundle(60, CONCEPTS)
    check("X3 identical vectors across captures pass",
          XC.assert_vectors_match(b20, b60, CONCEPTS) is True)
    b60_bad = src_bundle(60, CONCEPTS,
                         vecs={c: np.full(D, 9.9, dtype=np.float32) for c in CONCEPTS})
    try:
        XC.assert_vectors_match(b20, b60_bad, CONCEPTS)
        check("X3 mismatched source vectors REFUSE (never mixed silently)", False,
              "no exception")
    except RuntimeError:
        check("X3 mismatched source vectors REFUSE (never mixed silently)", True)
except Exception as e:
    check("X3 vector identity", False, f"raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------- X4/X5: collect() seams
def make_fake_gen(calls):
    def fake_gen(model, tok, concept, strength, layer, g, system, inject_mode="gen",
                 inject_override=None):
        calls.append(dict(concept=concept, strength=strength, layer=layer, system=system,
                          inject_mode=inject_mode, inject_override=inject_override))
        v, a = inject_override
        deg = dict(word_rate=0.0, repetition=0.1, non_latin=0.0, spacing=0.2)
        out = [dict(tokens=torch.tensor([1, 2, 3]), text="qx zjf wpl", deg=deg, accepted=True,
                    gen_topk=[dict(ids=np.asarray([1, 2], dtype=np.int32),
                                   logp=np.asarray([-1.0, -2.0], dtype=np.float16))] * 3),
               dict(tokens=torch.tensor([4, 4, 4, 4]), text="aaaa aaaa", accepted=False,
                    deg=dict(word_rate=0.0, repetition=0.9, non_latin=0.0, spacing=0.2),
                    gen_topk=None)]
        return out, v, a
    return fake_gen


try:
    check("X4 the DEFAULT generator is covert_collect.gen_clean (the SAME function object -- "
          "reuse, not reimplementation)", XC.GEN_FN is CC.gen_clean)
    calls = []
    with tempfile.TemporaryDirectory() as td:
        g = dict(concepts=CONCEPTS, doses=(20, 60), target_clean=1, max_gen=2, tokens=8,
                 gen_batch=2, gen_topk=2)
        sources = {20: src_bundle(20, CONCEPTS), 60: src_bundle(60, CONCEPTS)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            bp = XC.collect(model=None, tok=None, sources=sources, outdir=td, g=g,
                            gen_fn=make_fake_gen(calls))
        check("X4 one generation call per (dose, concept), inject_mode='gen', the composed "
              "system", len(calls) == 4
              and all(c["inject_mode"] == "gen" for c in calls)
              and all(c["system"] == XC.compose_expressed_system() for c in calls)
              and [c["strength"] for c in calls] == [20, 20, 60, 60])
        by_dose = {(c["strength"], c["concept"]): c["inject_override"] for c in calls}
        a20 = 20.0 / float(np.linalg.norm(sources[20]["inject_vectors"]["ocean"]))
        a60 = 60.0 / float(np.linalg.norm(sources[60]["inject_vectors"]["ocean"]))
        check("X4 per-dose primitives come from the DOSE'S OWN source capture (s20 <- e1, "
              "s60 <- main; alpha convention)",
              abs(float(by_dose[(20, "ocean")][1]) - a20) < 1e-6
              and abs(float(by_dose[(60, "ocean")][1]) - a60) < 1e-6
              and np.allclose(np.asarray(by_dose[(20, "ocean")][0]),
                              sources[20]["inject_vectors"]["ocean"]))
        b = torch.load(bp, map_location="cpu", weights_only=False)
        check("X5 bundle: capture schema + the Amendment-2 fields",
              b["model"] == "qwen2.5-1.5b" and b["inject"] == "gen"
              and b["streamset"] == "expressed"
              and b["system_text"] == XC.compose_expressed_system()
              and b["concepts"] == CONCEPTS and sorted(b["strengths"]) == [20, 60]
              and set(b["inject_alpha"]) == {"ocean|s20", "fear|s20", "ocean|s60", "fear|s60"}
              and set(b["inject_vectors"]) == set(CONCEPTS) and b["layer"] == 5)
        check("X5 streams: kept AND rejected recorded, concept_idx aligned, gen_topk rides "
              "the accepted",
              len(b["streams"]) == 8
              and sum(s["accepted"] for s in b["streams"]) == 4
              and all(s["concept_idx"] == CONCEPTS.index(s["concept"])
                      for s in b["streams"])
              and all(s["gen_topk"] for s in b["streams"] if s["accepted"])
              and len({s["gidx"] for s in b["streams"]}) == 8)
        check("X5 bundle lands at expressed/qwen2.5-1.5b-expressed.pt",
              bp == XC.bundle_path(td) and bp.endswith(
                  os.path.join("expressed", "qwen2.5-1.5b-expressed.pt")))
        n_before = len(calls)
        with contextlib.redirect_stdout(buf):
            XC.collect(model=None, tok=None, sources=sources, outdir=td, g=g,
                       gen_fn=make_fake_gen(calls))
        check("X5 resume: existing cell shards are NEVER regenerated", len(calls) == n_before)
    # cross-capture mismatch refuses inside collect too
    calls2 = []
    with tempfile.TemporaryDirectory() as td:
        bad_sources = {20: src_bundle(20, CONCEPTS),
                       60: src_bundle(60, CONCEPTS,
                                      vecs={c: np.full(D, 7.0, dtype=np.float32)
                                            for c in CONCEPTS})}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                XC.collect(model=None, tok=None, sources=bad_sources, outdir=td, g=g,
                           gen_fn=make_fake_gen(calls2))
            check("X5 collect refuses mismatched source captures ($0-fail gate)", False,
                  "no exception")
        except RuntimeError:
            check("X5 collect refuses mismatched source captures ($0-fail gate)",
                  not calls2, "generation ran before the gate")
except Exception as e:
    check("X4/X5 collect seams", False, f"raised {type(e).__name__}: {e}")


# ---------------------------------------------------------------- X6: gen_clean override seam
class HookLayer:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        h = types.SimpleNamespace(removed=False)
        h.remove = lambda: setattr(h, "removed", True)
        return h


class GenModel:
    def __init__(self, layer):
        self.model = types.SimpleNamespace(layers={0: layer})

    def generate(self, rep, **kw):
        b = rep.shape[0]
        gen = torch.tensor([[7, 8, 9, 2]]).repeat(b, 1)
        steps = tuple(torch.zeros(b, 11) for _ in range(4))
        return types.SimpleNamespace(sequences=torch.cat([rep, gen], dim=1), logits=steps)


class GenTok:
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        return "qx zjf wpl kbt"


try:
    orig_chat, orig_cvb = CC.K.chat_ids, CC.K.concept_vector_blog

    def _boom(*a, **k):
        raise AssertionError("concept_vector_blog called despite inject_override")
    CC.K.chat_ids = lambda tok, user, system=None, **kw: torch.ones(1, 3, dtype=torch.long)
    CC.K.concept_vector_blog = _boom
    try:
        layer = HookLayer()
        model = GenModel(layer)
        g = dict(target_clean=1, max_gen=2, tokens=4, gen_batch=2, gen_topk=2)
        vec = torch.arange(D, dtype=torch.float32)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out, v, a = CC.gen_clean(model, GenTok(), "ocean", 20, 0, g, "SYS",
                                     inject_mode="gen", inject_override=(vec, 0.5))
        check("X6 override drives the injection: stored vector/alpha returned, derivation "
              "NEVER called", torch.allclose(v, vec) and a == 0.5 and len(out) >= 1)
        hs = torch.zeros(1, 6, D)
        hooked = layer.hooks[0](None, None, hs)
        check("X6 generation-only convention: prompt positions clean, generated positions "
              "+ alpha*vec",
              torch.allclose(hooked[0, :3], torch.zeros(3, D))
              and torch.allclose(hooked[0, 3:], 0.5 * vec.expand(3, D)))
        layer2 = HookLayer()
        with contextlib.redirect_stdout(buf):
            out0, v0, a0 = CC.gen_clean(GenModel(layer2), GenTok(), "ocean", 0, 0, g, "SYS",
                                        inject_mode="gen", inject_override=(vec, 0.5))
        check("X6 strength 0 stays uninjected (no hook, no primitives returned)",
              v0 is None and a0 is None and not layer2.hooks)
    finally:
        CC.K.chat_ids, CC.K.concept_vector_blog = orig_chat, orig_cvb
except Exception as e:
    check("X6 gen_clean override", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- X7: cfg
try:
    full = XC.cfg(False)
    smoke = XC.cfg(True)
    check("X7 full cfg = the Amendment-2 cell: 12 concepts x (20, 60) x target 24, 128 tokens, "
          "gen_topk 64 (word-free filter rides gen_clean)",
          full["concepts"] == C.COVERT_CONCEPTS and full["doses"] == (20, 60)
          and full["target_clean"] == 24 and full["tokens"] == 128
          and full["gen_topk"] == 64)
    check("X7 smoke cfg is a tiny s20 slice",
          len(smoke["concepts"]) <= 3 and smoke["doses"] == (20,)
          and smoke["target_clean"] <= 4 and smoke["max_gen"] <= 16)
except Exception as e:
    check("X7 cfg", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- X8: marker safety
try:
    src = open(os.path.join(REPO, "src", "expressed_collect.py")).read()
    BAD = ("LRG_READY", "LRG_DONE", "LRG_FATAL", "LRX_READY", "LRX_DONE", "LRX_FATAL",
           "LR_READY", "LR_DONE", "LR_FATAL", "MC_READY", "MC_DONE", "MC_FATAL",
           "LR72_READY", "LR72_DONE", "LR72_FATAL", "COLLECT_DONE", "COLLECT_FATAL",
           "MODEL_READY")
    check("X8 module source carries no box marker substring", not any(m in src for m in BAD))
except Exception as e:
    check("X8 marker safety", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
