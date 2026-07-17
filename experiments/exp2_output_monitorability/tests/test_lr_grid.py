"""RED-first unit tests for the LR scale-grid build (prereg: reports/lr_scale_grid_prereg.md,
checklist Phase B: units B1, B2, B3, B5, B7). No model, no GPU -- stubs and synthetic tensors only.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_lr_grid.py

D* (B7)  deps_for: reader-family deps for the BOX env; the 4.46.3 pin must load the llama
         architecture (verified against the locally installed pin, not assumed). Amendment 4:
         the cross-family readers are Falcon3-{1B,3B,7B}-Instruct (registered ungated fallback;
         model_type=llama, so the same arch claim carries).
G* (B1)  alt-stream generation at 3B/7B: config-level reuse of exp3's collect_induction
         (arms=evoked_alt only, real-run feasibility floor, no smoke/pilot overrides; the s0
         neutral cell rides inside the pipeline), bundle path resolution repo-first.
T* (B3)  shared-tokenizer assert before teacher-forcing saved Qwen ids under another-size Qwen
         reader (identical ids on a sample of the stream texts, FATAL on mismatch).
E* (B5)  eos rule: eos-free PRIMARY + with-eos SECONDARY LL sums from ONE forward through the
         certified lr_reader numerics (synthetic stream where the two DIFFER).
S* (B2)  grid shard naming/resume/atomicity, marker collision safety (no marker a substring of
         another), conservative VRAM-aware batch defaults + the B8 util-gate seam, numerics
         reuse (lr_grid calls lr_reader, never reimplements), and the A1 Llama-context
         NotImplementedError seam (B4 pending the SCI review's A1 decision).
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOX_PATH = os.path.join(REPO, "experiments", "exp2_output_monitorability", "box_lr_grid.py")
try:
    BOX = load_module("box_lr_grid", BOX_PATH)
except Exception as e:
    BOX = None
    check("import box_lr_grid.py", False, f"{type(e).__name__}: {e}")

# ================================================================ D (B7): deps_for
if BOX is not None:
    try:
        grid_readers = ["qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
                        "falcon3-1b", "falcon3-3b", "falcon3-7b"]
        deps = BOX.deps_for(grid_readers)
        check("D1 grid readers (Qwen2.5 + Falcon3, both llama-arch-loadable) stay on the "
              "validated 4.46.3 pin",
              "transformers==4.46.3" in deps, f"deps = {deps}")
        check("D1 generation rides the same box: wordfreq present (word-free filter must be live)",
              "wordfreq" in deps, f"deps = {deps}")
        check("D1 base reader deps present (accelerate/numpy/huggingface_hub)",
              all(d in deps for d in ("accelerate", "numpy", "huggingface_hub")), f"deps = {deps}")
        deps3 = BOX.deps_for(["qwen3-1.7b"])
        check("D1 a qwen3 reader would bump off the pin (mirrors run_mc/run_labkit deps_for)",
              "transformers==4.46.3" not in deps3
              and any(d.startswith("transformers>=4.51") for d in deps3), f"deps = {deps3}")
    except Exception as e:
        check("D1 deps_for", False, f"raised {type(e).__name__}: {e}")

# D2: the pin actually loads the llama architecture -- checked against the INSTALLED
# transformers (the .venv used to run this test), never assumed. The qwen3 deps bug was ENV not
# code: reviews checked code while the box env couldn't load the arch. Amendment 4: the Falcon3
# readers ship model_type=llama / LlamaForCausalLM (verified from the cached configs,
# 2026-07-11), so these same checks certify the pin for them.
try:
    import transformers
    v = transformers.__version__
    if v == "4.46.3":
        from transformers import LlamaForCausalLM                                  # noqa: F401
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        check("D2 transformers==4.46.3 loads llama arch (LlamaForCausalLM imports; Falcon3's "
              "declared architecture)", True)
        check("D2 'llama' registered in CONFIG_MAPPING", "llama" in CONFIG_MAPPING)
        check("D2 'llama3' rope scaling init present (rope-scaling llama configs need it)",
              "llama3" in ROPE_INIT_FUNCTIONS)
    else:
        check("D2 local transformers matches the 4.46.3 pin (else re-verify the Llama arch claim)",
              False, f"installed {v} != 4.46.3 -- deps_for's pin claim is unverified locally")
except Exception as e:
    check("D2 Llama arch under the 4.46.3 pin", False, f"raised {type(e).__name__}: {e}")

# ================================================================ G (B1): alt generation 3B/7B
# Config-level reuse of the exp3 induction pipeline: the box invokes collect_induction.py
# UNCHANGED with --arms evoked_alt. Identical acceptance + word-free gates and the s0 neutral
# streams riding along are properties OF that pipeline (run_model collects the strength-0 neutral
# cell inside every arm; wordfreq-less real runs raise) -- the tests here pin that we invoke it
# unforked, with the real-run feasibility floor, and never with smoke/pilot sizing.
if BOX is not None:
    try:
        cmd, env = BOX.altgen_cmd("qwen2.5-3b")
        script = cmd[cmd.index("-u") + 1] if "-u" in cmd else cmd[1]
        check("G1 altgen invokes exp3's collect_induction.py (the script exists, unforked)",
              os.path.basename(script) == "collect_induction.py" and os.path.exists(script),
              f"cmd = {cmd}")
        check("G1 arms = evoked_alt ONLY (evoked exists at all sizes; named/secret not re-run)",
              "--arms" in cmd and cmd[cmd.index("--arms") + 1] == "evoked_alt"
              and cmd.count("evoked_alt") == 1 and "evoked" not in
              [c for i, c in enumerate(cmd) if cmd[max(0, i - 1)] != "--arms" and c == "evoked"],
              f"cmd = {cmd}")
        check("G1 feasibility gate at the real-run floor (24, = run_exp3's default)",
              "--min-per-class" in cmd and cmd[cmd.index("--min-per-class") + 1] == "24",
              f"cmd = {cmd}")
        check("G1 no smoke/pilot sizing (acceptance gates identical to the 1.5B alt run)",
              "--smoke" not in cmd and "--pilot" not in cmd, f"cmd = {cmd}")
        check("G1 --models routes the one slug", cmd[cmd.index("--models") + 1] == "qwen2.5-3b",
              f"cmd = {cmd}")
        check("G1 env: INTRO_MODEL=<slug>, INTRO_RUN_DIR under OUT/_ind/<slug>",
              env.get("INTRO_MODEL") == "qwen2.5-3b"
              and env.get("INTRO_RUN_DIR", "").endswith(os.path.join("_ind", "qwen2.5-3b")),
              f"env = {env}")
    except Exception as e:
        check("G1 altgen_cmd", False, f"raised {type(e).__name__}: {e}")

    try:
        p15 = BOX.alt_bundle_path("qwen2.5-1.5b")
        check("G2 1.5B alt bundle resolves to the existing repo copy (runs/_ind)",
              p15 == os.path.join(REPO, "runs", "_ind", "qwen2.5-1.5b", "data",
                                  "qwen2.5-1.5b-evoked_alt.pt") and os.path.exists(p15),
              f"path = {p15}")
        p3 = BOX.alt_bundle_path("qwen2.5-3b")
        check("G2 3B alt bundle (repo copy absent) resolves under OUT/_ind (S1 generates it)",
              p3 == os.path.join(BOX.OUT, "_ind", "qwen2.5-3b", "data",
                                 "qwen2.5-3b-evoked_alt.pt"),
              f"path = {p3}")
        gen_dir = os.path.dirname(BOX.altgen_cmd("qwen2.5-3b")[1]["INTRO_RUN_DIR"])
        check("G2 altgen writes where alt_bundle_path looks (INTRO_RUN_DIR/data/<slug>-evoked_alt.pt)",
              p3 == os.path.join(BOX.altgen_cmd("qwen2.5-3b")[1]["INTRO_RUN_DIR"],
                                 "data", "qwen2.5-3b-evoked_alt.pt"),
              f"bundle = {p3}, INTRO_RUN_DIR parent = {gen_dir}")
    except Exception as e:
        check("G2 alt_bundle_path", False, f"raised {type(e).__name__}: {e}")

    check("G3 altgen stage exists and resumes on an existing bundle (skip-if-present in source)",
          hasattr(BOX, "altgen_stage") and "alt_bundle_path" in
          __import__("inspect").getsource(BOX.altgen_stage) if hasattr(BOX, "altgen_stage")
          else False,
          "box_lr_grid has no altgen_stage")

# ================================================================ T (B3): shared-tokenizer assert
try:
    import lr_grid as G            # src/lr_grid.py (GPU module; imported CPU-side for pure fns)
except Exception as e:
    G = None
    check("import src/lr_grid.py", False, f"{type(e).__name__}: {e}")


class CountTok:
    """Stub tokenizer: text -> per-char ordinals (+shift); counts encode calls."""
    def __init__(self, shift=0, break_on=None):
        self.shift, self.break_on, self.calls = shift, break_on, 0

    def __call__(self, text, add_special_tokens=None):
        self.calls += 1
        ids = [ord(c) + self.shift for c in text]
        if self.break_on is not None and self.break_on in text:
            ids = ids + [999999]                       # divergent encode for one specific text
        class _R:                                       # noqa: E306
            pass
        r = _R()
        r.input_ids = ids
        return r


if G is not None:
    texts = [f"qx z fjm wpl kbt {i}" for i in range(50)]
    try:
        a, b = CountTok(), CountTok()
        G.assert_shared_tokenizer(a, b, texts, n=32)
        check("T1 identical tokenizers pass", True)
        check("T1 sample cap respected (n texts each, not the whole pool)",
              a.calls == 32 and b.calls == 32, f"calls = {a.calls}/{b.calls}")
    except Exception as e:
        check("T1 identical tokenizers pass", False, f"raised {type(e).__name__}: {e}")
    try:
        G.assert_shared_tokenizer(CountTok(), CountTok(shift=1), texts, n=8)
        check("T2 differing tokenizers raise (teacher-forcing saved ids would be meaningless)",
              False, "no exception raised")
    except RuntimeError as e:
        bad = ("LRG_READY", "LRG_DONE", "LRG_FATAL", "LR_READY", "LR_DONE", "LR_FATAL",
               "CUDA error", "CUDA out of memory", "Traceback (most recent call last)",
               "ModuleNotFoundError", "torch.cuda.OutOfMemoryError")
        check("T2 differing tokenizers raise (teacher-forcing saved ids would be meaningless)",
              True)
        check("T2 failure message is marker/FATAL-substring safe",
              not any(s in str(e) for s in bad), repr(str(e)))
    except Exception as e:
        check("T2 differing tokenizers raise (teacher-forcing saved ids would be meaningless)",
              False, f"wrong exception type {type(e).__name__}: {e}")
    try:
        G.assert_shared_tokenizer(CountTok(), CountTok(break_on=texts[20]), texts, n=32)
        check("T3 a single divergent text inside the sample trips the assert", False,
              "no exception raised")
    except RuntimeError:
        check("T3 a single divergent text inside the sample trips the assert", True)
    except Exception as e:
        check("T3 a single divergent text inside the sample trips the assert", False,
              f"wrong exception type {type(e).__name__}: {e}")

    try:
        check("T4 GRID_READERS carries the 6 prereg readers (Amendment 4: Falcon3 fallback)",
              set(G.GRID_READERS) == {"qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
                                      "falcon3-1b", "falcon3-3b", "falcon3-7b"},
              f"readers = {sorted(getattr(G, 'GRID_READERS', {}))}")
        import config as C_
        check("T4 Qwen reader hf_ids come from the config registry (single source of truth)",
              all(G.GRID_READERS[s] == C_.MODELS[s]["hf_id"]
                  for s in ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")))
        check("T4 Falcon3 hf_ids are the Amendment 4 Instruct models (ungated)",
              G.GRID_READERS["falcon3-1b"] == "tiiuae/Falcon3-1B-Instruct"
              and G.GRID_READERS["falcon3-3b"] == "tiiuae/Falcon3-3B-Instruct"
              and G.GRID_READERS["falcon3-7b"] == "tiiuae/Falcon3-7B-Instruct")
        check("T4 family() splits qwen/falcon3",
              G.family("qwen2.5-7b") == "qwen" and G.family("falcon3-7b") == "falcon3")
    except Exception as e:
        check("T4 GRID_READERS/family", False, f"raised {type(e).__name__}: {e}")

# ================================================================ E (B5): eos rule, ONE forward
# Registered: PRIMARY scoring excludes the terminal eos from all LL sums; the with-eos SECONDARY
# rides alongside (comparability with the 1.5B numbers). Both must come from the SAME forward
# through lr_reader's certified numerics. Synthetic stream chosen so the two DIFFER.
import numpy as np      # noqa: E402
import torch            # noqa: E402

import lr_reader as LR  # noqa: E402

if G is not None:
    try:
        got = G.noeos_lens([[5, 6, 9], [5, 6], np.array([9, 9]), torch.tensor([9, 5])], eos_id=9)
        check("E1 noeos_lens drops ONLY a terminal eos (mixed dtypes)",
              got == [2, 2, 1, 2], f"got {got}")
        check("E1 eos_id None -> lengths unchanged",
              G.noeos_lens([[5, 9]], eos_id=None) == [2])
    except Exception as e:
        check("E1 noeos_lens", False, f"raised {type(e).__name__}: {e}")

    Vv = 16                                            # > max synthetic token id (9)

    class _FlatModel:
        """Uniform logits -> every position's true-token logprob is exactly -ln(V)."""
        def __init__(self):
            self.calls = 0

        def __call__(self, ids, attention_mask=None, **kw):
            self.calls += 1
            class _O:                                   # noqa: E306
                pass
            o = _O()
            o.logits = torch.zeros((ids.shape[0], ids.shape[1], Vv))
            return o

    try:
        ctx = torch.ones((1, 3), dtype=torch.long)
        batch, lens = LR.pad_tokens([[5, 6, 9], [5, 6]], pad_id=0)   # stream 0 ends in eos=9
        lens_ne = G.noeos_lens([[5, 6, 9], [5, 6]], eos_id=9)
        m = _FlatModel()
        orig_ll = LR.ll_from_logits
        ll_free, ll_eos, use_kv = G.score_batch_dual(m, ctx, None, None, batch, lens, lens_ne,
                                                     use_kv=False)
        ln4 = float(np.log(Vv))
        check("E2 the synthetic eos-terminated stream: eos-free and with-eos DIFFER",
              abs(float(ll_free[0]) - float(ll_eos[0])) > 0.5,
              f"free={float(ll_free[0])} eos={float(ll_eos[0])}")
        check("E2 with-eos LL = -T*ln(V) (certified numerics, full length)",
              abs(float(ll_eos[0]) + 3 * ln4) < 1e-5 and abs(float(ll_eos[1]) + 2 * ln4) < 1e-5,
              f"ll_eos = {ll_eos.tolist()}")
        check("E2 eos-free LL = -(T-1)*ln(V) on the eos stream, unchanged on the other",
              abs(float(ll_free[0]) + 2 * ln4) < 1e-5 and abs(float(ll_free[1]) + 2 * ln4) < 1e-5,
              f"ll_free = {ll_free.tolist()}")
        check("E2 no-terminal-eos stream: the two sums are IDENTICAL",
              abs(float(ll_free[1]) - float(ll_eos[1])) < 1e-7)
        check("E2 ONE forward for both sums (never a second pass)", m.calls == 1,
              f"model called {m.calls}x")
        check("E2 use_kv passthrough", use_kv is False)
        check("E2 certified LR.ll_from_logits restored after the call",
              LR.ll_from_logits is orig_ll)
    except Exception as e:
        check("E2 score_batch_dual", False, f"raised {type(e).__name__}: {e}")

    # E3: a KV path that raises mid-run must still yield BOTH sums via the concat fallback
    # (lr_reader's attempt-6 gate), with the certified function restored.
    try:
        import contextlib
        import io
        _orig_kv = LR.score_batch_kv

        def _kv_raises(*a, **k):
            raise RuntimeError("synthetic KV failure")

        LR.score_batch_kv = _kv_raises
        m2 = _FlatModel()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ll_free2, ll_eos2, uk2 = G.score_batch_dual(m2, ctx, None, None, batch, lens,
                                                            lens_ne, use_kv=True)
        finally:
            LR.score_batch_kv = _orig_kv
        check("E3 raising KV path: concat fallback still returns both sums, use_kv flips off",
              uk2 is False and abs(float(ll_free2[0]) + 2 * float(np.log(Vv))) < 1e-5
              and abs(float(ll_eos2[0]) + 3 * float(np.log(Vv))) < 1e-5,
              f"uk={uk2} free={ll_free2.tolist()} eos={ll_eos2.tolist()}")
        check("E3 certified LR.ll_from_logits restored after the fallback",
              LR.ll_from_logits is orig_ll)
    except Exception as e:
        check("E3 KV-fallback dual", False, f"raised {type(e).__name__}: {e}")

    # E4: record wiring -- shards carry BOTH sums per context label.
    try:
        streams = [dict(gidx=7, concept="fear", strength=1, tokens=[5, 6, 9]),
                   dict(gidx=8, concept="neutral", strength=0, tokens=[5, 6])]
        recs = G.grid_records(streams)
        check("E4 grid_records: gidx/concept/strength/T + empty ll and ll_eos dicts",
              [r["gidx"] for r in recs] == [7, 8] and recs[0]["T"] == 3
              and recs[0]["ll"] == {} and recs[0]["ll_eos"] == {})
        G.record_lls(recs, 0, "fear", torch.tensor([-1.0, -2.0]), torch.tensor([-1.5, -2.0]))
        check("E4 record_lls stores eos-free under ll (PRIMARY) and with-eos under ll_eos",
              recs[0]["ll"]["fear"] == -1.0 and recs[0]["ll_eos"]["fear"] == -1.5
              and recs[1]["ll"]["fear"] == -2.0 and recs[1]["ll_eos"]["fear"] == -2.0)
    except Exception as e:
        check("E4 record wiring", False, f"raised {type(e).__name__}: {e}")

# ================================================================ S (B2): grid box + scoring loop
import ast                      # noqa: E402
import inspect                  # noqa: E402
from pathlib import Path        # noqa: E402

import config as C              # noqa: E402

GEN_MODELS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")
GRID_READER_LIST = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
                    "falcon3-1b", "falcon3-3b", "falcon3-7b")
SETS, CTXS = ("evoked", "evoked_alt"), ("N", "A", "B")


def print_strings(path):
    """[(concatenated string literals of one print(...) call, lineno)] -- test_marker_guard's
    helper, duplicated here because importing that module would run its checks + sys.exit."""
    with open(path) as f:
        tree = ast.parse(f.read())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            lits = [c.value for c in ast.walk(node)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            out.append(("".join(lits), node.lineno))
    return out


if G is not None and BOX is not None:
    # ---- S1: shard naming -- one atomic shard per (reader, generator, streamset, ctxset)
    try:
        names = [G.shard_path(Path("o"), r, m, ss, cs).name
                 for r in GRID_READER_LIST for m in GEN_MODELS for ss in SETS for cs in CTXS]
        check("S1 full grid = 108 shards, all names distinct",
              len(names) == 108 and len(set(names)) == 108, f"n={len(names)}")
        check("S1 shard names are filesystem-safe (no separators/spaces)",
              all("/" not in n and " " not in n for n in names))
        check("S1 evoked vs evoked_alt cells cannot collide (set name delimited)",
              G.shard_path(Path("o"), "qwen2.5-3b", "qwen2.5-7b", "evoked", "A").name
              != G.shard_path(Path("o"), "qwen2.5-3b", "qwen2.5-7b", "evoked_alt", "A").name)
    except Exception as e:
        check("S1 shard_path", False, f"raised {type(e).__name__}: {e}")

    # ---- S2: VRAM-aware conservative batch defaults + the B8 util-gate seam
    try:
        bat = {r: G.default_batch(r) for r in GRID_READER_LIST}
        check("S2 conservative defaults: every reader <= the certified 1.5B run's 16",
              all(1 <= b <= 16 for b in bat.values()), f"batches = {bat}")
        check("S2 big readers (7B) at the most conservative default",
              bat["qwen2.5-7b"] <= 8 and bat["falcon3-7b"] <= 8, f"batches = {bat}")
        check("S2 unknown size falls back conservative", G.default_batch("x-999b") <= 8)
        check("S2 util_gate_hook exists as the B8 seam (no-op, documented)",
              G.util_gate_hook("shard_done") is None and "B8" in (G.util_gate_hook.__doc__ or ""))
        check("S2 main() already calls the hook (B8 wires the body, not the call sites)",
              "util_gate_hook(" in inspect.getsource(G.main))
    except Exception as e:
        check("S2 batch defaults / util gate seam", False, f"raised {type(e).__name__}: {e}")

    # ---- S3: context construction -- Qwen readers = certified contexts; cross-family = A1
    # rendering (B4/B13 landed: the seam's NotImplementedError is replaced by the adjudicated
    # Amendment 1 implementation, deep-tested in test_llama_ctx.py; here we pin the ROUTING +
    # date pinning). Amendment 4: the cross-family readers are the falcon3 slugs; the date pin
    # is still passed (a verified no-op for Falcon3's date-free template).
    class RecTok:
        def __init__(self):
            self.msgs, self.kw = None, None

        def apply_chat_template(self, msgs, **kw):
            self.msgs, self.kw = msgs, kw
            return torch.ones((1, 4), dtype=torch.long)

    try:
        rec = RecTok()
        ids = G.ctx_ids_for("qwen2.5-3b", rec, "A", "fear", "cpu")
        check("S3 Qwen reader context == certified LR context (system + GEN_PROMPT)",
              ids.shape == (1, 4)
              and rec.msgs[0]["content"] == LR.context_system("A", "fear")
              and rec.msgs[-1]["content"] == C.GEN_PROMPT)
    except Exception as e:
        check("S3 Qwen reader context == certified LR context", False,
              f"raised {type(e).__name__}: {e}")
    try:
        lrec = RecTok()
        lids = G.ctx_ids_for("falcon3-1b", lrec, "A", "fear", "cpu")
        check("S3 falcon3 routes to the A1 rendering (reader's own template over persona + "
              "GEN_PROMPT, generation header on)",
              lids.shape == (1, 4)
              and lrec.msgs[0]["content"] == LR.context_system("A", "fear")
              and lrec.msgs[-1]["content"] == C.GEN_PROMPT
              and lrec.kw.get("add_generation_prompt") is True)
        check("S3 A1 render pins date_string (Amendment 1; a Falcon3 no-op, Amendment 4)",
              lrec.kw.get("date_string") == G.LLAMA_DATE, f"kw = {lrec.kw}")
    except Exception as e:
        check("S3 falcon3 A1 rendering routing", False, f"raised {type(e).__name__}: {e}")

    # ---- S4: bundle provenance gate
    try:
        good = dict(model="qwen2.5-3b", inject="evoked_alt", streams=[])
        G._assert_bundle(good, "p.pt", "qwen2.5-3b", "evoked_alt")
        check("S4 matching bundle passes provenance", True)
        for bad, why in ((dict(model="qwen2.5-7b", inject="evoked_alt"), "wrong generator"),
                         (dict(model="qwen2.5-3b", inject="evoked"), "wrong arm"),
                         (dict(model="qwen2.5-3b", inject="evoked_alt", variant="calm"),
                          "wrong prompt variant")):
            try:
                G._assert_bundle(bad, "p.pt", "qwen2.5-3b", "evoked_alt")
                check(f"S4 provenance mismatch raises ({why})", False, "no exception")
            except AssertionError:
                check(f"S4 provenance mismatch raises ({why})", True)
    except Exception as e:
        check("S4 provenance gate", False, f"raised {type(e).__name__}: {e}")

    # ---- J (control (b)): the INJECTED stream set (exp1 covert_collect capture) --------------
    # The scale-grid extension that measures LR on INJECTED streams at 3B/7B (the 1.5B LR run
    # read injected x A = 0.002). The injected streams come from covert_collect.pt (exp1 capture,
    # inject == 'gen', per-stream concept + tokens, strength == smax) -- a DIFFERENT bundle schema
    # than the _ind arms. They score under the concept's NATURAL PERSONA contexts (A/B wordings)
    # vs neutral -- the SAME construction lr_reader.select_streams("injected") + LR.CTX_SETS used
    # at 1.5B (NOT a "the secret word is X" context). Reuses the certified scoring unchanged.
    try:
        check("J1 'injected' is a whitelisted stream set (parse_bundle_spec accepts it)",
              "injected" in G.BUNDLE_SETS, f"BUNDLE_SETS = {G.BUNDLE_SETS}")
        gen, ss, path = G.parse_bundle_spec("qwen2.5-3b:injected:/runs/qwen2.5-3b/data/"
                                            "covert_collect.pt")
        check("J1 parse_bundle_spec round-trips an injected spec",
              gen == "qwen2.5-3b" and ss == "injected"
              and path == "/runs/qwen2.5-3b/data/covert_collect.pt", f"({gen},{ss},{path})")
    except Exception as e:
        check("J1 injected whitelist / parse", False, f"raised {type(e).__name__}: {e}")

    # J2: injected is NOT a secret arm -- it routes through the natural-persona A/B/N grid
    # (LR.CTX_SETS), exactly like evoked, never through the SW/SS/SM secret ctx.
    try:
        check("J2 injected is not a SECRET arm (routes through the natural A/B/N grid)",
              "injected" not in G.SECRET_CTX and "injected" not in G.SECRET_ARMS)
        check("J2 injected is NOT in SECRET_CTX so main() gives it LR.CTX_SETS (N,A,B)",
              G.SECRET_CTX.get("injected") is None)
    except Exception as e:
        check("J2 injected ctx routing", False, f"raised {type(e).__name__}: {e}")

    # J3: injected context construction == the certified natural-persona context (A/B via
    # LR.context_system, neutral via the concept-None baseline) -- byte-identical to the 1.5B
    # injected x A / x B / x N cells. ctx_ids_for on a Qwen reader must hand the natural persona
    # system + GEN_PROMPT (NEVER a secret-arm baseline).
    try:
        rec = RecTok()
        ids = G.ctx_ids_for("qwen2.5-3b", rec, "A", "fear", "cpu")
        check("J3 injected-under-A ctx == certified natural persona A (LR.context_system)",
              ids.shape == (1, 4)
              and rec.msgs[0]["content"] == LR.context_system("A", "fear")
              and rec.msgs[-1]["content"] == C.GEN_PROMPT)
    except Exception as e:
        check("J3 injected context construction", False, f"raised {type(e).__name__}: {e}")

    # J4: the DIFFERENT covert_collect schema. inject == 'gen' (not a streamset name), so
    # _assert_bundle's arm check must ACCEPT 'gen' for the injected set (the 1.5B lr_reader run's
    # _assert_provenance never checked inject at all); model + orig-variant still gate.
    try:
        cap = dict(model="qwen2.5-3b", inject="gen", variant="orig", streams=[])
        G._assert_bundle(cap, "cap.pt", "qwen2.5-3b", "injected")
        check("J4 injected capture (inject='gen') passes provenance for the injected set", True)
        for bad, why in ((dict(model="qwen2.5-7b", inject="gen", variant="orig"),
                          "wrong generator"),
                         (dict(model="qwen2.5-3b", inject="gen", variant="calm"),
                          "wrong prompt variant")):
            try:
                G._assert_bundle(bad, "cap.pt", "qwen2.5-3b", "injected")
                check(f"J4 injected provenance mismatch raises ({why})", False, "no exception")
            except AssertionError:
                check(f"J4 injected provenance mismatch raises ({why})", True)
    except Exception as e:
        check("J4 injected schema provenance", False, f"raised {type(e).__name__}: {e}")

    # J5: stream selection reuses lr_reader's certified select_streams("injected") -- the max
    # strength (word-free accepted) pool, len>=2, exactly the 1.5B run's construction.
    try:
        cap = dict(model="qwen2.5-3b", inject="gen", variant="orig", streams=[
            dict(gidx=0, concept="fear", strength=0, tokens=[5, 6, 7], accepted=True),
            dict(gidx=1, concept="fear", strength=60, tokens=[5, 6, 7], accepted=True),
            dict(gidx=2, concept="ocean", strength=60, tokens=[8, 9], accepted=True),
            dict(gidx=3, concept="ocean", strength=60, tokens=[8], accepted=True),  # len<2 drop
            dict(gidx=4, concept="anger", strength=40, tokens=[1, 2, 3], accepted=True),  # !smax
        ])
        sel = LR.select_streams(cap, "injected")
        gidxs = sorted(s["gidx"] for s in sel)
        check("J5 injected selection = accepted strength==smax, len>=2 (certified select_streams)",
              gidxs == [1, 2], f"selected gidxs = {gidxs}")
        check("J5 main() selects the injected pool via LR.select_streams (no reimplementation)",
              "LR.select_streams(" in inspect.getsource(G.main))
    except Exception as e:
        check("J5 injected stream selection", False, f"raised {type(e).__name__}: {e}")

    # ---- S5: marker safety (the LR attempt-4 / substring-collision bug class)
    ALL_BOX_MARKERS = ("LRG_READY", "LRG_DONE", "LRG_FATAL",
                       "LR_READY", "LR_DONE", "LR_FATAL",
                       "MC_READY", "MC_DONE", "MC_FATAL",
                       "ELICIT_READY", "ELICIT_DONE", "ELICIT_FATAL",
                       "GAUGE_READY", "GAUGE_DONE", "GAUGE_FATAL",
                       "ANALYZE_READY", "ANALYZE_DONE", "ANALYZE_FATAL",
                       "COLLECT_DONE", "COLLECT_FATAL", "MODEL_READY", "ALL_DONE",
                       "BASELINE_DONE")
    FATAL_SUBSTRINGS = ("CUDA error", "CUDA out of memory", "Traceback (most recent call last)",
                        "ModuleNotFoundError", "torch.cuda.OutOfMemoryError")
    coll = [(a, b) for a in ALL_BOX_MARKERS for b in ALL_BOX_MARKERS if a != b and a in b]
    check("S5 no box marker is a substring of another (grep-verified, incl. LRG_*)",
          not coll, f"collisions: {coll}")
    grid_src_path = os.path.join(REPO, "src", "lr_grid.py")
    hits = [f"line {ln}: {sub!r}" for text, ln in print_strings(grid_src_path)
            for sub in ALL_BOX_MARKERS + FATAL_SUBSTRINGS if sub in text]
    check("S5 src/lr_grid.py prints carry NO box marker / labkit FATAL substring",
          not hits, "; ".join(hits))
    with open(grid_src_path) as f:
        gsrc = f.read()
    check("S5 src/lr_grid.py never contains its box's own markers at all (M1 parity)",
          all(m not in gsrc for m in ("LRG_READY", "LRG_DONE", "LRG_FATAL")))
    with open(BOX_PATH) as f:
        bsrc = f.read()
    box_prints = print_strings(BOX_PATH)
    for m in ("LRG_READY", "LRG_DONE", "LRG_FATAL"):
        n = sum(m in text for text, _ in box_prints)
        check(f"S5 box owns {m}: printed exactly once", n == 1, f"printed {n}x")
    _done_at = bsrc.find("print(\"LRG_DONE\"")
    _guard_at = bsrc.find("if __name__ ==")
    check("S5 LRG_DONE is the box's final act (printed inside the __main__ guard)",
          _done_at > _guard_at >= 0, f"positions: print at {_done_at}, guard at {_guard_at}")
    bhits = [f"line {ln}: {sub!r}" for text, ln in box_prints
             for sub in FATAL_SUBSTRINGS if sub in text]
    check("S5 box prints carry no labkit FATAL substring", not bhits, "; ".join(bhits))

    # ---- S6: certified-code reuse + atomic/resume structure (source-level pins)
    try:
        msrc = inspect.getsource(G.main)
        check("S6 stream selection is lr_reader's (LR.select_streams)",
              "LR.select_streams(" in msrc)
        check("S6 scoring goes through score_batch_dual (certified numerics, both eos sums)",
              "score_batch_dual(" in msrc)
        check("S6 context prefill is lr_reader's (LR.prefill)", "LR.prefill(" in msrc)
        check("S6 registered KV self-check present at lr_reader parity (tol + concat reference)",
              "LR.SELFCHECK_TOL" in msrc and "LR.score_batch_concat(" in msrc)
        check("S6 B3 gate wired before scoring (assert_shared_tokenizer in main)",
              "assert_shared_tokenizer(" in msrc)
        check("S6 shards written tmp -> os.replace (atomic; presence = done)",
              '.with_suffix(".tmp")' in msrc and "os.replace(" in msrc)
        check("S6 resume: existing shard skipped with LRG_SKIP", "LRG_SKIP" in msrc)
        check("S6 no reimplemented numerics in lr_grid (no log_softmax/ll_from_logits def)",
              "log_softmax" not in gsrc and "def ll_from_logits" not in gsrc
              and "def score_batch_kv" not in gsrc)
    except Exception as e:
        check("S6 main structure", False, f"raised {type(e).__name__}: {e}")

    # ---- S7: box grid stage -- specs/env/shard bookkeeping consistent with lr_grid
    try:
        specs = BOX.bundle_specs()
        check("S7 12 bundle specs (evoked + evoked_alt + secret_word + secret_sustain at 3 "
              "sizes; B15)", len(specs) == 12, f"specs = {specs}")
        ALL_SETS = SETS + ("secret_word", "secret_sustain")
        parsed = [G.parse_bundle_spec(s) for s in specs]
        check("S7 specs parse round-trip (gen, streamset, abs path)",
              all(g in GEN_MODELS and ss in ALL_SETS and os.path.isabs(p)
                  for g, ss, p in parsed),
              f"parsed = {parsed}")
        check("S7 evoked bundles point at the fetched repo copies (runs/_ind)",
              all(p.startswith(os.path.join(REPO, "runs", "_ind"))
                  for g, ss, p in parsed if ss == "evoked"))
        shards = BOX.shards_for("qwen2.5-3b")
        check("S7 33 shards per injected-diagonal qwen reader (30 pre-control-(b) + 3 injected "
              "N/A/B)", len(shards) == 33, f"{len(shards)}")
        want = {G.shard_path(Path(os.path.join(BOX.OUT, "lr_grid")), "qwen2.5-3b", m, ss, cs).name
                for m in GEN_MODELS for ss in SETS for cs in CTXS}
        check("S7 box shard bookkeeping matches lr_grid.shard_path exactly (no drift)",
              want <= {os.path.basename(s) for s in shards})
        # B4/B13: 108 primary + 54 llama raw + 6 prose; B15 adds 72 secret primary (6 readers x
        # 3 gens x 2 sets x 2 ctx) + 36 llama secret raw + 2 E5 (1.5B reader) = 278. Control (b)
        # adds the injected self-diagonal at 3B/7B ONLY (2 readers x N/A/B = 6) = 284 total.
        check("S7 full run = 284 shard files (278 + 6 injected diagonal at 3B/7B)",
              sum(len(BOX.shards_for(r)) for r in BOX.READERS) == 284,
              f"{sum(len(BOX.shards_for(r)) for r in BOX.READERS)}")
        cmd, env = BOX.grid_cmd("falcon3-1b")
        check("S7 grid_cmd targets src/lr_grid.py with --reader + 12 --bundle specs (falcon "
              "readers get NO injected cell -- the diagonal is Qwen self-read)",
              os.path.basename(cmd[cmd.index("-u") + 1]) == "lr_grid.py"
              and cmd[cmd.index("--reader") + 1] == "falcon3-1b"
              and cmd.count("--bundle") == 12, f"cmd = {cmd}")
        check("S7 grid env sets INTRO_RUN_DIR only -- NEVER INTRO_MODEL (falcon slugs are not in "
              "the config registry; setting it would crash config's import assert)",
              env.get("INTRO_RUN_DIR") == BOX.OUT and "INTRO_MODEL" not in env, f"env = {env}")
        check("S7 fetch list: 3 evoked + the 1.5B alt + 3 secret_word + 2 injected (3B/7B "
              "covert_collect; B15's 3B/7B alt and all secret_sustain are S1/S1b-generated)",
              len(BOX.FETCHES) == 9
              and sum("evoked_alt" in f for f, _ in BOX.FETCHES) == 1
              and sum("secret_word" in f for f, _ in BOX.FETCHES) == 3
              and sum("-gen.pt" in f for f, _ in BOX.FETCHES) == 2, f"{BOX.FETCHES}")
    except Exception as e:
        check("S7 box grid stage", False, f"raised {type(e).__name__}: {e}")

    # ---- JB (control (b)): injected@{3b,7b} in the box S0 fetch + bundle_specs + shards --------
    try:
        # bundle_specs is reader-scoped: the injected covert_collect spec rides ONLY the 3B and
        # 7B readers (self-diagonal: reader == generator == that size), matching the self-read
        # framing (the question is self-legibility of injection). The 1.5B reader (its injected
        # cell exists from the certified LR run) and every falcon reader get NO injected spec.
        for r in ("qwen2.5-3b", "qwen2.5-7b"):
            specs = BOX.bundle_specs(reader=r)
            inj = [s for s in specs if G.parse_bundle_spec(s)[1] == "injected"]
            check(f"JB1 {r} bundle_specs carries exactly one injected spec (self-diagonal)",
                  len(inj) == 1, f"specs = {specs}")
            g, ss, p = G.parse_bundle_spec(inj[0])
            check(f"JB1 {r} injected spec is the reader's OWN covert_collect capture",
                  g == r and p.endswith(os.path.join(r, "data", "covert_collect.pt")),
                  f"({g},{ss},{p})")
        for r in ("qwen2.5-1.5b", "falcon3-1b", "falcon3-7b"):
            specs = BOX.bundle_specs(reader=r)
            check(f"JB1 {r} gets NO injected spec (diagonal is 3B/7B Qwen self-read only)",
                  not any(G.parse_bundle_spec(s)[1] == "injected" for s in specs),
                  f"specs = {specs}")
        # bundle_specs() with no reader stays the pre-control-(b) 12 (injected is reader-scoped).
        check("JB1 bundle_specs() (no reader) unchanged at 12 (injected is reader-scoped)",
              len(BOX.bundle_specs()) == 12, f"{len(BOX.bundle_specs())}")
        # shards: injected diagonal N/A/B (Qwen readers -> no _raw) present for 3B/7B, absent 1.5B.
        for r in ("qwen2.5-3b", "qwen2.5-7b"):
            names = {os.path.basename(s) for s in BOX.shards_for(r)}
            want_inj = {f"{r}__{r}__injected_{cs}.pt" for cs in CTXS}
            check(f"JB2 {r} shards include the injected self-diagonal N/A/B (no _raw)",
                  want_inj <= names and f"{r}__{r}__injected_A_raw.pt" not in names,
                  f"missing = {want_inj - names}")
        n15 = {os.path.basename(s) for s in BOX.shards_for("qwen2.5-1.5b")}
        check("JB2 1.5B reader has NO injected shard (its injected cell is the certified LR run)",
              not any("injected" in n for n in n15))
        # S0 fetch: the 3B/7B captures come from HF at the dataset-root <slug>-gen.pt path (the
        # 1.5B LR run's fetch name; box_confound already pulls these three) -> local
        # runs/<slug>/data/covert_collect.pt (where lr_reader/box_confound read them).
        inj_fetch = [(f, d) for f, d in BOX.FETCHES if "-gen.pt" in f]
        check("JB3 S0 fetches injected 3B + 7B captures from HF <slug>-gen.pt",
              {f for f, _ in inj_fetch} == {"qwen2.5-3b-gen.pt", "qwen2.5-7b-gen.pt"},
              f"{inj_fetch}")
        check("JB3 injected fetch dest = runs/<slug>/data/covert_collect.pt (local read path)",
              all(d.endswith(os.path.join(f.split("-gen")[0], "data", "covert_collect.pt"))
                  for f, d in inj_fetch), f"{inj_fetch}")
        check("JB3 the injected spec path == the S0 fetch dest (fetched file is what's scored)",
              {G.parse_bundle_spec(s)[2] for r in ("qwen2.5-3b", "qwen2.5-7b")
               for s in BOX.bundle_specs(reader=r)
               if G.parse_bundle_spec(s)[1] == "injected"} == {d for _, d in inj_fetch})
        # grid_cmd for a 3B/7B reader carries the extra injected --bundle (13 total).
        cmd3, _ = BOX.grid_cmd("qwen2.5-3b")
        check("JB4 grid_cmd for a 3B/7B reader carries 13 --bundle specs (12 + injected)",
              cmd3.count("--bundle") == 13, f"n={cmd3.count('--bundle')}")
    except Exception as e:
        check("JB injected box wiring", False, f"raised {type(e).__name__}: {e}")

# ================================================================ F (TECH-SF2): resume feasibility
# collect_induction saves the bundle BEFORE check_min_per_class runs, so a feasibility-FATALed
# run leaves a complete-looking bundle behind; altgen_stage's have-bundle resume branch must
# re-run the min-per-class check (CPU-cheap) before treating it as done, re-raising the same
# feasibility FATAL. collect_induction.py itself stays untouched.
import tempfile                 # noqa: E402

_F_CONCEPTS = ["celebration", "ocean", "fear", "silence"]


def _alt_bundle(n_per, tok_len=16):
    streams, gidx = [], 0
    for ci, c in enumerate(_F_CONCEPTS):
        for _ in range(n_per):
            streams.append(dict(gidx=gidx, concept=c, concept_idx=ci, arm="evoked_alt",
                                tokens=np.arange(4), text="x", deg={}, accepted=True,
                                strength=1, gen_topk=[0] * tok_len))
            gidx += 1
    return dict(model="qwen2.5-3b", inject="evoked_alt", concepts=_F_CONCEPTS,
                strengths=[1], streams=streams)


if BOX is not None:
    try:
        with tempfile.TemporaryDirectory() as td:
            ok_p = os.path.join(td, "ok.pt")
            thin_p = os.path.join(td, "thin.pt")
            torch.save(_alt_bundle(BOX.ALT_MIN_PER_CLASS), ok_p)
            torch.save(_alt_bundle(3), thin_p)
            BOX.assert_alt_bundle_feasible("qwen2.5-3b", path=ok_p)
            check("F1 a bundle meeting the real-run floor passes the resume re-check", True)
            try:
                BOX.assert_alt_bundle_feasible("qwen2.5-3b", path=thin_p)
                check("F1 a thin loaded bundle re-raises the SAME feasibility FATAL", False,
                      "no exception")
            except RuntimeError as e:
                check("F1 a thin loaded bundle re-raises the SAME feasibility FATAL",
                      "feasibility gate FAILED" in str(e), f"{e}")
            BOX.assert_alt_bundle_feasible("qwen2.5-3b", smoke=True, path=thin_p)
            check("F1 smoke keeps the floor off (min-per-class 0, matching altgen_cmd)", True)
        check("F1 altgen_stage's have-bundle branch runs the re-check before counting the "
              "bundle done",
              "assert_alt_bundle_feasible" in inspect.getsource(BOX.altgen_stage))
    except Exception as e:
        check("F1 resume feasibility re-check", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)


# W1 (smoke attempt 1, $0.01): the parity check must never read wordfreq.__version__ -- newer
# wordfreq drops the attribute on-box; package metadata is the robust read. Source-pinned.
_src_w = open(_os.path.join(_os.path.dirname(__file__), "..", "box_lr_grid.py")).read()
check("W1 parity check reads wordfreq version via importlib.metadata, never __version__",
      "wordfreq.__version__" not in _src_w and 'importlib.metadata.version("wordfreq")' in _src_w)
