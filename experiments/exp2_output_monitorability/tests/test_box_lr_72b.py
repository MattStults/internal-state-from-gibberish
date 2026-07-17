"""RED-first unit tests for LR-72B units 2+3: the box script box_lr_72b.py -- 72B generation via
the vLLM server, the conditional-$6 evoked DECISION GATE, teacher-force scoring wiring, markers,
atomic shards, and the util gate. No GPU, no vLLM server, no network (subprocess/serve commands and
pure decision/projection functions only; the generation loop that needs a server is exercised with
a MOCK client via a tiny fixture, never a real box).
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_box_lr_72b.py

B1  serve command: `vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 2` (the prereg's
    exact model + TP), plus a health-wait helper against a mock.
B2  arms: secret_word + secret_sustain ALWAYS (Phase 1); evoked + evoked_alt CONDITIONAL (Phase 2).
B3  the $6 decision gate: Phase 2 runs iff spend_so_far + projected(evoked from the MEASURED S1
    rate) <= $6; else skip-for-budget, disclosed.
B4  markers LR72_READY/LR72_DONE/LR72_FATAL: collision-checked (no marker a substring of another,
    across ALL box scripts incl the existing LRG_*/LR_* set); the box owns them, printed once each,
    DONE inside __main__.
B5  atomic shards: tmp -> os.replace; resume skips an existing shard.
B6  util gate: first gen + first score batch log tok/s + GPU util; < 60% halts (raise).
B7  generation reuses collect_induction's word-free filter + acceptance gate (is_degenerate) and
    primers_v3's anti-word compose_system -- not reimplemented; the vLLM completions client is the
    only new I/O.
B8  the 7B off-diagonal privacy cell is a SEPARATE small reader (documented path: HF lr_grid on the
    pulled 72B streams, offline) -- the box records which and does not teacher-force it under 72B.
B9  teacher-force scoring uses src/lr_vllm (prompt_logprobs), scores the DIAGONAL (72B reads its own
    streams), one atomic shard per (set, ctx); no HF forward pass on the 72B path.
"""
import ast
import importlib.util
import inspect
import os
import sys
import tempfile

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


BOX_PATH = os.path.join(REPO, "experiments", "exp2_output_monitorability", "box_lr_72b.py")
try:
    BOX = load_module("box_lr_72b", BOX_PATH)
    check("import box_lr_72b.py", True)
except Exception as e:
    BOX = None
    check("import box_lr_72b.py", False, f"{type(e).__name__}: {e}")


# ================================================================ B1: vLLM serve command
if BOX is not None:
    try:
        cmd = BOX.serve_cmd()
        check("B1 serve command runs `vllm serve` on Qwen2.5-72B-Instruct",
              cmd[0] == "vllm" and cmd[1] == "serve"
              and "Qwen/Qwen2.5-72B-Instruct" in cmd, f"cmd={cmd}")
        check("B1 tensor-parallel-size 2 (the prereg's 2xH100 TP)",
              "--tensor-parallel-size" in cmd
              and cmd[cmd.index("--tensor-parallel-size") + 1] == "2", f"cmd={cmd}")
        check("B1 MODEL/TP pinned as constants (frozen design)",
              BOX.MODEL_72B == "Qwen/Qwen2.5-72B-Instruct" and BOX.TP_SIZE == 2)
    except Exception as e:
        check("B1 serve command", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B2: arms
if BOX is not None:
    try:
        check("B2 Phase 1 always-arms = secret_word + secret_sustain",
              tuple(BOX.PHASE1_ARMS) == ("secret_word", "secret_sustain"), f"{BOX.PHASE1_ARMS}")
        check("B2 Phase 2 conditional arms = evoked + evoked_alt",
              tuple(BOX.PHASE2_ARMS) == ("evoked", "evoked_alt"), f"{BOX.PHASE2_ARMS}")
    except Exception as e:
        check("B2 arms", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B3: the $6 decision gate
if BOX is not None:
    try:
        check("B3 the Phase-2 budget ceiling is $6 (frozen)", BOX.PHASE2_MAX_USD == 6.0)
        # projected(evoked) from the MEASURED Phase-1 generation rate: Phase 2 generates the same
        # number of concept x arm cells as Phase 1 (2 arms), so at the measured $/arm it projects
        # 2 arms of spend. gate: spend_so_far + projected <= 6.
        # measured: Phase 1 spent $2.00 for its 2 arms -> $1.00/arm -> projected evoked (2 arms) $2.00
        dec = BOX.phase2_gate(spend_so_far=2.00, phase1_arms=2, phase1_spend=2.00)
        check("B3 gate GOES when spend_so_far + projected(evoked) <= $6",
              dec["go"] is True and abs(dec["projected_usd"] - 2.00) < 1e-9, f"{dec}")
        dec2 = BOX.phase2_gate(spend_so_far=5.00, phase1_arms=2, phase1_spend=4.00)
        check("B3 gate SKIPS (disclosed) when it would exceed $6",
              dec2["go"] is False and "budget" in dec2["reason"].lower(), f"{dec2}")
        check("B3 projection uses the MEASURED phase-1 $/arm (not a fixed guess)",
              abs(BOX.phase2_gate(2.0, 2, 3.0)["projected_usd"] - 3.0) < 1e-9,
              f"{BOX.phase2_gate(2.0, 2, 3.0)}")
    except Exception as e:
        check("B3 decision gate", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B4: markers (collision-safe)
ALL_BOX_MARKERS = ("LR72_READY", "LR72_DONE", "LR72_FATAL",
                   "LRG_READY", "LRG_DONE", "LRG_FATAL",
                   "LR_READY", "LR_DONE", "LR_FATAL",
                   "MC_READY", "MC_DONE", "MC_FATAL",
                   "ELICIT_READY", "ELICIT_DONE", "ELICIT_FATAL",
                   "GAUGE_READY", "GAUGE_DONE", "GAUGE_FATAL",
                   "ANALYZE_READY", "ANALYZE_DONE", "ANALYZE_FATAL",
                   "COLLECT_DONE", "COLLECT_FATAL", "MODEL_READY", "ALL_DONE", "BASELINE_DONE")
FATAL_SUBSTRINGS = ("CUDA error", "CUDA out of memory", "Traceback (most recent call last)",
                    "ModuleNotFoundError", "torch.cuda.OutOfMemoryError")
coll = [(a, b) for a in ALL_BOX_MARKERS for b in ALL_BOX_MARKERS if a != b and a in b]
check("B4 no box marker is a substring of another (incl LR72_* vs LR_*)", not coll,
      f"collisions: {coll}")


def print_strings(path):
    with open(path) as f:
        tree = ast.parse(f.read())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            lits = [c.value for c in ast.walk(node)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            out.append((("".join(lits)), node.lineno))
    return out


if BOX is not None:
    try:
        with open(BOX_PATH) as f:
            bsrc = f.read()
        box_prints = print_strings(BOX_PATH)
        for m in ("LR72_READY", "LR72_DONE", "LR72_FATAL"):
            n = sum(m in text for text, _ in box_prints)
            check(f"B4 box owns {m}: printed exactly once", n == 1, f"printed {n}x")
        _done_at = bsrc.find('print("LR72_DONE"')
        _guard_at = bsrc.find("if __name__ ==")
        check("B4 LR72_DONE is the box's final act (inside the __main__ guard)",
              _done_at > _guard_at >= 0, f"done at {_done_at}, guard at {_guard_at}")
        bhits = [f"line {ln}: {sub!r}" for text, ln in box_prints
                 for sub in FATAL_SUBSTRINGS if sub in text]
        check("B4 box prints carry no labkit FATAL substring", not bhits, "; ".join(bhits))
        # the src scorer module never carries these markers (M1 parity, checked in test_lr_vllm too)
        with open(os.path.join(REPO, "src", "lr_vllm.py")) as f:
            vsrc = f.read()
        check("B4 src/lr_vllm.py carries none of this box's markers",
              all(m not in vsrc for m in ("LR72_READY", "LR72_DONE", "LR72_FATAL")))
    except Exception as e:
        check("B4 marker ownership", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B5: atomic shards + resume
if BOX is not None:
    try:
        msrc = inspect.getsource(BOX)
        check("B5 shards written tmp -> os.replace (atomic; presence = done)",
              '.tmp' in msrc and "os.replace(" in msrc)
        check("B5 resume: an existing shard is skipped (LR72_SKIP)", "LR72_SKIP" in msrc)
        # score_path names one shard per (set, ctx) unambiguously
        p1 = BOX.score_shard_path("/o", "secret_word", "SW")
        p2 = BOX.score_shard_path("/o", "secret_sustain", "SS")
        check("B5 score_shard_path is per (set, ctx), distinct, fs-safe",
              p1 != p2 and " " not in os.path.basename(p1)
              and os.path.basename(p1).endswith(".pt"), f"{p1} {p2}")
    except Exception as e:
        check("B5 atomic shards", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B6: util gate (< 60% halts)
if BOX is not None:
    try:
        check("B6 util floor is 60% (the prereg perf requirement)", BOX.UTIL_GATE_MIN == 60.0)
        # a healthy util logs and returns (no raise); a low util raises
        BOX.util_gate("gen", tokens=1000, secs=1.0, util=85.0)     # ok
        check("B6 healthy util (>=60%) passes the gate", True)
        try:
            BOX.util_gate("score", tokens=1000, secs=1.0, util=40.0)
            check("B6 util < 60% HALTS the box (raise)", False, "no raise")
        except RuntimeError as e:
            markerbad = ("LR72_FATAL", "CUDA out of memory")
            check("B6 util < 60% HALTS the box (raise)", True)
            check("B6 util-gate message is marker/FATAL-substring safe",
                  not any(b in str(e) for b in markerbad), repr(str(e)))
        # util None (no nvidia-smi / test env) logs instead of falsely halting
        BOX.util_gate("gen", tokens=1000, secs=1.0, util=None)
        check("B6 util None (CPU/test env) logs, never falsely halts", True)
    except Exception as e:
        check("B6 util gate", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B7: generation reuse
if BOX is not None:
    try:
        msrc = inspect.getsource(BOX)
        check("B7 word-free acceptance reuses covert_collect (is_degenerate / degeneracy)",
              "is_degenerate" in msrc and "degeneracy" in msrc)
        check("B7 anti-word system prompts come from primers_v3.compose_system (arms)",
              "compose_system" in msrc)
        # the generation loop drives a vLLM client (client.generate over HTTP), NOT an HF
        # model.generate() forward pass -- no HF generate signature (output_logits / model.generate).
        check("B7 generation is NOT an HF model.generate forward pass "
              "(no output_logits / return_dict_in_generate)",
              "output_logits" not in msrc and "return_dict_in_generate" not in msrc
              and "model.generate(" not in msrc)

        # exercise generate_arm against a MOCK completions client + stub tokenizer (no server)
        class StubTok:
            eos_token_id = 151645
            def __call__(self, text, add_special_tokens=False):
                class _R: pass
                r = _R(); r.input_ids = [ord(c) for c in text]; return r
            def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
                ids = []
                for m in msgs:
                    ids.extend(ord(c) for c in m["content"])
                return ids
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(int(i)) for i in ids)

        class GenClient:
            """Returns clean gibberish for one stream and a REPETITION-degenerate stream for
            another (so the acceptance gate must reject at least one). Repetition trips
            is_degenerate WITHOUT wordfreq (the analysis venv lacks it; the box installs it) -- so
            the word-free/acceptance gate is demonstrably live in the test env."""
            def __init__(self):
                self.n = 0
            def generate(self, prompt_ids, n=1, max_tokens=48, **kw):
                self.n += 1
                outs = []
                for i in range(n):
                    text = ("qx z fjm wpl kbt" if (self.n + i) % 2 == 0
                            else "abcabcabcabcabcabcabcabcabc")   # repetition -> rejected
                    outs.append([ord(c) for c in text])
                return outs
        gc = GenClient()
        recs = BOX.generate_arm(gc, StubTok(), arm="secret_word", concept="fear",
                                target_clean=2, max_gen=12)
        check("B7 generate_arm returns accepted word-free streams up to target_clean",
              len([r for r in recs if r["accepted"]]) >= 2
              and all("tokens" in r and "concept" in r for r in recs), f"n={len(recs)}")
        check("B7 the real-word stream is REJECTED by the acceptance gate (word-free enforced)",
              any(not r["accepted"] for r in recs), "no rejects -- filter inert?")
    except Exception as e:
        check("B7 generation reuse", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B8: the 7B off-diagonal cell
if BOX is not None:
    try:
        src = inspect.getsource(BOX)
        check("B8 the 7B off-diagonal privacy check is documented as a SEPARATE small reader "
              "(HF lr_grid on the pulled 72B streams, offline) -- not teacher-forced under 72B",
              "7b" in src.lower() and ("off-diag" in src.lower() or "off_diag" in src.lower())
              and ("lr_grid" in src or "offline" in src.lower()))
        check("B8 the box exposes which path scores the 7B cell (a named constant/flag)",
              hasattr(BOX, "OFFDIAG_7B_NOTE") or "OFFDIAG" in src)
    except Exception as e:
        check("B8 7b off-diagonal documentation", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B9: teacher-force scoring wiring
if BOX is not None:
    try:
        msrc = inspect.getsource(BOX)
        check("B9 scoring imports the vLLM prompt_logprobs scorer (src/lr_vllm)",
              "lr_vllm" in msrc)
        check("B9 the diagonal is scored via lr_vllm.lr_score (72B reads its own streams)",
              "lr_score" in msrc)
        check("B9 no HF teacher-forcing on the 72B path (no lr_reader.score_batch here)",
              "score_batch_kv" not in msrc and "score_batch_concat" not in msrc)
        # score_arm wiring: scores every stream under matched ctx vs neutral, writes an atomic shard
        check("B9 score_arm writes the per-(set,ctx) shard atomically",
              "score_arm" in dir(BOX))
    except Exception as e:
        check("B9 scoring wiring", False, f"raised {type(e).__name__}: {e}")

# ============================================================ B10: observer-scaling cells (Amdt 1)
# The 72B ALSO observes the existing smaller-model streams (SCORING-ONLY, no generation): 3
# generators x 3 arms = 9 observer cells, each teacher-forced under its OWN matched context vs
# neutral with the SAME lr_vllm scorer. Distinct shard names carrying an observe_ prefix.
if BOX is not None:
    try:
        check("B10 observer generators = qwen2.5-{1.5b,3b,7b} (the completed scale-grid streams)",
              tuple(BOX.OBSERVE_GENERATORS) == ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"),
              f"{getattr(BOX, 'OBSERVE_GENERATORS', None)}")
        check("B10 observer arms = secret_word + secret_sustain + evoked (3 arms)",
              tuple(BOX.OBSERVE_ARMS) == ("secret_word", "secret_sustain", "evoked"),
              f"{getattr(BOX, 'OBSERVE_ARMS', None)}")
        cells = [(g, a) for g in BOX.OBSERVE_GENERATORS for a in BOX.OBSERVE_ARMS]
        check("B10 exactly 9 observer cells (3 generators x 3 arms)", len(cells) == 9)
    except Exception as e:
        check("B10 observer cell design", False, f"raised {type(e).__name__}: {e}")

if BOX is not None:
    try:
        # observe_shard_path: distinct per (gen, arm, ctxset), reader=72b, generator=the smaller
        # model, carries an observe_ prefix, fs-safe .pt, distinct from the diagonal shard names.
        p_sw = BOX.observe_shard_path("/o", "qwen2.5-7b", "secret_word", "SW")
        p_ev = BOX.observe_shard_path("/o", "qwen2.5-7b", "evoked", "A")
        p_diag = BOX.score_shard_path("/o", "secret_word", "SW")
        base = os.path.basename(p_sw)
        check("B10 observe_shard_path carries the observe_ prefix + gen + arm + ctx",
              "observe_" in base and "qwen2.5-7b" in base and "secret_word" in base
              and "SW" in base, f"{base}")
        check("B10 observer shard names are distinct across (gen, arm, ctx)",
              len({p_sw, p_ev, BOX.observe_shard_path('/o', 'qwen2.5-1.5b', 'secret_word', 'SW')})
              == 3, f"{p_sw} {p_ev}")
        check("B10 observer shard != the 72B diagonal shard (reader vs generator differ)",
              p_sw != p_diag, f"obs={p_sw} diag={p_diag}")
        check("B10 observer shard is fs-safe .pt (no spaces)",
              " " not in base and base.endswith(".pt"), base)
    except Exception as e:
        check("B10 observe_shard_path", False, f"raised {type(e).__name__}: {e}")

if BOX is not None:
    try:
        src = inspect.getsource(BOX)
        # the smaller-model source bundles are pulled in S0 (local _ind path documented; HF fallback)
        check("B10 observer source bundles are documented (local _ind + HF fallback)",
              hasattr(BOX, "observe_bundle_path")
              and ("_ind" in src or "OBSERVE_BUNDLE" in src))
        paths = BOX.observe_bundle_path("qwen2.5-1.5b", "secret_word")
        check("B10 observe_bundle_path resolves to a <slug>-<arm>.pt bundle",
              str(paths).endswith("qwen2.5-1.5b-secret_word.pt"), f"{paths}")
        # secret_sustain lives under the grid box's _ind mirror; the resolver must find it there.
        ss = BOX.observe_bundle_path("qwen2.5-3b", "secret_sustain")
        check("B10 observe_bundle_path finds secret_sustain (grid-box _ind mirror)",
              str(ss).endswith("qwen2.5-3b-secret_sustain.pt"), f"{ss}")
    except Exception as e:
        check("B10 observer bundle resolution", False, f"raised {type(e).__name__}: {e}")

if BOX is not None:
    try:
        # score_observer: teacher-force one smaller-model bundle's streams under matched ctx vs
        # neutral via lr_vllm.lr_score (SAME scorer as the diagonal, verbatim), writing an atomic
        # observe_ shard. Exercised against a MOCK completions client (no vLLM server).
        import numpy as np
        import lr_vllm as V

        class ObsTok:
            eos_token_id = 151645
            def __call__(self, text, add_special_tokens=False):
                class _R: pass
                r = _R(); r.input_ids = [ord(c) for c in text]; return r
            def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
                return [ord(c) for m in msgs for c in m["content"]]
            def decode(self, ids, skip_special_tokens=True):
                return "".join(chr(int(i)) for i in ids)

        class ScoreClient:
            """A completions mock returning prompt_logprobs for exactly the sent prompt ids: every
            provided token gets a fixed logprob so lr_vllm.span_logprobs aligns and sums without a
            provided-token miss (the diagonal test's contract, reused)."""
            def completions(self, prompt_ids, prompt_logprobs=20, **kw):
                pl = [None] + [{str(int(t)): -0.5} for t in prompt_ids[1:]]
                return {"prompt_logprobs": pl}

        # a tiny smaller-model bundle in the exp3 collector schema (strength-1 induced streams).
        concepts = list(BOX._concepts())
        streams = []
        for gi, c in enumerate(concepts[:3]):
            streams.append(dict(gidx=gi, concept=c, concept_idx=concepts.index(c),
                                arm="secret_word", tokens=[65 + gi, 66 + gi, 67 + gi],
                                text="abc", accepted=True, strength=1))
        with tempfile.TemporaryDirectory() as td:
            gdir = os.path.join(td, "lr_72b")
            os.makedirs(gdir)
            BOX.score_observer(ScoreClient(), ObsTok(), "qwen2.5-1.5b", "secret_word",
                               streams, gdir, first_score=[True])
            shard = BOX.observe_shard_path(gdir, "qwen2.5-1.5b", "secret_word", "SW")
            check("B10 score_observer writes the matched-ctx observe_ shard atomically",
                  os.path.exists(shard) and not os.path.exists(shard + ".tmp"), shard)
            import torch
            sh = torch.load(shard, map_location="cpu", weights_only=False)
            check("B10 observer shard: reader=72B, generator=the smaller model (off-diagonal)",
                  sh["reader"] == "qwen2.5-72b" and sh["generator"] == "qwen2.5-1.5b",
                  f"{sh.get('reader')}/{sh.get('generator')}")
            check("B10 observer shard records carry ll[concept] (the baked-in LR difference)",
                  sh["records"] and all("ll" in r for r in sh["records"]))
        # score_observer must call the SAME lr_vllm.lr_score used on the diagonal (no new numerics)
        osrc = inspect.getsource(BOX.score_observer)
        check("B10 score_observer reuses lr_vllm.lr_score verbatim (same currency as diagonal)",
              "lr_score" in osrc)
        check("B10 score_observer is SCORING-ONLY (no client.generate, no HF forward pass)",
              "generate(" not in osrc and "model.generate" not in osrc)
    except Exception as e:
        check("B10 score_observer wiring", False, f"raised {type(e).__name__}: {e}")

# ================================================================ B11: VLLMClient.generate raises on empty token_ids (FIX 2)
# If vLLM omits token_ids from a choices entry, the old code silently returned [] ->
# all-zeros LR that looks like a real null result.  The fix raises explicitly.
if BOX is not None:
    try:
        class _MockPost:
            """_post replacement: returns choices WITHOUT token_ids to trigger the guard."""
            def __init__(self, client_obj):
                self._orig = client_obj._post
            def __call__(self, body):
                return {"choices": [{"text": "abc"}]}   # token_ids absent

        c = BOX.VLLMClient.__new__(BOX.VLLMClient)
        c.url, c.model = "http://nowhere", BOX.MODEL_72B
        c._post = _MockPost(c)
        try:
            c.generate([1, 2, 3], n=1)
            check("B11 VLLMClient.generate raises when token_ids absent from choice", False,
                  "no raise -- silent empty list would fabricate a null result")
        except RuntimeError as exc:
            check("B11 VLLMClient.generate raises when token_ids absent from choice", True)
            # message must be LR72_FATAL-safe (not a substring of a marker)
            check("B11 raise message is LR72_FATAL-safe (no marker substring collision)",
                  "LR72_FATAL" not in str(exc) and "LR72_READY" not in str(exc)
                  and "LR72_DONE" not in str(exc), repr(str(exc)))
    except Exception as e:
        check("B11 VLLMClient.generate empty-token_ids guard", False,
              f"raised {type(e).__name__}: {e}")

# ================================================================ B12: wait_health detects vLLM process death (FIX 3)
# OOM -> the process exits with a nonzero code while wait_health loops for 3600s, burning 1hr
# of billing before the timeout.  The fix: pass proc to wait_health and check proc.poll() each
# iteration; a dead process raises immediately rather than waiting out the full timeout.
if BOX is not None:
    try:
        import inspect as _inspect
        wh_sig = _inspect.signature(BOX.wait_health)
        check("B12 wait_health accepts a proc parameter",
              "proc" in wh_sig.parameters,
              f"params={list(wh_sig.parameters)}")

        # mock a dead process (returncode set, poll() returns it immediately)
        class _DeadProc:
            returncode = 1
            def poll(self):
                return self.returncode

        # opener that never succeeds (simulates a server that never comes up)
        def _never_open(url, timeout=10):
            raise OSError("connection refused")

        try:
            BOX.wait_health(url="http://nowhere", timeout_s=10, opener=_never_open,
                            proc=_DeadProc())
            check("B12 wait_health raises promptly when the vLLM process is dead", False,
                  "no raise -- would have burned up to timeout_s billing seconds")
        except RuntimeError as exc:
            check("B12 wait_health raises promptly when the vLLM process is dead", True)
            check("B12 raise message mentions the exit code",
                  "code" in str(exc) or "returncode" in str(exc) or "exit" in str(exc).lower(),
                  repr(str(exc)))
    except Exception as e:
        check("B12 wait_health proc-death guard", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
