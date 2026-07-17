"""RED-first unit tests for scale-grid unit B12 (prereg Amendment 1, should-fixes 2 + 7): the B1
alt-generation stage additions. No model, no GPU, no API calls.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_alt_gauge.py

A1  gauge COLLECTION for the evoked_alt personas at 3B/7B: exp3-protocol persona-only
    free-association texts (arm=evoked_alt via primers_v2.compose_gauge_system), sampling
    byte-parity with collect_induction._collect_gauge (max_new_tokens=64, temperature=1.0,
    top_p=0.98), gauge_n = the exp3 real-run default (cfg(smoke=False)).
A2  blind JUDGE reuse: gauge_judge_alt scores the pulled sidecars with the SAME pinned judge +
    protocol as exp3 (imports gauge_judge.judge_model_gauge / JUDGE_MODEL -- the same objects,
    not copies); gauge-fail FLAGS (gauge_pass in the output json consumed by lr_grid_offline),
    never FATALs a run (no box marker / FATAL print anywhere in it).
A3  box wiring: gauge collection rides S1 per alt-generated size, resume-safe on the sidecar,
    sidecar under OUT/_ind/<slug>/data so labkit pulls it home.
A4  env/pipeline PARITY asserts vs the exp3 evoked collection at that size: transformers pin,
    wordfreq live (word-free filter), sampling cfg == collect_induction's real-run cfg, checked
    on-box BEFORE any alt stream is generated.
A5  BYTE-IDENTITY of the wording-A/B persona texts across sizes: sha256 over the frozen
    primers.EVOKED/EVOKED_ALT/NEUTRAL texts pinned in code and asserted at box start -- every
    size's generation and every reader's context reconstruction provably uses the same bytes.
"""
import hashlib
import importlib.util
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EXP3 = os.path.join(REPO, "experiments", "exp3_induction_and_scale")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, EXP3)

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOX_PATH = os.path.join(REPO, "experiments", "exp2_output_monitorability", "box_lr_grid.py")
GAC_PATH = os.path.join(EXP3, "gauge_alt_collect.py")
GJA_PATH = os.path.join(EXP3, "gauge_judge_alt.py")

try:
    BOX = load_module("box_lr_grid", BOX_PATH)
except Exception as e:
    BOX = None
    check("import box_lr_grid.py", False, f"{type(e).__name__}: {e}")

import collect_induction as CI   # noqa: E402
import primers as P              # noqa: E402

# ================================================================ A1: alt-gauge collection
try:
    GAC = load_module("gauge_alt_collect", GAC_PATH)
except Exception as e:
    GAC = None
    check("A1 gauge_alt_collect.py exists and imports", False, f"{type(e).__name__}: {e}")

if GAC is not None:
    try:
        src = inspect.getsource(GAC.collect_gauge_arm)
        ref = inspect.getsource(CI._collect_gauge)
        check("A1 gauge system is the ALT arm's (compose_gauge_system(..., arm=...) driven by "
              "the arm param, never hardcoded 'evoked')",
              "compose_gauge_system" in src and 'arm="evoked")' not in src)
        for lit in ("max_new_tokens=64", "temperature=1.0", "top_p=0.98", "do_sample=True"):
            check(f"A1 sampling parity with exp3's _collect_gauge: {lit}",
                  lit in src and lit in ref)
        check("A1 probe is the frozen GAUGE_PROBE", "GAUGE_PROBE" in src)
        check("A1 gauge_n default = the exp3 real-run cfg (cfg(smoke=False))",
              GAC.DEFAULT_GAUGE_N == CI.cfg(False)["gauge_n"],
              f"{getattr(GAC, 'DEFAULT_GAUGE_N', None)} vs {CI.cfg(False)['gauge_n']}")
        msrc = inspect.getsource(GAC.main)
        check("A1 sidecar saved atomically (tmp -> os.replace)",
              "os.replace(" in msrc and '.with_suffix(".tmp")' in msrc or ".tmp" in msrc)
        check("A1 sidecar records arm=evoked_alt + model for provenance",
              'arm="evoked_alt"' in msrc or "arm='evoked_alt'" in msrc)
    except Exception as e:
        check("A1 gauge_alt_collect content", False, f"raised {type(e).__name__}: {e}")

# ================================================================ A2: pinned-judge reuse
try:
    # gauge_judge imports anthropic lazily (inside main), so this import is offline-safe. Plain
    # import (NOT load_module) so gauge_judge_alt's `from gauge_judge import ...` resolves to the
    # SAME sys.modules instance and the identity check below is meaningful.
    import gauge_judge as GJ
    GJA = load_module("gauge_judge_alt", GJA_PATH)
except Exception as e:
    GJA = None
    check("A2 gauge_judge_alt.py exists and imports", False, f"{type(e).__name__}: {e}")

if GJA is not None:
    try:
        check("A2 judge protocol is exp3's judge_model_gauge (same function object)",
              GJA.judge_model_gauge is GJ.judge_model_gauge)
        check("A2 judge model is the SAME pinned snapshot", GJA.JUDGE_MODEL == GJ.JUDGE_MODEL
              and GJA.JUDGE_MODEL == "claude-haiku-4-5-20251001", f"{GJA.JUDGE_MODEL}")
        check("A2 targets the 3B/7B alt pools",
              tuple(GJA.MODELS) == ("qwen2.5-3b", "qwen2.5-7b"), f"{GJA.MODELS}")
        check("A2 reads the pulled sidecars (runs/lr_grid_box/_ind/<slug>/data)",
              "lr_grid_box" in GJA.sidecar_path("qwen2.5-3b")
              and GJA.sidecar_path("qwen2.5-3b").endswith("qwen2.5-3b-evoked_alt-gauge.pt"))
        check("A2 writes gauge_alt_results.json (lr_grid_offline's flag input)",
              GJA.OUT_JSON.endswith("gauge_alt_results.json"))
        gsrc = open(GJA_PATH).read()
        check("A2 gauge-fail FLAGS, never FATALs (no FATAL/marker strings in the judge script)",
              "FATAL" not in gsrc and "sys.exit(1)" not in gsrc)
    except Exception as e:
        check("A2 gauge_judge_alt content", False, f"raised {type(e).__name__}: {e}")

# ================================================================ A3: box wiring
if BOX is not None:
    try:
        p = BOX.alt_gauge_path("qwen2.5-3b")
        check("A3 sidecar path under OUT/_ind/<slug>/data (labkit pulls it home)",
              p == os.path.join(BOX.OUT, "_ind", "qwen2.5-3b", "data",
                                "qwen2.5-3b-evoked_alt-gauge.pt"), f"path = {p}")
        cmd, env = BOX.gauge_cmd("qwen2.5-3b")
        script = cmd[cmd.index("-u") + 1] if "-u" in cmd else cmd[1]
        check("A3 gauge_cmd invokes exp3's gauge_alt_collect.py",
              os.path.basename(script) == "gauge_alt_collect.py" and os.path.exists(script),
              f"cmd = {cmd}")
        check("A3 gauge env: INTRO_MODEL=<slug>, INTRO_RUN_DIR under OUT/_ind/<slug>",
              env.get("INTRO_MODEL") == "qwen2.5-3b"
              and env.get("INTRO_RUN_DIR", "").endswith(os.path.join("_ind", "qwen2.5-3b")),
              f"env = {env}")
        ssrc = inspect.getsource(BOX.altgen_stage)
        check("A3 gauge collection rides S1 (altgen_stage runs gauge_cmd, resume-safe on the "
              "sidecar)", "gauge_cmd(" in ssrc and "alt_gauge_path(" in ssrc)
    except Exception as e:
        check("A3 box gauge wiring", False, f"raised {type(e).__name__}: {e}")

# ================================================================ A4: parity asserts
if BOX is not None:
    try:
        check("A4 transformers parity pin = the evoked collections' validated 4.46.3",
              BOX.PARITY_TRANSFORMERS == "4.46.3")
        check("A4 sampling parity cfg == collect_induction's real-run cfg (single source drift "
              "check: if exp3 defaults ever move, the box fails loudly pre-spend)",
              BOX.PARITY_CFG == CI.cfg(False), f"{getattr(BOX, 'PARITY_CFG', None)}")
        psrc = inspect.getsource(BOX.alt_parity_check)
        check("A4 parity check asserts transformers pin + wordfreq live",
              "transformers" in psrc and "wordfreq" in psrc)
        ssrc = inspect.getsource(BOX.altgen_stage)
        check("A4 parity checked in S1 BEFORE generation", "alt_parity_check(" in ssrc
              and ssrc.index("alt_parity_check(") < ssrc.index("altgen_cmd("))
    except Exception as e:
        check("A4 parity asserts", False, f"raised {type(e).__name__}: {e}")

# ================================================================ A5: persona byte-identity
DIGEST = None
try:
    h = hashlib.sha256()
    for wording, d in (("A", P.EVOKED), ("B", P.EVOKED_ALT)):
        for c in sorted(d):
            h.update(f"{wording}|{c}|{d[c]}\n".encode())
    h.update(("N|" + P.NEUTRAL).encode())
    DIGEST = h.hexdigest()
except Exception as e:
    check("A5 independent digest of the frozen personas", False, f"{type(e).__name__}: {e}")

if BOX is not None and DIGEST is not None:
    try:
        check("A5 PERSONA_SHA256 pinned to the frozen primers texts",
              BOX.PERSONA_SHA256 == DIGEST, f"pinned {getattr(BOX, 'PERSONA_SHA256', None)!r} "
                                            f"vs computed {DIGEST!r}")
        check("A5 persona_digest() reproduces the pin (runtime = frozen bytes)",
              BOX.persona_digest() == BOX.PERSONA_SHA256)
        try:
            BOX.assert_personas()
            check("A5 assert_personas passes on the frozen texts", True)
        except Exception as e:
            check("A5 assert_personas passes on the frozen texts", False,
                  f"raised {type(e).__name__}: {e}")

        class _Tampered:
            EVOKED = dict(P.EVOKED, fear=P.EVOKED["fear"] + " ")
            EVOKED_ALT = P.EVOKED_ALT
            NEUTRAL = P.NEUTRAL
        try:
            BOX.assert_personas(primers_mod=_Tampered)
            check("A5 a single-byte persona drift is caught", False, "no exception")
        except (AssertionError, RuntimeError):
            check("A5 a single-byte persona drift is caught", True)
        msrc = inspect.getsource(BOX.main)
        check("A5 main() asserts persona byte-identity before any stage",
              "assert_personas(" in msrc
              and msrc.index("assert_personas(") < msrc.index("fetch_inputs("))
    except Exception as e:
        check("A5 persona byte-identity", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
