"""Offline checks for the prompt-variant sweep probe (no model, no GPU).

Verifies the candidate prompts are well-formed + distinct, the valid-subset accounting, and that the driver
wires --variants/--modes into the probe entrypoint. Run:  .venv/bin/python tests/test_sweep_probe.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
os.environ.setdefault("INTRO_MODEL", "qwen3-4b")
import baseline_clean as B  # noqa: E402

checks = []


def ck(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    checks.append(cond)


# 1. variants well-formed
V = B.PROMPT_VARIANTS
ck("4 variants present", set(V) == {"orig", "fragments", "calm", "antiloop"})
ck("all variant prompts non-trivial", all(len(p) > 60 for p in V.values()))
ck("variant prompts are distinct", len(set(V.values())) == 4)
ck("anti-loop variants forbid alphabet/single letters",
   all(("alphabet" in V[v].lower() or "single letter" in V[v].lower()) for v in ("fragments", "calm", "antiloop")))

# 2. valid-subset accounting (thinking): unclosed + empty excluded, clean over real gibberish only
streams = [
    dict(text="qxz fjm wplk bt rvnm wuqs hlk jmb", has_think=True, accepted=True),    # valid clean
    dict(text="z x z x z x z x z x z x z x z x z x", has_think=True, accepted=False),  # valid degenerate
    dict(text="<think>okay the user wants random letters", has_think=False, accepted=False),  # unclosed -> excl
    dict(text="ok", has_think=True, accepted=True),                                   # empty post-think -> excl
]
# assert against the PRODUCTION accounting (baseline_clean.valid_subset_accounting), not a re-implementation
valid, n_unclosed, n_empty = B.valid_subset_accounting(streams, think=True)
ck("valid subset = 2 (excludes unclosed + empty)", len(valid) == 2 and n_unclosed == 1 and n_empty == 1)
ck("clean over valid only = 0.5", sum(s["accepted"] for s in valid) / len(valid) == 0.5)
v2, u2, e2 = B.valid_subset_accounting(streams, think=False)
ck("nothink: all valid, no exclusions", len(v2) == len(streams) and u2 == 0 and e2 == 0)

# 3. driver wiring (--variants/--modes -> probe entrypoint) needs labkit (driver venv); run separately if present
try:
    spec = importlib.util.spec_from_file_location(
        "rl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness", "run_labkit.py"))
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    j = rl.build_job(["qwen3-4b"], False, None, False, True, "orig,fragments", "nothink,think")
    ep = j.payload["entrypoint"]
    ck("probe entrypoint runs baseline_clean", "baseline_clean.py" in ep)
    ck("entrypoint carries --variants + --modes", "--variants orig,fragments" in ep and "--modes nothink,think" in ep)
    ck("collect path unaffected (no --variants)", "--variants" not in rl.build_job(["qwen3-4b"], False).payload["entrypoint"])
except ModuleNotFoundError:
    print("  [SKIP] driver-wiring checks (labkit not in this venv -- run in .venv-driver)")

print(f"\n{'ALL PASS' if all(checks) else 'FAILURES'} ({sum(checks)}/{len(checks)})")
sys.exit(0 if all(checks) else 1)
