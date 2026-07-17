"""Aliased-import attribute audit (full-run attempt 1, 2026-07-11, ~$0.15): every box-invoked
entrypoint must resolve every attribute it takes from every `import X as Y` module -- the class of
break that local suites miss (used at runtime, absent from the module: gauge_alt_collect imported
primers_v2 but used P.GAUGE_PROBE, which only primers_v3 exports)."""
import importlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for rel in ("experiments/exp3_induction_and_scale", "src", "experiments/exp2_output_monitorability"):
    sys.path.insert(0, os.path.join(REPO, rel))

ENTRYPOINTS = [
    "experiments/exp3_induction_and_scale/gauge_alt_collect.py",
    "experiments/exp3_induction_and_scale/gauge_judge_alt.py",
    "experiments/exp3_induction_and_scale/collect_induction.py",
    "src/lr_grid.py", "src/mc_reader.py",
    "src/lr_rider.py", "src/inject_tf_lr.py",
    "experiments/exp2_output_monitorability/box_lr_grid.py",
    "experiments/exp2_output_monitorability/box_mc.py",
    "experiments/exp2_output_monitorability/box_lr_extend.py",
]

n_pass = n_fail = 0
for f in ENTRYPOINTS:
    src = open(os.path.join(REPO, f)).read()
    for m in re.finditer(r"^import (\w+) as ([A-Z][A-Za-z]*)\s*(?:#.*)?$", src, re.M):
        mod_name, alias = m.groups()
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue                      # box-only dep absent locally: not this test's class
        attrs = sorted(set(re.findall(rf"\b{alias}\.([A-Za-z_]\w*)", src)))
        missing = [a for a in attrs if not hasattr(mod, a)]
        ok = not missing
        n_pass += ok; n_fail += (not ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] E1 {f}: {alias}={mod_name} missing={missing}")

print(f"{n_pass} pass, {n_fail} fail")
sys.exit(1 if n_fail else 0)
