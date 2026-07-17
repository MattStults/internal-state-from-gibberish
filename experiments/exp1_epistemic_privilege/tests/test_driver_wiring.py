"""Driver wiring check (driver venv only -- imports labkit, NOT baseline_clean/numpy).
Run:  .venv-driver/bin/python tests/test_driver_wiring.py"""
import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location(
    "rl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "harness", "run_labkit.py"))
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)

ep = rl.build_job(["qwen3-4b"], False, None, False, True, "orig,fragments", "nothink,think").payload["entrypoint"]
collect_ep = rl.build_job(["qwen3-4b"], False).payload["entrypoint"]
# collect with a chosen variant
cvar = rl.build_job(["qwen3-4b"], False, None, False, False, "orig", "nothink", "fragments").payload["entrypoint"]
# collect with the all-position injection method + fixed-dose (--no-calibrate) -- the controlled A/B arm
cab = rl.build_job(["qwen3-4b"], False, None, False, False, "orig", "nothink", "orig", "all", True).payload["entrypoint"]
checks = [
    ("probe runs baseline_clean", "baseline_clean.py" in ep),
    ("carries --variants + --modes", "--variants orig,fragments" in ep and "--modes nothink,think" in ep),
    ("probe has no --variant (singular)", " --variant " not in ep),
    ("collect runs covert_collect, default --variant orig", "covert_collect.py" in collect_ep and "--variant orig" in collect_ep),
    ("collect carries chosen --variant", "--variant fragments" in cvar),
    ("collect defaults to --inject gen", "--inject gen" in collect_ep and "--no-calibrate" not in collect_ep),
    ("collect carries --inject all + --no-calibrate", "--inject all" in cab and "--no-calibrate" in cab),
]
for name, cond in checks:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
