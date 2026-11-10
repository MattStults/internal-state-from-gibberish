"""The exp1 analysis scripts must locate the repo root from their own path.

Regression: several scripts used `Path(__file__).resolve().parent.parent`, which is the experiment
folder, not the repo root, so `REPO / "runs" / ...` pointed at nothing. One of them
(check_bmc_significance.py) then printed "skipping" for every model instead of its table.

Run:  .venv/bin/python experiments/exp1_epistemic_privilege/tests/test_analysis_repo_root.py
"""
import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"
ROOT = HERE.parents[2]

checks = []


def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


check("this test's own idea of the repo root holds the Makefile and README",
      (ROOT / "Makefile").is_file() and (ROOT / "README.md").is_file(), str(ROOT))

# 1. every top-level `REPO = ...` in analysis/*.py evaluates to the repo root
found = 0
for script in sorted(ANALYSIS.glob("*.py")):
    for node in ast.parse(script.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "REPO"):
            found += 1
            value = eval(compile(ast.Expression(node.value), str(script), "eval"),
                         {"Path": Path, "__file__": str(script)})
            check(f"{script.name}: REPO resolves to the repo root", Path(value) == ROOT, f"got {value}")
check("at least three analysis scripts define REPO (the test is looking at something)", found >= 3, f"found {found}")

# 2. behaviour: the significance summary reads the committed results instead of skipping them
out = subprocess.run([sys.executable, str(ANALYSIS / "check_bmc_significance.py")],
                     capture_output=True, text=True)
check("check_bmc_significance.py exits 0", out.returncode == 0, out.stderr[-300:])
check("check_bmc_significance.py does not skip any model", "skipping" not in out.stdout,
      f"{out.stdout.count('skipping')} of 3 models skipped")
for slug in ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b"):
    check(f"check_bmc_significance.py prints a result block for {slug}", f"=== {slug} (s" in out.stdout)

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
