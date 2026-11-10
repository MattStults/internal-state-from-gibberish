"""Two repo-hygiene invariants.

1. Every test file reports failure through its exit status. `make test` stops on a non-zero exit,
   so a test file that only prints [FAIL] lines can never fail the suite.
2. The 14B-extension scorer records its input folder relative to the repo, so re-running it does
   not write the runner's home directory into the committed results JSON.

Run:  .venv/bin/python experiments/exp2_output_monitorability/tests/test_suite_hygiene.py
"""
import ast
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

checks = []


def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def exits_with_status(path):
    """True if the file has a sys.exit(...) call or `raise SystemExit` that is not inside a try block
    (an exit inside try/except is something the test is catching, not the file's own verdict)."""
    tree = ast.parse(path.read_text())
    inside_try = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                inside_try.add(id(child))
    for node in ast.walk(tree):
        if id(node) in inside_try:
            continue
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "exit"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "sys"):
            return True
        if isinstance(node, ast.Raise) and node.exc is not None and "SystemExit" in ast.dump(node.exc):
            return True
    return False


tests = sorted(ROOT.glob("experiments/*/tests/test_*.py"))
check("found the test files", len(tests) >= 40, f"found {len(tests)}")
for t in tests:
    check(f"{t.parent.parent.name}/tests/{t.name} sets an exit status", exits_with_status(t))

try:
    spec = importlib.util.spec_from_file_location("lr_extend_offline", HERE.parent / "analysis" / "lr_extend_offline.py")
    LEO = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LEO)
    rec = LEO.portable_path(LEO.BOX_DIR)
    check("scorer records its default input folder relative to the repo",
          rec == os.path.join("runs", "lr_extend_box"), f"got {rec!r}")
    check("recorded folder is not an absolute path", not os.path.isabs(rec), f"got {rec!r}")
    outside = os.path.join(os.sep, "somewhere", "else", "box")
    check("a folder outside the repo is recorded unchanged", LEO.portable_path(outside) == outside)
except Exception as e:  # noqa: BLE001
    check("scorer records its input folder relative to the repo", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    if not ok:
        print(f"  [FAIL] {name}" + (f"  -- {note}" if note else ""))
print(f"  [{'PASS' if all(ok for _, ok, _ in checks) else 'FAIL'}] {sum(ok for _, ok, _ in checks)}/{len(checks)} hygiene checks")
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
