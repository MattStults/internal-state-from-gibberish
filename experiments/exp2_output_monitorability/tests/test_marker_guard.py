"""Class-level box-marker / watchdog-substring guard over EVERY box-script reader module.

labkit substring-matches its ready/done markers AND a generic FATAL tuple against every log line
(labkit lifecycle/remote.py:20). Box attempt 4 (LR) died because the reader's per-shard progress
line contained the done marker as a substring ("LR_DONE" + "_SHARD") -- labkit declared the box
done, pulled and tore it down 1/9th of the way in. The same bug class can be rebuilt in ANY
reader module, so this guard scans the PRINT-STATEMENT string literals of each one for:

  - its own box's ready/done markers (only the box script, as its final line, may emit them);
  - labkit's generic FATAL substrings (the remote.py FATAL tuple, copied verbatim below) -- a
    printed literal containing one would FATAL a healthy run.

Any future src/*_reader.py is picked up automatically; if unregistered in MODULES it is checked
against ALL known box markers (register it with its box's markers when the box exists).
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_marker_guard.py
"""
import ast
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(REPO, "src")

# box-script reader module -> (ready, done) markers of the box that drives it
MODULES = {
    "lr_reader.py": ("LR_READY", "LR_DONE"),               # box_lr.py
    "elicit_reader.py": ("ELICIT_READY", "ELICIT_DONE"),   # box_elicit.py
    "mc_reader.py": ("MC_READY", "MC_DONE"),               # box_mc.py
    "state_trajectory.py": ("GAUGE_READY", "GAUGE_DONE"),  # box_gauge.py
}
ALL_MARKERS = tuple(m for pair in MODULES.values() for m in pair)
# labkit lifecycle/remote.py:20 -- the FATAL tuple, VERBATIM (keep in sync with the pinned tag)
FATAL = ("CUDA error", "CUDA out of memory", "Traceback (most recent call last)",
         "ModuleNotFoundError", "torch.cuda.OutOfMemoryError")

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def print_strings(path):
    """[(concatenated string literals of one print(...) call, lineno)] -- f-string constant parts
    included (a marker split across an f-string's literal chunks still concatenates in order)."""
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


targets = dict(MODULES)
for path in sorted(glob.glob(os.path.join(SRC, "*_reader.py"))):
    targets.setdefault(os.path.basename(path), None)       # future reader: all markers apply

for fname, markers in sorted(targets.items()):
    path = os.path.join(SRC, fname)
    if not os.path.exists(path):
        check(f"MG {fname}: registered module exists", False, f"{path} missing")
        continue
    forbid = (markers if markers is not None else ALL_MARKERS) + FATAL
    hits = [f"line {ln}: {sub!r}" for text, ln in print_strings(path)
            for sub in forbid if sub in text]
    check(f"MG {fname}: no box ready/done marker or labkit FATAL substring in print statements",
          not hits, "; ".join(hits))

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
