"""Run every exp2 test suite in a fresh interpreter each (the suites are stateful scripts) and
summarize exit codes. Local convenience; suites remain individually runnable.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/run_all.py [suite ...]
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = [f[:-3] for f in sorted(os.listdir(HERE))
           if f.startswith("test_") and f.endswith(".py")]


def main():
    suites = sys.argv[1:] or DEFAULT
    bad = []
    for s in suites:
        p = subprocess.run([sys.executable, os.path.join(HERE, f"{s}.py")],
                           capture_output=True, text=True)
        fails = p.stdout.count("[FAIL]")
        passes = p.stdout.count("[PASS]")
        ok = p.returncode == 0
        print(f"{'OK  ' if ok else 'BAD '} {s}: {passes} pass, {fails} fail "
              f"(exit {p.returncode})")
        if not ok:
            bad.append(s)
            for line in p.stdout.splitlines():
                if "[FAIL]" in line:
                    print(f"      {line.strip()}")
            if p.stderr.strip():
                print("      stderr tail: " + p.stderr.strip().splitlines()[-1])
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
