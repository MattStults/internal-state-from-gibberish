"""Run the whole offline analysis suite for one model, CPU-only -- so you don't have to set INTRO_MODEL
and invoke ten scripts by hand.

    .venv/bin/python analysis/run_all.py --model qwen2.5-3b

Runs the per-model analyses (which read runs/<model>/data/covert_collect.pt) in order. The cross-model
summaries (verify_v1_numbers, check_bmc_significance, concept_matched_control, derisk_learning_curve) loop
over all models themselves -- run those directly, or `make summary`.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PER_MODEL = ["check_injection", "analyze_v2", "analyze_v3_curves", "analyze_distinct", "analyze_reevocation",
             "show_calibration"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-3b", help="model slug (sets INTRO_MODEL for the suite)")
    args = ap.parse_args()
    env = {**os.environ, "INTRO_MODEL": args.model}
    failed = []
    for s in PER_MODEL:
        print(f"\n===== {s}  (model={args.model}) =====", flush=True)
        if subprocess.run([sys.executable, str(HERE / f"{s}.py")], env=env).returncode != 0:
            failed.append(s)
    print(f"\n{'OK' if not failed else 'FAILED: ' + ', '.join(failed)}  ({len(PER_MODEL) - len(failed)}/{len(PER_MODEL)})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
