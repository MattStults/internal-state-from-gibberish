"""On-box orchestrator for the ELICITED SELF-REPORT run (prereg: reports/elicited_report_prereg.md).

One box, N reader scales (--models): src/elicit_reader.py scores the FIXED 1.5B stream pool
(injected s60 / injected s0 / evoked s1 / evoked-neutral) under each reader -- prefilled-turn
elicitation (closed + open) plus the legacy passive '; secret word:' baseline. 12 atomic shards
per reader land in out/elicit/; scoring is OFFLINE (analysis/elicit_offline.py).

Two planned invocations (see harness/run_elicit.py):
  RTX3090 tier:   --models qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b
  RTX A6000 tier: --models qwen2.5-14b

Stages (idempotent; elicit_reader ELICIT_SKIPs existing shards):
  S0  HF-pull the inputs rsync excludes (*.pt): the exp1 1.5B capture and the exp3 evoked
      bundle, from ErrareHumanumEst/internal-state-from-gibberish (HF_TOKEN).
  S1  src/elicit_reader.py per reader scale (one subprocess per model -> clean GPU teardown),
      INTRO_RUN_DIR=out (shards -> out/elicit/).

Markers: ELICIT_READY / ELICIT_DONE / ELICIT_FATAL. Driven by harness/run_elicit.py (gated).
NEVER run this on the Mac -- it loads models.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

HF_DATASET = "ErrareHumanumEst/internal-state-from-gibberish"
SRC = "qwen2.5-1.5b"                                     # the stream pool's generating model
OUT = os.path.abspath(os.environ.get("INTRO_REPORT_DIR") or os.path.join(REPO, "out"))
CAP15 = os.path.join(REPO, "runs", SRC, "data", "covert_collect.pt")
EVOKED15 = os.path.join(REPO, "runs", "_ind", SRC, "data", f"{SRC}-evoked.pt")
FETCHES = [
    ("qwen2.5-1.5b-gen.pt", CAP15),
    (f"exp3/bundles/{SRC}-evoked.pt", EVOKED15),
]
STREAM_SETS = ("injected", "injected_s0", "evoked", "evoked_s0")
VARIANTS = ("closed", "open", "passive")
GPU_THREADS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}


def emit_step(step, **fields):
    print("LABKIT_STEP " + json.dumps({"step": int(step), **fields}), flush=True)


def start_heartbeat(period_s=120):
    """The gauge run's attempt-1 lesson: a silent HF weight download froze the log and the stall
    watchdog killed the box. A daemon heartbeat keeps the log growing through genuinely-quiet
    work; run_to still caps a true hang."""
    t0 = time.time()

    def beat():
        while True:
            time.sleep(period_s)
            print(f"HEARTBEAT t={int(time.time() - t0)}s", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def fetch_inputs():
    from huggingface_hub import hf_hub_download
    for fname, dest in FETCHES:
        if os.path.exists(dest):
            print(f"S0 have {os.path.relpath(dest, REPO)}", flush=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(hf_hub_download(HF_DATASET, fname, repo_type="dataset"), dest)
        print(f"S0 fetched {fname} -> {os.path.relpath(dest, REPO)}", flush=True)
    missing = [f for f, d in FETCHES if not os.path.exists(d)]
    if missing:
        raise RuntimeError(f"S0: inputs missing after HF pull: {missing}")


def shards_for(slug):
    return [os.path.join(OUT, "elicit", f"{slug}_{ss}_{v}.pt")
            for ss in STREAM_SETS for v in VARIANTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b",
                    help="comma-separated reader slugs, run smallest-first")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    os.makedirs(OUT, exist_ok=True)
    start_heartbeat()
    emit_step(0, phase="S0_fetch")
    fetch_inputs()
    print("ELICIT_READY", flush=True)

    for i, slug in enumerate(models):
        emit_step(1000 * (i + 1), phase="S1_elicit_reader", model=slug)
        if all(os.path.exists(s) for s in shards_for(slug)):
            print(f"S1 SKIP {slug}: all 12 shards exist", flush=True)
            continue
        cmd = [sys.executable, "-u", os.path.join(REPO, "src", "elicit_reader.py"),
               "--capture", CAP15, "--evoked", EVOKED15, "--batch", str(args.batch)]
        env = {**os.environ, **GPU_THREADS, "INTRO_MODEL": slug, "INTRO_RUN_DIR": OUT}
        print(f"RUN {' '.join(cmd)}  env=INTRO_MODEL={slug} INTRO_RUN_DIR={OUT}", flush=True)
        subprocess.run(cmd, env=env, cwd=REPO, check=True)

    missing = [s for slug in models for s in shards_for(slug) if not os.path.exists(s)]
    if missing:                                         # never report DONE with nothing to pull
        raise RuntimeError(f"elicit shards missing: {[os.path.relpath(m, OUT) for m in missing]}")
    emit_step(9000, phase="elicit_done", shards=12 * len(models))
    print(f"outputs OK in {OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
        print("ELICIT_DONE", flush=True)
    except Exception:
        print("ELICIT_FATAL", flush=True)
        raise
