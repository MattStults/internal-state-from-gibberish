"""On-box orchestrator for the LR-READER run (prereg: reports/lr_reader_prereg.md).

One 24GB box, ONE model (qwen2.5-1.5b): src/lr_reader.py scores every accepted saved stream
(injected s60 / evoked / evoked_alt) teacher-forced under the 25 reconstructed collection contexts
(12 wording-A personas, 12 wording-B, 1 neutral). 9 atomic shards land in out/lr/; scoring is
OFFLINE (analysis/lr_reader_offline.py).

Stages (idempotent; lr_reader LR_SKIPs existing shards):
  S0  HF-pull the inputs rsync excludes (*.pt): the exp1 1.5B capture and the exp3 evoked +
      evoked_alt bundles, from ErrareHumanumEst/internal-state-from-gibberish (HF_TOKEN).
  S1  src/lr_reader.py at 1.5B, INTRO_RUN_DIR=out (shards -> out/lr/).

Markers: LR_READY / LR_DONE / LR_FATAL. Driven by harness/run_lr.py (gated).
NEVER run this on the Mac -- it loads a model.
"""
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
M15 = "qwen2.5-1.5b"
OUT = os.path.abspath(os.environ.get("INTRO_REPORT_DIR") or os.path.join(REPO, "out"))
CAP15 = os.path.join(REPO, "runs", M15, "data", "covert_collect.pt")
EVOKED15 = os.path.join(REPO, "runs", "_ind", M15, "data", f"{M15}-evoked.pt")
ALT15 = os.path.join(REPO, "runs", "_ind", M15, "data", f"{M15}-evoked_alt.pt")
FETCHES = [
    ("qwen2.5-1.5b-gen.pt", CAP15),
    (f"exp3/bundles/{M15}-evoked.pt", EVOKED15),
    (f"exp3/bundles/{M15}-evoked_alt.pt", ALT15),
]
SHARDS = [os.path.join(OUT, "lr", f"{M15}_{ss}_{cs}.pt")
          for ss in ("injected", "evoked", "evoked_alt") for cs in ("N", "A", "B")]
GPU_THREADS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}


def emit_step(step, **fields):
    print("LABKIT_STEP " + json.dumps({"step": int(step), **fields}), flush=True)


def start_heartbeat(period_s=120):
    """The gauge run's attempt-1 lesson: a silent HF weight download froze the log and the stall
    watchdog killed the box. A daemon heartbeat keeps the log growing through genuinely-quiet work;
    run_to still caps a true hang."""
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


def main():
    os.makedirs(OUT, exist_ok=True)
    start_heartbeat()
    emit_step(0, phase="S0_fetch")
    fetch_inputs()
    print("LR_READY", flush=True)

    emit_step(1000, phase="S1_lr_reader", model=M15)
    cmd = [sys.executable, "-u", os.path.join(REPO, "src", "lr_reader.py"),
           "--capture", CAP15, "--evoked", EVOKED15, "--evoked-alt", ALT15]
    env = {**os.environ, **GPU_THREADS, "INTRO_MODEL": M15, "INTRO_RUN_DIR": OUT}
    print(f"RUN {' '.join(cmd)}  env=INTRO_MODEL={M15} INTRO_RUN_DIR={OUT}", flush=True)
    subprocess.run(cmd, env=env, cwd=REPO, check=True)

    missing = [s for s in SHARDS if not os.path.exists(s)]
    if missing:                                         # never report DONE with nothing to pull
        raise RuntimeError(f"lr shards missing: {[os.path.relpath(m, OUT) for m in missing]}")
    emit_step(2000, phase="lr_done", shards=len(SHARDS))
    print(f"outputs OK in {OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
        print("LR_DONE", flush=True)
    except Exception:
        print("LR_FATAL", flush=True)
        raise
