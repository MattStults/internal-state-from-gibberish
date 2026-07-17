"""On-box orchestrator for the GAUGE-TRAJECTORY run (task #20: Story A vs Story B).

Discriminates two readings of the E4 null (evoked z ~= 0.05 sigma under the anti-word task):
  Story A  the persona state IS installed during free behavior and the anti-word task DISPLACES it
           -> gauge z clearly elevated at early/mid cuts
  Story B  the persona never writes the injected-vector direction at all (v_read != v_write)
           -> gauge z ~= floor too

One 24GB box, two legs of src/state_trajectory.py --arm gauge:evoked (1.5B then 7B), re-forwarding
the evoked bundles' saved GAUGE free-association texts under the gauge context (induction text alone
+ GAUGE_PROBE). The 7B leg runs only if its bundle carries gauge texts. FIDELITY CAVEAT (documented
in state_trajectory): gauge texts are re-tokenized, so positions are those of the re-tokenized
stream, not bit-exact to generation time.

Stages (idempotent; state_trajectory TRAJ_SKIPs existing shards):
  S0  HF-pull the inputs rsync excludes (*.pt): exp1 1.5B+7B captures (projection vectors) and the
      exp3 evoked bundles (gauge texts), from ErrareHumanumEst/internal-state-from-gibberish (HF_TOKEN).
  S1  state_trajectory --arm gauge:evoked at 1.5B, then 7B, INTRO_RUN_DIR=out/gauge_traj.

Markers: GAUGE_READY / GAUGE_DONE / GAUGE_FATAL. Driven by harness/run_gauge.py (gated).
NEVER run this on the Mac -- it loads models.
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
M15, M7 = "qwen2.5-1.5b", "qwen2.5-7b"
OUT = os.path.abspath(os.environ.get("INTRO_REPORT_DIR") or os.path.join(REPO, "out"))
CAP15 = os.path.join(REPO, "runs", M15, "data", "covert_collect.pt")
CAP7 = os.path.join(REPO, "runs", M7, "data", "covert_collect.pt")
EVOKED15 = os.path.join(REPO, "runs", "_ind", M15, "data", f"{M15}-evoked.pt")
EVOKED7 = os.path.join(REPO, "runs", "_ind", M7, "data", f"{M7}-evoked.pt")
FETCHES = [
    ("qwen2.5-1.5b-gen.pt", CAP15),
    ("qwen2.5-7b-gen-matched.pt", CAP7),
    (f"exp3/bundles/{M15}-evoked.pt", EVOKED15),
    (f"exp3/bundles/{M7}-evoked.pt", EVOKED7),
]
GPU_THREADS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}


def emit_step(step, **fields):
    print("LABKIT_STEP " + json.dumps({"step": int(step), **fields}), flush=True)


def start_heartbeat(period_s=120):
    """Attempt 1 was killed by the stall watchdog during a silent HF weight download. A daemon
    heartbeat keeps the log growing through genuinely-quiet work; run_to still caps a true hang."""
    t0 = time.time()

    def beat():
        while True:
            time.sleep(period_s)
            print(f"HEARTBEAT t={int(time.time() - t0)}s", flush=True)

    threading.Thread(target=beat, daemon=True).start()


def run_sub(script_rel, args, env_extra, step_base):
    cmd = [sys.executable, "-u", os.path.join(REPO, script_rel)] + list(args)
    env = {**os.environ, **GPU_THREADS, "INTRO_STEP_BASE": str(step_base), **env_extra}
    shown = {k: v for k, v in env.items() if k.startswith("INTRO_")}
    print(f"RUN {' '.join(cmd)}  env={shown}", flush=True)
    subprocess.run(cmd, env=env, cwd=REPO, check=True)


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


def has_gauge(bundle_path):
    import torch
    b = torch.load(bundle_path, map_location="cpu", weights_only=False)
    return "gauge" in b


def main():
    os.makedirs(OUT, exist_ok=True)
    start_heartbeat()
    emit_step(0, phase="S0_fetch")
    fetch_inputs()
    print("GAUGE_READY", flush=True)

    traj_dir = os.path.join(OUT, "gauge_traj")
    legs = [(M15, EVOKED15, CAP15), (M7, EVOKED7, CAP7)]
    ran = []
    for i, (slug, bundle, vec) in enumerate(legs):
        emit_step(1000 + 10 * i, phase="S1_gauge_traj", model=slug)
        shard = os.path.join(traj_dir, "trajectory", f"{slug}_gauge-evoked.pt")
        if os.path.exists(shard):
            print(f"S1 SKIP: {os.path.relpath(shard, OUT)} exists", flush=True)
            ran.append(shard)
            continue
        if not has_gauge(bundle):                       # 7B leg is conditional on gauge texts
            print(f"S1 SKIP: {os.path.relpath(bundle, REPO)} carries no gauge texts", flush=True)
            continue
        run_sub("src/state_trajectory.py",
                ["--arm", "gauge:evoked", "--bundle", bundle, "--vectors-from", vec],
                {"INTRO_MODEL": slug, "INTRO_RUN_DIR": traj_dir}, 1000 + 10 * i)
        ran.append(shard)

    missing = [s for s in ran if not os.path.exists(s)]
    if not ran or missing:                              # never report DONE with nothing to pull
        raise RuntimeError(f"gauge shards missing: ran={ran} missing={missing}")
    emit_step(2000, phase="gauge_done", shards=len(ran))
    print(f"outputs OK in {OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
        print("GAUGE_DONE", flush=True)
    except Exception:
        print("GAUGE_FATAL", flush=True)
        raise
