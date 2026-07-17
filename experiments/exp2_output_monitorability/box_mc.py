"""On-box orchestrator for the MC-LETTER ELICITED reader run (prereg: reports/mc_reader_prereg.md).

One box, N reader scales (--models): src/mc_reader.py scores the FIXED 1.5B stream pool
(injected s60 / injected s0 / evoked s1 / evoked-neutral) under each reader -- MC-letter readout
over 12 cyclic Latin-square orderings, framings {elicited-MC, passive-MC} x reasoning {no-CoT
direct, greedy-capped-CoT cap=256}. 16 atomic shards per reader land in out/mc/; scoring is OFFLINE
(analysis/mc_offline.py).

Two planned invocations (see harness/run_mc.py):
  RTX3090 tier:   --models qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b,qwen3-1.7b
  RTX A6000 tier: --models qwen2.5-14b   (when the 14B supply sampler frees the tier)

OWN-POOL DIAGONAL (scale-grid checklist B6; prereg lr_scale_grid_prereg.md "MC self-report
diagonal"): --own-pool makes each reader read its OWN generator-size evoked bundle
(exp3/bundles/<slug>-evoked.pt -> runs/_ind/<slug>/data/), sets {evoked, evoked_s0} only (the
injected pool exists only at 1.5B), 8 shards per reader. Use a FRESH INTRO_REPORT_DIR: diagonal
shard filenames coincide with default-pool ones; mc_reader FATALs (assert_shard_source) rather
than resume across pools, but a clean dir avoids the trip entirely.

Stages (idempotent; mc_reader MC_SKIPs existing same-pool shards):
  S0  HF-pull the inputs rsync excludes (*.pt): the exp1 1.5B capture and the exp3 evoked
      bundle(s), from ErrareHumanumEst/internal-state-from-gibberish (HF_TOKEN).
  S1  src/mc_reader.py per reader scale (one subprocess per model -> clean GPU teardown),
      INTRO_RUN_DIR=out (shards -> out/mc/).

Markers: MC_READY / MC_DONE / MC_FATAL. Driven by harness/run_mc.py (gated).
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
DIAG_SETS = ("evoked", "evoked_s0")                      # own-pool diagonal: no injected pool >1.5B
FRAMINGS = ("elicited", "passive")
REASONINGS = ("direct", "cot")
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


def evoked_path(slug):
    return os.path.join(REPO, "runs", "_ind", slug, "data", f"{slug}-evoked.pt")


def fetches_for(models, own_pool=False):
    """S0 fetch list. Default: the fixed 1.5B pool (exp1 capture + evoked bundle). Own-pool
    diagonal: each reader's OWN generator-size evoked bundle, no capture (evoked-only sets;
    mc_reader's first_ids gate passes on the diagonal by reader == generator)."""
    if not own_pool:
        return list(FETCHES)
    return [(f"exp3/bundles/{slug}-evoked.pt", evoked_path(slug)) for slug in models]


def fetch_inputs(fetches):
    from huggingface_hub import hf_hub_download
    for fname, dest in fetches:
        if os.path.exists(dest):
            print(f"S0 have {os.path.relpath(dest, REPO)}", flush=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(hf_hub_download(HF_DATASET, fname, repo_type="dataset"), dest)
        print(f"S0 fetched {fname} -> {os.path.relpath(dest, REPO)}", flush=True)
    missing = [f for f, d in fetches if not os.path.exists(d)]
    if missing:
        raise RuntimeError(f"S0: inputs missing after HF pull: {missing}")


def sets_for(own_pool=False):
    return DIAG_SETS if own_pool else STREAM_SETS


def shards_for(slug, own_pool=False):
    return [os.path.join(OUT, "mc", f"{slug}_{ss}_{fr}_{rs}.pt")
            for ss in sets_for(own_pool) for fr in FRAMINGS for rs in REASONINGS]


def reader_cmd(slug, batch, own_pool=False):
    """The mc_reader subprocess argv. Default = the certified invocation VERBATIM; own-pool adds
    only the B6 pool wiring flags (--stream-source = the reader's own slug, evoked sets, own
    bundle, no capture)."""
    base = [sys.executable, "-u", os.path.join(REPO, "src", "mc_reader.py")]
    if own_pool:
        return base + ["--evoked", evoked_path(slug), "--stream-source", slug,
                       "--sets", ",".join(DIAG_SETS), "--batch", str(batch)]
    return base + ["--capture", CAP15, "--evoked", EVOKED15, "--batch", str(batch)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b,qwen3-1.7b",
                    help="comma-separated reader slugs, run smallest-first")
    ap.add_argument("--batch", type=int, default=24)  # util: batch-8 wasted 24GB; 24 is safe for <=7B+CoT-256 and ~3x the direct-read throughput (numerically identical -- reviewed padding path)
    ap.add_argument("--own-pool", action="store_true",
                    help="MC self-report diagonal: each reader reads its OWN generator-size "
                         "evoked bundle (sets evoked,evoked_s0; 8 shards/reader)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    os.makedirs(OUT, exist_ok=True)
    start_heartbeat()
    emit_step(0, phase="S0_fetch")
    fetch_inputs(fetches_for(models, args.own_pool))
    print("MC_READY", flush=True)

    n_shards = len(sets_for(args.own_pool)) * len(FRAMINGS) * len(REASONINGS)
    for i, slug in enumerate(models):
        emit_step(1000 * (i + 1), phase="S1_mc_reader", model=slug)
        if not args.own_pool and all(os.path.exists(s) for s in shards_for(slug)):
            # own-pool never short-circuits here: shard names do not carry the pool, so the
            # skip decision belongs to mc_reader's assert_shard_source (cross-pool = FATAL).
            print(f"S1 SKIP {slug}: all {n_shards} shards exist", flush=True)
            continue
        cmd = reader_cmd(slug, args.batch, args.own_pool)
        env = {**os.environ, **GPU_THREADS, "INTRO_MODEL": slug, "INTRO_RUN_DIR": OUT}
        print(f"RUN {' '.join(cmd)}  env=INTRO_MODEL={slug} INTRO_RUN_DIR={OUT}", flush=True)
        subprocess.run(cmd, env=env, cwd=REPO, check=True)

    missing = [s for slug in models for s in shards_for(slug, args.own_pool)
               if not os.path.exists(s)]
    if missing:                                         # never report DONE with nothing to pull
        raise RuntimeError(f"mc shards missing: {[os.path.relpath(m, OUT) for m in missing]}")
    emit_step(9000, phase="mc_done", shards=n_shards * len(models))
    print(f"outputs OK in {OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
        print("MC_DONE", flush=True)
    except Exception:
        print("MC_FATAL", flush=True)
        raise
