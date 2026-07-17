#!/usr/bin/env python3
"""Gated exp3 ANALYSIS driver: run the reader analysis on a cheap rented box instead of Matt's Mac (which the
local run crashed). Stages runs/_ind as a tar (labkit rsync excludes *.pt), ships code + tar (~250MB), the box
extracts embeds from HF and runs run_induction with per-bundle checkpointing; out/ (JSON + png) is pulled even
on failure. Routed through the experimentfactory gate (observe), same as the collection driver.

  Launch:  .venv-driver/bin/python harness/run_exp3_analysis.py [--gpu RTX4090] [--max-spend 15]
"""
import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EF = Path(os.environ.get("EXPERIMENTFACTORY_HOME", REPO.parent / "experiment_harness"))
sys.path.insert(0, str(EF))

import labkit                                                                      # noqa: E402
from experimentfactory import (authorized_run, Spec, GateBlocked, jsonl_recorder,   # noqa: E402
                               default_gate_log, evaluate, facts_from_spec, EXPERIMENT_SPEND_POLICY)

LABKIT_TAG = "v0.2.50"
THREAD_CAPS = {k: "16" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def stage_bundles():
    """Tar runs/_ind (labkit's rsync excludes *.pt, so bundles must ship inside a tar)."""
    tar_path = REPO / "runs" / "exp3_bundles.tar"
    with tarfile.open(tar_path, "w") as tf:                    # uncompressed: .pt payloads are already binary
        tf.add(REPO / "runs" / "_ind", arcname="runs/_ind")
    print(f"staged {tar_path} ({tar_path.stat().st_size/1e6:.0f} MB)", flush=True)
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX4090", help="cheapest reliable datacenter tier; the job is CPU-bound")
    ap.add_argument("--max-spend", type=float, default=15.0)
    ap.add_argument("--max-dph", type=float, default=1.00)
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()

    stage_bundles()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    status_path = REPO / "runs" / "exp3-analysis-status.json"
    events_path = REPO / "runs" / "exp3-analysis-events.jsonl"
    events_path.unlink(missing_ok=True)

    job = labkit.script_job(
        workdir=str(REPO), entrypoint="python3 -u experiments/exp3_induction_and_scale/box_analyze.py",
        env={"INTRO_REPORT_DIR": "out", **THREAD_CAPS},
        deps=["scikit-learn", "numpy", "matplotlib", "safetensors", "huggingface_hub"],   # torch on the image
        ready="ANALYZE_READY", done="ANALYZE_DONE", fatal="ANALYZE_FATAL",
        local_out=str(REPO / "experiments/exp3_induction_and_scale/reports/box"), pull_subdir="out",
        setup_to=900, stall_to=1800, run_to=5400)              # per-seed prints every ~1-2 min; 30min stall guard

    kwargs = dict(
        provider=labkit.VastProvider(owner="exp3-analysis", disk_gb=24.0,
                                     throttle_path=labkit.default_vast_throttle_path()),
        gpu=args.gpu, min_vram_mb=8000, pull_gb=2, est_run_s=3600,
        max_dph=args.max_dph, max_spend=args.max_spend, max_hours=2.0,
        max_acquire_tries=8, max_setup_retries=3, require_verified=True,
        mk={"min_reliability": 0.97, "min_inet_down": 600},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id="exp3-analysis")

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=2000,                                      # CPU job; GPU idles
        output_incremental=True,                               # per-bundle JSON checkpoint (the crash lesson)
        shakedown_done=False,                                  # honest: this entrypoint's first box run IS the shakedown
        eng_review="SHIP", sci_review="SHIP",                  # analysis unit reviewed at 9b24b40; observe mode
    )
    decision = evaluate(facts_from_spec(spec), EXPERIMENT_SPEND_POLICY)
    print(f"gate: authorized={decision.authorized} reasons={list(decision.reasons)}", flush=True)

    def _mock(**k):
        print(f"[DRY] would run: gpu={k['gpu']} max_spend=${k['max_spend']} run_id={k['run_id']}", flush=True)
        return "DRY_OK"

    res = authorized_run(spec, mode="observe", runner=_mock if args.dry else None,
                         recorder=jsonl_recorder(default_gate_log()))
    if isinstance(res, GateBlocked):
        print("BLOCKED ($0):", *res.reasons, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    if args.dry:
        print("dry ok", flush=True)
        return
    print(f"outcome={getattr(res, 'outcome', '?')} ok={getattr(res, 'ok', '?')} "
          f"spend=${getattr(res, 'spend_usd', '?')} partial={getattr(res, 'partial_pull', None)} "
          f"log={getattr(res, 'log_path', '?')}", flush=True)
    if not getattr(res, "ok", False):
        print("reasons:", getattr(res, "reasons", None), "error:", getattr(res, "error", None), file=sys.stderr)
        sys.exit(1)
    print("pulled -> experiments/exp3_induction_and_scale/reports/box/", flush=True)


if __name__ == "__main__":
    main()
