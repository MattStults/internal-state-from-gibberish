#!/usr/bin/env python3
"""Gated CONFOUND-CLOSING driver: one cheap 24GB box runs box_confound.py end-to-end (E2 pilot + gate,
E1 dose sweep, E3 prompt-only, E2 full, E5, the E4 trajectory re-forwards incl. the 7B leg, then the
exp2 CPU reanalysis riding post-GPU). Design frozen in
experiments/exp2_output_monitorability/reports/confound_closing_prereg.md.

The box HF-pulls every *.pt input (rsync excludes them); every stage is shard-resumable ON-BOX.
NOTE a partial pull does NOT reseed a relaunch: labkit's rsync-up excludes *.pt, so pulled shards
never travel back -- a fresh box redoes every unfinished stage from its own S0 pull.
Routed through the experimentfactory gate. BUILD/VALIDATE with --dry first.

  Launch:  .venv-driver/bin/python harness/run_confound.py [--gpu RTX3090] [--max-hours 6.0] [--dry]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EF = Path(os.environ.get("EXPERIMENTFACTORY_HOME", REPO.parent / "experiment_harness"))
sys.path.insert(0, str(EF))

import labkit                                                                       # noqa: E402
from experimentfactory import (authorized_run, Spec, GateBlocked, jsonl_recorder,    # noqa: E402
                               default_gate_log, evaluate, facts_from_spec, EXPERIMENT_SPEND_POLICY)

LABKIT_TAG = "v0.2.50"
# PARENT process runs the S5 CPU reanalysis in-process: BLAS single-threaded per worker so the joblib
# outer fan-out owns the cores (run_reanalysis lesson). box_confound raises the caps back to 8 for the
# GPU collect SUBPROCESSES (run_labkit convention).
THREAD_CAPS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
# run_labkit's Qwen2.5 collect deps (validated transformers pin) + run_reanalysis's pinned numeric stack
# for the S5 reader analysis (unpinned scipy 1.17.1 fails "SVD did not converge"; joblib needs
# return_as="generator_unordered", >=1.4).
DEPS = ["transformers==4.46.3", "accelerate", "numpy", "wordfreq",
        "scikit-learn==1.7.2", "scipy==1.15.3", "joblib==1.5.3",
        "matplotlib", "safetensors", "huggingface_hub"]


def hf_token():
    """Local HF token -> passed to the box so it PULLS the private captures/bundles from HF (rsync
    excludes *.pt; uploading ~1.5GB from the laptop's slow home link blew labkit's rsync ceiling before)."""
    for src in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if src:
            return src.strip()
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        return tok.read_text().strip()
    raise SystemExit("no HF token found (env HF_TOKEN or ~/.cache/huggingface/token) -- needed to pull the private dataset")


def main():
    ap = argparse.ArgumentParser()
    # RTX3090 = the cheap 24GB consumer tier: fits 1.5B AND the bf16 7B trajectory leg; on-box per-cell
    # shard resume + salvage pulls make the flakier cheap supply tolerable (a relaunch still redoes
    # unfinished stages -- pulled *.pt never rsync back up).
    ap.add_argument("--gpu", default="RTX3090", help="24GB consumer tier (7B bf16 leg needs the VRAM)")
    ap.add_argument("--min-vram", type=int, default=24000, help="7B bf16 trajectory leg needs ~16GB + headroom")
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--max-dph", type=float, default=0.20)   # cheap tier; the job is long, not hungry
    ap.add_argument("--max-hours", type=float, default=6.0,
                    help="wall-clock cap; run_to (watchdog) tracks this instead of fighting it")
    ap.add_argument("--min-bw", type=int, default=400, help="min host downlink Mbps (~18GB of weights + bundles)")
    ap.add_argument("--unverified", action="store_true",
                    help="allow residential/unverified hosts (cheapest tier; shard-resume covers flakiness)")
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()

    tok = hf_token()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    status_path = REPO / "runs" / "confound-status.json"
    events_path = REPO / "runs" / "confound-events.jsonl"
    events_path.unlink(missing_ok=True)
    print(f"live status -> {status_path}", flush=True)
    print(f"mid-run wakeup -> .venv-driver/bin/python -m labkit watch --events {events_path} "
          f"--status {status_path} --until warn", flush=True)

    job = labkit.script_job(
        workdir=str(REPO), entrypoint="python3 -u experiments/exp2_output_monitorability/box_confound.py",
        env={"INTRO_REPORT_DIR": "out", "HF_TOKEN": tok, **THREAD_CAPS},   # HF_TOKEN: box pulls the private data
        deps=DEPS,
        ready="CONFOUND_READY", done="CONFOUND_DONE", fatal="CONFOUND_FATAL",
        local_out=str(REPO / "runs" / "confound_box"), pull_subdir="out",   # partial pull = salvage only
        # (rsync-up excludes *.pt, so pulled shards do NOT reseed a relaunch; a fresh box redoes stages)
        # setup: pip + the S0 HF pull happen pre-READY but model downloads are per-stage; generous belts:
        # stall 45min covers the quiet 7B weight download mid-run between LABKIT_STEP lines.
        setup_to=1200, stall_to=2700, run_to=int(args.max_hours * 3600))

    kwargs = dict(
        # disk: docker (~8GB) + 1.5B+7B weights (~18GB) + ~1.5GB pulled bundles + E1's ~0.6GB capture
        # + HF cache duplication + S5 embed matrices etc.; 48 ran tight, 64 buys real headroom.
        provider=labkit.VastProvider(owner="confound", disk_gb=64.0,
                                     throttle_path=labkit.default_vast_throttle_path()),
        gpu=args.gpu, min_vram_mb=args.min_vram, pull_gb=4, est_run_s=int(args.max_hours * 3200),
        max_dph=args.max_dph, max_spend=args.max_spend, max_hours=args.max_hours,
        # per-PROJECT ledger (supported labkit param): max_spend caps THIS project's cumulative spend
        # instead of colliding with the shared lifetime ledger (~$27 across agents) -- the eng-review fix.
        ledger_path=str(REPO / "runs" / "confound-ledger.json"),
        max_acquire_tries=8, max_setup_retries=3, require_verified=not args.unverified,
        mk={"min_reliability": 0.97, "min_inet_down": args.min_bw},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id="confound")

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=16000,                                    # peak = the 7B bf16 trajectory leg
        output_incremental=True,                              # per-cell shards + per-bundle checkpoints
        shakedown_done=False,                                 # box_confound's first box run (observe mode surfaces it)
        eng_review="SHIP", sci_review="SHIP",                 # orchestration-only over reviewed measurement code
    )
    decision = evaluate(facts_from_spec(spec), EXPERIMENT_SPEND_POLICY)
    print(f"gate: authorized={decision.authorized} reasons={list(decision.reasons)}", flush=True)

    def _mock(**k):
        print(f"[DRY] would run: gpu={k['gpu']} max_spend=${k['max_spend']} max_hours={k['max_hours']} "
              f"run_id={k['run_id']}", flush=True)
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
    print("pulled -> runs/confound_box/  (pilot_verdict.json + e1_dose/e3_prompt/e2_full/e5_secret/e4_traj "
          "+ budget_results.json; score against the prereg bands offline)", flush=True)


if __name__ == "__main__":
    main()
