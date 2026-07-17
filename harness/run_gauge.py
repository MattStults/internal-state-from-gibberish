#!/usr/bin/env python3
"""Gated GAUGE-TRAJECTORY driver (task #20, Story A vs B): one cheap 24GB box runs box_gauge.py --
state_trajectory --arm gauge:evoked at 1.5B + 7B over the evoked bundles' saved gauge free-association
texts. Tiny run (~15-30 min): two model loads + ~104 single re-forwards each; score offline against the
neutral-gauge null (see reports/gauge_trajectory_verdict.json).

The box HF-pulls every *.pt input (rsync excludes them). Routed through the experimentfactory gate.
BUILD/VALIDATE with --dry first.

  Launch:  .venv-driver/bin/python harness/run_gauge.py [--gpu RTX3090] [--max-hours 1.0] [--dry]
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
THREAD_CAPS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
# run_labkit's validated Qwen2.5 collect deps; no S5-style CPU analysis rides this box.
DEPS = ["transformers==4.46.3", "accelerate", "numpy", "safetensors", "huggingface_hub"]


def hf_token():
    """Local HF token -> the box PULLS the private captures/bundles from HF (rsync excludes *.pt)."""
    for src in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if src:
            return src.strip()
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        return tok.read_text().strip()
    raise SystemExit("no HF token found (env HF_TOKEN or ~/.cache/huggingface/token) -- needed to pull the private dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="RTX3090", help="24GB consumer tier (7B bf16 leg needs the VRAM)")
    ap.add_argument("--min-vram", type=int, default=24000)
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--max-dph", type=float, default=0.20)
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--min-bw", type=int, default=400, help="min host downlink Mbps (~18GB of weights)")
    ap.add_argument("--unverified", action="store_true")
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()

    tok = hf_token()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    status_path = REPO / "runs" / "gauge-status.json"
    events_path = REPO / "runs" / "gauge-events.jsonl"
    events_path.unlink(missing_ok=True)
    print(f"live status -> {status_path}", flush=True)

    job = labkit.script_job(
        workdir=str(REPO), entrypoint="python3 -u experiments/exp2_output_monitorability/box_gauge.py",
        # HF_HUB_DISABLE_XET: attempt 1 died to the stall watchdog with the log frozen mid weight
        # download (hf-xet 1.5.1 prints no progress and can hang); the plain HTTP backend streams tqdm
        # to stderr, which keeps the log growing.
        env={"INTRO_REPORT_DIR": "out", "HF_TOKEN": tok, "HF_HUB_DISABLE_XET": "1", **THREAD_CAPS},
        deps=DEPS,
        ready="GAUGE_READY", done="GAUGE_DONE", fatal="GAUGE_FATAL",
        local_out=str(REPO / "runs" / "gauge_box"), pull_subdir="out",
        # stall 45min = box_confound's proven value for quiet weight downloads; box_gauge also
        # heartbeats every 2min, so run_to (not stall) is the real cap on a true hang.
        setup_to=1200, stall_to=2700, run_to=int(args.max_hours * 3600))

    kwargs = dict(
        # disk: docker (~8GB) + 1.5B+7B weights (~18GB) + ~130MB pulled bundles/captures
        provider=labkit.VastProvider(owner="gauge", disk_gb=48.0,
                                     throttle_path=labkit.default_vast_throttle_path()),
        gpu=args.gpu, min_vram_mb=args.min_vram, pull_gb=1, est_run_s=2400,
        max_dph=args.max_dph, max_spend=args.max_spend, max_hours=args.max_hours,
        # per-PROJECT ledger shared with the confound run (same project budget line)
        ledger_path=str(REPO / "runs" / "confound-ledger.json"),
        max_acquire_tries=8, max_setup_retries=3, require_verified=not args.unverified,
        mk={"min_reliability": 0.97, "min_inet_down": args.min_bw},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id="gauge")

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=16000,                                    # peak = the 7B bf16 leg
        output_incremental=True,                              # per-leg TRAJ shards
        shakedown_done=False,                                 # box_gauge's first box run (observe mode)
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
    print("pulled -> runs/gauge_box/gauge_traj/trajectory/  (score offline -> "
          "reports/gauge_trajectory_verdict.json)", flush=True)


if __name__ == "__main__":
    main()
