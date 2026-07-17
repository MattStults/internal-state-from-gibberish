#!/usr/bin/env python3
"""Gated MC-LETTER ELICITED reader driver (prereg: exp2 reports/mc_reader_prereg.md): one box runs
box_mc.py -- src/mc_reader.py per reader scale over the FIXED 1.5B stream pool (MC-letter readout,
12 cyclic Latin-square orderings, framings {elicited-MC, passive-MC} x reasoning {no-CoT direct,
greedy-capped-CoT cap=256}). 16 atomic shards per reader; score offline (analysis/mc_offline.py ->
reports/mc_reader_results.json).

TWO planned invocations (each --dry first):
  .venv-driver/bin/python harness/run_mc.py --dry
  .venv-driver/bin/python harness/run_mc.py
      # RTX3090 tier, readers qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b,qwen3-1.7b, run_id mc
  .venv-driver/bin/python harness/run_mc.py --models qwen2.5-14b --gpu "RTX A6000" \
      --max-dph 0.85 --max-hours 2 --dry   (then without --dry)
      # A6000 48GB tier, run_id mc-14b (min_vram/disk auto-raise for 14b); waits for the 14B
      # supply sampler to free the tier. Does NOT touch runs/lr-* or runs/elicit-14b-*.

The box HF-pulls every *.pt input (rsync excludes them). Routed through the experimentfactory gate.
BUILD/VALIDATE with --dry first; this driver does not launch a box under --dry.
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
# run_labkit's validated Qwen2.5 deps; no CPU analysis rides these boxes.
_BASE_DEPS = ["accelerate", "numpy", "safetensors", "huggingface_hub"]


def deps_for(slugs):
    """Mirror run_labkit.deps_for: the qwen3 architecture needs transformers>=4.51 (the 4.46.3 pin
    can't load it -> 'model type qwen3 not recognized'); Qwen2.5 stays on the validated 4.46.3 pin.
    Both reviews missed this (they checked code, not box env vs the qwen3 reader), so the qwen3
    cross-family reader is launched as its OWN run (--models qwen3-1.7b) to keep the Qwen2.5 readers
    on 4.46.3 -- version-matched to the LR ceiling for the workspace-tax subtraction."""
    tf = "transformers>=4.51,<5.0" if any(s.startswith("qwen3") for s in slugs) else "transformers==4.46.3"
    return [tf] + _BASE_DEPS
WEIGHTS_GB = {"1.5b": 3.5, "1.7b": 4.0, "3b": 6.5, "7b": 16.0, "14b": 29.0}   # bf16 footprint
VRAM_BY_SIZE = {"1.5b": 24000, "1.7b": 24000, "3b": 24000, "7b": 24000, "14b": 40000}


def disk_for(sizes):
    """Container disk GB: docker image (~10) + bf16 weights + slack (4). FLOOR 56 whenever a 14b
    reader rides: the 14B collect precedent (run_labkit.py) -- a 43GB container boots, then dies
    disk-full mid weight download, paying for the privilege."""
    gb = round(10.0 + sum(WEIGHTS_GB.get(z, 16.0) for z in sizes) + 4.0, 1)
    return max(gb, 56.0) if "14b" in sizes else gb


def vram_for(sizes):
    """min-vram MB: max over reader sizes, but FLOOR 40000 whenever a 14b reader rides (prereg:
    the A6000 48GB tier; 14B bf16 + KV + fp32 logit chunks needs the headroom)."""
    v = max(VRAM_BY_SIZE.get(z, 24000) for z in sizes)
    return max(v, 40000) if "14b" in sizes else v


def hf_token():
    """Local HF token -> the box PULLS the private capture/bundle from HF (rsync excludes *.pt)."""
    for src in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if src:
            return src.strip()
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        return tok.read_text().strip()
    raise SystemExit("no HF token found (env HF_TOKEN or ~/.cache/huggingface/token) -- needed to pull the private dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2.5-1.5b,qwen2.5-3b,qwen2.5-7b,qwen3-1.7b",
                    help="comma-separated reader slugs (the 14b run passes qwen2.5-14b alone)")
    ap.add_argument("--gpu", default="RTX3090",
                    help="RTX3090 = 24GB consumer tier; 'RTX A6000' = the 48GB tier for 14b")
    ap.add_argument("--min-vram", type=int, default=None, help="default: floored over reader sizes")
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--max-dph", type=float, default=0.25)
    ap.add_argument("--max-hours", type=float, default=2.0)
    ap.add_argument("--min-bw", type=int, default=400, help="min host downlink Mbps (weights)")
    ap.add_argument("--run-id", default=None, help="default: mc / mc-14b by model set")
    ap.add_argument("--unverified", action="store_true")
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()

    slugs = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [s.split("-")[-1].lower() for s in slugs]
    run_id = args.run_id or ("mc-14b" if sizes == ["14b"] else "mc")
    min_vram = args.min_vram or vram_for(sizes)
    disk_gb = disk_for(sizes)
    vram_est = 32000 if "14b" in sizes else 20000

    tok = hf_token()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    status_path = REPO / "runs" / f"{run_id}-status.json"
    events_path = REPO / "runs" / f"{run_id}-events.jsonl"
    events_path.unlink(missing_ok=True)
    print(f"live status -> {status_path}", flush=True)

    job = labkit.script_job(
        workdir=str(REPO),
        entrypoint=("python3 -u experiments/exp2_output_monitorability/box_mc.py "
                    f"--models {','.join(slugs)}"),
        # HF_HUB_DISABLE_XET: the gauge run's attempt 1 died to the stall watchdog with the log
        # frozen mid weight download (hf-xet prints no progress); the plain HTTP backend streams
        # tqdm to stderr, which keeps the log growing. box_mc also heartbeats every 2min.
        env={"INTRO_REPORT_DIR": "out", "HF_TOKEN": tok, "HF_HUB_DISABLE_XET": "1", **THREAD_CAPS},
        deps=deps_for(slugs),
        ready="MC_READY", done="MC_DONE", fatal="MC_FATAL",
        local_out=str(REPO / "runs" / "mc_box"), pull_subdir="out",
        # stall 45min = the proven value for quiet weight downloads; run_to generous (the real cap
        # on a true hang).
        setup_to=1800, stall_to=2700, run_to=int(args.max_hours * 3600))

    kwargs = dict(
        provider=labkit.VastProvider(owner=run_id, disk_gb=disk_gb,
                                     throttle_path=labkit.default_vast_throttle_path()),
        gpu=args.gpu, min_vram_mb=min_vram, pull_gb=1, est_run_s=int(args.max_hours * 1800),
        max_dph=args.max_dph, max_spend=args.max_spend, max_hours=args.max_hours,
        # per-PROJECT ledger shared with the confound/gauge/lr/elicit runs (same project budget line)
        ledger_path=str(REPO / "runs" / "confound-ledger.json"),
        max_acquire_tries=8, max_setup_retries=3, require_verified=not args.unverified,
        mk={"min_reliability": 0.97, "min_inet_down": args.min_bw},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id=run_id)

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=vram_est,
        output_incremental=True,                     # per-(reader, set, framing, reasoning) shards
        shakedown_done=False,                        # box_mc's first box runs (observe mode)
        eng_review="SHIP", sci_review="SHIP",        # orchestration-only over unit-tested scoring code
    )
    decision = evaluate(facts_from_spec(spec), EXPERIMENT_SPEND_POLICY)
    print(f"gate: authorized={decision.authorized} reasons={list(decision.reasons)}", flush=True)

    def _mock(**k):
        print(f"[DRY] would run: gpu={k['gpu']} min_vram={k['min_vram_mb']} disk={disk_gb}GB "
              f"max_dph=${k['max_dph']} max_spend=${k['max_spend']} max_hours={k['max_hours']} "
              f"run_id={k['run_id']} models={slugs}", flush=True)
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
    print("pulled -> runs/mc_box/mc/  (score offline: "
          ".venv/bin/python experiments/exp2_output_monitorability/analysis/mc_offline.py)", flush=True)


if __name__ == "__main__":
    main()
