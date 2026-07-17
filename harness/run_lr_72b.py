#!/usr/bin/env python3
"""Gated LR-72B driver (prereg: exp2 reports/lr_72b_prereg.md): ONE contiguous 2xH100(-NVL) box
self-hosts Qwen2.5-72B-Instruct under vLLM and runs box_lr_72b.py end-to-end -- S0 start the vLLM
server (wait /health) -> S1 generate secret_word + secret_sustain -> the conditional-$6 DECISION
GATE for Phase 2 (evoked + evoked_alt) -> S3 teacher-force-score the 72B self-read diagonal via
prompt_logprobs. No manual phases between create and teardown.

Same discipline as every other box (run_lr_grid pattern):
  - experimentfactory gate (authorized_run); BUILD/VALIDATE with --dry first ($0 mock runner).
  - per-PROJECT ledger runs/confound-ledger.json; ledger cap $10 for this run (Matt 2026-07-12;
    expected ~$5). The runtime $6 gate is a within-run scope decision (Phase 2), disclosed -- it
    is NOT the ledger cap.
  - deadman self-destruct: labkit arms provider-side autodestroy at create from max-hours (+
    teardown buffer), so an orphaned box kills itself even if this driver dies mid-run.
  - status heartbeat + monitor-ready events; a `labkit watch` hint printed at launch.
  - HIGH-BANDWIDTH host filter (min ~2 Gbps downlink): the 144GB weight download is the real idle
    risk on a 2xH100 box -- a slow host burns paid GPU time waiting on weights.

TWO planned invocations (each --dry first):
  .venv-driver/bin/python harness/run_lr_72b.py --smoke [--dry]
      # tiny slice: 2-3 streams; the box verifies prompt_logprobs teacher-forces + tokenizer
      #   parity + util logging (the smoke's whole job), pulls runs/lr_72b_smoke_box/.
  .venv-driver/bin/python harness/run_lr_72b.py [--dry]
      # full run -> pulls runs/lr_72b_box/ (lr_72b/ shards + the raw 72B secret streams).
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABKIT_TAG = "v0.2.50"
LEDGER_PATH = str(REPO / "runs" / "confound-ledger.json")
PROJECT_CAP = 5.0                     # the original prereg budget line (shared ledger)
# Ledger cap for THIS run: $10 (Matt 2026-07-12, prereg "ledger cap $10; expected ~$5"). The
# within-run $6 gate governs Phase 2 scope (evoked), NOT the ledger cap.
RUN_AUTHORIZED_USD = 10.0
PHASE2_MAX_USD = 6.0                  # shared with box_lr_72b.phase2_gate (the runtime scope gate)
DEADMAN_BUFFER_S = 1800
THREAD_CAPS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}

# 2xH100 tier (prereg: 2xH100(-NVL), tensor-parallel). H100 = 80GB/card; the 144GB bf16 weights
# split across the two cards under TP.
DEFAULT_GPU = "H100"
NUM_GPUS = 2
DEFAULT_MIN_VRAM = 80000              # per-card (H100 80GB); labkit multiplies by num_gpus
DEFAULT_MIN_BW = 2000                 # Mbps downlink floor (144GB pull -> minutes, not tens)

# DISK floor: the 72B bf16 weights are the dominant term (~144GB) and the real idle risk (a
# container that boots then dies disk-full mid weight download still bills). Qwen2.5-72B =
# 72.7B params -> ~145GB bf16; round to 144 and carry image + generous slack (HF cache dupes +
# shards + the pulled streams).
WEIGHTS_72B_GB = 144.0
IMAGE_GB = 12.0                       # vLLM image is heavier than the HF-only images
DISK_SLACK_GB = 20.0


def entrypoint_for(smoke=False):
    ep = "python3 -u experiments/exp2_output_monitorability/box_lr_72b.py"
    return ep + (" --smoke" if smoke else "")


def run_id_for(smoke=False):
    return "lr-72b-smoke" if smoke else "lr-72b"


def disk_for_72b():
    """Container disk floor = vLLM image + 72B bf16 weights + slack."""
    return round(IMAGE_GB + WEIGHTS_72B_GB + DISK_SLACK_GB, 1)


def deps_for():
    """Box env deps: vLLM (the server) + the generation-side deps that ride the same box
    (wordfreq so the word-free filter is live; transformers for the local tokenizer + primers).
    Pinned transformers to the validated LR-grid pin for tokenizer parity with the smaller sizes.
    vllm is unpinned-major here (the box image provides a CUDA-matched build); the driver's clean
    review confirms the launched image before the first paid launch."""
    return ["vllm", "transformers==4.46.3", "accelerate", "numpy", "huggingface_hub",
            "wordfreq"]


def provider_kwargs(run_id, disk_gb, max_hours, throttle_path=None):
    """VastProvider kwargs; default_deadman_s MUST ride max_hours (E1 blocker parity with
    run_lr_grid): labkit's create() clamps every requested deadman to min(default, requested) and
    the provider default is 6h -- without this override a long run self-destructs early."""
    return dict(owner=run_id, disk_gb=disk_gb, throttle_path=throttle_path,
                default_deadman_s=int(max_hours * 3600) + DEADMAN_BUFFER_S)


def is_rsync_flake(reasons=None, error=None):
    blobs = list(reasons or [])
    if error:
        blobs.append(str(error))
    pat = re.compile(r"rsync[^\n]*\b255\b")
    return any(pat.search(str(b)) for b in blobs)


def remaining_budget(ledger_path=LEDGER_PATH, cap=PROJECT_CAP):
    if not os.path.exists(ledger_path):
        return float(cap)
    with open(ledger_path) as f:
        led = json.load(f)
    return float(cap) - float(sum(led.values()))


def project_phase2(spend_so_far, phase1_arms, phase1_spend, max_usd=PHASE2_MAX_USD):
    """The $6 Phase-2 (evoked) projection -- the SAME rule as the box's runtime gate
    (box_lr_72b.phase2_gate), reused here so the driver's advice never diverges from what the box
    actually decides. Loads the box module lazily (stdlib-only at its top level)."""
    box = _load_box()
    return box.phase2_gate(spend_so_far=spend_so_far, phase1_arms=phase1_arms,
                           phase1_spend=phase1_spend, max_usd=max_usd)


def _load_box():
    p = REPO / "experiments" / "exp2_output_monitorability" / "box_lr_72b.py"
    if "box_lr_72b" in sys.modules:
        return sys.modules["box_lr_72b"]
    spec = importlib.util.spec_from_file_location("box_lr_72b", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["box_lr_72b"] = mod
    spec.loader.exec_module(mod)
    return mod


def hf_token():
    for src in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if src:
            return src.strip()
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        return tok.read_text().strip()
    raise SystemExit("no HF token found (env HF_TOKEN or ~/.cache/huggingface/token)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=DEFAULT_GPU, help="2xH100(-NVL) tier (prereg)")
    ap.add_argument("--num-gpus", type=int, default=NUM_GPUS, help="tensor-parallel-size (2)")
    ap.add_argument("--min-vram", type=int, default=DEFAULT_MIN_VRAM, help="per-card VRAM floor")
    ap.add_argument("--max-spend", type=float, default=None,
                    help="cumulative project cap against the shared ledger (default: current "
                         "ledger spend + the $10 authorized for this run)")
    ap.add_argument("--max-dph", type=float, default=6.0, help="2xH100 tier ceiling $/hr")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock cap -> labkit deadline -> provider deadman "
                         "(default 1.0 smoke / 4.0 full)")
    ap.add_argument("--min-bw", type=int, default=DEFAULT_MIN_BW,
                    help="min host downlink Mbps (the 144GB weight pull is the real idle risk)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny slice: the box verifies prompt_logprobs teacher-forces + tokenizer "
                         "parity + util logging (2-3 streams, no $6 Phase 2)")
    ap.add_argument("--unverified", action="store_true")
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()
    max_hours = args.max_hours or (1.0 if args.smoke else 4.0)
    spent = PROJECT_CAP - remaining_budget()
    max_spend = args.max_spend if args.max_spend is not None else round(
        spent + RUN_AUTHORIZED_USD, 2)
    run_id = run_id_for(args.smoke)
    local_out = REPO / "runs" / ("lr_72b_smoke_box" if args.smoke else "lr_72b_box")
    disk_gb = disk_for_72b()

    EF = Path(os.environ.get("EXPERIMENTFACTORY_HOME", REPO.parent / "experiment_harness"))
    sys.path.insert(0, str(EF))
    import labkit
    from experimentfactory import (authorized_run, Spec, GateBlocked, jsonl_recorder,
                                   default_gate_log, evaluate, facts_from_spec,
                                   EXPERIMENT_SPEND_POLICY)

    tok = hf_token()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                  text=True).strip()
    status_path = REPO / "runs" / f"{run_id}-status.json"
    events_path = REPO / "runs" / f"{run_id}-events.jsonl"
    events_path.unlink(missing_ok=True)
    print(f"live status -> {status_path}", flush=True)
    print(f"mid-run wakeup -> .venv-driver/bin/python -m labkit watch --events {events_path} "
          f"--status {status_path} --until warn", flush=True)

    job = labkit.script_job(
        workdir=str(REPO), entrypoint=entrypoint_for(args.smoke),
        # HF_HUB_DISABLE_XET: hf-xet freezes the log mid weight download; plain HTTP streams
        # progress. The box also heartbeats every 2 min through the quiet 144GB pull + serve.
        env={"INTRO_REPORT_DIR": "out", "HF_TOKEN": tok, "HF_HUB_DISABLE_XET": "1",
             "VLLM_WORKER_MULTIPROC_METHOD": "spawn", **THREAD_CAPS},
        deps=deps_for(),
        ready="LR72_READY", done="LR72_DONE", fatal="LR72_FATAL",
        local_out=str(local_out), pull_subdir="out",
        # the vLLM weight load + serve is the long quiet stretch -> a generous stall ceiling.
        setup_to=3600, stall_to=3600, run_to=int(max_hours * 3600))

    kwargs = dict(
        provider=labkit.VastProvider(**provider_kwargs(
            run_id, disk_gb, max_hours,
            throttle_path=labkit.default_vast_throttle_path())),
        gpu=args.gpu, num_gpus=args.num_gpus, min_vram_mb=args.min_vram, pull_gb=2,
        est_run_s=int(max_hours * 1800),
        max_dph=args.max_dph, max_spend=max_spend, max_hours=max_hours,
        ledger_path=LEDGER_PATH,
        max_acquire_tries=8, max_setup_retries=3, require_verified=not args.unverified,
        mk={"min_reliability": 0.97, "min_inet_down": args.min_bw},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id=run_id)

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=140000,                 # peak = the 72B bf16 weights across 2 cards
        output_incremental=True,            # atomic per-cell shards, resume-safe on-box
        shakedown_done=False,               # box_lr_72b's first box (observe mode surfaces it)
        eng_review="SHIP", sci_review="SHIP",
    )
    decision = evaluate(facts_from_spec(spec), EXPERIMENT_SPEND_POLICY)
    print(f"gate: authorized={decision.authorized} reasons={list(decision.reasons)}", flush=True)

    def _mock(**k):
        print(f"[DRY] would run: gpu={k['gpu']}x{k.get('num_gpus')} "
              f"min_vram={k['min_vram_mb']} disk={disk_gb}GB min_bw={args.min_bw}Mbps "
              f"max_dph=${k['max_dph']} max_spend=${k['max_spend']} max_hours={k['max_hours']} "
              f"run_id={k['run_id']} entry={entrypoint_for(args.smoke)!r}", flush=True)
        return "DRY_OK"

    res = None
    for attempt in (1, 2):
        res = authorized_run(spec, mode="observe", runner=_mock if args.dry else None,
                             recorder=jsonl_recorder(default_gate_log()))
        if isinstance(res, GateBlocked):
            print("BLOCKED ($0):", *res.reasons, sep="\n  ", file=sys.stderr)
            sys.exit(1)
        if args.dry:
            print("dry ok", flush=True)
            return
        if getattr(res, "ok", False):
            break
        if attempt == 1 and is_rsync_flake(getattr(res, "reasons", None),
                                           getattr(res, "error", None)):
            print("rsync-255 infra flake -- retrying once (on-box stages are shard-resume-safe)",
                  flush=True)
            continue
        break

    print(f"outcome={getattr(res, 'outcome', '?')} ok={getattr(res, 'ok', '?')} "
          f"spend=${getattr(res, 'spend_usd', '?')} partial={getattr(res, 'partial_pull', None)} "
          f"log={getattr(res, 'log_path', '?')}", flush=True)
    if not getattr(res, "ok", False):
        print("reasons:", getattr(res, "reasons", None), "error:", getattr(res, "error", None),
              file=sys.stderr)
        sys.exit(1)

    if args.smoke:
        # The smoke's whole job (prereg): confirm on a live box that prompt_logprobs teacher-forces
        # (the vLLM feature is present and returns per-prompt-token logprobs), the local tokenizer
        # matches vLLM's tokenization (parity: we send token ids; span_logprobs raises on a
        # provided-token miss), and the util logging fires (< 60% would have halted). These are
        # LOG assertions, verified in the pulled box log -- not an offline score.
        print("\nsmoke checklist (verify in the pulled box log runs/lr_72b_smoke_box/):", flush=True)
        print("  - LR72_UTIL lines for the first gen + first score batch (tok/s + GPU util; "
              "< 60% util would have HALTED the box)", flush=True)
        print("  - the score shards exist -> prompt_logprobs teacher-forced (span_logprobs did "
              "NOT raise a provided-token miss: tokenizer parity holds)", flush=True)
        print("  - an observe_ shard exists -> the OBSERVER path ran (Amendment 1: the 72B scored "
              "a few 1.5B secret_word streams, verifying the observer wiring, not just the diagonal)",
              flush=True)
        print("  - no LR72_FATAL (a parity or prompt_logprobs failure FATALs by design)",
              flush=True)
        return
    print("pulled -> runs/lr_72b_box/  (score offline: extend lr_grid_offline for the 72B cells; "
          "the 7B off-diagonal cell is scored via the HF src/lr_grid.py path on the pulled 72B "
          "secret streams)", flush=True)


if __name__ == "__main__":
    main()
