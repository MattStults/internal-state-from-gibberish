#!/usr/bin/env python3
"""Driver: collect introspection-leakage data for ONE or SEVERAL models on a rented Vast GPU via labkit
(>=v0.2.1). Wrapper-free single model; a batched loop for several (amortizes one acquisition + one pip).

Single:  .venv-driver/bin/python harness/run_labkit.py qwen2.5-7b [--smoke]
Batch :  .venv-driver/bin/python harness/run_labkit.py qwen3-1.7b qwen3-4b qwen3-8b [--smoke]

Single: env= sets INTRO_MODEL + INTRO_RUN_DIR=out + thread caps; covert_collect writes out/, pulled to
runs/<slug>/. Batch: a `for` loop sets INTRO_MODEL + INTRO_RUN_DIR=out/<slug> per model and prints
ALL_DONE only after the loop (per-model COLLECT_DONE must NOT be the watchdog `done=`); out/<slug>/...
pulls to runs/<slug>/. Run in the LIGHT driver venv (.venv-driver) only.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import labkit

REPO = Path(__file__).resolve().parent.parent


def deps_for(slugs):
    # Qwen3 arch needs transformers >= 4.51 (Qwen2.5 stays on the validated 4.46.3 pin); <5.0 avoids the
    # 5.x apply_chat_template breakage. Re-canary validates the chosen version before any Qwen3 collect.
    tf = "transformers>=4.51,<5.0" if any(s.startswith("qwen3") for s in slugs) else "transformers==4.46.3"
    return [tf, "accelerate", "scikit-learn", "numpy", "wordfreq"]
THREAD_CAPS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
# bf16 footprint: 0.5-3B fit ~12-16GB; 7/8B ~24GB; 14B ~36GB. min_vram = max over the batch.
VRAM_BY_SIZE = {"0.5b": 12000, "1.5b": 16000, "1.7b": 16000, "3b": 16000, "4b": 20000,
                "7b": 24000, "8b": 24000, "14b": 36000}


def build_job(slugs, smoke, effmags=None, stream=False, probe=False, variants="orig", modes="nothink",
              variant="orig", inject="gen", no_calibrate=False, run_to=None):
    # stream (v0.2.11): live-tail the box stdout to our stdout as it runs (watch calibration/collect
    # live; only useful if the DRIVER itself runs unbuffered -- launch with `python -u`). The full box
    # run.log is pulled to res.log_path regardless, so stream is purely for live visibility.
    # probe: run the cheap baseline-clean measurement (src/baseline_clean.py, no injection) instead of the
    # full collect -- decides the capability-floor redesign before spending on a full Qwen3 4B/8B collect.
    script = "src/baseline_clean.py" if probe else "src/covert_collect.py"
    done1 = "BASELINE_DONE" if probe else "COLLECT_DONE"
    flag = (f" --variants {variants} --modes {modes}") if probe else (   # sweep: probe only
        (" --smoke" if smoke else "") + f" --variant {variant} --inject {inject}"
        + (" --no-calibrate" if no_calibrate else ""))
    if len(slugs) == 1:
        s = slugs[0]
        env = {"INTRO_MODEL": s, "INTRO_RUN_DIR": "out", **THREAD_CAPS}
        if effmags and not probe:                     # tuning sweep / explicit eff_mags (collect only)
            env["INTRO_EFFMAGS"] = effmags
        return labkit.script_job(
            workdir=".", entrypoint=f"python3 -u {script}{flag}",
            env=env, stream=stream,
            deps=deps_for(slugs), ready="MODEL_READY", done=done1, fatal="COLLECT_FATAL",
            local_out=f"runs/{s}", pull_subdir="out",
            # 14B = ~30GB HF download + weight load BEFORE the MODEL_READY line -> setup_to must cover it
            # even on a mid-speed host (review blocker: 900s needs ~400Mbps sustained; 1800 is the belt).
            setup_to=1800 if s.split("-")[-1].lower() == "14b" else 900,
            # 7B/8B: 15GB+ download on a slow host + a 36%-util (sawtooth) collect can exceed 50min.
            # run_to (watchdog cap post-READY) must track --max-hours, not fight it (review blocker:
            # the hardcoded 4500s would kill a 3h 14B collect mid-run).
            run_to=run_to or (4500 if s.split("-")[-1].lower() in ("7b", "8b", "14b") else 3000))
    # batch: loop models sequentially, per-model env in the loop; ALL_DONE only after the loop.
    # INTRO_STEP_BASE=i*1000 keeps covert_collect's LABKIT_STEP `step` globally monotonic across the
    # per-model runs (the watchdog ignores a step <= the last seen, so a per-model reset would freeze progress).
    inner = ("i=0; for s in " + " ".join(slugs) + "; do echo \"BATCH_MODEL $s\"; "
             f"INTRO_MODEL=$s INTRO_RUN_DIR=out/$s INTRO_STEP_BASE=$((i*1000)) python3 -u {script}{flag} "
             "|| echo \"BATCH_FAIL $s\"; i=$((i+1)); done; echo ALL_DONE")
    return labkit.script_job(
        workdir=".", entrypoint="bash -c " + shlex.quote(inner),
        env=dict(THREAD_CAPS), stream=stream,        # INTRO_MODEL/RUN_DIR vary per loop iteration
        deps=deps_for(slugs), ready="MODEL_READY", done="ALL_DONE", fatal="COLLECT_FATAL",
        local_out="runs", pull_subdir="out",         # out/<slug>/... -> runs/<slug>/...
        setup_to=900, stall_to=900,                  # tolerate quiet inter-model downloads
        run_to=1800 * len(slugs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+", help="one slug (single) or several (batched in one rental)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--min-vram", type=int, default=None)
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--max-dph", type=float, default=1.50)
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--min-bw", type=int, default=600, help="min host downlink Mbps (model downloads dominate)")
    ap.add_argument("--gpu", default="RTX3090",
                    help="Vast gpu_name -- must be the SERVER-SIDE spelling, quoted if it has a space "
                         "('RTX A6000', 'A40', 'RTX 4090'). RTX3090 is the cheap consumer/residential tier; "
                         "the A6000/A40 48GB tier (needed for 14B) skews datacenter (pay more, fail less). "
                         "A name Vast doesn't recognize fails fast with no_offer.")
    ap.add_argument("--effmags", default=None, help="comma eff_mags for a tuning sweep (overrides scaling)")
    ap.add_argument("--stream", action="store_true",
                    help="v0.2.11 live-tail the box stdout as it runs (launch the driver with `python -u`)")
    ap.add_argument("--probe", action="store_true",
                    help="cheap baseline-clean measurement (src/baseline_clean.py, no injection) instead of full collect")
    ap.add_argument("--variants", default="orig", help="probe: comma list of system-prompt variants to sweep")
    ap.add_argument("--modes", default="nothink", help="probe: comma list of nothink,think (think no-op on Qwen2.5)")
    ap.add_argument("--variant", default="orig", help="collect: system-prompt variant ('orig' is canonical)")
    ap.add_argument("--inject", default="gen", choices=["gen", "all", "prompt"],
                    help="collect: injection method -- 'gen' generation-only (default) or 'all' all-position (legacy)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="collect: skip the auto-tuner -> resid-norm-scaled BASE_EFFMAGS (the original per-model doses)")
    ap.add_argument("--allow-unverified", action="store_true",
                    help="v0.2.18 escape hatch: allow community (non-datacenter) hosts. Default is "
                         "verified-only (no_offer reasons={'unverified': N} if a GPU has thin verified supply)")
    ap.add_argument("--setup-retries", type=int, default=3,
                    help="v0.2.16 slow-host failover budget (extra hosts to try when setup is slow-but-progressing)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock cap; default 1.0 single / 2.0 batch (tuned for <=7B -- 14B generation is "
                         "~2x slower per token, so pass ~3.0 for a 14B full collect)")
    args = ap.parse_args()

    batch = len(args.slugs) > 1
    sizes = [s.split("-")[-1].lower() for s in args.slugs]
    min_vram = args.min_vram or max(VRAM_BY_SIZE.get(z, 24000) for z in sizes)
    # container disk must hold docker image (~8GB) + HF weights + deps + the output .pt. 24GB was already
    # borderline at 7B (15GB weights); 14B's ~30GB weights need ~56 (review blocker: boots, then dies
    # disk-full mid-download, paying for the privilege).
    disk1 = 56.0 if "14b" in sizes else 24.0
    disk_gb = disk1 if not batch else min(120.0, disk1 + 18.0 * len(args.slugs))   # room for N downloads
    tag = args.slugs[0] if not batch else "batch-" + "_".join(s.replace("qwen", "q") for s in args.slugs)
    if args.probe:
        tag = "probe-" + tag

    status_path = REPO / "runs" / f"collect-status-{tag}.json"
    events_path = REPO / "runs" / f"collect-events-{tag}.jsonl"   # v0.2.41 severity-tagged on_event JSONL
    status_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.unlink(missing_ok=True)                           # fresh file so `labkit watch` won't trip on a prior run
    print(f"live status -> {status_path}  (poll from another shell)", flush=True)
    print(f"mid-run wakeup -> .venv-driver/bin/python -m labkit watch --events {events_path} "
          f"--status {status_path} --until warn   (v0.2.50: background it to get woken on a warn+ event)", flush=True)
    res = labkit.run_experiment(
        # throttle_path: several agents share this Vast key and Vast rate-limits per (key+IP), so pace API
        # calls across processes via the shared file lock (v0.2.9). offers_cache (v0.2.10) is automatic.
        provider=labkit.VastProvider(owner=f"introspect-collect-{tag}", disk_gb=disk_gb,
                                     throttle_path=labkit.default_vast_throttle_path()),
        gpu=args.gpu,
        min_vram_mb=min_vram, pull_gb=(40 if "14b" in sizes else 12) * len(args.slugs),
        est_run_s=900 * len(args.slugs),
        max_dph=args.max_dph, max_spend=args.max_spend,
        max_hours=args.max_hours if args.max_hours else (1.0 if not batch else 2.0),
        max_acquire_tries=args.tries, max_setup_retries=args.setup_retries,
        require_verified=not args.allow_unverified,   # v0.2.18: datacenter-only by default (kills the residential tier)
        mk={"min_reliability": 0.97, "min_inet_down": args.min_bw},
        status_path=str(status_path),   # v0.2.27 live status (phase/host/spend/util/last_error) + v0.2.34 boot heartbeat
        on_event=str(events_path),      # v0.2.41 severity-tagged events -> JSONL that `labkit watch` (v0.2.50) reads
        job=build_job(args.slugs, args.smoke, args.effmags, args.stream, args.probe, args.variants, args.modes,
                      args.variant, args.inject, args.no_calibrate,
                      # watchdog run-cap follows the wall-clock budget instead of fighting it
                      run_to=int(args.max_hours * 3600) if args.max_hours else None),
        run_id=f"collect-{tag}" + ("-smoke" if args.smoke else ""))

    # res.log_path (v0.2.11): the FULL box run.log, pulled locally on success AND failure -- the on-box
    # CAL trace + any traceback live here even when the bundle is empty.
    print(f"outcome={res.outcome} ok={res.ok} spend=${getattr(res, 'spend_usd', '?')} "
          f"offer={getattr(res, 'offer_id', '?')} dph={getattr(res, 'dph', '?')} "
          f"partial={getattr(res,'partial_pull',None)} log={getattr(res,'log_path','?')}")
    # res.util_verdict / res.perf (v0.2.13): GPU occupancy + setup-vs-run timing + cost reconciliation for the
    # WHOLE compute window. util_verdict='sawtooth' => under-batched (read it with results/perf.json's
    # generate/read split). vram_headroom high => room to raise batch sizes next run.
    perf = getattr(res, "perf", None) or {}
    print(f"profile: util_verdict={getattr(res,'util_verdict',None)} util_mean={perf.get('util_mean')} "
          f"vram_headroom={perf.get('vram_headroom')} setup_s={perf.get('setup_s')} run_s={perf.get('run_s')} "
          f"cost=${perf.get('total_cost_usd')} (overhead ${perf.get('overhead_cost_usd')})")
    if not res.ok:
        print("reasons:", getattr(res, "reasons", None), "| error:", getattr(res, "error", None), file=sys.stderr)
        # v0.2.8 terminal rate-limit outcomes: don't hammer a retry — back off (shared key) or check account.
        if res.outcome == "rate_limited":
            print("HINT: persistent Vast 429 (shared key). Wait several minutes before re-running; the "
                  "throttle_path lock paces concurrent agents but can't undo an already-tripped limit.", file=sys.stderr)
        elif res.outcome == "spend_capped":
            print("HINT: account spend_rate_limit hit (terminal). Verify the Vast account / use cheaper hosts; "
                  "re-running immediately will just re-trip it.", file=sys.stderr)
        subprocess.run(["osascript", "-e",
                        f'display notification "collect {tag}: {res.outcome}" with title "introspection_leakage"'],
                       check=False)
        sys.exit(1)
    print("pulled -> runs/{" + ",".join(args.slugs) + "}/  (analyses: INTRO_MODEL=<slug> python3 analysis/...)")


if __name__ == "__main__":
    main()
