#!/usr/bin/env python3
"""Gated exp3 induction-collection driver: runs collect_induction on a Vast GPU THROUGH the experimentfactory
spend gate (observe mode) -- NOT raw labkit.run_experiment (per experimentfactory/CLAUDE.md). One model per
invocation (contiguous create->run->destroy), so the 1.5B feasibility gate is a HUMAN checkpoint before 3B/7B.

experimentfactory has no packaging; it rides PYTHONPATH to the sibling checkout (its own README pattern).
labkit stays pip-pinned in .venv-driver. Run from the LIGHT driver venv only (no torch here).

  Validate the gate + Spec at $0 (no GPU):  .venv-driver/bin/python harness/run_exp3.py qwen2.5-1.5b --smoke --dry
  Shakedown (~$1, first GPU run of the code): .venv-driver/bin/python harness/run_exp3.py qwen2.5-1.5b --smoke
  Real 1.5B (feasibility gate on):            .venv-driver/bin/python harness/run_exp3.py qwen2.5-1.5b --min-per-class 24
  Then, only if 1.5B clears:                  .venv-driver/bin/python harness/run_exp3.py qwen2.5-3b --min-per-class 24
                                              .venv-driver/bin/python harness/run_exp3.py qwen2.5-7b --gpu RTXA6000 --min-per-class 24
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# experimentfactory has no packaging -> ride PYTHONPATH to the sibling checkout (its own README pattern).
# Override with EXPERIMENTFACTORY_HOME; default to the sibling of the ai_safety_projects dir.
EF = Path(os.environ.get("EXPERIMENTFACTORY_HOME",
                         REPO.parent / "experiment_harness"))
sys.path.insert(0, str(EF))

import labkit                                                                     # noqa: E402  pip-pinned in .venv-driver
from experimentfactory import (authorized_run, Spec, GateBlocked, jsonl_recorder,  # noqa: E402
                               default_gate_log, evaluate, facts_from_spec, EXPERIMENT_SPEND_POLICY)

LABKIT_TAG = "v0.2.50"                                    # must match harness/LABKIT.md pin
THREAD_CAPS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
MIN_VRAM = {"0.5b": 12000, "1.5b": 16000, "1.7b": 16000, "3b": 16000, "4b": 20000, "7b": 24000, "8b": 24000}
VRAM_EST = {"0.5b": 6000, "1.5b": 8000, "1.7b": 8000, "3b": 12000, "4b": 15000, "7b": 18000, "8b": 20000}  # peak < min_vram


def deps_for(slug):
    tf = "transformers>=4.51,<5.0" if slug.startswith("qwen3") else "transformers==4.46.3"   # Qwen2.5 stays on the pin
    return [tf, "accelerate", "scikit-learn", "numpy", "wordfreq"]


def build_labkit_kwargs(slug, arms, smoke, min_per_class, gpu, max_spend, max_dph, tries, min_bw):
    size = slug.split("-")[-1].lower()
    flag = (" --smoke" if smoke else "") + f" --min-per-class {min_per_class} --arms " + " ".join(arms)
    entry = f"python3 -u experiments/exp3_induction_and_scale/collect_induction.py --models {slug}{flag}"
    job = labkit.script_job(
        workdir=".", entrypoint=entry,
        env={"INTRO_MODEL": slug, "INTRO_RUN_DIR": "out", **THREAD_CAPS},   # INTRO_RUN_DIR=out -> C.DATA=out/data, pulled
        deps=deps_for(slug), ready="MODEL_READY", done="COLLECT_DONE", fatal="COLLECT_FATAL",
        local_out=f"runs/_ind/{slug}", pull_subdir="out", setup_to=900,
        run_to=4500 if size in ("7b", "8b") else 3000)
    status_path = REPO / "runs" / f"exp3-status-{slug}.json"
    events_path = REPO / "runs" / f"exp3-events-{slug}.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.unlink(missing_ok=True)
    max_hours = 2.0 if size in ("7b", "8b") else 1.0     # 7B/8B: 15GB download + collect can exceed 60min (run_to=4500s)
    return dict(
        provider=labkit.VastProvider(owner=f"exp3-induct-{slug}", disk_gb=24.0,
                                     throttle_path=labkit.default_vast_throttle_path()),
        gpu=gpu, min_vram_mb=MIN_VRAM.get(size, 24000), pull_gb=12, est_run_s=900,
        max_dph=max_dph, max_spend=max_spend, max_hours=max_hours,
        max_acquire_tries=tries, max_setup_retries=3, require_verified=True,   # fail over off a bad SSH host (was terminal)
        mk={"min_reliability": 0.97, "min_inet_down": min_bw},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id=f"exp3-{slug}" + ("-smoke" if smoke else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", help="one model slug, e.g. qwen2.5-1.5b (one model per invocation)")
    ap.add_argument("--arms", nargs="+", default=["evoked", "evoked_alt", "named", "secret_word"])
    ap.add_argument("--smoke", action="store_true", help="tiny shakedown run (feasibility gate auto-off)")
    ap.add_argument("--min-per-class", type=int, default=24, help="feasibility gate floor (real runs)")
    ap.add_argument("--gpu", default="RTX3090", help="Vast gpu_name (7B: use RTXA6000/A40 for VRAM headroom)")
    ap.add_argument("--max-spend", type=float, default=5.0)
    ap.add_argument("--max-dph", type=float, default=1.50)
    ap.add_argument("--tries", type=int, default=8)
    ap.add_argument("--min-bw", type=int, default=600)
    ap.add_argument("--dry", action="store_true",
                    help="validate the gate + Spec at $0 (mock runner; labkit never called, no GPU)")
    args = ap.parse_args()

    size = args.slug.split("-")[-1].lower()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    mpc = 0 if args.smoke else args.min_per_class          # smoke's tiny target_clean can't reach 24 -> gate off
    kwargs = build_labkit_kwargs(args.slug, args.arms, args.smoke, mpc, args.gpu, args.max_spend,
                                 args.max_dph, args.tries, args.min_bw)
    spec = Spec(
        labkit_kwargs=kwargs,
        seed=0,
        data_revision=sha,                                 # the frozen-primers commit driving this run
        labkit_tag=LABKIT_TAG,
        vram_est_mb=VRAM_EST.get(size, 20000),
        output_incremental=True,                           # per-arm bundles written as it goes -> partial pull recovers
        shakedown_done=not args.smoke,                     # the --smoke run IS the shakedown; real runs assert it ran
        # These reflect REAL fresh-context subagent reviews (collector atomic review; full-project review, B1/B2
        # closed). Honest for observe. Before ever flipping to mode="enforce", re-derive from a fresh review of
        # the THEN-current code -- do not trust these literals across material changes (experimentfactory rule).
        eng_review="SHIP",
        sci_review="SHIP",
    )

    decision = evaluate(facts_from_spec(spec), EXPERIMENT_SPEND_POLICY)   # $0: show what the gate concludes + why
    print(f"gate: authorized={decision.authorized} reasons={list(decision.reasons)}", flush=True)
    print(f"spec: model={args.slug} arms={args.arms} smoke={args.smoke} min_per_class={mpc} "
          f"data_rev={sha} shakedown_done={spec.shakedown_done}", flush=True)

    def _mock(**k):                                         # $0 validation runner (only with --dry)
        print(f"[DRY] would call labkit.run_experiment: gpu={k['gpu']} min_vram={k['min_vram_mb']} "
              f"max_spend=${k['max_spend']} max_hours={k['max_hours']} run_id={k['run_id']}", flush=True)
        return "DRY_OK"

    res = authorized_run(spec, mode="observe",             # OBSERVE: records the decision, always passes through
                         runner=_mock if args.dry else None,
                         recorder=jsonl_recorder(default_gate_log()))
    print(f"gate log -> {default_gate_log()}", flush=True)

    if isinstance(res, GateBlocked):                       # cannot happen in observe; handled for safety
        print("BLOCKED ($0):", *res.reasons, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    if args.dry:
        print("dry validation complete -- no spend, no GPU.", flush=True)
        return
    print(f"outcome={getattr(res, 'outcome', '?')} ok={getattr(res, 'ok', '?')} "
          f"spend=${getattr(res, 'spend_usd', '?')} partial={getattr(res, 'partial_pull', None)} "
          f"log={getattr(res, 'log_path', '?')}", flush=True)
    perf = getattr(res, "perf", None) or {}              # util_verdict='sawtooth' => under-batched (raise gen_batch)
    print(f"profile: util_verdict={getattr(res, 'util_verdict', None)} util_mean={perf.get('util_mean')} "
          f"vram_headroom={perf.get('vram_headroom')} run_s={perf.get('run_s')} "
          f"cost=${perf.get('total_cost_usd')}", flush=True)
    if not getattr(res, "ok", False):
        print("reasons:", getattr(res, "reasons", None), "error:", getattr(res, "error", None), file=sys.stderr)
        subprocess.run(["osascript", "-e",
                        f'display notification "exp3 {args.slug}: {getattr(res,"outcome","?")}" '
                        f'with title "introspection_leakage"'], check=False)
        sys.exit(1)
    print(f"pulled -> runs/_ind/{args.slug}/out/  (bundles under .../data/)", flush=True)


if __name__ == "__main__":
    main()
