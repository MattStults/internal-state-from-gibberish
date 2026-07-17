#!/usr/bin/env python3
"""Gated exp2+exp3 RE-ANALYSIS driver: recompute the reader analysis on a cheap rented box (never Matt's Mac --
the local nested-CV crashed it). This run adds the char (transcript) reader, seeded PCA, concept-bootstrap CIs,
the full-stream evaluation, and the bits-ladder cross-check. Reads the EXISTING streams (no re-collection).

Stages runs/_ind (exp3, ~250MB) and runs/_ab/*-gen.pt (exp2, ~960MB) as tars (labkit rsync excludes *.pt);
box_analyze.py unpacks both, extracts embeds from HF, runs run_induction + run_budget with per-bundle
checkpointing; out/ (JSONs + pngs) is pulled even on failure. Routed through the experimentfactory gate.

  Launch:  .venv-driver/bin/python harness/run_reanalysis.py [--gpu RTX4090] [--max-spend 15] [--dry]
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

import labkit                                                                       # noqa: E402
from experimentfactory import (authorized_run, Spec, GateBlocked, jsonl_recorder,    # noqa: E402
                               default_gate_log, evaluate, facts_from_spec, EXPERIMENT_SPEND_POLICY)

LABKIT_TAG = "v0.2.50"
# BLAS single-threaded per worker: the analysis now parallelizes the OUTER (reader x budget) fits across cores
# (joblib), so each worker must NOT also spawn 16 BLAS threads or N_cores workers x 16 threads thrash. One
# thread per worker + outer joblib fan-out = full saturation without oversubscription. (INTRO_JOBS=-1 default.)
THREAD_CAPS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
GEN_BUNDLES = [f"runs/_ab/qwen2.5-{m}-gen.pt" for m in ("1.5b", "3b", "7b")]


def hf_token():
    """Local HF token (the dataset is private) -> passed to the box so it can PULL the bundles from HF, which
    avoids uploading 1.2GB from the laptop's slow home link (that blew labkit's 600s rsync ceiling)."""
    for src in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if src:
            return src.strip()
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        return tok.read_text().strip()
    raise SystemExit("no HF token found (env HF_TOKEN or ~/.cache/huggingface/token) -- needed to pull the private dataset")


def clean_stale_tars():
    """Remove any leftover bundle tars so labkit doesn't ship the old 1.2GB up the slow link."""
    for t in ("exp2_bundles.tar", "exp3_bundles.tar"):
        (REPO / "runs" / t).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    # CPU-only job (nested-CV): the GPU idles, so pick the CHEAPEST box, not a fast card. RTX3090 = the cheap
    # consumer tier (matches the collection driver); per-bundle checkpointing + partial-pull make the flakier
    # cheap/residential supply safe. Do NOT default to RTX4090 -- it's ~5x the price for zero benefit here.
    ap.add_argument("--gpu", default="RTX3090", help="cheap consumer tier; job is CPU-bound so GPU model is irrelevant")
    ap.add_argument("--max-spend", type=float, default=15.0)
    ap.add_argument("--max-dph", type=float, default=0.30)   # cap $/hr low: this is CPU work on a cheap box
    ap.add_argument("--unverified", action="store_true",
                    help="allow residential/unverified hosts (cheapest tier; resume + per-bundle checkpointing cover the flakiness)")
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()

    clean_stale_tars()                                        # box pulls bundles from HF; ship only code + resume seed
    tok = hf_token()
    sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True).strip()
    status_path = REPO / "runs" / "reanalysis-status.json"
    events_path = REPO / "runs" / "reanalysis-events.jsonl"
    events_path.unlink(missing_ok=True)

    job = labkit.script_job(
        workdir=str(REPO), entrypoint="python3 -u experiments/exp3_induction_and_scale/box_analyze.py",
        env={"INTRO_REPORT_DIR": "out", "HF_TOKEN": tok, **THREAD_CAPS},   # HF_TOKEN: box pulls the private bundles
        # PIN the numeric stack to the versions the results were validated on: unpinned latest scipy (1.17.1)
        # fails "SVD did not converge" on 3B where the pinned scipy 1.15.3 runs clean (verified locally). This
        # also makes the box numbers reproducible against the analysis venv. transformers = char tokenizer.
        deps=["scikit-learn==1.7.2", "scipy==1.15.3", "joblib==1.5.3", "matplotlib", "safetensors", "huggingface_hub", "transformers"],  # joblib pinned: the parallel fan-out needs return_as="generator_unordered" (>=1.3, and 1.3.x rejects the value)
        ready="ANALYZE_READY", done="ANALYZE_DONE", fatal="ANALYZE_FATAL",
        local_out=str(REPO / "runs" / "reanalysis_box"), pull_subdir="out",
        setup_to=1800, stall_to=3600, run_to=14000)           # per-(reader,budget) prints -> lines every few min even
                                                              # on a weak CPU; 60min stall = generous belt on top

    kwargs = dict(
        provider=labkit.VastProvider(owner="reanalysis", disk_gb=32.0,
                                     throttle_path=labkit.default_vast_throttle_path()),
        gpu=args.gpu, min_vram_mb=4000, pull_gb=2, est_run_s=9000,   # we never touch the GPU; any card is fine
        max_dph=args.max_dph, max_spend=args.max_spend, max_hours=4.0,
        max_acquire_tries=8, max_setup_retries=3, require_verified=not args.unverified,   # --unverified -> cheapest residential tier
        mk={"min_reliability": 0.97, "min_inet_down": 200},   # the 1.2GB ship is one-time; don't pay for 600Mbps
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id="reanalysis")

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=2000,                                     # CPU job; GPU idles
        output_incremental=True,                              # per-bundle JSON checkpoint (the crash lesson)
        shakedown_done=False,                                 # this combined entrypoint's first box run
        eng_review="SHIP", sci_review="SHIP",                 # analysis code reviewed clean (commit 0019eff); observe mode
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
    print("pulled -> runs/reanalysis_box/  (place JSONs into the report dirs after review)", flush=True)


if __name__ == "__main__":
    main()
