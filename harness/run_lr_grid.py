#!/usr/bin/env python3
"""Gated LR SCALE-GRID driver (unit B9; prereg: exp2 reports/lr_scale_grid_prereg.md, checklist
reports/lr_scale_grid_checklist.md): ONE contiguous 48GB-tier box runs box_lr_grid.py end-to-end
-- S0 HF fetch -> S1 alt-generation at 3B/7B (+ B12 gauge) -> S1b secret_sustain generation at
1.5/3/7B (B15, no gauge) -> S2 the 6-reader LR grid (incl the B15 secret cells + E5) -> S3 the
MC self-report diagonal (box_mc --own-pool, B6 seams). No manual phases (perf checklist 1).

Safety rails (run_mc.py pattern):
  - experimentfactory gate (authorized_run); BUILD/VALIDATE with --dry first ($0 mock runner).
  - per-PROJECT ledger runs/confound-ledger.json; B10 AS RE-AMENDED at D4 (Matt, 2026-07-11):
    up to $25 authorized for this run (full scope, no trims) with the E-phase GO delegated --
    a smoke projection <= $25 advises GO on a clean E verdict, > $25 advises STOP (structural
    problem; registered trim order secret_sustain -> MC diagonal -> never LR cells, only with
    a disclosed recomputation) and exits 3.
  - deadman self-destruct: labkit arms provider-side autodestroy at create time from the
    max-hours deadline (+ teardown buffer), so an orphaned box kills itself even if this driver
    dies mid-run.
  - status heartbeat + monitor-ready events: status/events json(l) under runs/, `labkit watch`
    hint printed at launch.
  - rsync-255 = a retryable ssh-transport infra flake (never an experiment failure): one bounded
    relaunch, everything on-box is shard-resume-safe (a fresh box redoes unfinished stages;
    pulled *.pt never rsync back up).

TWO planned invocations (each --dry first):
  .venv-driver/bin/python harness/run_lr_grid.py --smoke [--dry]
      # D1 slice -> pulls runs/lr_grid_smoke_box/, prints the spend projection (B10 gate);
      # score: .venv/bin/python -c "import lr_grid_offline; lr_grid_offline.main(grid_dir=...)"
  .venv-driver/bin/python harness/run_lr_grid.py [--dry]
      # full grid -> pulls runs/lr_grid_box/ (lr_grid/ + mc_diag/mc/ + _ind sidecars)
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
# Checklist B10 AS RE-AMENDED at D4 (Matt, 2026-07-11): authorization RAISED to $25 for THIS
# run -- full scope, no trims -- after the D4 projection ($16.17 @$1.00/hr) fired the original
# $7.50 STOP. E-phase GO stays delegated (do not block on him). Projection <= $25 -> proceed on
# a clean E verdict; > $25 -> STOP (that magnitude signals a structural problem, not a budget
# question) and apply the registered trim order only with a disclosed recomputation.
RUN_AUTHORIZED_USD = 25.0
TRIM_ORDER = "secret_sustain -> MC diagonal -> never LR cells"
DEADMAN_BUFFER_S = 1800               # provider deadman = max_hours deadline + this teardown buffer
THREAD_CAPS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}

# 48GB tier (prereg perf checklist 5: L40S/A6000-class; every reader <= 16GB bf16 weights ->
# real batch headroom). Pass --gpu L40S to prefer that card on the same tier.
DEFAULT_GPU = "RTX A6000"
DEFAULT_MIN_VRAM = 44000

# bf16 weight footprints (GB) for the DISK floor: the box holds ALL grid readers' weights over
# one contiguous run (subprocesses share the HF cache, nothing is evicted). Amendment 4 sizes
# from the Falcon3 configs (verified 2026-07-11): 1B = 1.67B params -> 3.3GB bf16 (the "1B" name
# undersells it; a 2.1 figure floated at swap time was wrong-low and is NOT used -- floors must
# not under-price disk), 3B = 6.5, 7B = 14.9. The shared "3b"/"7b" keys keep the max of the
# qwen/falcon footprints (qwen2.5-7b = 16.0).
WEIGHTS_GB = {"1b": 3.3, "1.5b": 3.5, "3b": 6.5, "7b": 16.0}
IMAGE_GB = 10.0
DISK_SLACK_GB = 6.0                   # pulled bundles + shards + HF dupes


def _load_box():
    """box_lr_grid is stdlib-only at module level -- the single source for READERS/deps_for."""
    p = REPO / "experiments" / "exp2_output_monitorability" / "box_lr_grid.py"
    if "box_lr_grid" in sys.modules:
        return sys.modules["box_lr_grid"]
    spec = importlib.util.spec_from_file_location("box_lr_grid", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["box_lr_grid"] = mod
    spec.loader.exec_module(mod)
    return mod


BOX = _load_box()
deps_for = BOX.deps_for               # single source of truth (validated 4.46.3 pin, D1-tested)


def entrypoint_for(smoke=False):
    ep = "python3 -u experiments/exp2_output_monitorability/box_lr_grid.py"
    return ep + (" --smoke" if smoke else "")


def run_id_for(smoke=False):
    return "lr-grid-smoke" if smoke else "lr-grid"


def disk_for_grid(readers=None):
    """Container disk floor = docker image + SUM of ALL reader weights + slack (the 14B collect
    precedent: a container that boots then dies disk-full mid weight download still bills)."""
    readers = readers if readers is not None else list(BOX.READERS)
    return round(IMAGE_GB + sum(WEIGHTS_GB[r.split("-")[-1].lower()] for r in readers)
                 + DISK_SLACK_GB, 1)


def provider_kwargs(run_id, disk_gb, max_hours, throttle_path=None):
    """VastProvider constructor kwargs. default_deadman_s MUST ride max_hours (E1 blocker):
    labkit's create() clamps every requested deadman to min(default_deadman_s, requested) and
    the provider DEFAULT is 6h -- without this override the box self-destructs ~6h into a
    16.2h full run. deadman = the max_hours deadline + DEADMAN_BUFFER_S teardown buffer."""
    return dict(owner=run_id, disk_gb=disk_gb, throttle_path=throttle_path,
                default_deadman_s=int(max_hours * 3600) + DEADMAN_BUFFER_S)


def is_rsync_flake(reasons=None, error=None):
    """rsync exit 255 = the ssh transport died (host flake), not an experiment failure --
    checklist B9: retryable. Anything else (incl other rsync codes: real transfer errors)
    is NOT swallowed."""
    blobs = list(reasons or [])
    if error:
        blobs.append(str(error))
    pat = re.compile(r"rsync[^\n]*\b255\b")
    return any(pat.search(str(b)) for b in blobs)


def remaining_budget(ledger_path=LEDGER_PATH, cap=PROJECT_CAP):
    """cap - the ledger's cumulative spend (ledger = {instance_id: usd}); a missing ledger is a
    first run (full cap)."""
    if not os.path.exists(ledger_path):
        return float(cap)
    with open(ledger_path) as f:
        led = json.load(f)
    return float(cap) - float(sum(led.values()))


# ------------------------------------------------------------------ smoke spend projection
# Scale factors from smoke -> full, all named + documented (D4 recomputes from real smoke
# timing before any full launch; these are the B9/B10 printer):
ALT_SCALE_3B = (36 / 4) * (128 / 48)   # target_clean 4->36, stream tokens 48->128 (cfg parity)
ALT_7B_OVER_3B = 2.2                   # 7B/3B generation compute ratio (bf16 param ratio)
SS_1P5B_OVER_3B = 0.5                  # 1.5B/3B generation compute ratio (bf16 param ratio)
SMOKE_QWEN_SHARDS = 8                  # 1.5B reader x (evoked + alt) x 3 ctx + secret_word x 2
SMOKE_LLAMA_SHARDS = 8                 # xfam 1B reader x evoked x 3 ctx x {tmpl,raw} + 2 prose
FULL_QWEN_SHARDS = 30                  # 18 pre-B15 + 12 secret (2 sets x 3 gens x {N, matched})
FULL_LLAMA_SHARDS = 62                 # 38 pre-B15 + 12 secret + 12 secret raw (xfam readers)
E5_SHARDS = 2                          # the 1.5B reader's maintained_secret descriptive cell
INJECTED_SHARDS = 3                    # control (b): the injected N/A/B self-diagonal, on the
#                                        3B and 7B qwen readers ONLY (each reads its own capture)
# Amendment 4: the *_LLAMA_* names above are Llama-era (retained); the cross-family readers are
# Falcon3. Factors = bf16 param ratio vs the 1.5B anchor (Falcon3 real sizes from their configs:
# 1.67B / 3.23B / 7.46B -- the "1B" is bigger than Llama-3.2-1B was, so its factor rose 0.7->1.1).
READER_FACTOR = {"qwen2.5-1.5b": 1.0, "qwen2.5-3b": 2.0, "qwen2.5-7b": 4.7,
                 "falcon3-1b": 1.1, "falcon3-3b": 2.1, "falcon3-7b": 4.8}
MC_CELLS_FULL = 8                      # per diagonal reader (2 sets x 2 framings x 2 reasonings)
MC_COT_MULT = 6.0                      # a CoT cell ~ 6x a direct cell (256-token greedy gen)
SETUP_S = 900.0                        # boot + pip + weights not captured by step deltas
SLACK = 1.25


def parse_steps(log_text):
    """LABKIT_STEP json lines (box emit_step carries t = secs since box start) -> list of
    dicts, in order."""
    out = []
    for line in log_text.splitlines():
        m = re.search(r"LABKIT_STEP (\{.*\})", line)
        if m:
            try:
                out.append(json.loads(m.group(1)))
            except json.JSONDecodeError:
                pass
    return out


def _stage_durations(steps):
    """Consecutive-step deltas keyed by phase (+reader). The last step (done) closes the final
    stage."""
    dur = {}
    for a, b in zip(steps, steps[1:]):
        key = a.get("phase", str(a.get("step")))
        if a.get("reader"):
            key += f":{a['reader']}"
        dur[key] = dur.get(key, 0.0) + float(b.get("t", 0)) - float(a.get("t", 0))
    return dur


def smoke_projection(steps, dph):
    """Full-run spend projected from the D1 smoke's measured stage durations. Every scale
    factor is a named constant above; D4 re-derives before launch, and B10 (check_projection)
    gates the advice either way. B15: secret_sustain generation (S1b) is NOT exercised by the
    smoke -- its term EXTRAPOLATES from the measured alt-gen timing (same pipeline, same cfg;
    sizes scaled by the bf16 param-ratio constants), noted in the returned projection."""
    dur = _stage_durations(steps)
    alt_smoke = dur.get("S1_altgen", 0.0)
    altgen_s = alt_smoke * ALT_SCALE_3B * (1.0 + ALT_7B_OVER_3B)
    # S1b at 1.5B + 3B + 7B, extrapolated from the 3B alt-gen smoke timing (never measured).
    secretgen_s = alt_smoke * ALT_SCALE_3B * (SS_1P5B_OVER_3B + 1.0 + ALT_7B_OVER_3B)
    q_shard = dur.get("S2_lr_grid:qwen2.5-1.5b", 0.0) / SMOKE_QWEN_SHARDS
    l_shard = dur.get("S2_lr_grid:falcon3-1b", 0.0) / SMOKE_LLAMA_SHARDS
    lr_s = q_shard * E5_SHARDS                           # the 1.5B reader's E5 cell
    for r, f in READER_FACTOR.items():
        if r.startswith("qwen"):
            lr_s += q_shard * (f / READER_FACTOR["qwen2.5-1.5b"]) * FULL_QWEN_SHARDS
            # control (b): the 3B/7B qwen readers each add the injected self-diagonal.
            if r in ("qwen2.5-3b", "qwen2.5-7b"):
                lr_s += q_shard * (f / READER_FACTOR["qwen2.5-1.5b"]) * INJECTED_SHARDS
        else:
            lr_s += l_shard * (f / READER_FACTOR["falcon3-1b"]) * FULL_LLAMA_SHARDS
    mc_smoke = dur.get("S3_mc_diag", 0.0)                # one 3B direct cell
    per_reader = mc_smoke * (MC_CELLS_FULL / 2.0) * (1.0 + MC_COT_MULT) / 2.0
    mc_s = per_reader * (1.0 + ALT_7B_OVER_3B)           # 3B + 7B diagonal readers
    total_s = (altgen_s + secretgen_s + lr_s + mc_s) * SLACK + SETUP_S
    return dict(altgen_s=altgen_s, secretgen_s=secretgen_s, lr_s=lr_s, mc_s=mc_s,
                total_s=total_s,
                projected_usd=total_s / 3600.0 * float(dph),
                note="scale factors are named constants in run_lr_grid.py; D4 re-derives "
                     "from real smoke timing before any full launch. B15: secret_sustain "
                     "generation is NOT in the smoke slice -- its term is extrapolated from "
                     "the measured alt-gen timing (registered in checklist B15).")


def check_projection(projected_usd, authorized_usd=RUN_AUTHORIZED_USD):
    """B10 AS RE-AMENDED at D4 (Matt, 2026-07-11): projection <= the authorized $25 (full
    scope, no trims) -> proceed on a clean E verdict WITHOUT waiting for Matt; > $25 -> STOP
    -- that magnitude signals a structural problem, not a budget question -- and apply the
    registered trim order (secret_sustain -> MC diagonal -> never LR cells) only with a
    disclosed recomputation. Matt is still the escalation path outside the authorization."""
    if projected_usd > authorized_usd:
        return dict(go=False, message=(
            f"STOP: projected full-run spend ${projected_usd:.2f} exceeds the authorized "
            f"${authorized_usd:.2f} (Matt, 2026-07-11) -- a structural problem, not a budget "
            f"question. Apply the registered trim order ({TRIM_ORDER}) only with a DISCLOSED "
            "recomputation, or ask Matt."))
    return dict(go=True, message=(
        f"GO: projected ${projected_usd:.2f} within the authorized ${authorized_usd:.2f} -- "
        "proceed on a clean E-phase verdict without waiting for Matt (B10 as amended)."))


def hf_token():
    """Local HF token -> the box PULLS the private bundles from HF (rsync excludes *.pt)."""
    for src in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if src:
            return src.strip()
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        return tok.read_text().strip()
    raise SystemExit("no HF token found (env HF_TOKEN or ~/.cache/huggingface/token) -- needed "
                     "to pull the private dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=DEFAULT_GPU,
                    help="48GB tier (prereg perf checklist 5); L40S also qualifies")
    ap.add_argument("--min-vram", type=int, default=DEFAULT_MIN_VRAM)
    ap.add_argument("--max-spend", type=float, default=None,
                    help="cumulative project cap against the shared ledger (default: current "
                         "ledger spend + the $25 Matt authorized for this run, B10 re-amended "
                         "at D4)")
    ap.add_argument("--max-dph", type=float, default=0.85, help="48GB-tier ceiling $/hr")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock cap -> labkit deadline -> provider deadman "
                         "(default 1.5 smoke / 10.0 full -- raised from 8.0 for the B15 "
                         "secret cells + S1b generation; spend stays gated by max_spend/B10)")
    ap.add_argument("--min-bw", type=int, default=400, help="min host downlink Mbps (weights)")
    ap.add_argument("--smoke", action="store_true",
                    help="the registered D1 slice + spend projection (B10 gate)")
    ap.add_argument("--unverified", action="store_true")
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()
    max_hours = args.max_hours or (1.5 if args.smoke else 10.0)
    # max_spend with ledger_path caps the PROJECT'S CUMULATIVE spend (run_confound precedent):
    # default = what the ledger already holds + Matt's $25 per-run authorization (D4).
    spent = PROJECT_CAP - remaining_budget()
    max_spend = args.max_spend if args.max_spend is not None else round(
        spent + RUN_AUTHORIZED_USD, 2)
    run_id = run_id_for(args.smoke)
    local_out = REPO / "runs" / ("lr_grid_smoke_box" if args.smoke else "lr_grid_box")
    disk_gb = disk_for_grid()

    # lazy driver deps: unit tests import this module without the driver venv
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
    # monitor armed at launch (checklist B9): the watch command tails the same events/status
    print(f"mid-run wakeup -> .venv-driver/bin/python -m labkit watch --events {events_path} "
          f"--status {status_path} --until warn", flush=True)

    job = labkit.script_job(
        workdir=str(REPO), entrypoint=entrypoint_for(args.smoke),
        # HF_HUB_DISABLE_XET (checklist B8): hf-xet freezes the log mid weight download; the
        # plain HTTP backend streams progress. box_lr_grid also heartbeats every 2 min.
        env={"INTRO_REPORT_DIR": "out", "HF_TOKEN": tok, "HF_HUB_DISABLE_XET": "1",
             **THREAD_CAPS},
        deps=deps_for(list(BOX.READERS)),
        ready="LRG_READY", done="LRG_DONE", fatal="LRG_FATAL",
        local_out=str(local_out), pull_subdir="out",
        # stall 45 min = the proven quiet-weight-download ceiling; run_to = the true-hang cap.
        # labkit arms the provider deadman at create from max_hours, but create() CLAMPS it to
        # min(default_deadman_s, requested) -- provider_kwargs raises the 6h default (E1 blocker).
        setup_to=1800, stall_to=2700, run_to=int(max_hours * 3600))

    kwargs = dict(
        # E1 blocker: provider_kwargs overrides labkit's 6h default_deadman_s (create() clamps
        # every deadman to min(default, requested)) so the deadman covers max_hours + buffer.
        provider=labkit.VastProvider(**provider_kwargs(
            run_id, disk_gb, max_hours,
            throttle_path=labkit.default_vast_throttle_path())),
        gpu=args.gpu, min_vram_mb=args.min_vram, pull_gb=2,
        est_run_s=int(max_hours * 1800),
        max_dph=args.max_dph, max_spend=max_spend, max_hours=max_hours,
        ledger_path=LEDGER_PATH,       # per-PROJECT ledger shared with confound/gauge/lr/mc/elicit
        max_acquire_tries=8, max_setup_retries=3, require_verified=not args.unverified,
        mk={"min_reliability": 0.97, "min_inet_down": args.min_bw},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id=run_id)

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=20000,                  # peak = the 7B/8B bf16 readers + KV + fp32 chunks
        output_incremental=True,            # atomic per-cell shards, resume-safe on-box
        shakedown_done=False,               # box_lr_grid's first box (observe mode surfaces it)
        eng_review="SHIP", sci_review="SHIP",  # orchestration over unit-tested measurement code
    )
    decision = evaluate(facts_from_spec(spec), EXPERIMENT_SPEND_POLICY)
    print(f"gate: authorized={decision.authorized} reasons={list(decision.reasons)}", flush=True)

    def _mock(**k):
        print(f"[DRY] would run: gpu={k['gpu']} min_vram={k['min_vram_mb']} disk={disk_gb}GB "
              f"max_dph=${k['max_dph']} max_spend=${k['max_spend']} max_hours={k['max_hours']} "
              f"run_id={k['run_id']} entry={entrypoint_for(args.smoke)!r}", flush=True)
        return "DRY_OK"

    res = None
    for attempt in (1, 2):                  # bounded retry: rsync-255 infra flake ONLY
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
            print("rsync-255 infra flake -- retrying once (on-box stages are shard-resume-safe; "
                  "a fresh box redoes unfinished work)", flush=True)
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
        # B10: projection from the smoke's own step timings vs the remaining ledger.
        log_path = getattr(res, "log_path", None)
        steps = []
        if log_path and os.path.exists(str(log_path)):
            with open(str(log_path)) as f:
                steps = parse_steps(f.read())
        proj = smoke_projection(steps, dph=args.max_dph)
        verdict = check_projection(proj["projected_usd"])
        print(f"\nsmoke spend projection: altgen={proj['altgen_s']:.0f}s "
              f"secretgen={proj['secretgen_s']:.0f}s (extrapolated from alt-gen timing; "
              f"S1b is not in smoke) lr={proj['lr_s']:.0f}s mc={proj['mc_s']:.0f}s "
              f"total={proj['total_s']:.0f}s "
              f"-> ${proj['projected_usd']:.2f} at ${args.max_dph}/hr "
              f"(ledger so far ${PROJECT_CAP - remaining_budget():.2f}; authorized "
              f"${RUN_AUTHORIZED_USD:.2f} for this run)", flush=True)
        print(verdict["message"], flush=True)
        # SCI-SF3: the smoke scoring pass writes to the SMOKE results json (never the full-run
        # OUT_JSON), so the D2 anchor persists for the full run's anchor-consistency check.
        smoke_json = ("experiments/exp2_output_monitorability/reports/"
                      "lr_grid_smoke_results.json")
        print("score the smoke offline: .venv/bin/python -c \"import sys; "
              "sys.path.insert(0, 'experiments/exp2_output_monitorability/analysis'); "
              "import lr_grid_offline as L; "
              f"L.main(grid_dir='{local_out}/lr_grid', mc_diag_dir='{local_out}/mc_diag/mc', "
              f"out_json='{smoke_json}')\"", flush=True)
        if not verdict["go"]:
            sys.exit(3)
        return
    print("pulled -> runs/lr_grid_box/  (score offline: .venv/bin/python "
          "experiments/exp2_output_monitorability/analysis/lr_grid_offline.py; judge the alt "
          "gauge: .venv/bin/python experiments/exp3_induction_and_scale/gauge_judge_alt.py)",
          flush=True)


if __name__ == "__main__":
    main()
