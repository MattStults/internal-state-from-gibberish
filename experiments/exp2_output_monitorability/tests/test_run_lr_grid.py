"""RED-first unit tests for scale-grid unit B9: the gated driver harness/run_lr_grid.py + the
box's contiguous S3 (MC diagonal) stage and --smoke (D1 slice) mode. No GPU, no launch, no
labkit import required (driver deps load lazily inside main()).
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_run_lr_grid.py

R1  the driver module imports WITHOUT the driver venv (labkit/experimentfactory lazy in main).
R2  entrypoint/run-id/deps: box_lr_grid.py entrypoint (+ --smoke passthrough), deps from
    box_lr_grid.deps_for (single source), ledger = runs/confound-ledger.json, cap $5.
R3  disk floor = image + SUM of ALL reader weights + slack; every grid reader size priced.
R4  48GB tier defaults (A6000-class, min_vram >= 40GB).
R5  rsync-255 handling: is_rsync_flake spots the retryable infra flake, nothing else.
R6  parse_steps reads LABKIT_STEP lines (with t) from a box log.
R7  smoke_projection: positive spend from synthetic stage durations, parts itemized
    (altgen/lr/mc), linear in $/hr.
R8  remaining_budget = cap - ledger sum; check_projection STOPs (ask Matt) when the projection
    exceeds it (B10 rule), GOes otherwise.
R9  box S3: mc_diag_cmd runs box_mc --own-pool with an EXPLICIT model list (3B,7B -- never the
    default list) and a FRESH INTRO_REPORT_DIR (OUT/mc_diag); the full-run main wires it.
R10 box --smoke: the registered D1 slice -- tiny alt-gen at 3B (collect_induction --smoke,
    feasibility floor off), one LR cell per reader family (Qwen 1.5B on the 1.5B pools incl the
    D2 anchor cells; Falcon3 1B on the 1.5B evoked pool -- Amendment 4), ONE MC diagonal shard
    (mc_reader elicited x direct on the 3B evoked bundle, own-pool flags, no capture).
R11 mc_reader exposes --framings/--reasonings loop filters (orchestration only; the certified
    scoring bodies stay sha-pinned by test_mc_own_pool S1 -- rerun it as the regression).
R12 driver main() source: gate (authorized_run), monitor-ready (status_path + on_event + watch
    hint), deadman via max_hours (labkit arms autodestroy from the deadline), rsync-255 retry
    loop, smoke projection + B10 STOP wiring.
"""
import importlib.util
import inspect
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BOX = load_module("box_lr_grid",
                  os.path.join(REPO, "experiments", "exp2_output_monitorability",
                               "box_lr_grid.py"))
sys.modules["box_lr_grid"] = BOX      # the driver's loader reuses this instance (identity check)
DRV_PATH = os.path.join(REPO, "harness", "run_lr_grid.py")

# ================================================================ R1: lazy driver deps
try:
    DRV = load_module("run_lr_grid", DRV_PATH)
    check("R1 driver imports without labkit/experimentfactory (lazy in main)", True)
except Exception as e:
    DRV = None
    check("R1 driver imports without labkit/experimentfactory (lazy in main)", False,
          f"{type(e).__name__}: {e}")

# ================================================================ R2: entrypoint/deps/ledger
if DRV is not None:
    try:
        check("R2 entrypoint targets box_lr_grid.py",
              "box_lr_grid.py" in DRV.entrypoint_for(smoke=False))
        check("R2 --smoke passes through to the box",
              DRV.entrypoint_for(smoke=True).rstrip().endswith("--smoke"))
        check("R2 deps come from box_lr_grid.deps_for (single source, validated 4.46.3 pin)",
              DRV.deps_for is BOX.deps_for
              and "transformers==4.46.3" in DRV.deps_for(list(BOX.READERS)))
        check("R2 ledger = the shared project ledger; B10 as re-amended at D4: $25 authorized "
              "for this run (Matt 2026-07-11, full scope, no trims)",
              DRV.LEDGER_PATH.endswith(os.path.join("runs", "confound-ledger.json"))
              and DRV.PROJECT_CAP == 5.0 and DRV.RUN_AUTHORIZED_USD == 25.0)
        check("R2 run ids: lr-grid / lr-grid-smoke",
              DRV.run_id_for(False) == "lr-grid" and DRV.run_id_for(True) == "lr-grid-smoke")
    except Exception as e:
        check("R2 entrypoint/deps", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R3: disk floor
if DRV is not None:
    try:
        sizes = {r.split("-")[-1].lower() for r in BOX.READERS}
        check("R3 every grid reader size is priced in WEIGHTS_GB",
              sizes <= set(DRV.WEIGHTS_GB), f"sizes={sizes} priced={set(DRV.WEIGHTS_GB)}")
        want_min = 10.0 + sum(DRV.WEIGHTS_GB[r.split('-')[-1].lower()] for r in BOX.READERS)
        got = DRV.disk_for_grid()
        check("R3 disk floor = image + SUM of ALL reader weights + slack",
              got >= want_min + 2.0, f"got {got}, weights+image = {want_min}")
    except Exception as e:
        check("R3 disk floor", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R4: 48GB tier defaults
if DRV is not None:
    try:
        check("R4 default GPU is the 48GB tier (A6000-class per prereg perf checklist 5)",
              "A6000" in DRV.DEFAULT_GPU or "L40S" in DRV.DEFAULT_GPU, f"{DRV.DEFAULT_GPU}")
        check("R4 min VRAM covers the tier (>= 40GB)", DRV.DEFAULT_MIN_VRAM >= 40000)
    except Exception as e:
        check("R4 tier defaults", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R5: rsync-255 flake
if DRV is not None:
    try:
        check("R5 rsync-255 is a retryable infra flake",
              DRV.is_rsync_flake(error="pull failed: rsync exited with code 255") is True
              and DRV.is_rsync_flake(reasons=["ssh transport: rsync error 255"]) is True)
        check("R5 real failures are NOT retried as flakes",
              DRV.is_rsync_flake(error="CUDA out of memory") is False
              and DRV.is_rsync_flake(error="rsync exited with code 23") is False
              and DRV.is_rsync_flake() is False)
    except Exception as e:
        check("R5 rsync flake", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R6/R7: projection
if DRV is not None:
    try:
        log = "\n".join([
            'LABKIT_STEP {"step": 0, "phase": "S0_fetch", "t": 10}',
            "S0 fetched things",
            'LABKIT_STEP {"step": 500, "phase": "S1_altgen", "t": 100}',
            'LABKIT_STEP {"step": 1000, "phase": "S2_lr_grid", "reader": "qwen2.5-1.5b",'
            ' "t": 400}',
            'LABKIT_STEP {"step": 2000, "phase": "S2_lr_grid", "reader": "falcon3-1b",'
            ' "t": 1000}',
            'LABKIT_STEP {"step": 8000, "phase": "S3_mc_diag", "t": 1480}',
            'LABKIT_STEP {"step": 9000, "phase": "lr_grid_done", "t": 1780}',
        ])
        steps = DRV.parse_steps(log)
        check("R6 parse_steps reads LABKIT_STEP json lines with t",
              len(steps) == 6 and steps[0]["phase"] == "S0_fetch" and steps[-1]["t"] == 1780)
        proj = DRV.smoke_projection(steps, dph=0.80)
        check("R7 projection itemizes altgen/lr/mc parts and totals a positive spend",
              proj["projected_usd"] > 0
              and all(k in proj for k in ("altgen_s", "lr_s", "mc_s", "total_s")),
              f"{proj}")
        proj2 = DRV.smoke_projection(steps, dph=1.60)
        check("R7 projection is linear in $/hr",
              abs(proj2["projected_usd"] - 2 * proj["projected_usd"]) < 1e-9)
    except Exception as e:
        check("R6/R7 projection", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R8: budget + B10 STOP
if DRV is not None:
    try:
        with tempfile.TemporaryDirectory() as td:
            lp = os.path.join(td, "ledger.json")
            with open(lp, "w") as f:
                json.dump({"a": 1.0, "b": 0.88}, f)
            rem = DRV.remaining_budget(lp, cap=5.0)
            check("R8 remaining budget = cap - ledger sum", abs(rem - 3.12) < 1e-9, f"{rem}")
            check("R8 missing ledger -> full cap (first run)",
                  DRV.remaining_budget(os.path.join(td, "nope.json"), cap=5.0) == 5.0)
        # B10 AS RE-AMENDED at D4 (Matt 2026-07-11): threshold = the $25 per-run authorization
        # (full scope, no trims), GO delegated to a clean E verdict; over-authorization -> STOP
        # + the registered trim order. The D4 projection ($16.17) must sit on the GO side.
        stop = DRV.check_projection(25.80)
        go = DRV.check_projection(16.17)
        check("R8 B10 (re-amended): projection > $25 authorized -> STOP with the registered "
              "trim order (secret_sustain -> MC diagonal -> never LR cells)",
              stop["go"] is False and "trim order" in stop["message"]
              and "secret_sustain" in stop["message"] and "LR cells" in stop["message"],
              f"{stop}")
        check("R8 B10 (re-amended): the D4 projection $16.17 <= $25 -> GO on a clean E "
              "verdict without waiting for Matt",
              go["go"] is True and "E-phase" in go["message"], f"{go}")
    except Exception as e:
        check("R8 budget/B10", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R9: box S3 stage
try:
    cmd, env = BOX.mc_diag_cmd()
    check("R9 S3 = box_mc --own-pool with an EXPLICIT 3B/7B model list (B6 seams: no default "
          "list)", os.path.basename(cmd[cmd.index("-u") + 1]) == "box_mc.py"
          and "--own-pool" in cmd and "--models" in cmd
          and cmd[cmd.index("--models") + 1] == "qwen2.5-3b,qwen2.5-7b", f"cmd={cmd}")
    check("R9 S3 uses a FRESH INTRO_REPORT_DIR (OUT/mc_diag -- diagonal shard names coincide "
          "with default-pool ones)",
          env.get("INTRO_REPORT_DIR") == os.path.join(BOX.OUT, "mc_diag"), f"env={env}")
    check("R9 full-run main wires S3 after the reader loop",
          "mc_diag_cmd(" in inspect.getsource(BOX.main))
    exp = BOX.mc_diag_shards()
    check("R9 done-check covers the 16 diagonal MC shards (8 per reader)",
          len(exp) == 16 and all("mc_diag" in p for p in exp), f"n={len(exp)}")
except Exception as e:
    check("R9 box S3", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R10: box --smoke (D1 slice)
try:
    check("R10 smoke readers = one per family (Qwen 1.5B + Falcon3 1B; Amendment 4)",
          tuple(BOX.SMOKE_READERS) == ("qwen2.5-1.5b", "falcon3-1b"))
    cmd, env = BOX.altgen_cmd("qwen2.5-3b", smoke=True)
    check("R10 smoke alt-gen: tiny (--smoke) at 3B with the feasibility floor OFF",
          "--smoke" in cmd and cmd[cmd.index("--min-per-class") + 1] == "0", f"cmd={cmd}")
    dflt, _ = BOX.altgen_cmd("qwen2.5-3b")
    check("R10 default alt-gen unchanged (real-run floor 24, no --smoke)",
          "--smoke" not in dflt and dflt[dflt.index("--min-per-class") + 1] == "24")
    sq = BOX.smoke_bundle_specs("qwen2.5-1.5b")
    sl = BOX.smoke_bundle_specs("falcon3-1b")
    check("R10 Qwen smoke cells = the 1.5B pools BOTH wordings (the D2 eos-free diagonal "
          "anchor needs evoked + evoked_alt) + the B15 secret_word cell",
          len(sq) == 3 and any(":evoked:" in s for s in sq)
          and any(":evoked_alt:" in s for s in sq)
          and any(":secret_word:" in s for s in sq)
          and all(s.startswith("qwen2.5-1.5b:") for s in sq), f"{sq}")
    check("R10 falcon3 smoke cell = the 1.5B evoked pool only",
          sl == [s for s in sq if ":evoked:" in s], f"{sl}")
    cmd, env = BOX.mc_smoke_cmd()
    check("R10 ONE MC diagonal shard: mc_reader elicited x direct on the 3B evoked bundle, "
          "own-pool flags, no capture",
          os.path.basename(cmd[cmd.index("-u") + 1]) == "mc_reader.py"
          and cmd[cmd.index("--stream-source") + 1] == "qwen2.5-3b"
          and cmd[cmd.index("--sets") + 1] == "evoked"
          and cmd[cmd.index("--framings") + 1] == "elicited"
          and cmd[cmd.index("--reasonings") + 1] == "direct"
          and "--capture" not in cmd, f"cmd={cmd}")
    check("R10 smoke MC also lands in the fresh mc_diag report dir",
          env.get("INTRO_RUN_DIR") == os.path.join(BOX.OUT, "mc_diag")
          and env.get("INTRO_MODEL") == "qwen2.5-3b", f"env={env}")
    msrc = inspect.getsource(BOX.main)
    check("R10 main handles --smoke (D1 slice) and emits t in steps for the projection",
          "--smoke" in inspect.getsource(BOX) and "smoke" in msrc
          and '"t"' in inspect.getsource(BOX.emit_step) or "t=" in inspect.getsource(BOX.emit_step))
except Exception as e:
    check("R10 box smoke", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R11: mc_reader cell filters
try:
    import mc_reader as MR
    fr = MR.parse_subset("elicited", MR.FRAMINGS, "framing")
    rs = MR.parse_subset("direct", MR.REASONINGS, "reasoning")
    check("R11 parse_subset selects valid cells", fr == ("elicited",) and rs == ("direct",))
    try:
        MR.parse_subset("nope", MR.FRAMINGS, "framing")
        check("R11 unknown cell name raises", False, "no exception")
    except ValueError:
        check("R11 unknown cell name raises", True)
    src = open(os.path.join(REPO, "src", "mc_reader.py")).read()
    check("R11 CLI exposes --framings/--reasonings",
          '"--framings"' in src and '"--reasonings"' in src)
except Exception as e:
    check("R11 mc_reader filters", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R12: driver main source
if DRV is not None:
    try:
        msrc = inspect.getsource(DRV.main)
        check("R12 gated launch (authorized_run through the experimentfactory gate)",
              "authorized_run(" in msrc and "GateBlocked" in msrc)
        check("R12 monitor-ready: status_path + on_event + a watch hint at launch",
              "status_path" in msrc and "events_path" in msrc and "labkit watch" in msrc)
        check("R12 deadman: max_hours feeds the deadline (labkit arms autodestroy = deadline + "
              "buffer)", "max_hours" in msrc and "deadman" in inspect.getsource(DRV).lower())
        check("R12 rsync-255 retry loop wired", "is_rsync_flake(" in msrc)
        check("R12 smoke -> projection + B10 STOP wiring",
              "smoke_projection(" in msrc and "check_projection(" in msrc)
    except Exception as e:
        check("R12 driver main", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R14 [E1 blocker]: deadman
if DRV is not None:
    try:
        kw = DRV.provider_kwargs(run_id="lr-grid", disk_gb=36.7, max_hours=20.0,
                                 throttle_path="/tmp/throttle")
        check("R14 [E1 blocker] provider kwargs carry default_deadman_s covering the full run "
              "(labkit create() min()-clamps EVERY deadman to the provider default = 6h; "
              "unset -> the box self-kills ~6h into a 16.2h run)",
              kw.get("default_deadman_s") is not None
              and kw["default_deadman_s"] >= int(20.0 * 3600) + 600, f"kwargs={kw}")
        check("R14 deadman = max_hours deadline + teardown buffer; owner/disk/throttle intact",
              DRV.provider_kwargs("x", 24.0, 1.5)["default_deadman_s"]
              == int(1.5 * 3600) + DRV.DEADMAN_BUFFER_S
              and kw["owner"] == "lr-grid" and kw["disk_gb"] == 36.7
              and kw["throttle_path"] == "/tmp/throttle")
        msrc = inspect.getsource(DRV.main)
        check("R14 main() constructs the VastProvider FROM provider_kwargs (the override "
              "actually reaches the constructor)",
              "provider_kwargs(" in msrc and "VastProvider(**" in msrc)
    except Exception as e:
        check("R14 deadman clamp", False, f"raised {type(e).__name__}: {e}")

# ================================================================ R13 [SCI-SF3]: smoke out_json
if DRV is not None:
    try:
        msrc = inspect.getsource(DRV.main)
        check("R13 the smoke-mode offline-scoring command writes to "
              "reports/lr_grid_smoke_results.json (the D2 anchor persists for the full run's "
              "anchor check)",
              "lr_grid_smoke_results.json" in msrc and "out_json=" in msrc)
        check("R13 the smoke command never targets the full-run OUT_JSON "
              "(lr_grid_results.json)", "out_json='lr_grid_results.json'" not in msrc
              and "lr_grid_results.json" not in msrc)
    except Exception as e:
        check("R13 smoke out_json", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
