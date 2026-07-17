"""RED-first unit tests for LR-72B unit 4: the gated driver harness/run_lr_72b.py. No GPU, no
launch, no labkit import (driver deps lazy in main). Mirrors test_run_lr_grid's discipline.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_run_lr_72b.py

D1  the driver imports WITHOUT the driver venv (labkit/experimentfactory lazy in main).
D2  entrypoint targets box_lr_72b.py (+ --smoke passthrough); run ids lr-72b / lr-72b-smoke.
D3  ledger cap $10 (Matt 2026-07-12); the shared project ledger.
D4  disk floor covers the 144GB Qwen2.5-72B bf16 weights + image + slack (the real idle risk).
D5  HIGH-BANDWIDTH host filter: min downlink ~2000 Mbps (144GB weight pull must be minutes).
D6  2xH100 tier: min_vram covers 2 cards; the GPU request is H100-class x2.
D7  the $6 evoked projection helper: projects Phase-2 spend from a measured Phase-1 rate and
    advises the runtime gate (parity with box_lr_72b.phase2_gate).
D8  deadman: provider_kwargs carries default_deadman_s covering max_hours + buffer (E1-blocker
    parity); main constructs the provider FROM it.
D9  main() source: gated authorized_run, monitor-ready (status + events + watch hint), --dry mock,
    the smoke path prints the prompt_logprobs/tokenizer-parity/util smoke reminders.
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


DRV_PATH = os.path.join(REPO, "harness", "run_lr_72b.py")
try:
    DRV = load_module("run_lr_72b", DRV_PATH)
    check("D1 driver imports without labkit/experimentfactory (lazy in main)", True)
except Exception as e:
    DRV = None
    check("D1 driver imports without labkit/experimentfactory (lazy in main)", False,
          f"{type(e).__name__}: {e}")

# ================================================================ D2: entrypoint / run ids
if DRV is not None:
    try:
        check("D2 entrypoint targets box_lr_72b.py",
              "box_lr_72b.py" in DRV.entrypoint_for(smoke=False))
        check("D2 --smoke passes through to the box",
              DRV.entrypoint_for(smoke=True).rstrip().endswith("--smoke"))
        check("D2 run ids: lr-72b / lr-72b-smoke",
              DRV.run_id_for(False) == "lr-72b" and DRV.run_id_for(True) == "lr-72b-smoke")
    except Exception as e:
        check("D2 entrypoint/run ids", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D3: ledger cap $10
if DRV is not None:
    try:
        check("D3 the per-run authorization is $10 (Matt 2026-07-12; expected ~$5)",
              DRV.RUN_AUTHORIZED_USD == 10.0)
        check("D3 shared project ledger", DRV.LEDGER_PATH.endswith(
            os.path.join("runs", "confound-ledger.json")))
        with tempfile.TemporaryDirectory() as td:
            lp = os.path.join(td, "l.json")
            with open(lp, "w") as f:
                json.dump({"a": 1.5}, f)
            check("D3 remaining_budget = cap - ledger sum",
                  abs(DRV.remaining_budget(lp, cap=5.0) - 3.5) < 1e-9)
    except Exception as e:
        check("D3 ledger cap", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D4: disk floor for 144GB weights
if DRV is not None:
    try:
        disk = DRV.disk_for_72b()
        check("D4 disk floor covers the 144GB 72B bf16 weights + image + slack (real idle risk)",
              disk >= 144.0 + DRV.IMAGE_GB, f"disk={disk}")
        check("D4 WEIGHTS_72B_GB ~ 144 (Qwen2.5-72B bf16)", 140.0 <= DRV.WEIGHTS_72B_GB <= 150.0,
              f"{DRV.WEIGHTS_72B_GB}")
        check("D4 disk floor carries slack beyond the raw weights",
              disk >= DRV.WEIGHTS_72B_GB + DRV.IMAGE_GB + 5.0, f"disk={disk}")
    except Exception as e:
        check("D4 disk floor", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D5: high-bandwidth host filter
if DRV is not None:
    try:
        check("D5 default min downlink is high (~2000 Mbps: the 144GB pull must be minutes)",
              DRV.DEFAULT_MIN_BW >= 2000, f"{DRV.DEFAULT_MIN_BW}")
        msrc = inspect.getsource(DRV.main)
        check("D5 min_inet_down is wired into the labkit mk filter",
              "min_inet_down" in msrc)
    except Exception as e:
        check("D5 bandwidth filter", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D6: 2xH100 tier
if DRV is not None:
    try:
        check("D6 GPU request is H100-class", "H100" in DRV.DEFAULT_GPU, f"{DRV.DEFAULT_GPU}")
        check("D6 num_gpus = 2 (tensor-parallel-size 2)", DRV.NUM_GPUS == 2)
        check("D6 min VRAM covers a single H100 card (>= 80GB)",
              DRV.DEFAULT_MIN_VRAM >= 80000, f"{DRV.DEFAULT_MIN_VRAM}")
    except Exception as e:
        check("D6 2xH100 tier", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D7: $6 evoked projection
if DRV is not None:
    try:
        # the driver's projection helper must agree with the box's runtime gate (single rule).
        BOX = load_module("box_lr_72b",
                          os.path.join(REPO, "experiments", "exp2_output_monitorability",
                                       "box_lr_72b.py"))
        p = DRV.project_phase2(spend_so_far=2.0, phase1_arms=2, phase1_spend=2.0)
        g = BOX.phase2_gate(spend_so_far=2.0, phase1_arms=2, phase1_spend=2.0)
        check("D7 driver's Phase-2 projection matches the box's runtime $6 gate",
              abs(p["projected_usd"] - g["projected_usd"]) < 1e-9 and p["go"] == g["go"],
              f"drv={p} box={g}")
        check("D7 the $6 ceiling is shared (never re-hardcoded divergently)",
              DRV.PHASE2_MAX_USD == BOX.PHASE2_MAX_USD == 6.0)
    except Exception as e:
        check("D7 phase-2 projection", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D8: deadman
if DRV is not None:
    try:
        kw = DRV.provider_kwargs(run_id="lr-72b", disk_gb=170.0, max_hours=6.0,
                                 throttle_path="/tmp/t")
        check("D8 provider kwargs carry default_deadman_s covering max_hours + buffer (E1 parity)",
              kw.get("default_deadman_s") is not None
              and kw["default_deadman_s"] >= int(6.0 * 3600) + 600, f"{kw}")
        msrc = inspect.getsource(DRV.main)
        check("D8 main constructs the VastProvider FROM provider_kwargs",
              "provider_kwargs(" in msrc and "VastProvider(**" in msrc)
    except Exception as e:
        check("D8 deadman", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D9: main source
if DRV is not None:
    try:
        msrc = inspect.getsource(DRV.main)
        check("D9 gated launch (authorized_run through the experimentfactory gate)",
              "authorized_run(" in msrc and "GateBlocked" in msrc)
        check("D9 monitor-ready: status + events + a watch hint at launch",
              "status_path" in msrc and "events_path" in msrc and "labkit watch" in msrc)
        check("D9 --dry validates the gate at $0 (mock runner)", "--dry" in inspect.getsource(DRV)
              and "_mock" in msrc)
        check("D9 the smoke reminder names what to verify (prompt_logprobs teacher-forces + "
              "tokenizer parity + util)",
              ("prompt_logprobs" in msrc and "parity" in msrc.lower() and "util" in msrc.lower()))
    except Exception as e:
        check("D9 main source", False, f"raised {type(e).__name__}: {e}")

# ================================================================ D10: smoke touches an observer
# The smoke slice must exercise the observer path (Amendment 1): the box scores at least one
# smaller-model observer cell so the smoke verifies the observer scoring wiring, not just the
# diagonal. The box owns HOW; the driver's smoke reminder must NAME the observer check.
if DRV is not None:
    try:
        msrc = inspect.getsource(DRV.main)
        check("D10 the smoke reminder names the observer cell check (72B scores a smaller stream)",
              "observ" in msrc.lower())
        BOX = load_module("box_lr_72b_d10",
                          os.path.join(REPO, "experiments", "exp2_output_monitorability",
                                       "box_lr_72b.py"))
        boxsrc = inspect.getsource(BOX.main)
        check("D10 the box smoke path runs score_observer on at least one observer cell",
              "score_observer" in boxsrc)
        check("D10 observer cells are cheap SCORING-ONLY (no extra generation projection cost)",
              hasattr(BOX, "OBSERVE_GENERATORS") and hasattr(BOX, "OBSERVE_ARMS"))
    except Exception as e:
        check("D10 smoke touches observer", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
