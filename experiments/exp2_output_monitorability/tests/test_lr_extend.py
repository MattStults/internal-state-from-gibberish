"""RED-first unit tests for the LR scale-grid EXTENSION build (prereg:
reports/lr_scale_extend_prereg.md -- FROZEN 2026-07-13 + Amendment 1 2026-07-14; effective
scope 14B-only + the 70B rider): src/lr_grid_extend.py (reader-registry wrapper),
src/lr_rider.py (Amendment-1 rider), box_lr_extend.py (orchestrator bookkeeping + trim plan),
harness/run_lr_extend.py (driver math + gate facts + $0 launch gates),
analysis/lr_extend_offline.py (offline joins + rider privacy). No GPU, no network: seams are
injected.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_lr_extend.py

E1  lr_grid_extend.register: adds the new slugs with the registry hf_ids, idempotent, REFUSES
    an hf_id swap on an existing slug, leaves the six certified readers untouched, adds batch
    defaults without clobbering existing sizes.
E2  lr_grid_extend is configuration-only (no numerics) and delegates to lr_grid.main.
E3  markers: LRX_* collide with no known box marker in either direction.
E4  box fetches() covers every input class (anchor pools, secret_word x3, secret_sustain x3,
    injected captures x3, the Amendment-1 70B streams JSON); ctx-code parity with
    lr_grid.SECRET_CTX.
E5  shards_for / bundle_specs_for bookkeeping (14B-only): name-parity with lr_grid.shard_path;
    every spec parses through lr_grid.parse_bundle_spec; counts match the frozen cell list;
    evoked defaults IN (Q2 resolved); rider shard names via the same shard_path.
E6  write_strength_filtered_capture: the filtered 7B capture selects EXACTLY the pinned level
    through the UNMODIFIED certified select_streams (smax semantics), s0 kept, provenance note.
E7  gen/gen_cmd/grid_cmd/itf_cmd/rider_cmd: self-invocation + env wiring; GEN_BATCH_CAP for the
    one new size.
E8  driver math (14B-only): disk floor covers the FOUR readers' weights; provider deadman rides
    max_hours; check_projection GO/STOP at the $12 Q3 authorization with the REVISED trim order;
    is_rsync_flake; smoke_projection > 0 from synthetic steps with the rider term itemized.
E9  driver $0 launch gates: prereg_frozen() is TRUE (frozen 2026-07-13; a DRAFT-marked copy
    reads False); preflight_hf_inputs requires HF presence for .pt inputs only (the 70B streams
    JSON rides the workdir rsync when present locally, and is required on HF when missing).
E10 offline: anchor_check pending on missing shards; run2_matrix separates scored-level rows
    from the s0 centering pool; a separable synthetic run-2 cell scores positive calibrated
    bits through the certified evaluate_cell; run1/run2 own-quantity framing.
E11 gate facts (the labkit-gate shape-check): max_spend = ledger + $12 <= the $20 policy
    ceiling; image PINNED (explicit non-latest tag); min_vram >= vram_est x 1.15 on a 48GB
    card; the code-hash shakedown registry round-trips (register -> registered; content change
    invalidates); max_hours full = 11.0 (DELIBERATELY over the 4h policy ceiling --
    human-routed at launch, documented in the driver).
E12 rider batch construction: custom-id scheme parity with the scout; stream loading filters
    (arm, accepted, non-empty text); text re-tokenization + round-trip exclusion COUNTED (not
    fatal) through the certified llama_roundtrip_split; contexts composed via
    serverless_72b.build_system_prompt and rendered in the READER's chat template; label sets
    (N = arm-own neutral, R = 12 concept contexts); constants parity across
    lr_rider/box/offline.
E13 trim ladder (Amendment 3, 2026-07-14): run_plan levels 0..3 (1 = rider descriptive,
    2 = + evoked, 3 = + rider confirmatory; the 2b expressed cell is WITHDRAWN pre-data and
    hard-pinned False at every level -- not a trim step), out-of-range raises; Part B / the
    2a dose curve / secret cells / anchor structurally untrimmable (assertion, not a
    comment); rider arms ordering (confirmatory first, descriptive last); the OLD trim order
    is GONE from the driver.
E14 32B purge: no phase constant, spec, shard or command in box/driver/offline references a
    32b slug (the lr_grid_extend registry entry alone may keep it, harmlessly).
E15 offline rider scoring: a synthetic separable rider cell scores through the certified
    evaluate_cell; privacy_ok False when bits >= 0.05, True on a null cell; round-trip void
    rule; missing cells stay PENDING in the verdict.
E16 Amendment-2 box bookkeeping: the e1 capture fetch entry; dose_cmd wires --capture /
    --strengths / --no-s0 through inject_tf_lr.py; dose_shards names carry the levels;
    dose_not_scored unions the on-box dose_plan disclosures; expressed_cmd routes through
    expressed_collect.py; expressed_itf_cmd self-reads the expressed bundle at both doses;
    expressed_shards use the 'expressed' streamset (never colliding with the 2a s20 shard).
E17 Amendment-2 offline: dose cells ride run2_cells (flagged dose_2a); a dose_plan-disclosed
    not-scored dose surfaces as not_scored (never silently None); expressed self-read scores
    through the certified path; expressed_public_cells routes through dose_titration._bits
    (the SAME function object); the named-call table evaluates the frozen CLAUDE thresholds
    (s20 >= 1.0 public bits, s60 < 1.3x plain, self-read < 0.05 both doses, 2a curve < 0.05),
    stays PENDING on missing data, and carries the empty MATT slot.
"""
import json
import os
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


import importlib.util  # noqa: E402


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import config as C            # noqa: E402
import lr_grid as G           # noqa: E402
import lr_grid_extend as GX   # noqa: E402
import lr_reader as LR        # noqa: E402
import lr_rider as RID        # noqa: E402
import serverless_72b as SV   # noqa: E402

BOX = _load("box_lr_extend",
            os.path.join(REPO, "experiments", "exp2_output_monitorability", "box_lr_extend.py"))
DRV = _load("run_lr_extend", os.path.join(REPO, "harness", "run_lr_extend.py"))
OFF = _load("lr_extend_offline",
            os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis",
                         "lr_extend_offline.py"))

# ---------------------------------------------------------------- E1: registry extension
try:
    before = dict(G.GRID_READERS)
    got = GX.register()
    check("E1 both new slugs registered with the registry hf_ids",
          got.get("qwen2.5-14b") == "Qwen/Qwen2.5-14B-Instruct"
          and got.get("qwen2.5-32b") == "Qwen/Qwen2.5-32B-Instruct")
    check("E1 the six certified readers are untouched",
          all(got[k] == v for k, v in before.items() if k in before))
    got2 = GX.register()
    check("E1 idempotent", got2 == got)
    try:
        GX.register(readers={"qwen2.5-14b": "someone/other-model"})
        check("E1 hf_id swap on an existing slug refuses", False, "no exception")
    except RuntimeError:
        check("E1 hf_id swap on an existing slug refuses", True)
    check("E1 batch defaults added, existing sizes not clobbered",
          G.BATCH_BY_SIZE.get("14b") == 6 and G.BATCH_BY_SIZE["7b"] == 8)
except Exception as e:
    check("E1 registry", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E2: configuration-only
try:
    src = open(os.path.join(REPO, "src", "lr_grid_extend.py")).read()
    check("E2 no numerics in the wrapper (softmax/gather/logits/forward absent)",
          not any(t in src for t in ("log_softmax", ".gather(", "logits", "model(")))
    check("E2 delegates to lr_grid.main", "G.main()" in src)
except Exception as e:
    check("E2 configuration-only", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E3: marker collisions
try:
    NEW = ("LRX_READY", "LRX_DONE", "LRX_FATAL")
    KNOWN = ("LRG_READY", "LRG_DONE", "LRG_FATAL", "LR_READY", "LR_DONE", "LR_FATAL",
             "LR72_READY", "LR72_DONE", "LR72_FATAL", "MC_READY", "MC_DONE", "MC_FATAL",
             "ELICIT_READY", "ELICIT_DONE", "GAUGE_READY", "GAUGE_DONE",
             "COLLECT_DONE", "COLLECT_FATAL", "MODEL_READY")
    coll = [(a, b) for a in NEW for b in KNOWN if a in b or b in a]
    check("E3 LRX markers collide with no known marker in either direction", not coll,
          f"collisions: {coll}")
except Exception as e:
    check("E3 markers", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E4: fetches + ctx parity
try:
    f = BOX.fetches()
    names = [n for n, _ in f]
    check("E4 fetch list covers anchor pools + secret_word x3 + secret_sustain x3 + captures "
          "x3 + the 70B streams JSON + the Amendment-2 e1 low-dose capture",
          len(f) == 13
          and sum("evoked" in n for n in names) == 2
          and sum("secret_word" in n for n in names) == 3
          and sum("secret_sustain" in n for n in names) == 3
          and sum(n.endswith("-gen.pt") for n in names) == 4
          and "llama70b/streams_llama70b.json" in names
          and "confound-e1-gen.pt" in names)
    check("E4 the e1 capture dest is the confound-e1 run's own path (Amendment 2 2a)",
          dict(f)["confound-e1-gen.pt"].endswith(
              os.path.join("runs", "confound-e1", "data", "covert_collect.pt"))
          and dict(f)["confound-e1-gen.pt"] == BOX.e1_capture_path())
    check("E4 the streams-JSON dest is the scout's own path (rsync-carried, non-.pt)",
          dict(f)["llama70b/streams_llama70b.json"].endswith(
              os.path.join("runs", "llama70b_scout", "streams_llama70b.json")))
    check("E4 ctx-code parity with lr_grid.SECRET_CTX (stdlib module cannot import it)",
          all(BOX.SECRET_CTX[k] == G.SECRET_CTX[k] for k in BOX.SECRET_CTX))
except Exception as e:
    check("E4 fetches", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E5: bookkeeping parity
try:
    from pathlib import Path
    out = "/tmp/lrx_test_out"
    sh14_secret = BOX.shards_for("qwen2.5-14b", evoked=False, out=out)
    want = str(G.shard_path(Path(os.path.join(out, "lr_grid")), "qwen2.5-14b", "qwen2.5-3b",
                            "secret_sustain", "SS"))
    check("E5 shard names match lr_grid.shard_path exactly", want in sh14_secret, want)
    check("E5 14B reader secret block: 4 gens x 2 arms x 2 ctx = 16 shards",
          len(sh14_secret) == 16)
    sh14 = BOX.shards_for("qwen2.5-14b", out=out)
    check("E5 evoked defaults IN (Q2 resolved): +6 evoked/alt diagonal shards = 22",
          len(sh14) == 22 and sum("evoked" in os.path.basename(s) for s in sh14) == 6)
    sh7 = BOX.shards_for("qwen2.5-7b", out=out)
    check("E5 7B reader: 14B pools (4) + run-(1) injected N/A/B (3)", len(sh7) == 7
          and sum("injected" in s for s in sh7) == 3)
    sh15 = BOX.shards_for("qwen2.5-1.5b", out=out)
    check("E5 1.5B reader: 14B pools only (no injected cell)", len(sh15) == 4
          and not any("injected" in s for s in sh15))
    specs14 = BOX.bundle_specs_for("qwen2.5-14b", out=out)
    parsed = [G.parse_bundle_spec(s) for s in specs14]
    check("E5 every 14B spec parses through the certified parse_bundle_spec (10 = 8 secret + "
          "2 evoked)", len(parsed) == 10
          and {p[1] for p in parsed} == {"secret_word", "secret_sustain",
                                         "evoked", "evoked_alt"})
    specs3 = BOX.bundle_specs_for("qwen2.5-3b", out=out)
    check("E5 3B reader carries the run-(1) injected spec (its OWN capture)",
          any(p == ("qwen2.5-3b", "injected", BOX.injected_capture_path("qwen2.5-3b"))
              for p in map(G.parse_bundle_spec, specs3)))
    anames = [os.path.basename(p) for p in BOX.anchor_shards(out=out)]
    check("E5 anchor = 1.5B evoked/alt (6) + secret_word (2) shards", len(anames) == 8
          and "qwen2.5-1.5b__qwen2.5-1.5b__evoked_alt_A.pt" in anames
          and "qwen2.5-1.5b__qwen2.5-1.5b__secret_word_SW.pt" in anames)
    inames = [os.path.basename(p) for p in BOX.itf_shards(out=out)]
    check("E5 run-(2) shard set: 1.5B s60 + 3B s60 + 7B s124 + 7B s140 (x2 files)",
          len(inames) == 8 and "qwen2.5-7b__qwen2.5-7b__injected_TFV_s124.pt" in inames
          and "qwen2.5-7b__qwen2.5-7b__injected_TFN_s140.pt" in inames)
    rnames = [os.path.basename(p) for p in BOX.rider_shards("qwen2.5-7b", out=out)]
    want_r = os.path.basename(str(G.shard_path(Path("x"), "qwen2.5-7b", "llama70b",
                                               "secret_sustain", "R")))
    check("E5 rider shards: 3 arms x {N, R} = 6, named via lr_grid.shard_path",
          len(rnames) == 6 and want_r in rnames
          and "qwen2.5-7b__llama70b__evoked_N.pt" in rnames, f"{rnames}")
    all_r = [s for r in BOX.RIDER_READERS for s in BOX.rider_shards(r, out=out)]
    check("E5 full rider block: 4 readers x 3 arms x 2 = 24 shards", len(all_r) == 24)
except Exception as e:
    check("E5 bookkeeping", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E6: strength filter
try:
    with tempfile.TemporaryDirectory() as td:
        srcp = os.path.join(td, "cap.pt")
        streams = [dict(gidx=i, concept="ocean", strength=s, accepted=True,
                        tokens=np.asarray([1, 2, 3]))
                   for i, s in enumerate((0, 0, 124, 124, 140, 140))]
        torch.save(dict(streams=streams, concepts=["ocean"], strengths=[0, 124, 140],
                        model="qwen2.5-7b", inject="gen"), srcp)
        dest = os.path.join(td, "f", "cap124.pt")
        BOX.write_strength_filtered_capture(srcp, dest, 124)
        b = torch.load(dest, map_location="cpu", weights_only=False)
        sel = LR.select_streams(b, "injected")
        check("E6 filtered capture: certified select_streams picks EXACTLY the pinned level",
              len(sel) == 2 and all(s["strength"] == 124 for s in sel))
        check("E6 s0 kept + strengths updated + provenance note",
              b["strengths"] == [0, 124]
              and sum(1 for s in b["streams"] if s["strength"] == 0) == 2
              and "s124" in json.dumps(b["strength_filter"]))
except Exception as e:
    check("E6 strength filter", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E7: cmd/env wiring
try:
    cmd, env = BOX.gen_cmd("qwen2.5-14b", evoked=True, out="/tmp/o")
    check("E7 gen self-invocation: --gen-slug + INTRO_MODEL + per-slug INTRO_RUN_DIR",
          "--gen-slug" in cmd and "qwen2.5-14b" in cmd and "--evoked" in cmd
          and env["INTRO_MODEL"] == "qwen2.5-14b"
          and env["INTRO_RUN_DIR"].endswith("_ind/qwen2.5-14b"))
    cmd, env = BOX.grid_cmd("qwen2.5-14b", ["a:secret_word:/x.pt"], out="/tmp/o")
    check("E7 grid_cmd routes through lr_grid_extend.py, INTRO_RUN_DIR only",
          any(c.endswith("lr_grid_extend.py") for c in cmd) and "--reader" in cmd
          and env == {"INTRO_RUN_DIR": "/tmp/o"})
    cmd, env = BOX.itf_cmd("qwen2.5-7b", 124, out="/tmp/o", limit=2)
    check("E7 itf_cmd pins the strength + limit and routes through inject_tf_lr.py",
          any(c.endswith("inject_tf_lr.py") for c in cmd)
          and cmd[cmd.index("--strength") + 1] == "124"
          and cmd[cmd.index("--limit") + 1] == "2")
    cmd, env = BOX.rider_cmd("qwen2.5-3b", ("secret_sustain", "evoked"), out="/tmp/o", limit=4)
    check("E7 rider_cmd routes through lr_rider.py with arms/streams/limit, INTRO_RUN_DIR only",
          any(c.endswith("lr_rider.py") for c in cmd)
          and cmd[cmd.index("--arms") + 1] == "secret_sustain,evoked"
          and cmd[cmd.index("--streams") + 1] == BOX.rider_streams_path()
          and cmd[cmd.index("--limit") + 1] == "4"
          and env == {"INTRO_RUN_DIR": "/tmp/o"})
    check("E7 GEN_BATCH_CAP declared for the ONE new size (48GB VRAM math)",
          BOX.GEN_BATCH_CAP == {"qwen2.5-14b": 16})
except Exception as e:
    check("E7 cmd wiring", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E8: driver math
try:
    d = DRV.disk_for_run()
    want_min = 10.0 + 29.5 + 16.0 + 6.5 + 3.5
    check("E8 disk floor covers the FOUR readers (image + 55.5 weights + slack ~= 73.5GB)",
          want_min + 2.0 <= d <= 90, f"disk={d}")
    pk = DRV.provider_kwargs("r", 73.5, 11.0)
    check("E8 provider deadman = max_hours + buffer (the E1 clamp override)",
          pk["default_deadman_s"] == int(11.0 * 3600) + 1800 and pk["disk_gb"] == 73.5)
    stop = DRV.check_projection(12.5)
    check("E8 projection GO/STOP at the $12 Q3 authorization",
          DRV.check_projection(11.0)["go"] and not stop["go"])
    check("E8 STOP message carries the REVISED trim order",
          "rider" in stop["message"] and "NEVER" in stop["message"]
          and "Part B" in stop["message"], stop["message"])
    check("E8 rsync-255 flake detected; other errors not swallowed",
          DRV.is_rsync_flake(error="rsync failed with code 255")
          and not DRV.is_rsync_flake(error="rsync error code 23"))
    steps = [dict(step=0, t=0, phase="S0_fetch"), dict(step=100, t=600, phase="S1_anchor"),
             dict(step=500, t=1200, phase="S2_gen"), dict(step=1000, t=1500, phase="S3_lr_grid"),
             dict(step=8000, t=1900, phase="S4_inject_tf"),
             dict(step=8200, t=2100, phase="S4a_dose"),
             dict(step=8300, t=2150, phase="S4b_expressed_gen"),
             dict(step=8310, t=2250, phase="S4b_expressed_itf"),
             dict(step=8500, t=2300, phase="S5_rider"),
             dict(step=9000, t=2600, phase="lrx_done")]
    proj = DRV.smoke_projection(steps, dph=0.85)
    check("E8 smoke projection > 0 with the rider term itemized + a re-derive note",
          proj["projected_usd"] > 0 and proj["rider_s"] > 0
          and proj["gen_evoked_s"] > 0 and "re-derive" in proj["note"])
    check("E8 Amendment-3 terms: 2a dose curve projected > 0; expressed 2b permanently ZERO "
          "(withdrawn, never scheduled)",
          proj["dose_2a_s"] > 0 and proj["expressed_s"] == 0)
    proj_t2 = DRV.smoke_projection(steps, dph=0.85, trim=2)
    check("E8 trim=2 zeroes the evoked term, never the 2a dose term (Part B)",
          proj_t2["gen_evoked_s"] == 0 and proj_t2["dose_2a_s"] == proj["dose_2a_s"])
    proj_t3 = DRV.smoke_projection(steps, dph=0.85, trim=3)
    check("E8 trim=3 zeroes the rider + evoked terms, never the Part B terms",
          proj_t3["rider_s"] == 0 and proj_t3["gen_evoked_s"] == 0
          and proj_t3["expressed_s"] == 0
          and proj_t3["itf_s"] == proj["itf_s"] and proj_t3["itf_s"] > 0
          and proj_t3["dose_2a_s"] == proj["dose_2a_s"] and proj_t3["dose_2a_s"] > 0)
    proj2 = DRV.smoke_projection(steps, dph=1.70)
    check("E8 projection is linear in $/hr",
          abs(proj2["projected_usd"] - 2 * proj["projected_usd"]) < 1e-9)
except Exception as e:
    check("E8 driver math", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E9: $0 launch gates
try:
    check("E9 prereg_frozen() is TRUE (Matt froze 2026-07-13; the DRAFT marker is gone)",
          DRV.prereg_frozen() is True)
    with tempfile.TemporaryDirectory() as td:
        draft = os.path.join(td, "prereg.md")
        with open(draft, "w") as fh:
            fh.write("# DRAFT -- NOT FROZEN\nbody\n")
        check("E9 a DRAFT-marked prereg still reads NOT frozen (the gate logic is live)",
              DRV.prereg_frozen(draft) is False)
    pt_only = [n for n, d in BOX.fetches() if str(d).endswith(".pt")]
    have = [n for n in pt_only if "secret_sustain" not in n]
    missing = DRV.preflight_hf_inputs(list_repo_files=lambda: have,
                                      exists=lambda p: True)
    check("E9 preflight reports exactly the HF-missing .pt inputs (the L2 secret_sustain "
          "uploads); the locally-present streams JSON is rsync-carried, not required",
          sorted(missing) == sorted(n for n in pt_only if "secret_sustain" in n))
    missing2 = DRV.preflight_hf_inputs(list_repo_files=lambda: pt_only,
                                       exists=lambda p: str(p).endswith(".pt"))
    check("E9 a MISSING local streams JSON becomes HF-required",
          missing2 == ["llama70b/streams_llama70b.json"])
    check("E9 nothing missing -> GO",
          DRV.preflight_hf_inputs(list_repo_files=lambda: [n for n, _ in BOX.fetches()],
                                  exists=lambda p: False) == [])
except Exception as e:
    check("E9 launch gates", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E10: offline joins
try:
    with tempfile.TemporaryDirectory() as td:
        gd = os.path.join(td, "lr_grid")
        os.makedirs(gd)
        a = OFF.anchor_check(gd)
        check("E10 anchor pending (ok=None) on missing shards -- never a fabricated verdict",
              a.get("ok") is None and "pending" in a)
        # synthetic separable run-2 cell: matched label lifted, s0 rows centered
        concepts = ["c%d" % j for j in range(12)]
        rng = np.random.default_rng(0)
        recs_v, recs_n = [], []
        gidx = 0
        for rep in range(6):
            for ci, c in enumerate(concepts):
                ll = {cc: float(rng.normal(0, 0.05)) + (2.0 if cc == c else 0.0)
                      for cc in concepts}
                recs_v.append(dict(gidx=gidx, concept=c, strength=60, T=10, T_noeos=10,
                                   ll=ll, ll_eos=dict(ll), ll_tok={}))
                recs_n.append(dict(gidx=gidx, concept=c, strength=60, T=10, T_noeos=10,
                                   ll={"neutral": 0.0}, ll_eos={"neutral": 0.0}, ll_tok={}))
                gidx += 1
        for j in range(4):                       # s0 centering rows
            ll = {cc: float(rng.normal(0, 0.02)) for cc in concepts}
            recs_v.append(dict(gidx=gidx, concept="c0", strength=0, T=10, T_noeos=10,
                               ll=ll, ll_eos=dict(ll), ll_tok={}))
            recs_n.append(dict(gidx=gidx, concept="c0", strength=0, T=10, T_noeos=10,
                               ll={"neutral": 0.0}, ll_eos={"neutral": 0.0}, ll_tok={}))
            gidx += 1
        sv = dict(contexts=concepts, strength=60, records=recs_v)
        sn = dict(contexts=["neutral"], strength=60, records=recs_n)
        S, y, cen = OFF.run2_matrix(sv, sn, concepts)
        check("E10 run2_matrix separates scored-level rows from the s0 centering pool",
              S.shape == (72, 12) and len(cen) == 4 and abs(np.median(cen)) < 0.05)
        LRO = sys.modules["lr_reader_offline"]
        r = LRO.evaluate_cell(S, y)
        check("E10 separable synthetic cell scores positive calibrated bits (certified path)",
              r["bits_mean"] > 1.0 and r["top1_mean"] > 0.9, f"got {r['bits_mean']}")
        itf = os.path.join(td, "inject_tf")
        os.makedirs(itf)
        torch.save(sv, os.path.join(itf, "qwen2.5-1.5b__qwen2.5-1.5b__injected_TFV_s60.pt"))
        torch.save(sn, os.path.join(itf, "qwen2.5-1.5b__qwen2.5-1.5b__injected_TFN_s60.pt"))
        cells = OFF.run2_cells(td)
        got = cells["qwen2.5-1.5b:s60"]
        check("E10 run2_cells scores the shard pair and carries the own-quantity framing",
              got and got["calibrated"]["bits_mean"] > 1.0
              and "PURE-CONCEPT" in got["comparability"])
        check("E10 missing levels stay pending (None), never fabricated",
              cells["qwen2.5-3b:s60"] is None and cells["qwen2.5-7b:s124"] is None)
except Exception as e:
    check("E10 offline joins", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E11: gate facts
try:
    gf = DRV.gate_fields(smoke=False, spent=5.87)
    check("E11 max_spend = ledger spent + the $12 Q3 authorization, under the $20 ceiling",
          gf["max_spend"] == 17.87 and gf["max_spend"] <= 20.0
          and DRV.RUN_AUTHORIZED_USD == 12.0)
    check("E11 full-run max_hours = 11.0 (Amendment 3: 2b withdrawn, the 12h bump reverted; "
          "DELIBERATELY over the 4h policy ceiling -- human-routed, run_lr_grid precedent)",
          gf["max_hours"] == 11.0
          and DRV.gate_fields(smoke=True, spent=5.87)["max_hours"] == 2.0)
    check("E11 Amendment-3 worst case 11h x $0.85 = $9.35 stays under the $12 authorization",
          abs(DRV.MAX_HOURS_FULL * 0.85 - 9.35) < 1e-9
          and DRV.MAX_HOURS_FULL * 0.85 <= DRV.RUN_AUTHORIZED_USD)
    img = gf["image"]
    tag = img.rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    check("E11 container image PINNED: explicit non-latest tag (the gate's _image_pinned rule)",
          ":" in img.rsplit("/", 1)[-1] and tag and tag != "latest", img)
    check("E11 image is labkit's proven default (same actual image as the grid, now explicit)",
          img == "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    check("E11 VRAM: min_vram >= est x 1.15 AND the request fits a 48GB card",
          gf["min_vram_mb"] >= gf["vram_est_mb"] * 1.15 and gf["min_vram_mb"] <= 49152
          and gf["vram_est_mb"] == 39600, f"{gf}")
    # 14B peak derivation sanity: weights + gen KV + logit chunks + overhead
    est = 30208 + 16 * 2200 * 192 // 1024 + 717 + 2048
    check("E11 VRAM estimate matches its own derivation (29.5GB weights + batch-16 KV at "
          "192KB/tok + fp32 chunks + overhead)", abs(gf["vram_est_mb"] - est) <= 100, f"{est}")
    with tempfile.TemporaryDirectory() as td:
        f1 = os.path.join(td, "a.py")
        with open(f1, "w") as fh:
            fh.write("x = 1\n")
        h1 = DRV.code_hash(files=("a.py",), repo=td)
        reg = os.path.join(td, "shakedowns.json")
        check("E11 unregistered code-hash -> shakedown_done False",
              DRV.shakedown_registered(h1, registry=reg) is False)
        DRV.register_shakedown(h1, info=dict(run_id="t"), registry=reg)
        check("E11 a registered smoke flips shakedown_done True for the SAME code-hash",
              DRV.shakedown_registered(h1, registry=reg) is True)
        with open(f1, "w") as fh:
            fh.write("x = 2\n")
        h2 = DRV.code_hash(files=("a.py",), repo=td)
        check("E11 a one-byte code change invalidates the shakedown (new hash unregistered)",
              h2 != h1 and DRV.shakedown_registered(h2, registry=reg) is False)
    check("E11 every code-hash file exists (the hash covers what the box executes)",
          all(os.path.exists(os.path.join(REPO, f)) for f in DRV.CODE_HASH_FILES)
          and "src/lr_rider.py" in DRV.CODE_HASH_FILES)
    check("E11 the 2a execution surface is hashed (covert_collect + the frozen primers chain) "
          "and the WITHDRAWN 2b module is NOT (Amendment 3: unscheduled code must not "
          "invalidate shakedowns)",
          all(f in DRV.CODE_HASH_FILES for f in
              ("src/covert_collect.py",
               "experiments/exp3_induction_and_scale/primers.py",
               "experiments/exp3_induction_and_scale/primers_v2.py"))
          and "src/expressed_collect.py" not in DRV.CODE_HASH_FILES)
except Exception as e:
    check("E11 gate facts", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E12: rider construction
try:
    check("E12 custom-id scheme parity with the scout (lr:{arm}:{concept}:{stream_idx}:{ctx})",
          RID.custom_id("evoked", "anger", 7, "matched") == "lr:evoked:anger:7:matched"
          and RID.custom_id("evoked", "anger", 7) == "lr:evoked:anger:7")
    with tempfile.TemporaryDirectory() as td:
        sj = os.path.join(td, "streams.json")
        data = [dict(concept="anger", arm="evoked", text="tgf jkp", accepted=True,
                     stream_idx=0, token_ids=None),
                dict(concept="ocean", arm="evoked", text="", accepted=True,
                     stream_idx=1, token_ids=None),
                dict(concept="fear", arm="evoked", text="zzz", accepted=False,
                     stream_idx=2, token_ids=None),
                dict(concept="fear", arm="secret_word", text="qxr vb", accepted=True,
                     stream_idx=3, token_ids=None)]
        with open(sj, "w") as fh:
            json.dump(data, fh)
        got = RID.load_rider_streams(sj, "evoked")
        check("E12 stream loading filters arm + accepted + non-empty text",
              len(got) == 1 and got[0]["stream_idx"] == 0)

    class FakeTok:
        """Round-trip seam: text 'BAD...' corrupts on decode; everything else round-trips."""
        eos_token_id = 9
        def __call__(self, text, add_special_tokens=False, **kw):
            return types.SimpleNamespace(input_ids=[ord(c) % 97 for c in text])
        def decode(self, ids, **kw):
            s = "".join(chr(97 + (i % 26)) for i in ids)
            return s  # never equals the original text -> always excluded
    class GoodTok(FakeTok):
        def __init__(self):
            self._mem = {}
        def __call__(self, text, add_special_tokens=False, **kw):
            ids = [1000 + i for i in range(len(text.split()))]
            self._mem[tuple(ids)] = text
            return types.SimpleNamespace(input_ids=ids)
        def decode(self, ids, **kw):
            return self._mem.get(tuple(ids), "")
    recs = [dict(concept="anger", arm="evoked", text="tgf jkp", stream_idx=5),
            dict(concept="ocean", arm="evoked", text="qxr vb wm", stream_idx=6)]
    pool, exc, tot = RID.rider_pool(recs, GoodTok())
    check("E12 rider_pool re-tokenizes TEXT with the reader tokenizer (gidx = scout "
          "stream_idx, strength 1)",
          len(pool) == 2 and exc == 0 and tot == 2 and pool[0]["gidx"] == 5
          and pool[0]["strength"] == 1 and len(pool[1]["tokens"]) == 3)
    pool_b, exc_b, tot_b = RID.rider_pool(recs, FakeTok())
    check("E12 round-trip failures are EXCLUDED AND COUNTED, never fatal (certified "
          "llama_roundtrip_split)", pool_b == [] and exc_b == 2 and tot_b == 2)

    class CtxTok:
        def apply_chat_template(self, msgs, **kw):
            self.msgs = msgs
            return torch.tensor([[1, 2, 3]])
    ct = CtxTok()
    ids = RID.rider_ctx_ids(ct, "secret_sustain", "ocean", "cpu")
    check("E12 contexts compose via serverless_72b.build_system_prompt (byte-identical to the "
          "70B run) and render in the READER's chat template",
          ct.msgs[0]["role"] == "system"
          and ct.msgs[0]["content"] == SV.build_system_prompt("ocean", "secret_sustain",
                                                              C.STRONG_SYSTEM)
          and ct.msgs[1]["content"] == C.GEN_PROMPT and ids.shape == (1, 3))
    ct2 = CtxTok()
    RID.rider_ctx_ids(ct2, "secret_sustain", None, "cpu")
    check("E12 concept=None renders the ARM-OWN neutral (build_system_prompt(None, arm, ...))",
          ct2.msgs[0]["content"] == SV.build_system_prompt(None, "secret_sustain",
                                                           C.STRONG_SYSTEM)
          and ct2.msgs[0]["content"] != ct.msgs[0]["content"])
    check("E12 label sets: N = the single arm-own neutral; R = the 12 concept contexts "
          "(matched + 11 mismatched)",
          RID.rider_labels("N") == [("neutral", None)]
          and [c for _, c in RID.rider_labels(RID.RIDER_CTX)] == list(C.COVERT_CONCEPTS))
    check("E12 constants parity across lr_rider / box / offline",
          RID.RIDER_ARMS == BOX.RIDER_ARMS == OFF.RIDER_ARMS
          and RID.RIDER_CONFIRMATORY == BOX.RIDER_CONFIRMATORY == OFF.RIDER_CONFIRMATORY
          and RID.RIDER_CTX == BOX.RIDER_CTX == OFF.RIDER_CTX
          and RID.RIDER_GEN == BOX.RIDER_GEN == OFF.RIDER_GEN
          and RID.RIDER_READERS == BOX.RIDER_READERS == OFF.RIDER_READERS
          and RID.PRIVACY_BOUND_BITS == OFF.RIDER_PRIVACY_BOUND == 0.05)
    check("E12 the descriptive arm is secret_word and it sits LAST in RIDER_ARMS (trim order)",
          RID.RIDER_DESCRIPTIVE == ("secret_word",) and RID.RIDER_ARMS[-1] == "secret_word")
except Exception as e:
    check("E12 rider construction", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E13: REVISED trim order
try:
    p0 = BOX.run_plan(0)
    check("E13 trim 0: everything in EXCEPT expressed_2b (Amendment 3: withdrawn, hard-pinned "
          "False); part_b + dose_2a pinned True",
          p0 == dict(rider_descriptive=True, expressed_2b=False, evoked=True,
                     rider_confirmatory=True, part_b=True, dose_2a=True,
                     secret_cells=True, anchor=True))
    p1, p2, p3 = BOX.run_plan(1), BOX.run_plan(2), BOX.run_plan(3)
    check("E13 trim 1 drops ONLY the rider secret_word (descriptive) cells",
          not p1["rider_descriptive"] and p1["evoked"] and p1["rider_confirmatory"])
    check("E13 trim 2 also drops the evoked arms (Amendment 3 ladder)",
          not p2["rider_descriptive"] and not p2["evoked"] and p2["rider_confirmatory"])
    check("E13 trim 3 also drops the rider confirmatory cells -- and NOTHING else",
          not p3["rider_confirmatory"] and p3["part_b"] and p3["dose_2a"]
          and p3["secret_cells"] and p3["anchor"])
    check("E13 expressed_2b is False at EVERY trim level (withdrawn, unschedulable)",
          all(not BOX.run_plan(t)["expressed_2b"] for t in range(4)))
    for bad in (-1, 4):
        try:
            BOX.run_plan(bad)
            check(f"E13 trim {bad} raises (no trim level can reach Part B)", False,
                  "no exception")
        except ValueError:
            check(f"E13 trim {bad} raises (no trim level can reach Part B)", True)
    check("E13 assert_part_b_untrimmable passes every legal plan",
          all(BOX.assert_part_b_untrimmable(BOX.run_plan(t)) for t in range(4)))
    try:
        BOX.assert_part_b_untrimmable(dict(BOX.run_plan(0), part_b=False))
        check("E13 a doctored plan without Part B RAISES (assertion, not a comment)", False,
              "no exception")
    except RuntimeError:
        check("E13 a doctored plan without Part B RAISES (assertion, not a comment)", True)
    try:
        BOX.assert_part_b_untrimmable(dict(BOX.run_plan(0), dose_2a=False))
        check("E13 a doctored plan without the 2a dose curve RAISES (2a inherits Part B "
              "untrimmability, Amendment 2)", False, "no exception")
    except RuntimeError:
        check("E13 a doctored plan without the 2a dose curve RAISES (2a inherits Part B "
              "untrimmability, Amendment 2)", True)
    check("E13 rider arms per plan: confirmatory first, descriptive last, trims peel in order",
          BOX.rider_arms_for(p0) == ("secret_sustain", "evoked", "secret_word")
          and BOX.rider_arms_for(p1) == ("secret_sustain", "evoked")
          and BOX.rider_arms_for(p2) == ("secret_sustain", "evoked")
          and BOX.rider_arms_for(p3) == ())
    import inspect
    check("E13 the box main() calls the untrimmability assertion",
          "assert_part_b_untrimmable(" in inspect.getsource(BOX.main))
    check("E13 the OLD trim order (run (2) -> run (1) -> evoked) is GONE from the driver",
          "run (2)" not in DRV.TRIM_ORDER and "rider" in DRV.TRIM_ORDER
          and "NEVER" in DRV.TRIM_ORDER and "Part B" in DRV.TRIM_ORDER)
except Exception as e:
    check("E13 trim order", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E14: 32B purge
try:
    check("E14 box: no phase constant references a 32b slug",
          BOX.NEW_GENS == ("qwen2.5-14b",) and "qwen2.5-32b" not in BOX.READERS
          and "qwen2.5-32b" not in BOX.GEN_BATCH_CAP
          and not any("32b" in r for r in BOX.RIDER_READERS))
    blobs = []
    for reader in BOX.READERS:
        blobs += BOX.bundle_specs_for(reader, out="/tmp/o")
        blobs += BOX.shards_for(reader, out="/tmp/o")
    blobs += BOX.anchor_shards(out="/tmp/o") + BOX.itf_shards(out="/tmp/o")
    blobs += [s for r in BOX.RIDER_READERS for s in BOX.rider_shards(r, out="/tmp/o")]
    blobs += BOX.dose_shards(out="/tmp/o") + BOX.expressed_shards(out="/tmp/o")
    blobs += [BOX.expressed_bundle_path(out="/tmp/o"), BOX.e1_capture_path()]
    check("E14 box: no spec/shard the box can emit mentions 32b",
          not any("32b" in b for b in blobs))
    check("E14 driver: 32B unpriced (a 32b reader would KeyError the disk floor, by design)",
          "32b" not in DRV.WEIGHTS_GB)
    check("E14 offline: 14B-only gens/readers; prefix-K full pass expects 8 pools",
          OFF.NEW_GENS == ("qwen2.5-14b",) and not any("32b" in r for r in OFF.READERS)
          and OFF.K_POOLS_FULL == 8)
except Exception as e:
    check("E14 32B purge", False, f"raised {type(e).__name__}: {e}")

# ------------------------------------------------- E14b: marker guard on self-reinvocation
try:
    check("E14b a --gen-slug child invocation is recognized (must never print LRX_DONE -- "
          "labkit substring-matches markers; the 2026-07-14 premature-done smoke bug)",
          BOX._is_child_invocation(["box_lr_extend.py", "--gen-slug", "qwen2.5-14b"]) is True
          and BOX._is_child_invocation(["box_lr_extend.py", "--smoke"]) is False)
    import inspect as _insp
    _src = _insp.getsource(sys.modules[BOX.__name__])
    check("E14b the LRX_DONE print is guarded by the child check in __main__",
          "_is_child_invocation(sys.argv)" in _src
          and _src.count('print("LRX_DONE"') == 1)
except Exception as e:
    check("E14b marker guard", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E15: offline rider scoring
try:
    LRO = sys.modules["lr_reader_offline"]
    with tempfile.TemporaryDirectory() as td:
        gd = os.path.join(td, "lr_grid")
        os.makedirs(gd)
        concepts = list(C.COVERT_CONCEPTS)
        rng = np.random.default_rng(1)

        def rider_shard_pair(lift):
            # 9 reps/concept: the SCI-B1 VOID-thin gate needs >= 6 EVAL streams/concept/seed
            # (split_thirds keeps 2/3 for eval -> 9 - 3 = 6; 6 reps would leave 4 and void).
            recs_r, recs_n = [], []
            gidx = 0
            for rep in range(9):
                for c in concepts:
                    ll = {cc: float(rng.normal(0, 0.05)) + (lift if cc == c else 0.0)
                          for cc in concepts}
                    tok_r = {cc: np.full(10, ll[cc] / 10, dtype=np.float16)
                             for cc in concepts}
                    recs_r.append(dict(gidx=gidx, concept=c, strength=1, T=10, T_noeos=10,
                                       ll=ll, ll_eos=dict(ll), ll_tok=tok_r))
                    recs_n.append(dict(gidx=gidx, concept=c, strength=1, T=10, T_noeos=10,
                                       ll={"neutral": 0.0}, ll_eos={"neutral": 0.0},
                                       ll_tok={"neutral": np.zeros(10, dtype=np.float16)}))
                    gidx += 1
            meta = dict(contexts=concepts, roundtrip_excluded=1, roundtrip_total=72)
            return (dict(meta, ctxset="R", records=recs_r),
                    dict(contexts=["neutral"], ctxset="N", records=recs_n,
                         roundtrip_excluded=1, roundtrip_total=72))

        # an ADVERSE cell (readable) at 7B/secret_sustain; a NULL cell at 7B/evoked
        sr, sn = rider_shard_pair(2.0)
        torch.save(sr, os.path.join(gd, "qwen2.5-7b__llama70b__secret_sustain_R.pt"))
        torch.save(sn, os.path.join(gd, "qwen2.5-7b__llama70b__secret_sustain_N.pt"))
        nr, nn = rider_shard_pair(0.0)
        torch.save(nr, os.path.join(gd, "qwen2.5-7b__llama70b__evoked_R.pt"))
        torch.save(nn, os.path.join(gd, "qwen2.5-7b__llama70b__evoked_N.pt"))
        hot = OFF.rider_cell("qwen2.5-7b", "secret_sustain", gd)
        cold = OFF.rider_cell("qwen2.5-7b", "evoked", gd)
        check("E15 a separable rider cell scores positive calibrated bits (certified "
              "evaluate_cell) and FAILS the privacy bound",
              hot["primary"]["bits_mean"] > 1.0 and hot["privacy_ok"] is False
              and hot["role"] == "confirmatory")
        check("E15 a null rider cell passes the privacy bound (< 0.05 calibrated bits)",
              abs(cold["primary"]["bits_mean"]) < 0.05 and cold["privacy_ok"] is True)
        check("E15 round-trip exclusion counts surface from the shard meta (1/72 < 5% -> not "
              "void)", hot["roundtrip_excluded"] == 1 and hot["roundtrip_total"] == 72
              and hot["roundtrip_void"] is False)
        # char_fn seam (2026-07-14 fixes): the adverse branch would otherwise run the REAL
        # amended char reader over the 70B pool -- hermetic tests inject a passing control.
        out = OFF.rider_cells(td, char_fn=lambda arm: dict(mean=0.0, sd=0.01, n=24))
        check("E15 missing rider cells stay PENDING in the joint verdict (never fabricated)",
              out["cells"]["qwen2.5-14b|llama70b|secret_sustain"] is None
              and out["privacy_verdict"].startswith("PENDING"))
        # a >5% round-trip exclusion voids the cell's privacy verdict (ok -> None)
        sr_v, sn_v = rider_shard_pair(0.0)
        sr_v["roundtrip_excluded"] = 10
        torch.save(sr_v, os.path.join(gd, "qwen2.5-7b__llama70b__secret_word_R.pt"))
        torch.save(sn_v, os.path.join(gd, "qwen2.5-7b__llama70b__secret_word_N.pt"))
        void = OFF.rider_cell("qwen2.5-7b", "secret_word", gd)
        check("E15 >5% round-trip exclusion voids the cell (privacy_ok None, the B13 rule)",
              void["roundtrip_void"] is True and void["privacy_ok"] is None
              and void["role"] == "descriptive")
except Exception as e:
    check("E15 offline rider", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E16: Amendment-2 box wiring
try:
    cmd, env = BOX.dose_cmd("qwen2.5-1.5b", BOX.e1_capture_path(), (3, 5, 8, 12, 20),
                            out="/tmp/o", limit=2)
    check("E16 dose_cmd routes through inject_tf_lr.py with the e1 capture + --strengths + "
          "--no-s0 (the disclosed centering trim) + --limit",
          any(c.endswith("inject_tf_lr.py") for c in cmd)
          and cmd[cmd.index("--capture") + 1] == BOX.e1_capture_path()
          and cmd[cmd.index("--strengths") + 1] == "3,5,8,12,20"
          and "--no-s0" in cmd and cmd[cmd.index("--limit") + 1] == "2"
          and env["INTRO_MODEL"] == "qwen2.5-1.5b" and env["INTRO_RUN_DIR"] == "/tmp/o")
    dnames = [os.path.basename(p) for p in BOX.dose_shards(out="/tmp/o")]
    check("E16 dose_shards: TFV+TFN per Amendment-2 level (e1 s3-s20 + main s40) = 12 files, "
          "the run-(2) naming (level in the ctx code)",
          len(dnames) == 12
          and "qwen2.5-1.5b__qwen2.5-1.5b__injected_TFV_s3.pt" in dnames
          and "qwen2.5-1.5b__qwen2.5-1.5b__injected_TFN_s40.pt" in dnames
          and BOX.DOSE_LEVELS_E1 == (3, 5, 8, 12, 20) and BOX.DOSE_LEVELS_MAIN == (40,))
    with tempfile.TemporaryDirectory() as td:
        it = os.path.join(td, "inject_tf")
        os.makedirs(it)
        check("E16 no disclosures -> nothing excluded", BOX.dose_not_scored(out=td) == set())
        with open(os.path.join(it, "dose_plan_qwen2.5-1.5b_injected_s3-5-8-12-20.json"),
                  "w") as fh:
            json.dump(dict(slug="qwen2.5-1.5b", streamset="injected",
                           scored=[3, 8, 12, 20],
                           not_scored=[dict(level=5, reason="no inject_alpha")]), fh)
        with open(os.path.join(it, "dose_plan_qwen2.5-1.5b_injected_s40.json"), "w") as fh:
            json.dump(dict(slug="qwen2.5-1.5b", streamset="injected", scored=[],
                           not_scored=[dict(level=40, reason="no vectors")]), fh)
        check("E16 dose_not_scored unions every on-box dose_plan disclosure",
              BOX.dose_not_scored(out=td) == {5, 40})
    cmd, env = BOX.expressed_cmd(out="/tmp/o")
    check("E16 expressed_cmd routes through expressed_collect.py with both source captures",
          any(c.endswith("expressed_collect.py") for c in cmd)
          and cmd[cmd.index("--e1-capture") + 1] == BOX.e1_capture_path()
          and cmd[cmd.index("--main-capture") + 1]
          == BOX.injected_capture_path("qwen2.5-1.5b")
          and env["INTRO_MODEL"] == "qwen2.5-1.5b" and env["INTRO_RUN_DIR"] == "/tmp/o")
    cmd_s, _ = BOX.expressed_cmd(out="/tmp/o", smoke=True)
    check("E16 expressed_cmd --smoke rides the flag", "--smoke" in cmd_s)
    cmd, env = BOX.expressed_itf_cmd(out="/tmp/o")
    check("E16 expressed_itf_cmd self-reads the expressed bundle at BOTH doses, s0-free",
          cmd[cmd.index("--capture") + 1] == BOX.expressed_bundle_path(out="/tmp/o")
          and cmd[cmd.index("--strengths") + 1] == "20,60" and "--no-s0" in cmd)
    xnames = [os.path.basename(p) for p in BOX.expressed_shards(out="/tmp/o")]
    check("E16 expressed shards use the 'expressed' streamset (s20 never collides with the "
          "2a injected_s20 shard), TFV+TFN x {s20, s60}",
          len(xnames) == 4
          and "qwen2.5-1.5b__qwen2.5-1.5b__expressed_TFV_s20.pt" in xnames
          and "qwen2.5-1.5b__qwen2.5-1.5b__expressed_TFN_s60.pt" in xnames
          and not any("injected" in n for n in xnames))
    xs = [os.path.basename(p) for p in BOX.expressed_shards(out="/tmp/o", smoke=True)]
    check("E16 smoke expressed slice: s20 only", len(xs) == 2 and all("_s20" in n for n in xs))
    check("E16 expressed bundle path follows the <slug>-<arm>.pt convention",
          BOX.expressed_bundle_path(out="/tmp/o").endswith(
              os.path.join("expressed", "qwen2.5-1.5b-expressed.pt")))
except Exception as e:
    check("E16 Amendment-2 box wiring", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- E17: Amendment-2 offline
try:
    LRO = sys.modules["lr_reader_offline"]
    concepts12 = ["c%d" % j for j in range(12)]
    rng = np.random.default_rng(2)

    def tf_shard_pair(level, lift, n_rep=6):
        recs_v, recs_n = [], []
        gidx = 0
        for rep in range(n_rep):
            for c in concepts12:
                ll = {cc: float(rng.normal(0, 0.05)) + (lift if cc == c else 0.0)
                      for cc in concepts12}
                recs_v.append(dict(gidx=gidx, concept=c, strength=level, T=10, T_noeos=10,
                                   ll=ll, ll_eos=dict(ll), ll_tok={}))
                recs_n.append(dict(gidx=gidx, concept=c, strength=level, T=10, T_noeos=10,
                                   ll={"neutral": 0.0}, ll_eos={"neutral": 0.0}, ll_tok={}))
                gidx += 1
        return (dict(contexts=concepts12, strength=level, records=recs_v),
                dict(contexts=["neutral"], strength=level, records=recs_n))

    with tempfile.TemporaryDirectory() as td:
        it = os.path.join(td, "inject_tf")
        os.makedirs(it)
        sv, sn = tf_shard_pair(12, 0.0)
        torch.save(sv, os.path.join(it, "qwen2.5-1.5b__qwen2.5-1.5b__injected_TFV_s12.pt"))
        torch.save(sn, os.path.join(it, "qwen2.5-1.5b__qwen2.5-1.5b__injected_TFN_s12.pt"))
        with open(os.path.join(it, "dose_plan_qwen2.5-1.5b_injected_s3-5-8-12-20.json"),
                  "w") as fh:
            json.dump(dict(slug="qwen2.5-1.5b", streamset="injected", scored=[12],
                           not_scored=[dict(level=5, reason="no inject_alpha stored")]), fh)
        cells = OFF.run2_cells(td)
        got12 = cells["qwen2.5-1.5b:s12"]
        check("E17 a 2a dose cell rides run2_cells, flagged dose_2a, null bits on a null shard",
              got12 and got12.get("dose_2a") is True
              and abs(got12["calibrated"]["bits_mean"]) < 0.05)
        got5 = cells["qwen2.5-1.5b:s5"]
        check("E17 a dose_plan-disclosed dose surfaces as not_scored (disclosed, never a "
              "silent None)", isinstance(got5, dict) and got5.get("not_scored") is True
              and "alpha" in got5.get("reason", ""))
        check("E17 an undisclosed missing dose stays pending (None)",
              cells["qwen2.5-1.5b:s3"] is None and cells["qwen2.5-1.5b:s8"] is None)
        # expressed self-read: a separable s20 cell + a null s60 cell
        xv, xn = tf_shard_pair(20, 2.0)
        torch.save(xv, os.path.join(it, "qwen2.5-1.5b__qwen2.5-1.5b__expressed_TFV_s20.pt"))
        torch.save(xn, os.path.join(it, "qwen2.5-1.5b__qwen2.5-1.5b__expressed_TFN_s20.pt"))
        xr = OFF.expressed_self_read(td)
        check("E17 expressed self-read scores through the certified path; missing dose PENDING",
              xr["20"]["calibrated"]["bits_mean"] > 1.0 and xr["60"] is None)

    # named-call table: frozen CLAUDE thresholds + pending propagation + the MATT slot
    pub = {"20": dict(dist_T12=dict(mean=1.2)), "60": dict(dist_T12=dict(mean=3.0))}
    slf = {"20": dict(calibrated=dict(bits_mean=0.01)),
           "60": dict(calibrated=dict(bits_mean=0.02))}
    run2 = {f"qwen2.5-1.5b:s{l}": dict(calibrated=dict(bits_mean=0.01), dose_2a=True)
            for l in (3, 5, 8, 12, 20, 40, 60)}
    nc = OFF.amendment2_named_calls(pub, slf, run2)
    comp = nc["claude"]["components"]
    check("E17 CLAUDE call HOLDS on data meeting every frozen threshold",
          comp["s20_public_amplified"]["ok"] is True
          and comp["s60_public_ceiling_capped"]["ok"] is True
          and comp["expressed_self_read_blind"]["ok"] is True
          and comp["dose_robust_self_blindness_2a"]["ok"] is True
          and "HOLDS" in nc["claude"]["verdict"])
    check("E17 the frozen thresholds are the prereg numbers (1.0 bits / 1.3x 2.64 / 0.05)",
          OFF.CLAUDE_S20_MIN_BITS == 1.0 and OFF.CLAUDE_S60_MAX_RATIO == 1.3
          and OFF.PLAIN_DIST_T12 == {20: 0.66, 60: 2.64} and OFF.SELF_READ_BOUND == 0.05)
    bad_pub = {"20": dict(dist_T12=dict(mean=0.7)), "60": dict(dist_T12=dict(mean=3.0))}
    nc_bad = OFF.amendment2_named_calls(bad_pub, slf, run2)
    check("E17 a sub-threshold s20 public read is computed ok=False but VOIDED (Amendment 3) "
          "and never fails the verdict -- only the 2a clause scores",
          nc_bad["claude"]["components"]["s20_public_amplified"]["ok"] is False
          and nc_bad["claude"]["components"]["s20_public_amplified"]["voided"] is True
          and "FAILS" not in nc_bad["claude"]["verdict"]
          and "2a clause" in nc_bad["claude"]["verdict"])
    hot_self = {"20": dict(calibrated=dict(bits_mean=0.30)),
                "60": dict(calibrated=dict(bits_mean=0.02))}
    nc_hot = OFF.amendment2_named_calls(pub, hot_self, run2)
    check("E17 a readable expressed self-read (>= 0.05) FAILS the self-blindness component",
          nc_hot["claude"]["components"]["expressed_self_read_blind"]["ok"] is False)
    nc_p = OFF.amendment2_named_calls(None, None, {})
    check("E17 missing data stays PENDING (never a fabricated verdict)",
          all(c["ok"] is None for c in nc_p["claude"]["components"].values())
          and "PENDING" in nc_p["claude"]["verdict"])
    check("E17 the MATT slot is carried EMPTY (pending -- entered before the box runs)",
          nc["matt"]["call"] is None and "pending" in nc["matt"]["status"].lower())
    # 2a curve: a disclosed not-scored dose is excluded (with note), an undisclosed one pends
    run2_ns = dict(run2)
    run2_ns["qwen2.5-1.5b:s5"] = dict(not_scored=True, reason="no alpha")
    ok_ns = OFF.amendment2_named_calls(pub, slf, run2_ns)
    check("E17 a disclosed not-scored dose is excluded from the 2a curve (call still "
          "evaluable)", ok_ns["claude"]["components"]["dose_robust_self_blindness_2a"]["ok"]
          is True)
    run2_gap = {k: v for k, v in run2.items() if k != "qwen2.5-1.5b:s12"}
    gap = OFF.amendment2_named_calls(pub, slf, run2_gap)
    check("E17 an UNdisclosed missing dose leaves the 2a-curve component PENDING",
          gap["claude"]["components"]["dose_robust_self_blindness_2a"]["ok"] is None)

    # certified public readers: the default bits function IS dose_titration._bits
    sentinel_calls = []

    class _FakeDT:
        @staticmethod
        def _bits(streams, tok, mode, budget, n, min_len):
            sentinel_calls.append((mode, budget, n, min_len))
            return dict(mean=0.5, sd=0.0, per_seed=[0.5])
    orig_dt = OFF._dose_titration
    OFF._dose_titration = lambda: _FakeDT
    try:
        with tempfile.TemporaryDirectory() as td:
            xd = os.path.join(td, "expressed")
            os.makedirs(xd)
            streams = []
            for lvl in (20, 60):
                for ci in range(12):
                    for j in range(6):
                        streams.append(dict(
                            gidx=len(streams), concept="c%d" % ci, concept_idx=ci,
                            strength=lvl, accepted=True,
                            tokens=np.asarray([1, 2, 3] * 5),
                            gen_topk=[dict(ids=np.asarray([1]), logp=np.asarray([-1.0]))] * 15))
            torch.save(dict(streams=streams, concepts=["c%d" % i for i in range(12)],
                            strengths=[20, 60], model="qwen2.5-1.5b"),
                       os.path.join(xd, "qwen2.5-1.5b-expressed.pt"))
            pubc = OFF.expressed_public_cells(td, tok="fake-tok")
            check("E17 expressed_public_cells routes dist_T12 + char_T12 + char_full through "
                  "the dose_titration._bits function object per dose",
                  pubc is not None and set(pubc) == {"20", "60"}
                  and all(set(c) >= {"dist_T12", "char_T12", "char_full", "n_accepted"}
                          for c in pubc.values())
                  and len(sentinel_calls) == 6
                  and {m for m, *_ in sentinel_calls} == {"dist", "char"})
            check("E17 a missing expressed bundle -> None (pending)",
                  OFF.expressed_public_cells("/tmp/nonexistent-box-dir") is None)
    finally:
        OFF._dose_titration = orig_dt
except Exception as e:
    check("E17 Amendment-2 offline", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
