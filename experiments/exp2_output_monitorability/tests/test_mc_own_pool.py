"""RED-first unit tests for the MC self-report DIAGONAL pool wiring (scale-grid checklist B6;
prereg: reports/lr_scale_grid_prereg.md "MC self-report diagonal" + reports/mc_reader_prereg.md).
No model, no GPU.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_mc_own_pool.py

The certified MC-letter reader (src/mc_reader.py) reads a FIXED 1.5B stream pool. B6 extends the
POOL WIRING ONLY so a 3B reader reads its OWN 3B evoked streams and a 7B reader its OWN 7B evoked
streams (runs/_ind/qwen2.5-{size}/data/{size}-evoked.pt), same selection rules as the frozen
elicit_reader.select_streams (deterministic ascending-gidx cap, CAP_PER_CONCEPT=17, no RNG). The
certified SCORING BODIES must remain byte-identical (S1 pins their source hashes).

P1  bind_pools binds the evoked/evoked_s0 sets from a reader's OWN generator-size bundle
    (RED pre-change: mc_reader has no bind_pools -- the pool loader hardcodes the 1.5B path via
    _assert_provenance's fixed STREAM_SOURCE and main()'s unparameterized wiring).
P2  selection parity: pool selection is EXACTLY elicit_reader.select_streams (same object), and
    bind_pools output equals direct select_streams calls on the same bundle (cap 17, ascending
    gidx, accepted, len>=2; evoked_s0 = neutral, uncapped).
P3  provenance: an own-pool bind REJECTS a bundle whose model != the declared stream source
    (a 1.5B bundle offered as the 3B pool is FATAL, and vice versa).
P4  injected sets stay bound to the fixed exp1 1.5B capture: requesting them with a non-default
    stream source raises; requesting them with no capture raises.
P5  default invocation unchanged: all four sets from the 1.5B capture+bundle bind exactly as the
    certified run did.
G2  first_ids gate wiring: with capture first_ids the registered assert_first_ids runs unchanged
    (match passes / mismatch FATAL); without a capture, only the diagonal (reader == stream
    source) passes; an off-diagonal evoked-only run is FATAL, never a silent skip.
R1  resume guard: an existing shard from a DIFFERENT pool refuses to MC_SKIP (silent cross-pool
    resume would splice two experiments).
X1  box_mc own-pool wiring: per-size S0 fetches (exp3/bundles/<slug>-evoked.pt), 8 diagonal
    shards (evoked sets only), reader cmd carries --stream-source/--sets/--evoked and NO
    --capture; the default cmd is byte-parity with the certified invocation.
X2  mc_reader CLI exposes --stream-source and --sets; --capture is no longer required=True.
S1  certified scoring bodies byte-identical: sha256 of each scoring function's source pinned at
    the certified (pre-B6) state; frozen prereg constants re-asserted.
"""
import hashlib
import inspect
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


try:
    import mc_reader as MR
except Exception as e:
    MR = None
    check("import src/mc_reader.py", False, f"{type(e).__name__}: {e}")

try:
    import elicit_reader as ER
except Exception as e:
    ER = None
    check("import src/elicit_reader.py", False, f"{type(e).__name__}: {e}")

BM = None
try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "box_mc", os.path.join(REPO, "experiments", "exp2_output_monitorability", "box_mc.py"))
    BM = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(BM)
except Exception as e:
    BM = None
    check("import box_mc.py", False, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ fixtures
def fake_evoked(model, per_concept=20, n_neutral=6):
    """Evoked-bundle shape: strength-1 concept streams + strength-0 'neutral' streams (the s0
    analog riding in the same bundle, exp3 convention). Includes rejected + too-short streams so
    the selection filters are exercised, and > CAP_PER_CONCEPT per concept so the cap bites."""
    streams, g = [], 0
    for c in ("anger", "ocean"):
        for _ in range(per_concept):
            streams.append(dict(gidx=g, concept=c, strength=1, accepted=True,
                                tokens=[1, 2, 3])); g += 1
        streams.append(dict(gidx=g, concept=c, strength=1, accepted=False,
                            tokens=[1, 2, 3])); g += 1      # rejected: excluded
        streams.append(dict(gidx=g, concept=c, strength=1, accepted=True,
                            tokens=[1])); g += 1            # len<2: excluded
    for _ in range(n_neutral):
        streams.append(dict(gidx=g, concept="neutral", strength=0, accepted=True,
                            tokens=[4, 5])); g += 1
    return dict(model=model, streams=streams)


def fake_capture():
    """exp1-capture shape: injected s60 + s0 streams, variant 'orig', with first_ids."""
    streams, g = [], 0
    for c in ("anger", "ocean"):
        for strength in (60, 60, 0, 0):
            streams.append(dict(gidx=g, concept=c, strength=strength, accepted=True,
                                tokens=[7, 8, 9])); g += 1
    return dict(model="qwen2.5-1.5b", variant="orig", streams=streams,
                first_ids=list(range(12)))


# ------------------------------------------------------------------ P1: own-pool binding
if MR is not None and ER is not None:
    ev3b = fake_evoked("qwen2.5-3b")
    try:
        pools = MR.bind_pools(None, ev3b, ("evoked", "evoked_s0"), "qwen2.5-3b")
        ok_sets = sorted(pools) == ["evoked", "evoked_s0"]
        ev = pools["evoked"]
        cap_ok = (len(ev) == 2 * MR.CAP_PER_CONCEPT
                  and all(s["accepted"] and len(s["tokens"]) >= 2 and s["strength"] == 1
                          and s["concept"] != "neutral" for s in ev)
                  and [s["gidx"] for s in ev] == sorted(s["gidx"] for s in ev))
        check("P1 bind_pools binds evoked from the reader's OWN 3B bundle "
              "(17/concept, ascending gidx, accepted, len>=2)", ok_sets and cap_ok,
              f"sets={sorted(pools) if ok_sets else pools} n={len(ev)}")
        s0 = pools["evoked_s0"]
        check("P1 evoked_s0 = the 3B bundle's own neutral (strength-0) streams, uncapped",
              len(s0) == 6 and all(s["concept"] == "neutral" for s in s0)
              and [s["gidx"] for s in s0] == sorted(s["gidx"] for s in s0),
              f"n={len(s0)}")
    except Exception as e:
        check("P1 bind_pools binds the reader's OWN generator-size bundle", False,
              f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- P2: selection parity
    try:
        check("P2 mc_reader.select_streams IS elicit_reader.select_streams (frozen rules)",
              MR.select_streams is ER.select_streams)
        pools = MR.bind_pools(None, ev3b, ("evoked", "evoked_s0"), "qwen2.5-3b")
        direct = {ss: ER.select_streams(ev3b, ss, MR.CAP_PER_CONCEPT)
                  for ss in ("evoked", "evoked_s0")}
        check("P2 bind_pools output == direct select_streams on the same bundle",
              all(pools[ss] == direct[ss] for ss in direct))
    except Exception as e:
        check("P2 selection parity", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- P3: provenance
    try:
        MR.bind_pools(None, fake_evoked("qwen2.5-1.5b"), ("evoked",), "qwen2.5-3b")
        check("P3 a 1.5B bundle offered as the 3B pool is rejected", False, "no exception")
    except Exception:
        check("P3 a 1.5B bundle offered as the 3B pool is rejected", True)
    try:
        MR.bind_pools(None, fake_evoked("qwen2.5-3b"), ("evoked",), "qwen2.5-7b")
        check("P3 a 3B bundle offered as the 7B pool is rejected", False, "no exception")
    except Exception:
        check("P3 a 3B bundle offered as the 7B pool is rejected", True)

    # -------------------------------------------------------------- P4: injected stays 1.5B
    try:
        MR.bind_pools(fake_capture(), ev3b, ("injected", "evoked"), "qwen2.5-3b")
        check("P4 injected sets + non-default stream source raises (fixed 1.5B exp1 pool)",
              False, "no exception")
    except Exception:
        check("P4 injected sets + non-default stream source raises (fixed 1.5B exp1 pool)", True)
    try:
        MR.bind_pools(None, fake_evoked("qwen2.5-1.5b"), ("injected",), "qwen2.5-1.5b")
        check("P4 injected sets with no capture raises", False, "no exception")
    except Exception:
        check("P4 injected sets with no capture raises", True)

    # -------------------------------------------------------------- P5: default path unchanged
    try:
        cap15, ev15 = fake_capture(), fake_evoked("qwen2.5-1.5b")
        pools = MR.bind_pools(cap15, ev15, MR.STREAM_SETS, "qwen2.5-1.5b")
        direct = {ss: ER.select_streams(cap15 if ss.startswith("injected") else ev15, ss,
                                        MR.CAP_PER_CONCEPT) for ss in MR.STREAM_SETS}
        check("P5 default all-four-sets 1.5B binding identical to the certified run's selection",
              sorted(pools) == sorted(MR.STREAM_SETS)
              and all(pools[ss] == direct[ss] for ss in MR.STREAM_SETS))
    except Exception as e:
        check("P5 default path unchanged", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- G2: first_ids gate wiring
    try:
        mode = MR.first_ids_gate(list(range(12)), list(range(12)), "qwen2.5-1.5b", "qwen2.5-3b")
        check("G2 capture first_ids present: registered gate runs, match passes",
              mode == "capture", f"mode={mode}")
    except Exception as e:
        check("G2 capture first_ids present: registered gate runs, match passes", False,
              f"raised {type(e).__name__}: {e}")
    try:
        MR.first_ids_gate(list(range(12)), list(range(1, 13)), "qwen2.5-1.5b", "qwen2.5-3b")
        check("G2 capture first_ids mismatch is FATAL (gate unchanged)", False, "no exception")
    except Exception:
        check("G2 capture first_ids mismatch is FATAL (gate unchanged)", True)
    try:
        mode = MR.first_ids_gate(list(range(12)), [], "qwen2.5-3b", "qwen2.5-3b")
        check("G2 no capture + diagonal (reader == stream source) passes as 'diagonal'",
              mode == "diagonal", f"mode={mode}")
    except Exception as e:
        check("G2 no capture + diagonal (reader == stream source) passes as 'diagonal'", False,
              f"raised {type(e).__name__}: {e}")
    try:
        MR.first_ids_gate(list(range(12)), [], "qwen2.5-3b", "qwen2.5-7b")
        check("G2 no capture + OFF-diagonal is FATAL (unvalidated cross-scale transfer)",
              False, "no exception")
    except Exception:
        check("G2 no capture + OFF-diagonal is FATAL (unvalidated cross-scale transfer)", True)

    # -------------------------------------------------------------- R1: resume-source guard
    try:
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "qwen2.5-3b_evoked_elicited_direct.pt")
            torch.save(dict(model="qwen2.5-3b", stream_source="qwen2.5-1.5b", records=[]), p)
            try:
                MR.assert_shard_source(p, "qwen2.5-3b")
                check("R1 existing shard from a DIFFERENT pool refuses to resume", False,
                      "no exception")
            except Exception:
                check("R1 existing shard from a DIFFERENT pool refuses to resume", True)
            MR.assert_shard_source(p, "qwen2.5-1.5b")
            check("R1 existing shard from the SAME pool resumes", True)
    except Exception as e:
        check("R1 resume-source guard", False, f"raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------ X1: box_mc own-pool wiring
if BM is not None:
    try:
        f = BM.fetches_for(["qwen2.5-3b", "qwen2.5-7b"], own_pool=True)
        want = [(f"exp3/bundles/qwen2.5-{s}-evoked.pt",
                 os.path.join(BM.REPO, "runs", "_ind", f"qwen2.5-{s}", "data",
                              f"qwen2.5-{s}-evoked.pt")) for s in ("3b", "7b")]
        check("X1 own-pool S0 fetches = each reader's OWN exp3 evoked bundle", list(f) == want,
              f"got {f}")
        check("X1 default S0 fetches unchanged (fixed 1.5B capture + bundle)",
              list(BM.fetches_for(["qwen2.5-3b"], own_pool=False)) == list(BM.FETCHES))
        sh = BM.shards_for("qwen2.5-3b", own_pool=True)
        names = [os.path.basename(s) for s in sh]
        check("X1 own-pool shards = 8 diagonal cells (evoked sets only, all framing x reasoning)",
              len(sh) == 8 and all(("_evoked_" in n or "_evoked_s0_" in n) for n in names)
              and not any("injected" in n for n in names), f"{names}")
        check("X1 default shards unchanged (16 cells)",
              len(BM.shards_for("qwen2.5-3b", own_pool=False)) == 16)
        cmd = BM.reader_cmd("qwen2.5-3b", 24, own_pool=True)
        j = " ".join(cmd)
        check("X1 own-pool reader cmd: --stream-source + --sets evoked,evoked_s0 + own --evoked, "
              "no --capture",
              "--stream-source" in cmd and cmd[cmd.index("--stream-source") + 1] == "qwen2.5-3b"
              and "--sets" in cmd and cmd[cmd.index("--sets") + 1] == "evoked,evoked_s0"
              and "--evoked" in cmd
              and cmd[cmd.index("--evoked") + 1].endswith("qwen2.5-3b-evoked.pt")
              and "--capture" not in cmd, j)
        dflt = BM.reader_cmd("qwen2.5-3b", 24, own_pool=False)
        check("X1 default reader cmd byte-parity with the certified invocation",
              "--capture" in dflt and dflt[dflt.index("--capture") + 1] == BM.CAP15
              and dflt[dflt.index("--evoked") + 1] == BM.EVOKED15
              and "--stream-source" not in dflt and "--sets" not in dflt, " ".join(dflt))
    except Exception as e:
        check("X1 box_mc own-pool wiring", False, f"raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------ X2: mc_reader CLI seam
try:
    src = open(os.path.join(REPO, "src", "mc_reader.py")).read()
    check("X2 mc_reader CLI exposes --stream-source and --sets",
          '"--stream-source"' in src and '"--sets"' in src)
    check("X2 --capture is optional (required only when injected sets ride)",
          '"--capture", required=True' not in src.replace("'", '"'))
except Exception as e:
    check("X2 mc_reader CLI seam", False, f"raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------ S1: certified scoring bodies
# sha256 of inspect.getsource per scoring function, pinned at the certified pre-B6 state
# (commit 1a3d7cd, the mc_reader_prereg.md instrument). B6 is POOL WIRING ONLY: any drift in
# these bodies fails this test and voids the certification.
SCORING_BODY_SHA256 = {
    "latin_square_orderings": "188de3e87b51876fb0fa7b5147fcba19e21b334716c78bd4ce7b6226da31311b",
    "letter_to_concept": "47558a6a766cc210ad666dc0f259df93a08c85e94f24ea3ea01c62bcaa0d1530",
    "_mc_list": "9fd66d63bdb9c356ce287075a765bd9a20c7067b64f8c29feb64e00a9d985e25",
    "mc_message": "8bdf7772cf792fb6e5c5375a97ddce89d939df69fa983ef75616e98723395bfa",
    "build_mc_ids": "aecdf344e662db89ac72590d2fc4c5f2c1868512be4baf2fdf9bfdd364637bf1",
    "append_forced_answer": "74858b4a27623207d7f9d02260d2c7dd95adcd9b8004febf29e868796b0623d0",
    "letter_token_ids": "72e30b9a38ea1fdd2f458cde4594ccc8048f1de010c78f0aa81a1d5e7c93a281",
    "read_letter_logprobs": "f67a2cd736545fbd73eedc3cbc655aec989492fbaaff4c77c2f864efb369537f",
    "letter_mass": "16f089a155e0588143d92e91e26681f743fb2cbc4aed8f6e57eacf5a99d4ffef",
    "is_truncated": "d7eca9017724e8c292a3f58584785b3fb8ec6bff9ac0413066e43788c846c08d",
    "_pad_right": "caf9e8d4cf65bc3adc62e108ef8478e0bc7edd49456649b0394b8df6b8fdc52c",
    "_pad_left": "df5275f6cb79156583e0cf52438531e072ec2e9dbb8291081226f1598a2c5797",
    "forward_last_logprobs": "cafbe97f3961ce217aacf490d65619ad70b9e665c9d7aeaf794e87ac96fe0344",
    "greedy_cot": "c13c6463b725895fc4e99932d311cf546123985d2297a246e0dab67c851b15cd",
    "cot_read_seq": "9675d4c14edcfd38ca55508de48ec9fd5680cabc7b578d8472422e3d57760a29",
    "_topk": "e8475ec975dfccbbbc212fa6757cb5fca61b7f8927f05a1b1e48e1d405fed372",
    "_direct_cell": "0fe0cd9120e889de11355d89d30e11cce94aff96803f10f5e3615be84bcf0725",
    "_cot_cell": "96f19feb8393cf2cfebceebc2f930062062d5ed5ab87a3ef1bebac68225c8773",
    "pad_dev": "ba0cf430389aec350085f041ee752261b4a1bff9cfe1e5c1ab99ba0be786780f",
    "_base_records": "cddf2dc97ffb5910ac69b9494c76685a7bc3ebd234fc2b118d863756800245b5",
}
if MR is not None:
    for name, want in SCORING_BODY_SHA256.items():
        try:
            got = hashlib.sha256(inspect.getsource(getattr(MR, name)).encode()).hexdigest()
            check(f"S1 scoring body byte-identical: {name}", got == want,
                  f"sha256 drifted: {got}")
        except Exception as e:
            check(f"S1 scoring body byte-identical: {name}", False,
                  f"raised {type(e).__name__}: {e}")
    check("S1 frozen constants unchanged (CAP_PER_CONCEPT/COT_CAP/FORCED_ANSWER/SELFCHECK_TOL/"
          "TOPK_SAVE/FRAMINGS/REASONINGS/STREAM_SETS)",
          MR.CAP_PER_CONCEPT == 17 and MR.COT_CAP == 256
          and MR.FORCED_ANSWER == "\nAnswer: (" and MR.SELFCHECK_TOL == 0.02
          and MR.TOPK_SAVE == 16 and MR.FRAMINGS == ("elicited", "passive")
          and MR.REASONINGS == ("direct", "cot")
          and MR.STREAM_SETS == ("injected", "injected_s0", "evoked", "evoked_s0"))
    check("S1 frozen instruction strings unchanged (Amendment 1 verified passive wording)",
          MR.PASSIVE_INSTRUCTION == ("Which concept best matches the previous turn's output? "
                                     "Answer with a single letter.")
          and MR.ELICIT_MC_INSTRUCTION == "Answer with a single letter.")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
