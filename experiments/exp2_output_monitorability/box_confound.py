"""On-box orchestrator for the CONFOUND-CLOSING run (reports/confound_closing_prereg.md, frozen).

One 24GB box, one contiguous session, qwen2.5-1.5b everywhere except the E4 7B trajectory leg.
Stage order (EVERY stage idempotent: checks its outputs/shards and skips; the collectors' own
shard-resume covers mid-stage restarts):

  S0  HF-pull the inputs rsync excludes (*.pt): exp1 1.5B+7B captures (vectors+streams) and the
      exp3 evoked bundles, from ErrareHumanumEst/internal-state-from-gibberish (private -> HF_TOKEN).
  S1  E2-PILOT: collect_induction --pilot --arms sustained_s1..s3 (3 concepts, n=16, 128-tok streams).
  S2  PILOT GATE: state_trajectory --arm pilot:sustained_sK per wording, then OFFLINE gate math
      (acceptance >= 0.5 x neutral clean; z-retention(t>=32 vs t<=8) >= 0.5) -> out/pilot_verdict.json.
      No passer => verdict "no_qualifying_wording" (a reportable finding), S3c is skipped, run continues.
  S3  COLLECTIONS (1.5B): (a) E1 weak-dose sweep INTRO_EFFMAGS=3,5,8,12,20 --inject gen;
      (b) E3 prompt-only INTRO_EFFMAGS=40,60 --inject prompt (own INTRO_RUN_DIR so bundles don't collide);
      (c) E2 full sustained_<winner> + sustained_alt_<winner>; (d) E5 maintained_secret.
  S4  E4 RE-FORWARDS: state_trajectory evoked/s0/injected at 1.5B, then evoked/s0 at 7B (same box).
  S5  exp2 CPU reanalysis (prereg task #11): run_budget over the three _ab gen bundles (GPU idle),
      mirroring box_analyze.py's exp2 path (embeds extracted via loader.load_embed_matrix).

Everything lands under $INTRO_REPORT_DIR (driver sets out/) so labkit pulls it even on failure.
Collectors run as SUBPROCESSES with per-stage INTRO_MODEL / INTRO_RUN_DIR / INTRO_EFFMAGS env --
src/config.py reads env at import, and the measurement code paths stay byte-identical (orchestration
only). Markers: CONFOUND_READY / CONFOUND_DONE / CONFOUND_FATAL; LABKIT_STEP lines per stage
(covert_collect's own steps ride in each stage's INTRO_STEP_BASE window).

Driven by harness/run_confound.py (gated). NEVER run this on the Mac -- it loads models.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "analysis"))       # loader / run_budget (S5)

HF_DATASET = "ErrareHumanumEst/internal-state-from-gibberish"
M15, M7 = "qwen2.5-1.5b", "qwen2.5-7b"
WORDINGS = ("s1", "s2", "s3")
EARLY_CUTS, LATE_CUTS = (2, 4, 8), (32, 64, 127)         # prereg: retention = mean z(t>=32) / mean z(t<=8)
OUT = os.path.abspath(os.environ.get("INTRO_REPORT_DIR") or os.path.join(REPO, "out"))
os.environ["INTRO_REPORT_DIR"] = OUT                     # run_budget (S5) writes here too
# exp1 captures + exp3 evoked bundles, restored to the paths every downstream tool expects.
# qwen2.5-1.5b-gen.pt IS the exp1 1.5B capture (md5-verified identical to runs/qwen2.5-1.5b/data/
# covert_collect.pt); the 7B capture is the NAMEABILITY-MATCHED gen bundle (md5-verified identical).
CAP15 = os.path.join(REPO, "runs", M15, "data", "covert_collect.pt")
CAP7 = os.path.join(REPO, "runs", M7, "data", "covert_collect.pt")
EVOKED15 = os.path.join(REPO, "runs", "_ind", M15, "data", f"{M15}-evoked.pt")
EVOKED7 = os.path.join(REPO, "runs", "_ind", M7, "data", f"{M7}-evoked.pt")
FETCHES = [
    ("qwen2.5-1.5b-gen.pt", CAP15),
    ("qwen2.5-7b-gen-matched.pt", CAP7),
    (f"exp3/bundles/{M15}-evoked.pt", EVOKED15),
    (f"exp3/bundles/{M7}-evoked.pt", EVOKED7),
]
S5_MODELS = ("1.5b", "3b", "7b")                          # run_budget over the _ab gen bundles
EMBED_MODELS = {f"qwen2.5-{m}": f"Qwen/Qwen2.5-{m.upper()}-Instruct" for m in S5_MODELS}
# GPU subprocesses get the collect drivers' thread caps (run_labkit convention); the PARENT keeps the
# driver's caps=1 so S5's joblib outer fan-out owns the cores (run_reanalysis convention).
GPU_THREADS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}


def emit_step(step, **fields):
    """covert_collect.emit_step's format: labkit's watchdog parses LABKIT_STEP {json} into live status.
    Steps must be globally monotonic -> each stage owns a base-1000 window (children get INTRO_STEP_BASE)."""
    print("LABKIT_STEP " + json.dumps({"step": int(step), **fields}), flush=True)


def run_sub(script_rel, args, env_extra, step_base):
    """Run one measurement script as a subprocess with per-stage env. check=True: a child failure
    (incl. its own COLLECT_FATAL paths) raises -> CONFOUND_FATAL. Child stdout shares ours, so its
    per-cell prints keep the stall watchdog alive."""
    cmd = [sys.executable, "-u", os.path.join(REPO, script_rel)] + list(args)
    env = {**os.environ, **GPU_THREADS, "INTRO_STEP_BASE": str(step_base), **env_extra}
    shown = {k: v for k, v in env.items() if k.startswith("INTRO_")}
    print(f"RUN {' '.join(cmd)}  env={shown}", flush=True)
    subprocess.run(cmd, env=env, cwd=REPO, check=True)


# --------------------------------------------------------------------------------- S0: HF pull
def fetch_inputs():
    from huggingface_hub import hf_hub_download
    for fname, dest in FETCHES:
        if os.path.exists(dest):
            print(f"S0 have {os.path.relpath(dest, REPO)}", flush=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(hf_hub_download(HF_DATASET, fname, repo_type="dataset"), dest)
        print(f"S0 fetched {fname} -> {os.path.relpath(dest, REPO)}", flush=True)
    missing = [f for f, d in FETCHES if not os.path.exists(d)]
    if missing:                                           # refuse to 'succeed' on a paid run with no data
        raise RuntimeError(f"S0: inputs missing after HF pull: {missing}")


# ----------------------------------------------------------------------- S1/S2: pilot + gate math
def pilot_bundle(k):
    return os.path.join(OUT, "e2_pilot", "data", f"{M15}-sustained_{k}.pt")


def pilot_shard(k):
    return os.path.join(OUT, "e2_pilot", "trajectory", f"{M15}_pilot-sustained_{k}.pt")


def s1_pilot():
    emit_step(1000, phase="S1_e2_pilot")
    if all(os.path.exists(pilot_bundle(k)) for k in WORDINGS):
        print("S1 SKIP: all pilot bundles exist", flush=True)
        return
    run_sub("experiments/exp3_induction_and_scale/collect_induction.py",
            ["--models", M15, "--pilot", "--arms"] + [f"sustained_{k}" for k in WORDINGS],
            {"INTRO_MODEL": M15, "INTRO_RUN_DIR": os.path.join(OUT, "e2_pilot")}, 1000)


def _diff(proj_t, c):
    """Own-concept projection minus mean other-concept projection at one cut."""
    others = [v for cc, v in proj_t.items() if cc != c]
    return proj_t[c] - (sum(others) / len(others))


def wording_metrics(k):
    """Offline gate math (prereg E2 pilot) from the pilot bundle + trajectory shard. z(t) = own-minus-other
    projection standardized per (concept, cut) against the NEUTRAL (strength-0) pool of the same wording."""
    import torch
    b = torch.load(pilot_bundle(k), map_location="cpu", weights_only=False)
    ind = [s for s in b["streams"] if int(s["strength"]) == 1]
    neu = [s for s in b["streams"] if int(s["strength"]) == 0]
    acc = (sum(int(s["accepted"]) for s in ind) / len(ind)) if ind else 0.0
    neutral_clean = (sum(int(s["accepted"]) for s in neu) / len(neu)) if neu else 0.0

    sh = torch.load(pilot_shard(k), map_location="cpu", weights_only=False)
    recs = sh["records"]
    concepts = sorted({r["concept"] for r in recs if r.get("concept") and int(r["strength"]) == 1})
    # neutral standardization pool: mu/sd of the own-minus-other diff per (concept, cut)
    pool = {}
    for c in concepts:
        for r in recs:
            if int(r["strength"]) != 0:
                continue
            for t, pt in r["proj"].items():
                pool.setdefault((c, int(t)), []).append(_diff(pt, c))
    import numpy as np
    stats = {key: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
             for key, v in pool.items()}
    zs = {"early": [], "late": []}
    for r in recs:
        if int(r["strength"]) != 1 or not r.get("accepted", True):
            continue
        c = r["concept"]
        for t, pt in r["proj"].items():
            t = int(t)
            mu, sd = stats.get((c, t), (None, None))
            if mu is None or not sd or sd <= 0:
                continue
            z = (_diff(pt, c) - mu) / sd
            if t in EARLY_CUTS:
                zs["early"].append(z)
            elif t in LATE_CUTS:
                zs["late"].append(z)
    early = float(np.mean(zs["early"])) if zs["early"] else None
    late = float(np.mean(zs["late"])) if zs["late"] else None
    retention = (late / early) if (early is not None and late is not None and early > 0) else None
    accept_ok = bool(neutral_clean > 0 and acc >= 0.5 * neutral_clean)
    retain_ok = bool(retention is not None and retention >= 0.5)
    return dict(acceptance=round(acc, 3), neutral_clean=round(neutral_clean, 3), accept_gate=accept_ok,
                early_z=None if early is None else round(early, 3),
                late_z=None if late is None else round(late, 3),
                retention=None if retention is None else round(retention, 3), retention_gate=retain_ok,
                n_early=len(zs["early"]), n_late=len(zs["late"]), passed=bool(accept_ok and retain_ok))


def s2_pilot_gate():
    emit_step(2000, phase="S2_pilot_gate")
    vpath = os.path.join(OUT, "pilot_verdict.json")
    if os.path.exists(vpath):
        v = json.load(open(vpath))
        print(f"S2 SKIP: verdict exists -> {v['verdict']} winner={v.get('winner')}", flush=True)
        return v
    for i, k in enumerate(WORDINGS):                      # E4 measurement per wording (TRAJ_SKIP resumes)
        if not os.path.exists(pilot_bundle(k)):
            raise RuntimeError(f"S2: pilot bundle missing for {k}: {pilot_bundle(k)}")
        run_sub("src/state_trajectory.py",
                ["--arm", f"pilot:sustained_{k}", "--bundle", pilot_bundle(k), "--vectors-from", CAP15],
                {"INTRO_MODEL": M15, "INTRO_RUN_DIR": os.path.join(OUT, "e2_pilot")}, 2000 + 10 * i)
    wm = {k: wording_metrics(k) for k in WORDINGS}
    passers = [k for k in WORDINGS if wm[k]["passed"]]
    # best passer = strongest sustained state (highest late-window z), acceptance breaks ties
    winner = max(passers, key=lambda k: (wm[k]["late_z"], wm[k]["acceptance"])) if passers else None
    v = dict(verdict=("qualified" if winner else "no_qualifying_wording"), winner=winner, wordings=wm,
             gates=dict(acceptance=">= 0.5 x neutral clean fraction",
                        retention=">= 0.5 of early (mean z t<=8) at late cuts (t>=32)"),
             selection="max late_z among passers, tie-break acceptance")
    tmp = vpath + ".tmp"
    json.dump(v, open(tmp, "w"), indent=2)
    os.replace(tmp, vpath)                                # atomic
    print(f"S2 verdict={v['verdict']} winner={winner} {json.dumps(wm)}", flush=True)
    if winner is None:
        print("S2: NO qualifying wording -- prereg says this is itself a reportable finding; "
              "S3c (E2 full) will be skipped and the run continues.", flush=True)
    return v


# --------------------------------------------------------------------------------- S3: collections
def s3_collections(winner):
    # (a) E1 weak-dose sweep (gen-only). covert_collect's shard loop resumes mid-cell-grid on restart.
    emit_step(3000, phase="S3a_e1_dose")
    if os.path.exists(os.path.join(OUT, "e1_dose", "data", "covert_collect.pt")):
        print("S3a SKIP: e1_dose capture exists", flush=True)
    else:
        run_sub("src/covert_collect.py", ["--no-calibrate", "--inject", "gen"],
                {"INTRO_MODEL": M15, "INTRO_RUN_DIR": os.path.join(OUT, "e1_dose"),
                 "INTRO_EFFMAGS": "3,5,8,12,20"}, 3000)
    # (b) E3 prompt-only, own INTRO_RUN_DIR (config routes ALL output under it, so bundles can't collide).
    emit_step(4000, phase="S3b_e3_prompt")
    if os.path.exists(os.path.join(OUT, "e3_prompt", "data", "covert_collect.pt")):
        print("S3b SKIP: e3_prompt capture exists", flush=True)
    else:
        run_sub("src/covert_collect.py", ["--no-calibrate", "--inject", "prompt"],
                {"INTRO_MODEL": M15, "INTRO_RUN_DIR": os.path.join(OUT, "e3_prompt"),
                 "INTRO_EFFMAGS": "40,60"}, 4000)
    # (c) E2 full: only the qualified winner + its paraphrase (prereg gate). collect_induction skips
    # per-arm when the bundle exists and per-cell via shards otherwise.
    emit_step(5000, phase="S3c_e2_full", winner=winner)
    if winner is None:
        print("S3c SKIP: no qualifying wording (see pilot_verdict.json)", flush=True)
    else:
        arms = [f"sustained_{winner}", f"sustained_alt_{winner}"]
        if all(os.path.exists(os.path.join(OUT, "e2_full", "data", f"{M15}-{a}.pt")) for a in arms):
            print("S3c SKIP: e2_full bundles exist", flush=True)
        else:
            run_sub("experiments/exp3_induction_and_scale/collect_induction.py",
                    ["--models", M15, "--arms"] + arms,
                    {"INTRO_MODEL": M15, "INTRO_RUN_DIR": os.path.join(OUT, "e2_full")}, 5000)
    # (d) E5 maintained secret.
    emit_step(6000, phase="S3d_e5_secret")
    if os.path.exists(os.path.join(OUT, "e5_secret", "data", f"{M15}-maintained_secret.pt")):
        print("S3d SKIP: e5_secret bundle exists", flush=True)
    else:
        run_sub("experiments/exp3_induction_and_scale/collect_induction.py",
                ["--models", M15, "--arms", "maintained_secret"],
                {"INTRO_MODEL": M15, "INTRO_RUN_DIR": os.path.join(OUT, "e5_secret")}, 6000)


# --------------------------------------------------------------------------------- S4: E4 re-forwards
def s4_trajectories():
    traj_dir = os.path.join(OUT, "e4_traj")
    # (slug, arm, bundle, vectors_from) -- 1.5B legs first, then the single 7B load (24GB fits 7B bf16)
    legs = [(M15, "evoked", EVOKED15, CAP15),
            (M15, "s0", CAP15, CAP15),
            (M15, "injected", CAP15, CAP15),              # prereg reference: max strength in the capture (s60)
            (M7, "evoked", EVOKED7, CAP7),
            (M7, "s0", CAP7, CAP7)]
    for i, (slug, arm, bundle, vec) in enumerate(legs):
        emit_step(7000 + 10 * i, phase="S4_e4_traj", model=slug, arm=arm)
        # state_trajectory writes $INTRO_RUN_DIR/trajectory/<slug>_<arm>.pt -- match that path exactly
        shard = os.path.join(traj_dir, "trajectory", f"{slug}_{arm}.pt")
        if os.path.exists(shard):                         # state_trajectory would TRAJ_SKIP too; save the spawn
            print(f"S4 SKIP: {os.path.relpath(shard, OUT)} exists", flush=True)
            continue
        run_sub("src/state_trajectory.py",
                ["--arm", arm, "--bundle", bundle, "--vectors-from", vec],
                {"INTRO_MODEL": slug, "INTRO_RUN_DIR": traj_dir}, 7000 + 10 * i)


# --------------------------------------------------------------------------------- S5: exp2 reanalysis
def s5_exp2_reanalysis():
    emit_step(8000, phase="S5_exp2_reanalysis")
    outpath = os.path.join(OUT, "budget_results.json")
    done = set()
    if os.path.exists(outpath):
        done = {r["model"] for r in json.load(open(outpath))}
    todo = [m for m in S5_MODELS if f"qwen2.5-{m}" not in done]
    if not todo:
        print("S5 SKIP: budget_results.json complete", flush=True)
        return
    from huggingface_hub import hf_hub_download            # gen bundles -> runs/_ab (rsync excludes *.pt)
    bundles = []
    for m in S5_MODELS:
        dest = os.path.join(REPO, "runs", "_ab", f"qwen2.5-{m}-gen.pt")
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(hf_hub_download(HF_DATASET, f"qwen2.5-{m}-gen.pt", repo_type="dataset"), dest)
            print(f"S5 fetched qwen2.5-{m}-gen.pt", flush=True)
        bundles.append(dest)
    import numpy as np                                     # embeds (box_analyze pattern; partial safetensors)
    from loader import load_embed_matrix
    emb_dir = os.path.join(REPO, "artifacts")
    os.makedirs(emb_dir, exist_ok=True)
    for slug, hf_id in EMBED_MODELS.items():
        p = os.path.join(emb_dir, f"{slug}_embed.npy")
        if not os.path.exists(p):
            print(f"S5 extracting embed matrix {slug} <- {hf_id}", flush=True)
            np.save(p, load_embed_matrix(hf_id))
    os.environ["INTRO_EMBED_DIR"] = emb_dir
    sys.argv = ["run_budget"] + bundles
    import run_budget
    run_budget.main()                                      # per-bundle atomic checkpoint into $INTRO_REPORT_DIR


def main():
    os.makedirs(OUT, exist_ok=True)
    emit_step(0, phase="S0_fetch")
    fetch_inputs()
    print("CONFOUND_READY", flush=True)

    print("\n########## S1 E2 wording pilot ##########", flush=True)
    s1_pilot()
    print("\n########## S2 pilot gate ##########", flush=True)
    verdict = s2_pilot_gate()
    print("\n########## S3 collections (E1/E3/E2/E5) ##########", flush=True)
    s3_collections(verdict.get("winner"))
    print("\n########## S4 E4 state trajectories ##########", flush=True)
    s4_trajectories()
    print("\n########## S5 exp2 CPU reanalysis ##########", flush=True)
    s5_exp2_reanalysis()

    # never report DONE with nothing to pull (paid run): the pilot verdict + at least the dose capture
    for req in (os.path.join(OUT, "pilot_verdict.json"),
                os.path.join(OUT, "e1_dose", "data", "covert_collect.pt")):
        if not os.path.exists(req):
            raise RuntimeError(f"missing required output {req}")
    emit_step(9000, phase="confound_done")
    print(f"outputs OK in {OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
        print("CONFOUND_DONE", flush=True)
    except Exception:
        print("CONFOUND_FATAL", flush=True)
        raise
