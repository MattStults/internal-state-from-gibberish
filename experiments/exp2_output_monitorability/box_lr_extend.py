"""On-box orchestrator for the LR scale-grid EXTENSION run (prereg:
reports/lr_scale_extend_prereg.md -- FROZEN 2026-07-13 + Amendments 1-2 2026-07-14. The header
Decisions block + the Amendments GOVERN: effective scope is **14B-only** -- 32B is fully
descoped and this box must be UNABLE to generate or score it -- evoked/evoked_alt IN at 14B,
the 70B cross-family observer rider IN, Part B injection runs UNTRIMMABLE, the Amendment-2
(2a) dose curve UNTRIMMABLE with them, the (2b) expressed-injection cell WITHDRAWN pre-data
per Amendment 3 -- built code stays in the tree unscheduled).

ONE contiguous 48GB box (RTX 6000 Ada / L40S / A6000 class), phases in run order:
  S0   HF-pull the input bundles rsync excludes (*.pt): the existing secret_word +
       secret_sustain bundles at 1.5/3/7B (launch checklist L2), the 1.5B evoked/evoked_alt
       anchor pools, the three exp1 injected captures (<slug>-gen.pt) -- plus the Amendment-1
       70B streams JSON (rsync carries it with the workdir; the fetch entry is the manifest +
       the fallback when the local copy is missing).
  S1   ANCHOR (smoke AND full): the 1.5B reader re-measures the D2 eos-free diagonal anchor
       (evoked/evoked_alt at 1.5B) + the 1.5B secret_word regression cell, through the SAME
       src/lr_grid_extend.py entrypoint every other cell uses. The offline scorer asserts
       |anchor - 0.16648| <= 0.01 before any new cell is interpreted.
  S2   GENERATION at 14B: exp3 collect_induction.run_model (config-level reuse -- same
       pipeline, word-free filter, acceptance gates, s0 neutral riding), arms secret_word +
       secret_sustain + (evoked/evoked_alt unless trimmed -- prereg Q2 RESOLVED: IN), ONE
       disclosed diff: gen_batch capped (GEN_BATCH_CAP -- the same mechanism run_model already
       applies at 7B/8B; batch size does not change per-stream sampling). Self-subprocess
       (--gen-slug) for clean GPU teardown.
  S3   LR GRID: src/lr_grid_extend.py once per reader (subprocess per model). The 14B reader
       scores its own diagonal + ALL the old pools (hot-box reuse: one model load scores every
       cell of that reader). 7B/3B/1.5B readers score the 14B pools; 3B/7B additionally collect
       the 7a run-(1) injected self-diagonal cells (N/A/B) never collected by the grid run.
  S3b  the 7B run-(1) s124-primary pass against a strength-filtered capture copy under a
       SEPARATE out dir (out/s124) so shard names never collide.
  S4   INJECTION run (2): src/inject_tf_lr.py per slug (1.5B, 3B, 7B) -- teacher-forcing with
       the concept vector ACTIVELY re-injected vs neutral; 7B runs s124 (primary) + s140
       (descriptive). Part B (S3's injected cells + S3b + S4) is UNTRIMMABLE (Amendment 1):
       asserted in main(), not a comment -- it runs BEFORE the rider so a deadman kill eats the
       trimmable tail first.
  S4a  Amendment-2 (2a) DOSE CURVE at 1.5B (Part B, UNTRIMMABLE): the run-(2) pass over the e1
       low-dose capture (runs/confound-e1, s3/5/8/12/20) + the main capture's s40, one model
       load per capture (inject_tf_lr --strengths), --no-s0 (disclosed centering trim: the s60
       shard carries the main capture's centering). A dose missing stored vectors degrades to
       not-scored (dose_plan disclosure), NEVER regenerated.
  S4b  Amendment-2 (2b) EXPRESSED-INJECTION cell at 1.5B (WITHDRAWN per Amendment 3 --
       run_plan hard-pins expressed_2b False, this phase never schedules):
       src/expressed_collect.py generates {s20, s60} x 12 concepts x ~24 accepted under
       vector-injection + the verbatim sustain-s1 "this feeling" suffix over the word-free
       system (stored primitives: s20 from e1, s60 from main), then inject_tf_lr self-reads
       the bundle (streamset 'expressed'). Public char/dist scoring is OFFLINE ONLY.
  S5   the Amendment-1 70B RIDER: src/lr_rider.py per reader (qwen2.5-{1.5b,3b,7b,14b})
       teacher-forces the 810 Llama-70B TEXT streams under matched + arm-neutral + 11
       mismatched serverless_72b.build_system_prompt contexts in the READER'S chat template.
       Confirmatory arms (secret_sustain, evoked) run before the descriptive secret_word
       within each reader load.

Trim ladder (Amendment 3, 2026-07-14; supersedes Amendment 2's -- the 2b expressed cell is
WITHDRAWN pre-data and is not a trim level):
  --trim 1  drop the 70B-rider secret_word (descriptive) cells
  --trim 2  ALSO drop the evoked/evoked_alt arms (generation + scoring)
  --trim 3  ALSO drop the 70B-rider confirmatory cells (max 3)
  NEVER the 14B secret diagonals/off-diagonals/anchor, NEVER Part B, NEVER the (2a) dose curve
  (2a scores existing streams and joins Part B as the claim's evidence) -- enforced by
  run_plan/assert_part_b_untrimmable.

--smoke = the registered D-slice + rider slice: S0 + S1 anchor + a tiny 14B secret_word
generation + scoring that tiny bundle (times the 14B path) + a 2-stream run-(2) slice at 1.5B
(verifies the hook path) + a 2-stream e1 dose slice at s20 (verifies the 2a path) + a tiny
expressed gen + 2-stream self-read (verifies the 2b path) + a 4-stream/1-arm rider slice at
1.5B (verifies the Amendment-1 path; disclosed additions to the frozen L4 slice). The driver
projects full-run spend from this run's LABKIT_STEP timings.

Markers LRX_READY / LRX_DONE / LRX_FATAL (no existing box marker is a substring of these or vice
versa -- checked in tests/test_lr_extend.py). Driven by harness/run_lr_extend.py (gated). NEVER
run this on the Mac -- it loads models. Module import is stdlib-only (driver/tests import it).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "src")
EXP3 = os.path.join(REPO, "experiments", "exp3_induction_and_scale")

HF_DATASET = "ErrareHumanumEst/internal-state-from-gibberish"
OUT = os.path.abspath(os.environ.get("INTRO_REPORT_DIR") or os.path.join(REPO, "out"))

OLD_GENS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")     # certified-grid generators
NEW_GENS = ("qwen2.5-14b",)          # this run's generator/reader -- 14B ONLY (32B descoped:
#                                      no phase below may reference a 32b slug, guarded by test)
SECRET_ARMS = ("secret_word", "secret_sustain")
EVOKED_ARMS = ("evoked", "evoked_alt")                      # IN by default (Q2); --trim>=2 drops
# ctx-set codes: MUST stay in lockstep with lr_grid.SECRET_CTX (stdlib-only module: parity is
# pinned by tests/test_lr_extend.py E5, the box_lr_grid X7 pattern).
SECRET_CTX = {"secret_word": "SW", "secret_sustain": "SS"}
ANCHOR_READER = "qwen2.5-1.5b"
SMALL_READERS = ("qwen2.5-7b", "qwen2.5-3b", "qwen2.5-1.5b")   # old sizes reading the new pool
READERS = NEW_GENS + SMALL_READERS                             # every grid reader this box runs
# 7a run (1): the injected self-diagonal cells the grid box wired but never collected.
INJECTED_READERS = ("qwen2.5-3b", "qwen2.5-7b")
# 7a run (2) slugs + injection levels: primary first (7B primary = s124, the criterion-passing
# dose scale14b Amendment 1 pinned; smax s140 failed the capability gate -> descriptive).
ITF_LEVELS = {"qwen2.5-1.5b": (60,), "qwen2.5-3b": (60,), "qwen2.5-7b": (124, 140)}
ITF_PRIMARY = {"qwen2.5-1.5b": 60, "qwen2.5-3b": 60, "qwen2.5-7b": 124}
S124_SUBDIR = "s124"          # separate INTRO_RUN_DIR for the 7B run-(1) s124-primary pass
# ---- Amendment-2 (2026-07-14): (2a) run-(2) dose curve + (2b) expressed-injection cell, both
# 1.5B-only (the settled 7a small-scale-only scope). 2a scores EXISTING streams (e1 low-dose
# capture + the main capture's s40; s60 already rides ITF_LEVELS) -- Part B untrimmable. 2b is
# new generation, trimmable at --trim>=2.
DOSE_SLUG = "qwen2.5-1.5b"
DOSE_LEVELS_E1 = (3, 5, 8, 12, 20)     # runs/confound-e1 capture (vectors verified present)
DOSE_LEVELS_MAIN = (40,)               # the main capture's s40 (s60 = the existing S4 cell)
DOSE_SMOKE_LEVELS = (20,)              # 2-stream e1 slice: proves the multi-dose path
EXPRESSED_ARM = "expressed"
EXPRESSED_DOSES = (20, 60)
EXPRESSED_SMOKE_DOSES = (20,)
# S2 sizing: the ONE disclosed diff vs the exp3 real-run cfg -- gen_batch capped for the 48GB
# card (14B weights 29.5GB bf16 + KV at batch 32 would breach it; run_model itself caps 7B/8B).
GEN_BATCH_CAP = {"qwen2.5-14b": 16}
GPU_THREADS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}

# ---- Amendment-1 rider constants (stdlib mirror; parity with src/lr_rider.py is pinned by
# tests/test_lr_extend.py -- this module must stay importable without torch) --------------------
RIDER_GEN = "llama70b"
RIDER_ARMS = ("secret_sustain", "evoked", "secret_word")   # descriptive arm LAST
RIDER_CONFIRMATORY = ("secret_sustain", "evoked")
RIDER_DESCRIPTIVE = ("secret_word",)
RIDER_CTX = "R"
RIDER_READERS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b")
RIDER_SMOKE_READER = "qwen2.5-1.5b"
RIDER_SMOKE_ARM = "secret_sustain"
RIDER_SMOKE_LIMIT = 4

# ---- Trim order (Amendment 3: 2b WITHDRAWN pre-data; supersedes Amendment 2's order) ----------
TRIM_STEPS = ("rider_descriptive", "evoked", "rider_confirmatory")
UNTRIMMABLE = ("part_b", "dose_2a", "secret_cells", "anchor")


def run_plan(trim=0):
    """Trim order per Amendment 3 (2026-07-14): the 2b expressed cell is WITHDRAWN pre-data —
    expressed_2b is hard-pinned False (unscheduled; the built code stays in the tree unused).
    Remaining ladder (supersedes Amendment 2's, which superseded Amendment 1's): trim N drops
    the first N of TRIM_STEPS. Part B, the 2a dose curve, the 14B secret diagonals/
    off-diagonals and the anchor are structurally untrimmable: they are not plan keys a trim
    level can reach, and part_b/dose_2a are hard-pinned True + asserted."""
    trim = int(trim)
    if not 0 <= trim <= len(TRIM_STEPS):
        raise ValueError(f"trim {trim} out of range 0..{len(TRIM_STEPS)} "
                         f"(order: {' -> '.join(TRIM_STEPS)}; Part B / the 2a dose curve / "
                         "secret cells / anchor can NEVER be trimmed)")
    dropped = set(TRIM_STEPS[:trim])
    return dict(rider_descriptive="rider_descriptive" not in dropped,
                expressed_2b=False,   # Amendment 3: withdrawn pre-data, never scheduled
                evoked="evoked" not in dropped,
                rider_confirmatory="rider_confirmatory" not in dropped,
                part_b=True, dose_2a=True, secret_cells=True, anchor=True)


def assert_part_b_untrimmable(plan):
    """Amendments 1+2: no claim of the form 'injection does not show up in LR' may be made
    without runs (1) and (2) having executed, and the 2a dose curve (scoring on EXISTING
    streams) inherits that untrimmability. An assertion, not a comment."""
    for key in UNTRIMMABLE:
        if not plan.get(key, False):
            raise RuntimeError(f"trim plan drops {key!r} -- Part B / the 2a dose curve / the "
                               "secret cells / the anchor are UNTRIMMABLE (Amendments 1+2, "
                               "2026-07-14)")
    return True


def rider_arms_for(plan):
    """Rider arms surviving the trim plan, confirmatory first (descriptive is trimmed first,
    so it runs last)."""
    arms = []
    if plan["rider_confirmatory"]:
        arms += [a for a in RIDER_ARMS if a in RIDER_CONFIRMATORY]
    if plan["rider_descriptive"]:
        arms += [a for a in RIDER_ARMS if a in RIDER_DESCRIPTIVE]
    return tuple(arms)


def _load_box_lr_grid():
    """Sibling box_lr_grid (stdlib-only): deps_for + the B12 persona/parity gates are ITS
    function objects -- certified reuse, nothing duplicated."""
    if "box_lr_grid" in sys.modules:
        return sys.modules["box_lr_grid"]
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "box_lr_grid", os.path.join(HERE, "box_lr_grid.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["box_lr_grid"] = mod
    spec.loader.exec_module(mod)
    return mod


def deps_for(slugs):
    return _load_box_lr_grid().deps_for(list(slugs))


# ---- S0 inputs ------------------------------------------------------------------------------
def secret_word_path(slug):
    return os.path.join(REPO, "runs", "_ind", slug, "data", f"{slug}-secret_word.pt")


def secret_sustain_path(slug):
    """Local-first resolution (box_lr_72b.observe_bundle_path pattern): the grid-box _ind mirror
    holds the only local copies; the HF fetch destination is the top-level _ind path."""
    for d in (os.path.join(REPO, "runs", "lr_grid_box", "_ind"),
              os.path.join(REPO, "runs", "_ind")):
        p = os.path.join(d, slug, "data", f"{slug}-secret_sustain.pt")
        if os.path.exists(p):
            return p
    return os.path.join(REPO, "runs", "_ind", slug, "data", f"{slug}-secret_sustain.pt")


def injected_capture_path(slug):
    return os.path.join(REPO, "runs", slug, "data", "covert_collect.pt")


def e1_capture_path():
    """The Amendment-2 (2a) low-dose input: the confound-run E1 capture (1.5B, strengths
    0/3/5/8/12/20, stored inject_vectors + per-dose alphas verified present)."""
    return os.path.join(REPO, "runs", "confound-e1", "data", "covert_collect.pt")


def rider_streams_path():
    """The Amendment-1 input: the 810 Llama-70B TEXT streams. NOT rsync-excluded (JSON), so the
    workdir rsync carries it; the fetches() entry is the manifest + HF fallback."""
    return os.path.join(REPO, "runs", "llama70b_scout", "streams_llama70b.json")


def new_bundle_path(slug, arm, out=None):
    """A bundle GENERATED by this box's S2 (under OUT/_ind so labkit pulls it home); a repo copy
    wins if one ever exists (resume across boxes)."""
    repo_p = os.path.join(REPO, "runs", "_ind", slug, "data", f"{slug}-{arm}.pt")
    if os.path.exists(repo_p):
        return repo_p
    return os.path.join(out or OUT, "_ind", slug, "data", f"{slug}-{arm}.pt")


ANCHOR_POOLS = {
    "evoked": os.path.join(REPO, "runs", "_ind", "qwen2.5-1.5b", "data",
                           "qwen2.5-1.5b-evoked.pt"),
    "evoked_alt": os.path.join(REPO, "runs", "_ind", "qwen2.5-1.5b", "data",
                               "qwen2.5-1.5b-evoked_alt.pt"),
}


def fetches(smoke=False):
    """[(hf_name, local_dest)] -- the box-input manifest. *.pt entries are rsync-excluded so
    they live on HF ONLY (secret_sustain: the pre-launch L2 upload; the driver preflights the
    exact hf_hub_download call class before create -- the Amendment-4 lesson). The 70B streams
    JSON rides the workdir rsync; its entry keeps the manifest complete + covers a missing
    local copy."""
    f = [("exp3/bundles/qwen2.5-1.5b-evoked.pt", ANCHOR_POOLS["evoked"]),
         ("exp3/bundles/qwen2.5-1.5b-evoked_alt.pt", ANCHOR_POOLS["evoked_alt"])]
    f += [(f"exp3/bundles/{m}-secret_word.pt", secret_word_path(m)) for m in OLD_GENS]
    f += [(f"exp3/bundles/{m}-secret_sustain.pt", secret_sustain_path(m)) for m in OLD_GENS]
    f += [(f"{m}-gen.pt", injected_capture_path(m)) for m in OLD_GENS]
    f += [("confound-e1-gen.pt", e1_capture_path())]      # Amendment 2 (2a): the low-dose capture
    f += [("llama70b/streams_llama70b.json", rider_streams_path())]
    return f


def fetch_inputs(smoke=False):
    from huggingface_hub import hf_hub_download
    for fname, dest in fetches(smoke):
        if os.path.exists(dest):
            print(f"S0 have {os.path.relpath(dest, REPO)}", flush=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(hf_hub_download(HF_DATASET, fname, repo_type="dataset"), dest)
        print(f"S0 fetched {fname} -> {os.path.relpath(dest, REPO)}", flush=True)
    missing = [f for f, d in fetches(smoke) if not os.path.exists(d)]
    if missing:
        raise RuntimeError(f"S0: inputs missing after HF pull: {missing}")
    if not smoke:
        fetch_resume()
    assert_capture_levels()


RESUME_PREFIX = "lr_extend_resume/"


def fetch_resume(lister=None, downloader=None):
    """Cross-box resume (2026-07-14 incident: the S4 fatal stranded 112 completed shards —
    759MB of S1-S3b work — because rsync excludes *.pt, so a relaunched box would have
    REGENERATED the 14B pools and discarded them). Full runs restore every file under the HF
    ``lr_extend_resume/`` prefix into OUT before any stage runs; the existing per-stage
    skip-on-existing logic then resumes exactly where the dead box stopped, against the SAME
    generated pools (no cross-generation shard mixing — the restored tree includes its own gen
    bundles). SMOKE runs never restore (they must time real work for the projection). Stale
    resume state is cleared by deleting the HF prefix. lister/downloader injectable for tests.
    """
    if lister is None or downloader is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        lister = lister or (lambda: [f for f in api.list_repo_files(
            HF_DATASET, repo_type="dataset") if f.startswith(RESUME_PREFIX)])
        downloader = downloader or (
            lambda fname: hf_hub_download(HF_DATASET, fname, repo_type="dataset"))
    restored = skipped = 0
    for fname in lister():
        rel = fname[len(RESUME_PREFIX):]
        dest = os.path.join(OUT, rel)
        if os.path.exists(dest):
            skipped += 1
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(downloader(fname), dest)
        restored += 1
    print(f"S0 resume: restored {restored} prior output files from HF "
          f"({skipped} already present)", flush=True)
    return restored


def assert_capture_levels(load=None):
    """S0 fail-fast (2026-07-14 incident: the HF 7B capture was a stale s62/s93 vintage; the
    mismatch surfaced at S4, 2.8 GPU-hours in). Every injected capture's stored strength levels
    must cover what the run plan will request (ITF_LEVELS per slug; the e1 dose levels) BEFORE
    any model loads. torch import is deferred (this module must stay stdlib-importable);
    ``load`` is injectable for tests."""
    if load is None:
        import torch
        load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    problems = []
    for slug, levels in ITF_LEVELS.items():
        lv = set(int(x) for x in load(injected_capture_path(slug)).get("strengths", []))
        want = set(int(x) for x in levels)
        if not want <= lv:
            problems.append(f"{slug}: capture has {sorted(lv)}, run needs {sorted(want)}")
    e1 = set(int(x) for x in load(e1_capture_path()).get("strengths", []))
    e1_want = set(int(x) for x in DOSE_LEVELS_E1)
    if not e1_want <= e1:
        # NOT fatal: the registered 2a semantics are degrade-and-disclose (dose_plan),
        # never regenerate. Print loudly so the disclosure is in the log from minute one.
        print(f"S0 e1 dose coverage PARTIAL: capture has {sorted(e1)}, plan wants "
              f"{sorted(e1_want)} -- missing doses will ride dose_plan as not-scored",
              flush=True)
    if problems:
        raise RuntimeError("S0 capture-level assert FAILED (stale/wrong-vintage input): "
                           + "; ".join(problems))
    print("S0 capture levels OK: "
          + "; ".join(f"{s}⊇{list(l)}" for s, l in ITF_LEVELS.items()), flush=True)


# ---- S1 anchor -------------------------------------------------------------------------------
def anchor_bundle_specs():
    """The instrument-certification cells: the 1.5B reader on BOTH 1.5B evoked pools (the D2
    eos-free diagonal anchor = mean of evoked x B / evoked_alt x A) + the 1.5B secret_word pool
    (the 0.163 regression check)."""
    m = ANCHOR_READER
    return [f"{m}:evoked:{ANCHOR_POOLS['evoked']}",
            f"{m}:evoked_alt:{ANCHOR_POOLS['evoked_alt']}",
            f"{m}:secret_word:{secret_word_path(m)}"]


def anchor_shards(out=None):
    base = os.path.join(out or OUT, "lr_grid")
    m = ANCHOR_READER
    names = [f"{m}__{m}__{ss}_{cs}.pt" for ss in ("evoked", "evoked_alt")
             for cs in ("N", "A", "B")]
    names += [f"{m}__{m}__secret_word_{cs}.pt" for cs in ("N", "SW")]
    return [os.path.join(base, n) for n in names]


# ---- S2 generation ---------------------------------------------------------------------------
def gen_arms(evoked=True):
    return SECRET_ARMS + (EVOKED_ARMS if evoked else ())


def gen_cmd(slug, evoked=True, smoke=False, out=None):
    """Self-subprocess: this file with --gen-slug (clean GPU teardown per size). INTRO_MODEL
    must be the slug (config registry membership -- 14B is registered) and INTRO_RUN_DIR
    the per-slug OUT/_ind dir (bundle comes home in the pull)."""
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--gen-slug", slug]
    if evoked:
        cmd.append("--evoked")
    if smoke:
        cmd.append("--smoke")
    env = {"INTRO_MODEL": slug,
           "INTRO_RUN_DIR": os.path.join(out or OUT, "_ind", slug)}
    return cmd, env


def run_generation(slug, evoked=True, smoke=False):
    """The --gen-slug body (runs INSIDE the subprocess): collect_induction.run_model with the
    real-run cfg and the ONE disclosed diff -- gen_batch capped (GEN_BATCH_CAP). The B12
    persona byte-identity + env/pipeline parity gates run first ($0-fail, box_lr_grid's own
    function objects). Arms: secret_word + secret_sustain (+ evoked arms unless trimmed).
    min_per_class=0 (registered pin: acceptance reported, offline n-gate voids thin cells, a
    thin pool never FATALs the box)."""
    BLG = _load_box_lr_grid()
    BLG.assert_personas()
    BLG.alt_parity_check(slug)
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    if EXP3 not in sys.path:
        sys.path.insert(0, EXP3)
    import config as C
    import collect_induction as CI
    g = dict(CI.cfg(smoke))
    cap = GEN_BATCH_CAP.get(slug)
    if cap:
        g["gen_batch"] = min(g["gen_batch"], cap)
        print(f"S2 {slug}: gen_batch capped to {g['gen_batch']} (disclosed diff, prereg S2; "
              "sampling per-stream unchanged)", flush=True)
    arms = list(gen_arms(evoked))
    if smoke:
        arms = ["secret_word"]                    # the registered smoke slice: one arm, tiny cfg
    # CRIT 2a: emit a wall-clock step the moment the model is ready (weights downloaded +
    # loaded) so the driver's projection bounds the gen slice at it -- the one-time 14B
    # download must never be multiplied by the generation scale factors.
    CI.run_model(C.MODELS[slug]["hf_id"], slug, arms, g, smoke, min_per_class=0,
                 on_model_ready=lambda: emit_step(510, phase="S2_model_ready", slug=slug))


# ---- S3 LR grid ------------------------------------------------------------------------------
def bundle_specs_for(reader, evoked=True, out=None):
    """One reader's full grid cell list (hot-box reuse: every cell of a reader in ONE model
    load). The 14B reader: own diagonal + all old pools. Small readers: the 14B pools; 3B/7B
    additionally the run-(1) injected self-diagonal (Part B -- untrimmable)."""
    def secret_specs(gen):
        if gen in NEW_GENS:
            return [f"{gen}:{arm}:{new_bundle_path(gen, arm, out)}" for arm in SECRET_ARMS]
        return [f"{gen}:secret_word:{secret_word_path(gen)}",
                f"{gen}:secret_sustain:{secret_sustain_path(gen)}"]

    specs = []
    if reader in NEW_GENS:
        for gen in OLD_GENS + NEW_GENS:           # old pools + own diagonal
            specs += secret_specs(gen)
        if evoked:
            specs += [f"{reader}:{arm}:{new_bundle_path(reader, arm, out)}"
                      for arm in EVOKED_ARMS]
    else:
        for gen in NEW_GENS:
            specs += secret_specs(gen)
        if reader in INJECTED_READERS:            # 7a run (1): smax pass rides the main out dir
            specs.append(f"{reader}:injected:{injected_capture_path(reader)}")
    return specs


def grid_cmd(reader, specs, out=None, batch=None):
    cmd = [sys.executable, "-u", os.path.join(SRC, "lr_grid_extend.py"), "--reader", reader]
    for spec in specs:
        cmd += ["--bundle", spec]
    if batch:
        cmd += ["--batch", str(batch)]
    return cmd, {"INTRO_RUN_DIR": out or OUT}


def shards_for(reader, evoked=True, out=None):
    """Expected S3 shard paths per reader -- name-parity with lr_grid.shard_path (pinned by
    tests/test_lr_extend.py E5/E6, the grid's S7 pattern)."""
    base = os.path.join(out or OUT, "lr_grid")
    names = []

    def secret_names(gen):
        return [f"{reader}__{gen}__{ss}_{cs}.pt"
                for ss in SECRET_ARMS for cs in ("N", SECRET_CTX[ss])]

    if reader in NEW_GENS:
        for gen in OLD_GENS + NEW_GENS:
            names += secret_names(gen)
        if evoked:
            names += [f"{reader}__{reader}__{ss}_{cs}.pt"
                      for ss in EVOKED_ARMS for cs in ("N", "A", "B")]
    else:
        for gen in NEW_GENS:
            names += secret_names(gen)
        if reader in INJECTED_READERS:
            names += [f"{reader}__{reader}__injected_{cs}.pt" for cs in ("N", "A", "B")]
    return [os.path.join(base, n) for n in names]


# ---- S3b: the 7B run-(1) s124-primary pass (separate out dir; strength-filtered capture) ------
def write_strength_filtered_capture(src_path, dest_path, keep_strength):
    """A mechanical strength filter on the 7B capture (prereg Part B dose caveat): keep s0 and
    the pinned level only, so lr_reader.select_streams('injected') (strength == smax) selects
    exactly the s124 pool through the UNMODIFIED certified path. Metadata untouched beyond
    strengths; provenance disclosed in the written dict."""
    import torch
    b = torch.load(src_path, map_location="cpu", weights_only=False)
    keep = {0, int(keep_strength)}
    b["streams"] = [s for s in b["streams"] if int(s["strength"]) in keep]
    b["strengths"] = sorted(keep)
    b["strength_filter"] = dict(source=os.path.basename(src_path), kept=sorted(keep),
                                note="prereg lr_scale_extend Part B: s124 = the 7B "
                                     "criterion-passing primary dose (scale14b Amendment 1)")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".tmp"
    torch.save(b, tmp)
    os.replace(tmp, dest_path)
    return dest_path


def s124_out(out=None):
    return os.path.join(out or OUT, S124_SUBDIR)


def s124_capture_path(out=None):
    return os.path.join(s124_out(out), "qwen2.5-7b-gen-s124.pt")


def s124_shards(out=None):
    base = os.path.join(s124_out(out), "lr_grid")
    return [os.path.join(base, f"qwen2.5-7b__qwen2.5-7b__injected_{cs}.pt")
            for cs in ("N", "A", "B")]


# ---- S4: injection run (2) --------------------------------------------------------------------
def itf_cmd(slug, level, out=None, limit=None):
    cmd = [sys.executable, "-u", os.path.join(SRC, "inject_tf_lr.py"),
           "--capture", injected_capture_path(slug), "--slug", slug,
           "--strength", str(int(level))]
    if limit:
        cmd += ["--limit", str(int(limit))]
    return cmd, {"INTRO_MODEL": slug, "INTRO_RUN_DIR": out or OUT}


def itf_shards(out=None, slugs=None):
    base = os.path.join(out or OUT, "inject_tf")
    names = []
    for slug in (slugs if slugs is not None else ITF_LEVELS):
        for lvl in ITF_LEVELS[slug]:
            names += [f"{slug}__{slug}__injected_TFV_s{lvl}.pt",
                      f"{slug}__{slug}__injected_TFN_s{lvl}.pt"]
    return [os.path.join(base, n) for n in names]


# ---- S4a: Amendment-2 (2a) dose curve (Part B, untrimmable) ------------------------------------
def dose_cmd(slug, capture, levels, out=None, limit=None):
    """One multi-dose inject_tf_lr pass (ONE model load): --strengths + --no-s0 (the disclosed
    centering trim -- the main capture's centering rides the existing s60 shard). A dose whose
    stored vectors are missing degrades on-box to not-scored (dose_plan disclosure), never
    regenerated."""
    cmd = [sys.executable, "-u", os.path.join(SRC, "inject_tf_lr.py"),
           "--capture", capture, "--slug", slug,
           "--strengths", ",".join(str(int(l)) for l in levels), "--no-s0"]
    if limit:
        cmd += ["--limit", str(int(limit))]
    return cmd, {"INTRO_MODEL": slug, "INTRO_RUN_DIR": out or OUT}


def dose_shards(out=None, levels=None):
    """Expected 2a shard paths (run-(2) naming, level in the ctx code). The done-check drops
    any level the on-box dose_plan disclosed as not-scored (dose_not_scored)."""
    base = os.path.join(out or OUT, "inject_tf")
    lvls = levels if levels is not None else DOSE_LEVELS_E1 + DOSE_LEVELS_MAIN
    return [os.path.join(base, f"{DOSE_SLUG}__{DOSE_SLUG}__injected_TF{vn}_s{int(l)}.pt")
            for l in lvls for vn in ("V", "N")]


def dose_not_scored(out=None):
    """Union of the levels every on-box dose_plan JSON disclosed as not-scored (degraded:
    missing stored vectors/alpha). The done-check must not demand their shards."""
    import glob
    ns = set()
    for p in glob.glob(os.path.join(out or OUT, "inject_tf", "dose_plan_*.json")):
        with open(p) as f:
            plan = json.load(f)
        ns |= {int(m["level"]) for m in plan.get("not_scored", [])}
    return ns


# ---- S4b: Amendment-2 (2b) expressed-injection cell (trimmable at --trim>=2) -------------------
def expressed_bundle_path(out=None):
    return os.path.join(out or OUT, "expressed", f"{DOSE_SLUG}-{EXPRESSED_ARM}.pt")


def expressed_cmd(out=None, smoke=False):
    """The 2b generation subprocess: expressed_collect (covert_collect apparatus + verbatim
    sustain-s1 'this feeling' suffix + stored primitives: s20 from e1, s60 from main)."""
    cmd = [sys.executable, "-u", os.path.join(SRC, "expressed_collect.py"),
           "--out", out or OUT,
           "--e1-capture", e1_capture_path(),
           "--main-capture", injected_capture_path(DOSE_SLUG)]
    if smoke:
        cmd.append("--smoke")
    return cmd, {"INTRO_MODEL": DOSE_SLUG, "INTRO_RUN_DIR": out or OUT}


def expressed_itf_cmd(out=None, limit=None, smoke=False):
    """The 2b run-(2)-style self-read: inject_tf_lr over the expressed bundle (streamset
    'expressed' -> expressed_TF* shard names, its stored system_text as the context). s0-free
    (the bundle has no s0 pool by design)."""
    levels = EXPRESSED_SMOKE_DOSES if smoke else EXPRESSED_DOSES
    cmd = [sys.executable, "-u", os.path.join(SRC, "inject_tf_lr.py"),
           "--capture", expressed_bundle_path(out), "--slug", DOSE_SLUG,
           "--strengths", ",".join(str(int(l)) for l in levels), "--no-s0"]
    if limit:
        cmd += ["--limit", str(int(limit))]
    return cmd, {"INTRO_MODEL": DOSE_SLUG, "INTRO_RUN_DIR": out or OUT}


def expressed_shards(out=None, smoke=False):
    base = os.path.join(out or OUT, "inject_tf")
    levels = EXPRESSED_SMOKE_DOSES if smoke else EXPRESSED_DOSES
    return [os.path.join(base,
                         f"{DOSE_SLUG}__{DOSE_SLUG}__{EXPRESSED_ARM}_TF{vn}_s{int(l)}.pt")
            for l in levels for vn in ("V", "N")]


# ---- S5: the Amendment-1 70B rider ------------------------------------------------------------
def rider_cmd(reader, arms, out=None, limit=None):
    cmd = [sys.executable, "-u", os.path.join(SRC, "lr_rider.py"), "--reader", reader,
           "--arms", ",".join(arms), "--streams", rider_streams_path()]
    if limit:
        cmd += ["--limit", str(int(limit))]
    return cmd, {"INTRO_RUN_DIR": out or OUT}


def rider_shards(reader, arms=RIDER_ARMS, out=None):
    """Expected rider shard paths -- lr_grid.shard_path naming with gen slug RIDER_GEN and ctx
    codes N (arm-own neutral) / R (the 12-concept matched set)."""
    base = os.path.join(out or OUT, "lr_grid")
    return [os.path.join(base, f"{reader}__{RIDER_GEN}__{arm}_{cs}.pt")
            for arm in arms for cs in ("N", RIDER_CTX)]


# ---- progress / heartbeat ---------------------------------------------------------------------
_T0 = time.time()


def emit_step(step, **fields):
    """t = seconds since THIS process's import (legacy field); wall = epoch seconds, the
    cross-process timeline -- the S2 gen CHILD emits its model-ready step on the same wall
    clock as the parent's phase steps, so the driver's projection can subtract the one-time
    weights download from the gen slice (2026-07-14 review CRIT 2a)."""
    print("LABKIT_STEP " + json.dumps({"step": int(step), "t": int(time.time() - _T0),
                                       "wall": int(time.time()), **fields}), flush=True)


def start_heartbeat(period_s=120):
    def beat():
        while True:
            time.sleep(period_s)
            print(f"HEARTBEAT t={int(time.time() - _T0)}s", flush=True)
    threading.Thread(target=beat, daemon=True).start()


def _run(cmd, env):
    shown = " ".join(f"{k}={v}" for k, v in env.items())
    print(f"RUN {' '.join(cmd)}  env={shown}", flush=True)
    subprocess.run(cmd, env={**os.environ, **GPU_THREADS, **env}, cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="registered D-slice + rider slice: anchor + tiny 14B gen + its scoring "
                         "shard + a 2-stream run-(2) slice + a 2-stream e1 dose slice + a tiny "
                         "expressed 2b slice + a 4-stream rider slice at 1.5B; the driver "
                         "projects spend from steps")
    ap.add_argument("--trim", type=int, default=0,
                    help="trim ladder (Amendment 3, 2026-07-14): 1 = drop the 70B-rider "
                         "secret_word (descriptive) cells; 2 = ALSO drop the evoked/"
                         "evoked_alt arms; 3 = ALSO drop the 70B-rider confirmatory cells "
                         "(max 3). NEVER Part B / the 2a dose curve / secret cells / anchor; "
                         "the 2b expressed cell is WITHDRAWN (never scheduled, not a trim "
                         "level).")
    ap.add_argument("--evoked", action="store_true", help=argparse.SUPPRESS)  # S2 subprocess
    ap.add_argument("--gen-slug", default=None, help=argparse.SUPPRESS)       # S2 subprocess
    args = ap.parse_args()

    if args.gen_slug:                             # the S2 subprocess body
        run_generation(args.gen_slug, evoked=args.evoked, smoke=args.smoke)
        return

    plan = run_plan(args.trim)
    assert_part_b_untrimmable(plan)               # Amendments 1+2: an assertion, not a comment
    evoked = plan["evoked"]
    rider_arms = rider_arms_for(plan)
    print(f"LRX plan trim={args.trim}: evoked={evoked} rider_arms={list(rider_arms)} "
          f"expressed_2b={plan['expressed_2b']} part_b=True dose_2a=True (untrimmable)",
          flush=True)

    os.makedirs(OUT, exist_ok=True)
    start_heartbeat()
    _load_box_lr_grid().assert_personas()         # B12 byte-identity, before any stage
    emit_step(0, phase="S0_fetch", smoke=bool(args.smoke))
    fetch_inputs(smoke=args.smoke)
    print("LRX_READY", flush=True)

    # S1 anchor (smoke AND full -- the instrument certification is never skipped).
    emit_step(100, phase="S1_anchor")
    a_expect = anchor_shards()
    if all(os.path.exists(s) for s in a_expect):
        print(f"S1 SKIP anchor: all {len(a_expect)} shards exist", flush=True)
    else:
        _run(*grid_cmd(ANCHOR_READER, anchor_bundle_specs()))

    # S2 generation at 14B (smoke: tiny, secret_word only).
    for i, slug in enumerate(NEW_GENS):
        emit_step(500 + i, phase="S2_gen", slug=slug)
        want = ["secret_word"] if args.smoke else list(gen_arms(evoked))
        if all(os.path.exists(new_bundle_path(slug, a)) for a in want):
            print(f"S2 SKIP {slug}: bundles exist", flush=True)
            continue
        _run(*gen_cmd(slug, evoked=evoked, smoke=args.smoke))
        missing = [a for a in want if not os.path.exists(new_bundle_path(slug, a))]
        if missing:
            raise RuntimeError(f"S2: {slug} bundles missing after generation: {missing}")

    # S3 the grid (smoke: one 14B pass over its own tiny bundle -- times the big-reader path).
    if args.smoke:
        emit_step(1000, phase="S3_lr_grid", reader="qwen2.5-14b", smoke=True)
        spec = [f"qwen2.5-14b:secret_word:{new_bundle_path('qwen2.5-14b', 'secret_word')}"]
        expect = [os.path.join(OUT, "lr_grid", f"qwen2.5-14b__qwen2.5-14b__secret_word_{cs}.pt")
                  for cs in ("N", "SW")]
        if not all(os.path.exists(s) for s in expect):
            _run(*grid_cmd("qwen2.5-14b", spec))
    else:
        for i, reader in enumerate(READERS):
            emit_step(1000 * (i + 1), phase="S3_lr_grid", reader=reader)
            expect = shards_for(reader, evoked=evoked)
            if all(os.path.exists(s) for s in expect):
                print(f"S3 SKIP {reader}: all {len(expect)} shards exist", flush=True)
                continue
            _run(*grid_cmd(reader, bundle_specs_for(reader, evoked=evoked)))
        # S3b: the 7B run-(1) s124-primary pass (separate out dir, filtered capture). Part B.
        emit_step(7000, phase="S3b_s124")
        if all(os.path.exists(s) for s in s124_shards()):
            print("S3b SKIP: s124 shards exist", flush=True)
        else:
            cap = s124_capture_path()
            if not os.path.exists(cap):
                write_strength_filtered_capture(injected_capture_path("qwen2.5-7b"), cap, 124)
            spec = [f"qwen2.5-7b:injected:{cap}"]
            _run(*grid_cmd("qwen2.5-7b", spec, out=s124_out()))

    # S4 injection run (2) -- Part B, UNTRIMMABLE; runs BEFORE the rider so a deadman kill eats
    # the trimmable tail first (smoke: 1.5B, 2 streams -- proves the hook path end-to-end).
    assert plan["part_b"], "Part B must be in the plan (Amendment 1)"
    emit_step(8000, phase="S4_inject_tf", smoke=bool(args.smoke))
    if args.smoke:
        expect = itf_shards(slugs=["qwen2.5-1.5b"])
        if not all(os.path.exists(s) for s in expect):
            _run(*itf_cmd("qwen2.5-1.5b", ITF_PRIMARY["qwen2.5-1.5b"], limit=2))
    else:
        for slug in ITF_LEVELS:
            for lvl in ITF_LEVELS[slug]:
                have = [s for s in itf_shards(slugs=[slug]) if f"_s{lvl}." in s]
                if all(os.path.exists(s) for s in have):
                    print(f"S4 SKIP {slug} s{lvl}: shards exist", flush=True)
                    continue
                _run(*itf_cmd(slug, lvl))

    # S4a Amendment-2 (2a) dose curve -- Part B, UNTRIMMABLE (scores EXISTING streams; a dose
    # missing stored vectors degrades to not-scored on-box, never regenerated).
    assert plan["dose_2a"], "the 2a dose curve must be in the plan (Amendment 2)"
    emit_step(8200, phase="S4a_dose", smoke=bool(args.smoke))
    if args.smoke:
        if not all(os.path.exists(s) for s in dose_shards(levels=DOSE_SMOKE_LEVELS)):
            _run(*dose_cmd(DOSE_SLUG, e1_capture_path(), DOSE_SMOKE_LEVELS, limit=2))
    else:
        for capture, levels in ((e1_capture_path(), DOSE_LEVELS_E1),
                                (injected_capture_path(DOSE_SLUG), DOSE_LEVELS_MAIN)):
            ns = dose_not_scored()
            want = dose_shards(levels=[l for l in levels if l not in ns])
            # fix 13: all() over an EMPTY want is True -- a capture whose every level the
            # dose_plan disclosed as not-scored resume-skips instead of re-running.
            if all(os.path.exists(s) for s in want):
                print(f"S4a SKIP {os.path.basename(capture)} s{list(levels)}: "
                      f"{'all levels disclosed not-scored' if not want else 'shards exist'}",
                      flush=True)
                continue
            _run(*dose_cmd(DOSE_SLUG, capture, levels))

    # S4b Amendment-2 (2b) expressed-injection cell -- trims after the rider descriptive cells
    # (--trim>=2), before the evoked arms.
    if plan["expressed_2b"]:
        emit_step(8300, phase="S4b_expressed_gen", smoke=bool(args.smoke))
        if os.path.exists(expressed_bundle_path()):
            print("S4b SKIP generation: expressed bundle exists", flush=True)
        else:
            _run(*expressed_cmd(smoke=args.smoke))
        emit_step(8310, phase="S4b_expressed_itf", smoke=bool(args.smoke))
        want = expressed_shards(smoke=args.smoke)
        if all(os.path.exists(s) for s in want):
            print("S4b SKIP self-read: expressed shards exist", flush=True)
        else:
            _run(*expressed_itf_cmd(smoke=args.smoke, limit=(2 if args.smoke else None)))
    else:
        print("S4b SKIP expressed 2b (trimmed; disclosed)", flush=True)

    # S5 the Amendment-1 70B rider (smoke: 4 streams x 1 arm at 1.5B -- proves the text path).
    if args.smoke:
        emit_step(8500, phase="S5_rider", smoke=True)
        expect = rider_shards(RIDER_SMOKE_READER, arms=(RIDER_SMOKE_ARM,))
        if not all(os.path.exists(s) for s in expect):
            _run(*rider_cmd(RIDER_SMOKE_READER, (RIDER_SMOKE_ARM,),
                            limit=RIDER_SMOKE_LIMIT))
    elif rider_arms:
        for i, reader in enumerate(RIDER_READERS):
            emit_step(8500 + i, phase="S5_rider", reader=reader)
            expect = rider_shards(reader, arms=rider_arms)
            if all(os.path.exists(s) for s in expect):
                print(f"S5 SKIP {reader}: all {len(expect)} rider shards exist", flush=True)
                continue
            _run(*rider_cmd(reader, rider_arms))
    else:
        print("S5 SKIP rider (trimmed; disclosed)", flush=True)

    # done-check: never report done with nothing to pull.
    if args.smoke:
        expected = a_expect + itf_shards(slugs=["qwen2.5-1.5b"])
        expected += dose_shards(levels=DOSE_SMOKE_LEVELS)
        if plan["expressed_2b"]:
            expected += [expressed_bundle_path()] + expressed_shards(smoke=True)
        expected += rider_shards(RIDER_SMOKE_READER, arms=(RIDER_SMOKE_ARM,))
    else:
        expected = a_expect + [s for r in READERS for s in shards_for(r, evoked=evoked)]
        expected += s124_shards() + itf_shards()
        ns = dose_not_scored()                    # disclosed degradations are not demanded
        expected += dose_shards(levels=[l for l in DOSE_LEVELS_E1 + DOSE_LEVELS_MAIN
                                        if l not in ns])
        if plan["expressed_2b"]:
            expected += [expressed_bundle_path()] + expressed_shards()
        expected += [s for r in RIDER_READERS for s in rider_shards(r, arms=rider_arms)]
    missing = [s for s in expected if not os.path.exists(s)]
    if missing:
        raise RuntimeError(f"shards missing: {[os.path.relpath(m, OUT) for m in missing]}")
    emit_step(9000, phase="lrx_done", shards=len(expected), smoke=bool(args.smoke))
    print(f"outputs OK in {OUT}", flush=True)


def _is_child_invocation(argv):
    """True for self-reinvocations (--gen-slug): the CHILD must never print LRX_DONE — labkit
    substring-matches markers on log lines, so a child's trailer reads as the BOX being done
    (the 2026-07-14 smoke bug: teardown raced S3, S4/S4a/S5 never ran, the shakedown was
    registered for an incomplete smoke). LRX_FATAL stays unguarded: a child crash SHOULD
    fatal the box."""
    return "--gen-slug" in argv


if __name__ == "__main__":
    try:
        main()
        if not _is_child_invocation(sys.argv):
            print("LRX_DONE", flush=True)
    except Exception:
        print("LRX_FATAL", flush=True)
        raise
