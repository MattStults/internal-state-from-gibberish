"""On-box orchestrator for the LR SCALE-GRID run (prereg: reports/lr_scale_grid_prereg.md,
checklist: reports/lr_scale_grid_checklist.md).

One contiguous 48GB-tier box, four phases (perf checklist item 1 -- no manual phases):
  S0  HF-pull the stream bundles rsync excludes (*.pt): the exp3 evoked bundles at 1.5B/3B/7B,
      the 1.5B evoked_alt bundle, and (B15) the three existing secret_word bundles, from
      ErrareHumanumEst/internal-state-from-gibberish (HF_TOKEN); FULL runs also pull the E5
      maintained-secret pool (one descriptive cell -- a disclosed pre-launch upload).
  S1  B1 alt-stream generation at 3B/7B: exp3's collect_induction.py (config-level reuse:
      --arms evoked_alt only; its primers import widened to the drop-in-superset primers_v3 for
      B15, byte-identical for this arm), identical anti-word instruction, word-free filter and
      acceptance gates; the strength-0 neutral cell rides inside the pipeline per arm.
  S1b B15 (prereg Amendment 2) secret_sustain generation at 1.5B/3B/7B: the SAME pipeline,
      --arms secret_sustain (secret-word context + E2's piloted s1 sustain template, the word
      substituted -- primers_v3). Feasibility floor OFF (an auxiliary arm must never FATAL the
      shared box; acceptance is REPORTED and the offline per-concept-n gate voids thin cells).
      NO blind-judge gauge (registered: no persona to evoke, the manipulation is in context).
  S2  B2 LR grid: src/lr_grid.py once per reader (subprocess per model -> clean GPU teardown),
      readers x stream-sets x context wordings + the B15 secret cells (matched one-sentence
      contexts vs arm-own neutral; E5 rides the 1.5B reader only); atomic shards -> out/lr_grid/.
  S3  B9: the MC self-report diagonal at 3B/7B, contiguous on the same box -- box_mc.py
      --own-pool with an EXPLICIT model list and a FRESH INTRO_REPORT_DIR (out/mc_diag), per
      the B6 seams; mc_reader scoring stays byte-identical (sha-pinned).

--smoke = the REGISTERED D1 slice (tiny alt-gen at 3B, one LR cell per reader family incl the
D2 1.5B eos-free diagonal anchor, one MC diagonal shard); the driver projects full-run spend
from this run's LABKIT_STEP timings (B10: over-budget -> STOP and ask Matt).

Markers: LRG_READY / LRG_DONE / LRG_FATAL (collision-checked in tests/test_lr_grid.py: no marker
is a substring of another, across ALL box scripts' markers). Driven by a gated harness driver
(unit B9). NEVER run this on the Mac -- it loads models.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP3 = os.path.join(REPO, "experiments", "exp3_induction_and_scale")

HF_DATASET = "ErrareHumanumEst/internal-state-from-gibberish"
OUT = os.path.abspath(os.environ.get("INTRO_REPORT_DIR") or os.path.join(REPO, "out"))
GEN_MODELS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")   # stream generators (prereg Streams)
ALT_GEN = ("qwen2.5-3b", "qwen2.5-7b")                      # evoked_alt to GENERATE (1.5B exists)
# Readers (prereg: 6). Smallest-first within family; Qwen first (the self-diagonal core cells),
# the cross-family readers after -- their context rendering is the A1 seam (adjudicated in
# prereg Amendment 1). Amendment 4 (2026-07-11): the cross-family readers are the Falcon3
# Instruct models (the registered ungated fallback; the box 403'd on meta-llama -- the license
# was never actually accepted, and HfApi.model_info() is NOT an access preflight).
READERS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
           "falcon3-1b", "falcon3-3b", "falcon3-7b")
STREAM_SETS = ("evoked", "evoked_alt")
CTX_SETS = ("N", "A", "B")
# B15 (prereg Amendment 2): the secret stream sets, each scored under ONE matched ctx set + the
# arm-own neutral. This map MUST stay in lockstep with lr_grid.SECRET_CTX (this module is
# stdlib-only, so it cannot import lr_grid; parity is pinned by test_lr_grid_secret X7).
SECRET_SETS = ("secret_word", "secret_sustain")
SECRET_CTX = {"secret_word": "SW", "secret_sustain": "SS", "maintained_secret": "SM"}
SECRET_GEN = GEN_MODELS               # S1b: secret_sustain generated at ALL THREE sizes
# Control (b): measure LR on INJECTED streams at 3B and 7B (the certified 1.5B LR run already
# read injected x A = 0.002; its cell rides src/lr_reader's out/lr/). A SMALL diagonal add --
# each reader reads its OWN size's exp1 covert_collect capture (self-legibility of injection),
# contrasting the secret channel that GROWS with scale. Diagonal is Qwen self-read only (falcon
# readers get no injected cell). The captures are on HF at the dataset-root <slug>-gen.pt path
# (the 1.5B LR run's fetch name; box_confound already pulls all three) -> local
# runs/<slug>/data/covert_collect.pt.
INJECTED_READERS = ("qwen2.5-3b", "qwen2.5-7b")
E5_READER = "qwen2.5-1.5b"            # the ONE descriptive E5 cell rides the 1.5B reader only
# S3 (B9): the MC self-report diagonal rides this box -- EXPLICIT model list (B6 seam: never
# box_mc's default list; 1.5B x own already exists from the certified MC run).
DIAG_MODELS = ("qwen2.5-3b", "qwen2.5-7b")
# --smoke = the REGISTERED D1 slice: tiny alt-gen at 3B, one LR cell per reader family, one MC
# diagonal shard. The Qwen smoke reader runs BOTH 1.5B pools so smoke also records the D2
# eos-free 1.5B diagonal anchor (prereg Amendment 1, Blocker 2).
SMOKE_ALT_GEN = ("qwen2.5-3b",)
SMOKE_READERS = ("qwen2.5-1.5b", "falcon3-1b")        # Amendment 4: falcon3 = the xfam smoke
GPU_THREADS = {k: "8" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}

# ---- B7: box-env deps per reader family (the driver's deps= list) ---------------------------
# transformers==4.46.3 VERIFIED (locally, 2026-07-11, and re-checked by tests/test_lr_grid D2
# against the installed pin) to load the llama architecture: LlamaForCausalLM imports, 'llama'
# in CONFIG_MAPPING, 'llama3' rope-scaling init present. Amendment 4: the Falcon3-{1B,3B,7B}-
# Instruct readers DECLARE that same architecture (model_type=llama, arch LlamaForCausalLM --
# verified from their configs, 2026-07-11), so the whole grid (Qwen2.5 + Falcon3 readers) rides
# the single validated pin -- version-matched to the certified 1.5B LR run. A qwen3 reader
# (none in this grid) would need >=4.51 (arch not in 4.46.3; the ENV-not-code bug both MC
# reviews missed).
# wordfreq + scikit-learn ride because S1 alt-generation shares the box (collect_induction env:
# without wordfreq the word-free filter is INERT and run_model raises).
_BASE_DEPS = ["accelerate", "numpy", "safetensors", "huggingface_hub", "scikit-learn", "wordfreq"]


def deps_for(slugs):
    tf = ("transformers>=4.51,<5.0" if any(s.startswith("qwen3") for s in slugs)
          else "transformers==4.46.3")
    return [tf] + _BASE_DEPS


# ---- S0 inputs: rsync excludes *.pt -> the box HF-pulls them (box_mc/box_lr precedent) ------
EVOKED = {m: os.path.join(REPO, "runs", "_ind", m, "data", f"{m}-evoked.pt") for m in GEN_MODELS}
ALT15 = os.path.join(REPO, "runs", "_ind", "qwen2.5-1.5b", "data", "qwen2.5-1.5b-evoked_alt.pt")
# B15: the exp3 secret_word bundles EXIST at all 3 sizes, both locally (runs/_ind) and on the HF
# dataset (exp3/bundles/, listing verified 2026-07-11) -- fetched like the evoked ones.
SECRET_WORD = {m: os.path.join(REPO, "runs", "_ind", m, "data", f"{m}-secret_word.pt")
               for m in GEN_MODELS}
# Control (b): the exp1 injected captures for the 3B/7B diagonal. They live LOCALLY at
# runs/<slug>/data/covert_collect.pt (rsync excludes *.pt) and ARE on the HF dataset at the
# root path <slug>-gen.pt (the same file box_confound.py's S5 already pulls for all three sizes;
# md5-verified identical to the local captures) -- fetched exactly like the evoked/secret_word
# bundles. NOT an exp3/bundles/ _ind file (different schema: inject == 'gen').
INJECTED = {m: os.path.join(REPO, "runs", m, "data", "covert_collect.pt")
            for m in INJECTED_READERS}
FETCHES = [(f"exp3/bundles/{m}-evoked.pt", EVOKED[m]) for m in GEN_MODELS] + \
          [("exp3/bundles/qwen2.5-1.5b-evoked_alt.pt", ALT15)] + \
          [(f"exp3/bundles/{m}-secret_word.pt", SECRET_WORD[m]) for m in GEN_MODELS] + \
          [(f"{m}-gen.pt", INJECTED[m]) for m in INJECTED_READERS]
# B15: the E5 maintained-secret pool (ONE descriptive cell, 1.5B reader). IS on HF at the path
# below (upload verified independently by both C-phase reviews, 2026-07-11); smoke never scores
# E5, so it stays a full-run-only fetch and a smoke box cannot trip on it.
E5_BUNDLE = os.path.join(REPO, "runs", "confound_box", "e5_secret", "data",
                         "qwen2.5-1.5b-maintained_secret.pt")
FETCHES_FULL_ONLY = [("confound/bundles/qwen2.5-1.5b-maintained_secret.pt", E5_BUNDLE)]


def fetch_inputs(smoke=False):
    from huggingface_hub import hf_hub_download
    fetches = list(FETCHES) + ([] if smoke else FETCHES_FULL_ONLY)
    for fname, dest in fetches:
        if os.path.exists(dest):
            print(f"S0 have {os.path.relpath(dest, REPO)}", flush=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(hf_hub_download(HF_DATASET, fname, repo_type="dataset"), dest)
        print(f"S0 fetched {fname} -> {os.path.relpath(dest, REPO)}", flush=True)
    missing = [f for f, d in fetches if not os.path.exists(d)]
    if missing:
        raise RuntimeError(f"S0: inputs missing after HF pull: {missing}")


# ---- B12: persona byte-identity + env/pipeline parity (prereg Amendment 1, should-fix 7) ----
# sha256 over the FROZEN primers.EVOKED (wording A) / primers.EVOKED_ALT (wording B) / NEUTRAL
# texts, canonical serialization "<wording>|<concept>|<text>\n" in sorted-concept order + the
# neutral suffix. Pinned from the frozen exp3 primers.py (a pre-registration artifact): asserting
# the runtime digest at box start proves EVERY size's generation and every reader's context
# reconstruction uses byte-identical persona texts (the amendment's cross-size byte-identity
# assert). Independently recomputed by tests/test_alt_gauge.py A5.
PERSONA_SHA256 = "46eb3f51c13e6980a456d1561f2c0bef0a65411f5c11601f7ed18fe4fd7ab29b"
PARITY_TRANSFORMERS = "4.46.3"    # the exp3 evoked collections' validated pin (deps_for's pin)
# collect_induction.cfg(smoke=False) -- the exp3 REAL-RUN sizing the evoked pools were collected
# with. Pinned HERE (stdlib-only module) and cross-checked against the live cfg on-box
# (alt_parity_check) and locally (test A4): a drift in exp3 defaults fails loudly pre-spend and
# becomes a disclosed diff, never a silent one.
PARITY_CFG = dict(target_clean=36, max_gen=192, tokens=128, gen_batch=32, gen_topk=64, gauge_n=8)


def persona_digest(primers_mod=None):
    """Digest of the frozen A/B persona texts + NEUTRAL (see PERSONA_SHA256 for the format)."""
    if primers_mod is None:
        if EXP3 not in sys.path:
            sys.path.insert(0, EXP3)
        import primers as primers_mod
    h = hashlib.sha256()
    for wording, d in (("A", primers_mod.EVOKED), ("B", primers_mod.EVOKED_ALT)):
        for c in sorted(d):
            h.update(f"{wording}|{c}|{d[c]}\n".encode())
    h.update(("N|" + primers_mod.NEUTRAL).encode())
    return h.hexdigest()


def assert_personas(primers_mod=None):
    """B12 byte-identity gate: the personas about to be composed (S1 generation AND every reader
    context in S2) must hash to the pinned frozen digest. Any drift is terminal, pre-spend."""
    got = persona_digest(primers_mod)
    if got != PERSONA_SHA256:
        raise RuntimeError(f"persona byte-identity check failed: digest {got} != pinned "
                           f"{PERSONA_SHA256} -- the wording-A/B persona texts drifted from the "
                           "frozen exp3 primers; refusing to generate or score")
    print(f"S0 personas byte-identical (sha256 {PERSONA_SHA256[:12]}...)", flush=True)


def alt_parity_check(slug):
    """B12 / Amendment 1 should-fix 7: env + pipeline parity vs the exp3 evoked collection at
    this size, asserted ON-BOX before any alt stream is generated ($0-fail: raises before the
    model download). Same transformers pin, word-free filter live (wordfreq importable -- the
    collector itself also raises on a real run without it), same sampling cfg."""
    import transformers
    if transformers.__version__ != PARITY_TRANSFORMERS:
        raise RuntimeError(f"alt-generation parity: transformers {transformers.__version__} != "
                           f"the evoked collections' pin {PARITY_TRANSFORMERS}")
    try:
        import wordfreq
    except ImportError:
        raise RuntimeError("alt-generation parity: wordfreq missing -- the word-free filter "
                           "would be inert, breaking acceptance parity with the evoked pools")
    if EXP3 not in sys.path:
        sys.path.insert(0, EXP3)
    from collect_induction import cfg
    live = cfg(False)
    if live != PARITY_CFG:
        raise RuntimeError(f"alt-generation parity: collect_induction real-run cfg drifted: "
                           f"{live} != pinned {PARITY_CFG} -- disclose as an amendment, never "
                           "generate silently off-parity")
    # smoke attempt 1 lesson ($0.01): newer wordfreq ships no __version__ attribute -- read the
    # installed version from package metadata, which exists for any pip-installed dist.
    import importlib.metadata
    wf_ver = importlib.metadata.version("wordfreq")
    print(f"S1 parity OK {slug}: transformers={transformers.__version__} "
          f"wordfreq={wf_ver} cfg=pinned", flush=True)


def assert_alt_bundle_parity(slug):
    """Post-generation structural parity: the new alt bundle vs the exp3 evoked bundle at the
    SAME size (concepts identical + orig prompt variant), so a silently divergent pipeline cannot
    hand mismatched pools to the grid."""
    import torch
    alt = torch.load(alt_bundle_path(slug), map_location="cpu", weights_only=False)
    ev = torch.load(EVOKED[slug], map_location="cpu", weights_only=False)
    if list(alt.get("concepts", [])) != list(ev.get("concepts", [])):
        raise RuntimeError(f"alt bundle parity: concepts differ from the {slug} evoked bundle")
    v = alt.get("variant")
    if v not in (None, "orig"):
        raise RuntimeError(f"alt bundle parity: prompt variant {v!r}, not the published 'orig'")
    print(f"S1 alt bundle parity OK {slug}: {len(alt.get('concepts', []))} concepts, "
          f"variant orig", flush=True)


# ---- B1: alt-stream generation at 3B/7B (config-level reuse of the exp3 pipeline) -----------
ALT_MIN_PER_CLASS = 24        # run_exp3's real-run feasibility floor -- gates identical to 1.5B


def assert_alt_bundle_feasible(slug, smoke=False, path=None):
    """TECH-SF2: the have-bundle resume branch must NOT bypass the B2 feasibility gate.
    collect_induction runs check_min_per_class AFTER saving the bundle, so a run FATALed by the
    gate leaves a complete-looking bundle behind; a naive resume would treat that thin pool as
    done. Re-run the SAME check (CPU-cheap, the collector's own function object -- nothing
    reimplemented, collect_induction.py untouched) on the loaded bundle before the stage counts
    it: a failure re-raises the identical feasibility RuntimeError. Smoke keeps the floor off
    (min-per-class 0), matching altgen_cmd's smoke sizing."""
    import torch
    if EXP3 not in sys.path:
        sys.path.insert(0, EXP3)
    from collect_induction import check_min_per_class
    b = torch.load(path or alt_bundle_path(slug), map_location="cpu", weights_only=False)
    check_min_per_class(b["streams"], 0 if smoke else ALT_MIN_PER_CLASS)


def alt_bundle_path(slug, out=None):
    """The evoked_alt bundle for `slug`: the repo copy (1.5B exists; S0 fetches it) wins, else the
    S1-generated copy under OUT/_ind/<slug>/data (labkit pulls out/, so it comes home)."""
    repo_p = os.path.join(REPO, "runs", "_ind", slug, "data", f"{slug}-evoked_alt.pt")
    if os.path.exists(repo_p):
        return repo_p
    return os.path.join(out or OUT, "_ind", slug, "data", f"{slug}-evoked_alt.pt")


def altgen_cmd(slug, out=None, smoke=False):
    """B1: exp3's collect_induction.py, arms=evoked_alt only, real-run feasibility floor. (B15
    widened its primers import to the drop-in-superset primers_v3 -- byte-identical composition
    for this arm, pinned by test_lr_grid_secret P3/P4; nothing else in the pipeline changed.) The identical anti-word instruction, word-free filter, acceptance gates and the
    riding-along strength-0 neutral cell are all properties of that pipeline (run_model collects
    the neutral cell inside every arm; a wordfreq-less real run raises). Nothing forked, no
    sizing overrides (cfg(smoke=False) defaults = the 1.5B alt run's config). smoke=True (the
    registered D1 slice ONLY, never the real run): the pipeline's own --smoke sizing with the
    feasibility floor OFF -- a handful of throwaway streams to time the box, worthless as data."""
    cmd = [sys.executable, "-u",
           os.path.join(REPO, "experiments", "exp3_induction_and_scale", "collect_induction.py"),
           "--models", slug, "--arms", "evoked_alt",
           "--min-per-class", "0" if smoke else str(ALT_MIN_PER_CLASS)]
    if smoke:
        cmd.append("--smoke")
    env = {"INTRO_MODEL": slug,
           "INTRO_RUN_DIR": os.path.join(out or OUT, "_ind", slug)}
    return cmd, env


def alt_gauge_path(slug, out=None):
    """B12: the evoked_alt gauge SIDECAR for `slug` -- written by gauge_alt_collect under
    OUT/_ind/<slug>/data (labkit pulls out/, so it comes home for the offline pinned judge)."""
    return os.path.join(out or OUT, "_ind", slug, "data", f"{slug}-evoked_alt-gauge.pt")


def gauge_cmd(slug, out=None):
    """B12: exp3's gauge_alt_collect.py -- the evoked_alt blind-judge gauge texts, same probe +
    sampling as the exp3 evoked gauge (parity pinned in that script), one subprocess per size."""
    cmd = [sys.executable, "-u", os.path.join(EXP3, "gauge_alt_collect.py")]
    env = {"INTRO_MODEL": slug,
           "INTRO_RUN_DIR": os.path.join(out or OUT, "_ind", slug)}
    return cmd, env


def altgen_stage(smoke=False):
    """S1: generate the missing evoked_alt bundles, one subprocess per model (clean GPU teardown
    between sizes), each preceded by the B12 env/pipeline parity check and followed by the
    structural bundle-parity check + the B12 alt-gauge collection (blind-judge texts; judged
    OFFLINE by gauge_judge_alt -- a gauge fail flags cells, it never fails this box). Resume-safe
    at every level: an existing bundle skips generation, an existing sidecar skips the gauge;
    collect_induction itself resumes per-(arm, concept) atomic shards inside a partial run.
    smoke (D1 slice): tiny alt-gen at 3B only, no gauge (throwaway timing streams)."""
    for slug in (SMOKE_ALT_GEN if smoke else ALT_GEN):
        dest = alt_bundle_path(slug)
        if os.path.exists(dest):
            # TECH-SF2: never bypass the feasibility gate on resume -- the collector saves the
            # bundle before its gate runs, so an existing bundle can be a gate-FATALed one.
            assert_alt_bundle_feasible(slug, smoke=smoke)
            print(f"S1 have alt bundle {os.path.relpath(dest, REPO)} "
                  f"(feasibility re-checked)", flush=True)
        else:
            alt_parity_check(slug)                       # B12: $0-fail BEFORE generating
            cmd, env = altgen_cmd(slug, smoke=smoke)
            print(f"RUN {' '.join(cmd)}  env=INTRO_MODEL={env['INTRO_MODEL']} "
                  f"INTRO_RUN_DIR={env['INTRO_RUN_DIR']}", flush=True)
            subprocess.run(cmd, env={**os.environ, **GPU_THREADS, **env}, cwd=REPO, check=True)
            if not os.path.exists(dest):
                raise RuntimeError(f"S1: alt bundle missing after generation: {dest}")
            assert_alt_bundle_parity(slug)               # B12: structural parity vs evoked
            print(f"S1 generated {os.path.relpath(dest, REPO)}", flush=True)
        if smoke:
            continue                                     # D1: no gauge on throwaway streams
        side = alt_gauge_path(slug)
        if os.path.exists(side):
            print(f"S1 have alt gauge {os.path.basename(side)}", flush=True)
            continue
        gcmd, genv = gauge_cmd(slug)
        print(f"RUN {' '.join(gcmd)}  env=INTRO_MODEL={genv['INTRO_MODEL']} "
              f"INTRO_RUN_DIR={genv['INTRO_RUN_DIR']}", flush=True)
        subprocess.run(gcmd, env={**os.environ, **GPU_THREADS, **genv}, cwd=REPO, check=True)
        if not os.path.exists(side):
            raise RuntimeError(f"S1: alt gauge sidecar missing after collection: {side}")


# ---- B15 / S1b: secret_sustain generation at 1.5B/3B/7B (prereg Amendment 2) ----------------
def secret_bundle_path(slug, out=None):
    """The secret_sustain bundle for `slug`: repo copy wins (none exist yet -- the arm is new),
    else the S1b-generated copy under OUT/_ind/<slug>/data (labkit pulls out/)."""
    repo_p = os.path.join(REPO, "runs", "_ind", slug, "data", f"{slug}-secret_sustain.pt")
    if os.path.exists(repo_p):
        return repo_p
    return os.path.join(out or OUT, "_ind", slug, "data", f"{slug}-secret_sustain.pt")


def secretgen_cmd(slug, out=None):
    """S1b: the SAME exp3 pipeline as S1, --arms secret_sustain only (composition = secret-word
    sentence + E2's piloted s1 sustain template with the word substituted; primers_v3). The
    identical anti-word instruction, word-free filter and the riding s0 neutral cell are pipeline
    properties. Feasibility floor OFF (--min-per-class 0, a REGISTERED pin): Amendment 2's trim
    order makes this arm the FIRST casualty, so a low-acceptance secret arm must report thin
    pools (offline per-concept-n gate voids cells), never FATAL the box that carries the core LR
    grid. Acceptance-rate reporting rides the pipeline's own acceptance_report print."""
    cmd = [sys.executable, "-u",
           os.path.join(REPO, "experiments", "exp3_induction_and_scale", "collect_induction.py"),
           "--models", slug, "--arms", "secret_sustain", "--min-per-class", "0"]
    env = {"INTRO_MODEL": slug,
           "INTRO_RUN_DIR": os.path.join(out or OUT, "_ind", slug)}
    return cmd, env


def assert_secret_bundle_parity(slug):
    """Post-generation structural parity (the alt-gen precedent): the new secret_sustain bundle
    vs the exp3 evoked bundle at the SAME size (concepts identical + orig prompt variant)."""
    import torch
    sec = torch.load(secret_bundle_path(slug), map_location="cpu", weights_only=False)
    ev = torch.load(EVOKED[slug], map_location="cpu", weights_only=False)
    if list(sec.get("concepts", [])) != list(ev.get("concepts", [])):
        raise RuntimeError(f"secret bundle parity: concepts differ from the {slug} evoked bundle")
    v = sec.get("variant")
    if v not in (None, "orig"):
        raise RuntimeError(f"secret bundle parity: prompt variant {v!r}, not the published 'orig'")
    print(f"S1b secret bundle parity OK {slug}: {len(sec.get('concepts', []))} concepts, "
          f"variant orig", flush=True)


def secretgen_stage(smoke=False):
    """S1b: generate the secret_sustain bundles at all three sizes, one subprocess per model,
    each preceded by the SAME B12 env/pipeline parity check as S1 and followed by the structural
    bundle-parity check. NO blind-judge gauge for this arm -- registered in Amendment 2 (there is
    no persona to evoke; the manipulation is trivially present in context). NOT in smoke: the D1
    projection EXTRAPOLATES this stage from alt-gen timing (checklist B15)."""
    if smoke:
        print("S1b SKIP: secret_sustain generation is not in the smoke slice (registered; the "
              "driver projection extrapolates it from alt-gen timing)", flush=True)
        return
    for slug in SECRET_GEN:
        dest = secret_bundle_path(slug)
        if os.path.exists(dest):
            print(f"S1b have secret bundle {os.path.relpath(dest, REPO)}", flush=True)
            continue
        alt_parity_check(slug)                           # same $0-fail parity gate as S1
        cmd, env = secretgen_cmd(slug)
        print(f"RUN {' '.join(cmd)}  env=INTRO_MODEL={env['INTRO_MODEL']} "
              f"INTRO_RUN_DIR={env['INTRO_RUN_DIR']}", flush=True)
        subprocess.run(cmd, env={**os.environ, **GPU_THREADS, **env}, cwd=REPO, check=True)
        if not os.path.exists(dest):
            raise RuntimeError(f"S1b: secret bundle missing after generation: {dest}")
        assert_secret_bundle_parity(slug)
        print(f"S1b generated {os.path.relpath(dest, REPO)} (no gauge -- registered)",
              flush=True)


_T0 = time.time()


def emit_step(step, **fields):
    """Progress steps for labkit's watchdog AND the driver's smoke spend projection: every step
    carries t = seconds since box start, so stage durations are parseable from the log."""
    print("LABKIT_STEP " + json.dumps({"step": int(step), "t": int(time.time() - _T0),
                                       **fields}), flush=True)


def start_heartbeat(period_s=120):
    """The gauge run's attempt-1 lesson: a silent HF weight download froze the log and the stall
    watchdog killed the box. A daemon heartbeat keeps the log growing through genuinely-quiet
    work; run_to still caps a true hang."""
    t0 = time.time()

    def beat():
        while True:
            time.sleep(period_s)
            print(f"HEARTBEAT t={int(time.time() - t0)}s", flush=True)

    threading.Thread(target=beat, daemon=True).start()


# ---- B2: the LR grid (readers x stream-sets x context wordings) -----------------------------
def bundle_specs(out=None, reader=None):
    """--bundle specs for ONE reader invocation: evoked at 3 sizes (fetched repo copies) +
    evoked_alt at 3 sizes (1.5B fetched; 3B/7B resolved to wherever S1 put them) + (B15) the
    secret_word bundles (fetched) and the S1b-generated secret_sustain bundles at 3 sizes. The
    1.5B reader additionally gets the ONE E5 maintained-secret descriptive spec (Amendment 2)."""
    specs = [f"{m}:evoked:{EVOKED[m]}" for m in GEN_MODELS]
    specs += [f"{m}:evoked_alt:{alt_bundle_path(m, out)}" for m in GEN_MODELS]
    specs += [f"{m}:secret_word:{SECRET_WORD[m]}" for m in GEN_MODELS]
    specs += [f"{m}:secret_sustain:{secret_bundle_path(m, out)}" for m in GEN_MODELS]
    if reader == E5_READER:
        specs.append(f"{E5_READER}:maintained_secret:{E5_BUNDLE}")
    # Control (b): the injected self-diagonal rides ONLY the 3B/7B readers -- each reads its OWN
    # size's exp1 capture (reader == generator; self-legibility of injection).
    if reader in INJECTED_READERS:
        specs.append(f"{reader}:injected:{INJECTED[reader]}")
    return specs


def shards_for(reader, out=None):
    """Expected shard paths for one reader -- MUST stay name-identical to lr_grid.shard_path
    (cross-checked by test_lr_grid S7 / test_llama_ctx L9 / test_lr_grid_secret X5, so
    bookkeeping drift fails tests, not a paid box). Cross-family (Amendment 4: falcon3) readers
    additionally produce the A1 raw-text robustness SECONDARY (_raw) per grid cell plus gate 4's
    two prose-control shards (B4/B13). B15: every reader adds the secret cells ({N, matched} per
    secret set per size); the 1.5B reader adds the two E5 shards."""
    base = os.path.join(out or OUT, "lr_grid")
    xfam = reader.startswith("falcon3")
    names = [f"{reader}__prose__control_{cs}.pt" for cs in ("N", "A")] if xfam else []
    for m in GEN_MODELS:
        for ss in STREAM_SETS:
            for cs in CTX_SETS:
                names.append(f"{reader}__{m}__{ss}_{cs}.pt")
                if xfam:
                    names.append(f"{reader}__{m}__{ss}_{cs}_raw.pt")
        for ss in SECRET_SETS:
            for cs in ("N", SECRET_CTX[ss]):
                names.append(f"{reader}__{m}__{ss}_{cs}.pt")
                if xfam:
                    names.append(f"{reader}__{m}__{ss}_{cs}_raw.pt")
    if reader == E5_READER:
        for cs in ("N", SECRET_CTX["maintained_secret"]):
            names.append(f"{reader}__{E5_READER}__maintained_secret_{cs}.pt")
    # Control (b): the injected self-diagonal shards (N/A/B) for the 3B/7B Qwen readers only --
    # reader == generator, natural-persona ctx, so no _raw secondary (Qwen readers, CTX_SETS).
    if reader in INJECTED_READERS:
        for cs in CTX_SETS:
            names.append(f"{reader}__{reader}__injected_{cs}.pt")
    return [os.path.join(base, n) for n in names]


def grid_cmd(reader, batch=None, out=None, specs=None):
    """One src/lr_grid.py invocation (subprocess per reader -> clean GPU teardown). env sets
    INTRO_RUN_DIR only -- NEVER INTRO_MODEL: the reader is --reader (falcon slugs are not in the
    config registry, and config asserts INTRO_MODEL membership at import). specs overrides the
    full-grid bundle list (the --smoke slice passes its per-reader cells)."""
    cmd = [sys.executable, "-u", os.path.join(REPO, "src", "lr_grid.py"), "--reader", reader]
    for spec in (specs if specs is not None else bundle_specs(out, reader)):
        cmd += ["--bundle", spec]
    if batch:
        cmd += ["--batch", str(batch)]
    return cmd, {"INTRO_RUN_DIR": out or OUT}


def smoke_bundle_specs(reader, out=None):
    """D1: one LR cell per reader family on REAL pools -- Qwen 1.5B reads BOTH 1.5B pools (the
    D2 eos-free diagonal anchor = evoked x B + evoked_alt x A at 1.5B, plus the ~0.59 evoked x A
    regression check) AND (B15) the 1.5B secret_word pool (existing bundle, cheap -- so the D4
    projection covers the new cell type); Falcon3 1B (Amendment 4) reads the 1.5B evoked pool
    only."""
    m = "qwen2.5-1.5b"
    specs = [f"{m}:evoked:{EVOKED[m]}"]
    if reader.startswith("qwen"):
        specs.append(f"{m}:evoked_alt:{alt_bundle_path(m, out)}")
        specs.append(f"{m}:secret_word:{SECRET_WORD[m]}")
    return specs


def smoke_shards_for(reader, out=None):
    """Expected D1 outputs per smoke reader (name-parity with lr_grid.shard_path)."""
    base = os.path.join(out or OUT, "lr_grid")
    m = "qwen2.5-1.5b"
    if reader.startswith("qwen"):
        names = [f"{reader}__{m}__{ss}_{cs}.pt" for ss in STREAM_SETS for cs in CTX_SETS]
        names += [f"{reader}__{m}__secret_word_{cs}.pt"
                  for cs in ("N", SECRET_CTX["secret_word"])]
        return [os.path.join(base, n) for n in names]
    names = [f"{reader}__prose__control_{cs}.pt" for cs in ("N", "A")]
    for cs in CTX_SETS:
        names += [f"{reader}__{m}__evoked_{cs}.pt", f"{reader}__{m}__evoked_{cs}_raw.pt"]
    return [os.path.join(base, n) for n in names]


# ---- S3 (B9): the MC self-report diagonal, contiguous on the same box ------------------------
def mc_diag_cmd(out=None):
    """Full run: box_mc.py --own-pool as a subprocess, B6 seams honored -- EXPLICIT model list
    (never box_mc's default) and a FRESH INTRO_REPORT_DIR (OUT/mc_diag: diagonal shard filenames
    coincide with default-pool ones; mc_reader's assert_shard_source would FATAL on a cross-pool
    resume, a clean dir avoids the trip entirely)."""
    cmd = [sys.executable, "-u", os.path.join(HERE, "box_mc.py"),
           "--models", ",".join(DIAG_MODELS), "--own-pool"]
    return cmd, {"INTRO_REPORT_DIR": os.path.join(out or OUT, "mc_diag")}


def mc_smoke_cmd(out=None):
    """D1: ONE MC diagonal shard -- mc_reader directly (elicited x direct on the 3B evoked
    bundle, own-pool flags, no capture; the --framings/--reasonings cell filters are
    orchestration-only, scoring bodies sha-pinned)."""
    m = "qwen2.5-3b"
    cmd = [sys.executable, "-u", os.path.join(REPO, "src", "mc_reader.py"),
           "--evoked", EVOKED[m], "--stream-source", m,
           "--sets", "evoked", "--framings", "elicited", "--reasonings", "direct"]
    return cmd, {"INTRO_MODEL": m, "INTRO_RUN_DIR": os.path.join(out or OUT, "mc_diag")}


def mc_diag_shards(out=None):
    """Expected S3 outputs (8 per diagonal reader; box_mc's own naming)."""
    base = os.path.join(out or OUT, "mc_diag", "mc")
    return [os.path.join(base, f"{slug}_{ss}_{fr}_{rs}.pt")
            for slug in DIAG_MODELS for ss in ("evoked", "evoked_s0")
            for fr in ("elicited", "passive") for rs in ("direct", "cot")]


def mc_smoke_shard(out=None):
    return os.path.join(out or OUT, "mc_diag", "mc",
                        "qwen2.5-3b_evoked_elicited_direct.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readers", default=",".join(READERS),
                    help="comma-separated reader slugs (Qwen first: the self-diagonal core "
                         "cells; ignored under --smoke)")
    ap.add_argument("--batch", type=int, default=None,
                    help="override lr_grid's conservative per-size default (B8 smoke sets this)")
    ap.add_argument("--smoke", action="store_true",
                    help="the REGISTERED D1 slice: tiny alt-gen at 3B, one LR cell per reader "
                         "family (incl the D2 1.5B eos-free diagonal anchor), one MC diagonal "
                         "shard. The driver prints a spend projection from this run's steps.")
    args = ap.parse_args()
    readers = (list(SMOKE_READERS) if args.smoke
               else [r.strip() for r in args.readers.split(",") if r.strip()])

    os.makedirs(OUT, exist_ok=True)
    start_heartbeat()
    assert_personas()             # B12: byte-identity of the A/B persona texts, before any stage
    emit_step(0, phase="S0_fetch", smoke=bool(args.smoke))
    fetch_inputs(smoke=args.smoke)
    print("LRG_READY", flush=True)

    emit_step(500, phase="S1_altgen")
    altgen_stage(smoke=args.smoke)

    # S1b (B15): secret_sustain generation -- never in smoke (registered; projection
    # extrapolates it from the S1 alt-gen timing).
    emit_step(700, phase="S1b_secretgen", smoke=bool(args.smoke))
    secretgen_stage(smoke=args.smoke)

    for i, reader in enumerate(readers):
        emit_step(1000 * (i + 1), phase="S2_lr_grid", reader=reader)
        expect = smoke_shards_for(reader) if args.smoke else shards_for(reader)
        if all(os.path.exists(s) for s in expect):
            print(f"S2 SKIP {reader}: all {len(expect)} shards exist", flush=True)
            continue
        cmd, env = grid_cmd(reader, args.batch,
                            specs=smoke_bundle_specs(reader) if args.smoke else None)
        print(f"RUN {' '.join(cmd)}  env=INTRO_RUN_DIR={env['INTRO_RUN_DIR']}", flush=True)
        subprocess.run(cmd, env={**os.environ, **GPU_THREADS, **env}, cwd=REPO, check=True)

    # S3 (B9): the MC self-report diagonal rides the SAME box, contiguous (perf checklist 1).
    # Full run = box_mc --own-pool over the EXPLICIT 3B/7B list with a fresh INTRO_REPORT_DIR
    # (B6 seams); smoke = one mc_reader cell. mc_reader scoring stays byte-identical (sha-pinned).
    emit_step(8000, phase="S3_mc_diag", smoke=bool(args.smoke))
    mc_expect = [mc_smoke_shard()] if args.smoke else mc_diag_shards()
    if all(os.path.exists(s) for s in mc_expect):
        print(f"S3 SKIP: all {len(mc_expect)} MC diagonal shards exist", flush=True)
    else:
        cmd, env = mc_smoke_cmd() if args.smoke else mc_diag_cmd()
        shown = " ".join(f"{k}={v}" for k, v in env.items())
        print(f"RUN {' '.join(cmd)}  env={shown}", flush=True)
        subprocess.run(cmd, env={**os.environ, **GPU_THREADS, **env}, cwd=REPO, check=True)

    expected = [s for r in readers
                for s in (smoke_shards_for(r) if args.smoke else shards_for(r))] + mc_expect
    missing = [s for s in expected if not os.path.exists(s)]
    if missing:                                         # never report done with nothing to pull
        raise RuntimeError(f"shards missing: {[os.path.relpath(m, OUT) for m in missing]}")
    emit_step(9000, phase="lr_grid_done", shards=len(expected), smoke=bool(args.smoke))
    print(f"outputs OK in {OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
        print("LRG_DONE", flush=True)
    except Exception:
        print("LRG_FATAL", flush=True)
        raise
