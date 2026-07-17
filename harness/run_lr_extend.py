#!/usr/bin/env python3
"""Gated driver for the LR scale-grid EXTENSION run (prereg: exp2 reports/
lr_scale_extend_prereg.md -- FROZEN 2026-07-13 + Amendments 1-2 2026-07-14; the header Decisions
block + the Amendments GOVERN over the body's older 14B/32B dual text). Effective scope:
**14B-only** (32B fully descoped), evoked/evoked_alt IN at 14B, the 70B cross-family observer
RIDER (Amendment 1), Part B injection runs UNTRIMMABLE, the Amendment-2 (2a) dose curve
untrimmable with them, the (2b) expressed-injection cell IN unless trimmed. ONE contiguous 48GB
box runs box_lr_extend.py end-to-end: S0 fetch -> S1 anchor (1.5B D2 certification, 0.16648 +-
0.01) -> S2 generation at 14B (secret + evoked arms) -> S3 the 4-reader LR grid (14B diagonal +
off-diagonal block + the 7a run-(1) injected cells) -> S3b 7B s124 primary -> S4 the 7a run-(2)
inject-during-TF pass (UNTRIMMABLE) -> S4a the Amendment-2 dose curve at 1.5B (e1 s3-s20 + main
s40; UNTRIMMABLE) -> S4b the expressed-injection cell (1.5B gen {s20, s60} + self-read) -> S5
the Amendment-1 70B rider (4 Qwen readers teacher-force the 810 Llama-70B text streams).

Safety rails (run_lr_grid.py pattern, byte-parallel where the concern is identical):
  - experimentfactory gate (authorized_run); BUILD/VALIDATE with --dry first ($0 mock runner).
    Gate-facts retarget (2026-07-14, this file): max_spend = ledger + the $12 Q3 authorization
    (<= the $20 policy ceiling); container image PINNED (labkit's proven default, explicit);
    vram_est recomputed for the 14B peak (VRAM_EST_MB below) with the 15% margin satisfied on a
    48GB card; shakedown = a REGISTERED --smoke run for the current code-hash
    (runs/lr_extend_shakedowns.json -- the --smoke run registers itself on success, so the full
    run's shakedown fact is machine-checked, not asserted).
  - max_hours: the policy's 4h ceiling is DELIBERATELY exceeded by the ~12h contiguous full run
    (phase table below). Study result (2026-07-14): run_lr_grid's 10h run cleared this the same
    way every over-ceiling run does -- the gate runs in mode="observe" and the policy routes
    over-ceiling spend TO A HUMAN rather than auto-authorizing; Matt's interactive launch IS that
    authorization. The reason is printed and expected; chunking the box would violate the frozen
    "ONE contiguous box" design (resume-safe shards notwithstanding, weights re-download per
    chunk would burn the saving).
  - per-PROJECT ledger runs/confound-ledger.json; per-run authorization $12 (prereg Q3, Matt
    2026-07-13); smoke projection <= authorization advises GO, > advises STOP and exits 3.
    REVISED trim order (Amendment 2): 70B-rider secret_word (descriptive) -> the expressed 2b
    cell -> evoked arms -> 70B-rider confirmatory -> NEVER the 14B secret diagonals, NEVER
    Part B, NEVER the 2a dose curve (runs 1 and 2 + the dose curve are the evidence for any
    "injection does not show up in LR" claim; asserted in the box).
  - provider deadman armed from max_hours + buffer (the E1 clamp override via provider_kwargs).
  - launch preflights ($0): prereg freeze marker; HF presence of every fetch rsync cannot carry
    (*.pt -- the L2 secret_sustain uploads; the 70B streams JSON travels with the workdir rsync
    and is only required on HF when the local copy is missing).
  - rsync-255 = retryable ssh-transport flake (one bounded relaunch; on-box stages resume).

Hour math (prereg Part C revised by the Decisions block + Amendments 1-3; re-derive from the
real smoke's LABKIT_STEP timings + per-unit subprocess lines before full launch -- the B10
discipline):
  setup+weights 0.5 | anchor 0.3 | 14B gen secret 1.3 | 14B gen evoked 1.0-1.5 |
  14B diag+evoked scoring 1.4 | off-diagonal block 1.2 | Amendment-1 rider 0.7-1.2 |
  Part B run (1) 0.4 + run (2) 0.9 | Amendment-2 2a dose curve 0.2-0.3
  => 7.9-9.0 h, x1.25 slack => 9.9-11.3 h, CAPPED at max_hours 11.0 (Amendment 3: 2b
  withdrawn, Amendment 2's 12h bump reverted; the shards resume, so a worst-case deadman kill
  loses only the tail).
  Spend ceiling check: 11.0 h x $0.85/hr (48GB tier) = $9.35 <= the $12 authorization.
  VRAM unchanged: the 1.5B dose passes are tiny next to the 14B generation peak.

TWO planned invocations (each --dry first):
  .venv-driver/bin/python harness/run_lr_extend.py --smoke [--dry]
  .venv-driver/bin/python harness/run_lr_extend.py [--trim N] [--dry]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LABKIT_TAG = "v0.2.50"
LEDGER_PATH = str(REPO / "runs" / "confound-ledger.json")
PROJECT_CAP = 5.0                    # the original shared-ledger line (kept for spent-so-far math)
RUN_AUTHORIZED_USD = 12.0            # prereg Q3 RESOLVED (Matt, 2026-07-13): $12 for 14B-only
TRIM_ORDER = ("70B-rider secret_word (descriptive) -> evoked arms -> 70B-rider confirmatory "
              "-> NEVER the 14B secret diagonals, NEVER Part B, NEVER the 2a dose curve "
              "(Amendment 3, 2026-07-14: the 2b expressed cell is WITHDRAWN, not trimmable)")
DEADMAN_BUFFER_S = 1800
THREAD_CAPS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                "NUMEXPR_NUM_THREADS")}

PREREG = (REPO / "experiments" / "exp2_output_monitorability" / "reports"
          / "lr_scale_extend_prereg.md")

# 48GB tier (prereg Q5 RESOLVED: 1x RTX 6000 Ada / L40S / A6000 class; the 80GB tier died with
# the 32B descope). VRAM peak = the 14B generation stage (largest resident model):
#   bf16 weights 29.5 GB = 30208 MB
# + gen KV: GEN_BATCH_CAP[14b]=16 x ~2.2k tok x 192 KB/tok = 6600 MB
#   (Qwen2.5-14B: 48 layers x 8 KV heads x 128 head_dim x 2 bytes x 2 (K+V) = 196608 B/tok --
#    the same derivation as the old spec's 262 KB/tok at 32B: 64 layers x 8 x 128 x 2 x 2)
# + fp32 logit chunks <= 717 MB (0.7 GB, unchanged from the old spec's term)
# + CUDA context/framework overhead ~= 2048 MB
# = 39573 MB -> VRAM_EST_MB 39600. x1.15 margin = 45540 <= the 46000 MB request <= 48GB card
# (49152 MB physical). Scoring peaks lower (batch 6 x ~2.2k seq). It FITS -- no fudging needed.
DEFAULT_GPU = "RTX A6000"
DEFAULT_MIN_VRAM = 46000
VRAM_EST_MB = 39600
# Container image PINNED (gate fact image_pinned): labkit's proven default, made EXPLICIT so the
# run is reproducible against a labkit default bump (same actual image the grid ran on).
IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
MAX_HOURS_FULL = 11.0                # 7.9-9.0 h phase table x1.25 slack, capped (Amendment 3:
#                                      2b withdrawn, Amendment 2's 12h bump reverted;
#                                      11 x $0.85 = $9.35 <= the $12 authorization)
MAX_HOURS_SMOKE = 2.0
WEIGHTS_GB = {"1.5b": 3.5, "3b": 6.5, "7b": 16.0, "14b": 29.5}    # 32B descoped (no entry: a
#                                                                   32b reader must KeyError)
IMAGE_GB = 10.0
DISK_SLACK_GB = 8.0                  # bundles + captures (~1.5GB) + shards + HF dupes


def _load_box():
    p = REPO / "experiments" / "exp2_output_monitorability" / "box_lr_extend.py"
    if "box_lr_extend" in sys.modules:
        return sys.modules["box_lr_extend"]
    spec = importlib.util.spec_from_file_location("box_lr_extend", str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["box_lr_extend"] = mod
    spec.loader.exec_module(mod)
    return mod


BOX = _load_box()
deps_for = BOX.deps_for


def entrypoint_for(smoke=False, trim=0):
    ep = "python3 -u experiments/exp2_output_monitorability/box_lr_extend.py"
    if smoke:
        ep += " --smoke"
    if trim:
        ep += f" --trim {int(trim)}"
    return ep


def run_id_for(smoke=False):
    return "lr-extend-smoke" if smoke else "lr-extend"


def disk_for_run(readers=None):
    readers = readers if readers is not None else list(BOX.READERS)
    return round(IMAGE_GB + sum(WEIGHTS_GB[r.split("-")[-1].lower()] for r in readers)
                 + DISK_SLACK_GB, 1)


def provider_kwargs(run_id, disk_gb, max_hours, throttle_path=None):
    """default_deadman_s MUST ride max_hours (the E1 clamp: labkit create() clamps every
    requested deadman to min(default_deadman_s, requested); provider default is 6h)."""
    return dict(owner=run_id, disk_gb=disk_gb, throttle_path=throttle_path,
                default_deadman_s=int(max_hours * 3600) + DEADMAN_BUFFER_S)


def is_rsync_flake(reasons=None, error=None):
    blobs = list(reasons or [])
    if error:
        blobs.append(str(error))
    pat = re.compile(r"rsync[^\n]*\b255\b")
    return any(pat.search(str(b)) for b in blobs)


def remaining_budget(ledger_path=LEDGER_PATH, cap=PROJECT_CAP):
    if not os.path.exists(ledger_path):
        return float(cap)
    with open(ledger_path) as f:
        led = json.load(f)
    return float(cap) - float(sum(led.values()))


def prereg_frozen(path=PREREG):
    """$0 launch gate: the DRAFT marker must be gone (Matt removed it at the 2026-07-13 freeze).
    --dry and tests never hit this."""
    try:
        text = Path(path).read_text()
    except OSError:
        return False
    return "DRAFT — NOT FROZEN" not in text and "DRAFT -- NOT FROZEN" not in text


def preflight_hf_inputs(list_repo_files=None, exists=os.path.exists):
    """$0 launch gate for checklist L2: every fetch that rsync CANNOT carry to the box must be
    PRESENT on the HF dataset before a box is paid for. rsync ships the workdir but excludes
    *.pt (labkit DEFAULT_EXCLUDES), so .pt inputs are HF-only; a non-.pt input (the Amendment-1
    70B streams JSON) travels with the workdir rsync and is only required on HF when the local
    copy is missing. Presence check on our own private dataset (list_repo_files); the box's S0
    hf_hub_download still exercises the real access call class (the Amendment-4 lesson applies
    to gated third-party repos). Returns the missing HF paths (empty = GO)."""
    if list_repo_files is None:
        from huggingface_hub import HfApi
        list_repo_files = lambda: HfApi().list_repo_files(BOX.HF_DATASET,  # noqa: E731
                                                          repo_type="dataset")
    have = set(list_repo_files())
    missing = []
    for f, dest in BOX.fetches():
        rsync_carries = (not str(dest).endswith(".pt")) and exists(dest)
        if f not in have and not rsync_carries:
            missing.append(f)
    return missing


# ------------------------------------------------------------------ shakedown (code-hash smoke)
# Gate fact shakedown_done ("no cheap shakedown for this code-hash"): a --smoke run that reaches
# LRX_DONE registers the hash of the code that actually runs on the box; the full run's Spec
# then carries a MACHINE-CHECKED shakedown_done (registry lookup), never a hand-set True (the
# run_exp3 `not args.smoke` pattern asserted it on trust; this closes that gap).
SHAKEDOWN_REGISTRY = str(REPO / "runs" / "lr_extend_shakedowns.json")
CODE_HASH_FILES = (
    "experiments/exp2_output_monitorability/box_lr_extend.py",
    "harness/run_lr_extend.py",
    "src/lr_grid_extend.py",
    "src/lr_grid.py",
    "src/lr_reader.py",
    "src/lr_rider.py",
    "src/inject_tf_lr.py",
    "src/serverless_72b.py",
    "src/common.py",
    "src/config.py",
    "experiments/exp3_induction_and_scale/collect_induction.py",
    "experiments/exp3_induction_and_scale/primers_v3.py",
    # Amendment 2: the 2a dose curve executes these on-box (2b withdrawn per Amendment 3 --
    # expressed_collect.py deliberately NOT hashed: unscheduled code must not invalidate
    # shakedowns)
    "src/covert_collect.py",
    "experiments/exp3_induction_and_scale/primers.py",
    "experiments/exp3_induction_and_scale/primers_v2.py",
)


def code_hash(files=CODE_HASH_FILES, repo=REPO):
    """sha256 over the byte contents of every file the box actually executes (order pinned by
    the tuple). A one-byte change anywhere invalidates the shakedown -- by design."""
    h = hashlib.sha256()
    for rel in files:
        h.update(rel.encode())
        h.update(Path(repo, rel).read_bytes())
    return h.hexdigest()[:16]


def shakedown_registered(h, registry=SHAKEDOWN_REGISTRY):
    if not os.path.exists(registry):
        return False
    with open(registry) as f:
        return h in json.load(f)


def register_shakedown(h, info=None, registry=SHAKEDOWN_REGISTRY):
    reg = {}
    if os.path.exists(registry):
        with open(registry) as f:
            reg = json.load(f)
    reg[h] = dict(info or {}, t=int(time.time()))
    with open(registry, "w") as f:
        json.dump(reg, f, indent=1)
    return registry


def gate_fields(smoke=False, spent=None, min_vram=DEFAULT_MIN_VRAM):
    """The retargeted gate-facing numbers, in ONE place (main() builds kwargs/Spec from these;
    tests shape-check them against the policy ceilings: max_spend <= $20, vram margin 15%,
    image pinned; max_hours full deliberately exceeds the 4h ceiling -- human-routed, above)."""
    spent = (PROJECT_CAP - remaining_budget()) if spent is None else float(spent)
    return dict(
        max_spend=round(spent + RUN_AUTHORIZED_USD, 2),
        max_hours=MAX_HOURS_SMOKE if smoke else MAX_HOURS_FULL,
        image=IMAGE,
        min_vram_mb=int(min_vram),
        vram_est_mb=VRAM_EST_MB,
        shakedown_done=shakedown_registered(code_hash()),
    )


# ------------------------------------------------------------------ smoke spend projection
# Named scale factors, smoke slice -> full run, 14B-ONLY phase list (32B terms REMOVED;
# Amendment-1 rider term ADDED). Re-derived from the real smoke's LABKIT_STEP timings before any
# full launch (the B10 printer discipline).
#
# 2026-07-14 review CRIT 2: per-unit rates come from the PER-SHARD/PER-CTX timing lines the
# subprocesses already print (LRG ctx ... done t= / RIDER ctx ... done t= / ITF_SHARD_SAVED
# ... t=; each t is relative to that subprocess's own POST-MODEL-LOAD clock), never from
# whole-phase wall deltas dominated by process spawn + weights load. The S2 gen slice is
# bounded at the gen child's S2_model_ready wall-clock step, excluding the one-time 14B
# weights download. Model-load overhead re-enters once, as the fixed named term below.
# Phase-delta fallbacks are retained for logs predating these lines (over-estimate,
# disclosed).
GEN_SCALE_14B = (36 / 4) * (128 / 48)   # target_clean 4->36, tokens 48->128 (cfg parity)
GEN_ARMS_SECRET = 2                     # secret_word + secret_sustain (smoke gens 1 arm)
EVOKED_GEN_MULT = 1.0                   # evoked+evoked_alt gen ~= the secret-arm total (prereg
#                                         phase table: 1.0-1.5h vs 1.3h); trim>=2 zeroes it
SCORE_STREAM_SCALE = 9.0                # tiny smoke pool (~4/concept) -> real (~36/concept)
SHARDS_14B_SECRET = 16                  # 4 gens x 2 arms x 2 ctx (smoke scores 2)
SHARDS_14B_EVOKED = 6                   # 2 evoked arms x N/A/B (trim>=2 drops)
SMALL_READER_FACTOR = {"qwen2.5-7b": 4.7, "qwen2.5-3b": 2.0, "qwen2.5-1.5b": 1.0}
SMALL_SHARDS = {"qwen2.5-7b": 4 + 3 + 3, "qwen2.5-3b": 4 + 3, "qwen2.5-1.5b": 4}
#               ^ new-pool shards (1 new gen x 2 arms x 2 ctx) + run-(1) injected N/A/B on
#                 3B/7B + the 7B s124 N/A/B pass
READER_FACTOR_14B = 9.3                 # vs the 1.5B anchor reader (bf16 param ratio)
RIDER_READER_FACTOR_SUM = 1.0 + 2.0 + 4.7 + 9.3   # all four rider readers vs 1.5B
RIDER_STREAMS_PER_ARM = 270             # 810 Llama-70B streams / 3 arms
RIDER_CONTEXTS = 13                     # matched-set 12 concepts + arm-own neutral
RIDER_SMOKE_UNITS = 4 * 13              # the smoke rider slice: 4 streams x 13 ctx at 1.5B
# Run-(2) stream counts per slug = (accepted streams at the scored level) + (accepted s0
# centering pool), both scored under all 13 labels. MEASURED from the captures
# (runs/<slug>/data/covert_collect.pt, accepted + len>=2; re-verified 2026-07-14 -- the
# review caught the old '+60' s0 guess, real s0 pools are 417/445/443):
#   1.5B: 435 @ s60  + 417 s0;  3B: 422 @ s60 + 445 s0;
#   7B:  two passes -- 429 @ s124 + 443 s0, and 430 @ s140 + 443 s0.
ITF_STREAMS = {"qwen2.5-1.5b": 435 + 417, "qwen2.5-3b": 422 + 445,
               "qwen2.5-7b": (429 + 443) + (430 + 443)}
ITF_FACTOR = {"qwen2.5-1.5b": 1.0, "qwen2.5-3b": 2.0, "qwen2.5-7b": 4.7}
# Amendment-2 terms (1.5B only). 2a scores the MEASURED accepted pools s0-free (--no-s0,
# disclosed): e1 s3/5/8/12/20 = 421+405+454+437+433 = 2150 + the main capture's s40 = 449.
DOSE_2A_STREAMS = 2150 + 449            # per-stream cost = the S4 smoke's 1.5B run-(2) unit
# 2b: smoke gens 2 concepts x 2 accepted x 48 tok x 1 dose; full = 12 x 24 x 128 tok x 2 doses.
EXPRESSED_GEN_SCALE = (12 / 2) * (24 / 2) * (128 / 48) * (2 / 1)
EXPRESSED_ITF_STREAMS = 2 * 12 * 24     # the self-read pool (both doses, target accepted)
SETUP_S = 1800.0                        # boot + pip + ~55.5GB weights not in step deltas
SLACK = 1.25
# CRIT 2b: the per-unit rates exclude subprocess model loads, so the FULL run's per-process
# load overhead re-enters ONCE as a fixed term (not multiplied by any scale factor). Process
# counts per size over the full phase list:
#   14b: S2 gen + S3 reader + S5 rider                              = 3
#   7b:  S3 reader + S3b s124 + S4 (s124, s140 passes) + S5 rider   = 5
#   3b:  S3 reader + S4 + S5 rider                                  = 3
#   1.5b: S1 anchor + S3 reader + S4 + S4a (e1, main) + S5 rider    = 6
MODEL_LOAD_COUNTS = {"14b": 3, "7b": 5, "3b": 3, "1.5b": 6}
PROC_START_S = 90.0                     # python + torch + tokenizer per subprocess (estimate)
LOAD_S_PER_GB = 2.0                     # bf16 weights load from the local disk cache


def overhead_full_s():
    """Fixed full-run per-process overhead (CRIT 2b): loads are excluded from the per-unit
    rates and must not be silently dropped either."""
    return float(sum(n * (PROC_START_S + LOAD_S_PER_GB * WEIGHTS_GB[size])
                     for size, n in MODEL_LOAD_COUNTS.items()))


def parse_steps(log_text):
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
    """Per-phase durations from consecutive steps. Prefers the wall (epoch) field when both
    endpoints carry it -- the ONE cross-process timeline (the gen child's S2_model_ready step
    runs on a different process clock, so its legacy t is meaningless here); falls back to the
    legacy per-process t for old logs."""
    dur = {}
    for a, b in zip(steps, steps[1:]):
        key = a.get("phase", str(a.get("step")))
        if "wall" in a and "wall" in b:
            d = float(b["wall"]) - float(a["wall"])
        else:
            d = float(b.get("t", 0)) - float(a.get("t", 0))
        dur[key] = dur.get(key, 0.0) + d
    return dur


# CRIT 2b: the subprocess per-unit timing lines (each t= is on that subprocess's own
# post-model-load clock, so the units carry NO spawn/load overhead).
_RE_LRG14 = re.compile(r"LRG ctx qwen2\.5-14b/\S+ done\s+t=(\d+)s")
_RE_RIDER = re.compile(r"RIDER ctx \S+ done t=(\d+)s")
_RE_ITF = re.compile(r"ITF_SHARD_SAVED (\S+) n=(\d+) t=(\d+)s")


def parse_unit_times(log_text):
    """Per-unit timing from the pulled smoke run.log's subprocess lines. Returns any of:
      lrg14_s          -- last 14B 'LRG ctx' t: scoring work for the 2-shard smoke slice
      rider_last_s     -- last 'RIDER ctx' t: the 4-stream x 13-ctx rider slice at 1.5B
      itf_per_stream_s -- ITF_SHARD_SAVED V-shard t / n: per-stream all-13-labels at 1.5B
                          (prefers the S4 s60 shard; any injected V shard as fallback)
    Empty dict when the lines are absent (phase-delta fallback governs)."""
    if not log_text:
        return {}
    out = {}
    lrg = _RE_LRG14.findall(log_text)
    if lrg:
        out["lrg14_s"] = float(lrg[-1])
    rid = _RE_RIDER.findall(log_text)
    if rid:
        out["rider_last_s"] = float(rid[-1])
    itf = [(name, int(n), int(t)) for name, n, t in _RE_ITF.findall(log_text)
           if "_TFV_" in name and "__injected_" in name]
    if itf:
        pref = [x for x in itf if "_s60" in x[0]] or itf
        name, n, t = pref[-1]
        if n:
            out["itf_per_stream_s"] = float(t) / float(n)
    return out


def smoke_projection(steps, dph, trim=0, log_text=None):
    """Full-run seconds/dollars from the smoke's measured stage durations + per-unit
    subprocess timing lines (CRIT 2), 14B-only phase list (S1 anchor / S2 gen secret+evoked /
    S3 grid incl small readers + s124 / S4 inject-TF / S4a dose curve / S5 rider; the S4b
    expressed term is permanently zero -- Amendment 3 withdrew 2b, run_plan pins expressed_2b
    False). Every factor is a named constant above; re-derive from the real smoke before any
    full launch. trim mirrors the box plan: >=1 drops the rider secret_word term, >=2 the
    evoked terms, >=3 the whole rider; Part B and the 2a dose term are NEVER dropped
    (Amendments 1+2+3)."""
    plan = BOX.run_plan(trim)
    dur = _stage_durations(steps)
    units = parse_unit_times(log_text)
    anchor_s = dur.get("S1_anchor", 0.0)                 # runs identically in full (resume-skips)
    # CRIT 2a: generation work = the post-model-ready slice of S2 (the S2_model_ready step
    # splits the phase; its own S2_gen sub-delta is the one-time download+load, excluded).
    gen_work = dur.get("S2_model_ready")
    if gen_work is None:
        gen_work = dur.get("S2_gen", 0.0)   # legacy fallback: includes the weights download
        #                                     (over-estimates; disclosed in the note)
    gen_secret = gen_work * GEN_SCALE_14B * GEN_ARMS_SECRET
    gen_evoked = gen_secret * EVOKED_GEN_MULT if plan["evoked"] else 0.0
    if "lrg14_s" in units:
        smoke_shard = units["lrg14_s"] / 2.0             # 2 shards, post-load clock
    else:
        smoke_shard = dur.get("S3_lr_grid", 0.0) / 2.0   # fallback: includes the model load
    shard14 = smoke_shard * SCORE_STREAM_SCALE
    n14 = SHARDS_14B_SECRET + (SHARDS_14B_EVOKED if plan["evoked"] else 0)
    score_s = shard14 * n14
    per_1p5b_shard = shard14 / READER_FACTOR_14B
    for r, f in SMALL_READER_FACTOR.items():
        score_s += per_1p5b_shard * f * SMALL_SHARDS[r]
    if "itf_per_stream_s" in units:
        per_stream_1p5b = units["itf_per_stream_s"]      # V-shard t/n, post-load clock
    else:
        per_stream_1p5b = dur.get("S4_inject_tf", 0.0) / 4.0   # 2(+2 s0) streams, w/ load
    itf_s = sum(per_stream_1p5b * ITF_FACTOR[s] * ITF_STREAMS[s] for s in ITF_STREAMS)
    dose_2a_s = per_stream_1p5b * DOSE_2A_STREAMS        # Part B: NEVER trimmed (Amendment 2)
    if plan["expressed_2b"]:
        expressed_s = (dur.get("S4b_expressed_gen", 0.0) * EXPRESSED_GEN_SCALE
                       + per_stream_1p5b * EXPRESSED_ITF_STREAMS)
    else:
        expressed_s = 0.0
    if "rider_last_s" in units:
        rider_unit = units["rider_last_s"] / RIDER_SMOKE_UNITS   # post-load clock
    else:
        rider_unit = dur.get("S5_rider", 0.0) / RIDER_SMOKE_UNITS   # fallback: w/ model load
    n_rider_arms = len(BOX.rider_arms_for(plan))
    rider_s = (rider_unit * RIDER_STREAMS_PER_ARM * RIDER_CONTEXTS * n_rider_arms
               * RIDER_READER_FACTOR_SUM)
    overhead_s = overhead_full_s()                       # CRIT 2b: loads re-enter ONCE, fixed
    total_s = ((anchor_s + gen_secret + gen_evoked + score_s + itf_s + dose_2a_s + expressed_s
                + rider_s + overhead_s) * SLACK + SETUP_S)
    return dict(anchor_s=anchor_s, gen_secret_s=gen_secret, gen_evoked_s=gen_evoked,
                score_s=score_s, itf_s=itf_s, dose_2a_s=dose_2a_s, expressed_s=expressed_s,
                rider_s=rider_s, overhead_s=overhead_s, total_s=total_s,
                unit_sources={k: ("subprocess-line" if k in units else "phase-delta-fallback")
                              for k in ("lrg14_s", "rider_last_s", "itf_per_stream_s")},
                projected_usd=total_s / 3600.0 * float(dph),
                note="named constants in run_lr_extend.py (14B-only phase list; CRIT 2: "
                     "per-unit rates from the subprocess timing lines exclude spawn/model-load "
                     "overhead, which re-enters once via overhead_s; the S2 gen slice is "
                     "bounded at the child's model-ready step so the one-time weights download "
                     "is never multiplied; smoke slices underfill batches, so stream-count "
                     "scaling is conservative); re-derive from real smoke timing before any "
                     "full launch (B10 discipline).")


def require_shakedown(shakedown_done, smoke=False, dry=False):
    """TECH M1 (2026-07-14 review): a FULL (non-smoke, non-dry) launch with no registered
    shakedown for the current code-hash is BLOCKED at $0 (exit 3) -- the old NOTE-and-continue
    could burn a full box on unshaken code. --smoke and --dry are unaffected (the smoke IS the
    shakedown; --dry spends nothing)."""
    if smoke or dry or shakedown_done:
        return True
    print(f"BLOCKED ($0): no registered shakedown for code-hash {code_hash()} -- a FULL run "
          f"never launches unshaken (TECH M1). Run --smoke first; a clean smoke registers the "
          f"hash in {SHAKEDOWN_REGISTRY}.", file=sys.stderr)
    sys.exit(3)


def check_projection(projected_usd, authorized_usd=RUN_AUTHORIZED_USD):
    if projected_usd > authorized_usd:
        return dict(go=False, message=(
            f"STOP: projected full-run spend ${projected_usd:.2f} exceeds the "
            f"${authorized_usd:.2f} authorization (prereg Q3, Matt 2026-07-13) -- structural "
            f"problem, not a budget question. Apply the REVISED trim order ({TRIM_ORDER}) only "
            "with a DISCLOSED recomputation, or ask Matt."))
    return dict(go=True, message=(
        f"GO: projected ${projected_usd:.2f} within the ${authorized_usd:.2f} authorization "
        "(prereg Q3, confirmed at freeze)."))


def hf_token():
    for src in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        if src:
            return src.strip()
    tok = Path.home() / ".cache" / "huggingface" / "token"
    if tok.exists():
        return tok.read_text().strip()
    raise SystemExit("no HF token found (env HF_TOKEN or ~/.cache/huggingface/token)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=DEFAULT_GPU, help="48GB tier (prereg Q5; L40S/6000Ada ok)")
    ap.add_argument("--min-vram", type=int, default=DEFAULT_MIN_VRAM)
    ap.add_argument("--max-spend", type=float, default=None)
    ap.add_argument("--max-dph", type=float, default=0.85, help="48GB-tier ceiling $/hr")
    ap.add_argument("--max-hours", type=float, default=None,
                    help=f"wall-clock cap -> provider deadman (default {MAX_HOURS_SMOKE} smoke "
                         f"/ {MAX_HOURS_FULL} full)")
    ap.add_argument("--min-bw", type=int, default=400, help="min host downlink Mbps (~56GB)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--trim", type=int, default=0,
                    help="trim ladder (Amendment 3, 2026-07-14): 1 = drop 70B-rider "
                         "secret_word (descriptive); 2 = also drop the evoked arms; 3 = also "
                         "drop the 70B-rider confirmatory cells (max 3). Part B, the 2a dose "
                         "curve and the 14B secret cells can NEVER be trimmed; the 2b "
                         "expressed cell is WITHDRAWN (not a trim level). Only with a "
                         "disclosed recomputation.")
    ap.add_argument("--unverified", action="store_true")
    ap.add_argument("--dry", action="store_true", help="validate gate+Spec at $0 (mock runner)")
    args = ap.parse_args()
    BOX.run_plan(args.trim)                        # validate NOW ($0): bad trim never launches
    gf = gate_fields(smoke=args.smoke, min_vram=args.min_vram)
    max_hours = args.max_hours or gf["max_hours"]
    max_spend = args.max_spend if args.max_spend is not None else gf["max_spend"]
    run_id = run_id_for(args.smoke)
    local_out = REPO / "runs" / ("lr_extend_smoke_box" if args.smoke else "lr_extend_box")
    disk_gb = disk_for_run()

    if not args.dry:
        # $0 launch gates: prereg freeze + HF input presence (checklist L1/L2).
        if not prereg_frozen():
            raise SystemExit("BLOCKED ($0): lr_scale_extend_prereg.md still carries the DRAFT "
                             "marker -- Matt freezes it (named calls + Q1-Q5) before any launch.")
        missing = preflight_hf_inputs()
        if missing:
            raise SystemExit("BLOCKED ($0): input bundles missing from the HF dataset "
                             f"(launch checklist L2): {missing} -- upload before create.")

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
    print(f"mid-run wakeup -> .venv-driver/bin/python -m labkit watch --events {events_path} "
          f"--status {status_path} --until warn", flush=True)

    job = labkit.script_job(
        workdir=str(REPO), entrypoint=entrypoint_for(args.smoke, args.trim),
        env={"INTRO_REPORT_DIR": "out", "HF_TOKEN": tok, "HF_HUB_DISABLE_XET": "1",
             **THREAD_CAPS},
        deps=deps_for(list(BOX.READERS)),
        ready="LRX_READY", done="LRX_DONE", fatal="LRX_FATAL",
        local_out=str(local_out), pull_subdir="out",
        setup_to=3600, stall_to=2700, run_to=int(max_hours * 3600))

    kwargs = dict(
        provider=labkit.VastProvider(**provider_kwargs(
            run_id, disk_gb, max_hours,
            throttle_path=labkit.default_vast_throttle_path())),
        gpu=args.gpu, min_vram_mb=gf["min_vram_mb"], image=gf["image"], pull_gb=4,
        est_run_s=int(max_hours * 1800),
        max_dph=args.max_dph, max_spend=max_spend, max_hours=max_hours,
        ledger_path=LEDGER_PATH,
        max_acquire_tries=8, max_setup_retries=3, require_verified=not args.unverified,
        mk={"min_reliability": 0.97, "min_inet_down": args.min_bw},
        status_path=str(status_path), on_event=str(events_path),
        job=job, run_id=run_id)

    spec = Spec(
        labkit_kwargs=kwargs, seed=0, data_revision=sha, labkit_tag=LABKIT_TAG,
        vram_est_mb=gf["vram_est_mb"],      # 14B gen peak (derivation at VRAM_EST_MB above)
        output_incremental=True,
        shakedown_done=gf["shakedown_done"],  # machine-checked: registered --smoke for this hash
        eng_review="SHIP", sci_review="SHIP",
    )
    decision = evaluate(facts_from_spec(spec), EXPERIMENT_SPEND_POLICY)
    print(f"gate: authorized={decision.authorized} reasons={list(decision.reasons)}", flush=True)
    expected = [r for r in decision.reasons if r.startswith("max_hours")]
    if expected and not args.smoke:
        print("NOTE: the max_hours reason is EXPECTED on the full run (one contiguous "
              f"{MAX_HOURS_FULL}h box; the policy routes over-ceiling runs to a human -- "
              "Matt's launch is the authorization, the run_lr_grid precedent).", flush=True)
    if not gf["shakedown_done"]:
        print(f"NOTE: no registered shakedown for code-hash {code_hash()} -- run --smoke first "
              f"(a clean smoke registers it in {SHAKEDOWN_REGISTRY}).", flush=True)
    # TECH M1: a FULL launch without a registered shakedown is a hard $0 block (exit 3);
    # smoke and --dry pass through (the smoke is the shakedown, --dry spends nothing).
    require_shakedown(gf["shakedown_done"], smoke=args.smoke, dry=args.dry)

    def _mock(**k):
        print(f"[DRY] would run: gpu={k['gpu']} min_vram={k['min_vram_mb']} "
              f"image={k['image']} disk={disk_gb}GB max_dph=${k['max_dph']} "
              f"max_spend=${k['max_spend']} max_hours={k['max_hours']} "
              f"run_id={k['run_id']} entry={entrypoint_for(args.smoke, args.trim)!r}",
              flush=True)
        return "DRY_OK"

    res = None
    for attempt in (1, 2):
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
            print("rsync-255 infra flake -- retrying once (on-box stages resume)", flush=True)
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
        h = code_hash()
        register_shakedown(h, info=dict(run_id=run_id, data_revision=sha,
                                        spend=getattr(res, "spend_usd", None)))
        print(f"shakedown registered for code-hash {h} -> {SHAKEDOWN_REGISTRY}", flush=True)
        log_path = getattr(res, "log_path", None)
        steps, log_text = [], None
        if log_path and os.path.exists(str(log_path)):
            with open(str(log_path)) as f:
                log_text = f.read()
            steps = parse_steps(log_text)
        proj = smoke_projection(steps, dph=args.max_dph, trim=args.trim, log_text=log_text)
        verdict = check_projection(proj["projected_usd"])
        print(f"\nsmoke spend projection (14B-only + Amendment-1 rider + Amendment-2 "
              f"dose/expressed): "
              f"anchor={proj['anchor_s']:.0f}s gen_secret={proj['gen_secret_s']:.0f}s "
              f"gen_evoked={proj['gen_evoked_s']:.0f}s score={proj['score_s']:.0f}s "
              f"inject_tf={proj['itf_s']:.0f}s dose_2a={proj['dose_2a_s']:.0f}s "
              f"expressed={proj['expressed_s']:.0f}s rider={proj['rider_s']:.0f}s "
              f"overhead={proj['overhead_s']:.0f}s "
              f"total={proj['total_s']:.0f}s -> ${proj['projected_usd']:.2f} at "
              f"${args.max_dph}/hr (ledger so far ${PROJECT_CAP - remaining_budget():.2f}; "
              f"authorization ${RUN_AUTHORIZED_USD:.2f}, Q3)", flush=True)
        print(f"unit sources (CRIT 2: subprocess-line = post-model-load clocks): "
              f"{proj['unit_sources']}", flush=True)
        print(verdict["message"], flush=True)
        print("score the smoke offline: .venv/bin/python -c \"import sys; "
              "sys.path.insert(0, 'experiments/exp2_output_monitorability/analysis'); "
              "import lr_extend_offline as L; "
              f"L.main(box_dir='{local_out}', smoke=True)\"", flush=True)
        if not verdict["go"]:
            sys.exit(3)
        return
    print("pulled -> runs/lr_extend_box/  (score offline: .venv/bin/python "
          "experiments/exp2_output_monitorability/analysis/lr_extend_offline.py)", flush=True)


if __name__ == "__main__":
    main()
