"""On-box orchestrator for the LR-72B run (prereg: reports/lr_72b_prereg.md). Does the frontier
scale point of the private-secret-channel grid keep growing at 72B?

ONE contiguous 2xH100(-NVL) box, self-hosting Qwen2.5-72B-Instruct under vLLM (tensor-parallel).
Unlike every prior LR box -- which loads an HF model directly and teacher-forces with OUR OWN
forward pass (lr_reader.score_batch) -- the 72B point is served by vLLM and both generation and
teacher-forcing go over HTTP:
  - GENERATION = vLLM /v1/completions (sampling) with the SAME anti-word prompt, word-free filter
    and acceptance gate as the exp3 collector (covert_collect.degeneracy/is_degenerate + primers_v3
    compose_system) -- reused, not reimplemented.
  - TEACHER-FORCING = vLLM /v1/completions with `prompt_logprobs` (src/lr_vllm), summing the
    per-prompt-token logprobs over the gibberish span under each context -> the exact LR quantity.

Stages (perf checklist: no manual phases between create and teardown):
  S0  start the vLLM server (subprocess), wait for /health, prefetch weights (144GB -> the driver
      picks a high-downlink host so this is minutes).
  S1  Phase 1 (ALWAYS): generate secret_word + secret_sustain, ~24/concept, via vLLM.
  GATE the $6 DECISION: Phase 2 (evoked + evoked_alt) runs ONLY IF spend_so_far + projected(evoked
      from the MEASURED Phase-1 $/arm rate) <= $6 (Matt 2026-07-12). Else disclosed skip-for-budget.
  S2  Phase 2 (conditional): generate evoked + evoked_alt if gated in.
  S3  teacher-force-score all generated arms (the 72B self-read DIAGONAL) via prompt_logprobs; one
      atomic shard per (set, ctx). The one OFF-DIAGONAL 7B privacy check is a SEPARATE small reader
      -- NOT teacher-forced under 72B here; the 72B secret streams are pulled home and scored by the
      existing HF lr_grid path offline (documented in OFFDIAG_7B_NOTE) -- cheapest and reuses the
      certified numerics.
  S3b OBSERVER cells (prereg Amendment 1): while the box is up the 72B ALSO teacher-forces the
      existing smaller-model streams (qwen2.5-{1.5b,3b,7b} x {secret_word, secret_sustain, evoked},
      from the completed scale-grid) under each arm's matched context vs neutral -- 9 scoring-only
      cells, the SAME lr_vllm currency as the diagonal, distinct observe_ shards. No generation, no
      ceiling change. Tests whether the model-private secret channel SURVIVES a frontier observer.

Utilization (Matt: don't rent that machine and underutilize it): the FIRST gen batch and the FIRST
score batch log tok/s + GPU util; a < 60% configuration HALTS the box (util_gate) before the hour
burns, to retune batch/tensor-parallel.

Markers LR72_READY / LR72_DONE / LR72_FATAL (collision-checked in tests: no marker a substring of
another across ALL box scripts). Driven by a gated harness driver (harness/run_lr_72b.py). NEVER
run on the Mac -- it starts a 144GB server.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
EXP3 = os.path.join(REPO, "experiments", "exp3_induction_and_scale")
SRC = os.path.join(REPO, "src")

# ---- frozen design (prereg) -----------------------------------------------------------------
MODEL_72B = "Qwen/Qwen2.5-72B-Instruct"
TP_SIZE = 2                                    # 2xH100 tensor-parallel
VLLM_PORT = 8000
VLLM_URL = f"http://127.0.0.1:{VLLM_PORT}"
PHASE1_ARMS = ("secret_word", "secret_sustain")     # ALWAYS (prereg Streams, Phase 1)
PHASE2_ARMS = ("evoked", "evoked_alt")              # CONDITIONAL (the $6 gate, Phase 2)
PHASE2_MAX_USD = 6.0                                # Matt 2026-07-12: concept-transfer IF <= $6
TARGET_CLEAN = 24                                   # ~24 word-free streams / concept (prereg)
MAX_GEN = 192                                       # reject-resample cap per (arm, concept)
GEN_TOKENS = 128                                    # stream length (matches the exp3 real-run cfg)
GEN_BATCH = 24                                      # vLLM continuous batching packs these
GEN_TEMP, GEN_TOP_P = 1.0, 0.98                     # exp3 gen config
PROMPT_LOGPROBS = 20                                # top-k prompt_logprobs; high enough that the
#                                                     teacher-forced token is always present (the
#                                                     smoke verifies -- span_logprobs raises if not)
UTIL_GATE_MIN = 60.0                                # prereg perf requirement: < 60% util halts
# The one off-diagonal privacy cell (7B reads the 72B secret streams, expect ~0): the 7B reader is
# a SEPARATE, SMALL model -- NOT this 72B server. Simplest + reuses certified numerics: pull the
# 72B secret streams home and score the 7B-on-72B cell OFFLINE with the existing HF lr_grid path
# (src/lr_grid.py, --reader qwen2.5-7b, a secret_word bundle whose model is qwen2.5-72b). This box
# does NOT teacher-force it under 72B; the driver/offline scorer owns it.
OFFDIAG_7B_NOTE = ("off-diagonal 7B-on-72B privacy cell scored OFFLINE via the existing HF "
                   "src/lr_grid.py path (reader=qwen2.5-7b) on the pulled 72B secret streams -- "
                   "NOT teacher-forced under the 72B vLLM server")

# B15 matched context-set codes (parity with lr_grid.SECRET_CTX; secret arms score under ONE
# matched ctx vs the arm-own neutral). evoked/evoked_alt keep the natural A/B/N grid.
SECRET_CTX = {"secret_word": "SW", "secret_sustain": "SS"}
CTX_SETS = {"evoked": ("N", "A", "B"), "evoked_alt": ("N", "A", "B"),
            "secret_word": ("N", "SW"), "secret_sustain": ("N", "SS")}

# ---- Amendment 1 (observer-scaling): the 72B ALSO OBSERVES the completed scale-grid streams -----
# While the box is up the 72B teacher-forces the existing smaller-model streams (SCORING-ONLY, no
# generation, no ceiling change): 3 generators x 3 arms = 9 observer cells, each under its OWN
# matched context vs neutral (the SAME lr_vllm currency as the diagonal). The open question is
# whether the model-private secret channel SURVIVES a frontier observer (predicted ~0); a positive
# is the adverse-scaling surprise (prereg Amendment 1). These are the smaller-model GENERATORS the
# 72B observes.
OBSERVE_GENERATORS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")
OBSERVE_ARMS = ("secret_word", "secret_sustain", "evoked")
# Each observer arm scores under its ONE matched context vs the arm-own neutral (secret arms:
# SW/SS; evoked: its natural "A" persona direction) -- one observer cell = one (gen, arm) pair.
OBSERVE_CTX = {"secret_word": "SW", "secret_sustain": "SS", "evoked": "A"}
# Source bundles for the observed streams (exp3 collector schema: {model, concepts, streams:[...]}),
# from the completed lr_scale_grid run. secret_word/evoked live in the top-level per-model _ind
# mirror; secret_sustain (the Amendment-2 arm) was generated by the grid box and lives in its _ind
# mirror. The driver pulls whichever exists; if a bundle is absent locally it is fetched from HF
# (the ErrareHumanumEst/internal-state-from-gibberish dataset) -- resolved by observe_bundle_path.
OBSERVE_BUNDLE_DIRS = (
    os.path.join(REPO, "runs", "_ind"),                       # secret_word + evoked (top-level)
    os.path.join(REPO, "runs", "lr_grid_box", "_ind"),        # secret_sustain (grid-box mirror)
)


def _sys():
    if SRC not in sys.path:
        sys.path.insert(0, SRC)
    if EXP3 not in sys.path:
        sys.path.insert(0, EXP3)


# ============================================================ S0: the vLLM server ==============
def serve_cmd():
    """`vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 2` (prereg model + TP) on the
    pinned port. --dtype bfloat16 matches every prior LR box's weight dtype."""
    return ["vllm", "serve", MODEL_72B, "--tensor-parallel-size", str(TP_SIZE),
            "--port", str(VLLM_PORT), "--dtype", "bfloat16"]


def wait_health(url=VLLM_URL, timeout_s=3600, opener=urllib.request.urlopen, proc=None):
    """Poll <url>/health until the server is up (the 144GB weight load is the real wait -- the
    driver picks a high-downlink host to keep it minutes). opener is injectable for tests.
    proc (optional): the vLLM subprocess; if it exits before /health responds the function raises
    immediately rather than burning up to timeout_s of billing on a dead server."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"vLLM process exited (exit code {proc.returncode}) before /health responded; "
                "likely OOM or startup error -- check the vLLM stderr log")
        try:
            r = opener(url + "/health", timeout=10)
            code = getattr(r, "status", None) or r.getcode()
            if int(code) == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vLLM /health did not come up within {timeout_s}s")


class VLLMClient:
    """Thin vLLM /v1/completions client (stdlib urllib; no requests dep so the CPU analysis venv
    can import this module for tests). generate() samples; completions() teacher-forces with
    prompt_logprobs. Both send TOKEN IDS as the prompt (tokenizer parity: vLLM scores exactly our
    ids). The transport is a plain POST; tests inject a mock client instead."""
    def __init__(self, url=VLLM_URL, model=MODEL_72B):
        self.url, self.model = url, model

    def _post(self, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.url + "/v1/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())

    def generate(self, prompt_ids, n=1, max_tokens=GEN_TOKENS, temperature=GEN_TEMP,
                 top_p=GEN_TOP_P):
        """Sample n continuations of the prompt ids -> list of generated-token-id lists."""
        body = dict(model=self.model, prompt=[int(x) for x in prompt_ids], n=int(n),
                    max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p),
                    logprobs=0)
        resp = self._post(body)
        out = []
        for c in resp["choices"]:
            ids = list(c.get("token_ids") or [])
            if not ids:
                raise RuntimeError(
                    "vLLM returned empty token_ids for a choice; check that --logprobs is set "
                    "and the vLLM version supports token_ids in completions responses")
            out.append(ids)
        return out

    def completions(self, prompt_ids, prompt_logprobs=PROMPT_LOGPROBS, **kw):
        """Teacher-force: return {'prompt_logprobs': [...]} for lr_vllm.span_logprobs."""
        import lr_vllm as V
        body = V.completions_request(self.model, prompt_ids, prompt_logprobs=prompt_logprobs)
        return self._post(body)


# ============================================================ util gate =========================
_UTIL_STATE = {"gen_done": False, "score_done": False}


def gpu_util_sample():
    """GPU util percent via nvidia-smi (present on every box image). None on CPU/test envs."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        vals = [float(x) for x in out.strip().splitlines() if x.strip()]
        return max(vals) if vals else None      # 2xH100: the busier card (TP splits the work)
    except Exception:
        return None


def util_gate(stage, tokens=None, secs=None, util=None):
    """Prereg perf requirement: the FIRST gen batch and the FIRST score batch log tok/s + GPU util;
    a < 60% configuration HALTS the box (raise -> the box FATALs) rather than burning the run at
    partial occupancy. util is injectable (tests); None -> log, never falsely halt. The message is
    marker/FATAL-substring safe."""
    if util is None:
        util = gpu_util_sample()
    tps = (float(tokens) / float(secs)) if tokens and secs else None
    print(f"LR72_UTIL {stage}: {'%.0f' % tps if tps is not None else '?'} tok/s "
          f"util={'%.1f' % util if util is not None else '?'}% (floor {UTIL_GATE_MIN:.0f}%)",
          flush=True)
    if util is not None and util < UTIL_GATE_MIN:
        raise RuntimeError(
            f"utilization gate: GPU util {util:.0f}% < {UTIL_GATE_MIN:.0f}% at the first {stage} "
            "batch -- halting the box; retune the batch size / tensor-parallel and relaunch "
            "(prereg perf requirement)")
    return None


# ============================================================ S1/S2: generation ================
def compose_arm_system(arm, concept):
    """Anti-word system prompt for one (arm, concept) -- primers_v3.compose_system, the SAME frozen
    composition the exp3 collector uses (secret_word sentence, secret_sustain template, evoked/alt
    personas). concept None -> the arm's own strength-0 neutral baseline."""
    _sys()
    import primers_v3 as P
    import config as C
    return P.compose_system(concept, C.STRONG_SYSTEM, arm=arm)


def generate_arm(client, tok, arm, concept, target_clean=TARGET_CLEAN, max_gen=MAX_GEN,
                 gen_batch=GEN_BATCH, first_gen=None):
    """Reject-resampled word-free generation for ONE (arm, concept) via the vLLM client. Reuses the
    exp3 acceptance gate VERBATIM: covert_collect.degeneracy/is_degenerate (the word-free filter),
    so the accepted pool is genuinely word-free -- nothing reimplemented. Returns the per-stream
    records (kept AND rejected, exp3 schema: gidx/concept/arm/tokens/text/accepted/strength).
    first_gen (a mutable [flag]) fires the util gate on the FIRST gen batch."""
    _sys()
    import config as C
    from covert_collect import degeneracy, is_degenerate
    system = compose_arm_system(arm, concept if concept != "neutral" else None)
    prompt_ids = list(tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": C.GEN_PROMPT}],
        add_generation_prompt=True))
    out, nclean, gidx = [], 0, 0
    strength = 0 if concept == "neutral" else 1
    while nclean < target_clean and len(out) < max_gen:
        b = min(gen_batch, max_gen - len(out))
        t0 = time.time()
        gens = client.generate(prompt_ids, n=b, max_tokens=GEN_TOKENS)
        if first_gen is not None and not first_gen[0]:
            first_gen[0] = True
            util_gate("gen", tokens=b * GEN_TOKENS, secs=max(time.time() - t0, 1e-6))
        for row in gens:
            ids = _strip_eos(row, getattr(tok, "eos_token_id", None))
            text = tok.decode(ids, skip_special_tokens=True)
            d = degeneracy(text)
            acc = not is_degenerate(d)
            nclean += int(acc)
            out.append(dict(gidx=gidx, concept=(concept if strength else "neutral"), arm=arm,
                            tokens=list(ids), text=text, deg=d, accepted=bool(acc),
                            strength=strength))
            gidx += 1
    print(f"LR72 gen {arm}/{concept}: {nclean}/{target_clean} clean ({len(out)} gen)", flush=True)
    return out


def _strip_eos(ids, eos_id):
    """Trailing eos (at most one) stripped -- streams are stored eos-stripped like the HF path; the
    eos-free rule then applies at score time. Keeps non-terminal special ids."""
    ids = [int(x) for x in ids]
    if eos_id is not None and ids and ids[-1] == int(eos_id):
        ids = ids[:-1]
    return ids


# ============================================================ the $6 decision gate =============
def phase2_gate(spend_so_far, phase1_arms, phase1_spend, max_usd=PHASE2_MAX_USD):
    """The frozen conditional-$6 gate (prereg; Matt 2026-07-12). Phase 2 generates the same number
    of arms as Phase 1 (2: evoked + evoked_alt), so projected(evoked) = (phase1_spend/phase1_arms)
    * len(PHASE2_ARMS) -- the MEASURED Phase-1 $/arm rate, never a fixed guess. GO iff spend_so_far
    + projected <= $6; else a disclosed skip-for-budget (report secret-only)."""
    per_arm = float(phase1_spend) / max(int(phase1_arms), 1)
    projected = per_arm * len(PHASE2_ARMS)
    total = float(spend_so_far) + projected
    go = total <= max_usd
    return dict(go=bool(go), projected_usd=projected, total_if_run=total, max_usd=max_usd,
                per_arm=per_arm,
                reason=("within the $%.2f Phase-2 budget" % max_usd if go else
                        "Phase 2 skipped for budget: spend_so_far $%.2f + projected(evoked) $%.2f "
                        "= $%.2f > $%.2f (report secret-only, disclosed)"
                        % (spend_so_far, projected, total, max_usd)))


# ============================================================ S3: teacher-force scoring ========
def score_shard_path(outdir, arm, ctxset):
    """One atomic shard per (arm/streamset, ctxset). '72b' reader is implicit (the diagonal)."""
    return os.path.join(str(outdir), f"qwen2.5-72b__qwen2.5-72b__{arm}_{ctxset}.pt")


def _concepts():
    _sys()
    import config as C
    return list(C.COVERT_CONCEPTS)


def score_arm(client, tok, arm, streams, outdir, first_score=None):
    """Teacher-force-score one arm's streams (the 72B self-read diagonal) via prompt_logprobs, one
    atomic shard per (arm, ctxset). For each ctxset the LR = LL(stream | matched persona) -
    LL(stream | neutral) is lr_vllm.lr_score over the SAME gibberish span (nothing numeric
    reimplemented -- vLLM computes the logprobs, lr_vllm aligns+sums). Records carry the raw LR
    difference under each context label; the offline scorer feeds them through the certified
    calibration. Resume-safe: an existing shard is skipped (LR72_SKIP)."""
    import lr_vllm as V
    _sys()
    concepts = _concepts()
    eos_id = getattr(tok, "eos_token_id", None)
    ind = [s for s in streams if s["strength"] == 1 and s.get("accepted")]
    for ctxset in CTX_SETS[arm]:
        shard = score_shard_path(outdir, arm, ctxset)
        if os.path.exists(shard):
            print(f"LR72_SKIP {os.path.basename(shard)} (resume)", flush=True)
            continue
        # matched labels: N -> the arm-own neutral only; A/B/SW/SS -> per-concept persona contexts.
        labels = ["neutral"] if ctxset == "N" else concepts
        neutral_system = compose_arm_system(arm, None)
        recs = []
        for s in ind:
            noeos = _strip_eos(s["tokens"], eos_id)
            rec = dict(gidx=s["gidx"], concept=s["concept"], strength=1,
                       T=len(s["tokens"]), T_noeos=len(noeos), ll={}, ll_tok={})
            for label in labels:
                ctx_concept = None if label == "neutral" else label
                ctx_system = compose_arm_system(_ctx_arm(arm, ctxset), ctx_concept)
                t0 = time.time()
                # per-token LR-diff vector rides the SAME two prompt_logprobs calls (no extra POST):
                # the offline Amendment-5 position control consumes it; the scalar LR is its sum.
                lr, pertok = V.lr_score(client, tok, ctx_system, neutral_system,
                                        _gen_prompt(), s["tokens"], prompt_logprobs=PROMPT_LOGPROBS,
                                        drop_last_eos=True, eos_id=eos_id, return_pertok=True)
                if first_score is not None and not first_score[0]:
                    first_score[0] = True
                    util_gate("score", tokens=len(s["tokens"]) * 2,
                              secs=max(time.time() - t0, 1e-6))
                rec["ll"][label] = float(lr)
                rec["ll_tok"][label] = _f16(pertok)
            recs.append(rec)
        _atomic_save(shard, dict(reader="qwen2.5-72b", generator="qwen2.5-72b", streamset=arm,
                                 ctxset=ctxset, contexts=labels,
                                 stream_tokenization="vllm-prompt_logprobs-eosfree",
                                 score="ll = LL(matched persona) - LL(neutral), eos-free primary; "
                                       "ll_tok = the same per-token LR-diff vector (position ctrl)",
                                 records=recs))
        print(f"LR72_SHARD_SAVED {os.path.basename(shard)} n={len(recs)}", flush=True)


# ============================================================ Amendment 1: observer cells =======
def observe_shard_path(outdir, gen, arm, ctxset):
    """One atomic OBSERVER shard per (generator, arm, ctxset): the 72B is the reader, the smaller
    model is the generator (an OFF-DIAGONAL cell), so the name carries BOTH slugs plus an observe_
    prefix on the streamset -- distinct from every 72B-self-read diagonal shard (reader==generator)
    and from every other observer cell."""
    return os.path.join(str(outdir), f"qwen2.5-72b__{gen}__observe_{arm}_{ctxset}.pt")


def observe_bundle_path(gen, arm, dirs=OBSERVE_BUNDLE_DIRS):
    """Resolve the LOCAL source bundle for one (generator, arm) observed stream set:
    <dir>/<gen>/data/<gen>-<arm>.pt across the known _ind mirrors (top-level for secret_word/evoked,
    the grid-box mirror for secret_sustain). Returns the first existing path, else the primary
    top-level path (the driver fetches it from HF if absent locally -- a launch dependency the
    driver surfaces). No I/O beyond existence checks."""
    rel = os.path.join(gen, "data", f"{gen}-{arm}.pt")
    for d in dirs:
        p = os.path.join(d, rel)
        if os.path.exists(p):
            return p
    return os.path.join(dirs[0], rel)


def load_observe_streams(gen, arm, path=None):
    """Load one smaller-model bundle's INDUCED (strength-1, accepted) streams for observation. The
    bundle is the exp3 collector schema ({model, concepts, streams:[{gidx,concept,tokens,strength,
    accepted,...}]}); we keep only the induced streams (the diagonal-parity selection score_arm
    uses). Returns (streams, concepts)."""
    import torch
    p = path or observe_bundle_path(gen, arm)
    b = torch.load(p, map_location="cpu", weights_only=False)
    concepts = list(b.get("concepts") or _concepts())
    ind = [dict(gidx=s["gidx"], concept=s["concept"], arm=arm,
                tokens=[int(x) for x in list(s["tokens"])],
                text=s.get("text", ""), accepted=bool(s.get("accepted", True)), strength=1)
           for s in b.get("streams", [])
           if s.get("strength") == 1 and s.get("accepted", True)]
    return ind, concepts


def score_observer(client, tok, gen, arm, streams, outdir, first_score=None):
    """Amendment-1 OBSERVER cell: the 72B teacher-forces one smaller-model's streams under the arm's
    ONE matched persona context vs the arm-own neutral, via the SAME lr_vllm.lr_score currency the
    diagonal uses (nothing new numeric -- this is scoring-only, no generation, no HF forward pass).
    Writes one atomic observe_ shard (reader=72B, generator=the smaller model). Resume-safe."""
    import lr_vllm as V
    _sys()
    concepts = _concepts()
    eos_id = getattr(tok, "eos_token_id", None)
    ctxset = OBSERVE_CTX[arm]
    shard = observe_shard_path(outdir, gen, arm, ctxset)
    if os.path.exists(shard):
        print(f"LR72_SKIP {os.path.basename(shard)} (resume)", flush=True)
        return
    neutral_system = compose_arm_system(_ctx_arm(arm, ctxset), None)
    ind = [s for s in streams if s.get("strength") == 1 and s.get("accepted", True)]
    recs = []
    for s in ind:
        noeos = _strip_eos(s["tokens"], eos_id)
        rec = dict(gidx=s["gidx"], concept=s["concept"], strength=1,
                   T=len(s["tokens"]), T_noeos=len(noeos), ll={}, ll_tok={})
        for label in concepts:                       # the matched persona under each concept
            ctx_system = compose_arm_system(_ctx_arm(arm, ctxset), label)
            t0 = time.time()
            lr, pertok = V.lr_score(client, tok, ctx_system, neutral_system, _gen_prompt(),
                                    s["tokens"], prompt_logprobs=PROMPT_LOGPROBS,
                                    drop_last_eos=True, eos_id=eos_id, return_pertok=True)
            if first_score is not None and not first_score[0]:
                first_score[0] = True
                util_gate("score", tokens=len(s["tokens"]) * 2,
                          secs=max(time.time() - t0, 1e-6))
            rec["ll"][label] = float(lr)
            rec["ll_tok"][label] = _f16(pertok)
        recs.append(rec)
    _atomic_save(shard, dict(reader="qwen2.5-72b", generator=gen, streamset=arm, ctxset=ctxset,
                             contexts=concepts, observer=True,
                             stream_tokenization="vllm-prompt_logprobs-eosfree",
                             score="ll = LL(matched persona) - LL(neutral), eos-free primary; "
                                   "ll_tok = the same per-token LR-diff vector (position ctrl)",
                             records=recs))
    print(f"LR72_SHARD_SAVED {os.path.basename(shard)} n={len(recs)}", flush=True)


def _ctx_arm(arm, ctxset):
    """The arm whose persona composes the matched context. secret arms use their own arm; the
    evoked/alt natural A/B directions map to the evoked/evoked_alt personas."""
    if ctxset == "A":
        return "evoked"
    if ctxset == "B":
        return "evoked_alt"
    return arm                                  # SW/SS/N: the arm's own composition


def _gen_prompt():
    _sys()
    import config as C
    return C.GEN_PROMPT


def _f16(vec):
    """Store per-token LR-diff vectors as fp16 (Amendment 1's registered per-token storage dtype;
    the fp32 summed LR in ll[] stays the scoring source of truth)."""
    import numpy as np
    return np.asarray(vec, dtype=np.float16)


def _atomic_save(path, obj):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)                        # atomic: presence = done


# ============================================================ progress + heartbeat =============
_T0 = time.time()


def emit_step(step, **fields):
    print("LABKIT_STEP " + json.dumps({"step": int(step), "t": int(time.time() - _T0), **fields}),
          flush=True)


def start_heartbeat(period_s=120):
    def beat():
        while True:
            time.sleep(period_s)
            print(f"HEARTBEAT t={int(time.time() - _T0)}s", flush=True)
    threading.Thread(target=beat, daemon=True).start()


# ============================================================ main =============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny slice: 2-3 streams, verify prompt_logprobs teacher-forces + "
                         "tokenizer parity + util logging (no $6 Phase 2, no full pools)")
    args = ap.parse_args()
    _sys()
    import config as C
    from common import load_model             # tokenizer only; the model is vLLM's, not ours

    outdir = os.path.abspath(os.environ.get("INTRO_REPORT_DIR") or os.path.join(REPO, "out"))
    grid_dir = os.path.join(outdir, "lr_72b")
    os.makedirs(grid_dir, exist_ok=True)
    start_heartbeat()

    # S0: start the server, wait for /health. The tokenizer is loaded LOCALLY (CPU) for rendering +
    # the parity gate; the 72B weights live only in the vLLM process.
    emit_step(0, phase="S0_serve", smoke=bool(args.smoke))
    proc = subprocess.Popen(serve_cmd(), env={**os.environ})
    try:
        wait_health(proc=proc)
        print("LR72_READY", flush=True)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_72B)
        client = VLLMClient()

        target = 3 if args.smoke else TARGET_CLEAN
        concepts = C.COVERT_CONCEPTS[:2] if args.smoke else C.COVERT_CONCEPTS
        first_gen, first_score = [False], [False]

        # S1: Phase 1 secret arms (ALWAYS).
        emit_step(1000, phase="S1_phase1_gen")
        pools = {}
        for arm in PHASE1_ARMS:
            recs = []
            for c in concepts:
                recs += generate_arm(client, tok, arm, c, target_clean=target,
                                     first_gen=first_gen)
            recs += generate_arm(client, tok, arm, "neutral", target_clean=target,
                                 first_gen=first_gen)
            pools[arm] = recs

        # DECISION GATE: Phase 2 only if it stays within $6 at the MEASURED Phase-1 rate.
        phase1_spend = _measured_spend()
        emit_step(2000, phase="phase2_gate")
        dec = phase2_gate(spend_so_far=phase1_spend, phase1_arms=len(PHASE1_ARMS),
                          phase1_spend=phase1_spend)
        print("LR72 phase-2 gate: " + json.dumps(dec), flush=True)
        if dec["go"] and not args.smoke:
            emit_step(3000, phase="S2_phase2_gen")
            for arm in PHASE2_ARMS:
                recs = []
                for c in concepts:
                    recs += generate_arm(client, tok, arm, c, target_clean=target,
                                         first_gen=first_gen)
                recs += generate_arm(client, tok, arm, "neutral", target_clean=target,
                                     first_gen=first_gen)
                pools[arm] = recs
        else:
            print("LR72 Phase 2 skipped: " + dec["reason"], flush=True)

        # S3: teacher-force-score all generated arms (the 72B self-read diagonal).
        emit_step(4000, phase="S3_score")
        import lr_vllm as V
        for arm, recs in pools.items():
            ind = [s["tokens"] for s in recs if s["strength"] == 1 and s.get("accepted")][:32]
            if ind:
                V.assert_prompt_roundtrips(tok, ind)     # tokenizer-parity gate before scoring
            score_arm(client, tok, arm, recs, grid_dir, first_score=first_score)
        # The 7B off-diagonal privacy cell is scored OFFLINE (OFFDIAG_7B_NOTE) -- also persist the
        # raw 72B secret streams for that pull.
        _atomic_save(os.path.join(grid_dir, "qwen2.5-72b-secret_streams.pt"),
                     dict(model="qwen2.5-72b", pools={a: pools[a] for a in pools},
                          offdiag_7b=OFFDIAG_7B_NOTE))

        # S3b (Amendment 1): the 72B OBSERVES the completed smaller-model streams -- scoring-only,
        # rides the box that is already up (no generation, no ceiling change). 3 generators x 3 arms
        # = 9 observer cells. Smoke touches ONE cell (72B scores a few 1.5B secret_word streams) so
        # the smoke verifies the observer path too. A missing bundle is disclosed and skipped (never
        # fabricated); the driver pulls them (local _ind or HF) before serving.
        emit_step(5000, phase="S3b_observe")
        obs_cells = ([("qwen2.5-1.5b", "secret_word")] if args.smoke
                     else [(g, a) for g in OBSERVE_GENERATORS for a in OBSERVE_ARMS])
        for gen, arm in obs_cells:
            bp = observe_bundle_path(gen, arm)
            if not os.path.exists(bp):
                print(f"LR72 observe skip {gen}/{arm}: bundle not present at {bp} "
                      "(pull it local/HF and rerun; the cell is resume-safe)", flush=True)
                continue
            streams, _ = load_observe_streams(gen, arm, path=bp)
            if args.smoke:
                streams = streams[:3]
            if streams:
                V.assert_prompt_roundtrips(tok, [s["tokens"] for s in streams][:32])
            score_observer(client, tok, gen, arm, streams, grid_dir, first_score=first_score)

        emit_step(9000, phase="lr72_done", smoke=bool(args.smoke))
        print(f"outputs OK in {grid_dir}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()


def _measured_spend():
    """Phase-1 spend so far, read from the run status/ledger the driver writes (INTRO_SPEND_USD is
    the driver-injected running figure; absent -> 0.0, and the gate then treats Phase 2 as free,
    which the driver's own $10 cap still bounds)."""
    v = os.environ.get("INTRO_SPEND_USD")
    try:
        return float(v) if v is not None else 0.0
    except ValueError:
        return 0.0


if __name__ == "__main__":
    try:
        main()
        print("LR72_DONE", flush=True)
    except Exception:
        print("LR72_FATAL", flush=True)
        raise
