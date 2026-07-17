"""LR scale-grid reader (prereg: experiments/exp2_output_monitorability/reports/
lr_scale_grid_prereg.md) -- the multi-reader x multi-generator extension of the certified 1.5B LR
instrument.

EXTENDS src/lr_reader.py WITHOUT modifying it: every numeric (float32 log-softmax LL sums, KV
prefill + registered self-check, the concat reference path, padding) is lr_reader's own certified
function, CALLED from here -- nothing is reimplemented (guarded by test: this module defines no
log-softmax/gather numerics of its own).

New over the 1.5B run:
- reader != generator: the reader is picked by --reader from GRID_READERS (never INTRO_MODEL --
  Llama slugs are not in the config registry); stream bundles arrive as repeated
  --bundle <gen_slug>:<streamset>:<path> specs.
- B3 shared-tokenizer gate: before teacher-forcing SAVED Qwen token ids under a different-size
  Qwen reader, assert both tokenizers encode a sample of the actual stream texts identically;
  any mismatch raises (box FATAL) -- scoring saved ids under a differing tokenizer is meaningless.
- B5 eos rule (registered): PRIMARY scoring excludes the terminal eos from all LL sums; the
  with-eos SECONDARY is computed alongside from the SAME forward (comparability with the 1.5B
  numbers). Records carry ll (eos-free) and ll_eos.
- B11 (prereg Amendment 1, Blocker 1): records ALSO carry ll_tok -- per-token LL vectors (fp16,
  with-eos length; numerator and denominator alike, keyed by context label) captured from the
  SAME single forward via the same scoped swap, so eos-free / with-eos / prefix-K readouts are
  all offline-computable from a shard. The per-token values are LR.ll_from_logits' own output
  (each position scored as its own length-1 stream); nothing numeric is reimplemented.
- A1 (B4/B13, prereg Amendment 1 ADJUDICATED): Llama cells render the context in the READER'S
  OWN chat template (system = persona/neutral text, user = GEN_PROMPT, generation header on),
  date_string PINNED (LLAMA_DATE), with a render-diff assert per scored context; the raw-text
  render is the registered robustness SECONDARY (every Llama cell also produces a _raw shard).
  Streams: trailing eos STRIPPED, decoded with the GENERATOR tokenizer, re-tokenized with the
  reader's own; per-stream round-trip failures are EXCLUDED and COUNTED (never FATAL -- the >5%
  void rule is the offline scorer's). Gate 4's English-prose control cell (PROSE_CONTROL) rides
  every Llama reader before its grid cells.
- B8 seam: util_gate_hook() is the registered smoke-shard utilization gate's hook point (no-op
  until B8).
- Amendment 4 (2026-07-11, pre-data for every cross-family cell): the cross-family readers are
  tiiuae/Falcon3-{1B,3B,7B}-Instruct -- the REGISTERED ungated fallback, fired after the box's
  HF 403 on meta-llama (license never actually accepted). Every A1/"Llama" rule in this module
  applies unchanged with "Llama" read as "Falcon3"; the Llama-era names (llama_ctx_ids,
  llama_streams, LLAMA_DATE, ...) are RETAINED to keep the review surface minimal -- they now
  serve the falcon slugs. Falcon3 facts verified locally (transformers 4.46.3, 2026-07-11):
  model_type=llama (loads under the pin), eos = <|endoftext|> (id 11), template =
  <|system|>\\n..\\n<|user|>\\n..\\n<|assistant|>\\n with NO date injected -- apply_chat_template
  ACCEPTS AND IGNORES date_string, so the LLAMA_DATE pin is a harmless no-op (retained); the
  (system, user, assistant) render is prefix-stable.
- B15 (prereg Amendment 2): the secret stream sets (secret_word / secret_sustain /
  maintained_secret) score under ONE matched ctx set each (SW/SS/SM: the frozen arm composition
  reconstructed via primers_v3 -> primers_v2 -> primers) vs the arm's OWN s0 baseline as neutral
  ("The secret word is paper." [+ suffix]) -- see SECRET_ARMS. Same certified numerics, eos rule,
  per-token shards, and A1 Llama rendering (render-diff assert included) as every other cell.

RESUMABLE: one atomic shard per (reader, generator, streamset, ctxset):
$INTRO_RUN_DIR/lr_grid/<reader>__<gen>__<set>_<ctx>.pt; existing shard => skip (LRG_SKIP).
box_lr_grid.py OWNS the box ready/done/fatal markers ("LRG_" + READY/DONE/FATAL); those strings
must not appear ANYWHERE in this file, prints or comments (labkit substring-matches markers on
log lines; the LR attempt-4 collision) -- guarded by test_lr_grid S5, M1 parity.
Run on GPU via the orchestrator; NEVER on the Mac.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import config as C
import common as K
import lr_reader as LR

# Readers (prereg: 6). Qwen hf_ids from the config registry (single source of truth); the
# cross-family slugs are grid-local -- they are READERS only, never collectors, so they stay out
# of C.MODELS (whose entries carry injection/collection fields these models will never use).
# Amendment 4: Falcon3 Instruct (ungated) replaced the 403-gated meta-llama readers pre-data.
GRID_READERS = {
    "qwen2.5-1.5b": C.MODELS["qwen2.5-1.5b"]["hf_id"],
    "qwen2.5-3b": C.MODELS["qwen2.5-3b"]["hf_id"],
    "qwen2.5-7b": C.MODELS["qwen2.5-7b"]["hf_id"],
    "falcon3-1b": "tiiuae/Falcon3-1B-Instruct",
    "falcon3-3b": "tiiuae/Falcon3-3B-Instruct",
    "falcon3-7b": "tiiuae/Falcon3-7B-Instruct",
}
TOKCHECK_SAMPLE = 32                                   # B3: texts per generator to cross-encode


def family(slug):
    """Reader/generator family: 'falcon3' (Amendment 4; any non-qwen) routes to the A1
    cross-family context seam -- the Llama-era llama_* path below -- 'qwen' to the certified
    teacher-forcing path."""
    return "falcon3" if slug.startswith("falcon3") else "qwen"


def assert_shared_tokenizer(reader_tok, source_tok, texts, n=TOKCHECK_SAMPLE):
    """B3 gate (registered): before teacher-forcing SAVED token ids under a reader that is not
    the generating model, assert both tokenizers encode a sample of the ACTUAL stream texts (real
    gibberish, not prose) to identical ids. Identical encodes => the saved ids are valid
    teacher-forcing targets for the reader; any mismatch is terminal (raise -> the traceback
    FATALs the box, by design). The message itself stays marker/FATAL-substring safe."""
    for i, text in enumerate(texts[: max(1, int(n))]):
        a = list(reader_tok(text, add_special_tokens=False).input_ids)
        b = list(source_tok(text, add_special_tokens=False).input_ids)
        if a != b:
            d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]),
                     min(len(a), len(b)))
            raise RuntimeError(
                f"shared-tokenizer gate failed on sample {i}: reader ids != source ids "
                f"(lens {len(a)} vs {len(b)}, first divergence at token {d}) -- teacher-forcing "
                f"the saved ids under this reader would be meaningless")


# ------------------------------------------------------------------ B5: registered eos rule
def noeos_lens(token_lists, eos_id):
    """PRIMARY (eos-free) lengths: a stream whose LAST token is the tokenizer eos loses exactly
    that one position from its LL sum (the 1.5B run's flagged artifact: eos probability under the
    persona contexts is context-correlated, not concept evidence). Non-terminal eos ids are part
    of the saved stream and stay. eos_id None -> lengths unchanged."""
    out = []
    for t in token_lists:
        arr = np.asarray(t).reshape(-1)
        L = int(arr.size)
        drop = eos_id is not None and L > 0 and int(arr[-1]) == int(eos_id)
        out.append(L - 1 if drop else L)
    return out


def score_batch_dual(model, ctx, kv, first, batch, lens, lens_noeos, use_kv, pertok=None):
    """Both LL sums -- eos-free PRIMARY over `lens_noeos`, with-eos SECONDARY over `lens` -- from
    ONE forward. The forward and the numerics are lr_reader's certified paths, unmodified:
    LR.score_batch runs as-is (KV-reuse with the attempt-6 mid-run concat fallback), and
    LR.ll_from_logits (the certified float32 log-softmax masked sum) is evaluated on the same
    pred logits -- once per length vector -- via a scoped, exception-safe swap of the module
    global. Nothing numeric is duplicated here; the alternative (a second forward per batch)
    would double the grid's paid compute for identical numbers.

    B11 (Amendment 1, Blocker 1): pass `pertok` (a dict) to ALSO capture the per-token LL matrix
    from the same forward -- pertok["ll"] = [B, Tmax] float32, real positions carry that token's
    LL, padded positions 0. The values are the certified function's own: every position is scored
    as its own length-1 stream (a pure reshape of the SAME pred logits), so no numeric is
    reimplemented and no extra forward runs.

    Returns (ll_eosfree, ll_witheos, use_kv)."""
    stash = {}
    orig = LR.ll_from_logits

    def _dual(pred_logits, targets, lengths):
        if pertok is not None:
            B, T = targets.shape
            pos = (torch.arange(T, device=targets.device).unsqueeze(0)
                   < torch.as_tensor(lens, device=targets.device).unsqueeze(1))
            stash["pertok"] = orig(pred_logits.reshape(B * T, 1, -1),
                                   targets.reshape(B * T, 1),
                                   pos.reshape(B * T).long()).reshape(B, T)
        stash["noeos"] = orig(pred_logits, targets, lens_noeos)
        return orig(pred_logits, targets, lengths)

    LR.ll_from_logits = _dual
    try:
        # If the KV path raises mid-forward, LR.score_batch falls back to the concat path, which
        # re-enters _dual -- the stash (per-token included) is then the concat pass's, matching
        # the returned ll.
        ll_eos, use_kv = LR.score_batch(model, ctx, kv, first, batch, lens, use_kv)
    finally:
        LR.ll_from_logits = orig
    if pertok is not None:
        pertok["ll"] = stash["pertok"]
    return stash["noeos"], ll_eos, use_kv


def grid_records(streams, eos_id=None):
    """Per-stream shard records: ll = eos-free PRIMARY sums, ll_eos = with-eos SECONDARY, ll_tok =
    per-token fp16 LL vectors (B11; with-eos length), all keyed by context label. T_noeos = the
    eos-free length (B11: lets the offline scorer slice the stored vector without re-deriving the
    eos rule)."""
    tn = noeos_lens([s["tokens"] for s in streams], eos_id)
    return [dict(gidx=s["gidx"], concept=s.get("concept"), strength=s.get("strength"),
                 T=len(s["tokens"]), T_noeos=int(tn[i]), ll={}, ll_eos={}, ll_tok={})
            for i, s in enumerate(streams)]


def record_lls(recs, lo, label, ll_free, ll_eos):
    for i, (a, b) in enumerate(zip(ll_free.cpu().tolist(), ll_eos.cpu().tolist())):
        recs[lo + i]["ll"][label] = float(a)
        recs[lo + i]["ll_eos"][label] = float(b)


def record_toks(recs, lo, label, pertok_ll, lens):
    """B11: store each stream's per-token LL vector (TRUE with-eos length, fp16) under its context
    label. fp16 is the registered storage dtype (Amendment 1); the certified fp32 SUMS in ll /
    ll_eos remain the scoring source of truth -- the vectors serve the offline secondaries."""
    mat = pertok_ll.cpu().numpy()
    for i in range(len(lens)):
        recs[lo + i]["ll_tok"][label] = mat[i, : int(lens[i])].astype(np.float16)


# ------------------------------------------------------------------ B2: grid cells + batching
# Conservative VRAM-aware defaults (bf16 weights + KV for ~25 contexts x batch streams on the
# 48GB tier; every value <= the certified 1.5B run's 16). B8's smoke re-derives real batch sizes
# from MEASURED headroom (the MC batch=8 lesson: don't let a conservative default burn 2/3 of a
# card on the full run) -- these only need to not-OOM the first shard.
BATCH_BY_SIZE = {"1b": 16, "1.5b": 16, "3b": 12, "7b": 8, "8b": 8}

# Reader-keyed KV pin (TECH M2, 2026-07-14 review): a reader listed here SKIPS the once-per-
# process KV self-check and scores on the pinned path (False = the concat reference path).
# Registered use: the 14B reader's self-check straddles the 0.02 tolerance (0.02056 / 0.05228
# observed across processes), making the scoring path a per-process coin flip with up-to-tol
# cell differences -- lr_grid_extend.register() pins it to concat. Smaller readers stay
# unlisted and keep the certified self-check. Disclosed in every shard meta (kv_pinned).
KV_PIN = {}


def default_batch(slug):
    return BATCH_BY_SIZE.get(slug.split("-")[-1].lower(), 8)


UTIL_GATE_MIN = 50.0          # registered floor (prereg perf checklist 4)
# 2026-07-14 review CRIT 1: shards below this scored-token count never count as "the first
# full shard" -- smoke slices (--limit; ~0.4-2k tokens) and 1-context N slivers (~15-18k on
# the real pools) sample the trailing-1s util window mostly idle and repeatably false-halted
# the rider smoke at 44% ($0.21/attempt). Full 12-context shards run ~200k+ tokens (~10x the
# floor), so the gate still polices every reader's real work: it fires on the first shard
# that CLEARS the floor. Same exemption shape as the __prose__ control-cell skip below.
UTIL_GATE_TOKEN_FLOOR = 20000
_UTIL_STATE = {"first_done": False}      # per-process = per reader (one subprocess per reader)


def gpu_util_sample():
    """GPU utilization percent via nvidia-smi (present on every box image; torch.cuda's own
    utilization query would add a pynvml dep). Sampled immediately after the shard's last
    forward: the counter reports the trailing ~1s window, which is still shard work. None when
    unavailable (CPU/test environments) -- the gate then logs instead of falsely halting."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=10)
        vals = [float(x) for x in out.strip().splitlines() if x.strip()]
        return vals[0] if vals else None
    except Exception:
        return None


def util_gate_hook(stage, shard=None, t=None, tokens=None, secs=None):
    """B8 body (prereg perf checklist 4): the FIRST full shard per reader logs tokens/sec + GPU
    util, and a <UTIL_GATE_MIN% configuration HALTS the grid (raise -> box FATAL) rather than
    burning the full run at partial occupancy (the MC batch=8 lesson). One-shot per process --
    each reader runs in its own subprocess, so every reader's first shard is gated. Gate-4
    prose-control shards are tiny (12 streams) and never count as 'the first full shard'.
    main() calls this at shard boundaries (B2 wired the call sites); --batch is the plumbed
    remedy for a halt."""
    if stage != "shard_done" or _UTIL_STATE["first_done"]:
        return None
    if shard is not None and "__prose__" in str(shard):
        return None                                    # gate-4 control cell: not a real shard
    if tokens is not None and float(tokens) < UTIL_GATE_TOKEN_FLOOR:
        print(f"LRG_UTIL skip {shard}: {int(tokens)} tokens < {UTIL_GATE_TOKEN_FLOOR} floor "
              "(smoke slice / 1-context sliver -- not a full shard; the gate waits for one)",
              flush=True)
        return None                                    # CRIT 1: doesn't consume the slot
    _UTIL_STATE["first_done"] = True
    util = gpu_util_sample()
    tps = (float(tokens) / float(secs)) if tokens and secs else None
    print(f"LRG_UTIL first shard {shard}: "
          f"{'%.0f' % tps if tps is not None else '?'} tok/s "
          f"util={'%.1f' % util if util is not None else '?'}%"
          f" (floor {UTIL_GATE_MIN:.0f}%)", flush=True)
    if util is not None and util < UTIL_GATE_MIN:
        raise RuntimeError(
            f"utilization gate: GPU util {util:.0f}% < {UTIL_GATE_MIN:.0f}% after the first "
            f"full shard ({shard}) -- halting the grid; raise the per-reader batch via --batch "
            "and relaunch (registered perf checklist 4)")
    return None


def shard_path(outdir, reader, gen, streamset, ctxset, render="template"):
    """One atomic shard per grid cell. Double-underscore delimits the two model slugs (each
    contains single '-' and '.'), so reader/generator/set/ctx parse back unambiguously. The
    raw-text robustness SECONDARY (A1; Llama readers only) gets a _raw suffix -- a marked,
    separate file, never overwriting the primary."""
    suffix = "_raw" if render == "raw" else ""
    return outdir / f"{reader}__{gen}__{streamset}_{ctxset}{suffix}.pt"


# B15 (prereg Amendment 2): the secret stream sets and their matched context sets. SW/SS/SM are
# the ctx-set codes (shard-name suffixes); each maps to the frozen arm whose collection
# composition the matched contexts reconstruct. The NEUTRAL for these cells is the arm's OWN
# strength-0 baseline ("The secret word is paper." [+ the sustain suffix]) -- the context the s0
# streams were actually generated under, and the one-token-different denominator Amendment 2's
# minimal-wording cell describes (registered pin; the evoked NEUTRAL persona would be a register
# mismatch for this arm family).
SECRET_ARMS = {"SW": "secret_word", "SS": "secret_sustain", "SM": "maintained_secret"}
SECRET_CTX = {v: k for k, v in SECRET_ARMS.items()}      # streamset -> matched ctx-set code
# Control (b): the INJECTED stream set. The exp1 covert_collect.pt capture (inject == 'gen',
# per-stream concept + tokens, strength == smax), scored under the concept's NATURAL PERSONA
# contexts (A/B wordings) vs neutral -- the SAME construction the certified 1.5B LR run used
# (lr_reader.select_streams("injected") + LR.CTX_SETS; injected x A read ~0.002 at 1.5B). NOT a
# secret arm: it routes through the natural A/B/N grid, never a "the secret word is X" context.
INJECTED_SET = "injected"
BUNDLE_SETS = ("evoked", "evoked_alt", INJECTED_SET) + tuple(SECRET_CTX)


def parse_bundle_spec(spec):
    """'<gen_slug>:<streamset>:<path>' -> (gen, streamset, path). Colons cannot appear in the
    slugs or POSIX paths we use, and the path keeps any later colons via maxsplit."""
    gen, streamset, path = spec.split(":", 2)
    if streamset not in BUNDLE_SETS:
        raise ValueError(f"bundle spec {spec!r}: streamset must be one of {'|'.join(BUNDLE_SETS)}")
    return gen, streamset, path


def _primers_v3():
    """exp3's primers_v3, lazily (the Amendment 2 secret_sustain composition lives there; it
    delegates to the frozen primers_v2 -> primers chain for every earlier arm)."""
    exp3 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "experiments", "exp3_induction_and_scale")
    if exp3 not in sys.path:
        sys.path.insert(0, exp3)
    import primers_v3
    return primers_v3


def grid_context_system(ctxset, concept):
    """Context system text for one grid cell. A/B/N delegate to lr_reader's CERTIFIED
    context_system verbatim (identical objects, nothing re-derived); the Amendment 2 secret
    ctxsets (SW/SS/SM) compose through the frozen primers chain exactly as at collection, with
    concept None -> the arm's OWN s0 baseline (see SECRET_ARMS note)."""
    if ctxset in SECRET_ARMS:
        P3 = _primers_v3()
        c = None if concept in (None, "neutral") else concept
        return P3.compose_system(c, C.STRONG_SYSTEM, arm=SECRET_ARMS[ctxset])
    return LR.context_system(ctxset, concept)


# ------------------------------------------------------------------ B4/B13: A1 xfam rendering
# Llama 3.1/3.2 templates inject "Today Date" from a drifting default when `date_string` is not
# supplied. PINNED (Amendment 1): unpinned, the numerator and denominator of a ratio rendered on
# different days -- or any rerun -- would differ OUTSIDE the persona text. The value is the Llama
# template's own documented fallback constant, chosen for template-idiomatic formatting.
# Amendment 4: Falcon3's template injects NO date and apply_chat_template ACCEPTS AND IGNORES the
# date_string kwarg (verified under the 4.46.3 pin, 2026-07-11) -- the pin is retained as a
# harmless no-op, still guarding any future reader whose template does inject one.
LLAMA_DATE = "26 Jul 2024"
# The registered raw-text robustness SECONDARY (register-OOD by design; disagreement with the
# template primary is disclosed, the primary governs).
RAW_CTX_FMT = "{system}\n\n{user}\n\n"

# Gate 4 (prereg): the small English-prose control set for the Llama sanity check -- one pinned
# concept-expressing sentence per concept. The prereg registered the GATE (Llama LL must rank
# matched > mismatched contexts on English prose before any gibberish cell is interpreted); the
# texts themselves were not frozen there, so they are pinned HERE, pre-data (disclosed).
PROSE_CONTROL = {
    "celebration": "The whole street burst into cheers as confetti rained over the parade and "
                   "the band struck up a joyful tune.",
    "ocean": "Salt spray drifted over the waves as the tide rolled endlessly against the shore "
             "under gulls circling the deep blue water.",
    "fear": "Her heart pounded in the dark hallway; every creak of the floorboards made her "
            "flinch and hold her breath in dread.",
    "silence": "Not a sound stirred in the empty library; the hush was so complete she could "
               "hear her own pulse.",
    "deception": "He smiled warmly while feeding them the false story, covering every lie with "
                 "another carefully planted excuse.",
    "obedience": "The recruits followed every order instantly and without question, marching "
                 "exactly as they were commanded.",
    "debugging": "She stepped through the stack trace line by line, adding print statements "
                 "until the failing branch finally revealed the bug.",
    "security": "The vault sat behind steel doors, biometric locks and armed guards, every "
                "access logged and every badge checked twice.",
    "curiosity": "The child kept asking why, prying open the clock to see the gears and "
                 "wondering what made each piece tick.",
    "anger": "He slammed his fist on the table, face flushed and voice rising into a furious "
             "shout he could barely control.",
    "warmth": "She wrapped the blanket around them by the fire, cocoa in hand, and the whole "
              "room felt soft and kind.",
    "loneliness": "The apartment was empty again that night; he ate alone by the window, "
                  "missing voices that never called anymore.",
}


def llama_ctx_ids(tok, ctxset, concept, device, render="template"):
    """A1 ADJUDICATED rendering (prereg Amendment 1, 2026-07-11). PRIMARY render='template': the
    READER'S OWN chat template over (system = persona/neutral text, user = GEN_PROMPT) with the
    generation header on -- mirroring collection structure; the decoded stream text is then
    teacher-forced as the assistant content (main re-tokenizes it; assert_llama_prefix_stable
    guards that this equals the full [system, user, assistant] render). date_string PINNED
    (LLAMA_DATE). SECONDARY render='raw': RAW_CTX_FMT, no template."""
    system = grid_context_system(ctxset, concept)
    if render == "raw":
        ids = tok(RAW_CTX_FMT.format(system=system, user=C.GEN_PROMPT),
                  return_tensors="pt").input_ids
        return ids.to(device)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": C.GEN_PROMPT}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, date_string=LLAMA_DATE,
                                  return_tensors="pt")
    ids = out if torch.is_tensor(out) else out["input_ids"]
    return ids.to(device)


def assert_render_diff(tok, ctxset, concept, render="template"):
    """A1 render-diff assert: the numerator (persona) and denominator (neutral) context renders
    must differ ONLY in the persona text -- the persona appears exactly once, and substituting it
    with the neutral text reproduces the denominator render byte-for-byte. Trips on templates
    that echo the system text elsewhere or inject varying content (unpinned dates)."""
    per = grid_context_system(ctxset, concept)
    neu = grid_context_system(ctxset, None)
    if render == "raw":
        a = RAW_CTX_FMT.format(system=per, user=C.GEN_PROMPT)
        b = RAW_CTX_FMT.format(system=neu, user=C.GEN_PROMPT)
    else:
        def _render(s):
            return tok.apply_chat_template(
                [{"role": "system", "content": s}, {"role": "user", "content": C.GEN_PROMPT}],
                add_generation_prompt=True, tokenize=False, date_string=LLAMA_DATE)
        a, b = _render(per), _render(neu)
    if a.count(per) != 1 or a.replace(per, neu) != b:
        raise RuntimeError(
            f"render-diff assert failed for ctx {ctxset}:{concept} ({render}): numerator and "
            "denominator renders differ outside the persona text -- the ratio would not isolate "
            "the persona")


def assert_llama_prefix_stable(tok, sample_text="qx z fjm wpl kbt"):
    """Template prefix-stability guard (checklist B4; the Qwen3 chat-template lesson): the
    generation-header context render must be an exact TOKEN prefix of the full
    [system, user, assistant=text] render, with the re-tokenized stream text immediately after.
    Only then is teacher-forcing re-encoded stream ids after the prefilled context the same
    computation as the A1-registered full-conversation render."""
    msgs = [{"role": "system", "content": LR.context_system("A", None)},
            {"role": "user", "content": C.GEN_PROMPT}]
    ctx = list(tok.apply_chat_template(msgs, add_generation_prompt=True,
                                       date_string=LLAMA_DATE))
    full = list(tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": sample_text}], date_string=LLAMA_DATE))
    stream = list(tok(sample_text, add_special_tokens=False).input_ids)
    if full[: len(ctx)] != ctx or full[len(ctx): len(ctx) + len(stream)] != stream:
        raise RuntimeError(
            "reader chat template is not prefix-stable: the [system, user, assistant] render does "
            "not extend the generation-header context with the re-tokenized stream -- "
            "teacher-forcing after the prefill would score a different computation")


def llama_stream_texts(streams, src_tok):
    """Stream TEXTS for the Llama path (Amendment 1): the trailing eos is STRIPPED (at most ONE)
    before decoding with the GENERATOR tokenizer -- the eos-free primary at text level; the
    with-eos secondary is Qwen-readers-only. Non-terminal special ids are part of the saved
    stream and stay (skip_special_tokens=False)."""
    eos = getattr(src_tok, "eos_token_id", None)
    out = []
    for s in streams:
        ids = [int(x) for x in np.asarray(s["tokens"]).reshape(-1)]
        if eos is not None and ids and ids[-1] == int(eos):
            ids = ids[:-1]
        out.append(src_tok.decode(ids, skip_special_tokens=False))
    return out


def llama_roundtrip_split(reader_tok, texts):
    """B13 (Amendment 1, should-fix 8): per-stream round-trip EXCLUSION, counted, never FATAL.
    decode(encode(text)) must reproduce the text under the READER tokenizer, or teacher-forcing
    the re-encoded ids would score a corrupted stream. Shards record the counts; the >5%
    voids-the-affected-Llama-cells rule is the OFFLINE scorer's job."""
    kept, excluded = [], 0
    for i, t in enumerate(texts):
        ids = reader_tok(t, add_special_tokens=False).input_ids
        if reader_tok.decode(ids) == t:
            kept.append(i)
        else:
            excluded += 1
    return kept, excluded


def llama_streams(streams, reader_tok, src_tok):
    """The Llama scoring pool: decoded (eos-stripped) texts re-tokenized with the reader's own
    tokenizer, round-trip failures excluded (counted). The SAME re-tokenized ids feed the
    numerator and the denominator of every ratio (prereg: identical decode -> re-encode for
    both). Returns (streams, n_excluded, n_total)."""
    texts = llama_stream_texts(streams, src_tok)
    kept, excluded = llama_roundtrip_split(reader_tok, texts)
    out = [dict(gidx=streams[i]["gidx"], concept=streams[i].get("concept"),
                strength=streams[i].get("strength"),
                tokens=list(reader_tok(texts[i], add_special_tokens=False).input_ids),
                text=texts[i]) for i in kept]
    return out, excluded, len(streams)


def prose_streams(reader_tok):
    """Gate 4's control cell: the 12 pinned English-prose sentences, reader-tokenized, one
    labeled stream per concept."""
    return [dict(gidx=i, concept=c, strength=None,
                 tokens=list(reader_tok(PROSE_CONTROL[c], add_special_tokens=False).input_ids),
                 text=PROSE_CONTROL[c])
            for i, c in enumerate(C.COVERT_CONCEPTS)]


def ctx_ids_for(reader_slug, tok, ctxset, concept, device, render="template"):
    """Context ids for one scoring cell. Qwen readers reuse the certified construction verbatim
    (identical context text + chat template as the 1.5B run; render is always the template
    primary); Llama routes to the A1 rendering with its template/raw render switch. B15: the
    secret ctxsets ride the SAME chat_ids construction as collection (K.chat_ids over the
    grid_context_system text + GEN_PROMPT -- LR.ctx_ids' own body, with the arm's system text)."""
    if family(reader_slug) != "qwen":                   # Amendment 4: the falcon3 readers
        return llama_ctx_ids(tok, ctxset, concept, device, render=render)
    if ctxset in SECRET_ARMS:
        return K.chat_ids(tok, C.GEN_PROMPT, system=grid_context_system(ctxset, concept),
                          device=device)
    return LR.ctx_ids(tok, ctxset, concept, device)


def _assert_bundle(bundle, path, gen, streamset):
    """Grid provenance: the bundle must be the named generator's, the named arm's, and the
    published 'orig' prompt variant (lr_reader parity, generalized off C.ACTIVE -- here the
    reader is deliberately NOT the generating model). Control (b): the INJECTED capture carries
    inject == 'gen' (not a streamset name) -- exactly what the 1.5B lr_reader run scored (its
    _assert_provenance never checked inject at all); for the injected set the arm check accepts
    'gen'. model + orig-variant still gate."""
    m = bundle.get("model")
    assert m in (None, gen), f"{path}: bundle model {m!r} != generator {gen!r}"
    arm = bundle.get("inject")
    arm_ok = {"gen", None} if streamset == INJECTED_SET else {streamset, None}
    assert arm in arm_ok, f"{path}: bundle arm {arm!r} != streamset {streamset!r}"
    v = bundle.get("variant")
    assert v in (None, "orig"), f"{path}: prompt variant {v!r}, not the published 'orig'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=sorted(GRID_READERS),
                    help="reader slug (GRID_READERS; INTRO_MODEL is deliberately unused)")
    ap.add_argument("--bundle", action="append", required=True,
                    help="repeatable <gen_slug>:<streamset>:<path> stream-bundle spec")
    ap.add_argument("--batch", type=int, default=None,
                    help="override the conservative per-size default (B8 sets this from smoke)")
    ap.add_argument("--tok-sample", type=int, default=TOKCHECK_SAMPLE)
    args = ap.parse_args()
    reader = args.reader
    batch_n = args.batch or default_batch(reader)

    outdir = C.RUN_DIR / "lr_grid"
    outdir.mkdir(parents=True, exist_ok=True)
    model, tok = K.load_model(GRID_READERS[reader])
    eos_id = getattr(tok, "eos_token_id", None)
    use_kv = KV_PIN.get(reader, None)   # registered self-check once per process, UNLESS the
    #                                     reader is KV-pinned (TECH M2: the 14B tol straddle)
    if use_kv is not None:
        print(f"LRG_KV_PINNED {reader}: use_kv={use_kv} (reader-keyed pin, self-check "
              "skipped; disclosed in shard meta)", flush=True)
    tok_checked = set()
    llama = family(reader) != "qwen"   # Llama-era name; Amendment 4: now the falcon3 readers
    if llama:
        assert_llama_prefix_stable(tok)                 # B4: template guard before any cell
        print(f"LRG xfam prefix-stability OK ({reader}; date pin {LLAMA_DATE} is a Falcon3 "
              "no-op, Amendment 4)", flush=True)
    renders = ("template", "raw") if llama else ("template",)   # A1: raw = marked SECONDARY
    src_toks = {}
    t0 = time.time()

    # Gate 4's prose control rides FIRST for llama readers (sentinel spec; no bundle behind it).
    specs = (["prose:control:-"] if llama else []) + list(args.bundle)
    for spec in specs:
        if spec.startswith("prose:"):
            gen, streamset = "prose", "control"
            streams = prose_streams(tok)
            rt_excluded, rt_total = 0, len(streams)
            ctxsets, cell_renders = ("N", "A"), ("template",)   # gate 4 runs under A1 (a)
        else:
            gen, streamset, path = parse_bundle_spec(spec)
            bundle = torch.load(path, map_location="cpu", weights_only=False)
            _assert_bundle(bundle, path, gen, streamset)
            streams = LR.select_streams(bundle, streamset)  # accepted s1 + s0 neutral, len>=2
            del bundle
            rt_excluded, rt_total = 0, len(streams)
            # B15: secret stream sets score under their ONE matched ctx set + the arm-own
            # neutral; the evoked sets keep the full A/B/N grid.
            ctxsets = (("N", SECRET_CTX[streamset]) if streamset in SECRET_CTX
                       else LR.CTX_SETS)
            cell_renders = renders
            if llama:                                   # B4/B13: decode -> re-tokenize path
                if gen not in src_toks:
                    from transformers import AutoTokenizer
                    src_toks[gen] = AutoTokenizer.from_pretrained(C.MODELS[gen]["hf_id"])
                streams, rt_excluded, rt_total = llama_streams(streams, tok, src_toks[gen])
                print(f"LRG roundtrip {gen}/{streamset}: kept {len(streams)}/{rt_total} "
                      f"(excluded {rt_excluded}; the >5 percent void rule is offline's)",
                      flush=True)
        print(f"LRG set={gen}/{streamset} n={len(streams)} reader={reader}", flush=True)
        # B3 gate: identical encodes on the ACTUAL stream texts before saved-id teacher forcing
        # under a reader that is not the generating model. Llama readers never teacher-force the
        # saved Qwen ids (B4 re-tokenizes decoded text), so the gate is Qwen-cross-size only.
        if family(reader) == "qwen" and gen != reader and gen not in tok_checked:
            from transformers import AutoTokenizer
            src_tok = AutoTokenizer.from_pretrained(C.MODELS[gen]["hf_id"])
            assert_shared_tokenizer(tok, src_tok, [s["text"] for s in streams],
                                    n=args.tok_sample)
            tok_checked.add(gen)
            print(f"LRG_TOKCHECK_OK {gen} (sample {min(args.tok_sample, len(streams))})",
                  flush=True)
        for ctxset in ctxsets:
            for render in cell_renders:
                shard = shard_path(outdir, reader, gen, streamset, ctxset, render=render)
                if shard.exists():
                    print(f"LRG_SKIP {shard.name} (resume)", flush=True)
                    continue
                recs = grid_records(streams, eos_id=eos_id)
                shard_t0, shard_toks = time.time(), 0        # B8: first-shard throughput
                for label, concept in LR.contexts_for(ctxset):
                    # neutral cells score under the pool's own s0 context: the evoked NEUTRAL
                    # persona for evoked/alt/prose (certified behavior), the secret arm's own
                    # baseline for the B15 sets (SECRET_CTX routes concept=None there).
                    eff = (SECRET_CTX.get(streamset, "A") if ctxset == "N" else ctxset)
                    if llama and concept is not None:   # A1: per scored context, both renders
                        assert_render_diff(tok, eff, concept, render=render)
                    ctx = ctx_ids_for(reader, tok, eff, concept, model.device, render=render)
                    kv, first = LR.prefill(model, ctx)
                    for lo in range(0, len(streams), batch_n):
                        chunk = streams[lo: lo + batch_n]
                        batch, lens = LR.pad_tokens([s["tokens"] for s in chunk])
                        batch = batch.to(model.device)
                        shard_toks += int(sum(lens))         # B8: scored-token count
                        lens_ne = noeos_lens([s["tokens"] for s in chunk], eos_id)
                        if use_kv is None:              # lr_reader-parity registered self-check
                            ll_cc = LR.score_batch_concat(model, ctx, batch, lens)
                            try:                        # a raising KV path must not kill the box
                                ll_kv = LR.score_batch_kv(model, ctx, kv, first, batch, lens)
                                worst = float((torch.abs(ll_kv - ll_cc)
                                               / torch.as_tensor(lens,
                                                                 device=ll_kv.device)).max())
                                use_kv = worst <= LR.SELFCHECK_TOL
                                print(f"{'LRG_SELFCHECK_OK' if use_kv else 'LRG_SELFCHECK_FALLBACK'} "
                                      f"max|dLL|/T={worst:.5f} (tol {LR.SELFCHECK_TOL})",
                                      flush=True)
                            except Exception as e:      # type name ONLY (no FATAL substrings)
                                use_kv = False
                                print(f"LRG_SELFCHECK_FALLBACK exception {type(e).__name__}",
                                      flush=True)
                        tokout = {}                     # B11: per-token capture, same forward
                        ll_free, ll_eos, use_kv = score_batch_dual(model, ctx, kv, first, batch,
                                                                   lens, lens_ne, use_kv,
                                                                   pertok=tokout)
                        record_lls(recs, lo, label, ll_free, ll_eos)
                        record_toks(recs, lo, label, tokout["ll"], lens)
                    print(f"LRG ctx {gen}/{streamset}x{ctxset}:{label}:{render} done "
                          f"t={int(time.time() - t0)}s", flush=True)
                util_gate_hook("shard_done", shard=shard.name, t=time.time() - t0,   # B8 gate
                               tokens=shard_toks, secs=time.time() - shard_t0)
                tmp = shard.with_suffix(".tmp")
                torch.save(dict(reader=reader, reader_hf=GRID_READERS[reader], generator=gen,
                                streamset=streamset, ctxset=ctxset, render=render,
                                stream_tokenization=("reader-tokenized-prose" if gen == "prose"
                                                     else "xfam-retok-eosfree" if llama
                                                     else "saved-ids"),
                                roundtrip_excluded=rt_excluded, roundtrip_total=rt_total,
                                contexts=[lbl for lbl, _ in LR.contexts_for(ctxset)],
                                selfcheck_kv=use_kv, kv_pinned=reader in KV_PIN,
                                batch=batch_n,
                                eos_rule=("ll = eos-free PRIMARY; ll_eos = with-eos SECONDARY"
                                          if not llama else
                                          "xfam path: trailing eos stripped pre-decode; "
                                          "with-eos secondary is Qwen-readers-only"),
                                tok_rule="ll_tok = per-token LL vectors fp16, with-eos length "
                                         "(Amendment 1 Blocker 1); slice [:T_noeos] for "
                                         "eos-free, [:min(K, T_noeos)] for prefix-K",
                                records=recs), tmp)
                os.replace(tmp, shard)                                          # atomic
                print(f"LRG_SHARD_SAVED {shard.name} n={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
