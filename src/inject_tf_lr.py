"""Injection-LR run (2): teacher-forced LL of the exp1 INJECTED streams with the concept vector
ACTIVELY RE-INJECTED during scoring, vs the same forward with no injection (prereg Part B of
experiments/exp2_output_monitorability/reports/lr_scale_extend_prereg.md, DRAFT; scope settled
SMALL-ONLY in HANDOFF.md 7a; comparability constraints in
reports/NOTE_injection_LR_comparability.md -- run-2 bits are their OWN quantity, pure-concept
channel, never framed against the secret/evoked language-channel LR).

The quantity (the NOTE's measurement #1, the fair analog of the secret_word diagonal):
    E_{s~P(.|V)}[ LL(s|V) - LL(s|neutral) ] = KL(P_injected || P_neutral)
plus the full 12-column matrix S[i, j] = LL(s_i | inject v_j) - LL(s_i | no inject) so the
certified 12-way calibrated readout (lr_reader_offline.evaluate_cell) applies unchanged offline.

Reuse, not reimplementation:
  - the steering primitives are the capture's own stored inject_vectors / inject_alpha
    (src/covert_collect.py wrote them for exactly this), re-applied via common._injection_hook
    with prompt_len = the context length -- the generation-time 'gen' convention (vector on the
    STREAM positions only, prompt clean);
  - the scoring numerics are lr_grid.score_batch_dual over lr_reader's certified functions,
    forced onto the CONCAT reference path (use_kv=False, structural: a position-indexed hook
    under KV-cached scoring would mis-index stream positions -- KV is excluded by construction,
    not merely self-checked). eos-free PRIMARY + with-eos secondary + per-token fp16 vectors from
    the same forward, grid shard schema, so analysis/lr_grid_offline machinery reads the shards
    unmodified.
  - stream selection is lr_reader.select_streams("injected") (accepted, len>=2, strength==smax),
    with --strength overriding smax (the 7B s124-primary pin, prereg Part B).

Amendment 2 (2a, 2026-07-14) extensions: --strengths scores a DOSE LIST in one model load
(run_doses); a dose whose stored vectors/alphas are missing DEGRADES to not-scored -- disclosed
in a dose_plan JSON next to the shards AND in every scored shard's meta -- and is NEVER
regenerated. --no-s0 drops the s0 centering pool from the dose passes (disclosed; the main
capture's centering rides the existing s60 shard). Bundles carrying streamset='expressed' +
system_text (the 2b expressed-injection cell) shard as <slug>__<slug>__expressed_TF*_s<lvl>.pt
and score under their OWN stored generation context, token-identical to collection.

HF path with hooks ONLY -- vLLM cannot inject (no steering hooks under prompt_logprobs), which is
why this never rides the serverless scout. RESUMABLE: one atomic shard per (slug, label-set):
$INTRO_RUN_DIR/inject_tf/<slug>__<slug>__injected_TF{V,N}.pt. Run on GPU via box_lr_extend.py;
NEVER on the Mac. Marker-safe prints (no box READY/DONE/FATAL substrings).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import config as C
import common as K
import lr_reader as LR
import lr_grid as G


def select_injected(bundle, strength=None):
    """The scored pool: lr_reader.select_streams('injected') (accepted, len>=2, strength==smax)
    unless `strength` pins a specific level (the 7B s124-primary pin) -- then the SAME accepted/
    len>=2 filter at that level (a mechanical strength filter on the identical selection rule)."""
    if strength is None:
        return LR.select_streams(bundle, "injected")
    pool = [s for s in bundle["streams"] if s.get("accepted", True) and len(s["tokens"]) >= 2]
    lvls = sorted({int(s["strength"]) for s in pool})
    if int(strength) not in lvls:
        raise ValueError(f"strength {strength} not in capture levels {lvls}")
    return [s for s in pool if int(s["strength"]) == int(strength)]


def neutral_streams(bundle):
    """The capture's s0 (uninjected) streams -- the centering-gate pool (accepted, len>=2)."""
    return [s for s in bundle["streams"]
            if s.get("accepted", True) and len(s["tokens"]) >= 2 and int(s["strength"]) == 0]


def steering_primitives(bundle, strength):
    """(vectors, alphas) keyed by concept, from the capture's OWN stored primitives -- never
    re-derived. alpha keys are '<concept>|s<strength>' (covert_collect's schema)."""
    vecs, alphas = {}, {}
    concepts = list(bundle["concepts"])
    iv = bundle.get("inject_vectors") or {}
    ia = bundle.get("inject_alpha") or {}
    for c in concepts:
        if c not in iv:
            raise KeyError(f"capture has no inject_vector for {c!r}")
        key = f"{c}|s{int(strength)}"
        if key not in ia:
            raise KeyError(f"capture has no inject_alpha {key!r} (have {sorted(ia)[:4]}...)")
        vecs[c] = torch.as_tensor(np.asarray(iv[c]), dtype=torch.float32)
        alphas[c] = float(ia[key])
    return vecs, alphas


class HookSeam:
    """Injectable hook seam: register(model, layer, vec, alpha, prompt_len) -> handle with
    .remove(). The real seam registers common._injection_hook on model.model.layers[layer]
    (the exact generation-time apparatus); tests inject a fake."""
    @staticmethod
    def register(model, layer, vec, alpha, prompt_len):
        hook = K._injection_hook(vec, alpha, prompt_len=prompt_len)
        return model.model.layers[layer].register_forward_hook(hook)


def score_pool(model, ctx, streams, batch_n, eos_id, label_hooks, hook_seam=HookSeam,
               layer=None, device=None):
    """Score every stream under every label. label_hooks: {label: (vec, alpha) | None} -- None
    scores the plain forward (the 'neutral' denominator). One CONCAT forward per (label, batch):
    lr_grid.score_batch_dual with use_kv=False (kv/first unused on the concat path), pertok
    capture on -- eos-free primary, with-eos secondary and per-token fp16 vectors all from the
    certified functions. Returns lr_grid-schema records."""
    device = device or model.device
    recs = G.grid_records(streams, eos_id=eos_id)
    plen = int(ctx.shape[1])
    for label, prim in label_hooks.items():
        handle = None
        if prim is not None:
            vec, alpha = prim
            handle = hook_seam.register(model, layer, vec, alpha, plen)
        try:
            for lo in range(0, len(streams), batch_n):
                chunk = streams[lo: lo + batch_n]
                batch, lens = LR.pad_tokens([s["tokens"] for s in chunk])
                batch = batch.to(device)
                lens_ne = G.noeos_lens([s["tokens"] for s in chunk], eos_id)
                tokout = {}
                ll_free, ll_eos, _ = G.score_batch_dual(model, ctx, None, None, batch, lens,
                                                        lens_ne, use_kv=False, pertok=tokout)
                G.record_lls(recs, lo, label, ll_free, ll_eos)
                G.record_toks(recs, lo, label, tokout["ll"], lens)
        finally:
            if handle is not None:
                handle.remove()
    return recs


def assert_hook_live(model, ctx, stream, vec, alpha, layer, eos_id, hook_seam=HookSeam,
                     tol=1e-6):
    """$0-fail sanity gate before the full pass: one stream's LL under its own concept's
    re-injection must DIFFER from the no-hook LL (|dLL| > tol in fp32) or the hook is dead
    (wrong layer/dtype/device would silently produce the neutral answer twice)."""
    batch, lens = LR.pad_tokens([stream["tokens"]])
    batch = batch.to(model.device)
    ll_off = LR.score_batch_concat(model, ctx, batch, lens)
    handle = hook_seam.register(model, layer, vec, alpha, int(ctx.shape[1]))
    try:
        ll_on = LR.score_batch_concat(model, ctx, batch, lens)
    finally:
        handle.remove()
    d = float(torch.abs(ll_on.float() - ll_off.float()).max())
    if not d > tol:
        raise RuntimeError(f"injection hook is DEAD: |dLL| = {d} <= {tol} with the vector "
                           "registered -- refusing to score (wrong layer/device/dtype?)")
    print(f"ITF hook live: |dLL| = {d:.4f} nats on the probe stream", flush=True)
    return d


def shard_paths(outdir, slug, level, streamset="injected"):
    """Two atomic shards per (slug, injection level): _TFV (12 concept labels, vector active)
    and _TFN (label 'neutral', no vector) -- lr_grid's <reader>__<gen>__<set>_<ctx>.pt naming
    (with the level in the ctx code, so the 7B s124-primary and s140-descriptive passes never
    collide) so the offline cell join (cell_rows over ctx + N shards) reads them unmodified.
    streamset 'expressed' (the 2b bundle's own tag) keeps the expressed s20 self-read from ever
    colliding with the 2a injected s20 shard."""
    ss = str(streamset)
    return (os.path.join(str(outdir), f"{slug}__{slug}__{ss}_TFV_s{int(level)}.pt"),
            os.path.join(str(outdir), f"{slug}__{slug}__{ss}_TFN_s{int(level)}.pt"))


def _atomic_save(path, obj):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_slug(model, tok, bundle, slug, outdir, batch_n=8, strength=None, layer=None,
             hook_seam=HookSeam, include_s0=True, limit=None, extra_meta=None):
    """One slug's full run-2 pass -> the two shards. Contexts: the SAME system the streams were
    generated under -- the bundle's stored system_text when present (the 2b expressed cell:
    sustain-s1 + STRONG_SYSTEM), else the capture's variant (default orig) -- + GEN_PROMPT via
    common.chat_ids, token-identical to collection. Resume-safe (existing shard skipped).
    limit caps BOTH pools AFTER selection (smoke slice only, never a real run). include_s0=False
    drops the s0 centering pool (the disclosed Amendment-2 dose-pass trim). extra_meta (e.g. the
    dose_plan disclosure) is merged into both shards' meta."""
    variant = bundle.get("variant") or "orig"
    system = bundle.get("system_text") or C.PROMPT_VARIANTS[variant]
    streamset = bundle.get("streamset") or "injected"
    ctx = K.chat_ids(tok, C.GEN_PROMPT, system=system, device=model.device)
    eos_id = getattr(tok, "eos_token_id", None)
    layer = layer if layer is not None else int(bundle["layer"])
    concepts = list(bundle["concepts"])
    streams = select_injected(bundle, strength=strength)
    lvl = int(strength) if strength is not None else int(max(s["strength"] for s in streams))
    if limit:
        streams = streams[: int(limit)]
    if include_s0:
        s0 = neutral_streams(bundle)
        streams = streams + (s0[: int(limit)] if limit else s0)
    vecs, alphas = steering_primitives(bundle, lvl)
    probe = next(s for s in streams if s.get("concept") in concepts)
    assert_hook_live(model, ctx, probe, vecs[probe["concept"]], alphas[probe["concept"]],
                     layer, eos_id, hook_seam=hook_seam)
    p_v, p_n = shard_paths(outdir, slug, lvl, streamset=streamset)
    meta = dict(reader=slug, generator=slug, streamset=streamset,
                strength=lvl, layer=layer, variant=variant,
                stream_tokenization="saved-ids", include_s0=bool(include_s0),
                score="ll = eos-free PRIMARY; ll_eos = with-eos SECONDARY; ll_tok = per-token "
                      "fp16 (same forward). TFV labels = LL under ACTIVE re-injection of each "
                      "concept vector (stream positions only, gen convention); TFN label "
                      "'neutral' = the same forward with NO vector. run-2 bits are a "
                      "PURE-CONCEPT-channel quantity (NOTE_injection_LR_comparability #2): "
                      "never compare directly to secret/evoked language-channel LR.")
    if extra_meta:
        meta.update(extra_meta)
    t0 = time.time()
    if os.path.exists(p_n):
        print(f"ITF_SKIP {os.path.basename(p_n)} (resume)", flush=True)
    else:
        recs = score_pool(model, ctx, streams, batch_n, eos_id, {"neutral": None},
                          hook_seam=hook_seam, layer=layer)
        _atomic_save(p_n, dict(meta, ctxset="TFN", contexts=["neutral"], records=recs))
        print(f"ITF_SHARD_SAVED {os.path.basename(p_n)} n={len(recs)} "
              f"t={int(time.time() - t0)}s", flush=True)
    if os.path.exists(p_v):
        print(f"ITF_SKIP {os.path.basename(p_v)} (resume)", flush=True)
    else:
        hooks = {c: (vecs[c], alphas[c]) for c in concepts}
        recs = score_pool(model, ctx, streams, batch_n, eos_id, hooks,
                          hook_seam=hook_seam, layer=layer)
        _atomic_save(p_v, dict(meta, ctxset="TFV", contexts=concepts, records=recs))
        print(f"ITF_SHARD_SAVED {os.path.basename(p_v)} n={len(recs)} "
              f"t={int(time.time() - t0)}s", flush=True)
    return p_v, p_n


def available_doses(bundle, levels):
    """Amendment 2 (2a): split the requested dose list into scoreable levels (stored steering
    primitives present AND streams at that level) and DEGRADED ones (missing vectors/alpha or
    an absent level) -- the degraded doses are reported not-scored, NEVER regenerated."""
    scored, missing = [], []
    for lvl in levels:
        try:
            steering_primitives(bundle, int(lvl))
            select_injected(bundle, strength=int(lvl))
            scored.append(int(lvl))
        except (KeyError, ValueError) as e:
            missing.append(dict(level=int(lvl), reason=str(e)))
    return scored, missing


def run_doses(model, tok, bundle, slug, outdir, levels, batch_n=8, layer=None,
              hook_seam=HookSeam, include_s0=True, limit=None):
    """The Amendment-2 dose pass: score every scoreable level of `levels` in ONE model load
    (one run_slug pass per level, resume-safe shards). The dose plan -- scored + not-scored
    with reasons -- is disclosed in a dose_plan JSON next to the shards AND embedded in every
    scored shard's meta. A missing dose is never regenerated."""
    scored, missing = available_doses(bundle, levels)
    streamset = bundle.get("streamset") or "injected"
    plan = dict(slug=slug, streamset=streamset,
                requested=[int(l) for l in levels], scored=scored, not_scored=missing,
                note="Amendment 2 (2a): a dose missing stored inject vectors/alpha degrades "
                     "to not-scored (disclosed here and in the shard meta), NEVER regenerated")
    paths = []
    for lvl in scored:
        paths.append(run_slug(model, tok, bundle, slug, outdir, batch_n=batch_n,
                              strength=lvl, layer=layer, hook_seam=hook_seam,
                              include_s0=include_s0, limit=limit,
                              extra_meta=dict(dose_plan=plan)))
    tag = "-".join(str(l) for l in levels)
    dp = os.path.join(str(outdir), f"dose_plan_{slug}_{streamset}_s{tag}.json")
    tmp = dp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(plan, f, indent=1)
    os.replace(tmp, dp)
    print(f"ITF dose plan: scored={scored} "
          f"not_scored={[m['level'] for m in missing]} -> {os.path.basename(dp)}", flush=True)
    return paths, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="exp1 covert_collect.pt capture")
    ap.add_argument("--slug", required=True, help="model slug (reader == generator)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--strength", type=int, default=None,
                    help="pin the scored injection level (default: smax; 7B primary pins 124 "
                         "per the prereg Part B dose caveat)")
    ap.add_argument("--strengths", default=None,
                    help="comma dose list scored in ONE model load (Amendment 2 2a); a dose "
                         "missing stored vectors/alpha degrades to not-scored (disclosed), "
                         "never regenerated")
    ap.add_argument("--no-s0", action="store_true",
                    help="drop the s0 centering pool (the disclosed Amendment-2 dose-pass "
                         "trim; the main capture's centering rides the existing s60 shard)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap streams per pass (smoke slice only, never a real run)")
    args = ap.parse_args()
    if args.strength is not None and args.strengths:
        raise SystemExit("--strength and --strengths are mutually exclusive")

    outdir = C.RUN_DIR / "inject_tf"
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.capture, map_location="cpu", weights_only=False)
    m = bundle.get("model")
    assert m in (None, args.slug), f"capture model {m!r} != --slug {args.slug!r}"
    assert bundle.get("inject") == "gen", (
        f"capture inject={bundle.get('inject')!r}: run 2 re-applies the generation-only "
        "convention and only 'gen' captures match it")
    hf_id = C.MODELS[args.slug]["hf_id"]
    model, tok = K.load_model(hf_id)
    if args.strengths:
        levels = [int(x) for x in args.strengths.replace(" ", "").split(",") if x]
        run_doses(model, tok, bundle, args.slug, outdir, levels, batch_n=args.batch,
                  include_s0=not args.no_s0, limit=args.limit)
    else:
        run_slug(model, tok, bundle, args.slug, outdir, batch_n=args.batch,
                 strength=args.strength, include_s0=not args.no_s0, limit=args.limit)


if __name__ == "__main__":
    main()
