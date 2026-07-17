"""70B cross-family observer RIDER (Amendment 1, 2026-07-14, of exp2 reports/
lr_scale_extend_prereg.md -- post-freeze, pre-data, Matt-approved): Qwen2.5 readers
teacher-force the 810 EXISTING Llama-3.3-70B streams (runs/llama70b_scout/
streams_llama70b.json) under matched + arm-own-neutral + 11 mismatched contexts. The 70B
faithful-template discriminator found real self-read channels (secret_sustain 0.244 bits,
evoked 0.143 -- lr_72b_fullsweep_verdict.md addendum) whose PRIVACY was never measured; these
cells measure it. Confirmatory: secret_sustain + evoked (privacy bound < 0.05 calibrated bits);
secret_word rides DESCRIPTIVE (its 70B diagonal is null -- a privacy claim there is vacuous).

Construction (nothing numeric is new -- certified function objects only, the lr_grid-over-
lr_reader pattern):
  - STREAMS are TEXT-ONLY (every scout record has token_ids = null -- HANDOFF section 6: they
    predate the capture fix), so the TEXT path is the only path: re-tokenize with the READER'S
    OWN tokenizer, exactly the cross-family pattern lr_grid.py registered for Falcon-reads-Qwen
    (Amendment 1/B13 of the grid prereg). Round-trip discipline is the grid's own
    lr_grid.llama_roundtrip_split (same function object): decode(encode(text)) must reproduce
    the text or the stream is EXCLUDED and COUNTED -- reported in the shard meta, never FATAL
    (the >5% void rule stays the offline scorer's).
  - CONTEXTS compose through serverless_72b.build_system_prompt (concept-or-None, arm,
    STRONG_SYSTEM) -- byte-identical to what the 70B run scored under (it delegates to
    primers_v3.compose_system) -- and render in the READER'S chat template via the certified
    common.chat_ids(system + GEN_PROMPT, generation header on): the grid's secret-cell
    construction (lr_grid.ctx_ids_for's K.chat_ids branch).
  - SCORING is lr_grid.score_batch_dual over lr_reader's certified paths (eos-free PRIMARY +
    with-eos secondary + per-token fp16 ll_tok from the SAME forward), with the lr_reader-parity
    KV self-check once per process and the B8 util gate on the first full shard.
  - custom-id scheme: record custom_id = "lr:{arm}:{concept}:{stream_idx}" -- the scout's
    "lr:{arm}:{concept}:{stream_idx}:{context}" scheme with the context supplied by the ll key
    ('neutral' or the concept name), so every scored number joins back to the raw 70B batch
    lines (runs/llama70b_scout/lr_raw_batch_output.jsonl).

Shards: the grid schema/naming via lr_grid.shard_path --
$INTRO_RUN_DIR/lr_grid/<reader>__llama70b__<arm>_{R,N}.pt (ctx code R = the rider matched set:
12 concept contexts; N = the arm-own neutral). analysis/lr_extend_offline.py joins them with
the certified cell_rows + evaluate_cell, unmodified.

RESUMABLE (existing shard => skip). Run on GPU via box_lr_extend.py; NEVER on the Mac.
Marker-safe prints (no box READY/DONE/FATAL substrings).
"""
import argparse
import json
import os
import time

import torch

import config as C
import common as K
import lr_reader as LR
import lr_grid as G
import lr_grid_extend as GX
import serverless_72b as SV

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

RIDER_GEN = "llama70b"                                  # shard-name generator slug
RIDER_SOURCE = "runs/llama70b_scout/streams_llama70b.json"
STREAMS_JSON = os.path.join(REPO, "runs", "llama70b_scout", "streams_llama70b.json")
RIDER_ARMS = ("secret_sustain", "evoked", "secret_word")   # descriptive arm LAST (trim order)
RIDER_CONFIRMATORY = ("secret_sustain", "evoked")       # privacy bound < 0.05 calibrated bits
RIDER_DESCRIPTIVE = ("secret_word",)                    # 70B diagonal null -> descriptive
RIDER_CTX = "R"                                         # matched set: the 12 concept contexts
RIDER_READERS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-14b")
PRIVACY_BOUND_BITS = 0.05


def custom_id(arm, concept, stream_idx, context=None):
    """The scout's custom-id scheme (run_llama70b_scout build: 'lr:{arm}:{concept}:
    {stream_idx}:{context}'); context=None returns the per-record base (the shard stores the
    base; the ll key supplies the context)."""
    base = f"lr:{arm}:{concept}:{stream_idx}"
    return base if context is None else f"{base}:{context}"


def load_rider_streams(path, arm):
    """The 70B scout records for one arm: accepted, non-empty TEXT (token_ids are null by
    construction -- asserted nowhere because absence is the documented state; text is the only
    path either way). stream_idx is the scout's own globally-unique index (verified 0..809)."""
    with open(path) as f:
        data = json.load(f)
    return [s for s in data
            if s.get("arm") == arm and s.get("accepted") and (s.get("text") or "").strip()]


def rider_pool(records, reader_tok):
    """Reader-tokenized scoring pool: TEXT re-tokenized with the READER tokenizer, round-trip
    failures EXCLUDED via the certified lr_grid.llama_roundtrip_split (counted, never fatal).
    gidx = the scout stream_idx (the cell join key + custom-id component).
    Returns (pool, n_excluded, n_total)."""
    texts = [r["text"] for r in records]
    kept, excluded = G.llama_roundtrip_split(reader_tok, texts)
    pool = [dict(gidx=int(records[i]["stream_idx"]), concept=records[i]["concept"], strength=1,
                 tokens=list(reader_tok(texts[i], add_special_tokens=False).input_ids),
                 text=texts[i])
            for i in kept]
    return pool, excluded, len(records)


def rider_ctx_ids(tok, arm, concept, device):
    """Context ids for one rider label: serverless_72b.build_system_prompt (the EXACT
    composition the 70B run used; concept=None -> the arm-own neutral) rendered in the READER'S
    own chat template through the certified K.chat_ids (system + GEN_PROMPT, generation header
    on -- lr_grid's secret-cell construction)."""
    system = SV.build_system_prompt(concept, arm, C.STRONG_SYSTEM)
    return K.chat_ids(tok, C.GEN_PROMPT, system=system, device=device)


def rider_labels(ctxset):
    """(label, concept) pairs per shard: the matched set R scores every stream under ALL 12
    concept contexts (matched + 11 mismatched = the certified 12-way matrix's columns); N is the
    arm-own neutral (concept None)."""
    if ctxset == "N":
        return [("neutral", None)]
    return [(c, c) for c in C.COVERT_CONCEPTS]


def rider_shard_meta(reader, arm, rt_excluded, rt_total, contexts, use_kv, batch_n):
    return dict(
        reader=reader, reader_hf=G.GRID_READERS[reader],
        kv_pinned=reader in G.KV_PIN,       # TECH M2 disclosure: 14B rides the pinned concat
        #                                     path (tol-straddling self-check); others self-check
        generator=RIDER_GEN, generator_source=RIDER_SOURCE,
        streamset=arm, contexts=list(contexts),
        confirmatory=arm in RIDER_CONFIRMATORY, privacy_bound_bits=PRIVACY_BOUND_BITS,
        stream_tokenization="xfam-retok-text (scout token_ids are null; reader re-tokenizes "
                            "the text -- the grid's Falcon-reads-Qwen pattern)",
        roundtrip_excluded=int(rt_excluded), roundtrip_total=int(rt_total),
        ctx_composition="serverless_72b.build_system_prompt(concept|None, arm, STRONG_SYSTEM) "
                        "rendered by the READER chat template (common.chat_ids + GEN_PROMPT)",
        custom_id_scheme="record custom_id + ':' + ll key reproduces the scout's "
                         "lr:{arm}:{concept}:{stream_idx}:{context}",
        selfcheck_kv=use_kv, batch=batch_n,
        eos_rule="ll = eos-free PRIMARY; ll_eos = with-eos SECONDARY (re-tokenized text carries "
                 "no terminal eos, so the two coincide except on pathological round-trips)",
        tok_rule="ll_tok = per-token LL vectors fp16, with-eos length (Amendment 1 Blocker 1)")


def score_rider_arm(model, tok, reader, arm, pool, rt_excluded, rt_total, outdir, batch_n,
                    state, t0=None, gated=True):
    """Both shards (R matched-set + N neutral) for one (reader, arm) cell. The inner loop is
    lr_grid.main's, over the same certified function objects (prefill, pad_tokens, the KV
    self-check, score_batch_dual with per-token capture). gated=False (a --limit smoke slice)
    skips the B8 util-gate hook entirely -- a 4-stream slice samples the trailing-1s util
    window mostly idle and repeatably false-halted at 44% (2026-07-14 review CRIT 1); the
    hook's own UTIL_GATE_TOKEN_FLOOR additionally exempts sub-floor shards on full runs, so
    the gate fires on the first FULL-SIZE shard (the 12-context R shard, ~200k+ tokens)."""
    t0 = t0 or time.time()
    eos_id = getattr(tok, "eos_token_id", None)
    for ctxset in ("N", RIDER_CTX):
        shard = G.shard_path(outdir, reader, RIDER_GEN, arm, ctxset)
        if shard.exists():
            print(f"RIDER_SKIP {shard.name} (resume)", flush=True)
            continue
        recs = G.grid_records(pool, eos_id=eos_id)
        for i, s in enumerate(pool):
            recs[i]["custom_id"] = custom_id(arm, s["concept"], s["gidx"])
        labels = rider_labels(ctxset)
        shard_t0, shard_toks = time.time(), 0
        for label, concept in labels:
            ctx = rider_ctx_ids(tok, arm, concept, model.device)
            kv, first = LR.prefill(model, ctx)
            for lo in range(0, len(pool), batch_n):
                chunk = pool[lo: lo + batch_n]
                batch, lens = LR.pad_tokens([s["tokens"] for s in chunk])
                batch = batch.to(model.device)
                shard_toks += int(sum(lens))
                lens_ne = G.noeos_lens([s["tokens"] for s in chunk], eos_id)
                if state.get("use_kv") is None:      # lr_reader-parity registered self-check
                    ll_cc = LR.score_batch_concat(model, ctx, batch, lens)
                    try:
                        ll_kv = LR.score_batch_kv(model, ctx, kv, first, batch, lens)
                        worst = float((torch.abs(ll_kv - ll_cc)
                                       / torch.as_tensor(lens, device=ll_kv.device)).max())
                        state["use_kv"] = worst <= LR.SELFCHECK_TOL
                        print(f"RIDER_SELFCHECK{'_OK' if state['use_kv'] else '_FALLBACK'} "
                              f"max|dLL|/T={worst:.5f} (tol {LR.SELFCHECK_TOL})", flush=True)
                    except Exception as e:           # type name ONLY (no FATAL substrings)
                        state["use_kv"] = False
                        print(f"RIDER_SELFCHECK_FALLBACK exception {type(e).__name__}",
                              flush=True)
                tokout = {}
                ll_free, ll_eos, state["use_kv"] = G.score_batch_dual(
                    model, ctx, kv, first, batch, lens, lens_ne, state["use_kv"],
                    pertok=tokout)
                G.record_lls(recs, lo, label, ll_free, ll_eos)
                G.record_toks(recs, lo, label, tokout["ll"], lens)
            print(f"RIDER ctx {arm}x{ctxset}:{label} done t={int(time.time() - t0)}s",
                  flush=True)
        if gated:
            G.util_gate_hook("shard_done", shard=shard.name, t=time.time() - t0,
                             tokens=shard_toks, secs=time.time() - shard_t0)
        else:
            print(f"RIDER util gate skipped for {shard.name} (--limit smoke slice; "
                  "CRIT 1)", flush=True)
        tmp = shard.with_suffix(".tmp")
        torch.save(dict(rider_shard_meta(reader, arm, rt_excluded, rt_total,
                                         [lbl for lbl, _ in labels], state.get("use_kv"),
                                         batch_n),
                        ctxset=ctxset, records=recs), tmp)
        os.replace(tmp, shard)                                          # atomic
        print(f"RIDER_SHARD_SAVED {shard.name} n={len(recs)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True, choices=sorted(RIDER_READERS))
    ap.add_argument("--arms", default=",".join(RIDER_ARMS),
                    help="comma-separated rider arms (the box's trim plan decides)")
    ap.add_argument("--streams", default=STREAMS_JSON)
    ap.add_argument("--batch", type=int, default=None,
                    help="override the conservative per-size default")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap streams per arm (smoke slice only, never a real run)")
    args = ap.parse_args()
    arms = tuple(a for a in args.arms.split(",") if a)
    bad = [a for a in arms if a not in RIDER_ARMS]
    if bad:
        raise SystemExit(f"unknown rider arms {bad}; expected among {RIDER_ARMS}")

    GX.register()                                       # 14B reader slug + batch defaults
    outdir = C.RUN_DIR / "lr_grid"
    outdir.mkdir(parents=True, exist_ok=True)
    model, tok = K.load_model(G.GRID_READERS[args.reader])
    batch_n = args.batch or G.default_batch(args.reader)
    # self-check once per process, UNLESS the reader is KV-pinned (TECH M2: the 14B reader's
    # self-check straddles the tolerance -- lr_grid_extend registered the concat pin above).
    state = {"use_kv": G.KV_PIN.get(args.reader)}
    if state["use_kv"] is not None:
        print(f"RIDER_KV_PINNED {args.reader}: use_kv={state['use_kv']} (reader-keyed pin, "
              "self-check skipped; disclosed in shard meta)", flush=True)
    t0 = time.time()
    for arm in arms:
        records = load_rider_streams(args.streams, arm)
        if args.limit:
            records = records[: int(args.limit)]
        pool, rt_excluded, rt_total = rider_pool(records, tok)
        print(f"RIDER set={RIDER_GEN}/{arm} reader={args.reader} kept {len(pool)}/{rt_total} "
              f"(roundtrip excluded {rt_excluded}; the >5 percent void rule is offline's)",
              flush=True)
        score_rider_arm(model, tok, args.reader, arm, pool, rt_excluded, rt_total, outdir,
                        batch_n, state, t0=t0, gated=not args.limit)


if __name__ == "__main__":
    main()
