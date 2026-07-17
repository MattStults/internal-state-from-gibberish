"""LR-reader (prereg: experiments/exp2_output_monitorability/reports/lr_reader_prereg.md).

Teacher-forced log-likelihood of every saved word-free stream under the 25 reconstructed collection
contexts: 12 wording-A persona contexts (primers_v2.compose_system(c, STRONG_SYSTEM, arm="evoked")),
12 wording-B (arm="evoked_alt"), 1 shared neutral (compose_system(None, ..., arm="evoked")) -- each
tokenized exactly as at collection (common.chat_ids with C.GEN_PROMPT). The LR reader is an OBSERVER:
no injection hook anywhere, ever -- it just asks how likely the saved tokens are under each context.

Stream sets: injected (exp1 capture, accepted s==max), evoked / evoked_alt (exp3 bundles, accepted
strength-1 concept streams PLUS the s0 neutral streams for the prereg sanity gates). All accepted
streams (len>=2); no per-concept cap.

Efficiency: per context, the context KV cache is prefilled once and reused (expanded) across
right-padded stream batches. REGISTERED SELF-CHECK: the first batch is scored via KV-reuse AND via
a full concat forward; if max |dLL|/T > 0.02 nats/token the whole run falls back to the concat path
(LR_SELFCHECK_FALLBACK) -- correctness over speed.

RESUMABLE: one atomic shard per (streamset, contextset): $INTRO_RUN_DIR/lr/<slug>_<set>_<ctx>.pt;
existing shard => skip (print LR_SKIP). Run on GPU via the orchestrator; NEVER on the Mac.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import config as C
import common as K

CTX_ARMS = {"A": "evoked", "B": "evoked_alt"}
CTX_SETS = ("N", "A", "B")
SELFCHECK_TOL = 0.02          # registered: max |LL_kv - LL_concat| / T, nats per token


def _primers_v2():
    """exp3's primers_v2, lazily (frozen collection composition lives there)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "experiments", "exp3_induction_and_scale"))
    import primers_v2
    return primers_v2


def context_system(ctxset, concept):
    """System string for one scoring context. concept in (None, 'neutral') -> the shared NEUTRAL
    persona baseline (identical for A and B: compose_system(None, ...) resolves to primers.NEUTRAL
    for both persona arms -- hence 25 contexts, not 26)."""
    P = _primers_v2()
    if concept in (None, "neutral"):
        return P.compose_system(None, C.STRONG_SYSTEM, arm="evoked")
    return P.compose_system(concept, C.STRONG_SYSTEM, arm=CTX_ARMS[ctxset])


def ctx_ids(tok, ctxset, concept, device):
    """Context token ids EXACTLY as at collection: chat template over (system, GEN_PROMPT)."""
    return K.chat_ids(tok, C.GEN_PROMPT, system=context_system(ctxset, concept), device=device)


def select_streams(bundle, streamset):
    """Accepted streams with len>=2. injected: exp1 capture at max strength only. evoked/evoked_alt:
    strength-1 concept streams + the s0 neutral streams (prereg sanity-gate pool)."""
    pool = [s for s in bundle["streams"] if s.get("accepted", True) and len(s["tokens"]) >= 2]
    if streamset == "injected":
        smax = max(s["strength"] for s in pool)
        return [s for s in pool if s["strength"] == smax]
    return [s for s in pool if s["strength"] == 1 or s.get("concept") == "neutral"]


def pad_tokens(token_lists, pad_id=0):
    """list of (list[int] | np.ndarray | 1-D torch tensor) -> ([B, Tmax] long, true lengths).
    Right padding; padded positions are masked out of both attention and the LL sum."""
    arrs = [torch.as_tensor(np.asarray(t), dtype=torch.long) for t in token_lists]
    lens = [int(a.numel()) for a in arrs]
    batch = torch.full((len(arrs), max(lens)), int(pad_id), dtype=torch.long)
    for i, a in enumerate(arrs):
        batch[i, : a.numel()] = a
    return batch, lens


def ll_from_logits(pred_logits, targets, lengths):
    """pred_logits [B, Tmax, V] where [:, t] predicts targets[:, t]. float32 log-softmax, summed
    over each stream's true length (padded positions inert). Returns float32 [B] (cpu or device)."""
    lp = torch.log_softmax(pred_logits.float(), dim=-1)
    pick = lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)                      # [B, Tmax]
    mask = (torch.arange(pick.shape[1], device=pick.device).unsqueeze(0)
            < torch.as_tensor(lengths, device=pick.device).unsqueeze(1))
    return (pick * mask.to(pick.dtype)).sum(dim=1)


def _stream_attn(plen, batch, lens):
    attn = torch.ones((batch.shape[0], plen + batch.shape[1]), dtype=torch.long,
                      device=batch.device)
    ar = torch.arange(batch.shape[1], device=batch.device).unsqueeze(0)
    attn[:, plen:] = (ar < torch.as_tensor(lens, device=batch.device).unsqueeze(1)).long()
    return attn


@torch.no_grad()
def prefill(model, ctx):
    """One context prefill -> (legacy KV tuple, last-position logits [1, V] predicting token 0).
    Box attempt 2's lesson: depending on the transformers version/config, past_key_values comes
    back as a LEGACY TUPLE or as a Cache object -- accept both."""
    out = model(ctx, use_cache=True)
    pk = out.past_key_values
    if not isinstance(pk, tuple):
        pk = pk.to_legacy_cache()
    return pk, out.logits[:, -1].float()


@torch.no_grad()
def score_batch_kv(model, ctx, legacy_kv, first_logit, batch, lens):
    """KV-reuse path: expand the prefilled context cache across the batch, one forward over the
    stream tokens. pred[:, 0] = the context's last-position logits (predicting token 0)."""
    from transformers.cache_utils import DynamicCache
    B = batch.shape[0]
    past = DynamicCache.from_legacy_cache(tuple(
        (k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous())
        for k, v in legacy_kv))
    attn = _stream_attn(ctx.shape[1], batch, lens)
    step = model(batch, past_key_values=past, attention_mask=attn, use_cache=True).logits
    pred = torch.cat([first_logit.expand(B, -1).unsqueeze(1).to(step.dtype), step[:, :-1]], dim=1)
    return ll_from_logits(pred, batch, lens)


@torch.no_grad()
def score_batch_concat(model, ctx, batch, lens):
    """Reference path: full forward over [ctx + stream] (no KV plumbing)."""
    B = batch.shape[0]
    plen = ctx.shape[1]
    ids = torch.cat([ctx.expand(B, -1), batch], dim=1)
    logits = model(ids, attention_mask=_stream_attn(plen, batch, lens)).logits
    return ll_from_logits(logits[:, plen - 1: plen - 1 + batch.shape[1]], batch, lens)


def score_batch(model, ctx, kv, first, batch, lens, use_kv):
    """Post-selfcheck batch scoring (attempt-6 gate) -> (ll, use_kv). A KV path that raises
    MID-RUN (after a clean self-check) must not kill a paid box: fall back to the concat
    reference path for this and every later batch. The fallback print is collision-safe --
    exception TYPE only (the raw message could carry a watchdog fatal substring, e.g. a CUDA
    OOM), and no box-marker substring."""
    if use_kv:
        try:
            return score_batch_kv(model, ctx, kv, first, batch, lens), True
        except Exception as e:
            print(f"LR_KV_BATCH_FALLBACK {type(e).__name__} -- concat path from here on",
                  flush=True)
    return score_batch_concat(model, ctx, batch, lens), False


def contexts_for(ctxset):
    """[(label, concept)] for one context set; labels key the per-stream ll dict."""
    if ctxset == "N":
        return [("neutral", None)]
    return [(c, c) for c in C.COVERT_CONCEPTS]


def _assert_provenance(bundle, path):
    m = bundle.get("model")
    assert m in (None, C.ACTIVE), f"{path}: bundle model {m!r} != INTRO_MODEL {C.ACTIVE!r}"
    v = bundle.get("variant")
    assert v in (None, "orig"), f"{path}: prompt variant {v!r}, not the published 'orig'"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="exp1 capture .pt (injected streams)")
    ap.add_argument("--evoked", required=True, help="exp3 evoked bundle .pt")
    ap.add_argument("--evoked-alt", required=True, help="exp3 evoked_alt bundle .pt")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    outdir = C.RUN_DIR / "lr"
    outdir.mkdir(parents=True, exist_ok=True)
    sources = [("injected", args.capture), ("evoked", args.evoked),
               ("evoked_alt", args.evoked_alt)]
    model, tok = K.load_model(C.MODEL)
    use_kv = None                                       # decided once by the registered self-check
    t0 = time.time()

    for streamset, path in sources:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        _assert_provenance(bundle, path)
        streams = select_streams(bundle, streamset)
        del bundle
        print(f"LR set={streamset} n={len(streams)}", flush=True)
        for ctxset in CTX_SETS:
            shard = outdir / f"{C.ACTIVE}_{streamset}_{ctxset}.pt"
            if shard.exists():
                print(f"LR_SKIP {shard.name} (resume)", flush=True)
                continue
            recs = [dict(gidx=s["gidx"], concept=s.get("concept"), strength=s.get("strength"),
                         T=len(s["tokens"]), ll={}) for s in streams]
            for label, concept in contexts_for(ctxset):
                ctx = ctx_ids(tok, "A" if ctxset == "N" else ctxset, concept, model.device)
                kv, first = prefill(model, ctx)
                for lo in range(0, len(streams), args.batch):
                    chunk = streams[lo: lo + args.batch]
                    batch, lens = pad_tokens([s["tokens"] for s in chunk])
                    batch = batch.to(model.device)
                    if use_kv is None:                  # registered self-check, first batch only
                        ll_cc = score_batch_concat(model, ctx, batch, lens)
                        try:                            # a raising KV path must not kill the run
                            ll_kv = score_batch_kv(model, ctx, kv, first, batch, lens)
                            worst = float((torch.abs(ll_kv - ll_cc)
                                           / torch.as_tensor(lens, device=ll_kv.device)).max())
                            use_kv = worst <= SELFCHECK_TOL
                            print(f"{'LR_SELFCHECK_OK' if use_kv else 'LR_SELFCHECK_FALLBACK'} "
                                  f"max|dLL|/T={worst:.5f} (tol {SELFCHECK_TOL})", flush=True)
                        except Exception as e:
                            use_kv = False
                            # type name ONLY -- the raw message could carry a watchdog fatal
                            # substring (e.g. a CUDA OOM text) and kill a healthy fallback run
                            print(f"LR_SELFCHECK_FALLBACK exception {type(e).__name__}",
                                  flush=True)
                        ll = ll_kv if use_kv else ll_cc
                    else:
                        ll, use_kv = score_batch(model, ctx, kv, first, batch, lens, use_kv)
                    for i, v in enumerate(ll.cpu().tolist()):
                        recs[lo + i]["ll"][label] = float(v)
                print(f"LR ctx {streamset}x{ctxset}:{label} done t={int(time.time() - t0)}s",
                      flush=True)
            tmp = shard.with_suffix(".tmp")
            torch.save(dict(model=C.ACTIVE, streamset=streamset, ctxset=ctxset,
                            contexts=[l for l, _ in contexts_for(ctxset)],
                            selfcheck_kv=use_kv, records=recs), tmp)
            os.replace(tmp, shard)                                              # atomic
            # NB: nothing this module prints may contain box_lr's done marker as a substring --
            # labkit substring-matches markers on log lines, and the old per-shard line (the done
            # marker + "_SHARD") ended box attempt 4 after the first shard. Guarded by test M1.
            print(f"LR_SHARD_SAVED {shard.name} n={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
