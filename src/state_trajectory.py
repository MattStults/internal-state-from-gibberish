"""E4 (confound-closing prereg): per-position concept-state trajectory from teacher-forced re-forwards.

For each saved stream, re-forward [exact original context + saved tokens] (deterministic; nothing is
regenerated) and capture the read-layer residual at the position of generated token t for cuts
t in {2,4,8,16,32,64,127}. Project onto the SAVED unit concept vectors (steering primitives from the
exp1 capture -- never re-derived). Output per stream: {cut: {concept: h.v_hat}} raw projections; the
z(t) standardization (own-minus-other vs the s0 pool) happens OFFLINE from these primitives.

Arms (--arm):
  evoked           streams from runs/_ind/<slug>/data/<slug>-evoked.pt, context = primers_v2 evoked system
  injected         streams from the exp1 capture (s>0), context = word-free system, injection hook LIVE
                   (gen-mode: reproduces generation-time state exactly, teacher-forced)
  s0               exp1 capture s==0 streams, clean forward (the standardization pool)
  pilot:<arm>      any primers_v2 arm (e.g. pilot:sustained_s2) reading a pilot bundle -- the E2
                   wording-qualification gate's measurement
  gauge:<base>     bundle["gauge"] free-association texts (e.g. gauge:evoked), re-forwarded under the
                   GAUGE context they were generated in: compose_gauge_system(concept, arm=<base>)
                   (induction text ALONE, no anti-word block) + GAUGE_PROBE as the user message (NOT
                   GEN_PROMPT). Discriminates Story A (persona state IS installed in free behavior and
                   the anti-word task displaces it) from Story B (never installed in the injected-vector
                   basis at all). FIDELITY CAVEAT: the gauge bundle stores TEXTS, not token ids, so the
                   streams are RE-TOKENIZED (tok(text, add_special_tokens=False)); tokenizer merges may
                   not reproduce the exact sampled token boundaries, so per-position residuals are those
                   of the re-tokenized stream -- fine for the z(t) trajectory read, not bit-exact to
                   generation time. gidx is synthetic (bundle order); concept "neutral" -> the arm's
                   neutral gauge context.

RESUMABLE: one atomic shard per (slug, arm): $INTRO_RUN_DIR/trajectory/<slug>_<arm>.pt; existing shard
=> skip (print TRAJ_SKIP). Stream-by-stream forwards (per-stream contexts differ); CPU-offloaded accumulation.
Run on GPU via the orchestrator; INTRO_MODEL selects the model as usual.
"""
import argparse
import os
import sys

import numpy as np
import torch

import config as C
import common as K

CUTS = (2, 4, 8, 16, 32, 64, 127)


def _exp3_primers():
    """(primers, primers_v2) from exp3, lazily (the frozen module owns GAUGE_PROBE; v2 owns arms)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "experiments", "exp3_induction_and_scale"))
    import primers
    import primers_v2
    return primers, primers_v2


def system_text(arm, concept):
    """System string for one stream's re-forward. concept in (None, 'neutral') selects the arm's
    strength-0 baseline (collect_induction records neutral streams with concept='neutral'; passing
    that string into primers raises KeyError -- it must map to compose_system(None, ...))."""
    if concept in (None, "neutral"):
        concept = None
    if arm in ("injected", "s0"):
        return C.STRONG_SYSTEM
    _, P = _exp3_primers()
    if arm.startswith("gauge:"):
        # gauge texts were generated under the induction text ALONE (no anti-word block)
        return P.compose_gauge_system(concept, arm=arm.split(":", 1)[1])
    a = arm.split(":", 1)[1] if arm.startswith("pilot:") else arm
    return P.compose_system(concept, C.STRONG_SYSTEM, arm=a)


def user_text(arm):
    """User message for the re-forward: gauge streams were generated under GAUGE_PROBE (free
    association), everything else under the word-free GEN_PROMPT."""
    if arm.startswith("gauge:"):
        P0, _ = _exp3_primers()
        return P0.GAUGE_PROBE
    return C.GEN_PROMPT


def _ctx_ids(tok, arm, concept, device):
    return K.chat_ids(tok, user_text(arm), system=system_text(arm, concept), device=device)


def gauge_pool(bundle):
    """bundle["gauge"] ({concept: list[str]}, incl. "neutral") -> synthetic stream records with
    sequential gidx and the TEXT still attached (tokens need the tokenizer; see retokenized)."""
    out = []
    for concept in sorted(bundle["gauge"]):
        for text in bundle["gauge"][concept]:
            out.append(dict(gidx=len(out), concept=concept,
                            strength=0 if concept == "neutral" else 1,
                            accepted=True, text=text))
    return out


def retokenized(tok, text):
    """Gauge text -> list[int] token ids with NO special tokens: the ids must append directly after
    the chat template's assistant generation prompt, exactly like saved generated tokens do (any
    BOS/EOS the tokenizer added would shift every per-position cut). FIDELITY CAVEAT: re-tokenization
    of decoded text may not reproduce the exact token boundaries the model originally sampled."""
    return list(tok(text, add_special_tokens=False)["input_ids"])


def stream_token_ids(tokens, device=None):
    """Saved stream tokens -> [1, T] long tensor. Tokens are list[int]/np arrays (collect_induction)
    OR 1-D torch tensors (exp1 captures) -- torch.tensor([tensor]) raises TypeError on the latter."""
    return torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=device).unsqueeze(0)


def injection_input(raw_vectors, inject_alpha, concept, strength):
    """(vector, alpha) exactly as the injection hook must receive them: the RAW saved vector with the
    SAVED alpha (covert_collect saves alpha = strength/||v_raw||, so ||alpha*v_raw|| == strength).
    NEVER feed the unit-normalized projection vectors here -- that under-doses by ~||v_raw|| (the
    hook would add ||delta|| ~= 1.5 instead of the saved eff_mag 60)."""
    v = torch.as_tensor(np.asarray(raw_vectors[concept]), dtype=torch.float32)
    alpha = float(inject_alpha[f"{concept}|s{strength}"])
    return v, alpha


def balanced_pool(pool, per_concept):
    """Per-concept cap (neutral counts as its own class). Bundles are stored concept-major, so a
    global prefix cap (pool[:N]) silently drops every late-listed concept from the evoked legs."""
    out, counts = [], {}
    for s in pool:
        c = s.get("concept")
        if counts.get(c, 0) < per_concept:
            counts[c] = counts.get(c, 0) + 1
            out.append(s)
    return out


@torch.no_grad()
def trajectory(model, tok, layer, streams, arm, vectors, inject_alpha=None, raw_vectors=None):
    """vectors: {concept: unit np vector} (projections ONLY). raw_vectors: {concept: RAW saved vector}
    for the injected arm's hook. Returns [{gidx, concept, strength, proj: {cut: {concept: float}}}]."""
    vmat = torch.tensor(np.stack([vectors[c] for c in sorted(vectors)]), dtype=torch.float32)  # [K, d]
    names = sorted(vectors)
    out = []
    for s in streams:
        ctx = _ctx_ids(tok, arm, s.get("concept"), model.device)
        plen = ctx.shape[1]
        toks = stream_token_ids(s["tokens"], device=model.device)
        ids = torch.cat([ctx, toks], dim=1)
        hook = None
        if arm == "injected":
            v, alpha = injection_input(raw_vectors, inject_alpha, s["concept"], s["strength"])
            hook = model.model.layers[layer].register_forward_hook(
                K._injection_hook(v.to(model.device), alpha, prompt_len=plen))
        try:
            with K.Capture(model, layer) as capt:
                model(ids)
            h = capt.acts[0][0].float().cpu()                              # [plen+T, d]
        finally:
            if hook is not None:
                hook.remove()
        T = len(s["tokens"])
        proj = {}
        for t in CUTS:
            if t <= T:
                hv = h[plen + t - 1]                                        # residual at generated token t
                proj[t] = {n: float(hv @ vmat[i]) for i, n in enumerate(names)}
        out.append(dict(gidx=s["gidx"], concept=s.get("concept"), strength=s.get("strength"),
                        accepted=s.get("accepted", True), proj=proj, T=T))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--bundle", required=True, help="path to the stream bundle (.pt) to re-forward")
    ap.add_argument("--vectors-from", required=True,
                    help="exp1 capture .pt holding inject_vectors (+ inject_alpha for --arm injected)")
    ap.add_argument("--max-per-concept", type=int, default=25,
                    help="balanced cap per concept incl. neutral (accepted streams only); a global "
                         "prefix cap would drop late-listed concepts from concept-major bundles")
    args = ap.parse_args()

    outdir = C.RUN_DIR / "trajectory"
    outdir.mkdir(parents=True, exist_ok=True)
    shard = outdir / f"{C.ACTIVE}_{args.arm.replace(':', '-')}.pt"
    if shard.exists():
        print(f"TRAJ_SKIP {shard.name} (resume)", flush=True)
        return

    cap = torch.load(args.vectors_from, map_location="cpu", weights_only=False)
    # provenance: vectors must come from a capture of THIS model, taken under the published instrument
    cap_model = cap.get("model")
    assert cap_model in (None, C.ACTIVE), \
        f"--vectors-from capture is for model {cap_model!r}, but INTRO_MODEL={C.ACTIVE!r}"
    cap_variant = cap.get("variant")
    assert cap_variant in (None, "orig"), \
        f"--vectors-from capture used prompt variant {cap_variant!r}, not the published 'orig'"
    raw_vectors = cap["inject_vectors"]                    # RAW saved steering primitives (for the hook)
    vectors = {c: (v / np.linalg.norm(v)).astype(np.float32) for c, v in raw_vectors.items()}  # projections
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    bundle_variant = bundle.get("variant")
    assert bundle_variant in (None, "orig"), \
        f"--bundle was collected under prompt variant {bundle_variant!r}, not the published 'orig'"
    if args.arm.startswith("gauge:"):
        if "gauge" not in bundle:
            raise SystemExit(f"--bundle carries no 'gauge' texts (required for --arm {args.arm})")
        pool = balanced_pool(gauge_pool(bundle), args.max_per_concept)
    else:
        pool = [s for s in bundle["streams"] if s.get("accepted", True) and len(s["tokens"]) >= 2]
        if args.arm == "injected":
            smax = max(s["strength"] for s in pool)
            pool = [s for s in pool if s["strength"] == smax]
        elif args.arm == "s0":
            pool = [s for s in pool if s["strength"] == 0]
        pool = balanced_pool(pool, args.max_per_concept)

    model, tok = K.load_model(C.MODEL)
    if args.arm.startswith("gauge:"):                      # re-tokenize now that the tokenizer exists
        for s in pool:
            s["tokens"] = retokenized(tok, s.pop("text"))
        pool = [s for s in pool if len(s["tokens"]) >= 2]
    layer = C.resolve_layer(model.config.num_hidden_layers)
    print(f"TRAJ start arm={args.arm} n={len(pool)} layer={layer}", flush=True)
    recs = trajectory(model, tok, layer, pool, args.arm, vectors,
                      inject_alpha=cap.get("inject_alpha"), raw_vectors=raw_vectors)
    tmp = shard.with_suffix(".tmp")
    torch.save(dict(arm=args.arm, model=C.ACTIVE, layer=layer, cuts=list(CUTS),
                    concepts=sorted(vectors), records=recs), tmp)
    os.replace(tmp, shard)                                                  # atomic
    print(f"TRAJ_DONE {shard.name} n={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
