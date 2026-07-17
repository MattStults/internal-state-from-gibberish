"""exp3 induction collector: generate word-free streams under prompt-INDUCTION (no steering vector), 3 arms.

Adapts exp1's covert_collect: drops the injection hook, and for each arm in {evoked, named, secret_word} loops
concepts (strength 1 = induced) + a neutral baseline (strength 0), composing the system prompt via
`primers.compose_system`. Everything else -- the anti-word GEN_PROMPT, sampling, the word-free filter, the
per-step top-K (`gen_topk`) capture -- is reused verbatim from exp1, so the bundles load through exp2's OWN
`filter_streams` unchanged (strong dose = the induced arm, s0 = neutral control). Also collects a persona-only
gauge run (evoked arm) for the blind-judge manipulation check.

Pure assembly/accounting (stream_record / acceptance_report / assemble_bundle) is unit-tested offline; the
generation loop needs a GPU. wordfreq MUST be installed in the collection env or the word-rate filter is inert.

RUNBOOK (B2 feasibility gate -- do NOT skip): run the SMALLEST model FIRST with the word filter live and the
gate on, then authorize larger models only if it clears:
    .venv/bin/python .../collect_induction.py --models qwen2.5-1.5b --min-per-class 24
The gate RAISES on the cheap box if any induced concept has < 24 analyzable (>=12-token) word-free streams --
catching a persona that floods real words BEFORE the 3B/7B spend and before common_n_subsample crashes. If a
concept is marginal, raise target_clean (cheap) or revise its primer, then re-run 1.5B. Then:
    .venv/bin/python .../collect_induction.py --models qwen2.5-3b qwen2.5-7b --min-per-class 24
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                  # primers
import primers_v3 as P                                    # noqa: E402  (drop-in superset of the
# frozen primers.py + primers_v2.py: identical output for all pre-registered exp3 arms, the
# confound-closing prereg arms sustained_s1..s3 / sustained_alt_s1..s3 / maintained_secret, and
# the lr_scale_grid Amendment 2 arm secret_sustain -- the same import-widening as primers ->
# primers_v2; byte-identity for every earlier arm is pinned by test_lr_grid_secret P3)

INDUCED, NEUTRAL_S = 1, 0                                 # strength codes: induced concept vs neutral baseline
# evoked_alt = the frozen invariance paraphrase (B1): collected by default so a positive evoked result can run
# the pre-registered wording-swap check without a SECOND paid collection. It is only REQUIRED on >=1 model, so
# cost-conscious larger-model runs may pass --arms evoked named secret_word.
ARMS = ("evoked", "evoked_alt", "named", "secret_word")


# ---------------------------------------------------------------- pure assembly / accounting (unit-tested)
def stream_record(gidx, concept, concept_idx, arm, tokens, text, deg, accepted, strength, gen_topk):
    """One generated stream's record, in the schema exp2's filter_streams reads (accepted + strength +
    gen_topk + tokens + concept_idx), plus arm/text/deg for provenance."""
    return dict(gidx=int(gidx), concept=concept, concept_idx=int(concept_idx), arm=arm,
                tokens=np.asarray(tokens), text=text, deg=deg, accepted=bool(accepted),
                strength=int(strength), gen_topk=gen_topk)


def acceptance_report(records, induced_strength=INDUCED):
    """Per-concept (n_accepted, n_total) word-free acceptance over the INDUCED streams (the feasibility gate).
    Neutral (strength != induced_strength) rows are excluded."""
    from collections import defaultdict
    acc, tot = defaultdict(int), defaultdict(int)
    for r in records:
        if int(r["strength"]) != induced_strength:
            continue
        tot[r["concept"]] += 1
        acc[r["concept"]] += int(r["accepted"])
    return {c: (acc[c], tot[c]) for c in tot}


def analyzable_report(records, min_tokens=12, induced_strength=INDUCED):
    """Per-concept count of ACCEPTED INDUCED streams with >= min_tokens gen_topk steps -- the streams that
    survive the analyzer's >=T-token filter (run_budget drops len(gen_topk)<max_budget). THIS, not raw
    acceptance, is what must clear the n-per-class floor, so a persona that floods real words (low acceptance)
    or produces very short streams is caught here."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for r in records:
        if int(r["strength"]) != induced_strength or not r["accepted"]:
            continue
        gtk = r.get("gen_topk")
        if gtk is not None and len(gtk) >= min_tokens:
            cnt[r["concept"]] += 1
    return dict(cnt)


def check_min_per_class(records, min_per_class, min_tokens=12, induced_strength=INDUCED):
    """B2 feasibility gate: RAISE if any induced concept has < min_per_class analyzable (>= min_tokens) streams.
    min_per_class=0 disables it. Run the SMALLEST model first with this on, so a low-acceptance concept fails
    loudly on the cheap box -- not inside common_n_subsample after the full 3B/7B spend."""
    if not min_per_class:
        return
    concepts = {r["concept"] for r in records if int(r["strength"]) == induced_strength}
    rep = analyzable_report(records, min_tokens=min_tokens, induced_strength=induced_strength)
    short = {c: rep.get(c, 0) for c in concepts if rep.get(c, 0) < min_per_class}
    if short:
        raise RuntimeError(f"feasibility gate FAILED: {len(short)} concept(s) below {min_per_class} analyzable "
                           f"(>= {min_tokens}-token) word-free streams: {short}. Raise target_clean or revise "
                           f"the primer before scaling up.")


def assemble_bundle(model, arm, concepts, records, layer=None, hf_id=None):
    """Package one (model, arm) collection into an exp2-loader-compatible bundle. inject=arm so downstream can
    tell the arms apart; strengths carries both the induced (1) and neutral (0) codes present."""
    strengths = sorted({int(r["strength"]) for r in records})
    return dict(model=model, inject=arm, concepts=list(concepts), K=len(concepts),
                strengths=strengths, streams=records, layer=layer, hf_id=hf_id)


# ---------------------------------------------------------------- generation (GPU; reused exp1 inner logic)
def _generate(model, tok, system, g, concept, concept_idx, arm, strength, gidx0):
    """Reject-resampled word-free generation for ONE (arm, concept, strength) condition. No injection.
    Mirrors covert_collect.gen_clean: same sampling, same filter, same gen_topk capture."""
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    import common as K
    import config as C
    from covert_collect import degeneracy, is_degenerate

    ids = K.chat_ids(tok, C.GEN_PROMPT, system=system, device=model.device)
    plen = ids.shape[1]
    out, nclean, gidx = [], 0, gidx0
    with torch.no_grad():
        while nclean < g["target_clean"] and len(out) < g["max_gen"]:
            b = min(g["gen_batch"], g["max_gen"] - len(out))
            rep = ids.repeat(b, 1)
            gout = model.generate(rep, attention_mask=torch.ones_like(rep), max_new_tokens=g["tokens"],
                                  do_sample=True, temperature=1.0, top_p=0.98, pad_token_id=tok.eos_token_id,
                                  return_dict_in_generate=True, output_logits=True)
            gen, steps = gout.sequences, gout.logits
            for r in range(b):
                row = gen[r, plen:]
                eos = (row == tok.eos_token_id).nonzero()
                rlen = int(eos[0]) + 1 if len(eos) else int(row.shape[0])
                row = row[:rlen].cpu()
                text = tok.decode(row, skip_special_tokens=True)
                d = degeneracy(text)
                acc = not is_degenerate(d)
                nclean += int(acc)
                gtk = None
                if acc:
                    gtk = []
                    for s in range(min(rlen, len(steps))):
                        lp = torch.log_softmax(steps[s][r].float(), dim=-1)
                        vals, idx = lp.topk(g["gen_topk"])
                        gtk.append(dict(ids=idx.cpu().numpy().astype(np.int32),
                                        logp=vals.cpu().numpy().astype(np.float16)))
                out.append(stream_record(gidx, concept, concept_idx, arm, row.numpy(), text, d, acc,
                                         strength, gtk))
                gidx += 1
            print(f"  {arm} {concept} s{strength}: {nclean}/{g['target_clean']} clean ({len(out)} gen)",
                  flush=True)
    return out, gidx


def _collect_gauge(model, tok, g):
    """Persona-only gauge run (EVOKED arm): free-association responses for the blind K-way judge. No anti-word
    block, no filter -- we want coherent words. Returns {concept|None: [responses]}."""
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    import common as K
    gauge = {}
    with torch.no_grad():
        for concept in [None] + list(P.CONCEPTS):
            ids = K.chat_ids(tok, P.GAUGE_PROBE, system=P.compose_gauge_system(concept, arm="evoked"),
                             device=model.device)
            rep = ids.repeat(g["gauge_n"], 1)
            gout = model.generate(rep, attention_mask=torch.ones_like(rep), max_new_tokens=64,
                                  do_sample=True, temperature=1.0, top_p=0.98, pad_token_id=tok.eos_token_id)
            gauge[concept or "neutral"] = [tok.decode(gout[i, ids.shape[1]:], skip_special_tokens=True)
                                           for i in range(g["gauge_n"])]
    return gauge


def cfg(smoke, pilot=False):
    if pilot:
        # E2 wording-qualification pilot (confound-closing prereg): 3 concepts x small target_clean,
        # but FULL-length 128-token streams (smoke's 48 never reaches the E4 trajectory gate's late
        # cuts t >= 32/64/127, so retention would be unmeasurable). n_concepts caps the concept loop.
        # target_clean=16: n=8 made the per-(concept,cut) neutral mu/sd (hence the retention z's) too
        # noisy to gate wordings on; 16 keeps the pilot cheap but the gate math meaningful.
        return dict(target_clean=16, max_gen=96, tokens=128, gen_batch=16, gen_topk=32, gauge_n=3,
                    n_concepts=3)
    if smoke:
        return dict(target_clean=4, max_gen=24, tokens=48, gen_batch=8, gen_topk=32, gauge_n=3)
    # gen_batch 32: a 1.5-3B model badly under-fills a 24GB card at 16 (labkit util_verdict='sawtooth'); 32
    # roughly halves the generate-call count. run_model caps it back to 16 for 7B/8B (VRAM). The shakedown's
    # util_verdict / vram_headroom tells us whether to push higher (48+) before the real runs.
    return dict(target_clean=36, max_gen=192, tokens=128, gen_batch=32, gen_topk=64, gauge_n=8)


def run_model(hf_id, model_slug, arms, g, smoke, min_per_class=0, min_tokens=12,
              on_model_ready=None):
    import torch  # noqa: F401
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    import common as K
    import config as C
    if not getattr(__import__("covert_collect"), "HAVE_WORDFREQ", False):
        msg = "wordfreq NOT installed -- the word-rate filter is INERT, so word-free acceptance is unenforced."
        if not smoke:                                    # mirror exp1: a real collect MUST have the word filter
            raise RuntimeError(msg + " Install it first: pip install wordfreq.")
        print("WARNING: " + msg + " (smoke only)", flush=True)
    model, tok = K.load_model(hf_id)
    print("MODEL_READY", flush=True)                    # labkit watchdog ready marker
    if on_model_ready is not None:
        on_model_ready()   # caller's timing seam (box_lr_extend emits a wall-clock step here
        #                    so the driver's spend projection can exclude the one-time weights
        #                    download from the generation slice -- 2026-07-14 review CRIT 2a)
    if model_slug.split("-")[-1].lower() in ("7b", "8b"):
        g = {**g, "gen_batch": min(g.get("gen_batch", 16), 16)}   # VRAM: cap batch for big models
    outdir = C.DATA                                      # runs/<slug>/data, or $INTRO_RUN_DIR/data so labkit pulls it
    os.makedirs(outdir, exist_ok=True)
    shard_dir = outdir / "shards"
    os.makedirs(shard_dir, exist_ok=True)
    concepts = list(P.CONCEPTS)[: int(g.get("n_concepts") or len(P.CONCEPTS))]   # --pilot: first 3 only
    for arm in arms:
        # RESUMABLE (confound-closing prereg): a finished arm's bundle .pt => skip the whole arm; else
        # per-(arm, concept) atomic shards resume the contiguous prefix (gidx continuity preserved).
        if (outdir / f"{model_slug}-{arm}.pt").exists():
            print(f"[{model_slug}/{arm}] RESUMED: bundle exists, skipping arm", flush=True)
            continue
        records, gidx, resume_ok = [], 0, True

        def _cell(name, sysmsg, cidx, strength, records, gidx, resume_ok):
            sp = shard_dir / f"{arm}_{name}.pt"
            if resume_ok and sp.exists():
                sh = torch.load(sp, map_location="cpu", weights_only=False)
                print(f"[{model_slug}/{arm}] RESUMED cell {name} ({len(sh['recs'])} streams)", flush=True)
                return records + sh["recs"], sh["next_gidx"], True
            recs, gidx = _generate(model, tok, sysmsg, g, name if cidx >= 0 else "neutral", cidx, arm, strength, gidx)
            tmp = sp.with_suffix(".tmp")
            torch.save(dict(recs=recs, next_gidx=gidx), tmp)
            os.replace(tmp, sp)                                    # atomic: no torn shards
            return records + recs, gidx, False

        for ci, concept in enumerate(concepts):
            sysmsg = P.compose_system(concept, C.STRONG_SYSTEM, arm=arm)
            records, gidx, resume_ok = _cell(concept, sysmsg, ci, INDUCED, records, gidx, resume_ok)
        neutral_sys = P.compose_system(None, C.STRONG_SYSTEM, arm=arm)     # strength-0 control
        records, gidx, resume_ok = _cell("neutral", neutral_sys, -1, NEUTRAL_S, records, gidx, resume_ok)
        bundle = assemble_bundle(model_slug, arm, concepts, records, hf_id=hf_id)
        if arm == "evoked":
            bundle["gauge"] = _collect_gauge(model, tok, g)               # manipulation check (evoked only)
        outpath = outdir / f"{model_slug}-{arm}.pt"
        tmp = outpath.with_suffix(".tmp")
        torch.save(bundle, tmp)
        os.replace(tmp, outpath)                           # atomic: the arm-resume check trusts this file
        print(f"[{model_slug}/{arm}] acceptance: {acceptance_report(records)}", flush=True)
        print(f"  analyzable(>= {min_tokens}tok): {analyzable_report(records, min_tokens)}", flush=True)
        print(f"  wrote {outpath}", flush=True)
        check_min_per_class(records, min_per_class, min_tokens)     # B2 gate: fail loudly if below the n-floor


def main():
    sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
    import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5-1.5b"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pilot", action="store_true",
                    help="E2 wording-qualification pilot sizing (confound prereg): 3 concepts, "
                         "target_clean=16, FULL 128-token streams (the trajectory gate needs cuts >= 32).")
    ap.add_argument("--min-per-class", type=int, default=0,
                    help="B2 feasibility gate: raise if any induced concept has < N analyzable streams "
                         "(set 24 for a real run; run the smallest model first).")
    ap.add_argument("--min-tokens", type=int, default=12,
                    help="min gen_topk steps for a stream to count as analyzable (matches run_budget's filter).")
    args = ap.parse_args()
    g = cfg(args.smoke, args.pilot)
    try:
        for slug in args.models:
            hf_id = C.MODELS[slug]["hf_id"] if slug in getattr(C, "MODELS", {}) else slug   # dict -> hf_id string
            print(f"\n===== {slug} ({hf_id}) arms={args.arms} =====", flush=True)
            run_model(hf_id, slug, args.arms, g, args.smoke, args.min_per_class, args.min_tokens)
    except Exception:
        print("COLLECT_FATAL", flush=True)              # labkit fatal marker: terminal (incl. feasibility gate) -- don't retry host
        raise
    print("COLLECT_DONE", flush=True)                   # labkit watchdog done marker


if __name__ == "__main__":
    main()
