"""Data-COLLECTION pass for the covert-channel / introspection-leakage experiment.

Philosophy: the GPU run does the model-requiring measurements and
saves the RAW, non-composite primitives; ALL metric construction (B-C vs ceiling, the activation
probe, the offline n-gram reader, clean-stream stratification, multiple C controls, the overt-leak
audit) happens offline from the saved artifacts, with zero re-runs.

Apparatus: Qwen2.5 1.5B/3B/7B, a blog-faithful concept vector injected GENERATION-ONLY at the introspection
layer (the prompt + the "; secret word:" probe context are left clean; open-introspection post-02 convention),
12 concept "secret words", strong anti-word (word-free) prompt.

Changes forced by the design review:
  * Reject-resample: keep generating until TARGET_CLEAN non-degenerate streams / concept / strength
    (cap MAX_GEN); SAVE rejected streams + acceptance counts (the degeneration rate is itself a result).
  * Strengths {0(null), 40, 60} -- s80 is ~95-98% degenerate (can't resample affordably).
  * Two read arms, each saving grid activations + per-cut per-word logprob:
      B (clean): un-injected reader of the stream  -> the covert channel.
      A (source): reader with the injection hook LIVE -> NOT a stream-channel ceiling but
                  "source nameability" (the vector points at the word at the read token). Relabelled.
  * Save per (arm, stream, cut): first-token logP + full-vocab rank for EVERY one of the 12 words,
    PLUS top-K logits (so any distractor set / margin / top-1 is reconstructable offline).
  * Also saved (for the proposed follow-up, no extra forwards): the FINAL-layer residual h_final per cut
    (offline propagation/bottleneck), the exact injected vectors v_hat + per-(concept,strength) alpha
    (steering primitives), and per-step top-K next-token logprobs of accepted streams (sampling/distribution).
  * Pre-run gate: assert the 12 concept first-token ids are distinct (a collision corrupts arm C).

NOT saved on purpose (offline-derivable from the streams, no model): the n-gram one-vs-rest reader,
and every composite (within-K rank, top-1, softmax margins, net-detection).
Known limitation: we save FIRST-TOKEN logP per word (one cheap forward/cut), not the multi-token
full-word sequence logprob. B-C is a difference of the same first-token score across streams, so it is
a valid consistent measure; full-word logprob would be a re-run (top-K logits are saved as a hedge).

Run on GPU via the labkit driver:  python3 harness/run_labkit.py <slug> --gpu RTX_4090 [--smoke]
"""
import argparse
import json
import os
import re
import time

import numpy as np
import torch

import config as C
import common as K

try:
    from wordfreq import zipf_frequency
    HAVE_WORDFREQ = True
except Exception:
    HAVE_WORDFREQ = False

SUFFIX = "; secret word:"
STRONG_SYSTEM = C.STRONG_SYSTEM    # 'orig' variant (default); --variant selects others from C.PROMPT_VARIANTS
PROMPT = C.GEN_PROMPT
ZIPF_MIN = 2.5
_SPLIT = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")

# LABKIT_STEP {json} lines: labkit's watchdog (lifecycle/remote.py) parses these into status.json's
# step/metrics for LIVE per-model progress (no --stream firehose). INTRO_STEP_BASE keeps `step` globally
# monotonic across the batched per-model runs (the reporter ignores any step <= the last one it saw).
STEP_BASE = int(os.environ.get("INTRO_STEP_BASE", "0"))


def emit_step(cell, **fields):
    print("LABKIT_STEP " + json.dumps({"step": STEP_BASE + int(cell), **fields}), flush=True)


def cfg(smoke):
    # strengths SCALED per model (C.strengths(): 0 + base eff_mags x resid_norm/REF_NORM). The smoke
    # tests the SAME scaled operating points (so its arm-A check verifies the real full-run eff_mags).
    if smoke:
        return dict(concepts=C.COVERT_CONCEPTS[:3], strengths=C.strengths(), target_clean=4, max_gen=24,
                    tokens=48, grid=[8, 24, 47], gen_batch=8, topk=64, gen_topk=32)
    return dict(concepts=C.COVERT_CONCEPTS, strengths=C.strengths(), target_clean=30, max_gen=120,
                tokens=128, grid=[2, 4, 8, 16, 32, 64, 127], gen_batch=16, topk=256, gen_topk=64)


# ------------------------------------------------------------- degeneration (a measured covariate)
def degeneracy(text):
    runs = [m.group(0) for m in _SPLIT.finditer(text) if len(m.group(0)) >= 3]
    if runs and HAVE_WORDFREQ:
        wr = sum(1 for r in runs if zipf_frequency(r.lower(), "en") >= ZIPF_MIN) / len(runs)
    else:
        wr = 0.0
    tri = [text[i:i + 3] for i in range(len(text) - 2)]
    rep = 1 - len(set(tri)) / len(tri) if tri else 0.0
    alpha = [c for c in text if c.isalpha()]
    nonlat = sum(1 for c in alpha if not ("a" <= c.lower() <= "z")) / len(alpha) if alpha else 0.0
    spacing = sum(c.isspace() for c in text) / len(text) if text else 0.0
    return dict(word_rate=float(wr), repetition=float(rep), non_latin=float(nonlat), spacing=float(spacing))


def is_degenerate(d):
    # also reject streams that contain real words (word_rate) -- enforces the word-free constraint at
    # acceptance, so the "clean" population (incl. the s0 control) is genuinely word-free, not prose.
    return bool(d["repetition"] > 0.6 or d["non_latin"] > 0.3 or d["spacing"] > 0.5 or d["word_rate"] > 0.1)


# ------------------------------------------------------------- generation (reject-resampled, batched)
@torch.no_grad()
def gen_clean(model, tok, concept, strength, layer, g, system, inject_mode="gen",
              inject_override=None):
    """Generate until >= target_clean non-degenerate streams (cap max_gen). Returns the full list of
    (token_ids, text, deg_dict, accepted) -- kept AND rejected -- plus the injection vector used.
    `system` is the (variant-selected) word-free system prompt. inject_mode: 'gen' injects only the
    generated positions (prompt clean, default); 'all' = legacy ALL-position; 'prompt' injects ONLY the
    prompt prefill and leaves every generated position clean (a static, persona-like perturbation with a
    dose knob -- the confound-closing prereg's E3 arm). inject_override=(vector, alpha): use a capture's
    OWN stored steering primitives instead of re-deriving (Amendment 2 (2b) of the lr_scale_extend
    prereg: stored-vector reuse); ignored at strength 0."""
    inject = None
    if strength > 0:
        if inject_override is not None:
            v = torch.as_tensor(np.asarray(inject_override[0]), dtype=torch.float32)
            inject = (v, float(inject_override[1]))
        else:
            base = [w for w in C.BASELINE_WORDS if w.lower() != concept.lower()]
            v = K.concept_vector_blog(model, tok, concept, base, layer)
            inject = (v, strength / v.norm().item())
    ids = K.chat_ids(tok, PROMPT, system=system)
    plen = ids.shape[1]
    handle = (model.model.layers[layer].register_forward_hook(
                  K._injection_hook(*inject, prompt_len=(None if inject_mode == "all" else plen),
                                    prompt_only=(inject_mode == "prompt")))
              if inject else None)
    out, nclean = [], 0
    topk_g = g["gen_topk"]
    try:
        while nclean < g["target_clean"] and len(out) < g["max_gen"]:
            b = min(g["gen_batch"], g["max_gen"] - len(out))
            rep = ids.repeat(b, 1)
            gout = model.generate(rep, attention_mask=torch.ones_like(rep),
                                  max_new_tokens=g["tokens"], do_sample=True, temperature=1.0, top_p=0.98,
                                  pad_token_id=tok.eos_token_id,
                                  return_dict_in_generate=True, output_logits=True)
            gen = gout.sequences
            steps = gout.logits        # tuple(#generated steps) of [b, vocab] RAW next-token logits (pre-warp)
            for r in range(b):
                row = gen[r, plen:]
                eos = (row == tok.eos_token_id).nonzero()
                rlen = int(eos[0]) + 1 if len(eos) else int(row.shape[0])
                row = row[:rlen].cpu()
                text = tok.decode(row, skip_special_tokens=True)
                d = degeneracy(text)
                acc = not is_degenerate(d)
                nclean += int(acc)
                # per-step top-K next-token logprobs (TRUE distribution, pre warp), ACCEPTED streams only
                gtk = None
                if acc:
                    gtk = []
                    for s in range(min(rlen, len(steps))):
                        lp = torch.log_softmax(steps[s][r].float(), dim=-1)
                        vals, idx = lp.topk(topk_g)
                        gtk.append(dict(ids=idx.cpu().numpy().astype(np.int32),
                                        logp=vals.cpu().numpy().astype(np.float16)))
                out.append(dict(tokens=row, text=text, deg=d, accepted=acc, gen_topk=gtk))
            print(f"  gen {concept} s{strength}: {nclean}/{g['target_clean']} clean "
                  f"({len(out)} generated)", flush=True)
    finally:
        if handle:
            handle.remove()
    return out, (inject[0] if inject else None), (inject[1] if inject else None)


# ------------------------------------------------------------- reads (activations + per-word logP)
def _final_logits(model, prefixes):
    """Final-(real)-position logits for each prefix (list of 1D LongTensors of varying length), returned as a
    list of [vocab] tensors -- the per-cut read's hot path: ONE forward per cut (the trusted reference).
    Any registered injection hook applies per position, so it composes unchanged."""
    return [model(p.unsqueeze(0)).logits[0, -1] for p in prefixes]


@torch.no_grad()
def read_stream(model, tok, stream, layer, last_layer, grid, cfirst_ids, suffix_ids, ctx0, topk, inject_hook,
                perf=None):
    """One arm's read of one stream. Registers inject_hook if given (arm A). Returns (acts, reads, hfinal):
      acts:   {t: np.array(d_model)} introspection-layer (`layer`) residual at each grid cut,
      reads:  {t: dict(logp=[12], rank=[12], topk_ids=[K], topk_logp=[K])} at each grid cut -- the first-token
              logP / full-vocab rank of each concept word after [ctx + stream[:t] + "; secret word:"],
      hfinal: {t: np.array(d_model)} FINAL-layer residual at each grid cut (for offline propagation/bottleneck).
      One forward per cut (the trusted path). `perf` (optional): accumulate the acts-forward vs per-cut-forward
      time split + forward count (read_acts_s / read_cuts_s / n_read_forwards)."""
    dev = ctx0.device
    handle = (model.model.layers[layer].register_forward_hook(inject_hook) if inject_hook else None)
    try:
        # activations: one forward over [ctx + full stream]; capture the introspection-layer AND final-layer residuals
        _t = time.perf_counter()
        full = torch.cat([ctx0, stream.to(dev)]).unsqueeze(0)
        with K.Capture(model, layer) as cap, K.Capture(model, last_layer) as capF:
            model(full)
        if perf is not None:
            perf["read_acts_s"] += time.perf_counter() - _t; perf["n_read_forwards"] += 1
        gen = cap.acts[-1][0][ctx0.shape[0]:]                  # [gen_len, d]
        genF = capF.acts[-1][0][ctx0.shape[0]:]
        # only cuts the stream reaches, matching the per-cut read below so acts/reads/hfinal share a key set
        acts = {t: gen[t].float().cpu().numpy() for t in grid if t < gen.shape[0]}
        hfinal = {t: genF[t].float().cpu().numpy() for t in grid if t < genF.shape[0]}

        # per-cut prefill read: logits at the position after "; secret word:" (one forward per cut)
        reads = {}
        cfid = torch.tensor([cfirst_ids[c] for c in range(len(cfirst_ids))], device=dev)
        valid_t = [t for t in grid if t < stream.shape[0]]
        prefixes = [torch.cat([ctx0, stream[:t].to(dev), suffix_ids]) for t in valid_t]
        _t = time.perf_counter()
        logit_vecs = _final_logits(model, prefixes)
        if perf is not None:
            perf["read_cuts_s"] += time.perf_counter() - _t
            perf["n_read_forwards"] += len(prefixes)
        for t, logits in zip(valid_t, logit_vecs):
            logp = torch.log_softmax(logits, dim=-1)
            rank = (logits > logits[cfid][:, None]).sum(1) + 1  # full-vocab rank per concept word
            tk = torch.topk(logits, topk)
            reads[t] = dict(logp=logp[cfid].float().cpu().numpy(),       # .float(): bf16 -> numpy
                            rank=rank.cpu().numpy().astype(int),
                            topk_ids=tk.indices.cpu().numpy().astype(int),
                            topk_logp=logp[tk.indices].float().cpu().numpy())
        return acts, reads, hfinal
    finally:
        if handle:
            handle.remove()


def tune_effmags_from_evaluate(evaluate, *, lo, hi0, targets=C.CAL_TARGET_NATS,
                               rel_floor=C.CAPABILITY_REL_FLOOR, min_floor=C.CAPABILITY_MIN_FLOOR,
                               iters=8, base_effmags=None, bracket_meta=None,
                               n_cal=None, n_streams=None, cal_tokens=None):
    """PURE search core (no model, unit-testable): `evaluate(effmag) -> (nameability_nats, clean_frac)`.

    Tunes to arm-A NAMEABILITY in NATS (higher = stronger), NOT a rank proxy: under generation-only a fixed
    rank no longer maps to the nats we want (rank ~21 -> ~4.5 nats vs ~8 under all-position), so rank-targeting
    under-injected. First measures the uninjected baseline clean (evaluate(0)) -> the model's word-free CEILING,
    and sets the capability floor RELATIVE to it: floor = max(min_floor, rel_floor*base_clean) (the fix for
    low-ceiling models that can never clear an absolute floor). Then for each TARGET nats (medium, strong)
    BINARY-SEARCH eff_mag for the MINIMAL dose with nats >= target AND clean >= floor (nats INCREASES, clean
    DECREASES with eff_mag, so the floor caps the dose). If a target is UNREACHABLE while the model stays
    capable, CAP at the strongest clean dose (max-achievable nats) -- never under-inject via base_effmags. Only
    fall back to base if NO dose keeps clean >= floor (truly incapacitated). Returns ([med, strong], meta)."""
    base_effmags = list(C.BASE_EFFMAGS) if base_effmags is None else list(base_effmags)
    trace = []
    _, base_clean = evaluate(0.0)                       # uninjected -> capability ceiling (also sets s0 baseline)
    floor = max(min_floor, round(rel_floor * base_clean, 3))
    print(f"  CAL baseline clean (no injection) = {base_clean:.2f} -> relative floor = {floor:.2f} "
          f"(rel {rel_floor} x base, min {min_floor}); target nats = {list(targets)}", flush=True)
    cap = {"effmag": None, "nats": -1e9}               # strongest clean-ok dose (for unreachable targets)

    def bisect_one(target, hi):
        a, b, best, floor_hit = lo, hi, None, False
        for _ in range(iters):
            mid = (a + b) / 2.0
            nats, cf = evaluate(mid)
            print(f"  CAL nats->{target}: effmag={mid:.0f}  nats={nats:.2f}  clean={cf:.2f}", flush=True)
            trace.append(dict(target=target, effmag=round(mid, 1), nats=round(float(nats), 2),
                              clean=round(cf, 3), bracket=[round(a, 1), round(b, 1)], hi=round(hi, 1)))
            if cf >= floor and nats > cap["nats"]:     # remember the strongest dose we can actually run
                cap["effmag"], cap["nats"] = mid, float(nats)
            if cf < floor:              # incapacitating (vs this model's own ceiling) -> lower eff_mag
                b, floor_hit = mid, True
            elif nats < target:         # not yet nameable enough -> raise eff_mag
                a = mid
            else:                       # nats>=target with capacity retained -> accept, search lower for min dose
                best, b = mid, mid
        return best, floor_hit

    tuned = {}
    for target in targets:
        hi = hi0
        for _ in range(3):              # expand only if never incapacitated AND still too weak (reachable higher)
            best, floor_hit = bisect_one(target, hi)
            if best is not None or floor_hit:
                break
            hi *= 2.0
        tuned[target] = best

    def cal_meta(result, note=None):
        return dict(bracket=bracket_meta or dict(lo=lo, hi0=round(hi0, 1)),
                    target_nats=list(targets), rel_floor=rel_floor, min_floor=min_floor,
                    base_clean=round(base_clean, 3), floor=floor, n_cal=n_cal, n_streams=n_streams,
                    cal_tokens=cal_tokens, iters=iters, evaluations=trace,
                    cap=dict(effmag=round(cap["effmag"], 1), nats=round(cap["nats"], 2)) if cap["effmag"] is not None else None,
                    tuned={str(k): (round(v, 1) if v is not None else None) for k, v in tuned.items()},
                    result=result, note=note)

    if cap["effmag"] is None:           # no dose kept clean >= floor -> truly incapacitated
        print(f"  CAL WARNING: clean < floor {floor:.2f} at EVERY injection -> fallback to BASE_EFFMAGS {base_effmags}", flush=True)
        return base_effmags, cal_meta(base_effmags, note=f"fallback: clean<floor {floor:.2f} at every injection")

    med_t, strong_t = targets
    med_capped, strong_capped = tuned[med_t] is None, tuned[strong_t] is None
    # a hit target -> its tuned dose; a capped (unreachable-but-capable) target -> the strongest clean-ok dose,
    # FLOORED (int) so the strength never rounds ABOVE the verified-clean cap into incapacitation.
    med = round(tuned[med_t]) if not med_capped else int(cap["effmag"])
    strong = round(tuned[strong_t]) if not strong_capped else int(cap["effmag"])
    notes = []
    if strong_capped:
        notes.append(f"CAPPED: strong target {strong_t} nats unreachable while capable -> max-achievable "
                     f"{cap['nats']:.1f} nats at eff_mag {cap['effmag']:.0f}")
    if med_capped:
        notes.append(f"CAPPED: medium target {med_t} nats unreachable while capable -> max-achievable")
    if base_clean < 0.5:                # surface the comparability caveat in the trace, don't silently tune
        notes.append(f"LOW baseline clean {base_clean:.2f}: weak word-free capability, the clean population is "
                     "a survivor set -- report base_clean as a covariate / reconsider comparability")
    if strong <= med:                   # window collapsed them -> keep two DISTINCT operating points
        if med_capped and strong_capped:    # both pinned at the verified cap -> lower MED, don't push strong past it
            med = max(round(lo), round(0.80 * strong))
            notes.append("distinctness guard: both targets capped -> lowered med below the verified cap")
        else:
            strong = med + max(10, round(0.08 * med))
            notes.append("distinctness guard: med/strong collapsed -> forced apart")
    print(f"  CAL CONVERGED: med={med} strong={strong} (target nats {med_t}/{strong_t}; "
          f"cap {cap['nats']:.1f} nats @ {cap['effmag']:.0f}){'; ' + '; '.join(notes) if notes else ''}", flush=True)
    return [med, strong], cal_meta([med, strong], note="; ".join(notes) if notes else None)


@torch.no_grad()
def calibrate_effmags(model, tok, layer, concepts, ctx0, suffix_ids, cfirst,
                      targets=C.CAL_TARGET_NATS, rel_floor=C.CAPABILITY_REL_FLOOR,
                      min_floor=C.CAPABILITY_MIN_FLOOR, n_cal=3, n_streams=12, cal_tokens=128, iters=8):
    """On-box auto-tune. Builds the model-coupled `evaluate` (inject at `effmag`, generate word-free streams,
    measure arm-A NAMEABILITY in nats = arm-A logP(concept) minus the uninjected s0 baseline, + clean fraction)
    and hands it to the pure search core tune_effmags_from_evaluate. Tuning on nats (not a rank proxy) is the
    fix for the generation-only under-injection. Logs per-concept nats + median rank each step so a recurrence
    is diagnosable from run.log."""
    dev = ctx0.device
    gen_ids = ctx0.unsqueeze(0)
    plen = gen_ids.shape[1]
    vecs = {ci: K.concept_vector_blog(model, tok, concepts[ci],
                                      [w for w in C.BASELINE_WORDS if w.lower() != concepts[ci].lower()], layer)
            for ci in range(min(n_cal, len(concepts)))}
    vnorm = max(float(v.norm()) for v in vecs.values())
    rn = C.MODELS[C.ACTIVE].get("resid_norm") or vnorm    # bracket from RESIDUAL norm (the eff_mag scale), not ||v||
    lo, hi0 = 10.0, max(120.0, 1.5 * 60.0 * rn / C.REF_NORM)
    print(f"  CAL bracket: ||v||~{vnorm:.0f} resid_norm~{rn:.0f} -> [{lo:.0f},{hi0:.0f}]", flush=True)
    baseline_logp = {}        # per cal-concept uninjected arm-A logP(concept); set on the evaluate(0) call

    def evaluate(effmag):
        logps = {ci: [] for ci in vecs}
        ranks, clean, tot = [], 0, 0
        for ci, v in vecs.items():
            # generation-only: prompt_len=plen leaves ctx0 clean for both the generate and the arm-A read below
            h = model.model.layers[layer].register_forward_hook(
                K._injection_hook(v, effmag / v.norm().item(), prompt_len=plen))
            try:
                b = gen_ids.repeat(n_streams, 1)
                gen = model.generate(b, attention_mask=torch.ones_like(b), max_new_tokens=cal_tokens,
                                     do_sample=True, temperature=1.0, top_p=0.98, pad_token_id=tok.eos_token_id)
                for r in range(n_streams):
                    row = gen[r, plen:]
                    eos = (row == tok.eos_token_id).nonzero()
                    if len(eos):
                        row = row[:int(eos[0]) + 1]
                    tot += 1
                    if not is_degenerate(degeneracy(tok.decode(row, skip_special_tokens=True))):
                        clean += 1
                        logits = model(torch.cat([ctx0, row.to(dev), suffix_ids]).unsqueeze(0)).logits[0, -1]
                        logps[ci].append(float(torch.log_softmax(logits, dim=-1)[cfirst[ci]]))
                        ranks.append(int((logits > logits[cfirst[ci]]).sum().item()) + 1)
            finally:
                h.remove()
        clean_frac = clean / tot if tot else 0.0
        mean_lp = {ci: (float(np.mean(logps[ci])) if logps[ci] else None) for ci in vecs}
        if effmag <= 0:                  # uninjected -> establish the s0 baseline arm-A logP per concept
            baseline_logp.update({ci: (lp if lp is not None else -20.0) for ci, lp in mean_lp.items()})
            print("  CAL s0 baseline arm-A logP: "
                  + ", ".join(f"{concepts[ci]}={baseline_logp[ci]:.2f}" for ci in vecs), flush=True)
            return 0.0, clean_frac
        lifts = {ci: mean_lp[ci] - baseline_logp.get(ci, -20.0) for ci in vecs if mean_lp[ci] is not None}
        mean_nats = float(np.mean(list(lifts.values()))) if lifts else 0.0
        med_rank = float(np.median(ranks)) if ranks else 1e9
        print("    per-concept lift(nats): "
              + ", ".join(f"{concepts[ci]}={lifts.get(ci, float('nan')):.1f}" for ci in vecs)
              + f"  | med_rank={med_rank:.0f}  n_clean={len(ranks)}", flush=True)
        return mean_nats, clean_frac

    return tune_effmags_from_evaluate(
        evaluate, lo=lo, hi0=hi0, targets=targets, rel_floor=rel_floor, min_floor=min_floor,
        iters=iters, base_effmags=C.BASE_EFFMAGS,
        bracket_meta=dict(vnorm=round(vnorm, 1), resid_norm=round(rn, 1), lo=lo, hi0=round(hi0, 1)),
        n_cal=len(vecs), n_streams=n_streams, cal_tokens=cal_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true", help="skip on-box eff_mag auto-tuning")
    ap.add_argument("--inject", default="gen", choices=["gen", "all", "prompt"],
                    help="injection method: 'gen'=generation-only (prompt clean, default); 'all'=ALL-position (legacy); "
                         "'prompt'=prompt-prefill-only (static persona-like perturbation, prereg E3). "
                         "The controlled A/B: same dose (--no-calibrate), vary only this.")
    ap.add_argument("--variant", default="orig", choices=list(C.PROMPT_VARIANTS),
                    help="word-free system-prompt variant (the elicitation instrument); 'orig' is canonical")
    args = ap.parse_args()
    g = cfg(args.smoke)
    SYSTEM = C.PROMPT_VARIANTS[args.variant]
    inject_mode = args.inject                  # gen / all / prompt (see gen_clean)
    print(f"CONFIG {'SMOKE' if args.smoke else 'FULL'} variant={args.variant} inject={args.inject}: {g}", flush=True)
    if not HAVE_WORDFREQ:
        # Without wordfreq, degeneracy() reports word_rate=0 for every stream, so the word-free
        # acceptance filter silently passes streams that contain real words -- the "clean" population
        # (incl. the s0 control) would no longer be word-free. Fail loudly rather than collect bad data.
        msg = "wordfreq not installed: the word-free acceptance filter is DISABLED (word_rate forced to 0)."
        print(f"WARNING: {msg}", flush=True)
        if not args.smoke:
            raise RuntimeError(msg + " Install wordfreq before a real collect (see requirements-driver.txt).")

    model, tok = K.load_model(C.MODEL)
    print("MODEL_READY", flush=True)
    C.ensure_run_dirs()
    layer = C.resolve_layer(model.config.num_hidden_layers)
    last_layer = model.config.num_hidden_layers - 1   # final residual -> h_final capture (offline propagation/bottleneck)
    print(f"MODEL {C.ACTIVE} ({C.MODEL}) n_layers={model.config.num_hidden_layers} -> inject/read layer {layer}", flush=True)
    dev = next(model.parameters()).device
    concepts = g["concepts"]
    suffix_ids = torch.tensor(tok(SUFFIX, add_special_tokens=False).input_ids, device=dev)
    ctx0 = K.chat_ids(tok, PROMPT, system=SYSTEM)[0]

    # ---- pre-run gate: distinct first-token ids, checked over ALL 12 (even in smoke) so the cheap
    #      smoke catches a collision (e.g. deception/debugging sharing " de") before the full run ----
    all12 = C.COVERT_CONCEPTS
    all_ids = [tok(" " + c, add_special_tokens=False).input_ids[0] for c in all12]
    if len(set(all_ids)) != len(all_ids):
        dupes = [(all12[i], all12[j]) for i in range(len(all12)) for j in range(i + 1, len(all12))
                 if all_ids[i] == all_ids[j]]
        print("COLLECT_FATAL", flush=True)   # watchdog fast-fail (esp. the Qwen3 tokenizer switch)
        raise SystemExit(f"ABORT: concept first-token ids collide {dupes} -- arm C / per-word logP "
                         "would be corrupted. Swap a colliding concept, or switch to full-word logprob.")
    print(f"first-token gate OK over all 12: {dict(zip(all12, all_ids))}", flush=True)
    cfirst = {ci: tok(" " + c, add_special_tokens=False).input_ids[0] for ci, c in enumerate(concepts)}
    first_ids = [cfirst[ci] for ci in range(len(concepts))]   # first-token id per READ concept (logp array order)

    # ---- on-box eff_mag AUTO-TUNE (binary-search to the 3B arm-A band) unless a tuned override exists ----
    #      makes adding a model one autonomous run: gate -> calibrate -> collect (no smokes / manual sweeps).
    # ---- intra-script phase timing (labkit's res.perf gives setup_s/run_s + GPU util_verdict for the WHOLE
    #      compute window; only OUR code knows the calibrate/generate/read split inside run_s). Pairs with
    #      res.util_verdict='sawtooth' (=under-batched) to localize where to optimize. -> results/perf.json.
    PERF = {"calibrate_s": 0.0, "generate_s": 0.0, "read_s": 0.0,
            "read_acts_s": 0.0, "read_cuts_s": 0.0, "n_read_forwards": 0,
            "n_gen_calls": 0, "n_streams_read": 0, "n_read_calls": 0}
    cal_trace = None
    if C.MODELS[C.ACTIVE].get("effmags") is None and not args.no_calibrate:
        print("CALIBRATING eff_mags on-box (no tuned override in registry)...", flush=True)
        _t = time.perf_counter()
        tuned, cal_trace = calibrate_effmags(model, tok, layer, concepts, ctx0, suffix_ids, cfirst, cal_tokens=g["tokens"])
        PERF["calibrate_s"] = round(time.perf_counter() - _t, 1)
        g["strengths"] = [0] + tuned
        print(f"CALIBRATED strengths -> {g['strengths']}", flush=True)
        # persist the trace into the bundle NOW (before the long collect) so it survives even a mid-collect
        # failure (the bundle is pulled best-effort on failure too; run.log -> res.log_path as a backup).
        with open(C.RESULTS / "calibration_trace.json", "w") as f:
            json.dump(cal_trace, f, indent=2)
        print(f"saved calibration trace -> {C.RESULTS / 'calibration_trace.json'}", flush=True)

    streams_record = []   # ALL generated streams (kept + rejected) with metadata
    acts_record = {}      # (arm, global_stream_idx) -> {t: introspection-layer vec}
    reads_record = {}     # (arm, global_stream_idx) -> {t: read dict}
    hfinal_record = {}    # (arm, global_stream_idx) -> {t: final-layer vec}  (offline propagation/bottleneck)
    inject_vectors = {}   # concept -> v_hat (raw difference vector, strength-independent)  [steering primitives]
    inject_alpha = {}     # "concept|s{strength}" -> alpha  (resid += alpha*v reproduces strength*v_hat)
    counts = {}

    gidx = 0
    ncells = len(g["strengths"]) * len(concepts)
    cell = 0
    emit_step(0, model=C.ACTIVE, phase="collect_start", cells=ncells)
    # ---- RESUMABLE cell loop (confound-closing prereg operational requirement): each (concept, strength)
    # cell writes an ATOMIC shard on completion; a restarted run loads the contiguous prefix of existing
    # shards (skipping their GPU work entirely) and regenerates from the first missing cell. Prefix-only:
    # gidx continuity requires cells in order, and a crash can only leave an in-order prefix. A stray
    # later shard (shouldn't happen) is ignored and overwritten.
    shard_dir = C.DATA / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    resume_ok = True
    for strength in g["strengths"]:
        for ci, c in enumerate(concepts):
            spath = shard_dir / f"cell_s{strength}_{c}.pt"
            ckey = f"{c}_s{strength}"
            if resume_ok and spath.exists():
                sh = torch.load(spath, map_location="cpu", weights_only=False)
                streams_record.extend(sh["streams"])
                acts_record.update(sh["acts"]); reads_record.update(sh["reads"]); hfinal_record.update(sh["hfinal"])
                counts[ckey] = sh["counts"]
                if sh.get("vec") is not None:
                    inject_vectors[c] = sh["vec"]
                    inject_alpha[f"{c}|s{strength}"] = sh["alpha"]
                gidx = sh["next_gidx"]
                cell += 1
                print(f"=== {c} s{strength} === RESUMED from shard ({sh['counts']})", flush=True)
                emit_step(cell, model=C.ACTIVE, concept=c, concept_idx=ci, strength=strength,
                          clean=sh["counts"]["clean"], cells=ncells, resumed=True)
                continue
            resume_ok = False                                      # first missing cell: regenerate from here on
            print(f"=== {c} s{strength} ===", flush=True)
            _t = time.perf_counter()
            gen_out, vec, alpha = gen_clean(model, tok, c, strength, layer, g, SYSTEM, inject_mode=inject_mode)
            PERF["generate_s"] += time.perf_counter() - _t; PERF["n_gen_calls"] += 1
            counts[ckey] = dict(generated=len(gen_out),
                                clean=int(sum(o["accepted"] for o in gen_out)))
            # generation-only: prompt_len=len(ctx0) injects the stream + "; secret word:" cue, not the read context
            inj_hook = (None if inject_mode == "prompt" else
                        (K._injection_hook(vec, alpha, prompt_len=(ctx0.shape[0] if inject_mode == "gen" else None))
                         if vec is not None else None))
            cell_vec = vec.float().cpu().numpy() if vec is not None else None
            cell_alpha = float(alpha) if vec is not None else None
            if cell_vec is not None:                               # persist the exact steering primitives
                inject_vectors[c] = cell_vec
                inject_alpha[f"{c}|s{strength}"] = cell_alpha
            cell_streams, cell_acts, cell_reads, cell_hfinal = [], {}, {}, {}
            for o in gen_out:
                rec = dict(gidx=gidx, concept=c, concept_idx=ci, strength=strength,
                           tokens=o["tokens"], text=o["text"], deg=o["deg"], accepted=o["accepted"],
                           gen_topk=o.get("gen_topk"))
                cell_streams.append(rec)
                # only READ the KEPT (non-degenerate) streams -- the population the question is about
                if o["accepted"]:
                    _t = time.perf_counter()
                    aB, rB, hB = read_stream(model, tok, o["tokens"], layer, last_layer, g["grid"], cfirst,
                                             suffix_ids, ctx0, g["topk"], None, perf=PERF)   # arm B (clean)
                    cell_acts[("B", gidx)] = aB; cell_reads[("B", gidx)] = rB; cell_hfinal[("B", gidx)] = hB
                    if inj_hook is not None:                               # arm A (source/injected)
                        aA, rA, hA = read_stream(model, tok, o["tokens"], layer, last_layer, g["grid"], cfirst,
                                                 suffix_ids, ctx0, g["topk"], inj_hook, perf=PERF)
                        cell_acts[("A", gidx)] = aA; cell_reads[("A", gidx)] = rA; cell_hfinal[("A", gidx)] = hA
                    PERF["read_s"] += time.perf_counter() - _t
                    PERF["n_streams_read"] += 1; PERF["n_read_calls"] += (2 if inj_hook is not None else 1)
                gidx += 1
            streams_record.extend(cell_streams)
            acts_record.update(cell_acts); reads_record.update(cell_reads); hfinal_record.update(cell_hfinal)
            tmp = spath.with_suffix(".tmp")
            torch.save(dict(streams=cell_streams, acts=cell_acts, reads=cell_reads, hfinal=cell_hfinal,
                            vec=cell_vec, alpha=cell_alpha, counts=counts[ckey], next_gidx=gidx), tmp)
            os.replace(tmp, spath)                                 # atomic: a crash never leaves a torn shard
            print(f"  saved {counts[ckey]} [shard {spath.name}]", flush=True)
            cell += 1
            emit_step(cell, model=C.ACTIVE, concept=c, concept_idx=ci, strength=strength,
                      clean=counts[ckey]["clean"], cells=ncells)

    # ---- measure residual norm (median ||resid|| at the read layer, s0, mid cut) -> seeds eff_mag
    #      scaling for THIS model; fill it into config.MODELS[slug]['resid_norm'] for future scaled runs.
    s0_gidx = [r["gidx"] for r in streams_record if r["strength"] == 0 and r["accepted"]]
    ncut = next((t for t in g["grid"] if t >= 8), g["grid"][0])
    s0_norms = [float(np.linalg.norm(acts_record[("B", gi)][ncut]))
                for gi in s0_gidx if ("B", gi) in acts_record and ncut in acts_record[("B", gi)]]
    resid_norm = float(np.median(s0_norms)) if s0_norms else None
    print(f"RESID_NORM (s0, layer {layer}, cut {ncut}) = {resid_norm} -> set MODELS['{C.ACTIVE}'] resid_norm", flush=True)

    # ---- intra-script phase profile -> results/perf.json (read against labkit's res.util_verdict/res.perf)
    PERF["generate_s"] = round(PERF["generate_s"], 1)
    PERF["read_s"] = round(PERF["read_s"], 1)
    PERF["read_acts_s"] = round(PERF["read_acts_s"], 1)
    PERF["read_cuts_s"] = round(PERF["read_cuts_s"], 1)
    PERF["read_ms_per_stream"] = round(1000 * PERF["read_s"] / PERF["n_streams_read"], 1) if PERF["n_streams_read"] else None
    PERF["read_frac"] = (round(PERF["read_s"] / (PERF["read_s"] + PERF["generate_s"] + PERF["calibrate_s"]), 3)
                         if (PERF["read_s"] + PERF["generate_s"] + PERF["calibrate_s"]) else None)
    # what a future safe batched-read optimization would target: per-cut forwards are the multiplier.
    PERF["read_cuts_frac"] = (round(PERF["read_cuts_s"] / PERF["read_s"], 3) if PERF["read_s"] else None)
    PERF["read_forwards_per_stream"] = (round(PERF["n_read_forwards"] / PERF["n_streams_read"], 1)
                                        if PERF["n_streams_read"] else None)
    with open(C.RESULTS / "perf.json", "w") as f:
        json.dump(PERF, f, indent=2)
    print(f"PERF {PERF}", flush=True)

    # ---- serialize EVERYTHING raw (tmp + os.replace: a crash never leaves a torn final bundle) ----
    _final = C.DATA / "covert_collect.pt"
    _ftmp = _final.with_suffix(".tmp")
    torch.save({"streams": streams_record, "acts": acts_record, "reads": reads_record,
                "hfinal": hfinal_record, "inject_vectors": inject_vectors, "inject_alpha": inject_alpha,
                "concepts": concepts, "first_ids": first_ids, "grid": g["grid"], "strengths": g["strengths"],
                "layer": layer, "last_layer": last_layer, "resid_norm": resid_norm,
                "calibration": cal_trace, "variant": args.variant, "inject": args.inject, "model": C.ACTIVE, "hf_id": C.MODEL},
               _ftmp)
    os.replace(_ftmp, _final)                              # atomic
    with open(C.RESULTS / "covert_collect_meta.json", "w") as f:
        json.dump(dict(model=C.MODEL, slug=C.ACTIVE, layer=layer, smoke=args.smoke, variant=args.variant,
                       config=g, counts=counts, concepts=concepts, first_ids=first_ids, resid_norm=resid_norm,
                       ref_norm=C.REF_NORM, calibration=cal_trace, perf=PERF, have_wordfreq=HAVE_WORDFREQ,
                       inject=args.inject),
                  f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(C.STREAMS / "covert_collect_streams.txt", "w") as f:
        f.write("\n".join(f"[{r['concept']}][s{r['strength']}] acc={r['accepted']} "
                          f"wr={r['deg']['word_rate']:.2f} rep={r['deg']['repetition']:.2f}\n{r['text']}\n"
                          for r in streams_record))
    print(json.dumps({"counts": counts}, indent=2))
    print(f"saved {len(streams_record)} streams; {len(reads_record)} arm-reads "
          f"(arms {sorted(set(k[0] for k in reads_record))})", flush=True)
    emit_step(ncells + 1, model=C.ACTIVE, phase="collect_done", cells=ncells)
    print("COLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
