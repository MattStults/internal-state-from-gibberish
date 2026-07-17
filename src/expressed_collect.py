"""Amendment-2 (2b) expressed-injection generation cell (prereg
experiments/exp2_output_monitorability/reports/lr_scale_extend_prereg.md, Amendment 2,
2026-07-14): qwen2.5-1.5b, doses {s20, s60}. The concept vector is injected DURING generation
(the exp1 covert_collect apparatus, generation-only convention) AND the system prompt carries
the sustain-s1 suffix in its ORIGINAL "this feeling" wording -- primers_v2.SUSTAIN_SUFFIXES
['s1'] VERBATIM, never the primers_v3 secret_word-substituted derivative -- composed with the
word-free STRONG_SYSTEM per the frozen primers convention (primer first, anti-word block LAST).
The model is thereby instructed to communicate a state it was never told in words.

Reuse, not reimplementation:
  - generation = covert_collect.gen_clean (the injection-generation apparatus: word-free
    acceptance filter, reject-resample, gen_topk capture) through its inject_override seam with
    the STORED steering primitives -- s20 from the e1 low-dose capture (runs/confound-e1),
    s60 from the main 1.5B capture. $0-fail gates before any generation: the alpha scaling
    convention is VERIFIED per (concept, dose) (alpha == strength/||v||, covert_collect's own
    write) and the two source captures' vectors are asserted identical (deterministic
    extraction, same model/layer) -- a mismatch refuses, never mixes;
  - the sustain suffix is primers_v2's own object (byte-identity pinned by
    tests/test_expressed_2b.py X1).

Output: ONE capture-schema bundle <out>/expressed/qwen2.5-1.5b-expressed.pt (streamset
'expressed'; system_text stored so the run-(2) self-read scores under the token-identical
context; ~24 accepted streams/concept/dose). RESUMABLE per-(dose, concept) atomic cell shards
(covert_collect's prefix-resume pattern) -- an existing cell is never regenerated. Scored
OFFLINE only (analysis/lr_extend_offline.py): public char + dist via dose_titration._bits (the
certified function objects) and the run-(2)-style self-read via inject_tf_lr. Driven by
box_lr_extend.py S4b; GPU only, NEVER on the Mac. Marker-safe prints.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

import config as C
import common as K
import covert_collect as CC

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
_EXP3 = os.path.join(REPO, "experiments", "exp3_induction_and_scale")
if _EXP3 not in sys.path:
    sys.path.insert(0, _EXP3)
import primers_v2 as PV2  # noqa: E402

SLUG = "qwen2.5-1.5b"
DOSES = (20, 60)                    # s20 <- the e1 capture's primitives, s60 <- the main capture's
SUFFIX_KEY = "s1"                   # the E2 pilot winner, ORIGINAL "this feeling" wording
ARM = "expressed"
ALPHA_RTOL = 0.02                   # alpha == strength/||v|| within 2% (fp/bf16 norm slack)
GEN_FN = CC.gen_clean               # the reused apparatus -- the SAME function object (X4)


def default_sources():
    """dose -> source capture path (stored-vector reuse: s20 from e1, s60 from main)."""
    return {20: os.path.join(REPO, "runs", "confound-e1", "data", "covert_collect.pt"),
            60: os.path.join(REPO, "runs", SLUG, "data", "covert_collect.pt")}


def compose_expressed_system(strong_system=None):
    """SUSTAIN_SUFFIXES['s1'] VERBATIM (the original 'this feeling' wording, the exact object
    primers_v2 froze) + '\\n\\n' + the word-free anti-word block -- primer first, anti-word
    block LAST (primers.compose_system's frozen convention). No concept word anywhere,
    asserted at compose time."""
    strong = C.STRONG_SYSTEM if strong_system is None else strong_system
    sysm = PV2.SUSTAIN_SUFFIXES[SUFFIX_KEY] + "\n\n" + strong
    low = sysm.lower()
    bad = [c for c in C.COVERT_CONCEPTS if c.lower() in low]
    if bad:
        raise RuntimeError(f"expressed system contains concept word(s) {bad} -- the cell is "
                           "word-free by construction, refusing")
    return sysm


def stored_primitives(bundle, concept, level, rtol=ALPHA_RTOL):
    """The source capture's OWN stored (vector, alpha) for (concept, level), with the scaling
    convention VERIFIED: covert_collect wrote alpha = strength / ||v|| (resid += alpha*v
    reproduces strength * v_hat). A missing key raises (never re-derived); a drifted alpha
    refuses (the convention check the Amendment asked for)."""
    iv = bundle.get("inject_vectors") or {}
    ia = bundle.get("inject_alpha") or {}
    if concept not in iv:
        raise KeyError(f"source capture has no inject_vector for {concept!r}")
    key = f"{concept}|s{int(level)}"
    if key not in ia:
        raise KeyError(f"source capture has no inject_alpha {key!r} "
                       f"(have {sorted(ia)[:4]}...)")
    v = np.asarray(iv[concept], dtype=np.float32)
    alpha = float(ia[key])
    want = float(level) / float(np.linalg.norm(v))
    if not abs(alpha - want) <= rtol * max(want, 1e-9):
        raise RuntimeError(f"alpha convention violated for {key}: stored {alpha:.6f} vs "
                           f"strength/||v|| = {want:.6f} (rtol {rtol}) -- refusing to inject "
                           "an unverified dose")
    return v, alpha


def assert_vectors_match(b_a, b_b, concepts, atol=1e-4):
    """The two source captures must carry the SAME concept vectors (deterministic extraction,
    same model/layer -- verified true for e1 vs main). A mismatch refuses: mixed steering
    primitives would make the two doses different manipulations, not a dose contrast."""
    for c in concepts:
        va = np.asarray(b_a["inject_vectors"][c], dtype=np.float32)
        vb = np.asarray(b_b["inject_vectors"][c], dtype=np.float32)
        if va.shape != vb.shape or not np.allclose(va, vb, atol=atol):
            raise RuntimeError(f"inject_vectors differ across source captures for {c!r} -- "
                               "refusing to mix steering primitives")
    return True


def cfg(smoke):
    """Full = the registered Amendment-2 cell (~24 accepted/concept/dose, main-capture token
    length, gen_topk capture for the dist reader). Smoke = a tiny s20 slice (2 concepts x 2
    accepted) proving the path end-to-end."""
    if smoke:
        return dict(concepts=C.COVERT_CONCEPTS[:2], doses=(20,), target_clean=2, max_gen=8,
                    tokens=48, gen_batch=8, gen_topk=32)
    return dict(concepts=C.COVERT_CONCEPTS, doses=DOSES, target_clean=24, max_gen=96,
                tokens=128, gen_batch=16, gen_topk=64)


def bundle_path(outdir):
    return os.path.join(str(outdir), "expressed", f"{SLUG}-{ARM}.pt")


def _atomic_save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def collect(model, tok, sources, outdir, g, system=None, gen_fn=None):
    """The cell loop: per-(dose, concept) resume-safe atomic shards -> ONE expressed bundle.
    sources: {dose: loaded source-capture dict}. gen_fn is the generation seam (default
    GEN_FN = covert_collect.gen_clean, the same apparatus). Returns the bundle path."""
    gen_fn = gen_fn or GEN_FN
    system = system or compose_expressed_system()
    doses = [int(d) for d in g["doses"]]
    concepts = list(g["concepts"])
    if len(doses) > 1:
        assert_vectors_match(sources[doses[0]], sources[doses[-1]], concepts)
    layer = int(sources[doses[0]]["layer"])
    shard_dir = os.path.join(str(outdir), "expressed", "shards")
    os.makedirs(shard_dir, exist_ok=True)

    streams, inject_vectors, inject_alpha, counts = [], {}, {}, {}
    gidx = 0
    resume_ok = True                 # covert_collect's prefix-resume: cells in order only
    t0 = time.time()
    for lvl in doses:
        for ci, c in enumerate(concepts):
            v, alpha = stored_primitives(sources[lvl], c, lvl)
            inject_vectors.setdefault(c, v)
            inject_alpha[f"{c}|s{lvl}"] = alpha
            spath = os.path.join(shard_dir, f"cell_s{lvl}_{c}.pt")
            ckey = f"{c}_s{lvl}"
            if resume_ok and os.path.exists(spath):
                sh = torch.load(spath, map_location="cpu", weights_only=False)
                streams.extend(sh["streams"])
                counts[ckey] = sh["counts"]
                gidx = sh["next_gidx"]
                print(f"XPR cell {c} s{lvl} RESUMED from shard ({sh['counts']})", flush=True)
                continue
            resume_ok = False        # first missing cell: generate from here on
            print(f"XPR cell {c} s{lvl} generating...", flush=True)
            gen_out, _, _ = gen_fn(model, tok, c, lvl, layer, g, system,
                                   inject_mode="gen",
                                   inject_override=(torch.as_tensor(v), float(alpha)))
            cell_streams = []
            for o in gen_out:
                cell_streams.append(dict(gidx=gidx, concept=c, concept_idx=ci, strength=lvl,
                                         tokens=o["tokens"], text=o["text"], deg=o["deg"],
                                         accepted=o["accepted"],
                                         gen_topk=o.get("gen_topk")))
                gidx += 1
            counts[ckey] = dict(generated=len(gen_out),
                                clean=int(sum(o["accepted"] for o in gen_out)))
            _atomic_save(spath, dict(streams=cell_streams, counts=counts[ckey],
                                     next_gidx=gidx))
            streams.extend(cell_streams)
            print(f"XPR cell {c} s{lvl} saved {counts[ckey]} "
                  f"t={int(time.time() - t0)}s", flush=True)

    bp = bundle_path(outdir)
    _atomic_save(bp, dict(
        streams=streams, concepts=concepts, strengths=doses,
        inject_vectors=inject_vectors, inject_alpha=inject_alpha,
        layer=layer, inject="gen", variant="orig", model=SLUG,
        hf_id=C.MODELS[SLUG]["hf_id"], streamset=ARM, arm=f"{ARM}_{SUFFIX_KEY}",
        suffix_key=SUFFIX_KEY, system_text=system, counts=counts,
        vector_source={str(lvl): os.path.basename(str(sources[lvl].get("_path", "loaded")))
                       for lvl in doses},
        note="Amendment 2 (2b) expressed-injection cell: vector injected during generation "
             "(gen convention, stored primitives) + sustain-s1 in its ORIGINAL 'this feeling' "
             "wording over the word-free STRONG_SYSTEM. Scored offline: dose_titration._bits "
             "(public char+dist) + inject_tf_lr self-read. Pure-concept-channel caveats per "
             "NOTE_injection_LR_comparability apply."))
    n_acc = sum(1 for s in streams if s["accepted"])
    print(f"XPR bundle saved {os.path.relpath(bp, str(outdir))}: {len(streams)} streams "
          f"({n_acc} accepted) t={int(time.time() - t0)}s", flush=True)
    return bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output root (default INTRO_RUN_DIR / runs/<slug>)")
    ap.add_argument("--e1-capture", default=default_sources()[20],
                    help="the s20 primitives source (runs/confound-e1)")
    ap.add_argument("--main-capture", default=default_sources()[60],
                    help="the s60 primitives source (the main 1.5B capture)")
    args = ap.parse_args()
    g = cfg(args.smoke)
    outdir = args.out or str(C.RUN_DIR)
    paths = {20: args.e1_capture, 60: args.main_capture}
    sources = {}
    for lvl in g["doses"]:
        b = torch.load(paths[int(lvl)], map_location="cpu", weights_only=False)
        assert b.get("model") in (None, SLUG), \
            f"source capture model {b.get('model')!r} != {SLUG!r}"
        assert b.get("inject") == "gen", \
            f"source capture inject={b.get('inject')!r}: the 2b cell reuses the gen convention"
        b["_path"] = paths[int(lvl)]
        sources[int(lvl)] = b
    print(f"XPR config {'SMOKE' if args.smoke else 'FULL'}: doses={list(g['doses'])} "
          f"target={g['target_clean']}/concept/dose out={outdir}", flush=True)
    model, tok = K.load_model(C.MODELS[SLUG]["hf_id"])
    collect(model, tok, sources, outdir, g)


if __name__ == "__main__":
    main()
