"""Full-stream & convergence re-analysis (publish-prep) -- the reproducible source for the
BLOGPOST "two-regime" section and the exp2/exp3 report full-stream updates.

It recomputes, in the exp2 bits currency (`bits_recovered = H(C) - CE`, per-channel best-decoder
nested-CV; seeds 0/1/2 -> mean +/- sd), the three things the short-budget (T<=12) run could not show:

  A) convergence_injected_1p5b -- injected 1.5B, a FIXED cohort of streams that reach >=64 tokens,
     truncated to T=12/24/48/64 and read at full length. Same streams at every budget, so the
     reader curves are comparable without a population shift. This is where `char` overtakes.
  B) injected_scales -- dist / R_emb / char at T=12 vs full stream, 1.5B/3B/7B, on the realistic
     heterogeneous set (all streams at their own length, min_len floor). char@full lives here.
  C) natural_scales -- the same for the evoked (natural-induction) bundles, PLUS a length-controlled
     >=64-token cohort: does `char` accumulate given long *natural* tokens (it does on injected)?
     This is the control that separates "distribution-only" from "streams too short to accumulate".

Provenance note: like the rest of the analysis this reads the stream bundles + extracted token
embeddings, which are NOT in git (they come from the private HF dataset
ErrareHumanumEst/internal-state-from-gibberish -- see README). CPU-only. Run cores-capped, e.g.:

  LOKY_MAX_CPU_COUNT=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  INTRO_EMBED_DIR=artifacts python3 full_stream_convergence.py

Writes reports/full_stream_convergence.json. Deterministic given the same bundles + embeds.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_budget as RB                                   # noqa: E402
from loader import load_ab_streams                        # noqa: E402
from prep import common_n_subsample, build_vocab_index    # noqa: E402
from reader import best_reader_proba_by_budget            # noqa: E402
from info import bits_recovered                           # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODES = ("dist", "emb", "char")
VOCAB_SIZE = {"dist": 151936, "emb": None, "sampled": None, "char": None}
SEEDS = (0, 1, 2)
FULL = 100000                                             # "read the whole stream" budget


def _bits(streams, embed, tok, mode, budget, n, own=False):
    """bits_recovered pooled over SEEDS for one (mode, budget). own=True -> full stream.
    Returns {mean, sd, per_seed}. n = per-class subsample (common-N, so cross-cell comparable)."""
    B = FULL if own else budget
    per = []
    for seed in SEEDS:
        y = np.array([s["concept_idx"] for s in streams])
        idx = common_n_subsample(y, n=n, seed=seed)
        ss = [streams[i] for i in idx]
        y = y[idx]
        ids = ([int(t) for s in ss for st in s["gen_topk"] for t in st["ids"]] +
               [int(t) for s in ss for t in s["tokens"][:B]])
        vocab = build_vocab_index(ids, max_vocab=300, min_count=2)
        X = RB._features(ss, B, vocab, mode, vocab_size=VOCAB_SIZE[mode], embed=embed, tokenizer=tok)
        P = best_reader_proba_by_budget({B: X}, y, [B], kind=RB.KIND[mode], folds=5, seed=seed, n_jobs=1)[B]
        per.append(float(bits_recovered(y, P)))
    return {"mean": float(np.mean(per)), "sd": float(np.std(per)), "per_seed": per}


def _load(rel_bundle, scale, art):
    b = load_ab_streams(os.path.join(REPO, rel_bundle))
    return b, RB._load_tokenizer(b["model"]), np.load(os.path.join(art, f"qwen2.5-{scale}_embed.npy"))


def _min_per_class(streams):
    return int(np.bincount([s["concept_idx"] for s in streams]).min())


def main():
    art = os.environ.get("INTRO_EMBED_DIR", os.path.join(REPO, "artifacts"))
    out = {
        "currency": "bits_recovered = H(C) - CE; per-channel best-decoder nested-CV; seeds 0/1/2 -> mean+/-sd",
        "note": "Local cores-capped run. Bundles + embeds from HF (ErrareHumanumEst/internal-state-from-gibberish).",
        "analyses": {},
    }

    # ---- A) convergence: injected 1.5B, fixed >=64 cohort, truncated to each budget ----
    b, tok, emb = _load("runs/_ab/qwen2.5-1.5b-gen.pt", "1.5b", art)
    coh = [s for s in b["streams"] if len(s["gen_topk"]) >= 64]
    n = min(13, _min_per_class(coh))                      # >=64 survival binds n; 13 keeps a class margin
    conv = {"cohort": "streams >=64 tokens, fixed set truncated to each budget",
            "n_per_class": n, "n_streams": len(coh), "budgets": [12, 24, 48, 64, "full"], "readers": {}}
    for mode in MODES:
        conv["readers"][mode] = {str(T): _bits(coh, emb, tok, mode, T, n) for T in (12, 24, 48, 64)}
        conv["readers"][mode]["full"] = _bits(coh, emb, tok, mode, 64, n, own=True)
        print(f"[A] injected-1.5B convergence {mode:4}: "
              + " ".join(f"{conv['readers'][mode][k]['mean']:.3f}" for k in ("12", "24", "48", "64", "full")),
              flush=True)
    out["analyses"]["convergence_injected_1p5b"] = conv

    # ---- B) injected T=12 vs full, realistic heterogeneous set, all scales ----
    inj = {}
    for scale in ("1.5b", "3b", "7b"):
        b, tok, emb = _load(f"runs/_ab/qwen2.5-{scale}-gen.pt", scale, art)
        s12 = [s for s in b["streams"] if len(s["gen_topk"]) >= 12]
        sfull = [s for s in b["streams"] if len(s["gen_topk"]) >= 8]
        n12, nf = min(24, _min_per_class(s12)), min(24, _min_per_class(sfull))
        inj[scale] = {"n_T12": n12, "n_full": nf, "min_len_T12": 12, "min_len_full": 8, "readers": {}}
        for mode in MODES:
            inj[scale]["readers"][mode] = {"T12": _bits(s12, emb, tok, mode, 12, n12),
                                           "full": _bits(sfull, emb, tok, mode, FULL, nf, own=True)}
            r = inj[scale]["readers"][mode]
            print(f"[B] injected-{scale} {mode:4}: T12={r['T12']['mean']:.3f} full={r['full']['mean']:.3f}", flush=True)
    out["analyses"]["injected_scales"] = inj

    # ---- C) natural (evoked): T=12 vs full realistic, PLUS length-controlled >=64 cohort ----
    nat = {}
    for scale in ("1.5b", "3b", "7b"):
        b, tok, emb = _load(f"runs/_ind/qwen2.5-{scale}/data/qwen2.5-{scale}-evoked.pt", scale, art)
        s12 = [s for s in b["streams"] if len(s["gen_topk"]) >= 12]
        sfull = [s for s in b["streams"] if len(s["gen_topk"]) >= 8]
        coh = [s for s in b["streams"] if len(s["gen_topk"]) >= 64]
        n12, nf, nc = min(24, _min_per_class(s12)), min(24, _min_per_class(sfull)), min(24, _min_per_class(coh))
        nat[scale] = {"n_T12": n12, "n_full": nf, "realistic": {}}
        # the >=64-token length-control needs >=5/class (CV folds); natural streams rarely reach 64 at 3B/7B
        long_ok = nc >= 5
        nat[scale]["long_cohort_ge64"] = {"n_per_class": nc, "n_streams": len(coh),
                                          "feasible": long_ok, "readers": {}}
        if not long_ok:
            nat[scale]["long_cohort_ge64"]["note"] = (
                f"too few natural streams reach >=64 tokens (min {nc}/class < 5 CV folds) -- "
                "length-control only feasible at 1.5B")
        for mode in MODES:
            nat[scale]["realistic"][mode] = {"T12": _bits(s12, emb, tok, mode, 12, n12),
                                             "full": _bits(sfull, emb, tok, mode, FULL, nf, own=True)}
            rr = nat[scale]["realistic"][mode]
            line = f"[C] natural-{scale} {mode:4}: realistic T12={rr['T12']['mean']:.3f} full={rr['full']['mean']:.3f}"
            if long_ok:
                rc = {"T12": _bits(coh, emb, tok, mode, 12, nc), "T64": _bits(coh, emb, tok, mode, 64, nc),
                      "full": _bits(coh, emb, tok, mode, 64, nc, own=True)}
                nat[scale]["long_cohort_ge64"]["readers"][mode] = rc
                line += f" | >=64 full={rc['full']['mean']:.3f} (n={nc}/cls)"
            else:
                line += f" | >=64 SKIPPED (n={nc}/cls < 5)"
            print(line, flush=True)
    out["analyses"]["natural_scales"] = nat

    dst = os.path.join(REPO, "experiments", "exp2_output_monitorability", "reports", "full_stream_convergence.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
