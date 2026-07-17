"""exp2 runner v3: the predict-then-demonstrate token-budget analysis on the _ab bundles.

Per model (and bits-ladder task), for the distribution-access (gen_topk, dense) and sampled-token (one-hot
realized tokens, sparse) channels: builds features, fits a per-channel nested-CV BEST-decoder, and averages
bits_recovered(T) over several seeds (subsample x CV) -> mean +/- sd learning curves. Reports the per-token
rate (nulled if the fit window is past saturation), T*, verification, and the FEATURIZATION-MATCHED gap.

NOTE: the sampled channel here is a one-hot token-ID reader -- a valid LOWER bound, and the literal
Prediction-#2 contrast, but NOT the best token monitor. The safety-headline gap uses the embedding reader
(R_emb) -- see the embedding-reader unit. So `featurization_matched_gap_bits` may OVERSTATE token-monitoring
weakness.

Run: .venv/bin/python experiments/exp2_output_monitorability/analysis/run_budget.py [bundle.pt ...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
from features import embed_features, pooled_dist_features, char_features_from_tokens  # noqa: E402
from info import bits_recovered                       # noqa: E402
from loader import load_ab_streams                    # noqa: E402
from predict import predict_budget                    # noqa: E402
from prep import build_vocab_index, common_n_subsample, group_labels  # noqa: E402
from rate import RateFit                              # noqa: E402
from reader import best_reader_proba_by_budget        # noqa: E402

# four readers of the SAME streams: dist (full next-token distribution), emb (realized-token embeddings, same
# model access as dist), sampled (one-hot token id, the weak floor), char (exp1's transcript symbol-counter --
# LESS access than emb; tests the "no transcript monitor sees anything" claim). None is proven optimal; all
# are held-out lower bounds. dist/emb/char = dense best-decoder; sampled = sparse.
KIND = {"dist": "dense", "emb": "dense", "sampled": "sparse", "char": "dense"}
MODES = ("dist", "emb", "sampled", "char")


def _load_tokenizer(model_slug):
    """HF tokenizer (no weights, no forward pass) for decoding realized tokens -> text for the char reader."""
    sys.path.insert(0, os.path.join(REPO, "src"))
    from transformers import AutoTokenizer
    from config import MODELS
    return AutoTokenizer.from_pretrained(MODELS[model_slug]["hf_id"])


def _sampled_steps(tokens):
    return [{"ids": np.array([int(t)]), "logp": np.array([0.0])} for t in tokens]


def _features(streams, budget, vocab, mode, vocab_size=None, embed=None, tokenizer=None):
    if mode == "char":
        return char_features_from_tokens(streams, budget, tokenizer)
    rows = []
    for s in streams:
        if mode == "emb":
            rows.append(embed_features(s["tokens"], budget, embed))
        else:
            steps = s["gen_topk"] if mode == "dist" else _sampled_steps(s["tokens"])
            rows.append(pooled_dist_features(steps, budget, vocab, vocab_size=vocab_size))
    return np.array(rows)


def _fit_reader_task(mode, T, X, y, folds, seed):
    """One (reader, budget) nested-CV fit for the joblib fan-out; inner CV single-threaded (n_jobs=1) so the
    OUTER parallelism owns the cores. Module-level so loky can pickle it. Returns (mode, T, bits, top proba)."""
    proba = best_reader_proba_by_budget({T: X}, y, [T], kind=KIND[mode], folds=folds, seed=seed, n_jobs=1)[T]
    return mode, int(T), float(bits_recovered(y, proba)), proba


def _bits_one_seed(bundle_path, seed, budgets, max_vocab, min_count, folds, floor_vocab_size, n, n_groups,
                   min_len=None):
    b = load_ab_streams(bundle_path)
    maxb = max(budgets)
    # keep streams long enough for the top budget; full-stream mode passes a small min_len + a huge budget so
    # every accepted stream is kept and each reader pools over that stream's OWN full length.
    thr = maxb if min_len is None else min_len
    streams = [s for s in b["streams"] if len(s["gen_topk"]) >= thr]
    y_full = np.array([s["concept_idx"] for s in streams])
    idx = common_n_subsample(y_full, n=n, seed=seed)               # common N across models (cross-scale fair)
    streams = [streams[i] for i in idx]
    y = y_full[idx]
    if n_groups is not None:
        y = group_labels(y, n_groups, seed=seed)
    ids = [int(t) for s in streams for step in s["gen_topk"] for t in step["ids"]]
    ids += [int(t) for s in streams for t in s["tokens"][:maxb]]    # union vocab
    vocab = build_vocab_index(ids, max_vocab=max_vocab, min_count=min_count)
    embed = np.load(f"{os.environ.get('INTRO_EMBED_DIR', 'artifacts')}/{b['model']}_embed.npy")  # R_emb; artifacts/ is labkit-upload-excluded
    tok = _load_tokenizer(b["model"]) if "char" in MODES else None
    vsize = {"dist": floor_vocab_size, "emb": None, "sampled": None, "char": None}
    H = float(-np.sum([(c / len(y)) * np.log2(c / len(y)) for c in np.bincount(y) if c]))
    # Build every (reader, budget) feature matrix once (cheap), then FAN the independent nested-CV fits across
    # cores with the inner CV single-threaded (n_jobs=1). Flips the parallelism: nested n_jobs=-1 only saturated
    # the innermost level (~5 cores), leaving a many-core box mostly idle; this fans ~len(MODES)*len(budgets)
    # tasks out at once. Numerically identical -- each fit is independent + seeded per call; n_jobs never affects
    # results. Prints on each task's completion so the stall watchdog stays fed even on a slow box.
    from joblib import Parallel, delayed
    outer_jobs = int(os.environ.get("INTRO_JOBS", "-1"))
    fmats = {(mode, T): _features(streams, T, vocab, mode, vocab_size=vsize[mode], embed=embed, tokenizer=tok)
             for mode in MODES for T in budgets}
    jobs = [delayed(_fit_reader_task)(mode, T, X, y, folds, seed) for (mode, T), X in fmats.items()]
    try:
        done = Parallel(n_jobs=outer_jobs, return_as="generator_unordered")(jobs)   # joblib>=1.3: stream results
    except (TypeError, ValueError):
        # <1.3 has no return_as (TypeError); 1.3.x has it but rejects "generator_unordered" (ValueError).
        # Fallback returns a list (all fits complete before the loop), so progress prints land after the batch,
        # not during -- fine because joblib is pinned to 1.5.3 (deps), so on the box this fallback never fires.
        done = Parallel(n_jobs=outer_jobs)(jobs)
    bits = {m: {} for m in MODES}
    top_proba = {}                                                  # (n,K) held-out proba at the top budget, per mode
    for mode, T, b_val, proba in done:
        bits[mode][T] = b_val
        if T == maxb:
            top_proba[mode] = proba
        print(f"    seed {seed} [{mode}] T={T}: {b_val:+.3f}b", flush=True)
    return dict(bits=bits, H=H, model=b["model"], inject=b["inject"], n=len(y),
                top_T=int(maxb), y=y.astype(int), top_proba=top_proba,
                k=len(np.unique(y)), vocab=len(vocab))


def bootstrap_ci_from_per(per, n_boot=2000, seed=0):
    """Pool the top-budget held-out (y, proba) ACROSS seeds and concept-bootstrap the top-budget bits + the
    dist-based gaps (dist-R_emb, dist-char, dist-onehot). The generalization unit is the concept, so this is
    the honest CI the 3-seed sd understated. Returns None if per lacks proba (kept for old callers/tests)."""
    from info import concept_bootstrap_ci
    if not per or "top_proba" not in per[0]:
        return None
    y = np.concatenate([np.asarray(p["y"]) for p in per])
    modes = list(per[0]["top_proba"].keys())
    proba_by_mode = {m: np.concatenate([p["top_proba"][m] for p in per]) for m in modes}
    gaps = [("dist_minus_emb", "dist", "emb")]
    if "char" in modes:
        gaps.append(("dist_minus_char", "dist", "char"))
    if "sampled" in modes:
        gaps.append(("dist_minus_sampled", "dist", "sampled"))
    out = concept_bootstrap_ci(y, proba_by_mode, gaps=gaps, n_boot=n_boot, seed=seed)
    out["top_T"] = int(per[0]["top_T"])
    return out


def full_stream_bits(bundle_path, seeds=(0, 1, 2), min_len=8, full_budget=100000, n=24, n_groups=None):
    """Full-stream evaluation: every reader pooled over each stream's OWN full length (no common-T truncation)
    -- the faithful replication of exp1's char reader, which never truncated. Keeps all accepted streams with
    len >= min_len (short ones no longer dropped), subsamples n/class for cross-scale comparability. This is
    where the transcript (char) reader gets its fair full-length shot at the 'distribution-only' claim, and
    where 7B-evoked (median ~108 tokens) is read near its full length. Per-reader mean/sd + concept bootstrap."""
    per = [_bits_one_seed(bundle_path, s, (full_budget,), max_vocab=300, min_count=2, folds=5,
                          floor_vocab_size=151936, n=n, n_groups=n_groups, min_len=min_len) for s in seeds]
    H = float(np.mean([p["H"] for p in per]))
    readers = {m: {"bits_mean": float(np.mean([p["bits"][m][full_budget] for p in per])),
                   "bits_sd": float(np.std([p["bits"][m][full_budget] for p in per]))} for m in MODES}
    return {"model": per[0]["model"], "inject": per[0]["inject"], "H_bits": round(H, 3), "n": per[0]["n"],
            "min_len": min_len, "readers": readers, "bootstrap_ci": bootstrap_ci_from_per(per)}


def ladder_bits(bundle_path, seeds=(0, 1, 2), budgets=(2, 3, 4, 5, 6, 8, 10, 12), groups=(2, 4), n=24):
    """Bits-ladder CROSS-CHECK (not a curve): recovered bits at the top budget for coarser concept groupings
    (K-way = a log2(K)-bit task), averaged over seeds (each seed = a different balanced random grouping). The
    12-way rung is the main run. Validates that the magnitudes are internally consistent -- bits <= H, rise
    with task entropy, recovered FRACTION falls -- the one calibration the single 12-way task can't give."""
    top = max(budgets)
    out = {}
    for K in groups:
        per = [_bits_one_seed(bundle_path, s, budgets, max_vocab=300, min_count=2, folds=5,
                              floor_vocab_size=151936, n=n, n_groups=K) for s in seeds]
        readers = {m: {"bits_mean": float(np.mean([p["bits"][m][top] for p in per])),
                       "bits_sd": float(np.std([p["bits"][m][top] for p in per]))} for m in MODES}
        out[f"{K}way"] = {"task_bits": float(np.log2(K)), "top_T": int(top), "readers": readers}
    return out


def _fit_rate(budgets, bits_mean, window):
    T = np.array([t for t in budgets if window[0] <= t <= window[1]], dtype=float)
    bb = np.array([bits_mean[int(t)] for t in T])
    slope, intercept = np.polyfit(T, bb, 1)
    pred = slope * T + intercept
    ss_res, ss_tot = float(np.sum((bb - pred) ** 2)), float(np.sum((bb - bb.mean()) ** 2))
    return RateFit(float(slope), float(intercept), 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def run_model(bundle_path, seeds=(0, 1, 2), budgets=(2, 3, 4, 5, 6, 8, 10, 12), fit_window=(2, 6),
              required_bits=1.0, max_vocab=300, min_count=2, folds=5, floor_vocab_size=151936, n=24,
              n_groups=None):
    per = [_bits_one_seed(bundle_path, s, budgets, max_vocab, min_count, folds, floor_vocab_size, n, n_groups)
           for s in seeds]
    H = float(np.mean([p["H"] for p in per]))
    out = {"model": per[0]["model"], "inject": per[0]["inject"], "task_classes": per[0]["k"], "n": per[0]["n"],
           "n_per_class": per[0]["n"] // per[0]["k"], "H_bits": round(H, 3), "budgets": list(budgets),
           "seeds": list(seeds), "required_bits": required_bits, "readers": {}}
    for mode in MODES:
        curves = np.array([[p["bits"][mode][int(T)] for T in budgets] for p in per])   # (seeds, budgets)
        bm = {int(T): round(float(curves[:, i].mean()), 3) for i, T in enumerate(budgets)}
        bsd = {int(T): round(float(curves[:, i].std()), 3) for i, T in enumerate(budgets)}
        fit = _fit_rate(budgets, bm, fit_window)
        r = {"bits_mean": bm, "bits_sd": bsd, "r2": round(fit.r2, 3),
             "rate": round(fit.rate, 4) if fit.r2 >= 0.8 else None,        # null if window past saturation
             "intercept": round(fit.intercept, 3) if fit.r2 >= 0.8 else None}
        try:
            tstar = predict_budget(fit, required_bits, min_r2=0.8, ceiling=H)
            r["T_star"] = round(tstar, 1)
            vb = min([T for T in budgets if T >= tstar], default=None)
            r["verified"] = bool(vb and bm[vb] - bsd[vb] >= required_bits) if vb else "T* > budgets"
        except ValueError as e:
            r["T_star"], r["note"] = None, str(e)
        out["readers"][mode] = r
    d, e, s, c = (out["readers"]["dist"], out["readers"]["emb"], out["readers"]["sampled"],
                  out["readers"]["char"])
    out["best_monitor_gap_bits"] = {int(T): round(d["bits_mean"][int(T)] - e["bits_mean"][int(T)], 3)
                                    for T in budgets}                    # dist - R_emb = the honest sampling loss
    out["featurization_matched_gap_bits"] = {int(T): round(d["bits_mean"][int(T)] - s["bits_mean"][int(T)], 3)
                                             for T in budgets}           # dist - one-hot (weak floor)
    out["dist_minus_char_gap_bits"] = {int(T): round(d["bits_mean"][int(T)] - c["bits_mean"][int(T)], 3)
                                       for T in budgets}                 # dist - transcript char reader
    out["gap_note"] = ("best_monitor_gap = dist - R_emb (embedding token, same model access); "
                       "dist_minus_char = dist - transcript char reader (tests 'distribution-only'). "
                       "No reader is proven optimal -- all are held-out lower bounds.")
    out["bootstrap_ci"] = bootstrap_ci_from_per(per)      # concept-level CI on top-budget bits + gaps
    return out


def _print(res):
    print(f"  {res['model']} {res['inject']} {res['task_classes']}-way n={res['n']}({res['n_per_class']}/cls) "
          f"H={res['H_bits']}b seeds={res['seeds']}")
    for mode, r in res["readers"].items():
        m = {t: f"{r['bits_mean'][t]}±{r['bits_sd'][t]}" for t in r["bits_mean"]}
        print(f"  [{mode:7s}] {m} rate={r['rate']} r2={r['r2']} T*={r.get('T_star')} "
              f"verified={r.get('verified', r.get('note'))}")
    print(f"  best-monitor gap (dist-R_emb): {res['best_monitor_gap_bits']}")
    print(f"  featurization-matched gap (dist-onehot): {res['featurization_matched_gap_bits']}")
    print(f"  transcript gap (dist-char): {res['dist_minus_char_gap_bits']}")


def plot(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)
    for ax, res in zip(axes[0], results):
        T = res["budgets"]
        for mode, style in (("dist", "o-"), ("emb", "^-."), ("sampled", "s--"), ("char", "D:")):
            if mode not in res["readers"]:
                continue
            r = res["readers"][mode]
            ax.errorbar(T, [r["bits_mean"][t] for t in T], yerr=[r["bits_sd"][t] for t in T],
                        fmt=style, capsize=3, label=mode)
        ax.axhline(res["H_bits"], ls=":", c="grey", label="H(C)")
        ax.axhline(0, ls="-", c="k", lw=0.5)
        ax.set(title=f"{res['model']} {res['task_classes']}-way", xlabel="tokens T", ylabel="bits recovered")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    paths = sys.argv[1:] or [f"runs/_ab/qwen2.5-{m}-gen.pt" for m in ("1.5b", "3b", "7b")]
    base = os.environ.get("INTRO_REPORT_DIR", "experiments/exp2_output_monitorability/reports")
    os.makedirs(base, exist_ok=True)
    outpath = f"{base}/budget_results.json"
    results, done = [], set()
    if os.path.exists(outpath):                              # RESUME: keep completed models, recompute none
        results = json.load(open(outpath))
        for r in results:                                    # JSON stringifies int budget keys -> restore for plot/indexing
            for rd in r.get("readers", {}).values():
                for k in ("bits_mean", "bits_sd"):
                    if k in rd:
                        rd[k] = {int(t): v for t, v in rd[k].items()}
            for k in ("best_monitor_gap_bits", "featurization_matched_gap_bits", "dist_minus_char_gap_bits"):
                if k in r:
                    r[k] = {int(t): v for t, v in r[k].items()}
        done = {r["model"] for r in results}
        print(f"resuming: {len(done)} models already done, skipping {sorted(done)}", flush=True)
    for p in paths:
        slug = os.path.basename(p).split("-gen")[0]          # qwen2.5-1.5b-gen.pt -> qwen2.5-1.5b
        if slug in done:
            continue
        print(f"\n===== {p} =====", flush=True)
        res = run_model(p)                                   # T<=12 budget curve + char + bootstrap CI
        res["full_stream"] = full_stream_bits(p)             # each stream at its OWN full length (exp1 replication)
        res["ladder"] = ladder_bits(p, budgets=(12,))        # {2way,4way} cross-check at the TOP budget only
        results.append(res)
        _print(res)
        fs = res["full_stream"]["readers"]
        print(f"  full-stream ({res['full_stream']['n']}/cls, min_len {res['full_stream']['min_len']}): "
              + " ".join(f"{m}={fs[m]['bits_mean']:.3f}" for m in ("dist", "char", "emb", "sampled")), flush=True)
        for k, v in res["ladder"].items():
            rr = v["readers"]
            print(f"  ladder {k} ({v['task_bits']:.0f}b): "
                  + " ".join(f"{m}={rr[m]['bits_mean']:.3f}" for m in ("dist", "char", "emb")), flush=True)
        json.dump(results, open(outpath + ".tmp", "w"), indent=2)   # atomic per-bundle checkpoint (crash-safe)
        os.replace(outpath + ".tmp", outpath)
    plot(results, f"{base}/budget_curves.png")


if __name__ == "__main__":
    main()
