"""exp3 analysis: run exp2's frozen readers on the induced bundles, per arm, and apply the PREREG verdicts.

Reuses exp2's reviewed machinery verbatim -- run_budget._bits_one_seed (per-channel nested-CV best decoders on
dist / R_emb / one-hot) is called per seed on each (model, arm) bundle; nothing in exp2's analysis is edited.
This module adds only what exp3 pre-registered: per-seed dist-R_emb gap stats (the gap_present check needs the
PER-SEED gap, not the difference of means' sds), the frozen-threshold verdicts, and the state x label summary
across arms (evoked = clean hidden-state gap; named - evoked = label effect at fixed state; secret_word = the
covert-token channel, echo vs spread via one-hot vs R_emb).

Run offline ($0), from anywhere:  .venv/bin/python experiments/exp3_induction_and_scale/run_induction.py
Writes reports/induction_results.json + induction_curves.png. The evoked-arm GAUGE is scored by a separate
unit (gauge_judge.py); its verdict gates interpretation of evoked scale points, not this computation.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis"))

MODELS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")
ARMS = ("evoked", "named", "secret_word", "evoked_alt")   # analyzed if the bundle exists
SEEDS = (0, 1, 2)
BUDGETS = (2, 3, 4, 5, 6, 8, 10, 12)


# ---------------------------------------------------------------- pure (unit-tested)
def _intkey_budgets(results):
    """JSON stringifies int budget keys; restore them so plot() / top-budget indexing (which use int budgets)
    work on RESUMED bundles too -- mixing int-keyed fresh + str-keyed loaded bundles KeyError'd plot()."""
    for r in results:
        for rd in r.get("readers", {}).values():
            for k in ("bits_mean", "bits_sd"):
                if k in rd:
                    rd[k] = {int(t): v for t, v in rd[k].items()}
    return results


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for k successes in n trials (the PREREG gauge CI method)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, float(center - half)), min(1.0, float(center + half)))


def arm_summary(bits_by_seed, budgets, H):
    """bits_by_seed: list over seeds of {mode: {T: bits}}. Returns per-reader mean/sd curves plus the
    PER-SEED dist-R_emb gap stats at the top budget (what PREREG's gap_present is defined on)."""
    top = max(budgets)
    readers = {}
    modes = list(bits_by_seed[0].keys())
    for mode in modes:
        curves = np.array([[s[mode][T] for T in budgets] for s in bits_by_seed])   # (seeds, budgets)
        readers[mode] = {
            "bits_mean": {int(T): float(curves[:, i].mean()) for i, T in enumerate(budgets)},
            "bits_sd": {int(T): float(curves[:, i].std()) for i, T in enumerate(budgets)},
        }
    def _gap(other):
        g = np.array([s["dist"][top] - s[other][top] for s in bits_by_seed])       # per-seed gap
        return {"mean": float(g.mean()), "sd": float(g.std()), "per_seed": [float(x) for x in g]}
    out = {"readers": readers, "H_bits": float(H), "top_T": int(top), "gap_dist_emb": _gap("emb")}
    if "char" in readers:                                                          # dist - transcript char reader
        out["gap_dist_char"] = _gap("char")
    return out


def prereg_verdicts(summary, prereg):
    """Apply the frozen thresholds (primers.PREREG) to one arm's summary, at the top budget.
    recover[mode]: mean - sd >= recover_margin_bits (above the 0-bits shuffle floor).
    gap_present: gap mean >= gap_present_bits AND per-seed gap (mean - sd) > 0."""
    top = summary["top_T"]
    margin = prereg["recover_margin_bits"]
    recover = {m: bool(r["bits_mean"][top] - r["bits_sd"][top] >= margin)
               for m, r in summary["readers"].items()}
    g = summary["gap_dist_emb"]
    gap_present = bool(g["mean"] >= prereg["gap_present_bits"] and (g["mean"] - g["sd"]) > 0)
    return {"recover": recover, "gap_present": gap_present}


# ---------------------------------------------------------------- orchestration (reuses reviewed exp2 code)
def run_arm(bundle_path, seeds=SEEDS, budgets=BUDGETS, do_ladder=False):
    from run_budget import (_bits_one_seed, bootstrap_ci_from_per,   # exp2's reviewed per-seed reader run + CI
                            full_stream_bits, ladder_bits)
    import primers as P
    per = []
    for s in seeds:
        per.append(_bits_one_seed(bundle_path, s, budgets, max_vocab=300, min_count=2, folds=5,
                                  floor_vocab_size=151936, n=24, n_groups=None))
        print(f"  seed {s} done", flush=True)                 # progress inside the quiet nested-CV crunch (stall guard)
    summary = arm_summary([p["bits"] for p in per], budgets, H=float(np.mean([p["H"] for p in per])))
    summary.update(model=per[0]["model"], arm=per[0]["inject"], n=per[0]["n"], k=per[0]["k"],
                   seeds=list(seeds), budgets=list(budgets))
    summary["bootstrap_ci"] = bootstrap_ci_from_per(per)      # concept-level CI on top-budget bits + gaps
    summary["full_stream"] = full_stream_bits(bundle_path, seeds=seeds)   # each stream own length (char cross-check)
    if do_ladder:                                             # magnitude cross-check only on the headline arm
        summary["ladder"] = ladder_bits(bundle_path, seeds=seeds, budgets=(max(budgets),))  # top budget only
    summary["verdicts"] = prereg_verdicts(summary, P.PREREG)
    return summary


def plot(results, path):
    if not results:
        print("nothing to plot (no bundles found)")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in results})
    arms = [a for a in ARMS if any(r["arm"] == a for r in results)]
    fig, axes = plt.subplots(len(models), len(arms), figsize=(4.2 * len(arms), 3.4 * len(models)),
                             squeeze=False)
    for i, m in enumerate(models):
        for j, a in enumerate(arms):
            ax = axes[i][j]
            r = next((x for x in results if x["model"] == m and x["arm"] == a), None)
            if r is None:
                ax.set_axis_off()
                continue
            T = r["budgets"]
            for mode, style in (("dist", "o-"), ("emb", "^-."), ("sampled", "s--"), ("char", "D:")):
                if mode not in r["readers"]:
                    continue
                rd = r["readers"][mode]
                ax.errorbar(T, [rd["bits_mean"][t] for t in T], yerr=[rd["bits_sd"][t] for t in T],
                            fmt=style, capsize=3, label=mode)
            ax.axhline(r["H_bits"], ls=":", c="grey")
            ax.axhline(0, ls="-", c="k", lw=0.5)
            ax.set(title=f"{m} / {a}", xlabel="tokens T", ylabel="bits")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    os.environ.setdefault("INTRO_EMBED_DIR", os.path.join(REPO, "artifacts"))     # R_emb matrices
    base = os.environ.get("INTRO_REPORT_DIR", os.path.join(HERE, "reports"))      # box sets out/ for the pull
    ind_dir = os.environ.get("INTRO_IND_DIR", os.path.join(REPO, "runs", "_ind"))
    os.makedirs(base, exist_ok=True)
    outpath = os.path.join(base, "induction_results.json")
    results, done = [], set()
    if os.path.exists(outpath):                                    # RESUME: keep completed bundles, recompute none
        results = _intkey_budgets(json.load(open(outpath)))        # JSON stringifies int budget keys -> restore
        done = {(r["model"], r["arm"]) for r in results}
        print(f"resuming: {len(done)} bundles already done, skipping {sorted(done)}", flush=True)
    for m in MODELS:
        for a in ARMS:
            if (m, a) in done:
                continue
            p = os.path.join(ind_dir, m, "data", f"{m}-{a}.pt")
            if not os.path.exists(p):
                continue
            print(f"\n===== {m} / {a} =====", flush=True)
            r = run_arm(p, do_ladder=(a == "evoked"))          # ladder cross-check only on the headline arm
            results.append(r)
            json.dump(results, open(outpath + ".tmp", "w"), indent=2)   # checkpoint per bundle, atomically:
            os.replace(outpath + ".tmp", outpath)                       # a crash keeps all completed work intact
            top = r["top_T"]
            rr = r["readers"]
            char = f" char={rr['char']['bits_mean'][top]:.3f}" if "char" in rr else ""
            gc = f" gap(dist-char)={r['gap_dist_char']['mean']:.3f}" if "gap_dist_char" in r else ""
            print(f"  T={top}: dist={rr['dist']['bits_mean'][top]:.3f} emb={rr['emb']['bits_mean'][top]:.3f} "
                  f"onehot={rr['sampled']['bits_mean'][top]:.3f}{char}  gap(dist-emb)={r['gap_dist_emb']['mean']:.3f}"
                  f"±{r['gap_dist_emb']['sd']:.3f}{gc}  verdicts={r['verdicts']}", flush=True)
            if "full_stream" in r:
                fs = r["full_stream"]["readers"]
                print(f"  full-stream ({r['full_stream']['n']}/cls): "
                      + " ".join(f"{mm}={fs[mm]['bits_mean']:.3f}" for mm in ("dist", "char", "emb", "sampled")),
                      flush=True)
            if "ladder" in r:
                for k, v in r["ladder"].items():
                    lr = v["readers"]
                    print(f"  ladder {k} ({v['task_bits']:.0f}b): "
                          + " ".join(f"{mm}={lr[mm]['bits_mean']:.3f}" for mm in ("dist", "char", "emb")), flush=True)
    plot(results, os.path.join(base, "induction_curves.png"))
    # state x label contrasts at top budget, per model (evoked = clean; named-evoked = label given state)
    print("\n===== state x label (top budget, dist bits | gap dist-emb) =====")
    for m in MODELS:
        row = {r["arm"]: r for r in results if r["model"] == m}
        if not row:
            continue
        cells = []
        for a in ("evoked", "named", "secret_word"):
            if a in row:
                top = row[a]["top_T"]
                cells.append(f"{a}: {row[a]['readers']['dist']['bits_mean'][top]:.2f}b "
                             f"(gap {row[a]['gap_dist_emb']['mean']:.2f})")
        print(f"  {m}: " + " | ".join(cells))


if __name__ == "__main__":
    main()
