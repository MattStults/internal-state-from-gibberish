"""Headline figure: bits-of-concept-identity recovered vs. token budget.

Reads the committed reports/full_stream_convergence.json and renders a
two-panel figure:

  LEFT  -- injected concept, 1.5B, fixed >=64-token cohort
           (analyses.convergence_injected_1p5b): dist / emb / char readers
           across budgets {12, 24, 48, 64, full}.
  RIGHT -- naturally induced concept, 1.5B, realistic cohort
           (analyses.natural_scales['1.5b'].realistic): the same readers
           recover essentially nothing at any budget.

Output: reports/convergence_figure.png (same reports/ dir as the JSON).

Usage:
    .venv/bin/python experiments/exp2_output_monitorability/analysis/plot_convergence.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path(__file__).resolve().parent.parent / "reports"
JSON_PATH = REPORTS / "full_stream_convergence.json"
OUT_PATH = REPORTS / "convergence_figure.png"

H_CEILING = 3.585  # H(C) = log2(12): perfect recovery of concept identity

# 'full' streams cap at ~128 tokens in this cohort; plot it at x=128 with a
# labeled tick so the x-axis stays roughly to scale.
FULL_X = 128

# Okabe-Ito colorblind-safe palette
STYLE = {
    "dist": dict(color="#0072B2", marker="o", label="dist  (next-token distribution)"),
    "emb": dict(color="#E69F00", marker="s", label="$R_{emb}$  (reader-embedding)"),
    "char": dict(color="#009E73", marker="^", label="char  (raw transcript text)"),
}


def budget_x(b):
    return FULL_X if b == "full" else int(b)


def main():
    data = json.loads(JSON_PATH.read_text())

    inj = data["analyses"]["convergence_injected_1p5b"]
    nat = data["analyses"]["natural_scales"]["1.5b"]["realistic"]

    budgets = inj["budgets"]  # [12, 24, 48, 64, 'full']
    xs = [budget_x(b) for b in budgets]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"wspace": 0.06}
    )

    # ---------------- LEFT: injected ----------------
    for reader in ("dist", "emb", "char"):
        means = [inj["readers"][reader][str(b)]["mean"] for b in budgets]
        sds = [inj["readers"][reader][str(b)]["sd"] for b in budgets]
        st = STYLE[reader]
        ax_l.errorbar(
            xs, means, yerr=sds,
            color=st["color"], marker=st["marker"], label=st["label"],
            lw=2, ms=6, capsize=3, zorder=3,
        )

    # char/dist crossover happens between the 64-token budget and 'full'
    ax_l.axvspan(64, FULL_X, color="0.55", alpha=0.12, zorder=1)
    ax_l.annotate(
        "char overtakes dist",
        xy=(96, 2.62), ha="center", va="bottom",
        fontsize=9, color="0.35", style="italic",
    )

    ax_l.set_title(
        "Injected concept: distribution reads fast,\nthe transcript accumulates",
        fontsize=11.5,
    )

    # ---------------- RIGHT: natural ----------------
    nat_xs = [12, FULL_X]
    for reader in ("dist", "emb", "char"):
        means = [nat[reader]["T12"]["mean"], nat[reader]["full"]["mean"]]
        sds = [nat[reader]["T12"]["sd"], nat[reader]["full"]["sd"]]
        st = STYLE[reader]
        ax_r.errorbar(
            nat_xs, means, yerr=sds,
            color=st["color"], marker=st["marker"],
            lw=2, ms=6, capsize=3, zorder=3,
        )

    ax_r.set_title(
        "Naturally induced: almost nothing,\nat any budget",
        fontsize=11.5,
    )
    ax_r.annotate(
        "transcript readers $\\approx$ 0 at any budget;\nthe distribution's 0.45 bits decays to $\\approx$ 0 by the full stream",
        xy=(70, 0.72), ha="center", fontsize=9.5, color="0.35", style="italic",
    )

    # ---------------- shared cosmetics ----------------
    for ax in (ax_l, ax_r):
        ax.axhline(H_CEILING, color="0.3", ls="--", lw=1.2, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(["12", "24", "48", "64", "full"])
        ax.set_xlim(4, FULL_X + 10)
        ax.set_xlabel("token budget")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)

    ax_l.text(
        6, H_CEILING + 0.07, "H(C) = 3.585 bits  (perfect recovery)",
        fontsize=9, color="0.3",
    )
    ax_l.set_ylim(-0.35, 4.0)
    ax_l.set_ylabel("bits of concept identity recovered")
    ax_l.legend(loc="center left", bbox_to_anchor=(0.02, 0.72), frameon=False, fontsize=9.5)

    fig.suptitle(
        "How many bits about the model's internal concept leak into its output?"
        "  (Qwen2.5-1.5B, mean $\\pm$ sd over 3 seeds)",
        fontsize=12, y=1.04,
    )

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
