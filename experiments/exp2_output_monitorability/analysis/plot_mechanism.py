"""Mechanism figure: why injected concepts mark the transcript and evoked ones don't.

Two panels, style-matched to plot_convergence.py:

  LEFT  -- "Transcript marking needs dose x persistence": dist@T12 and
           char@full for per-token (generation-only) injection across
           strengths {0, 3, 5, 8, 12, 20} (reports/dose_titration_confound-e1.json)
           joined with strengths {40, 60} from the original 1.5B titration
           (reports/dose_titration.json; same protocol, different capture).
           Prompt-only injection char@full at {40, 60}
           (reports/dose_titration_confound-e3.json) plotted as crosses:
           it floors even at strong dose. Horizontal reference: the natural
           evoked dist level (0.447 bits) and the pre-registered
           natural-matched window dist@T12 in [0.30, 0.60].
  RIGHT -- "The state, measured per position": per-cut concept-projection
           z-scores (sigma vs the s0/neutral pool) from
           reports/e4_trajectory_verdict.json (injected 1.5B, evoked 1.5B,
           evoked 7B, all in-task) and reports/gauge_trajectory_verdict.json
           (evoked 1.5B under the free-association gauge).

Output: reports/mechanism_figure.png.

Usage:
    .venv/bin/python experiments/exp2_output_monitorability/analysis/plot_mechanism.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path(__file__).resolve().parent.parent / "reports"
OUT_PATH = REPORTS / "mechanism_figure.png"

NATURAL_DIST = 0.447  # natural evoked dist@T12 (confound_closing_prereg.md)
WINDOW = (0.30, 0.60)  # pre-registered natural-matched window for dist@T12

# Okabe-Ito colorblind-safe palette (shared with plot_convergence.py)
C_DIST = "#0072B2"    # blue
C_CHAR = "#009E73"    # green
C_PROMPT = "#D55E00"  # vermillion
C_GAUGE = "#E69F00"   # orange
C_7B = "#CC79A7"      # pink


def cell_stats(data, strength, key):
    cell = data["cells"][str(strength)][key]
    return cell["mean"], cell["sd"]


def series(data, strengths, key):
    means, sds = [], []
    for s in strengths:
        m, sd = cell_stats(data, s, key)
        means.append(m)
        sds.append(sd)
    return means, sds


def main():
    e1 = json.loads((REPORTS / "dose_titration_confound-e1.json").read_text())
    tit = json.loads((REPORTS / "dose_titration.json").read_text())
    e3 = json.loads((REPORTS / "dose_titration_confound-e3.json").read_text())
    e4 = json.loads((REPORTS / "e4_trajectory_verdict.json").read_text())
    gauge = json.loads((REPORTS / "gauge_trajectory_verdict.json").read_text())

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.28}
    )

    # ---------------- LEFT: dose x persistence ----------------
    e1_strengths = [0, 3, 5, 8, 12, 20]
    tit_strengths = [40, 60]
    xs = e1_strengths + tit_strengths

    for key, color, marker, label in (
        ("dist_T12", C_DIST, "o", "dist @12  (per-token injection)"),
        ("char_full", C_CHAR, "^", "char @full  (per-token injection)"),
    ):
        m1, s1 = series(e1, e1_strengths, key)
        m2, s2 = series(tit, tit_strengths, key)
        means, sds = m1 + m2, s1 + s2
        ax_l.errorbar(
            xs, means, yerr=sds,
            color=color, marker=marker, label=label,
            lw=2, ms=6, capsize=3, zorder=3,
        )
        print(f"LEFT {key} (gen): " + ", ".join(
            f"s{x}={m:.4f}+-{sd:.4f}" for x, m, sd in zip(xs, means, sds)))

    # prompt-only injection: floors even at strong dose
    pm, ps = series(e3, tit_strengths, "char_full")
    ax_l.errorbar(
        tit_strengths, pm, yerr=ps,
        color=C_PROMPT, marker="x", ls="none", ms=9, mew=2.2, capsize=3,
        label="char @full  (prompt-only injection)", zorder=4,
    )
    print("LEFT char_full (prompt-only): " + ", ".join(
        f"s{x}={m:.4f}+-{sd:.4f}" for x, m, sd in zip(tit_strengths, pm, ps)))

    # natural evoked reference + natural-matched window
    ax_l.axhspan(*WINDOW, color="0.55", alpha=0.12, zorder=1)
    ax_l.axhline(NATURAL_DIST, color="0.3", ls="--", lw=1.2, zorder=2)
    ax_l.annotate(
        "natural evoked level (dist = 0.447)\nnatural-matched window [0.30, 0.60]",
        xy=(0.0, 0.66), ha="left", va="bottom",
        fontsize=8.5, color="0.35", style="italic",
    )
    ax_l.annotate(
        "prompt-only floors\neven at strong dose",
        xy=(56, 0.10), fontsize=8, color=C_PROMPT,
        style="italic", ha="center", va="bottom",
    )

    ax_l.set_xscale("symlog", linthresh=3)
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels([str(x) for x in xs], fontsize=8.5)
    ax_l.set_xlim(-0.4, 75)
    ax_l.set_xlabel("injection strength (effective magnitude)")
    ax_l.set_ylabel("bits of concept identity recovered")
    ax_l.set_ylim(-0.15, 3.0)
    ax_l.set_title(
        "Transcript marking needs dose $\\times$ persistence",
        fontsize=11.5,
    )
    ax_l.legend(loc="upper left", bbox_to_anchor=(0.02, 0.99), frameon=False,
                fontsize=9)

    # ---------------- RIGHT: state trajectory ----------------
    def zcurve(block):
        cuts = sorted(int(c) for c in block["z_by_cut"])
        return cuts, [block["z_by_cut"][str(c)] for c in cuts]

    curves = [
        ("qwen2.5-1.5b_injected", e4["qwen2.5-1.5b_injected"], C_DIST, "o",
         "injected, in-task (1.5B)"),
        ("gauge-evoked", gauge["qwen2.5-1.5b_gauge-evoked"], C_GAUGE, "s",
         "evoked, free-association gauge (1.5B)"),
        ("qwen2.5-1.5b_evoked", e4["qwen2.5-1.5b_evoked"], C_CHAR, "^",
         "evoked, in-task (1.5B)"),
        ("qwen2.5-7b_evoked", e4["qwen2.5-7b_evoked"], C_7B, "D",
         "evoked, in-task (7B)"),
    ]
    for name, block, color, marker, label in curves:
        cuts, zs = zcurve(block)
        sems = None
        if "sem_by_cut" in block:
            sems = [block["sem_by_cut"][str(c)] for c in cuts]
        ax_r.errorbar(
            cuts, zs, yerr=sems,
            color=color, marker=marker, label=label,
            lw=2, ms=5.5, capsize=3, zorder=3,
        )
        print(f"RIGHT {name}: " + ", ".join(
            f"t{c}={z:.4f}" for c, z in zip(cuts, zs)))

    ax_r.set_xscale("log", base=2)
    cut_ticks = [2, 4, 8, 16, 32, 64, 127]
    ax_r.set_xticks(cut_ticks)
    ax_r.set_xticklabels([str(c) for c in cut_ticks])
    ax_r.minorticks_off()
    ax_r.set_xlabel("cut position $t$ (tokens into the stream)")

    # values span -0.03 .. 12.4: symlog keeps the near-zero evoked curves honest
    ax_r.set_yscale("symlog", linthresh=0.1)
    ax_r.set_ylim(-0.06, 30)
    ax_r.set_yticks([0, 0.1, 0.3, 1, 3, 10])
    ax_r.set_yticklabels(["0", "0.1", "0.3", "1", "3", "10"])
    ax_r.set_ylabel("concept projection $z$  ($\\sigma$ vs neutral pool)")

    ax_r.annotate(
        "injection = exogenous persistence:\nre-written at every position ($\\approx$12$\\sigma$, flat)",
        xy=(16, 14.5), ha="center", va="bottom",
        fontsize=8.5, color=C_DIST, style="italic",
    )
    ax_r.annotate(
        "persona: never installed\n($\\lesssim$0.4$\\sigma$ everywhere)",
        xy=(3.4, 0.75), ha="center", va="bottom",
        fontsize=8.5, color="0.35", style="italic",
    )
    ax_r.set_title("The state, measured per position", fontsize=11.5)
    ax_r.legend(loc="center right", bbox_to_anchor=(1.02, 0.68), frameon=False,
                fontsize=8, handletextpad=0.5)

    # ---------------- shared cosmetics ----------------
    for ax in (ax_l, ax_r):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.9", lw=0.8, zorder=0)

    fig.suptitle(
        "Why injected concepts mark the transcript and evoked ones don't"
        "  (Qwen2.5-1.5B unless noted, mean $\\pm$ sd over 3 seeds)",
        fontsize=12, y=1.04,
    )
    fig.text(
        0.5, -0.06,
        "Left: per-token series joins two runs of the same protocol -- the weak-dose sweep "
        "(strengths 0-20, confound-e1) and the original titration (40/60). "
        "Right: gauge sem shown where reported; gauge cuts stop at 64 (no full-length pool).",
        ha="center", fontsize=8, color="0.35", style="italic",
    )

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
