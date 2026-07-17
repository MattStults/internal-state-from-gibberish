"""Blog figures for the privileged-access / self-read arc.

Reads the committed results JSONs (never hardcodes what a JSON carries):
  - reports/lr_grid_results.json          (1.5B/3B/7B grid: diagonals, off-diagonals, Falcon)
  - reports/lr_extend_results.json        (14B extension, run-2 dose curve, 70B rider cells)
  - reports/dose_titration.json           (1.5B public curves s0/s40/s60)
  - reports/dose_titration_confound-e1.json (1.5B public curves s3-s20)
  - reports/lr_72b_fullsweep_results.json (Llama-70B metadata; sanity only)

Hardcoded (cross-checked annotations only, each with a source comment):
  - Llama-3.3-70B faithful-template calibrated bits (lr_72b_fullsweep_verdict.md,
    2026-07-14 addendum) -- the rescore JSON on disk carries raw records, not the
    calibrated summary.
  - grid templating top-3 prefix shares at 1.5/3/7B and the ~0.90 Llama share
    (lr_scale_grid_verdicts.md review addendum R3 -- computed ad hoc from the stream
    bundles, not present in any results JSON).

Output: reports/figs/*.png (>=150 dpi, colorblind-safe Okabe-Ito palette).
Run: .venv/bin/python experiments/exp2_output_monitorability/analysis/make_blog_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.abspath(os.path.join(HERE, "..", "reports"))
FIGS = os.path.join(REPORTS, "figs")
os.makedirs(FIGS, exist_ok=True)

def _load(name):
    with open(os.path.join(REPORTS, name)) as f:
        return json.load(f)

GRID = _load("lr_grid_results.json")
EXT = _load("lr_extend_results.json")
DOSE_MAIN = _load("dose_titration.json")          # 1.5B s0/40/60
DOSE_E1 = _load("dose_titration_confound-e1.json")  # 1.5B s3-20
FULL70 = _load("lr_72b_fullsweep_results.json")     # sanity only

# Okabe-Ito colorblind-safe palette
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERMIL = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GRAY = "#7F7F7F"

QSIZES = ["1.5b", "3b", "7b", "14b"]
QPARAMS = {"1.5b": 1.5e9, "3b": 3e9, "7b": 7e9, "14b": 14e9}
QLABEL = {"1.5b": "1.5B", "3b": "3B", "7b": "7B", "14b": "14B"}

def grid_diag_cell(size, arm):
    """Diagonal cell record from the grid JSON (1.5/3/7B)."""
    r = f"qwen2.5-{size}"
    key = f"{r}/{'secret_wordxSW' if arm == 'secret_word' else 'secret_sustainxSS'}"
    return GRID["readers"][r][key]

def ext_diag_cell(arm):
    """14B diagonal cell record from the extension JSON."""
    return EXT["cells"][f"qwen2.5-14b|qwen2.5-14b|{arm}"]

def diag_series(arm):
    """(primary, prefix16) lists over 1.5/3/7/14B, all read from JSONs."""
    prim, p16 = [], []
    for s in QSIZES[:3]:
        c = grid_diag_cell(s, arm)
        prim.append(c["bits_mean"])
        p16.append(c["bits_secondary_B"])
    c = ext_diag_cell(arm)
    prim.append(c["primary"]["bits_mean"])
    p16.append(c["secondary_B"]["bits_mean"])
    return prim, p16

# Cross-checked annotations (NOT in any results JSON -- see module docstring):
# Llama-3.3-70B faithful-template calibrated bits, lr_72b_fullsweep_verdict.md addendum:
LLAMA70_FAITHFUL = {"secret_word": -0.011, "secret_sustain": 0.244, "evoked": 0.143}
# templating top-3 4-char-prefix shares, lr_scale_grid_verdicts.md addendum R3 (secret_word):
TEMPL_GRID = {"1.5b": 0.016, "3b": 0.048, "7b": 0.141}
TEMPL_LLAMA70 = 0.90  # ~90% single-template share, lr_72b_fullsweep_verdict.md confound (b)
assert "llama" in FULL70["model"].lower()  # sanity: 70B artifact present & is the Llama scout


# ---------------------------------------------------------------- fig 1: scale curve
def fig_scale_curve():
    prim, p16 = diag_series("secret_word")
    x = [QPARAMS[s] for s in QSIZES]
    templ = [TEMPL_GRID["1.5b"], TEMPL_GRID["3b"], TEMPL_GRID["7b"],
             EXT["templating_14b"]["share"]]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(x, prim, "-o", color=C_BLUE, lw=2.2, ms=7, label="secret_word diagonal (primary)")
    ax.plot(x, p16, "--s", color=C_GREEN, lw=2.2, ms=7,
            label="secret_word diagonal (length-matched, prefix-16)")
    # 70B faithful-template null -- open marker, cross-family / different regime
    ax.plot([70e9], [LLAMA70_FAITHFUL["secret_word"]], "o", mfc="none", mec=C_VERMIL,
            ms=10, mew=2.2)
    ax.annotate("Llama-3.3-70B (cross-family,\nfaithful template): null\n(~90% template collapse)",
                xy=(70e9, LLAMA70_FAITHFUL["secret_word"]), xytext=(1.7e10, 0.10),
                fontsize=9, color=C_VERMIL,
                arrowprops=dict(arrowstyle="->", color=C_VERMIL, lw=1.2))
    ax.axhline(0, color=C_GRAY, lw=0.8, ls=":")

    ax2 = ax.twinx()
    xt = x + [70e9]
    ax2.plot(xt, templ + [TEMPL_LLAMA70], "-^", color=C_ORANGE, lw=1.6, ms=6, alpha=0.85,
             label="templating: top-3 prefix share")
    ax2.set_ylabel("top-3 4-char-prefix share (template collapse)", color=C_ORANGE)
    ax2.tick_params(axis="y", labelcolor=C_ORANGE)
    ax2.set_ylim(0, 1.0)
    ax2.axhspan(0.40, 0.60, color=C_ORANGE, alpha=0.08)
    ax2.text(1.6e9, 0.415, "regime gray zone (0.40–0.60)", fontsize=8, color=C_ORANGE)

    ax.set_xscale("log")
    ax.set_xticks(xt)
    ax.set_xticklabels([QLABEL[s] for s in QSIZES] + ["70B*"])
    ax.set_xlabel("generator parameters")
    ax.set_ylabel("calibrated bits (self-read LR, 12-way; ceiling 3.585)")
    ax.set_ylim(-0.05, 0.47)
    ax.set_title("The self-read secret channel rises with scale, then bends at 14B\n"
                 "as word-free generation collapses toward templates")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_scale_curve.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2: dose self-read
def fig_dose_selfread():
    doses = [3, 5, 8, 12, 20, 40, 60]
    self_read = [EXT["run2_inject_TF"][f"qwen2.5-1.5b:s{d}"]["calibrated"]["bits_mean"]
                 for d in doses]
    def pub(d, key):
        src = DOSE_E1 if d <= 20 else DOSE_MAIN
        return src["cells"][str(d)][key]["mean"]
    dist = [pub(d, "dist_T12") for d in doses]
    char = [pub(d, "char_full") for d in doses]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.axhline(np.log2(12), color=C_GRAY, lw=1.0, ls="--")
    ax.text(3.1, np.log2(12) + 0.05, "ceiling log$_2$12 = 3.585", fontsize=8, color=C_GRAY)
    ax.axvspan(3, 20, color=C_SKY, alpha=0.12)
    ax.text(4.4, 2.9, "private-ish at natural strength:\nself-read $>$ public readers",
            fontsize=9, color=C_BLUE)
    ax.plot(doses, self_read, "-o", color=C_BLUE, lw=2.4, ms=7,
            label="self-read under re-injection (own quantity, pure-concept channel)")
    ax.plot(doses, dist, "-s", color=C_VERMIL, lw=1.8, ms=6,
            label="public: trained dist reader @ 12 tokens")
    ax.plot(doses, char, "-^", color=C_ORANGE, lw=1.8, ms=6,
            label="public: trained char reader @ full stream")
    ax.axhline(0, color=C_GRAY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(doses)
    ax.set_xticklabels([f"s{d}" for d in doses])
    ax.set_xlabel("injection strength (Qwen2.5-1.5B)")
    ax.set_ylabel("calibrated bits (12-way)")
    ax.set_ylim(-0.15, 3.85)
    ax.set_title("The model reads its own injected mark when it feels it again:\n"
                 "re-injection self-read exceeds every public reader at every dose")
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_dose_selfread.png"), dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- fig 3: access matrix
def fig_access_matrix():
    readers = QSIZES                      # rows
    gens = QSIZES + ["llama70b"]          # cols
    bits = np.full((4, 5), np.nan)
    voided = np.zeros((4, 5), bool)

    # grid 3x3 block (secret_word_cells carries bits + voided)
    for rec in GRID["secret"]["secret_word_cells"]:
        if not rec["reader"].startswith("qwen"):
            continue  # Falcon rows summarized as an annotation below
        i = readers.index(rec["reader"].replace("qwen2.5-", ""))
        j = gens.index(rec["gen"].replace("qwen2.5-", ""))
        bits[i, j] = rec["bits"]
        voided[i, j] = rec["voided"]

    # extension cells: 14B row + 14B column (+ diagonal)
    for key, c in EXT["cells"].items():
        reader, gen, arm = key.split("|")
        if arm != "secret_word":
            continue
        i = readers.index(reader.replace("qwen2.5-", ""))
        j = gens.index(gen.replace("qwen2.5-", ""))
        bits[i, j] = c["primary"]["bits_mean"]
        voided[i, j] = c["voided"]

    # 70B rider column (secret_word cells, descriptive)
    for key, c in EXT["rider_70b"]["cells"].items():
        reader, gen, arm = key.split("|")
        if arm != "secret_word":
            continue
        i = readers.index(reader.replace("qwen2.5-", ""))
        bits[i, 4] = c["primary"]["bits_mean"]
        voided[i, 4] = c["voided"]

    falcon = [rec["bits"] for rec in GRID["secret"]["secret_word_cells"]
              if rec["reader"].startswith("falcon")]

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    vmax = np.nanmax(bits)
    im = ax.imshow(bits, cmap="Blues", vmin=-0.02, vmax=vmax, aspect="auto")
    for i in range(4):
        for j in range(5):
            if np.isnan(bits[i, j]):
                continue
            if voided[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor=C_GRAY, lw=0))
            strong = bits[i, j] > 0.5 * vmax
            ax.text(j, i, f"{bits[i, j]:+.3f}" + ("\n(void)" if voided[i, j] else ""),
                    ha="center", va="center", fontsize=10,
                    color="white" if strong else "black",
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Qwen 1.5B", "Qwen 3B", "Qwen 7B", "Qwen 14B",
                        "Llama-70B (cross-family)"], fontsize=9,
                       rotation=30, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Qwen 1.5B", "Qwen 3B", "Qwen 7B", "Qwen 14B"], fontsize=9)
    ax.set_xlabel("generator (whose word-free streams are read)")
    ax.set_ylabel("reader (whose likelihoods do the reading)")
    ax.set_title("Who can read the secret mark? secret_word calibrated bits, reader × generator\n"
                 "(only the generator reads its own mark; hatch = instrument-voided cell)")
    fig.colorbar(im, ax=ax, label="calibrated bits")
    ax.text(0.0, -0.38,
            f"Falcon3 1B/3B/7B readers of all Qwen pools: ≈ 0 "
            f"(range {min(falcon):+.4f}…{max(falcon):+.4f} bits; instrument-unresolvable, grid caveat 4).\n"
            f"Llama-70B column: descriptive rider cells (its own secret_word diagonal is null).",
            transform=ax.transAxes, fontsize=7.5, color=C_GRAY)
    fig.savefig(os.path.join(FIGS, "fig_access_matrix.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4: sustain amplification
def fig_sustain_amplification():
    w_prim, w_p16 = diag_series("secret_word")
    s_prim, s_p16 = diag_series("secret_sustain")
    x = [QPARAMS[s] for s in QSIZES]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(x, s_prim, "-o", color=C_VERMIL, lw=2.2, ms=7,
            label='secret_sustain ("imbue every keystroke"), primary')
    ax.plot(x, s_p16, "--o", color=C_VERMIL, lw=1.6, ms=6, alpha=0.65,
            label="secret_sustain, length-matched (prefix-16)")
    ax.plot(x, w_prim, "-s", color=C_BLUE, lw=2.2, ms=7, label="secret_word, primary")
    ax.plot(x, w_p16, "--s", color=C_BLUE, lw=1.6, ms=6, alpha=0.65,
            label="secret_word, length-matched (prefix-16)")
    ax.axhline(0, color=C_GRAY, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([QLABEL[s] for s in QSIZES])
    ax.set_xlabel("generator parameters (reader = generator)")
    ax.set_ylabel("calibrated bits (12-way; ceiling 3.585)")
    ax.set_title('Instructed per-token use amplifies the private mark —\n'
                 'and both channels bend at 14B')
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_sustain_amplification.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_scale_curve()
    fig_dose_selfread()
    fig_access_matrix()
    fig_sustain_amplification()
    print("wrote figures to", FIGS)
