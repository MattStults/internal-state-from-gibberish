"""Cumulative self-read bits vs token-prefix length, secret_word diagonal per scale.
Companion to make_blog_figs.py (kept standalone: loads 4 shard pairs, ~minutes CPU).
  Run: .venv/bin/python experiments/exp2_output_monitorability/analysis/make_prefix_fig.py"""
import importlib.util, os, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
spec = importlib.util.spec_from_file_location("lro", os.path.join(HERE, "lr_reader_offline.py"))
LRO = importlib.util.module_from_spec(spec); sys.modules["lro"] = LRO; spec.loader.exec_module(LRO)
SHARDS = {"1.5B": "runs/lr_grid_box/lr_grid/qwen2.5-1.5b__qwen2.5-1.5b__secret_word_SW.pt",
          "3B": "runs/lr_grid_box/lr_grid/qwen2.5-3b__qwen2.5-3b__secret_word_SW.pt",
          "7B": "runs/lr_grid_box/lr_grid/qwen2.5-7b__qwen2.5-7b__secret_word_SW.pt",
          "14B": "runs/lr_extend_box/lr_grid/qwen2.5-14b__qwen2.5-14b__secret_word_SW.pt"}
KS = [4, 8, 16, 32, 64, 128, None]
def bits_at(sh, concepts, K):
    S, y = [], []
    for r in sh["records"]:
        if r.get("strength") not in (None, 1) or r["concept"] not in concepts: continue
        Tn = int(r.get("T_noeos", r["T"])); hi = Tn if K is None else min(K, Tn)
        if hi < 1: continue
        S.append([float(np.asarray(r["ll_tok"][c][:hi], dtype=np.float64).sum()) for c in concepts])
        y.append(concepts.index(r["concept"]))
    return LRO.evaluate_cell(np.asarray(S), np.asarray(y))["bits_mean"]
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
colors = {"1.5B": "#88CCEE", "3B": "#DDCC77", "7B": "#CC6677", "14B": "#332288"}
for label, rel in SHARDS.items():
    sh = torch.load(os.path.join(REPO, rel), map_location="cpu", weights_only=False)
    cs = list(sh["contexts"]); med_T = int(np.median([r.get("T_noeos", r["T"]) for r in sh["records"] if r["concept"] in cs]))
    ks = [k for k in KS if k and k < med_T] + [None]   # drop K past the pool median; keep full
    xs = [k if k else med_T for k in ks]; ys = [bits_at(sh, cs, k) for k in ks]
    ax.plot(xs, ys, "o-", color=colors[label], label=f"{label} (median stream {med_T} tok)")
    ax.plot(xs[-1], ys[-1], marker="*", ms=13, color=colors[label], zorder=5)  # full stream
ax.set_xscale("log", base=2); ax.set_xlabel("token prefix length read (log scale; rightmost point = full stream)")
ax.set_ylabel("calibrated bits (12-way, held-out τ)"); ax.axhline(0, color="gray", lw=0.5)
ax.set_title("Self-read secret_word bits vs how much of the stream the reader sees")
ax.legend(fontsize=8); fig.tight_layout()
out = os.path.join(HERE, "..", "reports", "figs", "fig_prefix_curve.png")
fig.savefig(out, bbox_inches="tight"); print("wrote", os.path.abspath(out))
