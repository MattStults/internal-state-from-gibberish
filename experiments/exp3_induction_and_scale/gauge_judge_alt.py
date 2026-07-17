"""Blind-judge scoring of the 3B/7B EVOKED_ALT gauge sidecars (scale-grid unit B12; prereg
lr_scale_grid_prereg.md Amendment 1, should-fix 2).

Protocol = exp3's gauge_judge VERBATIM: judge_model_gauge is imported (the same function object,
not a copy), so the pinned judge snapshot, the deterministic per-item label shuffle, the
one-word-parse rule (unparseable = WRONG) and the Wilson-CI pass criterion are all byte-identical
to the registered exp3 gauge. Inputs are the sidecars gauge_alt_collect.py wrote on-box, pulled to
runs/lr_grid_box/_ind/<slug>/data/<slug>-evoked_alt-gauge.pt.

A gauge FAIL at a size FLAGS that size's alt-direction LR cells (analysis/lr_grid_offline.py reads
reports/gauge_alt_results.json and voids/flags accordingly, per the amendment's gate-style rule) --
this script never fails a run and exits 0 regardless of verdicts. Registered caveat carried in the
output: induction strength is non-monotone in size (evoked gauge 31/17/43 percent), so the 3B point
is never interpreted alone; endpoint comparisons govern.

Needs ANTHROPIC_API_KEY. Cost: 2 models x 13 rows x gauge_n short replies -- pennies.
Run:  .venv/bin/python experiments/exp3_induction_and_scale/gauge_judge_alt.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from gauge_judge import JUDGE_MODEL, judge_model_gauge   # noqa: E402  pinned protocol, verbatim

MODELS = ("qwen2.5-3b", "qwen2.5-7b")                    # the newly generated alt pools
OUT_JSON = os.path.join(HERE, "reports", "gauge_alt_results.json")
CAVEAT = ("Amendment 1 should-fix 2: induction strength is non-monotone in size (evoked gauge "
          "31/17/43%), so the 3B point is never interpreted as a readability change on its own; "
          "endpoint (1.5B<->7B) comparisons govern. A gauge FAIL here flags/voids that size's "
          "alt-direction cells in lr_grid_offline; it never invalidates the run.")


def sidecar_path(slug):
    """The pulled on-box sidecar (box_lr_grid S1 -> labkit pull -> runs/lr_grid_box/_ind)."""
    return os.path.join(REPO, "runs", "lr_grid_box", "_ind", slug, "data",
                        f"{slug}-evoked_alt-gauge.pt")


def main():
    import anthropic
    import torch
    client = anthropic.Anthropic()                       # key from env; no credential is created
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = dict(judge=JUDGE_MODEL, arm="evoked_alt", caveat=CAVEAT, models=[])
    for m in MODELS:
        p = sidecar_path(m)
        if not os.path.exists(p):
            print(f"[{m}] no alt-gauge sidecar at {p} -- skipped (flag stays 'pending')")
            continue
        side = torch.load(p, map_location="cpu", weights_only=False)
        r = judge_model_gauge(side["gauge"], client, m)
        r["arm"] = "evoked_alt"
        out["models"].append(r)
        json.dump(out, open(OUT_JSON, "w"), indent=2)     # checkpoint per model
        print(f"[{m}] alt-gauge judge={r['judge_served_by']} acc={r['acc']:.2f} ({r['n']} items) "
              f"ci95=[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] gauge_pass={r['gauge_pass']}",
              flush=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
