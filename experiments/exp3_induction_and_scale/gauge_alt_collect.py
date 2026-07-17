"""Persona-only gauge collection for the EVOKED_ALT arm (scale-grid unit B12; prereg
lr_scale_grid_prereg.md Amendment 1, should-fix 2).

The exp3 collector saves a blind-judge gauge only for the EVOKED arm (collect_induction attaches
bundle["gauge"] when arm == "evoked"), so the newly generated evoked_alt pools at 3B/7B would have
no manipulation check. This script collects the SAME gauge -- free-association responses under the
persona alone (compose_gauge_system) + the frozen GAUGE_PROBE, identical sampling to
collect_induction._collect_gauge (max_new_tokens=64, do_sample=True, temperature=1.0, top_p=0.98,
gauge_n = the real-run cfg default) -- for the ALT (wording-B) personas, and saves it as a SIDECAR
next to the alt bundle: $INTRO_RUN_DIR/data/<slug>-evoked_alt-gauge.pt. Judging stays OFFLINE and
API-side (gauge_judge_alt.py, same pinned judge as exp3); a gauge FAIL flags the affected size's
alt-direction cells in the offline scorer -- it never fails this collection.

Run on GPU via the orchestrator (box_lr_grid S1); NEVER on the Mac.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # primers_v2 / collect_induction
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))  # config / common

import primers_v3 as P  # v3 = drop-in superset of v2 (byte-identical earlier arms, test-pinned); v2 lacks GAUGE_PROBE -- burned full-run attempt 1 at t=18s                                     # noqa: E402
from collect_induction import cfg                          # noqa: E402

DEFAULT_GAUGE_N = cfg(False)["gauge_n"]                    # exp3 real-run parity (no overrides)


def collect_gauge_arm(model, tok, gauge_n, arm):
    """Persona-only gauge run for ONE arm: free-association responses for the blind K-way judge.
    Mirrors collect_induction._collect_gauge exactly (no anti-word block, no filter -- we want
    coherent words) with the arm parameterized instead of hardcoded. Returns
    {concept|neutral: [responses]}."""
    import torch
    import common as K
    gauge = {}
    with torch.no_grad():
        for concept in [None] + list(P.CONCEPTS):
            ids = K.chat_ids(tok, P.GAUGE_PROBE, system=P.compose_gauge_system(concept, arm=arm),
                             device=model.device)
            rep = ids.repeat(gauge_n, 1)
            gout = model.generate(rep, attention_mask=torch.ones_like(rep), max_new_tokens=64,
                                  do_sample=True, temperature=1.0, top_p=0.98,
                                  pad_token_id=tok.eos_token_id)
            gauge[concept or "neutral"] = [tok.decode(gout[i, ids.shape[1]:],
                                                      skip_special_tokens=True)
                                           for i in range(gauge_n)]
    return gauge


def main():
    import torch
    import common as K
    import config as C
    ap = argparse.ArgumentParser()
    ap.add_argument("--gauge-n", type=int, default=DEFAULT_GAUGE_N,
                    help="responses per concept (default: the exp3 real-run cfg)")
    args = ap.parse_args()

    slug = C.ACTIVE
    outdir = C.DATA
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir / f"{slug}-evoked_alt-gauge.pt"
    if outpath.exists():
        print(f"GAUGEALT_SKIP {outpath.name} (resume)", flush=True)
        return
    model, tok = K.load_model(C.MODEL)
    print("MODEL_READY", flush=True)
    gauge = collect_gauge_arm(model, tok, args.gauge_n, arm="evoked_alt")
    tmp = outpath.with_suffix(".tmp")
    torch.save(dict(model=slug, arm="evoked_alt", gauge_n=args.gauge_n,
                    probe=P.GAUGE_PROBE, gauge=gauge), tmp)
    os.replace(tmp, outpath)                               # atomic
    print(f"GAUGEALT_SAVED {outpath.name} concepts={len(gauge)} n_per={args.gauge_n}", flush=True)


if __name__ == "__main__":
    main()
