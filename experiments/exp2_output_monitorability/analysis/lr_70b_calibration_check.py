"""Independent verification of the 70B calibration claim, using the CERTIFIED evaluate_cell.
Builds the 12-way context matrix S per arm from lr_raw_batch_output.jsonl and scores calibrated bits.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "harness")
import run_llama70b_scout as SC          # _find_stream_span_lps
import lr_vllm as LRV                    # ll_over_span
import config as CFG                     # COVERT_CONCEPTS

HERE = "experiments/exp2_output_monitorability/analysis"
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
LRO = _load("lr_reader_offline", os.path.join(HERE, "lr_reader_offline.py"))

CONCEPTS = list(CFG.COVERT_CONCEPTS)
CIDX = {c: i for i, c in enumerate(CONCEPTS)}
print("12 concepts:", CONCEPTS)

# streams: (arm, stream_idx) -> {concept, text}
streams = json.load(open("runs/llama70b_scout/streams_llama70b.json"))
smap = {(s["arm"], s["stream_idx"]): s for s in streams}

# ll[(arm, stream_idx, ctxname)] = summed span logprob (nats)
ll = {}
n_lines = 0; n_err = 0
with open("runs/llama70b_scout/lr_raw_batch_output.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line); n_lines += 1
        cid = rec.get("custom_id", "")
        parts = cid.split(":")
        if len(parts) != 5: continue
        _, arm, concept, sidx, ctx = parts
        sidx = int(sidx)
        body = (rec.get("response") or {}).get("body")
        if body is None: n_err += 1; continue
        s = smap.get((arm, sidx))
        if s is None: continue
        span = SC._find_stream_span_lps(body, s["text"])
        if not span: continue
        ll[(arm, sidx, ctx)] = LRV.ll_over_span(span)
print(f"parsed {n_lines} lines, {n_err} missing-body; ll entries: {len(ll)}")

def build_arm(arm):
    """Return S (n,12), y (n,), and diagnostics for one arm."""
    S_rows, y_rows = [], []
    raw_diag, gen_lift, centered = [], [], []
    concept_of = []
    # unique streams in this arm
    sidxs = sorted({k[1] for k in ll if k[0] == arm})
    for sidx in sidxs:
        s = smap.get((arm, sidx))
        if s is None: continue
        c = s["concept"]; ci = CIDX[c]
        matched = ll.get((arm, sidx, "matched"))
        neutral = ll.get((arm, sidx, "neutral"))
        if matched is None or neutral is None: continue
        # 12-vector: column j = LL under concept_j context
        row = np.full(12, np.nan)
        row[ci] = matched
        ok = True
        for j, cj in enumerate(CONCEPTS):
            if j == ci: continue
            v = ll.get((arm, sidx, cj))
            if v is None: ok = False; break
            row[j] = v
        if not ok or np.isnan(row).any(): continue
        S_rows.append(row); y_rows.append(ci); concept_of.append(c)
        mism = np.array([row[j] for j in range(12) if j != ci])
        raw_diag.append(matched - neutral)
        gen_lift.append(mism.mean() - neutral)
        centered.append(matched - mism.mean())
    S = np.array(S_rows); y = np.array(y_rows)
    return S, y, np.array(raw_diag), np.array(gen_lift), np.array(centered), concept_of

def concept_bootstrap_ci(vals, concepts_list, B=10000, seed=20260713):
    from collections import defaultdict
    byc = defaultdict(list)
    for v, c in zip(vals, concepts_list): byc[c].append(v)
    ks = sorted(byc); cm = np.array([np.mean(byc[k]) for k in ks])
    rng = np.random.default_rng(seed)
    boots = cm[rng.integers(0, len(ks), size=(B, len(ks)))].mean(axis=1)
    return float(cm.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

print(f"\n{'arm':16s} {'n':>4} {'raw_diag':>9} {'gen_lift':>9} {'centered':>9} "
      f"{'centered 95% CI':>22} {'calib_bits':>12} {'top1':>7}")
for arm in ["secret_word", "secret_sustain", "evoked"]:
    S, y, raw, lift, cen, cof = build_arm(arm)
    if len(y) == 0:
        print(f"{arm:16s} NO DATA"); continue
    res = LRO.evaluate_cell(S, y)
    cm, lo, hi = concept_bootstrap_ci(cen, cof)
    frac_lift = lift.mean() / raw.mean() if raw.mean() else float('nan')
    print(f"{arm:16s} {len(y):>4} {raw.mean():>9.3f} {lift.mean():>9.3f} {cen.mean():>9.3f} "
          f"[{lo:>7.2f},{hi:>7.2f}] {res['bits_mean']:>+8.3f}±{res['bits_sd']:.3f} {res['top1_mean']*100:>6.1f}%")
    print(f"{'':16s}      generic-lift fraction of raw diag = {frac_lift*100:.1f}%  "
          f"(centered clears 0: {lo>0})")
print(f"\nchance top-1 = {100/12:.1f}%   |   7B reference: 0.405 calib bits, top1 25.1%")
