"""Show the on-box eff_mag auto-tuner's calibration trace, and (for qwen3-1.7b) validate it against the
INDEPENDENT manual tuning. The trace is saved by covert_collect into the pulled bundle
(results/calibration_trace.json, also embedded in covert_collect.pt / meta) as STRUCTURED data -- parseable,
unlike the raw box run.log (which labkit >=v0.2.11 also pulls to res.log_path as a human-readable backup).

Run:  INTRO_MODEL=<slug> python3 analysis/show_calibration.py

For qwen3-1.7b it overlays the manual sweep we did by hand before the auto-tuner existed
(160->rank189, 220->108, 280->23, 340->27; clean ~70-79%) and the hand-picked operating points [280,340],
so step (3) of the plan -- "use this run to validate the auto-tune results" -- is one command.
"""
import json

import _paths as P

# Manual reference for the validation model (independent hand-tuning, pre-auto-tuner).
MANUAL = {
    "qwen3-1.7b": dict(sweep={160: 189, 220: 108, 280: 23, 340: 27}, picked=[280, 340],
                       picked_clean={280: 0.79, 340: 0.71}),
}

cal = None
f = P.RESULTS / "calibration_trace.json"
if f.exists():
    cal = json.loads(f.read_text())
else:
    import torch
    d = torch.load(P.DATA / "covert_collect.pt", map_location="cpu", weights_only=False)
    cal = d.get("calibration")

if not cal:
    print(f"No calibration trace for {P.ACTIVE} -- this run used a registry effmags override or --no-calibrate")
    print("(only auto-tuned runs have a trace). Nothing to show.")
    raise SystemExit(0)

b = cal["bracket"]
# floor is now RELATIVE (max(min_floor, rel_floor*base_clean)); older traces carried an absolute clean_floor
floor = cal.get("floor", cal.get("clean_floor"))
print(f"model={P.ACTIVE}")
print(f"bracket: ||v||~{b['vnorm']}  resid_norm~{b['resid_norm']}  -> [{b['lo']:.0f}, {b['hi0']:.0f}]")
if "base_clean" in cal:
    print(f"baseline clean (uninjected) = {cal['base_clean']}  -> relative floor = {floor} "
          f"(rel {cal.get('rel_floor')} x base, min {cal.get('min_floor')})")
print(f"targets: arm-A nameability {cal.get('target_nats', cal.get('target_ranks'))} nats  floor={floor}  "
      f"(n_cal={cal['n_cal']} concepts x n_streams={cal['n_streams']}, cal_tokens={cal['cal_tokens']})\n")

print(f"{'target':>6} {'effmag':>7} {'nats':>6} {'clean':>6} {'bracket':>16}  verdict")
for e in cal["evaluations"]:
    nats = e.get("nats")                          # new traces log nats; old ones logged rank (shown as '-')
    if e["clean"] < floor:
        v = "clean<floor -> lower"
    elif nats is None or nats < e["target"]:
        v = "too weak -> raise"
    else:
        v = "in band -> accept/refine"
    print(f"{e['target']:>6} {e['effmag']:>7.0f} {('-' if nats is None else f'{nats:.1f}'):>6} {e['clean']:>6.2f} "
          f"{str(e['bracket']):>16}  {v}")

if cal.get("cap"):
    print(f"\ncap (max-achievable nameability while staying capable): "
          f"{cal['cap']['nats']} nats @ eff_mag {cal['cap']['effmag']}")
print(f"tuned per target: {cal['tuned']}")
print(f"RESULT eff_mags : {cal['result']}" + (f"   [{cal['note']}]" if cal.get("note") else ""))

man = MANUAL.get(P.ACTIVE)
if man:
    print("\n--- VALIDATION vs independent manual tuning ---")
    print(f"manual sweep (rank by eff_mag): {man['sweep']}")
    print(f"manual picked operating points: {man['picked']}  (clean {man['picked_clean']})")
    print(f"auto-tuner result             : {cal['result']}")
    lo, hi = min(man["picked"]), max(man["picked"])
    res = cal["result"]
    # outcome-based: do the auto eff_mags sit in the neighborhood of the hand-picked band (+/-25%)?
    ok = len(res) == 2 and 0.75 * lo <= res[0] <= 1.25 * hi and 0.75 * lo <= res[1] <= 1.25 * hi
    print(f"auto within +/-25% of the hand-picked [{lo},{hi}] band: {'YES' if ok else 'NO -- inspect'}")
    print("NOTE: confirm the OUTCOME too -- INTRO_MODEL=%s python3 analysis/check_injection.py" % P.ACTIVE)
    print("(arm-A rank in band + clean>=floor on the FULL 12-concept data is the independent gate).")
