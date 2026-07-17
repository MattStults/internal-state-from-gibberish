"""Amendment 6 (POST-DATA, 2026-07-13) re-scoring runner: the amended char-surface control
(lr_grid_offline.secret_char_bits_amended / char_control_pass_amended -- 10 seeds, one-sided)
on every secret cell of the LR scale grid with a bundle present (1.5B/3B/7B x secret_word /
secret_sustain + maintained_secret@1.5B), side by side with the FROZEN Amendment-5 3-seed
verdict read from reports/lr_grid_results.json (never recomputed, never overwritten).

Writes reports/lr_grid_char_amendment_results.json: per cell the old rule inputs+verdict, the
new 10-seed mean/sd/per_seed, the new verdict, and whether the "positive, mechanism-confounded"
label would change (the position control is NOT touched by this amendment -- a position-control
failure keeps the label regardless of the char verdict). CPU-only; reads bundles + the frozen
results json; touches nothing under runs/.

  Run: .venv/bin/python experiments/exp2_output_monitorability/analysis/lr_grid_char_amendment.py
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS_JSON = os.path.join(HERE, "..", "reports", "lr_grid_results.json")
OUT_JSON = os.path.join(HERE, "..", "reports", "lr_grid_char_amendment_results.json")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LGO = _load("lr_grid_offline", os.path.join(HERE, "lr_grid_offline.py"))

# every secret cell with a bundle: (generator, set, reader-of-record, ctx). The char control is
# a POOL-level readout; the cell whose LR bits set the materiality threshold (and which carried
# / could carry the Amendment-5 label) is the diagonal own-size reader -- the only positive
# secret cells in the frozen grid (off-diagonal secret cells read ~0 and are never labeled).
CELLS = [
    ("qwen2.5-1.5b", "secret_word", "SW"),
    ("qwen2.5-1.5b", "secret_sustain", "SS"),
    ("qwen2.5-3b", "secret_word", "SW"),
    ("qwen2.5-3b", "secret_sustain", "SS"),
    ("qwen2.5-7b", "secret_word", "SW"),
    ("qwen2.5-7b", "secret_sustain", "SS"),
    ("qwen2.5-1.5b", "maintained_secret", "SM"),
]

OLD_RULE = (f"Amendment 5 (frozen, 3 seeds): pass iff |char mean| <= "
            f"{LGO.AM5_CHAR_SD_MULT:g} sd")
NEW_RULE = (f"Amendment 6 (POST-DATA 2026-07-13, {len(LGO.AM6_CHAR_SEEDS)} seeds): FAIL iff "
            f"(mean - {LGO.AM5_CHAR_SD_MULT:g}*sd) > 0 OR mean >= "
            f"max({LGO.AM6_CHAR_ABS_FLOOR:g}, {LGO.AM6_CHAR_LR_FRAC:g} * LR bits_mean); "
            "PASS otherwise; None stays None (pending, disclosed)")


def build_cell(frozen_cell, cb_new):
    """Assemble one cell's old-vs-new record. frozen_cell: the FROZEN results-json cell dict
    (bits_mean, am5_char, am5_position, am5_label -- the 3-seed inputs+verdict are read, never
    recomputed). cb_new: the 10-seed secret_char_bits_amended dict (or None: pending).
    The label recompute changes ONLY the char clause: label iff bits_mean >= SECRET_SURPRISE
    AND (amended char verdict is False OR the frozen position verdict is False)."""
    bits = frozen_cell.get("bits_mean")
    old_char = frozen_cell.get("am5_char") or {}
    pos = frozen_cell.get("am5_position") or {}
    old = dict(mean=old_char.get("bits"), sd=old_char.get("sd"),
               passed=old_char.get("passed"), rule=OLD_RULE)
    new_pass = LGO.char_control_pass_amended(cb_new, bits)
    new = dict(mean=None, sd=None, per_seed=None, n=None, seeds=None)
    if isinstance(cb_new, dict):
        new.update({k: cb_new.get(k) for k in ("mean", "sd", "per_seed", "n", "seeds")})
        if cb_new.get("skipped"):
            new["skipped"] = cb_new["skipped"]
    new["passed"] = new_pass
    if new.get("mean") is not None and new.get("sd") is not None:
        new["stat_positive_lower"] = float(new["mean"]) - LGO.AM5_CHAR_SD_MULT * float(new["sd"])
    new["materiality_threshold"] = (
        None if bits is None
        else max(LGO.AM6_CHAR_ABS_FLOOR, LGO.AM6_CHAR_LR_FRAC * float(bits)))
    new["rule"] = NEW_RULE

    label_old = frozen_cell.get("am5_label")
    positive = bits is not None and bits >= LGO.SECRET_SURPRISE
    label_new = (LGO.AM5_LABEL if positive and (new_pass is False
                                                or pos.get("passed") is False) else None)
    return dict(lr_bits=bits, old=old, new=new,
                position=dict(share=pos.get("share"), passed=pos.get("passed")),
                label_old=label_old, label_new=label_new,
                label_changes=bool(label_old != label_new))


def main(results_json=RESULTS_JSON, out_json=OUT_JSON, char_fn=None):
    char_fn = char_fn or LGO.secret_char_bits_amended
    with open(results_json) as f:
        frozen = json.load(f)
    out = dict(
        amendment="Amendment 6 (POST-DATA, 2026-07-13): amended char-surface control, "
                  "triggered by the 3B char-control failure (lr_scale_grid_prereg.md)",
        old_rule=OLD_RULE, new_rule=NEW_RULE,
        note="The frozen Amendment-5 3-seed verdicts stay on the books "
             "(lr_grid_results.json, unmodified); this artifact reports both side by side. "
             "The position control is unchanged by Amendment 6.",
        cells={})
    for gen, ss, cs in CELLS:
        reader = "qwen2.5-1.5b" if ss == "maintained_secret" else gen
        cell_key = f"{gen}/{ss}x{cs}"
        frozen_cell = (frozen.get("readers", {}).get(reader) or {}).get(cell_key)
        if frozen_cell is None:
            out["cells"][f"{ss}@{gen}"] = dict(skipped="no frozen cell in results json")
            continue
        bp = LGO.E5_BUNDLE if ss == "maintained_secret" else LGO._bundle_path(gen, ss)
        print(f"{ss}@{gen}: bundle {os.path.relpath(bp, REPO)}", flush=True)
        cb = char_fn(bp)
        rec = build_cell(frozen_cell, cb)
        rec["reader"] = reader
        rec["bundle"] = os.path.relpath(bp, REPO) if os.path.exists(bp) else None
        out["cells"][f"{ss}@{gen}"] = rec
        o, nw = rec["old"], rec["new"]
        print(f"  LR {rec['lr_bits']:+.4f} | old(3 seeds) {o['mean']:+.4f} +/- {o['sd']:.4f} "
              f"-> {'pass' if o['passed'] else 'FAIL'} | new(10 seeds) "
              + (f"{nw['mean']:+.4f} +/- {nw['sd']:.4f}" if nw.get("mean") is not None
                 else "pending")
              + f" -> {'pass' if nw['passed'] else 'FAIL' if nw['passed'] is False else 'pending'}"
              f" | label {rec['label_old']!r} -> {rec['label_new']!r}"
              + ("  [CHANGES]" if rec["label_changes"] else ""), flush=True)
    out_path = os.path.abspath(out_json)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {out_path}")
    return out


if __name__ == "__main__":
    main()
