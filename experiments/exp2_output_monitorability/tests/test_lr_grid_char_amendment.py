"""RED-first unit tests for Amendment 6 (POST-DATA, 2026-07-13): the amended char-surface
control decision rule in analysis/lr_grid_offline.py + the re-scoring runner
analysis/lr_grid_char_amendment.py. CPU-only, synthetic bundles, no model, no real data.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_lr_grid_char_amendment.py

C1  char_control_pass_amended is a ONE-SIDED test for positive surface signal:
    FAIL iff (mean - 2*sd) > 0 OR mean >= max(0.02, 0.10 * lr_bits); PASS otherwise;
    None/missing stays None. The 3B artifact case (precise negative mean) now PASSES;
    a statistically-positive char mean FAILS; a large-but-noisy positive mean FAILS on
    the materiality clause; the materiality threshold scales with the cell's LR bits.
C2  the REGISTERED Amendment-5 rule and its inputs are untouched: char_control_pass keeps
    the frozen two-sided behavior (the 3B inputs still FAIL under it), AM5_CHAR_SD_MULT
    stays 2.0, and dose_titration.SEEDS stays (0, 1, 2) -- before AND after the amended
    reader runs (the frozen 3-seed artifacts must stay byte-identical).
C3  secret_char_bits_amended reuses the SAME certified building blocks with only the seed
    set parameterized: at seeds (0, 1, 2) its per_seed values EQUAL secret_char_bits's on
    the same bundle (same function objects -> same numbers); default is 10 seeds (0..9);
    a missing bundle -> None; a too-thin pool -> skipped dict (never a crash / fake number).
C4  source-level certified reuse: secret_char_bits_amended calls dose_titration's
    common_n_subsample / build_vocab_index / RB._features / best_reader_proba_by_budget /
    bits_recovered and never assigns dose_titration.SEEDS.
C5  the runner's build_cell: old verdict comes from the FROZEN 3-seed values (never
    recomputed), new verdict from the amended rule, and the "positive, mechanism-confounded"
    label recompute keeps the label when the POSITION control failed (the amendment fixes
    only the char rule) and drops it when char was the sole failure.
"""
import importlib.util
import inspect
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ANA = os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis")

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    LGO = load_module("lr_grid_offline", os.path.join(ANA, "lr_grid_offline.py"))
except Exception as e:
    LGO = None
    check("import analysis/lr_grid_offline.py", False, f"{type(e).__name__}: {e}")

CONCEPTS = ["celebration", "ocean", "fear", "silence", "deception", "obedience",
            "debugging", "security", "curiosity", "anger", "warmth", "loneliness"]

if LGO is not None:
    # ---- C1: the amended decision rule ------------------------------------------------------
    try:
        f = LGO.char_control_pass_amended
        # the 3B secret_word artifact case: precise NEGATIVE mean -- must PASS the amended rule
        # (it FAILED the frozen two-sided rule only because its 3-seed sd was tiny).
        check("C1 precise negative mean PASSES (the 3B artifact case)",
              f(dict(mean=-0.0118, sd=0.0023), 0.191) is True)
        # statistically positive surface signal: mean - 2sd > 0 -> FAIL.
        check("C1 statistically positive mean FAILS (mean - 2sd > 0)",
              f(dict(mean=0.030, sd=0.010), 0.191) is False)
        # large-but-noisy positive mean: not statistically positive, but >= the materiality
        # threshold max(0.02, 0.10*lr) -> FAIL.
        check("C1 large noisy positive mean FAILS on materiality (mean >= max(0.02, 0.10*lr))",
              f(dict(mean=0.050, sd=0.050), 0.191) is False)
        # the SAME mean/sd passes when the cell's LR bits are large (threshold 0.10*1.109 ~ 0.111)
        check("C1 materiality threshold scales with the cell's LR bits",
              f(dict(mean=0.050, sd=0.050), 1.109) is True)
        # exactly at the threshold -> FAIL (rule is 'mean >= threshold')
        check("C1 mean == threshold FAILS (>= is inclusive)",
              f(dict(mean=0.020, sd=0.050), 0.10) is False)
        # missing lr_bits: the 0.02 absolute floor governs alone
        check("C1 lr_bits None -> the 0.02 floor still governs",
              f(dict(mean=0.019, sd=0.050), None) is True
              and f(dict(mean=0.021, sd=0.050), None) is False)
        # None / thin-skip inputs stay None (pending, disclosed -- never a silent verdict)
        check("C1 None / skipped stays None",
              f(None, 0.191) is None
              and f(dict(mean=None, skipped="thin", n=3), 0.191) is None
              and f(dict(mean=0.01), 0.191) is None)
    except Exception as e:
        check("C1 char_control_pass_amended", False, f"raised {type(e).__name__}: {e}")

    # ---- C2: the frozen registered rule + its constants are untouched -----------------------
    try:
        check("C2 frozen char_control_pass still FAILS the 3B inputs (two-sided, unchanged)",
              LGO.char_control_pass(dict(mean=-0.0118, sd=0.0023)) is False
              and LGO.char_control_pass(dict(mean=-0.0209, sd=0.0157)) is True)
        check("C2 AM5_CHAR_SD_MULT stays 2.0", LGO.AM5_CHAR_SD_MULT == 2.0)
        DT = sys.modules.get("dose_titration") or load_module(
            "dose_titration", os.path.join(ANA, "dose_titration.py"))
        check("C2 dose_titration.SEEDS stays (0, 1, 2)", tuple(DT.SEEDS) == (0, 1, 2))
    except Exception as e:
        check("C2 frozen rule untouched", False, f"raised {type(e).__name__}: {e}")

    # ---- C3: the amended reader = the certified reader with the seed set parameterized ------
    class CharTok:
        """Concept-separable decode: token id 100+ci -> a distinct letter, repeated."""
        def decode(self, ids):
            return "".join(chr(97 + (int(t) - 100) % 26) * 2 for t in ids)

    try:
        with tempfile.TemporaryDirectory() as td:
            streams, gidx = [], 0
            for ci, c in enumerate(CONCEPTS):
                for _ in range(6):
                    streams.append(dict(
                        gidx=gidx, concept=c, concept_idx=ci, arm="secret_word",
                        tokens=np.array([100 + ci] * 12), text="x", deg={}, accepted=True,
                        strength=1,
                        gen_topk=[dict(ids=np.array([100 + ci], dtype=np.int32),
                                       logp=np.array([0.0], dtype=np.float16))] * 12))
                    gidx += 1
            bp = os.path.join(td, "qwen2.5-1.5b-secret_word.pt")
            torch.save(dict(model="qwen2.5-1.5b", inject="secret_word",
                            concepts=CONCEPTS, strengths=[0, 1], streams=streams), bp)
            DT = sys.modules["dose_titration"]
            seeds_before = tuple(DT.SEEDS)
            base = LGO.secret_char_bits(bp, tok=CharTok())
            amended3 = LGO.secret_char_bits_amended(bp, tok=CharTok(), seeds=(0, 1, 2))
            check("C3 seeds (0,1,2): per_seed values EQUAL secret_char_bits's (same certified "
                  "function objects -> same numbers)",
                  isinstance(amended3, dict)
                  and amended3.get("per_seed") == base.get("per_seed")
                  and amended3.get("n") == base.get("n"), f"{amended3} vs {base}")
            amended10 = LGO.secret_char_bits_amended(bp, tok=CharTok())
            check("C3 default is 10 seeds (0..9), mean/sd over the 10 per-seed values",
                  isinstance(amended10, dict) and len(amended10.get("per_seed", [])) == 10
                  and list(amended10.get("seeds", [])) == list(range(10))
                  and abs(amended10["mean"] - np.mean(amended10["per_seed"])) < 1e-12
                  and abs(amended10["sd"] - np.std(amended10["per_seed"])) < 1e-12,
                  f"{amended10}")
            check("C3 a separable synthetic pool recovers positive bits at every seed",
                  min(amended10["per_seed"]) > 0.5, f"{amended10.get('per_seed')}")
            check("C3 dose_titration.SEEDS unchanged after the amended reader ran",
                  tuple(DT.SEEDS) == seeds_before == (0, 1, 2))
            check("C3 a missing bundle -> None (pending)",
                  LGO.secret_char_bits_amended(os.path.join(td, "nope.pt")) is None)
            thin = [s for s in streams if s["concept_idx"] in (0, 1)][:6]
            bp2 = os.path.join(td, "thin.pt")
            torch.save(dict(model="qwen2.5-1.5b", inject="secret_word",
                            concepts=CONCEPTS, strengths=[0, 1], streams=thin), bp2)
            cell2 = LGO.secret_char_bits_amended(bp2, tok=CharTok())
            check("C3 a too-thin pool is SKIPPED with the n disclosed",
                  isinstance(cell2, dict) and cell2.get("mean") is None
                  and "skipped" in cell2, f"{cell2}")
    except Exception as e:
        check("C3 secret_char_bits_amended", False, f"raised {type(e).__name__}: {e}")

    # ---- C4: source-level certified reuse ----------------------------------------------------
    try:
        src = inspect.getsource(LGO.secret_char_bits_amended)
        check("C4 amended reader uses the certified building blocks (common_n_subsample, "
              "build_vocab_index, RB._features, best_reader_proba_by_budget, bits_recovered)",
              all(s in src for s in ("common_n_subsample", "build_vocab_index", "_features",
                                     "best_reader_proba_by_budget", "bits_recovered")))
        check("C4 amended reader never assigns dose_titration.SEEDS",
              "SEEDS =" not in src and ".SEEDS=" not in src.replace(" ", ""))
    except Exception as e:
        check("C4 source-level reuse", False, f"raised {type(e).__name__}: {e}")

# ---- C5: the runner's build_cell (frozen-old vs amended-new, label recompute) ---------------
try:
    RUN = load_module("lr_grid_char_amendment", os.path.join(ANA, "lr_grid_char_amendment.py"))
    frozen_cell = dict(
        bits_mean=0.1913,
        am5_char=dict(bits=-0.0118, sd=0.0023, passed=False),
        am5_position=dict(share=0.491, passed=True),
        am5_label="positive, mechanism-confounded")
    cb_new = dict(mean=-0.0048, sd=0.0175, per_seed=[-0.0048] * 10, n=24,
                  seeds=list(range(10)))
    cell = RUN.build_cell(frozen_cell, cb_new)
    check("C5 old verdict comes from the FROZEN 3-seed values (never recomputed)",
          cell["old"]["mean"] == -0.0118 and cell["old"]["sd"] == 0.0023
          and cell["old"]["passed"] is False)
    check("C5 new verdict from the amended rule on the 10-seed values",
          cell["new"]["passed"] is True and cell["new"]["mean"] == -0.0048)
    check("C5 char-only failure: the label DROPS under the amendment",
          cell["label_old"] == "positive, mechanism-confounded"
          and cell["label_new"] is None and cell["label_changes"] is True, f"{cell}")
    # position-control failure is NOT touched by this amendment: the label STAYS.
    frozen_pos = dict(
        bits_mean=0.1007,
        am5_char=dict(bits=-0.0167, sd=0.0046, passed=False),
        am5_position=dict(share=0.5037, passed=False),
        am5_label="positive, mechanism-confounded")
    cell2 = RUN.build_cell(frozen_pos, dict(mean=-0.0048, sd=0.0110,
                                            per_seed=[-0.0048] * 10, n=24,
                                            seeds=list(range(10))))
    check("C5 position failure keeps the label (the amendment fixes ONLY the char rule)",
          cell2["new"]["passed"] is True
          and cell2["label_new"] == "positive, mechanism-confounded"
          and cell2["label_changes"] is False, f"{cell2}")
    # a sub-threshold cell never carries the label in either regime
    frozen_low = dict(bits_mean=0.003,
                      am5_char=dict(bits=-0.0118, sd=0.0023, passed=False),
                      am5_position=dict(share=0.30, passed=True), am5_label=None)
    cell3 = RUN.build_cell(frozen_low, cb_new)
    check("C5 a < 0.05-bit cell is never labeled (either regime)",
          cell3["label_old"] is None and cell3["label_new"] is None
          and cell3["label_changes"] is False)
except Exception as e:
    check("C5 runner build_cell", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
