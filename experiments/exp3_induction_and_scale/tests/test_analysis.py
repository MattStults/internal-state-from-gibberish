"""RED-first unit test for exp3's PURE analysis logic (no model, no bundles, no GPU).

Covers (1) wilson_ci -- the pre-registered gauge CI method; (2) arm_summary -- per-seed bits curves ->
mean/sd + per-seed gap stats; (3) prereg_verdicts -- the frozen PREREG thresholds applied to a summary
(recover per reader, gap_present via the per-seed gap, exactly as pinned in primers.PREREG).
  Run: .venv/bin/python experiments/exp3_induction_and_scale/tests/test_analysis.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from run_induction import arm_summary, prereg_verdicts, wilson_ci  # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

# (1) wilson_ci: known values. 8/8 correct -> lower bound well above 1/12; 1/12 correct -> spans chance.
lo, hi = wilson_ci(8, 8)
check("wilson 8/8 low > chance", lo > 1 / 12 and hi == 1.0)
lo2, hi2 = wilson_ci(1, 12)
check("wilson 1/12 spans chance", lo2 < 1 / 12 < hi2)
lo3, _ = wilson_ci(0, 10)
check("wilson 0/n low is 0-ish", lo3 == 0.0 or lo3 < 1e-9)

# (2) arm_summary: per-seed curves -> mean/sd per reader + per-seed dist-emb gap stats at top budget.
budgets = (2, 4)
bits_by_seed = [  # seed -> {mode: {T: bits}}
    {"dist": {2: 1.0, 4: 2.0}, "emb": {2: 0.2, 4: 0.5}, "sampled": {2: 0.1, 4: 0.4}},
    {"dist": {2: 1.2, 4: 2.2}, "emb": {2: 0.3, 4: 0.7}, "sampled": {2: 0.1, 4: 0.5}},
]
s = arm_summary(bits_by_seed, budgets, H=3.585)
check("summary mean dist@4", np.isclose(s["readers"]["dist"]["bits_mean"][4], 2.1))
check("summary sd emb@4", np.isclose(s["readers"]["emb"]["bits_sd"][4], np.std([0.5, 0.7])))
# per-seed gaps at top budget: [2.0-0.5, 2.2-0.7] = [1.5, 1.5] -> mean 1.5, sd 0
check("per-seed gap mean", np.isclose(s["gap_dist_emb"]["mean"], 1.5))
check("per-seed gap sd", np.isclose(s["gap_dist_emb"]["sd"], 0.0))
check("summary carries H", s["H_bits"] == 3.585)

# (3) prereg_verdicts vs the frozen thresholds (recover_margin_bits=0.2, gap_present_bits=0.2)
prereg = dict(recover_margin_bits=0.2, gap_present_bits=0.2)
v = prereg_verdicts(s, prereg)
check("dist recovers (2.1-0.1 >= 0.2)", v["recover"]["dist"] is True)
check("sampled recovers (0.45-0.05 >= 0.2)", v["recover"]["sampled"] is True)
check("gap present (1.5>=0.2, 1.5-0>0)", v["gap_present"] is True)

# gap NOT present when mean under threshold
weak = arm_summary([{"dist": {2: 0.3, 4: 0.4}, "emb": {2: 0.2, 4: 0.3}, "sampled": {2: 0.1, 4: 0.2}},
                    {"dist": {2: 0.3, 4: 0.4}, "emb": {2: 0.2, 4: 0.3}, "sampled": {2: 0.1, 4: 0.2}}],
                   budgets, H=3.585)
vw = prereg_verdicts(weak, prereg)
check("gap absent when mean 0.1 < 0.2", vw["gap_present"] is False)
check("emb recovers (0.3-0 >= 0.2)", vw["recover"]["emb"] is True)

# recover must use mean MINUS SD (not bare mean): mean 0.5, sd 0.4 -> 0.1 < 0.2 -> False
sd_case = arm_summary([{"dist": {2: 0.1, 4: 0.9}, "emb": {2: 0.0, 4: 0.1}, "sampled": {2: 0.0, 4: 0.1}},
                       {"dist": {2: 0.1, 4: 0.1}, "emb": {2: 0.0, 4: 0.1}, "sampled": {2: 0.0, 4: 0.1}}],
                      budgets, H=3.585)
check("recover uses mean-sd (0.5-0.4 < 0.2 -> False)", prereg_verdicts(sd_case, prereg)["recover"]["dist"] is False)
# recover must use the MARGIN (not just > 0): mean 0.15, sd 0 -> below 0.2 -> False
mg_case = arm_summary([{"dist": {2: 0.1, 4: 0.15}, "emb": {2: 0.0, 4: 0.1}, "sampled": {2: 0.0, 4: 0.1}}] * 2,
                      budgets, H=3.585)
check("recover uses margin (0.15-0 < 0.2 -> False)", prereg_verdicts(mg_case, prereg)["recover"]["dist"] is False)

# gap NOT present when noisy: per-seed gaps [0.5, -0.1] -> mean 0.2 ok but mean-sd <= 0
noisy = arm_summary([{"dist": {2: 0.1, 4: 1.0}, "emb": {2: 0.1, 4: 0.5}, "sampled": {2: 0.0, 4: 0.1}},
                     {"dist": {2: 0.1, 4: 0.4}, "emb": {2: 0.1, 4: 0.5}, "sampled": {2: 0.0, 4: 0.1}}],
                    budgets, H=3.585)
vn = prereg_verdicts(noisy, prereg)
check("gap absent when per-seed unstable", vn["gap_present"] is False)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
