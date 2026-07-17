"""RED-first unit test for the gauge judge's PURE logic (no API calls, no bundles).

Covers build_judge_prompt (all 12 labels present; per-item deterministic shuffle so label order can't leak the
answer or vary across reruns) and parse_choice (robust matching of the judge's reply to a label; None on
garbage rather than a silent wrong pick).
  Run: .venv/bin/python experiments/exp3_induction_and_scale/tests/test_gauge.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gauge_judge import build_judge_prompt, parse_choice  # noqa: E402
from primers import CONCEPTS                              # noqa: E402

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

# (1) prompt: contains the response text and every label exactly once; no primer text leaks in
p = build_judge_prompt("glassy swells rolling forever", CONCEPTS, item_key="qwen2.5-1.5b|ocean|0")
check("prompt has response", "glassy swells rolling forever" in p)
check("prompt has all 12 labels once", all(p.count(c) == 1 for c in CONCEPTS))

# (2) shuffle: deterministic per item_key, different across items, not the canonical order for typical keys
p2 = build_judge_prompt("x", CONCEPTS, item_key="qwen2.5-1.5b|ocean|0")
p3 = build_judge_prompt("x", CONCEPTS, item_key="qwen2.5-1.5b|ocean|1")
order = lambda s: sorted(CONCEPTS, key=s.find)  # noqa: E731  label order as they appear in the prompt
check("same key -> same order", order(p2) == order(build_judge_prompt("x", CONCEPTS, item_key="qwen2.5-1.5b|ocean|0")))
check("different key -> different order", order(p2) != order(p3))

# (3) parse_choice: exact, case/punctuation tolerant, embedded-in-sentence; None on garbage/ambiguous
check("parse exact", parse_choice("ocean", CONCEPTS) == "ocean")
check("parse case+punct", parse_choice("  Ocean.\n", CONCEPTS) == "ocean")
check("parse in sentence", parse_choice("The concept is clearly loneliness", CONCEPTS) == "loneliness")
check("parse garbage -> None", parse_choice("I cannot tell", CONCEPTS) is None)
check("parse ambiguous (two labels) -> None", parse_choice("ocean or fear", CONCEPTS) is None)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
