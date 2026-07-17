"""RED-first unit test for the exp3 induction collector's PURE assembly/accounting (no model, no GPU).

The collector's generation loop needs a GPU, but its bundle-assembly and per-concept acceptance accounting are
pure and must (a) produce a bundle that exp2's OWN loader (filter_streams) consumes unchanged -- induced =
strength 1, neutral = strength 0, inject = arm -- and (b) count word-free acceptance per concept (the
feasibility gate). This test drives both with synthetic streams.
  Run: .venv/bin/python experiments/exp3_induction_and_scale/tests/test_collect.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, ".."))                                   # exp3: collect_induction, primers
sys.path.insert(0, os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis"))  # exp2 loader
from collect_induction import (assemble_bundle, acceptance_report, stream_record,  # noqa: E402
                               analyzable_report, check_min_per_class)
from loader import filter_streams                                              # noqa: E402  the exp2 loader

checks = []
def check(name, cond):
    checks.append((name, bool(cond)))

CONCEPTS = ["ocean", "fear", "silence"]


def _gtk(n=4):                                     # a synthetic gen_topk: n steps, top-2 each
    return [dict(ids=np.array([10 + s, 20 + s], dtype=np.int32),
                 logp=np.array([-0.5, -1.5], dtype=np.float16)) for s in range(n)]


def _mk(concept, strength, accepted, gidx):
    ci = CONCEPTS.index(concept)
    return stream_record(gidx=gidx, concept=concept, concept_idx=ci, arm="evoked",
                         tokens=[10, 11, 12, 13], text="qxz fjm", deg={"word_rate": 0.0},
                         accepted=accepted, strength=strength, gen_topk=_gtk() if accepted else None)


# a mixed batch: induced (s1) accepted/rejected + neutral (s0) per concept
records = []
g = 0
for c in CONCEPTS:
    for acc in (True, True, False):                # induced: 2 accepted, 1 rejected  -> 2/3
        records.append(_mk(c, 1, acc, g)); g += 1
    records.append(_mk(c, 0, True, g)); g += 1     # one neutral control

# (1) stream_record shape
r = records[0]
check("stream_record keys", all(k in r for k in
      ("gidx", "concept", "concept_idx", "arm", "tokens", "accepted", "strength", "gen_topk")))
check("stream_record concept_idx", r["concept_idx"] == 0 and r["concept"] == "ocean")

# (2) acceptance_report: per-concept accepted/total over the INDUCED (strength>0) streams only
rep = acceptance_report(records, induced_strength=1)
check("acc per concept ocean 2/3", rep["ocean"] == (2, 3))
check("acc per concept fear 2/3", rep["fear"] == (2, 3))
check("acc report excludes neutral", all(tot == 3 for (_, tot) in rep.values()))

# (3) assemble_bundle -> consumed by exp2's filter_streams UNCHANGED
bundle = assemble_bundle(model="qwen2.5-1.5b", arm="evoked", concepts=CONCEPTS, records=records)
check("bundle model", bundle["model"] == "qwen2.5-1.5b")
check("bundle inject=arm", bundle["inject"] == "evoked")
check("bundle strengths has 0 and 1", set(bundle["strengths"]) == {0, 1})
check("bundle concepts", bundle["concepts"] == CONCEPTS)

fs = filter_streams(bundle)                        # strong dose = 1 (induced); should drop neutral + rejected
check("filter_streams returns 6 induced-accepted", len(fs["streams"]) == 6)   # 2 accepted x 3 concepts
check("filtered carry gen_topk", all(s["gen_topk"] for s in fs["streams"]))
check("filtered tokens are int arrays", all(s["tokens"].dtype.kind == "i" for s in fs["streams"]))
check("filtered concept_idx present", sorted({s["concept_idx"] for s in fs["streams"]}) == [0, 1, 2])
check("filter_streams model/inject passthrough", fs["model"] == "qwen2.5-1.5b" and fs["inject"] == "evoked")

# (4) analyzable_report + check_min_per_class: the B2 feasibility gate. Only accepted INDUCED streams with
#     >= min_tokens gen_topk steps survive the analyzer's >=12-token filter, so those are what must clear n.
ar = [stream_record(gidx=g, concept="ocean", concept_idx=0, arm="evoked",
                    tokens=list(range(20)), text="q", deg={}, accepted=True, strength=1,
                    gen_topk=_gtk(n=14)) for g in range(100, 105)]        # 5 ocean, len-14 gen_topk (analyzable)
ar += [stream_record(gidx=g, concept="ocean", concept_idx=0, arm="evoked",
                     tokens=[1], text="q", deg={}, accepted=True, strength=1,
                     gen_topk=_gtk(n=4)) for g in range(105, 108)]        # 3 ocean, len-4 (too short -> dropped)
ar += [stream_record(gidx=200, concept="fear", concept_idx=1, arm="evoked", tokens=[1], text="q", deg={},
                     accepted=True, strength=1, gen_topk=_gtk(n=14))]     # 1 fear analyzable
rep2 = analyzable_report(ar, min_tokens=12)
check("analyzable counts len>=12 only (ocean 5)", rep2["ocean"] == 5)
check("analyzable fear 1", rep2["fear"] == 1)
raised = False
try:
    check_min_per_class(ar, min_per_class=3, min_tokens=12)               # fear has 1 < 3 -> must raise
except (RuntimeError, ValueError):
    raised = True
check("check_min_per_class raises on shortfall", raised)
ok2 = True
try:
    check_min_per_class(ar, min_per_class=0, min_tokens=12)               # 0 = off -> no raise
except Exception:
    ok2 = False
check("check_min_per_class off when 0", ok2)

for name, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
sys.exit(0 if all(c for _, c in checks) else 1)
