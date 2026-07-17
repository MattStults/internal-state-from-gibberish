"""exp3 gauge judge: score the evoked-arm manipulation check with a blind K-way judge (the PREREG gauge).

For each model's evoked bundle, the collector saved persona-only free-association responses
(bundle["gauge"]: {concept|neutral: [texts]}). A judge that has NEVER seen the persona is shown one response
plus the 12 concept labels (order shuffled deterministically per item so position can't leak) and must pick
which concept the response best expresses. The persona "took" at a scale iff judge accuracy's Wilson-95 lower
bound exceeds chance (1/12) -- primers.PREREG. The judge is a PINNED separate model (JUDGE_MODEL below):
independent of the Qwen family and of the R_emb embedding geometry, per the prereg independence requirement.

Pure logic (prompt build, choice parsing) is unit-tested offline; the API loop needs ANTHROPIC_API_KEY.
Cost: 3 models x 12 concepts x gauge_n responses, ~5 output tokens each -- pennies on the pinned judge.

Run:  .venv/bin/python experiments/exp3_induction_and_scale/gauge_judge.py
Writes reports/gauge_results.json. A scale point whose gauge FAILS is flagged confounded and does not count
toward Prediction 3 (see run_induction / proposed-experiment.md).
"""
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from primers import CONCEPTS, PREREG  # noqa: E402
from run_induction import wilson_ci   # noqa: E402  the tested PREREG CI method

JUDGE_MODEL = "claude-haiku-4-5-20251001"   # PINNED dated snapshot (an alias can be repointed; prereg needs a
                                            # fixed judge): separate family from Qwen, off R_emb geometry
MODELS = ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")


# ---------------------------------------------------------------- pure (unit-tested)
def build_judge_prompt(response_text, labels, item_key):
    """The blind judge prompt: the response + the K labels, shuffled deterministically per item_key (so label
    position can't leak the answer and reruns are reproducible). The judge never sees the persona."""
    order = list(labels)
    random.Random(int(hashlib.sha256(item_key.encode()).hexdigest(), 16) % (2**32)).shuffle(order)
    return ("A language model wrote the following short free-association passage:\n\n"
            f"---\n{response_text}\n---\n\n"
            "Which ONE of these concepts does the passage best express?\n"
            + "\n".join(f"- {c}" for c in order)
            + "\n\nAnswer with exactly one word: the concept name.")


def parse_choice(reply, labels):
    """Match the judge's reply to exactly one label (case/punctuation tolerant, may be embedded in a
    sentence). None if zero or several labels appear -- an unparseable reply counts as WRONG, never as a
    silent guess."""
    words = set(re.findall(r"[a-z]+", reply.lower()))
    hits = [c for c in labels if c.lower() in words]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------- orchestration (API)
def judge_model_gauge(gauge, client, model_slug):
    """Score one model's gauge dict. Returns per-concept counts, PER-ITEM raw replies (so the stochastic
    verdict is reconstructible offline -- save-the-primitives), overall accuracy + Wilson CI + pass."""
    missing = set(CONCEPTS) - set(gauge)
    assert not missing, f"gauge dict missing concepts (would silently shrink n): {sorted(missing)}"
    per_concept, items, served_by, k_total, n_total = {}, [], None, 0, 0
    for concept in CONCEPTS:                                # neutral rows have no true label -> not scored
        texts = gauge[concept]
        k = 0
        for i, text in enumerate(texts):
            prompt = build_judge_prompt(text, CONCEPTS, item_key=f"{model_slug}|{concept}|{i}")
            resp = client.messages.create(
                model=JUDGE_MODEL, max_tokens=16, temperature=0,   # deterministic-ish; replies persisted below
                messages=[{"role": "user", "content": prompt}],
            )
            served_by = served_by or resp.model                    # record the exact serving model snapshot
            reply = next((b.text for b in resp.content if b.type == "text"), "")
            parsed = parse_choice(reply, CONCEPTS)                 # None (unparseable) counts as WRONG
            items.append(dict(concept=concept, i=i, reply=reply, parsed=parsed,
                              correct=bool(parsed == concept)))
            k += int(parsed == concept)
        per_concept[concept] = (k, len(texts))
        k_total += k
        n_total += len(texts)
    lo, hi = wilson_ci(k_total, n_total)
    return dict(model=model_slug, judge=JUDGE_MODEL, judge_served_by=served_by,
                per_concept=per_concept, items=items,
                acc=(k_total / n_total if n_total else 0.0), n=n_total,
                ci95=[lo, hi], chance=PREREG["gauge_chance"],
                gauge_pass=bool(lo > PREREG["gauge_chance"]))


def main():
    import anthropic
    import torch
    client = anthropic.Anthropic()                          # key from env (existing credential; none created)
    base = os.path.join(HERE, "reports")
    os.makedirs(base, exist_ok=True)
    outpath = os.path.join(base, "gauge_results.json")
    out = []
    for m in MODELS:
        p = os.path.join(REPO, "runs", "_ind", m, "data", f"{m}-evoked.pt")
        if not os.path.exists(p):
            print(f"[{m}] no evoked bundle at {p} -- skipped")
            continue
        gauge = torch.load(p, map_location="cpu", weights_only=False).get("gauge")   # self-produced bundle
        if not gauge:
            print(f"[{m}] no gauge in bundle -- skipped")
            continue
        r = judge_model_gauge(gauge, client, m)
        out.append(r)
        json.dump(out, open(outpath, "w"), indent=2)        # checkpoint per model (API loop is ~100 calls each)
        print(f"[{m}] judge={r['judge_served_by']} acc={r['acc']:.2f} ({r['n']} items) "
              f"ci95=[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}] chance={r['chance']:.3f} "
              f"gauge_pass={r['gauge_pass']}", flush=True)
        print("  per-concept: " + " ".join(f"{c}:{k}/{n}" for c, (k, n) in r["per_concept"].items()), flush=True)
    print(f"wrote {outpath}")


if __name__ == "__main__":
    main()
