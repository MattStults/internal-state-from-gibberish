"""Baseline-clean probe + system-prompt VARIANT SWEEP (no injection): for the loaded model, measure the
fraction of non-degenerate word-free streams across candidate system prompts and thinking modes, all on ONE
model load (the download is the expensive part, so sweep in-rental).

Motivation: Qwen3 sustains only ~31-45% clean at 128 tokens (vs Qwen2.5 ~89-94%) -- it falls into
enumeration loops (recites the alphabet, then repeats). Forcing thinking ON makes it WORSE (3%, it plans the
enumeration). The remaining lever is the SYSTEM PROMPT: push varied multi-char fragments, forbid single
letters / alphabet recitation. This probe A/Bs prompt variants to find one that lifts Qwen3 toward Qwen2.5
WITHOUT hurting Qwen2.5 (a shared control -> whatever wins must be re-run on all models).

Run on GPU:
  INTRO_MODEL=qwen3-4b   python3 src/baseline_clean.py --variants orig,fragments,calm,antiloop --modes nothink,think
  INTRO_MODEL=qwen2.5-3b python3 src/baseline_clean.py --variants orig,fragments,calm,antiloop --modes nothink
"""
import argparse
import collections
import json
import time

import numpy as np
import torch

import config as C
import common as K
from covert_collect import degeneracy, is_degenerate   # the EXACT acceptance logic from the full collector

# Candidate system prompts + the user prompt live in config now (shared with covert_collect --variant, so
# the probe and the full collect use the IDENTICAL prompt text per variant).
PROMPT_VARIANTS = C.PROMPT_VARIANTS
PROMPT = C.GEN_PROMPT


def valid_subset_accounting(streams, think=True, min_chars=20):
    """The valid-subset accounting for the clean-fraction measurement: in thinking mode a stream counts only
    if it CLOSED its <think> block AND has >= min_chars of post-think text; unclosed (no </think>) and
    empty-post-think streams are EXCLUDED and counted separately (they are not word-free failures). Non-think
    mode: every stream is valid. Returns (valid_streams, n_unclosed, n_empty). Single source of truth so the
    unit test asserts against the production accounting, not a re-implementation of it."""
    if not think:
        return list(streams), 0, 0
    valid = [s for s in streams if s["has_think"] and len(s["text"].strip()) >= min_chars]
    n_unclosed = sum(1 for s in streams if not s["has_think"])
    n_empty = sum(1 for s in streams if s["has_think"] and len(s["text"].strip()) < min_chars)
    return valid, n_unclosed, n_empty


@torch.no_grad()
def measure_one(model, tok, system, think, n, tokens, batch):
    """Generate n word-free streams under `system` (thinking on/off) and measure clean%. For thinking mode,
    measure the POST-</think> gibberish only; unclosed/empty are excluded (NOT scored as gibberish)."""
    ids = K.chat_ids(tok, PROMPT, system=system, think=think)
    plen = ids.shape[1]
    max_new = tokens if not think else max(tokens, 2048)   # Qwen3 reasons at length before the gibberish
    streams = []
    while len(streams) < n:
        b = min(batch, n - len(streams))
        rep = ids.repeat(b, 1)
        gen = model.generate(rep, attention_mask=torch.ones_like(rep), max_new_tokens=max_new,
                             do_sample=True, temperature=1.0, top_p=0.98, pad_token_id=tok.eos_token_id)
        for r in range(b):
            row = gen[r, plen:]
            eos = (row == tok.eos_token_id).nonzero()
            if len(eos):
                row = row[:int(eos[0]) + 1]
            full = tok.decode(row, skip_special_tokens=True)
            if think:
                tb, sep, after = full.partition("</think>")
                think_text, has_think, text = tb, bool(sep), (after if sep else full)
            else:
                think_text, has_think, text = "", False, full
            dg = degeneracy(text)
            streams.append(dict(text=text, think=think_text, has_think=has_think,
                                deg=dg, accepted=not is_degenerate(dg)))
        # progress per batch so the box log keeps growing -- a silent 2048-token think cell would otherwise
        # trip labkit's stall watchdog (the v1 sweep bug: stalled entering the first think cell)
        print(f"  gen {len(streams)}/{n} ({'think' if think else 'nothink'})", flush=True)
    valid, n_unclosed, n_empty = valid_subset_accounting(streams, think=think)
    acc = sum(s["accepted"] for s in valid)
    why = collections.Counter()
    for s in valid:
        if s["accepted"]:
            continue
        dg = s["deg"]
        for k, thr in (("word_rate", 0.1), ("repetition", 0.6), ("non_latin", 0.3), ("spacing", 0.5)):
            if dg[k] > thr:
                why[k] += 1
    r = dict(thinking=think, n=len(streams), n_valid=len(valid),
             clean_frac=(round(acc / len(valid), 3) if valid else None),
             reject_reasons=dict(why),
             samples=[s["text"][:140] for s in valid[:3]])
    if think:
        r.update(n_unclosed_think=n_unclosed, n_empty_post_think=n_empty,
                 has_think_frac=round(sum(s["has_think"] for s in streams) / len(streams), 3),
                 median_post_think_chars=(int(np.median([len(s["text"]) for s in valid])) if valid else 0),
                 think_samples=[s["think"][:300] for s in streams[:3]])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="orig", help="comma list from " + ",".join(PROMPT_VARIANTS))
    ap.add_argument("--modes", default="nothink", help="comma list: nothink,think (think is a no-op on Qwen2.5)")
    ap.add_argument("--n", type=int, default=64, help="streams per (variant,mode) cell")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    modes = [m for m in args.modes.split(",") if m]
    assert all(v in PROMPT_VARIANTS for v in variants), f"unknown variant in {variants}; have {list(PROMPT_VARIANTS)}"
    assert all(m in ("nothink", "think") for m in modes), f"modes must be nothink/think, got {modes}"

    model, tok = K.load_model(C.MODEL)
    print("MODEL_READY", flush=True)
    C.ensure_run_dirs()
    print(f"SWEEP {C.ACTIVE} ({C.MODEL}): variants={variants} modes={modes} n={args.n} tokens={args.tokens}", flush=True)

    t0 = time.perf_counter()
    cells = []
    for mode in modes:
        think = (mode == "think")
        for vname in variants:
            r = measure_one(model, tok, PROMPT_VARIANTS[vname], think, args.n, args.tokens, args.batch)
            r["variant"] = vname
            cells.append(r)
            extra = (f" has_think={r['has_think_frac']} post_chars={r['median_post_think_chars']}"
                     if think else "")
            print(f"  [{vname:>10} | {mode:>7}] clean={r['clean_frac']} (n_valid={r['n_valid']}) "
                  f"reject={r['reject_reasons']}{extra}", flush=True)
    out = dict(model=C.ACTIVE, hf_id=C.MODEL, tokens=args.tokens, n=args.n,
               sweep_seconds=round(time.perf_counter() - t0, 1), cells=cells)
    with open(C.RESULTS / "baseline_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSUMMARY (clean_frac by variant x mode):", flush=True)
    for c in cells:
        print(f"  {c['variant']:>10} {('think' if c['thinking'] else 'nothink'):>7}: {c['clean_frac']}", flush=True)
    print("BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
