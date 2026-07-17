"""exp3 feasibility smoke: does an EVOKED concept persona + the anti-word prompt still yield WORD-FREE output?

The concern: a rich concept persona might override "OUTPUT ONLY RANDOM LETTERS" and produce prose, killing the
primary arm. This checks it on the already-cached Qwen2.5-1.5B-Instruct, tiny + bounded (no download, no GPU
spend). For a few concepts x {evoked, secret_word, neutral} it generates a handful of short streams under the
EXACT exp1 harness (same STRONG_SYSTEM, same GEN_PROMPT user turn, same temp/top_p, same word-free filter) and
reports the word-free ACCEPTANCE rate + sample text per condition.

CAVEAT: word-free acceptance here is only meaningful with `wordfreq` installed (else the word-rate filter is
inert and this checks non-degeneracy only, NOT real-word content). Install wordfreq before trusting the rates.

Run (backgrounded):  .venv/bin/python experiments/exp3_induction_and_scale/feasibility_smoke.py
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

import config as C                                   # noqa: E402  STRONG_SYSTEM, GEN_PROMPT
import common as K                                   # noqa: E402  chat_ids
from covert_collect import degeneracy, is_degenerate, HAVE_WORDFREQ  # noqa: E402  the word-free filter
import primers as P                                  # noqa: E402  compose_system (evoked / secret_word / neutral)

HF_ID = "Qwen/Qwen2.5-1.5B-Instruct"                 # already cached; no download
CONCEPTS = ["ocean", "fear", "debugging", "silence"] # object / affect / technical / loop-prone
CONDS = ["evoked", "secret_word", "neutral"]         # neutral = baseline (no concept)
N = 6                                                # streams per (concept, condition)
TOKENS = 48

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.bfloat16
print(f"device={DEVICE} dtype={DTYPE} wordfreq={HAVE_WORDFREQ} "
      f"(word_rate filter is inert without wordfreq!)", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(HF_ID)
model = AutoModelForCausalLM.from_pretrained(HF_ID, torch_dtype=DTYPE).to(DEVICE).eval()
print("model loaded", flush=True)


def system_for(concept, cond):
    if cond == "neutral":
        return P.compose_system(None, C.STRONG_SYSTEM, arm="evoked")   # matched neutral persona + anti-word
    return P.compose_system(concept, C.STRONG_SYSTEM, arm=cond)


@torch.no_grad()
def run(concept, cond):
    system = system_for(concept, cond)
    ids = K.chat_ids(tok, C.GEN_PROMPT, system=system, device=DEVICE)
    plen = ids.shape[1]
    rep = ids.repeat(N, 1)
    gout = model.generate(rep, attention_mask=torch.ones_like(rep), max_new_tokens=TOKENS,
                          do_sample=True, temperature=1.0, top_p=0.98, pad_token_id=tok.eos_token_id)
    texts, accepts = [], []
    for r in range(N):
        row = gout[r, plen:]
        eos = (row == tok.eos_token_id).nonzero()
        rlen = int(eos[0]) + 1 if len(eos) else int(row.shape[0])
        text = tok.decode(row[:rlen].cpu(), skip_special_tokens=True)
        acc = not is_degenerate(degeneracy(text))
        texts.append(text); accepts.append(acc)
    return accepts, texts


print("\n=== word-free acceptance (accepted / N) + a sample text per condition ===", flush=True)
for concept in CONCEPTS:
    print(f"\n--- {concept} ---", flush=True)
    for cond in CONDS:
        accepts, texts = run(concept, cond)
        rate = sum(accepts) / len(accepts)
        # prefer showing an ACCEPTED sample if any, else the first (to see what failed)
        i = next((j for j, a in enumerate(accepts) if a), 0)
        samp = texts[i].replace("\n", " ")[:110]
        tag = "acc" if accepts[i] else "REJ"
        print(f"  [{cond:11s}] word-free {sum(accepts)}/{len(accepts)}  e.g.({tag}): {samp!r}", flush=True)

print("\nDONE", flush=True)
