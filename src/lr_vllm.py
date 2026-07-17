"""vLLM prompt_logprobs LR scorer for the LR-72B run (prereg: experiments/exp2_output_monitorability/
reports/lr_72b_prereg.md).

KEY DIFFERENCE from every prior LR box: those load an HF model directly and teacher-force with our
OWN forward pass (lr_reader.score_batch + the certified float32 log-softmax in ll_from_logits).
The 72B point is too big to teacher-force that way on the tier we can afford, so we self-host it
under vLLM (`vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 2`) and teacher-force via
the server's native `prompt_logprobs` feature: vLLM returns, for EVERY prompt token, the logprob of
that provided token under the preceding prompt tokens. Summing those over the gibberish-stream span
under each context IS the exact LR quantity `Sum_t log P(stream_t | ctx, stream_<t>)` -- the same
currency as the scale-grid, computed by vLLM rather than by us. This module ONLY renders prompts,
aligns the returned logprobs to the gibberish span, and sums+subtracts; it reimplements NO softmax
numerics (guarded by test V8).

CORRECTNESS POINTS (each pinned by test_lr_vllm.py, verifiable against a MOCK response, no GPU):
  (a) INDEX ALIGNMENT: vLLM's `prompt_logprobs` is a list, one entry per prompt token; entry i is
      the distribution over token i GIVEN tokens < i, so entry 0 is null (the first token has no
      preceding context). We align the GIBBERISH span [start:end) -- the assistant tokens after the
      generation-header context prefix -- to those indices exactly, taking prompt_logprobs[i] for
      each i in the span. (V1/V2.)
  (b) PROVIDED TOKEN, NOT ARGMAX: each prompt_logprobs[i] is a dict keyed by token-id-string; we
      read the entry for the ACTUAL provided token id (the one we teacher-forced at position i),
      NOT the rank-1 argmax. A provided id missing from entry i is terminal (raise): it means the
      teacher-forced token was outside the returned top-logprobs set, so the sum would be wrong or
      the alignment is off. (V2/V3.)
  (c) TOKENIZER PARITY: the prompt must be tokenized identically to how vLLM tokenizes it. We send
      TOKEN IDS (completions_request's prompt = the id list), so vLLM scores exactly the ids we
      rendered -- no server-side re-tokenization to drift against. assert_prompt_roundtrips is the
      belt-and-suspenders check that the stream ids round-trip under the local tokenizer (the same
      decode->re-encode identity lr_grid's Llama path uses). (V6/V7.)

The box script (box_lr_72b.py) OWNS the ready/done/fatal markers ("LR72" + those suffixes); those
marker strings must NOT appear anywhere in this file (labkit substring-matches markers on log
lines; the LR attempt-4 collision) -- guarded by test V8. Run on the box via the orchestrator; the
scoring itself is cheap once the server is up.
"""

# --------------------------------------------------------------- prompt rendering + span index
def render_prompt_ids(tok, system, gen_prompt, stream_ids):
    """Render the teacher-forcing prompt as a TOKEN ID list and return (prompt_ids, (start, end)),
    where [start, end) indexes the gibberish stream tokens inside prompt_ids.

    The context prefix is the reader's own chat template over (system = persona/secret context,
    user = gen_prompt) WITH the generation header on -- byte/token-identical to how the stream was
    generated (K.chat_ids at collection). The gibberish then rides as the assistant turn. Qwen2.5's
    template is prefix-stable (the generation-header render is a strict token prefix of the full
    [system, user, assistant] render), so the stream span is exactly the context-prefix length
    onward -- the same prefix-stability lr_grid.assert_llama_prefix_stable relies on. We build the
    prompt by CONCATENATING the context prefix ids and the provided stream ids so the span is
    unambiguous and the stream tokens are the exact saved/re-encoded ids (no re-tokenization of a
    decoded assistant turn that a template might merge across the boundary)."""
    ctx_ids = list(tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": gen_prompt}],
        add_generation_prompt=True))
    start = len(ctx_ids)
    stream_ids = [int(x) for x in stream_ids]
    prompt_ids = ctx_ids + stream_ids
    return prompt_ids, (start, start + len(stream_ids))


# --------------------------------------------------------------- prompt_logprobs alignment
def _entry_logprob(entry, tok_id):
    """The logprob of a SPECIFIC token id from one prompt_logprobs entry (a dict keyed by
    token-id-string). vLLM keys are strings; the value is either a float logprob or a small dict
    carrying {'logprob': ...} (the OpenAI-style Logprob object serialization). None if the provided
    id is absent from this entry -- the caller treats that as terminal."""
    if entry is None:
        return None
    v = entry.get(str(tok_id), entry.get(int(tok_id)) if not isinstance(tok_id, str) else None)
    if v is None:
        return None
    if isinstance(v, dict):
        return float(v["logprob"])
    return float(v)


def span_logprobs(prompt_logprobs, prompt_ids, span):
    """Per-token logprobs of the ACTUAL provided tokens over the gibberish span [start, end).

    prompt_logprobs[i] is P(prompt token i | prompt tokens < i); index 0 is null. For each position
    i in the span we read the entry for prompt_ids[i] -- the token WE teacher-forced -- NOT the
    argmax. A provided id absent from prompt_logprobs[i] is terminal: the teacher-forced token fell
    outside vLLM's returned logprobs at that position (the request must use a prompt_logprobs value
    large enough to always include the provided token; the box pins that), so summing would silently
    score the wrong quantity. The message is marker/FATAL-substring safe (test V3/V6)."""
    start, end = span
    if len(prompt_logprobs) != len(prompt_ids):
        raise RuntimeError(
            f"prompt_logprobs length {len(prompt_logprobs)} != prompt length {len(prompt_ids)} -- "
            "the returned per-prompt-token logprobs do not align to the sent prompt; refusing to "
            "score a misaligned span")
    if start < 1:
        # index 0 is null (no preceding context); a stream span must begin after the context prefix
        raise RuntimeError(
            f"gibberish span starts at index {start} -- it must begin after a non-empty context "
            "prefix (prompt_logprobs[0] is null: the first prompt token has no logprob)")
    out = []
    for i in range(start, end):
        lp = _entry_logprob(prompt_logprobs[i], prompt_ids[i])
        if lp is None:
            raise RuntimeError(
                f"provided token id {prompt_ids[i]} absent from prompt_logprobs at position {i} "
                "(within the gibberish span) -- the teacher-forced token was outside the returned "
                "top-logprobs set; raise the request's prompt_logprobs value and retry")
        out.append(lp)
    return out


def ll_over_span(span_lps, drop_last_eos=False, span_ids=None, eos_id=None):
    """Sum the span logprobs -> one log-likelihood scalar. drop_last_eos (the registered eos-free
    PRIMARY, mirroring lr_grid.noeos_lens): if the stream's LAST token is the tokenizer eos, that
    final position is excluded from the sum (the 1.5B run's flagged artifact -- eos probability under
    persona contexts is context-correlated, not concept evidence). span_ids/eos_id gate the drop."""
    lps = list(span_lps)
    if drop_last_eos and eos_id is not None and span_ids is not None:
        ids = [int(x) for x in span_ids]
        if ids and ids[-1] == int(eos_id) and lps:
            lps = lps[:-1]
    return float(sum(lps))


# --------------------------------------------------------------- LR = LL(ctx) - LL(neutral)
def _span_lps(client, tok, system, gen_prompt, stream_ids, prompt_logprobs=1, **client_kw):
    """The per-token span logprobs of one gibberish stream under ONE context (system). Renders the
    prompt ids, POSTs them (token ids -> no server re-tokenization), aligns the returned logprobs
    to the stream span. `client.completions(prompt_ids, prompt_logprobs=..)` must return a dict with
    a 'prompt_logprobs' list; the transport is injectable so tests use a mock."""
    prompt_ids, span = render_prompt_ids(tok, system, gen_prompt, stream_ids)
    resp = client.completions(prompt_ids, prompt_logprobs=prompt_logprobs, **client_kw)
    plps = resp["prompt_logprobs"] if isinstance(resp, dict) else resp
    return span_logprobs(plps, prompt_ids, span)


def _score_ll(client, tok, system, gen_prompt, stream_ids, prompt_logprobs=1,
              drop_last_eos=False, eos_id=None, **client_kw):
    """LL of one gibberish stream under ONE context: the span logprobs, summed (eos-free primary)."""
    lps = _span_lps(client, tok, system, gen_prompt, stream_ids,
                    prompt_logprobs=prompt_logprobs, **client_kw)
    return ll_over_span(lps, drop_last_eos=drop_last_eos, span_ids=stream_ids, eos_id=eos_id)


def lr_score(client, tok, ctx_system, neutral_system, gen_prompt, stream_ids,
             prompt_logprobs=1, drop_last_eos=False, eos_id=None, return_pertok=False,
             **client_kw):
    """LR = LL(stream | ctx) - LL(stream | neutral) for ONE gibberish stream -- the frozen scale-grid
    currency, teacher-forced over the SAME stream span under both contexts. Two prompt_logprobs
    calls (numerator, denominator); the difference cancels the length-scale, isolating the persona.
    Returns the LR scalar (bits are the offline calibration's job; this is the raw nat-scale LL
    difference the offline scorer feeds through the certified readout).

    return_pertok=True additionally returns the PER-TOKEN LR difference vector over the (eos-free)
    span -- pertok[t] = logP(stream_t | ctx, stream_<t>) - logP(stream_t | neutral, stream_<t>) --
    which the offline Amendment-5 position-lift control consumes (matched vs mismatched early-token
    share). The summed LR equals the vector's sum (same eos rule), so the two are consistent."""
    ctx_lps = _span_lps(client, tok, ctx_system, gen_prompt, stream_ids,
                        prompt_logprobs=prompt_logprobs, **client_kw)
    neu_lps = _span_lps(client, tok, neutral_system, gen_prompt, stream_ids,
                        prompt_logprobs=prompt_logprobs, **client_kw)
    ll_ctx = ll_over_span(ctx_lps, drop_last_eos=drop_last_eos, span_ids=stream_ids, eos_id=eos_id)
    ll_neu = ll_over_span(neu_lps, drop_last_eos=drop_last_eos, span_ids=stream_ids, eos_id=eos_id)
    lr = ll_ctx - ll_neu
    if not return_pertok:
        return lr
    n = len(ctx_lps)
    if drop_last_eos and eos_id is not None and stream_ids is not None:
        ids = [int(x) for x in stream_ids]
        if ids and ids[-1] == int(eos_id):
            n -= 1
    pertok = [float(ctx_lps[t] - neu_lps[t]) for t in range(n)]
    return lr, pertok


# --------------------------------------------------------------- tokenizer / round-trip parity
def assert_prompt_roundtrips(tok, stream_id_lists, n=32):
    """Tokenizer-parity gate (mirrors lr_grid's Llama round-trip discipline, applied to the vLLM
    path): decode(stream_ids) then re-encode under the SAME tokenizer must reproduce the stream ids,
    for a sample of streams. Because we SEND token ids to vLLM (completions_request's prompt is the
    id list), the server scores exactly what we rendered -- but a stream whose ids do not round-trip
    would still be a corrupted/ambiguous target, so we refuse it. Terminal (raise) with a
    marker/FATAL-substring-safe message; the box turns the traceback into its fatal marker."""
    for k, ids in enumerate(stream_id_lists[: max(1, int(n))]):
        ids = [int(x) for x in ids]
        text = tok.decode(ids, skip_special_tokens=False)
        re_ids = list(tok(text, add_special_tokens=False).input_ids)
        if re_ids != ids:
            d = next((j for j in range(min(len(ids), len(re_ids))) if ids[j] != re_ids[j]),
                     min(len(ids), len(re_ids)))
            raise RuntimeError(
                f"tokenizer parity gate failed on stream {k}: decode->re-encode ids differ "
                f"(lens {len(ids)} vs {len(re_ids)}, first divergence at token {d}) -- the stream "
                "is not a stable teacher-forcing target under this tokenizer; refusing to score")


# --------------------------------------------------------------- request shape (the POST body)
def completions_request(model, prompt_ids, prompt_logprobs=1, max_tokens=0, temperature=0.0):
    """The pinned vLLM /v1/completions request body for TEACHER-FORCING (not generation):
      - prompt = the TOKEN ID list (tokenizer parity: vLLM scores exactly our ids, no re-tokenize);
      - max_tokens = 0 (score the prompt only, emit no continuation);
      - prompt_logprobs >= 1 (return per-prompt-token logprobs -- the teacher-forcing signal);
      - temperature = 0.0 (deterministic; scoring, not sampling).
    The box POSTs this to http://127.0.0.1:<port>/v1/completions. The value of prompt_logprobs is a
    top-k over the vocabulary; the box sets it high enough that the provided token is always present
    (span_logprobs raises otherwise, and the smoke catches it)."""
    return dict(model=model, prompt=[int(x) for x in prompt_ids], max_tokens=int(max_tokens),
                prompt_logprobs=int(prompt_logprobs), temperature=float(temperature))
