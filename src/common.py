"""Model loading, concept-vector extraction, residual-stream injection, activation reads.

Reads and injection both happen at the OUTPUT of decoder layer `layer` (via a forward
hook on model.model.layers[layer]) so the two are at exactly the same point in the stack.
"""
import torch


def load_model(name, dtype=torch.bfloat16, device="cuda"):
    # lazy import: the CPU analysis venv has no transformers; only the GPU collect path calls this.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device)
    else:  # cpu/mps: plain load + .to (no accelerate/device_map needed locally)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device)
    model.eval()
    return model, tok


def chat_ids(tok, user, system=None, add_generation_prompt=True, device="cuda", think=False):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    # Qwen3 defaults to "thinking" mode -> the model emits a <think></think> block before its answer,
    # which contaminates the word-free gibberish (the literal "think" trips the word-rate filter).
    # enable_thinking=False (default) moves the (empty) block into the PROMPT so the generation is pure
    # gibberish. think=True keeps native thinking ON (Qwen3) -- to test whether the FORCED non-thinking mode
    # is what drives the repetition looping (the caller must split </think> off the generated text).
    # Harmless for Qwen2.5 (template ignores the kwarg); fall back if a tokenizer rejects it.
    kw = dict(add_generation_prompt=add_generation_prompt, return_tensors="pt")
    try:
        out = tok.apply_chat_template(msgs, enable_thinking=think, **kw)
    except (TypeError, ValueError):
        out = tok.apply_chat_template(msgs, **kw)
    # newer transformers return a BatchEncoding dict here, not a bare tensor
    ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
    return ids.to(device)


class Capture:
    """Context manager capturing the output activations of model.model.layers[layer]."""
    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self.acts, self._h = [], None

    def __enter__(self):
        def hook(_m, _a, out):
            hs = out[0] if isinstance(out, tuple) else out
            self.acts.append(hs.detach())
        self._h = self.model.model.layers[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._h.remove()


def _injection_hook(vector, alpha, prompt_len=None, prompt_only=False):
    """Add alpha*vector at the layer output. prompt_len=None injects ALL positions (legacy);
    an int injects only GENERATED positions (index >= prompt_len) and leaves the prompt prefill
    clean -- the open-introspection post-02 convention (all-position injection steers the prompt
    and the prefill probe, inflating downstream readouts). prompt_only=True (requires prompt_len)
    inverts the mask: inject ONLY the prompt prefill and leave every generated position clean --
    a STATIC perturbation with persona-like persistence (confound-closing prereg E3: injection
    provenance, decaying influence, dose knob)."""
    if prompt_only and prompt_len is None:
        raise ValueError("prompt_only injection requires prompt_len")
    def hook(_m, _a, out):
        is_tuple = isinstance(out, tuple)
        hs = out[0] if is_tuple else out
        delta = alpha * vector.to(hs.dtype).to(hs.device)
        if prompt_only:
            L = hs.shape[1]
            if L > 1:                        # prefill (or full re-forward): prompt positions only
                hs = hs.clone()
                hs[:, :min(prompt_len, L), :] += delta
            # else: KV-cached decode step -> generated position -> leave clean
        elif prompt_len is None:
            hs = hs + delta
        else:
            L = hs.shape[1]
            if L == 1:                       # KV-cached decode step -> a generated position
                hs = hs + delta
            elif L > prompt_len:             # full forward over [prompt + generated]
                hs = hs.clone()
                hs[:, prompt_len:, :] += delta
            # else: pure prompt prefill (L <= prompt_len) -> leave clean
        return (hs,) + tuple(out[1:]) if is_tuple else hs
    return hook


@torch.no_grad()
def _last_token_act(model, tok, text, layer, device="cuda"):
    """Residual activation at the LAST token of a RAW (no chat template) prompt, at `layer`."""
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    with Capture(model, layer) as cap:
        model(ids)
    return cap.acts[-1][0, -1, :]


@torch.no_grad()
def concept_vector_blog(model, tok, concept_word, baseline_words, layer,
                        template="Tell me about {}.", device="cuda"):
    """Blog-faithful extraction (ostegm/open-introspection): RAW prompt 'Tell me about {word}.',
    LAST-token residual activation at `layer`, target minus the mean last-token activation over many
    baseline words. Un-normalized. Much larger-norm + cleaner direction than the mean-pooled,
    chat-templated diff-of-means -- which gave an inert/disruptive vector with no introspection window."""
    target = _last_token_act(model, tok, template.format(concept_word), layer, device)
    base = torch.stack([_last_token_act(model, tok, template.format(w), layer, device)
                        for w in baseline_words]).mean(0)
    return target - base
