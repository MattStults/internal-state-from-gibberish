"""exp2 loader: pull the strong-dose accepted streams from an _ab covert_collect bundle.

filter_streams is the pure, testable core (filter accepted + strength, normalize tokens); load_ab_streams is
the thin torch.load I/O wrapper (smoke-tested on a real bundle, kept out of the fast unit suite). Each
returned stream carries gen_topk (distribution-access features) and realized token ids (sampled-token
features) + its concept_idx label.
"""
import numpy as np


def filter_streams(bundle, strength=None):
    """bundle: a loaded covert_collect dict. Returns {streams: [{gen_topk, tokens (np.int64), concept_idx}],
    concepts, strength, inject, model} for the ACCEPTED streams at `strength` (default: the strong/max dose)
    that have gen_topk."""
    strong = max(bundle["strengths"]) if strength is None else strength
    streams = []
    for r in bundle["streams"]:
        if r.get("accepted") and r["strength"] == strong and r.get("gen_topk"):
            streams.append(dict(gen_topk=r["gen_topk"],
                                tokens=np.asarray(r["tokens"]).astype(np.int64),
                                concept_idx=int(r["concept_idx"])))
    return dict(streams=streams, concepts=bundle["concepts"], strength=strong,
                inject=bundle.get("inject"), model=bundle.get("model"))


def load_ab_streams(pt_path, strength=None):
    import torch
    return filter_streams(torch.load(pt_path, map_location="cpu", weights_only=False), strength)


def load_embed_matrix(hf_id):
    """Partial safetensors load of ONLY model.embed_tokens.weight (vocab, d) from HF -- downloads just the
    containing shard and extracts one tensor. No full model, no forward pass (so no freeze risk). Returns a
    (vocab, d) float32 numpy array: token id -> embedding, the R_emb featurizer's input."""
    import json

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    key = "model.embed_tokens.weight"
    try:
        idx = json.load(open(hf_hub_download(hf_id, "model.safetensors.index.json")))
        shard = idx["weight_map"][key]
    except Exception:
        shard = "model.safetensors"                       # single-file checkpoint
    with safe_open(hf_hub_download(hf_id, shard), framework="pt") as f:
        return f.get_tensor(key).float().numpy()
