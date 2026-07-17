"""OPTIONAL GPU cross-check for the re-evocation experiment. NOT required to run analyze_reevocation.py
(that recovers the exact injected direction offline from armA-armB). This (a) re-derives the 12
blog-faithful concept vectors and SAVES them (per the save-steering-primitives rule, for future runs),
and (b) lets us confirm the blog method reproduces the injected direction:
    cos( normalize(armA-armB from covert_collect.pt) , concept_vector_blog ) should be >= 0.99.
Run on GPU:  python3 derive_vectors.py
"""
import json
import numpy as np
import torch
import config as C
import common as K

def main():
    model, tok = K.load_model(C.MODEL)
    C.ensure_run_dirs()
    layer = C.resolve_layer(model.config.num_hidden_layers)
    concepts = C.COVERT_CONCEPTS
    vectors, norms = {}, {}
    for c in concepts:
        base = [w for w in C.BASELINE_WORDS if w.lower() != c.lower()]
        v = K.concept_vector_blog(model, tok, c, base, layer)        # deterministic, no sampling
        v = v.float().cpu().numpy()
        vectors[c] = v
        norms[c] = float(np.linalg.norm(v))
        print(f"  {c:12s} ||v||={norms[c]:.3f}", flush=True)
    torch.save({"vectors": vectors, "norms": norms, "layer": layer, "model": C.MODEL,
                "baseline_words": C.BASELINE_WORDS}, C.DATA / "concept_vectors.pt")
    print("wrote concept_vectors.pt", flush=True)

    # optional on-box cross-check if the collected data is present
    try:
        d = torch.load(C.DATA / "covert_collect.pt", map_location="cpu", weights_only=False)
        S = d["streams"]
        for ci, c in enumerate(concepts):
            for r in S:
                if r["concept_idx"] == ci and r["strength"] == 60 and r["accepted"]:
                    a = d["acts"].get(("A", r["gidx"]), {}).get(8)
                    b = d["acts"].get(("B", r["gidx"]), {}).get(8)
                    if a is not None and b is not None:
                        diff = (np.asarray(a, np.float64) - np.asarray(b, np.float64))
                        vb = vectors[c].astype(np.float64)
                        cos = float(diff @ vb / (np.linalg.norm(diff) * np.linalg.norm(vb)))
                        print(f"  cos(armA-armB, blog v) [{c}] = {cos:+.4f}  (expect >= 0.99)", flush=True)
                        break
    except FileNotFoundError:
        print("(covert_collect.pt not on box -> skip cross-check; do it offline)", flush=True)
    print("DERIVE_VECTORS_DONE", flush=True)

if __name__ == "__main__":
    main()
