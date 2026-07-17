"""On-box exp2+exp3 analysis entrypoint: PULL the stream bundles + R_emb embedding matrices from HF (the box's
fast net), then run run_induction (exp3) + run_budget (exp2) with checkpointed output into out/.

Why HF-pull: the bundles are already published on HF, so rsync-ing them UP from the laptop's slow home link
was redundant and blew labkit's 600s ship timeout. The box downloads them directly instead. Needs HF_TOKEN in
env (the dataset is private). The reader analysis is heavy CPU (nested-CV) -- never run it on the Mac.

Driven by harness/run_reanalysis.py (gated). Markers: ANALYZE_READY / ANALYZE_DONE / ANALYZE_FATAL.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis"))

# slug -> hf_id for the embed extraction (mirrors src/config.py MODELS; hardcoded so this file is standalone)
EMBED_MODELS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


HF_DATASET = "ErrareHumanumEst/internal-state-from-gibberish"   # stream bundles live here (private -> needs HF_TOKEN)


def _parse_model_arm(name):
    stem = name[:-3]                                         # strip .pt
    for arm in ("evoked_alt", "secret_word", "evoked", "named"):   # longest suffix first (secret_word/evoked_alt)
        if stem.endswith("-" + arm):
            return stem[:-(len(arm) + 1)], arm
    return None, None


def _fetch_bundles_from_hf():
    """Pull the stream bundles straight from HF (the box's fast net) instead of rsync-ing them UP from the
    laptop's slow home link -- the bundles are already published there, so uploading them was redundant and
    blew labkit's 600s rsync ceiling. Needs HF_TOKEN in env (the dataset is private)."""
    import shutil

    from huggingface_hub import hf_hub_download, list_repo_files
    n_ind = n_ab = 0
    for f in list_repo_files(HF_DATASET, repo_type="dataset"):
        if f.startswith("exp3/bundles/") and f.endswith(".pt"):        # exp3 induced arms -> runs/_ind/<m>/data/
            model, arm = _parse_model_arm(os.path.basename(f))
            if not model:
                continue
            dest = os.path.join(REPO, "runs", "_ind", model, "data", f"{model}-{arm}.pt")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(hf_hub_download(HF_DATASET, f, repo_type="dataset"), dest)
            n_ind += 1
        elif f.endswith("-gen.pt") and "/" not in f:                  # exp2 injected gen bundles -> runs/_ab/
            dest = os.path.join(REPO, "runs", "_ab", f)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy(hf_hub_download(HF_DATASET, f, repo_type="dataset"), dest)
            n_ab += 1
    print(f"fetched {n_ind} exp3 + {n_ab} exp2 bundles from HF", flush=True)
    return n_ind, n_ab


def main():
    import numpy as np
    from loader import load_embed_matrix                     # partial safetensors: just embed_tokens.weight

    n_ind, n_ab = _fetch_bundles_from_hf()                   # pull bundles from HF (not rsync'd up from the laptop)
    if not n_ind or not n_ab:                                # refuse to 'succeed' on a paid run with no data
        raise RuntimeError(f"no bundles fetched from HF (_ind={n_ind}, _ab gen={n_ab})")

    emb_dir = os.path.join(REPO, "artifacts")
    os.makedirs(emb_dir, exist_ok=True)
    for slug, hf_id in EMBED_MODELS.items():
        p = os.path.join(emb_dir, f"{slug}_embed.npy")
        if not os.path.exists(p):
            print(f"extracting embed matrix {slug} <- {hf_id}", flush=True)
            np.save(p, load_embed_matrix(hf_id))
    print("ANALYZE_READY", flush=True)

    out = os.environ.setdefault("INTRO_REPORT_DIR", os.path.join(REPO, "out"))   # labkit pulls out/
    os.makedirs(out, exist_ok=True)
    resume_dir = os.path.join(REPO, "runs", "reanalysis_resume")   # shipped already-done bundle JSONs -> RESUME
    if os.path.isdir(resume_dir):
        import shutil
        for f in os.listdir(resume_dir):
            if f.endswith(".json"):
                shutil.copy(os.path.join(resume_dir, f), os.path.join(out, f))
                print(f"seeded resume: {f}", flush=True)
    os.environ["INTRO_EMBED_DIR"] = emb_dir
    os.environ["INTRO_IND_DIR"] = os.path.join(REPO, "runs", "_ind")

    print("\n########## exp3 (prompt-induction) ##########", flush=True)
    import run_induction
    run_induction.main()

    print("\n########## exp2 (injected budget) ##########", flush=True)
    sys.argv = ["run_budget"] + [os.path.join(REPO, "runs", "_ab", f"qwen2.5-{m}-gen.pt")
                                 for m in ("1.5b", "3b", "7b")]
    import run_budget
    run_budget.main()

    for name in ("induction_results.json", "budget_results.json"):
        if not os.path.exists(os.path.join(out, name)):     # never report DONE with no output (paid run!)
            raise RuntimeError(f"analysis produced no {name} in {out}")
    print(f"outputs OK in {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
        print("ANALYZE_DONE", flush=True)
    except Exception:
        print("ANALYZE_FATAL", flush=True)
        raise
