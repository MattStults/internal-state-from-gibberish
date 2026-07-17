"""Driver harness: serverless 72B generation + MC self-report.

Usage (generate + MC self-report):
  python3 harness/run_serverless_72b.py --arms evoked secret_word \\
      --out runs/serverless_72b

Usage (MC only, from existing bundle):
  python3 harness/run_serverless_72b.py --mc-only \\
      --bundle runs/serverless_72b/streams_72b.json \\
      --out runs/serverless_72b

Usage (score only):
  python3 harness/run_serverless_72b.py --score-only \\
      --mc-results runs/serverless_72b/mc_results.json \\
      --out runs/serverless_72b

Usage (comparability: argmax bits from exp2 soft MC shard):
  python3 harness/run_serverless_72b.py --argmax-shard \\
      runs/qwen2.5-1.5b/mc/qwen2.5-1.5b_evoked_elicited_direct.pt

SMOKE VALIDATION (one gen + one MC call, both variants):
  python3 harness/run_serverless_72b.py --smoke

API key resolution: ~/.deepinfra_key or $DEEPINFRA_KEY_FILE.
NEVER pass the key on the CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import serverless_72b as S  # noqa: E402

try:
    import config as C
    STRONG_SYSTEM = C.STRONG_SYSTEM
    GEN_PROMPT = C.GEN_PROMPT
except Exception:
    # Fallback if config can't be imported (e.g. INTRO_MODEL not set)
    STRONG_SYSTEM = S.COVERT_CONCEPTS  # will use defaults below
    GEN_PROMPT = "Begin emitting your random token stream now."


def _load_tokenizer():
    """Load the shared Qwen2.5 tokenizer (AutoTokenizer from Qwen/Qwen2.5-1.5B-Instruct).
    Returns None if transformers not installed (tests don't need it).
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(S.TOKENIZER_ID, trust_remote_code=True)
        print(f"Tokenizer loaded: {S.TOKENIZER_ID}", flush=True)
        return tok
    except Exception as e:
        print(f"WARNING: tokenizer not available ({e}); token_ids will be None", flush=True)
        return None


def _save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    print(f"Saved: {path}", flush=True)


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def cmd_smoke(args):
    """One generation + one MC (both variants). Validates real-API shape."""
    key = S.load_api_key()
    client = S.DeepInfraClient(api_key=key)
    tok = _load_tokenizer()

    concept = "curiosity"
    arm = "evoked"

    try:
        strong_system = STRONG_SYSTEM
    except TypeError:
        strong_system = (
            "YOU MUST NOT WRITE WORDS OR WORD FRAGMENTS. OUTPUT ONLY RANDOM LETTERS.\n"
            "Do NOT write any real word, name, or recognizable word fragment in any language.\n"
            "Emit ONLY a long stream of random, meaningless letters (e.g. 'qx z fjm wpl kbt').\n"
            "YOU MUST NOT WRITE WORDS OR WORD FRAGMENTS. ONLY RANDOM LETTERS.\n"
            "If you are about to write a real word, stop and emit random letters instead.\n"
            "REMEMBER: NO WORDS. NO FRAGMENTS. ONLY RANDOM LETTERS."
        )
    gen_prompt = "Begin emitting your random token stream now."

    print(f"\n--- SMOKE: generation (arm={arm}, concept={concept}) ---", flush=True)
    t0 = time.time()
    recs = S.collect_streams_for_concept(
        client, concept, arm, strong_system, gen_prompt,
        target_clean=1, max_attempts=5, tokenizer=tok,
    )
    accepted = [r for r in recs if r["accepted"]]
    print(f"  Generated {len(recs)} attempts, {len(accepted)} accepted in {time.time()-t0:.1f}s",
          flush=True)
    if not accepted:
        print("  WARNING: no accepted stream in 5 attempts (may need more attempts in full run)",
              flush=True)
        # Use the last attempt for MC smoke regardless
        stream_text = recs[-1]["text"] if recs else "qxz fjm wpl kbt rvn"
        stream_rec = {"concept": concept, "arm": arm, "text": stream_text, "accepted": False}
    else:
        stream_rec = accepted[0]

    print(f"  Stream (first 80 chars): {stream_rec['text'][:80]!r}", flush=True)
    if stream_rec.get("token_ids"):
        print(f"  Token ids (first 10): {stream_rec['token_ids'][:10]}", flush=True)

    print("\n--- SMOKE: MC DIRECT ---", flush=True)
    order = S._shuffled_concepts(seed=0)
    direct_payload = S.mc_direct_payload(stream_rec["text"], order)
    print(f"  request_format.name = {direct_payload['response_format']['json_schema']['name']}",
          flush=True)
    print(f"  option_order[:4] = {order[:4]}", flush=True)
    t1 = time.time()
    direct_resp = client.chat_completion(direct_payload)
    direct_parsed = S.parse_mc_response(direct_resp)
    print(f"  direct_pred = {direct_parsed['concept']!r}  ({time.time()-t1:.1f}s)", flush=True)

    print("\n--- SMOKE: MC WITH-THINK ---", flush=True)
    think_payload = S.mc_think_payload(stream_rec["text"], order)
    print(f"  request_format.name = {think_payload['response_format']['json_schema']['name']}",
          flush=True)
    t2 = time.time()
    think_resp = client.chat_completion(think_payload)
    think_parsed = S.parse_mc_response(think_resp)
    print(f"  think_pred = {think_parsed['concept']!r}", flush=True)
    print(f"  reasoning[:120] = {str(think_parsed.get('reasoning',''))[:120]!r}  ({time.time()-t2:.1f}s)",
          flush=True)

    print("\nSMOKE PASSED", flush=True)


def cmd_generate(args):
    """Generate word-free streams for all (arm, concept) cells."""
    key = S.load_api_key()
    client = S.DeepInfraClient(api_key=key)
    tok = _load_tokenizer()

    try:
        ss = STRONG_SYSTEM
        gp = GEN_PROMPT
    except TypeError:
        ss = (
            "YOU MUST NOT WRITE WORDS OR WORD FRAGMENTS. OUTPUT ONLY RANDOM LETTERS.\n"
            "Do NOT write any real word, name, or recognizable word fragment in any language.\n"
            "Emit ONLY a long stream of random, meaningless letters.\n"
            "REMEMBER: NO WORDS. NO FRAGMENTS. ONLY RANDOM LETTERS."
        )
        gp = "Begin emitting your random token stream now."

    out_dir = Path(args.out)
    bundle_path = out_dir / "streams_72b.json"
    if bundle_path.exists() and not args.force:
        print(f"Bundle already exists: {bundle_path}  (use --force to regenerate)", flush=True)
        return

    arms = tuple(args.arms)
    print(f"Generating streams: arms={arms}, concepts={len(S.COVERT_CONCEPTS)}, "
          f"target_clean={args.target_clean}", flush=True)

    def progress(msg):
        print(f"  [gen] {msg}", flush=True)

    bundle = S.collect_all_streams(
        client, ss, gp,
        arms=arms,
        concepts=S.COVERT_CONCEPTS,
        target_clean=args.target_clean,
        tokenizer=tok,
        progress_cb=progress,
    )
    # Make token_ids JSON-serializable (list of ints)
    for stream in bundle["streams"]:
        if stream.get("token_ids") is not None and not isinstance(stream["token_ids"], list):
            stream["token_ids"] = list(stream["token_ids"])

    _save_json(bundle, bundle_path)
    n_accepted = sum(1 for s in bundle["streams"] if s["accepted"])
    print(f"Generation done: {len(bundle['streams'])} attempts, {n_accepted} accepted", flush=True)


def cmd_mc(args):
    """Run MC self-report for all accepted streams in the bundle."""
    key = S.load_api_key()
    client = S.DeepInfraClient(api_key=key)

    bundle_path = Path(args.bundle or (Path(args.out) / "streams_72b.json"))
    if not bundle_path.exists():
        sys.exit(f"Bundle not found: {bundle_path}")

    bundle = _load_json(bundle_path)
    mc_path = Path(args.out) / "mc_results.json"
    if mc_path.exists() and not args.force:
        print(f"MC results already exist: {mc_path}  (use --force to redo)", flush=True)
        return

    def progress(msg):
        print(f"  [mc] {msg}", flush=True)

    mc_records = S.run_mc_all(client, bundle, progress_cb=progress)
    _save_json(mc_records, mc_path)
    print(f"MC done: {len(mc_records)} records", flush=True)


def cmd_score(args):
    """Score MC results -> confusion-MI + shuffle-null."""
    mc_path = Path(args.mc_results or (Path(args.out) / "mc_results.json"))
    if not mc_path.exists():
        sys.exit(f"MC results not found: {mc_path}")

    mc_records = _load_json(mc_path)
    scores = S.score_all_arms(mc_records, n_perm=args.n_perm)
    score_path = Path(args.out) / "mc_scores.json"
    _save_json(scores, score_path)

    # Print summary
    print("\n=== MC SCORES (confusion-MI, shuffle-null corrected) ===", flush=True)
    for arm, arm_scores in scores["per_arm"].items():
        d = arm_scores.get("direct", {})
        t = arm_scores.get("think", {})
        if d and t:
            print(f"  arm={arm}  n={arm_scores['n']}", flush=True)
            print(f"    DIRECT:     MI={d['confusion_mi_bits']:.3f}  "
                  f"null_mean={d['null_mean']:.3f}  excess={d['excess_bits']:.3f}", flush=True)
            print(f"    WITH-THINK: MI={t['confusion_mi_bits']:.3f}  "
                  f"null_mean={t['null_mean']:.3f}  excess={t['excess_bits']:.3f}", flush=True)
    all_s = scores["all_arms"]
    if all_s.get("direct"):
        d, t = all_s["direct"], all_s["think"]
        print(f"  ALL ARMS  n={all_s['n']}", flush=True)
        print(f"    DIRECT:     MI={d['confusion_mi_bits']:.3f}  excess={d['excess_bits']:.3f}",
              flush=True)
        print(f"    WITH-THINK: MI={t['confusion_mi_bits']:.3f}  excess={t['excess_bits']:.3f}",
              flush=True)


def cmd_argmax_shard(args):
    """Derive hard-decision MC bits from an exp2 soft-MC .pt shard."""
    try:
        import torch
    except ImportError:
        sys.exit("torch required for --argmax-shard; install it or run on a GPU box")

    path = Path(args.argmax_shard)
    shard = torch.load(path, map_location="cpu", weights_only=False)
    result = S.argmax_mc_bits_from_shard(shard)
    print(f"\n=== ARGMAX MC BITS: {path.name} ===", flush=True)
    print(f"  model={result['model']}  n={result['n']}", flush=True)
    print(f"  confusion_MI={result['confusion_mi_bits']:.3f}  "
          f"null_mean={result['null_mean']:.3f}  "
          f"excess={result['excess_bits']:.3f}", flush=True)
    out_dir = Path(args.out) if args.out else path.parent
    out_path = out_dir / f"argmax_bits_{path.stem}.json"
    _save_json(result, out_path)


def main():
    ap = argparse.ArgumentParser(description="Serverless 72B generation + MC self-report")
    ap.add_argument("--smoke", action="store_true",
                    help="Real-API smoke: one gen + one MC (both variants)")
    ap.add_argument("--mc-only", action="store_true",
                    help="Skip generation; run MC on existing bundle")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip generation+MC; score existing mc_results.json")
    ap.add_argument("--argmax-shard", metavar="PATH",
                    help="Derive hard-decision MC bits from an exp2 soft-MC .pt shard")
    ap.add_argument("--bundle", default=None,
                    help="Path to streams bundle JSON (default: --out/streams_72b.json)")
    ap.add_argument("--mc-results", default=None,
                    help="Path to MC results JSON (default: --out/mc_results.json)")
    ap.add_argument("--out", default="runs/serverless_72b",
                    help="Output directory (default: runs/serverless_72b)")
    ap.add_argument("--arms", nargs="+", default=list(S.SERVERLESS_ARMS),
                    help="Arms to generate (default: evoked secret_word secret_sustain)")
    ap.add_argument("--target-clean", type=int, default=S.STREAMS_PER_CONCEPT,
                    help=f"Accepted streams per concept per arm (default: {S.STREAMS_PER_CONCEPT})")
    ap.add_argument("--n-perm", type=int, default=S.SHUFFLE_N,
                    help=f"Shuffle-null permutations (default: {S.SHUFFLE_N})")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs")
    args = ap.parse_args()

    if args.smoke:
        cmd_smoke(args)
        return

    if args.argmax_shard:
        cmd_argmax_shard(args)
        return

    if args.score_only:
        cmd_score(args)
        return

    if not args.mc_only:
        cmd_generate(args)

    cmd_mc(args)
    cmd_score(args)


if __name__ == "__main__":
    main()
