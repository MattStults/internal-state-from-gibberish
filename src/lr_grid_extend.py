"""LR scale-grid reader-registry EXTENSION (prereg: experiments/exp2_output_monitorability/
reports/lr_scale_extend_prereg.md, DRAFT until Matt freezes it).

Registers the qwen2.5-14b / qwen2.5-32b reader slugs (and conservative 80GB-tier batch defaults)
into src/lr_grid.py's GRID_READERS, then DELEGATES to lr_grid.main() -- the certified multi-reader
path runs byte-identically. This module is CONFIGURATION ONLY: it defines no scoring, no
log-softmax, no gather, no forward pass of any kind (guarded by tests/test_lr_extend.py E2), and
it never modifies src/lr_grid.py or src/lr_reader.py on disk -- the same extend-without-modifying
pattern lr_grid itself uses over lr_reader.

hf_ids resolve through the config registry when the slug is registered there (single source of
truth), with the literal Qwen hub ids as fallback so the wrapper also works before a registry
entry lands (the 32b entry is an additive config change flagged in the prereg).

Run on GPU via the box orchestrator (box_lr_extend.py); NEVER on the Mac.
  python3 -u src/lr_grid_extend.py --reader qwen2.5-14b --bundle <gen>:<set>:<path> ...
"""
import config as C
import lr_grid as G

# The extension readers. Registry-first (C.MODELS is the single source of truth for hf ids);
# literal fallback keeps the wrapper importable/testable before the additive registry entry.
EXTEND_READERS = {
    "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
}
# Conservative 80GB-tier batch defaults (weights 29.5 / 65.5 GB bf16 + KV for ~25 contexts x
# batch streams; only need to not-OOM the first shard -- the B8 util gate + --batch re-derive).
EXTEND_BATCH = {"14b": 6, "32b": 4}
# TECH M2 (2026-07-14 review): the 14B KV self-check STRADDLES the 0.02 tolerance across
# processes (max|dLL|/T = 0.02056 and 0.05228 observed), so which scoring path a 14B process
# takes was a per-process coin flip with up-to-tol cell differences. Pin the 14B reader to the
# concat reference path (use_kv=False) in EVERY grid/rider invocation; smaller readers keep
# the certified self-check. Disclosed in shard meta (kv_pinned).
EXTEND_KV_PIN = {"qwen2.5-14b": False}


def register(readers=None, batches=None, kv_pin=None):
    """Additively extend lr_grid's reader registry + per-size batch defaults + the reader-keyed
    KV pin IN MEMORY. Idempotent; never overwrites an existing registration with a different
    hf_id (raise -- a silent hf_id swap on a certified slug must be impossible)."""
    for slug, fallback in (readers or EXTEND_READERS).items():
        reg = C.MODELS.get(slug, {}).get("hf_id")
        if reg is not None and fallback is not None and reg != fallback:
            raise RuntimeError(f"lr_grid_extend: requested hf_id {fallback!r} for {slug} "
                               f"conflicts with the config registry's {reg!r} -- a silent "
                               "hf_id swap on a registered slug must be impossible")
        hf_id = reg or fallback
        prev = G.GRID_READERS.get(slug)
        if prev is not None and prev != hf_id:
            raise RuntimeError(f"lr_grid_extend: {slug} already registered to {prev!r}; "
                               f"refusing to swap to {hf_id!r}")
        G.GRID_READERS[slug] = hf_id
    for size, b in (batches or EXTEND_BATCH).items():
        G.BATCH_BY_SIZE.setdefault(size, b)
    for slug, pin in (kv_pin or EXTEND_KV_PIN).items():
        G.KV_PIN.setdefault(slug, pin)      # TECH M2: 14B scores on the concat path, pinned
    return dict(G.GRID_READERS)


def main():
    register()
    G.main()


if __name__ == "__main__":
    main()
