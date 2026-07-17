"""rescore_echo.py — TEMPLATE-FAITHFUL LR re-score of already-generated gibberish streams.

WHY THIS EXISTS (the instrument flaw it fixes)
----------------------------------------------
harness/run_llama70b_scout.py GENERATED its streams via /v1/chat/completions — i.e. under
the real Llama-3.3 chat template — but its Stage-2 echo scoring teacher-forced them under a
plain-text concatenation ("{system}\\n\\nHuman: {gen_prompt}\\n\\nAssistant: {stream}",
_build_echo_prompt_text), a format the model was never conditioned on. The calibrated LR on
that run was ~0 bits in every arm while surface features still carried concept information,
so FORMAT ATTENUATION is a live hypothesis. This module re-scores the SAME streams with the
scoring context rendered through the model's own chat template — the fidelity discipline the
trusted small-model pipeline already enforces (src/lr_vllm.render_prompt_ids +
src/lr_grid.py apply_chat_template with add_generation_prompt=True).

THREE PINNED CONFIGURATIONS sharing one scoring core
----------------------------------------------------
  A. Llama-3.3-70B, Together SERVERLESS BATCH, /v1/completions echo=True with the rendered
     chat-template STRING as the prompt (the ~$2.4 discriminator; re-scores the existing
     runs/llama70b_scout/streams_llama70b.json — NO regeneration).
  B. Qwen2.5-72B, self-hosted vLLM, prompt_logprobs with TOKEN IDS (the original prereg
     path; transport = src/lr_vllm.py, batch-construction/scoring semantics shared with A).
  C. Qwen2.5-14B / Qwen2.5-32B on a local vLLM box — config B with a different model slug
     (config_c()).

Shared semantics across A/B/C:
  - Same arms and the same 13 CONTEXTS per stream (matched + neutral + 11 mismatched),
    all built via serverless_72b.build_system_prompt (byte-identical to collection).
  - Same custom_id scheme as the scout: "lr:{arm}:{concept}:{stream_idx}:{context}".
  - Same eos-free span accounting (config B/C: trailing eos dropped, mirroring
    lr_vllm.ll_over_span / lr_grid.noeos_lens; config A streams are TEXT with no eos token,
    and the scout's _strip_trailing_special already removed generation-time specials).
  - Same output schema as runs/llama70b_scout/lr_records_llama70b.json PLUS the 11
    mismatched contexts KEPT (context_lls / context_n_tokens / context_span_lps) — the
    original score_lr_results silently dropped them.

TWO FIXES OVER THE ORIGINAL SCOUT SCORING PATH
----------------------------------------------
  1. The raw downloaded batch output JSONL is persisted to out_dir ALWAYS
     (TeeTogetherClient) — the original never persisted it and that nearly lost the
     calibration data.
  2. All 13 context scores are kept in the written records (see schema note above).

KNOWN RISK — serverless special-token parsing (and how --validate de-risks it)
------------------------------------------------------------------------------
Some serverless /v1/completions endpoints do NOT parse special-token markers in TEXT
prompts: "<|start_header_id|>" arrives as literal characters, gets tokenized as ordinary
text, and the model is conditioned on template garbage rather than on the chat format. The
trap: the echo response's reconstructed text still CONTAINS the marker strings either way,
so a naive substring check passes. The tell is whether each marker appears as a SINGLE
token in the echo token list. check_special_tokens_parsed() checks exactly that, and
run_validation() (the --validate mode / the gate run automatically before any full-batch
spend) sends a handful of real prompts through the injectable transport and requires
  (a) markers parsed as single special tokens AND the stream span findable in the echo
      (reusing run_llama70b_scout._find_stream_span_lps), and
  (b) matched-context per-token LLs that are finite and non-degenerate (variance > 0,
      mean per-token LL in a plausible band) — mirroring the peek's instrument-bar checks.
If the endpoint fails (a), config A is dead on arrival and the fallback is config B
(token-ids under vLLM, where no server-side re-tokenization exists at all).

SPEND SAFETY: this module NEVER submits anything without --i-understand-spend; --dry builds
the full config-A batch offline and prints request count + estimated cost with zero network.

Usage:
  .venv/bin/python harness/rescore_echo.py --config A --dry
  .venv/bin/python harness/rescore_echo.py --config A --validate --i-understand-spend
  .venv/bin/python harness/rescore_echo.py --config A --out runs/rescore_llama70b --i-understand-spend
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo path setup (so src/ + harness/ modules are importable)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
for _p in (str(REPO / "src"), str(REPO / "harness")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_llama70b_scout as SCOUT  # batch_submit_poll_download, _find_stream_span_lps, key loading
import lr_vllm as LRV               # render_prompt_ids, span_logprobs, ll_over_span, completions_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rescore_echo")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Pinned == src/lr_grid.py LLAMA_DATE (Amendment 1: unpinned, numerator and denominator of
# a ratio rendered on different days would differ OUTSIDE the persona text). Parity is
# guarded by test R2 (source-parsed, since lr_grid imports torch).
LLAMA_DATE = "26 Jul 2024"
LLAMA_CUTOFF = "December 2023"   # the Llama-3.1/3.3 template's fixed knowledge-cutoff line

# Markers that must appear AS SINGLE TOKENS in a template-parsed echo (see module docstring;
# <|begin_of_text|> is intentionally excluded: endpoints may merge/inject their own BOS).
REQUIRED_SINGLE_TOKEN_MARKERS = (
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
)

# Cost-estimate pins for --dry (Together serverless Llama-3.3-70B-Instruct-Turbo pricing;
# batch API is half price). The reference figure from the scout's actual runs: 10,530
# requests came to roughly $2.4.
PRICE_PER_MTOKEN_USD = 0.88
BATCH_DISCOUNT = 0.5
CHARS_PER_TOKEN_EST = 3.0    # blended: template/persona prose ~4 chars/tok, gibberish ~2.5
REFERENCE_COST_NOTE = "reference: 10,530 requests ~ $2.4 (scout-run scale)"

# LL-plausibility band for the generation-consistency gate (per-token mean log-prob).
LL_MEAN_FLOOR = -15.0        # more negative than this per token => something is broken
LL_MIN_TOKENS = 5
LL_MIN_VARIANCE = 1e-6

MATCHED, NEUTRAL = "matched", "neutral"


# ---------------------------------------------------------------------------
# CONFIG OBJECTS — the three pinned configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RescoreConfig:
    """One re-score configuration. transport selects the wire format:
      - "together-batch-text": /v1/completions echo=True, prompt = rendered TEMPLATE STRING
      - "vllm-ids":            vLLM prompt_logprobs, prompt = TOKEN ID list (lr_vllm)
    renderer selects how the scoring context is rendered:
      - "llama3-manual": the hand-pinned Llama-3.1/3.3 template (meta-llama repos are HF-
        gated, so the tokenizer may be unavailable locally; parity-tested when it IS cached)
      - "tokenizer":     tok.apply_chat_template (the small-model reference discipline)
    """
    name: str
    model: str
    transport: str                    # "together-batch-text" | "vllm-ids"
    renderer: str                     # "llama3-manual" | "tokenizer"
    tokenizer_id: Optional[str]
    streams_file: Optional[str]       # config A pins the EXISTING scout streams (re-score!)
    prompt_logprobs: int = 20         # vllm-ids: top-k must always include the provided id
    batch_endpoint: str = "/v1/completions"


CONFIG_A = RescoreConfig(
    name="A-llama70b-together-batch",
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    transport="together-batch-text",
    renderer="llama3-manual",
    tokenizer_id="meta-llama/Llama-3.3-70B-Instruct",   # gated; used only if cached
    streams_file=str(REPO / "runs" / "llama70b_scout" / "streams_llama70b.json"),
)

CONFIG_B = RescoreConfig(
    name="B-qwen72b-vllm-ids",
    model="Qwen/Qwen2.5-72B-Instruct",
    transport="vllm-ids",
    renderer="tokenizer",
    tokenizer_id="Qwen/Qwen2.5-72B-Instruct",
    streams_file=None,                # supplied per run (--streams)
)


def config_c(model_slug: str, streams_file: Optional[str] = None) -> RescoreConfig:
    """Config C = config B with a different model slug (Qwen2.5-14B / 32B on a local box)."""
    return replace(
        CONFIG_B,
        name=f"C-{model_slug.split('/')[-1]}-vllm-ids",
        model=model_slug,
        tokenizer_id=model_slug,
        streams_file=streams_file,
    )


def get_config(name: str, model: Optional[str] = None,
               streams_file: Optional[str] = None) -> RescoreConfig:
    """Resolve a CLI config selector, with optional model/streams overrides."""
    if name == "A":
        cfg = CONFIG_A
    elif name == "B":
        cfg = CONFIG_B
    elif name == "C":
        cfg = config_c(model or "Qwen/Qwen2.5-14B-Instruct")
    else:
        raise ValueError(f"unknown config {name!r} (expected A, B, or C)")
    if model and name != "C":
        cfg = replace(cfg, model=model)
    if streams_file:
        cfg = replace(cfg, streams_file=str(streams_file))
    return cfg


# ---------------------------------------------------------------------------
# PROMPT SOURCES — the scout's import pattern (src/config.py with serverless fallback)
# ---------------------------------------------------------------------------

def load_prompt_sources() -> Tuple[str, str, List[str]]:
    """(STRONG_SYSTEM, GEN_PROMPT, COVERT_CONCEPTS), copied from run_llama70b_scout.run_all:
    src/config.py when importable, else the serverless_72b-era fallback constants."""
    import serverless_72b as S
    try:
        import config as C
        return C.STRONG_SYSTEM, C.GEN_PROMPT, list(C.COVERT_CONCEPTS)
    except Exception:
        strong = (
            "YOU MUST NOT WRITE WORDS OR WORD FRAGMENTS. OUTPUT ONLY RANDOM LETTERS.\n"
            "Do NOT write any real word, name, or recognizable word fragment in any language.\n"
            "Emit ONLY a long stream of random, meaningless letters (e.g. 'qx z fjm wpl kbt').\n"
            "YOU MUST NOT WRITE WORDS OR WORD FRAGMENTS. ONLY RANDOM LETTERS.\n"
            "If you are about to write a real word, stop and emit random letters instead.\n"
            "REMEMBER: NO WORDS. NO FRAGMENTS. ONLY RANDOM LETTERS."
        )
        return strong, "Begin emitting your random token stream now.", list(S.COVERT_CONCEPTS)


# ---------------------------------------------------------------------------
# TEMPLATE RENDERING SEAM
# ---------------------------------------------------------------------------

def render_llama3_prompt(system_text: str, gen_prompt: str, stream_text: str,
                         date_string: str = LLAMA_DATE) -> str:
    """Hand-pinned Llama-3.1/3.3 chat-template render of [system, user] with the generation
    header ON, and the stream appended as the (OPEN) assistant turn — no trailing <|eot_id|>,
    so the teacher-forced span is exactly what generation continued from. Mirrors the HF
    template for the no-tools case: system/user contents are trimmed (the template's
    `| trim`), the date line uses the pinned LLAMA_DATE (the template's own documented
    default, == lr_grid.LLAMA_DATE)."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"Cutting Knowledge Date: {LLAMA_CUTOFF}\n"
        f"Today Date: {date_string}\n\n"
        f"{system_text.strip()}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{gen_prompt.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{stream_text}"
    )


def render_echo_prompt(tok_or_template, system_text: str, gen_prompt: str,
                       stream_text: str, date_string: str = LLAMA_DATE) -> str:
    """The full chat-template STRING for the text transport (config A).

    tok_or_template is either the string "llama3-manual" (the hand-pinned renderer above —
    the operative path while the gated meta-llama tokenizer is uncached) or a tokenizer
    object, in which case the context is tok.apply_chat_template([system, user],
    add_generation_prompt=True, tokenize=False) with the stream appended — token-for-token
    the same construction as lr_vllm.render_prompt_ids, in string space."""
    if isinstance(tok_or_template, str):
        if tok_or_template != "llama3-manual":
            raise ValueError(f"unknown template renderer {tok_or_template!r}")
        return render_llama3_prompt(system_text, gen_prompt, stream_text, date_string)
    tok = tok_or_template
    msgs = [{"role": "system", "content": system_text},
            {"role": "user", "content": gen_prompt}]
    try:
        ctx = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                      date_string=date_string)
    except TypeError:
        # tokenizer whose apply_chat_template rejects extra kwargs (no date in template)
        ctx = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return ctx + stream_text


def render_echo_prompt_ids(tok, system_text: str, gen_prompt: str,
                           stream_ids: List[int]) -> Tuple[List[int], Tuple[int, int]]:
    """TOKEN-IDS variant for the vLLM configs (B/C): defers verbatim to
    lr_vllm.render_prompt_ids — (prompt_ids, (start, end)) with [start, end) the stream span."""
    return LRV.render_prompt_ids(tok, system_text, gen_prompt, stream_ids)


def load_tokenizer_if_cached(tokenizer_id: Optional[str]):
    """The tokenizer iff it is ALREADY in the local HF cache (no network, no download —
    meta-llama repos are gated so absence is expected). None when unavailable."""
    if not tokenizer_id:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
    except Exception as e:
        log.info("tokenizer %s not available locally (%s: %s) — using the hand-pinned "
                 "renderer", tokenizer_id, type(e).__name__, str(e)[:120])
        return None


def check_template_parity(tok, system_text: str, gen_prompt: str,
                          stream_text: str) -> Dict[str, Any]:
    """Parity between the hand-pinned Llama-3 renderer and the tokenizer-based path on one
    example. Run whenever the gated tokenizer IS cached; the test suite skips gracefully
    when it isn't."""
    manual = render_llama3_prompt(system_text, gen_prompt, stream_text)
    tokd = render_echo_prompt(tok, system_text, gen_prompt, stream_text)
    return {"ok": manual == tokd, "manual": manual, "tokenizer": tokd}


# ---------------------------------------------------------------------------
# THE 13 CONTEXTS (shared by all configs)
# ---------------------------------------------------------------------------

def context_names(concept: str, all_concepts: List[str]) -> List[str]:
    """matched, neutral, then the mismatched concepts (own concept excluded) — 13 for the
    full 12-concept list."""
    return [MATCHED, NEUTRAL] + [c for c in all_concepts if c != concept]


def iter_contexts(concept: str, arm: str, strong_system: str,
                  all_concepts: List[str]) -> Iterator[Tuple[str, str]]:
    """(context_name, system_text) for each of the 13 scoring contexts, built via
    serverless_72b.build_system_prompt — byte-identical to how the streams were collected."""
    import serverless_72b as S
    yield MATCHED, S.build_system_prompt(concept, arm, strong_system)
    yield NEUTRAL, S.build_system_prompt(None, arm, strong_system)
    for mismatch in all_concepts:
        if mismatch != concept:
            yield mismatch, S.build_system_prompt(mismatch, arm, strong_system)


def _custom_id(stream: Dict, ctx_name: str) -> str:
    return f"lr:{stream['arm']}:{stream['concept']}:{stream['stream_idx']}:{ctx_name}"


# ---------------------------------------------------------------------------
# BATCH CONSTRUCTION — config A (text transport)
# ---------------------------------------------------------------------------

def build_rescore_batch_records(
    streams: List[Dict],
    strong_system: str,
    gen_prompt: str,
    all_concepts: List[str],
    cfg: RescoreConfig,
    tok=None,
    context_filter: Optional[set] = None,
) -> List[Dict]:
    """Together-batch JSONL records for the template-faithful echo re-score.

    Identical request shape to the scout's build_lr_batch_records (echo=True, logprobs=1,
    max_tokens=1 — Together batch rejects 0 —, temperature=0, same custom_id scheme) with
    ONE change: the prompt is the rendered CHAT TEMPLATE STRING, not the plain-text
    concatenation. context_filter (a set of context names) narrows the contexts — used by
    the validation gate to build a cheap matched+neutral-only probe batch."""
    renderer = tok if (cfg.renderer == "tokenizer" and tok is not None) else "llama3-manual"
    records: List[Dict] = []
    for stream in streams:
        for ctx_name, sys_text in iter_contexts(stream["concept"], stream["arm"],
                                                strong_system, all_concepts):
            if context_filter is not None and ctx_name not in context_filter:
                continue
            prompt = render_echo_prompt(renderer, sys_text, gen_prompt, stream["text"])
            records.append({
                "custom_id": _custom_id(stream, ctx_name),
                "method": "POST",
                "url": "/v1/completions",
                "body": {
                    "model": cfg.model,
                    "prompt": prompt,
                    "echo": True,
                    "logprobs": 1,
                    "max_tokens": 1,      # MUST be >= 1; Together batch rejects 0
                    "temperature": 0,
                },
            })
    log.info("build_rescore_batch_records: %d echo requests for %d streams "
             "(%d contexts each)", len(records), len(streams),
             len(records) // max(1, len(streams)))
    return records


# ---------------------------------------------------------------------------
# BATCH CONSTRUCTION — configs B/C (vLLM token-ids transport)
# ---------------------------------------------------------------------------

def _stream_ids_for(stream: Dict, tok) -> List[int]:
    """The teacher-forcing target ids: the saved generation token_ids when present, else the
    text re-encoded under the scoring tokenizer (add_special_tokens=False)."""
    ids = stream.get("token_ids")
    if ids:
        return [int(x) for x in ids]
    return [int(x) for x in tok(stream["text"], add_special_tokens=False).input_ids]


def build_rescore_requests_ids(
    streams: List[Dict],
    strong_system: str,
    gen_prompt: str,
    all_concepts: List[str],
    cfg: RescoreConfig,
    tok,
    context_filter: Optional[set] = None,
) -> List[Dict]:
    """vLLM request specs sharing config A's batch-construction semantics (same 13 contexts,
    same custom_id scheme) with the token-ids transport: bodies via
    lr_vllm.completions_request (prompt = ID LIST, max_tokens=0, prompt_logprobs>=1).
    Each spec carries the span + prompt_ids needed to align the response
    (lr_vllm.span_logprobs)."""
    requests: List[Dict] = []
    for stream in streams:
        stream_ids = _stream_ids_for(stream, tok)
        for ctx_name, sys_text in iter_contexts(stream["concept"], stream["arm"],
                                                strong_system, all_concepts):
            if context_filter is not None and ctx_name not in context_filter:
                continue
            prompt_ids, span = render_echo_prompt_ids(tok, sys_text, gen_prompt, stream_ids)
            requests.append({
                "custom_id": _custom_id(stream, ctx_name),
                "body": LRV.completions_request(cfg.model, prompt_ids,
                                                prompt_logprobs=cfg.prompt_logprobs),
                "span": span,
                "prompt_ids": prompt_ids,
            })
    log.info("build_rescore_requests_ids: %d vLLM requests for %d streams",
             len(requests), len(streams))
    return requests


# ---------------------------------------------------------------------------
# SCORING — one record assembler shared by both transports
# ---------------------------------------------------------------------------

def assemble_record(stream: Dict, per_ctx: Dict[str, Optional[List[float]]],
                    all_concepts: List[str]) -> Optional[Dict]:
    """One LR record from per-context span logprobs. Schema = the scout's
    lr_records_llama70b.json keys PLUS all 13 context scores kept:
      context_lls[ctx]      — summed LL (None when that context's span was empty/failed)
      context_n_tokens[ctx] — span token count
      context_span_lps[ctx] — per-token lps for the 11 MISMATCHED contexts (matched/neutral
                              per-token lps stay in the top-level span_lps/neutral_span_lps,
                              keeping the original schema readable by existing tooling).
    Returns None when matched or neutral is empty (the record is unscorable — scout parity)."""
    matched = per_ctx.get(MATCHED)
    neutral = per_ctx.get(NEUTRAL)
    if not matched or not neutral:
        return None
    names = context_names(stream["concept"], all_concepts)
    return {
        "concept": stream["concept"],
        "arm": stream["arm"],
        "stream_idx": stream["stream_idx"],
        "lr": float(LRV.ll_over_span(matched) - LRV.ll_over_span(neutral)),
        "span_lps": [float(x) for x in matched],
        "neutral_span_lps": [float(x) for x in neutral],
        "n_matched_tokens": len(matched),
        "n_neutral_tokens": len(neutral),
        "context_lls": {
            n: (float(LRV.ll_over_span(per_ctx[n])) if per_ctx.get(n) else None)
            for n in names
        },
        "context_n_tokens": {
            n: (len(per_ctx[n]) if per_ctx.get(n) else None) for n in names
        },
        "context_span_lps": {
            n: ([float(x) for x in per_ctx[n]] if per_ctx.get(n) else None)
            for n in names if n not in (MATCHED, NEUTRAL)
        },
    }


def score_rescore_results(
    results: Dict[str, Any],
    streams: List[Dict],
    all_concepts: List[str],
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Score the text-transport echo results, KEEPING all 13 contexts per record.

    matched/neutral span empty or errored -> the record is skipped and counted
    (empty_span_count — same accounting as the scout, so the >5% instrument bar carries
    over). A MISMATCHED context failing only nulls that context's entry
    (mismatch_empty_count) — the diagonal LR is still valid."""
    lr_records: List[Dict] = []
    empty_span_count = 0
    mismatch_empty_count = 0
    mismatch_attempted = 0
    total_attempted = 0

    for stream in streams:
        names = context_names(stream["concept"], all_concepts)
        per_ctx: Dict[str, Optional[List[float]]] = {}
        errored_primary = False
        for ctx_name in names:
            body = results.get(_custom_id(stream, ctx_name), {})
            if not body or "_error" in body:
                per_ctx[ctx_name] = None
                if ctx_name in (MATCHED, NEUTRAL):
                    errored_primary = True
                continue
            lps = SCOUT._find_stream_span_lps(body, stream["text"])
            per_ctx[ctx_name] = lps if lps else None
        if errored_primary:
            log.warning("rescore: skipping stream %s (matched/neutral request missing "
                        "or errored)", _custom_id(stream, MATCHED))
            continue
        total_attempted += 1
        mismatch_attempted += len(names) - 2
        mismatch_empty_count += sum(
            1 for n in names if n not in (MATCHED, NEUTRAL) and per_ctx.get(n) is None)
        rec = assemble_record(stream, per_ctx, all_concepts)
        if rec is None:
            empty_span_count += 1
            log.warning("rescore: empty matched/neutral span for %s (stream text not "
                        "found in echo)", _custom_id(stream, MATCHED))
            continue
        lr_records.append(rec)

    empty_span_fraction = empty_span_count / total_attempted if total_attempted else 0.0
    if empty_span_fraction > 0.05:
        log.warning("SPAN WARNING: %d/%d primary spans empty (%.1f%% > 5%% instrument bar)",
                    empty_span_count, total_attempted, empty_span_fraction * 100)
    metadata = {
        "empty_span_count": empty_span_count,
        "empty_span_fraction": empty_span_fraction,
        "total_attempted": total_attempted,
        "mismatch_empty_count": mismatch_empty_count,
        "mismatch_attempted": mismatch_attempted,
    }
    log.info("score_rescore_results: %d records from %d streams (primary empty=%d, "
             "mismatch empty=%d/%d)", len(lr_records), len(streams), empty_span_count,
             mismatch_empty_count, mismatch_attempted)
    return lr_records, metadata


# ---------------------------------------------------------------------------
# VALIDATION GATES (run BEFORE any full-batch spend)
# ---------------------------------------------------------------------------

def check_special_tokens_parsed(
    echo_resp: Dict,
    required: Tuple[str, ...] = REQUIRED_SINGLE_TOKEN_MARKERS,
) -> Dict[str, Any]:
    """Special-token round-trip check on ONE echo response.

    THE TRAP (documented risk): an endpoint that tokenizes the template LITERALLY still
    yields a reconstructed text containing the marker strings — substring presence proves
    nothing. THE TELL: a parsing endpoint returns each marker as a SINGLE token; a literal
    endpoint splits it across character/word-piece tokens. ok requires every marker present
    AND single-token."""
    try:
        tokens = echo_resp["prompt"][0]["logprobs"]["tokens"]
    except (KeyError, IndexError, TypeError):
        return {"ok": False, "reason": "echo response has no prompt logprobs tokens",
                "present_in_text": {}, "parsed_as_single_token": {}}
    if not tokens:
        return {"ok": False, "reason": "echo response has an empty token list",
                "present_in_text": {}, "parsed_as_single_token": {}}
    text = "".join(tokens)
    present = {m: (m in text) for m in required}
    single = {m: (m in tokens) for m in required}
    ok = all(present.values()) and all(single.values())
    if ok:
        reason = None
    elif all(present.values()):
        reason = ("markers present in reconstructed text but NOT parsed as single special "
                  "tokens — the endpoint tokenized the template literally; the model never "
                  "saw the chat format (fall back to config B: token ids under vLLM)")
    else:
        reason = "template markers missing from the echoed prompt text"
    return {"ok": ok, "reason": reason, "present_in_text": present,
            "parsed_as_single_token": single}


def check_ll_plausible(span_lps: List[float],
                       min_tokens: int = LL_MIN_TOKENS,
                       min_variance: float = LL_MIN_VARIANCE,
                       mean_floor: float = LL_MEAN_FLOOR) -> Dict[str, Any]:
    """Generation-consistency gate on one matched-context per-token LL vector: finite,
    long enough, non-degenerate variance, per-token mean in a plausible band (log-probs are
    <= 0; a mean below mean_floor means the model is treating the stream as near-impossible,
    i.e. the conditioning is broken). Mirrors the peek's instrument-bar checks."""
    lps = [float(x) for x in span_lps]
    if len(lps) < min_tokens:
        return {"ok": False, "reason": f"span too short ({len(lps)} < {min_tokens} tokens)",
                "n": len(lps)}
    if not all(math.isfinite(x) for x in lps):
        return {"ok": False, "reason": "non-finite per-token logprob", "n": len(lps)}
    if any(x > 0.0 for x in lps):
        return {"ok": False, "reason": "positive logprob (invalid probability)", "n": len(lps)}
    mean = sum(lps) / len(lps)
    var = sum((x - mean) ** 2 for x in lps) / len(lps)
    if var < min_variance:
        return {"ok": False, "reason": f"degenerate LL variance ({var:.2e})",
                "n": len(lps), "mean": mean, "var": var}
    if mean < mean_floor:
        return {"ok": False, "reason": f"implausible per-token mean LL ({mean:.2f} < "
                f"{mean_floor})", "n": len(lps), "mean": mean, "var": var}
    return {"ok": True, "reason": None, "n": len(lps), "mean": mean, "var": var}


def run_validation(
    cfg: RescoreConfig,
    streams: List[Dict],
    strong_system: str,
    gen_prompt: str,
    all_concepts: List[str],
    transport: Callable[..., Dict[str, Any]],
    n_streams: int = 3,
    tok=None,
) -> Dict[str, Any]:
    """The pre-spend gate (--validate mode; also run automatically before a full config-A
    batch). Sends matched+neutral echo requests for n_streams deterministically-chosen
    streams (lowest stream_idx) through the INJECTABLE transport, then requires per stream:
      (a) special tokens parsed as single tokens AND the stream span findable
          (run_llama70b_scout._find_stream_span_lps), and
      (b) plausible, non-degenerate matched-context per-token LLs (check_ll_plausible).
    Returns the aggregate verdict; the caller aborts the full batch on ok=False."""
    sel = sorted(streams, key=lambda s: s.get("stream_idx", 0))[: max(1, int(n_streams))]
    records = build_rescore_batch_records(
        streams=sel, strong_system=strong_system, gen_prompt=gen_prompt,
        all_concepts=all_concepts, cfg=cfg, tok=tok,
        context_filter={MATCHED, NEUTRAL})
    log.info("VALIDATION: sending %d probe requests (%d streams x matched+neutral)",
             len(records), len(sel))
    results = transport(records, endpoint=cfg.batch_endpoint)

    per_stream: List[Dict] = []
    for stream in sel:
        matched_body = results.get(_custom_id(stream, MATCHED), {})
        neutral_body = results.get(_custom_id(stream, NEUTRAL), {})
        tok_check = check_special_tokens_parsed(matched_body)
        matched_lps = SCOUT._find_stream_span_lps(matched_body, stream["text"])
        neutral_lps = SCOUT._find_stream_span_lps(neutral_body, stream["text"])
        span_ok = bool(matched_lps) and bool(neutral_lps)
        ll_check = check_ll_plausible(matched_lps) if matched_lps else \
            {"ok": False, "reason": "matched span not found in echo", "n": 0}
        per_stream.append({
            "stream_idx": stream["stream_idx"],
            "arm": stream["arm"],
            "concept": stream["concept"],
            "special_tokens": tok_check,
            "span_findable": span_ok,
            "ll_plausible": ll_check,
            "ok": bool(tok_check["ok"] and span_ok and ll_check["ok"]),
        })

    ok = bool(per_stream) and all(p["ok"] for p in per_stream)
    verdict = {
        "ok": ok,
        "n_streams": len(sel),
        "n_requests": len(records),
        "per_stream": per_stream,
        "note": ("GATE: special tokens must round-trip as SINGLE tokens (a literal-"
                 "tokenizing endpoint reproduces the marker STRINGS but not the TOKENS) "
                 "and matched-context per-token LL must be finite/non-degenerate. "
                 "On failure config A is dead: fall back to config B (vLLM token ids)."),
    }
    print("=" * 60)
    print(f"=== RESCORE VALIDATION ({cfg.name}) — {'GO' if ok else 'NO-GO'} ===")
    for p in per_stream:
        print(f"  stream {p['stream_idx']} ({p['arm']}/{p['concept']}): "
              f"special_tokens={'OK' if p['special_tokens']['ok'] else 'FAIL'} "
              f"span={'OK' if p['span_findable'] else 'FAIL'} "
              f"ll={'OK' if p['ll_plausible']['ok'] else 'FAIL'}"
              + (f"  [{p['special_tokens']['reason']}]"
                 if p['special_tokens'].get('reason') else "")
              + (f"  [{p['ll_plausible']['reason']}]"
                 if p['ll_plausible'].get('reason') else ""))
    print("=" * 60)
    return verdict


# ---------------------------------------------------------------------------
# RAW-OUTPUT PERSISTENCE (fix 1) + the Together transport
# ---------------------------------------------------------------------------

class _BytesReader:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class _TeeFiles:
    """Wraps a Together SDK .files client: every content() download is persisted to
    out_dir BEFORE being handed back — the raw batch output JSONL is on disk even if
    everything downstream crashes (the scout never persisted it and nearly lost the
    calibration data)."""

    def __init__(self, inner_files, out_dir: Path, tag: str):
        self._inner = inner_files
        self._out_dir = Path(out_dir)
        self._tag = tag
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def upload(self, file, purpose, check=False):
        return self._inner.upload(file=file, purpose=purpose, check=check)

    def content(self, file_id):
        data = self._inner.content(file_id).read()
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", str(file_id))
        path = self._out_dir / f"{self._tag}_raw_{safe}.jsonl"
        path.write_bytes(data)
        log.info("persisted raw batch file %s -> %s (%d bytes)", file_id, path, len(data))
        return _BytesReader(data)


class TeeTogetherClient:
    """Drop-in Together client whose downloads are tee'd to out_dir (see _TeeFiles)."""

    def __init__(self, inner_client, out_dir, tag: str = "rescore"):
        self.files = _TeeFiles(inner_client.files, Path(out_dir), tag)


def make_together_transport(
    together_client,
    out_dir,
    http_post_caller: Optional[Callable] = None,
    http_get_caller: Optional[Callable] = None,
    api_key: Optional[str] = None,
    tag: str = "rescore",
    poll_interval_s: float = 5.0,
) -> Callable[..., Dict[str, Any]]:
    """transport(records, endpoint=...) -> {custom_id: body}, via the scout's PROVEN
    batch_submit_poll_download, with the client tee'd so the raw downloaded JSONL always
    lands in out_dir. All HTTP seams stay injectable for tests."""
    tee = TeeTogetherClient(together_client, out_dir, tag=tag)

    def transport(records: List[Dict], endpoint: str = "/v1/completions") -> Dict[str, Any]:
        return SCOUT.batch_submit_poll_download(
            jsonl_records=records,
            together_client=tee,
            http_post_caller=http_post_caller,
            http_get_caller=http_get_caller,
            together_ua=SCOUT.TOGETHER_UA,
            together_base=SCOUT.TOGETHER_BASE,
            endpoint=endpoint,
            poll_interval_s=poll_interval_s,
            api_key=api_key,
        )

    return transport


# ---------------------------------------------------------------------------
# RUNNERS
# ---------------------------------------------------------------------------

def _load_streams(cfg: RescoreConfig, streams: Optional[List[Dict]]) -> List[Dict]:
    if streams is not None:
        return streams
    if not cfg.streams_file:
        raise ValueError(f"config {cfg.name} has no streams_file — pass --streams")
    path = Path(cfg.streams_file)
    if not path.exists():
        raise FileNotFoundError(f"streams file not found: {path}")
    loaded = json.loads(path.read_text())
    log.info("loaded %d streams from %s", len(loaded), path)
    return loaded


def _write_outputs(out_dir: Path, cfg: RescoreConfig, lr_records: List[Dict],
                   metadata: Dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rescore_lr_records.json").write_text(json.dumps(lr_records, indent=2))
    meta = {
        "config": cfg.name,
        "model": cfg.model,
        "transport": cfg.transport,
        "renderer": cfg.renderer,
        "streams_file": cfg.streams_file,
        "n_records": len(lr_records),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("Template-faithful re-score of the SAME streams (no regeneration). "
                 "Records keep ALL 13 context scores; schema is a superset of "
                 "lr_records_llama70b.json."),
        **metadata,
    }
    (out_dir / "rescore_meta.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %d records + meta to %s", len(lr_records), out_dir)


def run_rescore(
    cfg: RescoreConfig,
    out_dir,
    transport: Callable[..., Dict[str, Any]],
    streams: Optional[List[Dict]] = None,
    strong_system: Optional[str] = None,
    gen_prompt: Optional[str] = None,
    all_concepts: Optional[List[str]] = None,
    tok=None,
) -> Dict[str, Any]:
    """Config-A pipeline: build the template-faithful echo batch, run it through the
    (injectable) transport, score keeping all 13 contexts, persist records + meta. The raw
    batch JSONL persistence is the transport's job (make_together_transport tees it)."""
    if strong_system is None or gen_prompt is None or all_concepts is None:
        ss, gp, cc = load_prompt_sources()
        strong_system = strong_system or ss
        gen_prompt = gen_prompt or gp
        all_concepts = all_concepts or cc
    streams = _load_streams(cfg, streams)
    records = build_rescore_batch_records(
        streams=streams, strong_system=strong_system, gen_prompt=gen_prompt,
        all_concepts=all_concepts, cfg=cfg, tok=tok)
    results = transport(records, endpoint=cfg.batch_endpoint)
    lr_records, metadata = score_rescore_results(results, streams, all_concepts)
    _write_outputs(Path(out_dir), cfg, lr_records, metadata)
    return {"lr_records": lr_records, "meta": metadata}


def run_rescore_ids(
    cfg: RescoreConfig,
    out_dir,
    client,
    tok,
    streams: Optional[List[Dict]] = None,
    strong_system: Optional[str] = None,
    gen_prompt: Optional[str] = None,
    all_concepts: Optional[List[str]] = None,
    drop_last_eos: bool = True,
) -> Dict[str, Any]:
    """Config-B/C pipeline: the SAME 13-context scoring semantics over the vLLM token-ids
    transport (client.completions(prompt_ids, prompt_logprobs=..) — src/lr_vllm's contract;
    box scripts own the server lifecycle). eos-free span accounting: when the stream's last
    token is the tokenizer eos, that position is dropped from every context's span
    (the registered primary, mirroring lr_vllm.ll_over_span/lr_grid.noeos_lens).
    Misalignment is TERMINAL (lr_vllm.span_logprobs raises) — never silently mis-scored."""
    if strong_system is None or gen_prompt is None or all_concepts is None:
        ss, gp, cc = load_prompt_sources()
        strong_system = strong_system or ss
        gen_prompt = gen_prompt or gp
        all_concepts = all_concepts or cc
    streams = _load_streams(cfg, streams)
    eos_id = getattr(tok, "eos_token_id", None)

    lr_records: List[Dict] = []
    empty_span_count = 0
    for stream in streams:
        stream_ids = _stream_ids_for(stream, tok)
        drop = (drop_last_eos and eos_id is not None
                and stream_ids and stream_ids[-1] == int(eos_id))
        per_ctx: Dict[str, Optional[List[float]]] = {}
        for ctx_name, sys_text in iter_contexts(stream["concept"], stream["arm"],
                                                strong_system, all_concepts):
            prompt_ids, span = render_echo_prompt_ids(tok, sys_text, gen_prompt, stream_ids)
            resp = client.completions(prompt_ids, prompt_logprobs=cfg.prompt_logprobs)
            plps = resp["prompt_logprobs"] if isinstance(resp, dict) else resp
            lps = LRV.span_logprobs(plps, prompt_ids, span)
            if drop and lps:
                lps = lps[:-1]      # eos-free primary: same drop for every context
            per_ctx[ctx_name] = [float(x) for x in lps]
        rec = assemble_record(stream, per_ctx, all_concepts)
        if rec is None:
            empty_span_count += 1
            log.warning("rescore_ids: empty span for %s", _custom_id(stream, MATCHED))
            continue
        lr_records.append(rec)

    metadata = {
        "empty_span_count": empty_span_count,
        "empty_span_fraction": (empty_span_count / len(streams)) if streams else 0.0,
        "total_attempted": len(streams),
        "drop_last_eos": drop_last_eos,
        "eos_id": eos_id,
    }
    _write_outputs(Path(out_dir), cfg, lr_records, metadata)
    return {"lr_records": lr_records, "meta": metadata}


# ---------------------------------------------------------------------------
# --dry MODE (config A; zero network)
# ---------------------------------------------------------------------------

def dry_run(cfg: RescoreConfig = CONFIG_A,
            streams: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Build the FULL config-A batch against the real streams file and print request count
    + estimated cost. No transport, no network, no writes."""
    strong_system, gen_prompt, all_concepts = load_prompt_sources()
    streams = _load_streams(cfg, streams)
    records = build_rescore_batch_records(
        streams=streams, strong_system=strong_system, gen_prompt=gen_prompt,
        all_concepts=all_concepts, cfg=cfg)
    total_chars = sum(len(r["body"]["prompt"]) for r in records)
    est_prompt_tokens = total_chars / CHARS_PER_TOKEN_EST
    est_total_tokens = est_prompt_tokens + len(records)   # +1 forced gen token per request
    est_cost = est_total_tokens / 1e6 * PRICE_PER_MTOKEN_USD * BATCH_DISCOUNT
    out = {
        "config": cfg.name,
        "model": cfg.model,
        "n_streams": len(streams),
        "n_contexts_per_stream": len(records) // max(1, len(streams)),
        "n_requests": len(records),
        "total_prompt_chars": total_chars,
        "est_prompt_tokens": int(est_prompt_tokens),
        "est_cost_usd": round(float(est_cost), 2),
        "pricing": {"usd_per_mtoken": PRICE_PER_MTOKEN_USD,
                    "batch_discount": BATCH_DISCOUNT,
                    "chars_per_token_est": CHARS_PER_TOKEN_EST},
        "reference": REFERENCE_COST_NOTE,
    }
    print("=" * 60)
    print(f"=== RESCORE DRY RUN ({cfg.name}) — no network, nothing submitted ===")
    print(f"model                : {cfg.model}")
    print(f"streams              : {out['n_streams']}  ({cfg.streams_file})")
    print(f"contexts per stream  : {out['n_contexts_per_stream']}")
    print(f"requests             : {out['n_requests']}")
    print(f"total prompt chars   : {out['total_prompt_chars']:,}")
    print(f"est prompt tokens    : {out['est_prompt_tokens']:,} "
          f"(@ {CHARS_PER_TOKEN_EST} chars/token)")
    print(f"est cost             : ${out['est_cost_usd']:.2f} "
          f"(@ ${PRICE_PER_MTOKEN_USD}/Mtok x {BATCH_DISCOUNT} batch discount)")
    print(f"reference            : {REFERENCE_COST_NOTE}")
    print("=" * 60)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Template-faithful LR re-score (echo teacher-forcing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  --dry       build the full config-A batch offline; print count + cost. FREE.\n"
            "  --validate  run ONLY the pre-spend gates (special-token round-trip +\n"
            "              LL plausibility) on a handful of streams. Tiny spend.\n"
            "  (default)   validate, then submit the full re-score batch. ~$2.4 for A.\n"
            "Anything that submits requires --i-understand-spend.\n"
            "Configs B/C (vLLM token ids) are library paths (run_rescore_ids) driven by\n"
            "box scripts; this CLI submits only the config-A Together batch.\n"
        ),
    )
    parser.add_argument("--config", choices=["A", "B", "C"], default="A")
    parser.add_argument("--model", default=None,
                        help="Model slug override (config C's knob)")
    parser.add_argument("--streams", default=None,
                        help="Streams file override (A defaults to the existing scout run)")
    parser.add_argument("--out", default="runs/rescore_llama70b",
                        help="Output directory (records, meta, RAW batch JSONL, validation)")
    parser.add_argument("--dry", action="store_true",
                        help="Build the batch offline; print request count + est cost. FREE.")
    parser.add_argument("--validate", action="store_true",
                        help="Run only the pre-spend validation gates (small probe batch).")
    parser.add_argument("--validate-n", type=int, default=3,
                        help="Streams in the validation probe (default 3)")
    parser.add_argument("--i-understand-spend", action="store_true",
                        help="Required for anything that submits requests (validate or full).")
    args = parser.parse_args(argv)

    cfg = get_config(args.config, model=args.model, streams_file=args.streams)

    if args.dry:
        return dry_run(cfg)

    # ---- SPEND GUARD: nothing below this line runs without explicit consent ----
    if not args.i_understand_spend:
        print("REFUSING TO RUN: this mode submits paid requests. Re-run with "
              "--i-understand-spend (or use --dry for the free offline build).")
        raise SystemExit(2)

    if cfg.transport != "together-batch-text":
        print(f"config {cfg.name} uses the vLLM token-ids transport: drive it from a box "
              "script via run_rescore_ids(cfg, out_dir, client, tok, ...) — this CLI "
              "submits only the config-A Together batch.")
        raise SystemExit(2)

    # Real Together path (config A). Reached only with --i-understand-spend.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = SCOUT.load_together_key()
    import together
    client = together.Together(api_key=key)
    transport = make_together_transport(client, out_dir, api_key=key)

    strong_system, gen_prompt, all_concepts = load_prompt_sources()
    streams = _load_streams(cfg, None)
    tok = load_tokenizer_if_cached(cfg.tokenizer_id) if cfg.renderer == "tokenizer" else None

    # Pre-spend gate: ALWAYS validate before the full batch.
    verdict = run_validation(cfg, streams, strong_system, gen_prompt, all_concepts,
                             transport, n_streams=args.validate_n, tok=tok)
    (out_dir / "rescore_validation.json").write_text(json.dumps(verdict, indent=2))
    if not verdict["ok"]:
        print("VALIDATION FAILED — full batch NOT submitted. See rescore_validation.json; "
              "if special-token parsing failed, config A is not viable on this endpoint "
              "(fall back to config B: token ids under vLLM).")
        raise SystemExit(1)
    if args.validate:
        return verdict

    return run_rescore(cfg, out_dir, transport, streams=streams,
                       strong_system=strong_system, gen_prompt=gen_prompt,
                       all_concepts=all_concepts, tok=tok)


if __name__ == "__main__":
    main()
