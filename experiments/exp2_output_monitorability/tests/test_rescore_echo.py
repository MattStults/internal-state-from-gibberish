"""RED-first unit tests for harness/rescore_echo.py — the template-faithful LR re-score path.

NO real API calls. All transports are injected fakes (same discipline as
test_run_llama70b_scout.py). The suite additionally monkeypatches urllib.request.urlopen
to a raiser around the --dry path to PROVE no network is touched.

Test inventory
--------------
R0  import rescore_echo
R1  render_llama3_prompt: hand-pinned Llama-3.1/3.3 template structure (markers, date line,
    generation header, stream appended with NO trailing <|eot_id|>).
R2  LLAMA_DATE parity with src/lr_grid.py's pinned constant (parsed from source, no import —
    lr_grid pulls torch).
R3  tokenizer-vs-manual template parity on a fixed example; SKIPS gracefully (still PASS,
    with a note) when the gated meta-llama tokenizer is not in the local HF cache.
R4  render_echo_prompt dispatch: "llama3-manual" -> hand renderer; tokenizer object ->
    apply_chat_template context + stream appended.
R5  render_echo_prompt_ids defers to lr_vllm.render_prompt_ids (ctx ids + stream ids, span).
R6  iter_contexts: exactly 13 contexts (matched, neutral, 11 mismatched), built via
    serverless_72b.build_system_prompt; own concept excluded from mismatches.
R7  build_rescore_batch_records (config A): 13 records/stream, scout custom_id scheme
    lr:{arm}:{concept}:{stream_idx}:{context}, echo=True/logprobs=1/max_tokens>=1/temp=0,
    prompt is the rendered CHAT TEMPLATE STRING ending with the stream text.
R8  build_rescore_requests_ids (config B/C): same custom_id scheme, bodies via
    lr_vllm.completions_request (prompt = TOKEN ID list, max_tokens=0, prompt_logprobs>=1).
R9  score_rescore_results keeps ALL 13 context scores (the original score_lr_results dropped
    the 11 mismatched ones) while remaining a schema superset of lr_records_llama70b.json.
R10 mismatched-context empty span is tolerated: record kept, that context's ll = None,
    counted in metadata.
R11 matched/neutral empty span: record skipped + counted (scout parity).
R12 check_special_tokens_parsed: SUCCESS shape — markers appear as single tokens.
R13 check_special_tokens_parsed: FAILURE shape — markers present in reconstructed text but
    split across character tokens (the endpoint tokenized the template literally). This is
    the documented serverless special-token-parsing risk.
R14 check_ll_plausible: finite non-degenerate lps pass; NaN / zero-variance / empty fail.
R15 run_validation with injected fake transports: success transport -> verdict ok; literal-
    tokenization transport -> verdict NOT ok. Zero real HTTP.
R16 TeeTogetherClient persists the raw downloaded batch JSONL to out_dir ALWAYS.
R17 make_together_transport: full fake round-trip through the scout's
    batch_submit_poll_download AND the raw output file lands in out_dir.
R18 run_rescore (config A semantics, fake transport): writes rescore_lr_records.json with
    13-context records + rescore_meta.json.
R19 config objects: A pins the EXISTING scout streams file + together-batch-text; B is
    vllm-ids Qwen2.5-72B; config_c() is B with a different model slug.
R20 dry_run against the REAL streams file: 10,530 requests, cost estimate in a sane band,
    urllib.request.urlopen monkeypatched to raise -> proves no network call.
R21 spend guard: main() without --dry and without --i-understand-spend exits (code 2)
    BEFORE any key loading / client construction.
R22 main --dry works end to end and returns the dry dict.
R23 run_rescore_ids (config B/C semantics, fake vLLM client + fake tokenizer): same record
    schema, eos-free span accounting applied.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup (match test_run_llama70b_scout.py)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
HARNESS = REPO / "harness"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HARNESS))

checks: List = []


def check(name: str, cond: bool, note: str = "") -> None:
    checks.append((name, bool(cond), note))


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # sys.modules registration BEFORE exec: dataclasses under `from __future__ import
    # annotations` resolve string annotations via sys.modules[cls.__module__] (py3.10).
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RESCORE_PATH = str(HARNESS / "rescore_echo.py")
RE = None
try:
    RE = load_module("rescore_echo", RESCORE_PATH)
    check("R0 import rescore_echo", True)
except Exception as e:
    check("R0 import rescore_echo", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
STRONG = "YOU MUST NOT WRITE WORDS. OUTPUT ONLY RANDOM LETTERS."
GEN_PROMPT = "Begin emitting your random token stream now."
CONCEPTS3 = ["curiosity", "ocean", "fear"]

STREAMS2 = [
    {"concept": "curiosity", "arm": "evoked", "text": "qxz fjm wpl kbt",
     "accepted": True, "stream_idx": 0},
    {"concept": "ocean", "arm": "evoked", "text": "rvnm kbt xqz",
     "accepted": True, "stream_idx": 1},
]


def _echo_body(context_tokens: List[str], stream_tokens: List[str],
               ctx_lps: List[float], stream_lps: List[float]) -> Dict:
    """Together /v1/completions echo=True response shape."""
    return {"prompt": [{"logprobs": {
        "tokens": context_tokens + stream_tokens,
        "token_logprobs": ctx_lps + stream_lps,
    }}]}


# ---------------------------------------------------------------------------
# R1: hand-pinned Llama-3 template renderer
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        p = RE.render_llama3_prompt("SYS TEXT", "USER TEXT", "gib gib gib")
        check("R1 starts with <|begin_of_text|>", p.startswith("<|begin_of_text|>"))
        check("R1 system header present",
              "<|start_header_id|>system<|end_header_id|>\n\n" in p)
        check("R1 user header present",
              "<|start_header_id|>user<|end_header_id|>\n\n" in p)
        check("R1 cutting-knowledge line pinned",
              "Cutting Knowledge Date: December 2023\n" in p)
        check("R1 date line pinned to LLAMA_DATE",
              f"Today Date: {RE.LLAMA_DATE}\n\n" in p)
        check("R1 system text in render", "SYS TEXT" in p)
        check("R1 user text in render", "USER TEXT" in p)
        check("R1 generation header immediately before stream",
              p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\ngib gib gib"))
        check("R1 no trailing eot after stream (assistant turn left OPEN)",
              not p.rstrip().endswith("<|eot_id|>"))
        check("R1 exactly two <|eot_id|> (system + user closes)",
              p.count("<|eot_id|>") == 2, f"count={p.count('<|eot_id|>')}")
        # ordering: system < user < assistant headers
        i_s = p.find("<|start_header_id|>system")
        i_u = p.find("<|start_header_id|>user")
        i_a = p.find("<|start_header_id|>assistant")
        check("R1 header ordering system<user<assistant", 0 <= i_s < i_u < i_a)
        # date_string override honored
        p2 = RE.render_llama3_prompt("s", "u", "x", date_string="01 Jan 2000")
        check("R1 date_string override honored", "Today Date: 01 Jan 2000\n\n" in p2)
    except Exception as e:
        check("R1 render_llama3_prompt", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R2: LLAMA_DATE parity with src/lr_grid.py (source parse, no torch import)
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        grid_src = (REPO / "src" / "lr_grid.py").read_text()
        m = re.search(r'^LLAMA_DATE\s*=\s*"([^"]+)"', grid_src, re.M)
        check("R2 lr_grid.LLAMA_DATE found in source", m is not None)
        if m:
            check("R2 LLAMA_DATE parity with lr_grid pin",
                  RE.LLAMA_DATE == m.group(1),
                  f"rescore={RE.LLAMA_DATE!r} lr_grid={m.group(1)!r}")
    except Exception as e:
        check("R2 LLAMA_DATE parity", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R3: tokenizer parity (skip gracefully when the gated tokenizer isn't cached)
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        tok = RE.load_tokenizer_if_cached(RE.CONFIG_A.tokenizer_id)
        if tok is None:
            check("R3 tokenizer parity (SKIPPED: meta-llama tokenizer not in local cache)",
                  True, "gated repo not cached; hand renderer is the operative path")
        else:
            res = RE.check_template_parity(tok, "SYS TEXT", "USER TEXT", "gib gib gib")
            check("R3 tokenizer parity on fixed example", res.get("ok") is True,
                  f"manual={res.get('manual','')[:120]!r} tok={res.get('tokenizer','')[:120]!r}")
    except Exception as e:
        check("R3 tokenizer parity", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R4: render_echo_prompt dispatch
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        manual = RE.render_echo_prompt("llama3-manual", "S", "U", "streamtxt")
        check("R4 'llama3-manual' == render_llama3_prompt",
              manual == RE.render_llama3_prompt("S", "U", "streamtxt"))

        class _FakeTok:
            def apply_chat_template(self, msgs, add_generation_prompt=False,
                                    tokenize=True, **kw):
                assert add_generation_prompt and not tokenize
                return f"[CTX sys={msgs[0]['content']} user={msgs[1]['content']} GENHDR]"

        out = RE.render_echo_prompt(_FakeTok(), "S", "U", "streamtxt")
        check("R4 tokenizer path appends stream after generation header",
              out == "[CTX sys=S user=U GENHDR]streamtxt", f"out={out!r}")
    except Exception as e:
        check("R4 render_echo_prompt dispatch", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R5: render_echo_prompt_ids defers to lr_vllm.render_prompt_ids
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        class _FakeTokIds:
            def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
                return [1, 2, 3]

        ids, span = RE.render_echo_prompt_ids(_FakeTokIds(), "S", "U", [7, 8])
        check("R5 prompt_ids = ctx ids + stream ids", ids == [1, 2, 3, 7, 8], f"ids={ids}")
        check("R5 span indexes the stream tokens", span == (3, 5), f"span={span}")
    except Exception as e:
        check("R5 render_echo_prompt_ids", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R6: iter_contexts — 13 contexts via serverless_72b.build_system_prompt
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        import serverless_72b as S72
        all12 = list(S72.COVERT_CONCEPTS)
        ctxs = list(RE.iter_contexts("curiosity", "evoked", STRONG, all12))
        names = [n for n, _ in ctxs]
        check("R6 13 contexts total", len(ctxs) == 13, f"n={len(ctxs)}")
        check("R6 first is matched, second is neutral", names[:2] == ["matched", "neutral"])
        check("R6 own concept not among mismatches", "curiosity" not in names[2:])
        check("R6 11 mismatched contexts", len(names[2:]) == 11)
        sysmap = dict(ctxs)
        check("R6 matched == build_system_prompt(concept, arm, strong)",
              sysmap["matched"] == S72.build_system_prompt("curiosity", "evoked", STRONG))
        check("R6 neutral == build_system_prompt(None, arm, strong)",
              sysmap["neutral"] == S72.build_system_prompt(None, "evoked", STRONG))
        check("R6 mismatch ctx == build_system_prompt(mismatch, arm, strong)",
              sysmap["ocean"] == S72.build_system_prompt("ocean", "evoked", STRONG))
    except Exception as e:
        check("R6 iter_contexts", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R7: build_rescore_batch_records (config A / text transport)
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        recs = RE.build_rescore_batch_records(
            streams=STREAMS2, strong_system=STRONG, gen_prompt=GEN_PROMPT,
            all_concepts=CONCEPTS3, cfg=RE.CONFIG_A)
        # 3 concepts -> matched + neutral + 2 mismatched = 4 contexts per stream
        check("R7 (n_concepts+1) records per stream", len(recs) == 2 * 4, f"n={len(recs)}")
        cids = [r["custom_id"] for r in recs]
        check("R7 scout custom_id scheme lr:{arm}:{concept}:{idx}:{ctx}",
              "lr:evoked:curiosity:0:matched" in cids
              and "lr:evoked:curiosity:0:neutral" in cids
              and "lr:evoked:curiosity:0:ocean" in cids, f"cids={cids[:4]}")
        r0 = recs[0]
        b = r0["body"]
        check("R7 url is /v1/completions", r0["url"] == "/v1/completions")
        check("R7 echo=True", b.get("echo") is True)
        check("R7 logprobs=1", b.get("logprobs") == 1)
        check("R7 max_tokens >= 1 (Together batch rejects 0)", b.get("max_tokens", 0) >= 1)
        check("R7 temperature=0", b.get("temperature") == 0)
        check("R7 model from config A", b.get("model") == RE.CONFIG_A.model)
        check("R7 prompt is a chat-template STRING with special markers",
              isinstance(b["prompt"], str) and "<|start_header_id|>" in b["prompt"]
              and "<|eot_id|>" in b["prompt"])
        check("R7 prompt ends with the stream text",
              b["prompt"].endswith(STREAMS2[0]["text"]))
        # matched and neutral prompts differ only via the system text
        by_cid = {r["custom_id"]: r["body"]["prompt"] for r in recs}
        check("R7 matched != neutral prompt",
              by_cid["lr:evoked:curiosity:0:matched"] != by_cid["lr:evoked:curiosity:0:neutral"])
        # 13-context count on the full concept list
        import serverless_72b as S72
        recs13 = RE.build_rescore_batch_records(
            streams=STREAMS2[:1], strong_system=STRONG, gen_prompt=GEN_PROMPT,
            all_concepts=list(S72.COVERT_CONCEPTS), cfg=RE.CONFIG_A)
        check("R7 full concept list -> 13 requests per stream", len(recs13) == 13,
              f"n={len(recs13)}")
    except Exception as e:
        check("R7 build_rescore_batch_records", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R8: build_rescore_requests_ids (config B/C / vllm-ids transport)
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        class _FakeTokB:
            eos_token_id = 99
            def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
                return [1, 2, 3]
            def __call__(self, text, add_special_tokens=False):
                class R:  # noqa
                    input_ids = [41, 42, 43]
                return R()

        streams_ids = [dict(STREAMS2[0], token_ids=[7, 8])]
        reqs = RE.build_rescore_requests_ids(
            streams=streams_ids, strong_system=STRONG, gen_prompt=GEN_PROMPT,
            all_concepts=CONCEPTS3, cfg=RE.CONFIG_B, tok=_FakeTokB())
        check("R8 4 requests for 3 concepts", len(reqs) == 4, f"n={len(reqs)}")
        check("R8 same custom_id scheme",
              reqs[0]["custom_id"] == "lr:evoked:curiosity:0:matched",
              f"cid={reqs[0]['custom_id']}")
        b8 = reqs[0]["body"]
        check("R8 prompt is TOKEN ID list", b8["prompt"] == [1, 2, 3, 7, 8],
              f"prompt={b8['prompt']}")
        check("R8 max_tokens=0 (vLLM scores the prompt only)", b8["max_tokens"] == 0)
        check("R8 prompt_logprobs >= 1", b8.get("prompt_logprobs", 0) >= 1)
        check("R8 model from config B", b8["model"] == RE.CONFIG_B.model)
        check("R8 span carried for alignment", reqs[0]["span"] == (3, 5),
              f"span={reqs[0].get('span')}")
        # falls back to re-tokenizing text when token_ids absent
        reqs_txt = RE.build_rescore_requests_ids(
            streams=[STREAMS2[0]], strong_system=STRONG, gen_prompt=GEN_PROMPT,
            all_concepts=CONCEPTS3, cfg=RE.CONFIG_B, tok=_FakeTokB())
        check("R8 token_ids=None -> re-tokenize text",
              reqs_txt[0]["body"]["prompt"] == [1, 2, 3, 41, 42, 43],
              f"prompt={reqs_txt[0]['body']['prompt']}")
    except Exception as e:
        check("R8 build_rescore_requests_ids", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R9-R11: score_rescore_results — all 13 contexts kept, empty-span accounting
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        st = STREAMS2[0]           # curiosity / evoked / idx 0
        text = st["text"]
        ctx_names = ["matched", "neutral", "ocean", "fear"]
        # matched lps sum -3.0; neutral -6.0; ocean -9.0; fear -12.0
        sums = {"matched": -1.0, "neutral": -2.0, "ocean": -3.0, "fear": -4.0}
        results = {}
        for cn in ctx_names:
            results[f"lr:evoked:curiosity:0:{cn}"] = _echo_body(
                ["CTXPREFIX "], [text], [-0.5], [sums[cn]] * 3)
        lr_recs, meta = RE.score_rescore_results(results, [st], CONCEPTS3)
        check("R9 one record", len(lr_recs) == 1, f"n={len(lr_recs)}")
        if lr_recs:
            r = lr_recs[0]
            base_keys = {"concept", "arm", "stream_idx", "lr", "span_lps",
                         "neutral_span_lps", "n_matched_tokens", "n_neutral_tokens"}
            check("R9 schema superset of lr_records_llama70b.json",
                  base_keys.issubset(r.keys()),
                  f"missing={base_keys - set(r.keys())}")
            check("R9 lr = LL(matched) - LL(neutral)",
                  abs(r["lr"] - ((-1.0 * 3) - (-2.0 * 3))) < 1e-9, f"lr={r['lr']}")
            check("R9 context_lls has all 4 contexts (matched+neutral+2 mismatched)",
                  set(r.get("context_lls", {}).keys()) == set(ctx_names),
                  f"keys={sorted(r.get('context_lls', {}).keys())}")
            check("R9 mismatched context ll kept (ocean)",
                  abs(r["context_lls"]["ocean"] - (-9.0)) < 1e-9,
                  f"ocean={r['context_lls'].get('ocean')}")
            check("R9 mismatched per-token lps kept",
                  r.get("context_span_lps", {}).get("fear") == [-4.0, -4.0, -4.0],
                  f"fear={r.get('context_span_lps', {}).get('fear')}")
            check("R9 context_n_tokens kept",
                  r.get("context_n_tokens", {}).get("ocean") == 3)

        # R10: one MISMATCHED context unfindable -> record kept, ll None, counted
        results_r10 = dict(results)
        results_r10["lr:evoked:curiosity:0:ocean"] = _echo_body(
            ["TOTALLY"], ["DIFFERENT"], [-1.0], [-1.0])
        recs10, meta10 = RE.score_rescore_results(results_r10, [st], CONCEPTS3)
        check("R10 record kept when only a mismatch span is empty", len(recs10) == 1)
        if recs10:
            check("R10 empty mismatch ll is None",
                  recs10[0]["context_lls"]["ocean"] is None)
        check("R10 mismatch empty counted in metadata",
              meta10.get("mismatch_empty_count") == 1,
              f"meta={meta10}")

        # R11: MATCHED unfindable -> record skipped + counted (scout parity)
        results_r11 = dict(results)
        results_r11["lr:evoked:curiosity:0:matched"] = _echo_body(
            ["TOTALLY"], ["DIFFERENT"], [-1.0], [-1.0])
        recs11, meta11 = RE.score_rescore_results(results_r11, [st], CONCEPTS3)
        check("R11 record skipped when matched span empty", len(recs11) == 0)
        check("R11 primary empty_span_count == 1",
              meta11.get("empty_span_count") == 1, f"meta={meta11}")
    except Exception as e:
        check("R9-R11 score_rescore_results", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R12/R13: special-token round-trip check — both shapes
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        # SUCCESS: endpoint parsed the template — markers are single tokens
        good_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "system",
                       "<|end_header_id|>", "\n\n", "SYS", "<|eot_id|>",
                       "<|start_header_id|>", "user", "<|end_header_id|>", "\n\n",
                       "U", "<|eot_id|>", "<|start_header_id|>", "assistant",
                       "<|end_header_id|>", "\n\n", "gib", " gib"]
        good = {"prompt": [{"logprobs": {"tokens": good_tokens,
                                         "token_logprobs": [None] + [-1.0] * 18}}]}
        v12 = RE.check_special_tokens_parsed(good)
        check("R12 success shape -> ok True", v12.get("ok") is True, f"v={v12}")

        # FAILURE: endpoint tokenized the template LITERALLY — markers split
        lit_tokens = ["<", "|", "begin", "_of", "_text", "|", ">",
                      "<", "|", "start", "_header", "_id", "|", ">", "system",
                      "<", "|", "end", "_header", "_id", "|", ">", "\n\n", "SYS",
                      "<", "|", "e", "ot", "_id", "|", ">", "gib", " gib"]
        lit = {"prompt": [{"logprobs": {"tokens": lit_tokens,
                                        "token_logprobs": [None] + [-1.0] * 32}}]}
        v13 = RE.check_special_tokens_parsed(lit)
        check("R13 literal-tokenization shape -> ok False", v13.get("ok") is False,
              f"v={v13}")
        check("R13 markers present in reconstructed text (the trap)",
              all(v13.get("present_in_text", {}).values()),
              f"present={v13.get('present_in_text')}")
        check("R13 markers NOT single tokens (the tell)",
              not any(v13.get("parsed_as_single_token", {}).values()),
              f"single={v13.get('parsed_as_single_token')}")
        check("R13 reason names the risk",
              "special" in str(v13.get("reason", "")).lower(), f"reason={v13.get('reason')}")

        # degenerate response
        v_bad = RE.check_special_tokens_parsed({"prompt": []})
        check("R12/13 malformed echo -> ok False", v_bad.get("ok") is False)
    except Exception as e:
        check("R12/R13 special-token check", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R14: generation-consistency (LL plausibility) check
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        ok_lps = [-1.0, -2.5, -0.3, -4.0, -1.7, -2.2]
        v = RE.check_ll_plausible(ok_lps)
        check("R14 finite non-degenerate lps pass", v.get("ok") is True, f"v={v}")
        v_nan = RE.check_ll_plausible([-1.0, float("nan"), -2.0, -1.0, -1.0])
        check("R14 NaN fails", v_nan.get("ok") is False)
        v_flat = RE.check_ll_plausible([-1.0] * 10)
        check("R14 zero-variance (degenerate) fails", v_flat.get("ok") is False)
        v_empty = RE.check_ll_plausible([])
        check("R14 empty fails", v_empty.get("ok") is False)
        v_pos = RE.check_ll_plausible([0.5, -1.0, -2.0, -1.5, -0.7])
        check("R14 positive logprob fails", v_pos.get("ok") is False)
    except Exception as e:
        check("R14 check_ll_plausible", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R15: run_validation with injected fake transports (both shapes)
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        def _tok_pieces(text: str) -> List[str]:
            """Chunk stream text into fake 'tokens' whose join reproduces it (fine-grained
            so short fixture streams still clear the LL gate's min-token bar)."""
            step = 2
            return [text[i:i + step] for i in range(0, len(text), step)]

        def _make_transport(parse_specials: bool):
            calls: List = []

            def transport(records, endpoint="/v1/completions"):
                calls.append(len(records))
                out = {}
                for rec in records:
                    prompt = rec["body"]["prompt"]
                    # find the stream: everything after the generation header
                    marker = "<|end_header_id|>\n\n"
                    stream = prompt[prompt.rfind(marker) + len(marker):]
                    if parse_specials:
                        ctx_toks = ["<|begin_of_text|>", "<|start_header_id|>",
                                    "system", "<|end_header_id|>", "\n\nCTX",
                                    "<|eot_id|>", "<|start_header_id|>", "assistant",
                                    "<|end_header_id|>", "\n\n"]
                    else:
                        ctx_toks = ["<", "|", "begin", "_of", "_text", "|", ">",
                                    "<", "|", "start", "_header", "_id", "|", ">",
                                    "CTX", "\n\n"]
                    s_toks = _tok_pieces(stream)
                    lps = [None] + [-0.5] * (len(ctx_toks) - 1) + \
                          [-(1.0 + 0.13 * (i % 7)) for i in range(len(s_toks))]
                    out[rec["custom_id"]] = {"prompt": [{"logprobs": {
                        "tokens": ctx_toks + s_toks,
                        "token_logprobs": lps,
                    }}]}
                return out

            transport.calls = calls
            return transport

        good_tp = _make_transport(parse_specials=True)
        verdict = RE.run_validation(
            cfg=RE.CONFIG_A, streams=STREAMS2, strong_system=STRONG,
            gen_prompt=GEN_PROMPT, all_concepts=CONCEPTS3,
            transport=good_tp, n_streams=2)
        check("R15 success transport -> verdict ok", verdict.get("ok") is True,
              f"verdict={ {k: v for k, v in verdict.items() if k != 'per_stream'} }")
        check("R15 validation used the injected transport (matched+neutral only)",
              good_tp.calls and good_tp.calls[0] == 2 * 2,
              f"calls={good_tp.calls}")

        bad_tp = _make_transport(parse_specials=False)
        verdict_bad = RE.run_validation(
            cfg=RE.CONFIG_A, streams=STREAMS2, strong_system=STRONG,
            gen_prompt=GEN_PROMPT, all_concepts=CONCEPTS3,
            transport=bad_tp, n_streams=2)
        check("R15 literal-tokenization transport -> verdict NOT ok",
              verdict_bad.get("ok") is False)
    except Exception as e:
        check("R15 run_validation with fakes", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R16/R17: raw batch output persisted ALWAYS (tee client + transport)
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        class _BytesWrapper:
            def __init__(self, data: bytes):
                self._data = data
            def read(self):
                return self._data

        class _FakeFiles:
            def __init__(self, output: bytes):
                self._output = output
                self.upload_calls: List = []
            def upload(self, file, purpose, check=False):
                self.upload_calls.append(file)
                class U:  # noqa
                    id = "file-tee-001"
                return U()
            def content(self, file_id):
                return _BytesWrapper(self._output)

        class _FakeClient:
            def __init__(self, output: bytes):
                self.files = _FakeFiles(output)

        raw_line = json.dumps({"custom_id": "lr:evoked:curiosity:0:matched",
                               "response": {"body": {"x": 1}}})
        with tempfile.TemporaryDirectory() as td:
            fc = _FakeClient(raw_line.encode())
            tee = RE.TeeTogetherClient(fc, td)
            got = tee.files.content("output-file-xyz").read()
            check("R16 tee returns the downloaded bytes", got == raw_line.encode())
            persisted = list(Path(td).glob("*output-file-xyz*"))
            check("R16 raw JSONL persisted to out_dir", len(persisted) == 1,
                  f"files={[p.name for p in Path(td).iterdir()]}")
            if persisted:
                check("R16 persisted bytes identical", persisted[0].read_bytes() == raw_line.encode())
            up = tee.files.upload(file="/tmp/x.jsonl", purpose="batch-api", check=False)
            check("R16 upload delegates to inner client",
                  up.id == "file-tee-001" and fc.files.upload_calls == ["/tmp/x.jsonl"])

        # R17: full fake round-trip through the scout's batch_submit_poll_download
        def _post(url, headers, body):
            return {"job": {"id": "batch-tee", "status": "validating"}}

        def _get(url, headers):
            return (200, {"job": {"status": "COMPLETED",
                                  "output_file_id": "output-file-r17"}})

        with tempfile.TemporaryDirectory() as td:
            fc2 = _FakeClient(raw_line.encode())
            transport = RE.make_together_transport(
                fc2, out_dir=td, http_post_caller=_post, http_get_caller=_get,
                api_key="test-key", poll_interval_s=0.01)
            res = transport([{"custom_id": "lr:evoked:curiosity:0:matched",
                              "method": "POST", "url": "/v1/completions", "body": {}}])
            check("R17 transport returns parsed dict keyed by custom_id",
                  res.get("lr:evoked:curiosity:0:matched", {}).get("x") == 1, f"res={res}")
            raws = list(Path(td).glob("*output-file-r17*"))
            check("R17 raw batch output persisted by the transport", len(raws) == 1,
                  f"files={[p.name for p in Path(td).iterdir()]}")
    except Exception as e:
        check("R16/R17 raw persistence", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R18: run_rescore end-to-end with a fake transport
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        def _fake_transport(records, endpoint="/v1/completions"):
            out = {}
            for rec in records:
                prompt = rec["body"]["prompt"]
                marker = "<|end_header_id|>\n\n"
                stream = prompt[prompt.rfind(marker) + len(marker):]
                # per-context distinct lps derived from context name
                ctx = rec["custom_id"].rsplit(":", 1)[1]
                base = -1.0 if ctx == "matched" else (-2.0 if ctx == "neutral" else -3.0)
                out[rec["custom_id"]] = {"prompt": [{"logprobs": {
                    "tokens": ["CTX "] + [stream],
                    "token_logprobs": [None, base],
                }}]}
            return out

        with tempfile.TemporaryDirectory() as td:
            res18 = RE.run_rescore(
                cfg=RE.CONFIG_A, out_dir=td, transport=_fake_transport,
                streams=STREAMS2, strong_system=STRONG, gen_prompt=GEN_PROMPT,
                all_concepts=CONCEPTS3)
            rec_path = Path(td) / "rescore_lr_records.json"
            meta_path = Path(td) / "rescore_meta.json"
            check("R18 rescore_lr_records.json written", rec_path.exists())
            check("R18 rescore_meta.json written", meta_path.exists())
            if rec_path.exists():
                recs18 = json.loads(rec_path.read_text())
                check("R18 one record per stream", len(recs18) == 2, f"n={len(recs18)}")
                if recs18:
                    check("R18 records carry all context lls",
                          len(recs18[0].get("context_lls", {})) == 4,
                          f"keys={sorted(recs18[0].get('context_lls', {}).keys())}")
                    check("R18 lr = matched - neutral = 1.0",
                          abs(recs18[0]["lr"] - 1.0) < 1e-9, f"lr={recs18[0]['lr']}")
            check("R18 return dict carries lr_records + meta",
                  isinstance(res18.get("lr_records"), list) and "meta" in res18)
    except Exception as e:
        check("R18 run_rescore end-to-end", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R19: pinned config objects
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        A, B = RE.CONFIG_A, RE.CONFIG_B
        check("R19 A model is Llama-3.3-70B Turbo",
              A.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo", f"model={A.model}")
        check("R19 A transport together-batch-text", A.transport == "together-batch-text")
        check("R19 A renderer is the hand-pinned llama3 template",
              A.renderer == "llama3-manual", f"renderer={A.renderer}")
        check("R19 A defaults to the EXISTING scout streams file (re-score, no regen)",
              str(A.streams_file).endswith("runs/llama70b_scout/streams_llama70b.json"),
              f"streams={A.streams_file}")
        check("R19 A streams file exists on disk", Path(A.streams_file).exists())
        check("R19 B model is Qwen2.5-72B", B.model == "Qwen/Qwen2.5-72B-Instruct")
        check("R19 B transport vllm-ids", B.transport == "vllm-ids")
        check("R19 B renderer tokenizer (token-ids path)", B.renderer == "tokenizer")
        C = RE.config_c("Qwen/Qwen2.5-14B-Instruct")
        check("R19 config_c = B with a different model slug",
              C.transport == B.transport and C.model == "Qwen/Qwen2.5-14B-Instruct"
              and C.tokenizer_id == "Qwen/Qwen2.5-14B-Instruct")
        C32 = RE.config_c("Qwen/Qwen2.5-32B-Instruct")
        check("R19 config_c accepts the 32B slug", C32.model == "Qwen/Qwen2.5-32B-Instruct")
    except Exception as e:
        check("R19 config objects", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R20: --dry against the REAL streams file, provably no network
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        _orig_urlopen = urllib.request.urlopen

        def _no_network(*a, **k):
            raise AssertionError("NETWORK CALL ATTEMPTED during --dry")

        urllib.request.urlopen = _no_network
        try:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dry = RE.dry_run(RE.CONFIG_A)
            printed = buf.getvalue()
        finally:
            urllib.request.urlopen = _orig_urlopen

        check("R20 dry_run returns a dict", isinstance(dry, dict))
        check("R20 10,530 requests (810 streams x 13 contexts)",
              dry.get("n_requests") == 10530, f"n_requests={dry.get('n_requests')}")
        check("R20 n_streams == 810", dry.get("n_streams") == 810,
              f"n_streams={dry.get('n_streams')}")
        cost = dry.get("est_cost_usd")
        check("R20 cost estimate in a sane band (0.5, 10) USD",
              isinstance(cost, float) and 0.5 < cost < 10.0, f"cost={cost}")
        check("R20 request count printed", "10530" in printed.replace(",", ""),
              f"printed={printed[:300]!r}")
        check("R20 no network call was made (urlopen never hit)", True)
    except AssertionError as e:
        check("R20 dry_run made a NETWORK call", False, str(e))
    except Exception as e:
        check("R20 dry_run", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R21: spend guard — no --dry, no --i-understand-spend => exit BEFORE key/client
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        key_calls: List = []
        _orig_load_key = RE.SCOUT.load_together_key

        def _spy_key(*a, **k):
            key_calls.append(1)
            raise AssertionError("load_together_key called despite spend guard")

        RE.SCOUT.load_together_key = _spy_key
        try:
            try:
                RE.main(["--config", "A"])
                check("R21 spend guard exits", False, "main returned without exiting")
            except SystemExit as e:
                check("R21 spend guard exits with code 2", e.code == 2, f"code={e.code}")
            except AssertionError as e:
                check("R21 spend guard exits", False, f"guard too late: {e}")
        finally:
            RE.SCOUT.load_together_key = _orig_load_key
        check("R21 no key load before the guard", len(key_calls) == 0,
              f"key_calls={key_calls}")
    except Exception as e:
        check("R21 spend guard", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R22: main --dry end-to-end
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        import io, contextlib
        buf22 = io.StringIO()
        with contextlib.redirect_stdout(buf22):
            out22 = RE.main(["--config", "A", "--dry"])
        check("R22 main --dry returns the dry dict",
              isinstance(out22, dict) and out22.get("n_requests") == 10530,
              f"out={out22}")
    except SystemExit as e:
        check("R22 main --dry", e.code in (0, None), f"SystemExit code={e.code}")
    except Exception as e:
        check("R22 main --dry", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# R23: run_rescore_ids (config B/C semantics) with fake vLLM client + tokenizer
# ---------------------------------------------------------------------------
if RE is not None:
    try:
        class _FakeTok23:
            eos_token_id = 99
            def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
                return [1, 2, 3]
            def __call__(self, text, add_special_tokens=False):
                class R:  # noqa
                    input_ids = [7, 8]
                return R()

        class _FakeVllm:
            def __init__(self):
                self.calls: List = []
            def completions(self, prompt_ids, prompt_logprobs=1, **kw):
                self.calls.append(list(prompt_ids))
                # entry i keyed by the provided id; index 0 null
                plps = [None] + [{str(t): -1.5} for t in prompt_ids[1:]]
                return {"prompt_logprobs": plps}

        # last stream token IS the eos (99): eos-free accounting must drop it
        streams23 = [dict(STREAMS2[0], token_ids=[7, 8, 99])]
        fake_vllm = _FakeVllm()
        with tempfile.TemporaryDirectory() as td:
            res23 = RE.run_rescore_ids(
                cfg=RE.CONFIG_B, out_dir=td, client=fake_vllm, tok=_FakeTok23(),
                streams=streams23, strong_system=STRONG, gen_prompt=GEN_PROMPT,
                all_concepts=CONCEPTS3)
            recs23 = res23["lr_records"]
            check("R23 one record", len(recs23) == 1, f"n={len(recs23)}")
            check("R23 13-context semantics on 3 concepts -> 4 vLLM calls",
                  len(fake_vllm.calls) == 4, f"calls={len(fake_vllm.calls)}")
            if recs23:
                r23 = recs23[0]
                check("R23 eos-free: trailing eos token dropped from span (2 not 3)",
                      r23["n_matched_tokens"] == 2, f"n={r23['n_matched_tokens']}")
                check("R23 same record schema as config A",
                      {"lr", "span_lps", "neutral_span_lps", "context_lls",
                       "context_span_lps"}.issubset(r23.keys()),
                      f"keys={sorted(r23.keys())}")
                check("R23 lr = 0 for symmetric fake", abs(r23["lr"]) < 1e-9,
                      f"lr={r23['lr']}")
            check("R23 records file written",
                  (Path(td) / "rescore_lr_records.json").exists())
    except Exception as e:
        check("R23 run_rescore_ids", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
passed = sum(1 for _, ok, _ in checks if ok)
failed = sum(1 for _, ok, _ in checks if not ok)

print(f"\n{'='*60}")
print(f"test_rescore_echo.py: {passed} passed, {failed} failed")
print(f"{'='*60}")

for name, ok, note in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {note}" if note else ""))

sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
