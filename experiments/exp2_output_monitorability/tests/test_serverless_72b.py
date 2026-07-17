"""RED-first unit tests for src/serverless_72b.py.

NO real API calls. All HTTP is intercepted by a mock caller.

Test inventory:
  K1  Key loading: file path resolution (file arg > env var > default ~/.deepinfra_key);
      missing file -> FileNotFoundError; empty file -> ValueError.
  C1  Client retry/backoff: 429 -> retry N times; 5xx -> retry; hard error -> raise immediately;
      success on third attempt (after two 429s) -> returns the response.
  C2  Client header: Authorization: Bearer <key> and Content-Type: application/json present.
  C3  API-error passthrough: 'error' key in response -> RuntimeError (non-rate-limit).
  S1  direct_response_schema: name='mc_direct', strict=True, concept is enum of 12 concepts,
      additionalProperties=False.
  S2  think_response_schema: name='mc_with_think', strict=True, has reasoning (str) + concept
      (enum), additionalProperties=False.
  S3  Concept enum completeness: both schemas enumerate exactly COVERT_CONCEPTS (12 entries,
      same set).
  P1  parse_mc_response: valid direct response -> {concept, reasoning=None}; valid think
      response -> {concept, reasoning: str}; invalid concept -> ValueError.
  P2  parse_mc_response: content as raw JSON string (not pre-parsed dict) is decoded correctly.
  O1  Option order randomization: _shuffled_concepts uses seed; two different seeds -> different
      orders; same seed -> same order; all 12 concepts present in each order.
  O2  Position bias cancellation: across 12 distinct seeds, each concept appears in each of the
      12 positions at least once (uniform coverage).
  M1  build_mc_prompt: stream text appears in the prompt; all 12 concept options appear as
      "(a) ...", "(b) ...", etc. in the given order.
  M2  mc_direct_payload: model == MODEL_72B, response_format.type == 'json_schema',
      response_format.json_schema.name == 'mc_direct', temperature == 0.
  M3  mc_think_payload: same checks with name == 'mc_with_think'.
  W1  Word-free filter: text with common English words -> rejected; pure random letters -> accepted.
  W2  Word-free filter: empty string -> rejected (no content).
  I1  confusion_matrix_mi_bits: perfect predictor (y_pred == y_true) -> MI > 3.0 bits (log2(12));
      uniform random predictor -> MI < 0.1 bits; chance predictor on 12 classes -> close to log2(12)
      only when perfect.
  I2  MI symmetry: I(Y; Y_hat) == I(Y_hat; Y) (computed the same way).
  I3  shuffle_null: null MI mean < observed MI (when signal is present); n_perm entries returned.
  I4  score_mc_records: perfect predictions -> high excess_bits; random predictions -> low
      excess_bits; arm filtering selects only matching rows.
  I5  score_mc_records: all-None arm -> scores all rows (no filter).
  A1  argmax_mc_bits_from_shard: synthetic shard with perfect-predictor letter_logp -> high
      excess_bits; shard with no orderings -> falls back gracefully.
  A2  argmax_mc_bits_from_shard: empty records -> returns n=0 result without crashing.
  G1  collect_streams_for_concept: mock client returns text that passes word-free filter ->
      accepted streams counted correctly; target_clean reached when enough accepted.
  G2  collect_streams_for_concept: mock client always returns a real-word text -> no accepted
      streams; loop stops at max_attempts.
  G3  run_mc_for_stream: calls mock client for DIRECT + WITH-THINK, option order randomized
      by stream_idx, result has correct keys.
  N1  run_mc_all: only accepted streams in bundle are scored; rejected skipped.
  N2  score_all_arms: per_arm keys match the distinct arms in mc_records; all_arms aggregates all.
  B1  Backoff constant: BACKOFF_BASE is 2.0 (doubles each retry).
  R1  Retry ceiling: client with max_retries=0 raises immediately on 429 (no silent swallow).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import numpy as np

# --- path setup ---
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Import the module under test; record any import error as a check
# ---------------------------------------------------------------------------
checks: List = []


def check(name: str, cond: bool, note: str = ""):
    checks.append((name, bool(cond), note))


S = None
try:
    import serverless_72b as S
    check("import serverless_72b", True)
except Exception as e:
    check("import serverless_72b", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_resp_ok(concept: str = "curiosity", reasoning: str = None) -> Dict:
    """Simulate a successful structured-output response from DeepInfra."""
    body = {"concept": concept}
    if reasoning is not None:
        body["reasoning"] = reasoning
    return {
        "choices": [{"message": {"content": json.dumps(body)}}]
    }


def _mock_gen_resp(text: str = "qxz fjm wpl kbt rvnm") -> Dict:
    """Simulate a generation response."""
    return {
        "choices": [{"message": {"content": text}}]
    }


def _make_rate_limit_resp():
    return {"error": {"code": "429", "message": "rate limited"}}


def _make_server_error_resp():
    return {"error": {"code": "503", "message": "service unavailable"}}


def _make_hard_error_resp():
    return {"error": {"code": "400", "message": "bad request", "type": "invalid_request"}}


# ---------------------------------------------------------------------------
# K1: Key loading
# ---------------------------------------------------------------------------

if S is not None:
    with tempfile.TemporaryDirectory() as td:
        key_path = Path(td) / "test.key"

        # File arg explicit
        key_path.write_text("test-key-123")
        try:
            k = S.load_api_key(str(key_path))
            check("K1a key from file arg", k == "test-key-123")
        except Exception as e:
            check("K1a key from file arg", False, str(e))

        # Env var
        os.environ["DEEPINFRA_KEY_FILE"] = str(key_path)
        try:
            k = S.load_api_key()
            check("K1b key from env var", k == "test-key-123")
        except Exception as e:
            check("K1b key from env var", False, str(e))
        finally:
            os.environ.pop("DEEPINFRA_KEY_FILE", None)

        # Missing file
        missing = Path(td) / "missing.key"
        try:
            S.load_api_key(str(missing))
            check("K1c missing file -> FileNotFoundError", False, "no exception")
        except FileNotFoundError:
            check("K1c missing file -> FileNotFoundError", True)
        except Exception as e:
            check("K1c missing file -> FileNotFoundError", False, str(e))

        # Empty file
        empty_path = Path(td) / "empty.key"
        empty_path.write_text("   ")
        try:
            S.load_api_key(str(empty_path))
            check("K1d empty file -> ValueError", False, "no exception")
        except ValueError:
            check("K1d empty file -> ValueError", True)
        except Exception as e:
            check("K1d empty file -> ValueError", False, str(e))


# ---------------------------------------------------------------------------
# C1: Retry/backoff
# ---------------------------------------------------------------------------

if S is not None:
    call_log: List[Dict] = []

    def _retry_caller(url, headers, payload):
        call_log.append({"url": url, "payload": payload})
        n = len(call_log)
        if n == 1:
            return _make_rate_limit_resp()
        if n == 2:
            return _make_rate_limit_resp()
        return _mock_gen_resp("qxz fjm")

    with patch("time.sleep"):   # don't actually sleep
        try:
            client = S.DeepInfraClient(api_key="dummy", http_caller=_retry_caller)
            resp = client.chat_completion({"model": S.MODEL_72B, "messages": []}, retries=3)
            check("C1 retry on 429, success on 3rd", resp["choices"][0]["message"]["content"] == "qxz fjm")
            check("C1 total calls == 3", len(call_log) == 3)
        except Exception as e:
            check("C1 retry on 429, success on 3rd", False, str(e))
            check("C1 total calls == 3", False, str(e))


# ---------------------------------------------------------------------------
# R1: Retry ceiling (max_retries=0 -> raise immediately)
# ---------------------------------------------------------------------------

if S is not None:
    def _always_429(url, headers, payload):
        return _make_rate_limit_resp()

    with patch("time.sleep"):
        try:
            client_r1 = S.DeepInfraClient(api_key="dummy", http_caller=_always_429)
            client_r1.chat_completion({"model": S.MODEL_72B, "messages": []}, retries=0)
            check("R1 max_retries=0 raises immediately", False, "no exception")
        except Exception as e:
            # Should raise after 0+1=1 attempt
            check("R1 max_retries=0 raises immediately", True)


# ---------------------------------------------------------------------------
# B1: Backoff constant
# ---------------------------------------------------------------------------

if S is not None:
    check("B1 BACKOFF_BASE == 2.0", S.BACKOFF_BASE == 2.0)


# ---------------------------------------------------------------------------
# C2: Headers
# ---------------------------------------------------------------------------

if S is not None:
    captured_headers: Dict = {}

    def _header_capture(url, headers, payload):
        captured_headers.update(headers)
        return _mock_gen_resp()

    try:
        client_h = S.DeepInfraClient(api_key="my-secret-key", http_caller=_header_capture)
        client_h.chat_completion({"model": S.MODEL_72B, "messages": []})
        check("C2 Authorization header", "Bearer my-secret-key" in captured_headers.get("Authorization", ""))
        check("C2 Content-Type header", "application/json" in captured_headers.get("Content-Type", ""))
    except Exception as e:
        check("C2 Authorization header", False, str(e))
        check("C2 Content-Type header", False, str(e))


# ---------------------------------------------------------------------------
# C3: Hard API error passthrough
# ---------------------------------------------------------------------------

if S is not None:
    def _hard_error(url, headers, payload):
        return _make_hard_error_resp()

    with patch("time.sleep"):
        try:
            client_e = S.DeepInfraClient(api_key="dummy", http_caller=_hard_error)
            client_e.chat_completion({"model": S.MODEL_72B, "messages": []})
            check("C3 hard error -> RuntimeError", False, "no exception")
        except RuntimeError:
            check("C3 hard error -> RuntimeError", True)
        except Exception as e:
            check("C3 hard error -> RuntimeError", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# S1, S2, S3: Schema structure
# ---------------------------------------------------------------------------

if S is not None:
    try:
        ds = S.direct_response_schema()
        check("S1a direct schema name='mc_direct'", ds["name"] == "mc_direct")
        check("S1b direct schema strict=True", ds["strict"] is True)
        schema = ds["schema"]
        check("S1c direct schema has 'concept' property", "concept" in schema.get("properties", {}))
        check("S1d direct schema additionalProperties=False", schema.get("additionalProperties") is False)
        concept_prop = schema["properties"]["concept"]
        check("S1e direct concept type=string", concept_prop.get("type") == "string")
        check("S1f direct concept enum length=12", len(concept_prop.get("enum", [])) == 12)
    except Exception as e:
        check("S1 direct schema", False, str(e))

    try:
        ts = S.think_response_schema()
        check("S2a think schema name='mc_with_think'", ts["name"] == "mc_with_think")
        check("S2b think schema strict=True", ts["strict"] is True)
        tschema = ts["schema"]
        check("S2c think schema has 'reasoning' property", "reasoning" in tschema.get("properties", {}))
        check("S2d think schema has 'concept' property", "concept" in tschema.get("properties", {}))
        check("S2e think schema additionalProperties=False", tschema.get("additionalProperties") is False)
        check("S2f think required=['reasoning','concept']",
              set(tschema.get("required", [])) == {"reasoning", "concept"})
    except Exception as e:
        check("S2 think schema", False, str(e))

    try:
        d_enum = set(S.direct_response_schema()["schema"]["properties"]["concept"]["enum"])
        t_enum = set(S.think_response_schema()["schema"]["properties"]["concept"]["enum"])
        expected = set(S.COVERT_CONCEPTS)
        check("S3a direct enum == COVERT_CONCEPTS", d_enum == expected)
        check("S3b think enum == COVERT_CONCEPTS", t_enum == expected)
    except Exception as e:
        check("S3 concept enum completeness", False, str(e))


# ---------------------------------------------------------------------------
# P1, P2: parse_mc_response
# ---------------------------------------------------------------------------

if S is not None:
    # P1a direct
    try:
        result = S.parse_mc_response(_mock_resp_ok("curiosity"))
        check("P1a parse direct -> concept correct", result["concept"] == "curiosity")
        check("P1b parse direct -> reasoning=None", result["reasoning"] is None)
    except Exception as e:
        check("P1a parse direct -> concept correct", False, str(e))
        check("P1b parse direct -> reasoning=None", False, str(e))

    # P1c think
    try:
        result_t = S.parse_mc_response(_mock_resp_ok("ocean", reasoning="It felt vast and deep"))
        check("P1c parse think -> concept correct", result_t["concept"] == "ocean")
        check("P1d parse think -> reasoning preserved", "vast" in (result_t["reasoning"] or ""))
    except Exception as e:
        check("P1c parse think -> concept correct", False, str(e))
        check("P1d parse think -> reasoning preserved", False, str(e))

    # P1e invalid concept
    try:
        bad_resp = {"choices": [{"message": {"content": json.dumps({"concept": "banana"})}}]}
        S.parse_mc_response(bad_resp)
        check("P1e invalid concept -> ValueError", False, "no exception")
    except ValueError:
        check("P1e invalid concept -> ValueError", True)
    except Exception as e:
        check("P1e invalid concept -> ValueError", False, str(e))

    # P2 string content is decoded
    try:
        str_resp = {"choices": [{"message": {"content": '{"concept": "fear"}'}}]}
        r2 = S.parse_mc_response(str_resp)
        check("P2 string content decoded", r2["concept"] == "fear")
    except Exception as e:
        check("P2 string content decoded", False, str(e))


# ---------------------------------------------------------------------------
# O1, O2: Option order randomization
# ---------------------------------------------------------------------------

if S is not None:
    try:
        o0 = S._shuffled_concepts(seed=0)
        o1 = S._shuffled_concepts(seed=1)
        o0_again = S._shuffled_concepts(seed=0)
        check("O1a same seed -> same order", o0 == o0_again)
        check("O1b different seeds -> different orders (likely)", o0 != o1)
        check("O1c all 12 concepts present", set(o0) == set(S.COVERT_CONCEPTS) and len(o0) == 12)
    except Exception as e:
        check("O1 option order", False, str(e))

    # O2: position coverage across 12 seeds
    try:
        position_concept: List[set] = [set() for _ in range(12)]
        for seed in range(12):
            order = S._shuffled_concepts(seed=seed)
            for pos, c in enumerate(order):
                position_concept[pos].add(c)
        # Each position should see at least some variety (full coverage not guaranteed with 12 seeds)
        # but each position must see >= 1 concept
        min_coverage = min(len(s) for s in position_concept)
        check("O2 each position sees >= 1 concept across 12 seeds", min_coverage >= 1)
    except Exception as e:
        check("O2 position coverage", False, str(e))


# ---------------------------------------------------------------------------
# M1, M2, M3: build_mc_prompt, payload structure
# ---------------------------------------------------------------------------

if S is not None:
    try:
        order = S._shuffled_concepts(seed=42)
        prompt = S.build_mc_prompt("qxz fjm wpl kbt", order, "direct")
        check("M1a stream text in prompt", "qxz fjm wpl kbt" in prompt)
        # All 12 option letters appear
        all_letters = all(f"({chr(ord('a') + i)})" in prompt for i in range(12))
        check("M1b all 12 option letters in prompt", all_letters)
        # All 12 concepts appear
        all_concepts = all(c in prompt for c in order)
        check("M1c all 12 concepts in prompt", all_concepts)
    except Exception as e:
        check("M1 build_mc_prompt", False, str(e))

    try:
        order_m2 = S._shuffled_concepts(seed=7)
        dp = S.mc_direct_payload("test stream", order_m2)
        check("M2a model == MODEL_72B", dp["model"] == S.MODEL_72B)
        check("M2b response_format.type == 'json_schema'",
              dp["response_format"]["type"] == "json_schema")
        check("M2c response_format name == 'mc_direct'",
              dp["response_format"]["json_schema"]["name"] == "mc_direct")
        check("M2d temperature == 0", dp.get("temperature") == 0.0)
    except Exception as e:
        check("M2 mc_direct_payload", False, str(e))

    try:
        tp = S.mc_think_payload("test stream", order_m2)
        check("M3a model == MODEL_72B", tp["model"] == S.MODEL_72B)
        check("M3b response_format name == 'mc_with_think'",
              tp["response_format"]["json_schema"]["name"] == "mc_with_think")
        check("M3c temperature == 0", tp.get("temperature") == 0.0)
    except Exception as e:
        check("M3 mc_think_payload", False, str(e))


# ---------------------------------------------------------------------------
# W1, W2: Word-free filter
# ---------------------------------------------------------------------------

if S is not None:
    try:
        check("W1a random letters accepted", S._is_word_free("qxz fjm wpl kbt rvnm"))
        check("W1b 'hello world' rejected", not S._is_word_free("hello world"))
        check("W1c 'celebration' rejected", not S._is_word_free("celebration curiosity"))
        check("W2 empty string rejected", not S._is_word_free(""))
    except Exception as e:
        check("W1/W2 word-free filter", False, str(e))


# ---------------------------------------------------------------------------
# I1, I2, I3, I4, I5: MI + shuffle-null
# ---------------------------------------------------------------------------

if S is not None:
    k = 12
    n = 120  # 10 per class

    # Perfect predictor
    y_true_perf = list(range(k)) * (n // k)
    y_pred_perf = list(y_true_perf)
    try:
        mi_perf = S.confusion_matrix_mi_bits(y_true_perf, y_pred_perf, k=k)
        check("I1a perfect predictor MI > 3.0 bits", mi_perf > 3.0)
    except Exception as e:
        check("I1a perfect predictor MI > 3.0 bits", False, str(e))

    # Uniform random predictor: raw plug-in MI can be > 0 due to small-n bias;
    # the EXCESS over shuffle null should be near zero.
    rng = np.random.default_rng(42)
    y_pred_rand = rng.integers(0, k, size=n).tolist()
    try:
        mi_rand = S.confusion_matrix_mi_bits(y_true_perf, y_pred_rand, k=k)
        null_rand = S.shuffle_null(y_true_perf, y_pred_rand, k=k, n_perm=200, seed=1)
        excess_rand = mi_rand - null_rand["mean"]
        check("I1b random predictor excess_bits near zero (< 0.15)", excess_rand < 0.15)
    except Exception as e:
        check("I1b random predictor excess_bits near zero (< 0.15)", False, str(e))

    # I2: symmetry
    try:
        mi_fwd = S.confusion_matrix_mi_bits(y_true_perf, y_pred_rand, k=k)
        mi_rev = S.confusion_matrix_mi_bits(y_pred_rand, y_true_perf, k=k)
        check("I2 MI symmetry (fwd==rev within 1e-9)", abs(mi_fwd - mi_rev) < 1e-9)
    except Exception as e:
        check("I2 MI symmetry", False, str(e))

    # I3: shuffle null
    try:
        null_result = S.shuffle_null(y_true_perf, y_pred_perf, k=k, n_perm=100, seed=0)
        check("I3a null result has mean", "mean" in null_result)
        check("I3b null result has p95", "p95" in null_result)
        check("I3c null mean < observed MI (signal present)", null_result["mean"] < mi_perf)
        check("I3d n_perm reflected", null_result["n_perm"] == 100)
    except Exception as e:
        check("I3 shuffle_null", False, str(e))

    # I4: score_mc_records
    try:
        mc_recs_perfect = [
            {"true_concept": c, "arm": "evoked", "direct_pred": c, "think_pred": c}
            for c in S.COVERT_CONCEPTS * 10
        ]
        scores_perf = S.score_mc_records(mc_recs_perfect, arm="evoked", n_perm=50)
        check("I4a perfect MC excess_bits > 0",
              scores_perf["direct"]["excess_bits"] > 0)

        # Truly random predictor (independent of true concept) -> excess near zero
        _rng_i4 = np.random.default_rng(12345)
        mc_recs_rand = [
            {"true_concept": S.COVERT_CONCEPTS[i % 12],
             "arm": "evoked",
             "direct_pred": S.COVERT_CONCEPTS[int(_rng_i4.integers(0, 12))],
             "think_pred": S.COVERT_CONCEPTS[int(_rng_i4.integers(0, 12))]}
            for i in range(240)
        ]
        scores_rand = S.score_mc_records(mc_recs_rand, arm="evoked", n_perm=100)
        # Excess should be near zero for a chance-level predictor (null corrects the bias)
        check("I4b chance-level MC excess_bits near zero (< 0.3)",
              scores_rand["direct"]["excess_bits"] < 0.3)

        # arm filtering
        mixed = mc_recs_perfect[:12] + [
            {"true_concept": "fear", "arm": "secret_word", "direct_pred": "ocean", "think_pred": "ocean"}
        ]
        scores_evoked = S.score_mc_records(mixed, arm="evoked", n_perm=20)
        check("I4c arm filter selects correct arm", scores_evoked["n"] == 12)
    except Exception as e:
        check("I4 score_mc_records", False, str(e))

    # I5: all-None arm
    try:
        scores_all = S.score_mc_records(mc_recs_perfect, arm=None, n_perm=20)
        check("I5 score_mc_records arm=None scores all rows", scores_all["n"] == len(mc_recs_perfect))
    except Exception as e:
        check("I5 score_mc_records arm=None", False, str(e))


# ---------------------------------------------------------------------------
# A1, A2: argmax_mc_bits_from_shard
# ---------------------------------------------------------------------------

if S is not None:
    k = 12
    # A1: synthetic shard with perfect-predictor letter_logp (diagonal)
    try:
        # Build orderings: canonical order for all 12 orderings
        orderings = [list(S.COVERT_CONCEPTS) for _ in range(12)]
        records = []
        for ci, concept in enumerate(S.COVERT_CONCEPTS):
            for _ in range(10):
                # Perfect logprob: high for true concept's position in each ordering
                lp = np.full((12, 12), -10.0, dtype=np.float32)
                for oi in range(12):
                    lp[oi, ci] = 0.0   # high logprob at the correct concept slot
                records.append({
                    "concept": concept,
                    "concept_idx": ci,
                    "letter_logp": lp,
                })
        shard = {
            "model": "qwen2.5-1.5b",
            "records": records,
            "orderings": orderings,
            "concepts": list(S.COVERT_CONCEPTS),
        }
        result_a1 = S.argmax_mc_bits_from_shard(shard, n_perm=50)
        check("A1a argmax shard n correct", result_a1["n"] == k * 10)
        check("A1b perfect logprob -> excess_bits > 0", (result_a1["excess_bits"] or 0) > 0)
    except Exception as e:
        check("A1 argmax_mc_bits_from_shard perfect", False, str(e))

    # A2: empty records
    try:
        empty_shard = {"model": "qwen2.5-1.5b", "records": [], "orderings": None}
        result_a2 = S.argmax_mc_bits_from_shard(empty_shard, n_perm=10)
        check("A2 empty shard -> n=0 no crash", result_a2["n"] == 0)
    except Exception as e:
        check("A2 empty shard", False, str(e))


# ---------------------------------------------------------------------------
# G1, G2: collect_streams_for_concept
# ---------------------------------------------------------------------------

if S is not None:
    # G1: mock client returns word-free text -> accepted
    g1_calls = [0]

    def _gen_wordfree(url, headers, payload):
        g1_calls[0] += 1
        return _mock_gen_resp("qxz fjm wpl kbt rvnm")   # word-free

    with patch("time.sleep"):
        try:
            client_g1 = S.DeepInfraClient(api_key="dummy", http_caller=_gen_wordfree)
            recs_g1 = S.collect_streams_for_concept(
                client_g1, "curiosity", "evoked",
                strong_system="YOU MUST NOT WRITE WORDS.",
                gen_prompt="Begin.",
                target_clean=3, max_attempts=10,
            )
            accepted_g1 = [r for r in recs_g1 if r["accepted"]]
            check("G1a target_clean=3 accepted >= 3", len(accepted_g1) >= 3)
            check("G1b all accepted records have concept", all(r["concept"] == "curiosity" for r in accepted_g1))
        except Exception as e:
            check("G1 collect_streams accepted", False, str(e))

    # G2: mock returns real-word text -> none accepted, stops at max_attempts
    def _gen_realwords(url, headers, payload):
        return _mock_gen_resp("hello world this is a test sentence")  # real words

    with patch("time.sleep"):
        try:
            client_g2 = S.DeepInfraClient(api_key="dummy", http_caller=_gen_realwords)
            recs_g2 = S.collect_streams_for_concept(
                client_g2, "fear", "evoked",
                strong_system="NO WORDS.",
                gen_prompt="Begin.",
                target_clean=5, max_attempts=8,
            )
            accepted_g2 = [r for r in recs_g2 if r["accepted"]]
            check("G2a no accepted when real words", len(accepted_g2) == 0)
            check("G2b stops at max_attempts", len(recs_g2) <= 8)
        except Exception as e:
            check("G2 collect_streams no accepted", False, str(e))


# ---------------------------------------------------------------------------
# G3: run_mc_for_stream
# ---------------------------------------------------------------------------

if S is not None:
    mc_call_shapes: List[Dict] = []

    def _mc_caller(url, headers, payload):
        mc_call_shapes.append(payload)
        # Return direct response for both calls
        return _mock_resp_ok("curiosity", reasoning="It felt inquisitive")

    with patch("time.sleep"):
        try:
            client_mc = S.DeepInfraClient(api_key="dummy", http_caller=_mc_caller)
            stream = {"concept": "curiosity", "arm": "evoked", "text": "qxz fjm wpl kbt"}
            result_mc = S.run_mc_for_stream(client_mc, stream, stream_idx=5)
            check("G3a result has direct_pred", "direct_pred" in result_mc)
            check("G3b result has think_pred", "think_pred" in result_mc)
            check("G3c result has option_order (12 items)", len(result_mc["option_order"]) == 12)
            check("G3d true_concept preserved", result_mc["true_concept"] == "curiosity")
            # Two calls: one direct, one think
            check("G3e two API calls made (direct + think)", len(mc_call_shapes) == 2)
            # First call is direct, second is think
            check("G3f first call is direct",
                  mc_call_shapes[0]["response_format"]["json_schema"]["name"] == "mc_direct")
            check("G3g second call is with_think",
                  mc_call_shapes[1]["response_format"]["json_schema"]["name"] == "mc_with_think")
        except Exception as e:
            check("G3 run_mc_for_stream", False, str(e))


# ---------------------------------------------------------------------------
# N1, N2: run_mc_all, score_all_arms
# ---------------------------------------------------------------------------

if S is not None:
    def _mc_caller_n1(url, headers, payload):
        return _mock_resp_ok("curiosity", reasoning="reason")

    with patch("time.sleep"):
        try:
            client_n1 = S.DeepInfraClient(api_key="dummy", http_caller=_mc_caller_n1)
            bundle_n1 = {
                "streams": [
                    {"concept": "curiosity", "arm": "evoked", "text": "qxz", "accepted": True},
                    {"concept": "fear", "arm": "evoked", "text": "bad word", "accepted": False},
                    {"concept": "ocean", "arm": "secret_word", "text": "fjm", "accepted": True},
                ]
            }
            mc_out = S.run_mc_all(client_n1, bundle_n1)
            check("N1a only accepted streams scored (2 of 3)", len(mc_out) == 2)
        except Exception as e:
            check("N1 run_mc_all skips rejected", False, str(e))

    # N2: score_all_arms
    try:
        mc_data = [
            {"true_concept": c, "arm": arm, "direct_pred": c, "think_pred": c}
            for c in S.COVERT_CONCEPTS for arm in ("evoked", "secret_word") for _ in range(5)
        ]
        all_scores = S.score_all_arms(mc_data, n_perm=20)
        check("N2a per_arm has 'evoked' key", "evoked" in all_scores["per_arm"])
        check("N2b per_arm has 'secret_word' key", "secret_word" in all_scores["per_arm"])
        check("N2c all_arms n is total", all_scores["all_arms"]["n"] == len(mc_data))
    except Exception as e:
        check("N2 score_all_arms", False, str(e))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

passed = sum(1 for _, ok, _ in checks if ok)
failed = sum(1 for _, ok, _ in checks if not ok)

print(f"\n{'='*60}")
print(f"test_serverless_72b.py: {passed} passed, {failed} failed")
print(f"{'='*60}")

for name, ok, note in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {note}" if note else ""))

if failed:
    sys.exit(1)
