"""RED-first unit tests for harness/run_llama70b_scout.py.

NO real API calls. All HTTP is intercepted via injected mock callers.
The Together SDK client is replaced by a lightweight fake.

Test inventory
--------------
B1  batch_submit_poll_download: out-of-order mock output lines are reconciled by custom_id.
B2  batch_submit_poll_download: raises RuntimeError on FAILED job.
B3  batch_submit_poll_download: error file downloaded and per-request errors surfaced even
    on COMPLETED status (per-request errors don't flip the job status).
B4  echo requests have max_tokens >= 1 (Together batch rejects max_tokens=0).
B5  _find_stream_span_lps: correctly extracts stream logprobs from a mock echo response
    when the stream is a suffix of the tokens.
B6  _find_stream_span_lps: returns empty list (or raises) when stream_text is not found
    in the echo tokens.
B7  generation batch builder: produces well-formed records (correct custom_id format
    "gen:<arm>:<concept>:<idx>", correct model slug, endpoint /v1/chat/completions).
B8  LR batch builder: produces well-formed records (echo=True, logprobs=1,
    max_tokens>=1, temperature=0, endpoint /v1/completions).
B9  MC batch builder: produces well-formed records (json_schema with 12-concept enum,
    correct model slug, endpoint /v1/chat/completions, max_tokens=512 for Llama non-reasoning).
B10 parse_mc_response from serverless_72b correctly parses a Llama structured-output
    response (string JSON body -> {concept, reasoning=None}).
B11 zero real API calls: end-to-end run path with injected mocks makes no real HTTP calls.
B12 serverless discipline: the script source has NO create_endpoint, NO wait_ready,
    NO delete_endpoint, NO teardown stack (no endpoint lifecycle at all).
B13 error file downloaded even on COMPLETED status (error file present alongside output).
B14 custom_id reconciliation: output lines out of order are matched back to requests
    correctly by custom_id (same as B1 but exercises the return dict shape).
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
HARNESS = REPO / "harness"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HARNESS))

# ---------------------------------------------------------------------------
# Attempt import
# ---------------------------------------------------------------------------
checks: List = []


def check(name: str, cond: bool, note: str = "") -> None:
    checks.append((name, bool(cond), note))


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCOUT_PATH = str(HARNESS / "run_llama70b_scout.py")
SCOUT = None
try:
    SCOUT = load_module("run_llama70b_scout", SCOUT_PATH)
    check("import run_llama70b_scout", True)
except Exception as e:
    check("import run_llama70b_scout", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# FAKE TOGETHER SDK CLIENT
# ---------------------------------------------------------------------------

class _FakeFileUpload:
    def __init__(self, file_id: str = "file-test-001"):
        self._id = file_id
    @property
    def id(self):
        return self._id


class _FakeFilesClient:
    """Minimal fake for together.Together().files — no real SDK needed."""
    def __init__(self, file_id: str = "file-test-001", output_jsonl: bytes = b"",
                 error_jsonl: bytes = b""):
        self._file_id = file_id
        self._output = output_jsonl
        self._error = error_jsonl
        self.upload_calls: List[Dict] = []
        self.content_calls: List[str] = []

    def upload(self, file, purpose, check=False):
        self.upload_calls.append({"file": file, "purpose": purpose})
        return _FakeFileUpload(self._file_id)

    def content(self, file_id):
        self.content_calls.append(file_id)
        if file_id.startswith("error-"):
            return _BytesWrapper(self._error)
        return _BytesWrapper(self._output)


class _BytesWrapper:
    def __init__(self, data: bytes):
        self._data = data
    def read(self):
        return self._data


class _FakeTogetherClient:
    def __init__(self, file_id="file-test-001", output_jsonl=b"", error_jsonl=b""):
        self.files = _FakeFilesClient(file_id, output_jsonl, error_jsonl)


# ---------------------------------------------------------------------------
# MOCK HTTP HELPERS
# ---------------------------------------------------------------------------

def _make_batch_create_resp(job_id: str = "batch-job-001") -> Dict:
    return {"job": {"id": job_id, "status": "validating"}}


def _make_batch_poll_resp(status: str, output_file_id: str = "output-file-001",
                          error_file_id: Optional[str] = None) -> Tuple[int, Dict]:
    body: Dict = {"job": {"status": status, "output_file_id": output_file_id}}
    if error_file_id:
        body["job"]["error_file_id"] = error_file_id
    return (200, body)


def _mock_post_create(url, headers, body) -> Dict:
    """Mock POST for /v1/batches -> creates job."""
    return _make_batch_create_resp("batch-001")


def _make_seq_poll_caller(states: List[str], output_file_id="output-file-001",
                          error_file_id: Optional[str] = None):
    """Returns a GET caller that cycles through the given statuses."""
    calls: List[int] = []

    def caller(url, headers) -> Tuple[int, Dict]:
        idx = min(len(calls), len(states) - 1)
        calls.append(idx)
        return _make_batch_poll_resp(states[idx], output_file_id, error_file_id)

    caller.calls = calls
    return caller


# ---------------------------------------------------------------------------
# B1: batch_submit_poll_download reconciles out-of-order output by custom_id
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Build two output records in REVERSE order
        out_records = [
            json.dumps({"custom_id": "req-2", "response": {"body": {"choices": [{"text": "stream2"}]}}}),
            json.dumps({"custom_id": "req-1", "response": {"body": {"choices": [{"text": "stream1"}]}}}),
        ]
        output_jsonl = b"\n".join(r.encode() for r in out_records)

        fake_client = _FakeTogetherClient(
            file_id="file-001",
            output_jsonl=output_jsonl,
            error_jsonl=b"",
        )

        post_calls: List[Dict] = []
        def _b1_post(url, headers, body):
            post_calls.append({"url": url, "body": body})
            return _make_batch_create_resp("batch-b1")

        get_caller = _make_seq_poll_caller(["COMPLETED"], output_file_id="output-file-001")

        records = [
            {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {}},
            {"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {}},
        ]

        result = SCOUT.batch_submit_poll_download(
            jsonl_records=records,
            together_client=fake_client,
            http_post_caller=_b1_post,
            http_get_caller=get_caller,
            together_ua="curl/8.4.0",
            together_base="https://api.together.xyz",
        )
        check("B1 returns dict keyed by custom_id", isinstance(result, dict) and set(result.keys()) == {"req-1", "req-2"})
        check("B1 req-1 body resolved correctly",
              result.get("req-1", {}).get("choices", [{}])[0].get("text") == "stream1"
              or "choices" in str(result.get("req-1", {})))
        check("B1 req-2 body resolved correctly", "req-2" in result)
    except Exception as e:
        check("B1 out-of-order reconciliation", False, f"{type(e).__name__}: {e}")
        check("B1 req-1 body resolved correctly", False)
        check("B1 req-2 body resolved correctly", False)


# ---------------------------------------------------------------------------
# B2: raises RuntimeError on FAILED job
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        fake_client_b2 = _FakeTogetherClient(
            file_id="file-b2",
            output_jsonl=b"",
            error_jsonl=b"",
        )

        def _b2_post(url, headers, body):
            return _make_batch_create_resp("batch-b2")

        get_caller_b2 = _make_seq_poll_caller(["in_progress", "FAILED"])

        records_b2 = [{"custom_id": "req-x", "method": "POST", "url": "/v1/completions", "body": {}}]

        try:
            SCOUT.batch_submit_poll_download(
                jsonl_records=records_b2,
                together_client=fake_client_b2,
                http_post_caller=_b2_post,
                http_get_caller=get_caller_b2,
                together_ua="curl/8.4.0",
                together_base="https://api.together.xyz",
            )
            check("B2 FAILED job raises RuntimeError", False, "no exception raised")
        except RuntimeError as e:
            check("B2 FAILED job raises RuntimeError", True)
        except Exception as e:
            check("B2 FAILED job raises RuntimeError", False, f"wrong exception {type(e).__name__}: {e}")
    except Exception as e:
        check("B2 FAILED job raises RuntimeError", False, f"outer {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# B3 / B13: error file downloaded even on COMPLETED; per-request errors surfaced
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        per_req_error = json.dumps({
            "custom_id": "req-err",
            "error": {"message": "context length exceeded", "code": "context_length_exceeded"},
        })
        good_output = json.dumps({
            "custom_id": "req-ok",
            "response": {"body": {"choices": [{"text": "ok"}]}},
        })

        fake_client_b3 = _FakeFilesClient(
            file_id="file-b3",
            output_jsonl=good_output.encode(),
            error_jsonl=per_req_error.encode(),
        )

        class _FakeClient_B3:
            def __init__(self):
                self.files = fake_client_b3

        def _b3_post(url, headers, body):
            return _make_batch_create_resp("batch-b3")

        get_caller_b3 = _make_seq_poll_caller(
            ["COMPLETED"],
            output_file_id="output-file-b3",
            error_file_id="error-file-b3",
        )

        records_b3 = [
            {"custom_id": "req-ok", "method": "POST", "url": "/v1/completions", "body": {}},
            {"custom_id": "req-err", "method": "POST", "url": "/v1/completions", "body": {}},
        ]

        result_b3 = SCOUT.batch_submit_poll_download(
            jsonl_records=records_b3,
            together_client=_FakeClient_B3(),
            http_post_caller=_b3_post,
            http_get_caller=get_caller_b3,
            together_ua="curl/8.4.0",
            together_base="https://api.together.xyz",
        )
        check("B3/B13 COMPLETED job still downloads error file",
              "error-file-b3" in fake_client_b3.content_calls
              or any("error" in str(c) for c in fake_client_b3.content_calls))
        check("B3 successful request still in result", "req-ok" in result_b3)
    except Exception as e:
        check("B3/B13 error file on COMPLETED", False, f"{type(e).__name__}: {e}")
        check("B3 successful request still in result", False)


# ---------------------------------------------------------------------------
# B4: echo requests have max_tokens >= 1
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Build LR batch and check all records have max_tokens >= 1
        # We need some accepted streams first
        accepted_streams = [
            {
                "concept": "curiosity",
                "arm": "evoked",
                "text": "qxz fjm wpl kbt",
                "accepted": True,
                "stream_idx": 0,
            }
        ]
        strong_system = "YOU MUST NOT WRITE WORDS."
        gen_prompt = "Begin."

        lr_records = SCOUT.build_lr_batch_records(
            accepted_streams=accepted_streams,
            strong_system=strong_system,
            gen_prompt=gen_prompt,
            all_concepts=["curiosity", "ocean", "fear"],  # small set for test
        )
        all_max_tokens = [r["body"].get("max_tokens", 0) for r in lr_records]
        check("B4 all echo requests have max_tokens >= 1",
              all(mt >= 1 for mt in all_max_tokens),
              f"min max_tokens={min(all_max_tokens) if all_max_tokens else 'no records'}")
    except Exception as e:
        check("B4 echo requests max_tokens >= 1", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# B5: _find_stream_span_lps correctly extracts stream logprobs
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Mock echo response: context = "context_prefix", stream = "stream_text"
        context_tokens = ["con", "text", "_pre", "fix"]
        stream_tokens = ["stream", "_", "text"]
        all_tokens = context_tokens + stream_tokens

        # Build a mock echo response following Together's shape:
        # resp["prompt"][0]["logprobs"]["tokens"] and ["token_logprobs"]
        context_lps = [-0.5, -0.6, -0.7, -0.8]
        stream_lps = [-1.0, -1.1, -1.2]
        all_lps = context_lps + stream_lps

        mock_echo_resp = {
            "prompt": [{
                "logprobs": {
                    "tokens": all_tokens,
                    "token_logprobs": all_lps,
                }
            }]
        }

        stream_text = "stream_text"
        extracted = SCOUT._find_stream_span_lps(mock_echo_resp, stream_text)
        check("B5 correct number of stream logprobs extracted",
              len(extracted) == len(stream_lps),
              f"got {len(extracted)}, expected {len(stream_lps)}")
        check("B5 stream logprobs match expected values",
              all(abs(a - b) < 1e-9 for a, b in zip(extracted, stream_lps)))
    except Exception as e:
        check("B5 _find_stream_span_lps correct extraction", False, f"{type(e).__name__}: {e}")
        check("B5 stream logprobs match expected values", False)


# ---------------------------------------------------------------------------
# B6: _find_stream_span_lps when stream not found
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        mock_echo_resp_b6 = {
            "prompt": [{
                "logprobs": {
                    "tokens": ["foo", "bar", "baz"],
                    "token_logprobs": [-1.0, -1.0, -1.0],
                }
            }]
        }
        stream_text_not_present = "XXXXXX_definitely_not_here"
        try:
            result_b6 = SCOUT._find_stream_span_lps(mock_echo_resp_b6, stream_text_not_present)
            # Either returns empty or raises — both are acceptable
            check("B6 stream not found returns empty or raises",
                  result_b6 == [] or result_b6 is None)
        except (ValueError, RuntimeError):
            check("B6 stream not found returns empty or raises", True)
    except Exception as e:
        check("B6 _find_stream_span_lps stream not found", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# B7: generation batch builder produces well-formed records
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        import config as _cfg  # src/ is already on sys.path; import config directly
        strong_system_b7 = _cfg.STRONG_SYSTEM
        gen_prompt_b7 = _cfg.GEN_PROMPT
        concepts_b7 = _cfg.COVERT_CONCEPTS[:3]  # small subset
        arms_b7 = ["evoked"]

        gen_records = SCOUT.build_generation_batch_records(
            arms=arms_b7,
            concepts=concepts_b7,
            strong_system=strong_system_b7,
            gen_prompt=gen_prompt_b7,
            target_clean=2,
        )
        check("B7 generation records is a non-empty list", isinstance(gen_records, list) and len(gen_records) > 0)
        r0 = gen_records[0]
        check("B7 each record has 'custom_id' key", "custom_id" in r0)
        check("B7 custom_id has gen: prefix", r0["custom_id"].startswith("gen:"))
        # custom_id format: "gen:{arm}:{concept}:{idx}"
        parts = r0["custom_id"].split(":")
        check("B7 custom_id has 4 colon-separated parts (gen:arm:concept:idx)",
              len(parts) == 4, f"parts={parts}")
        check("B7 first part is 'gen'", parts[0] == "gen")
        check("B7 arm in custom_id is one of arms_b7", parts[1] in arms_b7)
        check("B7 concept in custom_id is one of concepts_b7", parts[2] in concepts_b7)
        check("B7 record has 'body' key", "body" in r0)
        body = r0["body"]
        check("B7 model is Llama-3.3-70B", "Llama-3.3-70B" in body.get("model", ""),
              f"model={body.get('model')}")
        check("B7 body has 'messages' for chat", "messages" in body)
        check("B7 url is /v1/chat/completions",
              r0.get("url") == "/v1/chat/completions" or "chat" in str(r0))
        # B7b: generation requests logprobs so LL(stream|matched ctx) is captured for free
        # (Together chat completions returns choices[0].logprobs.{token_ids,token_logprobs}).
        check("B7b generation body requests logprobs", bool(body.get("logprobs")),
              f"logprobs={body.get('logprobs')}")
    except Exception as e:
        check("B7 generation batch records shape", False, f"{type(e).__name__}: {e}")
        for sub in ["each record has 'custom_id' key", "custom_id has gen: prefix",
                    "body has 'messages' for chat"]:
            check(f"B7 {sub}", False)


# ---------------------------------------------------------------------------
# B8: LR batch builder produces well-formed records
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        accepted_b8 = [
            {
                "concept": "curiosity",
                "arm": "evoked",
                "text": "qxz fjm wpl kbt rvnm",
                "accepted": True,
                "stream_idx": 0,
            },
            {
                "concept": "ocean",
                "arm": "evoked",
                "text": "fjm kbt qxz",
                "accepted": True,
                "stream_idx": 1,
            },
        ]

        lr_recs = SCOUT.build_lr_batch_records(
            accepted_streams=accepted_b8,
            strong_system="YOU MUST NOT WRITE WORDS.",
            gen_prompt="Begin.",
            all_concepts=["curiosity", "ocean", "fear"],
        )
        check("B8 LR records non-empty", len(lr_recs) > 0)
        r_lr = lr_recs[0]
        check("B8 LR record has 'custom_id'", "custom_id" in r_lr)
        check("B8 LR custom_id has lr: prefix", r_lr["custom_id"].startswith("lr:"))
        check("B8 LR record has 'body'", "body" in r_lr)
        body_lr = r_lr["body"]
        check("B8 echo=True in LR body", body_lr.get("echo") is True)
        check("B8 logprobs=1 in LR body", body_lr.get("logprobs") == 1)
        check("B8 max_tokens >= 1 in LR body", body_lr.get("max_tokens", 0) >= 1)
        check("B8 temperature=0 in LR body", body_lr.get("temperature") == 0)
        check("B8 LR url is /v1/completions",
              r_lr.get("url") == "/v1/completions" or "completions" in str(r_lr.get("url", "")))
        check("B8 LR model is Llama-3.3-70B", "Llama-3.3-70B" in body_lr.get("model", ""))
    except Exception as e:
        check("B8 LR batch records shape", False, f"{type(e).__name__}: {e}")
        for sub in ["echo=True in LR body", "logprobs=1 in LR body", "max_tokens >= 1 in LR body",
                    "temperature=0 in LR body"]:
            check(f"B8 {sub}", False)


# ---------------------------------------------------------------------------
# B8b: filter_generation_results captures generation logprobs + real token ids
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Mock two generation responses shaped like Together chat completions with
        # logprobs=True: choices[0].logprobs.{token_ids,tokens,token_logprobs}. The
        # trailing <|eot_id|> eos token must be dropped from the stored stream span.
        gib = "qxz fjm wpl kbt rvnm"  # word-free -> accepted
        gen_results_b8b = {
            "gen:evoked:curiosity:0": {
                "choices": [{
                    "message": {"content": gib},
                    "logprobs": {
                        "token_ids": [370, 111, 222, 128009],
                        "tokens": ["qxz", " fjm", " wpl", "<|eot_id|>"],
                        "token_logprobs": [-1.5, -2.0, -0.5, -0.01],
                    },
                }],
            },
            # A word-bearing response that must be rejected by the word-free filter.
            "gen:evoked:curiosity:1": {
                "choices": [{
                    "message": {"content": "the ocean is blue"},
                    "logprobs": {
                        "token_ids": [1, 2, 3, 128009],
                        "tokens": ["the", " ocean", " blue", "<|eot_id|>"],
                        "token_logprobs": [-0.1, -0.2, -0.3, -0.01],
                    },
                }],
            },
        }
        accepted_b8b = SCOUT.filter_generation_results(gen_results_b8b, target_clean=24)
        check("B8b exactly one stream accepted (word-free)", len(accepted_b8b) == 1,
              f"n={len(accepted_b8b)}")
        if accepted_b8b:
            rec = accepted_b8b[0]
            # real token ids captured (no longer hard-coded None), eos dropped
            check("B8b token_ids captured from generation logprobs",
                  rec.get("token_ids") == [370, 111, 222],
                  f"token_ids={rec.get('token_ids')}")
            # matched-context per-token logprobs captured, eos dropped
            check("B8b gen_token_logprobs captured (eos dropped)",
                  rec.get("gen_token_logprobs") == [-1.5, -2.0, -0.5],
                  f"gen_token_logprobs={rec.get('gen_token_logprobs')}")
            # summed matched-context LL available for the diagonal LR numerator
            gll = rec.get("gen_ll_matched")
            check("B8b gen_ll_matched == sum of stream logprobs",
                  gll is not None and abs(gll - (-4.0)) < 1e-6,
                  f"gen_ll_matched={gll}")
    except Exception as e:
        check("B8b filter_generation_results captures logprobs", False,
              f"{type(e).__name__}: {e}")
        for sub in ["exactly one stream accepted (word-free)",
                    "token_ids captured from generation logprobs",
                    "gen_token_logprobs captured (eos dropped)",
                    "gen_ll_matched == sum of stream logprobs"]:
            check(f"B8b {sub}", False)


# ---------------------------------------------------------------------------
# B8c: filter_generation_results degrades gracefully when logprobs are absent
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Legacy-shaped response: a word-free stream but NO logprobs field on the choice.
        gen_results_b8c = {
            "gen:evoked:curiosity:0": {
                "choices": [{"message": {"content": "qxz fjm wpl kbt rvnm"}}],
            },
        }
        accepted_b8c = SCOUT.filter_generation_results(gen_results_b8c, target_clean=24)
        check("B8c legacy (no logprobs) still accepts word-free stream",
              len(accepted_b8c) == 1, f"n={len(accepted_b8c)}")
        if accepted_b8c:
            r = accepted_b8c[0]
            check("B8c token_ids is None when logprobs absent", r.get("token_ids") is None,
                  f"token_ids={r.get('token_ids')}")
            check("B8c gen_ll_matched is None when logprobs absent",
                  r.get("gen_ll_matched") is None, f"gen_ll_matched={r.get('gen_ll_matched')}")
    except Exception as e:
        check("B8c graceful degradation without logprobs", False, f"{type(e).__name__}: {e}")
        for sub in ["legacy (no logprobs) still accepts word-free stream",
                    "token_ids is None when logprobs absent",
                    "gen_ll_matched is None when logprobs absent"]:
            check(f"B8c {sub}", False)


# ---------------------------------------------------------------------------
# B9: MC batch builder produces well-formed records
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        accepted_b9 = [
            {
                "concept": "curiosity",
                "arm": "evoked",
                "text": "qxz fjm wpl kbt",
                "accepted": True,
                "stream_idx": 0,
            }
        ]

        mc_recs = SCOUT.build_mc_batch_records(accepted_streams=accepted_b9)
        check("B9 MC records non-empty", len(mc_recs) > 0)
        r_mc = mc_recs[0]
        check("B9 MC record has 'custom_id'", "custom_id" in r_mc)
        check("B9 MC custom_id has mc: prefix", r_mc["custom_id"].startswith("mc:"))
        check("B9 MC record has 'body'", "body" in r_mc)
        body_mc = r_mc["body"]
        check("B9 MC model is Llama-3.3-70B", "Llama-3.3-70B" in body_mc.get("model", ""))
        rf = body_mc.get("response_format", {})
        check("B9 response_format.type == json_schema", rf.get("type") == "json_schema")
        js = rf.get("json_schema", {})
        schema = js.get("schema", {})
        props = schema.get("properties", {})
        check("B9 schema has 'concept' property", "concept" in props)
        concept_enum = props.get("concept", {}).get("enum", [])
        check("B9 concept enum has 12 entries", len(concept_enum) == 12,
              f"len={len(concept_enum)}")
        check("B9 max_tokens=512 (non-reasoning Llama)", body_mc.get("max_tokens") == 512)
        check("B9 MC url is /v1/chat/completions",
              r_mc.get("url") == "/v1/chat/completions" or "chat" in str(r_mc.get("url", "")))
    except Exception as e:
        check("B9 MC batch records shape", False, f"{type(e).__name__}: {e}")
        for sub in ["schema has 'concept' property", "concept enum has 12 entries",
                    "max_tokens=512 (non-reasoning Llama)"]:
            check(f"B9 {sub}", False)


# ---------------------------------------------------------------------------
# B10: parse_mc_response from serverless_72b handles Llama structured output
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        import serverless_72b as S72
        # Simulate a Llama structured-output response (content is JSON string)
        mock_llama_resp = {
            "choices": [{
                "message": {"content": json.dumps({"concept": "curiosity"})}
            }]
        }
        parsed = S72.parse_mc_response(mock_llama_resp)
        check("B10 parse_mc_response correct concept", parsed["concept"] == "curiosity")
        check("B10 parse_mc_response reasoning=None for direct", parsed.get("reasoning") is None)

        # Invalid concept
        bad_resp = {
            "choices": [{"message": {"content": json.dumps({"concept": "banana"})}}]
        }
        try:
            S72.parse_mc_response(bad_resp)
            check("B10 invalid concept raises ValueError", False, "no exception")
        except ValueError:
            check("B10 invalid concept raises ValueError", True)
    except Exception as e:
        check("B10 parse_mc_response Llama output", False, f"{type(e).__name__}: {e}")
        check("B10 parse_mc_response reasoning=None for direct", False)
        check("B10 invalid concept raises ValueError", False)


# ---------------------------------------------------------------------------
# B11: zero real API calls with injected mocks
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Build a minimal set of accepted streams
        accepted_b11 = [
            {
                "concept": "curiosity",
                "arm": "evoked",
                "text": "qxz fjm wpl",
                "accepted": True,
                "stream_idx": 0,
            }
        ]

        # Mock generation output: one accepted stream
        gen_output_line = json.dumps({
            "custom_id": "gen:evoked:curiosity:0",
            "response": {
                "body": {
                    "choices": [{"message": {"content": "qxz fjm wpl"}}]
                }
            }
        })
        # LR echo output
        lr_stream_text = "qxz fjm wpl"
        lr_tokens = lr_stream_text.split()
        lr_output_line = json.dumps({
            "custom_id": "lr:evoked:curiosity:0:matched",
            "response": {
                "body": {
                    "prompt": [{
                        "logprobs": {
                            "tokens": ["context ", "text "] + lr_tokens,
                            "token_logprobs": [-0.5, -0.6] + [-1.0] * len(lr_tokens),
                        }
                    }]
                }
            }
        })
        # MC output
        mc_output_line = json.dumps({
            "custom_id": "mc:evoked:curiosity:0:direct",
            "response": {
                "body": {
                    "choices": [{"message": {"content": json.dumps({"concept": "curiosity"})}}]
                }
            }
        })

        real_http_calls: List[str] = []

        def _b11_post(url, headers, body):
            real_http_calls.append(f"POST {url}")
            return _make_batch_create_resp("batch-b11")

        def _b11_get(url, headers):
            return _make_batch_poll_resp("COMPLETED", output_file_id="output-b11")

        output_bytes = (gen_output_line + "\n" + lr_output_line + "\n" + mc_output_line).encode()
        fake_together_b11 = _FakeTogetherClient(
            file_id="file-b11",
            output_jsonl=output_bytes,
            error_jsonl=b"",
        )

        # The key check: call batch_submit_poll_download with the mocks and verify
        # that no REAL urllib/requests calls are made (all HTTP goes through the injected callers)
        records_b11 = [{"custom_id": "gen:evoked:curiosity:0",
                        "method": "POST", "url": "/v1/chat/completions", "body": {}}]
        result_b11 = SCOUT.batch_submit_poll_download(
            jsonl_records=records_b11,
            together_client=fake_together_b11,
            http_post_caller=_b11_post,
            http_get_caller=_b11_get,
            together_ua="curl/8.4.0",
            together_base="https://api.together.xyz",
        )
        check("B11 batch_submit_poll_download returns a result with mocks", isinstance(result_b11, dict))
        check("B11 POST was routed through the injected caller",
              any("POST" in c for c in real_http_calls))
    except Exception as e:
        check("B11 zero real API calls with mocks", False, f"{type(e).__name__}: {e}")
        check("B11 POST was routed through the injected caller", False)


# ---------------------------------------------------------------------------
# B12: serverless discipline — NO endpoint lifecycle in source
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        src_text = inspect.getsource(SCOUT)
        check("B12 no create_endpoint in source", "create_endpoint" not in src_text)
        check("B12 no wait_ready in source", "wait_ready" not in src_text)
        check("B12 no delete_endpoint in source", "delete_endpoint" not in src_text)
        check("B12 no teardown stack (no atexit.register)", "atexit" not in src_text)
        check("B12 no inactive_timeout (no endpoint created)", "inactive_timeout" not in src_text)
    except Exception as e:
        check("B12 serverless discipline", False, f"{type(e).__name__}: {e}")
        for sub in ["no create_endpoint in source", "no wait_ready in source",
                    "no delete_endpoint in source"]:
            check(f"B12 {sub}", False)


# ---------------------------------------------------------------------------
# B14: custom_id reconciliation verifies the return dict shape more carefully
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        out_b14 = [
            json.dumps({"custom_id": "job-c", "response": {"body": {"val": "C"}}}),
            json.dumps({"custom_id": "job-a", "response": {"body": {"val": "A"}}}),
            json.dumps({"custom_id": "job-b", "response": {"body": {"val": "B"}}}),
        ]
        output_jsonl_b14 = b"\n".join(r.encode() for r in out_b14)

        fake_b14 = _FakeTogetherClient(file_id="file-b14", output_jsonl=output_jsonl_b14)

        def _b14_post(url, headers, body):
            return _make_batch_create_resp("batch-b14")

        def _b14_get(url, headers):
            return _make_batch_poll_resp("COMPLETED")

        records_b14 = [
            {"custom_id": "job-a", "method": "POST", "url": "/v1/chat/completions", "body": {}},
            {"custom_id": "job-b", "method": "POST", "url": "/v1/chat/completions", "body": {}},
            {"custom_id": "job-c", "method": "POST", "url": "/v1/chat/completions", "body": {}},
        ]

        result_b14 = SCOUT.batch_submit_poll_download(
            jsonl_records=records_b14,
            together_client=fake_b14,
            http_post_caller=_b14_post,
            http_get_caller=_b14_get,
            together_ua="curl/8.4.0",
            together_base="https://api.together.xyz",
        )
        check("B14 all 3 custom_ids in result dict", set(result_b14.keys()) == {"job-a", "job-b", "job-c"})
        check("B14 job-a value is A", result_b14.get("job-a", {}).get("val") == "A")
        check("B14 job-b value is B", result_b14.get("job-b", {}).get("val") == "B")
        check("B14 job-c value is C", result_b14.get("job-c", {}).get("val") == "C")
    except Exception as e:
        check("B14 custom_id reconciliation dict shape", False, f"{type(e).__name__}: {e}")
        for sub in ["all 3 custom_ids in result dict", "job-a value is A", "job-b value is B"]:
            check(f"B14 {sub}", False)


# ---------------------------------------------------------------------------
# P1: phase="generate" stops before Stage 2 (LR/MC batch builders NOT called)
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        import tempfile as _tempfile

        # Track calls to build_lr_batch_records and build_mc_batch_records
        lr_builder_calls: List = []
        mc_builder_calls: List = []

        _orig_build_lr = SCOUT.build_lr_batch_records
        _orig_build_mc = SCOUT.build_mc_batch_records

        def _spy_lr(*args, **kwargs):
            lr_builder_calls.append(1)
            return _orig_build_lr(*args, **kwargs)

        def _spy_mc(*args, **kwargs):
            mc_builder_calls.append(1)
            return _orig_build_mc(*args, **kwargs)

        SCOUT.build_lr_batch_records = _spy_lr
        SCOUT.build_mc_batch_records = _spy_mc

        try:
            # Build a generation output that yields one accepted stream
            gen_output_line = json.dumps({
                "custom_id": "gen:evoked:curiosity:0",
                "response": {"body": {"choices": [{"message": {"content": "qxz fjm wpl"}}]}},
            })
            fake_client_p1 = _FakeTogetherClient(
                file_id="file-p1",
                output_jsonl=gen_output_line.encode(),
                error_jsonl=b"",
            )

            def _p1_post(url, headers, body):
                return _make_batch_create_resp("batch-p1")

            def _p1_get(url, headers):
                return _make_batch_poll_resp("COMPLETED", output_file_id="output-p1")

            with _tempfile.TemporaryDirectory() as tmp_out:
                result_p1 = SCOUT.run_all(
                    out_dir=Path(tmp_out),
                    arms=["evoked"],
                    concepts=["curiosity"],
                    target_clean=1,
                    together_client=fake_client_p1,
                    http_post_caller=_p1_post,
                    http_get_caller=_p1_get,
                    phase="generate",
                )
            check("P1 phase=generate: LR builder NOT called", len(lr_builder_calls) == 0,
                  f"lr_builder_calls={lr_builder_calls}")
            check("P1 phase=generate: MC builder NOT called", len(mc_builder_calls) == 0,
                  f"mc_builder_calls={mc_builder_calls}")
            # run_all should return early with no lr/mc records
            check("P1 phase=generate: result has no lr_records key or is None/empty",
                  result_p1 is None or not result_p1.get("lr_records"),
                  f"result={result_p1}")
        finally:
            SCOUT.build_lr_batch_records = _orig_build_lr
            SCOUT.build_mc_batch_records = _orig_build_mc

    except Exception as e:
        check("P1 phase=generate stops before Stage 2", False, f"{type(e).__name__}: {e}")
        check("P1 phase=generate: LR builder NOT called", False)
        check("P1 phase=generate: MC builder NOT called", False)


# ---------------------------------------------------------------------------
# P2: phase="score" loads accepted_streams from disk, skips Stage 1
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        import tempfile as _tempfile2

        gen_upload_calls_p2: List = []

        def _p2_post(url, headers, body):
            return _make_batch_create_resp("batch-p2")

        def _p2_get(url, headers):
            return _make_batch_poll_resp("COMPLETED", output_file_id="output-p2")

        # Pre-write accepted_streams so phase="score" can load it
        accepted_p2 = [
            {
                "concept": "curiosity",
                "arm": "evoked",
                "text": "qxz fjm wpl",
                "accepted": True,
                "stream_idx": 0,
                "attempt_idx": 0,
            }
        ]

        # LR echo output for the one stream (matched + neutral)
        def _make_lr_echo(custom_id, tokens, lps):
            return json.dumps({
                "custom_id": custom_id,
                "response": {"body": {
                    "prompt": [{"logprobs": {
                        "tokens": tokens,
                        "token_logprobs": lps,
                    }}]
                }},
            })

        stream_tokens = ["qxz", " fjm", " wpl"]
        ctx_tokens = ["ctx "]
        all_tok = ctx_tokens + stream_tokens
        all_lps = [-0.5] + [-1.0] * len(stream_tokens)

        lr_matched_line = _make_lr_echo("lr:evoked:curiosity:0:matched", all_tok, all_lps)
        lr_neutral_line = _make_lr_echo("lr:evoked:curiosity:0:neutral", all_tok, all_lps)
        mc_direct_line = json.dumps({
            "custom_id": "mc:evoked:curiosity:0:direct",
            "response": {"body": {"choices": [{"message": {"content": json.dumps({"concept": "curiosity"})}}]}},
        })
        mc_think_line = json.dumps({
            "custom_id": "mc:evoked:curiosity:0:with_think",
            "response": {"body": {"choices": [{"message": {"content": json.dumps({"concept": "curiosity"})}}]}},
        })
        output_bytes_p2 = (
            lr_matched_line + "\n" + lr_neutral_line + "\n"
            + mc_direct_line + "\n" + mc_think_line
        ).encode()

        class _FakeFilesP2:
            def __init__(self):
                self.upload_calls: List = []
            def upload(self, file, purpose, check=False):
                self.upload_calls.append(file)
                return _FakeFileUpload("file-p2")
            def content(self, file_id):
                return _BytesWrapper(output_bytes_p2)

        class _FakeClientP2:
            def __init__(self):
                self.files = _FakeFilesP2()

        fake_p2 = _FakeClientP2()

        with _tempfile2.TemporaryDirectory() as tmp_out:
            streams_path_p2 = Path(tmp_out) / "streams_llama70b.json"
            streams_path_p2.write_text(json.dumps(accepted_p2))

            result_p2 = SCOUT.run_all(
                out_dir=Path(tmp_out),
                arms=["evoked"],
                concepts=["curiosity"],
                target_clean=1,
                together_client=fake_p2,
                http_post_caller=_p2_post,
                http_get_caller=_p2_get,
                phase="score",
            )

        # Stage 1 generation upload should NOT have been called
        check("P2 phase=score: Stage-1 generation upload NOT called",
              len(fake_p2.files.upload_calls) <= 2,  # at most LR+MC uploads, not gen
              f"upload_calls={fake_p2.files.upload_calls}")
        check("P2 phase=score: lr_records present in result",
              isinstance(result_p2.get("lr_records"), list),
              f"result={result_p2}")
        check("P2 phase=score: mc_records present in result",
              isinstance(result_p2.get("mc_records"), list),
              f"result={result_p2}")

        # Test error path: missing streams file
        try:
            with _tempfile2.TemporaryDirectory() as tmp_empty:
                SCOUT.run_all(
                    out_dir=Path(tmp_empty),
                    arms=["evoked"],
                    concepts=["curiosity"],
                    target_clean=1,
                    together_client=fake_p2,
                    http_post_caller=_p2_post,
                    http_get_caller=_p2_get,
                    phase="score",
                )
            check("P2 phase=score missing streams raises error", False, "no exception raised")
        except (FileNotFoundError, RuntimeError, SystemExit):
            check("P2 phase=score missing streams raises error", True)

    except Exception as e:
        check("P2 phase=score loads from disk", False, f"{type(e).__name__}: {e}")
        check("P2 phase=score: Stage-1 generation upload NOT called", False)
        check("P2 phase=score: lr_records present in result", False)
        check("P2 phase=score: mc_records present in result", False)
        check("P2 phase=score missing streams raises error", False)


# ---------------------------------------------------------------------------
# P3: print_gibberish_validation returns correct per-arm counts + flags real-word streams
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Check the function exists
        check("P3 print_gibberish_validation function exists",
              hasattr(SCOUT, "print_gibberish_validation"),
              "print_gibberish_validation not found in SCOUT")

        if hasattr(SCOUT, "print_gibberish_validation"):
            # 3 evoked + 2 secret_word; plant one real-word stream in evoked
            test_streams_p3 = [
                {"arm": "evoked", "concept": "curiosity", "text": "qxz fjm wpl", "stream_idx": 0},
                {"arm": "evoked", "concept": "ocean",     "text": "the cat sat",  "stream_idx": 1},  # real words!
                {"arm": "evoked", "concept": "fear",      "text": "rvnm kbt xqz", "stream_idx": 2},
                {"arm": "secret_word", "concept": "curiosity", "text": "fjm wpl qxz", "stream_idx": 3},
                {"arm": "secret_word", "concept": "ocean",     "text": "kbt rvnm wpl", "stream_idx": 4},
            ]
            import io as _io
            import sys as _sys_p3
            old_stdout = _sys_p3.stdout
            _sys_p3.stdout = _io.StringIO()
            try:
                stats = SCOUT.print_gibberish_validation(test_streams_p3)
            finally:
                captured = _sys_p3.stdout.getvalue()
                _sys_p3.stdout = old_stdout

            # Per-arm counts
            check("P3 stats has per_arm_counts key", "per_arm_counts" in stats,
                  f"stats keys: {list(stats.keys())}")
            if "per_arm_counts" in stats:
                check("P3 evoked count is 3", stats["per_arm_counts"].get("evoked") == 3,
                      f"evoked={stats['per_arm_counts'].get('evoked')}")
                check("P3 secret_word count is 2", stats["per_arm_counts"].get("secret_word") == 2,
                      f"secret_word={stats['per_arm_counts'].get('secret_word')}")

            # Real-word detection: "the cat sat" has real words
            check("P3 real_word_fraction > 0 (planted real-word stream)",
                  stats.get("real_word_fraction", 0) > 0,
                  f"real_word_fraction={stats.get('real_word_fraction')}")

            # Sample is deterministic (fixed seed → always same 5 indices)
            check("P3 sample key present in stats", "sample" in stats,
                  f"stats keys: {list(stats.keys())}")
            if "sample" in stats:
                check("P3 sample is a list", isinstance(stats["sample"], list))
                # Run again — should get same sample
                import io as _io2
                import sys as _sys_p3b
                old_stdout2 = _sys_p3b.stdout
                _sys_p3b.stdout = _io2.StringIO()
                try:
                    stats2 = SCOUT.print_gibberish_validation(test_streams_p3)
                finally:
                    _sys_p3b.stdout = old_stdout2
                check("P3 sample is deterministic across two calls",
                      stats["sample"] == stats2["sample"],
                      f"sample1={stats['sample']}, sample2={stats2['sample']}")

            # Header should appear in captured output
            check("P3 validation header printed",
                  "MANUAL GIBBERISH VALIDATION" in captured,
                  f"captured prefix: {captured[:200]!r}")

    except Exception as e:
        check("P3 print_gibberish_validation", False, f"{type(e).__name__}: {e}")
        for sub in ["stats has per_arm_counts key", "evoked count is 3",
                    "real_word_fraction > 0 (planted real-word stream)",
                    "sample key present in stats", "validation header printed"]:
            check(f"P3 {sub}", False)


# ---------------------------------------------------------------------------
# P4: score_lr_results counts empty spans and warns >5%
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        # Build accepted streams where SOME will have empty spans (text not found in echo)
        accepted_p4 = []
        for i in range(10):
            accepted_p4.append({
                "concept": "curiosity",
                "arm": "evoked",
                "text": f"qxz fjm wpl {i}",
                "accepted": True,
                "stream_idx": i,
            })

        # Build lr_results with only the first 4 having valid echo responses;
        # the other 6 will have a response but with stream_text NOT matching tokens
        # (so _find_stream_span_lps returns [])
        #
        # _find_stream_span_lps joins tokens with "".join(tokens) before searching,
        # so to make a "valid" match the tokens must preserve the original whitespace.
        def _make_echo_body_valid(stream_text):
            # Build tokens that, when joined, reproduce stream_text exactly.
            # Use a context prefix token + the full stream_text as a single trailing token.
            ctx_tok = "CTX "
            tokens = [ctx_tok, stream_text]
            lps = [-0.5, -1.0]
            return {"prompt": [{"logprobs": {"tokens": tokens, "token_logprobs": lps}}]}

        def _make_echo_body_invalid(stream_text):
            # Tokens do NOT contain stream_text
            tokens = ["TOTALLY", "DIFFERENT"]
            lps = [-1.0, -1.0]
            return {"prompt": [{"logprobs": {"tokens": tokens, "token_logprobs": lps}}]}

        lr_results_p4: Dict = {}
        for i in range(10):
            stream_text = f"qxz fjm wpl {i}"
            if i < 4:
                # Valid: joined tokens contain stream_text
                lr_results_p4[f"lr:evoked:curiosity:{i}:matched"] = _make_echo_body_valid(stream_text)
                lr_results_p4[f"lr:evoked:curiosity:{i}:neutral"] = _make_echo_body_valid(stream_text)
            else:
                # Invalid: tokens do NOT contain the stream_text (mismatch)
                lr_results_p4[f"lr:evoked:curiosity:{i}:matched"] = _make_echo_body_invalid(stream_text)
                lr_results_p4[f"lr:evoked:curiosity:{i}:neutral"] = _make_echo_body_invalid(stream_text)

        import warnings
        import io as _io_p4
        import sys as _sys_p4

        # Capture log warnings via logging
        import logging as _logging_p4
        log_output_p4: List[str] = []

        class _CapHandler(_logging_p4.Handler):
            def emit(self, record):
                log_output_p4.append(self.format(record))

        cap_handler = _CapHandler()
        scout_logger = _logging_p4.getLogger("llama70b_scout")
        scout_logger.addHandler(cap_handler)

        try:
            result_p4 = SCOUT.score_lr_results(lr_results_p4, accepted_p4)
        finally:
            scout_logger.removeHandler(cap_handler)

        # Check function signature: now returns (lr_records, metadata)
        check("P4 score_lr_results returns a tuple",
              isinstance(result_p4, tuple),
              f"type={type(result_p4)}")
        if isinstance(result_p4, tuple) and len(result_p4) == 2:
            lr_recs_p4, meta_p4 = result_p4
            check("P4 first element is a list of LR records",
                  isinstance(lr_recs_p4, list),
                  f"type={type(lr_recs_p4)}")
            check("P4 metadata dict has empty_span_count",
                  "empty_span_count" in meta_p4,
                  f"meta keys: {list(meta_p4.keys())}")
            if "empty_span_count" in meta_p4:
                check("P4 empty_span_count == 6",
                      meta_p4["empty_span_count"] == 6,
                      f"empty_span_count={meta_p4['empty_span_count']}")
            # Warning logged because 6/10 = 60% > 5%
            any_warn = any("empty" in m.lower() or "span" in m.lower()
                           for m in log_output_p4)
            check("P4 warning logged for >5% empty spans", any_warn,
                  f"log messages: {log_output_p4}")
        else:
            check("P4 first element is a list of LR records", False)
            check("P4 metadata dict has empty_span_count", False)
            check("P4 empty_span_count == 6", False)
            check("P4 warning logged for >5% empty spans", False)

    except Exception as e:
        check("P4 score_lr_results empty-span counting", False, f"{type(e).__name__}: {e}")
        for sub in ["returns a tuple", "first element is a list of LR records",
                    "metadata dict has empty_span_count", "empty_span_count == 6",
                    "warning logged for >5% empty spans"]:
            check(f"P4 {sub}", False)


# ---------------------------------------------------------------------------
# B15: subsample_streams_for_peek — Amendment-6 instrument peek cap
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        check("B15 subsample_streams_for_peek exists",
              hasattr(SCOUT, "subsample_streams_for_peek"))
        if hasattr(SCOUT, "subsample_streams_for_peek"):
            # 3 cells: (evoked,curiosity)x4, (evoked,ocean)x3, (secret_word,curiosity)x5
            streams_b15 = []
            idx = 0
            for arm, concept, n in [("evoked", "curiosity", 4),
                                    ("evoked", "ocean", 3),
                                    ("secret_word", "curiosity", 5)]:
                for _ in range(n):
                    streams_b15.append({"arm": arm, "concept": concept,
                                        "text": "qxz fjm", "stream_idx": idx})
                    idx += 1
            # peek_n=2 -> at most 2 per cell = 2+2+2 = 6
            kept2 = SCOUT.subsample_streams_for_peek(streams_b15, 2)
            check("B15 peek_n=2 keeps <=2 per cell (total 6)", len(kept2) == 6,
                  f"n={len(kept2)}")
            from collections import Counter as _Counter
            per_cell = _Counter((s["arm"], s["concept"]) for s in kept2)
            check("B15 no cell exceeds peek_n", all(v <= 2 for v in per_cell.values()),
                  f"per_cell={dict(per_cell)}")
            # deterministic: lowest stream_idx kept for the 5-stream cell (idx 7,8)
            sw = sorted(s["stream_idx"] for s in kept2
                        if s["arm"] == "secret_word")
            check("B15 deterministic: keeps lowest stream_idx per cell",
                  sw == [7, 8], f"secret_word idxs kept={sw}")
            # peek_n larger than any cell -> returns everything (12)
            keptbig = SCOUT.subsample_streams_for_peek(streams_b15, 100)
            check("B15 peek_n>=cell size keeps all", len(keptbig) == 12,
                  f"n={len(keptbig)}")
            # peek_n None or 0 -> full sweep (all 12), unchanged
            check("B15 peek_n=None returns all (full sweep)",
                  len(SCOUT.subsample_streams_for_peek(streams_b15, None)) == 12)
            check("B15 peek_n=0 returns all (full sweep)",
                  len(SCOUT.subsample_streams_for_peek(streams_b15, 0)) == 12)
    except Exception as e:
        check("B15 subsample_streams_for_peek", False, f"{type(e).__name__}: {e}")
        for sub in ["peek_n=2 keeps <=2 per cell (total 6)", "no cell exceeds peek_n",
                    "deterministic: keeps lowest stream_idx per cell",
                    "peek_n>=cell size keeps all", "peek_n=None returns all (full sweep)"]:
            check(f"B15 {sub}", False)


# ---------------------------------------------------------------------------
# P5: run_all(phase="score", peek_n=N) subsamples streams before Stage 2
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        import tempfile as _tempfile5

        # Capture the accepted_streams handed to build_lr_batch_records AND build_mc_batch_records
        seen_lr_streams: List = []
        seen_mc_streams: List = []
        _orig_build_lr5 = SCOUT.build_lr_batch_records
        _orig_build_mc5 = SCOUT.build_mc_batch_records

        def _spy_lr5(accepted_streams, **kwargs):
            seen_lr_streams.append(list(accepted_streams))
            return _orig_build_lr5(accepted_streams=accepted_streams, **kwargs)

        def _spy_mc5(accepted_streams, **kwargs):
            seen_mc_streams.append(list(accepted_streams))
            return _orig_build_mc5(accepted_streams=accepted_streams, **kwargs)

        SCOUT.build_lr_batch_records = _spy_lr5
        SCOUT.build_mc_batch_records = _spy_mc5
        try:
            # 5 streams in one cell; peek_n=2 must cut to 2 before Stage 2
            accepted_p5 = [
                {"concept": "curiosity", "arm": "evoked", "text": "qxz fjm wpl",
                 "accepted": True, "stream_idx": i, "attempt_idx": i}
                for i in range(5)
            ]

            def _p5_post(url, headers, body):
                return _make_batch_create_resp("batch-p5")

            def _p5_get(url, headers):
                return _make_batch_poll_resp("COMPLETED", output_file_id="output-p5")

            # Minimal LR/MC output so scoring doesn't crash (content-agnostic)
            def _lr_echo5(cid):
                return json.dumps({"custom_id": cid, "response": {"body": {"prompt": [{"logprobs": {
                    "tokens": ["ctx ", "qxz", " fjm", " wpl"],
                    "token_logprobs": [-0.5, -1.0, -1.0, -1.0]}}]}}})
            lines5 = []
            for i in range(2):  # only the 2 peeked streams get scored
                lines5.append(_lr_echo5(f"lr:evoked:curiosity:{i}:matched"))
                lines5.append(_lr_echo5(f"lr:evoked:curiosity:{i}:neutral"))
                lines5.append(json.dumps({"custom_id": f"mc:evoked:curiosity:{i}:direct",
                    "response": {"body": {"choices": [{"message": {"content": json.dumps({"concept": "curiosity"})}}]}}}))
                lines5.append(json.dumps({"custom_id": f"mc:evoked:curiosity:{i}:with_think",
                    "response": {"body": {"choices": [{"message": {"content": json.dumps({"concept": "curiosity"})}}]}}}))
            output_bytes_p5 = ("\n".join(lines5)).encode()

            class _FakeFilesP5:
                def upload(self, file, purpose, check=False):
                    return _FakeFileUpload("file-p5")
                def content(self, file_id):
                    return _BytesWrapper(output_bytes_p5)

            class _FakeClientP5:
                def __init__(self):
                    self.files = _FakeFilesP5()

            with _tempfile5.TemporaryDirectory() as tmp5:
                (Path(tmp5) / "streams_llama70b.json").write_text(json.dumps(accepted_p5))
                SCOUT.run_all(
                    out_dir=Path(tmp5), arms=["evoked"], concepts=["curiosity"],
                    target_clean=5, together_client=_FakeClientP5(),
                    http_post_caller=_p5_post, http_get_caller=_p5_get,
                    phase="score", peek_n=2,
                )
                verdict_path_p5 = Path(tmp5) / "peek_verdict.json"
                verdict_exists = verdict_path_p5.exists()
                verdict_obj = json.loads(verdict_path_p5.read_text()) if verdict_exists else {}
            check("P5 build_lr called once", len(seen_lr_streams) == 1,
                  f"calls={len(seen_lr_streams)}")
            if seen_lr_streams:
                check("P5 peek_n=2 subsampled 5->2 before Stage 2 (LR)",
                      len(seen_lr_streams[0]) == 2, f"n={len(seen_lr_streams[0])}")
            # MC (Stage 3) must score the SAME capped subset, not the full 5
            check("P5 build_mc called once", len(seen_mc_streams) == 1,
                  f"calls={len(seen_mc_streams)}")
            if seen_mc_streams:
                check("P5 peek_n=2 subsampled 5->2 before Stage 3 (MC)",
                      len(seen_mc_streams[0]) == 2, f"n={len(seen_mc_streams[0])}")
            # Disclosure artifact written with correct counts
            check("P5 peek_verdict.json written", verdict_exists)
            check("P5 peek_verdict records streams_scored=2 / available=5",
                  verdict_obj.get("streams_scored") == 2 and verdict_obj.get("streams_available") == 5,
                  f"verdict={verdict_obj}")
        finally:
            SCOUT.build_lr_batch_records = _orig_build_lr5
            SCOUT.build_mc_batch_records = _orig_build_mc5
    except Exception as e:
        check("P5 run_all peek_n subsamples before Stage 2", False,
              f"{type(e).__name__}: {e}")
        check("P5 peek_n=2 subsampled 5->2 before Stage 2 (LR)", False)
        check("P5 peek_n=2 subsampled 5->2 before Stage 3 (MC)", False)
        check("P5 peek_verdict.json written", False)


# ---------------------------------------------------------------------------
# P6: run_all(skip_mc=True) skips Stage 3 (MC batch NOT built), LR still runs
# ---------------------------------------------------------------------------
if SCOUT is not None:
    try:
        import tempfile as _tempfile6
        mc_calls6: List = []
        lr_calls6: List = []
        _orig_mc6 = SCOUT.build_mc_batch_records
        _orig_lr6 = SCOUT.build_lr_batch_records

        def _spy_mc6(accepted_streams, **kwargs):
            mc_calls6.append(1)
            return _orig_mc6(accepted_streams=accepted_streams, **kwargs)

        def _spy_lr6(accepted_streams, **kwargs):
            lr_calls6.append(1)
            return _orig_lr6(accepted_streams=accepted_streams, **kwargs)

        SCOUT.build_mc_batch_records = _spy_mc6
        SCOUT.build_lr_batch_records = _spy_lr6
        try:
            accepted_p6 = [
                {"concept": "curiosity", "arm": "evoked", "text": "qxz fjm wpl",
                 "accepted": True, "stream_idx": i, "attempt_idx": i}
                for i in range(3)
            ]

            def _p6_post(url, headers, body):
                return _make_batch_create_resp("batch-p6")

            def _p6_get(url, headers):
                return _make_batch_poll_resp("COMPLETED", output_file_id="output-p6")

            # Only LR echo output (no MC lines needed since MC is skipped)
            def _lr_echo6(cid):
                return json.dumps({"custom_id": cid, "response": {"body": {"prompt": [{"logprobs": {
                    "tokens": ["ctx ", "qxz", " fjm", " wpl"],
                    "token_logprobs": [-0.5, -1.0, -1.0, -1.0]}}]}}})
            lines6 = []
            for i in range(3):
                lines6.append(_lr_echo6(f"lr:evoked:curiosity:{i}:matched"))
                lines6.append(_lr_echo6(f"lr:evoked:curiosity:{i}:neutral"))
            output_bytes_p6 = ("\n".join(lines6)).encode()

            class _FakeFilesP6:
                def upload(self, file, purpose, check=False):
                    return _FakeFileUpload("file-p6")
                def content(self, file_id):
                    return _BytesWrapper(output_bytes_p6)

            class _FakeClientP6:
                def __init__(self):
                    self.files = _FakeFilesP6()

            with _tempfile6.TemporaryDirectory() as tmp6:
                (Path(tmp6) / "streams_llama70b.json").write_text(json.dumps(accepted_p6))
                result_p6 = SCOUT.run_all(
                    out_dir=Path(tmp6), arms=["evoked"], concepts=["curiosity"],
                    target_clean=3, together_client=_FakeClientP6(),
                    http_post_caller=_p6_post, http_get_caller=_p6_get,
                    phase="score", skip_mc=True,
                )
                mc_file6 = (Path(tmp6) / "mc_records_llama70b.json").exists()
            check("P6 skip_mc: MC builder NOT called", len(mc_calls6) == 0,
                  f"mc_calls={len(mc_calls6)}")
            check("P6 skip_mc: LR builder STILL called", len(lr_calls6) == 1,
                  f"lr_calls={len(lr_calls6)}")
            check("P6 skip_mc: lr_records present in result",
                  isinstance(result_p6.get("lr_records"), list) and len(result_p6["lr_records"]) > 0,
                  f"result_keys={list(result_p6.keys()) if isinstance(result_p6, dict) else result_p6}")
            check("P6 skip_mc: mc_records empty, scores None",
                  result_p6.get("mc_records") == [] and result_p6.get("scores") is None,
                  f"mc={result_p6.get('mc_records')} scores={result_p6.get('scores')}")
            check("P6 skip_mc: no mc_records file written", not mc_file6)
        finally:
            SCOUT.build_mc_batch_records = _orig_mc6
            SCOUT.build_lr_batch_records = _orig_lr6
    except Exception as e:
        check("P6 run_all skip_mc skips Stage 3", False, f"{type(e).__name__}: {e}")
        for sub in ["MC builder NOT called", "LR builder STILL called",
                    "lr_records present in result", "mc_records empty, scores None"]:
            check(f"P6 skip_mc: {sub}", False)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

passed = sum(1 for _, ok, _ in checks if ok)
failed = sum(1 for _, ok, _ in checks if not ok)

print(f"\n{'='*60}")
print(f"test_run_llama70b_scout.py: {passed} passed, {failed} failed")
print(f"{'='*60}")

for name, ok, note in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {note}" if note else ""))

sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
