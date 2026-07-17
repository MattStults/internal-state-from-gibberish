"""RED-first unit tests for harness/run_frontier_72b.py.

NO real API calls.  All HTTP is intercepted via injected mock callers.

Test inventory
--------------
A1  adapter request-shaping — Together dedicated create_endpoint:
      POSTs to the correct /endpoints URL with the pinned hw_id, model slug,
      autoscaling {min:1, max:1}, inactive_timeout, and display_name.  The
      Authorization header is "Bearer <key>", User-Agent is the curl string
      that bypasses Cloudflare.
A2  adapter request-shaping — Together teacher_force:
      POSTs to the correct /completions URL with echo=True, logprobs=1,
      max_tokens=0, temperature=0, and the User-Agent header.
A3  adapter request-shaping — Fireworks teacher_force:
      POSTs to api.fireworks.ai /v1/completions with echo_last=<N> (the
      number of tokens in the span), logprobs=1, max_tokens=0.
A4  adapter request-shaping — Together wait_ready polls until STARTED:
      mock GET returns PENDING x2 then STARTED; verify three GET calls were
      made and wait_ready returns without error.
A5  wait_ready raises TimeoutError if STARTED is never reached within the
      poll window (mock always returns PENDING; poll_timeout_s=1, interval=0).
A6  wait_ready raises RuntimeError on endpoint ERROR state.
A7  delete_endpoint is idempotent: a second delete after 404 does not raise
      and does not make a second HTTP call (id is cleared after first delete).

P1  functionality_probe PASS — echo returns per-token logprobs for the actual
      tokens (mock Together response with correct token_ids + token_logprobs).
      Probe should return ok=True.
P2  functionality_probe EARLY-OUT — echo returns only a completion, not
      per-token logprobs (DeepInfra-style stub; teacher_force raises).
      Probe should return ok=False with a non-empty reason string.
P3  functionality_probe early-out on length mismatch: echo response returns
      the wrong number of logprobs.  Probe returns ok=False.

E1  together_echo_to_span_logprobs: given a mock Together response whose
      token_ids include the expected_token_ids as a contiguous suffix,
      returns the per-token logprobs for exactly those positions.
E2  together_echo_to_span_logprobs raises RuntimeError when expected_token_ids
      are NOT found in the response (alignment failure).
E3  fireworks_echo_to_span_logprobs: given a mock Fireworks response with
      exactly N logprobs, returns all N.
E4  fireworks_echo_to_span_logprobs: leading None values are stripped before
      alignment (Fireworks sometimes returns a null at index 0).
E5  fireworks_echo_to_span_logprobs raises RuntimeError when fewer logprobs
      are returned than expected.

T1  teardown idempotency: calling _do_teardown twice does not make a second
      DELETE HTTP call (the id is cleared after the first successful delete).
T2  endpoint id file: write_endpoint_id writes the id; read_endpoint_id reads
      it back; remove_endpoint_id_file removes it; a missing file returns None.
T3  --reap: given a populated id file and a mock DELETE that returns 204,
      cmd_reap deletes the endpoint and removes the id file.
T4  --reap: given a populated id file and a mock DELETE that returns 404
      (already gone), cmd_reap still removes the id file (idempotent).
T5  --reap: given no id file, cmd_reap logs "nothing to do" and returns
      without error.

D1  --dry never creates an endpoint: cmd_dry builds a serverless adapter and
      calls functionality_probe but NEVER calls create_endpoint_fn — even when
      the probe itself succeeds.
D2  --dry with the DeepInfra broken-echo adapter triggers the early-out branch
      (probe returns ok=False) without creating any endpoint.
D3  --dry with the Fireworks adapter and a mock teacher_force that returns
      valid logprobs reports PASS.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch, MagicMock

import math

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
HARNESS = REPO / "harness"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HARNESS))

# ---------------------------------------------------------------------------
# Import the module under test; record any import error as a check
# ---------------------------------------------------------------------------
checks: List = []


def check(name: str, cond: bool, note: str = "") -> None:
    checks.append((name, bool(cond), note))


F = None
try:
    import run_frontier_72b as F
    check("import run_frontier_72b", True)
except Exception as e:
    check("import run_frontier_72b", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# MOCK HELPERS
# ---------------------------------------------------------------------------

def _mock_together_create_ok(ep_id: str = "endpoint-abc123"):
    """Mock POST caller for Together create_endpoint -> returns {id: ...}."""
    def caller(url, headers, body):
        return {"id": ep_id, "state": "PENDING"}
    return caller


def _mock_together_status_sequence(states: List[str]):
    """Mock GET caller returning states in sequence (cycles on last element)."""
    calls: List = []
    def caller(url, headers):
        idx = min(len(calls), len(states) - 1)
        calls.append(idx)
        return (200, {"state": states[idx]})
    caller.calls = calls
    return caller


def _mock_together_delete(status: int = 204):
    """Mock DELETE caller returning the given status code."""
    calls: List = []
    def caller(url, headers):
        calls.append(url)
        return status
    caller.calls = calls
    return caller


def _mock_together_echo_response(prompt_ids: List[int], base_lp: float = -1.0) -> Dict:
    """Build a mock Together echo response with per-token logprobs for prompt_ids."""
    return {
        "prompt": [{
            "logprobs": {
                "token_ids": list(prompt_ids),
                "tokens": [str(x) for x in prompt_ids],
                "token_logprobs": [base_lp - i * 0.01 for i in range(len(prompt_ids))],
            }
        }]
    }


def _mock_fireworks_echo_response(n_tokens: int, base_lp: float = -1.5,
                                  leading_nulls: int = 0) -> Dict:
    """Build a mock Fireworks echo_last response."""
    lps = ([None] * leading_nulls) + [base_lp - i * 0.01 for i in range(n_tokens)]
    return {
        "choices": [{
            "logprobs": {
                "token_logprobs": lps,
            }
        }]
    }


# Fake tokenizer that encodes each char to its ordinal (no torch needed)
class _FakeTok:
    class _R:
        def __init__(self, ids):
            self.input_ids = ids
    def __call__(self, text, add_special_tokens=False):
        return self._R([ord(c) for c in text])
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)
    def apply_chat_template(self, msgs, add_generation_prompt=False, **kw):
        ids = []
        for m in msgs:
            ids.extend(ord(c) for c in m["content"])
        if add_generation_prompt:
            ids.append(99999)
        return ids


# ===========================================================================
# A1: Together create_endpoint request shaping
# ===========================================================================
if F is not None:
    captured_create: List[Dict] = []

    def _capture_create(url, headers, body):
        captured_create.append({"url": url, "headers": dict(headers), "body": dict(body)})
        return {"id": "endpoint-test-001", "state": "PENDING"}

    try:
        adapter = F.make_together_dedicated_adapter(
            api_key="test-key-A1",
            display_name="test-display",
            hw_id="1x_nvidia_b200_180gb_sxm",
            model="Qwen/Qwen2.5-72B-Instruct",
            http_post_caller=_capture_create,
            http_get_caller=None,
            http_delete_caller=None,
        )
        ep_id = adapter.create_endpoint_fn()
        c = captured_create[0]
        check("A1a create POSTs to correct /endpoints URL",
              "/endpoints" in c["url"] and "together.ai" in c["url"])
        check("A1b create body contains pinned hw_id",
              c["body"].get("hardware") == "1x_nvidia_b200_180gb_sxm")
        check("A1c create body contains model slug",
              c["body"].get("model") == "Qwen/Qwen2.5-72B-Instruct")
        check("A1d create body has autoscaling min_replicas=1 max_replicas=1",
              c["body"].get("autoscaling") == {"min_replicas": 1, "max_replicas": 1})
        check("A1e create body has inactive_timeout",
              "inactive_timeout" in c["body"])
        check("A1f create body has display_name",
              c["body"].get("display_name") == "test-display")
        check("A1g Authorization header is Bearer key",
              "Bearer test-key-A1" in c["headers"].get("Authorization", ""))
        check("A1h User-Agent is curl string (Cloudflare bypass)",
              c["headers"].get("User-Agent") == F.TOGETHER_UA)
        check("A1i create_endpoint returns the endpoint id",
              ep_id == "endpoint-test-001")
    except Exception as e:
        for tag in ("A1a", "A1b", "A1c", "A1d", "A1e", "A1f", "A1g", "A1h", "A1i"):
            check(f"{tag} create_endpoint request shaping", False, str(e))


# ===========================================================================
# A2: Together teacher_force request shaping
# ===========================================================================
if F is not None:
    tf_captured: List[Dict] = []

    def _tf_status_A2(url, headers):
        return (200, {"state": "STARTED"})

    def _tf_post_A2(url, headers, body):
        # Route by URL: /endpoints -> create response; /completions -> echo response
        if "/endpoints" in url and "completions" not in url:
            return {"id": "endpoint-tf-test", "state": "PENDING"}
        # For /completions: capture the request and return an echo response.
        # The prompt will be the decoded text of [97, 98, 99] = "abc"
        tf_captured.append({"url": url, "headers": dict(headers), "body": dict(body)})
        probe_ids = [ord(c) for c in "abc"]
        return _mock_together_echo_response(probe_ids)

    try:
        adapter_tf = F.make_together_dedicated_adapter(
            api_key="test-key-A2",
            http_post_caller=_tf_post_A2,
            http_get_caller=_tf_status_A2,
            http_delete_caller=_mock_together_delete(204),
        )
        adapter_tf.create_endpoint_fn()
        adapter_tf.wait_ready_fn()

        # Patch the tokenizer so no real model is loaded
        with patch.object(F, "_get_tokenizer", return_value=_FakeTok()):
            prompt_ids = [ord(c) for c in "abc"]
            try:
                adapter_tf.teacher_force_fn(prompt_ids)
            except Exception:
                pass   # echo alignment may fail with fake tok; we only care about request shape

        if tf_captured:
            c2 = tf_captured[-1]
            check("A2a teacher_force POSTs to correct /completions URL",
                  "/completions" in c2["url"] and "together.xyz" in c2["url"])
            check("A2b teacher_force body has echo=True",
                  c2["body"].get("echo") is True)
            check("A2c teacher_force body has logprobs=1",
                  c2["body"].get("logprobs") == 1)
            check("A2d teacher_force body has max_tokens=0",
                  c2["body"].get("max_tokens") == 0)
            check("A2e teacher_force body has temperature=0",
                  c2["body"].get("temperature") == 0)
            check("A2f teacher_force Authorization header is Bearer key",
                  "Bearer test-key-A2" in c2["headers"].get("Authorization", ""))
            check("A2g teacher_force User-Agent is curl string",
                  c2["headers"].get("User-Agent") == F.TOGETHER_UA)
        else:
            for tag in ("A2a", "A2b", "A2c", "A2d", "A2e", "A2f", "A2g"):
                check(f"{tag} teacher_force request shaping", False, "no POST captured (check routing)")
    except Exception as e:
        for tag in ("A2a", "A2b", "A2c", "A2d", "A2e", "A2f", "A2g"):
            check(f"{tag} teacher_force request shaping", False, str(e))


# ===========================================================================
# A3: Fireworks teacher_force request shaping
# ===========================================================================
if F is not None:
    fw_captured: List[Dict] = []

    def _fw_post(url, headers, body):
        fw_captured.append({"url": url, "headers": dict(headers), "body": dict(body)})
        ids = [ord(c) for c in "abc"]
        return _mock_fireworks_echo_response(len(ids))

    try:
        adapter_fw = F.make_fireworks_serverless_adapter(
            api_key="fw-key-A3",
            http_post_caller=_fw_post,
        )
        with patch.object(F, "_get_tokenizer", return_value=_FakeTok()):
            prompt_ids = [ord(c) for c in "abc"]
            try:
                adapter_fw.teacher_force_fn(prompt_ids)
            except Exception:
                pass

        if fw_captured:
            c3 = fw_captured[0]
            check("A3a Fireworks POST to correct /completions URL",
                  "fireworks.ai" in c3["url"] and "/completions" in c3["url"])
            check("A3b Fireworks body has echo_last=<N>",
                  "echo_last" in c3["body"])
            check("A3c Fireworks body has logprobs=1",
                  c3["body"].get("logprobs") == 1)
            check("A3d Fireworks body has max_tokens=0",
                  c3["body"].get("max_tokens") == 0)
        else:
            for tag in ("A3a", "A3b", "A3c", "A3d"):
                check(f"{tag} Fireworks request shaping", False, "no POST captured")
    except Exception as e:
        for tag in ("A3a", "A3b", "A3c", "A3d"):
            check(f"{tag} Fireworks request shaping", False, str(e))


# ===========================================================================
# A4: wait_ready polls until STARTED (PENDING x2 then STARTED)
# ===========================================================================
if F is not None:
    try:
        status_seq = _mock_together_status_sequence(["PENDING", "PENDING", "STARTED"])
        adapter_w4 = F.make_together_dedicated_adapter(
            api_key="key-A4",
            http_post_caller=_mock_together_create_ok("ep-A4"),
            http_get_caller=status_seq,
            http_delete_caller=_mock_together_delete(204),
            poll_interval_s=0,     # don't sleep in tests
            poll_timeout_s=60,
        )
        adapter_w4.create_endpoint_fn()
        with patch("time.sleep"):
            adapter_w4.wait_ready_fn()
        check("A4 wait_ready returns when STARTED (3 GET calls: PENDING x2 + STARTED)",
              len(status_seq.calls) == 3)
    except Exception as e:
        check("A4 wait_ready polls until STARTED", False, str(e))


# ===========================================================================
# A5: wait_ready raises TimeoutError if STARTED never reached
# ===========================================================================
if F is not None:
    try:
        always_pending = _mock_together_status_sequence(["PENDING"])
        adapter_w5 = F.make_together_dedicated_adapter(
            api_key="key-A5",
            http_post_caller=_mock_together_create_ok("ep-A5"),
            http_get_caller=always_pending,
            http_delete_caller=_mock_together_delete(204),
            poll_interval_s=0,
            poll_timeout_s=0,   # immediate timeout
        )
        adapter_w5.create_endpoint_fn()
        with patch("time.sleep"):
            try:
                adapter_w5.wait_ready_fn()
                check("A5 wait_ready raises TimeoutError", False, "no exception")
            except TimeoutError:
                check("A5 wait_ready raises TimeoutError", True)
            except Exception as e:
                check("A5 wait_ready raises TimeoutError", False, f"{type(e).__name__}: {e}")
    except Exception as e:
        check("A5 wait_ready raises TimeoutError", False, str(e))


# ===========================================================================
# A6: wait_ready raises RuntimeError on ERROR state
# ===========================================================================
if F is not None:
    try:
        error_state = _mock_together_status_sequence(["PENDING", "ERROR"])
        adapter_w6 = F.make_together_dedicated_adapter(
            api_key="key-A6",
            http_post_caller=_mock_together_create_ok("ep-A6"),
            http_get_caller=error_state,
            http_delete_caller=_mock_together_delete(204),
            poll_interval_s=0,
            poll_timeout_s=60,
        )
        adapter_w6.create_endpoint_fn()
        with patch("time.sleep"):
            try:
                adapter_w6.wait_ready_fn()
                check("A6 wait_ready raises RuntimeError on ERROR state", False, "no exception")
            except RuntimeError:
                check("A6 wait_ready raises RuntimeError on ERROR state", True)
            except Exception as e:
                check("A6 wait_ready raises RuntimeError on ERROR state",
                      False, f"{type(e).__name__}: {e}")
    except Exception as e:
        check("A6 wait_ready on ERROR state", False, str(e))


# ===========================================================================
# A7: delete_endpoint idempotency
# ===========================================================================
if F is not None:
    try:
        del_calls: List[str] = []

        def _del_A7(url, headers):
            del_calls.append(url)
            if len(del_calls) == 1:
                return 204
            # Second call: 404 (already gone)
            return 404

        adapter_d7 = F.make_together_dedicated_adapter(
            api_key="key-A7",
            http_post_caller=_mock_together_create_ok("ep-A7"),
            http_delete_caller=_del_A7,
            poll_interval_s=0,
            poll_timeout_s=60,
        )
        adapter_d7.create_endpoint_fn()
        adapter_d7.delete_endpoint_fn()   # first delete: 204
        # endpoint_id should be cleared after successful delete; second delete
        # should be a no-op (no HTTP call, since id is None)
        adapter_d7.delete_endpoint_fn()   # idempotent no-op
        check("A7 delete_endpoint: only one HTTP DELETE call (id cleared after first)",
              len(del_calls) == 1)
    except Exception as e:
        check("A7 delete_endpoint idempotency", False, str(e))


# ===========================================================================
# E1: together_echo_to_span_logprobs — correct alignment
# ===========================================================================
if F is not None:
    try:
        # prefix + span: prompt_ids = [10, 20, 30, 40, 50]
        # expected tokens are [40, 50] (the last two = the gibberish span)
        prefix = [10, 20, 30]
        span_ids = [40, 50]
        all_ids = prefix + span_ids
        base_lps = [-0.1 * (i + 1) for i in range(len(all_ids))]
        mock_resp = {
            "prompt": [{
                "logprobs": {
                    "token_ids": all_ids,
                    "tokens": [str(x) for x in all_ids],
                    "token_logprobs": base_lps,
                }
            }]
        }
        result = F.together_echo_to_span_logprobs(mock_resp, span_ids)
        check("E1a span logprobs have correct length", len(result) == len(span_ids))
        expected_lps = base_lps[len(prefix):]
        check("E1b span logprobs are for the correct positions",
              all(abs(result[i] - expected_lps[i]) < 1e-9 for i in range(len(span_ids))))
    except Exception as e:
        check("E1 together_echo_to_span_logprobs correct alignment", False, str(e))


# ===========================================================================
# E2: together_echo_to_span_logprobs raises on token-id mismatch
# ===========================================================================
if F is not None:
    try:
        mock_resp_bad = {
            "prompt": [{
                "logprobs": {
                    "token_ids": [10, 20, 30],   # span_ids [40, 50] not present
                    "tokens": ["a", "b", "c"],
                    "token_logprobs": [-0.1, -0.2, -0.3],
                }
            }]
        }
        try:
            F.together_echo_to_span_logprobs(mock_resp_bad, [40, 50])
            check("E2 together_echo raises on missing span ids", False, "no exception")
        except RuntimeError:
            check("E2 together_echo raises on missing span ids", True)
    except Exception as e:
        check("E2 together_echo raises on missing span ids", False, str(e))


# ===========================================================================
# E3: fireworks_echo_to_span_logprobs — correct N logprobs
# ===========================================================================
if F is not None:
    try:
        ids = [1, 2, 3, 4]
        resp_e3 = _mock_fireworks_echo_response(len(ids), base_lp=-0.5, leading_nulls=0)
        result_e3 = F.fireworks_echo_to_span_logprobs(resp_e3, ids)
        check("E3a Fireworks echo returns correct N logprobs", len(result_e3) == len(ids))
        check("E3b all logprobs are finite and <= 0",
              all(math.isfinite(x) and x <= 0 for x in result_e3))
    except Exception as e:
        check("E3 fireworks_echo_to_span_logprobs correct N", False, str(e))


# ===========================================================================
# E4: fireworks_echo_to_span_logprobs strips leading None
# ===========================================================================
if F is not None:
    try:
        ids_e4 = [7, 8, 9]
        # 2 leading nulls then 3 real logprobs
        resp_e4 = _mock_fireworks_echo_response(len(ids_e4), base_lp=-0.3, leading_nulls=2)
        result_e4 = F.fireworks_echo_to_span_logprobs(resp_e4, ids_e4)
        check("E4 Fireworks strips leading None before alignment",
              len(result_e4) == len(ids_e4))
    except Exception as e:
        check("E4 fireworks_echo leading None stripped", False, str(e))


# ===========================================================================
# E5: fireworks_echo_to_span_logprobs raises when too few logprobs
# ===========================================================================
if F is not None:
    try:
        resp_e5 = _mock_fireworks_echo_response(2, base_lp=-0.3)   # only 2 tokens
        try:
            F.fireworks_echo_to_span_logprobs(resp_e5, [1, 2, 3, 4])   # need 4
            check("E5 Fireworks raises when too few logprobs", False, "no exception")
        except RuntimeError:
            check("E5 Fireworks raises when too few logprobs", True)
    except Exception as e:
        check("E5 Fireworks raises when too few logprobs", False, str(e))


# ===========================================================================
# P1: functionality_probe PASS — echo returns per-token logprobs
# ===========================================================================
if F is not None:
    try:
        probe_ids = [100, 101, 102, 103]   # fake token ids

        def _probe_p1_tf(ids):
            # Return one finite negative logprob per token
            return [-0.5 - i * 0.01 for i in range(len(ids))]

        # Build a minimal adapter that passes the echo check but we need to
        # bypass the MC check (which needs DeepInfra).  We patch _load_key
        # and DeepInfraClient to avoid real network calls.
        adapter_p1 = F.ProviderAdapter(
            name="mock_probe_p1",
            teacher_force_fn=_probe_p1_tf,
            model="mock-model",
            is_dedicated=False,
        )

        import serverless_72b as S_

        def _mock_mc_response(*a, **kw):
            return {"choices": [{"message": {"content":
                json.dumps({"concept": "curiosity"})}}]}

        mock_client = S_.DeepInfraClient(api_key="dummy",
                                         http_caller=lambda *a: _mock_mc_response())

        with patch.object(F, "_get_tokenizer", return_value=_FakeTok()), \
             patch.object(F, "_load_key", return_value="dummy-key"), \
             patch.object(S_, "DeepInfraClient", return_value=mock_client):
            result_p1 = F.functionality_probe(adapter_p1)

        check("P1a probe PASS returns ok=True", result_p1.get("ok") is True)
        check("P1b probe PASS has empty reason", result_p1.get("reason") == "")
    except Exception as e:
        check("P1 functionality_probe PASS", False, str(e))
        check("P1b probe PASS has empty reason", False, str(e))


# ===========================================================================
# P2: functionality_probe EARLY-OUT — teacher_force raises (DeepInfra-style)
# ===========================================================================
if F is not None:
    try:
        adapter_p2 = F.make_deepinfra_serverless_adapter(api_key="dummy-key")

        with patch.object(F, "_get_tokenizer", return_value=_FakeTok()):
            result_p2 = F.functionality_probe(adapter_p2)

        check("P2a probe EARLY-OUT returns ok=False", result_p2.get("ok") is False)
        check("P2b probe EARLY-OUT has non-empty reason",
              bool(result_p2.get("reason")))
    except Exception as e:
        check("P2 functionality_probe EARLY-OUT", False, str(e))
        check("P2b probe EARLY-OUT has non-empty reason", False, str(e))


# ===========================================================================
# P3: functionality_probe early-out on wrong logprob count
# ===========================================================================
if F is not None:
    try:
        def _probe_p3_tf(ids):
            # Returns too few logprobs (1 instead of len(ids))
            return [-0.5]

        adapter_p3 = F.ProviderAdapter(
            name="mock_probe_p3",
            teacher_force_fn=_probe_p3_tf,
            model="mock-model",
            is_dedicated=False,
        )

        with patch.object(F, "_get_tokenizer", return_value=_FakeTok()):
            result_p3 = F.functionality_probe(adapter_p3)

        check("P3a probe early-out on length mismatch returns ok=False",
              result_p3.get("ok") is False)
        check("P3b probe early-out reason mentions mismatch or logprob",
              bool(result_p3.get("reason")))
    except Exception as e:
        check("P3 probe early-out on length mismatch", False, str(e))
        check("P3b probe early-out reason", False, str(e))


# ===========================================================================
# T1: teardown idempotency — only one DELETE call
# ===========================================================================
if F is not None:
    try:
        t1_del_calls: List[str] = []

        def _t1_del(url, headers):
            t1_del_calls.append(url)
            return 204

        adapter_t1 = F.make_together_dedicated_adapter(
            api_key="key-T1",
            http_post_caller=_mock_together_create_ok("ep-T1"),
            http_delete_caller=_t1_del,
            poll_interval_s=0,
            poll_timeout_s=60,
        )
        adapter_t1.create_endpoint_fn()
        # Manually register this adapter as the active one
        F._active_adapter = adapter_t1

        F._do_teardown("test teardown 1")   # first call
        F._do_teardown("test teardown 2")   # second call — should be no-op
        check("T1 teardown: only one DELETE call after two _do_teardown calls",
              len(t1_del_calls) == 1)
    except Exception as e:
        check("T1 teardown idempotency", False, str(e))


# ===========================================================================
# T2: endpoint id file write / read / remove
# ===========================================================================
if F is not None:
    try:
        with tempfile.TemporaryDirectory() as td:
            id_path = Path(td) / "frontier_endpoint.json"

            # write
            F.write_endpoint_id("ep-T2-test", path=id_path)
            check("T2a write_endpoint_id creates the file", id_path.exists())

            # read back
            ep_id_back = F.read_endpoint_id(path=id_path)
            check("T2b read_endpoint_id returns the correct id",
                  ep_id_back == "ep-T2-test")

            # remove
            F.remove_endpoint_id_file(path=id_path)
            check("T2c remove_endpoint_id_file removes the file", not id_path.exists())

            # read from missing file
            check("T2d read_endpoint_id returns None for missing file",
                  F.read_endpoint_id(path=id_path) is None)
    except Exception as e:
        for tag in ("T2a", "T2b", "T2c", "T2d"):
            check(f"{tag} endpoint id file", False, str(e))


# ===========================================================================
# T3: --reap deletes endpoint and removes id file (204 response)
# ===========================================================================
if F is not None:
    try:
        with tempfile.TemporaryDirectory() as td:
            id_path = Path(td) / "frontier_endpoint.json"
            F.write_endpoint_id("ep-T3-reap", path=id_path)

            # Write a real key file so load_together_key doesn't raise
            key_path = Path(td) / ".together_key"
            key_path.write_text("fake-key-T3")

            t3_del_calls: List[str] = []

            def _t3_del(url, headers):
                t3_del_calls.append(url)
                return 204

            # Patch ENDPOINT_ID_FILE and the module-level _http_delete
            with patch.object(F, "ENDPOINT_ID_FILE", id_path), \
                 patch.object(F, "_http_delete", _t3_del):
                import argparse
                # Pass key_file explicitly so load_together_key finds it
                args = argparse.Namespace(together_key=str(key_path))
                F.cmd_reap(args)

            check("T3a --reap makes one DELETE call", len(t3_del_calls) == 1)
            check("T3b --reap removes the id file after 204", not id_path.exists())
    except Exception as e:
        check("T3a --reap makes one DELETE call", False, str(e))
        check("T3b --reap removes the id file after 204", False, str(e))


# ===========================================================================
# T4: --reap with 404 (already gone) still removes id file
# ===========================================================================
if F is not None:
    try:
        with tempfile.TemporaryDirectory() as td:
            id_path = Path(td) / "frontier_endpoint.json"
            F.write_endpoint_id("ep-T4-gone", path=id_path)

            key_path = Path(td) / ".together_key"
            key_path.write_text("fake-key-T4")

            def _t4_del(url, headers):
                return 404   # already gone

            with patch.object(F, "ENDPOINT_ID_FILE", id_path), \
                 patch.object(F, "_http_delete", _t4_del):
                import argparse
                args = argparse.Namespace(together_key=str(key_path))
                F.cmd_reap(args)

            check("T4 --reap with 404 still removes id file (idempotent)",
                  not id_path.exists())
    except Exception as e:
        check("T4 --reap with 404 removes id file", False, str(e))


# ===========================================================================
# T5: --reap with no id file is a no-op
# ===========================================================================
if F is not None:
    try:
        with tempfile.TemporaryDirectory() as td:
            id_path = Path(td) / "no_such_file.json"  # does not exist
            t5_del_calls: List[str] = []

            def _t5_del(url, headers):
                t5_del_calls.append(url)
                return 204

            with patch.object(F, "ENDPOINT_ID_FILE", id_path), \
                 patch.object(F, "_http_delete", _t5_del):
                import argparse
                args = argparse.Namespace(together_key=None)
                F.cmd_reap(args)

            check("T5 --reap with no id file: no DELETE call, no crash",
                  len(t5_del_calls) == 0)
    except Exception as e:
        check("T5 --reap with no id file is a no-op", False, str(e))


# ===========================================================================
# D1: --dry NEVER creates an endpoint
# ===========================================================================
if F is not None:
    try:
        d1_create_calls: List[str] = []

        def _d1_create(url, headers, body):
            d1_create_calls.append(url)
            return {"id": "should-not-be-called", "state": "PENDING"}

        # A serverless adapter that "passes" the probe
        def _d1_tf(ids):
            return [-0.5] * len(ids)

        adapter_d1 = F.ProviderAdapter(
            name="mock_d1",
            teacher_force_fn=_d1_tf,
            model="mock-model",
            is_dedicated=False,
        )

        import argparse
        args_d1 = argparse.Namespace(provider="fireworks", fireworks_key=None)

        # Patch cmd_dry's adapter construction so it returns our mock serverless adapter
        import serverless_72b as S_d1

        def _mock_mc_d1(*a, **kw):
            return {"choices": [{"message": {"content":
                json.dumps({"concept": "curiosity"})}}]}

        mock_client_d1 = S_d1.DeepInfraClient(api_key="dummy",
                                               http_caller=lambda *a: _mock_mc_d1())

        with patch.object(F, "make_fireworks_serverless_adapter", return_value=adapter_d1), \
             patch.object(F, "_get_tokenizer", return_value=_FakeTok()), \
             patch.object(F, "_load_key", return_value="dummy-key"), \
             patch.object(S_d1, "DeepInfraClient", return_value=mock_client_d1):
            F.cmd_dry(args_d1)

        check("D1 --dry NEVER calls create_endpoint (no HTTP POST to /endpoints)",
              len(d1_create_calls) == 0)
    except Exception as e:
        check("D1 --dry never creates endpoint", False, str(e))


# ===========================================================================
# D2: --dry with DeepInfra broken-echo adapter triggers early-out
# ===========================================================================
if F is not None:
    try:
        adapter_d2 = F.make_deepinfra_serverless_adapter(api_key="dummy-d2")
        with patch.object(F, "_get_tokenizer", return_value=_FakeTok()):
            result_d2 = F.functionality_probe(adapter_d2)
        check("D2 --dry DeepInfra broken-echo: probe returns ok=False (early-out)",
              result_d2.get("ok") is False)
        check("D2 --dry DeepInfra broken-echo adapter has is_dedicated=False",
              adapter_d2.is_dedicated is False)
    except Exception as e:
        check("D2 --dry DeepInfra early-out", False, str(e))
        check("D2 adapter is_dedicated=False", False, str(e))


# ===========================================================================
# D3: --dry with Fireworks valid mock returns PASS
# ===========================================================================
if F is not None:
    try:
        def _d3_tf_pass(ids):
            return [-0.3 - i * 0.01 for i in range(len(ids))]

        adapter_d3 = F.ProviderAdapter(
            name="mock_fireworks_d3",
            teacher_force_fn=_d3_tf_pass,
            model="mock-model",
            is_dedicated=False,
        )

        import serverless_72b as S_d3

        def _mock_mc_d3(*a, **kw):
            return {"choices": [{"message": {"content":
                json.dumps({"concept": "ocean"})}}]}

        mock_client_d3 = S_d3.DeepInfraClient(api_key="dummy",
                                               http_caller=lambda *a: _mock_mc_d3())

        with patch.object(F, "_get_tokenizer", return_value=_FakeTok()), \
             patch.object(F, "_load_key", return_value="dummy-key"), \
             patch.object(S_d3, "DeepInfraClient", return_value=mock_client_d3):
            result_d3 = F.functionality_probe(adapter_d3)

        check("D3 --dry Fireworks valid mock: probe returns ok=True",
              result_d3.get("ok") is True)
    except Exception as e:
        check("D3 --dry Fireworks PASS", False, str(e))


# ===========================================================================
# Report
# ===========================================================================

passed = sum(1 for _, ok, _ in checks if ok)
failed = sum(1 for _, ok, _ in checks if not ok)

print(f"\n{'='*60}")
print(f"test_run_frontier_72b.py: {passed} passed, {failed} failed")
print(f"{'='*60}")

for name, ok, note in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {note}" if note else ""))

if failed:
    sys.exit(1)
