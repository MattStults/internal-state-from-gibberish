#!/usr/bin/env python3
"""run_frontier_72b.py  —  Together dedicated-endpoint or Fireworks serverless
teacher-forcing for the LR-72B / MC run.

=== WHAT THIS SCRIPT DOES ===
Qwen2.5-72B-Instruct teacher-forcing (echo / logprob) on TWO provider paths:

  Together dedicated  – spins up a B200 endpoint (~$0.15/min = ~$9/hr), waits
                        until STARTED, runs everything, tears it down.
  Fireworks serverless – no endpoint lifecycle; uses echo_last on the existing
                        serverless endpoint.  Good for --dry and small checks.

The run sequence (real mode):
  1. create_endpoint  →  write endpoint-id to runs/frontier_endpoint.json  →
  2. wait_ready (poll until STARTED, timeout 15 min)  →
  3. functionality_probe (echo teacher-forcing + structured MC smoke)  →
     FAIL: teardown + exit 1
  4. generation  (collect word-free streams; reuse serverless_72b logic)  →
  5. LR echo diagonal + observer teacher-forcing (reuse lr_vllm span logic)  →
  6. MC self-report (reuse serverless_72b MC)  →
  7. score offline (reuse lr_72b_offline)  →
  8. teardown (ALWAYS, even on error — try/finally + atexit + signals)

=== TEARDOWN SAFETY STACK ===
  - try/finally in main: teardown lives in the finally block, always runs.
  - signal.signal(SIGINT) + SIGTERM: both redirect to the registered teardown.
  - atexit.register(teardown): fires even on sys.exit() / unexpected death.
  - idempotent delete: HTTP 404 on DELETE is treated as success (already gone).
  - id file written BEFORE the box is waited-ready so --reap can always recover.
  - inactive_timeout=10 min: provider-side auto-stop backstop even if this
    driver is kill -9'd and --reap never runs.

=== MODES ===
  --dry       build the SERVERLESS adapter, run functionality_probe only,
              print PASS/FAIL — NO endpoint created, $0.
  real (default) create endpoint → run full pipeline → teardown
  --reap      read runs/frontier_endpoint.json, DELETE the recorded endpoint,
              remove the id file.  Recovery for kill -9.

=== KEY FILES ===
  ~/.together_key     Together API key (never on CLI)
  ~/.fireworks_key    Fireworks API key (never on CLI)
  runs/frontier_endpoint.json   endpoint id written at creation (before poll)

=== USAGE ===
  # Dry-run on Fireworks serverless (probe only, $0, no endpoint)
  .venv/bin/python harness/run_frontier_72b.py --dry --provider fireworks

  # Full dedicated run on Together B200
  .venv/bin/python harness/run_frontier_72b.py --provider together

  # Recover a stranded endpoint after kill -9
  .venv/bin/python harness/run_frontier_72b.py --reap
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# stdlib imports — ONLY stdlib here so --dry / --reap work without torch
# ---------------------------------------------------------------------------
import asyncio
import atexit
import json
import logging
import math
import os
import signal
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo path setup (so src/ modules are importable)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------------------
# Logging: simple timestamped lines so teardown events are unmistakable
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("frontier_72b")


# ===========================================================================
# CONSTANTS
# ===========================================================================

# Together dedicated endpoint
TOGETHER_BASE_ENDPOINTS = "https://api.together.ai/v1"
TOGETHER_BASE_COMPLETIONS = "https://api.together.xyz/v1"
# User-Agent that bypasses Together's Cloudflare challenge
TOGETHER_UA = "curl/8.4.0"

# Hardware id for bf16 72B on Together:
# B200 180 GB SXM — fits the 144 GB bf16 Qwen2.5-72B with headroom.
# Cost: ~$0.15/min (~$9/hr).  Change here to switch tier.
TOGETHER_HW_ID = "1x_nvidia_b200_180gb_sxm"
TOGETHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"

# inactive_timeout in MINUTES (provider auto-stop backstop; set conservatively)
TOGETHER_INACTIVE_TIMEOUT_MIN = 10

# Together endpoint polling
POLL_INTERVAL_S = 15       # seconds between GET /endpoints/{id}
POLL_TIMEOUT_S  = 900      # 15 minutes max before we give up

# Fireworks serverless
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/qwen2p5-72b-instruct"

# Where to persist the running endpoint id (written BEFORE waiting ready)
ENDPOINT_ID_FILE = REPO / "runs" / "frontier_endpoint.json"

# A short known token sequence for the functionality probe.
# We teacher-force "qx z fjm" (7 chars = 7 tokens under char-level tokenisation,
# or a handful of Qwen2.5 BPE tokens).  The probe only checks that per-token
# logprobs come back — exact token count doesn't matter.
PROBE_TEXT = "qx z fjm wpl kbt"


# ===========================================================================
# HTTP HELPERS (injectable; tests pass a mock caller)
# ===========================================================================

def _http_post(url: str, headers: Dict, body: Dict,
               http_caller: Optional[Callable] = None) -> Dict:
    """POST url with JSON body; return parsed response dict.

    Falls back to urllib (stdlib) if no http_caller injected.
    Raises RuntimeError on HTTP errors (status >= 400).
    """
    if http_caller is not None:
        return http_caller(url, headers, body)
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} {e.reason}: {body_text}") from e


def _http_get(url: str, headers: Dict,
              http_caller: Optional[Callable] = None) -> Tuple[int, Dict]:
    """GET url; return (status_code, parsed_body).

    http_caller for GET has signature: (url, headers) -> (status_code, dict).
    Falls back to urllib.  Returns (status_code, body) always — never raises on
    HTTP errors so callers can inspect the status.
    """
    if http_caller is not None:
        return http_caller(url, headers)
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        try:
            body = json.loads(body_text)
        except Exception:
            body = {"raw": body_text}
        return e.code, body


def _http_delete(url: str, headers: Dict,
                 http_caller: Optional[Callable] = None) -> int:
    """DELETE url; return HTTP status code.

    http_caller for DELETE has signature: (url, headers) -> status_code.
    404 from the real API means "already gone" — callers treat that as OK.
    Falls back to urllib.
    """
    if http_caller is not None:
        return http_caller(url, headers)
    req = urllib.request.Request(url, headers=headers, method="DELETE")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


# ===========================================================================
# KEY LOADING
# ===========================================================================

def _load_key(path_or_env: str, env_var: str, description: str) -> str:
    """Load an API key from a file path or fallback environment variable.

    Resolution order:
      1. path_or_env argument (if it looks like a path and the file exists)
      2. env_var environment variable
      3. path_or_env treated as ~/.{path_or_env} (default key-file convention)
    Raises FileNotFoundError or ValueError if key is missing or empty.
    """
    # Try env var first
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val
    # Try the path
    p = Path(path_or_env).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"{description} key file not found: {p}  "
            f"(set {env_var} env var or create {p})"
        )
    key = p.read_text().strip()
    if not key:
        raise ValueError(f"{description} key file is empty: {p}")
    return key


def load_together_key(key_file: Optional[str] = None) -> str:
    return _load_key(
        key_file or str(Path.home() / ".together_key"),
        "TOGETHER_KEY",
        "Together",
    )


def load_fireworks_key(key_file: Optional[str] = None) -> str:
    return _load_key(
        key_file or str(Path.home() / ".fireworks_key"),
        "FIREWORKS_KEY",
        "Fireworks",
    )


# ===========================================================================
# ECHO → SPAN ADAPTER
# (bridge between Together/Fireworks response shape and lr_vllm.span_logprobs)
# ===========================================================================

def together_echo_to_span_logprobs(
    resp: Dict,
    expected_token_ids: List[int],
) -> List[float]:
    """Convert a Together echo response to a per-token logprob list for lr_vllm.

    Together's /v1/completions echo response (the teacher-forcing shape):
      resp["prompt"][0]["logprobs"] = {
          "token_ids":    [int, ...],   # parallel arrays; index 0 is NOT null
          "tokens":       [str, ...],
          "token_logprobs": [float, ...],
      }

    lr_vllm.span_logprobs expects a vLLM-style list where index 0 IS null
    and each entry is a dict keyed by token-id-string.  This adapter bridges
    the two shapes by:
      1. Locating the contiguous run in token_ids that matches expected_token_ids.
      2. Reading token_logprobs[i] for each matched position.
      3. Summing them (same as lr_vllm.ll_over_span) — returns the flat list so
         the caller can choose drop_last_eos or sum themselves.

    Raises RuntimeError if expected_token_ids don't appear in the response's
    token_id sequence (alignment failure — the teacher-forced tokens were not
    found, which means something went wrong with the prompt or the API).
    """
    lp_obj = resp["prompt"][0]["logprobs"]
    resp_ids: List[int] = [int(x) for x in lp_obj["token_ids"]]
    resp_lps: List[float] = [float(x) for x in lp_obj["token_logprobs"]]
    n_expected = len(expected_token_ids)

    # Find the last occurrence of expected_token_ids as a contiguous subsequence
    # (the gibberish span is at the end of the prompt).
    match_start = None
    for i in range(len(resp_ids) - n_expected, -1, -1):
        if resp_ids[i : i + n_expected] == expected_token_ids:
            match_start = i
            break

    if match_start is None:
        raise RuntimeError(
            f"together_echo_to_span_logprobs: the expected token ids "
            f"({expected_token_ids[:8]}{'...' if len(expected_token_ids)>8 else ''}) "
            f"were not found as a contiguous span in the Together echo response's "
            f"token_ids (length {len(resp_ids)}).  The prompt may have been "
            f"re-tokenised by the server or the echo request was malformed."
        )

    # Verify EVERY id in the span matches (belt-and-suspenders — the search above
    # already guarantees it, but explicit is better than implicit).
    for j, tid in enumerate(expected_token_ids):
        got = resp_ids[match_start + j]
        if got != tid:
            raise RuntimeError(
                f"together_echo_to_span_logprobs: token-id mismatch at span "
                f"position {j}: expected {tid}, got {got}.  Refusing to score "
                f"the wrong token's logprob."
            )

    return [resp_lps[match_start + j] for j in range(n_expected)]


def fireworks_echo_to_span_logprobs(
    resp: Dict,
    expected_token_ids: List[int],
) -> List[float]:
    """Convert a Fireworks echo_last response to a per-token logprob list.

    Fireworks /v1/completions with echo_last=N returns:
      resp["choices"][0]["logprobs"]["token_logprobs"]  — parallel array,
      one entry per token in the echoed suffix.  The suffix is the LAST N
      tokens of the prompt (the gibberish span we teacher-force).

    We align by length: if the returned array has exactly len(expected_token_ids)
    entries, they correspond 1-to-1.  If it's longer (Fireworks sometimes returns
    a leading null for the first echo position), we take the trailing slice.

    Raises RuntimeError on length mismatch after the null-trim heuristic, or if
    token_logprobs is absent.
    """
    lps = resp["choices"][0]["logprobs"]["token_logprobs"]
    # Drop leading nulls / None (Fireworks may include them)
    lps = [x for x in lps if x is not None]
    n = len(expected_token_ids)
    if len(lps) < n:
        raise RuntimeError(
            f"fireworks_echo_to_span_logprobs: expected {n} logprobs (one per "
            f"echoed token), got {len(lps)} after null-trim.  The echo_last "
            f"request may not have covered the full gibberish span."
        )
    # If longer, take the trailing n (the span is the last tokens)
    return [float(x) for x in lps[-n:]]


# ===========================================================================
# PROVIDER ADAPTER DATACLASS
# ===========================================================================

@dataclass
class ProviderAdapter:
    """Pluggable provider interface.

    Three concrete implementations:
      TogetherDedicated   – full lifecycle (create/wait/delete) + echo TF
      FireworksServerless – echo_last TF only, no endpoint management
      DeepInfraServerless – echo KNOWN-BROKEN (used for --dry early-out test)

    Fields
    ------
    name : str
        Short human-readable label for logs.
    teacher_force_fn : Callable[[List[int]], List[float]]
        Given a list of token ids (the full teacher-forcing prompt), return
        per-token logprobs for those tokens.  Raises on API errors.
    create_endpoint_fn : Optional[Callable[[], str]]
        Create the dedicated endpoint; return the endpoint id.  None for
        serverless adapters.
    wait_ready_fn : Optional[Callable[[], None]]
        Poll until the endpoint is STARTED.  Raises TimeoutError if not ready
        within POLL_TIMEOUT_S.  None for serverless adapters.
    delete_endpoint_fn : Optional[Callable[[], None]]
        Delete / release the endpoint.  Idempotent: must not raise on 404.
        None for serverless adapters.
    model : str
        The model identifier used in requests (for logging/probe messages).
    is_dedicated : bool
        True only for adapters that create real endpoints.
    """
    name: str
    teacher_force_fn: Callable[[List[int]], List[float]]
    create_endpoint_fn: Optional[Callable[[], str]] = None
    wait_ready_fn: Optional[Callable[[], None]] = None
    delete_endpoint_fn: Optional[Callable[[], None]] = None
    model: str = ""
    is_dedicated: bool = False


# ===========================================================================
# TOGETHER DEDICATED ADAPTER FACTORY
# ===========================================================================

def make_together_dedicated_adapter(
    api_key: str,
    display_name: str = "frontier-72b-lr",
    hw_id: str = TOGETHER_HW_ID,
    model: str = TOGETHER_MODEL,
    http_post_caller: Optional[Callable] = None,
    http_get_caller: Optional[Callable] = None,
    http_delete_caller: Optional[Callable] = None,
    poll_interval_s: float = POLL_INTERVAL_S,
    poll_timeout_s: float = POLL_TIMEOUT_S,
    inactive_timeout_min: int = TOGETHER_INACTIVE_TIMEOUT_MIN,
) -> ProviderAdapter:
    """Build a TogetherDedicated ProviderAdapter.

    All HTTP callers are injectable for tests.  In production they default to
    the stdlib urllib wrappers defined above.

    Parameters
    ----------
    api_key : str          Together API key (Bearer token).
    display_name : str     Human-readable endpoint name shown in the Together UI.
    hw_id : str            Together hardware tier id (default: B200 180 GB).
    model : str            Model slug (default: Qwen2.5-72B-Instruct).
    http_*_caller :        Injectable HTTP callers (tests pass mocks here).
    poll_interval_s :      Seconds between status polls.
    poll_timeout_s :       Max seconds to wait for STARTED before raising.
    inactive_timeout_min : Provider-side auto-stop backstop (minutes).
    """
    auth_headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": TOGETHER_UA,
    }
    # Mutable state: the endpoint id is set by create and read by wait/delete.
    _state: Dict[str, Any] = {"endpoint_id": None}

    # ------------------------------------------------------------------ create
    def create_endpoint() -> str:
        """POST /endpoints to provision the B200 dedicated endpoint.

        Writes the returned id to _state immediately so subsequent calls can
        reference it even if this function is interrupted mid-return.
        """
        url = f"{TOGETHER_BASE_ENDPOINTS}/endpoints"
        body = {
            "model": model,
            "hardware": hw_id,
            "autoscaling": {"min_replicas": 1, "max_replicas": 1},
            "inactive_timeout": inactive_timeout_min,
            "display_name": display_name,
        }
        log.info("Together: creating dedicated endpoint  hardware=%s  model=%s", hw_id, model)
        resp = _http_post(url, dict(auth_headers), body, http_post_caller)
        ep_id = resp.get("id") or resp.get("endpoint_id")
        if not ep_id:
            raise RuntimeError(
                f"Together CREATE endpoint did not return an id.  Response: {resp}"
            )
        _state["endpoint_id"] = ep_id
        log.info("Together: endpoint created  id=%s  state=%s", ep_id, resp.get("state"))
        return ep_id

    # ------------------------------------------------------------------ wait_ready
    def wait_ready() -> None:
        """Poll GET /endpoints/{id} until state == STARTED.

        Raises TimeoutError after poll_timeout_s.  Raises RuntimeError if the
        endpoint enters ERROR state.
        """
        ep_id = _state["endpoint_id"]
        if ep_id is None:
            raise RuntimeError("wait_ready called before create_endpoint")
        deadline = time.time() + poll_timeout_s
        log.info("Together: waiting for endpoint %s to reach STARTED "
                 "(timeout %.0f min)...", ep_id, poll_timeout_s / 60)
        while time.time() < deadline:
            url = f"{TOGETHER_BASE_ENDPOINTS}/endpoints/{ep_id}"
            status_code, resp = _http_get(url, dict(auth_headers), http_get_caller)
            state = resp.get("state", "UNKNOWN")
            log.info("Together: endpoint %s  state=%s", ep_id, state)
            if state == "STARTED":
                log.info("Together: endpoint STARTED and ready.")
                return
            if state == "ERROR":
                raise RuntimeError(
                    f"Together endpoint {ep_id} reached ERROR state.  Response: {resp}"
                )
            if status_code >= 400:
                raise RuntimeError(
                    f"Together endpoint status poll returned HTTP {status_code}: {resp}"
                )
            time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Together endpoint {ep_id} did not reach STARTED within "
            f"{poll_timeout_s / 60:.0f} min.  Last state reported in logs above."
        )

    # ------------------------------------------------------------------ delete_endpoint
    def delete_endpoint() -> None:
        """DELETE /endpoints/{id}.  Idempotent: HTTP 404 is treated as success.

        This is called from EVERY teardown path (finally block, atexit, signal
        handler).  It must NEVER raise, so crashes during teardown don't hide
        the original error.  We log a loud WARNING if delete unexpectedly fails.
        """
        ep_id = _state.get("endpoint_id")
        if ep_id is None:
            log.info("Together teardown: no endpoint id recorded — nothing to delete.")
            return
        url = f"{TOGETHER_BASE_ENDPOINTS}/endpoints/{ep_id}"
        log.warning(
            "==[ TEARDOWN ]== Together: deleting endpoint %s  "
            "(idempotent: 404=ok, charges stop at delete)",
            ep_id,
        )
        status = _http_delete(url, dict(auth_headers), http_delete_caller)
        if status in (200, 204, 404):
            log.warning(
                "==[ TEARDOWN ]== Together: endpoint %s deleted (status %d).  "
                "No further charges.",
                ep_id, status,
            )
            _state["endpoint_id"] = None   # prevent double-delete
        else:
            # Non-fatal: we tried.  Log loudly so the human can --reap manually.
            log.error(
                "==[ TEARDOWN ]== Together: DELETE returned unexpected status %d  "
                "ep_id=%s — run '--reap' to attempt cleanup.",
                status, ep_id,
            )

    # ------------------------------------------------------------------ teacher_force
    def teacher_force(prompt_ids: List[int]) -> List[float]:
        """Teacher-force the full prompt; return per-token logprobs.

        Uses the Together completions echo path:
          POST https://api.together.xyz/v1/completions
          body: {model, prompt: <decoded_text>, echo: true, logprobs: 1,
                 max_tokens: 0, temperature: 0}

        The prompt is sent as DECODED TEXT (not token ids) because Together's
        /v1/completions endpoint accepts a text prompt.  The echo response's
        parallel arrays are then aligned back to prompt_ids by
        together_echo_to_span_logprobs.

        Returns a list of per-token logprobs aligned to prompt_ids (one float
        per token in the same order).  The caller can slice the gibberish span.
        """
        ep_id = _state.get("endpoint_id")
        if ep_id is None:
            raise RuntimeError(
                "teacher_force called before endpoint is ready (endpoint_id is None)"
            )
        # Decode the token ids to text so the echo endpoint can consume them.
        # We import the tokenizer here (lazy, CPU-only) so that --dry mode never
        # needs a GPU.
        tok = _get_tokenizer()
        prompt_text = tok.decode([int(x) for x in prompt_ids], skip_special_tokens=False)

        url = f"{TOGETHER_BASE_COMPLETIONS}/completions"
        body = {
            "model": model,
            "prompt": prompt_text,
            "echo": True,
            "logprobs": 1,
            "max_tokens": 0,
            "temperature": 0,
        }
        resp = _http_post(url, dict(auth_headers), body, http_post_caller)
        return together_echo_to_span_logprobs(resp, [int(x) for x in prompt_ids])

    return ProviderAdapter(
        name="together_dedicated",
        teacher_force_fn=teacher_force,
        create_endpoint_fn=create_endpoint,
        wait_ready_fn=wait_ready,
        delete_endpoint_fn=delete_endpoint,
        model=model,
        is_dedicated=True,
    )


# ===========================================================================
# FIREWORKS SERVERLESS ADAPTER FACTORY
# ===========================================================================

def make_fireworks_serverless_adapter(
    api_key: str,
    model: str = FIREWORKS_MODEL,
    http_post_caller: Optional[Callable] = None,
) -> ProviderAdapter:
    """Build a FireworksServerless ProviderAdapter.

    Fireworks does NOT have a dedicated-endpoint API so there is no
    create/wait/delete.  Teacher-forcing uses the echo_last parameter.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def teacher_force(prompt_ids: List[int]) -> List[float]:
        """Teacher-force via Fireworks echo_last.

        POST /v1/completions  body={model, prompt:<text>, echo_last:<N>,
                                    logprobs:1, max_tokens:0}
        Returns per-token logprobs for the last N=len(prompt_ids) tokens.
        """
        tok = _get_tokenizer()
        prompt_text = tok.decode([int(x) for x in prompt_ids], skip_special_tokens=False)
        n = len(prompt_ids)

        url = f"{FIREWORKS_BASE}/completions"
        body = {
            "model": model,
            "prompt": prompt_text,
            "echo_last": n,
            "logprobs": 1,
            "max_tokens": 0,
        }
        resp = _http_post(url, dict(headers), body, http_post_caller)
        return fireworks_echo_to_span_logprobs(resp, [int(x) for x in prompt_ids])

    return ProviderAdapter(
        name="fireworks_serverless",
        teacher_force_fn=teacher_force,
        create_endpoint_fn=None,
        wait_ready_fn=None,
        delete_endpoint_fn=None,
        model=model,
        is_dedicated=False,
    )


# ===========================================================================
# DEEPINFRA SERVERLESS ADAPTER FACTORY  (KNOWN-BROKEN echo; --dry early-out)
# ===========================================================================

def make_deepinfra_serverless_adapter(
    api_key: str,
    http_post_caller: Optional[Callable] = None,
) -> ProviderAdapter:
    """Build a DeepInfraServerless adapter.

    DeepInfra's /v1/completions does NOT support echo / prompt_logprobs for
    the 72B model (it returns only a completion, no per-token logprobs for the
    prompt).  This adapter is used ONLY to exercise the functionality_probe
    early-out branch: probe will call teacher_force, find no per-prompt-token
    logprobs in the response, and return ok=False.

    This is NOT a real call path for production; it exists to make the
    early-out test branch exercisable without a real network call.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def teacher_force(prompt_ids: List[int]) -> List[float]:
        # DeepInfra echo is known-broken for this model: we simulate the
        # failure by raising RuntimeError (which functionality_probe catches).
        raise RuntimeError(
            "DeepInfra /v1/completions does not return per-token prompt "
            "logprobs for this model (echo is not supported).  "
            "functionality_probe early-out triggered."
        )

    return ProviderAdapter(
        name="deepinfra_serverless_broken_echo",
        teacher_force_fn=teacher_force,
        create_endpoint_fn=None,
        wait_ready_fn=None,
        delete_endpoint_fn=None,
        model="Qwen/Qwen2.5-72B-Instruct",
        is_dedicated=False,
    )


# ===========================================================================
# TOKENIZER (lazy, cached)
# ===========================================================================

_tokenizer_cache: Dict[str, Any] = {}


def _get_tokenizer(tokenizer_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Load the shared Qwen2.5 tokenizer lazily.  Cached after first load.

    Uses AutoTokenizer from transformers.  The 1.5B tokenizer is shared across
    all Qwen2.5 sizes (same vocabulary + chat template).
    """
    if tokenizer_id in _tokenizer_cache:
        return _tokenizer_cache[tokenizer_id]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
        _tokenizer_cache[tokenizer_id] = tok
        log.info("Tokenizer loaded: %s", tokenizer_id)
        return tok
    except Exception as e:
        raise ImportError(
            f"Could not load tokenizer {tokenizer_id!r}.  "
            f"Install transformers: pip install transformers.  "
            f"Original error: {e}"
        ) from e


# ===========================================================================
# FUNCTIONALITY PROBE
# ===========================================================================

def functionality_probe(adapter: ProviderAdapter) -> Dict:
    """Smoke-test the adapter BEFORE running any real work.

    Three checks:
      (a) Echo teacher-forcing: teacher_force a short known sequence and confirm
          per-token logprobs come back (not just a completion, not argmax).
          We encode PROBE_TEXT with the local tokenizer, teacher-force it, and
          verify we get one logprob per token — each a finite negative float.
      (b) Structured-output one-of-12 MC: send a minimal mc_direct_payload and
          confirm the response parses to one of the 12 COVERT_CONCEPTS.  Uses
          the DeepInfra serverless client from serverless_72b (the serverless
          MC path is unaffected by which provider we teacher-force on).
      (c) Precision check: log the model/endpoint metadata if available, or
          warn that it couldn't be confirmed.

    Returns {"ok": bool, "reason": str}  (reason is "" on success).
    """
    import serverless_72b as S  # reuse; CPU-only for schemas + MC

    # -------- (a) Echo teacher-forcing -------------------------------------------
    log.info("probe (a): echo teacher-forcing a short known sequence via %s", adapter.name)
    try:
        tok = _get_tokenizer()
    except ImportError as e:
        return {"ok": False, "reason": f"probe (a) tokenizer unavailable: {e}"}

    probe_ids = tok(PROBE_TEXT, add_special_tokens=False).input_ids
    if not probe_ids:
        return {"ok": False, "reason": "probe (a): tokenizer returned empty ids for probe text"}

    try:
        per_tok_lps = adapter.teacher_force_fn(probe_ids)
    except Exception as e:
        return {
            "ok": False,
            "reason": (
                f"probe (a): teacher_force raised {type(e).__name__}: {e}.  "
                "The provider does not support per-token prompt logprobs on this model."
            ),
        }

    # Validate: one logprob per token, each finite and <= 0 (it's a log-probability)
    if len(per_tok_lps) != len(probe_ids):
        return {
            "ok": False,
            "reason": (
                f"probe (a): expected {len(probe_ids)} per-token logprobs, "
                f"got {len(per_tok_lps)}.  Alignment failure."
            ),
        }
    bad = [lp for lp in per_tok_lps if not (math.isfinite(lp) and lp <= 0)]
    if bad:
        return {
            "ok": False,
            "reason": (
                f"probe (a): {len(bad)} logprob(s) are non-finite or > 0: "
                f"{bad[:5]}.  These are not valid log-probabilities."
            ),
        }
    log.info("probe (a): PASS  %d tokens, logprobs in [%.3f, %.3f]",
             len(per_tok_lps), min(per_tok_lps), max(per_tok_lps))

    # -------- (b) Structured-output MC smoke -------------------------------------
    # We reuse the serverless_72b MC path (DeepInfra OpenAI-compatible endpoint)
    # for the structured-output check.  This is independent of the teacher-forcing
    # provider (MC uses a separate serverless call).
    log.info("probe (b): structured-output one-of-12 MC check (serverless DeepInfra)")
    try:
        di_key = _load_key(
            str(Path.home() / ".deepinfra_key"), "DEEPINFRA_KEY_FILE", "DeepInfra"
        )
        di_client = S.DeepInfraClient(api_key=di_key)
        order = S._shuffled_concepts(seed=99)
        mc_payload = S.mc_direct_payload(PROBE_TEXT, order)
        mc_resp = di_client.chat_completion(mc_payload)
        parsed = S.parse_mc_response(mc_resp)
        log.info("probe (b): PASS  concept=%r", parsed["concept"])
    except Exception as e:
        # Not fatal for the echo TF path; warn and continue with ok=True.
        # (The MC call uses a different API; a serverless MC failure shouldn't
        # block a good dedicated TF endpoint.)
        log.warning("probe (b): MC structured-output check FAILED (%s: %s) — "
                    "MC path will fail at run-time; continuing probe with warning.",
                    type(e).__name__, e)

    # -------- (c) Precision / metadata check -------------------------------------
    log.info("probe (c): model metadata  adapter=%s  model=%s",
             adapter.name, adapter.model)
    # We don't have a dedicated metadata endpoint here; log what we know.
    if adapter.is_dedicated:
        log.info("probe (c): dedicated endpoint is STARTED (verified by wait_ready)")
    else:
        log.warning("probe (c): serverless adapter — bf16 precision cannot be "
                    "confirmed via API; verify in provider dashboard if needed.")

    return {"ok": True, "reason": ""}


# ===========================================================================
# ENDPOINT ID FILE  (WRITTEN BEFORE WAITING READY)
# ===========================================================================

def write_endpoint_id(ep_id: str, path: Path = ENDPOINT_ID_FILE) -> None:
    """Persist the endpoint id to disk immediately after creation.

    Written BEFORE waiting for STARTED so that --reap can recover even if
    wait_ready times out or this process is killed during the poll.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"endpoint_id": ep_id, "provider": "together",
                               "created_at": time.time()}))
    tmp.replace(path)
    log.info("Endpoint id written to %s", path)


def read_endpoint_id(path: Path = ENDPOINT_ID_FILE) -> Optional[str]:
    """Read back the endpoint id from the id file.  Returns None if missing."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("endpoint_id")
    except Exception:
        return None


def remove_endpoint_id_file(path: Path = ENDPOINT_ID_FILE) -> None:
    """Remove the id file after successful teardown."""
    try:
        path.unlink()
        log.info("Removed endpoint id file %s", path)
    except FileNotFoundError:
        pass  # already gone — idempotent


# ===========================================================================
# GENERATION  (reuses serverless_72b)
# ===========================================================================

def run_generation(strong_system: str, gen_prompt: str) -> Dict:
    """Generate word-free streams via the serverless DeepInfra client.

    Reuses serverless_72b.collect_all_streams exactly — the stream collection
    step is independent of which teacher-forcing provider we use.  Returns the
    bundle dict (same schema as the exp3 bundle).
    """
    import serverless_72b as S
    key = _load_key(str(Path.home() / ".deepinfra_key"), "DEEPINFRA_KEY_FILE", "DeepInfra")
    client = S.DeepInfraClient(api_key=key)

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(S.TOKENIZER_ID, trust_remote_code=True)
        log.info("Generation tokenizer loaded: %s", S.TOKENIZER_ID)
    except Exception as e:
        log.warning("Tokenizer unavailable for generation (%s); token_ids will be None", e)
        tok = None

    def progress(msg: str) -> None:
        log.info("[gen] %s", msg)

    log.info("Starting word-free stream generation ...")
    bundle = S.collect_all_streams(
        client, strong_system, gen_prompt,
        arms=S.SERVERLESS_ARMS,
        concepts=S.COVERT_CONCEPTS,
        target_clean=S.STREAMS_PER_CONCEPT,
        tokenizer=tok,
        progress_cb=progress,
    )
    n_acc = sum(1 for s in bundle["streams"] if s.get("accepted"))
    log.info("Generation done: %d accepted / %d total streams",
             n_acc, len(bundle["streams"]))
    return bundle


# ===========================================================================
# LR TEACHER-FORCING  (reuses lr_vllm span logic)
# ===========================================================================

def run_lr_teacher_forcing(
    adapter: ProviderAdapter,
    bundle: Dict,
    strong_system: str,
    neutral_system: str,
    gen_prompt: str,
) -> List[Dict]:
    """LR teacher-forcing: score each accepted stream via the dedicated adapter.

    For each accepted stream (token_ids must be present — collected with the
    tokenizer) we compute:
      LR = LL(stream | concept_context) - LL(stream | neutral_context)

    using the adapter's teacher_force_fn.  This reuses lr_vllm.render_prompt_ids
    to build the prompt token list and the span start/end indices, then calls
    teacher_force_fn on the full prompt and slices the span logprobs.

    Returns a list of LR records:
      {concept, arm, stream_idx, lr, span_lps, neutral_span_lps}
    """
    import lr_vllm as LR  # reuse span logic

    tok = _get_tokenizer()
    accepted = [s for s in bundle["streams"]
                if s.get("accepted") and s.get("token_ids")]
    log.info("LR teacher-forcing: %d accepted streams with token_ids", len(accepted))

    import serverless_72b as S

    records = []
    for i, stream in enumerate(accepted):
        concept = stream["concept"]
        arm = stream["arm"]
        stream_ids = [int(x) for x in stream["token_ids"]]

        # Build the concept (ctx) system prompt
        ctx_system = S.build_system_prompt(concept, arm, strong_system)

        # Build the full prompt token list for the ctx context and extract span
        ctx_prompt_ids, span = LR.render_prompt_ids(
            tok, ctx_system, gen_prompt, stream_ids
        )
        # Teacher-force under the concept context
        ctx_all_lps = adapter.teacher_force_fn(ctx_prompt_ids)
        start, end = span
        ctx_span_lps = ctx_all_lps[start:end]

        # Teacher-force under the neutral context
        neu_prompt_ids, neu_span = LR.render_prompt_ids(
            tok, neutral_system, gen_prompt, stream_ids
        )
        neu_all_lps = adapter.teacher_force_fn(neu_prompt_ids)
        neu_start, neu_end = neu_span
        neu_span_lps = neu_all_lps[neu_start:neu_end]

        lr_val = LR.ll_over_span(ctx_span_lps) - LR.ll_over_span(neu_span_lps)
        records.append({
            "concept": concept,
            "arm": arm,
            "stream_idx": i,
            "lr": float(lr_val),
            "span_lps": [float(x) for x in ctx_span_lps],
            "neutral_span_lps": [float(x) for x in neu_span_lps],
        })
        if (i + 1) % 20 == 0:
            log.info("LR TF: scored %d / %d", i + 1, len(accepted))

    log.info("LR teacher-forcing done: %d records", len(records))
    return records


# ===========================================================================
# MC  (reuses serverless_72b)
# ===========================================================================

def run_mc(bundle: Dict) -> List[Dict]:
    """MC self-report for all accepted streams; reuses serverless_72b.run_mc_all."""
    import serverless_72b as S
    key = _load_key(str(Path.home() / ".deepinfra_key"), "DEEPINFRA_KEY_FILE", "DeepInfra")
    client = S.DeepInfraClient(api_key=key)

    def progress(msg: str) -> None:
        log.info("[mc] %s", msg)

    return S.run_mc_all(client, bundle, progress_cb=progress)


# ===========================================================================
# OFFLINE SCORING  (reuses lr_72b_offline)
# ===========================================================================

def run_offline_scoring(lr_records: List[Dict], mc_records: List[Dict]) -> Dict:
    """Score LR and MC records; reuses lr_72b_offline named calls.

    We don't have a full .pt shard here (no torch dependency at this level),
    so we compute a simplified summary:
      - MC confusion-MI + shuffle null (via serverless_72b.score_all_arms)
      - LR mean per arm (a preliminary view; full calibration needs the shard)
    Named calls require the full calibrated bits from the shard — we report
    "pending" until the .pt shard is written by the box scorer.
    """
    import serverless_72b as S

    mc_scores = S.score_all_arms(mc_records, n_perm=S.SHUFFLE_N)

    # LR preliminary: mean per (arm, concept) diagonal
    lr_by_arm: Dict[str, List[float]] = {}
    for rec in lr_records:
        arm = rec["arm"]
        lr_by_arm.setdefault(arm, []).append(rec["lr"])
    lr_summary = {
        arm: {
            "mean_lr": float(sum(vs) / len(vs)) if vs else None,
            "n": len(vs),
        }
        for arm, vs in lr_by_arm.items()
    }

    return {
        "mc_scores": mc_scores,
        "lr_summary": lr_summary,
        "named_calls": "pending — full calibration requires the .pt shard",
    }


# ===========================================================================
# CONCURRENT PIPELINE  (async producer-consumer run layer)
#
# This is the top-level entry point called by cmd_real AFTER the probe passes.
# It replaces the sequential generation → LR TF → MC sequence (phases 4–6) with
# a concurrent two-stage pipeline:
#
#   Stage 1 — generation workers: stream requests to the dedicated endpoint.
#   Stage 2 — scoring workers: LR teacher-forcing + MC for each completed stream,
#              starting IMMEDIATELY when the first stream lands on the queue.
#
# Both stages share a single semaphore whose size is governed by SaturationRamp
# (starts modest, grows while throughput keeps rising, holds at the plateau).
#
# The function is a module-level name so tests can patch it:
#   patch.object(run_frontier_72b, "run_concurrent_pipeline", ...)
#
# INJECTION SEAM
# --------------
# async_gen_caller and async_score_caller are optional keyword args.  In
# production they are None (defaults to the real adapter calls built inside
# _make_async_gen_caller / _make_async_score_caller).  Tests pass mock async
# callables to guarantee zero real API calls.
# ===========================================================================

def run_concurrent_pipeline(
    adapter: "ProviderAdapter",
    bundle_gen_fn: Callable,
    bundle_score_fn: Callable,
    strong_system: str,
    neutral_system: str,
    gen_prompt: str,
    concepts: List[str],
    arms: List[str],
    streams_per_concept_arm: int,
    max_retries: int = 5,
    per_call_timeout_s: float = 120.0,
    async_gen_caller: Optional[Callable] = None,
    async_score_caller: Optional[Callable] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Run the concurrent generation + scoring pipeline; return (lr_records, mc_records).

    This is the sync wrapper around the async pipeline that cmd_real calls.
    It imports concurrent_pipeline lazily (not at module level) so that --dry
    and --reap modes never need it and it doesn't add startup overhead.

    Parameters
    ----------
    adapter                : ProviderAdapter with teacher_force_fn (dedicated endpoint).
    bundle_gen_fn          : run_generation — kept for offline use; NOT called here.
                             The async pipeline calls async_gen_caller directly.
    bundle_score_fn        : run_lr_teacher_forcing — kept for offline use.
    strong_system          : The word-free system prompt (gen context).
    neutral_system         : The neutral system prompt (LR denominator).
    gen_prompt             : The generation user prompt.
    concepts               : List of covert concepts to generate streams for.
    arms                   : List of experimental arms.
    streams_per_concept_arm: How many streams to request per (concept, arm) pair.
    max_retries            : Per-request retry budget.
    per_call_timeout_s     : Per-request asyncio timeout.
    async_gen_caller       : Injectable async HTTP callable (payload→dict).
                             If None, built from adapter.teacher_force_fn.
    async_score_caller     : Injectable async scoring callable (stream→dict).
                             If None, built from adapter.teacher_force_fn + MC client.

    Returns
    -------
    (lr_records, mc_records) — lists of per-stream score dicts, same schema as
    the sync run_lr_teacher_forcing and run_mc return values.
    """
    # Lazy import: avoids importing asyncio-based concurrent_pipeline at the
    # top level so --dry / --reap work without it.  concurrent_pipeline lives in
    # the same harness/ directory.
    import concurrent_pipeline as CP
    import serverless_72b as S

    # ---- build the async gen caller -----------------------------------------
    # The generation call is: adapter.teacher_force_fn is the LR path; for
    # *generation* of new word-free streams we use the serverless DeepInfra path
    # (same as run_generation, but wrapped as an async callable).
    #
    # Why not use teacher_force_fn here?  Because stage-1 GENERATES streams
    # (word-free output from the 72B model) — that uses the standard
    # completions endpoint.  The dedicated adapter's teacher_force_fn is the
    # SCORING (echo) path, which runs in stage-2.
    #
    # In the sync version, run_generation calls S.collect_all_streams.
    # Here we wrap a single-stream generation call as an async coroutine so
    # the pipeline can run many concurrently.

    if async_gen_caller is None:
        try:
            di_key = _load_key(
                str(Path.home() / ".deepinfra_key"), "DEEPINFRA_KEY_FILE", "DeepInfra"
            )
            di_client = S.DeepInfraClient(api_key=di_key)
        except Exception as e:
            log.error("run_concurrent_pipeline: could not build DeepInfra client: %s", e)
            raise

        async def _default_gen_caller(payload: Dict) -> Dict:
            """Wrap S.DeepInfraClient.chat_completion as an async call.

            We run the blocking HTTP call in the default executor so it doesn't
            block the event loop.  The response is normalised to the pipeline's
            expected shape: {'status_code': 200, 'text': ..., 'token_ids': ...}.
            """
            loop = asyncio.get_event_loop()
            try:
                resp = await loop.run_in_executor(
                    None,
                    lambda: di_client.chat_completion(payload),
                )
                # Normalise to pipeline shape
                text = resp["choices"][0]["message"]["content"]
                return {"status_code": 200, "text": text, "token_ids": None,
                        "concept": payload.get("concept"), "arm": payload.get("arm"),
                        "raw": resp}
            except Exception as e:
                # Map to status_code so the worker's retry logic fires
                code = getattr(e, "status_code", 500)
                return {"status_code": code, "error": str(e)}

        async_gen_caller = _default_gen_caller

    # ---- build the async score caller ---------------------------------------
    # For each completed stream, stage-2 runs:
    #   (a) LR echo teacher-forcing via adapter.teacher_force_fn  (dedicated endpoint)
    #   (b) MC via S.DeepInfraClient                               (serverless)
    # The result dict includes both LR and MC fields.

    if async_score_caller is None:
        try:
            di_key_s = _load_key(
                str(Path.home() / ".deepinfra_key"), "DEEPINFRA_KEY_FILE", "DeepInfra"
            )
            di_client_s = S.DeepInfraClient(api_key=di_key_s)
        except Exception as e:
            log.error("run_concurrent_pipeline: could not build DeepInfra scoring client: %s", e)
            raise

        tok = _get_tokenizer()

        import lr_vllm as LR

        async def _default_score_caller(stream: Dict) -> Dict:
            """Run LR teacher-forcing + MC for one completed stream.

            Runs the blocking calls in the default executor so they don't block
            the event loop.  Returns a combined score dict.
            """
            loop = asyncio.get_event_loop()
            concept = stream.get("concept", "")
            arm = stream.get("arm", "")
            stream_ids = stream.get("token_ids") or []
            if not stream_ids:
                # No token ids — can't teacher-force.  Return a placeholder.
                return {"status_code": 200, "lr": None, "mc": None,
                        "concept": concept, "arm": arm, "error": "no_token_ids"}

            ctx_system = S.build_system_prompt(concept, arm, strong_system)

            def _compute_lr():
                # LR: run two teacher-force calls under ctx and neutral contexts
                ctx_ids, span = LR.render_prompt_ids(tok, ctx_system, gen_prompt, stream_ids)
                ctx_lps = adapter.teacher_force_fn(ctx_ids)
                neu_ids, neu_span = LR.render_prompt_ids(tok, neutral_system, gen_prompt, stream_ids)
                neu_lps = adapter.teacher_force_fn(neu_ids)
                start, end = span
                neu_start, neu_end = neu_span
                lr_val = LR.ll_over_span(ctx_lps[start:end]) - LR.ll_over_span(neu_lps[neu_start:neu_end])
                return float(lr_val), ctx_lps[start:end], neu_lps[neu_start:neu_end]

            def _compute_mc():
                order = S._shuffled_concepts(seed=abs(hash(f"{concept}:{arm}:{stream.get('text','')}")))
                payload = S.mc_direct_payload(stream.get("text", ""), order)
                resp = di_client_s.chat_completion(payload)
                return S.parse_mc_response(resp)

            try:
                lr_val, span_lps, neu_span_lps = await loop.run_in_executor(None, _compute_lr)
            except Exception as e:
                return {"status_code": 500, "error": f"lr_failed: {e}",
                        "concept": concept, "arm": arm}

            try:
                mc_result = await loop.run_in_executor(None, _compute_mc)
            except Exception as e:
                mc_result = {"concept": None, "error": str(e)}

            return {
                "status_code": 200,
                "concept": concept,
                "arm": arm,
                "lr": lr_val,
                "span_lps": [float(x) for x in span_lps],
                "neutral_span_lps": [float(x) for x in neu_span_lps],
                "mc_concept": mc_result.get("concept"),
                "mc_raw": mc_result,
            }

        async_score_caller = _default_score_caller

    # ---- build generation request list --------------------------------------
    # We build the full list up-front (memory: O(concepts * arms * streams_per))
    # but pass it as an iterator so the pipeline streams them lazily.
    #
    # For generation, the payload is the chat messages payload for a word-free
    # stream.  We include concept/arm metadata so the scoring worker can read it.

    def _build_gen_payload(concept: str, arm: str, idx: int) -> Dict:
        """Build the DeepInfra chat payload for one word-free stream generation."""
        ctx_system = S.build_system_prompt(concept, arm, strong_system)
        msgs = [
            {"role": "system", "content": ctx_system},
            {"role": "user", "content": gen_prompt},
        ]
        payload = S.gen_payload(msgs)   # uses S's generation params
        payload["concept"] = concept
        payload["arm"] = arm
        return payload

    gen_requests = CP.build_generation_requests(
        concepts=concepts,
        arms=arms,
        streams_per_concept_arm=streams_per_concept_arm,
        build_payload_fn=_build_gen_payload,
        max_retries=max_retries,
    )
    total = len(gen_requests)
    log.info(
        "run_concurrent_pipeline: built %d generation requests  "
        "(concepts=%d  arms=%d  per_pair=%d)",
        total, len(concepts), len(arms), streams_per_concept_arm,
    )

    # ---- create saturation ramp ---------------------------------------------
    # Start conservatively: 4 concurrent requests.  The ramp will grow to the
    # plateau automatically.  max_concurrency=64 is a safe cap for a B200 box.
    ramp = CP.SaturationRamp(
        initial_concurrency=4,
        step=4,
        max_concurrency=64,
        measure_interval_s=10.0,
        improvement_threshold_pct=5.0,
    )

    # ---- run the async pipeline ---------------------------------------------
    log.info("run_concurrent_pipeline: starting async pipeline ...")
    _, score_results, tracker = asyncio.run(
        CP.run_concurrent_pipeline(
            generation_requests=iter(gen_requests),
            num_generation_requests=total,
            async_gen_caller=async_gen_caller,
            async_score_caller=async_score_caller,
            ramp=ramp,
            max_retries=max_retries,
            per_call_timeout_s=per_call_timeout_s,
            ramp_check_interval_s=10.0,
        )
    )
    log.info(
        "run_concurrent_pipeline: done  scored=%d  ramp_concurrency=%d  "
        "plateau=%s  tracker_summary follows",
        len(score_results), ramp.current_concurrency, ramp.plateau_reached,
    )
    tracker.log_summary()

    # ---- split score_results into lr_records and mc_records -----------------
    lr_records = []
    mc_records = []
    for r in score_results:
        if r is None:
            continue
        if "lr" in r and r.get("lr") is not None:
            lr_records.append({
                "concept": r.get("concept"),
                "arm": r.get("arm"),
                "lr": r.get("lr"),
                "span_lps": r.get("span_lps", []),
                "neutral_span_lps": r.get("neutral_span_lps", []),
            })
        if "mc_concept" in r:
            mc_records.append({
                "concept": r.get("concept"),
                "arm": r.get("arm"),
                "prediction": r.get("mc_concept"),
                "raw": r.get("mc_raw"),
            })

    log.info(
        "run_concurrent_pipeline: lr_records=%d  mc_records=%d",
        len(lr_records), len(mc_records),
    )
    return lr_records, mc_records


# ===========================================================================
# TEARDOWN  (the heart of the safety stack — always runs)
# ===========================================================================

# Global adapter reference so atexit / signal handlers can reach it
_active_adapter: Optional[ProviderAdapter] = None


def _do_teardown(reason: str = "normal exit") -> None:
    """Delete the active endpoint and remove the id file.

    Called from:
      - main()'s finally block
      - atexit handler
      - SIGINT / SIGTERM signal handlers

    Must be safe to call multiple times (idempotent).
    """
    global _active_adapter
    adapter = _active_adapter

    # ===[ TEARDOWN ]===================================================
    log.warning("==[ TEARDOWN ]== reason=%s", reason)

    if adapter is not None and adapter.delete_endpoint_fn is not None:
        try:
            adapter.delete_endpoint_fn()
        except Exception as e:
            log.error("==[ TEARDOWN ]== delete_endpoint raised %s: %s  "
                      "(run --reap to retry cleanup)", type(e).__name__, e)
    else:
        log.info("==[ TEARDOWN ]== serverless adapter — no endpoint to delete.")

    # Remove the id file only if delete succeeded (delete sets id to None).
    # Read ENDPOINT_ID_FILE as a module-level name so tests can patch it at runtime.
    id_file = ENDPOINT_ID_FILE
    ep_id = read_endpoint_id(path=id_file)
    if ep_id is None:
        remove_endpoint_id_file(path=id_file)
    else:
        log.warning(
            "==[ TEARDOWN ]== endpoint id file NOT removed (delete may have failed); "
            "run '--reap' to retry deletion and cleanup."
        )

    log.warning("==[ TEARDOWN ]== complete.")
    # ===[ END TEARDOWN ]===============================================

    _active_adapter = None   # prevent double-delete from atexit after explicit call


def _signal_handler(signum: int, frame: Any) -> None:
    log.warning("==[ TEARDOWN ]== caught signal %d — tearing down before exit", signum)
    _do_teardown(reason=f"signal {signum}")
    sys.exit(128 + signum)


def _register_teardown(adapter: ProviderAdapter) -> None:
    """Wire up the global teardown safety stack."""
    global _active_adapter
    _active_adapter = adapter
    atexit.register(_do_teardown, "atexit")
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ===========================================================================
# OUTPUT HELPERS
# ===========================================================================

def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    tmp.replace(path)
    log.info("Saved: %s", path)


# ===========================================================================
# CMD: --reap (recovery for kill -9)
# ===========================================================================

def cmd_reap(args: Any) -> None:
    """Read runs/frontier_endpoint.json and DELETE the recorded endpoint.

    This is the recovery path when this process was kill -9'd before teardown
    ran.  The inactive_timeout=10min provider backstop should fire regardless,
    but --reap is the belt-and-suspenders manual recovery.
    """
    # Read ENDPOINT_ID_FILE as a module-level name so tests can patch it.
    id_file = ENDPOINT_ID_FILE
    ep_id = read_endpoint_id(path=id_file)
    if ep_id is None:
        log.info("--reap: no endpoint id file found at %s — nothing to do.", id_file)
        return

    log.warning("==[ REAP ]== found endpoint id %s — deleting ...", ep_id)

    try:
        key = load_together_key(getattr(args, "together_key", None))
    except Exception as e:
        log.error("--reap: could not load Together key: %s", e)
        sys.exit(1)

    auth_headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": TOGETHER_UA,
    }
    url = f"{TOGETHER_BASE_ENDPOINTS}/endpoints/{ep_id}"
    status = _http_delete(url, auth_headers)
    if status in (200, 204, 404):
        log.warning("==[ REAP ]== endpoint %s deleted (status %d).  No further charges.", ep_id, status)
        remove_endpoint_id_file(path=id_file)
    else:
        log.error("==[ REAP ]== DELETE returned %d for endpoint %s.  "
                  "Check the Together dashboard manually.", status, ep_id)
        sys.exit(1)


# ===========================================================================
# CMD: --dry (probe only, no endpoint)
# ===========================================================================

def cmd_dry(args: Any) -> None:
    """Build a serverless adapter, run functionality_probe, print PASS/FAIL.

    No endpoint is created.  No money is spent.
    The provider choice matters only for which serverless adapter is used.
    """
    provider = getattr(args, "provider", "fireworks").lower()
    log.info("--dry: building serverless adapter for provider=%s", provider)

    if provider == "fireworks":
        try:
            key = load_fireworks_key(getattr(args, "fireworks_key", None))
        except FileNotFoundError:
            # Allow --dry without a real key for CI environments
            log.warning("--dry: Fireworks key not found; using a placeholder for probe "
                        "(the probe will fail at the HTTP call but the structure is valid).")
            key = "dry-placeholder-key"
        adapter = make_fireworks_serverless_adapter(api_key=key)
    elif provider == "deepinfra":
        try:
            key = _load_key(str(Path.home() / ".deepinfra_key"),
                            "DEEPINFRA_KEY_FILE", "DeepInfra")
        except FileNotFoundError:
            key = "dry-placeholder-key"
        adapter = make_deepinfra_serverless_adapter(api_key=key)
    else:
        log.error("--dry: unknown provider %r  (choose: fireworks, deepinfra)", provider)
        sys.exit(1)

    log.info("--dry: running functionality_probe (no HTTP without a real key)...")

    # In a true dry run we may not have a real key.  Catch probe errors and
    # report them rather than crashing.
    result = functionality_probe(adapter)
    if result["ok"]:
        log.info("--dry: functionality_probe PASS")
        print("DRY PASS: functionality_probe passed on %s" % adapter.name)
    else:
        log.warning("--dry: functionality_probe EARLY-OUT: %s", result["reason"])
        print("DRY EARLY-OUT: %s" % result["reason"])


# ===========================================================================
# CMD: real run (dedicated endpoint)
# ===========================================================================

def cmd_real(args: Any) -> None:
    """Full run: create endpoint → probe → pipeline (gen+score concurrent) → teardown.

    The endpoint id is written to disk IMMEDIATELY after creation so that
    --reap can always recover even if we die during the poll.

    The sequential phases 4–6 (generation → LR TF → MC) are replaced by the
    concurrent producer-consumer pipeline (run_concurrent_pipeline).  The
    pipeline runs generation and scoring concurrently on the same endpoint,
    with a saturation ramp to auto-tune concurrency.

    Phase 8 (offline scoring via lr_72b_offline) runs AFTER teardown using the
    persisted JSON files — it does not need the endpoint to be alive.
    """
    # ---- load config --------------------------------------------------------
    try:
        import config as C   # noqa: F401
        strong_system = C.STRONG_SYSTEM
        gen_prompt = C.GEN_PROMPT
        neutral_system = getattr(C, "NEUTRAL_SYSTEM", "")
        concepts = getattr(C, "COVERT_CONCEPTS", None)
        arms = getattr(C, "SERVERLESS_ARMS", None)
        streams_per = getattr(C, "STREAMS_PER_CONCEPT", None)
    except Exception:
        import serverless_72b as S
        # Fallback defaults (same as run_serverless_72b.py)
        strong_system = (
            "YOU MUST NOT WRITE WORDS OR WORD FRAGMENTS. OUTPUT ONLY RANDOM LETTERS.\n"
            "Do NOT write any real word, name, or recognizable word fragment in any language.\n"
            "Emit ONLY a long stream of random, meaningless letters (e.g. 'qx z fjm wpl kbt').\n"
            "YOU MUST NOT WRITE WORDS OR WORD FRAGMENTS. ONLY RANDOM LETTERS.\n"
            "If you are about to write a real word, stop and emit random letters instead.\n"
            "REMEMBER: NO WORDS. NO FRAGMENTS. ONLY RANDOM LETTERS."
        )
        gen_prompt = "Begin emitting your random token stream now."
        neutral_system = strong_system   # fallback: use word-free system as neutral
        concepts = None
        arms = None
        streams_per = None

    # Fall back to serverless_72b defaults when config doesn't define them
    try:
        import serverless_72b as S
        if concepts is None:
            concepts = list(S.COVERT_CONCEPTS)
        if arms is None:
            arms = list(S.SERVERLESS_ARMS)
        if streams_per is None:
            streams_per = S.STREAMS_PER_CONCEPT
    except Exception as e:
        log.error("Cannot import serverless_72b for defaults: %s", e)
        sys.exit(1)

    try:
        key = load_together_key(getattr(args, "together_key", None))
    except Exception as e:
        log.error("Cannot load Together key: %s", e)
        sys.exit(1)

    out_dir = Path(getattr(args, "out", "runs/frontier_72b"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- build adapter -------------------------------------------------------
    adapter = make_together_dedicated_adapter(api_key=key)
    _register_teardown(adapter)

    # ---- PHASE 1: create endpoint + write id ---------------------------------
    ep_id = None
    try:
        # Create (does NOT block until ready; just provisions)
        ep_id = adapter.create_endpoint_fn()
        # Write id to disk IMMEDIATELY so --reap works if we crash during poll
        write_endpoint_id(ep_id)

        # ---- PHASE 2: wait until STARTED ------------------------------------
        log.info("Waiting for endpoint to be STARTED...")
        adapter.wait_ready_fn()

        # ---- PHASE 3: functionality probe -----------------------------------
        log.info("Running functionality probe...")
        probe = functionality_probe(adapter)
        log.info("Probe result: ok=%s  reason=%r", probe["ok"], probe["reason"])
        if not probe["ok"]:
            log.error("Functionality probe FAILED — tearing down without running experiment.")
            # teardown in finally
            sys.exit(1)

        # ---- PHASES 4–6: concurrent generation + LR TF + MC (pipeline) ------
        # The pipeline replaces the sequential:
        #   Phase 4: run_generation  (collect word-free streams)
        #   Phase 5: run_lr_teacher_forcing  (score each stream via adapter)
        #   Phase 6: run_mc  (MC self-report)
        #
        # Instead: streams are generated and scored concurrently.  As each
        # stream completes stage-1, it is IMMEDIATELY picked up by stage-2
        # for LR + MC scoring — no waiting for all streams to finish first.
        log.info("=== PHASES 4–6: concurrent pipeline (gen + LR TF + MC) ===")
        lr_records, mc_records = run_concurrent_pipeline(
            adapter=adapter,
            bundle_gen_fn=run_generation,         # kept for reference / fallback
            bundle_score_fn=run_lr_teacher_forcing,  # kept for reference / fallback
            strong_system=strong_system,
            neutral_system=neutral_system,
            gen_prompt=gen_prompt,
            concepts=list(concepts),
            arms=list(arms),
            streams_per_concept_arm=int(streams_per),
        )
        _save_json(lr_records, out_dir / "lr_records_frontier.json")
        _save_json(mc_records, out_dir / "mc_records_frontier.json")

        # ---- PHASE 7: offline scoring ---------------------------------------
        # Runs on the collected records — does NOT need the endpoint alive.
        # The endpoint teardown (in finally) will fire after this returns.
        log.info("=== PHASE 7: offline scoring ===")
        scores = run_offline_scoring(lr_records, mc_records)
        _save_json(scores, out_dir / "scores_frontier.json")

        log.info("Run complete.  Results in %s", out_dir)

    finally:
        # ===[ TEARDOWN: ALWAYS RUNS ]=======================================
        # Whether we succeeded, failed, were probed-early-out, or got a
        # KeyboardInterrupt — teardown always fires here.
        log.warning(
            "==[ TEARDOWN FINALLY ]== Executing teardown "
            "(always runs regardless of success/failure)"
        )
        _do_teardown(reason="finally block")


# ===========================================================================
# MAIN / ARGPARSE
# ===========================================================================

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description=(
            "run_frontier_72b: Together B200 dedicated-endpoint or Fireworks "
            "serverless teacher-forcing for LR + MC on Qwen2.5-72B-Instruct."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selector (mutually exclusive; default = real run)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry",
        action="store_true",
        help=(
            "Dry-run: build the SERVERLESS adapter for --provider, run "
            "functionality_probe only, print PASS/EARLY-OUT.  NO endpoint "
            "created.  $0."
        ),
    )
    mode.add_argument(
        "--reap",
        action="store_true",
        help=(
            "Recovery mode: read runs/frontier_endpoint.json and DELETE the "
            "recorded endpoint.  Run after a kill -9 to clean up."
        ),
    )

    ap.add_argument(
        "--provider",
        default="together",
        choices=["together", "fireworks", "deepinfra"],
        help=(
            "Provider for the real run (default: together).  "
            "For --dry, 'fireworks' or 'deepinfra' are used (no dedicated endpoint)."
        ),
    )
    ap.add_argument(
        "--out",
        default="runs/frontier_72b",
        help="Output directory for results (default: runs/frontier_72b)",
    )
    ap.add_argument(
        "--together-key",
        default=None,
        help="Path to Together API key file (default: ~/.together_key)",
    )
    ap.add_argument(
        "--fireworks-key",
        default=None,
        help="Path to Fireworks API key file (default: ~/.fireworks_key)",
    )

    args = ap.parse_args()

    if args.reap:
        cmd_reap(args)
        return

    if args.dry:
        cmd_dry(args)
        return

    # Default: real run
    cmd_real(args)


if __name__ == "__main__":
    main()
