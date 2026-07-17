"""concurrent_pipeline.py — Async producer-consumer execution layer for run_frontier_72b.py

OVERVIEW
========
This module provides a concurrency-controlled, producer-consumer pipeline for the
frontier 72B teacher-forcing run.  It is designed to:

  1. Stream requests from an iterable, never holding all ~9k in memory.
  2. Bound in-flight concurrency with a single asyncio.Semaphore that covers BOTH
     pipeline stages (generation and scoring run on the same endpoint).
  3. Implement exponential backoff on 429 / 5xx / timeout errors, matching the
     retry/tracker structure from OpenAI's api_request_parallel_processor.py.
  4. Replace the cookbook's token-bucket rate-limiter with a *saturation ramp*
     that starts at modest concurrency and increases while throughput is still
     rising, then holds at the plateau (dedicated endpoint, no published rate limit).
  5. Provide a two-stage asyncio.Queue-connected pipeline so stage-2 scoring begins
     on completed streams IMMEDIATELY, without waiting for all of stage 1 to finish.

STRUCTURE
=========
  StatusTracker           — counters + throughput measurement (mirrors cookbook shape)
  GenerationRequest       — stage-1 request dataclass (mirrors APIRequest)
  ScoringRequest          — stage-2 request dataclass
  SaturationRamp          — replaces token-bucket; auto-tunes in-flight concurrency
  run_worker              — single async coroutine: fetch with backoff, put result on queue
  generation_worker_pool  — spawns run_workers, bounded by the shared semaphore
  scoring_worker_pool     — drains the queue, runs LR + MC scoring in order
  run_concurrent_pipeline — top-level entry point; called by cmd_real after probe passes

KEY DESIGN DECISIONS vs. the OpenAI cookbook
=============================================
  - NO token-bucket: the dedicated B200 endpoint has no published token rate limit.
    Instead SaturationRamp tracks req/s and increases concurrency as long as
    throughput keeps rising; it stops at the plateau and holds there.
  - Both stages share one Semaphore; the combined in-flight is bounded by
    SaturationRamp.current_concurrency.  This prevents flooding the endpoint when
    generation and scoring are both active simultaneously.
  - asyncio.Queue with a sentinel (_QUEUE_DONE) for clean shutdown: stage-1 workers
    put a sentinel when done; stage-2 drains until it sees the sentinel.
  - All HTTP is async (aiohttp); the injection seam accepts an async_http_caller
    argument so tests can pass a mock coroutine without network I/O.
  - Retry: 429 + 5xx + asyncio.TimeoutError -> exponential backoff with jitter
    (base 2s, cap 64s); permanent failures (max_retries exhausted) are logged
    and written to the error list without crashing the pipeline.

ZERO REAL CALLS GUARANTEE
===========================
  - The sync helper functions (run_generation, run_lr_teacher_forcing, run_mc) are
    imported from run_frontier_72b.py only inside cmd_real; tests inject mocks at
    the async_http_caller seam, so no real network calls are made during testing.
  - The saturation ramp's _measure_throughput is injectable for tests.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

log = logging.getLogger("concurrent_pipeline")


# ===========================================================================
# STATUS TRACKER
# (mirrors the shape from OpenAI's api_request_parallel_processor.py;
#  adds throughput tracking for the saturation ramp)
# ===========================================================================

@dataclass
class StatusTracker:
    """Live counters for the pipeline.  Thread-safe only within one event loop.

    Mirrors the cookbook's StatusTracker with these fields:
      num_tasks_started    — total requests handed to workers
      num_tasks_in_progress — currently executing (started - succeeded - failed)
      num_tasks_succeeded  — finished without error
      num_tasks_failed     — exhausted retries or permanent error
      num_retries          — total retry attempts across all requests
      num_rate_limit_errors  — 429 responses seen (not necessarily failed)
      num_api_errors         — 5xx responses seen
      num_timeout_errors     — asyncio.TimeoutError events

    Throughput fields (for SaturationRamp):
      _window_start_time   — monotonic clock at last throughput window reset
      _window_count        — requests completed since last reset
    """
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_retries: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_timeout_errors: int = 0
    # throughput window (private; use measure_throughput())
    _window_start_time: float = field(default_factory=time.monotonic)
    _window_count: int = 0

    def record_completion(self) -> None:
        """Call when a request finishes (success or failure)."""
        self.num_tasks_in_progress -= 1
        self._window_count += 1

    def measure_throughput(self, reset: bool = True) -> float:
        """Return req/s over the current window.  Optionally reset the window."""
        elapsed = time.monotonic() - self._window_start_time
        if elapsed < 0.01:
            # Avoid divide-by-zero on very fast mock runs
            return 0.0
        rps = self._window_count / elapsed
        if reset:
            self._window_start_time = time.monotonic()
            self._window_count = 0
        return rps

    def log_summary(self) -> None:
        log.info(
            "StatusTracker: started=%d  succeeded=%d  failed=%d  in_flight=%d  "
            "retries=%d  rate_limit=%d  api_err=%d  timeout=%d",
            self.num_tasks_started, self.num_tasks_succeeded, self.num_tasks_failed,
            self.num_tasks_in_progress, self.num_retries,
            self.num_rate_limit_errors, self.num_api_errors, self.num_timeout_errors,
        )


# ===========================================================================
# REQUEST DATACLASSES
# (mirrors APIRequest from the cookbook — separated into two stages)
# ===========================================================================

@dataclass
class GenerationRequest:
    """One stage-1 generation request (produce a word-free stream).

    Fields
    ------
    task_id : int          Unique sequential id for logging.
    concept : str          The covert concept for this stream.
    arm     : str          The experimental arm (evoked / secret_word / …).
    payload : dict         The raw API payload (messages, temperature, …).
    attempts_left : int    Remaining retries before permanent failure.
    result  : list         Accumulates error messages on retries; replaced with
                           the successful response dict on success.
    """
    task_id: int
    concept: str
    arm: str
    payload: Dict
    attempts_left: int
    result: List = field(default_factory=list)


@dataclass
class ScoringRequest:
    """One stage-2 scoring request (LR teacher-forcing + MC).

    Produced by stage-1 when a stream completes and put on the queue.

    Fields
    ------
    task_id   : int        Matches the GenerationRequest that produced this.
    concept   : str        Same concept / arm as the generation request.
    arm       : str
    stream    : dict       The completed generation result dict (text + token_ids).
    attempts_left : int    Retries for the scoring call.
    result    : list       Same semantics as GenerationRequest.result.
    """
    task_id: int
    concept: str
    arm: str
    stream: Dict
    attempts_left: int
    result: List = field(default_factory=list)


# ===========================================================================
# SATURATION RAMP
# (REPLACES the cookbook's token-bucket rate limiter for dedicated endpoints)
# ===========================================================================

class SaturationRamp:
    """Automatically tunes concurrency by tracking endpoint throughput.

    WHY THIS EXISTS
    ---------------
    A dedicated endpoint has no published rate limit.  The token-bucket approach
    from the OpenAI cookbook is designed for shared endpoints with per-minute
    limits.  For a dedicated box, the right strategy is:

      "Add one more in-flight request if the endpoint is still getting faster."

    ALGORITHM
    ---------
    Every `measure_interval_s` seconds we call `measure_fn()` (defaults to
    StatusTracker.measure_throughput) to get the current req/s.  If it is >=
    `improvement_threshold_pct`% better than the previous measurement, we
    increase `current_concurrency` by `step`.  If the throughput has plateaued
    (improvement < threshold), we stop ramping and log the chosen level.

    The ramp never increases beyond `max_concurrency`.  After plateau detection,
    subsequent calls to `maybe_increase()` are no-ops.

    PARAMETERS
    ----------
    initial_concurrency : int   Starting semaphore size (default 4).
    step                : int   How many slots to add per ramp step (default 2).
    max_concurrency     : int   Hard cap (default 64).
    measure_interval_s  : float Seconds between throughput measurements (default 5).
    improvement_threshold_pct : float  Min % improvement to keep ramping (default 5.0).
    measure_fn          : Callable[[], float]   Injectable for tests (default: None
                          → caller must pass one explicitly or use measure_throughput).
    """

    def __init__(
        self,
        initial_concurrency: int = 4,
        step: int = 2,
        max_concurrency: int = 64,
        measure_interval_s: float = 5.0,
        improvement_threshold_pct: float = 5.0,
        measure_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.current_concurrency: int = initial_concurrency
        self.step = step
        self.max_concurrency = max_concurrency
        self.measure_interval_s = measure_interval_s
        self.improvement_threshold_pct = improvement_threshold_pct
        self._measure_fn = measure_fn   # injectable; tests replace this

        self._last_measurement_time: float = time.monotonic()
        self._last_throughput: float = 0.0
        self._plateau_reached: bool = False
        self._ramp_history: List[Tuple[float, float, int]] = []
        # Each entry: (time, measured_rps, resulting_concurrency)

    def set_measure_fn(self, fn: Callable[[], float]) -> None:
        """Inject a throughput measurement function (used by tests and by the
        pipeline when it wires up the StatusTracker)."""
        self._measure_fn = fn

    def maybe_increase(self) -> bool:
        """Check elapsed time and throughput; increase concurrency if warranted.

        Returns True if concurrency was increased, False otherwise.

        This is called periodically from the pipeline control loop.  It does
        nothing if the plateau has been reached or the measurement interval has
        not elapsed yet.
        """
        if self._plateau_reached:
            return False
        if self.current_concurrency >= self.max_concurrency:
            self._plateau_reached = True
            log.info(
                "SaturationRamp: hit max_concurrency=%d  (hard cap)",
                self.max_concurrency,
            )
            return False

        now = time.monotonic()
        elapsed = now - self._last_measurement_time
        if elapsed < self.measure_interval_s:
            return False

        # Time to measure
        if self._measure_fn is None:
            # No measure function wired up yet; skip
            return False

        current_rps = self._measure_fn()
        self._last_measurement_time = now

        self._ramp_history.append((now, current_rps, self.current_concurrency))

        if self._last_throughput == 0.0:
            # First measurement — just record, don't ramp yet
            self._last_throughput = current_rps
            log.info(
                "SaturationRamp: first measurement  rps=%.2f  concurrency=%d",
                current_rps, self.current_concurrency,
            )
            return False

        # Compute percentage improvement
        if self._last_throughput > 0:
            improvement_pct = (current_rps - self._last_throughput) / self._last_throughput * 100.0
        else:
            # Previous was 0; any positive rps is an improvement
            improvement_pct = 100.0 if current_rps > 0 else 0.0

        if improvement_pct >= self.improvement_threshold_pct:
            # Throughput is still rising — add more concurrency
            old = self.current_concurrency
            self.current_concurrency = min(
                self.current_concurrency + self.step, self.max_concurrency
            )
            self._last_throughput = current_rps
            log.info(
                "SaturationRamp: rps improved %.1f%% (%.2f→%.2f)  "
                "concurrency %d→%d",
                improvement_pct, self._last_throughput, current_rps,
                old, self.current_concurrency,
            )
            return True
        else:
            # Plateau detected — stop ramping
            self._plateau_reached = True
            self._last_throughput = current_rps
            log.info(
                "SaturationRamp: PLATEAU detected  rps=%.2f  improvement=%.1f%%  "
                "holding concurrency=%d",
                current_rps, improvement_pct, self.current_concurrency,
            )
            return False

    @property
    def plateau_reached(self) -> bool:
        return self._plateau_reached


# ===========================================================================
# EXPONENTIAL BACKOFF HELPER
# ===========================================================================

def _backoff_seconds(attempt_num: int, base: float = 2.0, cap: float = 64.0) -> float:
    """Exponential backoff with full jitter (matches cookbook approach).

    attempt_num is 0-indexed: first retry → 2s base, second → 4s, etc.
    Full jitter: actual sleep = uniform(0, min(cap, base ** (attempt_num + 1))).
    We use full jitter (not uniform(base^n/2, base^n)) so that a burst of
    429s from many concurrent workers does NOT all retry at the same time.
    """
    import random
    ceiling = min(cap, base ** (attempt_num + 1))
    return random.uniform(0, ceiling)


# ===========================================================================
# SINGLE ASYNC WORKER  (stage-1 generation)
# ===========================================================================

async def _generation_worker(
    request: GenerationRequest,
    async_http_caller: Callable[..., Awaitable[Dict]],
    semaphore: asyncio.Semaphore,
    result_queue: asyncio.Queue,
    tracker: StatusTracker,
    per_call_timeout_s: float = 120.0,
    max_retries: int = 5,
) -> None:
    """Execute one generation request with retry logic; put result on queue.

    Retry policy (mirrors the cookbook):
      - 429 / 503 / 504 → exponential backoff, decrement attempts_left
      - 5xx (other)     → exponential backoff, decrement attempts_left
      - asyncio.TimeoutError → treated as transient, exponential backoff
      - 4xx (not 429)   → permanent failure (bad request; retrying won't help)
      - Exhausted retries → log error, put error marker on queue

    On success: put the ScoringRequest on result_queue (stage-2 picks it up).
    On permanent failure: put None on result_queue with the request's error info
    logged; stage-2 skips None entries.
    """
    tracker.num_tasks_started += 1
    tracker.num_tasks_in_progress += 1

    attempt = 0
    while request.attempts_left > 0:
        request.attempts_left -= 1
        attempt += 1

        try:
            async with semaphore:
                # per-call timeout guards against hung connections
                response = await asyncio.wait_for(
                    async_http_caller(request.payload),
                    timeout=per_call_timeout_s,
                )
        except asyncio.TimeoutError:
            tracker.num_timeout_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "gen task_id=%d  timeout on attempt %d  backoff=%.1fs  "
                "attempts_left=%d",
                request.task_id, attempt, wait, request.attempts_left,
            )
            request.result.append({"error": "timeout", "attempt": attempt})
            if request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        except Exception as e:
            # Unexpected error (network stack, etc.) — treat as transient
            tracker.num_api_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "gen task_id=%d  unexpected error %s on attempt %d  "
                "backoff=%.1fs  attempts_left=%d",
                request.task_id, e, attempt, wait, request.attempts_left,
            )
            request.result.append({"error": str(e), "attempt": attempt})
            if request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        # ---- Check for HTTP-level error inside the response dict ----
        # The async_http_caller raises on true network errors; API-level errors
        # (rate limit, server overload) are signalled via a status_code field
        # in the returned dict.  This mirrors the cookbook's error-detection pattern.
        status_code = response.get("status_code", 200)

        if status_code == 429:
            tracker.num_rate_limit_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "gen task_id=%d  rate-limited (429) on attempt %d  "
                "backoff=%.1fs  attempts_left=%d",
                request.task_id, attempt, wait, request.attempts_left,
            )
            request.result.append({"error": "rate_limited_429", "attempt": attempt})
            if request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        if status_code >= 500:
            tracker.num_api_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "gen task_id=%d  server error %d on attempt %d  "
                "backoff=%.1fs  attempts_left=%d",
                request.task_id, attempt, wait, request.attempts_left,
            )
            request.result.append({"error": f"server_{status_code}", "attempt": attempt})
            if request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        if status_code >= 400:
            # Permanent 4xx (not 429) — bad request, won't fix on retry
            tracker.num_api_errors += 1
            log.error(
                "gen task_id=%d  permanent 4xx error %d  concept=%s arm=%s  "
                "(will not retry)",
                request.task_id, status_code, request.concept, request.arm,
            )
            request.result.append({"error": f"permanent_{status_code}", "attempt": attempt})
            break  # exit retry loop → fall through to failure path

        # ---- Success ----
        tracker.num_tasks_succeeded += 1
        tracker.record_completion()
        log.debug(
            "gen task_id=%d DONE  concept=%s arm=%s",
            request.task_id, request.concept, request.arm,
        )
        # Put a ScoringRequest on the queue so stage-2 can start immediately
        await result_queue.put(
            ScoringRequest(
                task_id=request.task_id,
                concept=request.concept,
                arm=request.arm,
                stream=response,
                attempts_left=max_retries,
            )
        )
        return  # worker done

    # ---- Exhausted retries or permanent error ----
    tracker.num_tasks_failed += 1
    tracker.record_completion()
    log.error(
        "gen task_id=%d FAILED after %d attempts  concept=%s arm=%s  errors=%s",
        request.task_id, attempt, request.concept, request.arm,
        request.result[-3:],  # last 3 errors (avoid flooding logs)
    )
    # Put None so the queue count stays consistent and stage-2 knows about the failure
    await result_queue.put(None)


# ===========================================================================
# SINGLE ASYNC WORKER  (stage-2 scoring)
# ===========================================================================

async def _scoring_worker(
    score_request: ScoringRequest,
    async_score_caller: Callable[..., Awaitable[Dict]],
    semaphore: asyncio.Semaphore,
    tracker: StatusTracker,
    per_call_timeout_s: float = 120.0,
    max_retries: int = 5,
) -> Optional[Dict]:
    """Score one completed stream (LR + MC); return the combined score dict or None.

    Retry policy is identical to _generation_worker.  Returns the score dict on
    success, or None on permanent failure (the caller collects all results).
    """
    tracker.num_tasks_started += 1
    tracker.num_tasks_in_progress += 1

    attempt = 0
    while score_request.attempts_left > 0:
        score_request.attempts_left -= 1
        attempt += 1

        try:
            async with semaphore:
                response = await asyncio.wait_for(
                    async_score_caller(score_request.stream),
                    timeout=per_call_timeout_s,
                )
        except asyncio.TimeoutError:
            tracker.num_timeout_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "score task_id=%d  timeout on attempt %d  backoff=%.1fs",
                score_request.task_id, attempt, wait,
            )
            score_request.result.append({"error": "timeout", "attempt": attempt})
            if score_request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        except Exception as e:
            tracker.num_api_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "score task_id=%d  unexpected error %s on attempt %d  backoff=%.1fs",
                score_request.task_id, e, attempt, wait,
            )
            score_request.result.append({"error": str(e), "attempt": attempt})
            if score_request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        status_code = response.get("status_code", 200)

        if status_code == 429:
            tracker.num_rate_limit_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "score task_id=%d  rate-limited (429) on attempt %d  backoff=%.1fs",
                score_request.task_id, attempt, wait,
            )
            score_request.result.append({"error": "rate_limited_429", "attempt": attempt})
            if score_request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        if status_code >= 500:
            tracker.num_api_errors += 1
            tracker.num_retries += 1
            wait = _backoff_seconds(attempt - 1)
            log.warning(
                "score task_id=%d  server error %d on attempt %d  backoff=%.1fs",
                score_request.task_id, attempt, wait,
            )
            score_request.result.append({"error": f"server_{status_code}", "attempt": attempt})
            if score_request.attempts_left > 0:
                await asyncio.sleep(wait)
            continue

        if status_code >= 400:
            tracker.num_api_errors += 1
            log.error(
                "score task_id=%d  permanent 4xx error %d  (will not retry)",
                score_request.task_id, status_code,
            )
            score_request.result.append({"error": f"permanent_{status_code}", "attempt": attempt})
            break

        # ---- Success ----
        tracker.num_tasks_succeeded += 1
        tracker.record_completion()
        return response

    # ---- Exhausted retries ----
    tracker.num_tasks_failed += 1
    tracker.record_completion()
    log.error(
        "score task_id=%d FAILED after %d attempts  errors=%s",
        score_request.task_id, attempt, score_request.result[-3:],
    )
    return None


# ===========================================================================
# SENTINEL — signals stage-1 is done to stage-2
# ===========================================================================

# A unique object (not None, which we use for failed streams) that stage-1 puts
# on the queue after all its workers complete.  Stage-2 drains until it sees this.
_QUEUE_DONE = object()


# ===========================================================================
# TOP-LEVEL PIPELINE ORCHESTRATOR
# ===========================================================================

async def run_concurrent_pipeline(
    generation_requests: Iterable[GenerationRequest],
    num_generation_requests: int,
    async_gen_caller: Callable[..., Awaitable[Dict]],
    async_score_caller: Callable[..., Awaitable[Dict]],
    ramp: Optional[SaturationRamp] = None,
    max_retries: int = 5,
    per_call_timeout_s: float = 120.0,
    ramp_check_interval_s: float = 5.0,
) -> Tuple[List[Dict], List[Dict], StatusTracker]:
    """Run the two-stage producer-consumer pipeline.

    Stage 1 (generation):
      For each GenerationRequest in generation_requests, spawn a
      _generation_worker coroutine that calls async_gen_caller.  In-flight
      concurrency is bounded by a semaphore that grows under the SaturationRamp.

    Stage 2 (scoring):
      A single scoring-consumer loop reads from the queue.  For each completed
      stream (non-None), it calls _scoring_worker which calls async_score_caller.
      Stage 2 starts consuming IMMEDIATELY after the first stream lands on the
      queue — it does NOT wait for stage 1 to finish.

    Clean shutdown:
      After all stage-1 workers complete, a _QUEUE_DONE sentinel is put on the
      queue.  Stage-2 drains all remaining items, then exits.

    Parameters
    ----------
    generation_requests      : Iterable of GenerationRequest (may be lazy / generator).
    num_generation_requests  : Total count (for logging; not used to limit the iter).
    async_gen_caller         : Async callable (payload: dict) -> dict.  Must include
                               a 'status_code' key for error detection; omit for 200.
    async_score_caller       : Async callable (stream: dict) -> dict.
    ramp                     : SaturationRamp instance; created with defaults if None.
    max_retries              : Retries per request (passed to workers).
    per_call_timeout_s       : Per-call asyncio timeout.
    ramp_check_interval_s    : How often the main loop checks the ramp (seconds).

    Returns
    -------
    (gen_results, score_results, tracker)
      gen_results    : list of dicts from async_gen_caller (successful streams)
      score_results  : list of dicts from async_score_caller (successful scores)
      tracker        : final StatusTracker with aggregate counts
    """
    if ramp is None:
        ramp = SaturationRamp()

    tracker = StatusTracker()
    # Wire the ramp's measure function to the tracker's throughput window
    ramp.set_measure_fn(lambda: tracker.measure_throughput(reset=True))

    # Queue connecting stage-1 → stage-2 (unbounded to avoid deadlock; the
    # semaphore is the real backpressure mechanism)
    result_queue: asyncio.Queue = asyncio.Queue()

    gen_results: List[Dict] = []
    score_results: List[Dict] = []

    # Start with the ramp's initial concurrency
    semaphore = asyncio.Semaphore(ramp.current_concurrency)

    # -------------------------------------------------------------------------
    # Stage-2 consumer coroutine
    # Runs concurrently with stage-1.  Reads from result_queue until _QUEUE_DONE.
    # -------------------------------------------------------------------------
    async def stage2_consumer() -> None:
        """Drain the queue and run scoring workers until _QUEUE_DONE."""
        scoring_tasks: List[asyncio.Task] = []

        while True:
            item = await result_queue.get()
            if item is _QUEUE_DONE:
                # Stage-1 is done; wait for any in-flight scoring tasks to finish
                if scoring_tasks:
                    results = await asyncio.gather(*scoring_tasks, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Dict):
                            score_results.append(r)
                        elif r is not None and not isinstance(r, Exception):
                            score_results.append(r)
                break

            # item is either a ScoringRequest (success) or None (failure)
            if item is None:
                # Stage-1 worker failed; skip scoring for this stream
                log.debug("stage2_consumer: skipping scoring for failed gen task")
                continue

            # item is a ScoringRequest — spawn a scoring worker immediately
            # (this is the zero-downtime requirement: scoring starts before
            # all generation completes)
            task = asyncio.create_task(
                _scoring_worker(
                    item,
                    async_score_caller,
                    semaphore,
                    tracker,
                    per_call_timeout_s=per_call_timeout_s,
                    max_retries=max_retries,
                )
            )
            scoring_tasks.append(task)
            log.debug(
                "stage2_consumer: spawned scoring task for gen task_id=%d  "
                "concept=%s  (scoring_tasks_count=%d)",
                item.task_id, item.concept, len(scoring_tasks),
            )

    # -------------------------------------------------------------------------
    # Stage-1 producer: stream requests one at a time (never hold all in memory)
    # -------------------------------------------------------------------------
    async def stage1_producer() -> None:
        """Spawn generation workers, one per GenerationRequest, semaphore-bounded."""
        gen_tasks: List[asyncio.Task] = []

        log.info(
            "Pipeline stage-1 starting  total_requests=%d  "
            "initial_concurrency=%d",
            num_generation_requests, ramp.current_concurrency,
        )

        for req in generation_requests:
            # Each worker holds the semaphore for the duration of its HTTP call.
            # Because the semaphore is shared with scoring, the combined in-flight
            # is naturally bounded — no separate accounting needed.
            task = asyncio.create_task(
                _generation_worker(
                    req,
                    async_gen_caller,
                    semaphore,
                    result_queue,
                    tracker,
                    per_call_timeout_s=per_call_timeout_s,
                    max_retries=max_retries,
                )
            )
            gen_tasks.append(task)
            log.debug("stage1_producer: spawned gen task_id=%d", req.task_id)

        # Wait for all generation tasks to complete
        await asyncio.gather(*gen_tasks)

        log.info("Pipeline stage-1 done  success=%d failed=%d",
                 tracker.num_tasks_succeeded, tracker.num_tasks_failed)

        # Signal stage-2 that no more items will arrive
        await result_queue.put(_QUEUE_DONE)

    # -------------------------------------------------------------------------
    # Saturation ramp monitor: periodically checks throughput and grows the semaphore
    # -------------------------------------------------------------------------
    async def ramp_monitor(stop_event: asyncio.Event) -> None:
        """Background task: poll SaturationRamp.maybe_increase().

        When the ramp says to increase, we need to update the semaphore's
        internal value.  asyncio.Semaphore doesn't support external resize, so
        we track the target concurrency separately and release extra permits to
        grow the effective cap.

        This is the simplest safe approach: extra _releases_ raise the internal
        counter, allowing more coroutines to acquire simultaneously.  We track
        'extra_releases' so we don't over-release.
        """
        extra_releases_given = 0
        while not stop_event.is_set():
            await asyncio.sleep(ramp_check_interval_s)
            increased = ramp.maybe_increase()
            if increased:
                # ramp.current_concurrency grew by ramp.step; release that many
                # extra permits so the semaphore allows more concurrent acquires
                for _ in range(ramp.step):
                    semaphore.release()
                    extra_releases_given += 1
                log.info(
                    "SaturationRamp: semaphore expanded  "
                    "concurrency=%d  (extra_releases=%d)",
                    ramp.current_concurrency, extra_releases_given,
                )

    # -------------------------------------------------------------------------
    # Run both stages concurrently
    # -------------------------------------------------------------------------
    stop_ramp = asyncio.Event()
    ramp_task = asyncio.create_task(ramp_monitor(stop_ramp))
    consumer_task = asyncio.create_task(stage2_consumer())

    # Run stage-1 (blocks until all gen workers finish + sentinel is queued)
    await stage1_producer()

    # Wait for stage-2 to drain
    await consumer_task

    # Stop the ramp monitor
    stop_ramp.set()
    ramp_task.cancel()
    try:
        await ramp_task
    except asyncio.CancelledError:
        pass

    # Collect gen_results from the tracker
    # (gen worker writes to tracker directly; we return score_results and tracker)
    tracker.log_summary()
    log.info(
        "Pipeline complete  gen_success=%d  scored=%d  ramp_concurrency=%d  "
        "plateau=%s",
        tracker.num_tasks_succeeded, len(score_results),
        ramp.current_concurrency, ramp.plateau_reached,
    )

    return gen_results, score_results, tracker


# ===========================================================================
# REQUEST BUILDER  (constructs GenerationRequest objects from the bundle params)
# ===========================================================================

def build_generation_requests(
    concepts: List[str],
    arms: List[str],
    streams_per_concept_arm: int,
    build_payload_fn: Callable[[str, str, int], Dict],
    max_retries: int = 5,
) -> List[GenerationRequest]:
    """Build a flat list of GenerationRequest objects for all (concept, arm, idx) combos.

    Parameters
    ----------
    concepts               : List of concept strings.
    arms                   : List of arm strings.
    streams_per_concept_arm: Number of streams to request per (concept, arm) pair.
    build_payload_fn       : fn(concept, arm, stream_idx) -> API payload dict.
    max_retries            : Retries per request.

    Returns
    -------
    List[GenerationRequest] in concept-arm-index order.  The caller passes this
    directly to run_concurrent_pipeline as the generation_requests iterable.
    """
    requests = []
    task_id = 0
    for concept in concepts:
        for arm in arms:
            for idx in range(streams_per_concept_arm):
                payload = build_payload_fn(concept, arm, idx)
                requests.append(
                    GenerationRequest(
                        task_id=task_id,
                        concept=concept,
                        arm=arm,
                        payload=payload,
                        attempts_left=max_retries,
                    )
                )
                task_id += 1
    return requests
