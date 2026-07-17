# Using labkit as this experiment's GPU harness

labkit is the shared GPU-lifecycle harness (instance create/run/destroy, leak-proof teardown, spend
tracking, watchdog). This experiment **depends on it as a pinned package — it is never vendored or
copied in**, and we never edit labkit internals. Experiment code lives here and runs on the box via
`script_job`.

## Install (driver side)

Pinned to **v0.2.50** (A24 pull-verification, severity events + proactive spend_threshold, `labkit watch`
mid-run tripwire, the v0.2.36 security fix — always pin ≥ v0.2.36; plus verified-datacenter-host default, 429
hardening, full box-log pull). Either install into the driver venv:

```
.venv-driver/bin/pip install \
  "labkit @ git+https://github.com/MattStults/experiment_harness.git@v0.2.50#subdirectory=labkit"
```

…or pin it reproducibly — it's already in **`requirements-driver.txt`**:

```
labkit @ git+https://github.com/MattStults/experiment_harness.git@v0.2.50#subdirectory=labkit
```

This repo's `.venv-driver/` is exactly this (verified: light — no torch). The driver is `harness/run_labkit.py`.

Then `import labkit` and drive via `run_experiment` / `script_job` (labkit README §1b).

## v0.2.11 notes (2026-06-28): surface the box's stdout

Directly fixes the gap we hit this session — the on-box calibration trace (and any traceback) used to die
with the box because only `out/` was pulled, not `/root/run.log`. Now:

- **Full box `run.log` pulled locally on success AND failure**, surfaced as **`res.log_path`** (lands at
  `local_out/run.log`). The driver now prints it. So even an empty-bundle failure leaves the complete box
  stdout to read.
- **`script_job(stream=True)`** live-tails the box stdout to our stdout as the job runs — exposed as
  `run_labkit.py --stream` (only useful if the driver itself runs unbuffered, i.e. launched `python -u`).
- `tail_chars` (the `log_tail` size) is now configurable; default 2000.

We keep saving the STRUCTURED `calibration_trace.json` in-bundle regardless — parseable JSON beats scraping
`run.log` text for the auto-tuner-vs-manual validation. `res.log_path` is the human-readable backup.

## v0.2.7–v0.2.10 notes (2026-06-28): the 429 rate-limit fix line

This whole line targets exactly the Vast **429 "Too Many Requests"** we hit on bursty sweep retries. Vast's
limiter is a *minimum interval between requests per (endpoint, key+IP)* with **no `Retry-After`** — the docs
say *prevent*, don't retry-through. So:

- **v0.2.8 — client-side min-interval throttle** on every API call (prevents the 429 at the source) +
  cached `GET /instances/` + two new terminal outcomes: **`rate_limited`** (persistent 429 → back off) and
  **`spend_capped`** (account `spend_rate_limit` → verify account). `run_labkit.py` now prints a back-off
  hint for each instead of blindly failing.
- **v0.2.9 — opt-in CROSS-PROCESS throttle (the one that matters for us).** Vast's limit identity is
  `key+IP`, so the *several agents sharing this Vast key on this machine* share one limit. Our driver now
  passes `VastProvider(throttle_path=labkit.default_vast_throttle_path())` (`~/.labkit/vast_throttle.json`),
  a shared file-lock that serializes API calls across those processes. Best-effort: falls back to the
  in-process throttle on any lock failure.
- **v0.2.10 — offer-search cache** (`offers_cache_ttl_s`, default 10s; automatic). The offer search is the
  heaviest / most-limited call; rapid acquisitions now share one search, a stale offer just fails over.

Net: the earlier "space runs out by hand to dodge 429" workaround is now the library's job. Still branch on
`res.outcome` — it can now be `rate_limited` / `spend_capped`.

## v0.2.6 notes (2026-06-27)

- **Breaking change (does NOT affect us):** `reuse=True` / `keep_warm=True` now require an explicit
  `owner=` — the default `"me"` is rejected before any spend (`outcome="error"`, `reasons.owner_required`).
  Our driver does one-shot runs (`reuse`/`keep_warm` default `False`), so this never triggers. (If we ever
  add warm-box reuse, pass a real `owner`.)
- **Host roulette is now fixed in the library** — `acquire` fails over to *distinct physical machines* and
  bans a non-booting machine for 6h (`~/.labkit/lemons.json`), so the CDI/GPU-container-runtime roulette we
  kept hitting self-heals. Our `--tries`/`--min-bw` are now backstops, not necessities.
- **Post-boot ssh-auth race fixed** — `wait_ssh` waited on the TCP port only, so the first rsync could land
  before sshd auth was ready and kill the paid run; now retried on the same host. (Drop any rsync-retry
  wrapper idea.)
- **Other (no action):** ledger now locked across processes (concurrent runs can't lose spend / leak the
  cap); an unbuilt mode returns `job_failed` not false `ok`. Always branch on `res.outcome`
  (ok/no_offer/over_budget/stalled/job_failed/rate_limited/spend_capped/error) + read `res.reasons`.

## Three things to get right

1. **Pin a tag, never `@main`.** When labkit ships a new tag and we're ready, bump the version and
   `pip install --force-reinstall …`. A labkit change must never silently alter a run mid-experiment.
2. **Keep the driver venv light — no heavy deps.** Just labkit (its `VastProvider` is stdlib-only).
   `torch`/`transformers`/vLLM/trl live on the rented box, via `script_job(deps=…)` or the box `image=`,
   **not** in the driver venv. In particular **do not** install `labkit[mi]` / `labkit[sft]` locally —
   they pull torch. (`#subdirectory=labkit` is required because labkit is a subfolder of the repo.)
3. **Private-repo auth.** pip installs from git, so the private repo matters:
   - On Matt's machine: already works (same keychain auth that pushes this repo).
   - Another machine / CI: use the SSH form (no token in the URL):
     `git+ssh://git@github.com/MattStults/experiment_harness.git@v0.1.0#subdirectory=labkit`

## Standing rules

- **Never vendor/copy labkit in.** Depend on the pinned package.
- **Experiment code lives in this repo and runs via `script_job`.**
- **Never edit labkit internals** (see labkit/CLAUDE.md → README).

## Driver pattern (sketch — migration is staged, see below)

The driver replaces `harness/gpu_run.py` + `harness/setup_and_run.sh`. A `script_job` ships this repo
(`workdir="."`), runs the entrypoint on the box, and pulls the run bundle:

With labkit **v0.2.6** (`env=` on `script_job`) the driver is **wrapper-free** — the box exports
`INTRO_MODEL` + `INTRO_RUN_DIR=out` (config honors `INTRO_RUN_DIR`, so the bundle is written straight into
`out/` for the pull) + the thread caps, and the entrypoint is just the script. This is implemented in
`harness/run_labkit.py`; the sketch:

```python
import labkit
SLUG = "qwen2.5-1.5b"
res = labkit.run_experiment(
    provider=labkit.VastProvider(owner="introspect-collect-<date>"),
    gpu="RTX3090", min_vram_mb=24000, pull_gb=10, max_dph=1.20, max_spend=10.0, max_hours=1.0,
    job=labkit.script_job(
        workdir=".",
        entrypoint="python3 -u src/covert_collect.py",           # no wrapper
        env={"INTRO_MODEL": SLUG, "INTRO_RUN_DIR": "out",         # write the bundle straight into out/
             "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
             "MKL_NUM_THREADS": "8", "NUMEXPR_NUM_THREADS": "8"}, # env= requires labkit >= v0.2.6
        deps=["transformers==4.46.3", "accelerate", "scikit-learn", "numpy", "wordfreq"],  # box deps, pinned
        ready="MODEL_READY", done="COLLECT_DONE", fatal="DEPS_INSTALL_FAILED",
        local_out=f"runs/{SLUG}", pull_subdir="out"),
    run_id=f"collect-{SLUG}")
if not res.ok: ...  # notify (osascript); res sets partial_pull if a failed run still wrote out/
```

**Status of the three gaps my earlier sketch worried about (re-checked against the installed v0.2.6):**
- **Partial-pull on failure — already there** (`modes/_runner.py` pulls `out/` best-effort even on a failed
  job, sets `verdict["partial_pull"]`). No workaround needed.
- **Output pull path — configurable** (`pull_subdir`).
- **Thread caps / `INTRO_RUN_DIR` — solved by `env=` in v0.2.6** (verified: `build_cmd` exports them before
  deps+entrypoint). The driver is wrapper-free; no `remote_entry.sh`.

**Still true:** the `script_job` path is mock-tested in labkit, so the first real run is also its hardware
validation — **stage it behind one ~$1 shakedown.** Keep the `transformers==4.46.3` pin in `deps=`.

*(Nit for the labkit owner: the installed package's `__version__`/metadata still reports `0.0.1` even at tag
v0.2.6 — the version string wasn't bumped. Functionally fine; worth bumping for clean `pip` reporting.)*
