# labkit feature request: make workdir transfer robust to slow hosts

Found running the introspection-leakage controlled A/B on v0.2.50. Three related transfer issues; (1) is the
one that actually cost paid runs.

## 1. `rsync_up`/`rsync_down` timeout should fail over to a new host, not hard-error the run

**Severity:** medium — terminates authorized paid runs on a transient host fault that the failover machinery
already exists to handle.

A 3-model batch acquired a host that booted cleanly, then the workdir upload stalled; after 600s:

```
TimeoutExpired: ['rsync','-az','-e','ssh ... -p 10810', --exclude __pycache__ --exclude .venv
  --exclude *.pt --exclude artifacts --exclude .git, './', 'root@ssh5.vast.ai:/root/labkit_run'] timed out after 600s
  remote.py:207 rsync_up -> remote.py:182 _run_checked
```

Outcome `error`, instance correctly torn down (verified `list_mine` → 0 live). labkit already fails over and
lemon-bans bad hosts on the **boot** path (host-roulette + `lemons.json`, 6h ban). A host too slow to receive
the workdir is the same "bad host" class but is **not** in the failover path — the `TimeoutExpired` goes
straight to `outcome=error`. **Request:** treat an `rsync_up`/`rsync_down` timeout as a host fault —
lemon-ban + fail over to the next offer within the existing `max_setup_retries` budget.

## 2. The 600s rsync timeout is hardcoded and very long

For a workdir that excludes `.pt`/`.venv`/`.git`/`artifacts` (normally a few hundred KB) 600s is a long
block before giving up. **Request:** make it configurable, and/or add a fast upload-stall detector (no bytes
moved in N s → fail over) so a slow host is abandoned in seconds, not 10 minutes. (`remote.py:182`.)

## 3. The `*.pt` exclude misses rsync's own temp files — and there's no way to exclude a results dir

Our own root cause for the stalls was a **240 MB orphaned rsync temp file**
(`runs/.../.covert_collect.pt.IkNFGiPdfg`) left by an earlier interrupted pull. Its name ends in a random
suffix, **not** `.pt`, so the `*.pt` exclude didn't catch it and it re-uploaded every attempt — turning a
~7 MB upload into ~250 MB and blowing the 600s budget. Two asks:

- The default exclude for `*.pt` (and any artifact glob) should also match rsync's in-progress temp form
  `.<name>.<ext>.<random>` — e.g. add `.*.pt.*` / exclude dotfiles — so a stale partial transfer can't
  silently bloat the next upload.
- Expose a **user exclude list** on `script_job` so callers can drop a local results/archive dir (we never
  need our `runs/` on the box) from the upload entirely. Today the exclude set is fixed.

## 4. Observability: mirror transfer errors into `status_path.last_error`

On the timeout, the run's `status.json` `last_error` stayed `null`; the error only reached `res.error`. The
`labkit watch` tripwire reads the status file, so it could report *that* the run ended but not *why*.
**Request:** mirror setup/transfer errors into `last_error`.

---

**Addendum (2026-07-01, exp3 runs), corrected:** one more genuine (1)-class hit — an **exit 255** at upload
on a verified host (3B collect; a plain relaunch on a fresh offer succeeded immediately, which is exactly
what the failover path would have done automatically). `max_setup_retries=3` was set and did **not** engage —
the retry budget covers slow-but-progressing setup, not a dead transfer. So (1) stands, now covering exit
255 as well as timeout.

**Related but ours, worth a labkit guard anyway:** three further 255s in a row turned out to be a
*driver-side* auth failure — the machine rebooted, emptying ssh-agent; labkit's ssh command passes **no
`-i`** and `BatchMode=yes`, so with no agent identity every host 255s instantly and indistinguishably from a
host fault. Two asks: (a) pass `-i <the key labkit registered with the provider>` explicitly instead of
relying on ambient agent state; (b) on repeated same-exit-code transfer failures across ≥2 distinct hosts,
surface "likely local auth/network" in `reasons` instead of a bare CalledProcessError — three paid
acquisitions were burned ((~$0.02, but wall-clock and attention too) misdiagnosing this as host roulette.

Context: same spirit as the 7B silent-drop report that became A24's pull-verification — this is the
upload-side analogue. Found on v0.2.50; check whether v0.2.52 already covers (1)/(4).
