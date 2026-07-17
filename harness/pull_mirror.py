#!/usr/bin/env python3
"""Driver-side periodic data mirror for long labkit runs (Matt's data-delivery requirement,
2026-07-11: "if we do the run and we don't get the data that's a huge deal").

Every --interval seconds, rsync the box's out/ tree down to a LOCAL MIRROR dir. Read-only with
respect to the box and the measurement path: touches nothing on-box, uses labkit's own
rsync_down over the live Lease from provider.list_mine(). The mirror is a BACKUP, deliberately
separate from the run's canonical pull dir (local_out) so resume/offline-scoring semantics are
untouched -- on a healthy run you never read the mirror; on a dead-host run it holds everything
up to the last sync (max exposure = one interval of work).

Guarantee stack this completes:
  1. atomic per-cell shards on-box (tmp -> os.replace; a torn file can never be mirrored)
  2. THIS: mirror every interval (default 600s)
  3. end-of-run canonical pull (labkit)
  4. labkit's recovery partial_pull on job failure
  5. resume-safe relaunch recomputes anything missing

Usage: .venv-driver/bin/python harness/pull_mirror.py --owner lr-grid \
           --mirror runs/lr_grid_box_mirror [--interval 600] [--once]
Exits 0 when no live boxes remain for the owner (run over -> canonical pull owns the data).
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import labkit                                                    # noqa: E402
from labkit.lifecycle import remote                              # noqa: E402

REMOTE_OUT = "/root/labkit_run/out/"


def sync_once(provider, owner, mirror):
    leases = [l for l in provider.list_mine(owner) if l.state in ("up", "init", "free")]
    if not leases:
        return 0, 0
    # list_mine() builds BARE leases (no ssh endpoint) -- fill ssh from the instance records the
    # same way vast.py's create path does (ssh_host:ssh_port), else rsync targets 'root@:' and
    # every sync fails 255 (burned the first hour of full-run mirroring, 2026-07-11).
    by_id = {str(i["id"]): i for i in provider._all_instances()}
    for lease in leases:
        inst = by_id.get(lease.instance_id, {})
        if inst.get("ssh_host"):
            lease.ssh = f"{inst.get('ssh_host')}:{inst.get('ssh_port')}"
    leases = [l for l in leases if l.ssh]      # not-yet-booted boxes: retry next interval
    ok = 0
    for lease in leases:
        dest = os.path.join(mirror, lease.instance_id)
        os.makedirs(dest, exist_ok=True)
        try:
            remote.rsync_down(lease, REMOTE_OUT, dest)
            n = sum(len(fs) for _, _, fs in os.walk(dest))
            print(f"MIRROR ok instance={lease.instance_id} files={n} -> {dest}", flush=True)
            ok += 1
        except (subprocess.SubprocessError, OSError, RuntimeError) as e:
            # transient ssh/rsync failure: skip this window, next interval retries; the box's own
            # atomic shards make every successfully-mirrored file complete and final.
            print(f"MIRROR skip instance={lease.instance_id}: {type(e).__name__}: {e}", flush=True)
    return len(leases), ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True, help="labkit owner tag of the run (e.g. lr-grid)")
    ap.add_argument("--mirror", required=True, help="local mirror dir (NOT the run's local_out)")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--grace", type=int, default=3,
                    help="consecutive empty list_mine() polls before concluding the run is over")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    provider = labkit.VastProvider(owner=args.owner,
                                   throttle_path=labkit.default_vast_throttle_path())
    os.makedirs(args.mirror, exist_ok=True)
    empty = 0
    while True:
        n, ok = sync_once(provider, args.owner, args.mirror)
        if args.once:
            return
        if n == 0:
            empty += 1
            if empty >= args.grace:
                print("MIRROR done: no live boxes for owner; canonical pull owns the data", flush=True)
                return
        else:
            empty = 0
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
