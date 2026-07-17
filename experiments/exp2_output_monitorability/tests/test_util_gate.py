"""RED-first unit tests for scale-grid unit B8 (prereg perf checklist 4): the smoke-shard
utilization gate. The FIRST full shard per reader logs tokens/sec + GPU util; a <50%-util
configuration HALTS the grid (raise -> box FATAL) instead of burning the full run at partial
occupancy. No GPU here -- the sampler is injected/monkeypatched.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_util_gate.py

U1  util below the 50% floor after the first full shard -> RuntimeError (marker-safe message).
U2  healthy util passes; the gate is one-shot per process (per-reader subprocess = per reader).
U3  unavailable sampler (no nvidia-smi / CPU env) -> log-only, never a false halt.
U4  prose-control shards (tiny gate-4 cells) do NOT consume the first-shard slot.
U5  non-shard_done stages are inert.
U6  the log line carries tokens/sec + util and no box-marker / labkit-FATAL substring.
U7  gpu_util_sample survives a machine without nvidia-smi (returns None, never raises).
U8  main() feeds the hook real measurements (tokens=, secs= at the call site).
U9  token-floor exemption (2026-07-14 review CRIT 1): shards under UTIL_GATE_TOKEN_FLOOR
    (smoke slices, 1-context N slivers) neither halt nor consume the first-shard slot; the
    next full-size shard is still gated.
U10 the floor is 20k tokens: real 12-context shards (~200k+) clear it ~10x; the rider smoke
    slice (~1-2k) and 1-context N slivers (~15-18k) are exempt.
"""
import contextlib
import inspect
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


import lr_grid as G   # noqa: E402

BAD = ("LRG_READY", "LRG_DONE", "LRG_FATAL", "LR_READY", "LR_DONE", "LR_FATAL",
       "MC_READY", "MC_DONE", "MC_FATAL", "CUDA error", "CUDA out of memory",
       "Traceback (most recent call last)", "ModuleNotFoundError",
       "torch.cuda.OutOfMemoryError")


def reset():
    G._UTIL_STATE["first_done"] = False


def call(sample, **fields):
    """Invoke the hook with an injected util sample, capturing stdout."""
    orig = G.gpu_util_sample
    G.gpu_util_sample = lambda: sample
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            r = G.util_gate_hook("shard_done", **fields)
    finally:
        G.gpu_util_sample = orig
    return r, buf.getvalue()


# ---------------------------------------------------------------- U1: low util halts
try:
    reset()
    try:
        call(30.0, shard="q__q__evoked_A.pt", tokens=120000, secs=60.0)
        check("U1 util < 50% after the first full shard raises (grid halt)", False,
              "no exception raised")
    except RuntimeError as e:
        check("U1 util < 50% after the first full shard raises (grid halt)", True)
        check("U1 halt message is marker/FATAL-substring safe and actionable (--batch)",
              not any(s in str(e) for s in BAD) and "--batch" in str(e), repr(str(e)))
except Exception as e:
    check("U1 low-util halt", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- U2: healthy + one-shot
try:
    reset()
    r, out = call(87.0, shard="q__q__evoked_A.pt", tokens=120000, secs=60.0)
    check("U2 healthy util passes (returns None, no raise)", r is None)
    r2, out2 = call(5.0, shard="q__q__evoked_B.pt", tokens=1, secs=1.0)
    check("U2 the gate is one-shot: later shards are not re-gated (per-reader subprocess "
          "restarts the state)", r2 is None and out2 == "", f"out2 = {out2!r}")
except Exception as e:
    check("U2 healthy/one-shot", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- U3: sampler unavailable
try:
    reset()
    r, out = call(None, shard="q__q__evoked_A.pt", tokens=120000, secs=60.0)
    check("U3 unavailable sampler -> log-only, never a false halt",
          r is None and "util=?" in out, f"out = {out!r}")
except Exception as e:
    check("U3 sampler unavailable", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- U4: prose shards don't count
try:
    reset()
    r, out = call(5.0, shard="falcon3-1b__prose__control_N.pt", tokens=300, secs=2.0)
    check("U4 a prose-control shard neither halts nor consumes the first-shard slot",
          r is None and G._UTIL_STATE["first_done"] is False, f"state = {G._UTIL_STATE}")
    try:
        call(30.0, shard="falcon3-1b__qwen2.5-1.5b__evoked_A.pt", tokens=90000, secs=30.0)
        check("U4 the NEXT real shard is still gated", False, "no exception raised")
    except RuntimeError:
        check("U4 the NEXT real shard is still gated", True)
except Exception as e:
    check("U4 prose skip", False, f"raised {type(e).__name__}: {e}")

# ------------------------------------------------- U9/U10: token-floor exemption (CRIT 1)
try:
    reset()
    r, out = call(30.0, shard="qwen2.5-1.5b__llama70b__secret_sustain_N.pt",
                  tokens=2000, secs=5.0)
    check("U9 a sub-floor shard (smoke slice / 1-ctx sliver) neither halts nor consumes the "
          "first-shard slot",
          r is None and G._UTIL_STATE["first_done"] is False, f"state = {G._UTIL_STATE}")
    check("U9 the exemption is disclosed in the log (marker-safe)",
          "floor" in out and not any(s in out for s in BAD), f"out = {out!r}")
    try:
        call(30.0, shard="qwen2.5-1.5b__llama70b__secret_sustain_R.pt",
             tokens=200000, secs=60.0)
        check("U9 the NEXT full-size shard is still gated (the gate polices real work)",
              False, "no exception raised")
    except RuntimeError:
        check("U9 the NEXT full-size shard is still gated (the gate polices real work)", True)
    check("U10 the floor is 20k tokens (full 12-ctx shards clear ~10x; N slivers/smoke exempt)",
          getattr(G, "UTIL_GATE_TOKEN_FLOOR", None) == 20000)
    reset()
    try:
        call(30.0, shard="q__q__evoked_A.pt", tokens=None, secs=None)
        check("U10 a missing token count never falsely exempts (gate still applies)", False,
              "no exception raised")
    except RuntimeError:
        check("U10 a missing token count never falsely exempts (gate still applies)", True)
except Exception as e:
    check("U9/U10 token floor", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- U5: other stages inert
try:
    reset()
    r = G.util_gate_hook("something_else")
    check("U5 non-shard_done stages are inert",
          r is None and G._UTIL_STATE["first_done"] is False)
except Exception as e:
    check("U5 stage filter", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- U6: log content + safety
try:
    reset()
    r, out = call(91.5, shard="q__q__evoked_A.pt", tokens=120000, secs=60.0)
    check("U6 log line carries tokens/sec and util",
          "tok/s" in out and "util=91" in out, f"out = {out!r}")
    check("U6 log line is marker/FATAL-substring safe",
          not any(s in out for s in BAD), f"out = {out!r}")
except Exception as e:
    check("U6 log content", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- U7: sampler robustness
try:
    v = G.gpu_util_sample()
    check("U7 gpu_util_sample never raises without nvidia-smi (None or a float)",
          v is None or isinstance(v, float), f"got {v!r}")
except Exception as e:
    check("U7 gpu_util_sample robustness", False, f"raised {type(e).__name__}: {e}")

# ---------------------------------------------------------------- U8: main call site enriched
msrc = inspect.getsource(G.main)
check("U8 main feeds tokens= and secs= into the util gate call site",
      "tokens=" in msrc and "secs=" in msrc and "util_gate_hook(" in msrc)
check("U8 UTIL_GATE_MIN is the registered 50% floor",
      getattr(G, "UTIL_GATE_MIN", None) == 50.0)

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
