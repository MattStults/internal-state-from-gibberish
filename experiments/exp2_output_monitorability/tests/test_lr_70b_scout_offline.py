"""RED-first unit tests for the OFFICIAL Llama-70B scout calibration adapter
(analysis/lr_70b_scout_offline.py -- HANDOFF §2 steps 3-5, §8.1). CPU-only, synthetic inputs only.

The adapter parses the Together serverless raw batch output (lr_raw_batch_output.jsonl), rebuilds
the per-arm (n x 12) context matrices S[i, j] = LL(stream | concept_j ctx) - LL(stream | neutral)
using the SCOUT'S OWN span finder (harness/run_llama70b_scout.py::_find_stream_span_lps -- imported,
never copied), and scores them through the CERTIFIED calibrator (lr_reader_offline.evaluate_cell --
same function object) + per_token_stats for gate 3. Nothing numeric is reimplemented.

  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_lr_70b_scout_offline.py

F1  certified reuse is literal: find_stream_span_lps IS the scout's function object;
    evaluate_cell / per_token_stats ARE lr_reader_offline's function objects.
F2  parse_raw_batch: custom_id lr:{arm}:{concept}:{stream_idx}:{ctx} parsed; span log-probs
    extracted through the scout's span finder match the planted values; failures counted by reason.
F3  build_arm: S is (n x 12); the true-concept column comes from the "matched" request; every
    column is neutral-subtracted; a stream missing ANY of the 13 contexts is dropped and counted.
F4  calibration goes through the certified evaluate_cell (identical bits to a direct call; a
    strong synthetic diagonal recovers high bits).
F5  raw diagonal reported in nats AND bits (bits = nats / ln 2) -- the label bug that motivated
    this adapter.
F6  generic-lift decomposition: a pure-lift synthetic (matched == mismatched >> 0 vs neutral)
    yields centered ~ 0, calibrated ~ 0 bits -- lift alone must NOT read as channel signal.
F7  concept bootstrap: seed 20260713, 10k resamples of the 12 concept means; pure-lift centered
    CI does not clear zero.
F8  gate 3: a large mismatched per-token median (> 0.02 nats/tok) voids the arm.
F9  length truncation: trunc=K sums only the first K span tokens.
F10 main end-to-end on a synthetic run dir: writes the results JSON with the required blocks
    (raw diag nats+bits, lift, centered+CI, gate3, calibrated, truncation table, drop counts,
    am5 not-computable disclosure, named-calls disposition).
F11 no reimplemented numerics (no local ce_bits / fit_temperature / softmax / span finder copy).
"""
import importlib.util
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ANALYSIS = os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, ANALYSIS)

import numpy as np  # noqa: E402

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M_PATH = os.path.join(ANALYSIS, "lr_70b_scout_offline.py")
try:
    M = load_module("lr_70b_scout_offline", M_PATH)
    check("import analysis/lr_70b_scout_offline.py", True)
except Exception as e:
    M = None
    check("import analysis/lr_70b_scout_offline.py", False, f"{type(e).__name__}: {e}")


CONCEPTS = ["anger", "celebration", "curiosity", "debugging", "deception", "fear",
            "loneliness", "obedience", "ocean", "security", "silence", "warmth"]
ARM = "secret_word"


# ---------------------------------------------------------------- synthetic run builders
def _echo_line(arm, concept, idx, ctx, stream_tokens, stream_lps, prefix=("CTXHDR ", "pad ")):
    """One raw-batch output line in the Together echo shape the scout's span finder reads:
    response.body.prompt[0].logprobs.{tokens, token_logprobs} with token_logprobs[0] = null."""
    tokens = list(prefix) + list(stream_tokens)
    lps = [None] + [-0.5] * (len(prefix) - 1) + [float(x) for x in stream_lps]
    body = {"prompt": [{"logprobs": {"tokens": tokens, "token_logprobs": lps}}]}
    return {"custom_id": f"lr:{arm}:{concept}:{idx}:{ctx}",
            "response": {"status_code": 200, "body": body}}


def _stream_tokens(idx, T=8):
    return [f"w{idx}x{t} " for t in range(T)]


def _make_run(dirpath, diag=0.0, lift=0.0, n_per=3, T=8, noise=0.01, seed=0,
              drop_ctx_for=None, arm=ARM):
    """Synthetic streams json + raw jsonl. Per stream: neutral lps sum ~ -T; matched sum =
    neutral + lift + diag; each mismatched sum = neutral + lift (generic lift only). Returns
    (streams_path, raw_path). drop_ctx_for: (stream_idx, ctx) line to omit (drop-count test)."""
    rng = np.random.default_rng(seed)
    streams, lines = [], []
    idx = 0
    for c in CONCEPTS:
        for _ in range(n_per):
            toks = _stream_tokens(idx, T)
            streams.append({"concept": c, "arm": arm, "text": "".join(toks),
                            "accepted": True, "stream_idx": idx, "attempt_idx": 0,
                            "token_ids": None})
            base = -1.0  # per-token neutral level
            for ctx in ["matched", "neutral"] + [k for k in CONCEPTS if k != c]:
                if drop_ctx_for is not None and (idx, ctx) == (drop_ctx_for[0], ctx) \
                        and ctx == drop_ctx_for[1]:
                    continue
                if ctx == "neutral":
                    tot = base * T
                elif ctx == "matched":
                    tot = base * T + lift + diag + rng.normal(0, noise)
                else:
                    tot = base * T + lift + rng.normal(0, noise)
                lps = [tot / T] * T
                lines.append(_echo_line(arm, c, idx, ctx, toks, lps))
            idx += 1
    sp = os.path.join(dirpath, "streams.json")
    rp = os.path.join(dirpath, "raw.jsonl")
    with open(sp, "w") as f:
        json.dump(streams, f)
    with open(rp, "w") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    return sp, rp


if M is not None:
    # ---- F1: literal certified reuse ------------------------------------------------------
    try:
        SC = sys.modules.get("run_llama70b_scout")
        LRO = sys.modules.get("lr_reader_offline")
        check("F1 span finder IS the scout's _find_stream_span_lps (same object, not a copy)",
              SC is not None and M.find_stream_span_lps is SC._find_stream_span_lps)
        check("F1 evaluate_cell IS lr_reader_offline.evaluate_cell (certified calibrator)",
              LRO is not None and M.evaluate_cell is LRO.evaluate_cell)
        check("F1 per_token_stats IS lr_reader_offline.per_token_stats (gate 3)",
              LRO is not None and M.per_token_stats is LRO.per_token_stats)
        check("F1 frozen constants: gate3 bound 0.02, bootstrap seed 20260713 / 10k, Ks 16/32/64/full",
              M.GATE3_BOUND == 0.02 and M.BOOT_SEED == 20260713 and M.BOOT_N == 10000
              and tuple(M.TRUNC_KS) == (16, 32, 64, None))
    except Exception as e:
        check("F1 certified reuse", False, f"raised {type(e).__name__}: {e}")

    # ---- F2: raw batch parsing ------------------------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as td:
            sp, rp = _make_run(td, diag=2.0, lift=1.0, n_per=2, T=8)
            streams = json.load(open(sp))
            span_lps, failures, n_lines = M.parse_raw_batch(rp, streams)
            check("F2 all lines parsed (13 contexts x streams)",
                  n_lines == 13 * len(streams) and len(span_lps) == n_lines,
                  f"n_lines={n_lines} spans={len(span_lps)} fails={len(failures)}")
            key = (ARM, 0, "neutral")
            ok = key in span_lps and abs(sum(span_lps[key]) - (-8.0)) < 1e-6
            check("F2 span log-probs match the planted per-token values (scout span finder)",
                  ok, f"{span_lps.get(key)}")
            # a no-body line is a counted failure, not a crash
            with open(rp, "a") as f:
                f.write(json.dumps({"custom_id": f"lr:{ARM}:anger:999:matched",
                                    "response": {}}) + "\n")
            _, fails2, _ = M.parse_raw_batch(rp, streams)
            check("F2 missing-body line counted as a failure by reason",
                  any("body" in r for _, r in fails2), f"{fails2[:2]}")
    except Exception as e:
        check("F2 parse_raw_batch", False, f"raised {type(e).__name__}: {e}")

    # ---- F3: matrix build, matched column, neutral subtraction, drops ----------------------
    try:
        with tempfile.TemporaryDirectory() as td:
            sp, rp = _make_run(td, diag=3.0, lift=1.0, n_per=2, T=8, noise=0.0)
            streams = json.load(open(sp))
            span_lps, _, _ = M.parse_raw_batch(rp, streams)
            S, y, T, n_drop, kept, labels = M.build_arm(ARM, streams, span_lps, CONCEPTS)
            check("F3 S is (n_streams x 12), y aligned",
                  S.shape == (len(streams), 12) and len(y) == S.shape[0] and n_drop == 0,
                  f"S={S.shape} drop={n_drop}")
            # matched column = (neutral + lift + diag) - neutral = lift + diag; others = lift
            check("F3 true-concept column comes from the matched request (= lift + diag)",
                  abs(S[0, y[0]] - 4.0) < 1e-6, f"{S[0, y[0]]}")
            off = S[0, [j for j in range(12) if j != y[0]]]
            check("F3 mismatched columns are neutral-subtracted (= lift)",
                  np.allclose(off, 1.0, atol=1e-6), f"{off[:3]}")
            # dropping one context for stream 5 drops exactly that stream, counted
            sp2, rp2 = _make_run(td + "/d", diag=3.0, lift=1.0, n_per=2, T=8, noise=0.0,
                                 drop_ctx_for=(5, "neutral")) if os.makedirs(td + "/d") is None \
                else (None, None)
            streams2 = json.load(open(sp2))
            span2, _, _ = M.parse_raw_batch(rp2, streams2)
            S2, y2, _, n_drop2, kept2, _ = M.build_arm(ARM, streams2, span2, CONCEPTS)
            check("F3 a stream missing any of the 13 contexts is dropped and counted",
                  n_drop2 == 1 and S2.shape[0] == len(streams2) - 1 and 5 not in kept2,
                  f"drop={n_drop2} n={S2.shape[0]}")
    except Exception as e:
        check("F3 build_arm", False, f"raised {type(e).__name__}: {e}")

    # ---- F4: certified calibration recovers a strong diagonal ------------------------------
    try:
        LRO = sys.modules["lr_reader_offline"]
        with tempfile.TemporaryDirectory() as td:
            sp, rp = _make_run(td, diag=8.0, lift=1.0, n_per=6, T=8, noise=0.05)
            streams = json.load(open(sp))
            span_lps, _, _ = M.parse_raw_batch(rp, streams)
            S, y, T, _, _, _ = M.build_arm(ARM, streams, span_lps, CONCEPTS)
            cell = M.evaluate_cell(S, y)
            direct = LRO.evaluate_cell(S, y)
            check("F4 adapter bits == direct certified evaluate_cell bits",
                  abs(cell["bits_mean"] - direct["bits_mean"]) < 1e-12, f"{cell['bits_mean']}")
            check("F4 strong synthetic diagonal recovers high calibrated bits (> 3)",
                  cell["bits_mean"] > 3.0, f"bits={cell['bits_mean']}")
    except Exception as e:
        check("F4 certified calibration", False, f"raised {type(e).__name__}: {e}")

    # ---- F5/F6/F7/F8: the pure-generic-lift arm ---------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as td:
            # matched == mismatched == +4 nats over neutral: 100% generic lift, zero signal.
            sp, rp = _make_run(td, diag=0.0, lift=4.0, n_per=6, T=8, noise=0.05)
            streams = json.load(open(sp))
            span_lps, _, _ = M.parse_raw_batch(rp, streams)
            res = M.score_arm(ARM, streams, span_lps, CONCEPTS)
            check("F5 raw diagonal reported in nats AND bits with bits = nats/ln2",
                  abs(res["raw_diag_nats"] - 4.0) < 0.1
                  and abs(res["raw_diag_bits"] - res["raw_diag_nats"] / np.log(2)) < 1e-9,
                  f"nats={res['raw_diag_nats']} bits={res['raw_diag_bits']}")
            check("F6 generic lift ~ raw diagonal (pure-lift synthetic)",
                  abs(res["generic_lift_nats"] - 4.0) < 0.1, f"{res['generic_lift_nats']}")
            check("F6 centered diagonal ~ 0 and calibrated ~ 0 bits (lift is not signal)",
                  abs(res["centered_diag_nats"]["mean"]) < 0.1
                  and abs(res["calibrated"]["bits_mean"]) < 0.15,
                  f"centered={res['centered_diag_nats']['mean']} "
                  f"bits={res['calibrated']['bits_mean']}")
            ci = res["centered_diag_nats"]
            check("F7 concept bootstrap (seed 20260713, 10k) CI present; pure-lift does not clear 0",
                  ci["seed"] == 20260713 and ci["n_boot"] == 10000
                  and ci["ci_lo"] <= 0 <= ci["ci_hi"] and ci["clears_zero"] is False,
                  f"CI=[{ci['ci_lo']}, {ci['ci_hi']}]")
            # gate 3: lift = 4 nats over T=8 tokens -> mismatched median 0.5 nats/tok >> 0.02
            check("F8 gate 3 voids the arm when mismatched per-token median > 0.02 nats/tok",
                  res["gate3"]["passed"] is False and res["voided"] is True
                  and res["gate3"]["mismatched_pt"] > 0.02, f"{res['gate3']}")
            # and a lift-free arm passes gate 3
            sp0, rp0 = _make_run(td + "/g", diag=0.5, lift=0.0, n_per=3, T=8, noise=0.01) \
                if os.makedirs(td + "/g") is None else (None, None)
            st0 = json.load(open(sp0))
            sl0, _, _ = M.parse_raw_batch(rp0, st0)
            res0 = M.score_arm(ARM, st0, sl0, CONCEPTS)
            check("F8 gate 3 passes a centered (lift-free) arm",
                  res0["gate3"]["passed"] is True and res0["voided"] is False, f"{res0['gate3']}")
    except Exception as e:
        check("F5-F8 lift decomposition / gate3", False, f"raised {type(e).__name__}: {e}")

    # ---- F9: truncation sums only the first K tokens ----------------------------------------
    try:
        with tempfile.TemporaryDirectory() as td:
            sp, rp = _make_run(td, diag=2.0, lift=1.0, n_per=2, T=8, noise=0.0)
            streams = json.load(open(sp))
            span_lps, _, _ = M.parse_raw_batch(rp, streams)
            Sf, yf, Tf, _, _, _ = M.build_arm(ARM, streams, span_lps, CONCEPTS)
            S4, y4, T4, _, _, _ = M.build_arm(ARM, streams, span_lps, CONCEPTS, trunc=4)
            # per-token-constant lps: truncating to 4 of 8 tokens halves every entry
            check("F9 trunc=K sums the first K span tokens (half the total at K = T/2)",
                  np.allclose(S4, Sf / 2.0, atol=1e-9) and float(T4.max()) == 4.0,
                  f"S4[0,y]={S4[0, y4[0]]} Sf[0,y]={Sf[0, yf[0]]}")
    except Exception as e:
        check("F9 truncation", False, f"raised {type(e).__name__}: {e}")

    # ---- F10: main end-to-end ----------------------------------------------------------------
    try:
        with tempfile.TemporaryDirectory() as td:
            sp, rp = _make_run(td, diag=1.0, lift=1.0, n_per=3, T=8, noise=0.05)
            out = os.path.join(td, "results.json")
            res = M.main(raw_path=rp, streams_path=sp, out_json=out)
            check("F10 main writes the results JSON", os.path.exists(out))
            j = json.load(open(out))
            arm = j["arms"][ARM]
            need = ["n_kept", "n_dropped", "raw_diag_nats", "raw_diag_bits",
                    "generic_lift_nats", "centered_diag_nats", "gate3", "voided",
                    "calibrated", "truncation"]
            check("F10 per-arm block has all required fields",
                  all(k in arm for k in need), f"missing={[k for k in need if k not in arm]}")
            check("F10 truncation table has K=16/32/64/full entries",
                  set(arm["truncation"]) == {"K16", "K32", "K64", "full"},
                  f"{list(arm['truncation'])}")
            check("F10 am5 controls disclosed as NOT computable on this run (no token ids/gen_topk)",
                  j.get("am5_controls", {}).get("char") is None
                  and j.get("am5_controls", {}).get("position") is None
                  and "not computable" in (j.get("am5_controls", {}).get("note") or "").lower(),
                  f"{j.get('am5_controls')}")
            check("F10 named-calls disposition block present (both readings, recommendation)",
                  "named_calls_disposition" in j
                  and "if_accepted_as_72b_test" in j["named_calls_disposition"]
                  and "recommended" in j["named_calls_disposition"],
                  f"{list(j.get('named_calls_disposition', {}))}")
    except Exception as e:
        check("F10 main end-to-end", False, f"raised {type(e).__name__}: {e}")

    # ---- F11: no reimplemented numerics -------------------------------------------------------
    try:
        with open(M_PATH) as f:
            src = f.read()
        check("F11 no reimplemented calibration (no local ce_bits/fit_temperature/log_softmax)",
              "def ce_bits" not in src and "def fit_temperature" not in src
              and "log_softmax" not in src)
        check("F11 no copied span finder (imports the scout's, defines none)",
              "def _find_stream_span_lps" not in src and "def find_stream_span_lps" not in src)
    except Exception as e:
        check("F11 no reimplemented numerics", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
