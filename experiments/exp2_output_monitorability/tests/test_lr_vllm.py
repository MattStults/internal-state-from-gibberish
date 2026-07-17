"""RED-first unit tests for LR-72B unit 1: the vLLM prompt_logprobs LR scorer (src/lr_vllm.py).

The 72B path teacher-forces via a self-hosted vLLM server's `prompt_logprobs` instead of our own
HF forward pass. This suite pins the CORRECTNESS points that a GPU cannot verify for us (we test
against a MOCK vLLM response, never a real server): index alignment of prompt_logprobs to the
gibberish token span, that the summed logprob is of the ACTUAL provided token (not the argmax),
tokenizer/round-trip parity, and the LR = LL(ctx) - LL(neutral) subtraction. No GPU, no network.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_lr_vllm.py

V1  render_prompt_ids: Qwen chat render (system=ctx, user=GEN_PROMPT, assistant=gibberish) that is
    an exact TOKEN PREFIX-extension -- the context render + the re-tokenized gibberish span, and it
    reports the [start,end) index of the gibberish span in the full prompt token list.
V2  span_logprobs: given a vLLM /v1/completions response's prompt_logprobs list, extract the
    per-token logprob of the ACTUAL provided token over the gibberish span, aligned by index.
    prompt_logprobs[i] is P(token i | tokens<i); index 0 is null. Asserts the returned entry
    carries the provided token id (guards the argmax-vs-provided-token bug).
V3  a returned-token-id MISMATCH (vLLM logged a different token than we teacher-forced) raises --
    scoring the wrong token's logprob would be a silent correctness failure.
V4  ll_over_span sums span_logprobs -> one LL scalar; noeos rule drops a terminal eos position.
V5  lr_score: LR = LL(ctx) - LL(neutral) over the SAME gibberish span (two prompts, same stream).
V6  tokenizer parity: assert_prompt_roundtrips -- the ids we send must re-tokenize/round-trip
    identically (send token_ids when the API takes them; else verify text round-trip), FATAL on
    drift, with a marker/FATAL-substring-safe message.
V7  request-shape builder: completions_request emits the pinned prompt_logprobs body
    (prompt=token_ids, max_tokens=0, prompt_logprobs>=1, temperature 0) -- the exact shape the
    box will POST; no network in the test (the transport is injectable).
V8  marker safety: the module prints/strings carry NO box marker (LR72_*) or labkit FATAL
    substring (the attempt-4 collision class); the scorer defines no log-softmax numerics of its
    own (vLLM computes the logprobs; we only align+sum).
"""
import ast
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V_PATH = os.path.join(REPO, "src", "lr_vllm.py")
try:
    V = load_module("lr_vllm", V_PATH)
    check("import src/lr_vllm.py", True)
except Exception as e:
    V = None
    check("import src/lr_vllm.py", False, f"{type(e).__name__}: {e}")


class StubTok:
    """Deterministic char-level tokenizer stub, chat-template aware. Encodes each char to its
    ordinal; the chat template wraps (system,user) with sentinel ids and (optionally) the
    assistant content, so a generation-header render is a strict TOKEN PREFIX of the full
    [system,user,assistant] render -- mirroring Qwen2.5's prefix-stable template."""
    SYS, USR, ASST, GEN = 900001, 900002, 900003, 900004  # role sentinel token ids
    eos_token_id = 151645

    def __call__(self, text, add_special_tokens=False):
        class _R:
            pass
        r = _R()
        r.input_ids = [ord(c) for c in text]
        return r

    def _wrap(self, msgs, add_generation_prompt):
        ids = []
        for m in msgs:
            role = {"system": self.SYS, "user": self.USR, "assistant": self.ASST}[m["role"]]
            ids.append(role)
            ids.extend(ord(c) for c in m["content"])
        if add_generation_prompt:
            ids.append(self.GEN)
        return ids

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True, **kw):
        ids = self._wrap(msgs, add_generation_prompt)
        if not tokenize:
            return "".join(chr(min(i, 0x10FFFF)) for i in ids)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


if V is not None:
    tok = StubTok()
    SYSTEM = "the secret word is paper"
    GEN_PROMPT = "emit random tokens"
    GIB = "qx z fjm"            # gibberish stream text
    gib_ids = [ord(c) for c in GIB]

    # ---- V1: render + span index ---------------------------------------------------------
    try:
        prompt_ids, span = V.render_prompt_ids(tok, SYSTEM, GEN_PROMPT, gib_ids)
        ctx_ids = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": GEN_PROMPT}],
            add_generation_prompt=True)
        s, e = span
        check("V1 gibberish span sits immediately after the generation-header context prefix",
              s == len(ctx_ids), f"span start {s} != ctx len {len(ctx_ids)}")
        check("V1 span covers exactly the gibberish tokens",
              e - s == len(gib_ids) and prompt_ids[s:e] == gib_ids, f"span {span}")
        check("V1 prompt is the context prefix + gibberish (prefix-stable render)",
              prompt_ids[:len(ctx_ids)] == ctx_ids and prompt_ids[s:e] == gib_ids)
    except Exception as ex:
        check("V1 render_prompt_ids", False, f"raised {type(ex).__name__}: {ex}")

    # ---- V2: span_logprobs alignment (provided token, not argmax) ------------------------
    # Build a vLLM prompt_logprobs list: one dict per prompt token, keyed by token-id-string ->
    # {logprob, rank, decoded_token}. Index 0 is null (first token has no preceding context).
    def _plps(prompt_ids, provided_lp, argmax_other=True):
        out = [None]
        for i in range(1, len(prompt_ids)):
            tid = prompt_ids[i]
            entry = {str(tid): {"logprob": provided_lp[i], "rank": 3, "decoded_token": chr(tid)}}
            if argmax_other:                     # a DIFFERENT token is the argmax (rank 1)
                entry[str(tid + 1)] = {"logprob": 0.0, "rank": 1, "decoded_token": "?"}
            out.append(entry)
        return out
    try:
        prompt_ids, span = V.render_prompt_ids(tok, SYSTEM, GEN_PROMPT, gib_ids)
        s, e = span
        lp = {i: -(i * 0.1) for i in range(len(prompt_ids))}   # distinct per-position logprobs
        resp_plps = _plps(prompt_ids, lp, argmax_other=True)
        span_lps = V.span_logprobs(resp_plps, prompt_ids, span)
        check("V2 one logprob per gibberish token", len(span_lps) == e - s, f"{len(span_lps)}")
        check("V2 the extracted logprobs are the PROVIDED tokens' (not the argmax rank-1)",
              all(abs(span_lps[j] - lp[s + j]) < 1e-9 for j in range(e - s)),
              f"got {span_lps}")
    except Exception as ex:
        check("V2 span_logprobs alignment", False, f"raised {type(ex).__name__}: {ex}")

    # ---- V3: token-id mismatch raises ----------------------------------------------------
    try:
        prompt_ids, span = V.render_prompt_ids(tok, SYSTEM, GEN_PROMPT, gib_ids)
        s, e = span
        bad = [None]
        for i in range(1, len(prompt_ids)):
            # every entry reports a DIFFERENT token id than the one we teacher-forced
            bad.append({str(prompt_ids[i] + 7): {"logprob": -1.0, "rank": 1}})
        try:
            V.span_logprobs(bad, prompt_ids, span)
            check("V3 a provided-token-id absent from prompt_logprobs[i] raises", False,
                  "no exception")
        except RuntimeError as ex:
            markerbad = ("LR72_READY", "LR72_DONE", "LR72_FATAL", "CUDA out of memory",
                         "Traceback (most recent call last)")
            check("V3 a provided-token-id absent from prompt_logprobs[i] raises", True)
            check("V3 mismatch message is marker/FATAL-substring safe",
                  not any(b in str(ex) for b in markerbad), repr(str(ex)))
    except Exception as ex:
        check("V3 token-id mismatch", False, f"raised {type(ex).__name__}: {ex}")

    # ---- V4: ll_over_span sum + eos rule -------------------------------------------------
    try:
        vals = [-0.5, -1.0, -0.25]
        check("V4 ll_over_span sums the span logprobs",
              abs(V.ll_over_span(vals) - (-1.75)) < 1e-9)
        # eos rule: a terminal eos token drops its (last) span position from the sum
        gib_eos = gib_ids + [tok.eos_token_id]
        prompt_ids, span = V.render_prompt_ids(tok, SYSTEM, GEN_PROMPT, gib_eos)
        s, e = span
        lp2 = {i: -1.0 for i in range(len(prompt_ids))}
        span_lps2 = V.span_logprobs(_plps(prompt_ids, lp2), prompt_ids, span)
        ll_eos = V.ll_over_span(span_lps2)
        ll_free = V.ll_over_span(span_lps2, drop_last_eos=True, span_ids=gib_eos,
                                 eos_id=tok.eos_token_id)
        check("V4 eos-free LL drops exactly the terminal eos position",
              abs(ll_eos - (-1.0 * len(gib_eos))) < 1e-9
              and abs(ll_free - (-1.0 * (len(gib_eos) - 1))) < 1e-9,
              f"eos={ll_eos} free={ll_free}")
    except Exception as ex:
        check("V4 ll_over_span", False, f"raised {type(ex).__name__}: {ex}")

    # ---- V5: lr_score = LL(ctx) - LL(neutral) over the SAME stream -----------------------
    class MockClient:
        """Records requests; returns a prompt_logprobs response whose provided-token logprob is
        -mag at every position (a per-system constant), so LL(ctx)-LL(neutral) is deterministic."""
        def __init__(self, mag_by_system):
            self.mag_by_system, self.requests = mag_by_system, []

        def completions(self, prompt_ids, prompt_logprobs=1, **kw):
            self.requests.append(dict(prompt_ids=list(prompt_ids),
                                      prompt_logprobs=prompt_logprobs, **kw))
            # recover which system was rendered from the sentinel-wrapped prompt: the chars after
            # the SYS sentinel up to the USR sentinel
            try:
                a = prompt_ids.index(StubTok.SYS) + 1
                b = prompt_ids.index(StubTok.USR)
                system = "".join(chr(c) for c in prompt_ids[a:b])
            except ValueError:
                system = ""
            mag = self.mag_by_system.get(system, 1.0)
            plps = [None]
            for i in range(1, len(prompt_ids)):
                plps.append({str(prompt_ids[i]): {"logprob": -mag, "rank": 1}})
            return dict(prompt_logprobs=plps)
    try:
        NEUTRAL = "neutral persona"
        client = MockClient({SYSTEM: 0.5, NEUTRAL: 2.0})   # ctx more likely -> positive LR
        lr = V.lr_score(client, tok, SYSTEM, NEUTRAL, GEN_PROMPT, gib_ids)
        # LL(ctx) = -0.5*n, LL(neutral) = -2.0*n -> LR = 1.5*n
        n = len(gib_ids)
        check("V5 LR = LL(ctx) - LL(neutral) over the same gibberish span",
              abs(lr - (1.5 * n)) < 1e-9, f"lr={lr} expected {1.5 * n}")
        check("V5 both prompts teacher-force the SAME stream tokens (span ids identical)",
              client.requests[0]["prompt_ids"][-n:] == gib_ids
              and client.requests[1]["prompt_ids"][-n:] == gib_ids)
        check("V5 prompt_logprobs requested (teacher-forcing, not generation)",
              all(r["prompt_logprobs"] >= 1 for r in client.requests))
        # return_pertok: the per-token LR difference vector (for the offline position control);
        # its sum equals the scalar LR (same eos rule).
        client2 = MockClient({SYSTEM: 0.5, NEUTRAL: 2.0})
        lr2, pertok = V.lr_score(client2, tok, SYSTEM, NEUTRAL, GEN_PROMPT, gib_ids,
                                 return_pertok=True)
        check("V5 return_pertok yields a per-token LR-diff vector whose sum == the scalar LR",
              len(pertok) == n and abs(sum(pertok) - lr2) < 1e-9
              and all(abs(p - 1.5) < 1e-9 for p in pertok), f"pertok={pertok} lr={lr2}")
    except Exception as ex:
        check("V5 lr_score", False, f"raised {type(ex).__name__}: {ex}")

    # ---- V6: tokenizer/round-trip parity -------------------------------------------------
    try:
        # a stream whose ids round-trip cleanly under this tokenizer passes
        V.assert_prompt_roundtrips(tok, [gib_ids])
        check("V6 clean round-trip passes", True)

        class DriftTok(StubTok):
            def __call__(self, text, add_special_tokens=False):
                class _R:
                    pass
                r = _R()
                r.input_ids = [ord(c) for c in text] + [123]   # re-encode adds a token
                return r
        try:
            V.assert_prompt_roundtrips(DriftTok(), [gib_ids])
            check("V6 a tokenizer that does not round-trip the span raises (parity FATAL)",
                  False, "no exception")
        except RuntimeError as ex:
            mbad = ("LR72_READY", "LR72_DONE", "LR72_FATAL", "CUDA out of memory")
            check("V6 a tokenizer that does not round-trip the span raises (parity FATAL)", True)
            check("V6 parity message is marker/FATAL-substring safe",
                  not any(b in str(ex) for b in mbad), repr(str(ex)))
    except Exception as ex:
        check("V6 tokenizer parity", False, f"raised {type(ex).__name__}: {ex}")

    # ---- V7: request-shape builder -------------------------------------------------------
    try:
        body = V.completions_request(model="Qwen/Qwen2.5-72B-Instruct", prompt_ids=[1, 2, 3])
        check("V7 request body sends token ids as the prompt (tokenizer parity: no re-tokenize)",
              body.get("prompt") == [1, 2, 3])
        check("V7 request asks for prompt_logprobs (teacher-forcing) and max_tokens 0 "
              "(no generation)",
              int(body.get("prompt_logprobs", 0)) >= 1 and body.get("max_tokens") == 0)
        check("V7 request is greedy/deterministic (temperature 0 -- scoring, not sampling)",
              float(body.get("temperature", 1.0)) == 0.0)
        check("V7 model pinned in the body", body.get("model") == "Qwen/Qwen2.5-72B-Instruct")
    except Exception as ex:
        check("V7 completions_request", False, f"raised {type(ex).__name__}: {ex}")

    # ---- V8: marker safety + no reimplemented numerics -----------------------------------
    try:
        BADMARK = ("LR72_READY", "LR72_DONE", "LR72_FATAL", "LRG_READY", "LRG_DONE", "LRG_FATAL",
                   "LR_READY", "LR_DONE", "LR_FATAL")
        FATAL_SUBSTR = ("CUDA error", "CUDA out of memory", "Traceback (most recent call last)",
                        "ModuleNotFoundError", "torch.cuda.OutOfMemoryError")
        with open(V_PATH) as f:
            vsrc = f.read()
        tree = ast.parse(vsrc)
        prints = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                lits = [c.value for c in ast.walk(node)
                        if isinstance(c, ast.Constant) and isinstance(c.value, str)]
                prints.append("".join(lits))
        hits = [p for p in prints for sub in BADMARK + FATAL_SUBSTR if sub in p]
        check("V8 lr_vllm prints carry no box marker / labkit FATAL substring", not hits,
              "; ".join(hits))
        check("V8 no reimplemented log-softmax numerics (vLLM computes the logprobs; we align+sum)",
              "log_softmax" not in vsrc and "logsumexp" not in vsrc)
        check("V8 the module never contains the LR72 box markers (M1 parity: box owns them)",
              all(m not in vsrc for m in ("LR72_READY", "LR72_DONE", "LR72_FATAL")))
    except Exception as ex:
        check("V8 marker safety", False, f"raised {type(ex).__name__}: {ex}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
