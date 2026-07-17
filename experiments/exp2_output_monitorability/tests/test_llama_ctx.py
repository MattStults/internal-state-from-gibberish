"""RED-first unit tests for scale-grid units B4 + B13 (prereg Amendment 1, A1 ADJUDICATED):
the Llama cross-family context rendering + decoded-stream path. No model, no GPU.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_llama_ctx.py

L1  PRIMARY render = the READER'S OWN chat template: system = persona/neutral text (lr_reader's
    certified context_system), user = GEN_PROMPT, add_generation_prompt -- with date_string
    PINNED (LLAMA_DATE): a drifting template default must NOT change the render between calls.
L2  render-diff assert: numerator (persona) and denominator (neutral) renders differ ONLY in the
    persona text; a template that duplicates the system text trips it.
L3  SECONDARY render = raw text (register-OOD robustness secondary): plain system + GEN_PROMPT
    string, no template; ctx_ids_for routes render= for llama readers and stays certified for
    Qwen readers.
L4  template prefix-stability guard (the Qwen3 lesson): the generation-header context render must
    be an exact token prefix of the full [system, user, assistant=text] render with the
    re-tokenized stream immediately after; a header-retokenizing template trips it.
L5  stream text derivation: the trailing eos is STRIPPED before decoding (at most ONE; a
    non-terminal eos is part of the saved stream and stays).
L6  round-trip policy (B13): per-stream EXCLUSION, counted, never FATAL; kept streams score the
    reader's own re-tokenization of the decoded text (identical ids for numerator and
    denominator by construction).
L7  shard naming: raw-secondary shards carry a _raw suffix, distinct from primary names; the
    prose-control (gate 4) shards parse as reader__prose__control_<ctx>.
L8  main() wiring: renders loop, roundtrip counts persisted in shards, prose control rides for
    llama readers, PROSE_CONTROL covers all 12 concepts.
L9  box bookkeeping: shards_for counts llama raw secondaries + prose shards; primary names stay
    in lockstep with lr_grid.shard_path.
F1  prereg Amendment 4 (2026-07-11): the cross-family readers are tiiuae/Falcon3-{1B,3B,7B}-
    Instruct (the registered ungated fallback; the box 403'd on meta-llama). Every A1/"Llama"
    rule above applies unchanged with "Llama" read as "Falcon3"; the Llama-era function names
    (llama_ctx_ids and friends) are retained and now SERVE the falcon slugs. F1 pins: the falcon
    slugs/hf-ids, family routing, the (system, user, assistant) triple round-tripping the
    prefix-stability guard under the REAL Falcon3-1B tokenizer when cached locally (else a
    pinned-string fixture of the template format <|system|>\\n..<|user|>\\n..<|assistant|>\\n..
    <|endoftext|>), eos = <|endoftext|>, and the date pin being a harmless no-op (Falcon3's
    template injects no date; the kwarg is accepted and ignored under the 4.46.3 pin).
"""
import importlib.util
import inspect
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


import config as C       # noqa: E402
import lr_grid as G      # noqa: E402
import lr_reader as LR   # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LlamaTok:
    """Llama-3-like chat-template stub. Char-ordinal tokenization (exact round-trip); the
    template injects a date line whose DEFAULT drifts (self.default_date) unless the caller pins
    date_string -- exactly the Llama 'Today Date' behavior the amendment pins against."""
    eos_token_id = 128009

    def __init__(self, echo_system_twice=False, retokenizing_header=False, break_texts=()):
        self.default_date = "26 Jul 2024"
        self.echo_system_twice = echo_system_twice
        self.retokenizing_header = retokenizing_header
        self.break_texts = set(break_texts)

    def render(self, msgs, add_generation_prompt, date_string):
        date = date_string if date_string is not None else self.default_date
        out = ""
        for m in msgs:
            if m["role"] == "system":
                sysline = m["content"] * (2 if self.echo_system_twice else 1)
                out += f"<|sys|>Today Date: {date}\n{sysline}<|eot|>"
            elif m["role"] == "user":
                out += f"<|usr|>{m['content']}<|eot|>"
            elif m["role"] == "assistant":
                pad = "~" if self.retokenizing_header else ""
                out += f"<|ast|>{pad}{m['content']}<|eot|>"
        if add_generation_prompt:
            out += "<|ast|>"
        return out

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True,
                            date_string=None, return_tensors=None, **kw):
        text = self.render(msgs, add_generation_prompt, date_string)
        if not tokenize:
            return text
        ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        ids = [ord(c) for c in text]
        if text in self.break_texts:
            ids = ids + [31]                     # decode() will emit an extra char -> round-trip fails
        class _R:                                # noqa: E306
            pass
        r = _R()
        r.input_ids = (torch.tensor([ids], dtype=torch.long)
                       if return_tensors == "pt" else ids)
        return r

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in np.asarray(ids).reshape(-1))


# ================================================================ L1: pinned-date template render
try:
    tok = LlamaTok()
    ids1 = G.llama_ctx_ids(tok, "A", "fear", "cpu")
    tok.default_date = "01 Jan 2031"             # a drifting template default...
    ids2 = G.llama_ctx_ids(tok, "A", "fear", "cpu")
    check("L1 llama_ctx_ids returns [1, T] ids from the reader's OWN template",
          torch.is_tensor(ids1) and ids1.dim() == 2 and ids1.shape[0] == 1)
    check("L1 date_string PINNED: a drifting template default cannot change the render",
          ids1.shape == ids2.shape and bool((ids1 == ids2).all()))
    check("L1 LLAMA_DATE is a pinned module constant used by the render",
          isinstance(getattr(G, "LLAMA_DATE", None), str)
          and G.LLAMA_DATE in tok.render(
              [{"role": "system", "content": "x"}], False, G.LLAMA_DATE))
    text = tok.apply_chat_template(
        [{"role": "system", "content": LR.context_system("A", "fear")},
         {"role": "user", "content": C.GEN_PROMPT}],
        add_generation_prompt=True, tokenize=False, date_string=G.LLAMA_DATE)
    check("L1 render structure: system = persona text, user = GEN_PROMPT, generation header on",
          "".join(chr(int(i)) for i in ids1[0].tolist()) == text)
except Exception as e:
    check("L1 llama template render", False, f"raised {type(e).__name__}: {e}")

# ================================================================ L2: render-diff assert
try:
    G.assert_render_diff(LlamaTok(), "A", "fear")
    check("L2 render-diff passes when renders differ only in the persona text", True)
except Exception as e:
    check("L2 render-diff passes when renders differ only in the persona text", False,
          f"raised {type(e).__name__}: {e}")
try:
    G.assert_render_diff(LlamaTok(echo_system_twice=True), "A", "fear")
    check("L2 a template that duplicates the persona text trips the assert", False,
          "no exception")
except RuntimeError as e:
    bad = ("LRG_READY", "LRG_DONE", "LRG_FATAL", "CUDA error", "CUDA out of memory",
           "Traceback (most recent call last)", "ModuleNotFoundError")
    check("L2 a template that duplicates the persona text trips the assert", True)
    check("L2 failure message is marker/FATAL-substring safe",
          not any(s in str(e) for s in bad), repr(str(e)))
except Exception as e:
    check("L2 a template that duplicates the persona text trips the assert", False,
          f"wrong exception type {type(e).__name__}: {e}")

# ================================================================ L3: raw secondary + routing
try:
    tok = LlamaTok()
    raw = G.llama_ctx_ids(tok, "A", "fear", "cpu", render="raw")
    want = G.RAW_CTX_FMT.format(system=LR.context_system("A", "fear"), user=C.GEN_PROMPT)
    check("L3 raw render = plain persona + GEN_PROMPT text, no template",
          "".join(chr(int(i)) for i in raw[0].tolist()) == want)
    tmpl = G.llama_ctx_ids(tok, "A", "fear", "cpu", render="template")
    check("L3 raw and template renders differ (the registered robustness contrast)",
          raw.shape != tmpl.shape or not bool((raw == tmpl).all()))
    via = G.ctx_ids_for("falcon3-1b", tok, "A", "fear", "cpu", render="raw")
    check("L3 ctx_ids_for routes render= for cross-family readers",
          via.shape == raw.shape and bool((via == raw).all()))

    class QwenRec:
        def apply_chat_template(self, msgs, **kw):
            self.msgs, self.kw = msgs, kw
            return torch.ones((1, 4), dtype=torch.long)
    qr = QwenRec()
    _ = G.ctx_ids_for("qwen2.5-3b", qr, "A", "fear", "cpu")
    check("L3 Qwen readers stay on the certified construction (no date_string kwarg injected)",
          "date_string" not in qr.kw and qr.msgs[0]["content"] == LR.context_system("A", "fear"))
except Exception as e:
    check("L3 raw secondary / routing", False, f"raised {type(e).__name__}: {e}")

# ================================================================ L4: prefix-stability guard
try:
    G.assert_llama_prefix_stable(LlamaTok())
    check("L4 prefix-stability guard passes on a stable template", True)
except Exception as e:
    check("L4 prefix-stability guard passes on a stable template", False,
          f"raised {type(e).__name__}: {e}")
try:
    G.assert_llama_prefix_stable(LlamaTok(retokenizing_header=True))
    check("L4 a header-retokenizing template trips the guard", False, "no exception")
except RuntimeError:
    check("L4 a header-retokenizing template trips the guard", True)
except Exception as e:
    check("L4 a header-retokenizing template trips the guard", False,
          f"wrong exception type {type(e).__name__}: {e}")

# ================================================================ L5: eos strip before decode
try:
    class SrcTok:
        eos_token_id = 9
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(64 + int(i)) for i in np.asarray(ids).reshape(-1))
    streams = [dict(gidx=0, tokens=[5, 6, 9]),       # terminal eos -> stripped
               dict(gidx=1, tokens=[5, 9, 6]),       # NON-terminal eos -> stays
               dict(gidx=2, tokens=np.array([9, 9]))]  # only the LAST one stripped
    texts = G.llama_stream_texts(streams, SrcTok())
    check("L5 trailing eos stripped before decode; non-terminal eos stays; at most ONE stripped",
          texts == ["EF", "EIF", "I"], f"got {texts}")
except Exception as e:
    check("L5 eos strip before decode", False, f"raised {type(e).__name__}: {e}")

# ================================================================ L6: round-trip exclusion
try:
    texts = ["abc", "def", "ghi"]
    tok = LlamaTok(break_texts=("def",))
    kept, excl = G.llama_roundtrip_split(tok, texts)
    check("L6 round-trip failures are EXCLUDED and COUNTED, never FATAL",
          kept == [0, 2] and excl == 1, f"kept={kept} excl={excl}")

    class Src2:
        eos_token_id = 999999
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(int(i)) for i in np.asarray(ids).reshape(-1))
    raw_streams = [dict(gidx=3, concept="fear", strength=1,
                        tokens=[ord(c) for c in "abc"]),
                   dict(gidx=4, concept="ocean", strength=1,
                        tokens=[ord(c) for c in "def"]),
                   dict(gidx=5, concept="anger", strength=1,
                        tokens=[ord(c) for c in "ghi"])]
    ls, excl2, total = G.llama_streams(raw_streams, tok, Src2())
    check("L6 llama_streams: kept streams carry the READER re-tokenization + text",
          len(ls) == 2 and total == 3 and excl2 == 1
          and ls[0]["tokens"] == [ord(c) for c in "abc"] and ls[0]["gidx"] == 3
          and ls[1]["concept"] == "anger")
except Exception as e:
    check("L6 round-trip exclusion", False, f"raised {type(e).__name__}: {e}")

# ================================================================ L7: shard naming
try:
    from pathlib import Path
    a = G.shard_path(Path("o"), "falcon3-1b", "qwen2.5-3b", "evoked", "A")
    b = G.shard_path(Path("o"), "falcon3-1b", "qwen2.5-3b", "evoked", "A", render="raw")
    check("L7 raw secondary shard = primary name + _raw, distinct",
          a.name != b.name and b.name == a.name.replace(".pt", "_raw.pt"))
    p = G.shard_path(Path("o"), "falcon3-1b", "prose", "control", "A")
    check("L7 prose gate-4 shard parses as reader__prose__control_A",
          p.name == "falcon3-1b__prose__control_A.pt")
    names = {G.shard_path(Path("o"), "falcon3-1b", m, ss, cs, render=r).name
             for m in ("qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b")
             for ss in ("evoked", "evoked_alt") for cs in ("N", "A", "B")
             for r in ("template", "raw")}
    check("L7 36 distinct per-xfam-reader grid shard names (18 primary + 18 raw)",
          len(names) == 36)
except Exception as e:
    check("L7 shard naming", False, f"raised {type(e).__name__}: {e}")

# ================================================================ L8: main wiring + prose set
try:
    msrc = inspect.getsource(G.main)
    check("L8 main loops renders (raw secondary produced alongside primary)",
          '"raw"' in msrc and "render" in msrc)
    check("L8 shards persist the roundtrip exclusion counts (offline >5% void rule input)",
          "roundtrip_excluded" in msrc and "roundtrip_total" in msrc)
    check("L8 prose control rides main for llama readers", "prose" in msrc)
    check("L8 PROSE_CONTROL covers exactly the 12 concepts",
          set(G.PROSE_CONTROL) == set(C.COVERT_CONCEPTS))
    check("L8 prose texts are real English prose (multi-word, non-empty)",
          all(len(t.split()) >= 8 for t in G.PROSE_CONTROL.values()))
    ps = G.prose_streams(LlamaTok())
    check("L8 prose_streams: one labeled stream per concept, reader-tokenized",
          len(ps) == 12 and all(s["concept"] in C.COVERT_CONCEPTS and len(s["tokens"]) > 0
                                for s in ps))
    check("L8 the A1 NotImplementedError seam is GONE (B4/B13 landed)",
          "NotImplementedError" not in inspect.getsource(G.llama_ctx_ids))
except Exception as e:
    check("L8 main wiring", False, f"raised {type(e).__name__}: {e}")

# ================================================================ L9: box bookkeeping
try:
    BOX = load_module("box_lr_grid",
                      os.path.join(REPO, "experiments", "exp2_output_monitorability",
                                   "box_lr_grid.py"))
    sh_q = BOX.shards_for("qwen2.5-3b")
    sh_l = BOX.shards_for("falcon3-1b")
    check("L9 qwen 3b reader: 18 pre-B15 + 12 secret + 3 control-(b) injected diagonal = 33",
          len(sh_q) == 33, f"{len(sh_q)}")
    check("L9 xfam (falcon3) readers: 38 pre-B15 + 12 secret + 12 secret raw = 62",
          len(sh_l) == 62, f"{len(sh_l)}")
    names = {os.path.basename(s) for s in sh_l}
    check("L9 xfam raw + prose names match lr_grid.shard_path exactly",
          "falcon3-1b__qwen2.5-3b__evoked_A_raw.pt" in names
          and "falcon3-1b__qwen2.5-3b__secret_word_SW_raw.pt" in names
          and "falcon3-1b__prose__control_A.pt" in names
          and "falcon3-1b__prose__control_N.pt" in names)
    total = sum(len(BOX.shards_for(r)) for r in BOX.READERS)
    check("L9 full run = 284 shard files (278 + 6 control-(b) injected diagonal at 3B/7B)",
          total == 284, f"{total}")
except Exception as e:
    check("L9 box bookkeeping", False, f"raised {type(e).__name__}: {e}")


# ================================================================ F1 (Amendment 4): Falcon3
# The cross-family readers are tiiuae/Falcon3-{1B,3B,7B}-Instruct (prereg Amendment 4: the
# registered ungated fallback fired after the box's HF 403 on meta-llama). The render/guard code
# is the SAME Llama-era A1 path; F1 pins that it serves the falcon slugs, against the REAL
# Falcon3-1B tokenizer when its files are cached locally, else a pinned-string fixture of the
# template format (verified against the real tokenizer under transformers 4.46.3, 2026-07-11).
class FalconFixtureTok:
    """Pinned fixture of the Falcon3-Instruct chat template (the no-tools branch):
    system -> '<|system|>\\n{content}\\n', user -> '<|user|>\\n{content}\\n',
    assistant (last) -> '<|assistant|>\\n{content}<|endoftext|>' (non-last adds '\\n'),
    generation prompt -> '<|assistant|>\\n'. NO date is injected; unknown kwargs
    (date_string) are accepted and ignored. Char-ordinal ids (exact round-trip)."""
    eos_token = "<|endoftext|>"
    eos_token_id = 11

    def render(self, msgs, add_generation_prompt):
        out = ""
        for i, m in enumerate(msgs):
            if m["role"] == "system":
                out += f"<|system|>\n{m['content']}\n"
            elif m["role"] == "user":
                out += f"<|user|>\n{m['content']}\n"
            elif m["role"] == "assistant":
                out += f"<|assistant|>\n{m['content']}<|endoftext|>"
                if i != len(msgs) - 1:
                    out += "\n"
        if add_generation_prompt:
            out += "<|assistant|>\n"
        return out

    def apply_chat_template(self, msgs, add_generation_prompt=False, tokenize=True,
                            return_tensors=None, **kw):      # date_string lands here, ignored
        text = self.render(msgs, add_generation_prompt)
        if not tokenize:
            return text
        ids = [ord(c) for c in text]
        return torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else ids

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        class _R:                                            # noqa: E306
            pass
        r = _R()
        ids = [ord(c) for c in text]
        r.input_ids = (torch.tensor([ids], dtype=torch.long)
                       if return_tensors == "pt" else ids)
        return r

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in np.asarray(ids).reshape(-1))


def _falcon_tok():
    """The REAL Falcon3-1B tokenizer if cached locally (no download), else the pinned fixture."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("tiiuae/Falcon3-1B-Instruct",
                                             local_files_only=True), "real"
    except Exception:
        return FalconFixtureTok(), "fixture"


try:
    ftok, fsrc = _falcon_tok()
    check("F1 GRID_READERS carries the Amendment 4 falcon slugs -> tiiuae Instruct hf-ids",
          all(s in getattr(G, "GRID_READERS", {})
              for s in ("falcon3-1b", "falcon3-3b", "falcon3-7b"))
          and G.GRID_READERS.get("falcon3-1b") == "tiiuae/Falcon3-1B-Instruct"
          and G.GRID_READERS.get("falcon3-7b") == "tiiuae/Falcon3-7B-Instruct",
          f"readers = {sorted(getattr(G, 'GRID_READERS', {}))}")
    check("F1 family() routes falcon slugs off the qwen (saved-ids) path",
          G.family("falcon3-1b") != "qwen" and G.family("falcon3-7b") != "qwen",
          f"family = {G.family('falcon3-1b')}")
    check(f"F1 falcon eos is <|endoftext|> [{fsrc} tokenizer]",
          getattr(ftok, "eos_token", None) == "<|endoftext|>")
    # the (system, user, assistant) triple round-trips the prefix-stability guard
    G.assert_llama_prefix_stable(ftok)
    check(f"F1 falcon (system,user,assistant) render round-trips the prefix-stability guard "
          f"[{fsrc} tokenizer]", True)
    # date pin = harmless no-op: the kwarg is accepted+ignored and no date lands in the render
    msgs = [{"role": "system", "content": LR.context_system("A", "fear")},
            {"role": "user", "content": C.GEN_PROMPT}]
    t_pin = ftok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                     date_string=G.LLAMA_DATE)
    t_no = ftok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    check("F1 date_string pin is a no-op for falcon (accepted, ignored, no date injected)",
          t_pin == t_no and G.LLAMA_DATE not in t_pin)
    # routing: ctx_ids_for on the falcon slug = the A1 template render, both render modes
    ids_direct = G.llama_ctx_ids(ftok, "A", "fear", "cpu")
    ids_routed = G.ctx_ids_for("falcon3-1b", ftok, "A", "fear", "cpu")
    check("F1 ctx_ids_for('falcon3-1b') routes to the A1 template render",
          ids_routed.shape == ids_direct.shape and bool((ids_routed == ids_direct).all()))
    # render-diff assert + raw secondary run identically on the falcon template
    G.assert_render_diff(ftok, "A", "fear", render="template")
    G.assert_render_diff(ftok, "A", "fear", render="raw")
    check("F1 render-diff assert passes under the falcon template (both renders)", True)
except Exception as e:
    check("F1 falcon template / routing (Amendment 4)", False,
          f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
