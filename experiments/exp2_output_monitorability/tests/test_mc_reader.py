"""RED-first unit tests for the MC-letter elicited reader (src/mc_reader.py +
analysis/mc_offline.py). No model, no GPU -- stub tokenizers and synthetic logits.
  Run: .venv/bin/python experiments/exp2_output_monitorability/tests/test_mc_reader.py

Prereg: reports/mc_reader_prereg.md.

C1  MC construction: letter options listed in the Latin-square ordering, single-letter answer
    instruction; the forced-answer string is '\nAnswer: (' and the read position follows '('.
L1  Latin-square balance: 12 cyclic orderings; each concept appears in each of the 12 letter
    slots exactly once, and each letter carries each concept exactly once. Ordering 0's letter
    (a) = concepts[0].
L2  ordering -> concept<->letter binding round-trips: the offline averager can invert each
    ordering's letter->concept map to place a letter logprob back on its concept.
B1  elicited-MC chat splice: head + stream ids VERBATIM + tail (elicitation user turn w/ the MC
    list + assistant header), prefix-property construction; passive-MC splice omits the elicit
    wording but carries the same MC list.
B2  prefix-property violation -> build FATALs.
E1  trailing-eos strip (crash-class guard b): both MC builders strip AT MOST ONE trailing eos
    before splicing; spliced ids token-exact to a real conversation render for eos/non-eos.
A1  letter answer-token ids: the 12 single letters after '(' are distinct single tokens; a
    tokenizer that makes a letter multi-token or collides two letters raises.
A2  letter-logprob arithmetic vs a reference: read at the forced-answer position picks each
    letter's own token logprob; mass-on-12-letters = sum of the 12 renormalization inputs.
F1  forced-answer splice: build appends '\nAnswer: (' and the read index is the last position.
T1  truncation flag: a CoT that reaches the cap without eos -> truncated True; one ending in eos
    -> False.
G1  tokenizer-compat gate reused from elicit_reader: missing capture first_ids FATAL, mismatch
    FATAL, match passes (cross-family reader guard).
R1  offline held-out-third temperature calibration (LR parity): delta scores -> bits ~ log2(12);
    uniform -> ~0; tau grid + stratified split reused from lr_reader_offline semantics.
R2  Latin-square averaging: per-ordering answer logprobs placed back on concepts and averaged;
    a concept fixed high across all orderings -> that concept wins.
D1  CoT quality: repetition score detects a looped token 3-gram; concept-mention flag.
M1  done-marker collision guard: 'MC_DONE' never appears in src/mc_reader.py print literals; the
    class-level marker guard registers the mc box markers.
H1  harness/run_mc.py: disk floors at 56GB and min-vram 40000 when a 14b reader rides; sub-14b
    tier unchanged; the cross-family qwen3-1.7b rides the 3090 tier (no 14b floor).
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3_induction_and_scale"))

import config as C              # noqa: E402
import primers_v2 as P          # noqa: E402

checks = []
def check(name, cond, note=""):
    checks.append((name, bool(cond), note))


try:
    import mc_reader as MR       # src/mc_reader.py (GPU module; pure fns imported CPU-side)
except Exception as e:
    MR = None
    check("import src/mc_reader.py", False, f"{type(e).__name__}: {e}")

AN = None
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mc_offline",
        os.path.join(REPO, "experiments", "exp2_output_monitorability", "analysis",
                     "mc_offline.py"))
    AN = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AN)
except Exception as e:
    AN = None
    check("import analysis/mc_offline.py", False, f"{type(e).__name__}: {e}")


# ------------------------------------------------------------------ stub tokenizers
class WordTok:
    """Qwen-ish chat template rendered to text, word-level incremental vocab. Prefix-stable.
    Splits '(' and letters so '(a' -> tokens '(', 'a' (single-letter tokens after '(')."""
    def __init__(self):
        self.vocab = {}
        self.eos_token_id = None

    def _split(self, text):
        for m in ("(", ")", "\n"):
            text = text.replace(m, f" {m} ")
        return [t for t in text.split(" ") if t != ""]

    def _ids(self, text):
        return [self.vocab.setdefault(t, len(self.vocab) + 10) for t in self._split(text)]

    def render(self, msgs, add_generation_prompt):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def apply_chat_template(self, msgs, add_generation_prompt=True, return_tensors="pt", **kw):
        return torch.tensor([self._ids(self.render(msgs, add_generation_prompt))])

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = self._ids(text)
        return r

    def decode(self, ids, **kw):
        inv = {v: k for k, v in self.vocab.items()}
        return " ".join(inv.get(int(i), "?") for i in ids)


class EosTok(WordTok):
    """WordTok with an eos id where <|im_start|>/<|im_end|> tokenize standalone."""
    def __init__(self):
        super().__init__()
        self.eos_token_id = self._ids("<|im_end|>")[0]

    def _split(self, text):
        for m in ("<|im_start|>", "<|im_end|>"):
            text = text.replace(m, f" {m} ")
        return super()._split(text)


class BrokenTok(EosTok):
    """Violates the prefix property: prepends a marker token only for >2-message lists."""
    def apply_chat_template(self, msgs, add_generation_prompt=True, return_tensors="pt", **kw):
        ids = super().apply_chat_template(msgs, add_generation_prompt, return_tensors, **kw)
        if len(msgs) > 2:
            return torch.cat([torch.tensor([[7]]), ids], dim=1)
        return ids


class MultiLetterTok(WordTok):
    """A letter after '(' tokenizes to >1 token (breaks the single-letter answer read)."""
    def _ids(self, text):
        out = []
        for t in self._split(text):
            if len(t) == 1 and t.isalpha():
                out.append(self.vocab.setdefault(t, len(self.vocab) + 10))
                out.append(self.vocab.setdefault(t + "_x", len(self.vocab) + 10))
            else:
                out.append(self.vocab.setdefault(t, len(self.vocab) + 10))
        return out


class MergeParenTok:
    """Mimics REAL Qwen BPE: message-INITIAL '(a' merges to ONE token distinct from the bare letter,
    but the forced-answer continuation '\\nAnswer: (' + letter emits the BARE letter token (as Qwen
    does: '\\nAnswer: (' ends in ' (' and the next token spelling the answer is a bare 'a'/'b'/...).
    letter_token_ids MUST return the bare-letter continuation ids (what the model predicts at the
    read position), NOT the merged '(a' ids. Regression guard for the message-mid/-initial mismatch."""
    def __init__(self):
        self.eos_token_id = None
        # bare letters a..l -> 64..75 (Qwen's real ids); merged '(x' -> a disjoint high id
        self.bare = {chr(ord("a") + i): 64 + i for i in range(12)}
        self.merged = {chr(ord("a") + i): 900 + i for i in range(12)}

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = self._ids(text)
        return r

    def _ids(self, text):
        # forced-answer prefix ends in ' (' (id 320); a letter right after ' (' is the BARE letter
        if text.startswith("\nAnswer: ("):
            ids = [198, 16141, 25, 320]
            for ch in text[len("\nAnswer: ("):]:
                ids.append(self.bare.get(ch, ord(ch)))
            return ids
        # message-initial '(x' merges to one token (the trap the old code fell into)
        if len(text) == 2 and text[0] == "(" and text[1] in self.merged:
            return [self.merged[text[1]]]
        if text == "(":
            return [7]
        return [ord(c) for c in text]


# ------------------------------------------------------------------ L1: Latin-square balance
if MR is not None:
    try:
        orderings = MR.latin_square_orderings(C.COVERT_CONCEPTS)
        n = len(C.COVERT_CONCEPTS)
        check("L1 twelve orderings", len(orderings) == n == 12)
        # each ordering is a permutation of the 12 concepts (the MC list order = letters a..l)
        ok_perm = all(sorted(o) == sorted(C.COVERT_CONCEPTS) for o in orderings)
        check("L1 each ordering is a permutation of the 12 concepts", ok_perm)
        # slot balance: concept c appears in each letter slot exactly once across orderings
        slots = {c: [] for c in C.COVERT_CONCEPTS}
        for o in orderings:
            for slot, c in enumerate(o):
                slots[c].append(slot)
        check("L1 each concept in each letter slot exactly once (cyclic Latin square)",
              all(sorted(v) == list(range(n)) for v in slots.values()))
        # letter balance: letter slot j carries each concept exactly once
        by_slot = list(zip(*orderings))
        check("L1 each letter slot carries each concept exactly once",
              all(sorted(col) == sorted(C.COVERT_CONCEPTS) for col in by_slot))
    except Exception as e:
        check("L1 latin_square_orderings", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- L2: ordering<->concept map
    try:
        orderings = MR.latin_square_orderings(C.COVERT_CONCEPTS)
        # letter_to_concept(ordering) maps letter index -> concept; must invert the list order
        for o in orderings:
            l2c = MR.letter_to_concept(o)
            check("L2 letter index i -> concept == ordering[i]",
                  all(l2c[i] == o[i] for i in range(len(o))))
            break
    except Exception as e:
        check("L2 letter_to_concept", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- C1: MC list + forced answer
    try:
        order0 = MR.latin_square_orderings(C.COVERT_CONCEPTS)[0]
        msg = MR.mc_message("injected", "elicited", order0)
        # letters a) .. l) present, in ordering sequence, each with its concept
        lines_ok = all(f"({chr(ord('a') + i)}) {order0[i]}" in msg for i in range(12))
        check("C1 MC list lists 12 lettered options in ordering sequence", lines_ok, msg[:120])
        check("C1 elicited-MC carries the elicitation wording",
              "figure out what the concept was" in msg or "previous turn was prefilled" in msg,
              msg[:120])
        msg_p = MR.mc_message("injected", "passive", order0)
        check("C1 passive-MC carries the MC list but NOT the elicitation prefilled-wording",
              all(f"({chr(ord('a') + i)}) {order0[i]}" in msg_p for i in range(12))
              and "previous turn was prefilled" not in msg_p, msg_p[:120])
        check("C1 forced-answer string is '\\nAnswer: ('",
              MR.FORCED_ANSWER == "\nAnswer: (", repr(MR.FORCED_ANSWER))
    except Exception as e:
        check("C1 mc_message", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- A1: letter answer-token ids
    try:
        tok = WordTok()
        lids = MR.letter_token_ids(tok, 12)
        check("A1 12 distinct single-letter answer-token ids after '('",
              len(lids) == 12 and len(set(lids)) == 12)
    except Exception as e:
        check("A1 letter_token_ids", False, f"raised {type(e).__name__}: {e}")
    try:
        MR.letter_token_ids(MultiLetterTok(), 12)                # a letter is multi-token
        check("A1 multi-token letter raises", False, "no exception")
    except Exception:
        check("A1 multi-token letter raises", True)
    # REGRESSION (real-Qwen mismatch): the read is at the last token of FORCED_ANSWER (' ('), so the
    # letter id MUST be the bare-letter continuation token, NOT the message-initial merged '(a' token.
    try:
        lids = MR.letter_token_ids(MergeParenTok(), 12)
        check("A1 letter ids are the forced-answer CONTINUATION (bare letters 64..75), not merged '(a'",
              lids == list(range(64, 76)), f"got {lids}")
    except Exception as e:
        check("A1 forced-answer-continuation letter ids", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- B1: elicited-MC splice
    try:
        tok = WordTok()
        order0 = MR.latin_square_orderings(C.COVERT_CONCEPTS)[0]
        stream = torch.tensor([90001, 90002, 90003])            # ids outside the stub vocab
        ids = MR.build_mc_ids(tok, "injected", "elicited", order0, stream, device="cpu")
        import common as K
        head = K.chat_ids(tok, C.GEN_PROMPT, system=C.STRONG_SYSTEM, device="cpu")
        H = head.shape[1]
        check("B1 elicited-MC ids start with head", ids[0, :H].tolist() == head[0].tolist())
        check("B1 stream tokens spliced VERBATIM at the assistant position",
              ids[0, H:H + 3].tolist() == [90001, 90002, 90003])
        tail = tok.decode(ids[0, H + 3:].tolist())
        check("B1 tail contains the MC list + assistant header",
              "(" in tail and "a" in tail and "assistant" in tail, tail[-60:])
    except Exception as e:
        check("B1 build_mc_ids (elicited)", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- B2: prefix violation FATALs
    try:
        order0 = MR.latin_square_orderings(C.COVERT_CONCEPTS)[0]
        MR.build_mc_ids(BrokenTok(), "injected", "elicited", order0, torch.tensor([90001]),
                        device="cpu")
        check("B2 prefix-property violation raises", False, "no exception")
    except Exception:
        check("B2 prefix-property violation raises", True)

    # -------------------------------------------------------------- E1: trailing-eos strip
    try:
        tok = EosTok()
        order0 = MR.latin_square_orderings(C.COVERT_CONCEPTS)[0]
        content = "gleaming rivulets hum"
        body = tok(content).input_ids
        ids = MR.build_mc_ids(tok, "injected", "elicited", order0, body + [tok.eos_token_id],
                              device="cpu")
        real = tok.apply_chat_template(
            [{"role": "system", "content": C.STRONG_SYSTEM},
             {"role": "user", "content": C.GEN_PROMPT},
             {"role": "assistant", "content": content},
             {"role": "user", "content": MR.mc_message("injected", "elicited", order0)}],
            add_generation_prompt=True)
        check("E1 eos-terminated MC splice token-exact to a real conversation render",
              ids[0].tolist() == real[0].tolist(),
              f"got tail={ids[0].tolist()[-10:]} real tail={real[0].tolist()[-10:]}")
        ids_plain = MR.build_mc_ids(tok, "injected", "elicited", order0, body, device="cpu")
        check("E1 non-eos MC splice == the eos-stripped splice",
              ids_plain[0].tolist() == ids[0].tolist())
    except Exception as e:
        check("E1 eos-strip splice", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- F1: forced-answer splice
    try:
        tok = WordTok()
        base = torch.tensor([[11, 12, 13]])
        spliced = MR.append_forced_answer(tok, base, device="cpu")
        forced = tok(MR.FORCED_ANSWER, add_special_tokens=False).input_ids
        check("F1 forced-answer appended verbatim, read at last position",
              spliced[0, :3].tolist() == [11, 12, 13]
              and spliced[0, 3:].tolist() == list(forced))
    except Exception as e:
        check("F1 append_forced_answer", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- A2: letter-logprob read
    try:
        # 3-letter toy: logits at the answer position, read each letter's token logprob
        letter_ids = [3, 5, 8]
        lp = torch.full((1, 10), -50.0)
        lp[0, 3] = np.log(0.5)
        lp[0, 5] = np.log(0.25)
        lp[0, 8] = np.log(0.125)
        vals = MR.read_letter_logprobs(lp, letter_ids)          # [1, 3]
        check("A2 read_letter_logprobs picks each letter's own token logprob",
              abs(float(vals[0, 0]) - np.log(0.5)) < 1e-5
              and abs(float(vals[0, 2]) - np.log(0.125)) < 1e-5, f"{vals.tolist()}")
        mass = MR.letter_mass(lp, letter_ids)                   # sum prob on the letters
        check("A2 letter_mass = total probability on the option letters",
              abs(float(mass[0]) - 0.875) < 1e-5, f"mass={float(mass[0])}")
    except Exception as e:
        check("A2 read_letter_logprobs/letter_mass", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- T1: truncation flag
    try:
        eos = 99
        # generated row that hit eos within the cap -> not truncated
        row_eos = torch.tensor([1, 2, eos, 7])
        check("T1 CoT ending in eos within cap -> truncated False",
              MR.is_truncated(row_eos, eos_id=eos, cap=8) is False)
        # generated row filling the cap without eos -> truncated
        row_full = torch.arange(8)
        check("T1 CoT reaching cap without eos -> truncated True",
              MR.is_truncated(row_full, eos_id=eos, cap=8) is True)
    except Exception as e:
        check("T1 is_truncated", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- G1: first_ids gate reuse
    try:
        MR.assert_first_ids(list(range(12)), list(range(12)))
        check("G1 exact first_ids match passes", True)
    except Exception as e:
        check("G1 exact first_ids match passes", False, f"raised {type(e).__name__}: {e}")
    for label, args in [("MISSING", (list(range(12)), [])),
                        ("mismatch", (list(range(12)), list(range(1, 13))))]:
        try:
            MR.assert_first_ids(*args)
            check(f"G1 {label} first_ids is FATAL", False, "no exception")
        except Exception:
            check(f"G1 {label} first_ids is FATAL", True)

# ------------------------------------------------------------------ R1/R2/D1: offline
if AN is not None:
    try:
        n, Kc = 24, 12
        y = np.arange(n) % Kc
        S_delta = np.full((n, Kc), -40.0)
        S_delta[np.arange(n), y] = 0.0
        bits, top1 = AN.calibrated_bits_top1(S_delta, y)        # held-out-third tau (LR parity)
        check("R1 delta scores -> bits ~ log2(12), top-1 = 1",
              bits > np.log2(12) - 0.3 and top1 == 1.0, f"bits={bits} top1={top1}")
        bits_u, _ = AN.calibrated_bits_top1(np.zeros((n, Kc)), y)
        check("R1 uniform scores -> bits ~ 0", abs(bits_u) < 0.1, f"bits={bits_u}")
    except Exception as e:
        check("R1 calibrated_bits_top1", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- R2: Latin-square averaging
    try:
        concepts = list(C.COVERT_CONCEPTS)
        orderings = [MR.letter_to_concept(o) for o in MR.latin_square_orderings(concepts)] \
            if MR is not None else None
        # synthesize per-ordering letter logprobs where the TRUE concept letter is always high.
        # shape [n_orderings, n_streams, 12 letters]; place back on concepts + average.
        if MR is not None:
            los = MR.latin_square_orderings(concepts)
            n_streams = 3
            true_idx = [0, 5, 11]
            per_ord = np.full((12, n_streams, 12), -10.0)
            for oi, order in enumerate(los):
                for si, ti in enumerate(true_idx):
                    # letter slot of the true concept in this ordering
                    slot = order.index(concepts[ti])
                    per_ord[oi, si, slot] = 0.0
            S = AN.latin_average(per_ord, los, concepts)        # [n_streams, 12 concepts]
            check("R2 latin_average places letters back on concepts (true concept wins)",
                  [int(S[si].argmax()) for si in range(n_streams)] == true_idx,
                  f"argmax={[int(S[si].argmax()) for si in range(n_streams)]}")
    except Exception as e:
        check("R2 latin_average", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- D1: CoT quality
    try:
        looped = "ab ab ab ab ab ab ab ab"
        clean = "the stream feels like cold water and distance from others"
        rl = AN.repetition_score(looped)
        rc = AN.repetition_score(clean)
        check("D1 repetition_score high on a looped CoT, low on varied text",
              rl > 0.5 and rc < 0.5, f"looped={rl} clean={rc}")
        check("D1 concept-mention flag detects a leaked concept word",
              AN.mentions_concept("i think it is loneliness honestly", C.COVERT_CONCEPTS)
              and not AN.mentions_concept("qx zt fjm", C.COVERT_CONCEPTS))
    except Exception as e:
        check("D1 CoT quality", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- AM1: char surface arbitration
    # Amendment 1 B1: char n-gram is the PRIMARY surface-matching discriminant. mc_offline must
    # extract the injected char bits from full_stream_convergence.json and score the MC-vs-char
    # delta (the decisive surface signature is MC bits ~ char bits).
    try:
        char_stub = {"analyses": {"convergence_injected_1p5b": {"readers": {"char": {
            "full": {"mean": 2.30, "sd": 0.1}}}}}}
        cb = AN.char_injected_bits(char_stub)
        check("AM1 char_injected_bits reads the injected char full-budget bits", abs(cb - 2.30) < 1e-9,
              f"got {cb}")
        d = AN.mc_vs_char_delta(0.55, char_stub)
        check("AM1 mc_vs_char_delta = MC bits - char bits (surface arbitration)",
              abs(d["delta"] - (0.55 - 2.30)) < 1e-9 and d["char_bits"] == 2.30, f"got {d}")
        check("AM1 char_injected_bits missing json -> None (not crash)",
              AN.char_injected_bits(None) is None and AN.char_injected_bits({}) is None)
    except Exception as e:
        check("AM1 char surface arbitration", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- AM2: legacy continuation leak
    # Amendment 1 B3(b): the legacy '; secret word:' passive continuation numbers (from
    # elicited_report_results.json 'passive' bits) ride as a SEPARATE raw-unprompted-leak row --
    # open-vocab continuation currency, explicitly NOT subtracted into any MC/workspace tax.
    try:
        elic_stub = {"readers": {"qwen2.5-1.5b": {
            "injected": {"passive": {"bits": -1.40, "top1": 0.167, "n": 204}},
            "evoked": {"passive": {"bits": -2.50, "top1": 0.088, "n": 204}}}}}
        blk = AN.legacy_leak_block(elic_stub)
        row = blk["rows"]["qwen2.5-1.5b"]
        check("AM2 legacy_leak_block carries injected+evoked passive continuation bits",
              abs(row["injected"]["bits"] - (-1.40)) < 1e-9
              and abs(row["evoked"]["bits"] - (-2.50)) < 1e-9, f"got {row}")
        check("AM2 legacy leak block is flagged different-currency / not-subtracted",
              "not subtracted" in blk["note"].lower()
              and "continuation" in blk["note"].lower(), blk["note"][:80])
        check("AM2 legacy_leak_block missing json -> pending string, not crash",
              isinstance(AN.legacy_leak_block(None), str))
    except Exception as e:
        check("AM2 legacy continuation leak", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- AM3: strata calibration status
    # Amendment 1 B2: a low-n stratum scored with raw tau=1 (no held-out calibration) must be
    # FLAGGED so it is never compared across scales as if calibrated.
    try:
        # n=4 (< 6) -> raw tau=1 path -> uncalibrated
        y_low = np.array([0, 1, 2, 3])
        S_low = np.zeros((4, 12))
        st_low = AN.calib_status(S_low, y_low)
        check("AM3 low-n stratum flagged uncalibrated (raw tau=1)", st_low == "uncalibrated_raw_tau",
              f"got {st_low}")
        y_ok = np.arange(24) % 12
        S_ok = np.zeros((24, 12))
        st_ok = AN.calib_status(S_ok, y_ok)
        check("AM3 sufficient-n stratum flagged calibrated", st_ok == "calibrated", f"got {st_ok}")
    except Exception as e:
        check("AM3 strata calibration status", False, f"raised {type(e).__name__}: {e}")

    # -------------------------------------------------------------- AM4: letter-position residual bias
    # Amendment 1 SHOULD-FIX: on the label-free evoked_s0 set, the per-letter argmax rate must be
    # ~uniform; flag any letter whose argmax rate > 2x(1/12).
    try:
        # per_ord [n_ord, n_streams, 12]: skew every ordering's argmax to letter 0 -> biased.
        per_ord_biased = np.full((12, 20, 12), -10.0)
        per_ord_biased[:, :, 0] = 0.0
        lb = AN.letter_position_bias(per_ord_biased)
        check("AM4 letter_position_bias flags a letter argmax rate > 2x uniform",
              lb["flagged"] and lb["max_rate"] > 2.0 / 12.0, f"got {lb}")
        # uniform-ish: rotate the high letter per ordering so argmax spreads across letters
        per_ord_flat = np.full((12, 12, 12), -10.0)
        for oi in range(12):
            for si in range(12):
                per_ord_flat[oi, si, (si) % 12] = 0.0
        lb2 = AN.letter_position_bias(per_ord_flat)
        check("AM4 letter_position_bias does not flag a uniform argmax spread",
              not lb2["flagged"], f"got {lb2}")
    except Exception as e:
        check("AM4 letter-position residual bias", False, f"raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------ M1: done-marker collision
with open(os.path.join(REPO, "src", "mc_reader.py")) as f:
    check("M1 src/mc_reader.py never contains the done-marker substring 'MC_DONE'",
          "MC_DONE" not in f.read())

# marker guard registers the mc box markers + still passes over all reader modules. The guard
# module calls sys.exit() at import (it is a standalone test), so parse its MODULES via AST and
# run it as a SUBPROCESS to read the pass/fail (never exec it in-process -- that would sys.exit us).
try:
    import ast as _ast
    import subprocess as _sp
    _mg_path = os.path.join(HERE, "test_marker_guard.py")
    _tree = _ast.parse(open(_mg_path).read())
    _mods = {}
    for _n in _ast.walk(_tree):
        if isinstance(_n, _ast.Assign) and any(
                getattr(t, "id", None) == "MODULES" for t in _n.targets):
            _mods = _ast.literal_eval(_n.value)
    check("M1 marker guard registers mc_reader.py with the MC box markers",
          _mods.get("mc_reader.py") == ("MC_READY", "MC_DONE"),
          f"MODULES[mc_reader.py]={_mods.get('mc_reader.py')}")
    _r = _sp.run([sys.executable, _mg_path], capture_output=True, text=True)
    check("M1 marker guard passes over every reader module (mc included)",
          _r.returncode == 0, _r.stdout[-300:] + _r.stderr[-300:])
except Exception as e:
    check("M1 marker guard integration", False, f"raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------ H1: 14b disk/vram floor
try:
    import types
    import importlib.util as _ilu
    for _name in ("labkit", "experimentfactory"):
        sys.modules.setdefault(_name, types.ModuleType(_name))
    _ef = sys.modules["experimentfactory"]
    for _attr in ("authorized_run", "Spec", "GateBlocked", "jsonl_recorder", "default_gate_log",
                  "evaluate", "facts_from_spec", "EXPERIMENT_SPEND_POLICY"):
        if not hasattr(_ef, _attr):
            setattr(_ef, _attr, lambda *a, **k: None)
    _spec = _ilu.spec_from_file_location("run_mc", os.path.join(REPO, "harness", "run_mc.py"))
    RM = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(RM)
    check("H1 run_mc disk floors at 56GB when a 14b reader rides",
          RM.disk_for(["14b"]) >= 56.0, f"disk={RM.disk_for(['14b'])}GB")
    check("H1 run_mc min-vram >= 40000 when a 14b reader rides",
          RM.vram_for(["14b"]) >= 40000, f"vram={RM.vram_for(['14b'])}")
    check("H1 sub-14b tier disk unchanged (10 + weights + 4)",
          abs(RM.disk_for(["1.5b", "3b", "7b"]) - 40.0) < 1e-9,
          f"disk={RM.disk_for(['1.5b', '3b', '7b'])}")
    check("H1 cross-family qwen3-1.7b rides the 3090 tier (no 14b floor)",
          RM.disk_for(["1.7b"]) < 56.0 and RM.vram_for(["1.7b"]) < 40000,
          f"disk={RM.disk_for(['1.7b'])} vram={RM.vram_for(['1.7b'])}")
except Exception as e:
    check("H1 run_mc disk/vram floor", False, f"raised {type(e).__name__}: {e}")

for name, ok, note in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {note}" if note and not ok else ""))
sys.exit(0 if all(ok for _, ok, _ in checks) and checks else 1)
