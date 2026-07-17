"""Amendment 2 arm for the LR scale-grid prereg (exp2 reports/lr_scale_grid_prereg.md):
`secret_sustain` = secret word + imbue-into-every-token, the cross cell nobody generated (E5
tested storage-without-use, E2 tested sustain-of-a-feeling; this instructs per-token USE of a
symbol).

`primers.py` (exp3 prereg) and `primers_v2.py` (confound-closing prereg) are FROZEN artifacts --
this module extends WITHOUT modifying either, exactly as primers_v2 extended primers.

Composition (registered): the frozen secret-word sentence ("The secret word is {concept}.") plus
E2's piloted WINNING sustain template -- the s1 wording (confound_closing_verdicts.md: "Pilot
winner: s1", acceptance 0.81, retention gate passed) -- with THE SINGLE SUBSTITUTION
"this feeling" -> "the secret word". The suffix is BYTE-DERIVED from the frozen
primers_v2.SUSTAIN_SUFFIXES["s1"] (never retyped), space-joined like the frozen sustained arms,
so the marginal contrast vs the secret_word arm isolates exactly the sustain instruction. The
suffix stays concept-TOKEN-free (it refers to "the secret word", as s1 referred to "this
feeling"); the concept token appears only where the secret_word arm already put it. The
strength-0 baseline substitutes the arm's own neutral filler (primers.SECRET_WORD_NEUTRAL,
"paper"), matching the secret_word arm's registered s0 design.

NO blind-judge gauge for this arm (registered in Amendment 2: there is no persona to evoke; the
manipulation is trivially present in context) -- compose_gauge_system exists only for drop-in
interface parity.
"""
import primers as P
import primers_v2 as P2

CONCEPTS = P.CONCEPTS
GAUGE_PROBE = P.GAUGE_PROBE   # re-export: collect_induction's evoked-arm gauge reads P.GAUGE_PROBE

# The single substitution, byte-derived from the frozen s1 (asserted, so a drift in the frozen
# artifact -- or a retyped suffix -- fails at import, never silently changes the manipulation).
assert P2.SUSTAIN_SUFFIXES["s1"].count("this feeling") == 1, \
    "frozen s1 suffix drifted: 'this feeling' must appear exactly once for the substitution"
SECRET_SUSTAIN_SUFFIX = P2.SUSTAIN_SUFFIXES["s1"].replace("this feeling", "the secret word")

NEW_ARMS = ("secret_sustain",)


def _text(concept, arm):
    if arm != "secret_sustain":
        raise ValueError(f"unknown v3 arm {arm!r}")
    word = P.SECRET_WORD_NEUTRAL if concept is None else concept
    # space join = the frozen sustained arms' join (primers_v2._text: persona + " " + suffix)
    return P.SECRET_WORD_TEMPLATE.format(word=word) + " " + SECRET_SUSTAIN_SUFFIX


def compose_system(concept, strong_system, arm="evoked"):
    """Drop-in superset of primers_v2.compose_system: the secret_sustain arm is handled here,
    everything else delegates to the frozen chain (byte-identical output for every pre-B15 arm,
    pinned by test_lr_grid_secret P3)."""
    if arm in NEW_ARMS:
        return _text(concept, arm) + "\n\n" + strong_system
    return P2.compose_system(concept, strong_system, arm=arm)


def compose_gauge_system(concept, arm="evoked"):
    if arm in NEW_ARMS:
        return _text(concept, arm)
    return P2.compose_gauge_system(concept, arm=arm)
