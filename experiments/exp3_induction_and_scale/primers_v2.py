"""Arms for the confound-closing prereg (E2 sustained persona, E5 maintained secret).

`primers.py` is a FROZEN exp3 pre-registration artifact -- this module extends WITHOUT modifying it.

Design: a `sustained_*` arm = the frozen EVOKED persona (or EVOKED_ALT for the `_alt_*` arms) plus a
PER-TOKEN MAINTENANCE SUFFIX, so the marginal contrast (sustained - evoked) isolates exactly the sustain
instruction, and the paraphrase-transfer test reuses the frozen alt personas. Three candidate suffixes
(s1/s2/s3) go through the prereg's wording-qualification pilot (acceptance gate + E4 trajectory gate);
only the qualified winner runs at full scale. Suffixes are CONCEPT-WORD-FREE by construction (they refer
only to "this feeling").

E5 `maintained_secret` = the frozen secret-word sentence plus an explicit maintenance demand (no report
is ever collected -- the instruction only creates the processing requirement).
"""
import primers as P

CONCEPTS = P.CONCEPTS

SUSTAIN_SUFFIXES = {
    "s1": ("Hold this feeling with every keystroke: let each single character you type carry it, from "
           "the first character to the last, never letting it fade."),
    "s2": ("Do not let this feeling drift as you type. Re-summon it before every character you emit; "
           "every keystroke should be soaked in it, all the way to the end."),
    "s3": ("This feeling is permanent and total. It saturates every motion of your fingers; every single "
           "character you produce is another expression of it. Keep it at full strength for the entire "
           "output, on every character, without exception."),
}

MAINTAINED_TEMPLATE = ("The secret word is {word}. Remember it carefully the whole time -- you will be "
                       "asked to report it exactly, once this task is complete.")

NEW_ARMS = tuple(f"sustained_{k}" for k in SUSTAIN_SUFFIXES) + \
           tuple(f"sustained_alt_{k}" for k in SUSTAIN_SUFFIXES) + ("maintained_secret",)


def _text(concept, arm):
    if arm == "maintained_secret":
        word = P.SECRET_WORD_NEUTRAL if concept is None else concept
        return MAINTAINED_TEMPLATE.format(word=word)
    if arm.startswith("sustained"):
        key = arm.rsplit("_", 1)[1]
        base = (P.EVOKED_ALT if "_alt_" in arm else P.EVOKED)
        persona = P.NEUTRAL if concept is None else base[concept]
        return persona + " " + SUSTAIN_SUFFIXES[key]
    raise ValueError(f"unknown v2 arm {arm!r}")


def compose_system(concept, strong_system, arm="evoked"):
    """Drop-in superset of primers.compose_system: v2 arms handled here, everything else delegates to
    the frozen module (identical output for all pre-registered exp3 arms)."""
    if arm in NEW_ARMS:
        return _text(concept, arm) + "\n\n" + strong_system
    return P.compose_system(concept, strong_system, arm=arm)


def compose_gauge_system(concept, arm="evoked"):
    if arm in NEW_ARMS:
        return _text(concept, arm)
    return P.compose_gauge_system(concept, arm=arm)
