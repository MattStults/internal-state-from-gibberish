"""Offline unit test for the auto-tuner (no model, no GPU, no inference).

Drives covert_collect.tune_effmags_from_evaluate with synthetic `evaluate(effmag)->(nameability_nats, clean)`
curves and asserts the OBJECTIVE FIX: the tuner searches for nameability in NATS (higher = stronger), not a
rank proxy, so it can't under-inject by hitting a rank that no longer corresponds to the nats we want.
Cases: (1) targets reachable -> tunes med/strong to the nats band; (2) a target UNREACHABLE while the model
stays capable -> CAP at the max-achievable nats dose (NOT fall back to under-injected base eff_mags -- the
exact failure we are fixing); (3) relative capability floor preserved; (4) truly incapacitated -> fall back.

Run:  INTRO_MODEL=qwen2.5-3b .venv/bin/python tests/test_calibrate_floor.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
os.environ.setdefault("INTRO_MODEL", "qwen2.5-3b")   # any registry slug; we never load it
import config as C                                    # noqa: E402
from covert_collect import tune_effmags_from_evaluate  # noqa: E402

MED, STRONG = C.CAL_TARGET_NATS                       # the nats band we tune to (med, strong)


def synth_nats(base_clean, clean_slope, nat_scale=10.0, nat_half=40.0):
    """A monotone synthetic model. effmag=0 -> (0 nameability lift, baseline clean). For effmag>0:
    nameability nats SATURATE upward (nat_scale*e/(e+nat_half)) and clean decreases linearly -- mirrors the
    real coupling (more injection -> more nameable but more degeneration). nat_scale caps the achievable nats."""
    def ev(e):
        if e <= 0:
            return (0.0, base_clean)
        nats = nat_scale * e / (e + nat_half)
        return (nats, max(0.0, base_clean - clean_slope * e))
    return ev


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return bool(cond)


def case_reaches_targets():
    """High-ceiling, both nats targets reachable while capable -> tunes med/strong onto the band."""
    print(f"case: targets reachable (base_clean=0.89, max nats=10 > strong={STRONG})")
    ev = synth_nats(0.89, 0.001, nat_scale=10.0)
    result, meta = tune_effmags_from_evaluate(ev, lo=10.0, hi0=400.0, targets=C.CAL_TARGET_NATS, iters=12)
    med, strong = result
    nm, cm = ev(med)
    ns, cs = ev(strong)
    return all([
        check("did NOT fall back to base", result != list(C.BASE_EFFMAGS)),
        check(f"floor relative (0.7*0.89={round(0.70*0.89,3)})", meta["floor"] == round(0.70 * 0.89, 3)),
        check(f"med reaches ~{MED} nats (got {nm:.1f})", nm >= MED - 0.6),
        check(f"strong reaches ~{STRONG} nats (got {ns:.1f})", ns >= STRONG - 0.6),
        check("med < strong (distinct, increasing)", med < strong),
        check(f"both stay clean>=floor (med={cm:.2f},strong={cs:.2f})", cm >= meta["floor"] and cs >= meta["floor"]),
        check("not capped (targets met)", not (meta.get("note") and "cap" in meta["note"].lower())),
    ])


def case_caps_when_unreachable():
    """THE FIX: strong nats target is unreachable (model saturates below it) but the model stays capable.
    The tuner must CAP at the max-achievable nats dose, NOT fall back to under-injected base eff_mags."""
    print(f"case: strong target unreachable (max nats=5.5 < strong={STRONG}), model stays capable")
    ev = synth_nats(0.89, 0.0004, nat_scale=5.5)      # nats saturate at 5.5 < STRONG(7); clean stays ok
    result, meta = tune_effmags_from_evaluate(ev, lo=10.0, hi0=400.0, targets=C.CAL_TARGET_NATS, iters=12)
    med, strong = result
    ns, cs = ev(strong)
    nm, _ = ev(med)
    return all([
        check("did NOT fall back to base (the bug we are fixing)", result != list(C.BASE_EFFMAGS)),
        check(f"med still reached ~{MED} nats (got {nm:.1f})", nm >= MED - 0.6),
        check(f"strong CAPPED near the achievable max ~5.5 (got {ns:.1f})", ns >= 5.5 - 0.8),
        check("strong is the strongest dose tried (a high eff_mag, not base)", strong > med and strong > 100),
        check("cap is surfaced in the note", bool(meta.get("note")) and "cap" in meta["note"].lower()),
    ])


def case_floor_relative_lowceiling():
    """Low-ceiling model (base_clean=0.38): floor must be relative (0.7*0.38) so it can still tune."""
    print("case: low-ceiling (base_clean=0.38) -> relative floor")
    ev = synth_nats(0.38, 0.0003, nat_scale=10.0)
    result, meta = tune_effmags_from_evaluate(ev, lo=10.0, hi0=400.0, targets=C.CAL_TARGET_NATS, iters=12)
    floor_expected = max(C.CAPABILITY_MIN_FLOOR, round(C.CAPABILITY_REL_FLOOR * 0.38, 3))
    return all([
        check("did NOT fall back to base", result != list(C.BASE_EFFMAGS)),
        check(f"floor is relative ({meta['floor']} == {floor_expected})", meta["floor"] == floor_expected),
        check("med < strong", result[0] < result[1]),
        check("LOW-baseline caveat noted", bool(meta.get("note")) and "LOW baseline" in meta["note"]),
    ])


def case_both_capped():
    """BOTH nats targets unreachable (model saturates below MED) but the model stays capable -> both pin at
    the SAME max-achievable dose. The distinctness guard must separate them by LOWERING med, NOT by raising
    strong PAST the verified-clean cap (which would push the strong operating point into incapacitation).
    Regression for LOW finding #1 (the 7B-with-low-nats-ceiling regime)."""
    print(f"case: both targets unreachable (max nats=3 < med={MED}), model stays capable")
    ev = synth_nats(0.89, 0.002, nat_scale=3.0)       # nats saturate at ~3 < MED(4); clean caps the dose
    result, meta = tune_effmags_from_evaluate(ev, lo=10.0, hi0=400.0, targets=C.CAL_TARGET_NATS, iters=14)
    med, strong = result
    cap_eff = meta["cap"]["effmag"]
    return all([
        check("did NOT fall back to base", result != list(C.BASE_EFFMAGS)),
        check("med < strong (distinct operating points)", med < strong),
        check(f"strong does NOT exceed the verified-clean cap eff_mag ({strong} <= {cap_eff})", strong <= cap_eff),
        check(f"strong stays clean>=floor (clean={ev(strong)[1]:.2f} >= {meta['floor']})", ev(strong)[1] >= meta["floor"]),
        check("cap surfaced in note", bool(meta.get("note")) and "cap" in meta["note"].lower()),
    ])


def case_incapacitated():
    """Clean below the floor even at minimal injection -> no feasible dose -> fall back to base eff_mags."""
    print("case: incapacitated (base_clean=0.20, steep clean collapse)")
    ev = synth_nats(0.20, 0.05, nat_scale=10.0)
    result, meta = tune_effmags_from_evaluate(ev, lo=10.0, hi0=400.0, targets=C.CAL_TARGET_NATS, iters=12)
    return all([
        check("fell back to BASE_EFFMAGS", result == list(C.BASE_EFFMAGS)),
        check("fallback note present", meta.get("note", "").startswith("fallback")),
    ])


if __name__ == "__main__":
    cases = [case_reaches_targets(), case_caps_when_unreachable(), case_both_capped(),
             case_floor_relative_lowceiling(), case_incapacitated()]
    print(f"\n{'ALL PASS' if all(cases) else 'FAILURES PRESENT'} ({sum(cases)}/{len(cases)} cases)")
    sys.exit(0 if all(cases) else 1)
