"""Score the gauge-trajectory shards (task #20: Story A vs Story B) -> reports/gauge_trajectory_verdict.json.

Same z(t) math as the E4 scorer (e4_trajectory_verdict.json), with one change of null: the pool is the
NEUTRAL GAUGE records' own-minus-other at the same cut (the s0 gibberish pool would standardize gauge
free-association projections against a different regime's variance).

Per record and cut: diff = proj[own] - mean(proj[other]); null mu/sd per (cut) from every neutral record
x every candidate concept; z = (diff - mu) / sd. Windows: early t<=8, mid t in {16,32}, late t in {64,127};
per-record window z = mean of its per-cut z, so the window SEM respects the record-level dependence.

VERDICT (pre-stated in task #20): STORY A if gauge z is clearly elevated (mean z > 3x SEM and > 0.5 sigma)
at early/mid cuts -- the persona state IS installed in free behavior, so the E4 anti-word null (~0.05 sigma)
means DISPLACEMENT. STORY B if gauge z ~= floor too -- the persona never writes the injected-vector
direction (v_read != v_write made real). Mixed/marginal -> ambiguous.

Run: .venv/bin/python experiments/exp2_output_monitorability/analysis/score_gauge.py
"""
import datetime
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TRAJ = os.path.join(REPO, "runs", "gauge_box", "gauge_traj", "trajectory")
OUT = os.path.join(REPO, "experiments", "exp2_output_monitorability", "reports",
                   "gauge_trajectory_verdict.json")
WINDOWS = {"early": (2, 4, 8), "mid": (16, 32), "late": (64, 127)}
ELEVATED_SIGMA = 0.5     # task #20 threshold: clearly elevated means > 0.5 sigma AND > 3x SEM


def zcurve(shard_path):
    sh = torch.load(shard_path, map_location="cpu", weights_only=False)
    names = sh["concepts"]
    # null: every NEUTRAL gauge record x every candidate concept, own-minus-other at the same cut
    null = {int(t): [] for t in sh["cuts"]}
    for r in sh["records"]:
        if r["concept"] not in (None, "neutral"):
            continue
        for t, p in r["proj"].items():
            v = np.array([p[n] for n in names])
            for i in range(len(names)):
                null[int(t)].append(v[i] - np.mean(np.delete(v, i)))
    mu = {t: float(np.mean(vs)) for t, vs in null.items() if vs}
    sd = {t: float(np.std(vs)) + 1e-9 for t, vs in null.items() if vs}
    n_null = {t: len(vs) for t, vs in null.items()}

    per_rec = []                                     # one {cut: z} per concept record
    for r in sh["records"]:
        if r["concept"] in (None, "neutral"):
            continue
        zs = {}
        for t, p in r["proj"].items():
            t = int(t)
            if t not in mu:
                continue
            v = np.array([p[n] for n in names])
            i = names.index(r["concept"])
            zs[t] = ((v[i] - np.mean(np.delete(v, i))) - mu[t]) / sd[t]
        if zs:
            per_rec.append(zs)

    z_by_cut, sem_by_cut = {}, {}
    for t in sorted(mu):
        vals = [zs[t] for zs in per_rec if t in zs]
        if vals:
            z_by_cut[t] = float(np.mean(vals))
            sem_by_cut[t] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None

    windows = {}
    for w, cuts in WINDOWS.items():
        rec_means = [float(np.mean([zs[t] for t in cuts if t in zs]))
                     for zs in per_rec if any(t in zs for t in cuts)]
        if rec_means:
            m = float(np.mean(rec_means))
            sem = float(np.std(rec_means, ddof=1) / np.sqrt(len(rec_means))) if len(rec_means) > 1 else None
            windows[w] = dict(z=m, sem=sem, n=len(rec_means),
                              elevated=bool(sem is not None and m > 3 * sem and m > ELEVATED_SIGMA))
    return dict(z_by_cut=z_by_cut, sem_by_cut=sem_by_cut, windows=windows,
                n_records=len(per_rec), n_null_per_cut=n_null)


def model_story(windows):
    """Elevated at early or mid -> A-like; both clearly floor-y -> B-like; else ambiguous."""
    em = [windows.get(w) for w in ("early", "mid") if windows.get(w)]
    if not em:
        return "no_data"
    if any(w["elevated"] for w in em):
        return "A_like"
    if all(abs(w["z"]) < ELEVATED_SIGMA for w in em):
        return "B_like"
    return "ambiguous"


def main():
    out = {}
    stories = {}
    for slug in ("qwen2.5-1.5b", "qwen2.5-7b"):
        p = os.path.join(TRAJ, f"{slug}_gauge-evoked.pt")
        if not os.path.exists(p):
            print(f"missing {p} -- skipped")
            continue
        r = zcurve(p)
        out[f"{slug}_gauge-evoked"] = r
        stories[slug] = model_story(r["windows"])
        zs = {t: round(v, 2) for t, v in sorted(r["z_by_cut"].items())}
        print(f"{slug} gauge:evoked z={zs} windows={ {w: (round(d['z'], 2), None if d['sem'] is None else round(d['sem'], 3)) for w, d in r['windows'].items()} } -> {stories[slug]}")

    calls = set(stories.values())
    verdict = ("STORY_A" if calls == {"A_like"} else
               "STORY_B" if calls == {"B_like"} else "ambiguous")
    out["verdict"] = dict(
        verdict=verdict, per_model=stories,
        criterion=f"elevated iff window mean z > 3x SEM AND > {ELEVATED_SIGMA} sigma at early/mid cuts; "
                  "STORY_A = elevated here while E4 anti-word evoked z ~= 0.05 sigma (displacement); "
                  "STORY_B = floor here too (persona never writes the injected-vector direction)")
    out["provenance"] = dict(
        date=str(datetime.date.today()),
        scorer="experiments/exp2_output_monitorability/analysis/score_gauge.py",
        shards=os.path.relpath(TRAJ, REPO),
        arm="gauge:evoked (state_trajectory): compose_gauge_system context + GAUGE_PROBE user msg",
        null="neutral gauge records' own-minus-other per cut (same shard), e4-scorer math otherwise",
        fidelity_caveat="gauge streams are RE-TOKENIZED texts (tok(text, add_special_tokens=False)); "
                        "token boundaries may differ from the originally sampled ones",
        reference="e4_trajectory_verdict.json: evoked-under-anti-word z ~= 0.05 sigma (1.5B), "
                  "0.08 sigma (7B); injected ~= 12 sigma")
    tmp = OUT + ".tmp"
    json.dump(out, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT)
    print(f"verdict={verdict}  wrote {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
