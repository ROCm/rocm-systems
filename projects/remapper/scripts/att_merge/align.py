#!/usr/bin/env python3
"""Cross-replay timeline alignment on CU 0 of every shader engine, scored on held-out shared events.

Replay r traces SIMD r. For an ordered replay pair (r, q) and one shader engine, a method maps r's shader clock
onto q's. Test events are seen in both replays:
  barrier: completion of the n-th s_barrier_wait of workgroup g. A barrier releases all of a workgroup's waves
           together, so replay r sees it on SIMD r and replay q on SIMD q; aligned, the two times should agree.
  occupancy: begin/end of each wave resident on CU 0 (every replay records all SIMDs' occupancy).
An event is never an anchor for its own prediction. Usage: align.py EXTRACT_DIR OUT.json
"""

import argparse
import os
import collections
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

SPACINGS = (1, 2, 4, 8, 16, 32, 64, 128)
# A1 uid = workgroup identity * waves per workgroup + wave (8 waves per workgroup in this kernel).
WAVES_PER_WORKGROUP = int(os.environ.get("REMAPPER_MERGE_WAVES", "8"))
TARGET_CU = int(os.environ.get("REMAPPER_MERGE_TARGET_CU", "0"))


def load(extract):
    sweeps = collections.defaultdict(dict)
    for p in sorted(Path(extract).glob("*.pkl")):
        r = pickle.loads(p.read_bytes())
        sweeps[r["sweep"]][r["traced_simd"]] = r
    return sweeps


def events(cell, se):
    e = cell["engines"][se]
    occ_cu0, occ_all = {}, {}
    for w in e["residencies"]:
        if w["uid"] is None:
            continue
        for kind in ("begin", "end"):
            occ_all[(w["uid"], kind)] = w[kind]
            if w["cu"] == TARGET_CU:
                occ_cu0[(w["uid"], kind)] = w[kind]
    bar = {}
    for w in e["detailed"]:
        if w["cu"] == TARGET_CU and w["uid"] is not None:
            for n, t in enumerate(w["barriers"]):
                bar[(w["uid"] // WAVES_PER_WORKGROUP, n)] = t
    first = min(w["begin"] for w in e["residencies"] if w["cu"] == TARGET_CU)
    return dict(
        occ=occ_cu0, occ_all=occ_all, bar=bar, dispatch=e["dispatch"][0], first=first
    )


def offsets(anchors, a, b):
    keys = sorted(anchors, key=lambda k: a[k])
    x = np.array([a[k] for k in keys], dtype=np.float64)
    return keys, x, np.array([b[k] for k in keys], dtype=np.float64) - x


def warp(anchors, a, b):
    """Piecewise-linear offset through the anchors; constant offset beyond the ends."""
    _, x, d = offsets(anchors, a, b)
    return lambda t: t + np.interp(t, x, d)


def loo(keys_set, a, b):
    """Leave-one-out prediction of each anchor from its neighbours in time."""
    keys, x, d = offsets(keys_set, a, b)
    pred = np.empty_like(d)
    pred[0], pred[-1] = d[1], d[-2]
    xi, xl, xr, dl, dr = x[1:-1], x[:-2], x[2:], d[:-2], d[2:]
    span = np.where(xr > xl, xr - xl, 1.0)
    pred[1:-1] = dl + (dr - dl) * (xi - xl) / span
    return dict(zip(keys, x + pred))


def scores(predicted, b, keys):
    return [abs(predicted[k] - b[k]) for k in keys]


def pair(a, b):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    bar = sorted(set(a["bar"]) & set(b["bar"]))
    occ = sorted(set(a["occ"]) & set(b["occ"]))
    if len(occ) < 3:
        return out, dict(shared_barriers=len(bar), shared_occupancy=len(occ))
    if len(bar) < 3:
        bar = []  # kernels without (enough) barriers: occupancy anchors only

    def score(name, fn, test="both"):
        if test in ("both", "barrier"):
            out[name]["barrier"] += [abs(fn(a["bar"][k]) - b["bar"][k]) for k in bar]
        if test in ("both", "occupancy"):
            out[name]["occupancy"] += [abs(fn(a["occ"][k]) - b["occ"][k]) for k in occ]

    score("dispatch timestamp", lambda t: t + b["dispatch"] - a["dispatch"])
    score("first wave on WGP", lambda t: t + b["first"] - a["first"])
    others = set(a["occ_all"]) & set(b["occ_all"]) - set(occ)
    if others:  # sparse launches have no identified waves off the traced WGP
        score("occupancy, all WGPs on SE", warp(others, a["occ_all"], b["occ_all"]))
    # Same-WGP occupancy: barriers test it directly; occupancy events are held out one at a time.
    score("occupancy, traced WGP", warp(occ, a["occ"], b["occ"]), "barrier")
    p = loo(occ, a["occ"], b["occ"])
    out["occupancy, traced WGP"]["occupancy"] += scores(p, b["occ"], occ)
    # Barrier anchors every s-th barrier of each workgroup; tested on the barriers that are not anchors.
    for s in SPACINGS if bar else ():
        name = f"barriers, every {s}" if s > 1 else "barriers, all others"
        if s == 1:
            p = loo(bar, a["bar"], b["bar"])
            out[name]["barrier"] += scores(p, b["bar"], bar)
            score(name, warp(bar, a["bar"], b["bar"]), "occupancy")
        else:
            anchors = [k for k in bar if k[1] % s == 0]
            test = [k for k in bar if k[1] % s]
            fn = warp(anchors, a["bar"], b["bar"])
            out[name]["barrier"] += [abs(fn(a["bar"][k]) - b["bar"][k]) for k in test]
            score(name, fn, "occupancy")
    both_a = {
        **{("o",) + k: v for k, v in a["occ"].items()},
        **{("b",) + k: v for k, v in a["bar"].items()},
    }
    both_b = {
        **{("o",) + k: v for k, v in b["occ"].items()},
        **{("b",) + k: v for k, v in b["bar"].items()},
    }
    keys = [("o",) + k for k in occ] + [("b",) + k for k in bar]
    p = loo(keys, both_a, both_b)
    out["occupancy + barriers"]["barrier"] += [
        abs(p[("b",) + k] - b["bar"][k]) for k in bar
    ]
    out["occupancy + barriers"]["occupancy"] += [
        abs(p[("o",) + k] - b["occ"][k]) for k in occ
    ]
    return out, dict(shared_barriers=len(bar), shared_occupancy=len(occ))


def summarize(xs):
    if not xs:
        return None
    xs = np.asarray(xs)
    return dict(
        n=int(xs.size),
        median=float(np.median(xs)),
        p90=float(np.percentile(xs, 90)),
        p99=float(np.percentile(xs, 99)),
        mean=float(xs.mean()),
    )


def main():
    global WAVES_PER_WORKGROUP, TARGET_CU
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("extract")
    ap.add_argument("out")
    ap.add_argument("--waves-per-workgroup", type=int, default=WAVES_PER_WORKGROUP)
    ap.add_argument("--target-cu", type=int, default=TARGET_CU)
    args = ap.parse_args()
    WAVES_PER_WORKGROUP, TARGET_CU = args.waves_per_workgroup, args.target_cu
    sweeps = load(args.extract)
    pooled = collections.defaultdict(lambda: collections.defaultdict(list))
    coverage, context = [], collections.defaultdict(list)
    for sweep, cells in sorted(sweeps.items()):
        if sorted(cells) != [0, 1, 2, 3]:
            continue
        for se in sorted(cells[0]["engines"]):
            ev = {r: events(cells[r], se) for r in range(4)}
            for r in range(4):
                waves = collections.defaultdict(list)
                for (g, n), t in ev[r]["bar"].items():
                    waves[g].append(t)
                for ts in waves.values():
                    context["barrier_interval"] += list(np.diff(sorted(ts)))
            for r in range(4):
                for q in range(4):
                    if r == q:
                        continue
                    res, cov = pair(ev[r], ev[q])
                    coverage.append(dict(sweep=sweep, se=se, r=r, q=q, **cov))
                    for m, sets in res.items():
                        for t, xs in sets.items():
                            pooled[m][t] += xs
    workgroups_seen = [
        len({g for g, _ in events(c, se)["bar"]})
        for cells in sweeps.values()
        for c in cells.values()
        for se in c["engines"]
    ]
    result = dict(
        methods={
            m: {t: summarize(xs) for t, xs in sets.items()}
            for m, sets in pooled.items()
        },
        coverage=dict(
            pairs=len(coverage),
            pairs_scored=sum(c["shared_barriers"] >= 3 for c in coverage),
            median_shared_barriers=float(
                np.median([c["shared_barriers"] for c in coverage])
            ),
            median_shared_occupancy=float(
                np.median([c["shared_occupancy"] for c in coverage])
            ),
            traced_workgroups_per_replay_and_se=float(np.median(workgroups_seen)),
        ),
        context=dict(barrier_interval=summarize(context["barrier_interval"])),
    )
    Path(args.out).write_text(json.dumps(result, indent=2))
    rows = [
        (m, v.get("barrier"), v.get("occupancy")) for m, v in result["methods"].items()
    ]
    print(json.dumps(result["coverage"]), json.dumps(result["context"]))
    print(
        f"{'method':32} {'barrier median':>15} {'p90':>10} {'occupancy median':>17} {'p90':>10}"
    )
    for m, bs, os_ in rows:
        f = lambda s, k: f"{s[k]:.0f}" if s else "-"
        print(
            f"{m:32} {f(bs, 'median'):>15} {f(bs, 'p90'):>10} {f(os_, 'median'):>17} {f(os_, 'p90'):>10}"
        )


if __name__ == "__main__":
    main()
