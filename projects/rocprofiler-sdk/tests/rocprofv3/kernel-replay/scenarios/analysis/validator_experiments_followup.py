#!/usr/bin/env python3
"""Follow-up experiments, driven by what E1-E4 exposed.

E2b  fine tolerance sweep around the analytic threshold, and value drift on a
     per-pass unique counter (which E2 suggested is never checked at all)
E5   duplicate / double-delivered records: E3 found this class escapes 100% of the
     time, so quantify it directly and characterise why
E6   multiplicity analysis: how many (dispatch, pass) records exist vs how many the
     validator observes, i.e. what the aggregation silently collapses
"""

import argparse
import copy
import csv
import inspect
import os
import statistics
import sys

sys.path.insert(0, os.environ.get("KR_TESTDIR", ""))

import replay_fixtures as fx  # noqa: E402
import validate  # noqa: E402

ARGS = {
    "expected_passes": fx.PASSES,
    "common_counters": fx.COMMON_COUNTERS,
    "pass_groups": fx.PASS_GROUPS,
}
CHECKS = [
    (n, f)
    for n, f in sorted(vars(validate).items())
    if n.startswith("test_") and inspect.isfunction(f)
]
SHORT = {n: n.replace("test_", "") for n, _ in CHECKS}


def detections(doc):
    out = {}
    for name, fn in CHECKS:
        kwargs = {
            p: (doc if p == "json_data" else ARGS[p])
            for p in inspect.signature(fn).parameters
        }
        try:
            fn(**kwargs)
            out[name] = False
        except AssertionError:
            out[name] = True
    return out


def firing(doc):
    return sorted(SHORT[k] for k, v in detections(doc).items() if v)


# ---- E2b -------------------------------------------------------------------


def _drift(doc, counter, frac, pass_idx, kernel="vecAdd"):
    kid_to_name = {v: k for k, v in fx._KID.items()}
    for rec in fx._records(doc):
        info = rec["dispatch_data"]["dispatch_info"]
        if kid_to_name.get(info["kernel_id"]) != kernel:
            continue
        if rec["replay_pass"] != pass_idx:
            continue
        for sub in rec["records"]:
            if sub["counter_id"]["handle"] == fx._CID[counter]:
                sub["value"] = float(sub["value"]) * (1.0 + frac)
    return doc


def e2b(outdir):
    tol = validate.COUNTER_TOLERANCE
    analytic = tol / (
        1.0 - tol
    )  # |a-b| <= tol*max(a,b) with b=a(1+f) -> f <= tol/(1-tol)
    fine = [round(x / 1000, 4) for x in range(0, 301, 2)]  # 0 .. 30% in 0.2% steps
    rows = []
    thresholds = {}
    for counter, kind in (("SQ_WAVES", "shared"), (fx.PASS_GROUPS[2], "unique")):
        first = None
        for frac in fine:
            doc = _drift(fx.golden(), counter, frac, pass_idx=2)
            fires = firing(doc)
            if fires and first is None and frac > 0:
                first = frac
            rows.append(
                {
                    "counter": counter,
                    "kind": kind,
                    "drift": frac,
                    "detected": int(bool(fires)),
                    "checks": ";".join(fires),
                }
            )
        thresholds[kind] = first
    # extreme drift on a unique counter
    extreme = []
    for frac in (1.0, 10.0, 100.0, 1000.0, -1.0):
        doc = _drift(fx.golden(), fx.PASS_GROUPS[2], frac, pass_idx=2)
        extreme.append((frac, firing(doc)))
    with open(os.path.join(outdir, "e2b_fine_tolerance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {
        "declared": tol,
        "analytic": analytic,
        "thresholds": thresholds,
        "extreme_unique": extreme,
    }


# ---- E5 duplicates ---------------------------------------------------------


def e5_duplicates(outdir):
    results = []

    def dup_all(doc):
        recs = fx._records(doc)
        fx._replace_records(doc, recs + [copy.deepcopy(r) for r in recs])
        return doc

    def dup_one_pass(doc):
        recs = fx._records(doc)
        target = [r for r in recs if r["replay_pass"] == 2]
        fx._replace_records(doc, recs + [copy.deepcopy(r) for r in target])
        return doc

    def dup_with_different_values(doc):
        """A duplicate that disagrees with the original -- double delivery with drift."""
        recs = fx._records(doc)
        extra = []
        for r in recs:
            if r["replay_pass"] == 2:
                clone = copy.deepcopy(r)
                for sub in clone["records"]:
                    sub["value"] = float(sub["value"]) * 3.0
                extra.append(clone)
        fx._replace_records(doc, recs + extra)
        return doc

    def triplicate(doc):
        recs = fx._records(doc)
        fx._replace_records(doc, recs * 3)
        return doc

    for name, mutate in (
        ("duplicate_every_record", dup_all),
        ("duplicate_one_pass", dup_one_pass),
        ("duplicate_with_drift", dup_with_different_values),
        ("triplicate_every_record", triplicate),
    ):
        doc = mutate(fx.golden())
        n_records = len(fx._records(doc))
        fires = firing(doc)
        results.append(
            {
                "mode": name,
                "records_in_json": n_records,
                "detected": int(bool(fires)),
                "checks": ";".join(fires),
            }
        )

    # why: how many records exist vs how many the validator's table retains
    doc = dup_all(fx.golden())
    raw = len(fx._records(doc))
    table = validate._records_by_dispatch(validate._sdk(doc))
    observed = sum(len(e["passes"]) for e in table.values())
    with open(os.path.join(outdir, "e5_duplicates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    return {"results": results, "raw_records": raw, "observed_by_validator": observed}


# ---- E6 multiplicity -------------------------------------------------------


def e6_multiplicity(outdir):
    """For a golden doc, count records per (dispatch, pass) and what the table keeps."""
    rows = []
    for factor in (1, 2, 3, 5):
        doc = fx.golden()
        recs = fx._records(doc)
        fx._replace_records(doc, [copy.deepcopy(r) for r in recs for _ in range(factor)])
        table = validate._records_by_dispatch(validate._sdk(doc))
        observed = sum(len(e["passes"]) for e in table.values())
        rows.append(
            {
                "duplication_factor": factor,
                "records_in_json": len(fx._records(doc)),
                "records_observed": observed,
                "collapsed": len(fx._records(doc)) - observed,
                "detected": int(bool(firing(doc))),
            }
        )
    with open(os.path.join(outdir, "e6_multiplicity.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/exp/out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("== E2b fine tolerance sweep ==")
    r = e2b(args.outdir)
    print(f"  declared COUNTER_TOLERANCE      : {r['declared']:.1%}")
    print(f"  analytic threshold tol/(1-tol)  : {r['analytic']:.2%}")
    for kind, first in r["thresholds"].items():
        print(
            f"  first detected drift ({kind:6s})  : "
            f"{'never within 30%' if first is None else f'{first:.2%}'}"
        )
    print("  extreme drift on a per-pass unique counter:")
    for frac, fires in r["extreme_unique"]:
        print(f"    {frac:+8.1%} -> {fires or 'NOT DETECTED'}")

    print("\n== E5 duplicate / double-delivered records ==")
    d = e5_duplicates(args.outdir)
    for row in d["results"]:
        print(
            f"  {row['mode']:24s} records={row['records_in_json']:3d} "
            f"-> {'detected: ' + row['checks'] if row['detected'] else 'NOT DETECTED'}"
        )
    print(
        f"  aggregation: {d['raw_records']} records in JSON, "
        f"{d['observed_by_validator']} observed by the validator "
        f"({d['raw_records'] - d['observed_by_validator']} silently collapsed)"
    )

    print("\n== E6 multiplicity ==")
    m = e6_multiplicity(args.outdir)
    print(
        f"  {'factor':>7} {'in_json':>8} {'observed':>9} {'collapsed':>10} {'detected':>9}"
    )
    for row in m["rows"]:
        print(
            f"  {row['duplication_factor']:7d} {row['records_in_json']:8d} "
            f"{row['records_observed']:9d} {row['collapsed']:10d} {row['detected']:9d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
