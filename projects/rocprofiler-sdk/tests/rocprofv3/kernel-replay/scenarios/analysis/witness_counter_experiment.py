#!/usr/bin/env python3
"""X1 and X2, the follow-ups the escape decomposition pointed at.

X1  A counter placed in two --pmc groups appears in two passes and therefore becomes
    cross-checkable. Quantify how much of the unique-counter blind spot that recovers,
    with and without a generalised agreement check.

X2  Non-vacuous fuzzing with a continuous perturbation magnitude, giving a detection
    curve rather than one rate. Vacuous mutations (which change nothing) are rejected at
    generation, so the denominator is meaningful.
"""

import argparse
import copy
import csv
import inspect
import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("KR_TESTDIR", ""))

import replay_fixtures as fx  # noqa: E402
import validate  # noqa: E402

TOL = validate.COUNTER_TOLERANCE
THRESH = TOL / (1.0 - TOL)  # analytic detection threshold for a one-sided drift


def wilson(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---- generalised agreement check (candidate addition) -------------------------


def check_repeated_counters_agree(json_data, **_):
    """Any counter appearing in more than one pass of a dispatch must agree there.

    Subsumes the shared-counter check and extends it to any counter a user happens to
    place in several --pmc groups, which is the only way a per-pass counter can be
    cross-validated at all.
    """
    table = validate._records_by_dispatch(validate._sdk(json_data))
    bad = []
    for did, entry in table.items():
        seen = {}
        for pass_idx, batch in entry["passes"].items():
            for name, value in batch.items():
                seen.setdefault(name, []).append((pass_idx, value))
        for name, observations in seen.items():
            if len(observations) < 2:
                continue
            values = [v for _p, v in observations]
            if not validate._approx_equal(min(values), max(values)):
                bad.append(f"dispatch {did} {name}: {[round(v, 2) for v in values]}")
    assert not bad, "counters collected in several passes disagree: " + "; ".join(bad[:6])


BASE_CHECKS = [
    (n, f)
    for n, f in sorted(vars(validate).items())
    if n.startswith("test_") and inspect.isfunction(f)
]


def detected(doc, pass_groups, extra_check=None):
    args = {
        "expected_passes": len(pass_groups),
        "common_counters": fx.COMMON_COUNTERS,
        "pass_groups": pass_groups,
    }
    checks = list(BASE_CHECKS) + ([("extra", extra_check)] if extra_check else [])
    for _name, fn in checks:
        kwargs = {}
        for param in inspect.signature(fn).parameters:
            if param == "json_data":
                kwargs[param] = doc
            elif param in args:
                kwargs[param] = args[param]
        if "kwargs" in inspect.signature(fn).parameters or any(
            p.kind == p.VAR_KEYWORD for p in inspect.signature(fn).parameters.values()
        ):
            pass
        try:
            fn(**kwargs)
        except AssertionError:
            return True
        except TypeError:
            fn(doc)
    return False


# ---- X1 -----------------------------------------------------------------------


def build_doc(pass_groups):
    """Golden-shaped doc whose pass i collects pass_groups[i] as its unique counter."""
    doc = fx.golden()
    recs = []
    for d, kernel in enumerate(fx.KERNELS):
        for p, unique in enumerate(pass_groups):
            counters = dict(fx.BASE[kernel])
            counters[unique] = 1000.0 * (p + 1) + d
            recs.append(fx._record(d + 1, kernel, p, counters))
    fx._replace_records(doc, recs)
    return doc


def corrupt(doc, counter, pass_idx, factor):
    for rec in fx._records(doc):
        if rec["replay_pass"] != pass_idx:
            continue
        for sub in rec["records"]:
            if sub["counter_id"]["handle"] == fx._CID[counter]:
                sub["value"] = float(sub["value"]) * factor
    return doc


def x1(outdir, trials, seed):
    rng = random.Random(seed)
    # witness layout: GRBM_COUNT appears in group 0 and group 2
    witness = [
        "GRBM_COUNT",
        "GRBM_GUI_ACTIVE",
        "GRBM_COUNT",
        "SQ_INSTS_SMEM",
        "SQ_INSTS_LDS",
    ]
    plain = list(fx.PASS_GROUPS)
    rows = []
    tally = {}
    for label, groups, target_pass, target in (
        ("plain/unique", plain, 2, plain[2]),
        ("witness/repeated", witness, 2, "GRBM_COUNT"),
        ("witness/non-repeated", witness, 3, "SQ_INSTS_SMEM"),
    ):
        for extra_label, extra in (
            ("base", None),
            ("with_agreement", check_repeated_counters_agree),
        ):
            hits = 0
            for _ in range(trials):
                factor = rng.choice([0.0, 0.25, 0.5, 2.0, 10.0, 100.0])
                doc = corrupt(build_doc(groups), target, target_pass, factor)
                if detected(doc, groups, extra):
                    hits += 1
            lo, hi = wilson(hits, trials)
            tally[(label, extra_label)] = (hits, trials, lo, hi)
            rows.append(
                {
                    "layout": label,
                    "checks": extra_label,
                    "target": target,
                    "pass": target_pass,
                    "detected": hits,
                    "trials": trials,
                    "rate": hits / trials,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    with open(os.path.join(outdir, "x1_witness_counter.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows, "tally": tally}


# ---- X2 -----------------------------------------------------------------------


def x2(outdir, per_bin, seed):
    rng = random.Random(seed)
    bins = [0.005, 0.02, 0.05, 0.08, 0.10, 0.111, 0.12, 0.15, 0.25, 0.5, 1.0, 5.0]
    rows = []
    for kind in ("shared", "unique"):
        counter = "SQ_WAVES" if kind == "shared" else fx.PASS_GROUPS[2]
        for frac in bins:
            hits = 0
            for _ in range(per_bin):
                sign = rng.choice([1.0, -1.0])
                doc = corrupt(build_doc(fx.PASS_GROUPS), counter, 2, 1.0 + sign * frac)
                if detected(doc, fx.PASS_GROUPS):
                    hits += 1
            lo, hi = wilson(hits, per_bin)
            rows.append(
                {
                    "kind": kind,
                    "magnitude": frac,
                    "detected": hits,
                    "trials": per_bin,
                    "rate": hits / per_bin,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "beyond_threshold": int(frac > THRESH),
                }
            )
    with open(os.path.join(outdir, "x2_detection_curve.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/exp/out")
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--per-bin", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260813)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    print(f"declared tolerance {TOL:.0%}, analytic one-sided threshold {THRESH:.2%}")

    print("\n== X1 witness counter (one counter placed in two --pmc groups) ==")
    r1 = x1(a.outdir, a.trials, a.seed)
    print(f"  {'layout':22s} {'checks':16s} {'detected':>9} {'rate':>7}  95% CI")
    for row in r1["rows"]:
        print(
            f"  {row['layout']:22s} {row['checks']:16s} "
            f"{row['detected']:4d}/{row['trials']:<4d} {row['rate']:6.1%}  "
            f"[{row['ci_lo']:.1%}..{row['ci_hi']:.1%}]"
        )

    print("\n== X2 detection curve, non-vacuous continuous magnitude ==")
    r2 = x2(a.outdir, a.per_bin, a.seed)
    print(f"  {'kind':8s} {'magnitude':>10} {'>thresh':>8} {'rate':>7}  95% CI")
    for row in r2["rows"]:
        print(
            f"  {row['kind']:8s} {row['magnitude']:10.3f} {row['beyond_threshold']:8d} "
            f"{row['rate']:6.1%}  [{row['ci_lo']:.1%}..{row['ci_hi']:.1%}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
