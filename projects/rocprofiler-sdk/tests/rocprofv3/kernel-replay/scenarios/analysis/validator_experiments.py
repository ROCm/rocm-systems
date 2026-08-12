#!/usr/bin/env python3
"""Experimental evaluation of the kernel-replay validation suite.

E1  per-check detection matrix over the catalogued failure modes
E2  tolerance sensitivity: at what injected error does each check start to fire
E3  randomised fault fuzzing: detection rate with Wilson confidence intervals
E4  validator cost scaling in passes, dispatches and counters

Emits CSV per experiment plus a summary to stdout. No GPU required: every input is
constructed, so the measurements are of the validation logic, not of hardware.
"""

import argparse
import copy
import csv
import inspect
import math
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("KR_TESTDIR", ""))

import replay_fixtures as fx  # noqa: E402
import validate  # noqa: E402

ARGS = {
    "expected_passes": fx.PASSES,
    "common_counters": fx.COMMON_COUNTERS,
    "pass_groups": fx.PASS_GROUPS,
}


def checks():
    out = []
    for name, fn in sorted(vars(validate).items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            out.append((name, fn))
    return out


CHECKS = checks()
SHORT = {name: name.replace("test_", "") for name, _ in CHECKS}


def run_check(fn, doc):
    kwargs = {}
    for param in inspect.signature(fn).parameters:
        kwargs[param] = doc if param == "json_data" else ARGS[param]
    try:
        fn(**kwargs)
    except AssertionError:
        return False  # rejected
    except Exception as err:  # a crash is not a detection
        return ("error", type(err).__name__)
    return True  # accepted


def detections(doc):
    """name -> True if the check REJECTED the document."""
    result = {}
    for name, fn in CHECKS:
        outcome = run_check(fn, doc)
        result[name] = outcome is False
    return result


# ---------------- E1: detection matrix ----------------------------------------


def e1_detection_matrix(outdir):
    modes = sorted(fx.FAILURE_MODES)
    rows = []
    matrix = {}
    for mode in modes:
        det = detections(fx.broken(mode))
        matrix[mode] = det
        rows.append({"mode": mode, **{SHORT[k]: int(v) for k, v in det.items()}})

    golden_det = detections(fx.golden())
    false_positives = [SHORT[k] for k, v in golden_det.items() if v]

    with open(os.path.join(outdir, "e1_detection_matrix.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode"] + [SHORT[n] for n, _ in CHECKS])
        w.writeheader()
        w.writerows(rows)

    # per-check sensitivity over the catalogue
    sens = {}
    for name, _ in CHECKS:
        hits = sum(1 for m in modes if matrix[m][name])
        sens[SHORT[name]] = hits / len(modes)

    # unique catches: modes only this check detects
    unique = {}
    for name, _ in CHECKS:
        u = [m for m in modes if matrix[m][name] and sum(matrix[m].values()) == 1]
        unique[SHORT[name]] = u

    # greedy minimum covering set
    remaining = set(modes)
    cover = []
    while remaining:
        best, best_gain = None, 0
        for name, _ in CHECKS:
            gain = sum(1 for m in remaining if matrix[m][name])
            if gain > best_gain:
                best, best_gain = name, gain
        if best is None:
            break
        cover.append(SHORT[best])
        remaining -= {m for m in remaining if matrix[m][best]}

    # redundancy: how many checks fire per mode
    per_mode = {m: sum(matrix[m].values()) for m in modes}

    return {
        "modes": modes,
        "matrix": matrix,
        "sensitivity": sens,
        "unique": unique,
        "cover": cover,
        "per_mode_depth": per_mode,
        "false_positives": false_positives,
        "uncovered": sorted(remaining),
    }


# ---------------- E2: tolerance sensitivity ------------------------------------


def _drift_shared_counter(doc, frac, which="SQ_WAVES", pass_idx=3, kernel="vecAdd"):
    kid = None
    for rec in fx._records(doc):
        info = rec["dispatch_data"]["dispatch_info"]
        if kid is None:
            kid = {v: k for k, v in fx._KID.items()}
        if kid.get(info["kernel_id"]) == kernel and rec["replay_pass"] == pass_idx:
            base = fx.BASE[kernel][which]
            fx._set(rec, which, base * (1.0 + frac))
    return doc


def _drift_unique_counter(doc, frac, pass_idx=2):
    for rec in fx._records(doc):
        if rec["replay_pass"] == pass_idx:
            name = fx.PASS_GROUPS[pass_idx]
            for sub in rec["records"]:
                if sub["counter_id"]["handle"] == fx._CID[name]:
                    sub["value"] *= 1.0 + frac
    return doc


def e2_tolerance(outdir):
    fracs = [
        0.0,
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.08,
        0.09,
        0.095,
        0.10,
        0.105,
        0.11,
        0.15,
        0.20,
        0.50,
        1.00,
    ]
    rows = []
    first_detect = {"shared": None, "unique": None}
    for frac in fracs:
        for kind, mutate in (
            ("shared", _drift_shared_counter),
            ("unique", _drift_unique_counter),
        ):
            doc = mutate(fx.golden(), frac)
            det = detections(doc)
            any_det = any(det.values())
            if any_det and first_detect[kind] is None and frac > 0:
                first_detect[kind] = frac
            rows.append(
                {
                    "injected_error": frac,
                    "target": kind,
                    "detected": int(any_det),
                    "checks_firing": ";".join(
                        sorted(SHORT[k] for k, v in det.items() if v)
                    ),
                }
            )
    with open(os.path.join(outdir, "e2_tolerance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {
        "rows": rows,
        "first_detect": first_detect,
        "declared_tolerance": validate.COUNTER_TOLERANCE,
    }


# ---------------- E3: randomised fuzzing --------------------------------------


def _wilson(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


FUZZ_OPS = [
    "drop_record",
    "duplicate_record",
    "perturb_value",
    "shift_pass",
    "swap_pass_groups",
    "drop_counter_entry",
    "retag_counter",
    "retag_kernel",
    "change_dispatch_id",
    "scale_dims",
]


def _apply_fuzz(doc, rng, op):
    recs = fx._records(doc)
    if not recs:
        return op
    rec = rng.choice(recs)
    if op == "drop_record":
        recs.remove(rec)
    elif op == "duplicate_record":
        recs.append(copy.deepcopy(rec))
    elif op == "perturb_value":
        if rec["records"]:
            sub = rng.choice(rec["records"])
            sub["value"] = float(sub["value"]) * rng.choice([0.5, 0.9, 1.1, 2.0, 0.0])
    elif op == "shift_pass":
        rec["replay_pass"] = rng.randrange(0, fx.PASSES + 2)
    elif op == "swap_pass_groups":
        a, b = rng.sample(range(fx.PASSES), 2)
        for r in recs:
            if r["replay_pass"] == a:
                r["replay_pass"] = b
            elif r["replay_pass"] == b:
                r["replay_pass"] = a
    elif op == "drop_counter_entry":
        if rec["records"]:
            rec["records"].remove(rng.choice(rec["records"]))
    elif op == "retag_counter":
        if rec["records"]:
            sub = rng.choice(rec["records"])
            sub["counter_id"]["handle"] = rng.randrange(1, len(fx.ALL_COUNTERS) + 1)
    elif op == "retag_kernel":
        rec["dispatch_data"]["dispatch_info"]["kernel_id"] = rng.choice(
            list(fx._KID.values())
        )
    elif op == "change_dispatch_id":
        rec["dispatch_data"]["dispatch_info"]["dispatch_id"] = rng.randrange(1, 5)
    elif op == "scale_dims":
        info = rec["dispatch_data"]["dispatch_info"]
        info["grid_size"]["x"] = int(
            info["grid_size"]["x"] * rng.choice([0.5, 1.03, 2.0])
        )
    return op


def e3_fuzz(outdir, trials, seed):
    rng = random.Random(seed)
    per_op = {op: {"n": 0, "detected": 0} for op in FUZZ_OPS}
    rows = []
    escapes = []
    for trial in range(trials):
        op = rng.choice(FUZZ_OPS)
        doc = fx.golden()
        _apply_fuzz(doc, rng, op)
        det = detections(doc)
        hit = any(det.values())
        per_op[op]["n"] += 1
        per_op[op]["detected"] += int(hit)
        rows.append(
            {
                "trial": trial,
                "op": op,
                "detected": int(hit),
                "checks_firing": ";".join(sorted(SHORT[k] for k, v in det.items() if v)),
            }
        )
        if not hit:
            escapes.append(op)
    with open(os.path.join(outdir, "e3_fuzz.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total_n = sum(v["n"] for v in per_op.values())
    total_d = sum(v["detected"] for v in per_op.values())
    summary = {}
    for op, v in per_op.items():
        lo, hi = _wilson(v["detected"], v["n"])
        summary[op] = {
            "n": v["n"],
            "detected": v["detected"],
            "rate": (v["detected"] / v["n"]) if v["n"] else 0.0,
            "ci": (lo, hi),
        }
    overall_ci = _wilson(total_d, total_n)
    return {
        "per_op": summary,
        "overall": {
            "n": total_n,
            "detected": total_d,
            "rate": total_d / total_n if total_n else 0,
            "ci": overall_ci,
        },
        "escape_counts": {op: escapes.count(op) for op in FUZZ_OPS if escapes.count(op)},
    }


# ---------------- E4: cost scaling --------------------------------------------


def _synth(n_dispatch, n_pass, n_extra_counters):
    """Golden-shaped document with configurable size."""
    doc = fx.golden()
    recs = []
    kernels = list(fx.KERNELS)
    for d in range(n_dispatch):
        kernel = kernels[d % len(kernels)]
        for p in range(n_pass):
            counters = dict(fx.BASE[kernel])
            counters[fx.PASS_GROUPS[p % len(fx.PASS_GROUPS)]] = 1000.0 * (p + 1) + d
            rec = fx._record(d + 1, kernel, p, counters)
            for extra in range(n_extra_counters):
                rec["records"].append(
                    {"counter_id": {"handle": 10_000 + extra}, "value": float(extra)}
                )
            recs.append(rec)
    fx._replace_records(doc, recs)
    return doc


def e4_scaling(outdir, repeats):
    rows = []
    configs = []
    for n_dispatch in (3, 30, 300, 3000):
        configs.append((n_dispatch, fx.PASSES, 0))
    for n_pass in (2, 5, 10, 20):
        configs.append((30, n_pass, 0))
    for extra in (0, 10, 50, 200):
        configs.append((30, fx.PASSES, extra))
    for n_dispatch, n_pass, extra in configs:
        doc = _synth(n_dispatch, n_pass, extra)
        n_records = len(fx._records(doc))
        # pass_groups must cover n_pass for the mapping check to be meaningful
        saved = ARGS["pass_groups"], ARGS["expected_passes"]
        ARGS["pass_groups"] = [
            fx.PASS_GROUPS[i % len(fx.PASS_GROUPS)] for i in range(n_pass)
        ]
        ARGS["expected_passes"] = n_pass
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            detections(doc)
            times.append(time.perf_counter() - t0)
        ARGS["pass_groups"], ARGS["expected_passes"] = saved
        rows.append(
            {
                "dispatches": n_dispatch,
                "passes": n_pass,
                "extra_counters": extra,
                "records": n_records,
                "median_ms": 1000 * statistics.median(times),
                "min_ms": 1000 * min(times),
                "max_ms": 1000 * max(times),
                "repeats": repeats,
            }
        )
    with open(os.path.join(outdir, "e4_scaling.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return {"rows": rows}


# ---------------- main --------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/exp/out")
    ap.add_argument("--fuzz-trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"validator checks discovered: {len(CHECKS)}")
    for name, _ in CHECKS:
        print(f"  {SHORT[name]}")

    print("\n== E1 detection matrix ==")
    e1 = e1_detection_matrix(args.outdir)
    print(f"  failure modes: {len(e1['modes'])}")
    print(f"  false positives on a correct replay: {e1['false_positives'] or 'none'}")
    print("  per-check sensitivity over the catalogue:")
    for name, s in sorted(e1["sensitivity"].items(), key=lambda kv: -kv[1]):
        print(f"    {s*100:5.1f}%  {name}")
    print("  checks that are the ONLY detector of some mode:")
    for name, modes in sorted(e1["unique"].items()):
        if modes:
            print(f"    {name}: {', '.join(modes)}")
    print(f"  greedy minimum covering set: {e1['cover']}")
    print(f"  uncovered modes: {e1['uncovered'] or 'none'}")
    depths = list(e1["per_mode_depth"].values())
    print(
        f"  detection depth per mode: median {statistics.median(depths)}, "
        f"min {min(depths)}, max {max(depths)}"
    )
    singles = [m for m, d in e1["per_mode_depth"].items() if d == 1]
    print(f"  modes caught by exactly one check ({len(singles)}): {', '.join(singles)}")

    print("\n== E2 tolerance sensitivity ==")
    e2 = e2_tolerance(args.outdir)
    print(f"  declared COUNTER_TOLERANCE: {e2['declared_tolerance']:.0%}")
    for kind, frac in e2["first_detect"].items():
        print(
            f"  first detected error on a {kind} counter: "
            f"{'never' if frac is None else f'{frac:.1%}'}"
        )

    print("\n== E3 randomised fuzzing ==")
    e3 = e3_fuzz(args.outdir, args.fuzz_trials, args.seed)
    o = e3["overall"]
    print(
        f"  {o['detected']}/{o['n']} detected = {o['rate']:.1%} "
        f"(95% Wilson CI {o['ci'][0]:.1%}..{o['ci'][1]:.1%})"
    )
    print("  by mutation class:")
    for op, s in sorted(e3["per_op"].items(), key=lambda kv: kv[1]["rate"]):
        print(
            f"    {s['rate']:6.1%}  [{s['ci'][0]:.1%}..{s['ci'][1]:.1%}]  n={s['n']:4d}  {op}"
        )
    if e3["escape_counts"]:
        print("  escapes by class:")
        for op, c in sorted(e3["escape_counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {c:4d}  {op}")

    print("\n== E4 cost scaling ==")
    e4 = e4_scaling(args.outdir, args.repeats)
    print(f"  {'records':>8} {'disp':>5} {'pass':>5} {'extra':>6} {'median_ms':>10}")
    for r in e4["rows"]:
        print(
            f"  {r['records']:8d} {r['dispatches']:5d} {r['passes']:5d} "
            f"{r['extra_counters']:6d} {r['median_ms']:10.3f}"
        )

    print(f"\nCSVs in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
