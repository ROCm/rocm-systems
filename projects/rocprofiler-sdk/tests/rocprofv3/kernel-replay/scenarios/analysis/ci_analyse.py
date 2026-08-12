#!/usr/bin/env python3
"""Statistical analysis of the collected CI measurements."""

import collections
import csv
import math
import statistics
import sys


def wilson(hits, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def two_prop_z(h1, n1, h2, n2):
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p1, p2 = h1 / n1, h2 / n2
    p = (h1 + h2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("inf"), 0.0
    z = (p1 - p2) / se
    # two-sided p via erfc
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "/tmp/exp/out/ci_measurements.csv")
    jobs = load("/tmp/exp/out/ci_jobs.csv")
    print(f"measurements: {len(rows)}   jobs: {len(jobs)}")
    print(
        f"branches: {len(set(r['branch'] for r in rows))}   "
        f"distinct tests: {len(set(r['test'] for r in rows))}"
    )

    # ---- Q1: duration stability of the kernel-replay tests -------------------
    print("\n== Q1 kernel-replay test duration stability ==")
    replay = [
        r for r in rows if "kernel-replay" in r["test"] or "kernel_replay" in r["test"]
    ]
    by_test = collections.defaultdict(list)
    for r in replay:
        if r["status"] == "Passed":
            by_test[r["test"]].append(float(r["seconds"]))
    print(f"  {'test':62s} {'n':>3} {'median':>7} {'CV':>7} {'min':>7} {'max':>7}")
    for test, vals in sorted(by_test.items(), key=lambda kv: -len(kv[1])):
        if len(vals) < 3:
            continue
        med = statistics.median(vals)
        cv = statistics.stdev(vals) / statistics.mean(vals) if len(vals) > 1 else 0.0
        print(
            f"  {test[:62]:62s} {len(vals):3d} {med:7.3f} {cv:6.1%} "
            f"{min(vals):7.3f} {max(vals):7.3f}"
        )

    # per-OS comparison for the generate test
    print("\n  per-OS medians, rocprofv3-test-kernel-replay-generate:")
    gen = collections.defaultdict(list)
    for r in replay:
        if r["test"].endswith("kernel-replay-generate") and r["status"] == "Passed":
            os_name = (
                r["job"].split("•")[-1].strip() if "•" in r["job"] else r["job"][:28]
            )
            gen[os_name].append(float(r["seconds"]))
    for os_name, vals in sorted(gen.items()):
        if vals:
            print(
                f"    {os_name:32s} n={len(vals):2d} median {statistics.median(vals):6.3f}s "
                f"range {min(vals):.3f}-{max(vals):.3f}"
            )

    # ---- Q2: hip-streams-per-thread flakiness -------------------------------
    print("\n== Q2 hip-streams-per-thread outcome across independent jobs ==")
    streams = [r for r in rows if "hip-streams-per-thread" in r["test"]]
    exe = [r for r in streams if r["test"].startswith("tests.integration.execute")]
    per_job = {}
    for r in exe:
        per_job[(r["job_id"], r["branch"], r["job"])] = r["status"]
    bad = sum(1 for v in per_job.values() if v != "Passed")
    n = len(per_job)
    lo, hi = wilson(bad, n)
    print(f"  jobs observing the execute test: {n}")
    print(f"  non-Passed outcomes: {bad}  = {bad/n:.1%} (95% Wilson {lo:.1%}..{hi:.1%})")
    by_status = collections.Counter(per_job.values())
    for status, count in by_status.most_common():
        print(f"    {status:28s} {count}")

    print("\n  split by job family:")
    fam = collections.defaultdict(lambda: [0, 0])
    for (jid, branch, job), status in per_job.items():
        key = "TheRock gfx94X" if job.startswith("Linux (") else "Core mi325"
        fam[key][1] += 1
        if status != "Passed":
            fam[key][0] += 1
    for key, (bad_k, n_k) in sorted(fam.items()):
        lo_k, hi_k = wilson(bad_k, n_k)
        print(
            f"    {key:16s} {bad_k}/{n_k} = {bad_k/n_k:5.1%} (95% {lo_k:.1%}..{hi_k:.1%})"
        )
    if len(fam) == 2:
        (k1, v1), (k2, v2) = sorted(fam.items())
        z, p = two_prop_z(v1[0], v1[1], v2[0], v2[1])
        print(f"    two-proportion test {k1} vs {k2}: z={z:.2f}, p={p:.4g}")

    print("\n  split by branch:")
    per_branch = collections.defaultdict(lambda: [0, 0])
    for (jid, branch, job), status in per_job.items():
        per_branch[branch.split("/")[-1]][1] += 1
        if status != "Passed":
            per_branch[branch.split("/")[-1]][0] += 1
    for branch, (bad_b, n_b) in sorted(per_branch.items()):
        print(f"    {branch[:44]:44s} {bad_b}/{n_b}")

    # ---- Q3: job-level pass rates -------------------------------------------
    print("\n== Q3 job-level CTest summaries ==")
    have = [j for j in jobs if j["tests_total"]]
    print(f"  jobs reporting a CTest summary: {len(have)}")
    fails = [int(j["tests_failed"]) for j in have]
    totals = [int(j["tests_total"]) for j in have]
    if have:
        print(
            f"  tests per job: median {statistics.median(totals):.0f}, "
            f"range {min(totals)}-{max(totals)}"
        )
        print(
            f"  failures per job: median {statistics.median(fails):.0f}, "
            f"mean {statistics.mean(fails):.2f}, max {max(fails)}"
        )
        clean = sum(1 for f in fails if f == 0)
        lo3, hi3 = wilson(clean, len(fails))
        print(
            f"  jobs with zero test failures: {clean}/{len(fails)} = "
            f"{clean/len(fails):.1%} (95% {lo3:.1%}..{hi3:.1%})"
        )
        dist = collections.Counter(fails)
        print(
            "  failure-count distribution: "
            + ", ".join(f"{k} failures x{v}" for k, v in sorted(dist.items()))
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
