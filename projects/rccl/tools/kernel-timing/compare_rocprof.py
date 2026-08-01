#!/usr/bin/env python3
"""Compares kernel durations reported by the drain API against a rocprof trace.

The two cannot be collected in one run -- rocprof replaces the dispatch
completion signals the drain API reads -- so this compares the duration
distribution of the same workload measured each way, per collective size.
"""
import csv
import re
import sys
from collections import defaultdict
from statistics import median


def quantile(xs, q):
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def ours(path):
    """Durations from the drain API, with the size each record was tagged with."""
    out = []
    for r in csv.DictReader(open(path)):
        m = re.search(r"size=(\d+)", r["Config"])
        out.append((int(r["Duration_ns"]), int(m.group(1)) if m else 0))
    return out


def rocprof(path):
    """Durations of the RCCL device kernels in a rocprof kernel trace."""
    return [
        (int(r["End_Timestamp"]) - int(r["Start_Timestamp"]), 0)
        for r in csv.DictReader(open(path))
        if "ncclDevKernel" in r["Kernel_Name"]
    ]


def band(xs, lo, hi):
    return sorted(d for d, _ in xs if lo <= d < hi)


def main(ours_csv, rocprof_csv):
    a, b = ours(ours_csv), rocprof(rocprof_csv)
    print(f"drain API: {len(a)} records")
    print(f"rocprof:   {len(b)} RCCL kernel dispatches")

    # rocprof rows carry no collective size, so both sets are split the same
    # way: by duration, at boundaries that sit in the empty gaps between the
    # three message sizes. The label comes from what the drain API records in
    # each band agree on.
    bounds = [(0, 60_000), (60_000, 300_000), (300_000, 10**12)]
    print(f"\n{'band':>16} {'size':>10} {'n drain':>8} {'n rp':>6} {'drain med':>11} {'rocprof med':>12} {'delta':>9} {'delta %':>8}")
    for lo, hi in bounds:
        ds, rs = band(a, lo, hi), band(b, lo, hi)
        if not ds or not rs:
            continue
        labels = {s for d, s in a if lo <= d < hi and s}
        dm, rm = median(ds), median(rs)
        name = f"{lo/1000:.0f}-{hi/1000:.0f}us" if hi < 10**12 else f">{lo/1000:.0f}us"
        sz = ",".join(str(s) for s in sorted(labels))
        print(
            f"{name:>16} {sz:>10} {len(ds):>8} {len(rs):>6} {dm/1000:>10.2f}u {rm/1000:>11.2f}u"
            f" {(dm-rm)/1000:>8.2f}u {100*(dm-rm)/rm:>7.2f}%"
        )
        print(
            f"{'':>16} {'':>10} p10 {quantile(ds,0.1)/1000:8.2f}u p90 {quantile(ds,0.9)/1000:8.2f}u"
            f"      p10 {quantile(rs,0.1)/1000:8.2f}u p90 {quantile(rs,0.9)/1000:8.2f}u"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
