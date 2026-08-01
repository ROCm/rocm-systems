#!/usr/bin/env python3
"""Durations at latency-bound sizes: drain API vs a rocprof kernel trace.

All the sizes involved take nearly the same time, so unlike compare_rocprof.py
there is nothing to bin -- the whole distribution is compared at once.
"""
import csv
import glob
import re
import sys
from collections import defaultdict
from statistics import median


def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main(ours_glob, rocprof_glob):
    ours = defaultdict(list)
    for r in csv.DictReader(open(glob.glob(ours_glob)[0])):
        m = re.search(r"size=(\d+)", r["Config"])
        if m and "phase=setup" not in r["Config"]:
            ours[int(m.group(1))].append(int(r["Duration_ns"]))

    rp = sorted(
        int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        for r in csv.DictReader(open(glob.glob(rocprof_glob)[0]))
        if "ncclDevKernel" in r["Kernel_Name"]
    )

    print(f"{'size':>6} {'n':>7} {'drain med':>11} {'p10':>9} {'p90':>9}")
    every = []
    for s in sorted(ours):
        d = sorted(ours[s])
        every += d
        print(f"{s:>6} {len(d):>7} {median(d)/1000:>10.2f}u {q(d,.1)/1000:>8.2f}u {q(d,.9)/1000:>8.2f}u")

    every.sort()
    print()
    print(f"drain   all sizes: n={len(every):>6} med {median(every)/1000:7.2f}u  p10 {q(every,.1)/1000:6.2f}u p90 {q(every,.9)/1000:6.2f}u")
    print(f"rocprof all sizes: n={len(rp):>6} med {median(rp)/1000:7.2f}u  p10 {q(rp,.1)/1000:6.2f}u p90 {q(rp,.9)/1000:6.2f}u")
    delta = median(every) - median(rp)
    print(f"delta: {delta/1000:+.2f}us ({100*delta/median(rp):+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
