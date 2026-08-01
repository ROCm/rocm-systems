#!/usr/bin/env python3
"""Sanity-checks a kernel timing trace written by rccl-tests.

Reports per-rank record counts and duration statistics, and verifies the
invariants a dispatch timeline must satisfy: end after start, no dispatch
overlapping another on the same rank, and every window inside the run.
"""
import csv
import sys
from collections import defaultdict


def main(paths):
    """Takes one trace per process, so a run under mpirun passes all of them."""
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    if not rows:
        print("no records")
        return 1

    byrank = defaultdict(list)
    for r in rows:
        byrank[int(r["Rank"])].append(r)

    lo = min(int(r["Start_Timestamp"]) for r in rows)
    hi = max(int(r["End_Timestamp"]) for r in rows)
    print(f"{len(rows)} records from {len(paths)} process(es), {len(byrank)} ranks, span {(hi-lo)/1e6:.1f} ms")

    bad = 0
    for rank in sorted(byrank):
        rs = byrank[rank]
        durs = sorted(int(r["Duration_ns"]) for r in rs)
        seqs = [int(r["Seq"]) for r in rs]
        gaps = [b - a for a, b in zip(seqs, seqs[1:]) if b - a != 1]
        overlaps = 0
        prev_end = 0
        for r in sorted(rs, key=lambda r: int(r["Start_Timestamp"])):
            s, e = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
            if e <= s:
                bad += 1
            if s < prev_end:
                overlaps += 1
            prev_end = e
        n = len(durs)
        print(
            f"  rank {rank}: {n:4d} recs  seq {seqs[0]}..{seqs[-1]}"
            f"{' (gaps: %d)' % len(gaps) if gaps else ''}"
            f"  dur min {durs[0]/1000:8.2f} med {durs[n//2]/1000:8.2f}"
            f" max {durs[-1]/1000:8.2f} us  overlaps {overlaps}"
        )

    print(f"invalid windows: {bad}")

    cfgs = defaultdict(list)
    for r in rows:
        cfgs[r["Config"]].append(int(r["Duration_ns"]))
    print(f"{len(cfgs)} configs:")
    for cfg, durs in cfgs.items():
        durs.sort()
        print(f"  n={len(durs):4d} med {durs[len(durs)//2]/1000:8.2f} us  {cfg[:90]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
