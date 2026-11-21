#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Compare two SelectorGolden CSVs decision by decision.

Exit code 0 when every point picks the same backend, protocol and channel count,
1 when any decision changed, 2 when the files were recorded in different setups
(rank count, arch, env) and cannot be compared.

Changed points are folded into size ranges: consecutive counts of the same
(coll, dtype, op, reg, inplace, capture) slice with the same old -> new change
print as one line.
"""
import argparse
import csv
import sys
from collections import OrderedDict, defaultdict

SLICE = ("coll", "dtype", "op", "reg", "inplace", "capture")
DECISION = ("algo", "proto", "channels", "status")
# Header keys that must match for a comparison to mean anything.
SETUP_KEYS = ("arch", "nranks", "nhosts", "ranks_per_host", "cta_policy", "warmup", "max_bytes")


def load(path):
    header, env, rows = {}, set(), OrderedDict()
    with open(path) as f:
        lines = []
        for line in f:
            if line.startswith("# env "):
                env.add(line[len("# env "):].strip())
            elif line.startswith("# ") and "=" in line:
                k, v = line[2:].strip().split("=", 1)
                header[k] = v
            elif not line.startswith("#"):
                lines.append(line)
    for r in csv.DictReader(lines):
        rows[tuple(r[k] for k in SLICE) + (int(r["count"]),)] = r
    return header, env, rows


def fmt_decision(r, ignore_channels):
    keys = [k for k in DECISION if not (ignore_channels and k == "channels")]
    s = "%s(%s)/%s" % (r["algo_name"], r["algo"], r["proto"])
    if not ignore_channels:
        s += "/ch%s" % r["channels"]
    if r["status"] != "0":
        s += "/err%s" % r["status"]
    return s, tuple(r[k] for k in keys)


def human(n):
    for unit in ("", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return ("%d%s" % (n, unit)) if n == int(n) else ("%.1f%s" % (n, unit))
        n /= 1024.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--ignore-channels", action="store_true",
                    help="compare backend and protocol only (channel counts can move with unrelated launch tuning)")
    ap.add_argument("--allow-setup-mismatch", action="store_true")
    ap.add_argument("--max-lines", type=int, default=200)
    args = ap.parse_args()

    hb, eb, base = load(args.baseline)
    hc, ec, cand = load(args.candidate)

    setup_diff = [(k, hb.get(k), hc.get(k)) for k in SETUP_KEYS if hb.get(k) != hc.get(k)]
    if eb != ec:
        setup_diff.append(("env", sorted(eb - ec), sorted(ec - eb)))
    if setup_diff:
        for k, a, b in setup_diff:
            print("setup differs: %s: %s -> %s" % (k, a, b))
        if not args.allow_setup_mismatch:
            return 2

    missing = [k for k in base if k not in cand]
    extra = [k for k in cand if k not in base]
    disagree = [k for k, r in cand.items() if r.get("ranks_agree") == "0"]

    # slice -> list of (count, bytes, old, new)
    changes = defaultdict(list)
    for k, rb in base.items():
        rc = cand.get(k)
        if rc is None:
            continue
        sb, kb = fmt_decision(rb, args.ignore_channels)
        sc, kc = fmt_decision(rc, args.ignore_channels)
        if kb != kc:
            changes[k[:-1]].append((k[-1], int(rb["bytes"]), sb, sc))

    total = sum(len(v) for v in changes.values())
    print("baseline  %s: %d points" % (args.baseline, len(base)))
    print("candidate %s: %d points" % (args.candidate, len(cand)))
    print("changed: %d   missing in candidate: %d   new in candidate: %d   ranks disagreeing in candidate: %d"
          % (total, len(missing), len(extra), len(disagree)))

    lines = []
    for sl, items in sorted(changes.items()):
        items.sort()
        run = [items[0]]
        for it in items[1:]:
            if (it[2], it[3]) == (run[-1][2], run[-1][3]):
                run.append(it)
            else:
                lines.append((sl, run))
                run = [it]
        lines.append((sl, run))
    for sl, run in lines[: args.max_lines]:
        lo, hi = run[0][1], run[-1][1]
        rng = human(lo) if lo == hi else "%s..%s" % (human(lo), human(hi))
        print("  %-13s %-8s %-4s %s inplace=%s capture=%s  bytes %-14s (%d pts)  %s -> %s"
              % (sl[0], sl[1], sl[2], sl[3], sl[4], sl[5], rng, len(run), run[0][2], run[0][3]))
    if len(lines) > args.max_lines:
        print("  ... %d more ranges" % (len(lines) - args.max_lines))

    return 1 if (total or missing or extra) else 0


if __name__ == "__main__":
    sys.exit(main())
