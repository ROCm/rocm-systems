#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""Split and score the RCCL execution-policy rule table from forced sweeps.

Every rule gets a score: the peak bus bandwidth its profile reached over its
byte range, divided by the platform's theoretical peak. Scores are only
comparable between rules covering the same sizes, so rules that can match the
same input are first split at each other's byte boundaries.

A piece is scored from the forced sweep of its transport, at the sizes where
the table's original priority order made it the selected rule of that
transport (the configuration the sweep measured). Pieces never selected in the
sweep are UNMEASURED.

Usage:
  execution_policy_scores.py RULES_CC --ipc DIR --shm DIR [--write] [--report]

DIR holds rccl-tests output files (all_reduce.txt, alltoall.txt, ...) from a
single-node sweep with -f 2. --write rewrites RULES_CC in place; --report
lists measured rules shadowed by an earlier, measurably slower rule matching
the same input, which the build rejects. Selection is first-match, so fixing
the order is the table author's decision.
"""

import argparse
import os
import re
import sys

SIZE_MAX = (1 << 64) - 1
INT_MIN, INT_MAX = -(1 << 31), (1 << 31) - 1
MARGIN = 1.03
# Facts of the sweep topology used to decide which rule each sweep measured.
SWEEP_ARCH, SWEEP_NODES, SWEEP_RANKS, SWEEP_CHANNELS = "gfx1201", 1, 8, 32

COLL_FILES = {
    "ALL_TO_ALL": ("alltoall", 1), "ALL_TO_ALL_V": ("alltoallv", 1),
    "ALL_GATHER": ("all_gather", 1), "ALL_REDUCE": ("all_reduce", 1),
    "BROADCAST": ("broadcast", 1), "REDUCE": ("reduce", 1),
    "REDUCE_SCATTER": ("reduce_scatter", 1), "GATHER": ("gather", 1),
    "SCATTER": ("scatter", 1),
    # P2P Send/Recv facts are rank-aggregate bytes; rccl-tests reports per peer.
    "ncclFuncSend": ("sendrecv", SWEEP_RANKS), "ncclFuncRecv": ("sendrecv", SWEEP_RANKS),
}
TABLE_START = "constexpr RuleSpec kCollectiveExecutionRules[] = {"


def split_top(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch in "{(":
            depth += 1
        elif ch in "})":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def strip_braces(text):
    text = text.strip()
    assert text[0] == "{" and text[-1] == "}", text
    return text[1:-1]


def eval_bytes(expr):
    return int(eval(expr.replace("ULL", "").replace("SIZE_MAX", str(SIZE_MAX)), {}, {}))


def fmt_bytes(value):
    if value == SIZE_MAX:
        return "SIZE_MAX"
    for v, suffix in ((value, ""), (value + 1, " - 1")):
        for shift in (30, 20, 10):
            if v >= (1 << shift) and v % (1 << shift) == 0:
                body = f"{v >> shift}ULL << {shift}"
                return f"({body}){suffix}" if suffix else body
    return str(value)


def parse_int_range(expr):
    expr = expr.strip()
    if expr == "kAnyInt":
        return (INT_MIN, INT_MAX)
    m = re.fullmatch(r"(EQ|GE)\((\d+)\)", expr)
    assert m, expr
    v = int(m.group(2))
    return (v, v) if m.group(1) == "EQ" else (v, INT_MAX)


class Rule:
    def __init__(self, text):
        parts = split_top(strip_braces(text))
        assert len(parts) in (2, 3), text
        self.match_fields = split_top(strip_braces(parts[0]))
        assert len(self.match_fields) == 9, parts[0]
        self.profile = parts[1]
        f = self.match_fields
        self.scope, self.arch, self.coll = f[0], f[1].strip('"'), f[2]
        if f[3] == "kAnyBytes":
            self.lo, self.hi = 0, SIZE_MAX
        else:
            lo, hi = split_top(strip_braces(f[3]))
            self.lo, self.hi = eval_bytes(lo), eval_bytes(hi)
        self.nodes, self.ranks, self.channels = (parse_int_range(x) for x in f[4:7])
        self.transport, self.in_place = f[7], f[8]
        self.score = None  # measured GB/s, or None when unmeasured

    def piece(self, lo, hi, in_place=None):
        other = Rule.__new__(Rule)
        other.__dict__.update(self.__dict__)
        other.match_fields = list(self.match_fields)
        other.lo, other.hi = lo, hi
        if in_place is not None:
            other.in_place = other.match_fields[8] = in_place
        return other

    def placements(self):
        return {"ANY_BOOL": ("out", "in"), "BOOL_FALSE": ("out",), "BOOL_TRUE": ("in",)}[self.in_place]

    def matches(self, arch, nodes, ranks, channels, size, placement, transports):
        inside = lambda r, v: r[0] <= v <= r[1]
        return (arch.startswith(self.arch) and inside(self.nodes, nodes) and inside(self.ranks, ranks)
                and inside(self.channels, channels) and self.lo <= size <= self.hi
                and placement in self.placements() and self.transport in transports)

    def render(self, peak_name):
        f = list(self.match_fields)
        f[3] = "kAnyBytes" if (self.lo, self.hi) == (0, SIZE_MAX) else \
            "{" + fmt_bytes(self.lo) + ", " + fmt_bytes(self.hi) + "}"
        score = "UNMEASURED" if self.score is None else f"EFFICIENCY({self.score:.2f}, {peak_name})"
        return (f"  {{{{{', '.join(f[:6])},\n    {', '.join(f[6:])}}},\n"
                f"   {self.profile}, {score}}},\n")


def can_match_same_input(a, b):
    overlap = lambda x, y: x[0] <= y[1] and y[0] <= x[1]
    placement = a.in_place == "ANY_BOOL" or b.in_place == "ANY_BOOL" or a.in_place == b.in_place
    return (a.scope == b.scope and a.coll == b.coll and a.lo <= b.hi and b.lo <= a.hi
            and overlap(a.nodes, b.nodes) and overlap(a.ranks, b.ranks)
            and overlap(a.channels, b.channels) and placement
            and (a.arch.startswith(b.arch) or b.arch.startswith(a.arch)))


def align(pieces):
    """Split [(slot, rule)] until co-matching pieces share identical or disjoint
    byte ranges, and a placement-specific rule never shares inputs with an
    ANY_BOOL one; otherwise a score would describe inputs it never measured."""
    changed = True
    while changed:
        changed = False
        result = []
        for slot, rule in pieces:
            cuts, split_placement = set(), False
            for _, other in pieces:
                if other is rule or not can_match_same_input(rule, other):
                    continue
                split_placement |= rule.in_place == "ANY_BOOL" and other.in_place != "ANY_BOOL"
                for cut in (other.lo, other.hi + 1 if other.hi != SIZE_MAX else None):
                    if cut is not None and rule.lo < cut <= rule.hi:
                        cuts.add(cut)
            if split_placement:
                changed = True
                result.extend((slot, rule.piece(rule.lo, rule.hi, p)) for p in ("BOOL_FALSE", "BOOL_TRUE"))
                continue
            if not cuts:
                result.append((slot, rule))
                continue
            changed = True
            bounds = [rule.lo] + sorted(cuts) + [rule.hi + 1]
            result.extend((slot, rule.piece(bounds[i], bounds[i + 1] - 1)) for i in range(len(bounds) - 1))
        pieces = result
    return pieces


def load_sweep(directory, name, multiplier, peak):
    """Return {size: {"out": GB/s, "in": GB/s}} with bus bandwidth from exact
    timings. Points above the platform peak are measurement artifacts and are
    dropped."""
    path = os.path.join(directory, name + ".txt")
    points, factor = {}, None
    if not os.path.exists(path):
        return points
    rows = []
    with open(path) as f:
        for line in f:
            cols = line.split()
            if len(cols) < 13 or not cols[0].isdigit() or cols[0] == "0":
                continue
            rows.append(cols)
    for cols in rows:
        try:
            algbw, busbw = float(cols[6]), float(cols[7])
        except ValueError:
            continue
        if algbw >= 1.0:
            factor = busbw / algbw
    if factor is None:
        return points
    for cols in rows:
        size = int(cols[0])
        entry = {}
        for key, col in (("out", 5), ("in", 9)):
            try:
                time_us = float(cols[col])
            except ValueError:
                continue
            if time_us <= 0:
                continue
            busbw = size / (time_us * 1e3) * factor
            if busbw > peak:
                print(f"dropping {path} size={size} {key}: {busbw:.2f} GB/s exceeds peak {peak}",
                      file=sys.stderr)
                continue
            entry[key] = busbw
        points[size * multiplier] = entry
    return points


def forced_winner(target, rule_list, size, placement, transport):
    for rule in rule_list:
        if rule.coll == target.coll and rule.scope == target.scope and rule.matches(
                SWEEP_ARCH, SWEEP_NODES, SWEEP_RANKS, SWEEP_CHANNELS, size, placement, (transport,)):
            return rule
    return None


def score_pieces(pieces, sweep_dirs, peak):
    rules = [rule for _, rule in pieces]
    cache = {}
    for rule in rules:
        if rule.transport not in sweep_dirs or rule.coll not in COLL_FILES or \
                not SWEEP_ARCH.startswith(rule.arch):
            continue
        name, multiplier = COLL_FILES[rule.coll]
        key = (rule.transport, name)
        if key not in cache:
            cache[key] = load_sweep(sweep_dirs[rule.transport], name, multiplier, peak)
        sweep = cache[key]
        peaks = []
        for placement in rule.placements():
            values = [point[placement] for size, point in sweep.items()
                      if rule.lo <= size <= rule.hi and placement in point
                      and forced_winner(rule, rules, size, placement, rule.transport) is rule]
            if not values:
                break
            peaks.append(max(values))
        else:
            rule.score = min(peaks)


def report(pieces):
    """Print rules that the build's firstMisorderedRule check rejects: a
    measured rule that outscores an earlier co-matching measured rule by more
    than MARGIN. Moving it is left to the table's author."""
    rules = [rule for _, rule in pieces]
    describe = lambda i, r: (f"#{i} {r.scope} {r.coll} {r.transport} [{fmt_bytes(r.lo)}, "
                             f"{fmt_bytes(r.hi)}] {r.in_place} {r.profile} {r.score:.2f} GB/s")
    count = 0
    for later_id, later in enumerate(rules):
        if not later.score:
            continue
        for earlier_id, earlier in enumerate(rules[:later_id]):
            if earlier.score and round(later.score, 2) > round(earlier.score, 2) * MARGIN and \
                    can_match_same_input(earlier, later):
                count += 1
                print(f"{describe(later_id, later)}\n  is shadowed by {describe(earlier_id, earlier)}")
    print(f"{count} misordered rule pairs", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rules_cc")
    parser.add_argument("--ipc", required=True)
    parser.add_argument("--shm", required=True)
    parser.add_argument("--peak-name", default="kGfx120xPeakBusBwGBps")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    source = open(args.rules_cc).read()
    start = source.index(TABLE_START) + len(TABLE_START)
    end = source.index("\n};", start)
    body = source[start:end]

    # Keep comments and blank lines in place; every rule entry becomes a slot.
    chunks, entry = [], None
    for line in body.split("\n"):
        stripped = line.strip()
        if entry is None and (not stripped or stripped.startswith("//")):
            chunks.append(("text", line))
            continue
        entry = (entry + "\n" + line) if entry is not None else line
        code = re.sub(r"//.*", "", entry)
        if code.count("{") == code.count("}") and code.rstrip().endswith(","):
            chunks.append(("rule", Rule(code.strip().rstrip(","))))
            entry = None
    assert entry is None, "unterminated rule entry"

    pieces = [(slot, rule) for slot, (kind, rule) in enumerate(chunks) if kind == "rule"]
    pieces = align(pieces)
    sweep_dirs = {"IPC": args.ipc, "SHM": args.shm}
    peak_match = re.search(r"constexpr double " + re.escape(args.peak_name) + r" = ([0-9.]+);", source)
    if not peak_match:
        sys.exit(f"{args.peak_name} is not defined in {args.rules_cc}")
    peak = float(peak_match.group(1))
    score_pieces(pieces, sweep_dirs, peak)

    if args.report:
        report(pieces)
    out = []
    for slot, (kind, value) in enumerate(chunks):
        if kind == "text":
            out.append(value + "\n")
        else:
            out.extend(rule.render(args.peak_name) for s, rule in pieces if s == slot)
    new_body = "".join(out).rstrip("\n")
    print(f"{sum(1 for k, _ in chunks if k == 'rule')} rules -> {len(pieces)} scored rules", file=sys.stderr)
    if args.write:
        with open(args.rules_cc, "w") as f:
            f.write(source[:start] + new_body + source[end:])


if __name__ == "__main__":
    main()
