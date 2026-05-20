#!/usr/bin/env python3
"""Merge per-rank RCCL bootstrap trace dumps into a single timeline.

Inputs:  one or more directories produced by NCCL_BOOTSTRAP_TRACE=1
         (env NCCL_BOOTSTRAP_TRACE_DIR controls the directory). When invoked
         on a multi-node MPI run, copy/symlink every node's dump dir into a
         common parent and pass that parent on the command line.

Outputs:
  --json <path>   Chrome Trace Format file (open in chrome://tracing or Perfetto)
  --csv <path>    Long-form CSV (one row per event)
  --summary       Print per-phase aggregate stats (count, min, p50, p99, max)

Each binary dump file starts with:
  uint32_t magic       = 0xB007F00D
  uint32_t version     = 1
  uint32_t rank
  uint32_t isRoot
  uint32_t count
  uint32_t reserved
followed by `count` Event records:
  uint64_t t_ns       (CLOCK_MONOTONIC_RAW absolute)
  uint32_t rank
  uint16_t phase
  uint16_t md
  uint32_t dur_us
  uint32_t bytes
"""

import argparse
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

# Phase IDs — keep in sync with src/include/bootstrap_trace.h
PHASE_NAMES = {
    0:   "init.total",
    1:   "listen.fwd",
    2:   "listen.rev",
    3:   "listen.root",
    4:   "send_to_root.conn",
    5:   "send_to_root.data",
    6:   "recv_from_root",
    7:   "forward_connect",
    8:   "reverse_connect",
    9:   "proxy_listen",
    10:  "peer_listen",
    11:  "ras_init",
    12:  "ring_allgather",
    13:  "ring_step",
    14:  "proxy_init",
    100: "root.total",
    101: "root.wait_first",
    102: "root.accept",
    103: "root.recv_info",
    104: "root.inline_send",
    105: "root.final_send",
    200: "tcp.connect",
    201: "tcp.accept",
    202: "tcp.ready",
}

MAGIC = 0xB007F00D
HDR_FMT = "<IIIIII"           # 6 x uint32_t
HDR_SIZE = struct.calcsize(HDR_FMT)
EVT_FMT = "<QIHHII"           # t_ns, rank, phase, md, dur_us, bytes
EVT_SIZE = struct.calcsize(EVT_FMT)
assert EVT_SIZE == 24


class Event:
    __slots__ = ("t_ns", "rank", "phase", "md", "dur_us", "bytes",
                 "is_root", "src_file")

    def __init__(self, t_ns, rank, phase, md, dur_us, bytes_, is_root, src_file):
        self.t_ns = t_ns
        self.rank = rank
        self.phase = phase
        self.md = md
        self.dur_us = dur_us
        self.bytes = bytes_
        self.is_root = is_root
        self.src_file = src_file

    def name(self):
        return PHASE_NAMES.get(self.phase, f"phase_{self.phase}")


def load_dump(path: Path):
    """Yield Event objects from a single .bin file."""
    data = path.read_bytes()
    if len(data) < HDR_SIZE:
        print(f"[warn] {path}: truncated (<{HDR_SIZE}B)", file=sys.stderr)
        return
    magic, version, rank, is_root, count, _ = struct.unpack_from(HDR_FMT, data, 0)
    if magic != MAGIC:
        print(f"[warn] {path}: bad magic 0x{magic:08x}", file=sys.stderr)
        return
    if version != 1:
        print(f"[warn] {path}: unknown version {version}", file=sys.stderr)
        return
    body = data[HDR_SIZE:]
    expected = count * EVT_SIZE
    if len(body) < expected:
        print(f"[warn] {path}: body truncated, expected {expected}, got {len(body)}",
              file=sys.stderr)
        count = len(body) // EVT_SIZE
    for i in range(count):
        t_ns, evt_rank, phase, md, dur_us, bytes_ = struct.unpack_from(
            EVT_FMT, body, i * EVT_SIZE)
        yield Event(t_ns, evt_rank, phase, md, dur_us, bytes_,
                    bool(is_root), path.name)


def collect(dirs):
    events = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"[warn] not a directory: {p}", file=sys.stderr)
            continue
        for f in sorted(p.glob("*.bin")):
            events.extend(load_dump(f))
    return events


def write_chrome_trace(events, path):
    """Chrome Trace Format JSON: one event per record, X = complete event."""
    out = {"traceEvents": [], "displayTimeUnit": "ms"}
    if not events:
        with open(path, "w") as f:
            json.dump(out, f)
        return

    base_ns = min(e.t_ns for e in events)
    for e in events:
        ts_us = (e.t_ns - base_ns) / 1000.0
        if e.is_root:
            pid = -1
            tid_label = f"root[{e.src_file}]"
            tid = abs(hash(tid_label)) % (1 << 31)
            args_extra = {"src_file": e.src_file}
        else:
            pid = e.rank
            tid_label = "main"
            tid = 1
            args_extra = {}
        args = {"md": e.md, "bytes": e.bytes, **args_extra}
        ev = {
            "name": e.name(),
            "cat": "bootstrap",
            "ts": ts_us,
            "pid": pid,
            "tid": tid,
            "args": args,
        }
        if e.dur_us > 0:
            ev["ph"] = "X"
            ev["dur"] = e.dur_us
        else:
            ev["ph"] = "i"
            ev["s"] = "t"
        out["traceEvents"].append(ev)

    seen_pid_meta = set()
    for e in events:
        pid = -1 if e.is_root else e.rank
        if pid in seen_pid_meta:
            continue
        seen_pid_meta.add(pid)
        label = "ROOT" if e.is_root else f"rank {e.rank}"
        out["traceEvents"].append({
            "name": "process_name", "ph": "M", "pid": pid, "tid": 0,
            "args": {"name": label},
        })
        out["traceEvents"].append({
            "name": "process_sort_index", "ph": "M", "pid": pid, "tid": 0,
            "args": {"sort_index": -1 if e.is_root else e.rank},
        })

    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[ok] Chrome trace: {path} ({len(out['traceEvents'])} events, "
          f"{len(seen_pid_meta)} processes)")


def write_csv(events, path):
    with open(path, "w") as f:
        f.write("t_ns,rank,is_root,phase_id,phase_name,md,dur_us,bytes,src\n")
        for e in events:
            f.write(f"{e.t_ns},{e.rank},{int(e.is_root)},"
                    f"{e.phase},{e.name()},{e.md},{e.dur_us},{e.bytes},{e.src_file}\n")
    print(f"[ok] CSV: {path} ({len(events)} rows)")


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = (len(sorted_vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def print_summary(events):
    by_phase = defaultdict(list)
    for e in events:
        if e.dur_us > 0:
            by_phase[(e.phase, e.is_root)].append(e.dur_us)

    rows = []
    for (phase, is_root), durs in by_phase.items():
        durs.sort()
        rows.append((
            ("ROOT " if is_root else "rank ") + PHASE_NAMES.get(phase, f"phase_{phase}"),
            len(durs),
            durs[0],
            percentile(durs, 0.50),
            percentile(durs, 0.99),
            durs[-1],
            sum(durs) / len(durs),
        ))
    rows.sort(key=lambda r: -r[6])
    print()
    print(f"{'phase':<28} {'n':>6} {'min':>10} {'p50':>10} {'p99':>10} {'max':>10} {'avg':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r[0]:<28} {r[1]:>6} {r[2]:>10.1f} {r[3]:>10.1f} "
              f"{r[4]:>10.1f} {r[5]:>10.1f} {r[6]:>10.1f}")
    print(f"\nUnits: microseconds. {len(events)} total events from "
          f"{len(set(e.rank for e in events if not e.is_root))} ranks "
          f"+ {sum(1 for e in events if e.is_root and e.phase == 100)} root threads.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+",
                    help="One or more dump directories")
    ap.add_argument("--json", default=None,
                    help="Output Chrome Trace Format JSON (chrome://tracing, Perfetto)")
    ap.add_argument("--csv", default=None,
                    help="Output long-form CSV")
    ap.add_argument("--summary", action="store_true",
                    help="Print aggregate per-phase stats")
    args = ap.parse_args()

    events = collect(args.dirs)
    if not events:
        print("[err] no events collected", file=sys.stderr)
        sys.exit(1)
    events.sort(key=lambda e: e.t_ns)
    print(f"[info] loaded {len(events)} events from "
          f"{len(set(e.src_file for e in events))} files")

    if args.json:
        write_chrome_trace(events, args.json)
    if args.csv:
        write_csv(events, args.csv)
    if args.summary or (not args.json and not args.csv):
        print_summary(events)


if __name__ == "__main__":
    main()
