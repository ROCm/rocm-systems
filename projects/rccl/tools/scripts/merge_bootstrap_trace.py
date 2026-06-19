#!/usr/bin/env python3
"""Merge per-rank RCCL bootstrap trace dumps into a single timeline.

Inputs:  one or more directories produced by NCCL_BOOTSTRAP_TRACE=1
         (env NCCL_BOOTSTRAP_TRACE_DIR controls the directory). When invoked
         on a multi-node MPI run, copy/symlink every node's dump dir into a
         common parent and pass that parent on the command line.

Outputs:
  --json <path>      Chrome Trace Format file (open in chrome://tracing or Perfetto)
  --perfetto <path>  Same format, named for clarity; phase slices + TCP_INFO
                     counter tracks (rtt/rttvar/cwnd/retrans). Open at
                     https://ui.perfetto.dev
  --csv <path>       Long-form CSV (one row per event)
  --net-csv <path>   TCP_INFO net-sample CSV
  --summary          Print per-phase aggregate stats (count, min, p50, p99, max)

Binary dump format (little-endian):

  v1 header (legacy, 6 x uint32):
    magic = 0xB007F00D, version = 1, rank, isRoot, count, reserved
  followed by `count` Event records.

  v2 header (10 x uint32):
    magic = 0xB007F00D, version = 2, endian_mark = 0x01020304,
    rank, isRoot, eventCount, netCount, eventOverflow, netOverflow, reserved
  followed by `eventCount` Event records, then `netCount` NetSample records.

  Event (24B):     t_ns(Q) rank(I) phase(H) md(H) dur_us(I) bytes(I)
  NetSample (48B): t_ns(Q) rank(I) phase(H) md(H) rtt_us(I) rttvar_us(I)
                   rto_us(I) snd_cwnd(I) total_retrans(I) lost(I) unacked(I)
                   ca_state(H) reserved(H)

The endian_mark lets the reader detect a byte-swapped dump (cross-architecture)
and warn rather than silently misread (rec_26 C2).
"""

import argparse
import json
import os
import re
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
    300: "deploy.comm_init.total",
    301: "deploy.hip_ctx",
    302: "deploy.kernel_load",
    303: "deploy.comm_split_allgather",
    304: "deploy.bootstrap",
    305: "deploy.allgather.peer",
    306: "deploy.topo.detect",
    307: "deploy.topo.paths",
    308: "deploy.graph_search",
    309: "deploy.allgather3",
    310: "deploy.topo.postset",
    311: "deploy.buffers",
    312: "deploy.proxy_create",
    313: "deploy.transport_connect",
    314: "deploy.proxy_connect",
    315: "deploy.tuner_load",
    316: "deploy.dev_comm_setup",
    317: "deploy.intranode_barrier",
    318: "deploy.kernel_launch",
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
ENDIAN_MARK = 0x01020304
HDR_V1_FMT = "<IIIIII"        # 6 x uint32_t (legacy)
HDR_V1_SIZE = struct.calcsize(HDR_V1_FMT)
HDR_V2_FMT = "<IIIIIIIIII"    # 10 x uint32_t
HDR_V2_SIZE = struct.calcsize(HDR_V2_FMT)
EVT_FMT = "<QIHHII"           # t_ns, rank, phase, md, dur_us, bytes
EVT_SIZE = struct.calcsize(EVT_FMT)
assert EVT_SIZE == 24
# t_ns, rank, phase, md, rtt, rttvar, rto, cwnd, retrans, lost, unacked, ca, rsv
NET_FMT = "<QIHHIIIIIIIHH"
NET_SIZE = struct.calcsize(NET_FMT)
assert NET_SIZE == 48

TEXT_PHASE_IDS = {name: phase for phase, name in PHASE_NAMES.items()}
TEXT_PHASE_IDS.update({
    "deploy.exec": 1000,
    "mpi.init": 1001,
    "mpi.init_thread": 1002,
    "pmix.init": 1003,
    "deploy.main": 1004,
})
TEXT_PHASE_NAMES = {phase: name for name, phase in TEXT_PHASE_IDS.items()}
DEPLOY_RE = re.compile(
    r"DEPLOY_TRACE\s+rank=(?P<rank>-?\d+)\s+pid=(?P<pid>\d+)\s+"
    r"phase=(?P<phase>[A-Za-z0-9_.-]+)\s+t_ns=(?P<t_ns>\d+)\s+"
    r"dur_us=(?P<dur_us>\d+)\s+md=(?P<md>\d+)\s+bytes=(?P<bytes>\d+)"
    r"(?:\s+detail=(?P<detail>\S+))?"
)
# TCP_INFO sample line emitted by bootstrap_trace.cc logDump():
#   BNETSTAT rank=R root=0|1 p=PID name=NAME t_ns=ABS md=MD rtt_us=.. rttvar_us=..
#            rto_us=.. cwnd=.. retrans=.. lost=.. unacked=.. ca_state=..
BNETSTAT_RE = re.compile(
    r"BNETSTAT\s+rank=(?P<rank>-?\d+)\s+root=(?P<root>\d+)\s+p=(?P<p>\d+)\s+"
    r"name=(?P<name>[A-Za-z0-9_.-]+)\s+t_ns=(?P<t_ns>\d+)\s+md=(?P<md>\d+)\s+"
    r"rtt_us=(?P<rtt>\d+)\s+rttvar_us=(?P<rttvar>\d+)\s+rto_us=(?P<rto>\d+)\s+"
    r"cwnd=(?P<cwnd>\d+)\s+retrans=(?P<retrans>\d+)\s+lost=(?P<lost>\d+)\s+"
    r"unacked=(?P<unacked>\d+)\s+ca_state=(?P<ca>\d+)"
)


class NetSample:
    __slots__ = ("t_ns", "rank", "phase", "md", "rtt_us", "rttvar_us", "rto_us",
                 "snd_cwnd", "total_retrans", "lost", "unacked", "ca_state",
                 "is_root", "src_file")

    def __init__(self, t_ns, rank, phase, md, rtt, rttvar, rto, cwnd,
                 retrans, lost, unacked, ca, is_root, src_file):
        self.t_ns = t_ns
        self.rank = rank
        self.phase = phase
        self.md = md
        self.rtt_us = rtt
        self.rttvar_us = rttvar
        self.rto_us = rto
        self.snd_cwnd = cwnd
        self.total_retrans = retrans
        self.lost = lost
        self.unacked = unacked
        self.ca_state = ca
        self.is_root = is_root
        self.src_file = src_file

    def name(self):
        return PHASE_NAMES.get(self.phase, TEXT_PHASE_NAMES.get(self.phase, f"phase_{self.phase}"))


class Event:
    __slots__ = ("t_ns", "rank", "phase", "md", "dur_us", "bytes",
                 "is_root", "src_file", "detail")

    def __init__(self, t_ns, rank, phase, md, dur_us, bytes_, is_root, src_file, detail=""):
        self.t_ns = t_ns
        self.rank = rank
        self.phase = phase
        self.md = md
        self.dur_us = dur_us
        self.bytes = bytes_
        self.is_root = is_root
        self.src_file = src_file
        self.detail = detail

    def name(self):
        return PHASE_NAMES.get(self.phase, TEXT_PHASE_NAMES.get(self.phase, f"phase_{self.phase}"))


def load_dump(path: Path, net_out=None):
    """Yield Event objects from a .bin file (v1 or v2).

    A file may contain MULTIPLE concatenated [header|events|nets] segments
    because dumpThreadBuffer() appends one segment per flush across the
    deployment timeline. Iterate segments until EOF. NetSample records (v2)
    are appended to `net_out` if provided.
    """
    data = path.read_bytes()
    pos = 0
    nseg = 0
    while pos + 8 <= len(data):
        magic, version = struct.unpack_from("<II", data, pos)
        if magic != MAGIC:
            swapped = struct.unpack_from(">I", data, pos)[0]
            if swapped == MAGIC:
                print(f"[warn] {path}: byte-swapped dump (big-endian producer); skipping",
                      file=sys.stderr)
            elif nseg == 0:
                print(f"[warn] {path}: bad magic 0x{magic:08x}", file=sys.stderr)
            return
        if version == 1:
            _, _, rank, is_root, count, _ = struct.unpack_from(HDR_V1_FMT, data, pos)
            hdr_size, net_count, ev_ovf, net_ovf = HDR_V1_SIZE, 0, 0, 0
        elif version == 2:
            (_, _, endian, rank, is_root, count, net_count,
             ev_ovf, net_ovf, _) = struct.unpack_from(HDR_V2_FMT, data, pos)
            if endian != ENDIAN_MARK:
                print(f"[warn] {path}: endian marker mismatch; skipping", file=sys.stderr)
                return
            hdr_size = HDR_V2_SIZE
        else:
            print(f"[warn] {path}: unknown version {version}", file=sys.stderr)
            return
        if ev_ovf or net_ovf:
            print(f"[warn] {path} seg{nseg}: ring overflow (events={ev_ovf}, net={net_ovf})",
                  file=sys.stderr)
        base = pos + hdr_size
        for i in range(count):
            off = base + i * EVT_SIZE
            if off + EVT_SIZE > len(data):
                return
            t_ns, evt_rank, phase, md, dur_us, bytes_ = struct.unpack_from(EVT_FMT, data, off)
            yield Event(t_ns, evt_rank, phase, md, dur_us, bytes_, bool(is_root), path.name)
        net_base = base + count * EVT_SIZE
        if net_out is not None:
            for i in range(net_count):
                off = net_base + i * NET_SIZE
                if off + NET_SIZE > len(data):
                    return
                (t_ns, nrank, phase, md, rtt, rttvar, rto, cwnd,
                 retrans, lost, unacked, ca, _) = struct.unpack_from(NET_FMT, data, off)
                net_out.append(NetSample(t_ns, nrank, phase, md, rtt, rttvar, rto,
                                         cwnd, retrans, lost, unacked, ca,
                                         bool(is_root), path.name))
        pos = net_base + net_count * NET_SIZE
        nseg += 1


def load_text_log(path: Path, net_out=None):
    """Yield DEPLOY_TRACE events from a text log; collect BNETSTAT net samples."""
    for line in path.read_text(errors="ignore").splitlines():
        m = DEPLOY_RE.search(line)
        if m:
            phase_name = m.group("phase")
            phase = TEXT_PHASE_IDS.setdefault(phase_name, 2000 + len(TEXT_PHASE_IDS))
            TEXT_PHASE_NAMES[phase] = phase_name
            yield Event(
                int(m.group("t_ns")),
                int(m.group("rank")),
                phase,
                int(m.group("md")),
                int(m.group("dur_us")),
                int(m.group("bytes")),
                False,
                path.name,
                m.group("detail") or "",
            )
            continue
        if net_out is not None:
            mn = BNETSTAT_RE.search(line)
            if mn:
                pname = mn.group("name")
                phase = TEXT_PHASE_IDS.get(pname)
                if phase is None:
                    phase = next((pid for pid, nm in PHASE_NAMES.items() if nm == pname), None)
                if phase is None:
                    # Fall back to the numeric phase id from the line (rec_27 N6)
                    # so an unknown name never collapses to a misleading phase_-1.
                    phase = int(mn.group("p"))
                net_out.append(NetSample(
                    int(mn.group("t_ns")), int(mn.group("rank")), phase,
                    int(mn.group("md")), int(mn.group("rtt")), int(mn.group("rttvar")),
                    int(mn.group("rto")), int(mn.group("cwnd")), int(mn.group("retrans")),
                    int(mn.group("lost")), int(mn.group("unacked")), int(mn.group("ca")),
                    bool(int(mn.group("root"))), path.name))


def collect(dirs, net_out=None):
    events = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"[warn] not a directory: {p}", file=sys.stderr)
            continue
        for f in sorted(p.glob("*.bin")):
            events.extend(load_dump(f, net_out))
        for f in sorted(p.glob("*.log")) + sorted(p.glob("*.out")) + sorted(p.glob("*.err")):
            events.extend(load_text_log(f, net_out))
    return events


def write_chrome_trace(events, path, net_samples=None):
    """Chrome Trace Format JSON (loads directly in Perfetto / chrome://tracing).

    Phase events become duration ('X') / instant ('i') slices per rank; TCP_INFO
    net samples become counter ('C') tracks (rtt_us, rttvar_us, snd_cwnd,
    total_retrans) per rank so the network metrics render as graphs aligned with
    the bootstrap phase timeline.
    """
    net_samples = net_samples or []
    out = {"traceEvents": [], "displayTimeUnit": "ms"}
    if not events and not net_samples:
        with open(path, "w") as f:
            json.dump(out, f)
        return

    all_t = [e.t_ns for e in events] + [s.t_ns for s in net_samples]
    base_ns = min(all_t)
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
        if e.detail:
            args["detail"] = e.detail
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

    # TCP_INFO counter tracks: each net sample emits a 'C' event whose args
    # become separate series in the rank's counter track in Perfetto.
    n_counters = 0
    for s in net_samples:
        ts_us = (s.t_ns - base_ns) / 1000.0
        pid = -1 if s.is_root else s.rank
        out["traceEvents"].append({
            "name": "tcp_info",
            "cat": "net",
            "ph": "C",
            "ts": ts_us,
            "pid": pid,
            "tid": 0,
            "args": {
                "rtt_us": s.rtt_us,
                "rttvar_us": s.rttvar_us,
                "snd_cwnd": s.snd_cwnd,
                "total_retrans": s.total_retrans,
                "lost": s.lost,
                "ca_state": s.ca_state,
            },
        })
        n_counters += 1

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
    print(f"[ok] Perfetto/Chrome trace: {path} ({len(out['traceEvents'])} events, "
          f"{len(seen_pid_meta)} processes, {n_counters} tcp_info counters)")


def write_csv(events, path):
    with open(path, "w") as f:
        f.write("t_ns,rank,is_root,phase_id,phase_name,md,dur_us,bytes,src,detail\n")
        for e in events:
            f.write(f"{e.t_ns},{e.rank},{int(e.is_root)},"
                    f"{e.phase},{e.name()},{e.md},{e.dur_us},{e.bytes},{e.src_file},{e.detail}\n")
    print(f"[ok] CSV: {path} ({len(events)} rows)")


def write_net_csv(samples, path):
    with open(path, "w") as f:
        f.write("t_ns,rank,is_root,phase_id,phase_name,md,rtt_us,rttvar_us,"
                "rto_us,snd_cwnd,total_retrans,lost,unacked,ca_state,src\n")
        for s in samples:
            f.write(f"{s.t_ns},{s.rank},{int(s.is_root)},{s.phase},{s.name()},{s.md},"
                    f"{s.rtt_us},{s.rttvar_us},{s.rto_us},{s.snd_cwnd},"
                    f"{s.total_retrans},{s.lost},{s.unacked},{s.ca_state},{s.src_file}\n")
    print(f"[ok] net CSV: {path} ({len(samples)} samples)")


def print_net_summary(samples):
    if not samples:
        return
    by_phase = defaultdict(list)
    for s in samples:
        by_phase[s.phase].append(s)
    print("\nTCP_INFO samples by phase (rtt/rttvar/rto in us; retrans cumulative):")
    print(f"{'phase':<20} {'n':>5} {'rtt_p50':>9} {'rtt_p99':>9} {'rttvar_p99':>11} "
          f"{'retrans':>8} {'lost':>6} {'non-Open':>9}")
    print("-" * 84)
    for phase, rows in sorted(by_phase.items()):
        rtt = sorted(r.rtt_us for r in rows)
        rttvar = sorted(r.rttvar_us for r in rows)
        name = PHASE_NAMES.get(phase, TEXT_PHASE_NAMES.get(phase, f"phase_{phase}"))
        print(f"{name:<20} {len(rows):>5} {percentile(rtt,0.5):>9.0f} "
              f"{percentile(rtt,0.99):>9.0f} {percentile(rttvar,0.99):>11.0f} "
              f"{sum(r.total_retrans for r in rows):>8} {sum(r.lost for r in rows):>6} "
              f"{sum(1 for r in rows if r.ca_state != 0):>9}")


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
    ap.add_argument("--perfetto", default=None,
                    help="Output Perfetto/Chrome trace (.json) with phase slices + "
                         "TCP_INFO counter tracks; open at https://ui.perfetto.dev")
    ap.add_argument("--csv", default=None,
                    help="Output long-form CSV")
    ap.add_argument("--net-csv", default=None,
                    help="Output TCP_INFO net-sample CSV")
    ap.add_argument("--summary", action="store_true",
                    help="Print aggregate per-phase stats")
    args = ap.parse_args()

    net_samples = []
    events = collect(args.dirs, net_samples)
    if not events and not net_samples:
        print("[err] no events collected", file=sys.stderr)
        sys.exit(1)
    events.sort(key=lambda e: e.t_ns)
    net_samples.sort(key=lambda s: s.t_ns)
    print(f"[info] loaded {len(events)} events + {len(net_samples)} net samples from "
          f"{len(set(e.src_file for e in events))} files")

    if args.json:
        write_chrome_trace(events, args.json, net_samples)
    if args.perfetto:
        write_chrome_trace(events, args.perfetto, net_samples)
    if args.csv:
        write_csv(events, args.csv)
    if args.net_csv:
        write_net_csv(net_samples, args.net_csv)
    if args.summary or not (args.json or args.perfetto or args.csv or args.net_csv):
        print_summary(events)
        print_net_summary(net_samples)


if __name__ == "__main__":
    main()
