#!/usr/bin/env python3
"""Network-counter report for ACCL profiler output (ACCL_NET_COUNTERS=1).

Usage:
    python3 net_tel_report.py /path/to/output/ [more dirs or .jsonl files]
    python3 net_tel_report.py /path/to/output/ --trace net_trace.json

Prints, per collective and message size:
  - a network verdict: NETWORK_SUSPECT / CLEAN / INCONCLUSIVE, with the
    counters, side and ranks behind it;
  - exec time with the read skew removed: a rank's kernel also waits for a
    peer whose counter read finished later, so
    corrected = exec - (max launch_wait over the comm's ranks - own launch_wait).

--trace writes Chrome trace JSON (chrome://tracing, ui.perfetto.dev): one
process per host, one thread per rank, a slice per counter window and per read.
Timestamps are each host's CLOCK_MONOTONIC and are not aligned across hosts.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# RDMA hw counters (ionic names) that mean the network misbehaved, and the side
# of a transfer whose NIC counts them. Loss shows up on the receiver
# (out-of-sequence, out-of-buffer) and as NAKs/retransmits on the sender, so a
# verdict pools every rank of a size.
NET_LOSS_COUNTERS = {
    'tx_rdma_retx_pkts': 'sender',
    'tx_rdma_retx_bytes': 'sender',
    'tx_rdma_ack_timeout': 'sender',
    'req_tx_retry_excd_err': 'sender',
    'req_rx_pkt_seq_err': 'sender (NAK from receiver)',
    'req_rx_impl_nak_seq_err': 'sender (NAK from receiver)',
    'req_rx_rnr_retry_err': 'sender (RNR from receiver)',
    'resp_rx_outouf_seq': 'receiver',
    'resp_rx_outof_buf': 'receiver',
    'resp_tx_pkt_seq_err': 'receiver',
    'resp_tx_rnr_retry_err': 'receiver',
    'tx_rdma_ccl_cts_retx_pkts': 'receiver (CTS)',
    'tx_rdma_ccl_cts_ack_timeout': 'receiver (CTS)',
    'resp_rx_ccl_cts_outouf_seq': 'sender (CTS)',
}
NET_CONGESTION_COUNTERS = {
    'rx_rdma_ecn_pkts': 'receiver',
    'tx_rdma_cnp_pkts': 'receiver',
    'rx_rdma_cnp_pkts': 'sender',
}
VOLUME = ('tx_rdma_ucast_bytes', 'rx_rdma_ucast_bytes')
MIN_USABLE = 3

FILE_RE = re.compile(r'accl_profiler_rank(\d+)_(.+)_pid(\d+)_(0x[0-9a-fA-F]+)\.jsonl$')


@dataclass
class Rec:
    rank: int
    host: str
    comm: str
    coll: str
    sn: int
    size: int
    exec_us: float
    net: Optional[dict]
    corrected_us: Optional[float] = None


@dataclass
class Verdict:
    verdict: str
    n_records: int
    n_usable: int
    n_overlapped: int
    n_unavailable: int
    tx_bytes_per_coll: float
    # counter -> (sum of deltas, records hit, sorted ranks)
    drivers: Dict[str, Tuple[int, int, List[int]]] = field(default_factory=dict)
    pause_frames: int = 0
    reason: str = ''


def fmt_size(b: int) -> str:
    for unit, thresh in [("G", 1024**3), ("M", 1024**2), ("K", 1024)]:
        if b >= thresh:
            val = b / thresh
            return f"{val:.0f}{unit}" if val == int(val) else f"{val:.1f}{unit}"
    return f"{b}B"


def expand_inputs(paths: List[str]) -> List[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.jsonl'))))
        else:
            files.append(p)
    return files


def load(paths: List[str], warmup: int = 5) -> Tuple[List[Rec], List[dict]]:
    """Records and summary lines from every file; drops the first `warmup`
    records of each (file, coll, size), ordered by sequence number."""
    recs: List[Rec] = []
    summaries: List[dict] = []
    for path in expand_inputs(paths):
        m = FILE_RE.search(os.path.basename(path))
        host, comm = (m.group(2), m.group(4)) if m else ('?', path)
        groups: Dict[Tuple[str, int], List[Rec]] = defaultdict(list)
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('{'):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'summary' in obj:
                    summaries.append(obj['summary'])
                    continue
                cp, hdr = obj.get('coll_perf'), obj.get('header')
                if not cp or not hdr:
                    continue
                r = Rec(rank=hdr.get('rank', 0), host=host, comm=comm,
                        coll=cp.get('coll', ''), sn=cp.get('coll_sn', 0),
                        size=cp.get('coll_msg_size_bytes', 0),
                        exec_us=cp.get('coll_exec_time_us', 0.0),
                        net=obj.get('net_counters'))
                groups[(r.coll, r.size)].append(r)
        for g in groups.values():
            g.sort(key=lambda r: r.sn)
            recs.extend(g[warmup:] if warmup > 0 else g)
    return recs, summaries


def rdma_deltas(net: dict) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for dev in net.get('devices') or []:
        for k, v in (dev.get('rdma') or {}).items():
            out[k] += v
    return out


def pause_frames(net: dict) -> int:
    total = 0
    for dev in net.get('devices') or []:
        for k, v in (dev.get('ethtool') or {}).items():
            if k != 'stale' and 'pause' in k and 'us_count' not in k:
                total += v
    return total


def usable(r: Rec) -> bool:
    return bool(r.net) and not r.net.get('overlapped') and not r.net.get('unavailable')


def verdicts(recs: List[Rec]) -> Dict[Tuple[str, int], Verdict]:
    """Per (coll, msg_size) verdict from clean windows only. Overlapped windows
    hold other work; ethtool deltas are stale on ionic (about 5 s refresh) and
    never drive a verdict."""
    groups: Dict[Tuple[str, int], List[Rec]] = defaultdict(list)
    for r in recs:
        groups[(r.coll, r.size)].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        good = [r for r in rs if usable(r)]
        n_over = sum(1 for r in rs if r.net and r.net.get('overlapped'))
        n_unav = sum(1 for r in rs if not r.net or r.net.get('unavailable'))
        deltas = [(r, rdma_deltas(r.net)) for r in good]
        drivers = {}
        for name in list(NET_LOSS_COUNTERS) + list(NET_CONGESTION_COUNTERS):
            hits = [(r.rank, d[name]) for r, d in deltas if d.get(name, 0) > 0]
            if hits:
                drivers[name] = (sum(v for _, v in hits), len(hits),
                                 sorted({rk for rk, _ in hits}))
        tx = sum(d.get('tx_rdma_ucast_bytes', 0) for _, d in deltas)
        traffic = sum(d.get(k, 0) for _, d in deltas for k in VOLUME)
        known = any('tx_rdma_ucast_bytes' in d for _, d in deltas)
        v = Verdict('CLEAN', len(rs), len(good), n_over, n_unav,
                    tx / len(good) if good else 0.0, drivers,
                    sum(pause_frames(r.net) for r in good))
        loss = [k for k in drivers if k in NET_LOSS_COUNTERS]
        if len(good) < MIN_USABLE or 2 * len(good) < len(rs):
            v.verdict = 'INCONCLUSIVE'
            v.reason = f"{len(good)}/{len(rs)} usable ({n_over} overlapped, {n_unav} unavailable)"
        elif not known and not drivers:
            # Counter names are ionic's; another driver would read as clean.
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'unknown counter names (not ionic)'
        elif loss:
            v.verdict = 'NETWORK_SUSPECT'
            v.reason = 'loss/retransmit'
        elif drivers:
            v.verdict = 'NETWORK_SUSPECT'
            v.reason = 'congestion signalled (ECN/CNP), no loss'
        elif traffic == 0:
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'no NIC traffic in the windows'
        out[key] = v
    return out


def correct_skew(recs: List[Rec]) -> None:
    """Sets corrected_us. Ranks of one comm launch collective sn after their own
    read; a rank whose read ended earlier waits inside its kernel for the
    slowest one, so that difference is taken back out."""
    groups: Dict[Tuple[str, str, int], List[Rec]] = defaultdict(list)
    for r in recs:
        if r.net and 'launch_wait_us' in r.net:
            groups[(r.comm, r.coll, r.sn)].append(r)
    for rs in groups.values():
        latest = max(r.net['launch_wait_us'] for r in rs)
        for r in rs:
            r.corrected_us = max(0.0, r.exec_us - (latest - r.net['launch_wait_us']))


def print_summaries(summaries: List[dict]) -> None:
    nets = [s['net_counters'] for s in summaries if 'net_counters' in s]
    if not nets:
        print("No net_counters in any summary line (run with ACCL_NET_COUNTERS=1).")
        return
    tot = defaultdict(int)
    for n in nets:
        for k in ('records', 'overlapped', 'unavailable', 'records_with_read_error',
                  'queued_at_finalize', 'late_proxy_ops', 'reads'):
            tot[k] += n.get(k, 0)
    read_mean = (sum(n.get('read_us_mean', 0) * n.get('reads', 0) for n in nets) /
                 tot['reads']) if tot['reads'] else 0.0
    print(f"Summaries: {len(nets)} comm-rank files, {tot['records']} records, "
          f"{tot['overlapped']} overlapped, {tot['unavailable']} unavailable, "
          f"{tot['records_with_read_error']} with read errors, "
          f"{tot['queued_at_finalize']} not written by finalize, "
          f"{tot['late_proxy_ops']} late proxy ops")
    print(f"Counter reads: {tot['reads']}, mean {read_mean:.0f} us, "
          f"max {max(n.get('read_us_max', 0) for n in nets)} us")
    incomplete = sum(1 for s in summaries if not s.get('complete', True))
    if incomplete:
        print(f"WARNING: {incomplete} file(s) report complete=false", file=sys.stderr)


def print_verdicts(vs: Dict[Tuple[str, int], Verdict]) -> None:
    print(f"\n{'='*140}")
    print("  NETWORK VERDICT per message size "
          "(clean windows only; ethtool pause frames are stale context)")
    print(f"{'='*140}")
    print(f"{'Coll':>14s} {'Size':>8s} │ {'Usable':>9s} │ {'Verdict':>15s} │ "
          f"{'TxMB/coll':>10s} {'Pause*':>7s} │ Drivers (sum, records hit, side, ranks)")
    for (coll, sz), v in vs.items():
        parts = []
        for name, (total, hits, ranks) in sorted(v.drivers.items(), key=lambda kv: -kv[1][0]):
            side = NET_LOSS_COUNTERS.get(name) or NET_CONGESTION_COUNTERS.get(name, '')
            mb = v.tx_bytes_per_coll * v.n_usable / 1e6
            rate = f", {total / mb:.3g}/MB" if mb > 0 else ''
            parts.append(f"{name}={total} ({hits}/{v.n_usable}{rate}, {side}, "
                         f"ranks {','.join(map(str, ranks))})")
        detail = '; '.join(parts) if parts else (v.reason or '-')
        if parts and v.reason:
            detail = f"[{v.reason}] {detail}"
        print(f"{coll:>14s} {fmt_size(sz):>8s} │ {v.n_usable:>4d}/{v.n_records:<4d} │ "
              f"{v.verdict:>15s} │ {v.tx_bytes_per_coll / 1e6:10.3f} {v.pause_frames:7d} │ {detail}")


def print_timing(recs: List[Rec]) -> None:
    groups: Dict[Tuple[str, int], List[Rec]] = defaultdict(list)
    for r in recs:
        if r.corrected_us is not None:
            groups[(r.coll, r.size)].append(r)
    if not groups:
        return
    print(f"\n{'='*140}")
    print("  EXEC TIME with counter-read skew removed (us; skew = exec - corrected)")
    print(f"{'='*140}")
    print(f"{'Coll':>14s} {'Size':>8s} │ {'n':>5s} │ {'ExecRaw':>10s} {'ExecCorr':>10s} │ "
          f"{'SkewMean':>9s} {'SkewP99':>9s} {'SkewMax':>9s} │ {'LaunchWait':>10s}")
    for (coll, sz), rs in sorted(groups.items()):
        skew = sorted(r.exec_us - r.corrected_us for r in rs)
        wait = statistics.mean(r.net['launch_wait_us'] for r in rs)
        print(f"{coll:>14s} {fmt_size(sz):>8s} │ {len(rs):5d} │ "
              f"{statistics.mean(r.exec_us for r in rs):10.1f} "
              f"{statistics.mean(r.corrected_us for r in rs):10.1f} │ "
              f"{statistics.mean(skew):9.1f} {skew[int(0.99 * (len(skew) - 1))]:9.1f} "
              f"{skew[-1]:9.1f} │ {wait:10.1f}")


def build_trace(recs: List[Rec]) -> dict:
    hosts = sorted({r.host for r in recs})
    pid = {h: i + 1 for i, h in enumerate(hosts)}
    events = []
    for h in hosts:
        events.append({"ph": "M", "name": "process_name", "pid": pid[h],
                       "args": {"name": h}})
    for h, rank in sorted({(r.host, r.rank) for r in recs}):
        events.append({"ph": "M", "name": "thread_name", "pid": pid[h], "tid": rank,
                       "args": {"name": f"rank {rank}"}})
    for r in recs:
        n = r.net
        if not n or 'before_ts_us' not in n:
            continue
        d = rdma_deltas(n)
        symptoms = {k: v for k, v in d.items()
                    if k in NET_LOSS_COUNTERS or k in NET_CONGESTION_COUNTERS}
        args = {"sn": r.sn, "overlapped": n.get('overlapped'),
                "unavailable": n.get('unavailable'), "exec_us": r.exec_us,
                "launch_wait_us": n.get('launch_wait_us'), **symptoms,
                **{k: d[k] for k in VOLUME if k in d}}
        if r.corrected_us is not None:
            args["exec_corrected_us"] = round(r.corrected_us, 2)
        cat = "overlapped" if n.get('overlapped') else (
            "unavailable" if n.get('unavailable') else ("suspect" if symptoms else "window"))
        events.append({"ph": "X", "cat": cat, "name": f"{r.coll} {fmt_size(r.size)}",
                       "pid": pid[r.host], "tid": r.rank, "ts": n['before_ts_us'],
                       "dur": max(0, n['after_ts_us'] - n['before_ts_us']), "args": args})
        events.append({"ph": "X", "cat": "read", "name": "counter read",
                       "pid": pid[r.host], "tid": r.rank, "ts": n['after_ts_us'],
                       "dur": n.get('after_read_us', 0)})
        events.append({"ph": "C", "name": f"rank {r.rank} NIC", "pid": pid[r.host],
                       "ts": n['after_ts_us'],
                       "args": {"tx_MB": d.get('tx_rdma_ucast_bytes', 0) / 1e6,
                                "loss": sum(v for k, v in symptoms.items()
                                            if k in NET_LOSS_COUNTERS)}})
    return {"traceEvents": events, "displayTimeUnit": "ms",
            "otherData": {"clock": "CLOCK_MONOTONIC per host, not aligned across hosts"}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ACCL network-counter report")
    ap.add_argument('inputs', nargs='+', help='ACCL output dirs or .jsonl files')
    ap.add_argument('--warmup', type=int, default=5,
                    help='records to drop per (file, coll, size) (default 5)')
    ap.add_argument('--trace', metavar='FILE', help='also write Chrome trace JSON')
    args = ap.parse_args(argv)

    recs, summaries = load(args.inputs, args.warmup)
    with_net = [r for r in recs if r.net is not None]
    print(f"Loaded {len(recs)} records, {len(with_net)} with net_counters")
    print_summaries(summaries)
    if not with_net:
        return 1
    correct_skew(with_net)
    print_verdicts(verdicts(with_net))
    print_timing(with_net)
    if args.trace:
        with open(args.trace, 'w', encoding='utf-8') as f:
            json.dump(build_trace(with_net), f)
        print(f"\nTrace written to {args.trace}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
