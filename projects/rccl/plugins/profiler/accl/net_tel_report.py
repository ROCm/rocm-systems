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
    corrected = max(0, exec - (max launch_wait over the comm's ranks - own)).
    This assumes the ranks reach the launch together, as after the app's
    per-collective barrier (rccl-tests -z 1); records clamped at 0 show where
    that does not hold.

--trace writes Chrome trace JSON (chrome://tracing, ui.perfetto.dev): one
process per host, one thread per rank process, a slice per counter window and
per read, and a per-rank NIC counter track.
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
# Volume, including AINIC's hardware CTS control traffic: never a symptom.
VOLUME_ALL = VOLUME + ('tx_rdma_ucast_pkts', 'rx_rdma_ucast_pkts',
                       'tx_rdma_ccl_cts_bytes', 'tx_rdma_ccl_cts_pkts',
                       'rx_rdma_ccl_cts_bytes', 'rx_rdma_ccl_cts_pkts')
MIN_USABLE = 3

FILE_RE = re.compile(r'accl_profiler_rank(\d+)_(.+)_pid(\d+)_(0x[0-9a-fA-F]+)\.jsonl$')


@dataclass
class Rec:
    rank: int
    n_ranks: int
    host: str
    pid: int
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
    # non-zero counters on neither list: shown, never drive the verdict
    other: Dict[str, Tuple[int, int, List[int]]] = field(default_factory=dict)
    pause_rx: int = 0
    pause_tx: int = 0
    n_no_net: int = 0
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
            files.extend(sorted(glob.glob(os.path.join(p, '**', '*.jsonl'), recursive=True)))
        else:
            files.append(p)
    return files


def load(paths: List[str], warmup: int = 5) -> Tuple[List[Rec], List[dict]]:
    """Records and summary lines from every file. Drops the first `warmup`
    sequence numbers of each (comm, coll, size), the same ones on every rank, so
    skew groups stay whole."""
    recs: List[Rec] = []
    summaries: List[dict] = []
    for path in expand_inputs(paths):
        m = FILE_RE.search(os.path.basename(path))
        host, pid, comm = (m.group(2), int(m.group(3)), m.group(4)) if m else ('?', 0, path)
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
                recs.append(Rec(rank=hdr.get('rank', 0), n_ranks=hdr.get('n_ranks', 1),
                                host=host, pid=pid, comm=comm,
                                coll=cp.get('coll', ''), sn=cp.get('coll_sn', 0),
                                size=cp.get('coll_msg_size_bytes', 0),
                                exec_us=cp.get('coll_exec_time_us', 0.0),
                                net=obj.get('net_counters')))
    if warmup <= 0:
        return recs, summaries
    sns: Dict[Tuple[str, str, int], set] = defaultdict(set)
    for r in recs:
        sns[(r.comm, r.coll, r.size)].add(r.sn)
    drop = {k: set(sorted(v)[:warmup]) for k, v in sns.items()}
    emptied = [k for k, v in sns.items() if len(v) <= warmup]
    if emptied:
        print(f"WARNING: warmup={warmup} consumed every record of {len(emptied)} "
              f"(comm, coll, size) group(s)", file=sys.stderr)
    return [r for r in recs if r.sn not in drop[(r.comm, r.coll, r.size)]], summaries


def rdma_deltas(net: dict) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for dev in net.get('devices') or []:
        for k, v in (dev.get('rdma') or {}).items():
            out[k] += v
    return out


def pause_frames(net: dict) -> Tuple[int, int]:
    """(rx, tx) pause/PFC frame counts; durations and transitions excluded."""
    rx = tx = 0
    for dev in net.get('devices') or []:
        for k, v in (dev.get('ethtool') or {}).items():
            if k == 'stale' or not ('pause' in k or 'pfc' in k):
                continue
            if any(x in k for x in ('us_count', 'duration', 'transition')):
                continue
            if 'rx' in k:
                rx += v
            elif 'tx' in k:
                tx += v
    return rx, tx


def usable(r: Rec) -> bool:
    return bool(r.net) and not r.net.get('overlapped') and not r.net.get('unavailable')


def _hits(deltas, name):
    hits = [(r.rank, d[name]) for r, d in deltas if d.get(name, 0) > 0]
    if not hits:
        return None
    return (sum(v for _, v in hits), len(hits), sorted({rk for rk, _ in hits}))


def verdicts(recs: List[Rec]) -> Dict[Tuple[str, str, int], Verdict]:
    """Per (comm, coll, msg_size) verdict from clean windows only. Overlapped
    windows hold other work; ethtool deltas are stale on ionic (about 5 s
    refresh) and never drive a verdict. Every rank must contribute a clean
    window: loss is often visible only on the receiver."""
    groups: Dict[Tuple[str, str, int], List[Rec]] = defaultdict(list)
    for r in recs:
        groups[(r.comm, r.coll, r.size)].append(r)
    out = {}
    for key, all_rs in sorted(groups.items()):
        rs = [r for r in all_rs if r.net is not None]
        if not rs:
            continue
        good = [r for r in rs if usable(r)]
        n_over = sum(1 for r in rs if r.net.get('overlapped'))
        n_unav = sum(1 for r in rs if r.net.get('unavailable'))
        deltas = [(r, rdma_deltas(r.net)) for r in good]
        drivers = {}
        for name in list(NET_LOSS_COUNTERS) + list(NET_CONGESTION_COUNTERS):
            h = _hits(deltas, name)
            if h:
                drivers[name] = h
        other = {}
        for name in sorted({k for _, d in deltas for k in d}):
            if name in NET_LOSS_COUNTERS or name in NET_CONGESTION_COUNTERS or name in VOLUME_ALL:
                continue
            h = _hits(deltas, name)
            if h:
                other[name] = h
        tx = sum(d.get('tx_rdma_ucast_bytes', 0) for _, d in deltas)
        traffic = sum(d.get(k, 0) for _, d in deltas for k in VOLUME)
        known = any('tx_rdma_ucast_bytes' in d for _, d in deltas)
        pauses = [pause_frames(r.net) for r in good]
        v = Verdict('CLEAN', len(rs), len(good), n_over, n_unav,
                    tx / len(good) if good else 0.0, drivers, other,
                    sum(p[0] for p in pauses), sum(p[1] for p in pauses),
                    len(all_rs) - len(rs))
        loss = [k for k in drivers if k in NET_LOSS_COUNTERS]
        unseen = sorted({(r.host, r.rank) for r in rs} - {(r.host, r.rank) for r in good})
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
        elif unseen:
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'no clean window on rank(s) ' + ','.join(str(rk) for _, rk in unseen)
        elif traffic == 0:
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'no NIC traffic in the windows'
        out[key] = v
    return out


def correct_skew(recs: List[Rec]) -> None:
    """Sets corrected_us. Ranks of one comm launch collective sn after their own
    read; a rank whose read ended earlier waits inside its kernel for the
    slowest one, so that difference is taken back out. Only whole groups with
    no timed-out read: a timed-out wait is not skew."""
    groups: Dict[Tuple[str, str, int], List[Rec]] = defaultdict(list)
    for r in recs:
        if r.net and 'launch_wait_us' in r.net:
            groups[(r.comm, r.coll, r.sn)].append(r)
    for rs in groups.values():
        if len(rs) < rs[0].n_ranks or any(r.net.get('unavailable') for r in rs):
            continue
        latest = max(r.net['launch_wait_us'] for r in rs)
        for r in rs:
            r.corrected_us = max(0.0, r.exec_us - (latest - r.net['launch_wait_us']))


def print_summaries(summaries: List[dict], have_records: bool) -> None:
    nets = [s['net_counters'] for s in summaries if 'net_counters' in s]
    if not nets:
        if have_records:
            print("No summary line with net_counters (did the run reach finalize?)")
        else:
            print("No net_counters in any record (run with ACCL_NET_COUNTERS=1).")
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
    if tot['reads']:
        print(f"Counter reads: {tot['reads']}, mean {read_mean:.0f} us, "
              f"max {max(n.get('read_us_max', 0) for n in nets)} us")
    incomplete = sum(1 for s in summaries if not s.get('complete', True))
    if incomplete:
        print(f"WARNING: {incomplete} file(s) report complete=false", file=sys.stderr)


def print_verdicts(vs: Dict[Tuple[str, str, int], Verdict]) -> None:
    multi = len({k[0] for k in vs}) > 1
    print(f"\n{'='*140}")
    print("  NETWORK VERDICT per message size "
          "(clean windows only; ethtool pause frames rx/tx are stale context)")
    print(f"{'='*140}")
    print(f"{'Coll':>14s} {'Size':>8s} │ {'Usable':>9s} │ {'Verdict':>15s} │ "
          f"{'TxMB/coll':>10s} {'Pause*rx/tx':>13s} │ Drivers (sum, records hit, side, ranks)")
    for (comm, coll, sz), v in vs.items():
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
        if v.other:
            detail += ' | other: ' + ', '.join(
                f"{k}={t} ({h}/{v.n_usable}, ranks {','.join(map(str, rk))})"
                for k, (t, h, rk) in v.other.items())
        if v.n_no_net:
            detail += f' | {v.n_no_net} record(s) without net_counters'
        label = f"{comm} {coll}" if multi else coll
        print(f"{label:>14s} {fmt_size(sz):>8s} │ {v.n_usable:>4d}/{v.n_records:<4d} │ "
              f"{v.verdict:>15s} │ {v.tx_bytes_per_coll / 1e6:10.3f} "
              f"{f'{v.pause_rx}/{v.pause_tx}':>13s} │ {detail}")


CLAMP_WARN = 0.05


def print_timing(recs: List[Rec]) -> None:
    groups: Dict[Tuple[str, str, int], List[Rec]] = defaultdict(list)
    for r in recs:
        if r.corrected_us is not None:
            groups[(r.comm, r.coll, r.size)].append(r)
    if not groups:
        return
    multi = len({k[0] for k in groups}) > 1
    print(f"\n{'='*140}")
    print("  EXEC TIME with counter-read skew removed (us, medians; skew = exec - corrected).")
    print("  Assumes ranks reach each launch together (per-collective barrier, e.g. "
          "rccl-tests -z 1); '!' marks sizes where it does not hold.")
    print(f"{'='*140}")
    print(f"{'Coll':>14s} {'Size':>8s} │ {'n':>5s} {'clamped':>7s} │ {'ExecRaw':>10s} {'ExecCorr':>10s} │ "
          f"{'Skew':>9s} {'SkewP99':>9s} {'SkewMax':>9s} │ {'LaunchWait':>10s}")
    for (comm, coll, sz), rs in sorted(groups.items()):
        skew = sorted(r.exec_us - r.corrected_us for r in rs)
        clamped = sum(1 for r in rs if r.corrected_us == 0.0 and r.exec_us > 0)
        mark = '!' if clamped > CLAMP_WARN * len(rs) else ' '
        label = f"{comm} {coll}" if multi else coll
        print(f"{label:>14s} {fmt_size(sz):>8s} │ {len(rs):5d} {clamped:6d}{mark} │ "
              f"{statistics.median(r.exec_us for r in rs):10.1f} "
              f"{statistics.median(r.corrected_us for r in rs):10.1f} │ "
              f"{statistics.median(skew):9.1f} {skew[int(0.99 * (len(skew) - 1))]:9.1f} "
              f"{skew[-1]:9.1f} │ {statistics.median(r.net['launch_wait_us'] for r in rs):10.1f}")


def build_trace(recs: List[Rec]) -> dict:
    hosts = sorted({r.host for r in recs})
    pid = {h: i + 1 for i, h in enumerate(hosts)}
    # One thread per rank process and comm; the OS pid keeps comms apart.
    tids: Dict[Tuple[str, int, str, int], int] = {}
    for r in sorted(recs, key=lambda r: (r.host, r.rank, r.comm, r.pid)):
        tids.setdefault((r.host, r.pid, r.comm, r.rank), len(tids) + 1)
    multi = len({r.comm for r in recs}) > 1
    events = []
    for h in hosts:
        events.append({"ph": "M", "name": "process_name", "pid": pid[h],
                       "args": {"name": h}})
    for (h, _, comm, rank), tid in tids.items():
        events.append({"ph": "M", "name": "thread_name", "pid": pid[h], "tid": tid,
                       "args": {"name": f"rank {rank}" + (f" {comm}" if multi else "")}})
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
        tid = tids[(r.host, r.pid, r.comm, r.rank)]
        events.append({"ph": "X", "cat": cat, "name": f"{r.coll} {fmt_size(r.size)}",
                       "pid": pid[r.host], "tid": tid, "ts": n['before_ts_us'],
                       "dur": max(0, n['after_ts_us'] - n['before_ts_us']), "args": args})
        events.append({"ph": "X", "cat": "read", "name": "counter read",
                       "pid": pid[r.host], "tid": tid, "ts": n['after_ts_us'],
                       "dur": n.get('after_read_us', 0)})
        events.append({"ph": "C", "name": f"rank {r.rank} NIC" + (f" {r.comm}" if multi else ""),
                       "pid": pid[r.host],
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
    print_summaries(summaries, bool(with_net))
    if not with_net:
        return 1
    correct_skew(with_net)
    print_verdicts(verdicts(recs))
    print_timing(with_net)
    if args.trace:
        with open(args.trace, 'w', encoding='utf-8') as f:
            json.dump(build_trace(with_net), f)
        print(f"\nTrace written to {args.trace}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
