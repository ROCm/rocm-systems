#!/usr/bin/env python3
"""ACCL NIC counter report and Chrome/Perfetto trace.

Only coordinated windows with verified proxy registration are eligible for
CLEAN. Every rank must contribute at least three complete collective windows.
Unknown non-volume counter changes produce INCONCLUSIVE.

Trace lanes distinguish host-observed collective activity, whole-collective
NIC deltas, and deduplicated raw snapshots. GPU duration is a detail field;
CPU and GPU clocks are not implicitly aligned. Host clocks remain independent.
AINIC hardware ucast byte counters include CTS control traffic.
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

# The counters shown, with their category and the side of a transfer whose NIC
# counts them (ionic names, then mlx5). Drops show up on the receiver
# (out-of-buffer, out-of-sequence) and as NAKs/timeouts on the sender, so a
# verdict pools every rank of a size.
DROP_COUNTERS = {
    'resp_rx_outof_buf': ('OOB', 'receiver'),
    'out_of_buffer': ('OOB', 'receiver'),
    'req_rx_rnr_retry_err': ('RNR', 'sender'),
    'resp_tx_rnr_retry_err': ('RNR', 'receiver'),
    'rnr_nak_retry_err': ('RNR', 'sender'),
    'req_rx_pkt_seq_err': ('NAK', 'sender, NAK received'),
    'req_rx_impl_nak_seq_err': ('NAK', 'sender, NAK received'),
    'resp_tx_pkt_seq_err': ('NAK', 'receiver, NAK sent'),
    'packet_seq_err': ('NAK', 'sender, NAK received'),
    'resp_rx_outouf_seq': ('OOS', 'receiver'),
    'resp_rx_ccl_cts_outouf_seq': ('OOS', 'sender, CTS'),
    'out_of_sequence': ('OOS', 'receiver'),
    'tx_rdma_ack_timeout': ('ACK timeout', 'sender'),
    'tx_rdma_ccl_cts_ack_timeout': ('ACK timeout', 'receiver, CTS'),
    'local_ack_timeout_err': ('ACK timeout', 'sender'),
}
CONGESTION_COUNTERS = {
    'rx_rdma_ecn_pkts': ('ECN', 'receiver'),
    'np_ecn_marked_roce_packets': ('ECN', 'receiver'),
    'tx_rdma_cnp_pkts': ('CNP sent', 'receiver'),
    'np_cnp_sent': ('CNP sent', 'receiver'),
    'rx_rdma_cnp_pkts': ('CNP received', 'sender'),
    'rp_cnp_handled': ('CNP received', 'sender'),
}
NET_LOSS_COUNTERS = {k: f'{c}, {side}' for k, (c, side) in DROP_COUNTERS.items()}
NET_CONGESTION_COUNTERS = {k: f'{c}, {side}' for k, (c, side) in CONGESTION_COUNTERS.items()}
VOLUME = ('tx_rdma_ucast_bytes', 'rx_rdma_ucast_bytes')
# Volume, including AINIC's hardware CTS control traffic: never a symptom.
VOLUME_ALL = VOLUME + ('tx_rdma_ucast_pkts', 'rx_rdma_ucast_pkts',
                       'tx_rdma_ccl_cts_bytes', 'tx_rdma_ccl_cts_pkts',
                       'rx_rdma_ccl_cts_bytes', 'rx_rdma_ccl_cts_pkts')
# PFC ethtool stats per priority. ionic: frames_{rx,tx}_pri_N and
# {rx,tx}_pripause_N_1us_count; mlx5: {rx,tx}_prioN_pause(_duration).
PFC_FRAMES_RE = re.compile(r'^(?:frames_(rx|tx)_pri_?(\d)|(rx|tx)_prio(\d)_pause)$')
PFC_TIME_RE = re.compile(r'^(?:(rx|tx)_pripause_(\d)_1us_count|(rx|tx)_prio(\d)_pause_duration)$')
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


def pfc_key(name: str, pattern) -> Optional[Tuple[str, int]]:
    """('rx'|'tx', priority) for a per-priority PFC stat, else None."""
    m = pattern.match(name)
    if not m:
        return None
    direction = m.group(1) or m.group(3)
    return direction, int(m.group(2) or m.group(4))


def pfc_frames(net: dict) -> Dict[Tuple[str, int], int]:
    """Per-window PFC frames by (direction, priority). Stale on ionic."""
    out: Dict[Tuple[str, int], int] = defaultdict(int)
    for dev in net.get('devices') or []:
        for k, v in (dev.get('ethtool') or {}).items():
            key = pfc_key(k, PFC_FRAMES_RE)
            if key and v:
                out[key] += v
    return out


def pause_frames(net: dict) -> Tuple[int, int]:
    """(rx, tx) PFC frames over all priorities."""
    f = pfc_frames(net)
    return (sum(v for (d, _), v in f.items() if d == 'rx'),
            sum(v for (d, _), v in f.items() if d == 'tx'))


def pfc_pause_summary(recs: List[Rec]) -> List[dict]:
    """Whole-run PFC per (host, device, priority) from the first and last raw
    snapshot of each device, so ionic's ~5 s ethtool refresh averages out.
    rx pause time: this host's NIC was paused by its switch port. tx pause
    time: this NIC paused its peer. Percentages are of the snapshot span."""
    first: Dict[Tuple[str, str], Tuple[int, dict]] = {}
    last: Dict[Tuple[str, str], Tuple[int, dict]] = {}
    for r in recs:
        if not r.net:
            continue
        for snap in (r.net.get('snapshot_before'), r.net.get('snapshot_after')):
            for dev in (snap or {}).get('devices') or []:
                eth = dev.get('ethtool')
                if not eth:
                    continue
                key = (r.host, dev['name'])
                if key not in first or dev['read_start_us'] < first[key][0]:
                    first[key] = (dev['read_start_us'], eth)
                if key not in last or dev['read_end_us'] > last[key][0]:
                    last[key] = (dev['read_end_us'], eth)
    rows = []
    for key in sorted(first):
        (t0, a), (t1, b) = first[key], last[key]
        span = t1 - t0
        per: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for name, v in b.items():
            delta = v - a.get(name, v)
            if delta <= 0:
                continue
            fk, tk = pfc_key(name, PFC_FRAMES_RE), pfc_key(name, PFC_TIME_RE)
            if fk:
                per[fk[1]][f'{fk[0]}_frames'] += delta
            elif tk:
                per[tk[1]][f'{tk[0]}_pause_us'] += delta
        for prio, c in sorted(per.items()):
            rows.append({'host': key[0], 'device': key[1], 'priority': prio, 'span_us': span,
                         'rx_frames': c['rx_frames'], 'tx_frames': c['tx_frames'],
                         'rx_pause_us': c['rx_pause_us'], 'tx_pause_us': c['tx_pause_us'],
                         'host_paused_pct': 100.0 * c['rx_pause_us'] / span if span > 0 else 0.0,
                         'peer_paused_pct': 100.0 * c['tx_pause_us'] / span if span > 0 else 0.0})
    return rows


def print_pfc(recs: List[Rec]) -> None:
    rows = pfc_pause_summary(recs)
    if not rows:
        return
    print(f"\n{'='*140}")
    print("  PFC per priority, whole run (first to last snapshot of each NIC)")
    print("  Host paused: pause frames received (rx pause time). Peer paused by host: "
          "pause frames sent (tx pause time).")
    print(f"{'='*140}")
    print(f"{'Host':>20s} {'Device':>8s} {'Prio':>4s} | {'RxFrames':>10s} {'TxFrames':>10s} | "
          f"{'HostPaused%':>11s} {'PeerPaused%':>11s} | {'Span_s':>8s}")
    for row in rows:
        print(f"{row['host']:>20s} {row['device']:>8s} {row['priority']:>4d} | "
              f"{row['rx_frames']:10d} {row['tx_frames']:10d} | "
              f"{row['host_paused_pct']:10.3f}% {row['peer_paused_pct']:10.3f}% | "
              f"{row['span_us'] / 1e6:8.2f}")


def measured(r: Rec) -> bool:
    """A delta that belongs to this window alone (passive or strict)."""
    return (bool(r.net) and not r.net.get('overlapped') and not r.net.get('unavailable')
            and not r.net.get('counter_discontinuities'))


def usable(r: Rec) -> bool:
    """Measured, completion-verified and coordinated: eligible for CLEAN."""
    return (measured(r) and r.net.get('completion_verified', False)
            and r.net.get('coordinated', False))


def _hits(deltas, name):
    hits = [(r.rank, d[name]) for r, d in deltas if d.get(name, 0) > 0]
    if not hits:
        return None
    return (sum(v for _, v in hits), len(hits), sorted({rk for rk, _ in hits}))


def verdicts(recs: List[Rec]) -> Dict[Tuple[str, str, int], Verdict]:
    """Per (comm, coll, msg_size) verdict. Loss or congestion in any measured
    window (passive mode included) is NETWORK_SUSPECT. CLEAN needs coordinated,
    completion-verified windows (ACCL_NET_STRICT=1) from every rank: loss is
    often visible only on the receiver. Overlapped windows hold other work;
    ethtool deltas are stale on ionic and never drive a verdict."""
    groups: Dict[Tuple[str, str, int], List[Rec]] = defaultdict(list)
    for r in recs:
        groups[(r.comm, r.coll, r.size)].append(r)
    out = {}
    for key, all_rs in sorted(groups.items()):
        rs = [r for r in all_rs if r.net is not None]
        if not rs:
            continue
        seen = [r for r in rs if measured(r)]
        good = [r for r in seen if usable(r)]
        n_over = sum(1 for r in rs if r.net.get('overlapped'))
        n_unav = sum(1 for r in rs if r.net.get('unavailable'))
        deltas = [(r, rdma_deltas(r.net)) for r in seen]
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
        pauses = [pause_frames(r.net) for r in seen]
        v = Verdict('CLEAN', len(rs), len(seen), n_over, n_unav,
                    tx / len(seen) if seen else 0.0, drivers, other,
                    sum(p[0] for p in pauses), sum(p[1] for p in pauses),
                    len(all_rs) - len(rs))
        loss = [k for k in drivers if k in NET_LOSS_COUNTERS]
        expected = set(range(max(r.n_ranks for r in all_rs)))
        unseen = sorted(expected - {r.rank for r in good})
        by_sn = defaultdict(set)
        for r in good:
            by_sn[r.sn].add(r.rank)
        complete_sns = sum(expected <= ranks for ranks in by_sn.values())
        passive = '' if all(usable(r) for r, d in deltas
                            if any(d.get(k, 0) > 0 for k in drivers)) else ' (uncoordinated windows)'
        if len(seen) < MIN_USABLE or 2 * len(seen) < len(rs):
            v.verdict = 'INCONCLUSIVE'
            v.reason = f"{len(seen)}/{len(rs)} usable ({n_over} overlapped, {n_unav} unavailable)"
        elif not known and not drivers:
            # Counter names are ionic's; another driver would read as clean.
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'unknown counter names (not ionic)'
        elif loss:
            v.verdict = 'NETWORK_SUSPECT'
            v.reason = 'loss/retransmit' + passive
        elif drivers:
            v.verdict = 'NETWORK_SUSPECT'
            v.reason = 'congestion signalled (ECN/CNP), no loss' + passive
        elif not good:
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'no symptoms, but CLEAN needs coordinated windows (ACCL_NET_STRICT=1)'
        elif unseen:
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'no clean window on rank(s) ' + ','.join(str(rk) for rk in unseen)
        elif complete_sns < MIN_USABLE:
            v.verdict = 'INCONCLUSIVE'
            v.reason = f'only {complete_sns} complete collective(s) across all {len(expected)} ranks'
        elif other:
            v.verdict = 'INCONCLUSIVE'
            v.reason = 'unclassified non-volume counters changed; inspect the listed counters'
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
          "(PFC frames rx/tx are stale context: ionic refreshes them about every 5 s)")
    print(f"{'='*140}")
    print(f"{'Coll':>14s} {'Size':>8s} | {'Usable':>9s} | {'Verdict':>15s} | "
          f"{'HW-TxMB/coll':>12s} {'PFC rx/tx':>13s} | Drivers (sum, records hit, category, side, ranks)")
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
        print(f"{label:>14s} {fmt_size(sz):>8s} | {v.n_usable:>4d}/{v.n_records:<4d} | "
              f"{v.verdict:>15s} | {v.tx_bytes_per_coll / 1e6:10.3f} "
              f"{f'{v.pause_rx}/{v.pause_tx}':>13s} | {detail}")


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
    print(f"{'Coll':>14s} {'Size':>8s} | {'n':>5s} {'clamped':>7s} | {'ExecRaw':>10s} {'ExecCorr':>10s} | "
          f"{'Skew':>9s} {'SkewP99':>9s} {'SkewMax':>9s} | {'LaunchWait':>10s}")
    for (comm, coll, sz), rs in sorted(groups.items()):
        skew = sorted(r.exec_us - r.corrected_us for r in rs)
        clamped = sum(1 for r in rs if r.corrected_us == 0.0 and r.exec_us > 0)
        mark = '!' if clamped > CLAMP_WARN * len(rs) else ' '
        label = f"{comm} {coll}" if multi else coll
        print(f"{label:>14s} {fmt_size(sz):>8s} | {len(rs):5d} {clamped:6d}{mark} | "
              f"{statistics.median(r.exec_us for r in rs):10.1f} "
              f"{statistics.median(r.corrected_us for r in rs):10.1f} | "
              f"{statistics.median(skew):9.1f} {skew[int(0.99 * (len(skew) - 1))]:9.1f} "
              f"{skew[-1]:9.1f} | {statistics.median(r.net['launch_wait_us'] for r in rs):10.1f}")


def build_trace(recs: List[Rec]) -> dict:
    """Host-observed activity, attribution windows, and deduplicated reads.

    Hardware deltas annotate the entire collective; counter plots contain
    interval totals, never an instantaneous rate or a point-in-time diagnosis.
    Host clocks are deliberately not presented as synchronized across machines.
    """
    hosts = sorted({r.host for r in recs})
    pids = {h: i + 1 for i, h in enumerate(hosts)}
    keys = sorted({(r.host, r.pid, r.comm, r.rank) for r in recs})
    tids = {key: 10 * i + 1 for i, key in enumerate(keys)}
    origin = {}
    for h in hosts:
        stamps = [r.net['before_ts_us'] for r in recs
                  if r.host == h and r.net and 'before_ts_us' in r.net]
        origin[h] = min(stamps) if stamps else 0
    multi_comm = len({r.comm for r in recs}) > 1
    events = []
    for h in hosts:
        events.append({'ph': 'M', 'name': 'process_name', 'pid': pids[h],
                       'args': {'name': h + ' (host-local clock)'}})
    for h, os_pid, comm, rank in keys:
        base = tids[(h, os_pid, comm, rank)]
        for lane, title in enumerate(('Collective (host observed)',
                                      'Counter window (includes idle)', 'Snapshot reads')):
            events.append({'ph': 'M', 'name': 'thread_name', 'pid': pids[h], 'tid': base + lane,
                           'args': {'name': f'rank {rank} · {title}' + (f' · {comm}' if multi_comm else '')}})
            events.append({'ph': 'M', 'name': 'thread_sort_index', 'pid': pids[h], 'tid': base + lane,
                           'args': {'sort_index': rank * 10 + lane}})
    seen_reads = set()
    # Series each rank's symptom tracks need: only what the run ever saw.
    has_drops: Dict[tuple, set] = defaultdict(set)
    has_cong: Dict[tuple, set] = defaultdict(set)
    has_pfc: Dict[tuple, set] = defaultdict(set)
    for r in recs:
        if r.net and measured(r):
            key = (r.host, r.pid, r.comm, r.rank)
            for k, v in rdma_deltas(r.net).items():
                if v and k in DROP_COUNTERS:
                    has_drops[key].add(DROP_COUNTERS[k][0])
                if v and k in CONGESTION_COUNTERS:
                    has_cong[key].add(CONGESTION_COUNTERS[k][0])
            for (dr, p) in pfc_frames(r.net):
                has_pfc[key].add(f'{dr} prio {p}')
    for r in recs:
        n = r.net
        if not n or 'before_ts_us' not in n:
            continue
        base = tids[(r.host, r.pid, r.comm, r.rank)]
        pid, zero = pids[r.host], origin[r.host]
        d = rdma_deltas(n)
        symptoms = {k: v for k, v in d.items()
                    if v and (k in NET_LOSS_COUNTERS or k in NET_CONGESTION_COUNTERS)}
        unclassified = {k: v for k, v in d.items() if v and k not in NET_LOSS_COUNTERS and k not in NET_CONGESTION_COUNTERS and k not in VOLUME_ALL}
        quality = 'overlapped' if n.get('overlapped') else (
            'unavailable' if not measured(r) else ('suspect' if symptoms else (
                'unclassified' if unclassified else ('window' if usable(r) else 'uncoordinated'))))
        color = 'grey' if quality in ('overlapped', 'unavailable') else (
            'terrible' if symptoms else ('bad' if unclassified else ('good' if quality == 'window' else 'yellow')))
        pfc = pfc_frames(n)
        drops = {f'{DROP_COUNTERS[k][0]} ({k})': v for k, v in sorted(d.items())
                 if v and k in DROP_COUNTERS}
        congestion = {f'{CONGESTION_COUNTERS[k][0]} ({k})': v for k, v in sorted(d.items())
                      if v and k in CONGESTION_COUNTERS}
        args = {'tx_bytes': d.get('tx_rdma_ucast_bytes'),
                'rx_bytes': d.get('rx_rdma_ucast_bytes'),
                **({'drops': drops} if drops else {}),
                **({'ecn_cnp': congestion} if congestion else {}),
                **({'pfc_frames (stale)': {f'{dr} prio {p}': v for (dr, p), v in sorted(pfc.items())}}
                   if pfc else {}),
                'sn': r.sn, 'rank': r.rank, 'message_bytes': r.size,
                'gpu_exec_us': r.exec_us, 'launch_wait_us': n.get('launch_wait_us', 0),
                'window': quality, 'overlapped': n.get('overlapped'),
                'unavailable': n.get('unavailable'),
                'completion_verified': n.get('completion_verified', False),
                'coordinated': n.get('coordinated', False)}
        if r.corrected_us is not None:
            args['exec_corrected_us'] = round(r.corrected_us, 2)
            args['correction_warning'] = 'heuristic estimate; not a recovered measurement'
        name = f'{r.coll} {fmt_size(r.size)} · #{r.sn}'
        before, after = n.get('snapshot_before'), n.get('snapshot_after')
        left = before['read_end_us'] if before else n['before_ts_us']
        right = after['read_start_us'] if after else n['after_ts_us']
        events.append({'ph': 'X', 'cat': quality, 'name': name,
                       'pid': pid, 'tid': base + 1, 'ts': left - zero,
                       'dur': max(0, right - left), 'cname': color, 'args': args})
        begin, finish = n.get('collective_start_cpu_us'), n.get('collective_complete_cpu_us')
        if begin is not None and finish is not None:
            events.append({'ph': 'X', 'cat': 'collective', 'name': name,
                           'pid': pid, 'tid': base, 'ts': begin - zero,
                           'dur': max(0, finish - begin), 'cname': color,
                           'args': {**args, 'placement': 'host callbacks: Coll start to observed GPU+proxy completion'}})
        snapshots = [x for x in (before, after) if x]
        if not snapshots:
            snapshots = [{'id': ('legacy', r.sn), 'read_start_us': n['after_ts_us'],
                          'read_end_us': n['after_ts_us'] + n.get('after_read_us', 0)}]
        for snap in snapshots:
            key = (r.host, r.pid, r.comm, str(snap['id']))
            if key in seen_reads:
                continue
            seen_reads.add(key)
            events.append({'ph': 'X', 'cat': 'read', 'name': f'Counter snapshot S{snap["id"]}',
                           'pid': pid, 'tid': base + 2, 'ts': snap['read_start_us'] - zero,
                           'dur': max(0, snap['read_end_us'] - snap['read_start_us']),
                           'cname': 'thread_state_iowait', 'args': snap})
        if measured(r):
            ts = (after['read_end_us'] if after else n['after_ts_us']) - zero
            track = f'rank {r.rank}' + (f' · {r.comm}' if multi_comm else '')
            events.append({'ph': 'C', 'name': f'{track} MB per collective', 'pid': pid, 'ts': ts,
                           'args': {'tx': d.get('tx_rdma_ucast_bytes', 0) / 1e6,
                                    'rx': d.get('rx_rdma_ucast_bytes', 0) / 1e6}})
            # Symptom tracks only for ranks where the run ever saw one.
            key = (r.host, r.pid, r.comm, r.rank)
            for title, cats, has in (('drops', DROP_COUNTERS, has_drops), ('ECN/CNP', CONGESTION_COUNTERS, has_cong)):
                if has.get(key):
                    per_cat = {c: 0 for c in sorted(has[key])}
                    for k, v in d.items():
                        if k in cats:
                            per_cat[cats[k][0]] += v
                    events.append({'ph': 'C', 'name': f'{track} {title}', 'pid': pid, 'ts': ts,
                                   'args': per_cat})
            if has_pfc.get(key):
                series = {name: 0 for name in sorted(has_pfc[key])}
                for (dr, p), v in pfc.items():
                    series[f'{dr} prio {p}'] = v
                events.append({'ph': 'C', 'name': f'{track} PFC frames (stale)', 'pid': pid, 'ts': ts,
                               'args': series})
    return {'traceEvents': events, 'displayTimeUnit': 'ms',
            'otherData': {'clock': 'CLOCK_MONOTONIC, normalized independently per host; hosts are NOT synchronized',
                          'host_origins_us': origin,
                          'activity': 'host observed; exact GPU duration is a detail field',
                          'counter_tracks': 'per-collective interval totals; not rates'}}


def print_raw_timing(recs):
    groups = defaultdict(list)
    for r in recs:
        groups[(r.comm, r.coll, r.size)].append(r)
    print('\nEXEC TIME (measured GPU duration; no skew subtraction), medians in us')
    for (comm, coll, size), rs in sorted(groups.items()):
        waits = [r.net.get('launch_wait_us', 0) for r in rs if r.net]
        print(f'{comm} {coll:>14} {fmt_size(size):>8}  GPU={statistics.median(r.exec_us for r in rs):.2f}'
              f'  launch_wait={statistics.median(waits) if waits else 0:.2f}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ACCL network-counter report")
    ap.add_argument('inputs', nargs='+', help='ACCL output dirs or .jsonl files')
    ap.add_argument('--warmup', type=int, default=5,
                    help='records to drop per (file, coll, size) (default 5)')
    ap.add_argument('--trace', metavar='FILE', help='also write Chrome trace JSON')
    ap.add_argument('--estimate-read-skew', action='store_true', help='also show a heuristic timing estimate (requires simultaneous arrival)')
    args = ap.parse_args(argv)

    recs, summaries = load(args.inputs, args.warmup)
    with_net = [r for r in recs if r.net is not None]
    print(f"Loaded {len(recs)} records, {len(with_net)} with net_counters")
    print_summaries(summaries, bool(with_net))
    if not with_net:
        return 1
    if args.estimate_read_skew:
        print('WARNING: skew subtraction is an estimate, not a recovered measurement.', file=sys.stderr)
        correct_skew(with_net)
    print_raw_timing(with_net)
    print_verdicts(verdicts(recs))
    print_pfc(with_net)
    print_timing(with_net)
    if args.trace:
        with open(args.trace, 'w', encoding='utf-8') as f:
            json.dump(build_trace(with_net), f)
        print(f"\nTrace written to {args.trace}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
