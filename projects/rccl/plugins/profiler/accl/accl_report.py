#!/usr/bin/env python3
"""ACCL Profiler — triage report with timing decomposition.

Usage:
    # Compare baseline vs candidate
    python3 accl_report.py compare --baseline base.jsonl --candidate cand.jsonl

    # Single run analysis
    python3 accl_report.py single --input file.jsonl

    # Merge multi-rank files
    python3 accl_report.py compare --baseline dir/baseline/ --candidate dir/candidate/
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Record:
    coll: str
    sn: int
    msg_size: int
    rank: int
    n_ranks: int
    algo: str
    proto: str
    n_channels: int
    exec_time_us: float
    algobw_gbs: float
    busbw_gbs: float
    timing_source: str
    # Decomposition
    launch_overhead_us: float = 0
    gpu_kernel_avg_us: float = 0
    gpu_kernel_min_us: float = 0
    gpu_kernel_max_us: float = 0
    proxy_gpu_wait_us: float = 0
    proxy_network_us: float = 0
    proxy_peer_wait_us: float = 0
    proxy_flush_us: float = 0
    proxy_gpu_recv_wait_us: float = 0
    n_proxy_ops: int = 0
    n_send_ops: int = 0
    n_recv_ops: int = 0
    kernel_events: list = field(default_factory=list)


def parse_jsonl(filepath: str, warmup: int = 5) -> List[Record]:
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cp = obj.get('coll_perf')
            hdr = obj.get('header')
            if not cp or not hdr:
                continue

            sn = cp.get('coll_sn', 0)
            n_ranks = hdr.get('n_ranks', 1)
            if sn <= warmup * n_ranks:
                continue

            decomp = cp.get('decomposition', {})
            records.append(Record(
                coll=cp.get('coll', ''),
                sn=sn,
                msg_size=cp.get('coll_msg_size_bytes', 0),
                rank=hdr.get('rank', 0),
                n_ranks=n_ranks,
                algo=cp.get('coll_algo', ''),
                proto=cp.get('coll_proto', ''),
                n_channels=cp.get('coll_n_channels', 0),
                exec_time_us=cp.get('coll_exec_time_us', 0),
                algobw_gbs=cp.get('coll_algobw_gbs', 0),
                busbw_gbs=cp.get('coll_busbw_gbs', 0),
                timing_source=cp.get('coll_timing_source', ''),
                launch_overhead_us=decomp.get('launch_overhead_us', 0),
                gpu_kernel_avg_us=decomp.get('gpu_kernel_avg_us', 0),
                gpu_kernel_min_us=decomp.get('gpu_kernel_min_us', 0),
                gpu_kernel_max_us=decomp.get('gpu_kernel_max_us', 0),
                proxy_gpu_wait_us=decomp.get('proxy_gpu_wait_us', 0),
                proxy_network_us=decomp.get('proxy_network_us', 0),
                proxy_peer_wait_us=decomp.get('proxy_peer_wait_us', 0),
                proxy_flush_us=decomp.get('proxy_flush_us', 0),
                proxy_gpu_recv_wait_us=decomp.get('proxy_gpu_recv_wait_us', 0),
                n_proxy_ops=decomp.get('n_proxy_ops', 0),
                n_send_ops=decomp.get('n_send_ops', 0),
                n_recv_ops=decomp.get('n_recv_ops', 0),
            ))
    return records


def load_dir_or_file(path: str, warmup: int) -> List[Record]:
    if os.path.isdir(path):
        records = []
        for fn in sorted(os.listdir(path)):
            if fn.endswith('.jsonl'):
                records.extend(parse_jsonl(os.path.join(path, fn), warmup))
        return records
    return parse_jsonl(path, warmup)


def fmt_size(b: int) -> str:
    if b >= 1024*1024*1024:
        return f"{b/(1024*1024*1024):.0f}G"
    if b >= 1024*1024:
        return f"{b/(1024*1024):.0f}M"
    if b >= 1024:
        return f"{b/1024:.0f}K"
    return f"{b}B"


@dataclass
class SizeAgg:
    msg_size: int
    n_samples: int
    mean_exec_us: float
    mean_busbw: float
    mean_algobw: float
    # Decomposition means
    mean_launch_us: float
    mean_kernel_us: float
    mean_proxy_gpu_wait_us: float
    mean_proxy_network_us: float
    mean_proxy_peer_wait_us: float
    mean_proxy_flush_us: float
    mean_proxy_gpu_recv_wait_us: float
    mean_n_proxy_ops: float
    algo: str
    proto: str
    n_channels: int


def aggregate(records: List[Record]) -> Dict[Tuple[str, int], SizeAgg]:
    groups: Dict[Tuple[str, int], List[Record]] = defaultdict(list)
    for r in records:
        groups[(r.coll, r.msg_size)].append(r)

    result = {}
    for key, recs in sorted(groups.items()):
        n = len(recs)
        result[key] = SizeAgg(
            msg_size=key[1],
            n_samples=n,
            mean_exec_us=sum(r.exec_time_us for r in recs) / n,
            mean_busbw=sum(r.busbw_gbs for r in recs) / n,
            mean_algobw=sum(r.algobw_gbs for r in recs) / n,
            mean_launch_us=sum(r.launch_overhead_us for r in recs) / n,
            mean_kernel_us=sum(r.gpu_kernel_avg_us for r in recs) / n,
            mean_proxy_gpu_wait_us=sum(r.proxy_gpu_wait_us for r in recs) / n,
            mean_proxy_network_us=sum(r.proxy_network_us for r in recs) / n,
            mean_proxy_peer_wait_us=sum(r.proxy_peer_wait_us for r in recs) / n,
            mean_proxy_flush_us=sum(r.proxy_flush_us for r in recs) / n,
            mean_proxy_gpu_recv_wait_us=sum(r.proxy_gpu_recv_wait_us for r in recs) / n,
            mean_n_proxy_ops=sum(r.n_proxy_ops for r in recs) / n,
            algo=recs[0].algo,
            proto=recs[0].proto,
            n_channels=recs[0].n_channels,
        )
    return result


def classify_bottleneck(agg: SizeAgg) -> str:
    """Classify the dominant bottleneck for a size point."""
    total = agg.mean_exec_us + agg.mean_launch_us
    if total <= 0:
        return "unknown"

    if agg.mean_n_proxy_ops == 0:
        return "gpu-compute (no proxy)"

    proxy_total = (agg.mean_proxy_network_us + agg.mean_proxy_peer_wait_us +
                   agg.mean_proxy_flush_us + agg.mean_proxy_gpu_recv_wait_us +
                   agg.mean_proxy_gpu_wait_us)
    kernel_pct = agg.mean_kernel_us / total * 100
    proxy_pct = proxy_total / total * 100
    launch_pct = agg.mean_launch_us / total * 100

    if proxy_pct > 40:
        net_dom = agg.mean_proxy_network_us + agg.mean_proxy_peer_wait_us
        flush_dom = agg.mean_proxy_flush_us + agg.mean_proxy_gpu_recv_wait_us
        if net_dom > flush_dom:
            return "NETWORK"
        return "PROXY-FLUSH/GDR"
    if agg.mean_proxy_gpu_wait_us / total * 100 > 20:
        return "GPU-SCHEDULING"
    if launch_pct > 40:
        return "LAUNCH-OVERHEAD"
    if kernel_pct > 50:
        return "GPU-COMPUTE"
    return "mixed"


def print_single_report(records: List[Record]):
    aggs = aggregate(records)
    colls = sorted(set(k[0] for k in aggs.keys()))

    for coll in colls:
        print(f"\n{'='*140}")
        print(f"  {coll}")
        print(f"{'='*140}")
        print(f"{'Size':>8s} │ {'ExecUs':>10s} {'BusBW':>8s} │ "
              f"{'Kernel':>8s} {'PeerWait':>8s} {'Flush':>8s} {'GPUWait':>8s} {'Launch':>8s} │ "
              f"{'Algo':>8s} {'Proto':>8s} {'nCh':>4s} {'nProxy':>6s} │ Bottleneck")
        print(f"{'─'*8}─┼─{'─'*10}─{'─'*8}─┼─"
              f"{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}─{'─'*8}─┼─"
              f"{'─'*8}─{'─'*8}─{'─'*4}─{'─'*6}─┼─{'─'*20}")

        for (c, sz), agg in sorted(aggs.items()):
            if c != coll:
                continue
            bottleneck = classify_bottleneck(agg)
            print(f"{fmt_size(sz):>8s} │ "
                  f"{agg.mean_exec_us:10.1f} {agg.mean_busbw:8.4f} │ "
                  f"{agg.mean_kernel_us:8.1f} "
                  f"{agg.mean_proxy_peer_wait_us:8.1f} "
                  f"{agg.mean_proxy_flush_us:8.1f} "
                  f"{agg.mean_proxy_gpu_wait_us + agg.mean_proxy_gpu_recv_wait_us:8.1f} "
                  f"{agg.mean_launch_us:8.1f} │ "
                  f"{agg.algo:>8s} {agg.proto:>8s} {agg.n_channels:4d} "
                  f"{agg.mean_n_proxy_ops:6.0f} │ {bottleneck}")


def pct_delta(base: float, cand: float) -> str:
    if base == 0:
        return "    N/A"
    d = (cand - base) / base * 100
    return f"{d:+7.1f}%"


def print_compare_report(base_recs: List[Record], cand_recs: List[Record],
                         threshold: float = 10.0):
    base_aggs = aggregate(base_recs)
    cand_aggs = aggregate(cand_recs)
    colls = sorted(set(list(k[0] for k in base_aggs.keys()) +
                       list(k[0] for k in cand_aggs.keys())))

    summary_results = []

    for coll in colls:
        print(f"\n{'='*140}")
        print(f"  {coll}")
        print(f"{'='*140}")

        # Header
        print(f"{'Size':>8s} │ {'Exec(B)':>10s} {'Exec(C)':>10s} {'Δ%':>7s} │ "
              f"{'BusBW(B)':>10s} {'BusBW(C)':>10s} {'Δ%':>7s} │ "
              f"{'Kernel(B)':>10s} {'Kernel(C)':>10s} {'Δ%':>7s} │ "
              f"{'Net(B)':>8s} {'Net(C)':>8s} {'Δ%':>7s} │ Flag")
        print(f"{'─'*8}─┼─{'─'*10}─{'─'*10}─{'─'*7}─┼─"
              f"{'─'*10}─{'─'*10}─{'─'*7}─┼─"
              f"{'─'*10}─{'─'*10}─{'─'*7}─┼─"
              f"{'─'*8}─{'─'*8}─{'─'*7}─┼─{'─'*20}")

        coll_flags = []
        busbw_auc_b = 0
        busbw_auc_c = 0
        peak_bw_b = 0
        peak_bw_c = 0

        sizes = sorted(set(
            [k[1] for k in base_aggs if k[0] == coll] +
            [k[1] for k in cand_aggs if k[0] == coll]
        ))

        for sz in sizes:
            bkey = (coll, sz)
            b = base_aggs.get(bkey)
            c = cand_aggs.get(bkey)
            if not b or not c:
                continue

            exec_d = pct_delta(b.mean_exec_us, c.mean_exec_us)
            bw_d = pct_delta(b.mean_busbw, c.mean_busbw)
            kern_d = pct_delta(b.mean_kernel_us, c.mean_kernel_us)
            net_d = pct_delta(b.mean_proxy_network_us, c.mean_proxy_network_us) \
                    if b.mean_proxy_network_us > 0 else "    N/A"

            # Flag logic
            flag = ""
            if b.mean_busbw > 0:
                bw_change = (c.mean_busbw - b.mean_busbw) / b.mean_busbw * 100
                if bw_change < -threshold:
                    # Determine cause
                    if b.mean_proxy_network_us > 0 and c.mean_proxy_network_us > 0:
                        net_change = (c.mean_proxy_network_us - b.mean_proxy_network_us) / b.mean_proxy_network_us * 100
                        kern_change = (c.mean_kernel_us - b.mean_kernel_us) / b.mean_kernel_us * 100 if b.mean_kernel_us > 0 else 0
                        if net_change > 15:
                            flag = "<< NETWORK regression"
                        elif kern_change > 15:
                            flag = "<< GPU KERNEL regression"
                        else:
                            flag = "<< REGRESSION"
                    else:
                        flag = "<< REGRESSION"
                    coll_flags.append((sz, flag))
                elif bw_change < -5:
                    flag = "<< bw warning"

            # AUC
            busbw_auc_b += b.mean_busbw
            busbw_auc_c += c.mean_busbw
            if b.mean_busbw > peak_bw_b:
                peak_bw_b = b.mean_busbw
            if c.mean_busbw > peak_bw_c:
                peak_bw_c = c.mean_busbw

            print(f"{fmt_size(sz):>8s} │ "
                  f"{b.mean_exec_us:10.1f} {c.mean_exec_us:10.1f} {exec_d:>7s} │ "
                  f"{b.mean_busbw:10.4f} {c.mean_busbw:10.4f} {bw_d:>7s} │ "
                  f"{b.mean_kernel_us:10.1f} {c.mean_kernel_us:10.1f} {kern_d:>7s} │ "
                  f"{b.mean_proxy_network_us:8.1f} {c.mean_proxy_network_us:8.1f} {net_d:>7s} │ {flag}")

        # Decomposition comparison for flagged sizes
        if coll_flags:
            print(f"\n  Decomposition for flagged sizes:")
            print(f"  {'Size':>8s} │ {'Component':>16s} │ {'Base(us)':>10s} {'Cand(us)':>10s} {'Δ%':>8s} │ Diagnosis")
            print(f"  {'─'*8}─┼─{'─'*16}─┼─{'─'*10}─{'─'*10}─{'─'*8}─┼─{'─'*30}")
            for sz, _ in coll_flags:
                b = base_aggs.get((coll, sz))
                c = cand_aggs.get((coll, sz))
                if not b or not c:
                    continue
                components = [
                    ("Launch",   b.mean_launch_us, c.mean_launch_us),
                    ("GPU Kernel", b.mean_kernel_us, c.mean_kernel_us),
                    ("Network",  b.mean_proxy_network_us, c.mean_proxy_network_us),
                    ("GPU Wait", b.mean_proxy_gpu_wait_us, c.mean_proxy_gpu_wait_us),
                    ("Peer Wait", b.mean_proxy_peer_wait_us, c.mean_proxy_peer_wait_us),
                    ("Flush",    b.mean_proxy_flush_us, c.mean_proxy_flush_us),
                    ("GPU Recv", b.mean_proxy_gpu_recv_wait_us, c.mean_proxy_gpu_recv_wait_us),
                ]
                for comp_name, bv, cv in components:
                    if bv == 0 and cv == 0:
                        continue
                    d = pct_delta(bv, cv)
                    diag = ""
                    if bv > 0 and cv > bv * 1.15:
                        diag = "*** REGRESSED"
                    print(f"  {fmt_size(sz):>8s} │ {comp_name:>16s} │ {bv:10.1f} {cv:10.1f} {d:>8s} │ {diag}")
                print()

        # Algo/proto change detection
        for sz in sizes:
            b = base_aggs.get((coll, sz))
            c = cand_aggs.get((coll, sz))
            if not b or not c:
                continue
            if b.algo != c.algo or b.proto != c.proto or b.n_channels != c.n_channels:
                print(f"  ⚠ TUNING CHANGE at {fmt_size(sz)}: "
                      f"{b.algo}/{b.proto}/{b.n_channels}ch → "
                      f"{c.algo}/{c.proto}/{c.n_channels}ch")

        # Summary
        auc_d = pct_delta(busbw_auc_b, busbw_auc_c)
        peak_d = pct_delta(peak_bw_b, peak_bw_c)
        verdict = "PASS" if not coll_flags else "FAIL"
        print(f"\n  AUC BusBW: {busbw_auc_b:.2f} → {busbw_auc_c:.2f}  ({auc_d})")
        print(f"  Peak BusBW: {peak_bw_b:.4f} → {peak_bw_c:.4f}  ({peak_d})")
        print(f"  VERDICT: {verdict}")

        summary_results.append((coll, verdict, auc_d, peak_d, len(coll_flags)))

    # Final summary table
    print(f"\n{'='*80}")
    print(f"  EXECUTIVE SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Collective':>15s} │ {'Verdict':>8s} │ {'AUC Δ':>8s} │ {'Peak Δ':>8s} │ Flags")
    print(f"  {'─'*15}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}")
    for coll, verdict, auc_d, peak_d, nflags in summary_results:
        print(f"  {coll:>15s} │ {verdict:>8s} │ {auc_d:>8s} │ {peak_d:>8s} │ {nflags}")


def main():
    parser = argparse.ArgumentParser(description="ACCL Profiler Report")
    sub = parser.add_subparsers(dest='cmd')

    p_single = sub.add_parser('single')
    p_single.add_argument('--input', required=True)
    p_single.add_argument('--warmup', type=int, default=5)

    p_compare = sub.add_parser('compare')
    p_compare.add_argument('--baseline', required=True)
    p_compare.add_argument('--candidate', required=True)
    p_compare.add_argument('--threshold', type=float, default=10.0)
    p_compare.add_argument('--warmup', type=int, default=5)
    p_compare.add_argument('--output', help='Output file (default: stdout)')

    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        sys.exit(1)

    if args.output if hasattr(args, 'output') and args.output else None:
        sys.stdout = open(args.output, 'w')

    if args.cmd == 'single':
        records = load_dir_or_file(args.input, args.warmup)
        print(f"Loaded {len(records)} records")
        print_single_report(records)
    elif args.cmd == 'compare':
        base = load_dir_or_file(args.baseline, args.warmup)
        cand = load_dir_or_file(args.candidate, args.warmup)
        print(f"Baseline: {len(base)} records, Candidate: {len(cand)} records")
        print_compare_report(base, cand, args.threshold)


if __name__ == '__main__':
    main()
