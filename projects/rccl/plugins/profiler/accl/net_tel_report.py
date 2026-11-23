#!/usr/bin/env python3
"""Report on the NIC counters the ACCL profiler records per collective.

One run: is the network clean? With --baseline: did it get worse?
Rules and accuracy notes: net_tel_report.py --help.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Dict, List, Optional, Tuple

HELP_EPILOG = """\
counter groups:
  PFC     pause frames (frames_rx/tx_pripause)
  CNP     congestion notifications and ECN marks
  Errors  drops: out of buffer, out of sequence, NAKs, RNR, ACK timeouts,
          MTU discards, retries exceeded

classification:
  one run:      0 -> CLEAN, below threshold -> NEGLIGIBLE, else ACTIVE
                (groups in the executive summary use 1000)
  with baseline:
    delta% = (run - baseline) / baseline * 100
    both 0 -> CLEAN
    if max(run, baseline) < threshold:
      one side 0 or |delta%| > 10 -> IGNORED, else NEGLIGIBLE
    otherwise:
      |delta%| < 1 -> NO CHANGE, <= -10 -> IMPROVED,
      >= +10 -> REGRESSION, else NO CHANGE
  thresholds: tx PFC 100 x nodes, rx PFC 5000, others 1000;
              groups when comparing: PFC 50000, CNP 10000, Errors 1000
  per-size rows apply the thresholds to the per-iteration average,
  so they do not depend on how many iterations ran
  nodes: HOT > 50% of the busiest node, MODERATE > 10%, else COLD

accuracy:
  Whole-run totals are always exact.
  Per-size counts depend on how the run was made:
    default            each rank reads its counters on its own; a few counts
                       can land in a neighbouring collective
    ACCL_NET_STRICT=1  all ranks pause together at every collective boundary,
                       so every count belongs to its own collective (slower;
                       needs the matching librccl, one GPU per process)
  Counters are read around each collective, so busbw is lower than in a
  normal benchmark. Compare only against a baseline made the same way
  (counters on, same mode).
"""

# --- Counters -------------------------------------------------------------
# Groups as rccl-tests' counter report defines them, plus the ionic drops and
# ECN the plugin also records, and mlx5 names. Order is the report's row order.
GROUPS: "OrderedDict[str, List[str]]" = OrderedDict([
    ("PFC", ["frames_rx_pripause", "frames_tx_pripause"]),
    ("CNP", ["rx_rdma_cnp_pkts", "tx_rdma_cnp_pkts", "rx_rdma_ecn_pkts",
             "rp_cnp_handled", "np_cnp_sent", "np_ecn_marked_roce_packets"]),
    ("Errors", ["rx_rdma_mtu_discard_pkts", "resp_rx_outof_buf", "tx_rdma_ack_timeout",
                "req_tx_retry_excd_err", "resp_rx_outouf_seq", "req_rx_pkt_seq_err",
                "req_rx_impl_nak_seq_err", "resp_tx_pkt_seq_err", "req_rx_rnr_retry_err",
                "resp_tx_rnr_retry_err", "resp_rx_ccl_cts_outouf_seq",
                "tx_rdma_ccl_cts_ack_timeout", "out_of_buffer", "out_of_sequence",
                "packet_seq_err", "local_ack_timeout_err", "rnr_nak_retry_err"]),
])
VOLUME = ["tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes"]
ETH_COUNTERS = set(GROUPS["PFC"])
RX_PFC = "frames_rx_pripause"
TX_PFC = "frames_tx_pripause"
# Node classification looks at these, first non-zero wins when comparing.
NODE_KEY_COUNTERS = ["frames_rx_pripause", "rp_cnp_handled", "rx_rdma_cnp_pkts",
                     "np_cnp_sent", "tx_rdma_cnp_pkts", "tx_rdma_ack_timeout",
                     "resp_rx_outouf_seq"]
# Per-priority PFC (ionic: frames_{rx,tx}_pri_N, {rx,tx}_pripause_N_1us_count;
# mlx5: {rx,tx}_prioN_pause, {rx,tx}_prioN_pause_duration).
PFC_FRAMES_RE = re.compile(r'^(?:frames_(rx|tx)_pri_?(\d)|(rx|tx)_prio(\d)_pause)$')
PFC_TIME_RE = re.compile(r'^(?:(rx|tx)_pripause_(\d)_1us_count|(rx|tx)_prio(\d)_pause_duration)$')

# --- Thresholds and classification ----------------------------------------
TX_PFC_PER_NODE = 100
THRESHOLDS = {RX_PFC: 5000, "__group_PFC__": 50000, "__group_CNP__": 10000,
              "__group_Errors__": 1000}
DEFAULT_THRESHOLD = 1000
GROUP_SINGLE_THRESHOLD = 1000


def threshold(counter: str, nodes: int = 1) -> int:
    if counter == TX_PFC:
        return TX_PFC_PER_NODE * max(nodes, 1)
    return THRESHOLDS.get(counter, DEFAULT_THRESHOLD)


def classify_single(total: float, counter: str, nodes: int = 1) -> str:
    if total == 0:
        return "CLEAN"
    return "NEGLIGIBLE" if total < threshold(counter, nodes) else "ACTIVE"


def classify_group_single(total: float) -> str:
    if total == 0:
        return "CLEAN"
    return "NEGLIGIBLE" if total < GROUP_SINGLE_THRESHOLD else "ACTIVE"


def classify_compare(run: float, base: float, counter: str, nodes: int = 1) -> str:
    """run vs baseline. Small totals never count as a trend."""
    if run == 0 and base == 0:
        return "CLEAN"
    if max(run, base) < threshold(counter, nodes):
        if run == 0 or base == 0:
            return "IGNORED"
        return "NEGLIGIBLE" if abs((run - base) / base * 100.0) <= 10.0 else "IGNORED"
    if base == 0:
        return "REGRESSION"
    if run == 0:
        return "IMPROVED"
    pct = (run - base) / base * 100.0
    if abs(pct) < 1.0:
        return "NO CHANGE"
    if pct <= -10.0:
        return "IMPROVED"
    if pct >= 10.0:
        return "REGRESSION"
    return "NO CHANGE"


def classify_node(count: float, busiest: float) -> str:
    if busiest == 0:
        return "COLD"
    share = count / busiest
    return "HOT" if share > 0.5 else ("MODERATE" if share > 0.1 else "COLD")


def shown(status: str) -> str:
    return f"{status} (low total)" if status in ("IGNORED", "NEGLIGIBLE") else status


# --- Formatting -------------------------------------------------------------
def fmt_count(n: float) -> str:
    """Integers with K/M/G suffix, two decimals, a bare .00 dropped."""
    n = int(round(n))
    if abs(n) < 1000:
        return str(n)
    for unit, div in (("G", 10**9), ("M", 10**6), ("K", 10**3)):
        if abs(n) >= div:
            s = format((Decimal(n) / Decimal(div)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")
            return (s[:-3] if s.endswith(".00") else s) + unit
    return str(n)


def fmt_avg(x: float) -> str:
    if x >= 1000 or x == 0:
        return fmt_count(x)
    return f"{x:.1f}" if x >= 1 else f"{x:.2g}"


def fmt_delta(run: float, base: float) -> str:
    if run == 0 and base == 0:
        return "-"
    if base == 0:
        return "+inf"
    return f"{(run - base) / base * 100:+.1f}%"


def fmt_size(b: int) -> str:
    for unit, div in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if b >= div and b % div == 0:
            return f"{b // div}{unit}"
    return f"{b}B" if b < 1024 else str(b)


def parse_size(text: str) -> int:
    m = re.fullmatch(r'\s*(\d+)\s*([KMG]?)B?\s*', text, re.I)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size {text!r}")
    return int(m.group(1)) << {"": 0, "K": 10, "M": 20, "G": 30}[m.group(2).upper()]


# --- Loading ----------------------------------------------------------------
FILE_RE = re.compile(r'accl_profiler_rank(\d+)_(.+)_pid(\d+)_(0x[0-9a-fA-F]+)\.jsonl$')


@dataclass
class Window:
    """One collective on one rank."""
    host: str
    rank: int
    n_ranks: int
    comm: str
    coll: str
    sn: int
    size: int
    busbw: float
    exec_us: float
    net: dict
    pid: int = 0

    def usable(self) -> bool:
        n = self.net
        return not n.get("overlapped") and not n.get("unavailable") and not n.get("counter_discontinuities")

    def exact(self) -> bool:
        return bool(self.net.get("coordinated"))


@dataclass
class Run:
    label: str
    comm: str
    windows: List[Window] = field(default_factory=list)   # after warmup
    all_windows: List[Window] = field(default_factory=list)
    hosts: List[str] = field(default_factory=list)
    n_ranks: int = 0


def expand(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)))
        else:
            out.append(p)
    return out


def load(paths: List[str], warmup: int, label: str) -> List[Run]:
    """One Run per communicator. Warmup drops the first `warmup` sequence
    numbers of each (collective, size), the same ones on every rank."""
    by_comm: Dict[str, List[Window]] = defaultdict(list)
    for path in expand(paths):
        m = FILE_RE.search(os.path.basename(path))
        host, pid, comm = (m.group(2), int(m.group(3)), m.group(4)) if m else ("?", 0, path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cp, hdr, net = obj.get("coll_perf"), obj.get("header"), obj.get("net_counters")
                if not cp or not hdr or net is None:
                    continue
                by_comm[comm].append(Window(
                    host=host, rank=hdr.get("rank", 0), n_ranks=hdr.get("n_ranks", 1), comm=comm,
                    coll=cp.get("coll", ""), sn=cp.get("coll_sn", 0),
                    size=cp.get("coll_msg_size_bytes", 0), busbw=cp.get("coll_busbw_gbs", 0.0),
                    exec_us=cp.get("coll_exec_time_us", 0.0), net=net, pid=pid))
    runs = []
    for comm, ws in sorted(by_comm.items()):
        sns: Dict[Tuple[str, int], List[int]] = defaultdict(list)
        for w in ws:
            sns[(w.coll, w.size)].append(w.sn)
        dropped = {k: set(sorted(set(v))[:warmup]) for k, v in sns.items()}
        emptied = [k for k, v in sns.items() if len(set(v)) <= warmup]
        if emptied and warmup > 0:
            print(f"WARNING: {label}: --warmup {warmup} leaves no collectives at "
                  f"{', '.join(fmt_size(k[1]) for k in sorted(emptied))}; those sizes are skipped",
                  file=sys.stderr)
        run = Run(label=label, comm=comm, all_windows=ws,
                  windows=[w for w in ws if w.sn not in dropped[(w.coll, w.size)]],
                  hosts=sorted({w.host for w in ws}), n_ranks=max(w.n_ranks for w in ws))
        runs.append(run)
    return runs


# --- Measurements -------------------------------------------------------------
def owners(ws: List[Window]) -> Dict[Tuple[str, str], int]:
    """Each (host, NIC) counted once: the lowest rank that reads it."""
    own: Dict[Tuple[str, str], int] = {}
    for w in ws:
        for d in w.net.get("devices") or []:
            key = (w.host, d["name"])
            own[key] = min(own.get(key, w.rank), w.rank)
    return own


def window_counts(w: Window, own: Dict[Tuple[str, str], int]) -> Dict[str, int]:
    """Counter deltas of one window, over the NICs this rank owns."""
    out: Dict[str, int] = defaultdict(int)
    for d in w.net.get("devices") or []:
        if own.get((w.host, d["name"])) != w.rank:
            continue
        for k, v in (d.get("rdma") or {}).items():
            out[k] += v
        for k, v in (d.get("ethtool") or {}).items():
            if k in ETH_COUNTERS:
                out[k] += v
    return out


@dataclass
class NodeTotals:
    host: str
    duration_s: float
    devices: Dict[str, Dict[str, int]]   # NIC -> counter -> whole-run total


def whole_run(run: Run) -> List[NodeTotals]:
    """Per node, per NIC: last snapshot minus first, over the whole run."""
    first: Dict[Tuple[str, str], dict] = {}
    last: Dict[Tuple[str, str], dict] = {}
    for w in run.all_windows:
        for snap in (w.net.get("snapshot_before"), w.net.get("snapshot_after")):
            for d in (snap or {}).get("devices") or []:
                if not d.get("ok", True):
                    continue
                key = (w.host, d["name"])
                if key not in first or d["read_start_us"] < first[key]["read_start_us"]:
                    first[key] = d
                if key not in last or d["read_end_us"] > last[key]["read_end_us"]:
                    last[key] = d
    nodes: Dict[str, NodeTotals] = {}
    if first:
        for (host, dev), a in sorted(first.items()):
            b = last[(host, dev)]
            vals: Dict[str, int] = {}
            for src in ("rdma", "ethtool"):
                fa, fb = a.get(src) or {}, b.get(src) or {}
                for k, v in fb.items():
                    if k in fa and (src == "rdma" or k in ETH_COUNTERS):
                        vals[k] = max(0, v - fa[k])
            node = nodes.setdefault(host, NodeTotals(host, 0.0, {}))
            node.devices[dev] = vals
            node.duration_s = max(node.duration_s, (b["read_end_us"] - a["read_start_us"]) / 1e6)
        return list(nodes.values())
    # Records without raw snapshots: sum the per-window deltas.
    own = owners(run.all_windows)
    for w in run.all_windows:
        node = nodes.setdefault(w.host, NodeTotals(w.host, 0.0, {}))
        for d in w.net.get("devices") or []:
            if own.get((w.host, d["name"])) != w.rank:
                continue
            dev = node.devices.setdefault(d["name"], defaultdict(int))
            for k, v in list((d.get("rdma") or {}).items()) + [
                    (k, v) for k, v in (d.get("ethtool") or {}).items() if k in ETH_COUNTERS]:
                dev[k] += v
    return list(nodes.values())


def totals(nodes: List[NodeTotals]) -> Dict[str, int]:
    out: Dict[str, int] = OrderedDict()
    for n in nodes:
        for dev in n.devices.values():
            for k, v in dev.items():
                out[k] = out.get(k, 0) + v
    return out


def per_node(nodes: List[NodeTotals], counter: str) -> Dict[str, int]:
    return {n.host: sum(d.get(counter, 0) for d in n.devices.values()) for n in nodes}


def group_total(t: Dict[str, float], group: str) -> float:
    return sum(t.get(c, 0) for c in GROUPS[group])


@dataclass
class SizeRow:
    coll: str
    size: int
    iters_used: int
    iters_total: int
    busbw: float
    per_iter: Dict[str, float]   # counter -> average per iteration (all ranks summed)


def per_size(run: Run) -> "OrderedDict[Tuple[str, int], SizeRow]":
    own = owners(run.all_windows)
    groups: Dict[Tuple[str, int], List[Window]] = defaultdict(list)
    for w in run.windows:
        groups[(w.coll, w.size)].append(w)
    rows: "OrderedDict[Tuple[str, int], SizeRow]" = OrderedDict()
    for key in sorted(groups):
        ws = groups[key]
        good = [w for w in ws if w.usable()]
        used_sns = {w.sn for w in good}
        sums: Dict[str, int] = defaultdict(int)
        for w in good:
            for k, v in window_counts(w, own).items():
                sums[k] += v
        n = len(used_sns)
        rows[key] = SizeRow(coll=key[0], size=key[1], iters_used=n,
                            iters_total=len({w.sn for w in ws}),
                            busbw=statistics.mean(w.busbw for w in good) if good else 0.0,
                            per_iter={k: v / n for k, v in sums.items()} if n else {})
    return rows


def accuracy_line(run: Run) -> str:
    good = [w for w in run.windows if w.usable()]
    exact = sum(1 for w in good if w.exact())
    if good and exact == len(good):
        return ("Counter accuracy: exact per collective (all ranks paused together at each "
                "collective boundary, ACCL_NET_STRICT=1)")
    if exact == 0:
        return ("Counter accuracy: whole-run totals exact; per-size counts approximate (a few "
                "counts can move to a neighbouring collective; run with ACCL_NET_STRICT=1 for "
                "exact per-size counts)")
    return (f"Counter accuracy: {exact} collectives exact, {len(good) - exact} approximate "
            f"(see --help)")


def strict(run: Run) -> bool:
    good = [w for w in run.windows if w.usable()]
    return bool(good) and sum(w.exact() for w in good) * 2 > len(good)


def avg_busbw(rows) -> float:
    vals = [r.busbw for r in rows.values() if r.iters_used]
    return statistics.mean(vals) if vals else 0.0


def pfc_ratio(t: Dict[str, int], bw: float, nodes: List[NodeTotals]) -> Optional[float]:
    """rx PFC frames/s per GB/s of busbw."""
    durs = [n.duration_s for n in nodes if n.duration_s > 0]
    if not durs or bw <= 0:
        return None
    rx = t.get(RX_PFC, 0)
    return 0.0 if rx <= 0 else (rx / statistics.mean(durs)) / bw


# --- Report sections ------------------------------------------------------------
LINE = "=" * 100


def header(run: Run, base: Optional[Run], rows, brows) -> None:
    colls = sorted({r.coll for r in rows.values()})
    sizes = [r.size for r in rows.values()]
    title = f"=== ACCL Net Counter Report: {run.label} ({', '.join(colls)}, " \
            f"{len(run.hosts)} nodes x {run.n_ranks // max(len(run.hosts), 1)} ranks)"
    print(title + (f"  vs  baseline {base.label} ===" if base else " ==="))
    used = sum(1 for w in run.windows if w.usable())
    if sizes:
        iters = max(r.iters_total for r in rows.values())
        print(f"  Message sizes {fmt_size(min(sizes))}..{fmt_size(max(sizes))}, "
              f"{iters} iterations per size after warmup, {used}/{len(run.windows)} usable windows")
    print(f"  {accuracy_line(run)}")
    if base:
        print(f"  Baseline: {accuracy_line(base)}")
    print()


def mismatches(run: Run, base: Run, rows, brows) -> Tuple[str, str]:
    size_msg = mode_msg = ""
    s1, s2 = set(rows), set(brows)
    if s1 != s2:
        def label(s):
            sizes = sorted(k[1] for k in s)
            return f"{fmt_size(sizes[0])}..{fmt_size(sizes[-1])}" if sizes else "none"
        size_msg = f"message sizes differ (run={label(s1)}, baseline={label(s2)})"
        print("WARNING: run and baseline have different message sizes and cannot be fully "
              f"compared ({size_msg}). Showing data side by side for reference.", file=sys.stderr)
    if strict(run) != strict(base):
        mode_msg = "accuracy modes differ (ACCL_NET_STRICT set in only one run)"
        print(f"WARNING: {mode_msg}: busbw is not comparable; counters are shown side by side.",
              file=sys.stderr)
    return size_msg, mode_msg


def executive(run_t, base_t, bw, base_bw, run_nodes, base_nodes, size_msg, mode_msg) -> None:
    print("=== Executive Summary ===")
    if base_t is None:
        print(f"  Performance : {bw:.2f} GB/s (avg busbw)")
        for g in GROUPS:
            total = group_total(run_t, g)
            detail = f"total: {fmt_count(total)}" if total else ""
            if g == "PFC" and total:
                detail = f"rx_pfc total: {fmt_count(run_t.get(RX_PFC, 0))}"
            print(f"  {g + ':':12s} {classify_group_single(total)}"
                  f"{'   (' + detail + ')' if detail else ''}")
        r = pfc_ratio(run_t, bw, run_nodes)
        if r:
            print(f"  PFC/throughput ratio: {int(r)} rx_pfc frames/s per GB/s")
        print()
        return
    if mode_msg:
        perf = "NOT COMPARABLE"
    elif bw > 0 and base_bw > 0:
        perf = "NO CHANGE" if abs(bw - base_bw) / base_bw * 100 < 3 else "NEEDS REVIEW"
    elif bw == 0 and base_bw == 0:
        perf = "NO DATA"
    else:
        perf = "NEEDS REVIEW"
    print(f"  Performance : {perf}   (run={bw:.2f} GB/s, baseline={base_bw:.2f} GB/s, "
          f"delta={fmt_delta(bw, base_bw)})")
    nodes = max(len(run_nodes), len(base_nodes), 1)
    statuses = []
    for g, members in GROUPS.items():
        t1, t0 = group_total(run_t, g), group_total(base_t, g)
        st = classify_compare(t1, t0, f"__group_{g}__", nodes)
        big = max(t1, t0)
        activity = "" if big == 0 else ("NEGLIGIBLE" if big < 1000 else "ACTIVE")
        label = f"{activity} / {st}" if activity and st != "CLEAN" else st
        if g == "PFC" and big:
            r1, r0 = run_t.get(RX_PFC, 0), base_t.get(RX_PFC, 0)
            detail = f"rx_pfc total: {fmt_count(r0)} -> {fmt_count(r1)}, {fmt_delta(r1, r0)}"
        elif big == 0:
            present = [c for c in members if c in run_t or c in base_t]
            detail = f"all {len(present)} counters zero in both runs"
        else:
            detail = f"total: {fmt_count(t0)} -> {fmt_count(t1)}, {fmt_delta(t1, t0)}"
        print(f"  {g + ':':12s} {label}   ({detail})")
        statuses.append(st)
    r1, r0 = pfc_ratio(run_t, bw, run_nodes), pfc_ratio(base_t, base_bw, base_nodes)
    if (run_t.get(RX_PFC, 0) or base_t.get(RX_PFC, 0)) and r1 is not None and r0 is not None:
        print(f"  PFC/throughput ratio: {int(r0)} vs {int(r1)} rx_pfc frames/s per GB/s "
              f"({fmt_delta(r1, r0)})")
    if size_msg:
        verdict = f"INVALID COMPARISON - {size_msg}"
    elif "REGRESSION" in statuses:
        verdict = "REGRESSION DETECTED"
    elif perf == "NO CHANGE" and all(s in ("CLEAN", "NO CHANGE", "NEGLIGIBLE", "IMPROVED", "IGNORED")
                                     for s in statuses):
        verdict = "NO REGRESSION"
    else:
        verdict = "NEEDS REVIEW"
    print(f"\n  OVERALL VERDICT: {verdict}\n")


def performance(rows, brows) -> None:
    print("=== Per-Size Performance (ACCL busbw, GPU-timed) ===")
    keys = sorted(set(rows) | set(brows or {}))
    multi = len({k[0] for k in keys}) > 1
    if brows is None:
        print(f"  {'Size':>10}  {'busbw':>8}\n  {'':>10}  {'(GB/s)':>8}\n  {'-'*10}  {'-'*8}")
        for k in keys:
            print(f"  {(k[0] + ' ' if multi else '') + fmt_size(k[1]):>10}  {rows[k].busbw:8.2f}")
    else:
        print(f"  {'Size':>10}  {'run':>10}  {'baseline':>10}  {'delta':>8}")
        print(f"  {'':>10}  {'(GB/s)':>10}  {'(GB/s)':>10}\n  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
        for k in keys:
            a = rows[k].busbw if k in rows else 0.0
            b = brows[k].busbw if k in brows else 0.0
            print(f"  {(k[0] + ' ' if multi else '') + fmt_size(k[1]):>10}  {a:10.2f}  {b:10.2f}  "
                  f"{fmt_delta(a, b):>8}")
        if set(rows) != set(brows):
            print("  * A size present in only one run shows 0.00 for the other.")
    print()


def ordered_counters(*tables: Dict[str, int]) -> List[str]:
    present = set().union(*[set(t) for t in tables])
    order = VOLUME + [c for g in GROUPS.values() for c in g]
    return [c for c in order if c in present]


def counter_summary(run_t, base_t, nodes: int) -> None:
    names = ordered_counters(run_t, *([base_t] if base_t is not None else []))
    if not names:
        print("  (no counters recorded)\n")
        return
    w = max(len(n) for n in names) + 2
    print("=== NIC Counter Summary (whole run, all ranks) ===")
    if base_t is None:
        print(f"  {'Counter':<{w}} {'Total':>12}  Status")
        for c in names:
            v = run_t.get(c, 0)
            st = "(volume)" if c in VOLUME else classify_single(v, c, nodes)
            print(f"  {c:<{w}} {fmt_count(v):>12}  {st}")
    else:
        print(f"  {'Counter':<{w}} {'run Total':>12}  {'baseline Total':>14}  {'Delta':>8}  Status")
        for c in names:
            v1, v0 = run_t.get(c, 0), base_t.get(c, 0)
            st = "(volume)" if c in VOLUME else shown(classify_compare(v1, v0, c, nodes))
            print(f"  {c:<{w}} {fmt_count(v1):>12}  {fmt_count(v0):>14}  {fmt_delta(v1, v0):>8}  {st}")
    print()


def pfc_ratio_section(run_t, base_t, bw, base_bw, run_nodes, base_nodes) -> None:
    r1 = pfc_ratio(run_t, bw, run_nodes)
    if base_t is None:
        if not r1:
            return
        durs = [n.duration_s for n in run_nodes if n.duration_s > 0]
        fps = run_t.get(RX_PFC, 0) / statistics.mean(durs)
        print("=== PFC / Throughput Ratio ===")
        print(f"  PFC frames/s  : {fps:.1f}\n  Avg busbw     : {bw:.2f} GB/s\n  frames/s/GB/s : {r1:.1f}\n")
        return
    r0 = pfc_ratio(base_t, base_bw, base_nodes)
    if not (run_t.get(RX_PFC, 0) or base_t.get(RX_PFC, 0)) or r1 is None or r0 is None:
        return
    print("=== PFC / Throughput Ratio ===")
    print(f"  {'':20s} {'run':>10}  {'baseline':>10}  {'delta':>8}")
    print(f"  {'frames/s/GB/s':20s} {r1:>10.1f}  {r0:>10.1f}  {fmt_delta(r1, r0):>8}\n")


def node_classification(run_nodes, base_nodes, run_t, base_t) -> None:
    if base_t is None:
        for c in NODE_KEY_COUNTERS:
            if not run_t.get(c):
                continue
            counts = per_node(run_nodes, c)
            busiest = max(counts.values())
            w = max(len(n) for n in counts) + 2
            print(f"=== Node Classification ({c}) ===")
            print(f"  {'Node':<{w}} {'Count':>12}  Class")
            for node, v in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {node:<{w}} {fmt_count(v):>12}  {classify_node(v, busiest)}")
            print()
        return
    c = next((c for c in NODE_KEY_COUNTERS if run_t.get(c) or base_t.get(c)), None)
    if c is None:
        return
    p1, p0 = per_node(run_nodes, c), per_node(base_nodes, c)
    nodes = list(OrderedDict.fromkeys(list(p1) + list(p0)))
    busiest = max(p1.values(), default=0)
    w = max(len(n) for n in nodes) + 2
    print(f"=== Node Classification ({c}) ===")
    print(f"  {'Node':<{w}} {'run count':>12}  {'base count':>12}  {'delta':>8}  {'status':<22}  class")
    for node in sorted(nodes, key=lambda n: -p1.get(n, 0)):
        v1, v0 = p1.get(node, 0), p0.get(node, 0)
        print(f"  {node:<{w}} {fmt_count(v1):>12}  {fmt_count(v0):>12}  {fmt_delta(v1, v0):>8}  "
              f"{shown(classify_compare(v1, v0, c, 1)):<22}  {classify_node(v1, busiest)}")
    print()


def size_breakdown(row: SizeRow, group: str) -> str:
    parts = sorted(((c, v) for c, v in row.per_iter.items() if c in GROUPS[group] and v),
                   key=lambda x: -x[1])
    return ", ".join(f"{c} {fmt_avg(v)}" for c, v in parts)


def per_size_table(rows, brows, nodes: int, only: Optional[List[int]], approximate: bool) -> None:
    keys = [k for k in rows if not only or k[1] in only]
    if brows is not None:
        keys = sorted(set(keys) | {k for k in brows if not only or k[1] in only})
    if not keys:
        return
    multi = len({k[0] for k in keys}) > 1
    print("=== Per-Size Counters (per iteration, summed over ranks; thresholds on the "
          "per-iteration average) ===")
    if approximate:
        print("  Counts near size boundaries may move between neighbouring sizes; whole-run "
              "totals are exact.")
    notes = []
    if brows is None:
        print(f"  {'Size':>10}  {'iters':>7}  " + "  ".join(f"{g:<22}" for g in GROUPS).rstrip())
        for k in keys:
            r = rows[k]
            cells = []
            for g in GROUPS:
                v = group_total(r.per_iter, g)
                st = classify_group_single(v)
                cells.append(f"{(fmt_avg(v) + '  ') if v else ''}{st}")
                if st == "ACTIVE":
                    notes.append(f"  {g} at {fmt_size(r.size)}: {size_breakdown(r, g)} "
                                 f"(per iteration)")
            print(f"  {(k[0] + ' ' if multi else '') + fmt_size(k[1]):>10}  "
                  f"{f'{r.iters_used}/{r.iters_total}':>7}  " + "  ".join(f"{c:<22}" for c in cells).rstrip())
    else:
        print(f"  {'Size':>10}  {'iters':>7}  " +
              "  ".join(f"{g + ' run / base':<26}{'status':<22}" for g in GROUPS).rstrip())
        for k in keys:
            r, b = rows.get(k), brows.get(k)
            cells = []
            for g in GROUPS:
                v1 = group_total(r.per_iter, g) if r else 0.0
                v0 = group_total(b.per_iter, g) if b else 0.0
                st = classify_compare(v1, v0, f"__group_{g}__", nodes)
                cells.append(f"{fmt_avg(v1) + ' / ' + fmt_avg(v0):<26}{shown(st):<22}")
                if st == "REGRESSION" and r:
                    notes.append(f"  {g} at {fmt_size(r.size)}: {size_breakdown(r, g)} "
                                 f"(per iteration)")
            iters = f"{r.iters_used}/{r.iters_total}" if r else "-"
            print(f"  {(k[0] + ' ' if multi else '') + fmt_size(k[1]):>10}  {iters:>7}  " + "  ".join(cells).rstrip())
    for n in notes:
        print(n)
    print()


def pfc_priorities(run: Run) -> List[dict]:
    """Whole-run PFC per (host, NIC, priority) from the first and last raw
    snapshot. rx pause time: this NIC was paused by its switch port; tx pause
    time: this NIC paused its peer."""
    first: Dict[Tuple[str, str], Tuple[int, dict]] = {}
    last: Dict[Tuple[str, str], Tuple[int, dict]] = {}
    for w in run.all_windows:
        for snap in (w.net.get("snapshot_before"), w.net.get("snapshot_after")):
            for d in (snap or {}).get("devices") or []:
                eth = d.get("ethtool")
                if not eth:
                    continue
                key = (w.host, d["name"])
                if key not in first or d["read_start_us"] < first[key][0]:
                    first[key] = (d["read_start_us"], eth)
                if key not in last or d["read_end_us"] > last[key][0]:
                    last[key] = (d["read_end_us"], eth)
    rows = []
    for key in sorted(first):
        (t0, a), (t1, b) = first[key], last[key]
        span = t1 - t0
        per: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for name, v in b.items():
            d = v - a.get(name, v)
            if d <= 0:
                continue
            for pattern, suffix in ((PFC_FRAMES_RE, "frames"), (PFC_TIME_RE, "pause_us")):
                m = pattern.match(name)
                if m:
                    per[int(m.group(2) or m.group(4))][f"{m.group(1) or m.group(3)}_{suffix}"] += d
        for prio, c in sorted(per.items()):
            rows.append({"host": key[0], "device": key[1], "priority": prio, "span_us": span,
                         "rx_frames": c["rx_frames"], "tx_frames": c["tx_frames"],
                         "host_paused_pct": 100.0 * c["rx_pause_us"] / span if span > 0 else 0.0,
                         "peer_paused_pct": 100.0 * c["tx_pause_us"] / span if span > 0 else 0.0})
    return rows


def pfc_section(run: Run, base: Optional[Run]) -> None:
    print("=== PFC Pause Time per Priority (whole run) ===")
    rows = pfc_priorities(run)
    if not rows:
        print("  (no PFC frames" + (" in this run" if base else "") + ")\n")
        return
    print("  HostPaused%: share of the run this NIC was paused by its switch (pause frames received).")
    print("  PeerPaused%: share of the run this NIC paused its peer (pause frames sent).")
    print(f"  {'Host':>20s} {'NIC':>8s} {'Prio':>4s}  {'RxFrames':>10s} {'TxFrames':>10s}  "
          f"{'HostPaused%':>11s} {'PeerPaused%':>11s}")
    for r in rows:
        print(f"  {r['host']:>20s} {r['device']:>8s} {r['priority']:>4d}  {fmt_count(r['rx_frames']):>10s} "
              f"{fmt_count(r['tx_frames']):>10s}  {r['host_paused_pct']:10.3f}% {r['peer_paused_pct']:10.3f}%")
    print()


def detail(nodes: List[NodeTotals]) -> None:
    for n in nodes:
        print(f"=== Detail: {n.host} (duration={n.duration_s:.3f}s) ===")
        names = ordered_counters(*n.devices.values())
        if not names:
            print("  (no counters)\n")
            continue
        devs = list(n.devices)
        cw = max(len(c) for c in names) + 2
        dw = max(8, *(len(d) for d in devs))
        print(f"  {'Counter':<{cw}}" + "".join(f"  {d:>{dw}}" for d in devs))
        print("  " + "-" * cw + "".join("  " + "-" * dw for _ in devs))
        for c in names:
            print(f"  {c:<{cw}}" + "".join(f"  {fmt_count(n.devices[d].get(c, 0)):>{dw}}" for d in devs))
        print()


def report(run: Run, base: Optional[Run], args) -> None:
    rows = per_size(run)
    brows = per_size(base) if base else None
    run_nodes = whole_run(run)
    base_nodes = whole_run(base) if base else []
    run_t = totals(run_nodes)
    base_t = totals(base_nodes) if base else None
    bw = avg_busbw(rows)
    base_bw = avg_busbw(brows) if base else 0.0
    header(run, base, rows, brows)
    size_msg = mode_msg = ""
    if base:
        size_msg, mode_msg = mismatches(run, base, rows, brows)
    executive(run_t, base_t, bw, base_bw, run_nodes, base_nodes, size_msg, mode_msg)
    performance(rows, brows)
    counter_summary(run_t, base_t, max(len(run_nodes), len(base_nodes), 1))
    pfc_ratio_section(run_t, base_t, bw, base_bw, run_nodes, base_nodes)
    node_classification(run_nodes, base_nodes, run_t, base_t)
    per_size_table(rows, brows, max(len(run_nodes), len(base_nodes), 1), args.sizes,
                   approximate=not all(w.exact() for w in run.windows if w.usable()))
    pfc_section(run, base)
    if args.detail:
        detail(run_nodes)


# --- Trace ---------------------------------------------------------------------
# Trace labels: (category, the side that counts it).
TRACE_LABELS = {
    "resp_rx_outof_buf": ("OOB", "receiver"), "out_of_buffer": ("OOB", "receiver"),
    "req_rx_rnr_retry_err": ("RNR", "sender"), "resp_tx_rnr_retry_err": ("RNR", "receiver"),
    "rnr_nak_retry_err": ("RNR", "sender"),
    "req_rx_pkt_seq_err": ("NAK", "sender, NAK received"),
    "req_rx_impl_nak_seq_err": ("NAK", "sender, NAK received"),
    "resp_tx_pkt_seq_err": ("NAK", "receiver, NAK sent"),
    "packet_seq_err": ("NAK", "sender, NAK received"),
    "resp_rx_outouf_seq": ("OOS", "receiver"), "resp_rx_ccl_cts_outouf_seq": ("OOS", "sender, CTS"),
    "out_of_sequence": ("OOS", "receiver"),
    "tx_rdma_ack_timeout": ("ACK timeout", "sender"),
    "tx_rdma_ccl_cts_ack_timeout": ("ACK timeout", "receiver, CTS"),
    "local_ack_timeout_err": ("ACK timeout", "sender"),
    "rx_rdma_mtu_discard_pkts": ("MTU discard", "receiver"),
    "req_tx_retry_excd_err": ("Retry exceeded", "sender"),
    "rx_rdma_ecn_pkts": ("ECN", "receiver"), "np_ecn_marked_roce_packets": ("ECN", "receiver"),
    "tx_rdma_cnp_pkts": ("CNP sent", "receiver"), "np_cnp_sent": ("CNP sent", "receiver"),
    "rx_rdma_cnp_pkts": ("CNP received", "sender"), "rp_cnp_handled": ("CNP received", "sender"),
}


def rank_counts(w: Window) -> Tuple[Dict[str, int], Dict[str, int]]:
    """RDMA deltas and per-priority PFC frames over every NIC this rank reads."""
    rdma: Dict[str, int] = defaultdict(int)
    pfc: Dict[str, int] = defaultdict(int)
    for d in w.net.get("devices") or []:
        for k, v in (d.get("rdma") or {}).items():
            rdma[k] += v
        for k, v in (d.get("ethtool") or {}).items():
            m = PFC_FRAMES_RE.match(k)
            if m and v:
                pfc[f"{m.group(1) or m.group(3)} prio {m.group(2) or m.group(4)}"] += v
    return rdma, pfc


def build_trace(runs: List[Run]) -> dict:
    """Perfetto/Chrome trace: one process per host, three rows per rank
    (collective, counter window, counter reads) and counter tracks per rank.
    Counter tracks are per-collective totals, not rates. Each host's clock
    starts at 0; hosts are not time-aligned."""
    ws = [w for run in runs for w in run.all_windows if "before_ts_us" in w.net]
    hosts = sorted({w.host for w in ws})
    pids = {h: i + 1 for i, h in enumerate(hosts)}
    keys = sorted({(w.host, w.pid, w.comm, w.rank) for w in ws})
    tids = {k: 10 * i + 1 for i, k in enumerate(keys)}
    origin = {h: min(w.net["before_ts_us"] for w in ws if w.host == h) for h in hosts}
    multi = len({w.comm for w in ws}) > 1
    events: List[dict] = [{"ph": "M", "name": "process_name", "pid": pids[h],
                           "args": {"name": f"{h} (host-local clock)"}} for h in hosts]
    for k in keys:
        for lane, title in enumerate(("Collective (host observed)", "Counter window (includes idle)",
                                      "Snapshot reads")):
            tid = tids[k] + lane
            events.append({"ph": "M", "name": "thread_name", "pid": pids[k[0]], "tid": tid,
                           "args": {"name": f"rank {k[3]} · {title}" + (f" · {k[2]}" if multi else "")}})
            events.append({"ph": "M", "name": "thread_sort_index", "pid": pids[k[0]], "tid": tid,
                           "args": {"sort_index": k[3] * 10 + lane}})
    # Symptom tracks only for ranks that ever saw that symptom.
    seen: Dict[tuple, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for w in ws:
        if not w.usable():
            continue
        rdma, pfc = rank_counts(w)
        key = (w.host, w.pid, w.comm, w.rank)
        for c, v in rdma.items():
            if v and c in TRACE_LABELS:
                track = "ECN/CNP" if c in GROUPS["CNP"] else "drops"
                seen[key][track].add(TRACE_LABELS[c][0])
        seen[key]["PFC frames (stale)"].update(pfc)
    reads = set()
    for w in ws:
        n, key = w.net, (w.host, w.pid, w.comm, w.rank)
        pid, zero, tid = pids[w.host], origin[w.host], tids[key]
        rdma, pfc = rank_counts(w)
        drops = {f"{TRACE_LABELS[c][0]} ({c})": v for c, v in sorted(rdma.items())
                 if v and c in TRACE_LABELS and c not in GROUPS["CNP"]}
        cnp = {f"{TRACE_LABELS[c][0]} ({c})": v for c, v in sorted(rdma.items())
               if v and c in TRACE_LABELS and c in GROUPS["CNP"]}
        if n.get("overlapped"):
            quality = "overlapped"
        elif not w.usable():
            quality = "unavailable"
        elif drops or cnp:
            quality = "suspect"
        else:
            quality = "exact" if w.exact() else "approximate"
        color = {"suspect": "terrible", "exact": "good", "approximate": "yellow"}.get(quality, "grey")
        args = {"tx_bytes": rdma.get("tx_rdma_ucast_bytes"), "rx_bytes": rdma.get("rx_rdma_ucast_bytes"),
                **({"drops": drops} if drops else {}), **({"ecn_cnp": cnp} if cnp else {}),
                **({"pfc_frames (stale)": dict(sorted(pfc.items()))} if pfc else {}),
                "sn": w.sn, "rank": w.rank, "message_bytes": w.size, "gpu_exec_us": w.exec_us,
                "launch_wait_us": n.get("launch_wait_us", 0), "window": quality,
                "overlapped": n.get("overlapped"), "unavailable": n.get("unavailable"),
                "completion_verified": n.get("completion_verified", False),
                "coordinated": n.get("coordinated", False)}
        name = f"{w.coll} {fmt_size(w.size)} · #{w.sn}"
        before, after = n.get("snapshot_before"), n.get("snapshot_after")
        left = before["read_end_us"] if before else n["before_ts_us"]
        right = after["read_start_us"] if after else n["after_ts_us"]
        events.append({"ph": "X", "cat": quality, "name": name, "pid": pid, "tid": tid + 1,
                       "ts": left - zero, "dur": max(0, right - left), "cname": color, "args": args})
        begin, done = n.get("collective_start_cpu_us"), n.get("collective_complete_cpu_us")
        if begin is not None and done is not None:
            events.append({"ph": "X", "cat": "collective", "name": name, "pid": pid, "tid": tid,
                           "ts": begin - zero, "dur": max(0, done - begin), "cname": color,
                           "args": {**args, "placement": "host callbacks: Coll start to observed "
                                                         "GPU+proxy completion"}})
        for snap in (before, after):
            if not snap or (key, snap["id"]) in reads:
                continue
            reads.add((key, snap["id"]))
            events.append({"ph": "X", "cat": "read", "name": f"Counter snapshot S{snap['id']}",
                           "pid": pid, "tid": tid + 2, "ts": snap["read_start_us"] - zero,
                           "dur": max(0, snap["read_end_us"] - snap["read_start_us"]),
                           "cname": "thread_state_iowait", "args": snap})
        if not w.usable():
            continue
        ts = (after["read_end_us"] if after else n["after_ts_us"]) - zero
        track = f"rank {w.rank}" + (f" · {w.comm}" if multi else "")
        events.append({"ph": "C", "name": f"{track} MB per collective", "pid": pid, "ts": ts,
                       "args": {"tx": rdma.get("tx_rdma_ucast_bytes", 0) / 1e6,
                                "rx": rdma.get("rx_rdma_ucast_bytes", 0) / 1e6}})
        for title, series in seen[key].items():
            if not series:
                continue
            vals = {c: 0 for c in sorted(series)}
            if title == "PFC frames (stale)":
                vals.update(pfc)
            else:
                for c, v in rdma.items():
                    if c in TRACE_LABELS and (c in GROUPS["CNP"]) == (title == "ECN/CNP"):
                        vals[TRACE_LABELS[c][0]] += v
            events.append({"ph": "C", "name": f"{track} {title}", "pid": pid, "ts": ts, "args": vals})
    return {"traceEvents": events, "displayTimeUnit": "ms",
            "otherData": {"clock": "CLOCK_MONOTONIC, normalized independently per host; "
                                   "hosts are NOT synchronized",
                          "host_origins_us": origin,
                          "counter_tracks": "per-collective interval totals; not rates"}}


def pair(run: Run, bases: List[Run]) -> Optional[Run]:
    """Baseline communicator for `run`: same collectives and rank count,
    same accuracy mode preferred, else the largest."""
    if not bases:
        return None
    colls = lambda r: {w.coll for w in r.windows}
    score = lambda b: (colls(b) == colls(run), b.n_ranks == run.n_ranks, strict(b) == strict(run),
                       b.n_ranks, len(b.windows))
    best = max(bases, key=score)
    if len(bases) > 1:
        print(f"WARNING: baseline has {len(bases)} communicators; comparing {run.label} "
              f"{run.comm} with {best.comm}. Point --baseline at one run to choose.", file=sys.stderr)
    return best


# --- CLI -----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="net_tel_report.py",
        usage="%(prog)s RUN [--baseline BASE] [--warmup N] [--sizes LIST] [--detail] [--trace FILE]",
        description="Report on the NIC counters the ACCL profiler records per collective "
                    "(ACCL_NET_COUNTERS=1). One run: is the network clean? With --baseline: "
                    "did it get worse?",
        epilog=HELP_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", metavar="RUN", nargs="+", help="ACCL output directory or .jsonl files of the run")
    ap.add_argument("--baseline", metavar="BASE", nargs="+", help="ACCL output of a known-good run to compare against")
    ap.add_argument("--warmup", metavar="N", type=int, default=5,
                    help="collectives dropped at the start of each size (default 5)")
    ap.add_argument("--sizes", metavar="LIST", type=lambda s: [parse_size(x) for x in s.split(",")],
                    help="message sizes for the per-size table, e.g. 64M,256M")
    ap.add_argument("--detail", action="store_true", help="per-node, per-NIC counter tables")
    ap.add_argument("--trace", metavar="FILE", help="also write a Perfetto/Chrome trace")
    args = ap.parse_args(argv)
    missing = [p for p in args.run + (args.baseline or []) if not os.path.exists(p)]
    if missing:
        ap.error(f"no such file or directory: {', '.join(missing)}")

    runs = load(args.run, args.warmup, label=" ".join(args.run))
    if not runs:
        print("No records with net_counters (run with ACCL_NET_COUNTERS=1).", file=sys.stderr)
        return 1
    bases = load(args.baseline, args.warmup, label=" ".join(args.baseline)) if args.baseline else []
    if args.baseline and not bases:
        print("No records with net_counters in the baseline.", file=sys.stderr)
        return 1
    runs.sort(key=lambda r: (-r.n_ranks, r.comm))
    for run in runs:
        report(run, pair(run, bases), args)
    if args.trace:
        with open(args.trace, "w", encoding="utf-8") as f:
            json.dump(build_trace(runs), f)
        print(f"Trace written to {args.trace}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
