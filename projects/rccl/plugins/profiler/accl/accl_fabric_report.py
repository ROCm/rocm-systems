#!/usr/bin/env python3
"""ACCL fabric report: what did the network show while each collective ran?

Joins two outputs of the same run:
  * accl profiler JSONL (one file per rank and communicator): when each
    collective ran and how its time splits between GPU and proxy;
  * RCCL network telemetry JSON (RCCL_TELEMETRY_ENABLE=1 and
    RCCL_TELEMETRY_SAMPLE_MS, one file per process): NIC counters over time.

Both stamp time with CLOCK_MONOTONIC, so on one host they line up directly.
For every (collective, message size) on every rank the NIC counters are
charged to the time those collectives were running.

Counters are only sampled every few ms, so the amount that fell inside the
collectives is reported as a range: everything counted in the sample windows
the collectives touched is an upper bound, and for durations, that total minus
the time those windows spent outside the collectives is a lower bound. A
finding is "yes" when its lower bound crosses the threshold and "maybe" when
only its upper bound does.

Findings, per NIC:
  FABRIC-CONGESTION  the switch paused our transmit (PFC frames received)
  HOST-BACKPRESSURE  our NIC paused the switch: this host could not take data in
  FABRIC-LOSS        out-of-sequence drops, sequence-error NAKs, ACK timeouts
  HOST-RNR           receiver not ready / out of receive buffers
  LINK-SHARED        the link was busy and mostly with traffic that is not RCCL
  CONGESTION-MARKS   ECN-marked packets or CNPs above a share of all packets
Rx pause is not proof of a fabric fault: a receiver that pushes back makes
the switch pause the senders. The per-collective summary therefore blames a
host first when any of its NICs paused the switch.

Usage:
    python3 accl_fabric_report.py report --accl DIR --telemetry DIR
    python3 accl_fabric_report.py trace  --accl DIR --telemetry DIR -o trace.json
"""
import argparse
import bisect
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accl_report  # noqa: E402  (Record, aggregate, classify_bottleneck, fmt_size)

NUM_PFC_PRIO = 8
PRIO_KEYS = ("pfc_rx_pause_us", "pfc_tx_pause_us", "pfc_rx_frames", "pfc_tx_frames",
             "prio_rx_bytes", "prio_tx_bytes")
SCALAR_KEYS = ("tx_bytes", "rx_bytes", "nic_tx_bytes", "nic_rx_bytes", "nic_tx_packets",
               "nic_rx_packets", "ecn_marked_pkts", "cnp_sent", "cnp_rcvd")
# Sampled error counters, by what they point at.
LOSS_FABRIC = ("oos_drop_count", "seq_err_naks_rcvd", "local_ack_timeout_err")
LOSS_HOST = ("out_of_buffer", "rnr_retry_err")
INF = math.inf
COVERAGE = "telemetry samples do not cover all of these collectives"
# Collectives closer together than this run as one busy stretch.
MERGE_GAP_US = 1000.0


# ============================================================================
# Loading
# ============================================================================

@dataclass
class CollRecord:
    host: str
    pid: int
    rank: int
    comm_hash: str
    rec: accl_report.Record
    exec_start_us: int
    exec_end_us: int
    enqueue_us: int = 0
    # device_id -> SW counter deltas over the collective's network window
    net: Dict[int, dict] = field(default_factory=dict)
    net_start_us: int = 0
    net_end_us: int = 0


def load_accl(path: str, warmup: int) -> Tuple[List[CollRecord], int]:
    """Returns (records, number skipped because the plugin predates host_ts)."""
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".jsonl")] \
        if os.path.isdir(path) else [path]
    out, no_ts = [], 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cp, hdr = obj.get("coll_perf"), obj.get("header")
                if not cp or not hdr or cp.get("coll_sn", 0) < warmup:
                    continue
                ts = cp.get("host_ts")
                if not ts or "host" not in hdr:
                    no_ts += 1
                    continue
                d = cp.get("decomposition", {})
                rec = accl_report.Record(
                    coll=cp.get("coll", ""), sn=cp.get("coll_sn", 0),
                    msg_size=cp.get("coll_msg_size_bytes", 0), rank=hdr.get("rank", 0),
                    n_ranks=hdr.get("n_ranks", 1), algo=cp.get("coll_algo", ""),
                    proto=cp.get("coll_proto", ""), n_channels=cp.get("coll_n_channels", 0),
                    exec_time_us=cp.get("coll_exec_time_us", 0),
                    algobw_gbs=cp.get("coll_algobw_gbs", 0), busbw_gbs=cp.get("coll_busbw_gbs", 0),
                    timing_source=cp.get("coll_timing_source", ""),
                    enqueue_to_kernel_us=d.get("enqueue_to_kernel_us", 0),
                    gpu_kernel_avg_us=d.get("gpu_kernel_avg_us", 0),
                    gpu_kernel_min_us=d.get("gpu_kernel_min_us", 0),
                    gpu_kernel_max_us=d.get("gpu_kernel_max_us", 0),
                    proxy_gpu_wait_us=d.get("proxy_gpu_wait_us", 0),
                    proxy_network_us=d.get("proxy_network_us", 0),
                    proxy_peer_wait_us=d.get("proxy_peer_wait_us", 0),
                    proxy_flush_us=d.get("proxy_flush_us", 0),
                    proxy_gpu_recv_wait_us=d.get("proxy_gpu_recv_wait_us", 0),
                    n_proxy_ops=d.get("n_proxy_ops", 0),
                    n_send_ops=d.get("n_send_ops", 0), n_recv_ops=d.get("n_recv_ops", 0))
                netobj = cp.get("net") or {}
                net = {dev["device_id"]: dev for dev in netobj.get("devices", [])}
                out.append(CollRecord(host=hdr["host"], pid=int(hdr.get("pid", 0)), rank=rec.rank,
                                      comm_hash=hdr.get("comm_hash", ""), rec=rec,
                                      exec_start_us=ts.get("exec_start_us", 0),
                                      enqueue_us=ts.get("enqueue_us", 0),
                                      exec_end_us=ts.get("exec_end_us", 0), net=net,
                                      net_start_us=netobj.get("window_start_us", 0),
                                      net_end_us=netobj.get("window_end_us", 0)))
    return out, no_ts


class Series:
    """A cumulative counter sampled over time; linear between samples."""

    def __init__(self, ts: List[int], vals: List[float], resets: int = 0):
        self.ts, self.vals, self.resets = ts, vals, resets

    def at(self, t: float) -> float:
        i = bisect.bisect_left(self.ts, t)
        if i <= 0:
            return self.vals[0]
        if i >= len(self.ts):
            return self.vals[-1]
        t0, t1 = self.ts[i - 1], self.ts[i]
        v0, v1 = self.vals[i - 1], self.vals[i]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0) if t1 > t0 else v1

    def delta(self, a: float, b: float) -> float:
        return max(0.0, self.at(b) - self.at(a))

    def total(self) -> float:
        return self.vals[-1] - self.vals[0]


class SeriesMap:
    """name -> Series, built from the samples on first use: a run has dozens of
    counters per NIC and most are never looked at. Per-priority series are
    named "<key>[<prio>]"."""

    def __init__(self, samples: Optional[Dict[str, list]] = None):
        self._samples, self._built = samples or {"ts_us": []}, {}

    def get(self, name: str, default=None) -> Optional[Series]:
        if name not in self._built:
            key, _, idx = name.partition("[")
            self._built[name] = _series(self._samples, key, int(idx[:-1]) if idx else None)
        v = self._built[name]
        return default if v is None else v

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __getitem__(self, name: str) -> Series:
        v = self.get(name)
        if v is None:
            raise KeyError(name)
        return v

    def built(self) -> List[Series]:
        return [v for v in self._built.values() if v is not None]


@dataclass
class Device:
    device_id: int
    roce: str
    hw_type: str
    link_gbps: float = 0.0
    series: SeriesMap = field(default_factory=SeriesMap)


@dataclass
class Telemetry:
    host: str
    pid: int
    sample_interval_ms: int
    hist_interval_ns: int
    hist_max_buckets: int
    devices: Dict[int, Device] = field(default_factory=dict)
    truncated: bool = False   # sampler buffer filled before the run ended


def _series(samples, key, idx: Optional[int] = None) -> Optional[Series]:
    """Every series here is a cumulative counter. A single sample lower than the
    one before is a bad read and is dropped (Thor2 sysfs bytes have returned 0
    between good readings); a drop that persists is a counter reset, and the
    count carries on from where it was."""
    pts = []
    for t, v in zip(samples["ts_us"], samples.get(key, ())):
        if isinstance(v, list):
            v = v[idx] if idx is not None and idx < len(v) else None
        if v is None or v < 0:
            continue  # -1 = counter not available on this NIC
        pts.append((t, float(v)))
    ts, vals, offset, resets = [], [], 0.0, 0
    for i, (t, v) in enumerate(pts):
        if vals and v + offset < vals[-1]:
            raw_last = vals[-1] - offset
            nxt = pts[i + 1][1] if i + 1 < len(pts) else None
            if nxt is None or nxt >= raw_last:
                continue
            offset, resets = vals[-1], resets + 1
        ts.append(t)
        vals.append(v + offset)
    return Series(ts, vals, resets) if len(ts) >= 2 else None


def load_telemetry(path: str, wanted: Optional[Dict[Tuple[str, int], set]] = None
                   ) -> Dict[Tuple[str, int], Telemetry]:
    """wanted: per process, the device ids worth keeping. Older telemetry sampled
    every NIC on the node; keeping only the ones a process drove, stored as one
    list per counter, is what keeps a long run's files from filling memory."""
    files = [os.path.join(path, f) for f in sorted(os.listdir(path))
             if f.startswith("rccl_telemetry_") and f.endswith(".json")] \
        if os.path.isdir(path) else [path]
    out = {}
    for fp in files:
        try:
            with open(fp, encoding="utf-8") as f:
                obj = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: skipping {fp}: {e}", file=sys.stderr)
            continue
        tel = Telemetry(host=obj.get("host_name", ""), pid=int(obj.get("process_id", 0)),
                        sample_interval_ms=int(obj.get("sample_interval_ms", 0)),
                        hist_interval_ns=int(obj.get("histogram_bucket_interval_ns", 30000)),
                        hist_max_buckets=int(obj.get("histogram_max_buckets", 5)),
                        truncated=bool(obj.get("hw_samples_truncated", False)))
        meta = {d["device_id"]: d for d in obj.get("devices", [])}
        keep = (wanted or {}).get((tel.host, tel.pid))
        by_dev = defaultdict(list)
        for s in obj.pop("hw_samples", []):
            if keep is None or s["device_id"] in keep:
                by_dev[s["device_id"]].append(s)
        del obj
        for dev_id, samples in by_dev.items():
            samples.sort(key=lambda s: s["ts_us"])
            cols = {k: [s.get(k) for s in samples] for k in samples[0]}
            m = meta.get(dev_id, {})
            tel.devices[dev_id] = Device(dev_id, m.get("roce_device") or samples[0].get("roce_device", ""),
                                         m.get("hw_type", ""), float(m.get("link_gbps") or 0),
                                         SeriesMap(cols))
        by_dev.clear()
        out[(tel.host, tel.pid)] = tel
    return out


# ============================================================================
# Counter windows and bounds
# ============================================================================

def union(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping intervals; back-to-back collectives become one span."""
    merged: List[List[int]] = []
    for a, b in sorted(i for i in intervals if i[1] > i[0]):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _overlap(spans: List[Tuple[int, int]], a: float, b: float) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in spans)


@dataclass
class Bound:
    """Amount of a counter that fell inside the spans."""
    lb: float        # at least this much
    ub: float        # at most this much; INF when part of the spans was not sampled
    fill: float      # share of the touched counter windows the spans occupy


def stretches(spans: List[Tuple[int, int]], gap_us: float = MERGE_GAP_US) -> List[Tuple[int, int]]:
    """Back-to-back collectives leave gaps of a few us between their exec
    intervals. Joined across gaps no longer than gap_us they form the busy
    stretches a slowly refreshing counter can be compared against at all."""
    out: List[List[int]] = []
    for a, b in union(spans):
        if out and a - out[-1][1] <= gap_us:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def bound(ser: Optional[Series], spans: List[Tuple[int, int]], pad_us: float,
          duration: bool) -> Optional[Bound]:
    """Bound what `ser` counted inside `spans`. A counter that refreshes pad_us
    late shows in sample window [a, b] what happened in [a - pad_us, b]. That
    increment can belong to the spans if the two overlap (upper bound), and does
    if a busy stretch holds all of it (lower bound). A duration counter's lower
    bound also gives up the gap time between collectives in that window, since
    the pause could have fallen there; event counts in those short gaps are
    charged to the collectives."""
    if ser is None or not spans:
        return None
    spans = union(spans)          # sorted and disjoint, so one forward pass will do
    runs = stretches(spans)
    busy = float(sum(b - a for a, b in spans))
    total = inner = window = inside = 0.0
    j = r = 0
    for k in range(len(ser.ts) - 1):
        a, b = ser.ts[k], ser.ts[k + 1]
        if b <= a:
            continue
        lo = a - pad_us
        while j < len(spans) and spans[j][1] <= lo:
            j += 1
        while r < len(runs) and runs[r][1] <= lo:
            r += 1
        if j == len(spans):
            break
        if spans[j][0] >= b:
            continue
        dv = ser.vals[k + 1] - ser.vals[k]
        total += dv
        window += b - a
        busy_lo_b = 0.0
        m = j
        while m < len(spans) and spans[m][0] < b:
            sa, sb = spans[m]
            inside += max(0.0, min(b, sb) - max(a, sa))
            busy_lo_b += max(0.0, min(b, sb) - max(lo, sa))
            m += 1
        if r < len(runs) and runs[r][0] <= lo and b <= runs[r][1]:
            gaps = (b - lo) - busy_lo_b
            inner += max(0.0, dv - gaps) if duration else dv
    sampled = _overlap(spans, ser.ts[0], ser.ts[-1])
    uncovered = max(0.0, busy - sampled)
    fill = inside / window if window > 0 else 0.0
    lb = inner
    if duration:
        if pad_us == 0:
            lb = max(lb, total - (window - inside))
        ub = min(total, sampled) + uncovered
    else:
        ub = total if uncovered <= 0.01 * busy else INF
    return Bound(lb, ub, fill)


def refresh_us(dev: Device, interval_us: float) -> float:
    """How often this NIC's ethtool counters really change, measured on its byte
    counter while RCCL was sending. Some drivers refresh far slower than the
    sampler reads; a counter that steps once a second cannot be split finer."""
    nic, sw = dev.series.get("nic_tx_bytes"), dev.series.get("tx_bytes")
    if nic is None or sw is None or interval_us <= 0:
        return interval_us
    moving = [sw.ts[i] for i in range(1, len(sw.ts)) if sw.vals[i] > sw.vals[i - 1]]
    if len(moving) < 5:
        return interval_us
    t0, t1 = moving[0], moving[-1]
    changes = [nic.ts[i] for i in range(1, len(nic.ts))
               if nic.vals[i] > nic.vals[i - 1] and t0 <= nic.ts[i] <= t1]
    if len(changes) < 3:
        return max(interval_us, t1 - t0)   # effectively stale while RCCL sent
    gaps = sorted(b - a for a, b in zip(changes, changes[1:]))
    return max(interval_us, gaps[len(gaps) // 2])


# ============================================================================
# Which PFC priority, and in which unit
# ============================================================================

@dataclass
class Lane:
    """The PFC priority RCCL's traffic uses, and how that was decided."""
    prio: Optional[int]
    source: str                     # given | traffic | most-paused | none
    share: float = 0.0              # share of the used NICs' bytes on that priority
    other_pause: Dict[int, float] = field(default_factory=dict)  # pause on other priorities
    all_zero: bool = False          # PFC counters exist and never moved, any priority


def _sum_prio(tels, used, key, p):
    tot, seen = 0.0, False
    for k, tel in tels.items():
        for d in used.get(k, []):
            s = tel.devices[d].series.get(f"{key}[{p}]")
            if s:
                seen, tot = True, tot + s.total()
    return tot, seen


def pick_lane(tels: Dict[Tuple[str, int], Telemetry], used: Dict[Tuple[str, int], List[int]],
              override: Optional[int] = None) -> Lane:
    nbytes = [sum(_sum_prio(tels, used, k, p)[0] for k in ("prio_tx_bytes", "prio_rx_bytes"))
              for p in range(NUM_PFC_PRIO)]
    pause, seen = [], False
    for p in range(NUM_PFC_PRIO):
        tot = 0.0
        for k in ("pfc_rx_pause_us", "pfc_tx_pause_us", "pfc_rx_frames", "pfc_tx_frames"):
            v, s = _sum_prio(tels, used, k, p)
            tot, seen = tot + v, seen or s
        pause.append(tot)
    if override is not None:
        prio, source = override, "given"
    elif any(nbytes):
        prio, source = max(range(NUM_PFC_PRIO), key=lambda p: nbytes[p]), "traffic"
    elif any(pause):
        prio, source = max(range(NUM_PFC_PRIO), key=lambda p: pause[p]), "most-paused"
    elif seen:
        prio, source = None, "none"
    else:
        return Lane(None, "none")
    share = nbytes[prio] / sum(nbytes) if prio is not None and sum(nbytes) else 0.0
    other = {p: v for p, v in enumerate(pause) if v and p != prio}
    return Lane(prio, source, share, other, all_zero=seen and not any(pause))


def pause_unit(dev: Device, prio: int) -> Optional[str]:
    """"time" when the NIC reports pause time, "frames" when its pause-time
    counters never move although pause frames do (Thor2 on MI355X reported
    0 us for millions of frames), None when it reports neither."""
    def moved(k):
        return any((s := dev.series.get(f"{k}[{p}]")) is not None and s.total() > 0
                   for p in range(NUM_PFC_PRIO))
    has_time = f"pfc_rx_pause_us[{prio}]" in dev.series or f"pfc_tx_pause_us[{prio}]" in dev.series
    has_frames = f"pfc_rx_frames[{prio}]" in dev.series or f"pfc_tx_frames[{prio}]" in dev.series
    if has_time and (moved("pfc_rx_pause_us") or moved("pfc_tx_pause_us")
                     or not (moved("pfc_rx_frames") or moved("pfc_tx_frames"))):
        return "time"
    if has_frames:
        return "frames"
    return None


# ============================================================================
# Analysis
# ============================================================================

@dataclass
class Thresholds:
    pause_pct: float = 5.0      # % of wall time paused
    pause_fps: float = 100.0    # pause frames per second, when only frames are counted
    marks_pct: float = 1.0      # % of packets ECN-marked or answered with a CNP
    link_busy_pct: float = 80.0  # link use that counts as busy
    link_other_pct: float = 20.0  # non-RCCL share of the link that counts as shared


@dataclass
class Finding:
    name: str
    level: str      # yes | maybe
    detail: str


@dataclass
class NicStats:
    roce: str
    unit: Optional[str]              # "%" or "/s" for the pause columns
    rx_pause: Optional[Tuple[float, float]] = None   # (lb, ub) in unit
    tx_pause: Optional[Tuple[float, float]] = None
    loss: Optional[Tuple[float, float]] = None       # fabric-loss events (lb, ub)
    rnr: Optional[Tuple[float, float]] = None        # host-not-ready events (lb, ub)
    marks_pct: Optional[float] = None
    link_pct: Optional[float] = None                 # busier direction, % of link speed
    rccl_pct: Optional[float] = None                 # RCCL's share of that traffic
    nic_stale: bool = False
    sw_tx_per_coll: float = 0.0
    cmpl_hist: List[int] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)


@dataclass
class GroupResult:
    host: str
    pid: int
    rank: int
    coll: str
    msg_size: int
    n: int
    busy_us: float
    mean_busbw: float
    accl_class: str
    nics: List[NicStats]
    verdict: str = ""
    inconclusive: List[str] = field(default_factory=list)
    # Share of this group's collectives whose network window overlaps another
    # collective's on the same process; their per-collective bytes are shared.
    overlap_pct: float = 0.0
    spans: List[Tuple[int, int]] = field(default_factory=list)


def _level(lb: float, ub: float, thr: float, strict: bool = False) -> Optional[str]:
    hit = (lambda v: v > thr) if strict else (lambda v: v >= thr)
    if hit(lb):
        return "yes"
    if ub == INF or hit(ub):
        return "maybe"
    return None


def _rate(b: Bound, busy: float, unit: str) -> Tuple[float, float]:
    """(lb, ub) of a bound as % of busy time or per second of busy time."""
    scale = (100.0 / busy) if unit == "%" else (1e6 / busy)
    return b.lb * scale, (INF if b.ub == INF else b.ub * scale)


def _rccl_bytes(tels, host, roce, spans, key) -> float:
    """RCCL's own bytes on one NIC during spans, summed over every process of the
    job on that host that drives it."""
    tot = 0.0
    for (h, _), tel in tels.items():
        if h != host:
            continue
        for dev in tel.devices.values():
            if dev.roce == roce and key in dev.series:
                tot += sum(dev.series[key].delta(a, b) for a, b in spans)
    return tot


def analyze(recs: List[CollRecord], tels: Dict[Tuple[str, int], Telemetry],
            lane: Lane, thr: Thresholds) -> List[GroupResult]:
    by_proc = defaultdict(list)
    for r in recs:
        by_proc[(r.host, r.pid)].append(r)

    results = []
    for key, proc_recs in sorted(by_proc.items()):
        tel = tels.get(key)
        used = devices_used(proc_recs, tel) if tel else []
        overlapped = _overlapping_windows(proc_recs)
        groups = defaultdict(list)
        for r in proc_recs:
            groups[(r.rec.coll, r.rec.msg_size)].append(r)
        interval = (tel.sample_interval_ms * 1000.0) if tel else 0.0

        for (coll, size), g in sorted(groups.items()):
            spans = union([(r.exec_start_us, r.exec_end_us) for r in g])
            busy = float(sum(b - a for a, b in spans))
            agg = accl_report.aggregate([r.rec for r in g])[(coll, size)]
            gr = GroupResult(host=key[0], pid=key[1], rank=g[0].rank, coll=coll, msg_size=size,
                             n=len(g), busy_us=busy, mean_busbw=agg.mean_busbw,
                             accl_class=accl_report.classify_bottleneck(agg), nics=[], spans=spans)
            with_win = [r for r in g if r.net_start_us]
            if with_win:
                gr.overlap_pct = 100.0 * sum(id(r) in overlapped for r in with_win) / len(with_win)
            if tel is None or interval <= 0:
                gr.inconclusive.append("no telemetry samples for this process")
            elif not used:
                gr.inconclusive.append("no NIC of this process carried traffic in telemetry")
            for dev_id in used:
                gr.nics.append(_nic_stats(tel.devices[dev_id], g, spans, busy, interval,
                                          lane, thr, tels, key[0], gr))
            gr.verdict = _rank_verdict(gr)
            results.append(gr)
    tag_echo(results)
    return results


def _nic_stats(dev, g, spans, busy, interval, lane, thr, tels, host, gr) -> NicStats:
    s = dev.series
    pad = refresh_us(dev, interval)
    pad = pad if pad > 1.5 * interval else 0.0
    unit = pause_unit(dev, lane.prio) if lane.prio is not None else None
    ns = NicStats(roce=dev.roce or f"dev{dev.device_id}",
                  unit={"time": "%", "frames": "/s"}.get(unit))
    busy_s = busy / 1e6

    # PFC pause on RCCL's priority
    if unit is not None and busy > 0:
        k = "pause_us" if unit == "time" else "frames"
        limit = thr.pause_pct if unit == "time" else thr.pause_fps
        for direction, name, what in (("rx", "FABRIC-CONGESTION", "switch paused our transmit"),
                                      ("tx", "HOST-BACKPRESSURE", "our NIC paused the switch")):
            b = bound(s.get(f"pfc_{direction}_{k}[{lane.prio}]"), spans, pad, unit == "time")
            if b is None:
                continue
            r = _rate(b, busy, ns.unit)
            setattr(ns, f"{direction}_pause", r)
            lvl = _level(r[0], r[1], limit)
            if lvl:
                ns.findings.append(Finding(name, lvl, f"{what} {_fmt_range(r, ns.unit)}"))

    # Loss and receiver-not-ready events: any is worth reporting.
    for attr, keys, name, what in (("loss", LOSS_FABRIC, "FABRIC-LOSS", "drops/NAKs/timeouts"),
                                   ("rnr", LOSS_HOST, "HOST-RNR", "receiver-not-ready events")):
        bs = [bound(s.get(k), spans, pad, False) for k in keys]
        bs = [b for b in bs if b is not None]
        if not bs:
            continue
        lb = sum(b.lb for b in bs)
        ub = INF if any(b.ub == INF for b in bs) else sum(b.ub for b in bs)
        setattr(ns, attr, (lb, ub))
        lvl = _level(lb, ub, 0.0, strict=True)
        if lvl:
            ns.findings.append(Finding(name, lvl, f"{_fmt_range((lb, ub), '')} {what}"))

    # ECN marks and CNPs as a share of packets, from the same counter windows.
    shares = []
    for mark, pkts in (("ecn_marked_pkts", "nic_rx_packets"), ("cnp_rcvd", "nic_tx_packets")):
        bm, bp = bound(s.get(mark), spans, pad, False), bound(s.get(pkts), spans, pad, False)
        if bm is not None and bp is not None and bp.ub not in (0, INF):
            shares.append(100.0 * bm.ub / bp.ub)
    if shares:
        ns.marks_pct = max(shares)
        if ns.marks_pct >= thr.marks_pct:
            ns.findings.append(Finding("CONGESTION-MARKS", "yes",
                                       f"{ns.marks_pct:.1f}% of packets ECN-marked or CNP'd"))

    # Link use, and how much of it was RCCL. Port counters include all traffic.
    if busy > 0:
        stale = False
        best = None
        for d in ("tx", "rx"):
            nic, sw = s.get(f"nic_{d}_bytes"), s.get(f"{d}_bytes")
            if nic is None:
                continue
            port = sum(nic.delta(a, b) for a, b in spans)
            ours = _rccl_bytes(tels, host, dev.roce, spans, f"{d}_bytes")
            if sw is not None and ours > 0 and port == 0:
                stale = True
                continue
            if port > 0 and dev.link_gbps > 0:
                util = 100.0 * port * 8 / (busy_s * dev.link_gbps * 1e9)
                share = min(100.0, 100.0 * ours / port)
                if best is None or util > best[0]:
                    best = (util, share)
        ns.nic_stale = stale and best is None
        if best:
            ns.link_pct, ns.rccl_pct = best
            other = best[0] * (100.0 - best[1]) / 100.0
            if best[0] >= thr.link_busy_pct and other >= thr.link_other_pct:
                # Rates over many counter refreshes are firm; over a few, rough.
                ns.findings.append(Finding(
                    "LINK-SHARED", "yes" if busy >= 5 * max(pad, interval) else "maybe",
                    f"link {best[0]:.0f}% busy, RCCL {best[1]:.0f}% of it"))

    # Per-collective RCCL bytes and completion latency, exact from accl.
    with_net = [r for r in g if dev.device_id in r.net]
    hist: List[int] = []
    for r in with_net:
        d = r.net[dev.device_id]
        ns.sw_tx_per_coll += d.get("tx_bytes", 0)
        h = d.get("completion_hist", [])
        hist += [0] * (len(h) - len(hist))
        for i, c in enumerate(h):
            hist[i] += c
    if with_net:
        ns.sw_tx_per_coll /= len(with_net)
    ns.cmpl_hist = hist

    # Spans the samples did not cover leave every finding open.
    probe = next((s[k] for k in ("tx_bytes", "nic_tx_bytes") if k in s), None)
    b = bound(probe, spans, 0, False)
    if (b is None or b.ub == INF) and COVERAGE not in gr.inconclusive:
        gr.inconclusive.append(COVERAGE)
    return ns


def devices_used(recs: List[CollRecord], tel: Telemetry) -> List[int]:
    """NICs this process drove: those the per-collective deltas name, else those
    whose SW byte counters moved during the run."""
    ids = sorted({d for r in recs for d in r.net})
    if ids:
        return [d for d in ids if d in tel.devices]
    return sorted(dev.device_id for dev in tel.devices.values()
                  if any(dev.series.get(k) and dev.series[k].total() > 0 for k in ("tx_bytes", "rx_bytes")))


def _overlapping_windows(proc_recs: List[CollRecord]) -> set:
    wins = sorted((r.net_start_us, r.net_end_us, id(r)) for r in proc_recs
                  if r.net_start_us and r.net_end_us > r.net_start_us)
    out, reach, reach_id = set(), 0, None
    for a, b, rid in wins:
        if a < reach:
            out.update((rid, reach_id))
        if b > reach:
            reach, reach_id = b, rid
    return out


ORDER = ["HOST-BACKPRESSURE", "HOST-RNR", "FABRIC-LOSS", "FABRIC-CONGESTION",
         "LINK-SHARED", "CONGESTION-MARKS"]


def _rank_verdict(gr: GroupResult) -> str:
    found = {}
    for ns in gr.nics:
        for f in ns.findings:
            if f.name not in found or f.level == "yes":
                found[f.name] = f.level
    yes = [n for n in ORDER if found.get(n) == "yes"]
    maybe = [n for n in ORDER if found.get(n) == "maybe"]
    if yes:
        return "+".join(yes) + (f" (maybe {'+'.join(maybe)})" if maybe else "")
    if maybe or gr.inconclusive:
        why = ([f"possible {'+'.join(maybe)}"] if maybe else []) + gr.inconclusive[:1]
        return "INCONCLUSIVE (" + "; ".join(why) + ")"
    return f"CLEAN (accl: {gr.accl_class})"


def _pushback(gs: List[GroupResult]) -> List[Tuple[int, str, Optional[str], Tuple[float, float]]]:
    """NICs that sent any pause frames, proven or possible, whatever the threshold:
    one receiver that pushes back can stall many senders."""
    return [(g.rank, g.host.split(".")[0], ns.unit, ns.tx_pause) for g in gs for ns in g.nics
            if ns.tx_pause and (ns.tx_pause[1] == INF or ns.tx_pause[1] > 0)]


def _by_size(results: List[GroupResult]) -> Dict[Tuple[str, int], List[GroupResult]]:
    by = defaultdict(list)
    for g in results:
        by[(g.coll, g.msg_size)].append(g)
    return by


def tag_echo(results: List[GroupResult]):
    """Mark rx pause on a rank as a possible echo when some NIC of the same
    collective and size pushed back."""
    for gs in _by_size(results).values():
        pushers = {r for r, _, _, _ in _pushback(gs)}
        if not pushers:
            continue
        for g in gs:
            if g.rank not in pushers and "FABRIC-CONGESTION" in g.verdict:
                g.verdict = g.verdict.replace("FABRIC-CONGESTION", "FABRIC-CONGESTION(echo?)", 1)


def _echo_summary(gs, cong, fmt, thr: "Thresholds") -> str:
    """Senders were paused x, some receiver pushed back y. A receiver stopped for
    y of the time can hold its senders back for about that long, not more, so
    roughly x - y is left for the fabric. A rule of thumb: who sends to whom is
    not known, and hosts' clocks are only lined up by collective and size."""
    push = _pushback(gs)
    proven = [p for p in push if p[3][0] > 0]
    who = fmt(sorted({(r, h) for r, h, _, _ in (proven or push)}))
    verb = "paused the switch" if proven else "may have paused the switch"
    senders = [ns for g in gs for ns in g.nics
               if ns.rx_pause and any(f.name == "FABRIC-CONGESTION" and f.level == "yes" for f in ns.findings)]
    # Only the NICs being compared need a common unit; a quiet NIC elsewhere may
    # report pause time while these count frames.
    units = {ns.unit for ns in senders} | {p[2] for p in push}
    unit = units.pop() if len(units) == 1 else None
    xs = [ns.rx_pause[0] for ns in senders]
    x = max(xs) if xs else 0.0
    y = max(p[3][1] for p in push)
    sender = f"rx pauses on {len(cong)} rank(s) (at least {_fmt_num(x, unit)})"
    if unit is None or y == INF:
        return (f"HOST{'' if proven else '?'}: {who} {verb}; {sender} may be backpressure from it "
                "(amounts not comparable)")
    limit = thr.pause_pct if unit == "%" else thr.pause_fps
    if x - y >= limit:
        return (f"FABRIC+HOST{'' if proven else '?'}: switch paused senders at least {_fmt_num(x, unit)}; {who} {verb} "
                f"(at most {_fmt_num(y, unit)}), which explains at most that much, so roughly "
                f"{_fmt_num(x - y, unit)} or more is the fabric")
    return (f"HOST{'' if proven else '?'}: {who} {verb} (up to {_fmt_num(y, unit)}); {sender} "
            "are likely backpressure from it")


def job_summary(results: List[GroupResult], lane: Lane, thr: Optional["Thresholds"] = None
                ) -> Dict[Tuple[str, int], str]:
    """One conclusion per (collective, size) across all ranks. A NIC that paused
    the switch can explain rx pauses elsewhere, so pushback is checked first."""
    thr = thr or Thresholds()
    out = {}
    for k, gs in _by_size(results).items():
        def ranks(name, level="yes"):
            return sorted({(g.rank, g.host.split(".")[0]) for g in gs for ns in g.nics
                           for f in ns.findings if f.name == name and f.level == level})

        def fmt(rs):
            return ", ".join(f"rank {r} ({h})" for r, h in rs[:4]) + (" ..." if len(rs) > 4 else "")

        bp, rnr = ranks("HOST-BACKPRESSURE"), ranks("HOST-RNR")
        loss, cong = ranks("FABRIC-LOSS"), ranks("FABRIC-CONGESTION")
        shared, marks = ranks("LINK-SHARED"), ranks("CONGESTION-MARKS")
        if cong and _pushback(gs):
            msg = _echo_summary(gs, cong, fmt, thr)
        elif bp:
            msg = f"HOST: {fmt(bp)} paused the switch (could not take data in fast enough)"
        elif rnr:
            msg = f"HOST: receiver-not-ready events on {fmt(rnr)}"
        elif loss:
            msg = f"FABRIC: drops/NAKs/timeouts on {fmt(loss)}"
        elif cong:
            msg = f"FABRIC: switch paused sending on {len(cong)} rank(s), no NIC pushed back: {fmt(cong)}"
        elif shared:
            msg = f"LINK-SHARED: other traffic held the link on {fmt(shared)}"
        elif marks:
            msg = f"CONGESTION-MARKS on {fmt(marks)}"
        elif any(g.verdict.startswith("INCONCLUSIVE") for g in gs):
            maybes = sorted({f.name for g in gs for ns in g.nics for f in ns.findings if f.level == "maybe"})
            reasons = sorted({x for g in gs for x in g.inconclusive})
            msg = "INCONCLUSIVE: " + "; ".join(([f"possible {', '.join(maybes)}"] if maybes else [])
                                               + reasons[:2])
        else:
            msg = "CLEAN: no fabric signal in any counter"
            if lane.all_zero:
                msg += " (PFC counters zero on every priority: no pauses, or PFC is disabled)"
        if lane.source == "most-paused" and (cong or bp):
            msg += f" [priority {lane.prio} unconfirmed, see header]"
        out[k] = msg
    return out


# ============================================================================
# Text report
# ============================================================================

def _fmt_num(v: float, unit: Optional[str]) -> str:
    return f"{v:.1f}%" if unit == "%" else f"{v:.0f}{unit or ''}"


def _fmt_range(r: Optional[Tuple[float, float]], unit: str) -> str:
    if r is None:
        return "-"
    lb, ub = r
    nd = 1 if unit == "%" else 0
    if ub == INF:
        return f">={lb:.{nd}f}{unit}"
    if ub - lb <= max(0.5 if unit == "%" else 5, 0.1 * ub):
        return f"{(lb + ub) / 2:.{nd}f}{unit}"
    return f"{lb:.{nd}f}-{ub:.{nd}f}{unit}"


def hist_quantile(hist: List[int], q: float, interval_ns: int) -> Optional[float]:
    """Upper edge (us) of the bucket holding quantile q."""
    total = sum(hist)
    if total == 0:
        return None
    acc = 0
    for b, c in enumerate(hist):
        acc += c
        if acc >= q * total:
            return (b + 1) * interval_ns / 1000.0
    return len(hist) * interval_ns / 1000.0


def fmt_quantile(v: Optional[float]) -> str:
    """The last configured bucket also holds everything slower than its edge,
    so a quantile that lands there is a lower bound."""
    if v is None:
        return "-"
    if v >= HIST_MAX * HIST_NS / 1000.0:
        return f">={(HIST_MAX - 1) * HIST_NS / 1000.0:.0f}"
    return f"{v:.0f}"


def print_report(results: List[GroupResult], lane: Lane, thr: Thresholds, per_nic: bool,
                 warnings: List[str]):
    hosts = {r.host for r in results}
    ranks = {(r.host, r.rank) for r in results}
    print(f"ACCL fabric report: {len(ranks)} rank(s) on {len(hosts)} host(s)")
    if lane.prio is None:
        print("PFC: no pause counters for RCCL's priority" +
              (" (all PFC counters stayed zero)" if lane.all_zero else ""))
    elif lane.source == "given":
        print(f"PFC priority: {lane.prio} (--pfc-prio)")
    elif lane.source == "traffic":
        print(f"PFC priority: {lane.prio} (carries {100 * lane.share:.0f}% of the job's NIC bytes)")
    else:
        print(f"PFC priority: {lane.prio} UNCONFIRMED: no per-priority byte counters, so this is "
              "the most-paused priority, which may not be RCCL's. Pass --pfc-prio.")
    if lane.other_pause:
        print("  Pause on other priorities (does not slow RCCL's): " +
              ", ".join(f"prio {p}" for p in sorted(lane.other_pause)))
    if lane.all_zero:
        print("  Every PFC counter stayed zero: either nothing paused or PFC is disabled, so a")
        print("  CLEAN result cannot rule out a lossy fabric on its own; check the loss columns.")
    print(f"Thresholds: pause {thr.pause_pct:g}% of wall time (or {thr.pause_fps:g} frames/s where the NIC "
          f"counts frames only), marks {thr.marks_pct:g}% of packets, link busy {thr.link_busy_pct:g}% "
          f"with >= {thr.link_other_pct:g}% non-RCCL traffic")
    print("  rxPause = switch paused our transmit   txPause = our NIC paused the switch")
    print("  a-b = at least a, at most b (counters are sampled, not continuous)\n")

    hdr = (f"{'Coll':<12} {'Size':>5} {'Rank':>4} {'Host':<14} {'N':>4} {'wall ms':>8} "
           f"{'busBW':>6} {'NIC':<10} {'rxPause':>10} {'txPause':>10} {'loss':>6} {'rnr':>5} "
           f"{'mark%':>5} {'link%':>5} {'rccl%':>5} {'B/coll':>8} {'cmpl p50/p99':>13}  Verdict")
    print(hdr)
    print("-" * len(hdr))
    for g in sorted(results, key=lambda r: (r.coll, r.msg_size, r.host, r.rank)):
        nics = g.nics if per_nic else sorted(g.nics, key=lambda n: -len(n.findings))[:1]
        first = (f"{g.coll[:12]:<12} {accl_report.fmt_size(g.msg_size):>5} {g.rank:>4} "
                 f"{g.host.split('.')[0][:14]:<14} {g.n:>4} {g.busy_us / 1000:>8.2f} {g.mean_busbw:>6.1f}")
        if not nics:
            print(f"{first} {'-':<10}  {g.verdict}")
            continue
        for i, n in enumerate(nics):
            lead = first if i == 0 else " " * len(first)
            p50 = fmt_quantile(hist_quantile(n.cmpl_hist, 0.5, HIST_NS))
            p99 = fmt_quantile(hist_quantile(n.cmpl_hist, 0.99, HIST_NS))
            link = "stale" if n.nic_stale else ("-" if n.link_pct is None else f"{n.link_pct:.0f}")
            bpc = accl_report.fmt_size(int(n.sw_tx_per_coll)) + ("*" if g.overlap_pct > 10 else "")
            print(f"{lead} {n.roce[:10]:<10} {_fmt_range(n.rx_pause, n.unit or ''):>10} "
                  f"{_fmt_range(n.tx_pause, n.unit or ''):>10} {_fmt_range(n.loss, ''):>6} "
                  f"{_fmt_range(n.rnr, ''):>5} {'-' if n.marks_pct is None else f'{n.marks_pct:.1f}':>5} "
                  f"{link:>5} {'-' if n.rccl_pct is None else f'{n.rccl_pct:.0f}':>5} {bpc:>8} "
                  f"{p50 + '/' + p99:>13}  {g.verdict if i == 0 else ''}")

    print("\nSummary per collective and size (all ranks):")
    for (coll, size), msg in sorted(job_summary(results, lane, thr).items()):
        print(f"  {coll} {accl_report.fmt_size(size)}: {msg}")

    print("\nFindings:")
    print("  FABRIC-CONGESTION  switch paused our transmit; can be backpressure from a receiver,")
    print("                     marked (echo?) when a NIC of the same collective and size pushed back")
    print("  HOST-BACKPRESSURE  our NIC paused the switch: this host could not take data in")
    print("  FABRIC-LOSS        out-of-sequence drops, sequence-error NAKs, ACK timeouts")
    print("  HOST-RNR           receiver not ready / out of receive buffers")
    print("  LINK-SHARED        link busy, mostly with traffic that is not this job's RCCL")
    print("  CONGESTION-MARKS   ECN-marked packets or CNPs above the threshold share")
    print("  CLEAN              every counter below threshold; time went elsewhere")
    print("  INCONCLUSIVE       missing samples, or only an upper bound crosses a threshold")
    print("  Echo split (FABRIC+HOST): pause x on senders, pushback y on a receiver; y can hold")
    print("  senders back about that long, so x - y is left to the fabric. A rule of thumb: who")
    print("  sends to whom is not known, and hosts are matched by collective and size, not clock.")
    notes = list(warnings)
    if any(g.overlap_pct > 10 for g in results):
        notes.append("B/coll*: more than 10% of these collectives overlapped another one's network "
                     "window on the same rank, so their per-collective bytes include a neighbour's.")
    if any(n.nic_stale for g in results for n in g.nics):
        notes.append("link 'stale': the NIC byte counter did not move while RCCL sent; this driver "
                     "does not refresh it during traffic, so link use is unknown.")
    merged = [(g.busy_us, sum(y - x for x, y in stretches(g.spans)) - g.busy_us) for g in results if g.spans]
    if merged and sum(m for _, m in merged) > 0:
        share = 100.0 * sum(m for _, m in merged) / sum(b + m for b, m in merged)
        notes.append(f"Collectives less than {MERGE_GAP_US / 1000:g} ms apart are compared with the counters "
                     f"as one busy stretch; the gaps inside are {share:.2f}% of that time. Pause time in "
                     "them is left out of the lower bounds; pause frames in them are counted.")
    if any(n.unit == "/s" for g in results for n in g.nics):
        notes.append("Pause in /s: these NICs count pause frames but their pause-time counters never "
                     "move, so pause is pause frames per second, a rate of stop messages, not time.")
    if notes:
        print("\nNotes:")
        for n in notes:
            print(f"  - {n}")


# Set from the first telemetry file in load(); bucket layout is a run-wide setting.
HIST_NS = 30000
HIST_MAX = 5


# ============================================================================
# Perfetto / Chrome trace
# ============================================================================

def _lanes(recs: List[CollRecord]) -> Dict[int, int]:
    """id(record) -> lane. Back-to-back collectives overlap a little (the next
    starts before the last one's proxy work ends), and slices on one track must
    nest, so each goes on the first lane whose last slice has ended."""
    ends: List[int] = []
    out = {}
    for r in sorted(recs, key=lambda x: (x.exec_start_us, x.exec_end_us)):
        for i, e in enumerate(ends):
            if e <= r.exec_start_us:
                ends[i] = r.exec_end_us
                out[id(r)] = i
                break
        else:
            ends.append(r.exec_end_us)
            out[id(r)] = len(ends) - 1
    return out


def _queue_depth(recs: List[CollRecord], bin_us: int = 1000) -> List[Tuple[int, int]]:
    """(time, collectives enqueued but not yet started), the most per bin_us."""
    steps = sorted([(r.enqueue_us, 1) for r in recs if r.enqueue_us and r.exec_start_us] +
                   [(r.exec_start_us, -1) for r in recs if r.enqueue_us and r.exec_start_us])
    out, depth, cur_bin, peak = [], 0, None, 0
    for t, d in steps:
        depth += d
        b = t - t % bin_us
        if b != cur_bin:
            if cur_bin is not None:
                out.append((cur_bin, peak))
            cur_bin, peak = b, depth
        peak = max(peak, depth)
    if cur_bin is not None:
        out += [(cur_bin, peak), (cur_bin + bin_us, depth)]
    return out


def _refresh_points(dev: Device, interval_us: float) -> Optional[List[int]]:
    """Where this NIC's hardware counters really change: the steps of its byte
    counter while it refreshes slower than it is sampled, else None (use every
    sample). Rates over 10 ms samples of a counter that steps once a second are
    0 most of the time and 100x too high once a second."""
    nic = dev.series.get("nic_tx_bytes")
    if nic is None or refresh_us(dev, interval_us) <= 1.5 * interval_us:
        return None
    # Only between steps: the first step also holds what the NIC counted before
    # sampling began, so a segment from the first sample to it would read high.
    pts = [nic.ts[i] for i in range(1, len(nic.ts)) if nic.vals[i] != nic.vals[i - 1]]
    return pts if len(pts) >= 3 else None


def write_trace(recs: List[CollRecord], tels: Dict[Tuple[str, int], Telemetry],
                used: Dict[Tuple[str, int], List[int]], lane: Lane, out_path: str):
    """Chrome JSON trace: one process per rank. Collectives are slices, on as
    many lanes as they overlap, each with a "network" child for its network
    window; counters show the queue and, per NIC, rates. CLOCK_MONOTONIC has a
    different origin on every host, so each host is shifted to start at 0;
    lining hosts up is therefore approximate."""
    events = []
    # Each host starts at 0 from its earliest event, enqueues included: a
    # negative timestamp is an error in Perfetto.
    host_t0 = {}
    for r in recs:
        for t in (r.enqueue_us, r.exec_start_us):
            if t:
                host_t0[r.host] = min(host_t0.get(r.host, t), t)

    by_proc = defaultdict(list)
    for r in recs:
        if r.exec_start_us and r.exec_end_us > r.exec_start_us:
            by_proc[(r.host, r.pid)].append(r)
    procs = sorted({(r.host, r.pid, r.rank) for r in recs})
    pid_of = {}
    for i, (host, pid, rank) in enumerate(procs, start=1):
        pid_of[(host, pid)] = i
        events.append({"ph": "M", "name": "process_name", "pid": i,
                       "args": {"name": f"rank {rank} ({host}, pid {pid})"}})
        events.append({"ph": "M", "name": "process_sort_index", "pid": i, "args": {"sort_index": rank}})

    for key, prs in by_proc.items():
        p, t0 = pid_of[key], host_t0[key[0]]
        tel = tels.get(key)
        lanes = _lanes(prs)
        for k in range(max(lanes.values()) + 1):
            events.append({"ph": "M", "name": "thread_name", "pid": p, "tid": k + 1,
                           "args": {"name": "collectives" if k == 0 else f"collectives (overlap {k + 1})"}})
            events.append({"ph": "M", "name": "thread_sort_index", "pid": p, "tid": k + 1,
                           "args": {"sort_index": k}})
        for r in prs:
            tid = lanes[id(r)] + 1
            args = {"seq": r.rec.sn, "algo": r.rec.algo, "proto": r.rec.proto,
                    "exec_us": r.rec.exec_time_us, "busbw_gbs": r.rec.busbw_gbs,
                    "queued_us": (r.exec_start_us - r.enqueue_us) if r.enqueue_us else None,
                    "gpu_kernel_us": r.rec.gpu_kernel_avg_us,
                    "proxy_network_us": r.rec.proxy_network_us,
                    "proxy_peer_wait_us": r.rec.proxy_peer_wait_us,
                    "proxy_gpu_wait_us": r.rec.proxy_gpu_wait_us,
                    "proxy_flush_us": r.rec.proxy_flush_us}
            net = {}
            for dev_id, d in r.net.items():
                name = tel.devices[dev_id].roce if tel and dev_id in tel.devices else f"dev{dev_id}"
                net[f"{name} tx_bytes"] = d.get("tx_bytes", 0)
                net[f"{name} rx_bytes"] = d.get("rx_bytes", 0)
            args.update(net)
            events.append({"ph": "X", "pid": p, "tid": tid,
                           "name": f"{r.rec.coll} {accl_report.fmt_size(r.rec.msg_size)}",
                           "ts": r.exec_start_us - t0, "dur": r.exec_end_us - r.exec_start_us,
                           "args": args})
            a = max(r.net_start_us, r.exec_start_us)
            b = min(r.net_end_us, r.exec_end_us)
            if r.net_start_us and b > a:
                events.append({"ph": "X", "pid": p, "tid": tid, "name": "network",
                               "ts": a - t0, "dur": b - a, "args": net})
        for t, depth in _queue_depth(prs):
            # Bins start on 1 ms marks, which can fall just before the first enqueue.
            events.append({"ph": "C", "pid": p, "name": "collectives queued",
                           "ts": max(0, t - t0), "args": {"waiting to start": depth}})

    for (host, pid), tel in sorted(tels.items()):
        if (host, pid) not in pid_of or host not in host_t0:
            continue
        p, t0 = pid_of[(host, pid)], host_t0[host]
        interval = tel.sample_interval_ms * 1000.0
        for dev in (tel.devices[d] for d in used.get((host, pid), [])):
            s = dev.series
            hw_points = _refresh_points(dev, interval)
            if hw_points:
                hw_points = [t for t in hw_points if t >= t0]

            def rate_track(title, parts, scale, hw=True):
                # Rate between consecutive points: the NIC's refresh steps for
                # hardware counters that refresh slowly, else every sample.
                have = [(label, s[k]) for label, k in parts if k in s]
                if not have:
                    return
                pts = hw_points if hw and hw_points else [t for t in have[0][1].ts if t >= t0]
                for a, b in zip(pts, pts[1:]):
                    if b > a:
                        events.append({"ph": "C", "pid": p, "name": f"{dev.roce} {title}", "ts": a - t0,
                                       "args": {label: round(ser.delta(a, b) / (b - a) * scale, 4)
                                                for label, ser in have}})

            unit = pause_unit(dev, lane.prio) if lane.prio is not None else None
            if unit == "time":
                rate_track(f"PFC pause % (prio {lane.prio})",
                           [("rx: switch paused us", f"pfc_rx_pause_us[{lane.prio}]"),
                            ("tx: we paused switch", f"pfc_tx_pause_us[{lane.prio}]")], 100.0)
            if unit is not None:
                rate_track(f"PFC pause frames/s (prio {lane.prio})",
                           [("rx: from switch", f"pfc_rx_frames[{lane.prio}]"),
                            ("tx: to switch", f"pfc_tx_frames[{lane.prio}]")], 1e6)
            rate_track("NIC GB/s (all port traffic)", [("tx", "nic_tx_bytes"), ("rx", "nic_rx_bytes")], 1e6 / 1e9)
            rate_track("RCCL GB/s", [("tx", "tx_bytes"), ("rx", "rx_bytes")], 1e6 / 1e9, hw=False)
            rate_track("ECN/CNP per ms", [("ecn marked", "ecn_marked_pkts"),
                                          ("cnp sent", "cnp_sent"), ("cnp rcvd", "cnp_rcvd")], 1e3)
            rate_track("loss events per s", [(k, k) for k in LOSS_FABRIC + LOSS_HOST], 1e6)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, f)
    print(f"Wrote {len(events)} events to {out_path}. Open it in https://ui.perfetto.dev")
    if len(host_t0) > 1:
        print("Note: each host's clock is shifted to start at 0; cross-host alignment is approximate.")


# ============================================================================
# Main
# ============================================================================

def _comm_overlap_warnings(recs: List[CollRecord]) -> List[str]:
    """Counters cannot tell two communicators apart, so time where two of them
    run at once is charged to both."""
    by_proc = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by_proc[(r.host, r.pid)][r.comm_hash].append((r.exec_start_us, r.exec_end_us))
    out = []
    for (host, pid), comms in by_proc.items():
        spans = {c: union(v) for c, v in comms.items()}
        cs = sorted(spans)
        for i, a in enumerate(cs):
            for b in cs[i + 1:]:
                both = sum(_overlap(spans[b], x, y) for x, y in spans[a])
                if both > 0:
                    out.append(f"{host.split('.')[0]} pid {pid}: communicators {a} and {b} ran at the "
                               f"same time for {both / 1000:.1f} ms; NIC counters in that time are "
                               "charged to both.")
    return out


def load(args):
    global HIST_NS, HIST_MAX
    warnings = []
    recs, no_ts = load_accl(args.accl, args.warmup)
    if no_ts:
        warnings.append(f"{no_ts} accl record(s) have no host_ts (plugin built before this report); "
                        "they are ignored.")
    if not recs:
        sys.exit("No usable accl records found in " + args.accl)
    # Processes whose records name their NICs only need those NICs' samples.
    wanted = defaultdict(set)
    for r in recs:
        wanted[(r.host, r.pid)].update(r.net)
    tels = load_telemetry(args.telemetry, {k: v for k, v in wanted.items() if v})
    if not tels:
        warnings.append("no rccl_telemetry_*.json found; every result is INCONCLUSIVE.")
    else:
        first = next(iter(tels.values()))
        HIST_NS, HIST_MAX = first.hist_interval_ns, first.hist_max_buckets
        trunc = [t for t in tels.values() if t.truncated]
        if trunc:
            warnings.append(f"telemetry sampling stopped early in {len(trunc)} process(es): the "
                            "hw_samples buffer filled. Raise RCCL_TELEMETRY_SAMPLE_MS for long runs.")
        if not any(t.sample_interval_ms for t in tels.values()):
            warnings.append("telemetry has no hw_samples; set RCCL_TELEMETRY_SAMPLE_MS (e.g. 10).")
    missing = {(r.host, r.pid) for r in recs} - set(tels)
    if missing and tels:
        warnings.append(f"{len(missing)} process(es) have accl records but no telemetry file "
                        "(check RCCL_TELEMETRY_OUTPUT_DIR points at shared storage).")
    warnings += _comm_overlap_warnings(recs)
    used = {}
    by_proc = defaultdict(list)
    for r in recs:
        by_proc[(r.host, r.pid)].append(r)
    for key, rs in by_proc.items():
        if key in tels:
            used[key] = devices_used(rs, tels[key])
    slow = []
    for (host, pid), ids in used.items():
        interval = tels[(host, pid)].sample_interval_ms * 1000.0
        for d in ids:
            r = refresh_us(tels[(host, pid)].devices[d], interval)
            if interval and r > 1.5 * interval:
                slow.append((r, interval))
    if slow:
        lo, hi = min(r for r, _ in slow) / 1e6, max(r for r, _ in slow) / 1e6
        warnings.append(f"{len(slow)} NIC(s) refresh their counters only every {lo:.2f}-{hi:.2f} s, slower "
                        f"than the {slow[0][1] / 1000:.0f} ms sampling, so counts cannot be placed finer than "
                        f"that. For a firm result run each collective size for at least ~{5 * hi:.0f} s.")
    return recs, tels, used, pick_lane(tels, used, args.pfc_prio), warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("report", "trace"):
        p = sub.add_parser(name)
        p.add_argument("--accl", required=True, help="accl JSONL file or directory")
        p.add_argument("--telemetry", required=True, help="rccl_telemetry JSON file or directory")
        p.add_argument("--pfc-prio", type=int, choices=range(NUM_PFC_PRIO), default=None,
                       help="PFC priority RCCL traffic uses (default: the one carrying most bytes)")
        p.add_argument("--warmup", type=int, default=5, help="skip collectives with seq < N")
        if name == "report":
            d = Thresholds()
            p.add_argument("--threshold-pct", type=float, default=d.pause_pct,
                           help="pause share of wall time that counts (default %(default)s)")
            p.add_argument("--threshold-fps", type=float, default=d.pause_fps,
                           help="pause frames per second that count, on NICs that count frames "
                                "only (default %(default)s)")
            p.add_argument("--threshold-marks-pct", type=float, default=d.marks_pct,
                           help="ECN-marked or CNP share of packets that counts (default %(default)s)")
            p.add_argument("--link-busy-pct", type=float, default=d.link_busy_pct,
                           help="link use that counts as busy (default %(default)s)")
            p.add_argument("--link-other-pct", type=float, default=d.link_other_pct,
                           help="non-RCCL share of the link that counts as shared (default %(default)s)")
            p.add_argument("--per-nic", action="store_true", help="one row per NIC, not just the worst")
        else:
            p.add_argument("-o", "--output", default="accl_fabric_trace.json")
    args = ap.parse_args()

    recs, tels, used, lane, warnings = load(args)
    if args.cmd == "report":
        thr = Thresholds(args.threshold_pct, args.threshold_fps, args.threshold_marks_pct,
                         args.link_busy_pct, args.link_other_pct)
        results = analyze(recs, tels, lane, thr)
        resets = sum(s.resets for t in tels.values() for d in t.devices.values() for s in d.series.built())
        if resets:
            warnings.append(f"{resets} NIC counter reset(s) during the run; counts continue across them.")
        print_report(results, lane, thr, args.per_nic, warnings)
    else:
        write_trace(recs, tels, used, lane, args.output)


if __name__ == "__main__":
    main()
