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

Observations, per NIC (what a counter showed; not yet a diagnosis):
  FABRIC-CONGESTION  the switch paused our transmit (PFC frames received)
  HOST-BACKPRESSURE  our NIC paused the switch: this host could not take data in
  OUT-OF-SEQ         packets this NIC received out of sequence (receiver side)
  SEQ-NAK            sequence-error NAKs this NIC got back (sender side)
  ACK-TIMEOUT        this NIC's sends timed out waiting for an ACK (sender side)
  RX-NO-BUFFER       this NIC had no receive buffer posted (this host's receiver)
  PEER-RNR           receiver-not-ready NAKs this NIC got: one of its peers was not ready
  CQ-ERROR           RCCL saw completion errors in these collectives (from accl)
  LINK-SHARED        the link was busy and mostly with traffic that is not RCCL
  CONGESTION-MARKS   ECN-marked share of received packets, or CNPs per second, above a threshold
Rx pause is not proof of a fabric fault: a receiver that pushes back makes
the switch pause the senders. The per-collective summary therefore blames a
host first when any of its NICs paused the switch.

Drop diagnosis, per collective and size: the delivery/recovery signals
(OUT-OF-SEQ, SEQ-NAK, ACK-TIMEOUT, completion errors) are weighed against
endpoint evidence (receive buffers, peer readiness, backpressure, NIC
progress) and land in HOST_LIKELY, FABRIC_LIKELY, MIXED or INCONCLUSIVE with
a confidence; see diagnose_drops() for the rules.

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
# Delivery and recovery symptoms: (telemetry column, observation, what it counts).
# Each is kept apart. One lost packet can show up as an out-of-sequence packet
# at its receiver, a sequence-error NAK at its sender and a retransmission, so
# their sum is not a count of lost packets. None of them alone proves a fabric
# fault: a timeout also follows a lost ACK, a long delay, or an endpoint that
# stopped answering.
SYMPTOMS = (("oos_drop_count", "OUT-OF-SEQ", "packets received out of sequence (receiver side)"),
            ("seq_err_naks_rcvd", "SEQ-NAK", "sequence-error NAKs received (sender side)"),
            ("local_ack_timeout_err", "ACK-TIMEOUT", "ACK timeouts (sender side)"))
# Receiver not ready, by where the counter sits. out_of_buffer counts on the
# receiver that had no buffer posted: this host. rnr_retry_err counts on the
# sender that got RNR NAKs back: the not-ready receiver is one of its peers.
RNR_LOCAL = ("out_of_buffer",)
RNR_PEER = ("rnr_retry_err",)
CONFIDENCE = ("high", "medium", "low")
INF = math.inf
PAUSE_QUANTA_BITS = 65535 * 512   # longest pause one PFC frame can ask for, in bit times
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
    # Why this record's file cannot be trusted to be complete, "" if it can.
    accl_incomplete: str = ""
    # Bytes of the receive that opened the network window: counted before the
    # start snapshot, so in no NIC's delta. One number per collective.
    rx_before: int = 0


def load_accl(path: str, warmup: int) -> Tuple[List[CollRecord], int]:
    """Returns (records, number skipped for lacking usable host timing)."""
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".jsonl")] \
        if os.path.isdir(path) else [path]
    out, no_ts = [], 0
    for fp in files:
        first = len(out)
        summary = None
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "summary" in obj:
                    summary = obj["summary"]
                    continue
                cp, hdr = obj.get("coll_perf"), obj.get("header")
                if not cp or not hdr or cp.get("coll_sn", 0) < warmup:
                    continue
                ts = cp.get("host_ts")
                # No host_ts: plugin predates this report. exec_start_us 0: no
                # kernel or network activity was seen, so the record has no
                # span; taking it as starting at 0 would stretch the group's
                # busy time to the machine's uptime.
                if not ts or "host" not in hdr or not 0 < ts.get("exec_start_us", 0) < ts.get("exec_end_us", 0):
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
                                      net_end_us=netobj.get("window_end_us", 0),
                                      rx_before=int(netobj.get("rx_bytes_before_window", 0))))
        why = _accl_incomplete(summary)
        for r in out[first:]:
            r.accl_incomplete = why
    return out, no_ts


def _accl_incomplete(summary: Optional[dict]) -> str:
    """Why an accl file may miss collectives or proxy data, "" if its end-of-run
    summary says nothing was lost. A missing summary means the process did not
    reach finalize; data lost there could hide exactly the slow collective."""
    if summary is None:
        return "accl wrote no end-of-run summary (the process may not have finished)"
    if summary.get("complete", True) and not summary.get("write_error"):
        return ""
    lost = [f"{k}={v}" for k, v in sorted(summary.items())
            if k not in ("complete", "write_error") and not k.endswith("_size") and k != "max_proxy_ops_per_coll"
            and isinstance(v, (int, float)) and v]
    return "accl lost data" + (f" ({', '.join(lost)})" if lost else "") + \
        (", write error" if summary.get("write_error") else "")


class Series:
    """A cumulative counter sampled over time; linear between samples."""

    def __init__(self, ts: List[int], vals: List[float], resets: int = 0,
                 reset_ts: Optional[List[int]] = None):
        self.ts, self.vals, self.resets = ts, vals, resets
        self.reset_ts = reset_ts or []

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
    qos: Dict = field(default_factory=dict)   # DCB settings + RCCL's traffic class (telemetry "qos")
    rx_merged: bool = False   # RCCL rx bytes pooled across a multi-NIC connection: not this NIC's own
    # The local_ack_timeout_err column was read from to_retransmits (retransmits
    # after a timeout), the driver having no local_ack_timeout_err.
    ack_retransmits: bool = False


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
    """Every series here is a cumulative counter. A sample lower than the one
    before is a bad read and is dropped when the counter later climbs back to
    where it was (Thor2 sysfs bytes have returned 0 between good readings, more
    than once in a row); a drop it never recovers from is a counter reset, and
    the count carries on from where it was."""
    pts = []
    for t, v in zip(samples["ts_us"], samples.get(key, ())):
        if isinstance(v, list):
            v = v[idx] if idx is not None and idx < len(v) else None
        if v is None or v < 0:
            continue  # -1 = counter not available on this NIC
        pts.append((t, float(v)))
    # later_max[i]: highest raw value after sample i, -inf past the end.
    later_max = [-math.inf] * len(pts)
    for i in range(len(pts) - 2, -1, -1):
        later_max[i] = max(later_max[i + 1], pts[i + 1][1])
    ts, vals, offset, resets, reset_ts = [], [], 0.0, 0, []
    for i, (t, v) in enumerate(pts):
        if vals and v + offset < vals[-1]:
            raw_last = vals[-1] - offset
            if i == len(pts) - 1 or later_max[i] >= raw_last:
                continue   # bad read: the counter comes back
            offset, resets = vals[-1], resets + 1
            reset_ts.append(t)
        ts.append(t)
        vals.append(v + offset)
    return Series(ts, vals, resets, reset_ts) if len(ts) >= 2 else None


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
                                         SeriesMap(cols), m.get("qos") or {}, bool(m.get("rx_bytes_merged")),
                                         _ack_from_retransmits(m.get("hw_type", ""), m.get("hw_counters")))
        by_dev.clear()
        out[(tel.host, tel.pid)] = tel
    return out


def _ack_from_retransmits(hw_type: str, hwc) -> bool:
    """Whether the sampled ACK-timeout column came from Thor2's to_retransmits
    (retransmits after a timeout): the start/exit readings show which of the
    two names the driver has. AINIC's to_retransmits is its ACK-timeout
    counter (tx_rdma_ack_timeout), so only Thor2 is affected."""
    if hw_type != "thor2" or not isinstance(hwc, dict):
        return False
    own, alt = hwc.get("local_ack_timeout_err"), hwc.get("to_retransmits")
    return (own is None or own < 0) and alt is not None and alt >= 0


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


def _overlap_spans(x: List[Tuple[int, int]], y: List[Tuple[int, int]]) -> float:
    """Total overlap of two sorted, disjoint span lists, in one pass."""
    i = j = 0
    tot = 0.0
    while i < len(x) and j < len(y):
        tot += max(0.0, min(x[i][1], y[j][1]) - max(x[i][0], y[j][0]))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return tot


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
    # A counter that shows events pad_us late had not yet shown, by its last
    # sample, what happened in the pad_us before it: that time is not covered,
    # however short, as events there could all still be unseen.
    tail = _overlap(spans, ser.ts[-1] - pad_us, ser.ts[-1]) if pad_us else 0.0
    uncovered = max(0.0, busy - sampled) + tail
    fill = inside / window if window > 0 else 0.0
    lb = inner
    if duration:
        if pad_us == 0:
            lb = max(lb, total - (window - inside))
        ub = min(total, sampled) + uncovered
    else:
        ub = total if uncovered <= 0.01 * busy and tail == 0 else INF
    return Bound(lb, ub, fill)


def _rccl_moving(dev: Device) -> Optional[Tuple[int, int]]:
    """First and last sample time at which RCCL's own byte count grew."""
    sw = dev.series.get("tx_bytes")
    if sw is None:
        return None
    moving = [sw.ts[i] for i in range(1, len(sw.ts)) if sw.vals[i] > sw.vals[i - 1]]
    return (moving[0], moving[-1]) if len(moving) >= 5 else None


def refresh_us(dev: Device, interval_us: float, key: str = "nic_tx_bytes") -> float:
    """How often one counter really changes, measured while RCCL was sending.
    Counter families refresh differently (ethtool vs IB sysfs, and per driver),
    so each is measured on its own changes. A counter cannot change faster than
    it refreshes, so a low gap between its changes bounds the refresh from
    above. A counter that barely moved (pause on a quiet run) cannot be
    measured; it gets the slowest refresh of this NIC's byte counters, which
    only widens its ranges."""
    if interval_us <= 0:
        return interval_us
    win = _rccl_moving(dev)
    ser = dev.series.get(key)
    if win is None or ser is None:
        return interval_us
    t0, t1 = win
    changes = [ser.ts[i] for i in range(1, len(ser.ts))
               if ser.vals[i] > ser.vals[i - 1] and t0 <= ser.ts[i] <= t1]
    if len(changes) >= 3:
        gaps = sorted(b - a for a, b in zip(changes, changes[1:]))
        return max(interval_us, gaps[len(gaps) // 10])
    if key in ("nic_tx_bytes", "nic_rx_bytes"):
        return max(interval_us, t1 - t0)   # a byte counter that stood still while RCCL sent: stale
    refs = [refresh_us(dev, interval_us, k) for k in ("nic_tx_bytes", "nic_rx_bytes") if k in dev.series]
    return max(refs) if refs else interval_us


# ============================================================================
# Which PFC priority, and in which unit
# ============================================================================

@dataclass
class Lane:
    """The PFC priority RCCL's traffic uses, and how that was decided."""
    prio: Optional[int]
    source: str                     # given | config | traffic | unknown | none
    share: float = 0.0              # share of the used NICs' bytes on that priority
    other_pause: Dict[int, float] = field(default_factory=dict)  # pause on other priorities
    all_zero: bool = False          # PFC counters exist and never moved, any priority
    no_pfc: bool = False            # PFC is not enabled on this priority (config or missing counters)
    why: str = ""                   # how the priority was decided, or why it could not be
    sent_with: str = ""             # traffic class / SL RCCL's QPs used, from telemetry
    pfc_config: Optional[List[int]] = None   # priorities with PFC on, from the NIC settings
    seen_pause: List[int] = field(default_factory=list)   # priorities that paused at all
    warnings: List[str] = field(default_factory=list)
    # (host, pid, device_id) -> (priority, PFC off on it), decided NIC by NIC
    per_dev: Dict[Tuple[str, int, int], Tuple[Optional[int], bool]] = field(default_factory=dict)

    def prio_for(self, host: str, pid: int, dev_id: int) -> Optional[int]:
        return self.per_dev.get((host, pid, dev_id), (self.prio, self.no_pfc))[0]

    def no_pfc_for(self, host: str, pid: int, dev_id: int) -> bool:
        return self.per_dev.get((host, pid, dev_id), (self.prio, self.no_pfc))[1]


def _sum_prio(tels, used, key, p):
    tot, seen = 0.0, False
    for k, tel in tels.items():
        for d in used.get(k, []):
            s = tel.devices[d].series.get(f"{key}[{p}]")
            if s:
                seen, tot = True, tot + s.total()
    return tot, seen


RCCL_UDP_PORT = 4791   # RoCE v2


# NICs on which a DCB app entry for RoCE's UDP port was seen to set the
# priority (Thor2: DSCP-0 traffic went on the port entry's priority). mlx5
# ignores such entries; on others it is unverified.
UDP_PORT_ENTRY_HONOURED = {"thor2"}


def config_prio(q: Dict, hw_type: str = "") -> Tuple[Optional[int], str]:
    """RCCL's priority from the NIC's DCB app table and RCCL's traffic class, or
    None with the reason it cannot be told. Never a guess: an entry for RoCE's
    UDP port or for RCCL's DSCP decides it, nothing else does."""
    if not q:
        return None, "no NIC settings recorded (older telemetry)"
    if q.get("dcb") != "ok":
        return None, {"unsupported": "the NIC driver does not expose its DCB settings",
                      "not_read": "the NIC's DCB settings were not read (RCCL never used it)"}.get(
                          q.get("dcb"), "reading the NIC's DCB settings failed")
    tc = q.get("traffic_class", -1)
    if tc is None or tc < 0:
        return None, "RCCL's traffic class was not recorded"
    dscp = tc >> 2
    apps = q.get("app", [])
    udp = {a["priority"] for a in apps if a.get("selector") == "dgram" and a.get("protocol") == RCCL_UDP_PORT
           and hw_type in UDP_PORT_ENTRY_HONOURED}
    ds = {a["priority"] for a in apps if a.get("selector") == "dscp" and a.get("protocol") == dscp}
    if len(udp | ds) > 1:
        parts = ([f"RoCE's UDP port to {sorted(udp)}"] if udp else []) + ([f"DSCP {dscp} to {sorted(ds)}"] if ds else [])
        return None, f"the NIC maps {' and '.join(parts)}; which one applies is up to the NIC"
    if udp:
        return next(iter(udp)), f"NIC maps RoCE (UDP {RCCL_UDP_PORT}) to it"
    if ds:
        return next(iter(ds)), f"NIC maps RCCL's DSCP {dscp} (traffic class {tc}) to it"
    if q.get("app_truncated"):
        return None, f"the NIC's app table was too long to record and RCCL's DSCP {dscp} was not in the part kept"
    if any(a.get("selector") == "dscp" for a in apps):
        return None, f"RCCL's DSCP {dscp} is not in the NIC's DSCP map, so a driver default applies"
    return None, (f"the NIC has no DCB app entry for RCCL's DSCP {dscp}, so the VLAN priority "
                  f"(SL {q.get('sl', '?')}) or a driver default applies")


def _pick(tels: Dict[Tuple[str, int], Telemetry], used: Dict[Tuple[str, int], List[int]],
          override: Optional[int] = None) -> Lane:
    nbytes = [sum(_sum_prio(tels, used, k, p)[0] for k in ("prio_tx_bytes", "prio_rx_bytes"))
              for p in range(NUM_PFC_PRIO)]
    pause, seen, seen_p = [], False, []
    for p in range(NUM_PFC_PRIO):
        tot, sp = 0.0, False
        for k in ("pfc_rx_pause_us", "pfc_tx_pause_us", "pfc_rx_frames", "pfc_tx_frames"):
            v, s = _sum_prio(tels, used, k, p)
            tot, sp = tot + v, sp or s
        pause.append(tot)
        seen_p.append(sp)
        seen = seen or sp
    devs = [tels[k].devices[d] for k, ds in used.items() for d in ds if d in tels[k].devices]
    # A NIC a process only received on has no sender QP, so no traffic class.
    devs = [d for d in devs if d.qos.get("traffic_class", -1) >= 0] or devs
    qos = [d.qos for d in devs]
    cfg = {config_prio(d.qos, d.hw_type) for d in devs} or {(None, "no NIC settings recorded")}
    cfg_prio = {p for p, _ in cfg}
    traffic = max(range(NUM_PFC_PRIO), key=lambda p: nbytes[p]) if any(nbytes) else None
    warnings = []
    if any(q.get("traffic_class_mixed") for q in qos):
        warnings.append("RCCL's data QPs on one NIC used different traffic classes")
    if override is not None:
        prio, source, why = override, "given", "--pfc-prio"
    elif len(cfg_prio) == 1 and None not in cfg_prio:
        prio, source, why = cfg_prio.pop(), "config", min(w for _, w in cfg)   # stable across runs
        # Byte counters include other traffic (storage, CNPs), so the busiest
        # priority need not be RCCL's. Only a configured priority that carried
        # almost nothing is overruled.
        if traffic is not None and traffic != prio:
            if nbytes[prio] < 0.01 * sum(nbytes):
                warnings.append(f"the NIC settings put RCCL on priority {prio}, but it carried almost no "
                                f"bytes and priority {traffic} carried most; using priority {traffic}")
                prio, source, why = traffic, "traffic", "per-priority byte counters"
            else:
                warnings.append(f"priority {traffic} carried more bytes than RCCL's priority {prio}: "
                                "other traffic shares these NICs")
    elif traffic is not None:
        prio, source, why = traffic, "traffic", "per-priority byte counters"
    else:
        reasons = sorted({w for p, w in cfg if p is None})
        if len(cfg_prio - {None}) > 1:
            reasons.append(f"NICs disagree: {sorted(cfg_prio - {None})}")
        prio, source = None, ("unknown" if seen else "none")
        why = "; ".join(reasons) + ("; and these NICs have no per-priority byte counters" if not any(nbytes) else "")
    share = nbytes[prio] / sum(nbytes) if prio is not None and sum(nbytes) else 0.0
    # With RCCL's priority unknown, no pause can be called "not RCCL's".
    other = {p: v for p, v in enumerate(pause) if v and p != prio} if prio is not None else {}
    seen_pause = [p for p, v in enumerate(pause) if v]
    pfc_cfg = [q["pfc_enabled"][prio] for q in qos
               if prio is not None and q.get("dcb") == "ok" and len(q.get("pfc_enabled", [])) > prio]
    no_pfc = prio is not None and ((bool(pfc_cfg) and not any(pfc_cfg)) or (seen and not seen_p[prio]))
    tcs = sorted({(q.get("traffic_class"), q.get("sl")) for q in qos if q.get("traffic_class", -1) >= 0})
    sent_with = ", ".join(f"traffic class {t} (DSCP {t >> 2}), SL {l}" for t, l in tcs)
    pfcs = {tuple(p for p, on in enumerate(q["pfc_enabled"]) if on) for q in qos if q.get("dcb") == "ok"}
    return Lane(prio, source, share, other, all_zero=seen and not any(pause), no_pfc=no_pfc,
                why=why, warnings=warnings, sent_with=sent_with, seen_pause=seen_pause,
                pfc_config=list(pfcs.pop()) if len(pfcs) == 1 else None)


def pick_lane(tels: Dict[Tuple[str, int], Telemetry], used: Dict[Tuple[str, int], List[int]],
              override: Optional[int] = None) -> Lane:
    """RCCL's PFC priority, decided per NIC: NICs can be set up differently.
    The job-level fields hold when every NIC agrees; otherwise prio is None,
    source is "per-nic", and lane.prio_for() gives each NIC's own."""
    job = _pick(tels, used, override)
    per = {(k[0], k[1], d): _pick(tels, {k: [d]}, override) for k, ds in used.items() for d in ds}
    job.per_dev = {key: (ln.prio, ln.no_pfc) for key, ln in per.items()}
    prios = {ln.prio for ln in per.values()}
    if len(prios) == 1:
        p = prios.pop()
        first = next(iter(per.values()))
        if p != job.prio and p is not None:
            # The job-wide byte majority can differ from what every NIC says;
            # recompute every job-level field for the NICs' priority.
            per_dev = job.per_dev
            job = _pick(tels, used, p)
            job.per_dev, job.source, job.why = per_dev, first.source, first.why
        job.no_pfc = all(ln.no_pfc for ln in per.values())
    elif len(per) > 1:
        job.prio, job.source, job.no_pfc = None, "per-nic", False
        job.why = "; ".join(f"{h.split('.')[0]} dev {d}: " + (f"priority {ln.prio} ({ln.why})" if ln.prio is not None
                                                               else f"unknown ({ln.why})")
                            for (h, _, d), ln in sorted(per.items()))
    return job


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
    ecn_pct: float = 1.0        # % of received packets ECN-marked
    cnp_ps: float = 100.0       # CNPs sent + received per second
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
    unit: Optional[str]              # "%" for the pause columns, None without pause counters
    frames_only: bool = False        # pause is bounded from pause frames, not measured
    tx_paused: bool = False          # this NIC provably paused the switch at least once
    rx_pause: Optional[Tuple[float, float]] = None   # (lb, ub) in unit
    tx_pause: Optional[Tuple[float, float]] = None
    # Delivery symptoms by telemetry column: (lb, ub), INF ub when the samples
    # do not cover the collectives; a column the NIC does not give is absent.
    symptoms: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    rnr: Optional[Tuple[float, float]] = None        # this NIC had no receive buffer (lb, ub)
    rnr_peer: Optional[Tuple[float, float]] = None   # RNR NAKs this NIC got from a peer (lb, ub)
    # Physical bytes the port moved inside the collectives, (tx lb, rx lb); None
    # without port counters. RCCL's own byte counts are posted work, not wire.
    port_lb: Optional[Tuple[float, float]] = None
    resets: List[str] = field(default_factory=list)  # counters used here that reset in the run
    sample_gap_us: float = 0.0                       # time in sampler stalls (gaps over 3 intervals) in the spans
    lag_us: float = 0.0                              # slowest refresh lag among the drop counters
    sharers: int = 1                                 # processes of this job driving this NIC
    device_id: int = -1
    ack_retransmits: bool = False                    # see Device.ack_retransmits
    ecn_pct: Optional[Tuple[float, float]] = None    # ECN-marked % of received packets (lb, ub)
    cnp_ps: Optional[Tuple[float, float]] = None     # CNPs sent + received per second (lb, ub)
    link_pct: Optional[float] = None                 # busier direction, % of link speed
    rccl_pct: Optional[float] = None                 # RCCL's share of that traffic
    nic_stale: bool = False
    sw_tx_per_coll: float = 0.0
    sw_rx_per_coll: float = 0.0                      # received inside the window only
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
    # Kept apart so sequential communicators or an algorithm/protocol switch at
    # one size are not merged into one row.
    comm: str = ""
    algo: str = ""
    proto: str = ""
    # All bytes received per collective, every NIC: a lower bound (see analyze).
    rx_per_coll_lb: Optional[float] = None
    n_ranks: int = 0                 # ranks in the communicator, per accl
    other_overlap_us: float = 0.0    # time other collectives of this process (other groups) also ran
    # Share of the proxy's step time spent waiting for this rank's GPU to produce
    # data (send side) and to take it (receive side, with GDR flush). accl sums
    # step times and several steps are in flight at once, so these are shares
    # of step time, not of wall time; the network time they are set against
    # covers both sides.
    gpu_send_wait_pct: float = 0.0
    gpu_recv_wait_pct: float = 0.0
    telemetry: bool = False          # this process wrote telemetry samples
    unsampled: List[int] = field(default_factory=list)   # of those, NICs telemetry did not sample
    # RCCL completion errors in these collectives' network windows, all NICs (accl).
    cq_errors: int = 0


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
            groups[(r.rec.coll, r.rec.msg_size, r.comm_hash, r.rec.algo, r.rec.proto)].append(r)
        interval = (tel.sample_interval_ms * 1000.0) if tel else 0.0

        for (coll, size, comm, algo, proto), g in sorted(groups.items()):
            spans = union([(r.exec_start_us, r.exec_end_us) for r in g])
            busy = float(sum(b - a for a, b in spans))
            agg = accl_report.aggregate([r.rec for r in g])[(coll, size)]
            gr = GroupResult(host=key[0], pid=key[1], rank=g[0].rank, coll=coll, msg_size=size,
                             n=len(g), busy_us=busy, mean_busbw=agg.mean_busbw,
                             accl_class=accl_report.classify_bottleneck(agg), nics=[], spans=spans,
                             comm=comm, algo=algo, proto=proto)
            for why in sorted({r.accl_incomplete for r in g if r.accl_incomplete}):
                gr.inconclusive.append(why)
            # Per collective: every NIC's in-window receive, plus the receive that
            # opened the window, added once (it is in no NIC's delta). Still a
            # lower bound: one completion can finish several grouped receives,
            # and only the first one's size is recorded.
            with_rx = [r for r in g if r.net]
            if with_rx:
                gr.rx_per_coll_lb = sum(sum(d.get("rx_bytes", 0) for d in r.net.values()) + r.rx_before
                                        for r in with_rx) / len(with_rx)
            with_win = [r for r in g if r.net_start_us]
            if with_win:
                gr.overlap_pct = 100.0 * sum(id(r) in overlapped for r in with_win) / len(with_win)
            gr.n_ranks = max(r.rec.n_ranks for r in g)
            mine = {id(r) for r in g}
            others = union([(r.exec_start_us, r.exec_end_us) for r in proc_recs if id(r) not in mine])
            gr.other_overlap_us = _overlap_spans(spans, others)
            send = agg.mean_proxy_gpu_wait_us + agg.mean_proxy_peer_wait_us + agg.mean_proxy_network_us
            recv = agg.mean_proxy_gpu_recv_wait_us + agg.mean_proxy_flush_us + agg.mean_proxy_network_us
            if send > 0:
                gr.gpu_send_wait_pct = 100.0 * agg.mean_proxy_gpu_wait_us / send
            if recv > 0:
                gr.gpu_recv_wait_pct = 100.0 * (agg.mean_proxy_gpu_recv_wait_us + agg.mean_proxy_flush_us) / recv
            gr.cq_errors = sum(int(d.get("cq_errors", 0)) for r in g for d in r.net.values())
            gr.telemetry = tel is not None and interval > 0
            if not gr.telemetry:
                gr.inconclusive.append("no telemetry samples for this process")
            elif not used:
                gr.inconclusive.append("no NIC of this process carried traffic in telemetry")
            for dev_id in used:
                ns = _nic_stats(tel.devices[dev_id], g, spans, busy, interval, lane, thr, tels, key[0], gr)
                gr.nics.append(ns)
                # CLEAN only means something for what was measured: a counter the
                # NIC did not give cannot be read as "nothing happened".
                missing = [what for what, have in (
                    ("delivery counters" + ("" if not ns.symptoms else " (" + ", ".join(
                        name for k, name, _ in SYMPTOMS if k not in ns.symptoms) + ")"),
                     len(ns.symptoms) == len(SYMPTOMS)),
                    ("receiver-not-ready counters" + ("" if ns.rnr is None and ns.rnr_peer is None else
                                                      " (RX-NO-BUFFER)" if ns.rnr is None else " (PEER-RNR)"),
                     ns.rnr is not None and ns.rnr_peer is not None),
                    ("ECN/CNP counters", ns.ecn_pct is not None or ns.cnp_ps is not None),
                    ("link use" + (" (port counter did not refresh)" if ns.nic_stale else ""),
                     ns.link_pct is not None),
                    # Pause is excused only where its state is known some other way:
                    # PFC off on RCCL's priority, or the priority unknown (said apart).
                    ("pause counters", ns.rx_pause is not None or ns.tx_pause is not None
                     or lane.no_pfc_for(key[0], key[1], dev_id)
                     or (lane.prio_for(key[0], key[1], dev_id) is None and lane.source != "none"))) if not have]
                if missing:
                    gr.inconclusive.append(f"{ns.roce}: not measured: {', '.join(missing)}")
            if gr.telemetry:
                gr.unsampled = sorted({d for r in g for d in r.net} - {ns.device_id for ns in gr.nics})
                if gr.unsampled:
                    gr.inconclusive.append(f"accl names NIC(s) {gr.unsampled} that telemetry did not sample")
            gr.verdict = _rank_verdict(gr)
            unjudged = lane.source != "none" and any(lane.prio_for(key[0], key[1], d) is None for d in used)
            if unjudged and gr.verdict.startswith("CLEAN"):
                gr.verdict = gr.verdict.replace("CLEAN (", "CLEAN (pause not judged; ", 1)
            results.append(gr)
    tag_echo(results)
    return results


def _nic_stats(dev, g, spans, busy, interval, lane, thr, tels, host, gr) -> NicStats:
    s = dev.series
    pads: Dict[str, float] = {}

    def pad_of(key: str) -> float:
        """Window padding for one counter: its own refresh lag, if slower than sampling."""
        if key not in pads:
            r = refresh_us(dev, interval, key)
            pads[key] = r if r > 1.5 * interval else 0.0
        return pads[key]
    prio = lane.prio_for(host, gr.pid, dev.device_id)
    unit = pause_unit(dev, prio) if prio is not None else None
    ns = NicStats(roce=dev.roce or f"dev{dev.device_id}", unit="%" if unit else None,
                  frames_only=unit == "frames", device_id=dev.device_id, ack_retransmits=dev.ack_retransmits)
    busy_s = busy / 1e6

    # PFC pause on RCCL's priority
    if unit is not None and busy > 0:
        k = "pause_us" if unit == "time" else "frames"
        # A pause frame stops the link for at most its quanta: 16 bits of 512 bit
        # times. Frames therefore bound pause time from above, never from below;
        # measured on AINIC, one frame stood for anywhere from 3 to 53 us.
        max_us = PAUSE_QUANTA_BITS / (dev.link_gbps * 1e3) if dev.link_gbps else INF
        for direction, name, what in (("rx", "FABRIC-CONGESTION", "switch paused our transmit"),
                                      ("tx", "HOST-BACKPRESSURE", "our NIC paused the switch")):
            key = f"pfc_{direction}_{k}[{prio}]"
            b = bound(s.get(key), spans, pad_of(key), unit == "time")
            if b is None:
                continue
            if unit == "time":
                r = _rate(b, busy, "%")
                detail = f"{what} {_fmt_range(r, '%')} of the time"
            else:
                ub = 0.0 if b.ub == 0 else (INF if INF in (b.ub, max_us) else min(100.0, 100.0 * b.ub * max_us / busy))
                r = (0.0, ub)
                fps = _rate(b, busy, "/s")
                most = "an unknown share (link speed unknown)" if ub == INF else f"{ub:.1f}%"
                detail = (f"{what}: {_fmt_range(fps, '/s')} pause frames, so at most {most} of the "
                          "time (frames do not give the length)")
            setattr(ns, f"{direction}_pause", r)
            if direction == "tx":
                ns.tx_paused = b.lb > 0      # a counted frame or pause time proves it
            lvl = _level(r[0], r[1], thr.pause_pct)
            if lvl:
                ns.findings.append(Finding(name, lvl, detail))

    # Delivery symptoms and receiver-not-ready events, each on its own: any is
    # worth reporting, and none is added to another.
    rows = [(k, name, what) for k, name, what in SYMPTOMS] + \
        [(RNR_LOCAL[0], "RX-NO-BUFFER", "times this NIC had no receive buffer posted"),
         (RNR_PEER[0], "PEER-RNR", "receiver-not-ready NAKs from a peer")]
    for k, name, what in rows:
        ser = s.get(k)
        b = bound(ser, spans, pad_of(k), False)
        if b is None:
            continue
        if any(a - pad_of(k) <= t <= b + pad_of(k) for t in ser.reset_ts for a, b in spans):
            ns.resets.append(k)
        ns.lag_us = max(ns.lag_us, pad_of(k))
        r = (b.lb, b.ub)
        if k == RNR_LOCAL[0]:
            ns.rnr = r
        elif k == RNR_PEER[0]:
            ns.rnr_peer = r
        else:
            ns.symptoms[k] = r
        lvl = _level(b.lb, b.ub, 0.0, strict=True)
        if lvl:
            ns.findings.append(Finding(name, lvl, f"{_fmt_range(r, '')} {what}"))
    # NIC progress: bytes the port provably moved inside the collectives.
    port = [bound(s.get(f"nic_{d}_bytes"), spans, pad_of(f"nic_{d}_bytes"), False) for d in ("tx", "rx")]
    if all(b is not None for b in port):
        ns.port_lb = (port[0].lb, port[1].lb)
    # Sampler stalls: gaps well past both the setting and the sampler's own
    # usual pace (reading many counters can make every sample slower than set).
    ts = s["tx_bytes"].ts if "tx_bytes" in s else []
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]))
    if gaps:
        stall = 3 * max(interval, gaps[len(gaps) // 2])
        ns.sample_gap_us = sum(_overlap(spans, a, b) for a, b in zip(ts, ts[1:]) if b - a > stall)
    ns.sharers = 1 if not dev.roce else sum(1 for (h, _), t in tels.items() if h == host and any(
        d.roce == dev.roce and "tx_bytes" in d.series and sum(d.series["tx_bytes"].delta(a, b) for a, b in spans) > 0
        for d in t.devices.values())) or 1

    # ECN marks as a share of received packets: a switch marks a share of the
    # packets it queues. CNPs as a rate instead: a NIC sends at most one per flow
    # every few tens of us, however many marks arrive, so their share of packets
    # stays far below 1% even when a flow is slowed the whole time.
    lvls = []
    bm = bound(s.get("ecn_marked_pkts"), spans, pad_of("ecn_marked_pkts"), False)
    bp = bound(s.get("nic_rx_packets"), spans, pad_of("nic_rx_packets"), False)
    if bm is not None and bp is not None and bp.ub not in (0, INF):
        ns.ecn_pct = (100.0 * bm.lb / bp.ub,
                      0.0 if bm.ub == 0 else INF if bm.ub == INF or bp.lb == 0 else 100.0 * bm.ub / bp.lb)
        lvls.append((_level(*ns.ecn_pct, thr.ecn_pct), f"{_fmt_range(ns.ecn_pct, '%')} of packets ECN-marked"))
    bs = [b for b in (bound(s.get(k), spans, pad_of(k), False) for k in ("cnp_sent", "cnp_rcvd")) if b is not None]
    if bs and busy > 0:
        tot = Bound(sum(b.lb for b in bs), INF if any(b.ub == INF for b in bs) else sum(b.ub for b in bs),
                    min(b.fill for b in bs))
        ns.cnp_ps = _rate(tot, busy, "/s")
        lvls.append((_level(*ns.cnp_ps, thr.cnp_ps), f"{_fmt_range(ns.cnp_ps, '/s')} CNPs sent or received"))
    hit = [(l, d) for l, d in lvls if l]
    if hit:
        level = "yes" if any(l == "yes" for l, _ in hit) else "maybe"
        ns.findings.append(Finding("CONGESTION-MARKS", level, "; ".join(d for _, d in hit)))

    # Link use, and how much of it was RCCL. Port counters include all traffic.
    if busy > 0:
        stale = False
        best = None
        # RCCL's rx bytes on a multi-NIC connection all land on its first NIC,
        # so on such NICs only the send direction can say how much was RCCL's.
        for d in (("tx",) if dev.rx_merged else ("tx", "rx")):
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
                    "LINK-SHARED", "yes" if busy >= 5 * max(pad_of("nic_tx_bytes"), pad_of("nic_rx_bytes"),
                                                            interval) else "maybe",
                    f"link {best[0]:.0f}% busy, RCCL {best[1]:.0f}% of it"))

    # Per-collective RCCL bytes and completion latency, exact from accl.
    with_net = [r for r in g if dev.device_id in r.net]
    hist: List[int] = []
    for r in with_net:
        d = r.net[dev.device_id]
        ns.sw_tx_per_coll += d.get("tx_bytes", 0)
        ns.sw_rx_per_coll += d.get("rx_bytes", 0)
        h = d.get("completion_hist", [])
        hist += [0] * (len(h) - len(hist))
        for i, c in enumerate(h):
            hist[i] += c
    if with_net:
        ns.sw_tx_per_coll /= len(with_net)
        ns.sw_rx_per_coll /= len(with_net)
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


ORDER = ["HOST-BACKPRESSURE", "RX-NO-BUFFER", "PEER-RNR", "OUT-OF-SEQ", "SEQ-NAK", "ACK-TIMEOUT",
         "CQ-ERROR", "FABRIC-CONGESTION", "LINK-SHARED", "CONGESTION-MARKS"]
DROP_RELATED = {"RX-NO-BUFFER", "PEER-RNR", "CQ-ERROR"} | {name for _, name, _ in SYMPTOMS}


def _rank_verdict(gr: GroupResult) -> str:
    found = {"CQ-ERROR": "yes"} if gr.cq_errors else {}
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


def _pushback(gs: List[GroupResult]) -> List[Tuple[int, str, Optional[str], Tuple[float, float], bool]]:
    """NICs that sent any pause frames, proven or possible, whatever the threshold:
    one receiver that pushes back can stall many senders."""
    return [(g.rank, g.host.split(".")[0], ns.unit, ns.tx_pause, ns.tx_paused) for g in gs for ns in g.nics
            if ns.tx_pause and (ns.tx_pause[1] == INF or ns.tx_pause[1] > 0)]


def _by_size(results: List[GroupResult]) -> Dict[Tuple[str, int, str, str, str], List[GroupResult]]:
    """All ranks' groups of one collective instance: same size, communicator,
    algorithm and protocol."""
    by = defaultdict(list)
    for g in results:
        by[(g.coll, g.msg_size, g.comm, g.algo, g.proto)].append(g)
    return by


def tag_echo(results: List[GroupResult]):
    """Mark rx pause on a rank as a possible echo when some NIC of the same
    collective and size pushed back."""
    for gs in _by_size(results).values():
        pushers = {r for r, *_ in _pushback(gs)}
        if not pushers:
            continue
        for g in gs:
            if g.rank not in pushers and "FABRIC-CONGESTION" in g.verdict:
                g.verdict = g.verdict.replace("FABRIC-CONGESTION", "FABRIC-CONGESTION(echo?)", 1)


def _echo_summary(gs, cong, fmt, thr: "Thresholds") -> str:
    """Senders were paused by the switch while some receiver paused the switch.
    A receiver that stops taking data makes the switch pause the senders behind
    it, so its pause can cause some or all of theirs. These counters cannot say
    how much: each rank's share is of its own busy time, who sends to whom is
    not known, and switch buffering spreads one stop over many senders. So both
    sides are reported and neither is subtracted from the other."""
    push = _pushback(gs)
    proven = [p for p in push if p[4]]
    who = fmt(sorted({(r, h) for r, h, *_ in (proven or push)}))
    verb = "paused the switch" if proven else "may have paused the switch"
    senders = [(g, ns) for g in gs for ns in g.nics
               if ns.rx_pause and any(f.name == "FABRIC-CONGESTION" and f.level == "yes" for f in ns.findings)]
    pushers = [(g, ns) for g in gs for ns in g.nics if ns.tx_pause and (ns.tx_pause[1] == INF or ns.tx_pause[1] > 0)]
    # Shares are of each rank's own busy time; milliseconds are what both sides share.
    x = max((ns.rx_pause[0] for _, ns in senders), default=0.0)
    x_ms = max((ns.rx_pause[0] * g.busy_us / 1e5 for g, ns in senders), default=0.0)
    y = max(ns.tx_pause[1] for _, ns in pushers)
    y_ms = max((INF if ns.tx_pause[1] == INF else ns.tx_pause[1] * g.busy_us / 1e5) for g, ns in pushers)
    # Backpressure is a full explanation candidate when the receiver provably
    # paused the switch above the threshold, or for about as long as the senders were paused.
    host_firm = any(p[4] and p[3][0] >= thr.pause_pct for p in push) or (bool(proven) and 2 * y_ms >= x_ms)
    sender = (f"the switch paused {len(cong)} sender rank(s) (at least {_fmt_num(x, '%')} of their own time, "
              f"{x_ms:.0f} ms)")
    receiver = f"{who} {verb} (" + ("for an unknown time" if y == INF else
                                    f"up to {_fmt_num(y, '%')} of its own time, {y_ms:.0f} ms") + ")"
    if host_firm:
        return (f"FABRIC+HOST?: {sender}; {receiver}. That backpressure can cause some or all of the "
                "senders' pause; these counters cannot split the two")
    return (f"FABRIC?: {sender}; {receiver}, so part of the senders' pause may be backpressure "
            "from there")


# ============================================================================
# Drop diagnosis: delivery symptoms against endpoint evidence
# ============================================================================

@dataclass
class DropDiagnosis:
    bucket: str                    # HOST_LIKELY | FABRIC_LIKELY | MIXED | INCONCLUSIVE | NONE
    confidence: str = ""           # high | medium | low; "" for INCONCLUSIVE
    leaning: str = ""              # what the rules gave before data quality took it to INCONCLUSIVE
    signals: List[str] = field(default_factory=list)
    endpoints: List[str] = field(default_factory=list)
    unknown: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    competing: List[str] = field(default_factory=list)
    quality: List[str] = field(default_factory=list)
    slowness: str = ""

    def headline(self) -> str:
        if self.bucket == "INCONCLUSIVE":
            return "INCONCLUSIVE" + (f" (leaning {self.leaning})" if self.leaning else "")
        return f"{self.bucket} ({self.confidence})"


def _where(g: GroupResult, ns: Optional[NicStats] = None) -> str:
    return f"rank {g.rank} ({g.host.split('.')[0]}" + (f" {ns.roce})" if ns else ")")


def _list(xs: List[str], n: int = 3) -> str:
    return ", ".join(xs[:n]) + (f" and {len(xs) - n} more" if len(xs) > n else "")


def _evidence(rows, what_ok, what_bad, where_bad=None):
    """Fold one kind of endpoint evidence over every NIC of the collective.
    rows: (place, state, detail), state one of
      pressure  proven at or over its threshold
      some      possibly over it: only the upper bound reaches it
      low       proven, but under the threshold; counts as fine
      unknown   not measured, or the samples do not cover the collectives
      ok        measured, covered and under the threshold
    Returns (state, text, unknown places). The state is the worst one seen,
    but the text names every place that is not ok, so none hides another."""
    by = defaultdict(list)
    for place, state, detail in rows:
        by[state].append((place, detail))

    def places(state):
        # Places that share a reason are listed under it once.
        why = defaultdict(list)
        for place, detail in by[state]:
            why[detail].append(place)
        return "; ".join(_list(ps) + (f": {w}" if w else "") for w, ps in why.items())
    parts = []
    if by["pressure"]:
        parts.append(f"{what_bad} at {places('pressure')}" + (f" ({where_bad})" if where_bad else ""))
    if by["some"]:
        parts.append(f"possibly {what_bad} at {places('some')}")
    if by["low"]:
        parts.append(f"under the threshold at {places('low')}")
    if by["unknown"]:
        parts.append(f"unknown at {places('unknown')}")
    fine = len(by["ok"]) + len(by["low"])
    if by["ok"] and not parts:
        parts.append(what_ok)
    elif by["ok"]:
        parts.append(f"fine at the other {len(by['ok'])} NIC(s)")
    state = next((st for st in ("pressure", "some", "unknown") if by[st]), "ok" if fine else "unknown")
    return state, "; ".join(parts), [f"{p}: {w}" for p, w in by["unknown"]]


def diagnose_drops(gs: List[GroupResult], lane: Lane, thr: Thresholds,
                   baseline: Optional[Dict[Tuple[str, int, str, str, int], float]] = None) -> DropDiagnosis:
    """Bucket one collective and size, over all its ranks.

    Signals (observations): OUT-OF-SEQ at a receiver, SEQ-NAK and ACK-TIMEOUT
    at a sender, and RCCL completion errors. Each is shown on its own with its
    bounds; "in window" means the lower bound is above 0 (it counted inside the
    collectives given the counter's measured refresh lag), "possibly" that only
    the upper bound is. Overlap in time is not cause.

    Endpoint evidence, each "ok" only where measured, covered and zero (pause:
    under the threshold):
      receive buffers   out_of_buffer on every NIC: the receiver had buffers
      peer readiness    rnr_retry_err on every NIC: no sender saw a not-ready
                        peer (the peer itself is not identified)
      backpressure      PFC tx pause on RCCL's priority; unknown with PFC off
                        (an overloaded receiver then drops, and port discards
                        are not sampled) or with the priority unknown
      NIC progress      port byte counters provably moved in both directions
                        (RCCL's own byte counts are posted work, not the wire)
    Endpoints are healthy only when all four are ok on every rank of the
    communicator. "No error reported" is not health: a missing counter is unknown.

    Rules ("receiver not ready" = receive buffers or peer readiness under pressure):
      no symptom in window:
          receiver not ready                 -> HOST_LIKELY (medium); those
                                                counters are drop/recovery events
          a symptom or receiver-not-ready
          count only possibly in window      -> INCONCLUSIVE
          a signal not measured somewhere    -> INCONCLUSIVE
          else                               -> NONE (high; medium if backpressure
                                                or NIC progress is not known).
                                                Pause alone is flow control, not drops
      only completion errors                 -> INCONCLUSIVE (local and remote causes)
      symptom and endpoint pressure (any of the three):
          fabric corroborated (rx pause with no pushback, ECN/CNP) -> MIXED (medium)
          else                               -> HOST_LIKELY (low)
      symptom, no endpoint pressure:
          only ACK timeouts:
              endpoints healthy              -> FABRIC_LIKELY (low; medium if corroborated)
              else                           -> INCONCLUSIVE
          possible endpoint pressure         -> INCONCLUSIVE (leaning MIXED)
          endpoints healthy                  -> FABRIC_LIKELY (medium; high if corroborated)
          endpoint health unknown            -> FABRIC_LIKELY (low; medium if corroborated)
    Then each kind of data-quality issue (accl lost data, a counter reset in
    the window, sampling gaps, drop counters refreshing slower than a fifth of
    the collectives' time, overlapping collectives or communicators, a NIC
    shared with other processes or traffic) lowers the confidence one step;
    below low the bucket becomes INCONCLUSIVE, the rules' answer kept as "leaning"."""
    d = DropDiagnosis(bucket="INCONCLUSIVE")
    nics = [(g, ns) for g in gs for ns in g.nics]
    expected = max((g.n_ranks for g in gs), default=0)
    seen = {g.rank for g in gs}
    # Blind: no telemetry, or accl names NICs telemetry did not sample. A
    # process with telemetry whose records name no NIC had no network traffic.
    blind = [(_where(g), "no telemetry") for g in gs if not g.telemetry] + \
        [(_where(g), f"telemetry has no samples of NIC(s) {g.unsampled}, which accl names") for g in gs
         if g.unsampled]
    if expected > len(seen):
        blind.append((f"{expected - len(seen)} of {expected} ranks", "no accl records"))

    # --- signals ---------------------------------------------------------
    definite, possible = set(), set()
    for col, name, what in SYMPTOMS:
        hits, maybe, unk = [], [], [f"{p}: {w}" for p, w in blind]
        for g, ns in nics:
            r = ns.symptoms.get(col)
            if r is None:
                unk.append(_where(g, ns) + ": not measured")
            elif r[0] > 0:
                hits.append(f"{_fmt_range(r, '')} at {_where(g, ns)}")
            elif r[1] == INF:
                unk.append(_where(g, ns) + ": samples do not cover the collectives (or the counter had not "
                           "refreshed yet)")
            elif r[1] > 0:
                maybe.append(f"0-{r[1]:.0f} at {_where(g, ns)}")
        text = f"{name}, {what}: "
        if hits:
            definite.add(name)
            text += "in window " + _list(hits)
        if maybe:
            possible.add(name)
            text += ("; " if hits else "") + "possibly in window " + _list(maybe)
        if not hits and not maybe:
            text += ("none where measured" if nics else "not measured anywhere") if unk else "none on any NIC"
        alt = [_where(g, ns) for g, ns in nics if col == "local_ack_timeout_err" and ns.ack_retransmits
               and col in ns.symptoms]
        if alt:
            text += f" (at {_list(alt)} this counts retransmits after a timeout: the driver has no timeout counter)"
        d.signals.append(text)
        if unk:
            d.unknown.append(f"{name}: {_list(unk)}")
    cq = [g for g in gs if g.cq_errors]
    if cq:
        definite.add("CQ-ERROR")
        d.signals.append("CQ-ERROR, RCCL completion errors (accl; overlapping windows can count one twice): "
                         + _list([f"{g.cq_errors} at {_where(g)}" for g in cq]))
    if len(definite) + len(possible) > 1:
        d.signals.append("these are separate counters: one lost packet can show in several of them, "
                         "so they are not added up into a count of lost packets")

    # --- endpoint evidence -----------------------------------------------
    def rows(get):
        out = [(p, "unknown", w) for p, w in blind]
        for g, ns in nics:
            r = get(ns)
            if r is None:
                out.append((_where(g, ns), "unknown", "no counter"))
            elif r[0] > 0:
                out.append((_where(g, ns), "pressure", _fmt_range(r, "")))
            elif r[1] == INF:
                out.append((_where(g, ns), "unknown", "samples do not cover the collectives"))
            elif r[1] > 0:
                out.append((_where(g, ns), "some", f"0-{r[1]:.0f}"))
            else:
                out.append((_where(g, ns), "ok", ""))
        return out

    ev = {}
    ev["receive buffers"] = _evidence(rows(lambda ns: ns.rnr), "no receiver ran out of buffers",
                                      "receive buffers ran out", "that host's own receiver")
    ev["peer readiness"] = _evidence(rows(lambda ns: ns.rnr_peer), "no sender got RNR NAKs",
                                     "RNR NAKs seen", "seen by the sender; the not-ready peer is not identified")
    for name, k in (("RX-NO-BUFFER", "receive buffers"), ("PEER-RNR", "peer readiness")):
        if ev[k][2]:
            d.unknown.append(f"{name}: {_list(ev[k][2])}")
    bp = [(p, "unknown", w) for p, w in blind]
    for g, ns in nics:
        w = _where(g, ns)
        if ns.tx_pause is None:
            why = ("PFC off on RCCL's priority: an overloaded receiver drops instead, and port "
                   "discards are not sampled") if lane.no_pfc_for(g.host, g.pid, ns.device_id) else \
                "RCCL's PFC priority unknown" if lane.prio_for(g.host, g.pid, ns.device_id) is None else \
                "no pause counters"
            bp.append((w, "unknown", why))
            continue
        # Same rule as the HOST-BACKPRESSURE finding; frames only ever bound it from above.
        lvl = _level(ns.tx_pause[0], ns.tx_pause[1], thr.pause_pct)
        shown = _fmt_bound(ns.tx_pause) if ns.frames_only else _fmt_range(ns.tx_pause, "%")
        if lvl == "yes":
            bp.append((w, "pressure", f"paused the switch {shown} of the time"))
        elif lvl == "maybe":
            bp.append((w, "some", f"paused the switch {shown} of the time"))
        elif ns.tx_pause[0] > 0 or ns.tx_paused:
            bp.append((w, "low", f"paused the switch {shown} of the time"))
        else:
            bp.append((w, "ok", ""))
    ev["backpressure"] = _evidence(bp, "no NIC paused the switch", "receiver backpressure")
    prog = [(p, "unknown", w) for p, w in blind]
    for g, ns in nics:
        if ns.port_lb is None:
            prog.append((_where(g, ns), "unknown", "no port byte counters"))
        elif min(ns.port_lb) > 0:
            prog.append((_where(g, ns), "ok", ""))
        else:
            prog.append((_where(g, ns), "unknown", "port bytes did not provably move inside the collectives"
                         " (counter lag or stale counter)"))
    ev["NIC progress"] = _evidence(prog, "every port moved bytes both ways inside the collectives", "")
    for k, (state, text, _) in ev.items():
        d.endpoints.append(f"{k}: {text}")
    # Context, not evidence: overlapping steps make these shares rough, and
    # healthy runs already show about half.
    snd, rcv = [g.gpu_send_wait_pct for g in gs], [g.gpu_recv_wait_pct for g in gs]
    d.endpoints.append(f"accl proxy waiting on the GPU, share of its step time (context only): send "
                       f"{min(snd):.0f}-{max(snd):.0f}%, receive {min(rcv):.0f}-{max(rcv):.0f}% across ranks")
    if max(snd + rcv) > 80:
        d.competing.append("the proxy waited on the GPU for most of its step time on some rank: the "
                           "endpoints' compute or scheduling may set the pace, not the network")

    states = {k: v[0] for k, v in ev.items()}
    # Only a receiver that was not ready counts as a host cause of drops. Pause
    # is flow control: on a lossless priority it kept the receiver from
    # dropping, so it only stops the endpoints counting as healthy.
    receiver = [k for k in ("receive buffers", "peer readiness") if states[k] == "pressure"]
    pressure = receiver
    some = [k for k in ("receive buffers", "peer readiness") if states[k] == "some"]
    blind_rx = all(ns.rnr is None and ns.rnr_peer is None for _, ns in nics)    # nothing known anywhere
    healthy = all(v == "ok" for v in states.values()) and not blind
    not_ok = ", ".join(k for k, v in states.items() if v != "ok") or "ranks without telemetry"

    # --- fabric corroboration --------------------------------------------
    pushback = bool(_pushback(gs))
    corr = sorted({f"{f.name} at {_where(g, ns)}" for g, ns in nics for f in ns.findings
                   if f.level == "yes" and (f.name == "CONGESTION-MARKS" or
                                            (f.name == "FABRIC-CONGESTION" and not pushback))})

    # --- rules -----------------------------------------------------------
    sym = definite - {"CQ-ERROR"}
    names = ", ".join(sorted(sym))
    sym_unknown = [u for u in d.unknown if u.split(":")[0] in {n for _, n, _ in SYMPTOMS}]
    if not sym and receiver:
        # Completion errors next to a receiver that was not ready are no
        # reason for less: RNR retries running out end in completion errors.
        open_sym = sorted(possible) + (["some not measured"] if sym_unknown else [])
        d.bucket, d.confidence = "HOST_LIKELY", "low" if open_sym else "medium"
        d.reasons.append(f"receiver not ready ({', '.join(receiver)}) with no other delivery symptom "
                         "in the window" + (f" (open: {', '.join(open_sym)})" if open_sym else ""))
    elif not definite:
        if possible or some:
            d.reasons.append("signals only possibly fall inside the collectives: their counters cannot "
                             "place them inside rather than next to them")
        elif d.unknown:
            d.reasons.append("no delivery symptom seen, but not every signal was measured on every rank")
        else:
            known = states["backpressure"] != "unknown" and states["NIC progress"] != "unknown"
            d.bucket, d.confidence = "NONE", "high" if known else "medium"
            d.reasons.append("no delivery or recovery symptom on any NIC, every signal measured" +
                             ("" if known else f" ({not_ok} not known)"))
            if states["backpressure"] in ("pressure", "some"):
                d.reasons.append("pause is flow control, not drops: see the summary line")
    elif not sym:
        d.reasons.append("only completion errors: they have local causes (protection, access, length) "
                         "as well as remote ones, and RCCL does not record which")
    elif pressure:
        if corr:
            d.bucket, d.confidence = "MIXED", "medium"
            d.reasons.append(f"delivery symptoms ({names}) together with endpoint pressure "
                             f"({', '.join(pressure)}) and fabric congestion ({_list(corr)})")
            d.competing.append("either side alone: a receiver short of buffers also makes senders "
                               "retransmit, and congestion can drop packets with every host healthy")
        else:
            d.bucket, d.confidence = "HOST_LIKELY", "low"
            d.reasons.append(f"delivery symptoms ({names}) together with endpoint pressure "
                             f"({', '.join(pressure)}); a receiver that is not ready also makes senders "
                             "retransmit, and no congestion marks or unexplained rx pause were seen")
            d.competing.append("loss in the fabric on the same paths: the symptoms alone cannot rule it out")
    elif sym == {"ACK-TIMEOUT"} and not healthy:
        d.reasons.append(f"only ACK timeouts, and endpoint health is not established ({not_ok})")
    elif sym == {"ACK-TIMEOUT"}:
        d.bucket, d.confidence = "FABRIC_LIKELY", "medium" if corr else "low"
        d.reasons.append("only ACK timeouts, with healthy endpoints" +
                         (f" and fabric congestion ({_list(corr)})" if corr else ""))
    elif some:
        d.reasons.append(f"delivery symptoms ({names}) and endpoint pressure ({', '.join(some)}) whose "
                         "counters only possibly reach the threshold")
        d.leaning = "MIXED"
    elif blind_rx:
        d.reasons.append(f"delivery symptoms ({names}), but nothing is known about the receivers' "
                         "readiness (no receive-buffer or RNR counters)")
        d.leaning = "FABRIC_LIKELY"
    else:
        base = (1 if healthy else 2) - (1 if corr else 0)
        d.bucket, d.confidence = "FABRIC_LIKELY", CONFIDENCE[base]
        d.reasons.append(f"delivery symptoms ({names}) with " +
                         ("healthy endpoints" if healthy else
                          ("no receiver short of buffers or RNR seen, but endpoint health not established "
                           f"({not_ok})")) +
                         (f"; fabric congestion seen ({_list(corr)})" if corr else ""))
    if "ACK-TIMEOUT" in definite:
        d.competing.append("a timeout also follows a lost ACK on the return path, a long delay, or an "
                           "endpoint that stopped answering, not only a lost packet")
    if "CQ-ERROR" in definite:
        d.competing.append("completion errors also come from local faults (protection, access, length)")
    if d.bucket == "FABRIC_LIKELY":
        d.competing.append("an endpoint fault these counters do not show (port discards, PCIe, NIC firmware)")
        if any(lane.no_pfc_for(g.host, g.pid, ns.device_id) for g, ns in nics):
            d.competing.append("with PFC off on RCCL's priority, a receiver host that cannot keep up drops "
                               "packets too, and those port discards are not sampled")
    if states["backpressure"] == "pressure" and definite:
        d.competing.append("a receiver paused the switch: a slow host, though PFC kept it from dropping; "
                           "its pause can also echo as rx pause elsewhere")
    elif pushback and definite:
        d.competing.append("a receiver paused the switch: rx pause elsewhere may be its echo, not "
                           "fabric congestion")

    # --- data quality ----------------------------------------------------
    q = {}
    for g in gs:
        for why in g.inconclusive:
            if why.startswith("accl"):
                q.setdefault("accl", f"{_where(g)}: {why}")
        # Counters cannot split work that ran at the same time. Overlap among
        # this group's own collectives does not matter here: it is all charged
        # to the group.
        if g.other_overlap_us > 0.01 * g.busy_us:
            q.setdefault("overlap", f"other collectives ran at the same time at {_where(g)} for "
                         f"{g.other_overlap_us / 1000:.1f} ms ({100 * g.other_overlap_us / g.busy_us:.0f}% "
                         "of these collectives' time)")
        for ns in g.nics:
            if ns.resets:
                q.setdefault("reset", f"{', '.join(ns.resets)} reset during the collectives at {_where(g, ns)}")
            if ns.sample_gap_us > 0.1 * g.busy_us:
                q.setdefault("gap", f"sampler stalls cover {ns.sample_gap_us / 1000:.0f} ms of "
                             f"{g.busy_us / 1000:.0f} ms at {_where(g, ns)}")
            if ns.lag_us > 0.2 * g.busy_us:
                q.setdefault("lag", f"drop counters refresh ~{ns.lag_us / 1000:.0f} ms late at {_where(g, ns)} "
                             f"against {g.busy_us / 1000:.0f} ms of collectives")
            if ns.sharers > 1:
                q.setdefault("procs", f"{ns.sharers} processes of this job drive {_where(g, ns)}")
            if ns.rccl_pct is not None and ns.rccl_pct < 100 - thr.link_other_pct:
                q.setdefault("traffic", f"only {ns.rccl_pct:.0f}% of the traffic at {_where(g, ns)} was this "
                             "job's RCCL")
    d.quality = list(q.values())
    if d.bucket == "NONE":
        # A counter that stayed at 0 is 0 whoever else used the NIC, and its
        # lag is already in the bounds; only lost accl data or a reset can
        # hide a count.
        q = {k: v for k, v in q.items() if k in ("accl", "reset")}
    if d.confidence:
        i = CONFIDENCE.index(d.confidence) + len(q)
        if i >= len(CONFIDENCE):
            d.leaning, d.bucket, d.confidence = d.bucket, "INCONCLUSIVE", ""
            d.reasons.append("data quality leaves no confidence: " + "; ".join(d.quality))
        else:
            d.confidence = CONFIDENCE[i]

    # --- was it slow? ----------------------------------------------------
    key = (gs[0].coll, gs[0].msg_size, gs[0].algo, gs[0].proto, expected)
    bw = sum(g.mean_busbw for g in gs) / len(gs)
    if baseline and baseline.get(key, 0) > 0:
        d.slowness = (f"busBW {bw:.1f} GB/s against {baseline[key]:.1f} GB/s in the baseline run "
                      f"({100 * (bw / baseline[key] - 1):+.0f}%)")
    else:
        d.slowness = "no matched baseline (same collective, size, algorithm, protocol and rank count)"
    d.slowness += "; seeing these events does not by itself show the collective was slow"
    return d


def job_summary(results: List[GroupResult], lane: Lane, thr: Optional["Thresholds"] = None,
                drops: Optional[Dict[Tuple[str, int, str, str, str], DropDiagnosis]] = None
                ) -> Dict[Tuple[str, int, str, str, str], str]:
    """One conclusion per (collective, size) across all ranks. A NIC that paused
    the switch can explain rx pauses elsewhere, so pushback is checked first,
    then a drop diagnosis that points at a host. Every finding the headline
    does not lead with, proven or possible, is listed after it, so the order
    hides none."""
    thr = thr or Thresholds()
    if drops is None:
        drops = {k: diagnose_drops(gs, lane, thr) for k, gs in _by_size(results).items()}
    out = {}
    for k, gs in _by_size(results).items():
        def ranks(name, level="yes"):
            hit = {(g.rank, g.host.split(".")[0]) for g in gs for ns in g.nics
                   for f in ns.findings if f.name == name and f.level == level}
            if name == "CQ-ERROR" and level == "yes":
                hit |= {(g.rank, g.host.split(".")[0]) for g in gs if g.cq_errors}
            return sorted(hit)

        def fmt(rs):
            return ", ".join(f"rank {r} ({h})" for r, h in rs[:4]) + (" ..." if len(rs) > 4 else "")

        bp, cong = ranks("HOST-BACKPRESSURE"), ranks("FABRIC-CONGESTION")
        shared, marks = ranks("LINK-SHARED"), ranks("CONGESTION-MARKS")
        dd = drops[k]
        host_drops = dd.bucket in ("HOST_LIKELY", "MIXED")
        drop_seen = any(ranks(n) for n in DROP_RELATED)
        named = set()
        if cong and _pushback(gs):
            msg = _echo_summary(gs, cong, fmt, thr)
            named = {"FABRIC-CONGESTION", "HOST-BACKPRESSURE"}
        elif bp:
            msg = f"HOST: {fmt(bp)} paused the switch (could not take data in fast enough)"
            named = {"HOST-BACKPRESSURE"}
        elif drop_seen and host_drops:
            msg = f"DROPS: {dd.headline()} from " + "; ".join(
                f"{n} on {fmt(ranks(n))}" for n in ORDER if n in DROP_RELATED and ranks(n)) + \
                ", see the drop diagnosis"
            named = {n for n in DROP_RELATED if ranks(n)}
        elif cong:
            msg = f"FABRIC: switch paused sending on {len(cong)} rank(s), no NIC pushed back: {fmt(cong)}"
            named = {"FABRIC-CONGESTION"}
        elif shared:
            msg = f"LINK-SHARED: other traffic held the link on {fmt(shared)}"
            named = {"LINK-SHARED"}
        elif marks:
            msg = f"CONGESTION-MARKS on {fmt(marks)}"
            named = {"CONGESTION-MARKS"}
        elif drop_seen:
            msg = f"DROPS: {dd.headline()} from " + "; ".join(
                f"{n} on {fmt(ranks(n))}" for n in ORDER if n in DROP_RELATED and ranks(n)) + \
                ", see the drop diagnosis"
            named = {n for n in DROP_RELATED if ranks(n)}
        elif any(g.verdict.startswith("INCONCLUSIVE") for g in gs):
            maybes = sorted({f.name for g in gs for ns in g.nics for f in ns.findings if f.level == "maybe"})
            reasons = sorted({x for g in gs for x in g.inconclusive})
            msg = "INCONCLUSIVE: " + "; ".join(([f"possible {', '.join(maybes)}"] if maybes else [])
                                               + reasons[:2])
            named = set(maybes)
        else:
            msg = "CLEAN: no fabric signal in any counter"
            if lane.no_pfc:
                msg += f" (no PFC on priority {lane.prio}: congestion shows only as loss or CNPs)"
            elif lane.all_zero:
                msg += " (PFC counters zero on every priority: no pauses, or PFC is disabled)"
        if lane.prio is None and lane.source == "unknown" and msg.startswith("CLEAN"):
            msg = msg.split(" (")[0] + " apart from pause, which is not judged: RCCL's priority is unknown"
        also = []
        for n in ORDER:
            if n in named:
                continue
            if ranks(n):
                also.append(f"{n} on {fmt(ranks(n))}")
            elif ranks(n, "maybe"):
                also.append(f"possible {n} on {fmt(ranks(n, 'maybe'))}")
        if not msg.startswith("DROPS") and dd.bucket != "NONE":
            also.append(f"drops {dd.headline()}")
        if also:
            msg += "; also: " + "; ".join(also)
        out[k] = msg
    return out


# ============================================================================
# Text report
# ============================================================================

def _fmt_secs(sec: float) -> str:
    return f"{sec * 1000:.0f} ms" if sec < 1 else f"{sec:.1f} s"


def _fmt_num(v: float, unit: Optional[str]) -> str:
    return f"{v:.1f}%" if unit == "%" else f"{v:.0f}{unit or ''}"


def _fmt_bound(r: Optional[Tuple[float, float]]) -> str:
    """A frames-only pause share: an upper bound, never a measured value."""
    if r is None:
        return "-"
    return "0%" if r[1] == 0 else ("0-?%" if r[1] == INF else f"0-{r[1]:.1f}%")


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


def _fmt_counts(rs: List[Optional[Tuple[float, float]]]) -> str:
    """Counters side by side, "-" where the NIC does not give one."""
    return "/".join("-" if r is None else _fmt_range(r, "") for r in rs)


def print_report(results: List[GroupResult], lane: Lane, thr: Thresholds, per_nic: bool,
                 warnings: List[str], baseline: Optional[Dict[Tuple[str, int, str, str, int], float]] = None):
    hosts = {r.host for r in results}
    procs = {(r.host, r.pid) for r in results}
    print(f"ACCL fabric report: {len(procs)} process(es) on {len(hosts)} host(s)")
    if lane.sent_with:
        print(f"RCCL sends with {lane.sent_with}")
    if lane.pfc_config is not None:
        print(f"PFC enabled on priorities {lane.pfc_config or 'none'} (NIC settings)")
    if lane.prio is None and lane.source == "unknown":
        print(f"PFC priority: UNKNOWN, so pause is not judged: {lane.why}.")
        print("  Pass --pfc-prio N once you know which priority RCCL's traffic class maps to.")
        if lane.seen_pause:
            print("  Pause was seen on priorities " + ", ".join(map(str, lane.seen_pause)) +
                  "; whether any is RCCL's is not known.")
    elif lane.source == "per-nic":
        print("PFC priority differs by NIC: " + lane.why)
    elif lane.prio is None:
        print("PFC: these NICs report no PFC counters")
    elif lane.source == "traffic":
        print(f"PFC priority: {lane.prio} (carries {100 * lane.share:.0f}% of the job's NIC bytes)")
    else:
        print(f"PFC priority: {lane.prio} ({lane.why})")
    for w in lane.warnings:
        print(f"  WARNING: {w}")
    if lane.other_pause:
        print("  Pause on other priorities (does not slow RCCL's): " +
              ", ".join(f"prio {p}" for p in sorted(lane.other_pause)))
    if lane.no_pfc:
        print(f"  PFC is not enabled on priority {lane.prio}, so RCCL's traffic is lossy: congestion shows as")
        print("  loss and CNPs, never as pause.")
    elif lane.all_zero:
        print("  Every PFC counter stayed zero: either nothing paused or PFC is disabled, so a")
        print("  CLEAN result cannot rule out a lossy fabric on its own; check the oos/nak/to columns.")
    print(f"Thresholds: pause {thr.pause_pct:g}% of wall time, ECN {thr.ecn_pct:g}% of packets or {thr.cnp_ps:g} CNPs/s, link busy {thr.link_busy_pct:g}% "
          f"with >= {thr.link_other_pct:g}% non-RCCL traffic")
    print("  rxPause = switch paused our transmit   txPause = our NIC paused the switch")
    print("  a-b = at least a, at most b (counters are sampled, not continuous)")
    print("  B/coll = RCCL bytes sent per collective; rxwin/coll = received inside the network window on")
    print("  that NIC; rx/coll = all RCCL bytes received per collective, all NICs, a lower bound")
    print("  oos/nak/to = out-of-sequence packets / sequence-error NAKs / ACK timeouts, each on its own")
    print("  nobuf/rnr = times this NIC had no receive buffer / RNR NAKs it got from a peer; - = not measured\n")

    hdr = (f"{'Coll':<12} {'Size':>5} {'Rank':>4} {'Host':<20} {'N':>4} {'wall ms':>8} "
           f"{'busBW':>6} {'NIC':<10} {'rxPause':>10} {'txPause':>10} {'oos/nak/to':>15} {'nobuf/rnr':>9} "
           f"{'ecn%':>9} {'cnp/s':>11} {'link%':>5} {'rccl%':>5} {'B/coll':>8} {'rxwin/coll':>10} {'rx/coll':>9} {'cmpl p50/p99':>13}  Verdict")
    print(hdr)
    print("-" * len(hdr))
    for g in sorted(results, key=lambda r: (r.coll, r.msg_size, r.host, r.rank)):
        nics = g.nics if per_nic else sorted(g.nics, key=lambda n: -len(n.findings))[:1]
        first = (f"{g.coll[:12]:<12} {accl_report.fmt_size(g.msg_size):>5} {g.rank:>4} "
                 f"{g.host.split('.')[0][-20:]:<20} {g.n:>4} {g.busy_us / 1000:>8.2f} {g.mean_busbw:>6.1f}")
        if not nics:
            print(f"{first} {'-':<10}  {g.verdict}")
            continue
        for i, n in enumerate(nics):
            lead = first if i == 0 else " " * len(first)
            rxp, txp = (_fmt_bound(n.rx_pause), _fmt_bound(n.tx_pause)) if n.frames_only else \
                (_fmt_range(n.rx_pause, n.unit or ''), _fmt_range(n.tx_pause, n.unit or ''))
            p50 = fmt_quantile(hist_quantile(n.cmpl_hist, 0.5, HIST_NS))
            p99 = fmt_quantile(hist_quantile(n.cmpl_hist, 0.99, HIST_NS))
            link = "stale" if n.nic_stale else ("-" if n.link_pct is None else f"{n.link_pct:.0f}")
            bpc = accl_report.fmt_size(int(n.sw_tx_per_coll)) + ("*" if g.overlap_pct > 10 else "")
            rxwin = accl_report.fmt_size(int(n.sw_rx_per_coll))
            # Once per collective, all NICs: shown on the first NIC row only.
            rxall = (">=" + accl_report.fmt_size(int(g.rx_per_coll_lb))) if i == 0 and g.rx_per_coll_lb is not None \
                else ("-" if i == 0 else "")
            print(f"{lead} {n.roce[:10]:<10} {rxp:>10} "
                  f"{txp:>10} {_fmt_counts([n.symptoms.get(k) for k, _, _ in SYMPTOMS]):>15} "
                  f"{_fmt_counts([n.rnr, n.rnr_peer]):>9} {_fmt_range(n.ecn_pct, '%'):>9} {_fmt_range(n.cnp_ps, '/s'):>11} "
                  f"{link:>5} {'-' if n.rccl_pct is None else f'{n.rccl_pct:.0f}':>5} {bpc:>8} {rxwin:>10} {rxall:>9} "
                  f"{p50 + '/' + p99:>13}  {g.verdict if i == 0 else ''}")

    print("\nSummary per collective and size (all ranks):")
    drops = {k: diagnose_drops(gs, lane, thr, baseline) for k, gs in _by_size(results).items()}
    summary = job_summary(results, lane, thr, drops)
    variants = defaultdict(int)
    for coll, size, *_ in summary:
        variants[(coll, size)] += 1
    for (coll, size, comm, algo, proto), msg in sorted(summary.items()):
        label = f"{coll} {accl_report.fmt_size(size)}"
        if variants[(coll, size)] > 1:   # name what tells them apart
            label += f" ({algo}/{proto}, comm {comm})"
        print(f"  {label}: {msg}")

    print("\nDrop diagnosis per collective and size (delivery symptoms against endpoint evidence):")
    for (coll, size, comm, algo, proto), dd in sorted(drops.items()):
        label = f"{coll} {accl_report.fmt_size(size)}"
        if variants[(coll, size)] > 1:
            label += f" ({algo}/{proto}, comm {comm})"
        print(f"  {label}: {dd.headline()}")
        for title, items in (("signals", dd.signals), ("endpoints", dd.endpoints),
                             ("not measured", dd.unknown), ("why", dd.reasons),
                             ("could also be", dd.competing), ("data quality", dd.quality),
                             ("slowness", [dd.slowness])):
            for i, it in enumerate(items):
                print(f"    {title if i == 0 else '':<13} {it}")

    print("\nFindings:")
    print("  FABRIC-CONGESTION  switch paused our transmit; can be backpressure from a receiver,")
    print("                     marked (echo?) when a NIC of the same collective and size pushed back")
    print("  HOST-BACKPRESSURE  our NIC paused the switch: this host could not take data in")
    print("  OUT-OF-SEQ         packets this NIC received out of sequence (receiver side)")
    print("  SEQ-NAK            sequence-error NAKs this NIC got back (sender side)")
    print("  ACK-TIMEOUT        this NIC's sends timed out waiting for an ACK (sender side)")
    print("  RX-NO-BUFFER       this NIC had no receive buffer posted (this host's receiver)")
    print("  PEER-RNR           a host this rank sends to was not ready (RNR NAKs seen here)")
    print("  CQ-ERROR           RCCL saw completion errors in these collectives (from accl)")
    print("  LINK-SHARED        link busy, mostly with traffic that is not this job's RCCL")
    print("  CONGESTION-MARKS   ECN-marked share of packets, or CNPs per second, above the threshold")
    print("  CLEAN              every counter below threshold; time went elsewhere")
    print("  INCONCLUSIVE       missing samples, or only an upper bound crosses a threshold")
    print("  Drop buckets: HOST_LIKELY, FABRIC_LIKELY, MIXED or INCONCLUSIVE, with a confidence;")
    print("  NONE = no delivery symptom, every signal measured. These are likelihoods from")
    print("  counters, not the failing component.")
    print("  Echo (FABRIC+HOST? / FABRIC?): senders paused while a receiver paused the switch; both")
    print("  are shown and neither is subtracted: who sends to whom is not known, and hosts are")
    print("  matched by collective and size, not clock.")
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
    if any(n.frames_only for g in results for n in g.nics):
        notes.append("Pause on NICs that count only pause frames is an upper bound (0-x%): frames x the "
                     "longest pause one frame can ask for at the link speed (84 us at 400 Gb/s). It rules "
                     "pause out when x is small, but cannot prove it: on AINIC one frame stood for 3 to 53 us.")
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
    # One track per process: a process in several communicators has a rank in
    # each, so it is named by its lowest one.
    lowest = {}
    for r in recs:
        lowest[(r.host, r.pid)] = min(lowest.get((r.host, r.pid), r.rank), r.rank)
    procs = sorted((h, p, rk) for (h, p), rk in lowest.items())
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

            prio = lane.prio_for(host, pid, dev.device_id)
            unit = pause_unit(dev, prio) if prio is not None else None
            if unit == "time":
                rate_track(f"PFC pause % (prio {prio})",
                           [("rx: switch paused us", f"pfc_rx_pause_us[{prio}]"),
                            ("tx: we paused switch", f"pfc_tx_pause_us[{prio}]")], 100.0)
            if unit is not None:
                rate_track(f"PFC pause frames/s (prio {prio})",
                           [("rx: from switch", f"pfc_rx_frames[{prio}]"),
                            ("tx: to switch", f"pfc_tx_frames[{prio}]")], 1e6)
            rate_track("NIC GB/s (all port traffic)", [("tx", "nic_tx_bytes"), ("rx", "nic_rx_bytes")], 1e6 / 1e9)
            rate_track("RCCL GB/s", [("tx", "tx_bytes"), ("rx", "rx_bytes")], 1e6 / 1e9, hw=False)
            rate_track("ECN/CNP per ms", [("ecn marked", "ecn_marked_pkts"),
                                          ("cnp sent", "cnp_sent"), ("cnp rcvd", "cnp_rcvd")], 1e3)
            rate_track("delivery and RNR events per s",
                       [(k, k) for k in [c for c, _, _ in SYMPTOMS] + list(RNR_LOCAL + RNR_PEER)], 1e6)

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
        warnings.append(f"{no_ts} accl record(s) have no usable host timing and are ignored: either no "
                        "host_ts (plugin built before this report) or no kernel or network activity was seen.")
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
        every = _fmt_secs(hi) if _fmt_secs(lo) == _fmt_secs(hi) else f"{_fmt_secs(lo)} to {_fmt_secs(hi)}"
        warnings.append(f"{len(slow)} NIC(s) refresh their counters only every {every}, slower "
                        f"than the {slow[0][1] / 1000:.0f} ms sampling, so counts cannot be placed finer than "
                        f"that. For a firm result run each collective size for at least ~{_fmt_secs(5 * hi)}.")
    return recs, tels, used, pick_lane(tels, used, args.pfc_prio), warnings


def load_baseline(path: str, warmup: int) -> Dict[Tuple[str, int, str, str, int], float]:
    """Mean busBW per (collective, size, algorithm, protocol, ranks) of a healthy
    run. Only busBW is needed, so records without host timing count too."""
    if not os.path.exists(path):
        sys.exit(f"--baseline: {path} does not exist")
    files = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".jsonl")] \
        if os.path.isdir(path) else [path]
    by = defaultdict(list)
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                cp, hdr = obj.get("coll_perf"), obj.get("header")
                if cp and hdr and cp.get("coll_sn", 0) >= warmup:
                    by[(cp.get("coll", ""), cp.get("coll_msg_size_bytes", 0), cp.get("coll_algo", ""),
                        cp.get("coll_proto", ""), hdr.get("n_ranks", 1))].append(cp.get("coll_busbw_gbs", 0))
    return {k: sum(v) / len(v) for k, v in by.items()}


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
            p.add_argument("--threshold-ecn-pct", type=float, default=d.ecn_pct,
                           help="ECN-marked share of received packets that counts (default %(default)s)")
            p.add_argument("--threshold-cnps", type=float, default=d.cnp_ps,
                           help="CNPs sent + received per second that count (default %(default)s)")
            p.add_argument("--link-busy-pct", type=float, default=d.link_busy_pct,
                           help="link use that counts as busy (default %(default)s)")
            p.add_argument("--link-other-pct", type=float, default=d.link_other_pct,
                           help="non-RCCL share of the link that counts as shared (default %(default)s)")
            p.add_argument("--per-nic", action="store_true", help="one row per NIC, not just the worst")
            p.add_argument("--baseline", default=None,
                           help="accl JSONL of a healthy run of the same job, to compare busBW against")
        else:
            p.add_argument("-o", "--output", default="accl_fabric_trace.json")
    args = ap.parse_args()

    recs, tels, used, lane, warnings = load(args)
    if args.cmd == "report":
        thr = Thresholds(args.threshold_pct, args.threshold_ecn_pct, args.threshold_cnps,
                         args.link_busy_pct, args.link_other_pct)
        results = analyze(recs, tels, lane, thr)
        resets = sum(s.resets for t in tels.values() for d in t.devices.values() for s in d.series.built())
        if resets:
            warnings.append(f"{resets} NIC counter reset(s) during the run; counts continue across them.")
        print_report(results, lane, thr, args.per_nic, warnings,
                     load_baseline(args.baseline, args.warmup) if args.baseline else None)
    else:
        write_trace(recs, tels, used, lane, args.output)


if __name__ == "__main__":
    main()
