#!/usr/bin/env python3
"""Local POC correlator: LTTng runtime tracepoints + CP dispatch log -> Perfetto/Chrome JSON.

    join.py <run_dir>     (reads <run_dir>/lttng, dlog.csv, clocks.csv; writes trace.json)

Joins, as in the dispatch-tracing design (section 5), with the doorbell-slot key from
rocprofiler-sdk's in-process reader:
  API call      one event per call: [start_timestamp, event time] on (pid, tid)
  dispatch      kernel_dispatch probe -> innermost API call on its thread containing it
  log records   (doorbell_off & 1023, dispatch_id) == (doorbell_slot, packet_index & 0xffffffff)
  GPU time      linear fit of GPU clock -> CLOCK_MONOTONIC from collector clock pairs
"""
import bisect
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import bt2

run = Path(sys.argv[1])


def field(ev, name, default=None):
    try:
        v = ev[name]
    except KeyError:
        return default
    return str(v) if isinstance(v, bt2._StringFieldConst) else int(v)


# ------------------------------------------------------------------ runtime tracepoints
api_calls = []   # dict(name, pid, tid, start, end, args)
probes = []      # dict(pid, tid, t, slot, index, kernel, queue_id, grid, wg)
for msg in bt2.TraceCollectionMessageIterator(str(run / "lttng")):
    if type(msg) is not bt2._EventMessageConst:
        continue
    ev = msg.event
    if not ev.name.startswith("rocm_hip:"):
        continue
    name = ev.name.split(":", 1)[1].split(";", 1)[0]
    t = msg.default_clock_snapshot.value  # LTTng monotonic clock, ns
    pid, tid = field(ev, "vpid"), field(ev, "vtid")
    if name == "kernel_dispatch":
        probes.append(dict(pid=pid, tid=tid, t=t, slot=field(ev, "doorbell_slot"),
                           index=field(ev, "packet_index"), kernel=field(ev, "kernel_name"),
                           queue_id=field(ev, "queue_id"),
                           grid=[field(ev, f"grid_{a}") for a in "xyz"],
                           wg=[field(ev, f"workgroup_{a}") for a in "xyz"]))
    else:
        skip = {"start_timestamp", "_ms_ActivityId", "__ms_ActivityId_length",
                "_ms_ActivityId_padding"}
        args = {f: (str(v) if isinstance(v, bt2._StringFieldConst) else int(v))
                for f, v in ev.payload_field.items() if f not in skip}
        api_calls.append(dict(name=name, pid=pid, tid=tid, start=field(ev, "start_timestamp"),
                              end=t, args=args))

# ------------------------------------------------------------------ dispatch log
gpu_dispatches = []  # dict(gpu, slot, id, start_tick, end_tick)
open_start = {}
rows = sorted(csv.DictReader(open(run / "dlog.csv")), key=lambda r: int(r["gpu_ts"]))
unpaired_end = 0
for r in rows:
    key = (int(r["gpu_id"]), int(r["doorbell_off"]), int(r["dispatch_id"]))
    if r["record_type"] == "1":
        open_start[key] = int(r["gpu_ts"])
    elif r["record_type"] == "2":
        s = open_start.pop(key, None)
        if s is None:
            unpaired_end += 1
            continue
        gpu_dispatches.append(dict(gpu=key[0], slot=key[1] & 1023, doorbell_off=key[1], id=key[2],
                                   start_tick=s, end_tick=int(r["gpu_ts"])))

# ------------------------------------------------------------------ clock fit per GPU
fit = {}
samples = defaultdict(list)
for r in csv.DictReader(open(run / "clocks.csv")):
    mid = (int(r["mono_before"]) + int(r["mono_after"])) / 2
    samples[int(r["gpu_id"])].append((int(r["gpu_clock"]), mid))
for gpu, pts in samples.items():
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if len(pts) >= 2 and max(xs) > min(xs):
        slope, intercept = statistics.linear_regression(xs, ys)
        resid = max(abs(slope * x + intercept - y) for x, y in pts)
        fit[gpu] = (slope, intercept, resid, len(pts))


def to_ns(gpu, tick):
    slope, intercept, _, _ = fit[gpu]
    return slope * tick + intercept


# ------------------------------------------------------------------ joins
calls_by_tid = defaultdict(list)
for i, c in enumerate(api_calls):
    c["id"] = i + 1  # the collector assigns correlation ids (collector design 8.5)
    calls_by_tid[(c["pid"], c["tid"])].append(c)
for lst in calls_by_tid.values():
    lst.sort(key=lambda c: c["start"])


def enclosing_call(p):
    best = None
    for c in calls_by_tid.get((p["pid"], p["tid"]), []):
        if c["start"] <= p["t"] <= c["end"] and (best is None or c["start"] >= best["start"]):
            best = c
    return best


log_by_key = defaultdict(list)
for d in gpu_dispatches:
    log_by_key[(d["slot"], d["id"])].append(d)
for lst in log_by_key.values():
    lst.sort(key=lambda d: d["start_tick"])

matched, no_log, no_call = [], 0, 0
for p in sorted(probes, key=lambda p: p["t"]):
    p["call"] = enclosing_call(p)
    if p["call"] is None:
        no_call += 1
    cands = log_by_key.get((p["slot"], p["index"] & 0xFFFFFFFF), [])
    if not cands:
        no_log += 1
        continue
    d = cands.pop(0)  # oldest unmatched record with this key
    p["gpu"] = d["gpu"]
    p["gpu_start"] = to_ns(d["gpu"], d["start_tick"])
    p["gpu_end"] = to_ns(d["gpu"], d["end_tick"])
    matched.append(p)
log_left = sum(len(v) for v in log_by_key.values())

# ------------------------------------------------------------------ Perfetto / Chrome JSON
t0 = min([c["start"] for c in api_calls] + [p["t"] for p in probes])
us = lambda ns: (ns - t0) / 1000.0
events = []
pids = {c["pid"] for c in api_calls}
for pid in pids:
    events.append(dict(ph="M", name="process_name", pid=pid, args=dict(name=f"app (pid {pid})")))
for c in api_calls:
    events.append(dict(ph="X", name=c["name"], cat="hip_api", pid=c["pid"], tid=c["tid"],
                       ts=us(c["start"]), dur=(c["end"] - c["start"]) / 1000.0,
                       args=dict(c["args"], correlation_id=c["id"])))
gpu_pid_base = 1_000_000
for gpu in {p["gpu"] for p in matched}:
    events.append(dict(ph="M", name="process_name", pid=gpu_pid_base + gpu,
                       args=dict(name=f"GPU {gpu}")))
for p in matched:
    gpid = gpu_pid_base + p["gpu"]
    events.append(dict(ph="M", name="thread_name", pid=gpid, tid=p["slot"],
                       args=dict(name=f"queue {p['queue_id']} (doorbell slot {p['slot']})")))
    cid = p["call"]["id"] if p["call"] else None
    events.append(dict(ph="X", name=p["kernel"], cat="kernel", pid=gpid, tid=p["slot"],
                       ts=us(p["gpu_start"]), dur=(p["gpu_end"] - p["gpu_start"]) / 1000.0,
                       args=dict(packet_index=p["index"], grid=p["grid"], workgroup=p["wg"],
                                 correlation_id=cid,
                                 api=p["call"]["name"] if p["call"] else None)))
    if p["call"]:
        fid = f"{cid}-{p['index']}-{p['slot']}"
        events.append(dict(ph="s", id=fid, name="launch", cat="launch", pid=p["pid"],
                           tid=p["tid"], ts=us(p["t"])))
        events.append(dict(ph="f", bp="e", id=fid, name="launch", cat="launch", pid=gpid,
                           tid=p["slot"], ts=us(p["gpu_start"])))
json.dump(dict(traceEvents=events, displayTimeUnit="ns"), open(run / "trace.json", "w"))

# ------------------------------------------------------------------ report
lat = sorted(p["gpu_start"] - p["t"] for p in matched)
print(f"API calls: {len(api_calls)}   kernel_dispatch probes: {len(probes)}   "
      f"log dispatches: {len(gpu_dispatches)} (unpaired END: {unpaired_end}, open START: {len(open_start)})")
print(f"matched probe->log: {len(matched)}   probes without log: {no_log}   "
      f"log without probe: {log_left}   probes without API call: {no_call}")
for gpu, (slope, intercept, resid, n) in fit.items():
    if any(p["gpu"] == gpu for p in matched):
        print(f"clock fit GPU {gpu}: {1e9 / slope / 1e6:.3f} MHz, max residual {resid:.0f} ns over {n} pairs")
if lat:
    print(f"probe -> GPU start (ns): min {lat[0]:.0f}  median {lat[len(lat) // 2]:.0f}  max {lat[-1]:.0f}"
          f"   negative: {sum(1 for x in lat if x < 0)}")
    durs = sorted(p["gpu_end"] - p["gpu_start"] for p in matched)
    print(f"kernel duration (us): min {durs[0] / 1e3:.1f}  median {durs[len(durs) // 2] / 1e3:.1f}  max {durs[-1] / 1e3:.1f}")
by_kernel = defaultdict(list)
for p in matched:
    by_kernel[(p["kernel"], p["call"]["name"] if p["call"] else None)].append(p)
for (k, api), lst in sorted(by_kernel.items()):
    print(f"  {len(lst):4d} x {k[:60]:60s} from {api}")
print(f"wrote {run / 'trace.json'} ({len(events)} events)")
