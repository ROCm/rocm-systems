#!/usr/bin/env python3
"""Local POC correlator: LTTng runtime tracepoints + CP dispatch log -> Perfetto/Chrome JSON.

    join.py <run_dir> [--provider=rocm_hip] [--rocprofv3-csv=<file>] [--probes-src=<file>]
        Reads <run_dir>/lttng, and dlog.csv and clocks.csv when present; writes trace.json.
        The LTTng session needs the vpid and vtid contexts. --rocprofv3-csv also writes the
        API calls in rocprofv3's hip_api_trace.csv format; --probes-src (the generated
        hip_trace_probes.cpp) gives each API's table, hence its rocprofv3 domain.

Joins, as in the dispatch-tracing design (section 5) and the collector design (8.5), with the
doorbell-slot key from rocprofiler-sdk's in-process reader:
  API call      <api>_enter / <api>_exit pair on (pid, tid), paired in call order; the collector
                assigns the correlation id and the enclosing call as parent. Also reads this
                branch's one-event probes (<api> with start_timestamp)
  dispatch      kernel_dispatch probe -> innermost API call on its thread containing it
  log records   (doorbell_off & 1023, dispatch_id) == (doorbell_slot, packet_index & 0xffffffff)
  GPU time      linear fit of GPU clock -> CLOCK_MONOTONIC from collector clock pairs
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import bt2

def option(name, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{name}=")), default)


args = [a for a in sys.argv[1:] if not a.startswith("--")]
provider = option("provider", "rocm_hip")
rocprofv3_csv = option("rocprofv3-csv")  # also write rocprofv3's <prefix>_hip_api_trace.csv
probes_src = option("probes-src")  # generated hip_trace_probes.cpp: which table each API is in
run = Path(args[0])


def value(v):
    """A babeltrace2 field as a Python value."""
    if isinstance(v, bt2._StringFieldConst):
        return str(v)
    if isinstance(v, (bt2._ArrayFieldConst,)):
        return [value(x) for x in v]
    if isinstance(v, bt2._RealFieldConst):
        return float(v)
    return int(v)


def field(ev, name, default=None):
    try:
        return value(ev[name])
    except KeyError:
        return default


def payload(ev):
    """The event's own fields: not the transport's (opcode, activity id) or the length and
    padding fields LTTng adds for strings and arrays."""
    return {f: value(v) for f, v in ev.payload_field.items()
            if not f.startswith("_") and not f.endswith("_padding")}


# ------------------------------------------------------------------ runtime tracepoints
api_calls = []     # dict(id, parent, name, pid, tid, start, end, args, exit)
probes = []        # dict(pid, tid, t, slot, index, kernel, queue_id, grid, wg)
stacks = defaultdict(list)  # (pid, tid) -> open calls, innermost last
unmatched_exit = 0
for msg in bt2.TraceCollectionMessageIterator(str(run / "lttng")):
    if type(msg) is not bt2._EventMessageConst:
        continue
    ev = msg.event
    if not ev.name.startswith(provider + ":"):
        continue
    name = ev.name.split(":", 1)[1].split(";", 1)[0]  # drop the keyword suffix (;k0;)
    t = msg.default_clock_snapshot.value  # LTTng monotonic clock, ns
    pid, tid = field(ev, "vpid"), field(ev, "vtid")
    if pid is None or tid is None:
        sys.exit("events have no vpid/vtid: add them with lttng add-context -u -t vpid -t vtid")
    stack = stacks[(pid, tid)]
    if name == "kernel_dispatch":
        probes.append(dict(pid=pid, tid=tid, t=t, slot=field(ev, "doorbell_slot"),
                           index=field(ev, "packet_index"), kernel=field(ev, "kernel_name"),
                           queue_id=field(ev, "queue_id"),
                           grid=[field(ev, f"grid_{a}") for a in "xyz"],
                           wg=[field(ev, f"workgroup_{a}") for a in "xyz"]))
    elif name.endswith("_enter"):
        call = dict(id=len(api_calls) + 1, parent=stack[-1]["id"] if stack else None,
                    name=name[: -len("_enter")], pid=pid, tid=tid, start=t, end=None,
                    args=payload(ev), exit=None)
        api_calls.append(call)  # the collector assigns correlation ids (collector design 8.5)
        stack.append(call)
    elif name.endswith("_exit"):
        api = name[: -len("_exit")]
        # Pairs nest in call order on a thread (RC-API-1): the exit closes the innermost call.
        # An exit for another API means an enter was lost; close calls up to the matching one.
        depth = next((i for i in range(len(stack) - 1, -1, -1) if stack[i]["name"] == api), None)
        if depth is None:
            unmatched_exit += 1
            continue
        call = stack[depth]
        del stack[depth:]
        call["end"] = t
        call["exit"] = payload(ev)
    elif field(ev, "start_timestamp") is not None:
        # This branch's own tltrace probes: one event per call, written on return with the start
        # time as a field (before the Nov 4 enter/exit decision). A complete call, not nested.
        args = payload(ev)
        start = args.pop("start_timestamp")
        api_calls.append(dict(id=len(api_calls) + 1, parent=stack[-1]["id"] if stack else None,
                              name=name, pid=pid, tid=tid, start=start, end=t, args=args,
                              exit={}))
open_calls = [c for c in api_calls if c["end"] is None]  # no exit: crash, hang, or trace ended

# ------------------------------------------------------------------ dispatch log
gpu_dispatches = []  # dict(gpu, slot, id, start_tick, end_tick)
open_start = {}
dlog = run / "dlog.csv"
rows = sorted(csv.DictReader(open(dlog)), key=lambda r: int(r["gpu_ts"])) if dlog.exists() else []
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
clocks = run / "clocks.csv"
for r in (csv.DictReader(open(clocks)) if clocks.exists() else []):
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
for c in api_calls:
    if c["end"] is not None:
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
t0 = min([c["start"] for c in api_calls] + [p["t"] for p in probes], default=0)
us = lambda ns: (ns - t0) / 1000.0
events = []
pids = {c["pid"] for c in api_calls}
for pid in pids:
    events.append(dict(ph="M", name="process_name", pid=pid, args=dict(name=f"app (pid {pid})")))
for c in api_calls:
    if c["end"] is None:
        continue
    events.append(dict(ph="X", name=c["name"], cat="hip_api", pid=c["pid"], tid=c["tid"],
                       ts=us(c["start"]), dur=(c["end"] - c["start"]) / 1000.0,
                       args=dict(correlation_id=c["id"], parent_id=c["parent"],
                                 enter=c["args"], exit=c["exit"])))
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

# ------------------------------------------------------------------ rocprofv3 CSV
if rocprofv3_csv:
    # rocprofv3's hip_api_trace.csv: one row per completed call, in correlation-id order. The
    # domain is the dispatch table the call goes through; the generated probes name it in each
    # exit helper (exit_<Table>_<api>). Without them, the compiler table's APIs are the __hip*
    # ones.
    compiler = set()
    if probes_src:
        import re

        compiler = set(re.findall(r"exit_HipCompilerDispatchTable_(\w+)\(", open(probes_src).read()))

    def domain(name):
        is_compiler = name in compiler if compiler else name.startswith("__hip")
        return "HIP_COMPILER_API_EXT" if is_compiler else "HIP_RUNTIME_API_EXT"

    with open(rocprofv3_csv, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        w.writerow(["Domain", "Function", "Process_Id", "Thread_Id", "Correlation_Id",
                    "Start_Timestamp", "End_Timestamp"])
        for c in api_calls:
            if c["end"] is not None:
                w.writerow([domain(c["name"]), c["name"], c["pid"], c["tid"], c["id"], c["start"],
                            c["end"]])

# ------------------------------------------------------------------ report
lat = sorted(p["gpu_start"] - p["t"] for p in matched)
nested = sum(1 for c in api_calls if c["parent"] is not None)
print(f"API calls: {len(api_calls)} ({nested} nested, {len(open_calls)} without exit, "
      f"{unmatched_exit} exits without enter)   kernel_dispatch probes: {len(probes)}   "
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
print(f"wrote {run / 'trace.json'} ({len(events)} events)"
      + (f" and {rocprofv3_csv}" if rocprofv3_csv else ""))
