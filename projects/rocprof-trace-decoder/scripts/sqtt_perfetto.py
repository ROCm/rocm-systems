#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Export SQTT auto-instrumentation traces to Perfetto/Chrome JSON Trace Event
Format.

One JSON file per ui_* trace directory (each is its own SQTT time domain;
no cross-collection time alignment is attempted).

Open the resulting files at https://ui.perfetto.dev (drag & drop).

Usage:
    # Auto-discover ui_*/ dirs and code objects under base_dir
    python3 sqtt_perfetto.py build/test/ --demangle

    # Write all JSONs into one directory
    python3 sqtt_perfetto.py build/test/ -o /tmp/perfetto_out

    # Convert cycles to nanoseconds using a known GPU clock rate
    python3 sqtt_perfetto.py build/test/ --clock-rate-ghz 2.1

    # Filter to one wave
    python3 sqtt_perfetto.py build/test/ --cu 0 --simd 0 --wave 0

    # Open the first generated trace in ui.perfetto.dev
    python3 sqtt_perfetto.py build/test/ --show
"""

import argparse
import json
import os
import sys
import webbrowser
from collections import defaultdict
from typing import Optional

from sqtt_data import (
    FuncMap,
    ShaderRecord,
    WaveSpan,
    build_kernel_to_co,
    decode_marker,
    demangle_name,
    discover_base_dir,
    find_wave_span_at,
    load_funcmaps,
    load_occupancy,
    load_shaderdata,
    make_funcmap_resolver,
    merge_funcmaps,
    preprocess_records,
)


# ---------------------------------------------------------------------------
# Wave grouping helpers
# ---------------------------------------------------------------------------

WaveKey = tuple[int, int, int, int, int]  # (se, cu, simd, wave_id, instance)


def _build_slot_launches(
    wave_spans: dict[tuple, list[WaveSpan]],
) -> dict[tuple, list[int]]:
    """Sorted launch times per (SE, CU, SIMD, wave_slot). Used to assign instance
    indices to records that share a physical wave slot across relaunches."""
    slot_launches: dict[tuple, list[int]] = defaultdict(list)
    for wk, spans in wave_spans.items():
        for span in spans:
            slot_launches[wk].append(span.launch_time)
    for wk in slot_launches:
        slot_launches[wk].sort()
    return slot_launches


def _wave_instance(slot_launches: dict[tuple, list[int]],
                   shader_engine: int, cu: int, simd: int, wave_id: int, time: int) -> int:
    """Index of the most recent launch at or before `time` for this slot."""
    launches = slot_launches.get((shader_engine, cu, simd, wave_id), [])
    if not launches:
        return 0
    lo, hi = 0, len(launches) - 1
    result = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if launches[mid] <= time:
            result = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def group_by_wave_instance(
    records: list[ShaderRecord],
    wave_spans: dict[tuple, list[WaveSpan]],
    filter_cu: Optional[int] = None,
    filter_simd: Optional[int] = None,
    filter_wave: Optional[int] = None,
) -> dict[WaveKey, list[ShaderRecord]]:
    """Group records by (SE, CU, SIMD, wave_id, instance), preserving time order."""
    slot_launches = _build_slot_launches(wave_spans)
    grouped: dict[WaveKey, list[ShaderRecord]] = defaultdict(list)
    for rec in records:
        if rec.flags & 2:
            continue
        if filter_cu is not None and rec.cu != filter_cu:
            continue
        if filter_simd is not None and rec.simd != filter_simd:
            continue
        if filter_wave is not None and rec.wave_id != filter_wave:
            continue
        inst = _wave_instance(slot_launches, rec.shader_engine, rec.cu, rec.simd,
                              rec.wave_id, rec.time)
        grouped[(rec.shader_engine, rec.cu, rec.simd, rec.wave_id, inst)].append(rec)
    for k in grouped:
        grouped[k].sort(key=lambda r: r.time)
    return grouped


def pack_tid(shader_engine: int, cu: int, simd: int, wave_id: int, instance: int) -> int:
    """Pack a shader-engine-aware wave key into a 64-bit thread id.

    The lower 32 bits retain the old CU/SIMD/wave/instance layout; shader
    engine occupies the high bits so equal wave slots on two engines do not
    share a Perfetto track.
    """
    return (shader_engine << 32) | (cu << 20) | (simd << 16) | (wave_id << 8) | instance


# ---------------------------------------------------------------------------
# Per-wave event emission
# ---------------------------------------------------------------------------

# Synthetic pid for records that fall outside any known wave span.
ORPHAN_PID = 0
ORPHAN_NAME = "unknown dispatch"


def _funcmap_for_dispatch(
    dispatch_id: Optional[int],
    dispatches: dict[str, str],
    kernel_to_co: dict[str, int],
    per_co: dict[int, FuncMap],
    merged_fm: FuncMap,
) -> FuncMap:
    """Pick the funcmap to use for a given dispatch (per code object,
    falling back to the merged map)."""
    if dispatch_id is None:
        return merged_fm
    raw_name = dispatches.get(str(dispatch_id), "")
    if not raw_name:
        return merged_fm
    co_id = kernel_to_co.get(raw_name)
    if co_id is None:
        return merged_fm
    return per_co.get(co_id, merged_fm)


def emit_wave_events(
    wave_records: list[ShaderRecord],
    wave_key: WaveKey,
    dispatches: dict[str, str],
    kernel_to_co: dict[str, int],
    per_co: dict[int, FuncMap],
    merged_fm: FuncMap,
    wave_spans: dict[tuple, list[WaveSpan]],
    ts_scale: float,
) -> tuple[list[dict], int]:
    """Walk one wave's records and emit Perfetto B/E/i events.

    Returns (events, unmatched_exits). Mirrors the control flow of
    build_stacks in sqtt_flamegraph.py (dispatch-boundary close, exit_prev
    pop / enter push, funcmap resolution) but emits trace events instead
    of accumulating cycle counts.
    """
    events: list[dict] = []
    unmatched = 0
    shader_engine, cu, simd, wave_id, instance = wave_key
    tid = pack_tid(shader_engine, cu, simd, wave_id, instance)
    base_wk = wave_key[:4]
    spans = wave_spans.get(base_wk, [])

    stack: list[tuple[str, str]] = []   # (name, mtype) pairs we've pushed
    cur_dispatch: Optional[int] = None
    cur_funcmap: FuncMap = merged_fm

    def _ts(cycles: int) -> float:
        return cycles * ts_scale

    for rec in wave_records:
        # Update dispatch context. When the dispatch changes mid-stream
        # we synthesize E events for anything still on the stack -- a
        # function entered under dispatch A doesn't outlive A.
        span = find_wave_span_at(spans, rec.time)
        dispatch_id = span.dispatch_id if span else ORPHAN_PID

        if dispatch_id != cur_dispatch:
            close_pid = cur_dispatch if cur_dispatch is not None else ORPHAN_PID
            for _ in stack:
                events.append({"ph": "E", "ts": _ts(rec.time),
                               "pid": close_pid, "tid": tid})
            stack.clear()
            cur_dispatch = dispatch_id
            cur_funcmap = _funcmap_for_dispatch(
                dispatch_id, dispatches, kernel_to_co, per_co, merged_fm)

        marker_id, enter, exit_prev = decode_marker(rec.value, cur_funcmap)
        if marker_id == 0 and not exit_prev:
            continue   # noop / reserved value

        name, mtype = cur_funcmap.resolve(marker_id)

        # Point markers (barriers, memory ops, user points) are instantaneous
        # -- emit as Perfetto instant events with thread scope so they appear
        # inline in the wave's track. They don't mutate the call stack.
        if mtype == "point":
            events.append({"ph": "i", "ts": _ts(rec.time),
                           "pid": dispatch_id, "tid": tid,
                           "name": name, "cat": "sqtt.point", "s": "t"})
            continue

        # exit_prev pops first, then enter pushes -- matches build_stacks
        # ordering. Fused exit+enter (combined marker) does both.
        if exit_prev:
            if stack:
                events.append({"ph": "E", "ts": _ts(rec.time),
                               "pid": dispatch_id, "tid": tid})
                stack.pop()
            else:
                unmatched += 1

        if enter:
            events.append({"ph": "B", "ts": _ts(rec.time),
                           "pid": dispatch_id, "tid": tid,
                           "name": name, "cat": f"sqtt.{mtype}"})
            stack.append((name, mtype))

    # Close anything still open at end of wave. Use last_record.time + 1
    # so the synthetic E sorts strictly after any real event at the same
    # cycle.
    if stack:
        end_ts = _ts(wave_records[-1].time + 1)
        close_pid = cur_dispatch if cur_dispatch is not None else ORPHAN_PID
        for _ in stack:
            events.append({"ph": "E", "ts": end_ts,
                           "pid": close_pid, "tid": tid})

    return events, unmatched


# ---------------------------------------------------------------------------
# Metadata events (process / thread names)
# ---------------------------------------------------------------------------

def build_metadata_events(
    pids_seen: set[int],
    tids_seen: set[tuple],
    dispatches: dict[str, str],
    do_demangle: bool,
) -> list[dict]:
    """Emit Perfetto metadata events naming each process (dispatch) and
    each thread (wave instance).

    tids_seen is a set of (pid, tid, wave_key) triples so we can recover
    the (SE, CU, SIMD, wave_id, instance) tuple for the thread name.
    """
    metadata: list[dict] = []

    for pid in sorted(pids_seen):
        if pid == ORPHAN_PID:
            name = ORPHAN_NAME
        else:
            raw = dispatches.get(str(pid), "")
            display = demangle_name(raw) if (raw and do_demangle) else raw
            name = f"dispatch {pid}: {display}" if display else f"dispatch {pid}"
        metadata.append({
            "ph": "M", "name": "process_name", "pid": pid,
            "args": {"name": name},
        })

    for pid, tid, wave_key in sorted(tids_seen):
        shader_engine, cu, simd, wave_id, instance = wave_key
        thread_name = f"SE{shader_engine}/CU{cu}/SIMD{simd}/W{wave_id}#{instance}"
        metadata.append({
            "ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
            "args": {"name": thread_name},
        })

    return metadata


# ---------------------------------------------------------------------------
# Per-trace-dir export
# ---------------------------------------------------------------------------

def export_trace_dir(
    trace_dir: str,
    per_co: dict[int, FuncMap],
    kernel_to_co: dict[str, int],
    merged_fm: FuncMap,
    do_demangle: bool,
    clock_rate_ghz: Optional[float],
    filter_cu: Optional[int],
    filter_simd: Optional[int],
    filter_wave: Optional[int],
) -> Optional[tuple[list[dict], int, int]]:
    """Process one ui_* directory. Returns (json_events, n_events,
    unmatched_exits) or None if the directory has no usable data.

    json_events is the full list of trace events (data events + metadata)
    ready to wrap in `{"displayTimeUnit": "ns", "traceEvents": ...}`.
    """
    records = load_shaderdata(trace_dir)
    if not records:
        return None
    dispatches, wave_spans = load_occupancy(trace_dir)
    resolver = make_funcmap_resolver(per_co, dispatches, kernel_to_co, wave_spans, merged_fm)
    records, _addr = preprocess_records(records, merged_fm, resolver)
    if not records:
        return None

    grouped = group_by_wave_instance(
        records, wave_spans,
        filter_cu=filter_cu, filter_simd=filter_simd, filter_wave=filter_wave)
    if not grouped:
        return None

    # ts_scale: cycles -> displayed time unit. Default leaves cycles as-is
    # and labels them ns (proportions correct, absolute values nominal).
    # With --clock-rate-ghz we convert to real ns.
    ts_scale = (1.0 / clock_rate_ghz) if clock_rate_ghz else 1.0

    all_events: list[dict] = []
    pids_seen: set[int] = set()
    tids_seen: set[tuple] = set()
    total_unmatched = 0

    for wave_key, recs in grouped.items():
        evs, unmatched = emit_wave_events(
            recs, wave_key, dispatches, kernel_to_co, per_co, merged_fm,
            wave_spans, ts_scale)
        total_unmatched += unmatched
        all_events.extend(evs)
        shader_engine, cu, simd, wave_id, instance = wave_key
        tid = pack_tid(shader_engine, cu, simd, wave_id, instance)
        for ev in evs:
            pids_seen.add(ev["pid"])
            tids_seen.add((ev["pid"], tid, wave_key))

    if not all_events:
        return None

    metadata = build_metadata_events(pids_seen, tids_seen, dispatches,
                                     do_demangle)
    # Metadata first so Perfetto picks up names before rendering tracks.
    return (metadata + all_events, len(all_events), total_unmatched)


def write_perfetto_json(
    events: list[dict],
    output_path: str,
) -> int:
    """Write the trace event list as a Chrome JSON Trace. Returns file size."""
    payload = {
        "displayTimeUnit": "ns",
        "traceEvents": events,
    }
    # Compact separators — these files can get large.
    with open(output_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return os.path.getsize(output_path)


def output_path_for(trace_dir: str, out_dir: Optional[str]) -> str:
    """Resolve the output JSON path. With out_dir, files live in out_dir/.
    Without, files are siblings of the ui_* directory.
    """
    name = os.path.basename(os.path.normpath(trace_dir)) + ".perfetto.json"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, name)
    return os.path.join(os.path.dirname(os.path.abspath(trace_dir)), name)


# ---------------------------------------------------------------------------
# Self-test (no rocprofv3 needed)
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Minimal in-memory smoke test of the event-emission pipeline.

    Builds synthetic ShaderRecord/WaveSpan/FuncMap inputs, runs
    export_trace_dir's core logic via emit_wave_events, and asserts the
    resulting event list contains the expected B/E/i/M shape. Returns 0
    on success, nonzero on failure.
    """
    # Funcmap: id 1 = "compute" (function), id 2 = "barrier" (point).
    fm = FuncMap()
    fm.markers[1] = ("compute", "function")
    fm.markers[2] = ("barrier", "point")

    # Encode marker values: (id << 2) | enter | exit_prev
    enter_compute = (1 << 2) | 0x2
    exit_prev = 0x1
    point_barrier = (2 << 2)

    # One wave (SE=0, cu=0, simd=1, wave_id=3) running dispatch 7.
    recs = [
        ShaderRecord(time=100, value=enter_compute, cu=0, simd=1, wave_id=3, flags=0),
        ShaderRecord(time=200, value=point_barrier, cu=0, simd=1, wave_id=3, flags=0),
        ShaderRecord(time=300, value=exit_prev,    cu=0, simd=1, wave_id=3, flags=0),
    ]
    wave_spans = {(0, 0, 1, 3): [WaveSpan(launch_time=0, retire_time=1000, dispatch_id=7)]}
    dispatches = {"7": "compute_kernel<int>"}

    grouped = group_by_wave_instance(recs, wave_spans)
    assert len(grouped) == 1, f"expected 1 wave group, got {len(grouped)}"
    wave_key = next(iter(grouped))
    assert wave_key == (0, 0, 1, 3, 0), wave_key

    events, unmatched = emit_wave_events(
        grouped[wave_key], wave_key, dispatches, kernel_to_co={},
        per_co={}, merged_fm=fm, wave_spans=wave_spans, ts_scale=1.0)
    assert unmatched == 0, f"unmatched exits: {unmatched}"

    # Expect: B(compute), i(barrier), E(compute)
    phs = [e["ph"] for e in events]
    assert phs == ["B", "i", "E"], phs
    assert events[0]["name"] == "compute" and events[0]["cat"] == "sqtt.function"
    assert events[1]["name"] == "barrier" and events[1]["cat"] == "sqtt.point"
    assert events[1]["s"] == "t"
    assert all(e["pid"] == 7 for e in events)
    expected_tid = pack_tid(0, 0, 1, 3, 0)
    assert all(e["tid"] == expected_tid for e in events)

    # B/E pairing (per (pid, tid))
    depth = 0
    for e in events:
        if e["ph"] == "B":
            depth += 1
        elif e["ph"] == "E":
            depth -= 1
        assert depth >= 0, "negative stack depth"
    assert depth == 0, f"unbalanced B/E (final depth {depth})"

    # Metadata events
    pids_seen = {e["pid"] for e in events}
    tids_seen = {(e["pid"], expected_tid, wave_key) for e in events}
    md = build_metadata_events(pids_seen, tids_seen, dispatches, do_demangle=False)
    md_phs = {e["name"] for e in md if e["ph"] == "M"}
    assert "process_name" in md_phs and "thread_name" in md_phs, md_phs

    # Round-trip the full payload through json to be sure it serializes
    payload = {"displayTimeUnit": "ns", "traceEvents": md + events}
    encoded = json.dumps(payload)
    reparsed = json.loads(encoded)
    assert reparsed["traceEvents"][-1]["ph"] == "E"

    print("self-test OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export SQTT auto-instrumentation traces to Perfetto JSON")
    parser.add_argument("base_dir", nargs="?",
        help="Base directory containing trace output. Auto-discovers "
             "ui_*/shaderdata_*.json and *code_object_id*.out files "
             "recursively.")
    parser.add_argument("--code-objects", "-c", nargs="+",
        help="Code object files (.out) with .sqtt_funcmap sections. "
             "If omitted, auto-detected from base_dir.")
    parser.add_argument("--output-dir", "-o", default=None,
        help="Directory to write *.perfetto.json files into. "
             "Default: write next to each ui_* directory.")
    parser.add_argument("--demangle", "-d", action="store_true",
        help="Demangle C++ function and dispatch names")
    parser.add_argument("--clock-rate-ghz", type=float, default=None,
        help="Convert SQTT cycles to ns using this GPU clock rate. "
             "Default: emit cycles as-is and label them ns.")
    parser.add_argument("--cu", type=int, default=None,
        help="Filter to specific CU/WGP ID")
    parser.add_argument("--simd", type=int, default=None,
        help="Filter to specific SIMD ID")
    parser.add_argument("--wave", type=int, default=None,
        help="Filter to specific wave ID")
    parser.add_argument("--show", "-s", action="store_true",
        help="Open ui.perfetto.dev in a browser after export. "
             "(The user must drag-and-drop the generated JSON file.)")
    parser.add_argument("--self-test", action="store_true",
        help="Run an internal smoke test on synthetic records and exit.")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.base_dir:
        parser.error("base_dir is required (or use --self-test)")

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory", file=sys.stderr)
        return 1

    trace_dirs, discovered_cos = discover_base_dir(base_dir)
    code_objects = args.code_objects or discovered_cos

    if not trace_dirs:
        print(f"Error: no shaderdata_*.json files found under {base_dir}",
              file=sys.stderr)
        return 1
    if not code_objects:
        print("Error: no code objects found. Use --code-objects.",
              file=sys.stderr)
        return 1

    print(f"Found {len(trace_dirs)} trace dir(s), "
          f"{len(code_objects)} code object(s)", file=sys.stderr)

    # Funcmaps: load unmangled to build the kernel->code_object map (which
    # keys on mangled names from occupancy.json), then optionally a
    # demangled copy for display.
    per_co_raw = load_funcmaps(code_objects, do_demangle=False)
    kernel_to_co = build_kernel_to_co(per_co_raw)
    per_co = (load_funcmaps(code_objects, do_demangle=True)
              if args.demangle else per_co_raw)

    merged_fm = merge_funcmaps(per_co.values())

    written: list[tuple[str, int, int]] = []
    skipped: list[str] = []
    total_unmatched = 0

    for td in trace_dirs:
        result = export_trace_dir(
            td, per_co, kernel_to_co, merged_fm,
            do_demangle=args.demangle,
            clock_rate_ghz=args.clock_rate_ghz,
            filter_cu=args.cu, filter_simd=args.simd, filter_wave=args.wave)
        if result is None:
            skipped.append(td)
            continue
        events, n_events, unmatched = result
        total_unmatched += unmatched
        out_path = output_path_for(td, args.output_dir)
        size = write_perfetto_json(events, out_path)
        written.append((out_path, size, n_events))

    if not written:
        print("Error: no perfetto traces produced", file=sys.stderr)
        return 1

    print(f"Wrote {len(written)} perfetto trace(s):", file=sys.stderr)
    for path, size, n_events in written:
        size_mb = size / (1024 * 1024)
        print(f"  {path} ({size_mb:.1f} MB, {n_events} events)",
              file=sys.stderr)
    if skipped:
        print(f"Skipped {len(skipped)} empty trace dir(s)", file=sys.stderr)
    if total_unmatched:
        print(f"Note: {total_unmatched} unmatched exit marker(s) "
              "(stack was empty when an exit was encountered)",
              file=sys.stderr)
    if not args.clock_rate_ghz:
        print("Note: timestamps are SQTT cycles displayed as ns. "
              "Pass --clock-rate-ghz to convert to real time.",
              file=sys.stderr)
    print("Open at https://ui.perfetto.dev (drag & drop)", file=sys.stderr)

    if args.show:
        webbrowser.open("https://ui.perfetto.dev")

    return 0


if __name__ == "__main__":
    sys.exit(main())
