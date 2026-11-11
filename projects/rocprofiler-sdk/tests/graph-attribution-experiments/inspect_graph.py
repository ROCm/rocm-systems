"""Inspect per-kernel graph visibility in an SDK JSON-tool trace (one process)."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(document, window=None):
    data = document["rocprofiler-sdk-json-tool"]
    records = data["buffer_records"]
    symbols = {
        r["payload"]["kernel_id"]: r["payload"]["kernel_name"]
        for r in data["callback_records"]["kernel_symbols"]
    }
    api_calls = {r["correlation_id"]["internal"]: r for r in records["hip_api_traces"]}
    names = next(
        k["operations"] for k in records["names"] if k["kind"] == "HIP_RUNTIME_API"
    )
    rows = []
    groups = defaultdict(list)
    for r in records["kernel_dispatch"]:
        if window and not (
            window[0] <= r["start_timestamp"] <= r["end_timestamp"] <= window[1]
        ):
            continue
        info = r["dispatch_info"]
        graph = info["graph_exec_id"]["handle"]
        if not graph:
            continue
        capture = info["capture_correlation_id"]
        api = api_calls.get(capture) if capture else None
        launch_id = r["correlation_id"]["internal"]
        row = {
            "graph_exec_id": graph,
            "launch_correlation_id": launch_id,
            "graph_node_ordinal": info["graph_node_id"]["handle"],
            "dispatch_id": info["dispatch_id"],
            "kernel_name": symbols.get(info["kernel_id"], "<missing symbol>"),
            "start_ns": r["start_timestamp"],
            "end_ns": r["end_timestamp"],
            "duration_ns": r["end_timestamp"] - r["start_timestamp"],
            "capture_correlation_id": capture,
            "capture_api": names[api["operation"]] if api else "",
            "capture_thread": api["thread_id"] if api else "",
            "capture_external_id": (
                api["correlation_id"].get("external", "") if api else ""
            ),
            "capture_resolved": bool(api),
        }
        rows.append(row)
        groups[(graph, launch_id)].append(row)
    duration = sum(r["duration_ns"] for r in rows)
    resolved = [r for r in rows if r["capture_resolved"]]
    summary = {
        "missing_kernel_names": sum(r["kernel_name"] == "<missing symbol>" for r in rows),
        "graph_dispatches": len(rows),
        "graph_execs": len({key[0] for key in groups}),
        "graph_launches": len(groups),
        "graph_internal_kernels_visible": bool(rows),
        "capture_resolved_dispatches": len(resolved),
        "capture_coverage_by_count": len(resolved) / len(rows) if rows else None,
        "capture_coverage_by_summed_kernel_time": (
            sum(r["duration_ns"] for r in resolved) / duration if duration else None
        ),
        "launch_kernel_counts": dict(Counter(len(rs) for rs in groups.values())),
        "unresolved_kernel_counts": dict(
            Counter(r["kernel_name"] for r in rows if not r["capture_resolved"])
        ),
        "layer_op_labels_verified": False,
        "limitations": [
            "Capture correlation identifies a HIP call, not a Python layer without framework labels.",
            "External IDs in the test JSON tool may be thread IDs; do not treat them as layer names.",
            "Graph node IDs are enqueue ordinals with upstream determinism constraints.",
            "Summed kernel time can double count overlap; it is not wall time.",
            "Fused kernel internals are not separate graph nodes.",
            "This trace alone does not prove a 500k prompt; verify request/server evidence separately.",
        ],
    }
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--request", type=Path, help="Request JSON with CLOCK_BOOTTIME boundaries"
    )
    args = p.parse_args()
    request = json.loads(args.request.read_text()) if args.request else None
    window = (
        (request["clocks_start_ns"]["7"], request["clocks_end_ns"]["7"])
        if request
        else None
    )
    rows, summary = analyze(json.loads(args.trace.read_text()), window)
    summary["request_window_boottime_ns"] = window
    if request:
        summary["request_meta_info"] = request["meta_info"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (args.out / "graph-kernels.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0])
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
