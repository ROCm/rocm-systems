#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Performance validation for rocprofv3 kernel-replay JSON output:
#   - Total tool duration bounded (init_time .. fini_time).
#   - Per-pass kernel intervals stable (no pass-index blow-up).
#   - Replay overhead vs single-pass baseline within scaling ratio.

import json
import sys
from collections import defaultdict

import pytest

MIN_GBPS = 8.0
# Footprint heuristic for the kernel-replay test app: n floats * 4 bytes per buffer, ~3 buffers
# plus HIP runtime inventory. Used only for a loose ceiling, not exact accounting.
BUFFER_COUNT = 3
OVERHEAD_MARGIN = 8.0
MAX_TOOL_SECONDS = 120.0
MAX_PASS_SPAN_RATIO = 4.0


def _sdk(json_data):
    tool = json_data["rocprofiler-sdk-tool"]
    if isinstance(tool, list):
        tool = tool[0]
    return tool


def _counter_records(sdk):
    records = sdk.get("callback_records", {}).get("counter_collection")
    assert records, "no counter_collection records"
    return records


def _pass_index(record):
    for key in ("replay_pass", "n"):
        if key in record and record[key] is not None:
            return int(record[key])
    raise AssertionError(f"no replay pass index in {record.keys()}")


def tool_duration_seconds(sdk) -> float:
    meta = sdk.get("metadata", {})
    init_t = meta.get("init_time")
    fini_t = meta.get("fini_time")
    assert init_t is not None and fini_t is not None
    return (int(fini_t) - int(init_t)) / 1e9


def per_pass_spans_ns(records):
    if not any("replay_pass" in r or "n" in r for r in records):
        return {}
    by_pass = defaultdict(list)
    for rec in records:
        dd = rec.get("dispatch_data", {})
        start = dd.get("start_timestamp")
        end = dd.get("end_timestamp")
        if start is None or end is None:
            continue
        by_pass[_pass_index(rec)].append(int(end) - int(start))
    return {p: sum(v) / len(v) for p, v in sorted(by_pass.items())}


def model_tool_ceiling_seconds(passes: int, n_elems: int, dispatches: int) -> float:
    footprint = BUFFER_COUNT * n_elems * 4
    per_dispatch = passes * footprint
    dma = per_dispatch / (MIN_GBPS * 1e9)
    return dispatches * dma * OVERHEAD_MARGIN + 5.0  # fixed startup allowance


def test_kernel_replay_performance(json_data, expected_passes, request):
    n_elems = request.config.getoption("--n-elems")
    baseline_json = request.config.getoption("--baseline-json")
    max_scaling_ratio = request.config.getoption("--max-scaling-ratio")
    sdk = _sdk(json_data)
    records = _counter_records(sdk)

    duration = tool_duration_seconds(sdk)
    assert (
        duration <= MAX_TOOL_SECONDS
    ), f"tool duration {duration:.1f}s exceeds hard cap {MAX_TOOL_SECONDS}s"

    dispatches = len(
        {r["dispatch_data"]["dispatch_info"]["dispatch_id"] for r in records}
    )
    ceiling = model_tool_ceiling_seconds(expected_passes, n_elems, dispatches)
    assert duration <= ceiling, (
        f"tool duration {duration:.1f}s exceeds cost-model ceiling {ceiling:.1f}s "
        f"(P={expected_passes} dispatches={dispatches})"
    )
    print(
        f"[kr-perf-json] PASS duration={duration:.2f}s <= ceiling={ceiling:.1f}s "
        f"(P={expected_passes})"
    )

    spans = per_pass_spans_ns(records)
    if spans:
        min_span = min(spans.values())
        max_span = max(spans.values())
        if min_span > 0:
            ratio = max_span / min_span
            assert (
                ratio <= MAX_PASS_SPAN_RATIO
            ), f"per-pass span ratio {ratio:.2f} > {MAX_PASS_SPAN_RATIO}: {spans}"
            print(f"[kr-perf-json] PASS per-pass span ratio={ratio:.2f} spans_ns={spans}")
    else:
        print("[kr-perf-json] SKIP per-pass span check (replay_pass not in JSON records)")

    if baseline_json:
        with open(baseline_json, encoding="utf-8") as f:
            base_data = json.load(f)
        base_dur = tool_duration_seconds(_sdk(base_data))
        if base_dur > 0:
            scale = duration / base_dur
            assert scale <= max_scaling_ratio, (
                f"duration scaling {scale:.2f} > {max_scaling_ratio} "
                f"(P={expected_passes} {duration:.1f}s vs baseline {base_dur:.1f}s)"
            )
            print(
                f"[kr-perf-json] PASS duration scaling ratio={scale:.2f} "
                f"(P={expected_passes} vs baseline)"
            )


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
