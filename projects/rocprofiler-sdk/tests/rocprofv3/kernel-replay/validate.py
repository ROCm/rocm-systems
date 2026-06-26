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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Validates kernel-replay counter collection (JSON output) produced by:
#   rocprofv3 --pmc <counter> --kernel-replay --kernel-replay-passes N -- kernel-replay ...
#
# Integration-test equivalent of the manual snap/restore loop. Key validation metrics:
#   1. Counters across replay passes are approximately the same for a given kernel (restore makes
#      every pass run against identical inputs, so per-pass counters should match within tolerance).
#   2. Counters differ between different kernels (vecAdd vs saxpy), proving the counters are
#      meaningful per-dispatch and not a constant artifact.
#   3. We expect exactly N replay passes per dispatch in the output file.
# (The app verifies its own results in the generate step, guarding the data itself.)

import collections
import sys

import pytest

# Tolerate the in-flight rename of the per-pass index field (replay_pass <-> n).
_PASS_KEYS = ("replay_pass", "n")

# Expected launch dimensions for the kernel-replay app's fixed configs (grid_size = grid blocks x
# block threads, workgroup_size = block threads): vecAdd<<<1024,1024>>>, saxpy<<<512,512>>>.
EXPECTED_DIMS = {
    "vecAdd": {"grid_size": 1024 * 1024, "workgroup_size": 1024},
    "saxpy": {"grid_size": 512 * 512, "workgroup_size": 512},
}

# Acceptable relative deviation for the launch-dimension checks.
DIM_TOLERANCE = 0.05

# Counters requested via --pmc; all must appear in the collected counter records.
EXPECTED_COUNTERS = ("SQ_WAVES", "SQ_INSTS_VALU")

# Acceptable relative deviation when comparing counter values (same kernel across passes, and to
# decide whether two kernels' counters are "the same").
COUNTER_TOLERANCE = 0.10


def _within_tolerance(actual, expected):
    return abs(actual - expected) <= DIM_TOLERANCE * expected


def _approx_equal(a, b, tol=COUNTER_TOLERANCE):
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= tol * scale


def _sdk(json_data):
    assert "rocprofiler-sdk-tool" in json_data, "missing rocprofiler-sdk-tool in JSON"
    tool = json_data["rocprofiler-sdk-tool"]
    if isinstance(tool, list):
        assert len(tool) > 0, "empty rocprofiler-sdk-tool array"
        tool = tool[0]
    return tool


def _counter_records(sdk):
    callback = sdk.get("callback_records", {})
    records = callback.get("counter_collection")
    assert records, "no counter_collection records in callback_records"
    return records


def _pass_index(record):
    for key in _PASS_KEYS:
        if key in record and record[key] is not None:
            return int(record[key])
    raise AssertionError(
        f"no replay-pass field {_PASS_KEYS} in counter record keys={list(record.keys())}"
    )


def _dispatch_info(record):
    return record["dispatch_data"]["dispatch_info"]


def _dispatch_id(record):
    return int(_dispatch_info(record)["dispatch_id"])


def _aggregated_named_counters(record, counter_id_to_name):
    """counter name -> summed value across all dimension instances in this record."""
    agg = collections.defaultdict(float)
    for sub in record.get("records", []):
        name = counter_id_to_name.get(int(sub["counter_id"]["handle"]))
        if name is not None:
            agg[name] += float(sub["value"])
    assert agg, "counter record has no named counter values"
    return dict(agg)


def _counter_id_to_name(sdk):
    counters = sdk.get("counters")
    assert counters, "missing counters section in JSON"
    entries = counters.values() if isinstance(counters, dict) else counters
    mapping = {}
    for counter in entries:
        cid = counter.get("id", {})
        handle = cid.get("handle") if isinstance(cid, dict) else cid
        name = counter.get("name")
        if handle is not None and name:
            mapping[int(handle)] = name
    assert mapping, "no named counters found"
    return mapping


def _kernel_id_to_name(sdk):
    symbols = sdk.get("kernel_symbols")
    if not symbols:
        callback = sdk.get("callback_records", {})
        symbols = callback.get("kernel_symbols") if isinstance(callback, dict) else None
    assert symbols, "missing kernel_symbols in JSON"

    entries = symbols.values() if isinstance(symbols, dict) else symbols
    mapping = {}
    for sym in entries:
        kid = sym.get("kernel_id")
        name = (
            sym.get("formatted_kernel_name")
            or sym.get("demangled_kernel_name")
            or sym.get("kernel_name")
        )
        if kid is not None and name:
            mapping[int(kid)] = name
    assert mapping, "no named kernel symbols found"
    return mapping


def test_every_dispatch_replayed_n_passes(json_data, expected_passes):
    records = _counter_records(_sdk(json_data))
    passes_by_dispatch = collections.defaultdict(set)
    for rec in records:
        passes_by_dispatch[_dispatch_id(rec)].add(_pass_index(rec))

    assert passes_by_dispatch, "no dispatches found in counter records"
    want = set(range(expected_passes))
    for dispatch_id, passes in passes_by_dispatch.items():
        assert (
            passes == want
        ), f"dispatch {dispatch_id} replay passes={sorted(passes)}, expected {sorted(want)}"


def test_counters_consistent_across_passes(json_data):
    # Metric 1: for a given kernel (dispatch), each counter is approximately the same across all
    # replay passes -- the determinism that snapshot/restore guarantees.
    sdk = _sdk(json_data)
    counter_id_to_name = _counter_id_to_name(sdk)

    # dispatch_id -> counter_name -> [value per pass]
    by_dispatch = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in _counter_records(sdk):
        for name, value in _aggregated_named_counters(rec, counter_id_to_name).items():
            by_dispatch[_dispatch_id(rec)][name].append(value)

    assert by_dispatch, "no counter values found"
    for dispatch_id, counters in by_dispatch.items():
        for name, values in counters.items():
            assert _approx_equal(min(values), max(values)), (
                f"dispatch {dispatch_id} counter {name} varies across replay passes beyond "
                f"{COUNTER_TOLERANCE:.0%}: {values}"
            )


def test_counters_differ_between_kernels(json_data):
    # Metric 2: vecAdd and saxpy are different kernels, so at least one shared counter must differ
    # beyond tolerance (otherwise the counters aren't actually measuring the dispatch).
    sdk = _sdk(json_data)
    counter_id_to_name = _counter_id_to_name(sdk)
    kernel_id_to_name = _kernel_id_to_name(sdk)

    per_kernel = {}
    for rec in _counter_records(sdk):
        name = kernel_id_to_name.get(int(_dispatch_info(rec)["kernel_id"]), "") or ""
        for target in ("vecAdd", "saxpy"):
            if target in name and target not in per_kernel:
                per_kernel[target] = _aggregated_named_counters(rec, counter_id_to_name)

    assert (
        "vecAdd" in per_kernel and "saxpy" in per_kernel
    ), f"need both vecAdd and saxpy counter records; found {sorted(per_kernel)}"
    vecadd, saxpy = per_kernel["vecAdd"], per_kernel["saxpy"]
    shared = set(vecadd) & set(saxpy)
    assert shared, "no shared counters between vecAdd and saxpy to compare"
    differing = [c for c in shared if not _approx_equal(vecadd[c], saxpy[c])]
    assert differing, (
        f"vecAdd and saxpy counters are indistinguishable within {COUNTER_TOLERANCE:.0%}: "
        f"vecAdd={vecadd} saxpy={saxpy}"
    )


def test_replayed_kernels_present(json_data):
    sdk = _sdk(json_data)
    id_to_name = _kernel_id_to_name(sdk)
    names = {
        id_to_name.get(int(_dispatch_info(rec)["kernel_id"]))
        for rec in _counter_records(sdk)
    }
    assert any(n and "vecAdd" in n for n in names), f"vecAdd not found in {names}"
    assert any(n and "saxpy" in n for n in names), f"saxpy not found in {names}"


def test_expected_counters_present(json_data):
    sdk = _sdk(json_data)
    id_to_name = _counter_id_to_name(sdk)

    seen = set()
    for rec in _counter_records(sdk):
        for sub in rec.get("records", []):
            name = id_to_name.get(int(sub["counter_id"]["handle"]))
            if name:
                seen.add(name)

    for counter in EXPECTED_COUNTERS:
        assert (
            counter in seen
        ), f"counter {counter} not found in collected counters: {sorted(seen)}"


def test_launch_dimensions(json_data):
    sdk = _sdk(json_data)
    id_to_name = _kernel_id_to_name(sdk)

    checked = set()
    for rec in _counter_records(sdk):
        info = _dispatch_info(rec)
        name = id_to_name.get(int(info["kernel_id"]), "") or ""
        for key, expected in EXPECTED_DIMS.items():
            if key not in name:
                continue
            grid = int(info["grid_size"]["x"])
            workgroup = int(info["workgroup_size"]["x"])
            assert _within_tolerance(
                grid, expected["grid_size"]
            ), f"{key} grid_size {grid} not within {DIM_TOLERANCE:.0%} of {expected['grid_size']}"
            assert _within_tolerance(workgroup, expected["workgroup_size"]), (
                f"{key} workgroup_size {workgroup} not within {DIM_TOLERANCE:.0%} of "
                f"{expected['workgroup_size']}"
            )
            checked.add(key)

    missing = set(EXPECTED_DIMS) - checked
    assert not missing, f"expected kernels not found for dimension check: {missing}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
