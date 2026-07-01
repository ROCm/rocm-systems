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

# Validates multi-pass kernel-replay counter collection (JSON output) produced by:
#   rocprofv3 --pmc <grp0> --pmc <grp1> ... --kernel-replay -- kernel-replay
# where each --pmc group shares the same sanity counters (SQ_WAVES, SQ_INSTS_VALU) plus one unique
# counter. The number of groups drives the number of replay passes.
#
# Key validation metrics:
#   1. Every dispatch is replayed exactly N times (replay_pass 0..N-1, one per --pmc group).
#   2. The shared sanity counters are CONSTANT across a given kernel's replay passes -- the
#      determinism snapshot/restore guarantees (each pass runs against identical inputs).
#   3. Each pass collects a DISTINCT batch (the unique counters differ pass-to-pass), proving real
#      multi-pass rather than one repeated batch.
#   4. Counters differ between the three kernels (vecAdd / saxpy / vecScale), so they are real
#      per-dispatch measurements, not a constant artifact.
# (The app verifies its own results in the generate step, guarding the restored data itself.)

import collections
import sys

import pytest

# Tolerate the in-flight rename of the per-pass index field (replay_pass <-> n).
_PASS_KEYS = ("replay_pass", "n")

# Launch dims of the kernel-replay app's three kernels (grid_size = grid blocks x block threads):
#   vecAdd<<<1024,1024>>>, saxpy<<<512,512>>>, vecScale<<<256,256>>>.
EXPECTED_DIMS = {
    "vecAdd": {"grid_size": 1024 * 1024, "workgroup_size": 1024},
    "saxpy": {"grid_size": 512 * 512, "workgroup_size": 512},
    "vecScale": {"grid_size": 256 * 256, "workgroup_size": 256},
}

EXPECTED_KERNELS = ("vecAdd", "saxpy", "vecScale")

# Union of every counter across all --pmc groups; all must appear in the collected records.
EXPECTED_COUNTERS = (
    "SQ_WAVES",
    "SQ_INSTS_VALU",
    "GRBM_COUNT",
    "GRBM_GUI_ACTIVE",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
)

DIM_TOLERANCE = 0.05
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


def _records_by_dispatch(sdk):
    """dispatch_id -> {"kernel": name, "passes": {pass_index: {counter_name: value}}}."""
    counter_id_to_name = _counter_id_to_name(sdk)
    kernel_id_to_name = _kernel_id_to_name(sdk)
    table = {}
    for rec in _counter_records(sdk):
        did = _dispatch_id(rec)
        entry = table.setdefault(
            did,
            {
                "kernel": kernel_id_to_name.get(
                    int(_dispatch_info(rec)["kernel_id"]), ""
                ),
                "passes": {},
            },
        )
        entry["passes"][_pass_index(rec)] = _aggregated_named_counters(
            rec, counter_id_to_name
        )
    assert table, "no counter records found"
    return table


def test_every_dispatch_replayed_n_passes(json_data, expected_passes):
    table = _records_by_dispatch(_sdk(json_data))
    want = set(range(expected_passes))
    for dispatch_id, entry in table.items():
        passes = set(entry["passes"])
        assert (
            passes == want
        ), f"dispatch {dispatch_id} ({entry['kernel']}) passes={sorted(passes)}, expected {sorted(want)}"


def test_common_counters_constant_across_passes(json_data, common_counters):
    # Metric 2: the shared sanity counters appear in every pass and are constant for a kernel.
    table = _records_by_dispatch(_sdk(json_data))
    for dispatch_id, entry in table.items():
        passes = entry["passes"]
        for counter in common_counters:
            values = [batch[counter] for batch in passes.values() if counter in batch]
            assert len(values) == len(passes), (
                f"dispatch {dispatch_id} ({entry['kernel']}) common counter {counter} missing in "
                f"some passes: present in {len(values)}/{len(passes)}"
            )
            assert _approx_equal(min(values), max(values)), (
                f"dispatch {dispatch_id} ({entry['kernel']}) counter {counter} varies across "
                f"replay passes beyond {COUNTER_TOLERANCE:.0%}: {values}"
            )


def test_each_pass_collects_distinct_batch(json_data, expected_passes, common_counters):
    # Metric 3: ignoring the shared counters, each pass contributes a distinct unique counter.
    table = _records_by_dispatch(_sdk(json_data))
    common = set(common_counters)
    for dispatch_id, entry in table.items():
        unique_per_pass = set()
        for batch in entry["passes"].values():
            unique_per_pass.update(c for c in batch if c not in common)
        assert len(unique_per_pass) == expected_passes, (
            f"dispatch {dispatch_id} ({entry['kernel']}) expected {expected_passes} distinct "
            f"per-pass counters, got {sorted(unique_per_pass)}"
        )


def test_counters_differ_between_kernels(json_data, common_counters):
    # Metric 4: each kernel has a distinct signature over the shared counters.
    table = _records_by_dispatch(_sdk(json_data))
    signatures = {}
    for entry in table.values():
        first_pass = entry["passes"][min(entry["passes"])]
        sig = tuple(round(first_pass.get(c, float("nan")), 3) for c in common_counters)
        signatures[entry["kernel"]] = sig
    values = list(signatures.values())
    assert len(set(values)) == len(
        values
    ), f"kernels are not distinguishable by common counters {common_counters}: {signatures}"


def test_replayed_kernels_present(json_data):
    names = {entry["kernel"] for entry in _records_by_dispatch(_sdk(json_data)).values()}
    for kernel in EXPECTED_KERNELS:
        assert any(kernel in (n or "") for n in names), f"{kernel} not found in {names}"


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
        assert counter in seen, f"counter {counter} not collected; seen={sorted(seen)}"


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
