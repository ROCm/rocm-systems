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
#   rocprofv3 --pmc <grp0> --pmc <grp1> ... --replay-mode kernel --kernel-replay-beta-enabled -- kernel-replay
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
#   5. dispatch_id identity: a replayed dispatch keeps ONE dispatch_id across its passes (only the
#      replay_pass index advances), and dispatch_id increments sequentially across dispatches -- one
#      id minted per logical dispatch, reused by every pass (never one id per pass).
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

# Counters fixed by launch geometry and the instruction stream: replaying one dispatch against
# restored inputs reproduces them bit-for-bit, so they are compared exactly. A relative band would
# only hide structural loss -- these values are summed over every hardware instance, so a pass that
# drops 1 of 32 instance records is off by 3%, well inside COUNTER_TOLERANCE. GRBM_* cycle counters
# are absent on purpose: they track cache and memory-controller state the snapshot cannot restore,
# so they do need a band if ever used as a common counter.
EXACT_ACROSS_PASSES = frozenset(
    {
        "SQ_WAVES",
        "SQ_INSTS_VALU",
        "SQ_INSTS_SALU",
        "SQ_INSTS_SMEM",
        "SQ_INSTS_LDS",
    }
)


def _within_tolerance(actual, expected):
    return abs(actual - expected) <= DIM_TOLERANCE * expected


def _approx_equal(a, b, tol=COUNTER_TOLERANCE):
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= tol * scale


def _pass_tolerance(counter):
    return 0.0 if counter in EXACT_ACROSS_PASSES else COUNTER_TOLERANCE


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


def _dispatch_passes(sdk):
    """dispatch_id -> {"kernel_id": int, "passes": [replay_pass, ...]} from raw counter records.

    Kept separate from _records_by_dispatch, which keys passes in a dict (hiding a duplicate pass
    index) and groups by counter name. Here every pass occurrence is retained alongside the
    kernel_id so a dispatch_id can be asserted stable: one kernel, each replay pass exactly once.
    """
    table = {}
    for rec in _counter_records(sdk):
        info = _dispatch_info(rec)
        did = int(info["dispatch_id"])
        kid = int(info["kernel_id"])
        entry = table.setdefault(did, {"kernel_id": kid, "passes": []})
        assert entry["kernel_id"] == kid, (
            f"dispatch_id {did} appears with two kernel_ids {entry['kernel_id']} and {kid}: "
            "a dispatch_id must identify a single logical dispatch"
        )
        entry["passes"].append(_pass_index(rec))
    assert table, "no counter records found"
    return table


def test_dispatch_id_constant_across_replay_passes(json_data, expected_passes):
    # A replayed dispatch carries ONE dispatch_id across all of its passes; only the replay_pass
    # index advances, taking each value 0..N-1 exactly once. This is the reserve-one-id-per-dispatch
    # invariant: minting an id per pass would fan a single dispatch across N ids (shrinking each
    # pass sequence to one entry), and a CONFIG/pass id mismatch would drop or duplicate a pass.
    table = _dispatch_passes(_sdk(json_data))
    want = list(range(expected_passes))
    for did, entry in table.items():
        passes = sorted(entry["passes"])
        assert passes == want, (
            f"dispatch {did} (kernel_id={entry['kernel_id']}) replay_pass sequence {passes} != "
            f"{want}: dispatch_id must stay constant while replay_pass covers 0..N-1 once each"
        )


def test_dispatch_ids_increment_sequentially(json_data):
    # Across distinct dispatches the dispatch_id increments sequentially -- one fresh id per logical
    # dispatch, none skipped, none reused. Replay passes reuse their dispatch's reserved id (they
    # must not consume new ids), so the observed ids form a contiguous ascending run. A regression
    # that minted per pass, or double-minted at CONFIG, would leave gaps or duplicates here.
    table = _dispatch_passes(_sdk(json_data))
    ids = sorted(table)
    assert len(ids) == len(set(ids)), f"duplicate dispatch_ids observed: {ids}"
    assert ids == list(
        range(ids[0], ids[0] + len(ids))
    ), f"dispatch_ids are not sequential (expected a contiguous ascending run): {ids}"


def test_dispatch_id_nonzero_every_pass(json_data):
    # dispatch_id 0 is the "unset" sentinel (make_dispatch_info leaves it 0): the replay path
    # reserves a real, nonzero id before CONFIG and threads it through every pass, so no pass record
    # may carry 0. This is the C9 regression directly -- CONFIG/pass callbacks and pass records
    # previously saw 0 instead of the reserved id, which a contiguous-run check alone can mask.
    zero = [
        _pass_index(rec)
        for rec in _counter_records(_sdk(json_data))
        if _dispatch_id(rec) == 0
    ]
    assert (
        not zero
    ), f"{len(zero)} pass record(s) carry dispatch_id==0 (replay_pass indices {zero})"


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
            # A counter stuck at zero is constant, so the spread check below cannot see it.
            assert min(values) > 0, (
                f"dispatch {dispatch_id} ({entry['kernel']}) common counter {counter} is not > 0 "
                f"in every replay pass: {values}"
            )
            tolerance = _pass_tolerance(counter)
            assert _approx_equal(min(values), max(values), tolerance), (
                f"dispatch {dispatch_id} ({entry['kernel']}) counter {counter} varies across "
                f"replay passes (allowed spread {tolerance:.0%}): {values}"
            )


def test_each_pass_collects_distinct_batch(json_data, expected_passes, common_counters):
    # Metric 3: each pass collects exactly one unique (non-common) counter, and those per-pass
    # counters differ — proving real multi-pass batch rotation rather than one pass carrying
    # every unique counter while the rest collect none.
    table = _records_by_dispatch(_sdk(json_data))
    common = set(common_counters)
    for dispatch_id, entry in table.items():
        unique_by_pass = {}
        for pass_index, batch in entry["passes"].items():
            unique = {c for c in batch if c not in common}
            assert len(unique) == 1, (
                f"dispatch {dispatch_id} ({entry['kernel']}) replay_pass {pass_index} "
                f"expected exactly one non-common counter, got {sorted(unique)} "
                f"(full batch {sorted(batch)})"
            )
            unique_by_pass[pass_index] = next(iter(unique))
        assert len(set(unique_by_pass.values())) == expected_passes, (
            f"dispatch {dispatch_id} ({entry['kernel']}) expected {expected_passes} "
            f"distinct per-pass unique counters, got {unique_by_pass}"
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
