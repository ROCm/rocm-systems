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

# The range is three launches of rr_v3_step. Each --pmc group is one pass, so every
# launch inside the range must show passes 0..N-1 and a SQ_WAVES value on each pass.

import collections

import pytest

_PASS_KEYS = ("replay_pass", "n")
_KERNEL = "rr_v3_step"


def _sdk(json_data):
    assert "rocprofiler-sdk-tool" in json_data, "missing rocprofiler-sdk-tool in JSON"
    tool = json_data["rocprofiler-sdk-tool"]
    if isinstance(tool, list):
        assert tool, "empty rocprofiler-sdk-tool array"
        tool = tool[0]
    return tool


def _pass_index(record):
    for key in _PASS_KEYS:
        if key in record and record[key] is not None:
            return int(record[key])
    raise AssertionError(f"no replay-pass field in {list(record.keys())}")


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


def _range_dispatches(json_data):
    sdk = _sdk(json_data)
    names = _kernel_id_to_name(sdk)
    counters = _counter_id_to_name(sdk)
    records = sdk.get("callback_records", {}).get("counter_collection")
    assert records, "no counter_collection records"
    table = {}
    for record in records:
        info = record["dispatch_data"]["dispatch_info"]
        name = names.get(int(info["kernel_id"]), "")
        if _KERNEL not in name:
            continue
        did = int(info["dispatch_id"])
        entry = table.setdefault(did, {"passes": set(), "counters": collections.defaultdict(set)})
        index = _pass_index(record)
        entry["passes"].add(index)
        for sub in record.get("records", []):
            counter = counters.get(int(sub["counter_id"]["handle"]))
            if counter:
                entry["counters"][index].add(counter)
    return table


def test_range_replays_each_dispatch(json_data, expected_passes, expected_dispatches):
    table = _range_dispatches(json_data)
    assert len(table) == expected_dispatches, (
        f"expected {expected_dispatches} {_KERNEL} dispatches, found {len(table)}"
    )
    want = set(range(expected_passes))
    for did, entry in table.items():
        assert entry["passes"] == want, (
            f"dispatch {did} passes {sorted(entry['passes'])} != {sorted(want)}"
        )
        for index in want:
            assert "SQ_WAVES" in entry["counters"][index], (
                f"dispatch {did} pass {index} has no SQ_WAVES ({sorted(entry['counters'][index])})"
            )
