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

"""Synthetic rocprofv3 kernel-replay results, plus mutations that each represent one
replay failure mode.

Used by test_validate_selftest.py to check that validate.py actually rejects broken
replays. Needs no GPU: the JSON is constructed, not measured.
"""

import copy

PASSES = 5
COMMON_COUNTERS = ["SQ_WAVES", "SQ_INSTS_VALU"]

# The counter unique to each --pmc group, in group order. Must match REPLAY_PASS_GROUPS
# in CMakeLists.txt.
PASS_GROUPS = [
    "GRBM_COUNT",
    "GRBM_GUI_ACTIVE",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
]

ALL_COUNTERS = COMMON_COUNTERS + PASS_GROUPS

# Launch config of the kernel-replay app's three kernels, matching EXPECTED_DIMS.
KERNELS = {
    "vecAdd": (1024 * 1024, 1024),
    "saxpy": (512 * 512, 512),
    "vecScale": (256 * 256, 256),
}

# Distinct per-kernel values so the kernels are separable by the shared counters.
BASE = {
    "vecAdd": {"SQ_WAVES": 16384.0, "SQ_INSTS_VALU": 900000.0},
    "saxpy": {"SQ_WAVES": 4096.0, "SQ_INSTS_VALU": 220000.0},
    "vecScale": {"SQ_WAVES": 1024.0, "SQ_INSTS_VALU": 55000.0},
}

_CID = {name: idx + 1 for idx, name in enumerate(ALL_COUNTERS)}
_KID = {name: idx + 100 for idx, name in enumerate(KERNELS)}


def _record(dispatch_id, kernel, pass_idx, counters):
    grid, workgroup = KERNELS[kernel]
    return {
        "replay_pass": pass_idx,
        "dispatch_data": {
            "dispatch_info": {
                "dispatch_id": dispatch_id,
                "kernel_id": _KID[kernel],
                "grid_size": {"x": grid, "y": 1, "z": 1},
                "workgroup_size": {"x": workgroup, "y": 1, "z": 1},
            }
        },
        "records": [
            {"counter_id": {"handle": _CID[name]}, "value": value}
            for name, value in counters.items()
        ],
    }


def golden():
    """A correct PASSES-way replay of one dispatch per kernel."""
    records = []
    for dispatch_idx, kernel in enumerate(KERNELS):
        for pass_idx in range(PASSES):
            counters = dict(BASE[kernel])
            counters[PASS_GROUPS[pass_idx]] = 1000.0 * (pass_idx + 1) + dispatch_idx
            records.append(_record(dispatch_idx + 1, kernel, pass_idx, counters))
    return {
        "rocprofiler-sdk-tool": [
            {
                "counters": [
                    {"id": {"handle": handle}, "name": name}
                    for name, handle in _CID.items()
                ],
                "kernel_symbols": [
                    {"kernel_id": kid, "formatted_kernel_name": name}
                    for name, kid in _KID.items()
                ],
                "callback_records": {"counter_collection": records},
            }
        ]
    }


def _records(doc):
    return doc["rocprofiler-sdk-tool"][0]["callback_records"]["counter_collection"]


def _replace_records(doc, records):
    doc["rocprofiler-sdk-tool"][0]["callback_records"]["counter_collection"] = records


def _set(record, name, value):
    for sub in record["records"]:
        if sub["counter_id"]["handle"] == _CID[name]:
            sub["value"] = value
            return
    record["records"].append({"counter_id": {"handle": _CID[name]}, "value": value})


def _drop(record, name):
    record["records"] = [
        sub for sub in record["records"] if sub["counter_id"]["handle"] != _CID[name]
    ]


def _dispatch_id(record):
    return record["dispatch_data"]["dispatch_info"]["dispatch_id"]


def _kernel_id(record):
    return record["dispatch_data"]["dispatch_info"]["kernel_id"]


# --- failure modes -----------------------------------------------------------------


def loop_exited_early(doc):
    _replace_records(
        doc,
        [
            rec
            for rec in _records(doc)
            if not (_dispatch_id(rec) == 1 and rec["replay_pass"] == PASSES - 1)
        ],
    )


def same_group_every_pass(doc):
    for rec in _records(doc):
        for name in PASS_GROUPS:
            _drop(rec, name)
        _set(rec, PASS_GROUPS[0], 1234.0)


def restore_broken(doc):
    for rec in _records(doc):
        if _kernel_id(rec) == _KID["vecAdd"] and rec["replay_pass"] == 3:
            _set(rec, "SQ_WAVES", BASE["vecAdd"]["SQ_WAVES"] * 2.0)


def one_pass_too_many(doc):
    extra = copy.deepcopy(
        next(
            rec
            for rec in _records(doc)
            if _dispatch_id(rec) == 1 and rec["replay_pass"] == PASSES - 1
        )
    )
    extra["replay_pass"] = PASSES
    _records(doc).append(extra)


def pass_index_one_based(doc):
    for rec in _records(doc):
        rec["replay_pass"] += 1


def pass_groups_swapped(doc):
    """Passes 1 and 2 collect each other's group. Union of counters is unchanged."""
    for rec in _records(doc):
        pass_idx = rec["replay_pass"]
        if pass_idx not in (1, 2):
            continue
        other = 2 if pass_idx == 1 else 1
        value = next(
            (
                sub["value"]
                for sub in rec["records"]
                if sub["counter_id"]["handle"] == _CID[PASS_GROUPS[pass_idx]]
            ),
            1.0,
        )
        _drop(rec, PASS_GROUPS[pass_idx])
        _set(rec, PASS_GROUPS[other], value)


def pass_groups_skewed(doc):
    """Pass 2 collects two groups, pass 1 collects none. Union is still complete."""
    for rec in _records(doc):
        if rec["replay_pass"] == 1:
            _drop(rec, PASS_GROUPS[1])
        elif rec["replay_pass"] == 2:
            _set(rec, PASS_GROUPS[1], 4242.0)


def dispatch_ids_collapsed(doc):
    for rec in _records(doc):
        rec["dispatch_data"]["dispatch_info"]["dispatch_id"] = 1


def counter_values_all_zero(doc):
    for rec in _records(doc):
        for sub in rec["records"]:
            sub["value"] = 0.0


def pass_index_never_incremented(doc):
    for rec in _records(doc):
        rec["replay_pass"] = 0


def only_first_dispatch_replayed(doc):
    _replace_records(
        doc,
        [
            rec
            for rec in _records(doc)
            if _dispatch_id(rec) == 1 or rec["replay_pass"] == 0
        ],
    )


def launch_dims_wrong(doc):
    for rec in _records(doc):
        if _kernel_id(rec) == _KID["saxpy"]:
            rec["dispatch_data"]["dispatch_info"]["grid_size"]["x"] = 12345


def kernels_indistinguishable(doc):
    for rec in _records(doc):
        _set(rec, "SQ_WAVES", 1000.0)
        _set(rec, "SQ_INSTS_VALU", 2000.0)


def one_counter_never_collected(doc):
    for rec in _records(doc):
        _drop(rec, PASS_GROUPS[-1])


def shared_counter_missing_in_one_pass(doc):
    for rec in _records(doc):
        if rec["replay_pass"] == 2:
            _drop(rec, "SQ_INSTS_VALU")


# id -> (description, mutation)
FAILURE_MODES = {
    "loop_exited_early": ("replay loop stopped one pass short", loop_exited_early),
    "same_group_every_pass": (
        "pass->group mapping stuck, one batch repeated",
        same_group_every_pass,
    ),
    "restore_broken": (
        "restore failed, so a pass saw different inputs",
        restore_broken,
    ),
    "one_pass_too_many": ("off-by-one in the loop bound", one_pass_too_many),
    "pass_index_one_based": ("pass index reported 1..N", pass_index_one_based),
    "pass_groups_swapped": (
        "two passes collected each other's group",
        pass_groups_swapped,
    ),
    "pass_groups_skewed": (
        "one pass collected two groups, another none",
        pass_groups_skewed,
    ),
    "dispatch_ids_collapsed": (
        "dispatch_id shared across distinct dispatches",
        dispatch_ids_collapsed,
    ),
    "counter_values_all_zero": (
        "counter readout returned zero everywhere",
        counter_values_all_zero,
    ),
    "pass_index_never_incremented": (
        "replay_pass never advanced",
        pass_index_never_incremented,
    ),
    "only_first_dispatch_replayed": (
        "replay applied to one dispatch only",
        only_first_dispatch_replayed,
    ),
    "launch_dims_wrong": ("dispatch_info launch dims corrupted", launch_dims_wrong),
    "kernels_indistinguishable": (
        "all kernels report identical shared counters",
        kernels_indistinguishable,
    ),
    "one_counter_never_collected": (
        "a requested counter is absent from every pass",
        one_counter_never_collected,
    ),
    "shared_counter_missing_in_one_pass": (
        "a shared counter is absent from one pass",
        shared_counter_missing_in_one_pass,
    ),
}


def broken(mode):
    doc = golden()
    FAILURE_MODES[mode][1](doc)
    return doc
