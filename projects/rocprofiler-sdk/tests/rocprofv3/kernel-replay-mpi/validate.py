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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Validates kernel replay under MPI (see CMakeLists.txt). For every profiled rank:
#   - the app's execution counters (kept in unrestored pinned host memory) show that the
#     replayed compute kernel really ran G times for each launch in --kernel-iteration-range and
#     once for every other launch, and that the excluded collective stand-in ran exactly once per
#     launch -- replaying it would have run it G times;
#   - the profiler output holds exactly one logical dispatch per launch in the range, each with
#     replay passes [0, G), and no record at all for the excluded kernel.
# Ranks that were not profiled must have replayed nothing.

import glob
import json
import os

import pytest

COMPUTE_KERNEL = "compute_kernel"
EXCLUDED_KERNEL = "peer_visible_kernel"


def _read_counts(output_dir, rank):
    path = os.path.join(output_dir, f"counts.rank{rank}.txt")
    assert os.path.isfile(path), f"rank {rank} wrote no execution counts: {path}"
    counts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key, _, value = line.strip().partition("=")
            counts[key] = int(value)
    return counts


def _sdk(json_data):
    tool = json_data["rocprofiler-sdk-tool"]
    return tool[0] if isinstance(tool, list) else tool


def _load_rank_json(output_dir, rank):
    pattern = os.path.join(output_dir, f"rank.{rank}", "**", "out_results.json")
    paths = glob.glob(pattern, recursive=True)
    assert len(paths) == 1, f"expected one results file for rank {rank}, got {paths}"
    with open(paths[0], "r", encoding="utf-8") as f:
        return _sdk(json.load(f))


def _kernel_names(sdk):
    symbols = sdk.get("kernel_symbols")
    if not symbols:
        symbols = sdk.get("callback_records", {}).get("kernel_symbols")
    assert symbols, "missing kernel_symbols"
    entries = symbols.values() if isinstance(symbols, dict) else symbols
    names = {}
    for sym in entries:
        name = (
            sym.get("formatted_kernel_name")
            or sym.get("demangled_kernel_name")
            or sym.get("kernel_name")
        )
        if sym.get("kernel_id") is not None and name:
            names[int(sym["kernel_id"])] = name
    return names


def _passes_by_dispatch(sdk):
    """(kernel name, dispatch_id) -> sorted list of replay pass indices."""
    names = _kernel_names(sdk)
    records = sdk.get("callback_records", {}).get("counter_collection", [])
    table = {}
    for rec in records:
        info = rec["dispatch_data"]["dispatch_info"]
        key = (names.get(int(info["kernel_id"]), ""), int(info["dispatch_id"]))
        table.setdefault(key, []).append(int(rec.get("replay_pass", rec.get("n"))))
    return {key: sorted(passes) for key, passes in table.items()}


@pytest.fixture
def all_ranks(mpi_ranks):
    return list(range(mpi_ranks))


def test_execution_counts_match_the_filters(
    output_dir, all_ranks, profiled_ranks, iters, groups, range_first, range_last
):
    replayed = range_last - range_first + 1
    for rank in all_ranks:
        counts = _read_counts(output_dir, rank)

        # a collective-like kernel is never replayed, on any rank
        assert counts["peer_execs"] == iters, (
            f"rank {rank}: excluded kernel executed {counts['peer_execs']} times for "
            f"{iters} launches -- it was replayed"
        )

        if rank in profiled_ranks:
            want = iters + replayed * (groups - 1)
        else:
            want = iters
        assert counts["compute_execs"] == want, (
            f"rank {rank}: compute kernel executed {counts['compute_execs']} times, "
            f"expected {want} ({'profiled' if rank in profiled_ranks else 'unprofiled'})"
        )


def test_profiled_ranks_report_each_replayed_dispatch_once(
    output_dir, profiled_ranks, groups, range_first, range_last
):
    replayed = range_last - range_first + 1
    for rank in profiled_ranks:
        passes = _passes_by_dispatch(_load_rank_json(output_dir, rank))

        compute = {k: v for k, v in passes.items() if COMPUTE_KERNEL in k[0]}
        assert (
            len(compute) == replayed
        ), f"rank {rank}: {len(compute)} logical compute dispatches, expected {replayed}"
        for key, got in compute.items():
            assert got == list(
                range(groups)
            ), f"rank {rank} dispatch {key}: replay passes {got}, expected 0..{groups - 1}"

        excluded = [k for k in passes if EXCLUDED_KERNEL in k[0]]
        assert not excluded, f"rank {rank}: excluded kernel has counter records {excluded}"


def test_unprofiled_ranks_produce_no_profiler_output(
    output_dir, all_ranks, profiled_ranks
):
    for rank in all_ranks:
        if rank in profiled_ranks:
            continue
        pattern = os.path.join(output_dir, f"rank.{rank}", "**", "*_results.json")
        assert not glob.glob(pattern, recursive=True), f"rank {rank} was profiled"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-x"] + sys.argv[1:]))
