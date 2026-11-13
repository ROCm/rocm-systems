#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""rocprofv3 --collector writes what in-process rocprofv3 --hip-trace writes.

The two runs are separate runs of the same application, so pointer values, process and
thread ids, and times differ. Compared:
- the same output files for each traced process, and as many traced processes;
- the CSV header, and each row's Process_Id is its process;
- the agent information;
- the calls (rocpd): on each thread, the in-process calls are a contiguous run of the
  collector's calls, with the same functions and arguments (pointers masked). Extra calls
  are allowed before or after that run only: calls made before rocprofv3 wraps the HIP
  tables, and teardown after it finalizes, which the collector records and rocprofv3 does
  not;
- the JSON output's records: the same kinds, except HIP event and HIP graph records,
  which the collector does not produce yet.
"""

import json
import sys

import pytest

# Record kinds the collector does not produce yet (rocprofv3 --collector drops them with a
# warning when --hip-trace implies them).
NOT_PRODUCED = {"hip_event", "hip_graph"}


def _find_slice(inner, outer):
    """Index i such that outer[i:i + len(inner)] == inner, or None."""
    for i in range(len(outer) - len(inner) + 1):
        if outer[i : i + len(inner)] == inner:
            return i
    return None


def _pair_processes(in_process, collector):
    """Pair the processes of the two runs: by number of calls (the processes of one run
    are separate instances, so any pairing of equal ones is equivalent)."""

    def num_calls(process):
        return sum(len(v) for v in process.calls().values())

    return list(zip(sorted(in_process, key=num_calls), sorted(collector, key=num_calls)))


def test_same_files(in_process, collector):
    assert len(collector) == len(in_process), "number of traced processes"
    assert sorted(p.files for p in collector) == sorted(p.files for p in in_process)


def test_csv_header_and_process(in_process, collector):
    reference, _ = in_process[0].csv_rows("hip_api_trace.csv")
    for process in collector:
        header, rows = process.csv_rows("hip_api_trace.csv")
        assert header == reference
        assert rows, f"no calls for process {process.pid}"
        assert {r["Process_Id"] for r in rows} == {process.pid}
        ids = sorted(int(r["Correlation_Id"]) for r in rows)
        assert ids == list(range(1, len(ids) + 1)), "correlation ids 1..n per process"
        assert all(int(r["Start_Timestamp"]) <= int(r["End_Timestamp"]) for r in rows)


def test_agent_info(in_process, collector):
    reference = in_process[0].csv_rows("agent_info.csv")
    for process in collector:
        assert process.csv_rows("agent_info.csv") == reference


def test_calls_and_arguments(in_process, collector):
    for ref_process, process in _pair_processes(in_process, collector):
        ref_threads = ref_process.calls()
        threads = dict(process.calls())
        extra_calls = 0
        # Longest threads first, so a short thread does not take a long thread's match.
        for ref_tid, ref_calls in sorted(ref_threads.items(), key=lambda x: -len(x[1])):
            match = None
            for tid, calls in threads.items():
                start = _find_slice(ref_calls, calls)
                if start is not None:
                    match = (tid, start, len(calls))
                    break
            if match is None:
                names = [c[0] for c in ref_calls]
                pytest.fail(
                    f"process {process.pid}: no thread has the {len(ref_calls)} calls of "
                    f"in-process thread {ref_tid} in order with the same arguments "
                    f"(first calls {names[:5]})"
                )
            tid, start, count = match
            extra_calls += count - len(ref_calls)
            del threads[tid]
        # Threads the in-process run did not record at all: their calls are all before
        # rocprofv3 wrapped the tables or after it finalized.
        for calls in threads.values():
            extra_calls += len(calls)
        sys.stderr.write(
            f"process {process.pid}: {sum(len(v) for v in ref_threads.values())} in-process "
            f"calls matched; {extra_calls} more recorded by the collector\n"
        )


def test_json_records(in_process, collector):
    for ref_process, process in _pair_processes(in_process, collector):
        with open(ref_process.path("results.json")) as f:
            ref = json.load(f)["rocprofiler-sdk-tool"][0]["buffer_records"]
        with open(process.path("results.json")) as f:
            doc = json.load(f)["rocprofiler-sdk-tool"][0]["buffer_records"]
        ref_kinds = {k for k, v in ref.items() if v}
        kinds = {k for k, v in doc.items() if v}
        assert (
            kinds <= ref_kinds
        ), f"record kinds only in the collector's output: {kinds - ref_kinds}"
        assert ref_kinds - kinds <= NOT_PRODUCED
        assert len(doc["hip_api"]) >= len(ref["hip_api"])


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
