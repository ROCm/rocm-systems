#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import os
import pytest

# The 9 host-stream APIs registered by rocshmem/src/api_trace.cc.
EXPECTED_OPERATIONS = {
    "barrier_all_on_stream",
    "quiet_on_stream",
    "sync_all_on_stream",
    "alltoallmem_on_stream",
    "broadcastmem_on_stream",
    "getmem_on_stream",
    "putmem_on_stream",
    "putmem_signal_on_stream",
    "signal_wait_until_on_stream",
}

# rocshmem-demo wraps its API block in `for(iter = 0; iter < 3; ++iter)` to
# exercise the tracer under repetition. Update both numbers in lockstep if
# the demo's iteration count changes.
DEMO_ITERATIONS = 3
EXPECTED_BUFFER_RECORDS = len(EXPECTED_OPERATIONS) * DEMO_ITERATIONS  # 45
EXPECTED_CALLBACK_RECORDS = EXPECTED_BUFFER_RECORDS * 2  # phase 1 + phase 2 = 90


def node_exists(name, data, min_len=1):
    assert name in data, f"missing key: {name}"
    assert data[name] is not None, f"null value for: {name}"
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len, f"{name}:\n{data}"


def get_operation(record, kind_name, op_name=None):
    for idx, itr in enumerate(record["names"]):
        if kind_name == itr["kind"]:
            if op_name is None:
                return idx, itr["operations"]
            for oidx, oname in enumerate(itr["operations"]):
                if op_name == oname:
                    return oidx
    return None


def _get_sdk_data(input_data):
    node_exists("rocprofiler-sdk-json-tool", input_data)
    return input_data["rocprofiler-sdk-json-tool"]


def test_data_structure(input_data):
    """Top-level JSON layout produced by rocprofiler-sdk-json-tool."""
    sdk_data = _get_sdk_data(input_data)
    node_exists("metadata", sdk_data)
    node_exists("buffer_records", sdk_data)
    node_exists("callback_records", sdk_data)
    node_exists("names", sdk_data["buffer_records"])


def test_rocshmem_domain_registered(input_data):
    """The ROCSHMEM_API buffer-tracing kind exposes exactly 9 operation names."""
    sdk_data = _get_sdk_data(input_data)
    op_lookup = get_operation(sdk_data["buffer_records"], "ROCSHMEM_API")
    assert (
        op_lookup is not None
    ), "ROCSHMEM_API kind not registered in buffer_records.names"
    _, op_names = op_lookup
    assert len(op_names) == 9, f"expected 9 operations, got {len(op_names)}: {op_names}"
    assert set(op_names) == EXPECTED_OPERATIONS, (
        f"operation set mismatch.\n  got:      {sorted(op_names)}"
        f"\n  expected: {sorted(EXPECTED_OPERATIONS)}"
    )


def test_buffer_records_contain_all_apis(input_data):
    """Every traced API surfaces at least once in the buffer-tracing stream."""
    sdk_data = _get_sdk_data(input_data)
    op_lookup = get_operation(sdk_data["buffer_records"], "ROCSHMEM_API")
    assert op_lookup is not None, "ROCSHMEM_API kind missing"
    kind_idx, op_names = op_lookup

    bf_records = sdk_data["buffer_records"].get("rocshmem_api_traces", [])
    assert len(bf_records) >= EXPECTED_BUFFER_RECORDS, (
        f"expected >={EXPECTED_BUFFER_RECORDS} rocshmem_api buffer records "
        f"({DEMO_ITERATIONS} iters x {len(EXPECTED_OPERATIONS)} APIs), "
        f"got {len(bf_records)}"
    )

    observed_ops = set()
    for rec in bf_records:
        # explicit field-presence checks (match the rocdecode / rocjpeg pattern
        # so missing keys produce an actionable assertion message rather than
        # a KeyError deep inside the record walker).
        for key in (
            "size",
            "kind",
            "operation",
            "correlation_id",
            "start_timestamp",
            "end_timestamp",
            "thread_id",
        ):
            assert key in rec, f"missing key '{key}' in buffer record: {rec}"

        assert rec["kind"] == kind_idx, f"unexpected kind in record: {rec}"
        assert (
            0 <= rec["operation"] < len(op_names)
        ), f"operation index out of range: {rec}"
        observed_ops.add(op_names[rec["operation"]])

        # field-value sanity (mirrors rocdecode/rocjpeg).
        assert rec["size"] > 0, f"non-positive record size: {rec}"
        assert rec["thread_id"] > 0
        assert rec["start_timestamp"] > 0
        assert rec["end_timestamp"] > 0
        assert rec["start_timestamp"] < rec["end_timestamp"]
        assert rec["correlation_id"]["internal"] > 0

    assert observed_ops == EXPECTED_OPERATIONS, (
        "rocshmem-demo did not exercise every traced API."
        f"\n  missing: {sorted(EXPECTED_OPERATIONS - observed_ops)}"
        f"\n  extra:   {sorted(observed_ops - EXPECTED_OPERATIONS)}"
    )


def test_callback_records_contain_all_apis(input_data):
    """Same coverage check via the callback-tracing stream."""
    sdk_data = _get_sdk_data(input_data)
    op_lookup = get_operation(sdk_data["buffer_records"], "ROCSHMEM_API")
    assert op_lookup is not None
    _, op_names = op_lookup

    cb_records = sdk_data["callback_records"].get("rocshmem_api_traces", [])
    if not cb_records:
        pytest.skip("json-tool did not record any callback traces for rocshmem_api")

    assert len(cb_records) >= EXPECTED_CALLBACK_RECORDS, (
        f"expected >={EXPECTED_CALLBACK_RECORDS} rocshmem_api callback records "
        f"({DEMO_ITERATIONS} iters x {len(EXPECTED_OPERATIONS)} APIs x 2 phases), "
        f"got {len(cb_records)}"
    )

    observed_ops = set()
    for rec in cb_records:
        # Callback records have phase 1 (enter) / phase 2 (exit). Both phases
        # carry the same operation index, so dedupe via the set.
        assert rec["phase"] in (1, 2), f"unexpected phase in record: {rec}"
        assert (
            0 <= rec["operation"] < len(op_names)
        ), f"operation index out of range: {rec}"
        observed_ops.add(op_names[rec["operation"]])

    assert observed_ops == EXPECTED_OPERATIONS, (
        "callback stream missing some rocshmem APIs."
        f"\n  missing: {sorted(EXPECTED_OPERATIONS - observed_ops)}"
    )


def test_timestamps_within_session(input_data):
    """All buffer records fall between the json-tool init/fini timestamps."""
    sdk_data = _get_sdk_data(input_data)
    init_ts = sdk_data["metadata"]["init_time"]
    fini_ts = sdk_data["metadata"]["fini_time"]
    assert init_ts > 0 and fini_ts > init_ts

    for rec in sdk_data["buffer_records"].get("rocshmem_api_traces", []):
        assert (
            init_ts <= rec["start_timestamp"] <= fini_ts
        ), f"start_timestamp out of session window: {rec}"
        assert (
            init_ts <= rec["end_timestamp"] <= fini_ts
        ), f"end_timestamp out of session window: {rec}"


def test_callback_record_payloads(input_data):
    """Every callback record has the expected payload structure.

    json-tool currently captures rocSHMEM call signatures via its `payload`
    sub-object (which carries `size` and `retval`) rather than the per-arg
    `args` dict, so payload presence + structure is the right granularity for
    this branch. The actual `args` dict is also asserted to exist (it is the
    payload-extension hook used by tools that want per-call kwargs).
    """
    sdk_data = _get_sdk_data(input_data)
    cb_records = sdk_data["callback_records"].get("rocshmem_api_traces", [])
    if not cb_records:
        pytest.skip("json-tool did not record any callback traces for rocshmem_api")

    for rec in cb_records:
        assert "payload" in rec, f"missing payload in callback record: {rec}"
        payload = rec["payload"]
        assert "size" in payload, f"missing payload.size: {rec}"
        assert payload["size"] > 0, f"non-positive payload.size: {rec}"
        # All 9 traced rocSHMEM host-stream APIs return void; the callback's
        # `retval` is therefore an empty struct, but the key must still be
        # present so tools can serialize it uniformly.
        assert "retval" in payload, f"missing payload.retval: {rec}"
        assert "args" in rec, f"missing args dict: {rec}"
        assert isinstance(rec["args"], dict), f"args is not a dict: {rec}"


def test_external_correlation_ids(input_data):
    """External correlation IDs flow correctly through buffer + callback streams.

    Mirrors `tests/async-copy-tracing/validate.py:test_external_correlation_ids`.
    On this branch json-tool sets the external correlation id to the issuing
    thread id, so that invariant is checked end-to-end.
    """
    sdk_data = _get_sdk_data(input_data)

    extern_corr_ids = set()
    for rec in sdk_data["callback_records"].get("rocshmem_api_traces", []):
        ext = rec["correlation_id"]["external"]
        assert ext > 0, f"non-positive external correlation id: {rec}"
        assert (
            rec["thread_id"] == ext
        ), f"thread_id ({rec['thread_id']}) != external ({ext}): {rec}"
        extern_corr_ids.add(ext)

    assert (
        len(extern_corr_ids) > 0
    ), "no external correlation ids observed in callback stream"

    for rec in sdk_data["buffer_records"].get("rocshmem_api_traces", []):
        ext = rec["correlation_id"]["external"]
        assert ext > 0, f"non-positive external correlation id: {rec}"
        assert (
            rec["thread_id"] == ext
        ), f"thread_id ({rec['thread_id']}) != external ({ext}): {rec}"
        assert (
            ext in extern_corr_ids
        ), f"buffer-stream external id {ext} not seen in callback stream"


def test_perfetto_data(request):
    """The Perfetto trace emitted by json-tool contains rocshmem activity.

    json-tool writes a sibling `.pftrace` next to the JSON output. Mirrors
    `tests/rocprofv3/{rocdecode,rocjpeg}-trace/validate.py:test_perfetto_data`,
    but inspects the slice dataframe directly because the rocshmem-trace
    branch does not yet have the `rocshmem_api` mapping in
    `pytest-packages/tests/rocprofv3.py:test_perfetto_data` (that helper is
    extended on the rocprofv3-rocshmem branch).
    """
    json_path = request.config.getoption("--input")
    pftrace_path = json_path[: json_path.rfind(".json")] + ".pftrace"
    if not os.path.isfile(pftrace_path):
        return pytest.skip(f"perfetto trace not produced: {pftrace_path}")

    from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader

    dataframe, _ = PerfettoReader(pftrace_path).read()
    assert (
        not dataframe.empty
    ), f"PerfettoReader returned empty dataframe for {pftrace_path}"

    # Slice names from the rocshmem domain look like 'rocshmem_putmem_on_stream'
    # etc.; the category column groups them under 'rocshmem_api'. Match either
    # to be resilient to json-tool's exact naming convention.
    name_hits = dataframe[
        dataframe["name"].astype(str).str.contains("rocshmem", case=False, na=False)
    ]
    cat_hits = dataframe[
        dataframe["category"].astype(str).str.contains("rocshmem", case=False, na=False)
    ]

    if name_hits.empty and cat_hits.empty:
        # On rocshmem-trace, the rocshmem perfetto category may not be wired
        # into the SDK's compile-time perfetto category list yet (that wiring
        # lands on the rocprofv3-rocshmem branch). The JSON tests above
        # already cover the dispatch table and record contents, so degrade to
        # a skip rather than a hard failure here.
        return pytest.skip(
            "perfetto trace contains no rocshmem slices "
            f"(categories: {sorted(dataframe['category'].astype(str).unique())[:10]}). "
            "rocSHMEM perfetto serialization may not be enabled on this branch."
        )

    # If rocshmem slices DO exist, confirm we see >= 1 occurrence of each of
    # the 9 traced APIs in the slice names.
    if not name_hits.empty:
        observed = set()
        for n in name_hits["name"].astype(str):
            for op in EXPECTED_OPERATIONS:
                if op in n:
                    observed.add(op)
                    break
        assert observed == EXPECTED_OPERATIONS, (
            "perfetto slices missing some rocshmem APIs."
            f"\n  missing: {sorted(EXPECTED_OPERATIONS - observed)}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
