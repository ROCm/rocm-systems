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


def test_size_entries(input_data):
    """Every ``size`` field across the JSON is > 0 (except size-named args).

    Verbatim equivalent of ``tests/rocdecode/validate.py:test_size_entries``
    and ``tests/rocjpeg/validate.py:test_size_entries``. Recursively walks
    the entire input dict/list tree and asserts every dict carrying a
    ``size`` key has a positive value. Skips the assertion when ``size``
    appears inside a function ``args`` block: user-provided argument names
    can legitimately collide with the field name and may be a string or a
    non-positive integer.
    """

    def check_size(data, bt):
        if "size" in data.keys():
            if isinstance(data["size"], str) and bt.endswith('["args"]'):
                pass
            else:
                assert data["size"] > 0, f"origin: {bt}"

    def iterate_data(data, bt):
        if isinstance(data, (list, tuple)):
            for i, itr in enumerate(data):
                if isinstance(itr, dict):
                    check_size(itr, f"{bt}[{i}]")
                iterate_data(itr, f"{bt}[{i}]")
        elif isinstance(data, dict):
            check_size(data, f"{bt}")
            for key, itr in data.items():
                iterate_data(itr, f'{bt}["{key}"]')

    iterate_data(input_data, "input_data")


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


def test_rocshmem_ext_domain_registered(input_data):
    """The ROCSHMEM_API_EXT buffer-tracing kind exposes the same 9 operation names."""
    sdk_data = _get_sdk_data(input_data)
    op_lookup = get_operation(sdk_data["buffer_records"], "ROCSHMEM_API_EXT")
    assert (
        op_lookup is not None
    ), "ROCSHMEM_API_EXT kind not registered in buffer_records.names"
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
    if not bf_records:
        pytest.skip("rocshmem tracing unavailable (no buffer records captured)")
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


def test_buffer_ext_records_contain_args_and_retval(input_data):
    """EXT buffer records carry the same coverage as the basic buffer stream,
    plus an ``args`` payload and a ``retval`` field."""
    sdk_data = _get_sdk_data(input_data)
    op_lookup = get_operation(sdk_data["buffer_records"], "ROCSHMEM_API_EXT")
    assert op_lookup is not None, "ROCSHMEM_API_EXT kind missing"
    kind_idx, op_names = op_lookup

    ext_records = sdk_data["buffer_records"].get("rocshmem_api_ext_traces", [])
    if not ext_records:
        pytest.skip("rocshmem ext tracing unavailable (no buffer ext records captured)")

    assert len(ext_records) >= EXPECTED_BUFFER_RECORDS, (
        f"expected >={EXPECTED_BUFFER_RECORDS} rocshmem_api_ext buffer records "
        f"({DEMO_ITERATIONS} iters x {len(EXPECTED_OPERATIONS)} APIs), "
        f"got {len(ext_records)}"
    )

    observed_ops = set()
    for rec in ext_records:
        for key in (
            "size",
            "kind",
            "operation",
            "correlation_id",
            "start_timestamp",
            "end_timestamp",
            "thread_id",
            "args",
            "retval",
        ):
            assert key in rec, f"missing key '{key}' in ext buffer record: {rec}"

        assert rec["kind"] == kind_idx, f"unexpected kind in ext record: {rec}"
        assert (
            0 <= rec["operation"] < len(op_names)
        ), f"operation index out of range: {rec}"
        observed_ops.add(op_names[rec["operation"]])

        assert rec["size"] > 0
        assert rec["thread_id"] > 0
        assert rec["start_timestamp"] > 0
        assert rec["end_timestamp"] > 0
        assert rec["start_timestamp"] < rec["end_timestamp"]
        assert rec["correlation_id"]["internal"] > 0

        # args serializes as a list of {type, name, value} dicts (one per
        # parameter). Every host-stream rocSHMEM API takes at least one
        # argument, so the list must be non-empty and well-formed.
        assert isinstance(rec["args"], list), f"args is not a list: {rec}"
        assert len(rec["args"]) > 0, f"empty args in ext record: {rec}"
        for arg in rec["args"]:
            assert isinstance(arg, dict), f"arg is not a dict: {rec}"
            for key in ("type", "name", "value"):
                assert key in arg, f"missing arg.{key}: {rec}"
                assert isinstance(arg[key], str), f"arg.{key} is not a string: {rec}"

    assert observed_ops == EXPECTED_OPERATIONS, (
        "rocshmem-demo did not exercise every traced API in EXT stream."
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


def test_timestamps(input_data):
    """Callback phase-1/phase-2 ordering and buffer start <= end.

    Mirrors ``tests/rocdecode/validate.py:test_timestamps`` and
    ``tests/rocjpeg/validate.py:test_timestamps``. For every rocshmem
    callback record, the ``phase 1`` (enter) timestamp must be <= the
    matching ``phase 2`` (exit) timestamp identified by correlation id.
    For every buffer record, ``start_timestamp <= end_timestamp``.

    Complements ``test_timestamps_within_session`` which checks records
    fall inside the json-tool init/fini window. Both checks are kept
    because they protect against orthogonal bugs (relative ordering vs.
    absolute window placement).
    """
    sdk_data = _get_sdk_data(input_data)

    cb_start = {}
    for rec in sdk_data["callback_records"].get("rocshmem_api_traces", []):
        cid = rec["correlation_id"]["internal"]
        phase = rec["phase"]
        if phase == 1:
            cb_start[cid] = rec["timestamp"]
        elif phase == 2:
            assert cid in cb_start, f"phase 2 without matching phase 1: {rec}"
            assert cb_start[cid] <= rec["timestamp"], (
                f"phase 2 timestamp ({rec['timestamp']}) precedes "
                f"phase 1 ({cb_start[cid]}) for corr_id {cid}: {rec}"
            )
        else:
            assert phase in (1, 2), f"unexpected phase {phase}: {rec}"

    for rec in sdk_data["buffer_records"].get("rocshmem_api_traces", []):
        assert (
            rec["start_timestamp"] <= rec["end_timestamp"]
        ), f"buffer record start > end: {rec}"


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

    Mirrors `tests/async-copy-tracing/validate.py:test_external_correlation_ids`
    and ``tests/rocjpeg/validate.py:test_external_correlation_ids``. On this
    branch json-tool sets the external correlation id to the issuing thread id,
    so that invariant is checked end-to-end.

    Skip only when BOTH the callback and buffer streams are empty (i.e.
    rocSHMEM tracing isn't supported at all on this host). If either stream
    has records, run the assertions: the "buffer populated, callback empty"
    partial state would otherwise mask a real bug where the SDK loses
    callback delivery while still emitting buffer records.
    """
    sdk_data = _get_sdk_data(input_data)

    cb_records = sdk_data["callback_records"].get("rocshmem_api_traces", [])
    bf_records = sdk_data["buffer_records"].get("rocshmem_api_traces", [])
    if not cb_records and not bf_records:
        pytest.skip("rocshmem tracing unavailable (no records captured)")

    extern_corr_ids = set()
    for rec in cb_records:
        ext = rec["correlation_id"]["external"]
        assert ext > 0, f"non-positive external correlation id: {rec}"
        assert (
            rec["thread_id"] == ext
        ), f"thread_id ({rec['thread_id']}) != external ({ext}): {rec}"
        extern_corr_ids.add(ext)

    # If buffer records exist, callbacks should have produced ids too. Failing
    # here surfaces the partial-state SDK bug that the old over-eager skip hid.
    if bf_records:
        assert (
            len(extern_corr_ids) > 0
        ), "buffer-stream rocshmem records exist but callback stream produced no external ids"

    for rec in bf_records:
        ext = rec["correlation_id"]["external"]
        assert ext > 0, f"non-positive external correlation id: {rec}"
        assert (
            rec["thread_id"] == ext
        ), f"thread_id ({rec['thread_id']}) != external ({ext}): {rec}"
        assert (
            ext in extern_corr_ids
        ), f"buffer-stream external id {ext} not seen in callback stream"


def test_internal_correlation_ids(input_data):
    """Internal correlation ids are unique per stream and densely allocated.

    Mirrors ``tests/rocdecode/validate.py:test_internal_correlation_ids`` and
    ``tests/rocjpeg/validate.py:test_internal_correlation_ids``. The same id
    appears at most once in each *buffer* stream (one API call -> one record),
    but it surfaces twice in the *callback* stream (phase 1 + phase 2), so the
    union has duplicates by construction. After dedup, the unique-id set must
    densely cover 1..N within the process.

    No skip-on-empty: even when rocSHMEM tracing isn't supported on the host
    (rocshmem_api_traces empty), this test still validates the HSA + HIP
    correlation-id invariants. Matches rocdecode/rocjpeg semantics.
    """
    sdk_data = _get_sdk_data(input_data)

    api_corr_ids = []
    for titr in ("hsa_api_traces", "hip_api_traces", "rocshmem_api_traces"):
        for itr in sdk_data["callback_records"].get(titr, []):
            api_corr_ids.append(itr["correlation_id"]["internal"])
        for itr in sdk_data["buffer_records"].get(titr, []):
            api_corr_ids.append(itr["correlation_id"]["internal"])

    api_corr_ids_sorted = sorted(api_corr_ids)
    api_corr_ids_unique = set(api_corr_ids)

    # Memory-allocation records reuse the corr_id of the triggering API, so
    # every alloc corr_id must already be in the api set.
    for itr in sdk_data["buffer_records"].get("memory_allocations", []):
        assert (
            itr["correlation_id"]["internal"] in api_corr_ids_unique
        ), f"memory_allocation corr_id not seen in any api stream: {itr}"

    # Buffer + callback duplication guarantees len(all) > len(unique).
    assert len(api_corr_ids) != len(
        api_corr_ids_unique
    ), "expected duplicate correlation ids (callback + buffer share ids)"

    # Per-process corr_ids should be densely allocated 1..N.
    assert max(api_corr_ids_sorted) == len(api_corr_ids_unique), (
        f"max corr_id ({max(api_corr_ids_sorted)}) != "
        f"unique corr_id count ({len(api_corr_ids_unique)}); allocator gap?"
    )


def test_retired_correlation_ids(input_data):
    """Every API correlation id is eventually retired with retired_ts > end_ts.

    Mirrors ``tests/rocdecode/validate.py:test_retired_correlation_ids``. The
    SDK populates the ``retired_correlation_ids`` buffer stream once the
    pipeline drains; every API + memory-allocation corr_id must appear there
    with a timestamp strictly greater than the originating record's
    ``end_timestamp``.

    No skip-on-empty: even when rocSHMEM tracing isn't supported on the host
    (rocshmem_api_traces empty), this test still validates the HSA + HIP
    retired-id invariants. Matches rocdecode/rocjpeg semantics.
    """
    sdk_data = _get_sdk_data(input_data)

    def _sort_dict(inp):
        return dict(sorted(inp.items()))

    api_corr_ids = {}
    for titr in ("hsa_api_traces", "hip_api_traces", "rocshmem_api_traces"):
        for itr in sdk_data["buffer_records"].get(titr, []):
            corr_id = itr["correlation_id"]["internal"]
            assert (
                corr_id not in api_corr_ids
            ), f"duplicate corr_id {corr_id} across buffer streams: {itr}"
            api_corr_ids[corr_id] = itr

    alloc_corr_ids = {}
    for itr in sdk_data["buffer_records"].get("memory_allocations", []):
        corr_id = itr["correlation_id"]["internal"]
        assert (
            corr_id not in alloc_corr_ids
        ), f"duplicate corr_id {corr_id} in memory_allocations: {itr}"
        alloc_corr_ids[corr_id] = itr

    retired_corr_ids = {}
    for itr in sdk_data["buffer_records"].get("retired_correlation_ids", []):
        corr_id = itr["internal_correlation_id"]
        assert (
            corr_id not in retired_corr_ids
        ), f"duplicate corr_id {corr_id} in retired stream: {itr}"
        retired_corr_ids[corr_id] = itr

    api_corr_ids = _sort_dict(api_corr_ids)
    alloc_corr_ids = _sort_dict(alloc_corr_ids)
    retired_corr_ids = _sort_dict(retired_corr_ids)

    for cid, itr in alloc_corr_ids.items():
        assert (
            cid in retired_corr_ids
        ), f"memory_allocation corr_id {cid} never retired: {itr}"
        retired_ts = retired_corr_ids[cid]["timestamp"]
        end_ts = itr["end_timestamp"]
        assert retired_ts - end_ts > 0, (
            f"retired_ts ({retired_ts}) <= end_ts ({end_ts}) "
            f"for alloc corr_id {cid}: {itr}"
        )

    for cid, itr in api_corr_ids.items():
        assert cid in retired_corr_ids, f"api corr_id {cid} never retired: {itr}"
        retired_ts = retired_corr_ids[cid]["timestamp"]
        end_ts = itr["end_timestamp"]
        assert retired_ts - end_ts > 0, (
            f"retired_ts ({retired_ts}) <= end_ts ({end_ts}) "
            f"for api corr_id {cid}: {itr}"
        )

    assert len(api_corr_ids) == len(retired_corr_ids), (
        f"corr_id count mismatch: api={len(api_corr_ids)}, "
        f"retired={len(retired_corr_ids)}. This assumes alloc corr_ids are a subset of "
        f"api corr_ids (allocs reuse the triggering API's id) and that the retired stream "
        f"contains no ids outside the api set."
    )


def test_perfetto_data(request, input_data):
    """The Perfetto trace emitted by json-tool contains rocSHMEM activity.

    json-tool writes a sibling `.pftrace` next to the JSON output. Mirrors
    `tests/rocprofv3/{rocdecode,rocjpeg}-trace/validate.py:test_perfetto_data`,
    but inspects the slice dataframe directly because the shared
    `rocprofv3.test_perfetto_data` helper may not yet include a
    `rocshmem_api` mapping in all builds.
    """
    sdk_data = _get_sdk_data(input_data)
    bf_records = sdk_data["buffer_records"].get("rocshmem_api_traces", [])

    json_path = request.config.getoption("--input")
    pftrace_path = json_path[: json_path.rfind(".json")] + ".pftrace"
    if not os.path.isfile(pftrace_path):
        if bf_records:
            pytest.fail(
                f"rocshmem buffer records exist but no perfetto trace was produced: "
                f"{pftrace_path}"
            )
        return pytest.skip(f"perfetto trace not produced: {pftrace_path}")

    from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader

    dataframe, _ = PerfettoReader(pftrace_path).read()
    assert (
        not dataframe.empty
    ), f"PerfettoReader returned empty dataframe for {pftrace_path}"

    # Look for rocSHMEM activity two ways: by entry name (e.g.
    # 'rocshmem_putmem_on_stream') and by category. Matching either covers
    # variations in how json-tool labels things.
    name_hits = dataframe[
        dataframe["name"].astype(str).str.contains("rocshmem", case=False, na=False)
    ]
    cat_hits = dataframe[
        dataframe["category"].astype(str).str.contains("rocshmem", case=False, na=False)
    ]

    if name_hits.empty and cat_hits.empty:
        if bf_records:
            # The JSON shows rocSHMEM was traced (buffer records exist),
            # but perfetto contains nothing about it - real bug.
            pytest.fail(
                "rocSHMEM was traced (buffer records exist) but perfetto "
                "has no rocSHMEM activity at all (categories seen: "
                f"{sorted(dataframe['category'].astype(str).unique())[:10]})."
            )
        # No rocSHMEM in either the JSON or perfetto: tracing is not
        # enabled in this build (e.g. older rocprofiler-register). The
        # JSON tests above already cover the SDK-level checks, so skip.
        return pytest.skip(
            "perfetto has no rocSHMEM activity (categories seen: "
            f"{sorted(dataframe['category'].astype(str).unique())[:10]}). "
            "rocSHMEM tracing may not be enabled in this build."
        )

    # If perfetto did record rocSHMEM, every one of the 9 traced API calls
    # should appear at least once in its entry names.
    if not name_hits.empty:
        observed = set()
        for n in name_hits["name"].astype(str):
            for op in EXPECTED_OPERATIONS:
                if op in n:
                    observed.add(op)
                    break

        # `observed` is the set of rocSHMEM API calls (of the 9 we trace)
        # whose names appeared in the perfetto trace. Three outcomes:
        # * all 9 names found: pass.
        # * some names found, others missing: bug - if any call is named
        #   in the trace, all 9 should be. Fail.
        # * no names found, but perfetto did record rocSHMEM activity:
        #   the trace shows rocSHMEM ran but doesn't yet name each
        #   individual call. This is a known intermediate state in some
        #   builds. The JSON tests above already verify each call is
        #   recorded, so skip.
        if not observed:
            return pytest.skip(
                "perfetto recorded rocSHMEM activity but doesn't name each "
                f"call (sample names: "
                f"{sorted(name_hits['name'].astype(str).unique())[:5]}). "
                "Per-call naming may not yet be enabled in this build."
            )
        assert observed == EXPECTED_OPERATIONS, (
            "perfetto is missing some rocSHMEM API calls."
            f"\n  found:   {sorted(observed)}"
            f"\n  missing: {sorted(EXPECTED_OPERATIONS - observed)}"
        )
        return

    # Reached here only when perfetto labels rocSHMEM via a 'rocshmem'
    # category but no entry name mentions rocshmem. If rocSHMEM was
    # traced (buffer records exist), the perfetto output regressed -
    # rocshmem should appear in entry names, not just the category.
    if bf_records:
        pytest.fail(
            "rocSHMEM was traced (buffer records exist) and perfetto has a "
            "'rocshmem' category, but no entry name mentions rocshmem. "
            "Did entry naming regress in the perfetto output?"
        )


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
