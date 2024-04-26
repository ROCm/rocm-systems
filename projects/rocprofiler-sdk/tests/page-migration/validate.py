#!/usr/bin/env python3

from collections import defaultdict
import os
import sys
import pytest


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len


def to_dict(key_values):
    a = defaultdict()
    for kv in key_values:
        a[kv["key"]] = kv["value"]
    return a


def get_operation(record, kind_name, op_name=None):
    for idx, itr in enumerate(record["names"]):
        if kind_name == itr["kind"]:
            if op_name is None:
                return idx, itr["operations"]
            else:
                for oidx, oname in enumerate(itr["operations"]):
                    if op_name == oname:
                        return oidx

    return None


def dict_from_value_key(d):
    ret_d = defaultdict()

    for k, v in d.items():
        assert v not in ret_d
        ret_d[v] = k
    return ret_d


def sort_by_timestamp(lines):
    timestamp_line_map = {}

    for log_line in lines:
        timestamp = log_line.split(" ")[1]
        timestamp_line_map[timestamp] = log_line

    timestamps_sorted = sorted([l.split(" ")[1] for l in lines])
    return timestamps_sorted, timestamp_line_map


# ------------------------------ Tests ------------------------------ #


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    data = input_data

    node_exists("rocprofiler-sdk-json-tool", data)

    sdk_data = data["rocprofiler-sdk-json-tool"]

    node_exists("metadata", sdk_data)
    node_exists("pid", sdk_data["metadata"])
    node_exists("main_tid", sdk_data["metadata"])
    node_exists("init_time", sdk_data["metadata"])
    node_exists("fini_time", sdk_data["metadata"])
    node_exists("validate_page_migration", sdk_data["metadata"])

    assert sdk_data["metadata"]["validate_page_migration"] is True

    node_exists("agents", sdk_data)
    node_exists("call_stack", sdk_data)
    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)

    node_exists("names", sdk_data["callback_records"])
    node_exists("code_objects", sdk_data["callback_records"])
    node_exists("kernel_symbols", sdk_data["callback_records"])
    node_exists("hsa_api_traces", sdk_data["callback_records"])
    node_exists("hip_api_traces", sdk_data["callback_records"], 0)
    node_exists("marker_api_traces", sdk_data["callback_records"], 0)

    node_exists("names", sdk_data["buffer_records"])
    node_exists("kernel_dispatches", sdk_data["buffer_records"])
    node_exists("memory_copies", sdk_data["buffer_records"], 0)
    node_exists("hsa_api_traces", sdk_data["buffer_records"])
    node_exists("hip_api_traces", sdk_data["buffer_records"], 0)
    node_exists("marker_api_traces", sdk_data["buffer_records"], 0)
    node_exists("retired_correlation_ids", sdk_data["buffer_records"])
    node_exists("page_migration", sdk_data["buffer_records"])


def test_timestamps(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    cb_start = {}
    cb_end = {}
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            cid = itr["correlation_id"]["internal"]
            phase = itr["phase"]
            if phase == 1:
                cb_start[cid] = itr["timestamp"]
            elif phase == 2:
                cb_end[cid] = itr["timestamp"]
                assert cb_start[cid] <= itr["timestamp"]
            else:
                assert phase == 1 or phase == 2

        for itr in sdk_data["buffer_records"][titr]:
            assert itr["start_timestamp"] <= itr["end_timestamp"]

    for titr in ["kernel_dispatches", "memory_copies"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["start_timestamp"] < itr["end_timestamp"]
            assert itr["correlation_id"]["internal"] > 0
            assert itr["correlation_id"]["external"] > 0
            assert sdk_data["metadata"]["init_time"] < itr["start_timestamp"]
            assert sdk_data["metadata"]["init_time"] < itr["end_timestamp"]
            assert sdk_data["metadata"]["fini_time"] > itr["start_timestamp"]
            assert sdk_data["metadata"]["fini_time"] > itr["end_timestamp"]

            # api_start = cb_start[itr["correlation_id"]["internal"]]
            # api_end = cb_end[itr["correlation_id"]["internal"]]
            # assert api_start < itr["start_timestamp"]
            # assert api_end <= itr["end_timestamp"]


def test_internal_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    api_corr_ids = []
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            api_corr_ids.append(itr["correlation_id"]["internal"])

        for itr in sdk_data["buffer_records"][titr]:
            api_corr_ids.append(itr["correlation_id"]["internal"])

    api_corr_ids_sorted = sorted(api_corr_ids)
    api_corr_ids_unique = list(set(api_corr_ids))

    for itr in sdk_data["buffer_records"]["kernel_dispatches"]:
        assert itr["correlation_id"]["internal"] in api_corr_ids_unique

    for itr in sdk_data["buffer_records"]["memory_copies"]:
        assert itr["correlation_id"]["internal"] in api_corr_ids_unique

    len_corr_id_unq = len(api_corr_ids_unique)
    assert len(api_corr_ids) != len_corr_id_unq
    assert max(api_corr_ids_sorted) == len_corr_id_unq


def test_external_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    extern_corr_ids = []
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            extern_corr_ids.append(itr["correlation_id"]["external"])

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            assert itr["thread_id"] in extern_corr_ids
            assert itr["correlation_id"]["external"] in extern_corr_ids

    for itr in sdk_data["buffer_records"]["kernel_dispatches"]:
        assert itr["correlation_id"]["external"] > 0
        assert itr["correlation_id"]["external"] in extern_corr_ids

    for itr in sdk_data["buffer_records"]["memory_copies"]:
        assert itr["correlation_id"]["external"] > 0
        assert itr["correlation_id"]["external"] in extern_corr_ids


def test_kernel_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    symbol_info = {}
    for itr in sdk_data["callback_records"]["kernel_symbols"]:
        phase = itr["phase"]
        payload = itr["payload"]
        kern_id = payload["kernel_id"]

        assert phase == 1 or phase == 2
        assert kern_id > 0
        if phase == 1:
            assert len(payload["kernel_name"]) > 0
            symbol_info[kern_id] = payload
        elif phase == 2:
            assert payload["kernel_id"] in symbol_info.keys()
            assert payload["kernel_name"] == symbol_info[kern_id]["kernel_name"]

    for itr in sdk_data["buffer_records"]["kernel_dispatches"]:
        assert itr["dispatch_info"]["kernel_id"] in symbol_info.keys()

    for itr in sdk_data["callback_records"]["kernel_dispatches"]:
        assert itr["payload"]["dispatch_info"]["kernel_id"] in symbol_info.keys()


def test_retired_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    def _sort_dict(inp):
        return dict(sorted(inp.items()))

    api_corr_ids = {}
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["buffer_records"][titr]:
            corr_id = itr["correlation_id"]["internal"]
            assert corr_id not in api_corr_ids.keys()
            api_corr_ids[corr_id] = itr

    async_corr_ids = {}
    for titr in ["kernel_dispatches", "memory_copies"]:
        for itr in sdk_data["buffer_records"][titr]:
            corr_id = itr["correlation_id"]["internal"]
            assert corr_id not in async_corr_ids.keys()
            async_corr_ids[corr_id] = itr

    retired_corr_ids = {}
    for itr in sdk_data["buffer_records"]["retired_correlation_ids"]:
        corr_id = itr["internal_correlation_id"]
        assert corr_id not in retired_corr_ids.keys()
        retired_corr_ids[corr_id] = itr

    api_corr_ids = _sort_dict(api_corr_ids)
    async_corr_ids = _sort_dict(async_corr_ids)
    retired_corr_ids = _sort_dict(retired_corr_ids)

    for cid, itr in async_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        ts = retired_corr_ids[cid]["timestamp"]
        assert (ts - itr["end_timestamp"]) > 0, f"correlation-id: {cid}, data: {itr}"

    for cid, itr in api_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        ts = retired_corr_ids[cid]["timestamp"]
        assert (ts - itr["end_timestamp"]) > 0, f"correlation-id: {cid}, data: {itr}"

    assert len(api_corr_ids.keys()) == (len(retired_corr_ids.keys()))


def get_allocated_pages(callback_records):
    # Get how many pages we allocated
    op_idx = get_operation(callback_records, "HIP_RUNTIME_API", "hipHostRegister")
    rt_idx, rt_data = get_operation(callback_records, "HIP_RUNTIME_API")

    assert op_idx is not None, f"{rt_idx}:\n{rt_data}"

    host_register_record = []
    for itr in callback_records["hip_api_traces"]:
        if itr["kind"] == rt_idx and itr["operation"] == op_idx and itr["phase"] == 2:
            assert "sizeBytes" in itr["args"].keys(), f"{itr}"
            assert "hostPtr" in itr["args"].keys(), f"{itr}"
            host_register_record.append(itr)

    assert len(host_register_record) == 1
    alloc_size = int(host_register_record[0]["args"]["sizeBytes"], 10)
    start_addr = int(host_register_record[0]["args"]["hostPtr"], 16)
    end_addr = start_addr + alloc_size

    return start_addr, end_addr, alloc_size


def test_page_migration_data(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]
    buffer_records = sdk_data["buffer_records"]
    callback_records = sdk_data["callback_records"]
    page_migration_buffers = buffer_records["page_migration"]

    _, bf_op_names = get_operation(buffer_records, "PAGE_MIGRATION")
    assert bf_op_names[0] == "PAGE_MIGRATION_NONE"
    assert "PAGE_MIGRATION_PAGE_MIGRATE" in bf_op_names
    assert len(bf_op_names) == 5

    node_ids = set(x["gpu_id"] for x in sdk_data["agents"])
    start_addr, end_addr, alloc_size = get_allocated_pages(callback_records)

    assert start_addr < end_addr and start_addr + alloc_size == end_addr
    assert int(alloc_size) == 16 * 4096  # We allocated 16 pages in the test

    # PID must be same
    assert len(set(r["pid"] for r in page_migration_buffers)) == 1

    for r in page_migration_buffers:
        op = r["operation"]

        assert r["size"] == 136
        assert op != 0 and bf_op_names[op] != "PAGE_MIGRATION_NONE"
        assert bf_op_names[op].lower().replace("page_migration_", "") in r.keys()

        if "page_migrate" in r:
            assert r["page_migrate"]["from_node"] in node_ids
            assert r["page_migrate"]["to_node"] in node_ids
            assert r["page_migrate"]["prefetch_node"] in node_ids
            assert r["page_migrate"]["preferred_node"] in node_ids
            assert r["page_migrate"]["trigger"] >= 0

        if "queue_suspend" in r:
            assert r["queue_suspend"]["trigger"] >= 0
            assert r["queue_suspend"]["node_id"] in node_ids

        if "unmap_from_gpu" in r:
            assert r["unmap_from_gpu"]["trigger"] >= 0
            # unmap is "instantaneous"
            assert 0 < r["start_timestamp"] == r["end_timestamp"]
        else:
            assert 0 < r["start_timestamp"] < r["end_timestamp"]

    # Check for events with our page
    for r in page_migration_buffers:

        if "page_migrate" in r and r["page_migrate"]["start_addr"] == start_addr:
            assert end_addr == r["page_migrate"]["end_addr"]

        if "unmap_from_gpu" in r and r["unmap_from_gpu"]["start_addr"] == start_addr:
            assert end_addr == r["unmap_from_gpu"]["end_addr"]

    # TODO: Check if a migrate a->b is paired up with b->a
    # It may not always be reported towards app finalization


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
