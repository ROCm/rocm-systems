#!/usr/bin/env python3

import sys
import pytest

# The app's two captures, in capture order (tests/bin/hip-graph-multistream): its only kernel launches.
# A replayed dispatch must carry the correlation id of the launch that ran its kernel.
CAPTURES = (
    ("ms_a0", "ms_b0", "ms_c0", "ms_b1", "ms_c1", "ms_b2", "ms_c2", "ms_d0"),
    ("ms_a0", "ms_b0", "ms_c0", "ms_e0", "ms_b1", "ms_c1", "ms_e1", "ms_d0"),
)
REPLAYS = 3

LAUNCH_APIS = (
    "hipLaunchKernel",
    "hipExtLaunchKernel",
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipHccModuleLaunchKernel",
    "hipLaunchCooperativeKernel",
    "hipLaunchKernelExC",
)


def get_operation(record, kind_name, op_name):
    for itr in record["names"]:
        if kind_name == itr["kind"]:
            for oidx, oname in enumerate(itr["operations"]):
                if op_name == oname:
                    return oidx
    return None


def launch_calls(sdk_data):
    """the internal correlation ids of the app's kernel launch calls, in call order"""
    calls = set()
    for records in (sdk_data["callback_records"], sdk_data["buffer_records"]):
        ops = {get_operation(records, "HIP_RUNTIME_API", name) for name in LAUNCH_APIS}
        ops.discard(None)
        for itr in records["hip_api_traces"]:
            if itr["operation"] in ops:
                calls.add(itr["correlation_id"]["internal"])
    return sorted(calls)


def kernel_names(sdk_data):
    names = {}
    for itr in sdk_data["callback_records"]["kernel_symbols"]:
        payload = itr["payload"]
        names[payload["kernel_id"]] = payload["kernel_name"]
    return names


def graph_dispatches(sdk_data):
    """{graph_exec_id: [dispatch_info, ...] in dispatch id order}"""
    by_exec = {}
    for itr in sdk_data["buffer_records"]["kernel_dispatch"]:
        info = itr["dispatch_info"]
        exec_id = info["graph_exec_id"]["handle"]
        if exec_id != 0:
            by_exec.setdefault(exec_id, []).append(info)
    for infos in by_exec.values():
        infos.sort(key=lambda x: x["dispatch_id"])
    return by_exec


def test_launches(input_data):
    """the app's kernel launches are its captures' and nothing else"""
    calls = launch_calls(input_data["rocprofiler-sdk-json-tool"])
    assert len(calls) == sum(len(c) for c in CAPTURES), f"{len(calls)} launch calls"


def test_replays(input_data):
    """each graph replayed REPLAYS times, every node once per replay"""
    by_exec = graph_dispatches(input_data["rocprofiler-sdk-json-tool"])
    assert len(by_exec) == len(CAPTURES), f"graph execs {sorted(by_exec)}"
    for exec_id, capture in zip(sorted(by_exec), CAPTURES):
        assert len(by_exec[exec_id]) == REPLAYS * len(capture), f"exec {exec_id}"


def test_capture_correlation_across_streams(input_data):
    """every replayed dispatch carries the launch that captured its kernel, although the runtime
    enqueues a graph captured across streams in another order than its capture's"""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    calls = launch_calls(sdk_data)
    names = kernel_names(sdk_data)
    launched = [k for capture in CAPTURES for k in capture]  # call order
    kernel_of_call = dict(zip(calls, launched))
    by_exec = graph_dispatches(sdk_data)
    reordered = 0
    for exec_id, capture in zip(sorted(by_exec), CAPTURES):
        for info in by_exec[exec_id]:
            name = names.get(info["kernel_id"], "")
            capture_id = info["capture_correlation_id"]
            assert (
                capture_id in kernel_of_call
            ), f"exec {exec_id}: {name} carries {capture_id}"
            assert (
                kernel_of_call[capture_id] in name
            ), f"exec {exec_id}: {name} carries the launch of {kernel_of_call[capture_id]}"
        # the replay order differs from the capture's: the case this test exists for
        first = [names.get(i["kernel_id"], "") for i in by_exec[exec_id][: len(capture)]]
        reordered += any(k not in n for k, n in zip(capture, first))
    assert reordered > 0, "no graph replayed in another order than its capture's"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
