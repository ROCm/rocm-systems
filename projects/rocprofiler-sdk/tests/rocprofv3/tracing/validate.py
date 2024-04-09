#!/usr/bin/env python3

import sys
import pytest


def test_agent_info(agent_info_input_data):
    logical_node_id = max([int(itr["Logical_Node_Id"]) for itr in agent_info_input_data])

    assert logical_node_id + 1 == len(agent_info_input_data)

    for row in agent_info_input_data:
        agent_type = row["Agent_Type"]
        assert agent_type in ("CPU", "GPU")
        if agent_type == "CPU":
            assert int(row["Cpu_Cores_Count"]) > 0
            assert int(row["Simd_Count"]) == 0
            assert int(row["Max_Waves_Per_Simd"]) == 0
        else:
            assert int(row["Cpu_Cores_Count"]) == 0
            assert int(row["Simd_Count"]) > 0
            assert int(row["Max_Waves_Per_Simd"]) > 0


def test_hsa_api_trace(hsa_input_data):
    functions = []
    correlation_ids = []
    for row in hsa_input_data:
        assert row["Domain"] in (
            "HSA_CORE_API",
            "HSA_AMD_EXT_API",
            "HSA_IMAGE_EXT_API",
            "HSA_FINALIZE_EXT_API",
        )
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) >= int(row["Process_Id"])
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])
        correlation_ids.append(int(row["Correlation_Id"]))

    correlation_ids = sorted(list(set(correlation_ids)))

    hsa_api_calls_offset = 2  # roctxRangePush is first
    num_marker_api_calls = 7  # seven marker API calls, only six entries in
    # marker csv data because roctxRangePush + roctxRangePop is one entry

    # all correlation ids are unique
    assert len(correlation_ids) == len(hsa_input_data)
    # correlation ids are numbered from 1 to N
    assert correlation_ids[0] == hsa_api_calls_offset
    assert correlation_ids[-1] == len(correlation_ids) + num_marker_api_calls

    functions = list(set(functions))
    assert "hsa_amd_memory_async_copy_on_engine" in functions


def test_kernel_trace(kernel_input_data):
    valid_kernel_names = (
        "_Z15matrixTransposePfS_i.kd",
        "matrixTranspose(float*, float*, int)",
    )

    assert len(kernel_input_data) == 1
    for row in kernel_input_data:
        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"]) > 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert row["Kernel_Name"] in valid_kernel_names
        assert int(row["Correlation_Id"]) > 0
        assert int(row["Workgroup_Size_X"]) == 4
        assert int(row["Workgroup_Size_Y"]) == 4
        assert int(row["Workgroup_Size_Z"]) == 1
        assert int(row["Grid_Size_X"]) == 1024
        assert int(row["Grid_Size_Y"]) == 1024
        assert int(row["Grid_Size_Z"]) == 1
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])


def test_memory_copy_trace(agent_info_input_data, memory_copy_input_data):

    def get_agent(node_id):
        for row in agent_info_input_data:
            if row["Logical_Node_Id"] == node_id:
                return row
        return None

    for row in memory_copy_input_data:
        assert row["Kind"] == "MEMORY_COPY"

    assert len(memory_copy_input_data) == 2

    def test_row(idx, direction):
        assert direction in ("HOST_TO_DEVICE", "DEVICE_TO_HOST")
        row = memory_copy_input_data[idx]
        assert row["Direction"] == direction
        src_agent = get_agent(row["Source_Agent_Id"])
        dst_agent = get_agent(row["Destination_Agent_Id"])
        assert src_agent is not None and dst_agent is not None, f"{agent_info_input_data}"
        if direction == "HOST_TO_DEVICE":
            assert src_agent["Agent_Type"] == "CPU"
            assert dst_agent["Agent_Type"] == "GPU"
        else:
            assert src_agent["Agent_Type"] == "GPU"
            assert dst_agent["Agent_Type"] == "CPU"
        assert int(row["Correlation_Id"]) > 0
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    test_row(0, "HOST_TO_DEVICE")
    test_row(1, "DEVICE_TO_HOST")


def test_marker_api_trace(marker_input_data):
    functions = []
    for row in marker_input_data:
        assert row["Domain"] in [
            "MARKER_CORE_API",
            "MARKER_CONTROL_API",
            "MARKER_NAME_API",
        ]
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) == 0 or int(row["Thread_Id"]) >= int(
            row["Process_Id"]
        )
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])

    functions = list(set(functions))
    assert "main" in functions


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
