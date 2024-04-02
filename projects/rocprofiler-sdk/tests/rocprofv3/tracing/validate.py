#!/usr/bin/env python3

import sys
import subprocess
import pytest


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


def test_memory_copy_trace(memory_copy_input_data):
    for row in memory_copy_input_data:
        assert row["Kind"] == "MEMORY_COPY"

    assert len(memory_copy_input_data) == 2

    row = memory_copy_input_data[0]
    assert row["Direction"] == "HOST_TO_DEVICE"
    output = subprocess.check_output(
        'rocminfo | grep "Node: *'
        + row["Source_Agent_Id"]
        + '" -A 1 | grep "Device Type" | sed \'s/.*: *//\'',
        shell=True,
    )
    assert int(str(output).find("CPU")) >= 0
    output = subprocess.check_output(
        'rocminfo | grep "Node: *'
        + row["Destination_Agent_Id"]
        + '" -A 1 | grep "Device Type" | sed \'s/.*: *//\'',
        shell=True,
    )
    assert int(str(output).find("GPU")) >= 0
    assert int(row["Correlation_Id"]) > 0
    assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    row = memory_copy_input_data[1]
    assert row["Direction"] == "DEVICE_TO_HOST"
    output = subprocess.check_output(
        'rocminfo | grep "Node: *'
        + row["Source_Agent_Id"]
        + '" -A 1 | grep "Device Type" | sed \'s/.*: *//\'',
        shell=True,
    )
    assert int(str(output).find("GPU")) >= 0
    output = subprocess.check_output(
        'rocminfo | grep "Node: *'
        + row["Destination_Agent_Id"]
        + '" -A 1 | grep "Device Type" | sed \'s/.*: *//\'',
        shell=True,
    )
    assert int(str(output).find("CPU")) >= 0
    assert int(row["Correlation_Id"]) > 0
    assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])


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
