#!/usr/bin/env python3

import sys
import pytest


def test_hsa_api_trace(hsa_input_data):
    functions = []
    correlation_ids = []
    for row in hsa_input_data:
        assert row["Domain"] == "HSA_API"
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) >= int(row["Process_Id"])
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])
        correlation_ids.append(int(row["Correlation_Id"]))

    correlation_ids = sorted(list(set(correlation_ids)))

    # all correlation ids are unique
    assert len(correlation_ids) == len(hsa_input_data)
    # correlation ids are numbered from 1 to N
    assert correlation_ids[0] == 1
    assert correlation_ids[-1] == len(correlation_ids)

    functions = list(set(functions))
    assert "hsa_amd_memory_async_copy_on_engine" in functions


def test_kernel_trace(kernel_input_data):
    for row in kernel_input_data:
        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"]) > 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert row["Kernel_Name"] == "matrixTranspose(float*, float*, int)"
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

    row = memory_copy_input_data[0]
    assert row["Direction"] == "HOST_TO_DEVICE"
    assert int(row["Source_Agent_Id"]) == 0
    assert int(row["Destination_Agent_Id"]) == 1
    assert int(row["Correlation_Id"]) > 0
    assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    row = memory_copy_input_data[1]
    assert row["Direction"] == "DEVICE_TO_HOST"
    assert int(row["Source_Agent_Id"]) == 1
    assert int(row["Destination_Agent_Id"]) == 0
    assert int(row["Correlation_Id"]) > 0
    assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
