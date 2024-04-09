#!/usr/bin/env python3

import sys
import pytest


def test_hsa_api_trace(hsa_trace_input_data):
    functions = []
    correlation_ids = []
    for row in hsa_trace_input_data:
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

    # deterministic call counts
    num_queue_create_calls = 2
    num_queue_destroy_calls = 2
    num_hsa_mem_free_calls = 2

    # signal create/destroy calls
    # although the app explicitly only creates 3 signals
    # but hsa_init() internally calls hsa_signal_create
    num_hsa_signal_create_calls = 4
    num_hsa_signal_destroy_calls = 4

    # all correlation ids are unique
    assert len(correlation_ids) == len(hsa_trace_input_data)

    functions = list(functions)
    assert "hsa_shut_down" in functions
    assert functions.count("hsa_queue_create") == num_queue_create_calls
    assert functions.count("hsa_queue_destroy") == num_queue_destroy_calls
    assert functions.count("hsa_memory_free") == num_hsa_mem_free_calls
    assert functions.count("hsa_signal_create") == num_hsa_signal_create_calls
    assert functions.count("hsa_signal_destroy") == num_hsa_signal_destroy_calls


def test_kernel_trace(kernel_trace_input_data):
    valid_kernel_names = ("copyA", "copyB", "copyC")

    assert len(kernel_trace_input_data) == 3
    for row in kernel_trace_input_data:
        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"]) > 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert row["Kernel_Name"] in valid_kernel_names
        assert int(row["Correlation_Id"]) > 0
        assert int(row["Workgroup_Size_X"]) == 64
        assert int(row["Workgroup_Size_Y"]) == 1
        assert int(row["Workgroup_Size_Z"]) == 1
        assert int(row["Grid_Size_X"]) == 64
        assert int(row["Grid_Size_Y"]) == 1
        assert int(row["Grid_Size_Z"]) == 1
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
