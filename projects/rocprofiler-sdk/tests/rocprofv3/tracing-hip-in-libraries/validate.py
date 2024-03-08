#!/usr/bin/env python3

import sys
import pytest


class dim3(object):
    def __init__(self, x, y, z):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def as_tuple(self):
        return (self.x, self.y, self.z)


def test_api_trace(hsa_input_data, hip_input_data):
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

    for row in hip_input_data:
        assert row["Domain"] in [
            "HIP_RUNTIME_API",
            "HIP_COMPILER_API",
        ]
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) == 0 or int(row["Thread_Id"]) >= int(
            row["Process_Id"]
        )
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])
        functions.append(row["Function"])
        correlation_ids.append(int(row["Correlation_Id"]))

    correlation_ids = sorted(list(set(correlation_ids)))

    # all correlation ids are unique
    assert len(correlation_ids) == (len(hsa_input_data) + len(hip_input_data))
    # correlation ids are numbered from 1 to N
    assert correlation_ids[0] == 1
    assert correlation_ids[-1] == len(correlation_ids)

    functions = list(set(functions))
    for itr in (
        "hsa_amd_memory_async_copy_on_engine",
        "hsa_agent_get_info",
        "hsa_agent_iterate_isas",
        "hsa_signal_create",
        "hsa_agent_get_info",
        "hsa_executable_symbol_get_info",
    ):
        assert itr in functions
    if hip_input_data:
        for itr in (
            "hipGetLastError",
            "hipLaunchKernel",
            "hipStreamSynchronize",
            "hipMemcpyAsync",
            "hipFree",
            "hipStreamDestroy",
            "hipDeviceSynchronize",
            "hipDeviceReset",
            "hipSetDevice",
        ):
            assert itr in functions


def test_kernel_trace(kernel_input_data):
    valid_kernel_names = sorted(
        [
            "__amd_rocclr_fillBufferAligned",
            "(anonymous namespace)::transpose(int const*, int*, int, int)",
            "void (anonymous namespace)::addition_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::divide_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::multiply_kernel<float>(float*, float const*, float const*, int, int)",
            "void (anonymous namespace)::subtract_kernel<float>(float*, float const*, float const*, int, int)",
        ]
    )

    kernels = []
    for row in kernel_input_data:
        kernel_name = row["Kernel_Name"]

        assert row["Kind"] == "KERNEL_DISPATCH"
        assert int(row["Agent_Id"]) > 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Kernel_Id"]) > 0
        assert int(row["Correlation_Id"]) > 0
        assert kernel_name in valid_kernel_names

        if kernel_name not in kernels:
            kernels.append(kernel_name)

        workgrp_size = dim3(
            row["Workgroup_Size_X"], row["Workgroup_Size_Y"], row["Workgroup_Size_Z"]
        )
        grid_size = dim3(row["Grid_Size_X"], row["Grid_Size_Y"], row["Grid_Size_Z"])

        if kernel_name == "__amd_rocclr_fillBufferAligned":
            assert workgrp_size.as_tuple() > (1, 1, 1)
            assert grid_size.as_tuple() > (1, 1, 1)
        elif "transpose" in kernel_name:
            assert workgrp_size.as_tuple() == (32, 32, 1)
            assert grid_size.as_tuple() == (9920, 9920, 1)
        else:
            assert workgrp_size.as_tuple() == (64, 1, 1)
            assert grid_size.as_tuple() == (4096, 2048, 1)

        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    kernels = sorted(list(set(kernels)))
    assert kernels == valid_kernel_names


def test_memory_copy_trace(memory_copy_input_data):
    for row in memory_copy_input_data:
        assert row["Kind"] == "MEMORY_COPY"
        assert row["Direction"] in ("HOST_TO_DEVICE", "DEVICE_TO_HOST")
        if row["Direction"] == "HOST_TO_DEVICE":
            assert int(row["Source_Agent_Id"]) == 0
        elif row["Direction"] == "DEVICE_TO_HOST":
            assert int(row["Destination_Agent_Id"]) == 0
        assert int(row["Correlation_Id"]) > 0
        assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"])

    assert len(memory_copy_input_data) == 120


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
