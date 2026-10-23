# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

execute_process(
    COMMAND
        "${AMDCXX}" -x hip --offload-arch=gfx950 "--rocm-path=${ROCM_PATH}"
        -fPIC -c "${CMAKE_CURRENT_LIST_DIR}/hip_toolchain_probe.hip" -o
        "${CMAKE_CURRENT_BINARY_DIR}/hip_toolchain_probe.o"
    RESULT_VARIABLE _rj_hip_probe_result
    OUTPUT_VARIABLE _rj_hip_probe_output
    ERROR_VARIABLE _rj_hip_probe_output
)
set(RJ_HAS_HIP_KERNEL_TOOLCHAIN FALSE)
if(_rj_hip_probe_result EQUAL 0)
    set(RJ_HAS_HIP_KERNEL_TOOLCHAIN TRUE)
else()
    message(
        STATUS
        "HIP kernel compilation probe failed - device kernel and HIP tests will be disabled:\n${_rj_hip_probe_output}"
    )
endif()
