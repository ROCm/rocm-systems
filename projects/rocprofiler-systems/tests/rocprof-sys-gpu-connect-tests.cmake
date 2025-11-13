# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# -------------------------------------------------------------------------------------- #
#
# GPU connectivity tests (transferBench)
#
# -------------------------------------------------------------------------------------- #

set(_gpu_connect_environment
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,xgmi,pcie"
    "ROCPROFSYS_SAMPLING_CPUS=none"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_PROCESS_SAMPLING_FREQ=10"
    "ROCPROFSYS_CPU_FREQ_ENABLED=OFF"
)

set(_gpu_connect_rocpd_validation_rules
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/gpu-connect/validation-rules.json"
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/gpu-connect/amd-smi-rules.json"
)

# Enable ROCPD for tests only if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _gpu_connect_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

set(skip_validation FALSE)

if(EXISTS "${PROJECT_BINARY_DIR}/transferBench")
    execute_process(
        COMMAND ${PROJECT_BINARY_DIR}/transferBench
        OUTPUT_VARIABLE _transfer_output
        ERROR_VARIABLE _transfer_output
        RESULT_VARIABLE _transfer_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(_transfer_output MATCHES "Error: No valid transfers created")
        set(skip_validation TRUE)
    endif()
endif()

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_SAMPLING SKIP_RUNTIME
    NAME transferbench
    TARGET transferBench
    GPU ON
    ENVIRONMENT "${_base_environment};${_gpu_connect_environment}"
    LABELS "transferbench;xgmi;pcie"
    SYS_RUN_SKIP_REGEX "Error: No valid transfers created"
)

if(NOT skip_validation)
    rocprofiler_systems_add_validation_test(
        NAME transferbench-sys-run
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "transferbench;perfetto"
        ARGS --counter-names "XGMI Read Data" "XGMI Write Data" -p
    )

    if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
        set_property(TEST transferbench-sys-run APPEND PROPERTY LABELS rocpd)

        rocprofiler_systems_add_validation_test(
            NAME transferbench-sys-run
            ROCPD_FILE "rocpd.db"
            LABELS "transferbench;rocpd"
            ARGS --validation-rules
                ${_gpu_connect_rocpd_validation_rules}
        )
    endif()
else()
    message(STATUS "TransferBench: No valid transfers created, skipping tests")
endif()
