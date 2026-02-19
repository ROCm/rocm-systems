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

# Skip all tests if no valid GPU is detected
if(NOT _VALID_GPU)
    message(
        STATUS
        "transferBench requires a GPU and no valid GPUs were found; skipping GPU connect tests"
    )
    return()
endif()

# Skip if transferBench target is not available
if(NOT TARGET transferBench)
    message(WARNING "transferBench not available; GPU connect tests will be skipped")
    return()
endif()

# Use legacy trace mode for AMD SMI counters - cached mode doesn't support real-time counter tracking
set(_gpu_connect_environment
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_TRACE_LEGACY=ON"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api"
    "ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,xgmi,pcie"
    "ROCPROFSYS_SAMPLING_CPUS=none"
    "ROCPROFSYS_USE_SAMPLING=OFF"
    "ROCPROFSYS_PROCESS_SAMPLING_FREQ=50"
    "ROCPROFSYS_CPU_FREQ_ENABLED=OFF"
)

set(_gpu_connect_rocpd_validation_rules
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/gpu-connect/validation-rules.json"
    "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/gpu-connect/amd-smi-rules.json"
)

# Enable ROCPD for tests only if valid ROCm is installed
if(ENABLE_ROCPD_TEST)
    list(APPEND _gpu_connect_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# Add a runtime validation test that checks if transferBench can run successfully
# This test runs before all other GPU connect tests and acts as a fixture
add_test(
    NAME transferbench-validation-check
    COMMAND $<TARGET_FILE:transferBench>
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
)

# Set this test as a fixture that must pass for GPU connect tests to run
set_tests_properties(
    transferbench-validation-check
    PROPERTIES
        LABELS "transferbench;validation"
        FIXTURES_SETUP transferbench_available
        FAIL_REGULAR_EXPRESSION "Error: No valid transfers created"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_SAMPLING SKIP_RUNTIME
    NAME transferbench
    TARGET transferBench
    GPU ON
    ENVIRONMENT "${_base_environment};${_gpu_connect_environment}"
    LABELS "transferbench;xgmi;pcie"
    SYS_RUN_SKIP_REGEX "Error: No valid transfers created"
)

# Make this test depend on the transferBench validation fixture
if(TEST transferbench-sys-run)
    set_property(
        TEST transferbench-sys-run
        APPEND
        PROPERTY FIXTURES_REQUIRED transferbench_available
    )
endif()

# Add validation test to check for XGMI data transfers
rocprofiler_systems_add_validation_test(
    NAME transferbench-sys-run
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "transferbench;perfetto"
    ARGS --counter-names "XGMI Read Data" "XGMI Write Data" -p
)
if(TEST validate-transferbench-sys-run-perfetto)
    set_property(
        TEST validate-transferbench-sys-run-perfetto
        APPEND
        PROPERTY FIXTURES_REQUIRED transferbench_available
    )
endif()

# Add ROCPD validation if enabled
if(${ENABLE_ROCPD_TEST} AND TEST transferbench-sys-run)
    set_property(TEST transferbench-sys-run APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME transferbench-sys-run
        ROCPD_FILE "rocpd.db"
        LABELS "transferbench;rocpd"
        ARGS --validation-rules
            ${_gpu_connect_rocpd_validation_rules}
    )

    # Make ROCPD validation test depend on the transferBench fixture
    if(TEST validate-transferbench-sys-run-rocpd)
        set_property(
            TEST validate-transferbench-sys-run-rocpd
            APPEND
            PROPERTY FIXTURES_REQUIRED transferbench_available
        )
    endif()
endif()
