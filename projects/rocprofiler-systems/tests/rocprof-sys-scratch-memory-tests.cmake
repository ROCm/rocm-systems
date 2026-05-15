# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

find_package(ROCmVersion)

if(NOT ROCmVersion_FOUND)
    message(
        WARNING
        "ROCmVersion_FOUND not found, skipping tests in ${CMAKE_CURRENT_LIST_FILE}"
    )
    return()
endif()

# -------------------------------------------------------------------------------------- #
#
# scratch-memory tests
#
# -------------------------------------------------------------------------------------- #

if(NOT TARGET scratch-memory)
    return()
endif()

set(_scratch_memory_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_api,hsa_api,kernel_dispatch,memory_copy,memory_allocation,scratch_memory"
)

# Enable ROCPD for tests only if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _scratch_memory_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

rocprofiler_systems_add_test(
    SKIP_RUNTIME
    NAME scratch-memory
    TARGET scratch-memory
    GPU ON
    LABELS "scratch-memory"
    ENVIRONMENT "${_scratch_memory_environment}"
    DISABLED ${ROCPROFSYS_INSIDE_DOCKER}
)

if(TEST scratch-memory-sampling AND NOT ROCPROFSYS_INSIDE_DOCKER)
    # Validate that the test detects GPU agents and runs kernels
    set(_scratch_memory_pass_regex
        "Detected [1-9][0-9]* agents"
        "Running test_primary_then_uso"
        "Running test_gridx"
        "Running Small"
        "Running Medium"
        "Running Large"
    )

    set(_scratch_memory_fail_regex "hip error" "HSA error" "ROCPROFSYS_ABORT_FAIL_REGEX")

    set_property(
        TEST scratch-memory-sampling
        APPEND
        PROPERTY PASS_REGULAR_EXPRESSION "${_scratch_memory_pass_regex}"
    )
    set_property(
        TEST scratch-memory-sampling
        APPEND
        PROPERTY FAIL_REGULAR_EXPRESSION "${_scratch_memory_fail_regex}"
    )
endif()

# Add perfetto validation to ensure kernel dispatch events are captured (skip when in Docker)
if(NOT ROCPROFSYS_INSIDE_DOCKER)
    rocprofiler_systems_add_validation_test(
        NAME scratch-memory-sampling
        PERFETTO_METRIC "rocm_scratch_memory"
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "scratch-memory"
        ARGS -p
    )
endif()

# Add ROCPD validation if enabled (skip when in Docker)
if(
    NOT ROCPROFSYS_INSIDE_DOCKER
    AND ${ENABLE_ROCPD_TEST}
    AND ${_VALID_GPU}
    AND TEST scratch-memory-sampling
)
    set_property(TEST scratch-memory-sampling APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME scratch-memory-sampling
        ROCPD_FILE "rocpd.db"
        ARGS --validation-rules
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/default-rules.json"
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/scratch-memory/sdk-metrics-rules.json"
        LABELS "scratch-memory;rocpd"
    )
endif()
