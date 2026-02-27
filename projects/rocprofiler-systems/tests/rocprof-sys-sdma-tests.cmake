# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# SDMA tests
#
# -------------------------------------------------------------------------------------- #

set(_sdma_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation,scratch_memory"
    "ROCPROFSYS_AMD_SMI_METRICS=sdma_usage"
)

# Check AMD SMI version - sdma_usage metric requires version >= 26.3.0
set(_DISABLE_SDMA_TEST ON)
find_package(amd_smi QUIET HINTS ${ROCmVersion_DIR} PATHS ${ROCmVersion_DIR})

if(amd_smi_FOUND AND amd_smi_VERSION VERSION_GREATER_EQUAL "26.3.0")
    set(_DISABLE_SDMA_TEST OFF)
else()
    if(amd_smi_FOUND)
        rocprofiler_systems_message(
            STATUS
            "AMD SMI version ${amd_smi_VERSION} is less than 26.3.0. Disabling SDMA tests..."
        )
    else()
        rocprofiler_systems_message(
            STATUS "AMD SMI package not found. Disabling SDMA tests..."
        )
    endif()
endif()

rocprofiler_systems_add_test(
    SKIP_RUNTIME SKIP_REWRITE
    NAME sdma-test
    TARGET sdma-test
    MPI OFF
    GPU ON
    NUM_PROCS 1
    ENVIRONMENT "${_sdma_environment}"
    DISABLED ${_DISABLE_SDMA_TEST}
)
