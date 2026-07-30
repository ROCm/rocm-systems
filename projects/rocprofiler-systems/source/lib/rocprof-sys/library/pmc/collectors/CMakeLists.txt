# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# PMC Collectors

# Add subdirectories
add_subdirectory(common)
add_subdirectory(base)
add_subdirectory(gpu)
add_subdirectory(cpu)

# GPU perf counter collector requires ROCm >= 6.4 (device counting service API change)
if(
    (ROCPROFSYS_ROCM_VERSION_MAJOR GREATER 6)
    OR (
        ROCPROFSYS_ROCM_VERSION_MAJOR EQUAL 6
        AND ROCPROFSYS_ROCM_VERSION_MINOR GREATER_EQUAL 4
    )
)
    add_subdirectory(gpu_perf_counter)
else()
    set(pmc_gpu_perf_counter_sources "")
endif()

# Conditionally compile NIC collector based on AINIC support
# ROCPROFSYS_BUILD_AINIC is set in cmake/Packages.cmake based on:
# - ROCPROFSYS_USE_AINIC option being ON
# - AMD SMI version >= 26.3
if(ROCPROFSYS_BUILD_AINIC)
    add_subdirectory(nic)
else()
    set(pmc_nic_sources "")
    # NIC device tests use a mock backend with no AMD SMI dependency
    if(ROCPROFSYS_BUILD_TESTING)
        add_subdirectory(nic/tests)
    endif()
endif()

# Collect all sources
set(pmc_collectors_sources
    ${pmc_common_sources}
    ${pmc_base_sources}
    ${pmc_gpu_sources}
    ${pmc_cpu_sources}
    ${pmc_gpu_perf_counter_sources}
    ${pmc_nic_sources}
)

# Add to parent variable
set(pmc_collectors_sources ${pmc_collectors_sources} PARENT_SCOPE)
