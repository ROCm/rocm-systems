# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Collectors

# Add subdirectories
add_subdirectory(common)
add_subdirectory(base)
add_subdirectory(gpu)

# Conditionally compile NIC collector based on AINIC support
# ROCPROFSYS_BUILD_AINIC is set in cmake/Packages.cmake based on:
# - ROCPROFSYS_USE_AINIC option being ON
# - AMD SMI version >= 26.3
if(ROCPROFSYS_BUILD_AINIC)
    add_subdirectory(nic)
else()
    set(pmc_nic_sources "")
endif()

# Collect all sources
set(pmc_collectors_sources
    ${pmc_common_sources}
    ${pmc_base_sources}
    ${pmc_gpu_sources}
    ${pmc_nic_sources}
)

# Add to parent variable
set(pmc_collectors_sources ${pmc_collectors_sources} PARENT_SCOPE)
