# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Collectors

# Add subdirectories
add_subdirectory(common)
add_subdirectory(base)
add_subdirectory(gpu)
add_subdirectory(nic)

# Collect all sources
set(pmc_collectors_sources
    ${pmc_common_sources}
    ${pmc_base_sources}
    ${pmc_gpu_sources}
    ${pmc_nic_sources}
)

# Add to parent variable
set(pmc_collectors_sources ${pmc_collectors_sources} PARENT_SCOPE)
