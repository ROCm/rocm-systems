# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Collectors

# Add subdirectories
add_subdirectory(common)
add_subdirectory(base)
if(ROCPROFSYS_USE_ROCM)
    add_subdirectory(gpu)
endif()

# Collect all sources
set(pmc_collectors_sources ${pmc_common_sources} ${pmc_base_sources} ${pmc_gpu_sources})

# Add to parent variable
set(pmc_collectors_sources ${pmc_collectors_sources} PARENT_SCOPE)
