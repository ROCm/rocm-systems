# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Rocprofiler-SDK Device Provider Sources
set(pmc_gpu_perf_counter_provider_sources ${CMAKE_CURRENT_LIST_DIR}/provider.hpp)

# Add to parent variable
set(pmc_gpu_perf_counter_provider_sources
    ${pmc_gpu_perf_counter_provider_sources}
    PARENT_SCOPE
)
