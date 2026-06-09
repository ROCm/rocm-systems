# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Rocprofiler-SDK Driver Sources
set(pmc_gpu_perf_counter_driver_sources ${CMAKE_CURRENT_LIST_DIR}/driver.hpp)

# Add to parent variable
set(pmc_gpu_perf_counter_driver_sources
    ${pmc_gpu_perf_counter_driver_sources}
    PARENT_SCOPE
)
