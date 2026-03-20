# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Add tests subdirectory
add_subdirectory(tests)

# PMC GPU Collector Sources
set(pmc_gpu_sources
    ${CMAKE_CURRENT_LIST_DIR}/types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device.hpp
    ${CMAKE_CURRENT_LIST_DIR}/collector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/gpu_traits.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto_policy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cache_policy.hpp
)

# Add to parent variable (will be used by collectors CMakeLists.txt)
set(pmc_gpu_sources ${pmc_gpu_sources} PARENT_SCOPE)
