# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

# PMC CPU Collector Sources
set(pmc_cpu_sources
    ${CMAKE_CURRENT_LIST_DIR}/types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/sample.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cpu_traits.hpp
    ${CMAKE_CURRENT_LIST_DIR}/collector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cache_policy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto_policy.hpp
)

set(pmc_cpu_sources ${pmc_cpu_sources} PARENT_SCOPE)
