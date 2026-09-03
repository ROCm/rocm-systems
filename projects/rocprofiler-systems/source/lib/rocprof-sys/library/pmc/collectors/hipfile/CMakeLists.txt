# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(ROCPROFSYS_BUILD_TESTING)
    add_subdirectory(tests)
endif()

# PMC hipFile Collector Sources (header-only)
set(pmc_hipfile_sources
    ${CMAKE_CURRENT_LIST_DIR}/types.hpp
    ${CMAKE_CURRENT_LIST_DIR}/device.hpp
    ${CMAKE_CURRENT_LIST_DIR}/hipfile_traits.hpp
    ${CMAKE_CURRENT_LIST_DIR}/cache_policy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/perfetto_policy.hpp
    ${CMAKE_CURRENT_LIST_DIR}/collector.hpp
)

# Add to parent variable (used by collectors CMakeLists.txt)
set(pmc_hipfile_sources ${pmc_hipfile_sources} PARENT_SCOPE)
