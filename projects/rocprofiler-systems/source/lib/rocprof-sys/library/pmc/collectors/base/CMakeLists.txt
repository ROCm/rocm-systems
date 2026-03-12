# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# PMC Base Collector Sources
set(pmc_base_sources
    ${CMAKE_CURRENT_LIST_DIR}/collector.hpp
    ${CMAKE_CURRENT_LIST_DIR}/traits_check.hpp
)

# Add to parent variable (will be used by collectors CMakeLists.txt)
set(pmc_base_sources ${pmc_base_sources} PARENT_SCOPE)
